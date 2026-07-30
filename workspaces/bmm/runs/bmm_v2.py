import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Batched fp32 matmul out[b] = lhs[b] @ rhs[b], b in 0..15.

    Shapes: v1 lhs (16,4096,64)=(B,M,K); v2 rhs (16,64,4096)=(B,K,N);
    out (16,4096,4096)=(B,M,N). K=64 <= 128 so the whole contraction is one
    Tensor-Engine pass per output tile (single nc_matmul, no K-accumulation).

    Structure: for each batch, all 32 m-tiles are transposed first into a resident
    packed SBUF buffer, then all main matmuls run against the resident rhs with no
    transpose interleaved between them. Separating the lhs transpose from the main
    matmul (instead of transpose->matmul per m-tile as in bmm_v1) removes the serial
    transpose->copy dependency from the head of each matmul burst; with the whole
    batch transposed up front, the resulting matmul run is long enough that the
    compiler hides every PSUM->SBUF copy+store behind the next matmul, so the tensor
    engine is fed with little idle time between instructions. This is a pure schedule
    change: identical single-pass K=64 math to bmm_v1 (transpose-before-use is exact,
    so transposing all m-tiles up front then matmul is bit-identical to interleaving).

    Layout per batch:
      - transpose all 32 lhs subtiles [m_in=128, k=64] -> [k=64, m_in=128] via the
        identity nc_matmul, packing into one resident [k=64, 32*128=4096] SBUF buffer.
      - for each of the 32 subtiles and each 1024-wide n-group: two [128,512]
        nc_matmuls into two PSUM banks, copied into one [128,1024] SBUF tile and
        stored with a single 1024-wide nl.store.

    rhs[b] stays fully resident [64,4096] (loaded once per batch). Bank budget per
    batch: the transpose PSUM bank drains before the 2 output PSUM banks are live, so
    at most 3 of the 8 physical PSUM banks are used at once. Resident SBUF: packed
    transpose buffer 64 x 4096 x 4B = 16 KB/partition + rhs 16 KB/partition = 32
    KB/partition, well within the 192 KB budget (no spill; HBM read/write stay at the
    once-each floor).
    """
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    B = 16
    M = 4096
    K = 64
    N = 4096
    M_TILES = 32
    M_SUB = 32                      # m-subtiles per m-block
    M_BLOCKS = M_TILES // M_SUB     # 1 (whole batch is one block)
    N_CHUNK = 512
    WIDE = 1024
    N_WIDE = N // WIDE              # 4 wide store groups
    G = WIDE // N_CHUNK             # 2 PSUM banks feed one wide store

    out = nl.ndarray((B, M, N), dtype=np.float32, buffer=nl.shared_hbm)

    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]],
        dtype=np.float32)

    for b in nl.affine_range(B):
        rhs_sb = nl.ndarray((par_dim(K), N), dtype=np.float32, buffer=nl.sbuf)
        rhs_sb[nl.arange(K)[:, None], nl.arange(N)[None, :]] = nl.load(
            v2[b, nl.arange(K)[:, None], nl.arange(N)[None, :]], dtype=np.float32)

        for mblk in nl.affine_range(M_BLOCKS):
            # Transpose all M_SUB subtiles up front, packed into [k=64, M_SUB*128].
            lhs_t_pack = nl.ndarray((par_dim(K), M_SUB * 128), dtype=np.float32,
                                    buffer=nl.sbuf)
            for s in nl.affine_range(M_SUB):
                mt = M_SUB * mblk + s
                lhs_sb = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
                lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]] = nl.load(
                    v1[b, 128 * mt + nl.arange(128)[:, None], nl.arange(K)[None, :]],
                    dtype=np.float32)
                psum_t = nl.ndarray((par_dim(K), 128), dtype=np.float32, buffer=nl.psum)
                psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                    lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]],
                    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    is_transpose=True, is_moving_onezero=True)
                lhs_t_pack[nl.arange(K)[:, None], 128 * s + nl.arange(128)[None, :]] = nl.copy(
                    psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]],
                    dtype=np.float32)

            # Main matmuls for all subtiles, 1024-wide stores, no transpose interleaved.
            for s in nl.affine_range(M_SUB):
                mt = M_SUB * mblk + s
                for w in nl.affine_range(N_WIDE):
                    acc = nl.ndarray((G, par_dim(128), N_CHUNK), dtype=np.float32,
                                     buffer=nl.psum)
                    for j in nl.affine_range(G):
                        c = G * w + j
                        acc[j, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.nc_matmul(
                            lhs_t_pack[nl.arange(K)[:, None], 128 * s + nl.arange(128)[None, :]],
                            rhs_sb[nl.arange(K)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])
                    out_sb = nl.ndarray((par_dim(128), WIDE), dtype=np.float32,
                                        buffer=nl.sbuf)
                    for j in nl.affine_range(G):
                        out_sb[nl.arange(128)[:, None], N_CHUNK * j + nl.arange(N_CHUNK)[None, :]] = nl.copy(
                            acc[j, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                            dtype=np.float32)
                    nl.store(
                        out[b, 128 * mt + nl.arange(128)[:, None],
                            WIDE * w + nl.arange(WIDE)[None, :]],
                        value=out_sb[nl.arange(128)[:, None], nl.arange(WIDE)[None, :]])

    return out
