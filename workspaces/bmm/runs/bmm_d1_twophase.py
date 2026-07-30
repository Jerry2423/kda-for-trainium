import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Batched fp32 matmul out[b] = lhs[b] @ rhs[b] via transpose-then-matmul separation.

    Local reschedules of v1 (multi-bank PSUM, issue-before-drain, wide stores) were
    all measured to compile to v1's exact instruction stream. The NKIBench baseline,
    which runs the identical main matmuls at ~1.70x lower per-instruction stall, has
    a different LOOP STRUCTURE: it separates the lhs transpose from the main matmul
    by transposing all m-subtiles up front, so the serial transpose->copy at the head of each
    m-tile's matmul burst is lifted OUT of the matmul stream.

    This candidate mirrors that. The 32 m-tiles are grouped into blocks of 8:
      - transpose all 8 lhs subtiles of the block, packing the transposed tiles
               into one resident [k=64, 8*128=1024] SBUF buffer.
      - run all main matmuls for the 8 subtiles (each still a single-pass K=64
               nc_matmul), coalescing the two 512-wide PSUM halves into one 1024-wide
               store, with NO transpose interleaved.

    rhs[b] stays fully resident [64,4096] (loaded once per batch, v1's minimal-read
    layout). Pure fp32; correctness identical to v1 (each output tile is a single '='
    write of one K=64 nc_matmul; no K-accumulation reorder; transpose-before-use is
    exact, so packing 8 tiles then matmul == transpose-then-matmul per tile).

    Bank budget per m-block: 1 transpose PSUM bank (drained) then 2 output PSUM banks
    = at most 2 live output banks + the drained transpose bank <= 8 physical.
    Packed transpose buffer: 64 part x 1024 x 4B = 4 KB/partition; rhs 16 KB/partition.
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
    M_SUB = 8                       # m-subtiles per m-block
    M_BLOCKS = M_TILES // M_SUB     # 4
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
