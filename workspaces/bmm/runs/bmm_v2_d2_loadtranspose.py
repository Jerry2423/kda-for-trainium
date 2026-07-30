import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Batched fp32 matmul out[b] = lhs[b] @ rhs[b] via a transposed HBM load.

    Same transpose-then-matmul structure as bmm_v2 (whole batch transposed up front
    into a resident pack, then all main matmuls), but the lhs transpose is produced by
    a direct transposed HBM load (nl.load_transpose2d) instead of the identity-matmul
    on the tensor engine. v1[b, 128*mt:+128, 0:64] is [m_in=128, k=64]; the transposed
    load yields [k=64, m_in=128] directly, removing the identity-transpose matmuls and
    the transpose PSUM bank + its copy.

    Since the identity-transpose in bmm_v2 is already hidden behind the matmul stream
    (the kernel is limited by the output write bandwidth, not the tensor engine),
    removing it is expected to be latency-neutral; bmm_v2's identity-transpose is the
    fallback if this does not lower latency or raises HBM traffic.

    Pure fp32; correctness identical to bmm_v2 (a silent axis swap in the transposed
    load would fail the L2 gate). rhs[b] resident [64,4096]; single-pass K=64 main
    matmul; 1024-wide coalesced stores.
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
    M_SUB = 32                      # whole batch is one block
    M_BLOCKS = M_TILES // M_SUB     # 1
    N_CHUNK = 512
    WIDE = 1024
    N_WIDE = N // WIDE              # 4
    G = WIDE // N_CHUNK             # 2

    out = nl.ndarray((B, M, N), dtype=np.float32, buffer=nl.shared_hbm)

    for b in nl.affine_range(B):
        rhs_sb = nl.ndarray((par_dim(K), N), dtype=np.float32, buffer=nl.sbuf)
        rhs_sb[nl.arange(K)[:, None], nl.arange(N)[None, :]] = nl.load(
            v2[b, nl.arange(K)[:, None], nl.arange(N)[None, :]], dtype=np.float32)

        for mblk in nl.affine_range(M_BLOCKS):
            # Transposed HBM load of all M_SUB subtiles, packed into [k=64, M_SUB*128].
            lhs_t_pack = nl.ndarray((par_dim(K), M_SUB * 128), dtype=np.float32,
                                    buffer=nl.sbuf)
            for s in nl.affine_range(M_SUB):
                mt = M_SUB * mblk + s
                # v1[b, 128*mt:+128, 0:64] = [m_in=128, k=64] -> transposed load [k=64, m_in=128].
                lhs_t_pack[nl.arange(K)[:, None], 128 * s + nl.arange(128)[None, :]] = nl.load_transpose2d(
                    v1[b, 128 * mt + nl.arange(128)[:, None], nl.arange(K)[None, :]],
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
