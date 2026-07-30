import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """fp32 dense GEMM out = lhs @ rhs (M=4096, K=5120, N=12288), M-blocked.

    Same tiled contract as the phase-1 kernel:
        v1 (32,128,40,128)=[m_tile,m_in,k_tile,k_in] lhs;
        v2 (40,128,12288)=[k_tile,k_in,n] rhs; out (32,128,12288)=[m_tile,m_in,n].

    Optimization vs phase-1 (matmul_v1): process a BLOCK of B M-tiles together so
    each rhs K-tile [k_in,512] is loaded from HBM ONCE and reused across the B
    stationary lhsT tiles, instead of being re-read for every M-tile. This cuts
    rhs HBM traffic ~B-fold (phase-1 re-read rhs 32x = ~8GB; here 32/B x). The B
    output-row tiles accumulate into B DISTINCT [128,512] fp32 PSUM banks and store
    to B DISTINCT out row-blocks out[mblock*B+mb, :, n0:n0+512].

    nc_matmul(stationary, moving)=stationary.T @ moving; contraction k_in on the
    partition axis of both (both in SBUF). lhs tile [m_in,k_in] is transposed to
    [k_in,m_in] via the identity idiom before use as stationary; rhs [k_in,512] is
    the moving operand directly.
    """
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    M_TILES = 32
    K_TILES = 40
    N = 12288
    N_CHUNK = 512
    N_CHUNKS = N // N_CHUNK      # 24
    B = 8                        # M-block factor (M_TILES % B must be 0)
    M_BLOCKS = M_TILES // B      # number of M-blocks (M_TILES/B)

    out = nl.ndarray((32, 128, 12288), dtype=np.float32, buffer=nl.shared_hbm)

    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]],
        dtype=np.float32)

    for mblk in nl.affine_range(M_BLOCKS):
        # Transposed lhs for all B members of this M-block:
        #   lhs_t[mb, kt] = [k_in(par)=128, m_in(free)=128]
        # (mb, kt) are leading index dims; par_dim on the partition axis.
        lhs_t = nl.ndarray((B, K_TILES, par_dim(128), 128), dtype=np.float32,
                           buffer=nl.sbuf)
        for mb in nl.affine_range(B):
            for kt in nl.affine_range(K_TILES):
                lhs_sb = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                    buffer=nl.sbuf)
                lhs_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
                    v1[mblk * B + mb, nl.arange(128)[:, None], kt,
                       nl.arange(128)[None, :]],
                    dtype=np.float32)
                psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                    buffer=nl.psum)
                psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                    lhs_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    is_transpose=True, is_moving_onezero=True)
                lhs_t[mb, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                    psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    dtype=np.float32)

        for c in nl.affine_range(N_CHUNKS):
            # B distinct PSUM accumulators, one per block member.
            acc = nl.zeros((B, par_dim(128), N_CHUNK), dtype=np.float32,
                           buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # Load this rhs K-tile ONCE, reuse across all B block members.
                rhs_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                    buffer=nl.sbuf)
                rhs_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.load(
                    v2[kt, nl.arange(128)[:, None],
                       N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)
                for mb in nl.affine_range(B):
                    acc[mb, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                        lhs_t[mb, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        rhs_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

            for mb in nl.affine_range(B):
                out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                    buffer=nl.sbuf)
                out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.copy(
                    acc[mb, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)
                # Distinct out row-block per member: [m_in(par), n(free)] ->
                # out[mblk*B+mb, :, n0:n0+512], matching out[mt,mi,n]=lhs@rhs.
                nl.store(
                    out[mblk * B + mb, nl.arange(128)[:, None],
                        N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                    value=out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

    return out
