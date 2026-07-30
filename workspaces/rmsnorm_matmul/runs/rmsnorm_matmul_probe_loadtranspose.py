import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """fp32 rmsnorm_matmul that produces the transposed x via a direct HBM load.

    Loads x's [m_in,128] K-sub-tile directly from HBM already transposed to [k_in,m_in]
    with nki.language.load_transpose2d, so no separate transpose instruction runs on the
    PE, the Vector engine, or the DMA-transpose path. The norm still reads the normal x
    load ([m_in,k]) for the free-axis reduce (that load is unchanged); the only change vs
    the post-scale eviction-fold kernel is how xT is produced. load_transpose2d is
    documented experimental, so it may not lower on an older remote target.
    """
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    M_TILES = 32
    K = 1024
    K_TILES = 8
    N = 2048
    N_CHUNK = 512
    N_CHUNKS = N // N_CHUNK
    INV_K = np.float32(1.0 / K)

    out = nl.ndarray((32, 128, 2048), dtype=np.float32, buffer=nl.shared_hbm)

    bias_zero = nl.zeros((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)

    w_sb = nl.ndarray((K_TILES, par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        w_sb[kt, nl.arange(128)[:, None], nl.arange(N)[None, :]] = nl.load(
            v2[kt, nl.arange(128)[:, None], nl.arange(N)[None, :]], dtype=np.float32)

    for mt in nl.affine_range(M_TILES):
        # Normal x load [m_in(par), k(free)] — used only for the norm's free-axis reduce.
        x_sb = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        x_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]] = nl.load(
            v1[mt, nl.arange(128)[:, None], nl.arange(K)[None, :]], dtype=np.float32)

        sq = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        sq[nl.arange(128)[:, None], nl.arange(K)[None, :]] = nisa.activation(
            op=nl.square, data=x_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]],
            bias=bias_zero[nl.arange(128)[:, None], 0], scale=1.0, dtype=np.float32)
        sumsq = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        sumsq[nl.arange(128)[:, None], 0] = nisa.tensor_reduce(
            nl.add, data=sq[nl.arange(128)[:, None], nl.arange(K)[None, :]],
            axis=[1], dtype=np.float32)
        inv_rms = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        inv_rms[nl.arange(128)[:, None], 0] = nisa.activation(
            op=nl.rsqrt, data=sumsq[nl.arange(128)[:, None], 0],
            bias=bias_zero[nl.arange(128)[:, None], 0], scale=INV_K, dtype=np.float32)

        # ---- load each raw x K-sub-tile from HBM already transposed ----
        # v1[mt, :, 128*kt:128*(kt+1)] is [m_in, k_in]; transposed load -> [k_in, m_in].
        xT = nl.ndarray((K_TILES, par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            xT[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load_transpose2d(
                v1[mt, nl.arange(128)[:, None], 128 * kt + nl.arange(128)[None, :]],
                dtype=np.float32)

        for c in nl.affine_range(N_CHUNKS):
            acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                    xT[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    w_sb[kt, nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])

            out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
            out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.tensor_scalar(
                data=acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                op0=nl.multiply, operand0=inv_rms[nl.arange(128)[:, None], 0],
                dtype=np.float32)
            nl.store(
                out[mt, nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                value=out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

    return out
