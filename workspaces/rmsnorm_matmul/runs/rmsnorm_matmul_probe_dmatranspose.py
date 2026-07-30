import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """fp32 rmsnorm_matmul that transposes x on the DMA Engine (in-place call form).

    Identical to the post-scale rmsnorm_matmul kernel except the raw-x K-sub-tile
    transpose ([m_in,128] -> [k_in,128]) is done with nisa.dma_transpose(dst, src,
    axes=(1,0)) on the DMA Engine instead of the identity nc_matmul on the PE. The API
    docs state every dma_transpose lowering path (hardware-DGE and software-DGE) requires
    a 2-byte dtype, so a 4-byte fp32 transpose is not expected to lower on this target.
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

        # ---- transpose raw x K-sub-tiles via DMA Engine (fp32) ----
        xT = nl.ndarray((K_TILES, par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            nisa.dma_transpose(
                xT[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                x_sb[nl.arange(128)[:, None], 128 * kt + nl.arange(128)[None, :]],
                axes=(1, 0))

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
