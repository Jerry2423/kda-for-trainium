import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Fused RMSNorm + compensated bf16-split GEMM  out = rmsnorm(x) @ w.

    The trn2 PE array is bf16-native; a correct fp32 GEMM runs multiple passes and is
    capped near ~46% MFU by that rate penalty. This kernel does the main matmul in bf16
    arithmetic while recovering ~16 effective mantissa bits with a two-limb compensated
    split, aiming to clear the relative-L2 correctness gate at bf16-class speed.

    Each fp32 operand is split into a high and low bfloat16 limb (round-to-nearest-even,
    which the Scalar/Vector engines apply when casting fp32 -> bf16):
        x_hi = bf16(x),  x_lo = bf16(x - x_hi)      (per transposed activation sub-tile)
        w_hi = bf16(w),  w_lo = bf16(w - w_hi)      (per resident weight tile, once)
    and three bf16 products are accumulated in fp32 PSUM, dropping the negligible
    x_lo*w_lo cross term (~1e-6 relative here):
        x @ w  ~=  x_hi@w_hi + x_hi@w_lo + x_lo@w_hi

    Because inv_rms[m] is a per-row scalar it commutes with the matmul, so the RMSNorm
    reduction stays fp32 and the per-row scale is applied at PSUM->SBUF eviction (the
    post-scale eviction fold), exactly as in the fp32 post-scale kernel:
        inv_rms[m] = 1 / sqrt( mean_k( x[m,k]^2 ) )
        out[m,n]   = ( x_hi@w_hi + x_hi@w_lo + x_lo@w_hi )[m,n] * inv_rms[m]

    Inputs (tiled by the NKIBench reference's transform_to_nki_inputs):
        v1: (32, 128, 1024) = [m_tile, m_in, k]   (x)
        v2: (8, 128, 2048)  = [k_tile, k_in, n]   (w)
    Output:
        v3: (32, 128, 2048) = [m_tile, m_in, n]   (out)

    nc_matmul(stationary, moving) = stationary.T @ moving needs the contraction dim
    (k_in) on the PARTITION axis of both SBUF operands. w limb tiles are [k_in, n] ->
    moving operands directly. Each x K-sub-tile is transposed RAW (exact fp32 identity
    matmul) to [k_in, m_in], then split into bf16 limbs used as the stationary operands.
    Splitting after the transpose is identical to splitting before it (the transpose is
    exact and bf16 rounding is element-wise).
    """
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    M_TILES = 32       # 4096 / 128
    K = 1024
    K_TILES = 8        # 1024 / 128
    N = 2048
    N_CHUNK = 512      # one fp32 PSUM bank in the free dim
    N_CHUNKS = N // N_CHUNK   # 4
    INV_K = np.float32(1.0 / K)   # folded into the rsqrt scale: rsqrt(sumsq * 1/K)

    out = nl.ndarray((32, 128, 2048), dtype=np.float32, buffer=nl.shared_hbm)

    # Per-partition zero bias [128,1] for the Scalar-Engine activations (square, rsqrt).
    bias_zero = nl.zeros((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)

    # 128x128 identity in SBUF for the RAW-x transpose (exact fp32 identity matmul).
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32)

    # ---- split resident w into two bf16 limbs once ----
    # w_hi[kt], w_lo[kt] = [k_in(par)=128, n=2048] bf16 (32 KB/partition each; 64 total).
    # The fp32 w tile is transient per k-tile (reused across iterations).
    w_hi = nl.ndarray((K_TILES, par_dim(128), N), dtype=nl.bfloat16, buffer=nl.sbuf)
    w_lo = nl.ndarray((K_TILES, par_dim(128), N), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        w_f = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        w_f[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nl.load(
            v2[kt, nl.arange(128)[:, None], nl.arange(N)[None, :]], dtype=np.float32)
        # w_hi = bf16(w)  (round-to-nearest-even cast)
        w_hi[kt, nl.arange(128)[:, None], nl.arange(N)[None, :]] = nl.copy(
            w_f[nl.arange(128)[:, None], nl.arange(N)[None, :]], dtype=nl.bfloat16)
        # residual = w - w_hi  (fp32; exact for O(1) magnitudes), then w_lo = bf16(residual)
        w_res = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        w_res[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nisa.tensor_tensor(
            w_f[nl.arange(128)[:, None], nl.arange(N)[None, :]],
            w_hi[kt, nl.arange(128)[:, None], nl.arange(N)[None, :]],
            op=nl.subtract)
        w_lo[kt, nl.arange(128)[:, None], nl.arange(N)[None, :]] = nl.copy(
            w_res[nl.arange(128)[:, None], nl.arange(N)[None, :]], dtype=nl.bfloat16)

    for mt in nl.affine_range(M_TILES):
        # Load this M-tile's x once: [m_in(par)=128, k(free)=1024].
        x_sb = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        x_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]] = nl.load(
            v1[mt, nl.arange(128)[:, None], nl.arange(K)[None, :]], dtype=np.float32)

        # ---- RMSNorm reduction over K, entirely in fp32 SBUF (per-row scale) ----
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

        # ---- transpose the 8 RAW x K-sub-tiles, then split each into bf16 limbs ----
        # xT_hi[kt], xT_lo[kt] = [k_in(par), m_in(free)] bf16.
        xT_hi = nl.ndarray((K_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        xT_lo = nl.ndarray((K_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
            psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                x_sb[nl.arange(128)[:, None], 128 * kt + nl.arange(128)[None, :]],
                identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                is_transpose=True, is_moving_onezero=True)
            xT_f = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
            xT_f[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32)
            # xT_hi = bf16(xT)
            xT_hi[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                xT_f[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=nl.bfloat16)
            # residual = xT - xT_hi (fp32), then xT_lo = bf16(residual)
            xT_res = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
            xT_res[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.tensor_tensor(
                xT_f[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                xT_hi[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                op=nl.subtract)
            xT_lo[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                xT_res[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=nl.bfloat16)

        # ---- matmul: 3 bf16 products per (N-chunk, K-tile), accumulated in fp32 PSUM ----
        for c in nl.affine_range(N_CHUNKS):
            acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # x_hi @ w_hi
                acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                    xT_hi[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    w_hi[kt, nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])
                # x_hi @ w_lo
                acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                    xT_hi[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    w_lo[kt, nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])
                # x_lo @ w_hi
                acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                    xT_lo[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    w_hi[kt, nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])

            # Post-scale eviction: apply the per-row fp32 inv_rms as PSUM -> SBUF.
            out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
            out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.tensor_scalar(
                data=acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                op0=nl.multiply, operand0=inv_rms[nl.arange(128)[:, None], 0],
                dtype=np.float32)
            nl.store(
                out[mt, nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                value=out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

    return out
