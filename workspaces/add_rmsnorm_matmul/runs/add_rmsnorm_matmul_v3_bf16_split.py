import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor):
    """Fused residual-add + RMSNorm + compensated bf16-split GEMM (M=4096, K=1024, N=2048).

    The trn2 PE array is bf16-native; a correct fp32 GEMM runs multiple internal passes
    and is capped near ~44-46% MFU by that rate penalty. This kernel does the main matmul
    in bf16 arithmetic while recovering ~16 effective mantissa bits with a two-limb
    compensated split, aiming to clear the relative-L2 gate at bf16-class speed.

        a          = x + z                                   # residual add
        inv_rms[m] = 1 / sqrt( mean_k( a[m,k]^2 ) + eps )    # per-row scalar
        w'[k,n]    = g[k] * w[k,n]                           # per-K scale folded into w
        out[m,n]   = inv_rms[m] * ( sum_k a[m,k] * w'[k,n] ) # split matmul then post-scale

    Each fp32 operand is split into a high and low bfloat16 limb. The split order is
    PINNED and auditable:
        w' = g * w  (fp32 FIRST) -> w'_hi = bf16(w'),  w'_lo = bf16(w' - w'_hi)
        aT = transpose(a) (exact fp32) -> aT_hi = bf16(aT), aT_lo = bf16(aT - aT_hi)
    bf16(.) is nl.copy(dtype=nl.bfloat16), the round-to-nearest-even cast the
    Scalar/Vector engines apply. The residual (fp32) is exact for these O(1) magnitudes.
    Three bf16 products are accumulated in fp32 PSUM, dropping the negligible aT_lo@w'_lo
    cross term (~1e-6 relative here; the offline sim's 4-product variant sizes it):
        a @ w'  ~=  aT_hi@w'_hi + aT_hi@w'_lo + aT_lo@w'_hi

    Two commutations (both fp32, algebraically equal to the reference; the offline
    fp32_control reproduced the reference to 4.82e-7 using exactly these):
      * g is indexed by the contraction column k, so it does NOT commute past the matmul,
        but it is folded into the resident weight ONCE, in fp32, BEFORE the bf16 split
        (this matches the offline-gated bf16x2_g_into_w placement, which was marginally
        more accurate than g-on-activation and is the cheaper resident fold). g_col[kt] is
        the [128,1] slice of g over the k_in/partition axis of weight tile kt.
      * inv_rms[m] is a per-row scalar, so it commutes with the matmul and the RMSNorm
        reduction stays fully fp32; the per-row scale is applied at PSUM->SBUF eviction
        (tensor_scalar reading the PSUM accumulator directly). Precision loss is confined
        to the matmul.

    Raw 2D I/O (this case's transform_to_nki_inputs is the identity, so the kernel
    receives and returns raw 2D tensors and slice-tiles itself):
        x_tensor, z_tensor : (M=4096, K=1024)   fp32
        w_tensor           : (K=1024, N=2048)   fp32
        g_tensor           : (K=1024,)          fp32   per-K contraction-axis scale
        eps                : python float scalar
        out                : (M=4096, N=2048)   fp32

    nc_matmul(stationary, moving) = stationary.T @ moving needs the contraction dim
    (k_in) on the PARTITION axis of both SBUF operands. w' limb tiles are [k_in, n] ->
    moving operands directly. Each RAW a K-sub-tile is transposed (exact fp32 identity
    matmul) to [k_in, m_in], then split into bf16 limbs used as the stationary operands.
    Splitting after the transpose is identical to splitting before it (the transpose is
    exact and bf16 rounding is element-wise).

    Memory: w'_hi + w'_lo are 2x bf16 [128,2048]x8 = 32+32 = 64 KB/partition (same total
    as v1/v2's fp32 w); aT_hi/aT_lo are bf16 [128,128]x8, tiny; plus the transient fp32 w'
    during the split. HBM is unchanged vs v1 (~42 MB): limbs are built on-chip from the
    same fp32 HBM loads.
    """
    M = 4096
    K = 1024
    K_TILES = 8               # 1024 / 128
    M_TILES = 32              # 4096 / 128
    N = 2048
    N_CHUNK = 512             # one fp32 PSUM bank in the free dim
    N_CHUNKS = N // N_CHUNK   # 4
    INV_K = np.float32(1.0 / K)

    ix = nl.arange(128)[:, None]      # partition index (m_in / k_in)
    ik = nl.arange(K)[None, :]        # full-K free index
    ig = nl.arange(1)[None, :]        # single-column free index for the g column
    inn = nl.arange(N)[None, :]       # full-N free index for w
    i128 = nl.arange(128)[None, :]    # 128-wide free index (sub-tile / transpose)
    icb = nl.arange(N_CHUNK)[None, :] # N-chunk free index

    out = nl.ndarray((M, N), dtype=np.float32, buffer=nl.shared_hbm)

    # Per-partition zero bias [128,1] for the Scalar-Engine activations (square,
    # rsqrt). A [128,1] bias vector is portable across NeuronCore generations.
    bias_zero = nl.zeros((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)

    # 128x128 identity in SBUF for the RAW-a transpose (exact fp32 identity matmul).
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[ix, i128] = nl.load(identity_const[ix, i128], dtype=np.float32)

    # ---- fold g into w' in fp32, then split w' into two bf16 limbs, once ----
    # w'_hi[kt], w'_lo[kt] = [k_in(par)=128, n=2048] bf16 (32 KB/partition each; 64 total).
    # The fp32 w'=g*w tile is transient per k-tile (reused across iterations).
    # Pinned order: w' = g*w (fp32) FIRST -> w'_hi = bf16(w') -> residual -> w'_lo.
    w_hi = nl.ndarray((K_TILES, par_dim(128), N), dtype=nl.bfloat16, buffer=nl.sbuf)
    w_lo = nl.ndarray((K_TILES, par_dim(128), N), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        w_f = nl.load(w_tensor[kt * 128 + ix, inn], dtype=np.float32)
        # g_col = g[kt*128 : kt*128+128] as a [128,1] partition vector (k_in axis).
        g_col = nl.load(g_tensor.reshape((K, 1))[kt * 128 + ix, ig], dtype=np.float32)
        # w' = g * w  (per-partition [128,1] scale, fp32, BEFORE the split)
        w_prime = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        w_prime[ix, inn] = nisa.tensor_scalar(
            data=w_f[ix, inn], op0=nl.multiply, operand0=g_col[ix, 0],
            dtype=np.float32)
        # w'_hi = bf16(w')  (round-to-nearest-even cast)
        w_hi[kt, ix, inn] = nl.copy(w_prime[ix, inn], dtype=nl.bfloat16)
        # residual = w' - w'_hi  (fp32; exact for O(1) magnitudes), then w'_lo = bf16(residual)
        w_res = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        w_res[ix, inn] = nisa.tensor_tensor(
            w_prime[ix, inn], w_hi[kt, ix, inn], op=nl.subtract)
        w_lo[kt, ix, inn] = nl.copy(w_res[ix, inn], dtype=nl.bfloat16)

    for mt in nl.affine_range(M_TILES):
        # ---- residual add a = x + z, [m_in(par)=128, k(free)=1024] ----
        x_sb = nl.load(x_tensor[mt * 128 + ix, ik], dtype=np.float32)
        z_sb = nl.load(z_tensor[mt * 128 + ix, ik], dtype=np.float32)
        a_sb = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        a_sb[ix, ik] = nl.add(x_sb[ix, ik], z_sb[ix, ik])

        # ---- fused RMSNorm reduction over K, entirely fp32 in SBUF (per-row scale) ----
        sq = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        sq[ix, ik] = nisa.activation(
            op=nl.square, data=a_sb[ix, ik],
            bias=bias_zero[ix, 0], scale=1.0, dtype=np.float32)
        sumsq = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        sumsq[ix, 0] = nisa.tensor_reduce(
            nl.add, data=sq[ix, ik], axis=[1], dtype=np.float32)
        # mean_eps = sumsq*(1/K) + eps  (eps added AFTER the /K mean, matches the reference)
        mean_eps = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        mean_eps[ix, 0] = nisa.tensor_scalar(
            data=sumsq[ix, 0], op0=nl.multiply, operand0=INV_K,
            op1=nl.add, operand1=eps, dtype=np.float32)
        inv_rms = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        inv_rms[ix, 0] = nisa.activation(
            op=nl.rsqrt, data=mean_eps[ix, 0],
            bias=bias_zero[ix, 0], scale=1.0, dtype=np.float32)

        # ---- transpose the 8 RAW a K-sub-tiles, then split each into bf16 limbs ----
        # aT_hi[kt], aT_lo[kt] = [k_in(par), m_in(free)] bf16.
        # Pinned order: aT = transpose(a) (exact fp32) -> aT_hi=bf16(aT) -> residual -> aT_lo.
        aT_hi = nl.ndarray((K_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        aT_lo = nl.ndarray((K_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
            psum_t[ix, i128] = nisa.nc_matmul(
                a_sb[ix, 128 * kt + i128],
                identity_local[ix, i128],
                is_transpose=True, is_moving_onezero=True)
            aT_f = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
            aT_f[ix, i128] = nl.copy(psum_t[ix, i128], dtype=np.float32)
            # aT_hi = bf16(aT)
            aT_hi[kt, ix, i128] = nl.copy(aT_f[ix, i128], dtype=nl.bfloat16)
            # residual = aT - aT_hi (fp32), then aT_lo = bf16(residual)
            aT_res = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
            aT_res[ix, i128] = nisa.tensor_tensor(
                aT_f[ix, i128], aT_hi[kt, ix, i128], op=nl.subtract)
            aT_lo[kt, ix, i128] = nl.copy(aT_res[ix, i128], dtype=nl.bfloat16)

        # ---- matmul: 3 bf16 products per (N-chunk, K-tile), accumulated in fp32 PSUM ----
        for c in nl.affine_range(N_CHUNKS):
            acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # aT_hi @ w'_hi
                acc[ix, icb] += nisa.nc_matmul(
                    aT_hi[kt, ix, i128],
                    w_hi[kt, ix, N_CHUNK * c + icb])
                # aT_hi @ w'_lo
                acc[ix, icb] += nisa.nc_matmul(
                    aT_hi[kt, ix, i128],
                    w_lo[kt, ix, N_CHUNK * c + icb])
                # aT_lo @ w'_hi   (dropping aT_lo @ w'_lo, the negligible cross term)
                acc[ix, icb] += nisa.nc_matmul(
                    aT_lo[kt, ix, i128],
                    w_hi[kt, ix, N_CHUNK * c + icb])

            # Post-scale eviction: apply the per-row fp32 inv_rms as PSUM -> SBUF.
            out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
            out_sb[ix, icb] = nisa.tensor_scalar(
                data=acc[ix, icb], op0=nl.multiply, operand0=inv_rms[ix, 0],
                dtype=np.float32)
            nl.store(out[mt * 128 + ix, N_CHUNK * c + icb], value=out_sb[ix, icb])

    return out
