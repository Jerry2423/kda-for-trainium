import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor):
    """Fused compensated bf16-split GEMM + residual-add + RMSNorm (M=4096, K=2048, N=2048).

        y[m,n]     = sum_k x[m,k] * w[k,n]  +  z[m,n]       # split GEMM then residual add
        inv_rms[m] = 1 / sqrt( mean_n( y[m,n]^2 ) + eps )   # per-row scalar, over N
        out[m,n]   = y[m,n] * inv_rms[m] * g[n]             # norm + per-N scale

    This is the promoted bf16x2 3-product kernel with ONLY the main GEMM loop nest reordered
    for stationary-operand (weight-load) reuse. The arithmetic, the two-limb compensated
    split, the residual add, the RMSNorm, eps/rsqrt, and the output scale are all byte-for-byte
    the promoted kernel -- only the ORDER in which the same bf16 matmuls are issued changes.

    Why reorder. nc_matmul(stationary, moving) loads the stationary operand into the PE array;
    consecutive matmuls that reuse the SAME stationary skip that reload. Here the transposed
    activation limbs xT_hi[kt] / xT_lo[kt] ([k_in=128, m_in=128], free=128) are the stationary
    operand and the weight limbs w_hi[kt,c] / w_lo[kt,c] ([k_in=128, n=512], free=512) stream.
    The promoted kernel is N-chunk-outer -- it fully accumulates one [128,512] chunk over all
    16 K-tiles before the next chunk -- so each stationary limb is reloaded once per chunk
    (4x per M-tile) with an in-chunk reuse run of only 2 (P1->P2 share xT_hi[kt], then P3
    changes to xT_lo[kt]).

    This kernel is K-tile-outer with 4 live [128,512] fp32 PSUM accumulators (one per N-chunk),
    grouped by shared stationary limb. For each K-tile kt:
      * hi-pass: xT_hi[kt] stays stationary for 8 consecutive matmuls -- P1 (xT_hi@w_hi) and
        P2 (xT_hi@w_lo) across all 4 chunks;
      * lo-pass: xT_lo[kt] stays stationary for 4 consecutive matmuls -- P3 (xT_lo@w_hi)
        across all 4 chunks.
    So the stationary-reuse run grows 2 -> 8 (xT_hi) and -> 4 (xT_lo), cutting stationary
    loads from 128 to 32 per M-tile. This op's K=2048 (16 K-tiles) doubles the accumulation
    depth over which the reuse can be grouped vs the K=1024 siblings.

    Per-bank accumulation is UNCHANGED. Because the kt loop is outer and each kt runs its
    hi-pass then lo-pass before advancing, every individual accumulator acc[c] still receives
    its products in the exact order P1_kt, P2_kt, P3_kt for kt = 0..15 -- identical to the
    N-chunk-outer kernel's per-chunk order. Only the cross-bank issue interleaving (the
    stationary-reuse lever) differs, so the fp32 sum reduced into each bank is the same one,
    in the same order, and rel-L2 is expected to match the promoted kernel (1.544749e-5).

    The two-limb split (from the promoted kernel, unchanged):
        w  (fp32) -> w_hi  = bf16(w),   w_lo  = bf16(w  - w_hi)     # once, at load, resident
        xT = transpose(x) (exact fp32)
                  -> xT_hi = bf16(xT),  xT_lo = bf16(xT - xT_hi)    # per transposed sub-tile
    bf16(.) is nl.copy(dtype=nl.bfloat16), round-to-nearest-even; the fp32 residual subtract is
    exact for these O(1) magnitudes. Three bf16 products are accumulated in fp32 PSUM in the
    FIXED order hi@hi, hi@lo, lo@hi, dropping the negligible xT_lo@w_lo cross term:
        x @ w  ~=  xT_hi@w_hi + xT_hi@w_lo + xT_lo@w_hi

    Everything OUTSIDE the matmul stays byte-for-byte fp32: the residual add y = x@w + z, the
    RMSNorm reduction over N, eps/rsqrt, and the output scale. g is on the OUTPUT free axis (N),
    applied AFTER the norm as a [1,N]->[128,N] broadcast multiply -- never folded into w (that
    would scale y before the norm and break rms = sqrt(mean(y^2))). inv_rms is not commuted out:
    the norm reduces over N, so the full [128,N] row is assembled before inv_rms is known.

    Raw 2D I/O (this case's transform_to_nki_inputs is the identity):
        x_tensor : (M=4096, K=2048)   fp32
        w_tensor : (K=2048, N=2048)   fp32
        z_tensor : (M=4096, N=2048)   fp32   residual, added to the GEMM output
        g_tensor : (N=2048,)          fp32   per-N output-axis scale
        eps      : python float scalar
        out      : (M=4096, N=2048)   fp32

    PSUM: 4 live [128,512] fp32 accumulator banks (512 <= 2048 elem/bank) + 1 [128,128]
    transpose bank at the source level; the actual allocation/liveness is compiler-decided.
    HBM is unchanged (~84 MB read / 34 MB write): the limbs are built on-chip and the matmuls
    are only reordered, not added.
    """
    M = 4096
    K = 2048
    K_TILES = 16              # 2048 / 128
    M_TILES = 32              # 4096 / 128
    N = 2048
    N_CHUNK = 512             # one fp32 PSUM bank in the free dim
    N_CHUNKS = N // N_CHUNK   # 4
    INV_N = np.float32(1.0 / N)

    ix = nl.arange(128)[:, None]      # partition index (m_in / k_in)
    ik = nl.arange(K)[None, :]        # full-K free index
    iw = nl.arange(1)[:, None]        # single-row partition index for g
    inn = nl.arange(N)[None, :]       # full-N free index for w / y
    i128 = nl.arange(128)[None, :]    # 128-wide free index (sub-tile / transpose)
    icb = nl.arange(N_CHUNK)[None, :] # N-chunk free index

    out = nl.ndarray((M, N), dtype=np.float32, buffer=nl.shared_hbm)

    # Per-partition zero bias [128,1] for the Scalar-Engine activations (square,
    # rsqrt). A [128,1] bias vector is portable across NeuronCore generations
    # (a scalar bias is only accepted from NeuronCore-v3+).
    bias_zero = nl.zeros((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)

    # g as a [1,N] row, broadcast to [128,N] along the partition axis at use. g is on
    # the output (free) axis here, so it is a plain broadcast multiply -- never folded
    # into w.
    g_tile = nl.load(g_tensor.reshape((1, N))[iw, inn], dtype=np.float32)

    # 128x128 identity in SBUF, used as the moving operand to transpose the x
    # sub-tiles on the Tensor Engine. Loaded once, reused for all tiles.
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[ix, i128] = nl.load(identity_const[ix, i128], dtype=np.float32)

    # ---- split w into two bf16 limbs, once, fully resident ----
    # w_hi[kt], w_lo[kt] = [k_in(par)=128, n=2048] bf16 (32 KB/partition each; 64 total).
    # Pinned order: w (fp32) FIRST -> w_hi = bf16(w) -> residual = w - w_hi (fp32) -> w_lo.
    # The fp32 w tile is transient per k-tile (reused across iterations).
    w_hi = nl.ndarray((K_TILES, par_dim(128), N), dtype=nl.bfloat16, buffer=nl.sbuf)
    w_lo = nl.ndarray((K_TILES, par_dim(128), N), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        w_f = nl.load(w_tensor[kt * 128 + ix, inn], dtype=np.float32)
        # w_hi = bf16(w)  (round-to-nearest-even cast)
        w_hi[kt, ix, inn] = nl.copy(w_f[ix, inn], dtype=nl.bfloat16)
        # residual = w - w_hi  (fp32; exact for O(1) magnitudes), then w_lo = bf16(residual)
        w_res = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        w_res[ix, inn] = nisa.tensor_tensor(
            w_f[ix, inn], w_hi[kt, ix, inn], op=nl.subtract)
        w_lo[kt, ix, inn] = nl.copy(w_res[ix, inn], dtype=nl.bfloat16)

    for mt in nl.affine_range(M_TILES):
        # ---- load this M-tile of x, [m_in(par)=128, k(free)=2048] ----
        x_sb = nl.load(x_tensor[mt * 128 + ix, ik], dtype=np.float32)

        # ---- transpose the 16 x K-sub-tiles, then split each into bf16 limbs ----
        # xT_hi[kt], xT_lo[kt] = [k_in(par), m_in(free)] bf16.
        # Pinned order: xT = transpose(x) (exact fp32) -> xT_hi=bf16(xT) -> residual -> xT_lo.
        xT_hi = nl.ndarray((K_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        xT_lo = nl.ndarray((K_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
            psum_t[ix, i128] = nisa.nc_matmul(
                x_sb[ix, 128 * kt + i128],
                identity_local[ix, i128],
                is_transpose=True, is_moving_onezero=True)
            xT_f = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
            xT_f[ix, i128] = nl.copy(psum_t[ix, i128], dtype=np.float32)
            # xT_hi = bf16(xT)
            xT_hi[kt, ix, i128] = nl.copy(xT_f[ix, i128], dtype=nl.bfloat16)
            # residual = xT - xT_hi (fp32), then xT_lo = bf16(residual)
            xT_res = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
            xT_res[ix, i128] = nisa.tensor_tensor(
                xT_f[ix, i128], xT_hi[kt, ix, i128], op=nl.subtract)
            xT_lo[kt, ix, i128] = nl.copy(xT_res[ix, i128], dtype=nl.bfloat16)

        # ---- GEMM (3 bf16 products), K-tile-outer with 4 live PSUM accumulators ----
        # 4 [128,512] fp32 PSUM banks (one per N-chunk), zeroed ONCE per M-tile before the
        # K-tile loop and never reused across M-tiles. For each K-tile we run the hi-pass
        # (xT_hi[kt] stationary for P1+P2 over all 4 chunks) then the lo-pass (xT_lo[kt]
        # stationary for P3 over all 4 chunks), so the stationary limb is reused across
        # 8 and 4 consecutive matmuls respectively. Each bank still accumulates its products
        # in the exact order P1_kt, P2_kt, P3_kt across kt = 0..15 (per-bank order identical
        # to the N-chunk-outer kernel; only the cross-bank issue order differs).
        y = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        acc = nl.zeros((N_CHUNKS, par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
        for kt in nl.affine_range(K_TILES):
            # hi-pass: xT_hi[kt] stationary for 8 consecutive matmuls (P1, P2 over 4 chunks)
            for c in nl.affine_range(N_CHUNKS):
                # xT_hi @ w_hi
                acc[c, ix, icb] += nisa.nc_matmul(
                    xT_hi[kt, ix, i128],
                    w_hi[kt, ix, N_CHUNK * c + icb])
                # xT_hi @ w_lo
                acc[c, ix, icb] += nisa.nc_matmul(
                    xT_hi[kt, ix, i128],
                    w_lo[kt, ix, N_CHUNK * c + icb])
            # lo-pass: xT_lo[kt] stationary for 4 consecutive matmuls (P3 over 4 chunks)
            for c in nl.affine_range(N_CHUNKS):
                # xT_lo @ w_hi   (dropping xT_lo @ w_lo, the negligible cross term)
                acc[c, ix, icb] += nisa.nc_matmul(
                    xT_lo[kt, ix, i128],
                    w_hi[kt, ix, N_CHUNK * c + icb])
        # ---- residual add + eviction: y[:,chunk] = acc[chunk] + z_tile ----
        for c in nl.affine_range(N_CHUNKS):
            z_tile = nl.load(z_tensor[mt * 128 + ix, N_CHUNK * c + icb], dtype=np.float32)
            y[ix, N_CHUNK * c + icb] = nl.add(acc[c, ix, icb], z_tile[ix, icb])

        # ---- single fused RMSNorm over N (free axis), non-clobbering (byte-for-byte v1) ----
        # sq = y^2  (Scalar Engine, fp32) -- a SEPARATE temp so y stays live for the
        # output scale below.
        sq = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        sq[ix, inn] = nisa.activation(
            op=nl.square, data=y[ix, inn],
            bias=bias_zero[ix, 0], scale=1.0, dtype=np.float32)
        # sumsq = sum_n(y^2)  (single full-N free-axis reduce, Vector Engine)
        sumsq = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        sumsq[ix, 0] = nisa.tensor_reduce(
            nl.add, data=sq[ix, inn], axis=[1], dtype=np.float32)
        # mean_eps = sumsq*(1/N) + eps  == mean_n(y^2) + eps
        # (eps is added AFTER the /N mean, matching the reference; it is NOT scaled by 1/N.)
        mean_eps = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        mean_eps[ix, 0] = nisa.tensor_scalar(
            data=sumsq[ix, 0], op0=nl.multiply, operand0=INV_N,
            op1=nl.add, operand1=eps, dtype=np.float32)
        # inv_rms = 1 / sqrt(mean_eps)
        inv_rms = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        inv_rms[ix, 0] = nisa.activation(
            op=nl.rsqrt, data=mean_eps[ix, 0],
            bias=bias_zero[ix, 0], scale=1.0, dtype=np.float32)

        # ---- output scale + store: out = y * inv_rms * g (reads the still-live y) ----
        # Full-width [128,N] elementwise pass with NO PSUM-bank constraint (measured ~6.4%
        # faster than a per-512-chunk store; the 512-chunking is a PSUM constraint for the
        # GEMM accumulator only, not this pure-SBUF epilogue). Byte-for-byte v1.
        out_sb = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        # y * inv_rms  (per-row [128,1] scale broadcast across the free axis)
        out_sb[ix, inn] = nisa.tensor_scalar(
            data=y[ix, inn], op0=nl.multiply, operand0=inv_rms[ix, 0],
            dtype=np.float32)
        # * g  (per-N free-axis scale; g broadcast [1,N] -> [128,N])
        out_sb[ix, inn] = nl.multiply(out_sb[ix, inn], g_tile.broadcast_to((128, N)))
        nl.store(out[mt * 128 + ix, inn], value=out_sb[ix, inn])

    return out
