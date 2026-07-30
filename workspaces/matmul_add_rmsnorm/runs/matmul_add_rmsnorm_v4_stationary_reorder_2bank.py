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

    2-BANK diagnostic variant of the stationary-reuse GEMM reorder. Same arithmetic and same
    two-limb compensated split as the promoted kernel; ONLY the GEMM loop nest differs, and this
    one holds HALF the live PSUM set of the 4-bank reorder.

    The 4-bank reorder (matmul_add_rmsnorm_v3_stationary_reorder.py) grouped the stationary reuse
    over ALL 4 N-chunks at once, holding 4 live [128,512] fp32 PSUM accumulators across the whole
    K-tile loop. It measured a small PE-idle rise (PE 91.7% -> 90.3%, TRUE PE-active +0.3%) with
    every regression sentinel flat (matmul_instruction_count / psum_read / HBM all ==) -- a
    bit-exact pure reschedule that the compiler dislikes, consistent with the enlarged-live-set
    pressure that constrains the affine_range software pipeline (the batched-matmul precedent).

    This variant groups the stationary reuse over 2 chunks at a time (an outer chunk-PAIR loop
    with the K-tile loop inside), holding only 2 live [128,512] accumulator banks + 1 [128,128]
    transpose bank = 3 banks. It keeps a stationary-reuse run of 4 (hi-pass: xT_hi[kt] across 2
    chunks = 4 matmuls; lo-pass: xT_lo[kt] across 2 chunks = 2 matmuls) while halving the live
    PSUM set. It is a diagnostic: if the 4-bank PE-idle rise was the enlarged live set, this
    recovers toward the promoted kernel; if it was the reorder itself, this also fails to help.

    Per-bank accumulation order is UNCHANGED (bit-exact expectation, like the 4-bank variant):
    within each chunk pair the kt loop is outer and each kt runs its hi-pass then lo-pass before
    advancing, so every accumulator acc[j] still receives P1_0,P2_0,P3_0,...,P1_15,P2_15,P3_15 --
    identical to the promoted kernel's per-chunk order. Only the cross-bank issue interleaving
    (over 2 chunks instead of 4) differs.

    The two-limb split (from the promoted kernel, unchanged):
        w  (fp32) -> w_hi  = bf16(w),   w_lo  = bf16(w  - w_hi)     # once, at load, resident
        xT = transpose(x) (exact fp32)
                  -> xT_hi = bf16(xT),  xT_lo = bf16(xT - xT_hi)    # per transposed sub-tile
    Three bf16 products accumulated in fp32 PSUM in the FIXED order hi@hi, hi@lo, lo@hi, dropping
    the negligible xT_lo@w_lo cross term:
        x @ w  ~=  xT_hi@w_hi + xT_hi@w_lo + xT_lo@w_hi

    Everything OUTSIDE the matmul stays byte-for-byte fp32: residual add y = x@w + z, RMSNorm
    over N, eps/rsqrt, output scale. g on the OUTPUT free axis (never folded into w); inv_rms not
    commuted out.

    Raw 2D I/O (this case's transform_to_nki_inputs is the identity):
        x_tensor : (M=4096, K=2048)   fp32
        w_tensor : (K=2048, N=2048)   fp32
        z_tensor : (M=4096, N=2048)   fp32   residual
        g_tensor : (N=2048,)          fp32   per-N output scale
        eps      : python float scalar
        out      : (M=4096, N=2048)   fp32
    """
    M = 4096
    K = 2048
    K_TILES = 16              # 2048 / 128
    M_TILES = 32              # 4096 / 128
    N = 2048
    N_CHUNK = 512             # one fp32 PSUM bank in the free dim
    N_CHUNKS = N // N_CHUNK   # 4
    CHUNKS_PER_PAIR = 2       # 2 live banks per chunk-pair
    N_PAIRS = N_CHUNKS // CHUNKS_PER_PAIR  # 2
    INV_N = np.float32(1.0 / N)

    ix = nl.arange(128)[:, None]      # partition index (m_in / k_in)
    ik = nl.arange(K)[None, :]        # full-K free index
    iw = nl.arange(1)[:, None]        # single-row partition index for g
    inn = nl.arange(N)[None, :]       # full-N free index for w / y
    i128 = nl.arange(128)[None, :]    # 128-wide free index (sub-tile / transpose)
    icb = nl.arange(N_CHUNK)[None, :] # N-chunk free index

    out = nl.ndarray((M, N), dtype=np.float32, buffer=nl.shared_hbm)

    bias_zero = nl.zeros((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)

    g_tile = nl.load(g_tensor.reshape((1, N))[iw, inn], dtype=np.float32)

    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[ix, i128] = nl.load(identity_const[ix, i128], dtype=np.float32)

    # ---- split w into two bf16 limbs, once, fully resident ----
    w_hi = nl.ndarray((K_TILES, par_dim(128), N), dtype=nl.bfloat16, buffer=nl.sbuf)
    w_lo = nl.ndarray((K_TILES, par_dim(128), N), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        w_f = nl.load(w_tensor[kt * 128 + ix, inn], dtype=np.float32)
        w_hi[kt, ix, inn] = nl.copy(w_f[ix, inn], dtype=nl.bfloat16)
        w_res = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        w_res[ix, inn] = nisa.tensor_tensor(
            w_f[ix, inn], w_hi[kt, ix, inn], op=nl.subtract)
        w_lo[kt, ix, inn] = nl.copy(w_res[ix, inn], dtype=nl.bfloat16)

    for mt in nl.affine_range(M_TILES):
        # ---- load this M-tile of x, [m_in(par)=128, k(free)=2048] ----
        x_sb = nl.load(x_tensor[mt * 128 + ix, ik], dtype=np.float32)

        # ---- transpose the 16 x K-sub-tiles, then split each into bf16 limbs ----
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
            xT_hi[kt, ix, i128] = nl.copy(xT_f[ix, i128], dtype=nl.bfloat16)
            xT_res = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
            xT_res[ix, i128] = nisa.tensor_tensor(
                xT_f[ix, i128], xT_hi[kt, ix, i128], op=nl.subtract)
            xT_lo[kt, ix, i128] = nl.copy(xT_res[ix, i128], dtype=nl.bfloat16)

        # ---- GEMM (3 bf16 products), 2-chunk grouping with 2 live PSUM accumulators ----
        # Outer chunk-PAIR loop; inside each pair, 2 [128,512] banks accumulate across the K-tile
        # loop with the hi-pass (xT_hi[kt] stationary over 2 chunks = 4 matmuls) then the lo-pass
        # (xT_lo[kt] stationary over 2 chunks = 2 matmuls). Live PSUM = 2 accumulators + the
        # transpose bank. Per-bank accumulation order preserved (bit-exact expectation).
        y = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        for cp in nl.affine_range(N_PAIRS):
            acc = nl.zeros((CHUNKS_PER_PAIR, par_dim(128), N_CHUNK),
                           dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # hi-pass: xT_hi[kt] stationary for 4 consecutive matmuls (P1,P2 over 2 chunks)
                for j in nl.affine_range(CHUNKS_PER_PAIR):
                    c = cp * CHUNKS_PER_PAIR + j
                    # xT_hi @ w_hi
                    acc[j, ix, icb] += nisa.nc_matmul(
                        xT_hi[kt, ix, i128],
                        w_hi[kt, ix, N_CHUNK * c + icb])
                    # xT_hi @ w_lo
                    acc[j, ix, icb] += nisa.nc_matmul(
                        xT_hi[kt, ix, i128],
                        w_lo[kt, ix, N_CHUNK * c + icb])
                # lo-pass: xT_lo[kt] stationary for 2 consecutive matmuls (P3 over 2 chunks)
                for j in nl.affine_range(CHUNKS_PER_PAIR):
                    c = cp * CHUNKS_PER_PAIR + j
                    # xT_lo @ w_hi   (dropping xT_lo @ w_lo, the negligible cross term)
                    acc[j, ix, icb] += nisa.nc_matmul(
                        xT_lo[kt, ix, i128],
                        w_hi[kt, ix, N_CHUNK * c + icb])
            # residual add + eviction for this chunk pair
            for j in nl.affine_range(CHUNKS_PER_PAIR):
                c = cp * CHUNKS_PER_PAIR + j
                z_tile = nl.load(z_tensor[mt * 128 + ix, N_CHUNK * c + icb], dtype=np.float32)
                y[ix, N_CHUNK * c + icb] = nl.add(acc[j, ix, icb], z_tile[ix, icb])

        # ---- single fused RMSNorm over N (free axis), non-clobbering (byte-for-byte v1) ----
        sq = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        sq[ix, inn] = nisa.activation(
            op=nl.square, data=y[ix, inn],
            bias=bias_zero[ix, 0], scale=1.0, dtype=np.float32)
        sumsq = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        sumsq[ix, 0] = nisa.tensor_reduce(
            nl.add, data=sq[ix, inn], axis=[1], dtype=np.float32)
        mean_eps = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        mean_eps[ix, 0] = nisa.tensor_scalar(
            data=sumsq[ix, 0], op0=nl.multiply, operand0=INV_N,
            op1=nl.add, operand1=eps, dtype=np.float32)
        inv_rms = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        inv_rms[ix, 0] = nisa.activation(
            op=nl.rsqrt, data=mean_eps[ix, 0],
            bias=bias_zero[ix, 0], scale=1.0, dtype=np.float32)

        # ---- output scale + store: out = y * inv_rms * g (reads the still-live y) ----
        out_sb = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        out_sb[ix, inn] = nisa.tensor_scalar(
            data=y[ix, inn], op0=nl.multiply, operand0=inv_rms[ix, 0],
            dtype=np.float32)
        out_sb[ix, inn] = nl.multiply(out_sb[ix, inn], g_tile.broadcast_to((128, N)))
        nl.store(out[mt * 128 + ix, inn], value=out_sb[ix, inn])

    return out
