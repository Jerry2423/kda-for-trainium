import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor):
    """Compensated bf16x2 4-PRODUCT split (accuracy-repair fallback for v3).

    Identical to add_rmsnorm_matmul_v3_bf16_split.py EXCEPT it keeps the fourth
    (normally-dropped) bf16 cross term aT_lo@w'_lo, accumulating all four bf16 products
    in fp32 PSUM:
        a @ w'  ~=  aT_hi@w'_hi + aT_hi@w'_lo + aT_lo@w'_hi + aT_lo@w'_lo

    This 4-product accuracy-repair variant was measured after the 3-product split landed
    near the review threshold (on-device rel-L2 ~1.53e-5). It trades a 4th bf16 pass
    (~+8% PE) for the offline-predicted rel-L2 improvement (offline: 3-product 4.451e-6 ->
    4-product 3.483e-6). Same PINNED split order, same fp32 RMSNorm + g-into-w' fold +
    inv_rms post-scale eviction as the 3-product kernel; the ONLY delta is the extra
    product. Everything else (signature, raw-2D I/O, invariants) is unchanged.
    """
    M = 4096
    K = 1024
    K_TILES = 8               # 1024 / 128
    M_TILES = 32              # 4096 / 128
    N = 2048
    N_CHUNK = 512             # one fp32 PSUM bank in the free dim
    N_CHUNKS = N // N_CHUNK   # 4
    INV_K = np.float32(1.0 / K)

    ix = nl.arange(128)[:, None]
    ik = nl.arange(K)[None, :]
    ig = nl.arange(1)[None, :]
    inn = nl.arange(N)[None, :]
    i128 = nl.arange(128)[None, :]
    icb = nl.arange(N_CHUNK)[None, :]

    out = nl.ndarray((M, N), dtype=np.float32, buffer=nl.shared_hbm)

    bias_zero = nl.zeros((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)

    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[ix, i128] = nl.load(identity_const[ix, i128], dtype=np.float32)

    # ---- fold g into w' in fp32, then split w' into two bf16 limbs, once ----
    w_hi = nl.ndarray((K_TILES, par_dim(128), N), dtype=nl.bfloat16, buffer=nl.sbuf)
    w_lo = nl.ndarray((K_TILES, par_dim(128), N), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        w_f = nl.load(w_tensor[kt * 128 + ix, inn], dtype=np.float32)
        g_col = nl.load(g_tensor.reshape((K, 1))[kt * 128 + ix, ig], dtype=np.float32)
        w_prime = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        w_prime[ix, inn] = nisa.tensor_scalar(
            data=w_f[ix, inn], op0=nl.multiply, operand0=g_col[ix, 0],
            dtype=np.float32)
        w_hi[kt, ix, inn] = nl.copy(w_prime[ix, inn], dtype=nl.bfloat16)
        w_res = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        w_res[ix, inn] = nisa.tensor_tensor(
            w_prime[ix, inn], w_hi[kt, ix, inn], op=nl.subtract)
        w_lo[kt, ix, inn] = nl.copy(w_res[ix, inn], dtype=nl.bfloat16)

    for mt in nl.affine_range(M_TILES):
        x_sb = nl.load(x_tensor[mt * 128 + ix, ik], dtype=np.float32)
        z_sb = nl.load(z_tensor[mt * 128 + ix, ik], dtype=np.float32)
        a_sb = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        a_sb[ix, ik] = nl.add(x_sb[ix, ik], z_sb[ix, ik])

        sq = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        sq[ix, ik] = nisa.activation(
            op=nl.square, data=a_sb[ix, ik],
            bias=bias_zero[ix, 0], scale=1.0, dtype=np.float32)
        sumsq = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        sumsq[ix, 0] = nisa.tensor_reduce(
            nl.add, data=sq[ix, ik], axis=[1], dtype=np.float32)
        mean_eps = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        mean_eps[ix, 0] = nisa.tensor_scalar(
            data=sumsq[ix, 0], op0=nl.multiply, operand0=INV_K,
            op1=nl.add, operand1=eps, dtype=np.float32)
        inv_rms = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        inv_rms[ix, 0] = nisa.activation(
            op=nl.rsqrt, data=mean_eps[ix, 0],
            bias=bias_zero[ix, 0], scale=1.0, dtype=np.float32)

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
            aT_hi[kt, ix, i128] = nl.copy(aT_f[ix, i128], dtype=nl.bfloat16)
            aT_res = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
            aT_res[ix, i128] = nisa.tensor_tensor(
                aT_f[ix, i128], aT_hi[kt, ix, i128], op=nl.subtract)
            aT_lo[kt, ix, i128] = nl.copy(aT_res[ix, i128], dtype=nl.bfloat16)

        # ---- matmul: 4 bf16 products per (N-chunk, K-tile), accumulated in fp32 PSUM ----
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
                # aT_lo @ w'_hi
                acc[ix, icb] += nisa.nc_matmul(
                    aT_lo[kt, ix, i128],
                    w_hi[kt, ix, N_CHUNK * c + icb])
                # aT_lo @ w'_lo  (the 4th cross term v3 drops; kept here for accuracy)
                acc[ix, icb] += nisa.nc_matmul(
                    aT_lo[kt, ix, i128],
                    w_lo[kt, ix, N_CHUNK * c + icb])

            out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
            out_sb[ix, icb] = nisa.tensor_scalar(
                data=acc[ix, icb], op0=nl.multiply, operand0=inv_rms[ix, 0],
                dtype=np.float32)
            nl.store(out[mt * 128 + ix, N_CHUNK * c + icb], value=out_sb[ix, icb])

    return out
