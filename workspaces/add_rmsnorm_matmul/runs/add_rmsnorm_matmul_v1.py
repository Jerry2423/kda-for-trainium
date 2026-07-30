import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor):
    """Fused residual-add + RMSNorm + dense fp32 GEMM (M=4096, K=1024, N=2048).

        a          = x + z                                   # residual add
        inv_rms[m] = 1 / sqrt( mean_k( a[m,k]^2 ) + eps )    # per-row scalar
        y[m,k]     = a[m,k] * inv_rms[m] * g[k]              # norm + per-K scale
        out[m,n]   = sum_k y[m,k] * w[k,n]                   # dense GEMM

    Raw 2D I/O (this case's transform_to_nki_inputs is the identity, so the kernel
    receives and returns raw 2D tensors and slice-tiles itself):
        x_tensor, z_tensor : (M=4096, K=1024)   fp32
        w_tensor           : (K=1024, N=2048)   fp32
        g_tensor           : (K=1024,)          fp32   per-K contraction-axis scale
        eps                : python float scalar
        out                : (M=4096, N=2048)   fp32

    The Tensor Engine's nc_matmul(stationary, moving) = stationary.T @ moving, and
    requires the contraction dim (k_in) on the PARTITION axis of both operands, both
    living in SBUF. w tiles are [k_in(par), n(free)] -> used directly as the moving
    operand. The normalized activation is [m_in(par), k(free)]; each normalized
    [128,128] K-sub-tile is transposed to [k_in(par), m_in(free)] via the identity
    nc_matmul (is_transpose=True) idiom before use as the stationary operand.

    w is only 8 MB (64 KB/partition) so it is loaded fully resident once and reused
    across all 32 M-tiles; each x/z tile is loaded exactly once. (The NKIBench
    baseline instead reloads all of w inside the M-loop -- 32*4*8 = 1024 weight
    loads, ~256 MB of redundant HBM reads.) Loop order is M-outer: per M-tile we
    add the residual, compute inv_rms, scale the row and apply g, transpose the 8
    K-sub-tiles, then stream all 4 N-chunks (width 512 = one fp32 PSUM bank),
    accumulating over the 8 K-tiles into a [m_in, 512] PSUM tile.

    g is per-contraction-column (indexed by k) so it does NOT commute past the
    matmul; it is applied inline on the activation (broadcast [1,K] -> [128,K] along
    the partition axis, then a free-axis multiply). inv_rms is per-row (commutes with
    the matmul) but is likewise applied inline here for a clean, obviously-correct
    kernel.
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
    iw = nl.arange(1)[:, None]        # single-row partition index for g
    inn = nl.arange(N)[None, :]       # full-N free index for w
    i128 = nl.arange(128)[None, :]    # 128-wide free index (sub-tile / transpose)
    icb = nl.arange(N_CHUNK)[None, :] # N-chunk free index

    out = nl.ndarray((M, N), dtype=np.float32, buffer=nl.shared_hbm)

    # Per-partition zero bias [128,1] for the Scalar-Engine activations (square,
    # rsqrt). A [128,1] bias vector is portable across NeuronCore generations
    # (a scalar bias is only accepted from NeuronCore-v3+).
    bias_zero = nl.zeros((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)

    # g as a [1,K] row, broadcast to [128,K] along the partition axis at use.
    g_tile = nl.load(g_tensor.reshape((1, K))[iw, ik], dtype=np.float32)

    # 128x128 identity in SBUF, used as the moving operand to transpose the
    # normalized sub-tiles on the Tensor Engine. Loaded once, reused for all tiles.
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[ix, i128] = nl.load(identity_const[ix, i128], dtype=np.float32)

    # Load all of w fully resident once: w_sb[kt] = [k_in(par)=128, n=2048].
    # 8 * 128 * 2048 * 4B = 8 MB = 64 KB/partition (budget 192 KB).
    w_sb = nl.ndarray((K_TILES, par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        w_sb[kt, ix, inn] = nl.load(w_tensor[kt * 128 + ix, inn], dtype=np.float32)

    for mt in nl.affine_range(M_TILES):
        # ---- residual add a = x + z, [m_in(par)=128, k(free)=1024] ----
        x_sb = nl.load(x_tensor[mt * 128 + ix, ik], dtype=np.float32)
        z_sb = nl.load(z_tensor[mt * 128 + ix, ik], dtype=np.float32)
        a_sb = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        a_sb[ix, ik] = nl.add(x_sb[ix, ik], z_sb[ix, ik])

        # ---- fused RMSNorm over K, entirely in SBUF ----
        # sq = a^2  (Scalar Engine, computed in fp32)
        sq = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        sq[ix, ik] = nisa.activation(
            op=nl.square, data=a_sb[ix, ik],
            bias=bias_zero[ix, 0], scale=1.0, dtype=np.float32)
        # sumsq = sum_k(a^2)  (single full-1024-wide free-axis reduce, Vector Engine)
        sumsq = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        sumsq[ix, 0] = nisa.tensor_reduce(
            nl.add, data=sq[ix, ik], axis=[1], dtype=np.float32)
        # mean_eps = sumsq*(1/K) + eps  == mean_k(a^2) + eps
        # (eps is added AFTER the /K mean, matching the reference; it is NOT scaled
        # by 1/K).
        mean_eps = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        mean_eps[ix, 0] = nisa.tensor_scalar(
            data=sumsq[ix, 0], op0=nl.multiply, operand0=INV_K,
            op1=nl.add, operand1=eps, dtype=np.float32)
        # inv_rms = 1 / sqrt(mean_eps)
        inv_rms = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        inv_rms[ix, 0] = nisa.activation(
            op=nl.rsqrt, data=mean_eps[ix, 0],
            bias=bias_zero[ix, 0], scale=1.0, dtype=np.float32)
        # norm = a * inv_rms  (per-row [128,1] scale broadcast across the free axis)
        norm = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        norm[ix, ik] = nisa.tensor_scalar(
            data=a_sb[ix, ik], op0=nl.multiply, operand0=inv_rms[ix, 0],
            dtype=np.float32)
        # y = norm * g  (per-K free-axis scale; g broadcast [1,K] -> [128,K])
        y = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        y[ix, ik] = nl.multiply(norm[ix, ik], g_tile.broadcast_to((128, K)))

        # ---- transpose the 8 normalized K-sub-tiles -> yT[kt] = [k_in(par), m_in(free)] ----
        yT = nl.ndarray((K_TILES, par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
            psum_t[ix, i128] = nisa.nc_matmul(
                y[ix, 128 * kt + i128],
                identity_local[ix, i128],
                is_transpose=True, is_moving_onezero=True)
            yT[kt, ix, i128] = nl.copy(psum_t[ix, i128], dtype=np.float32)

        # ---- matmul: stream all of N, accumulate over K into [m_in, 512] PSUM ----
        for c in nl.affine_range(N_CHUNKS):
            acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # nc_matmul(stationary=yT[kt] [k_in,m_in], moving=w_sb[kt] [k_in,512])
                #   = stationary.T @ moving = [m_in,k_in] @ [k_in,512] = [m_in,512]
                acc[ix, icb] += nisa.nc_matmul(
                    yT[kt, ix, i128],
                    w_sb[kt, ix, N_CHUNK * c + icb])

            out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
            out_sb[ix, icb] = nl.copy(acc[ix, icb], dtype=np.float32)
            nl.store(out[mt * 128 + ix, N_CHUNK * c + icb], value=out_sb[ix, icb])

    return out
