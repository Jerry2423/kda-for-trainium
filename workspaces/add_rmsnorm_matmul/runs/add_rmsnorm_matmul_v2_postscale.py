import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor):
    """Fused residual-add + RMSNorm + dense fp32 GEMM (M=4096, K=1024, N=2048).

    Pure-fp32 refactor of the v1 kernel via two algebraic commutations that move
    the per-tile scaling work off the activation and onto the weight / the eviction:

        a          = x + z                                   # residual add
        inv_rms[m] = 1 / sqrt( mean_k( a[m,k]^2 ) + eps )    # per-row scalar
        w'[k,n]    = g[k] * w[k,n]                           # per-K scale folded into w
        out[m,n]   = inv_rms[m] * ( sum_k a[m,k] * w'[k,n] ) # matmul then post-scale

    This is algebraically the v1 math regrouped:
      * g is indexed by the contraction column k, so it does NOT commute past the
        matmul -- but it can be folded into the resident weight ONCE. w_sb[kt] is
        [k_in(par), n(free)] and g[kt*128 + k_in] varies along its PARTITION axis, so
        w'[kt] = tensor_scalar(w_sb[kt], multiply, g_col[kt]) is a per-partition [128,1]
        scale applied 8 times at load, replacing v1's 32x [128,K] free-axis g multiply.
      * inv_rms[m] is per-row, so it commutes with the matmul and is applied at
        PSUM->SBUF eviction as a tensor_scalar reading the PSUM accumulator directly
        (this REPLACES v1's nl.copy eviction, so it is free, and removes v1's inline
        norm = a*inv_rms [128,K] pass).
    So RAW a = x+z is transposed and fed to the matmul; the norm reduction stays fully
    fp32 and only the narrow [128,1] inv_rms / [128,1] g columns and the [128,512]
    output tiles carry the scales.

    Raw 2D I/O (this case's transform_to_nki_inputs is the identity, so the kernel
    receives and returns raw 2D tensors and slice-tiles itself):
        x_tensor, z_tensor : (M=4096, K=1024)   fp32
        w_tensor           : (K=1024, N=2048)   fp32
        g_tensor           : (K=1024,)          fp32   per-K contraction-axis scale
        eps                : python float scalar
        out                : (M=4096, N=2048)   fp32

    The Tensor Engine's nc_matmul(stationary, moving) = stationary.T @ moving, and
    requires the contraction dim (k_in) on the PARTITION axis of both operands, both
    living in SBUF. w' tiles are [k_in(par), n(free)] -> used directly as the moving
    operand. RAW a is [m_in(par), k(free)]; each [128,128] K-sub-tile is transposed to
    [k_in(par), m_in(free)] via the identity nc_matmul (is_transpose=True) idiom before
    use as the stationary operand.

    w' is only 8 MB (64 KB/partition) so it is folded and left fully resident once and
    reused across all 32 M-tiles; each x/z tile is loaded exactly once. Loop order is
    M-outer: per M-tile we add the residual, compute inv_rms, transpose the 8 RAW
    K-sub-tiles, then stream all 4 N-chunks (width 512 = one fp32 PSUM bank),
    accumulating over the 8 K-tiles into a [m_in, 512] PSUM tile and post-scaling by
    inv_rms at eviction.
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
    # rsqrt). A [128,1] bias vector is portable across NeuronCore generations
    # (a scalar bias is only accepted from NeuronCore-v3+).
    bias_zero = nl.zeros((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)

    # 128x128 identity in SBUF, used as the moving operand to transpose the RAW
    # activation sub-tiles on the Tensor Engine. Loaded once, reused for all tiles.
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[ix, i128] = nl.load(identity_const[ix, i128], dtype=np.float32)

    # ---- fold g into resident w once: w'[kt] = g_col[kt] * w[kt] ----
    # w'[kt] = [k_in(par)=128, n=2048] fp32; 8 * 128 * 2048 * 4B = 8 MB = 64 KB/part.
    # g_col[kt] = g[kt*128 : kt*128+128] as a [128,1] partition vector (g is indexed by
    # the contraction column k, which is the PARTITION axis of the weight tile), so the
    # multiply is a natural per-partition tensor_scalar broadcast across the free axis n.
    w_prime = nl.ndarray((K_TILES, par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        w_f = nl.load(w_tensor[kt * 128 + ix, inn], dtype=np.float32)
        g_col = nl.load(g_tensor.reshape((K, 1))[kt * 128 + ix, ig], dtype=np.float32)
        w_prime[kt, ix, inn] = nisa.tensor_scalar(
            data=w_f[ix, inn], op0=nl.multiply, operand0=g_col[ix, 0],
            dtype=np.float32)

    for mt in nl.affine_range(M_TILES):
        # ---- residual add a = x + z, [m_in(par)=128, k(free)=1024] ----
        x_sb = nl.load(x_tensor[mt * 128 + ix, ik], dtype=np.float32)
        z_sb = nl.load(z_tensor[mt * 128 + ix, ik], dtype=np.float32)
        a_sb = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        a_sb[ix, ik] = nl.add(x_sb[ix, ik], z_sb[ix, ik])

        # ---- fused RMSNorm reduction over K, entirely fp32 in SBUF (per-row scale) ----
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

        # ---- transpose the 8 RAW a K-sub-tiles -> aT[kt] = [k_in(par), m_in(free)] ----
        aT = nl.ndarray((K_TILES, par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
            psum_t[ix, i128] = nisa.nc_matmul(
                a_sb[ix, 128 * kt + i128],
                identity_local[ix, i128],
                is_transpose=True, is_moving_onezero=True)
            aT[kt, ix, i128] = nl.copy(psum_t[ix, i128], dtype=np.float32)

        # ---- matmul: stream all of N, accumulate over K into [m_in, 512] PSUM ----
        for c in nl.affine_range(N_CHUNKS):
            acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # nc_matmul(stationary=aT[kt] [k_in,m_in], moving=w'[kt] [k_in,512])
                #   = stationary.T @ moving = [m_in,k_in] @ [k_in,512] = [m_in,512]
                #   = the UN-normalized (a @ w') partial, accumulated over K.
                acc[ix, icb] += nisa.nc_matmul(
                    aT[kt, ix, i128],
                    w_prime[kt, ix, N_CHUNK * c + icb])

            # Post-scale eviction: read the PSUM accumulator directly and apply the
            # per-row [128,1] inv_rms as the result moves PSUM->SBUF. inv_rms is aligned
            # to the output partition axis (m_in) and broadcasts across the free axis (n),
            # so each output row is scaled exactly once -- folding the norm into eviction.
            out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
            out_sb[ix, icb] = nisa.tensor_scalar(
                data=acc[ix, icb], op0=nl.multiply, operand0=inv_rms[ix, 0],
                dtype=np.float32)
            nl.store(out[mt * 128 + ix, N_CHUNK * c + icb], value=out_sb[ix, icb])

    return out
