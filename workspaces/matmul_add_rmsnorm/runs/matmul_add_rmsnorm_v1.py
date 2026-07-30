import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor):
    """Fused dense fp32 GEMM + residual-add + RMSNorm (M=4096, K=2048, N=2048).

        y[m,n]     = sum_k x[m,k] * w[k,n]  +  z[m,n]       # GEMM then residual add
        inv_rms[m] = 1 / sqrt( mean_n( y[m,n]^2 ) + eps )   # per-row scalar, over N
        out[m,n]   = y[m,n] * inv_rms[m] * g[n]             # norm + per-N scale

    Raw 2D I/O (this case's transform_to_nki_inputs is the identity, so the kernel
    receives and returns raw 2D tensors and slice-tiles itself):
        x_tensor : (M=4096, K=2048)   fp32
        w_tensor : (K=2048, N=2048)   fp32
        z_tensor : (M=4096, N=2048)   fp32   residual, added to the GEMM output
        g_tensor : (N=2048,)          fp32   per-N output-axis scale
        eps      : python float scalar
        out      : (M=4096, N=2048)   fp32

    This op is the MIRROR of the rmsnorm-first siblings: they do norm -> GEMM (reduce
    over the K contraction axis, and g/transpose sit on K); this does GEMM -> add ->
    norm, so the RMS reduces over the N free axis of the natural matmul output and g is
    a free-axis broadcast that never touches the contraction. The only transpose needed
    is x -> xT for the Tensor Engine.

    The Tensor Engine's nc_matmul(stationary, moving) = stationary.T @ moving, and
    requires the contraction dim (k_in) on the PARTITION axis of both operands, both
    living in SBUF. w tiles are [k_in(par), n(free)] -> used directly as the moving
    operand. x tiles are [m_in(par), k(free)]; each [128,128] K-sub-tile is transposed
    to [k_in(par), m_in(free)] via the identity nc_matmul (is_transpose=True) idiom
    before use as the stationary operand.

    w is 16 MB (128 KB/partition) so it is loaded fully resident once and reused across
    all 32 M-tiles; each x/z tile is loaded exactly once. (The NKIBench baseline instead
    reloads all of w inside the M-loop -- 32*4*16 = 2048 weight loads, ~537 MB of
    redundant HBM reads.) Loop order is M-outer: per M-tile we transpose the 16 x
    K-sub-tiles, then an N-chunk loop streams all 4 N-chunks (width 512 = one fp32 PSUM
    bank), accumulating over the 16 K-tiles into a [m_in, 512] PSUM tile and writing
    acc + z into a full [m_in, N] SBUF row; then a single full-N square+reduce yields
    inv_rms; then the output scale y * inv_rms * g is applied full-width over the [m_in, N]
    row (no PSUM constraint there) and stored in one pass -- measured ~6% faster than a
    per-chunk output store.
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

    # Load all of w fully resident once: w_sb[kt] = [k_in(par)=128, n=2048].
    # 16 * 128 * 2048 * 4B = 16 MB = 128 KB/partition (budget ~192 KB).
    w_sb = nl.ndarray((K_TILES, par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        w_sb[kt, ix, inn] = nl.load(w_tensor[kt * 128 + ix, inn], dtype=np.float32)

    for mt in nl.affine_range(M_TILES):
        # ---- load this M-tile of x, [m_in(par)=128, k(free)=2048] ----
        x_sb = nl.load(x_tensor[mt * 128 + ix, ik], dtype=np.float32)

        # ---- transpose the 16 x K-sub-tiles -> xT[kt] = [k_in(par), m_in(free)] ----
        xT = nl.ndarray((K_TILES, par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
            psum_t[ix, i128] = nisa.nc_matmul(
                x_sb[ix, 128 * kt + i128],
                identity_local[ix, i128],
                is_transpose=True, is_moving_onezero=True)
            xT[kt, ix, i128] = nl.copy(psum_t[ix, i128], dtype=np.float32)

        # ---- GEMM + residual add: assemble the full row y = x@w + z into SBUF ----
        # (RMSNorm needs the whole N-row before reducing, so all four N-chunks are
        # written before the norm; each chunk accumulates its 16 K-tiles in one PSUM
        # bank, then adds z before eviction to the [m_in, N] y buffer.)
        y = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        for c in nl.affine_range(N_CHUNKS):
            acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # nc_matmul(stationary=xT[kt] [k_in,m_in], moving=w_sb[kt] [k_in,512])
                #   = stationary.T @ moving = [m_in,k_in] @ [k_in,512] = [m_in,512]
                acc[ix, icb] += nisa.nc_matmul(
                    xT[kt, ix, i128],
                    w_sb[kt, ix, N_CHUNK * c + icb])
            z_tile = nl.load(z_tensor[mt * 128 + ix, N_CHUNK * c + icb], dtype=np.float32)
            y[ix, N_CHUNK * c + icb] = nl.add(acc[ix, icb], z_tile[ix, icb])

        # ---- single fused RMSNorm over N (free axis), non-clobbering ----
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
        # (eps is added AFTER the /N mean, matching the reference; it is NOT scaled
        # by 1/N.)
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
        # The output scale is a pure SBUF->SBUF elementwise pass with NO PSUM-bank
        # constraint (unlike the GEMM accumulator above, which is chunked to 512 to fit
        # one fp32 PSUM bank), so it is applied full-width [128,N] in two ops rather than
        # per 512-wide chunk. This is algebraically identical (y * g / rms) and measured
        # ~6% faster than a per-chunk store: fewer/wider Vec+Scalar ops and a single store
        # per M-tile keep the Vector Engine from serializing against the PE-bound matmul
        # (see profile/ digest; the per-chunk variant is a recorded measured-reject).
        out_sb = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        # y * inv_rms  (per-row [128,1] scale broadcast across the free axis)
        out_sb[ix, inn] = nisa.tensor_scalar(
            data=y[ix, inn], op0=nl.multiply, operand0=inv_rms[ix, 0],
            dtype=np.float32)
        # * g  (per-N free-axis scale; g broadcast [1,N] -> [128,N])
        out_sb[ix, inn] = nl.multiply(out_sb[ix, inn], g_tile.broadcast_to((128, N)))
        nl.store(out[mt * 128 + ix, inn], value=out_sb[ix, inn])

    return out
