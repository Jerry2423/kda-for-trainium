import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Fused RMSNorm + dense fp32 GEMM  out = rmsnorm(x) @ w  (M=4096, K=1024, N=2048).

    Post-scale variant. RMSNorm's per-row factor commutes with the matmul because
    inv_rms[m] is a per-row scalar:
        (x[m,:] * inv_rms[m]) @ w  ==  (x[m,:] @ w) * inv_rms[m]
    so instead of scaling x before the matmul, we transpose RAW x, matmul into PSUM,
    and apply the per-row inv_rms when moving the result PSUM->SBUF (the eviction op
    reads the PSUM accumulator directly as a tensor_scalar source). This decouples the
    transpose from the norm (the transpose no longer depends on the normalized values),
    and removes the full-width [128,1024] input-scale pass — the per-row scale now runs
    once on the narrower [128,512] output tiles at eviction.

        inv_rms[m]      = 1 / sqrt( mean_k( x[m,k]^2 ) )   # per-row scalar
        out[m,n]        = ( sum_k x[m,k] * w[k,n] ) * inv_rms[m]

    Inputs (tiled by the NKIBench reference's transform_to_nki_inputs):
        v1: (32, 128, 1024) = [m_tile, m_in, k]   (x)
        v2: (8, 128, 2048)  = [k_tile, k_in, n]   (w)
    Output:
        v3: (32, 128, 2048) = [m_tile, m_in, n]   (out)

    The Tensor Engine's nc_matmul(stationary, moving) = stationary.T @ moving, and
    requires the contraction dim (k_in) on the PARTITION axis of both operands, both
    living in SBUF. w tiles are [k_in(par), n(free)] -> used directly as the moving
    operand. x tiles are [m_in(par), k(free)]; each RAW [128,128] K-sub-tile is
    transposed to [k_in(par), m_in(free)] via the identity nc_matmul (is_transpose=True)
    idiom before use as the stationary operand.

    w is only 8 MB (64 KB/partition) so it is loaded fully resident once and reused
    across all 32 M-tiles; each x tile is loaded exactly once. Loop order is M-outer:
    per M-tile we compute inv_rms, transpose the 8 raw K-sub-tiles, then stream all 4
    N-chunks (width 512 = one fp32 PSUM bank), accumulating over the 8 K-tiles into a
    [m_in, 512] PSUM tile and scaling by inv_rms at eviction.
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

    # Per-partition zero bias [128,1] for the Scalar-Engine activations (square,
    # rsqrt). A per-partition [128,1] bias vector is portable across NeuronCore
    # generations (a scalar bias is only accepted from NeuronCore-v3+).
    bias_zero = nl.zeros((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)

    # 128x128 identity in SBUF, used as the moving operand to transpose the
    # raw x sub-tiles on the Tensor Engine. Loaded once, reused for all tiles.
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]],
        dtype=np.float32)

    # Load all of w fully resident in SBUF once: w_sb[kt] = [k_in(par)=128, n=2048].
    # 8 * 128 * 2048 * 4B = 8 MB = 64 KB/partition (budget 192 KB).
    w_sb = nl.ndarray((K_TILES, par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        w_sb[kt, nl.arange(128)[:, None], nl.arange(N)[None, :]] = nl.load(
            v2[kt, nl.arange(128)[:, None], nl.arange(N)[None, :]], dtype=np.float32)

    for mt in nl.affine_range(M_TILES):
        # Load this M-tile's x once: [m_in(par)=128, k(free)=1024].
        x_sb = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        x_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]] = nl.load(
            v1[mt, nl.arange(128)[:, None], nl.arange(K)[None, :]], dtype=np.float32)

        # ---- RMSNorm reduction over K, entirely in SBUF (produces the per-row scale) ----
        # sq = x^2  (Scalar Engine, computed in fp32)
        sq = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
        sq[nl.arange(128)[:, None], nl.arange(K)[None, :]] = nisa.activation(
            op=nl.square, data=x_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]],
            bias=bias_zero[nl.arange(128)[:, None], 0], scale=1.0, dtype=np.float32)
        # sumsq = sum_k(x^2)  (single full-1024-wide free-axis reduce, Vector Engine)
        sumsq = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        sumsq[nl.arange(128)[:, None], 0] = nisa.tensor_reduce(
            nl.add, data=sq[nl.arange(128)[:, None], nl.arange(K)[None, :]],
            axis=[1], dtype=np.float32)
        # inv_rms = rsqrt(sumsq * 1/K) = 1/sqrt(mean_k(x^2))  (folds 1/K into scale)
        inv_rms = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
        inv_rms[nl.arange(128)[:, None], 0] = nisa.activation(
            op=nl.rsqrt, data=sumsq[nl.arange(128)[:, None], 0],
            bias=bias_zero[nl.arange(128)[:, None], 0], scale=INV_K, dtype=np.float32)

        # ---- transpose the 8 RAW x K-sub-tiles -> xT[kt] = [k_in(par), m_in(free)] ----
        xT = nl.ndarray((K_TILES, par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
            psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                x_sb[nl.arange(128)[:, None], 128 * kt + nl.arange(128)[None, :]],
                identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                is_transpose=True, is_moving_onezero=True)
            xT[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                dtype=np.float32)

        # ---- matmul: stream all of N, accumulate over K into [m_in, 512] PSUM ----
        for c in nl.affine_range(N_CHUNKS):
            acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # nc_matmul(stationary=xT[kt] [k_in,m_in], moving=w_sb[kt] [k_in,512])
                #   = stationary.T @ moving = [m_in,k_in] @ [k_in,512] = [m_in,512]
                #   = the UN-normalized (x @ w) partial, accumulated over K.
                acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                    xT[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    w_sb[kt, nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])

            # Post-scale eviction: read the PSUM accumulator directly and apply the
            # per-row [128,1] inv_rms as the result moves PSUM->SBUF. inv_rms is aligned
            # to the output partition axis (m_in) and broadcasts across the free axis (n),
            # so this scales each output row exactly once — folding the norm into eviction.
            out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
            out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.tensor_scalar(
                data=acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                op0=nl.multiply, operand0=inv_rms[nl.arange(128)[:, None], 0],
                dtype=np.float32)
            # Output tile [m_in(par), n(free)] -> v3[mt, :, n0:n0+512]
            # (v3[mt,mi,n] == out[mt*128+mi, n]).
            nl.store(
                out[mt, nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                value=out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

    return out
