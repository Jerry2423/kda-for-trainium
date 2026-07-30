# swiglu_v1 — first correct fp32 NKI kernel for the fused SwiGLU feed-forward op.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (swiglu_M4096_N3072_K1024_0.py) = 2.074257 ms.
#
#   up   = x @ w_up                          # (M,N)
#   gate = x @ w_gate                        # (M,N)
#   h    = (gate * sigmoid(gate)) * up       # SiLU(gate) * up, elementwise on (M,N)
#   out  = h @ w_down                        # (M,K)
#
# with M=4096, K=1024, N=3072, all fp32.
#
# Structure (M-outer, one 128-row M-tile at a time): reshape v1 to a no-copy
# (32,128,1024) view, load a 128x128 identity once, then for each of 32 M-tiles:
#   - load the x tile [m_in(par)=128, k(free)=1024] and transpose it ONCE into 8
#     [k_in,m_in] sub-tiles (the identity-matmul transpose idiom). This single
#     transpose is SHARED as the stationary operand by BOTH the up and gate GEMMs.
#   - stream w_up / w_gate over 6 N-chunks of 512 (one fp32 PSUM bank), accumulating
#     over the 8 K-tiles into two PSUM banks; copy each PSUM->SBUF, apply the fused
#     nl.silu on the gate projection and multiply by up, into a resident h_sbuf
#     [128,3072] (no HBM spill of h, unlike the baseline's _spill_163/_reload_166).
#   - transpose h into 24 [n_in,m_in] sub-tiles, then stream w_down over 2 K-out
#     chunks of 512, accumulating over the 24 N-tiles into the output tile.
#
# The Tensor Engine's nc_matmul(stationary, moving) = stationary.T @ moving, with the
# contraction dim on the PARTITION axis of both operands, both resident in SBUF, and
# the moving free dim <= 512 (one fp32 PSUM bank). Weights are streamed from HBM
# (B=1) — the three 12 MB weights (288 KB/partition combined) do not all fit resident.

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2, v3, v4):
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    M_TILES = 32       # 4096 / 128
    K_TILES = 8        # 1024 / 128   (contraction of the up/gate GEMMs)
    N = 3072
    N_TILES = 24       # 3072 / 128   (contraction of the down GEMM)
    K_OUT = 1024       # down-GEMM output width
    CHUNK = 512        # one fp32 PSUM bank in the free dim
    N_CHUNKS = N // CHUNK        # 6   (up/gate output tiling)
    K_OUT_CHUNKS = K_OUT // CHUNK  # 2  (down-GEMM output tiling)

    ix = nl.arange(128)[:, None]      # partition index (m_in / k_in / n_in)
    i128 = nl.arange(128)[None, :]    # 128-wide free index (sub-tile / transpose)
    ich = nl.arange(CHUNK)[None, :]   # 512-wide N/K-out chunk free index

    # v1 = x, tiled (8,4,128,8,128) which is row-major identical to (32,128,1024):
    # x3[mt, m_in, k] = x[mt*128 + m_in, k]. A pure no-copy reshape view (no DMA).
    x3 = v1.reshape((M_TILES, 128, 1024))

    # Output v5, tiled (32,128,1024): v5[mt, m_in, kp] = out[mt*128 + m_in, kp].
    v5 = nl.ndarray((M_TILES, 128, K_OUT), dtype=np.float32, buffer=nl.shared_hbm)

    # 128x128 identity in SBUF, the moving operand for the identity-matmul transpose.
    # Loaded once, reused for every x and h sub-tile transpose.
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[ix, i128] = nl.load(identity_const[ix, i128], dtype=np.float32)

    for mt in nl.affine_range(M_TILES):
        # ---- load x tile and transpose ONCE into 8 shared xT sub-tiles ----
        # x_sb = [m_in(par)=128, k(free)=1024]; k is on the free axis, so each
        # [128,128] K-sub-tile is transposed to xT[kt] = [k_in(par), m_in(free)].
        x_sb = nl.load(x3[mt, ix, nl.arange(1024)[None, :]], dtype=np.float32)
        xT = nl.ndarray((K_TILES, par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
            psum_t[ix, i128] = nisa.nc_matmul(
                x_sb[ix, 128 * kt + i128],
                identity_local[ix, i128],
                is_transpose=True, is_moving_onezero=True)
            xT[kt, ix, i128] = nl.copy(psum_t[ix, i128], dtype=np.float32)

        # ---- up + gate projections, N-chunk by N-chunk, SiLU fused at eviction ----
        # h_sbuf holds SiLU(gate)*up fully resident: [128, 3072] fp32 = 12 KB/partition.
        # No HBM spill/reload of h (removes the baseline's _spill_163/_reload_166).
        h_sbuf = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        for c in nl.affine_range(N_CHUNKS):
            up_acc = nl.zeros((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.psum)
            gate_acc = nl.zeros((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # w_up / w_gate tiles are [k_in(par)=128, 512(free)] — moving operands
                # directly (contraction k_in already on the partition axis).
                w_up_sb = nl.load(
                    v2[kt, ix, CHUNK * c + ich], dtype=np.float32)
                w_gate_sb = nl.load(
                    v4[kt, ix, CHUNK * c + ich], dtype=np.float32)
                # nc_matmul(stationary=xT[kt] [k_in,m_in], moving=w [k_in,512])
                #   = stationary.T @ moving = [m_in,k_in] @ [k_in,512] = [m_in,512]
                up_acc[ix, ich] += nisa.nc_matmul(xT[kt, ix, i128], w_up_sb[ix, ich])
                gate_acc[ix, ich] += nisa.nc_matmul(xT[kt, ix, i128], w_gate_sb[ix, ich])

            # PSUM -> SBUF copy BEFORE activation (activation must not read a raw PSUM
            # bank; matches the baseline and every sibling).
            up_sb = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
            up_sb[ix, ich] = nl.copy(up_acc[ix, ich], dtype=np.float32)
            gate_sb = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
            gate_sb[ix, ich] = nl.copy(gate_acc[ix, ich], dtype=np.float32)
            # Fused SiLU on the Scalar Engine: sg = gate * sigmoid(gate), one op.
            # Algebraically identical to the reference's gate/(1+exp(-gate)).
            sg = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
            sg[ix, ich] = nisa.activation(op=nl.silu, data=gate_sb[ix, ich], dtype=np.float32)
            # h = SiLU(gate) * up  (Vector-Engine multiply) into the resident tile.
            h_sbuf[ix, CHUNK * c + ich] = nl.multiply(sg[ix, ich], up_sb[ix, ich])

        # ---- transpose h into 24 hT sub-tiles, then the down projection ----
        # h_sbuf = [m_in(par), n(free)]; the down GEMM contracts over n, so transpose
        # each [128,128] N-sub-tile to hT[nt] = [n_in(par), m_in(free)].
        hT = nl.ndarray((N_TILES, par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
        for nt_ in nl.affine_range(N_TILES):
            psum_h = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
            psum_h[ix, i128] = nisa.nc_matmul(
                h_sbuf[ix, 128 * nt_ + i128],
                identity_local[ix, i128],
                is_transpose=True, is_moving_onezero=True)
            hT[nt_, ix, i128] = nl.copy(psum_h[ix, i128], dtype=np.float32)

        for c2 in nl.affine_range(K_OUT_CHUNKS):
            out_acc = nl.zeros((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.psum)
            for nt_ in nl.affine_range(N_TILES):
                # w_down tile is [n_in(par)=128, 512(free)] — moving operand directly
                # (contraction n_in already on the partition axis).
                w_down_sb = nl.load(v3[nt_, ix, CHUNK * c2 + ich], dtype=np.float32)
                # nc_matmul(stationary=hT[nt] [n_in,m_in], moving=w_down [n_in,512])
                #   = [m_in,n_in] @ [n_in,512] = [m_in,512]
                out_acc[ix, ich] += nisa.nc_matmul(hT[nt_, ix, i128], w_down_sb[ix, ich])

            out_sb = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
            out_sb[ix, ich] = nl.copy(out_acc[ix, ich], dtype=np.float32)
            nl.store(v5[mt, ix, CHUNK * c2 + ich], value=out_sb[ix, ich])

    return v5
