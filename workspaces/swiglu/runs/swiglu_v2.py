# swiglu_v2 — compensated bf16x2 split GEMM for the fused SwiGLU feed-forward op.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (swiglu_M4096_N3072_K1024_0.py) = 2.074257 ms.
# Parent: swiglu_v1 (fp32 throughout, 2.2079 ms / 0.939x — PE-bound at the trn2
# fp32-emulation floor: MFU=44%, PE=95%). This is v1's loop nest and layout with a
# two-limb compensated bf16 split inserted on all three GEMMs.
#
#   up   = x @ w_up                          # (M,N)
#   gate = x @ w_gate                        # (M,N)
#   h    = (gate * sigmoid(gate)) * up       # SiLU(gate) * up, elementwise on (M,N)
#   out  = h @ w_down                        # (M,K)
#
# with M=4096, K=1024, N=3072, all fp32 I/O.
#
# Why: the trn2 PE array is bf16-native; a correct fp32 GEMM runs multiple internal
# passes and is capped near ~44% MFU by that rate penalty. The three GEMMs are ~95%
# of PE work, so the only lever on this PE-floored kernel is to stop paying the fp32
# emulation tax. Each fp32 operand is split into a high and low bfloat16 limb; three
# bf16 products accumulate in one fp32 PSUM bank, recovering ~16 effective mantissa
# bits at bf16 matmul rate (dropping the negligible lo@lo cross term):
#
#     a_hi = bf16(a),  a_lo = bf16(a - a_hi)          # round-to-nearest-even
#     b_hi = bf16(b),  b_lo = bf16(b - b_hi)
#     a @ b  ~=  a_hi@b_hi + a_hi@b_lo + a_lo@b_hi     # fp32 PSUM accumulation
#
# The two identity-matmul transposes stay EXACT fp32; operands are split into limbs
# AFTER the transpose (splitting after an exact transpose is identical to splitting
# before — the transpose is exact and bf16 rounding is element-wise). The fused
# nl.silu + nl.multiply producing h_sbuf stay fp32, identical to v1.
#
# An offline numpy sim (runs/offline_bf16_split_sim.py) that draws real distinct
# per-seed inputs proves the all-3-bf16x2 worst-case rel-L2 = 7.72e-6 over seeds
# [42,0,21,63,84], 2.6x under the 2e-5 gate, essentially seed-independent.
#
# Structure (unchanged from v1): M-outer, one 128-row M-tile at a time. Per M-tile:
#   - load x tile [m_in(par)=128, k(free)=1024], transpose ONCE into 8 fp32 [k_in,m_in]
#     sub-tiles (identity-matmul), then split each into xT_hi/xT_lo (bf16) ONCE. These
#     limbs are the SHARED stationary operand for BOTH the up and gate GEMMs.
#   - stream w_up / w_gate over 6 N-chunks of 512 (one fp32 PSUM bank), building each
#     weight chunk's bf16 limbs on-chip, and accumulating 3 bf16 products per K-tile
#     into two fp32 PSUM banks; PSUM->SBUF copy, fused nl.silu on gate * up, into a
#     resident h_sbuf [128,3072] fp32 (no HBM spill of h, as in v1).
#   - transpose h into 24 fp32 [n_in,m_in] sub-tiles, split into hT_hi/hT_lo (bf16),
#     then stream w_down over 2 K-out chunks of 512, building its limbs on-chip and
#     accumulating 3 bf16 products over the 24 N-tiles into the output tile.
#
# Weights are streamed from HBM (B=1): the three 12 MB fp32 weights do not fit
# resident, so their bf16 limbs are rebuilt per streamed chunk (unlike the sibling
# add_rmsnorm_matmul_v3, whose 2048-wide weight held its limbs resident). HBM traffic
# is unchanged from v1 (~607 MB read / 17 MB write): limbs are built on-chip from the
# same fp32 loads.

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
    # Loaded once, reused for every x and h sub-tile transpose. Transpose stays fp32.
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[ix, i128] = nl.load(identity_const[ix, i128], dtype=np.float32)

    for mt in nl.affine_range(M_TILES):
        # ---- load x tile, transpose ONCE, then split each sub-tile into bf16 limbs ----
        # x_sb = [m_in(par)=128, k(free)=1024]; each [128,128] K-sub-tile transposes to
        # xT = [k_in(par), m_in(free)] (exact fp32), then splits into xT_hi/xT_lo (bf16).
        # These limbs are SHARED by both the up and gate GEMMs (split once, reused).
        x_sb = nl.load(x3[mt, ix, nl.arange(1024)[None, :]], dtype=np.float32)
        xT_hi = nl.ndarray((K_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        xT_lo = nl.ndarray((K_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
            psum_t[ix, i128] = nisa.nc_matmul(
                x_sb[ix, 128 * kt + i128],
                identity_local[ix, i128],
                is_transpose=True, is_moving_onezero=True)
            # exact fp32 transpose result; split into two bf16 limbs.
            xT_f = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
            xT_f[ix, i128] = nl.copy(psum_t[ix, i128], dtype=np.float32)
            xT_hi[kt, ix, i128] = nl.copy(xT_f[ix, i128], dtype=nl.bfloat16)
            xT_res = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
            xT_res[ix, i128] = nisa.tensor_tensor(
                xT_f[ix, i128], xT_hi[kt, ix, i128], op=nl.subtract)
            xT_lo[kt, ix, i128] = nl.copy(xT_res[ix, i128], dtype=nl.bfloat16)

        # ---- up + gate projections, N-chunk by N-chunk, SiLU fused at eviction ----
        # h_sbuf holds SiLU(gate)*up fully resident: [128, 3072] fp32 = 12 KB/partition.
        # No HBM spill/reload of h (removes the baseline's _spill_163/_reload_166).
        h_sbuf = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        for c in nl.affine_range(N_CHUNKS):
            up_acc = nl.zeros((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.psum)
            gate_acc = nl.zeros((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # w_up / w_gate chunks are [k_in(par)=128, 512(free)] fp32 — build each
                # chunk's bf16 limbs on-chip (hi = RNE bf16 cast; lo = bf16 of the exact
                # fp32 residual). Limbs are transient per (c, kt): only one chunk's limbs
                # are live at a time (B=1 streaming).
                w_up_f = nl.load(v2[kt, ix, CHUNK * c + ich], dtype=np.float32)
                w_up_hi = nl.ndarray((par_dim(128), CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                w_up_hi[ix, ich] = nl.copy(w_up_f[ix, ich], dtype=nl.bfloat16)
                w_up_res = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
                w_up_res[ix, ich] = nisa.tensor_tensor(
                    w_up_f[ix, ich], w_up_hi[ix, ich], op=nl.subtract)
                w_up_lo = nl.ndarray((par_dim(128), CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                w_up_lo[ix, ich] = nl.copy(w_up_res[ix, ich], dtype=nl.bfloat16)

                w_gate_f = nl.load(v4[kt, ix, CHUNK * c + ich], dtype=np.float32)
                w_gate_hi = nl.ndarray((par_dim(128), CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                w_gate_hi[ix, ich] = nl.copy(w_gate_f[ix, ich], dtype=nl.bfloat16)
                w_gate_res = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
                w_gate_res[ix, ich] = nisa.tensor_tensor(
                    w_gate_f[ix, ich], w_gate_hi[ix, ich], op=nl.subtract)
                w_gate_lo = nl.ndarray((par_dim(128), CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                w_gate_lo[ix, ich] = nl.copy(w_gate_res[ix, ich], dtype=nl.bfloat16)

                # 3 bf16 products each into the fp32 PSUM accumulator, using the SHARED
                # xT_hi/xT_lo stationary limbs (dropping the negligible xT_lo@w_lo term).
                up_acc[ix, ich] += nisa.nc_matmul(xT_hi[kt, ix, i128], w_up_hi[ix, ich])
                up_acc[ix, ich] += nisa.nc_matmul(xT_hi[kt, ix, i128], w_up_lo[ix, ich])
                up_acc[ix, ich] += nisa.nc_matmul(xT_lo[kt, ix, i128], w_up_hi[ix, ich])

                gate_acc[ix, ich] += nisa.nc_matmul(xT_hi[kt, ix, i128], w_gate_hi[ix, ich])
                gate_acc[ix, ich] += nisa.nc_matmul(xT_hi[kt, ix, i128], w_gate_lo[ix, ich])
                gate_acc[ix, ich] += nisa.nc_matmul(xT_lo[kt, ix, i128], w_gate_hi[ix, ich])

            # PSUM -> SBUF copy BEFORE activation (activation must not read a raw PSUM
            # bank; matches v1). SiLU + multiply stay fp32.
            up_sb = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
            up_sb[ix, ich] = nl.copy(up_acc[ix, ich], dtype=np.float32)
            gate_sb = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
            gate_sb[ix, ich] = nl.copy(gate_acc[ix, ich], dtype=np.float32)
            # Fused SiLU on the Scalar Engine: sg = gate * sigmoid(gate), one op.
            sg = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
            sg[ix, ich] = nisa.activation(op=nl.silu, data=gate_sb[ix, ich], dtype=np.float32)
            # h = SiLU(gate) * up  (Vector-Engine multiply) into the resident tile.
            h_sbuf[ix, CHUNK * c + ich] = nl.multiply(sg[ix, ich], up_sb[ix, ich])

        # ---- transpose h into 24 hT sub-tiles, split into bf16 limbs, then down ----
        # h_sbuf = [m_in(par), n(free)]; the down GEMM contracts over n, so transpose
        # each [128,128] N-sub-tile to hT = [n_in(par), m_in(free)] (exact fp32), then
        # split into hT_hi/hT_lo (bf16).
        hT_hi = nl.ndarray((N_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        hT_lo = nl.ndarray((N_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        for nt_ in nl.affine_range(N_TILES):
            psum_h = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
            psum_h[ix, i128] = nisa.nc_matmul(
                h_sbuf[ix, 128 * nt_ + i128],
                identity_local[ix, i128],
                is_transpose=True, is_moving_onezero=True)
            hT_f = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
            hT_f[ix, i128] = nl.copy(psum_h[ix, i128], dtype=np.float32)
            hT_hi[nt_, ix, i128] = nl.copy(hT_f[ix, i128], dtype=nl.bfloat16)
            hT_res = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
            hT_res[ix, i128] = nisa.tensor_tensor(
                hT_f[ix, i128], hT_hi[nt_, ix, i128], op=nl.subtract)
            hT_lo[nt_, ix, i128] = nl.copy(hT_res[ix, i128], dtype=nl.bfloat16)

        for c2 in nl.affine_range(K_OUT_CHUNKS):
            out_acc = nl.zeros((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.psum)
            for nt_ in nl.affine_range(N_TILES):
                # w_down chunk is [n_in(par)=128, 512(free)] fp32 — build its bf16 limbs
                # on-chip, transient per (c2, nt_).
                w_down_f = nl.load(v3[nt_, ix, CHUNK * c2 + ich], dtype=np.float32)
                w_down_hi = nl.ndarray((par_dim(128), CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                w_down_hi[ix, ich] = nl.copy(w_down_f[ix, ich], dtype=nl.bfloat16)
                w_down_res = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
                w_down_res[ix, ich] = nisa.tensor_tensor(
                    w_down_f[ix, ich], w_down_hi[ix, ich], op=nl.subtract)
                w_down_lo = nl.ndarray((par_dim(128), CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                w_down_lo[ix, ich] = nl.copy(w_down_res[ix, ich], dtype=nl.bfloat16)

                # 3 bf16 products into the fp32 PSUM accumulator, using the hT_hi/hT_lo
                # stationary limbs (dropping the negligible hT_lo@w_down_lo term).
                out_acc[ix, ich] += nisa.nc_matmul(hT_hi[nt_, ix, i128], w_down_hi[ix, ich])
                out_acc[ix, ich] += nisa.nc_matmul(hT_hi[nt_, ix, i128], w_down_lo[ix, ich])
                out_acc[ix, ich] += nisa.nc_matmul(hT_lo[nt_, ix, i128], w_down_hi[ix, ich])

            out_sb = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
            out_sb[ix, ich] = nl.copy(out_acc[ix, ich], dtype=np.float32)
            nl.store(v5[mt, ix, CHUNK * c2 + ich], value=out_sb[ix, ich])

    return v5
