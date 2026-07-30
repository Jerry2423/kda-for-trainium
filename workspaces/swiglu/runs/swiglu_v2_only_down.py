# swiglu_v2_only_down — fallback-ladder rung: bf16x2 split on the DOWN GEMM ONLY.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (swiglu_M4096_N3072_K1024_0.py) = 2.074257 ms.
# Parent: swiglu_v1 (fp32). This is v1 with the compensated bf16x2 3-product split
# applied ONLY to the down GEMM (h @ w_down); the up and gate GEMMs stay pure fp32
# exactly as in v1.
#
# Why this rung exists: swiglu_v2 (bf16x2 on ALL three GEMMs) was numerically correct
# but 2.3x SLOWER — its all-GEMM split raised PE-active AND exposed a large per-M-tile
# weight-limb rebuild on Vec/Scl. This only-down rung converts exactly ONE of the three
# GEMMs to bf16x2 (offline distinct-seed worst rel-L2 = 4.45e-6, the smallest/safest ladder
# rung). Measured: 2.1372 ms / 0.971x — correct, and its true PE-active (1.95 ms) is actually
# LOWER than v1's fp32 2.09 ms (the down-GEMM bf16x2 is PE-cheaper than fp32 here, and keeping
# up/gate fp32 avoids most limb rebuild). But it is DMA-BOUND (DMA=100%, HBMrd 656 MB): at B=1
# the fp32 w_down chunk is reloaded per M-tile to rebuild its bf16 limbs on-chip. That sub-
# baseline PE floor with a DMA bottleneck is exactly what the M-block regime (swiglu_v3_mblock)
# amortizes — B=4 M-blocking of this kernel reaches 2.0219 ms / 1.026x (PROMOTED), the first
# kernel to beat baseline. This file is the B=1 parent of that win.
#
#   up   = x @ w_up                          # (M,N)  — fp32 (unchanged from v1)
#   gate = x @ w_gate                        # (M,N)  — fp32 (unchanged from v1)
#   h    = (gate * sigmoid(gate)) * up       # SiLU(gate) * up, fp32 (unchanged)
#   out  = h @ w_down                        # (M,K)  — bf16x2 3-product split
#
# with M=4096, K=1024, N=3072, all fp32 I/O.
#
# The down GEMM split: hT is transposed exactly (fp32 identity matmul) then split once
# into hT_hi/hT_lo (bf16); each w_down chunk is loaded fp32 and split on-chip into
# w_down_hi/w_down_lo (bf16); the accumulator sums the 3 bf16 products
# hT_hi@w_down_hi + hT_hi@w_down_lo + hT_lo@w_down_hi in one fp32 PSUM bank (dropping the
# negligible lo@lo cross term). Transpose, PSUM accumulation, SiLU, multiply stay fp32.

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
    N_CHUNKS = N // CHUNK          # 6   (up/gate output tiling)
    K_OUT_CHUNKS = K_OUT // CHUNK  # 2   (down-GEMM output tiling)

    ix = nl.arange(128)[:, None]      # partition index (m_in / k_in / n_in)
    i128 = nl.arange(128)[None, :]    # 128-wide free index (sub-tile / transpose)
    ich = nl.arange(CHUNK)[None, :]   # 512-wide N/K-out chunk free index

    x3 = v1.reshape((M_TILES, 128, 1024))
    v5 = nl.ndarray((M_TILES, 128, K_OUT), dtype=np.float32, buffer=nl.shared_hbm)

    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[ix, i128] = nl.load(identity_const[ix, i128], dtype=np.float32)

    for mt in nl.affine_range(M_TILES):
        # ---- load x tile and transpose ONCE into 8 shared fp32 xT sub-tiles (as v1) ----
        x_sb = nl.load(x3[mt, ix, nl.arange(1024)[None, :]], dtype=np.float32)
        xT = nl.ndarray((K_TILES, par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
            psum_t[ix, i128] = nisa.nc_matmul(
                x_sb[ix, 128 * kt + i128],
                identity_local[ix, i128],
                is_transpose=True, is_moving_onezero=True)
            xT[kt, ix, i128] = nl.copy(psum_t[ix, i128], dtype=np.float32)

        # ---- up + gate projections in fp32 (unchanged from v1), SiLU fused at eviction ----
        h_sbuf = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        for c in nl.affine_range(N_CHUNKS):
            up_acc = nl.zeros((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.psum)
            gate_acc = nl.zeros((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                w_up_sb = nl.load(v2[kt, ix, CHUNK * c + ich], dtype=np.float32)
                w_gate_sb = nl.load(v4[kt, ix, CHUNK * c + ich], dtype=np.float32)
                up_acc[ix, ich] += nisa.nc_matmul(xT[kt, ix, i128], w_up_sb[ix, ich])
                gate_acc[ix, ich] += nisa.nc_matmul(xT[kt, ix, i128], w_gate_sb[ix, ich])

            up_sb = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
            up_sb[ix, ich] = nl.copy(up_acc[ix, ich], dtype=np.float32)
            gate_sb = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
            gate_sb[ix, ich] = nl.copy(gate_acc[ix, ich], dtype=np.float32)
            sg = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
            sg[ix, ich] = nisa.activation(op=nl.silu, data=gate_sb[ix, ich], dtype=np.float32)
            h_sbuf[ix, CHUNK * c + ich] = nl.multiply(sg[ix, ich], up_sb[ix, ich])

        # ---- transpose h into 24 hT sub-tiles, split into bf16 limbs (down GEMM only) ----
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
                w_down_f = nl.load(v3[nt_, ix, CHUNK * c2 + ich], dtype=np.float32)
                w_down_hi = nl.ndarray((par_dim(128), CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                w_down_hi[ix, ich] = nl.copy(w_down_f[ix, ich], dtype=nl.bfloat16)
                w_down_res = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
                w_down_res[ix, ich] = nisa.tensor_tensor(
                    w_down_f[ix, ich], w_down_hi[ix, ich], op=nl.subtract)
                w_down_lo = nl.ndarray((par_dim(128), CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                w_down_lo[ix, ich] = nl.copy(w_down_res[ix, ich], dtype=nl.bfloat16)

                out_acc[ix, ich] += nisa.nc_matmul(hT_hi[nt_, ix, i128], w_down_hi[ix, ich])
                out_acc[ix, ich] += nisa.nc_matmul(hT_hi[nt_, ix, i128], w_down_lo[ix, ich])
                out_acc[ix, ich] += nisa.nc_matmul(hT_lo[nt_, ix, i128], w_down_hi[ix, ich])

            out_sb = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
            out_sb[ix, ich] = nl.copy(out_acc[ix, ich], dtype=np.float32)
            nl.store(v5[mt, ix, CHUNK * c2 + ich], value=out_sb[ix, ich])

    return v5
