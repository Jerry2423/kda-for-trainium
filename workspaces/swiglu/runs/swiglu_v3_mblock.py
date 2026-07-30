# swiglu_v3_mblock — M-tile-block regime layered on the down-GEMM bf16x2 split.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (swiglu_M4096_N3072_K1024_0.py) = 2.074257 ms.
# Parent: swiglu_v2_only_down (bf16x2 on the down GEMM only; 2.1372 ms / 0.971x).
#
# swiglu_v2_only_down is DMA-bound (DMA=100%, HBMrd 656 MB) with a true PE-active floor
# of ~1.95 ms — below the 2.074 ms baseline. Its DMA pressure comes from streaming each
# fp32 weight chunk once PER M-tile (B=1) and, for the down GEMM, reloading the fp32
# w_down chunk to rebuild its bf16 limbs on-chip. This kernel processes B M-tiles inside
# one weight-chunk stream so each fp32 weight chunk is loaded ONCE per M-block and reused
# across the B M-tiles (and the down-GEMM bf16 limbs are built once per chunk per M-block),
# cutting weight-DMA volume ~B-fold and pushing latency toward the PE floor.
#
#   up   = x @ w_up                          # (M,N)  — fp32
#   gate = x @ w_gate                        # (M,N)  — fp32
#   h    = (gate * sigmoid(gate)) * up       # SiLU(gate) * up, fp32
#   out  = h @ w_down                        # (M,K)  — bf16x2 3-product split
#
# with M=4096, K=1024, N=3072, all fp32 I/O. Set M_BLOCK below (2 or 4).
#
# PSUM budget: the up/gate phase uses 2 fp32 PSUM banks per M-tile (up_acc + gate_acc),
# so B M-tiles need 2B of the 8 banks -> B <= 4. The down phase reuses banks after up/gate
# are evicted. Per-M-tile resident activation state: h_sbuf [128,3072] fp32 (12 KB/part) +
# hT_hi/hT_lo [24,128,128] bf16 (12 KB/part) + xT [8,128,128] fp32 (4 KB/part) ~= 28 KB/part;
# at B=4 that is ~112 KB/part, within the ~200 KB budget.
#
# Loop structure (M-block outer): for each block of B M-tiles,
#   1. load+transpose x for all B tiles -> B * 8 fp32 xT sub-tiles (shared by up+gate).
#   2. up/gate: for each N-chunk, for each K-tile, load w_up/w_gate chunk ONCE and matmul
#      into all B M-tiles' up_acc/gate_acc; evict per (block-tile, N-chunk) into h_sbuf[b].
#   3. transpose+split h for all B tiles -> B * 24 hT_hi/hT_lo bf16 sub-tiles.
#   4. down: for each K-out chunk, for each N-tile, load w_down chunk ONCE, build its bf16
#      limbs ONCE, and issue the 3 bf16 products into all B M-tiles' out_acc; store.

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim

M_BLOCK = 4   # M-tiles processed per weight stream (2 or 4; PSUM caps at 4)


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
    B = M_BLOCK
    M_BLOCKS = M_TILES // B
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

    for mb in nl.affine_range(M_BLOCKS):
        # ---- 1. load x and transpose ONCE per M-tile into B*8 shared fp32 xT sub-tiles ----
        xT = nl.ndarray((B, K_TILES, par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
        for b in nl.affine_range(B):
            mt = mb * B + b
            x_sb = nl.load(x3[mt, ix, nl.arange(1024)[None, :]], dtype=np.float32)
            for kt in nl.affine_range(K_TILES):
                psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
                psum_t[ix, i128] = nisa.nc_matmul(
                    x_sb[ix, 128 * kt + i128],
                    identity_local[ix, i128],
                    is_transpose=True, is_moving_onezero=True)
                xT[b, kt, ix, i128] = nl.copy(psum_t[ix, i128], dtype=np.float32)

        # ---- 2. up/gate (fp32): load each w chunk ONCE, matmul into all B M-tiles ----
        # h_sbuf holds SiLU(gate)*up for all B M-tiles resident: [B,128,3072] fp32.
        h_sbuf = nl.ndarray((B, par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        for c in nl.affine_range(N_CHUNKS):
            up_acc = nl.zeros((B, par_dim(128), CHUNK), dtype=np.float32, buffer=nl.psum)
            gate_acc = nl.zeros((B, par_dim(128), CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # each fp32 weight chunk loaded ONCE per M-block, reused across the B tiles
                w_up_sb = nl.load(v2[kt, ix, CHUNK * c + ich], dtype=np.float32)
                w_gate_sb = nl.load(v4[kt, ix, CHUNK * c + ich], dtype=np.float32)
                for b in nl.affine_range(B):
                    up_acc[b, ix, ich] += nisa.nc_matmul(xT[b, kt, ix, i128], w_up_sb[ix, ich])
                    gate_acc[b, ix, ich] += nisa.nc_matmul(xT[b, kt, ix, i128], w_gate_sb[ix, ich])

            for b in nl.affine_range(B):
                up_sb = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
                up_sb[ix, ich] = nl.copy(up_acc[b, ix, ich], dtype=np.float32)
                gate_sb = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
                gate_sb[ix, ich] = nl.copy(gate_acc[b, ix, ich], dtype=np.float32)
                sg = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
                sg[ix, ich] = nisa.activation(op=nl.silu, data=gate_sb[ix, ich], dtype=np.float32)
                h_sbuf[b, ix, CHUNK * c + ich] = nl.multiply(sg[ix, ich], up_sb[ix, ich])

        # ---- 3. transpose+split h for all B M-tiles -> B*24 hT_hi/hT_lo bf16 sub-tiles ----
        hT_hi = nl.ndarray((B, N_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        hT_lo = nl.ndarray((B, N_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        for b in nl.affine_range(B):
            for nt_ in nl.affine_range(N_TILES):
                psum_h = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
                psum_h[ix, i128] = nisa.nc_matmul(
                    h_sbuf[b, ix, 128 * nt_ + i128],
                    identity_local[ix, i128],
                    is_transpose=True, is_moving_onezero=True)
                hT_f = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
                hT_f[ix, i128] = nl.copy(psum_h[ix, i128], dtype=np.float32)
                hT_hi[b, nt_, ix, i128] = nl.copy(hT_f[ix, i128], dtype=nl.bfloat16)
                hT_res = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
                hT_res[ix, i128] = nisa.tensor_tensor(
                    hT_f[ix, i128], hT_hi[b, nt_, ix, i128], op=nl.subtract)
                hT_lo[b, nt_, ix, i128] = nl.copy(hT_res[ix, i128], dtype=nl.bfloat16)

        # ---- 4. down (bf16x2): load each w_down chunk ONCE, build its limbs ONCE per ----
        # M-block, and issue the 3 bf16 products into all B M-tiles' accumulators.
        for c2 in nl.affine_range(K_OUT_CHUNKS):
            out_acc = nl.zeros((B, par_dim(128), CHUNK), dtype=np.float32, buffer=nl.psum)
            for nt_ in nl.affine_range(N_TILES):
                w_down_f = nl.load(v3[nt_, ix, CHUNK * c2 + ich], dtype=np.float32)
                w_down_hi = nl.ndarray((par_dim(128), CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                w_down_hi[ix, ich] = nl.copy(w_down_f[ix, ich], dtype=nl.bfloat16)
                w_down_res = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
                w_down_res[ix, ich] = nisa.tensor_tensor(
                    w_down_f[ix, ich], w_down_hi[ix, ich], op=nl.subtract)
                w_down_lo = nl.ndarray((par_dim(128), CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                w_down_lo[ix, ich] = nl.copy(w_down_res[ix, ich], dtype=nl.bfloat16)
                for b in nl.affine_range(B):
                    out_acc[b, ix, ich] += nisa.nc_matmul(hT_hi[b, nt_, ix, i128], w_down_hi[ix, ich])
                    out_acc[b, ix, ich] += nisa.nc_matmul(hT_hi[b, nt_, ix, i128], w_down_lo[ix, ich])
                    out_acc[b, ix, ich] += nisa.nc_matmul(hT_lo[b, nt_, ix, i128], w_down_hi[ix, ich])

            for b in nl.affine_range(B):
                mt = mb * B + b
                out_sb = nl.ndarray((par_dim(128), CHUNK), dtype=np.float32, buffer=nl.sbuf)
                out_sb[ix, ich] = nl.copy(out_acc[b, ix, ich], dtype=np.float32)
                nl.store(v5[mt, ix, CHUNK * c2 + ich], value=out_sb[ix, ich])

    return v5
