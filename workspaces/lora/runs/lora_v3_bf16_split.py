import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2, v3, v4):
    """Compensated bf16x2 3-product split of the BASE GEMM in lora  out = x@w + (x@a)@b.

    Localized diff on the fp32 M-block kernel (lora_v2_mblk4). Same tiled contract, same
    B=4 M-block (M-block == m_hi, the 4 members == m_lo), same N_CHUNK=512, same 4-sub-tile
    store. Only the BASE GEMM x@w changes precision; the down-projection x@a and the fused
    up-projection (x@a)@b stay fp32.

    Shapes (M=4096, K=5120, N=12288, R=128) in the NKIBench reference's tiled layout:
        v1 (x): (8, 4, 128, 40, 128) = [m_hi, m_lo, m_in, k_tile, k_in]
        v2 (w): (40, 128, 12288)     = [k_tile, k_in, n]
        v3 (a): (40, 128, 128)       = [k_tile, k_in, r]
        v4 (b): (128, 12288)         = [r, n]
        v5 (out): (8, 4, 128, 96, 128) = [m_hi, m_lo, m_in, n_tile, n_in]

    Why split only the base. The base GEMM x@w is 96.6% of the MACs; lora_v2_mblk4 is
    PE-bound at the fp32 systolic floor (PE 98%, MFU 48%, DMA idle 33%) for this base GEMM,
    which is shape-identical to the sibling matmul (matmul_v3_bf16_split took the same base
    from 1.017x to 1.274x). The trn2 PE array is bf16-native and emulates fp32 at ~2 passes
    at ~1.8-2x the per-instruction rate, so replacing one fp32 base matmul with three
    bf16-rate products (dropping the negligible lo@lo cross term) lowers TRUE PE-active
    (sibling RAW PE-active -24% on this base). The low-rank path is only 3.4% of MACs AND
    carries 99.6% of the output magnitude, so keeping it fp32 costs almost nothing and
    removes all doubt; the offline sim (runs/offline_lora_bf16_split_sim.py) authorizes the
    base-only split at composite rel-L2 3.930e-7 (base-only 4.453e-6 diluted 11.4x), device
    quadrature ~6.26e-7 (the fp32 floor 4.874e-7 DOMINATES) -- ~32x under the 2e-5 gate.

    Base split (per member m_lo, per K-tile kt):
        lhs_hi = bf16(lhs_t),  lhs_lo = bf16(lhs_t - lhs_hi)   (round-to-nearest-even)
        w_hi   = bf16(w_chunk), w_lo   = bf16(w_chunk - w_hi)
        acc[m_lo] += lhs_hi@w_hi + lhs_hi@w_lo + lhs_lo@w_hi   (base x@w only)
    The low limb is produced by a residual subtract into a bf16 destination
    (nisa.tensor_tensor upcasts the mixed fp32/bf16 operands to fp32 internally and
    downcasts the result -- no separate fp32 residual buffer).

    Two hard correctness-structure guards vs a naive port:
      * Down-projection operand lifetime. The sibling matmul_v3_bf16_split keeps NO resident
        fp32 lhs_t (only the two bf16 limbs), so lora's fp32 down-projection x@a would have
        no fp32 operand. Fix: fold the fp32 down-proj accumulation INTO the transpose/split
        loop, consuming the transient fp32 lhs_t_f (the copy of the transpose PSUM tile)
        BEFORE the bf16 limbs are built from it and it is freed. No full resident fp32 lhs_t
        survives; SBUF ~= 108 KB/partition (2 bf16 limbs 80 KB, same bytes as D1's fp32
        lhs_t, + a 20 KB + tT 2 KB + transients ~6 KB).
      * tT_psum accumulated EXACTLY ONCE per M-block. tT_psum[m_lo] is zeroed once per
        member and K-accumulated over the 40 K-tiles in the M-block PROLOGUE, before the
        N-chunk loop, then copied to fp32 SBUF (tT). The N-chunk loop only READS the
        completed tT (as tT.T @ b_chunk). Accumulating tT_psum inside the N-chunk loop would
        re-run the down-projection 24x (for N_CHUNK=512 over N=12288) and silently corrupt
        the output while looking structurally plausible.

    nc_matmul(stationary, moving) = stationary.T @ moving; contraction (k_in resp. r) on the
    partition axis of both operands, both SBUF-resident. PSUM: 4 base-acc banks ([128,512]
    fp32) during the N loop; 1 transpose bank + 1 down-proj accumulator bank during the
    prologue -> peak 4-5 of 8.
    """
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    M_HI, M_LO = 8, 4          # 8*4 = 32 M-tiles of 128 rows
    K_TILES = 40               # 5120 / 128
    R = 128                    # low-rank dim
    N = 12288
    N_CHUNK = 512              # wide base-GEMM chunk
    N_CHUNKS = N // N_CHUNK     # 24
    B = M_LO                   # M-block factor == the 4 m_lo members
    SUBTILES = N_CHUNK // 128   # 4 output n_tiles per 512-wide chunk

    out = nl.ndarray((8, 4, 128, 96, 128), dtype=np.float32, buffer=nl.shared_hbm)

    # 128x128 identity in SBUF: the moving operand that transposes lhs tiles on the
    # Tensor Engine (is_transpose=True). Loaded once, reused for all tiles.
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]],
        dtype=np.float32)

    # a fully resident in SBUF, fp32: a_local[kt] = v3[kt] = [k_in(par), r(free)]. The
    # stationary operand of the fp32 down-projection, reused across all M-blocks.
    a_local = nl.ndarray((K_TILES, par_dim(128), R), dtype=np.float32, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        a_local[kt, nl.arange(128)[:, None], nl.arange(R)[None, :]] = nl.load(
            v3[kt, nl.arange(128)[:, None], nl.arange(R)[None, :]], dtype=np.float32)

    for m_hi in nl.affine_range(M_HI):
        # Prologue for all B members of this M-block. For each member we transpose x per
        # K-tile, and from the transient fp32 transpose scratch we (a) accumulate the fp32
        # down-projection tT_psum BEFORE freeing it, then (b) build the resident bf16 limbs.
        #   lhs_hi[m_lo, kt] = lhs_lo[m_lo, kt] = [k_in(par)=128, m_in(free)=128]  (bf16)
        #   tT[m_lo]         = [R(par)=128, m_in(free)=128]  (fp32, completed before N loop)
        lhs_hi = nl.ndarray((B, K_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        lhs_lo = nl.ndarray((B, K_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        tT = nl.ndarray((B, par_dim(R), 128), dtype=np.float32, buffer=nl.sbuf)
        for m_lo in nl.affine_range(B):
            # Down-projection accumulator: zeroed ONCE per member, K-accumulated over the
            # 40 K-tiles in this prologue only. The N loop never touches it again.
            tT_psum = nl.zeros((par_dim(R), 128), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # Load x tile from HBM into SBUF: [m_in(par)=128, k_in(free)=128].
                lhs_sb = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
                lhs_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
                    v1[m_hi, m_lo, nl.arange(128)[:, None], kt, nl.arange(128)[None, :]],
                    dtype=np.float32)
                # Transpose -> PSUM [k_in(par), m_in(free)], then copy to a bounded transient
                # fp32 scratch (one [128,128] tile, freed each iteration -- no resident fp32).
                psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
                psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                    lhs_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    is_transpose=True, is_moving_onezero=True)
                lhs_t_f = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
                lhs_t_f[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                    psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32)

                # (a) fp32 down-projection FIRST, consuming the fp32 lhs_t_f while it lives.
                # nc_matmul(stationary=a_local[kt] [k_in,R], moving=lhs_t_f [k_in,m_in])
                #   = a_local[kt].T @ lhs_t_f = [R,k_in] @ [k_in,m_in] = [R, m_in].
                tT_psum[nl.arange(R)[:, None], nl.arange(128)[None, :]] += nisa.nc_matmul(
                    a_local[kt, nl.arange(128)[:, None], nl.arange(R)[None, :]],
                    lhs_t_f[nl.arange(128)[:, None], nl.arange(128)[None, :]])

                # (b) build the resident bf16 limbs of the base operand from the same lhs_t_f.
                lhs_hi[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                    lhs_t_f[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=nl.bfloat16)
                # lhs_lo = bf16(lhs_t - lhs_hi): exact fp32 residual, downcast to bf16.
                lhs_lo[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.tensor_tensor(
                    lhs_t_f[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    lhs_hi[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    op=nl.subtract)

            # Copy the completed down-projection to fp32 SBUF BEFORE the N loop (the N loop
            # reuses PSUM banks for the base accumulators).
            tT[m_lo, nl.arange(R)[:, None], nl.arange(128)[None, :]] = nl.copy(
                tT_psum[nl.arange(R)[:, None], nl.arange(128)[None, :]], dtype=np.float32)

        # Per N-chunk: base x@w via the 3-product bf16 split into B distinct banks + fused
        # fp32 low-rank (x@a)@b.
        for c in nl.affine_range(N_CHUNKS):
            n0 = N_CHUNK * c
            acc = nl.zeros((B, par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # Load this w K-tile ONCE, split into bf16 limbs, reuse across all B members.
                w_chunk = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
                w_chunk[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.load(
                    v2[kt, nl.arange(128)[:, None], n0 + nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)
                w_hi = nl.ndarray((par_dim(128), N_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                w_hi[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.copy(
                    w_chunk[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    dtype=nl.bfloat16)
                w_lo = nl.ndarray((par_dim(128), N_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                w_lo[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.tensor_tensor(
                    w_chunk[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    w_hi[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    op=nl.subtract)
                for m_lo in nl.affine_range(B):
                    # Pinned 3-product order into the member's fp32 PSUM bank; drop lo@lo.
                    acc[m_lo, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                        lhs_hi[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        w_hi[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])
                    acc[m_lo, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                        lhs_hi[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        w_lo[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])
                    acc[m_lo, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                        lhs_lo[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        w_hi[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

            # Fuse the fp32 low-rank residual into each member's bank (no HBM round-trip).
            # Load b chunk ONCE, reuse across all B members: [r(par)=128, 512(free)].
            b_chunk = nl.ndarray((par_dim(R), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
            b_chunk[nl.arange(R)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.load(
                v4[nl.arange(R)[:, None], n0 + nl.arange(N_CHUNK)[None, :]], dtype=np.float32)
            for m_lo in nl.affine_range(B):
                # Single fp32 nc_matmul: tT.T @ b_chunk = (x@a) @ b = [m_in,512], same bank.
                acc[m_lo, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                    tT[m_lo, nl.arange(R)[:, None], nl.arange(128)[None, :]],
                    b_chunk[nl.arange(R)[:, None], nl.arange(N_CHUNK)[None, :]])

                out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
                out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.copy(
                    acc[m_lo, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)
                # 512-wide -> 4 sub-tile stores into the reshaped N axis.
                for j in nl.static_range(SUBTILES):
                    nl.store(
                        out[m_hi, m_lo, nl.arange(128)[:, None], SUBTILES * c + j,
                            nl.arange(128)[None, :]],
                        value=out_sb[nl.arange(128)[:, None],
                                     128 * j + nl.arange(128)[None, :]])

    return out
