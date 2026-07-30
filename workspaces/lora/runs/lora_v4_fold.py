import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2, v3, v4):
    """LoRA weight-fold: out = x@w + (x@a)@b == x@(w + a@b) == x@w', via one bf16x2 GEMM.

    Uses the algebraic identity to absorb the low-rank term into the weights:
        w' = w + a@b   (materialized ONCE to fp32 HBM, a (K,N) tensor)
        out = x @ w'   (the sibling matmul_v3_bf16_split base GEMM on the folded weights)
    So the fused down-projection x@a, up-projection (x@a)@b, resident-a, and tT machinery of
    the incumbent lora_v3_bf16_split are all DELETED -- the main loop is literally the
    shape-identical sibling matmul_v3_bf16_split reading w' instead of w.

    Shapes (M=4096, K=5120, N=12288, R=128) in the NKIBench reference's tiled layout:
        v1 (x): (8, 4, 128, 40, 128) = [m_hi, m_lo, m_in, k_tile, k_in]
        v2 (w): (40, 128, 12288)     = [k_tile, k_in, n]
        v3 (a): (40, 128, 128)       = [k_tile, k_in, r]
        v4 (b): (128, 12288)         = [r, n]
        out:    (8, 4, 128, 96, 128) = [m_hi, m_lo, m_in, n_tile, n_in]

    Why fold, and why it is a genuine coin-flip vs the incumbent. The incumbent D2
    (lora_v3_bf16_split) is cleanly PE-bound at the base-GEMM systolic floor; its ~0.647 ms
    low-rank tail (down-proj + fused up-proj) sits on top of the base GEMM (sibling
    matmul_v3 base wall 10.656 ms vs D2 11.3034 ms). The fold REMOVES that tail -- the main
    loop is exactly the sibling base GEMM -- but pays back the a@b materialization (40
    a-transposes + 960 fp32 moving-512 matmuls, ~0.2 ms PE, ADDITIVE on a PE-bound op) plus
    ~250 MB net HBM write (read w once + write w' once, on top of the main w' stream). The
    net is genuinely uncertain and settled only by measurement; the offline sim + a strict
    same-session A/B gate decide promote vs record-as-reject.

    Numeric argument (why the fold is safe but MUST be re-gated). D2 keeps the low-rank fp32,
    so the base split error (4.453e-6 in isolation) is DILUTED ~11.4x in the composite to
    3.93e-7 (the low-rank term carries 99.6% of the output magnitude). The fold routes the
    ENTIRE output -- including that dominant low-rank part -- through the single bf16x2 GEMM
    x@w', so its rel-L2 is the UNDILUTED pure-GEMM value ~4.45e-6. Predicted device
    quadrature sqrt(fp32_floor 4.874e-7^2 + fold 4.45e-6^2) ~= 4.48e-6, still ~4.5x under the
    2e-5 gate (offline sim runs/offline_lora_bf16_split_sim.py, route [F], authorized the
    fold at worst-over-seeds 4.456e-6 < 8e-6, with an fp32 reassociation control 6.08e-7 that
    confirms the folded error is bf16-dominated, not a reassociation artifact).

    Two hard correctness-structure guards:
      * w' is fp32 in HBM. w_prime is an fp32 shared_hbm scratch; the main loop splits it
        into bf16 limbs only at the GEMM consumption point (identical to how D2 splits w).
        Storing w' as bf16 would route the 99.6%-dominant low-rank term through one weak
        rounding -> catastrophic (~2.3e-3, fails the gate). Never downcast w' to bf16 in HBM.
      * RAW ordering. The prologue is a SEPARATE pass that fully materializes every w' tile
        to HBM before the main loop's first w' read. There is no speculative preload of any
        w' tile before its fp32 store completes; the two loop nests share no live w' tile.

    Prologue (a@b -> HBM), separate pass:
        aT[kt] = transpose(v3[kt]) = [R(par)=128, k_in(free)=128]  (identity-transpose idiom)
        per K-tile kt (40), per N-chunk c (24 of width 512):
            ab_chunk = nc_matmul(stationary=aT[kt] [R,k_in], moving=b_chunk [R,512])
                     = aT[kt].T @ b_chunk = [k_in,R] @ [R,512] = a[kt] @ b[:,cols] = [k_in,512]
            w_prime[kt, :, cols] = v2[kt, :, cols] + ab_chunk   (fp32 add, STORE fp32 to HBM)
    The prologue nests kt (outer) over c (inner); b_chunk [R,512] is loaded per (kt, c) pair
    (b is small, 6.29 MB, and the resident aT is reused across all 24 chunks of each kt). PSUM
    peak in the prologue: 1 transpose bank + 1 ab accumulator [128,512] -> <=2 of 8.

    Main loop = matmul_v3_bf16_split on w' (down/up-proj/resident-a/tT DELETED):
        per m_hi M-block (8 blocks, B=4 members == m_lo):
            build resident bf16 lhs limbs (lhs_hi, lhs_lo)[m_lo,kt]=[k_in,m_in] from the
            transient fp32 transpose scratch (no resident fp32 lhs_t survives)
            per N-chunk c (24 of width 512):
                per K-tile kt (40): load w'[kt,:,cols] fp32, split -> w'_hi, w'_lo (bf16)
                    per member m_lo (4): acc[m_lo] += lhs_hi@w'_hi + lhs_hi@w'_lo + lhs_lo@w'_hi
                store acc[m_lo] as 4 sub-tiles -> out[m_hi, m_lo, :, 4c+j, :]
    PSUM peak in the main loop: 4 base-acc banks [128,512] + 1 transient transpose bank
    = 5 of 8. The main-loop resident set is SMALLER than D2 (no resident a 20 KB, no tT), so
    SBUF fits comfortably (2 bf16 lhs limbs 80 KB + transients).

    nc_matmul(stationary, moving) = stationary.T @ moving; contraction on the partition axis
    of both operands, both SBUF-resident.
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
    N_CHUNK = 512              # wide base-GEMM chunk (one fp32 PSUM bank)
    N_CHUNKS = N // N_CHUNK     # 24
    B = M_LO                   # M-block factor == the 4 m_lo members
    SUBTILES = N_CHUNK // 128   # 4 output n_tiles per 512-wide chunk

    out = nl.ndarray((8, 4, 128, 96, 128), dtype=np.float32, buffer=nl.shared_hbm)

    # Internal fp32 HBM scratch for the folded weights w' = w + a@b, in the SAME
    # [k_tile, k_in, n] tiled layout as v2 (w). fp32 in HBM by construction -- the main loop
    # splits it into bf16 limbs only at the GEMM consumption point (never stored as bf16).
    w_prime = nl.ndarray((40, 128, 12288), dtype=np.float32, buffer=nl.shared_hbm)

    # 128x128 identity in SBUF: the moving operand that transposes tiles on the Tensor Engine
    # (is_transpose=True). Loaded once, reused for the a-transpose and the lhs transposes.
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]],
        dtype=np.float32)

    # ==================================================================================
    # PROLOGUE: materialize w' = w + a@b to fp32 HBM (a SEPARATE pass; every w' tile is
    # fully written before the main loop reads any w' tile -- RAW correctness).
    # ==================================================================================
    # a transposed and held resident: aT[kt] = v3[kt].T = [R(par)=128, k_in(free)=128].
    # a[kt] = v3[kt] = [k_in, R]; transposing puts the contraction dim R on the partition
    # axis so nc_matmul(aT[kt], b_chunk) = aT[kt].T @ b_chunk = a[kt] @ b_chunk contracts R.
    aT = nl.ndarray((K_TILES, par_dim(R), 128), dtype=np.float32, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        a_sb = nl.ndarray((par_dim(128), R), dtype=np.float32, buffer=nl.sbuf)
        a_sb[nl.arange(128)[:, None], nl.arange(R)[None, :]] = nl.load(
            v3[kt, nl.arange(128)[:, None], nl.arange(R)[None, :]], dtype=np.float32)
        aT_psum = nl.ndarray((par_dim(R), 128), dtype=np.float32, buffer=nl.psum)
        aT_psum[nl.arange(R)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
            a_sb[nl.arange(128)[:, None], nl.arange(R)[None, :]],
            identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
            is_transpose=True, is_moving_onezero=True)
        aT[kt, nl.arange(R)[:, None], nl.arange(128)[None, :]] = nl.copy(
            aT_psum[nl.arange(R)[:, None], nl.arange(128)[None, :]], dtype=np.float32)

    for kt in nl.affine_range(K_TILES):
        for c in nl.affine_range(N_CHUNKS):
            n0 = N_CHUNK * c
            # b chunk for this (kt-independent) N-block: [r(par)=128, 512(free)].
            b_chunk = nl.ndarray((par_dim(R), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
            b_chunk[nl.arange(R)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.load(
                v4[nl.arange(R)[:, None], n0 + nl.arange(N_CHUNK)[None, :]], dtype=np.float32)
            # ab_chunk = aT[kt].T @ b_chunk = a[kt] @ b[:,cols] = [k_in,512], fp32 PSUM.
            ab_psum = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
            ab_psum[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.nc_matmul(
                aT[kt, nl.arange(R)[:, None], nl.arange(128)[None, :]],
                b_chunk[nl.arange(R)[:, None], nl.arange(N_CHUNK)[None, :]])
            # w_chunk = w[kt,:,cols] fp32.
            w_chunk = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
            w_chunk[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.load(
                v2[kt, nl.arange(128)[:, None], n0 + nl.arange(N_CHUNK)[None, :]],
                dtype=np.float32)
            # w' = w + a@b, fp32 add into an fp32 SBUF tile, then STORE fp32 to HBM.
            wp_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
            wp_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.tensor_tensor(
                w_chunk[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                ab_psum[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                op=nl.add)
            nl.store(
                w_prime[kt, nl.arange(128)[:, None], n0 + nl.arange(N_CHUNK)[None, :]],
                value=wp_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

    # ==================================================================================
    # MAIN LOOP: out = x @ w' (the sibling matmul_v3_bf16_split base GEMM on w').
    # ==================================================================================
    for m_hi in nl.affine_range(M_HI):
        # Resident bf16 limbs for all B members of this M-block, built from the transient
        # fp32 transpose scratch (freed each iteration -- no resident fp32 lhs_t survives).
        #   lhs_hi[m_lo, kt] = lhs_lo[m_lo, kt] = [k_in(par)=128, m_in(free)=128]  (bf16)
        lhs_hi = nl.ndarray((B, K_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        lhs_lo = nl.ndarray((B, K_TILES, par_dim(128), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        for m_lo in nl.affine_range(B):
            for kt in nl.affine_range(K_TILES):
                lhs_sb = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
                lhs_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
                    v1[m_hi, m_lo, nl.arange(128)[:, None], kt, nl.arange(128)[None, :]],
                    dtype=np.float32)
                # Transpose x tile -> PSUM [k_in(par), m_in(free)], copy to bounded transient
                # fp32 scratch (one [128,128] tile, freed each iteration -- no resident fp32).
                psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
                psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                    lhs_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    is_transpose=True, is_moving_onezero=True)
                lhs_t_f = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
                lhs_t_f[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                    psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32)
                # Build the resident bf16 limbs of the base operand from lhs_t_f.
                lhs_hi[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                    lhs_t_f[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=nl.bfloat16)
                # lhs_lo = bf16(lhs_t - lhs_hi): exact fp32 residual, downcast to bf16.
                lhs_lo[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.tensor_tensor(
                    lhs_t_f[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    lhs_hi[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    op=nl.subtract)

        # Per N-chunk: base x@w' via the 3-product bf16 split into B distinct PSUM banks.
        for c in nl.affine_range(N_CHUNKS):
            n0 = N_CHUNK * c
            acc = nl.zeros((B, par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # Load this w' K-tile ONCE, split into bf16 limbs, reuse across all B members.
                # w' is fp32 in HBM; the split to bf16 happens HERE, at the consumption point.
                wp_chunk = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
                wp_chunk[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.load(
                    w_prime[kt, nl.arange(128)[:, None], n0 + nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)
                wp_hi = nl.ndarray((par_dim(128), N_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                wp_hi[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.copy(
                    wp_chunk[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    dtype=nl.bfloat16)
                wp_lo = nl.ndarray((par_dim(128), N_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                wp_lo[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.tensor_tensor(
                    wp_chunk[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    wp_hi[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    op=nl.subtract)
                for m_lo in nl.affine_range(B):
                    # Pinned 3-product order into the member's fp32 PSUM bank; drop lo@lo.
                    acc[m_lo, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                        lhs_hi[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        wp_hi[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])
                    acc[m_lo, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                        lhs_hi[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        wp_lo[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])
                    acc[m_lo, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                        lhs_lo[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        wp_hi[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

            for m_lo in nl.affine_range(B):
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
