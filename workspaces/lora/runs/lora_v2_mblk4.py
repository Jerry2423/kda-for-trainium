import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2, v3, v4):
    """Fused low-rank residual GEMM  out = x@w + (x@a)@b  (fp32), M-blocked B=4.

    Shapes (M=4096, K=5120, N=12288, R=128) in the NKIBench reference's tiled layout:
        v1 (x): (8, 4, 128, 40, 128) = [m_hi, m_lo, m_in, k_tile, k_in]
                row m = (m_hi*4 + m_lo)*128 + m_in, col k = k_tile*128 + k_in
        v2 (w): (40, 128, 12288)     = [k_tile, k_in, n]   (contraction k_in on partition)
        v3 (a): (40, 128, 128)       = [k_tile, k_in, r]   (contraction k_in on partition)
        v4 (b): (128, 12288)         = [r, n]              (contraction r on partition)
    Output:
        v5 (out): (8, 4, 128, 96, 128) = [m_hi, m_lo, m_in, n_tile, n_in]
                  row m = (m_hi*4 + m_lo)*128 + m_in, col n = n_tile*128 + n_in

    Optimization vs the phase-1 kernel (lora_v1). lora_v1 was M-outer over all 32 M-tiles
    with N_CHUNK=128, re-streaming all of w (and b) from HBM once per M-tile -> HBM read
    7813 MB (~32x the single-pass ideal), PE-bound at MFU 18%. The base GEMM x@w
    (M4096/N12288/K5120) is 96.6% of the MACs and is shape-identical to the sibling
    matmul operator, so this kernel ports matmul_v2_b4's fp32 M-block recipe and grafts
    the cheap fused low-rank residual on top:

      * N_CHUNK=512 (4x wider, 4x fewer base matmuls than lora_v1's 128).
      * M-block of B members processed together, so each w K-tile [k_in,512] and the
        b chunk [r,512] are loaded from HBM ONCE per N-chunk and reused across all B
        members' distinct [128,512] fp32 PSUM banks, instead of being re-read per M-tile.
        This collapses the w/b reload traffic ~B-fold (predicted HBM read ~2.1 GB).

    The 2-level (m_hi, m_lo) M-index makes the block a natural, arithmetic-free fit: the
    M-block IS m_hi (8 blocks) and the B=4 members ARE m_lo (0..3) -- no flat-index
    floor-div/mod, no divisibility concern (it is structurally 8x4).

    Per m_hi block:
      1. Transpose x once per member (shared by both GEMMs): lhs_t[m_lo, kt] = [k_in, m_in]
         via the identity nc_matmul(is_transpose=True) idiom.
      2. Down-projection tT[m_lo] = (x@a)^T = [R, m_in], K-accumulated into a zero-init
         PSUM tile then copied to fp32 SBUF BEFORE the N loop (the N loop reuses PSUM banks
         for the base accumulators). One tT per member, all B resident.
      3. Per N-chunk (24 of width 512): the base x@w is K-accumulated into B distinct PSUM
         banks (w K-tile loaded once, reused across the B members), then the low-rank
         residual (x@a)@b = tT.T @ b_chunk is FUSED into each member's bank (b chunk loaded
         once, reused across the B members). The (x@a)/(x@a)@b intermediate never touches
         HBM. The 512-wide result is stored as 4 sub-tile writes into v5's reshaped N axis.

    nc_matmul(stationary, moving) = stationary.T @ moving; contraction (k_in resp. r) on
    the partition axis of both operands, both SBUF-resident. Pure fp32 throughout (matches
    the sibling matmul_v2_b4, which passed the same 2e-5 gate). PSUM: 4 base-acc banks
    ([128,512] fp32) during the N loop; 1 transpose bank + 1 down-proj bank during the
    prologue -> peak 4-5 of 8. SBUF: fp32 lhs_t (4*40*128*4 = 80 KB/part) + a (20 KB) +
    tT (2 KB) + transients (~6 KB) ~= 108 KB/part < ~192 KB.
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
    N_CHUNK = 512              # wide base-GEMM chunk (4x lora_v1's 128)
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

    # a fully resident in SBUF: a_local[kt] = v3[kt] = [k_in(par), r(free)], reused across
    # all M-blocks (~20 KB/partition). The stationary operand of the down-projection.
    a_local = nl.ndarray((K_TILES, par_dim(128), R), dtype=np.float32, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        a_local[kt, nl.arange(128)[:, None], nl.arange(R)[None, :]] = nl.load(
            v3[kt, nl.arange(128)[:, None], nl.arange(R)[None, :]], dtype=np.float32)

    for m_hi in nl.affine_range(M_HI):
        # 1. Transpose x once per member (shared by both GEMMs), and 2. the fp32
        # down-projection tT[m_lo] = (x@a)^T for all B members of this M-block.
        #   lhs_t[m_lo, kt] = [k_in(par)=128, m_in(free)=128]
        #   tT[m_lo]        = [R(par)=128, m_in(free)=128]
        lhs_t = nl.ndarray((B, K_TILES, par_dim(128), 128), dtype=np.float32,
                           buffer=nl.sbuf)
        tT = nl.ndarray((B, par_dim(R), 128), dtype=np.float32, buffer=nl.sbuf)
        for m_lo in nl.affine_range(B):
            for kt in nl.affine_range(K_TILES):
                # Load x tile from HBM into SBUF: [m_in(par)=128, k_in(free)=128].
                lhs_sb = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
                lhs_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
                    v1[m_hi, m_lo, nl.arange(128)[:, None], kt, nl.arange(128)[None, :]],
                    dtype=np.float32)
                # Transpose -> PSUM [k_in(par), m_in(free)], then copy to SBUF.
                psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
                psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                    lhs_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    is_transpose=True, is_moving_onezero=True)
                lhs_t[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                    psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    dtype=np.float32)

            # Down-projection tT[m_lo] = (x@a)^T = [R, m_in], K-accumulated into a
            # zero-init PSUM tile then copied to fp32 SBUF BEFORE the N loop.
            # nc_matmul(stationary=a_local[kt] [k_in,R], moving=lhs_t[m_lo,kt] [k_in,m_in])
            #   = a_local[kt].T @ lhs_t = [R,k_in] @ [k_in,m_in] = [R, m_in].
            tT_psum = nl.zeros((par_dim(R), 128), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                tT_psum[nl.arange(R)[:, None], nl.arange(128)[None, :]] += nisa.nc_matmul(
                    a_local[kt, nl.arange(128)[:, None], nl.arange(R)[None, :]],
                    lhs_t[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]])
            tT[m_lo, nl.arange(R)[:, None], nl.arange(128)[None, :]] = nl.copy(
                tT_psum[nl.arange(R)[:, None], nl.arange(128)[None, :]], dtype=np.float32)

        # 3. Per N-chunk: base x@w into B distinct banks + fused low-rank (x@a)@b.
        for c in nl.affine_range(N_CHUNKS):
            n0 = N_CHUNK * c
            acc = nl.zeros((B, par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # Load this w K-tile ONCE, reuse across all B block members.
                w_chunk = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
                w_chunk[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.load(
                    v2[kt, nl.arange(128)[:, None], n0 + nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)
                for m_lo in nl.affine_range(B):
                    # nc_matmul(stationary=lhs_t[m_lo,kt] [k_in,m_in], moving=w_chunk [k_in,512])
                    #   = [m_in,k_in] @ [k_in,512] = [m_in,512]  (base x@w)
                    acc[m_lo, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                        lhs_t[m_lo, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        w_chunk[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

            # Fuse the low-rank residual into each member's bank (no HBM round-trip).
            # Load b chunk ONCE, reuse across all B block members: [r(par)=128, 512(free)].
            b_chunk = nl.ndarray((par_dim(R), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
            b_chunk[nl.arange(R)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.load(
                v4[nl.arange(R)[:, None], n0 + nl.arange(N_CHUNK)[None, :]], dtype=np.float32)
            for m_lo in nl.affine_range(B):
                # nc_matmul(stationary=tT[m_lo] [r,m_in], moving=b_chunk [r,512])
                #   = tT.T @ b_chunk = (x@a) @ b = [m_in,512]  -> same layout, same bank.
                acc[m_lo, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                    tT[m_lo, nl.arange(R)[:, None], nl.arange(128)[None, :]],
                    b_chunk[nl.arange(R)[:, None], nl.arange(N_CHUNK)[None, :]])

                out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
                out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.copy(
                    acc[m_lo, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)
                # 512-wide -> 4 sub-tile stores: out_sb[:,128j:128j+128] into the reshaped
                # N axis v5[m_hi, m_lo, :, 4c+j, :] (chunk c covers n_tiles [4c, 4c+4)).
                for j in nl.static_range(SUBTILES):
                    nl.store(
                        out[m_hi, m_lo, nl.arange(128)[:, None], SUBTILES * c + j,
                            nl.arange(128)[None, :]],
                        value=out_sb[nl.arange(128)[:, None],
                                     128 * j + nl.arange(128)[None, :]])

    return out
