import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2, v3, v4):
    """Fused low-rank residual GEMM  out = x@w + (x@a)@b  (fp32).

    Shapes (M=4096, K=5120, N=12288, R=128) in the NKIBench reference's tiled layout:
        v1 (x): (8, 4, 128, 40, 128) = [m_hi, m_lo, m_in, k_tile, k_in]
                row m = (m_hi*4 + m_lo)*128 + m_in, col k = k_tile*128 + k_in
        v2 (w): (40, 128, 12288)     = [k_tile, k_in, n]   (contraction k_in on partition)
        v3 (a): (40, 128, 128)       = [k_tile, k_in, r]   (contraction k_in on partition)
        v4 (b): (128, 12288)         = [r, n]              (contraction r on partition)
    Output:
        v5 (out): (8, 4, 128, 96, 128) = [m_hi, m_lo, m_in, n_tile, n_in]
                  row m = (m_hi*4 + m_lo)*128 + m_in, col n = n_tile*128 + n_in

    The Tensor Engine's nc_matmul(stationary, moving) = stationary.T @ moving, with the
    contraction dim on the PARTITION axis of both operands, both operands SBUF-resident.
    w/a/b already have their contraction dim (k_in resp. r) on the partition axis -> used
    directly. x tiles arrive as [m_in(par), k_in(free)] -> each is transposed to
    [k_in(par), m_in(free)] via the identity nc_matmul(is_transpose=True) idiom. This
    transposed x tile (lhs_t) is the shared operand for BOTH x@w and x@a.

    Loop order is M-outer over the 2-level (m_hi, m_lo) M-index (8x4 = 32 tiles), driven as
    nested affine_range loops so both are affine indices into v1/v5. Per M-tile:
      1. Transpose x once (shared): lhs_t[kt] = [k_in, m_in], 40 identity transposes.
      2. Down-projection tT = (x@a)^T = [R, m_in], accumulated over K into a zero-init
         PSUM tile, then copied to fp32 SBUF BEFORE the N loop (the N loop reuses PSUM).
      3. Per n-tile (96 of width 128): accumulate the base GEMM x@w over the 40 K-tiles
         into one PSUM bank, then FUSE the low-rank residual (x@a)@b = tT.T @ b into the
         SAME bank before a single copy + store. The (x@a)/(x@a)@b intermediate never
         touches HBM.

    Pure fp32 throughout (matches sibling matmul_v1, which passed the same 2e-5 gate).
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
    N_CHUNK = 128              # one output n_tile per chunk (direct store, no strided write)
    N_TILES = N // N_CHUNK     # 96

    out = nl.ndarray((8, 4, 128, 96, 128), dtype=np.float32, buffer=nl.shared_hbm)

    # 128x128 identity in SBUF: the moving operand that transposes lhs tiles on the
    # Tensor Engine (is_transpose=True). Loaded once, reused for all tiles.
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]],
        dtype=np.float32)

    # a fully resident in SBUF: a_local[kt] = v3[kt] = [k_in(par), r(free)], reused across
    # all 32 M-tiles (~20 KB/partition). Used as the stationary operand of the down-proj.
    a_local = nl.ndarray((K_TILES, par_dim(128), R), dtype=np.float32, buffer=nl.sbuf)
    for kt in nl.affine_range(K_TILES):
        a_local[kt, nl.arange(128)[:, None], nl.arange(R)[None, :]] = nl.load(
            v3[kt, nl.arange(128)[:, None], nl.arange(R)[None, :]], dtype=np.float32)

    for m_hi in nl.affine_range(M_HI):
        for m_lo in nl.affine_range(M_LO):
            # 1. Transpose x once (shared by both GEMMs).
            # lhs_t[kt] = [k_in(par)=128, m_in(free)=128].
            lhs_t = nl.ndarray((K_TILES, par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
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
                lhs_t[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                    psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    dtype=np.float32)

            # 2. Down-projection tT = (x@a)^T = [R, m_in], accumulated over K.
            # nc_matmul(stationary=a_local[kt] [k_in,R], moving=lhs_t[kt] [k_in,m_in])
            #   = a_local[kt].T @ lhs_t[kt] = [R,k_in] @ [k_in,m_in] = [R, m_in].
            # PSUM tile explicitly zero-initialized before the K loop.
            tT_psum = nl.zeros((par_dim(R), 128), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                tT_psum[nl.arange(R)[:, None], nl.arange(128)[None, :]] += nisa.nc_matmul(
                    a_local[kt, nl.arange(128)[:, None], nl.arange(R)[None, :]],
                    lhs_t[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]])
            # Copy tT to fp32 SBUF BEFORE the N loop (which reuses PSUM banks for acc).
            tT = nl.ndarray((par_dim(R), 128), dtype=np.float32, buffer=nl.sbuf)
            tT[nl.arange(R)[:, None], nl.arange(128)[None, :]] = nl.copy(
                tT_psum[nl.arange(R)[:, None], nl.arange(128)[None, :]], dtype=np.float32)

            # 3. Per n-tile: base GEMM x@w + fused low-rank (x@a)@b into ONE output bank.
            for nt_ in nl.affine_range(N_TILES):
                n0 = N_CHUNK * nt_
                acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
                for kt in nl.affine_range(K_TILES):
                    # Load w tile from HBM into SBUF: [k_in(par)=128, 128(free)].
                    w_tile = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
                    w_tile[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.load(
                        v2[kt, nl.arange(128)[:, None], n0 + nl.arange(N_CHUNK)[None, :]],
                        dtype=np.float32)
                    # nc_matmul(stationary=lhs_t[kt] [k_in,m_in], moving=w_tile [k_in,128])
                    #   = [m_in,k_in] @ [k_in,128] = [m_in,128]  (base x@w)
                    acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                        lhs_t[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        w_tile[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

                # Fuse the low-rank residual into the SAME PSUM bank (no HBM round-trip).
                # Load b tile: [r(par)=128, 128(free)].
                b_tile = nl.ndarray((par_dim(R), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
                b_tile[nl.arange(R)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.load(
                    v4[nl.arange(R)[:, None], n0 + nl.arange(N_CHUNK)[None, :]], dtype=np.float32)
                # nc_matmul(stationary=tT [r,m_in], moving=b_tile [r,128])
                #   = tT.T @ b_tile = (x@a) @ b = [m_in,128]  -> same layout, same bank.
                acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                    tT[nl.arange(R)[:, None], nl.arange(128)[None, :]],
                    b_tile[nl.arange(R)[:, None], nl.arange(N_CHUNK)[None, :]])

                out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
                out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.copy(
                    acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]], dtype=np.float32)
                # Store [m_in(par), n_in(free)] into v5[m_hi, m_lo, :, nt, :].
                nl.store(
                    out[m_hi, m_lo, nl.arange(128)[:, None], nt_, nl.arange(N_CHUNK)[None, :]],
                    value=out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

    return out
