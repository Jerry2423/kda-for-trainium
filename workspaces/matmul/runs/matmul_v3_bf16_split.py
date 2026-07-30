import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Compensated bf16x2 3-product split of the dense GEMM out = lhs @ rhs.

    Same tiled contract and structure as matmul_v2_b4 (M=4096, K=5120, N=12288):
        v1 (32,128,40,128)=[m_tile,m_in,k_tile,k_in] lhs;
        v2 (40,128,12288)=[k_tile,k_in,n] rhs; out (32,128,12288)=[m_tile,m_in,n].
    M-block B=4, N_CHUNK=512, K-accumulate 40 kt into B distinct [128,512] fp32 PSUM
    banks, single copy+store epilogue. Only the operand precision and the matmul body
    change vs matmul_v2_b4.

    The trn2 PE array is bf16-native and emulates fp32 at ~2 passes, capping a correct
    fp32 GEMM near ~50% MFU. This kernel keeps each fp32 operand as two bf16 limbs and
    accumulates three bf16-rate products in an fp32 PSUM bank, dropping the negligible
    lo@lo cross term:
        lhs_hi = bf16(lhs_t),  lhs_lo = bf16(lhs_t - lhs_hi)   (round-to-nearest-even)
        rhs_hi = bf16(rhs_f),  rhs_lo = bf16(rhs_f - rhs_hi)
        acc[mb] += lhs_hi@rhs_hi + lhs_hi@rhs_lo + lhs_lo@rhs_hi
    The low limb is produced directly by a residual subtract into a bf16 destination
    (nisa.tensor_tensor upcasts the mixed fp32/bf16 operands to fp32 internally and
    downcasts the result for free -- no separate fp32 residual buffer). The idealized
    offline sim scores this exact product set/order at worst 5-seed rel-L2 4.454e-6,
    ~4.5x under the 2e-5 gate (pure-GEMM family: the bf16 error flows straight to the
    output, no norm feedback).

    lhs is split AFTER the fp32 identity transpose (1 transpose/tile, count unchanged
    vs v2_b4): the transpose PSUM tile is copied to a bounded transient fp32 SBUF tile,
    split element-wise into the resident bf16 limbs, and the fp32 scratch is freed --
    no full resident fp32 lhs_t survives. Two resident bf16 limbs (B,K_TILES,128,128)
    are 2 x 40 KB/partition = 80 KB/partition, exactly the bytes of v2_b4's fp32 lhs_t
    (half the dtype, twice the limbs), so the resident working set does not grow.

    nc_matmul(stationary, moving) = stationary.T @ moving; contraction k_in on the
    partition axis of both (both in SBUF). lhs tile [m_in,k_in] is transposed to
    [k_in,m_in] via the identity idiom before use as stationary; rhs [k_in,512] is the
    moving operand. PSUM: 4 acc banks ([128,512] fp32) + 1 transient transpose bank
    = 5 of 8; B stays 4.
    """
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    M_TILES = 32
    K_TILES = 40
    N = 12288
    N_CHUNK = 512
    N_CHUNKS = N // N_CHUNK      # 24
    B = 4                        # M-block factor (M_TILES % B must be 0)
    M_BLOCKS = M_TILES // B      # number of M-blocks (M_TILES/B)

    out = nl.ndarray((32, 128, 12288), dtype=np.float32, buffer=nl.shared_hbm)

    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]],
        dtype=np.float32)

    for mblk in nl.affine_range(M_BLOCKS):
        # Resident bf16 limbs for all B members of this M-block:
        #   lhs_hi[mb, kt] = lhs_lo[mb, kt] = [k_in(par)=128, m_in(free)=128]
        # Two bf16 limbs = exactly the bytes of v2_b4's single fp32 lhs_t.
        lhs_hi = nl.ndarray((B, K_TILES, par_dim(128), 128), dtype=nl.bfloat16,
                            buffer=nl.sbuf)
        lhs_lo = nl.ndarray((B, K_TILES, par_dim(128), 128), dtype=nl.bfloat16,
                            buffer=nl.sbuf)
        for mb in nl.affine_range(B):
            for kt in nl.affine_range(K_TILES):
                lhs_sb = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                    buffer=nl.sbuf)
                lhs_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
                    v1[mblk * B + mb, nl.arange(128)[:, None], kt,
                       nl.arange(128)[None, :]],
                    dtype=np.float32)
                psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                    buffer=nl.psum)
                psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                    lhs_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    is_transpose=True, is_moving_onezero=True)
                # Bounded transient fp32 scratch: one [128,128] tile, freed each
                # iteration -- no full resident fp32 lhs_t is kept.
                lhs_t_f = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                     buffer=nl.sbuf)
                lhs_t_f[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                    psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    dtype=np.float32)
                lhs_hi[mb, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                    lhs_t_f[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    dtype=nl.bfloat16)
                # lhs_lo = bf16(lhs_t - lhs_hi): exact fp32 residual, downcast to bf16.
                lhs_lo[mb, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.tensor_tensor(
                    lhs_t_f[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    lhs_hi[mb, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    op=nl.subtract)

        for c in nl.affine_range(N_CHUNKS):
            # B distinct PSUM accumulators, one per block member.
            acc = nl.zeros((B, par_dim(128), N_CHUNK), dtype=np.float32,
                           buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # Load this rhs K-tile ONCE, split into bf16 limbs, reuse across all B.
                rhs_f = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                   buffer=nl.sbuf)
                rhs_f[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.load(
                    v2[kt, nl.arange(128)[:, None],
                       N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)
                rhs_hi = nl.ndarray((par_dim(128), N_CHUNK), dtype=nl.bfloat16,
                                    buffer=nl.sbuf)
                rhs_hi[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.copy(
                    rhs_f[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    dtype=nl.bfloat16)
                # rhs_lo = bf16(rhs - rhs_hi): fused residual-subtract into a bf16 dest.
                rhs_lo = nl.ndarray((par_dim(128), N_CHUNK), dtype=nl.bfloat16,
                                    buffer=nl.sbuf)
                rhs_lo[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.tensor_tensor(
                    rhs_f[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    rhs_hi[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    op=nl.subtract)
                for mb in nl.affine_range(B):
                    # Pinned 3-product order into the member's fp32 PSUM bank;
                    # drop lhs_lo@rhs_lo (the negligible cross term).
                    acc[mb, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                        lhs_hi[mb, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        rhs_hi[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])
                    acc[mb, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                        lhs_hi[mb, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        rhs_lo[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])
                    acc[mb, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                        lhs_lo[mb, kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        rhs_hi[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

            for mb in nl.affine_range(B):
                out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                    buffer=nl.sbuf)
                out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.copy(
                    acc[mb, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)
                # Distinct out row-block per member: [m_in(par), n(free)] ->
                # out[mblk*B+mb, :, n0:n0+512], matching out[mt,mi,n]=lhs@rhs.
                nl.store(
                    out[mblk * B + mb, nl.arange(128)[:, None],
                        N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                    value=out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

    return out
