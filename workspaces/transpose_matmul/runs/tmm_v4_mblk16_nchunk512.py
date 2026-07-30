import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """bf16-split transpose-matmul, 2048-row M-block, 512-wide N chunks + 192 tail.

    A wider-N-chunk screen composed on tmm_v3_mblk16_bf16_split: N is tiled as 21 full
    512-wide chunks (0..10751) plus one 192-wide tail chunk (10752..10943), i.e. 22
    chunk-groups instead of the 24 that N_CHUNK=456 produces. Since each output column is
    still computed exactly once, the PE COLUMN work is the same either way; the only possible
    gain is fixed per-matmul issue/fill overhead (22 chunk-groups x 16 subtiles x 16 kt x 3
    products = 33792 matmul instructions vs 36864 at N_CHUNK=456, -8.3%). Whether that fixed
    overhead is a meaningful fraction of a PE-bound wall is measured, not assumed.

    The tail is handled as a SEPARATE 192-wide chunk, NOT a padded-to-512 masked chunk: 192 is
    a valid moving width (<=512, one fp32 PSUM bank), so a distinct narrow tile is
    correct-by-construction, does strictly less matmul work than padding-then-masking 512
    columns, and avoids the tail-mask off-by-one arithmetic (the single largest bug surface the
    fp32 tmm_v1 kernel deliberately removed by choosing the exact divisor 456). Both the full
    and tail chunks reuse the identical limb-build + 3-product body; only the moving width differs.

    Everything else is tmm_v3_mblk16: compensated bf16x2 3-product split (lhs_hi/lhs_lo
    resident per 2048-row block, rhs_hi/rhs_lo per chunk; hi@hi + hi@lo + lo@hi in one fp32
    PSUM bank, dropping lo@lo; low limb produced directly by the residual subtract into a bf16
    destination). Layout: reshape (K,.)->(128,16,.) puts K on the partition axis so
    nc_matmul(stationary, moving) = stationary.T @ moving computes lhs^T @ rhs directly.
    Shapes: lhs->v1 (128,16,4096); rhs->v2 (128,16,10944); out->v3 (32,128,10944).

    HBM: rhs is re-streamed once per M-block (2 blocks => rhs read x2), so reads stay well
    below the fp32 v1 floor. bf16 limbs = same bytes as the fp32 tiles they replace; limbs
    built on-chip from the same loads, no extra reads.
    """
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    M = 4096
    K = 2048
    N = 10944
    K_IN = 128                        # contraction lanes on the partition axis
    K_TILES = 16                      # kt tiles; K_IN * K_TILES = 2048 = K
    M_BLK = 16                        # m-subtiles per block (each subtile 128 rows)
    M_BLOCKS = 2                      # M / (M_BLK*128) = 4096 / 2048
    M_BLK_ROWS = M_BLK * 128          # 2048 rows per block
    N_CHUNK = 512                     # full-chunk moving width (one fp32 PSUM bank)
    N_FULL = N // N_CHUNK             # 21 full 512-wide chunks (21*512 = 10752)
    N_TAIL = N - N_FULL * N_CHUNK     # 192-wide tail (10944 - 10752)

    ki = nl.arange(K_IN)[:, None]          # partition index (contraction lane k_in)
    im = nl.arange(M_BLK_ROWS)[None, :]    # 2048-wide free index (m-rows of the block)
    i128 = nl.arange(128)[None, :]         # 128-wide free index (subtile / output)

    out = nl.ndarray((32, 128, N), dtype=np.float32, buffer=nl.shared_hbm)

    def process_chunk(mb, lhs_hi, lhs_lo, n0, width):
        """Build the rhs limbs for the [n0, n0+width) slice and accumulate the 3 products."""
        iw = nl.arange(width)[None, :]
        rhs_hi = nl.ndarray((par_dim(K_IN), K_TILES, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        rhs_lo = nl.ndarray((par_dim(K_IN), K_TILES, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            rhs_f = nl.load(v2[ki, kt, n0 + iw], dtype=np.float32)
            rhs_hi[ki, kt, iw] = nl.copy(rhs_f[ki, iw], dtype=nl.bfloat16)
            # rhs_lo = bf16(rhs - rhs_hi): fused residual-subtract into a bf16 destination.
            rhs_lo[ki, kt, iw] = nisa.tensor_tensor(
                rhs_f[ki, iw], rhs_hi[ki, kt, iw], op=nl.subtract)

        for s in nl.affine_range(M_BLK):
            acc = nl.zeros((par_dim(128), width), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # lhs_hi @ rhs_hi
                acc[nl.arange(128)[:, None], iw] += nisa.nc_matmul(
                    lhs_hi[ki, kt, 128 * s + i128], rhs_hi[ki, kt, iw])
                # lhs_hi @ rhs_lo
                acc[nl.arange(128)[:, None], iw] += nisa.nc_matmul(
                    lhs_hi[ki, kt, 128 * s + i128], rhs_lo[ki, kt, iw])
                # lhs_lo @ rhs_hi   (dropping lhs_lo @ rhs_lo, the negligible cross term)
                acc[nl.arange(128)[:, None], iw] += nisa.nc_matmul(
                    lhs_lo[ki, kt, 128 * s + i128], rhs_hi[ki, kt, iw])

            out_sb = nl.ndarray((par_dim(128), width), dtype=np.float32, buffer=nl.sbuf)
            out_sb[nl.arange(128)[:, None], iw] = nl.copy(
                acc[nl.arange(128)[:, None], iw], dtype=np.float32)
            # out[16*mb+s, mi, n0+nj] -> logical row m = mb*2048 + s*128 + mi.
            nl.store(
                out[M_BLK * mb + s, nl.arange(128)[:, None], n0 + iw],
                value=out_sb[nl.arange(128)[:, None], iw])

    for mb in nl.affine_range(M_BLOCKS):
        # Resident bf16 limbs of the lhs block for these 2048 m-rows: [k_in=128, kt=16, 2048],
        # built once per block. Two bf16 limbs = 128 KB/partition.
        lhs_hi = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        lhs_lo = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            lhs_f = nl.load(v1[ki, kt, M_BLK_ROWS * mb + im], dtype=np.float32)
            lhs_hi[ki, kt, im] = nl.copy(lhs_f[ki, im], dtype=nl.bfloat16)
            # lhs_lo = bf16(lhs - lhs_hi): exact fp32 residual, downcast to bf16 destination.
            lhs_lo[ki, kt, im] = nisa.tensor_tensor(
                lhs_f[ki, im], lhs_hi[ki, kt, im], op=nl.subtract)

        # 21 full 512-wide chunks, then the 192-wide tail (both mask-free).
        for c in nl.affine_range(N_FULL):
            process_chunk(mb, lhs_hi, lhs_lo, N_CHUNK * c, N_CHUNK)
        process_chunk(mb, lhs_hi, lhs_lo, N_CHUNK * N_FULL, N_TAIL)

    return out
