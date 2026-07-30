import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Compensated bf16-split transpose-matmul with a 2048-row M-block: out = lhs^T @ rhs.

    Same compensated bf16x2 3-product split as tmm_v2_bf16_split (numerically identical), but
    M_BLK=16 (2048 rows/block, 2 blocks) instead of M_BLK=8 (1024 rows, 4 blocks). The only
    reason to enlarge the M-block here is HBM READ traffic: the rhs chunk is re-streamed once
    per M-block, so the rhs read multiple equals the block count. tmm_v2 at M_BLK=8 re-reads
    rhs 4x and the bf16 split's larger resident working set makes the compiler re-fetch ~15%
    extra rhs tiles, pushing reads to 448 MB (vs the fp32 tmm_v1 floor 392 MB). Halving the
    block count to 2 halves the rhs read multiple (rhs 89.65 MB x2 = 179.3 vs x4 = 358.6),
    bringing total reads back below the v1 floor while keeping the matmul work, correctness,
    and the bf16 per-instruction PE win unchanged.

    SBUF budget: the resident lhs limbs double to 2x[128,16,2048] bf16 = 128 KB/partition
    (still exactly the same bytes as a hypothetical fp32 lhs_blk of the same shape); rhs limbs
    ~28.5 KB; transient fp32 build tiles small (lhs_f [128,2048] = 8 KB, rhs_f [128,456] ~1.8
    KB, freed across kt); out_sb ~1.8 KB. Peak ~168 KB/partition, under the 192 KB budget --
    watch for spill (hbm_write / psum_read_sbuf_write_count must stay at the v1 floor) since
    enlarging the resident set can constrain the affine_range software pipeline.

    Numeric method (unchanged from tmm_v2): lhs_hi=bf16(lhs), lhs_lo=bf16(lhs-lhs_hi);
    rhs_hi=bf16(rhs), rhs_lo=bf16(rhs-rhs_hi); accumulate hi@hi + hi@lo + lo@hi in one fp32
    PSUM bank, dropping the negligible lo@lo (offline worst 3-product rel-L2 4.453e-6 << 2e-5).
    The low limb is produced directly by the residual subtract into a bf16 destination (exact
    fp32 residual internally, downcast for free -- no separate fp32 residual buffer). Layout,
    the reshape mapping k=k_in*16+kt onto the partition axis, the fp32 PSUM accumulation, and
    the copy+store epilogue are otherwise the tmm_v1 structure.

    Shapes: lhs (K,M)=(2048,4096) -> v1 (128,16,4096); rhs (K,N)=(2048,10944) -> v2
    (128,16,10944); out (M,N)=(4096,10944) -> v3 (32,128,10944). N_CHUNK=456 = exact divisor
    of N (456*24), <=512 (one fp32 PSUM bank) => no tail masking anywhere.
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
    K_IN = 128                      # contraction lanes on the partition axis
    K_TILES = 16                    # kt tiles; K_IN * K_TILES = 2048 = K
    M_BLK = 16                      # m-subtiles per block (each subtile 128 rows)
    M_BLOCKS = 2                    # M / (M_BLK*128) = 4096 / 2048
    M_BLK_ROWS = M_BLK * 128        # 2048 rows per block
    N_CHUNK = 456                   # exact divisor of N, <= 512 fp32 PSUM width
    N_CHUNKS = N // N_CHUNK         # 24; 456 * 24 = 10944 exactly

    ki = nl.arange(K_IN)[:, None]        # partition index (contraction lane k_in)
    im = nl.arange(M_BLK_ROWS)[None, :]  # 2048-wide free index (m-rows of the block)
    inc = nl.arange(N_CHUNK)[None, :]    # 456-wide free index (n-slice)
    i128 = nl.arange(128)[None, :]       # 128-wide free index (subtile / output)

    out = nl.ndarray((32, 128, N), dtype=np.float32, buffer=nl.shared_hbm)

    for mb in nl.affine_range(M_BLOCKS):
        # Resident bf16 limbs of the lhs block for these 2048 m-rows: [k_in=128, kt=16, 2048],
        # built once per block. Two bf16 limbs = 128 KB/partition.
        lhs_hi = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        lhs_lo = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            lhs_f = nl.load(
                v1[ki, kt, M_BLK_ROWS * mb + im], dtype=np.float32)
            lhs_hi[ki, kt, im] = nl.copy(lhs_f[ki, im], dtype=nl.bfloat16)
            # lhs_lo = bf16(lhs - lhs_hi): exact fp32 residual, downcast to bf16 destination.
            lhs_lo[ki, kt, im] = nisa.tensor_tensor(
                lhs_f[ki, im], lhs_hi[ki, kt, im], op=nl.subtract)

        for c in nl.affine_range(N_CHUNKS):
            rhs_hi = nl.ndarray((par_dim(K_IN), K_TILES, N_CHUNK),
                                dtype=nl.bfloat16, buffer=nl.sbuf)
            rhs_lo = nl.ndarray((par_dim(K_IN), K_TILES, N_CHUNK),
                                dtype=nl.bfloat16, buffer=nl.sbuf)
            for kt in nl.affine_range(K_TILES):
                rhs_f = nl.load(
                    v2[ki, kt, N_CHUNK * c + inc], dtype=np.float32)
                rhs_hi[ki, kt, inc] = nl.copy(rhs_f[ki, inc], dtype=nl.bfloat16)
                # rhs_lo = bf16(rhs - rhs_hi): fused residual-subtract into a bf16 destination.
                rhs_lo[ki, kt, inc] = nisa.tensor_tensor(
                    rhs_f[ki, inc], rhs_hi[ki, kt, inc], op=nl.subtract)

            for s in nl.affine_range(M_BLK):
                acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32,
                               buffer=nl.psum)
                for kt in nl.affine_range(K_TILES):
                    # lhs_hi @ rhs_hi
                    acc[nl.arange(128)[:, None], inc] += nisa.nc_matmul(
                        lhs_hi[ki, kt, 128 * s + i128],
                        rhs_hi[ki, kt, inc])
                    # lhs_hi @ rhs_lo
                    acc[nl.arange(128)[:, None], inc] += nisa.nc_matmul(
                        lhs_hi[ki, kt, 128 * s + i128],
                        rhs_lo[ki, kt, inc])
                    # lhs_lo @ rhs_hi   (dropping lhs_lo @ rhs_lo, the negligible cross term)
                    acc[nl.arange(128)[:, None], inc] += nisa.nc_matmul(
                        lhs_lo[ki, kt, 128 * s + i128],
                        rhs_hi[ki, kt, inc])

                out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                    buffer=nl.sbuf)
                out_sb[nl.arange(128)[:, None], inc] = nl.copy(
                    acc[nl.arange(128)[:, None], inc],
                    dtype=np.float32)
                # out[16*mb+s, mi, 456*c+nj] -> logical row m = mb*2048 + s*128 + mi.
                nl.store(
                    out[M_BLK * mb + s, nl.arange(128)[:, None],
                        N_CHUNK * c + inc],
                    value=out_sb[nl.arange(128)[:, None], inc])

    return out
