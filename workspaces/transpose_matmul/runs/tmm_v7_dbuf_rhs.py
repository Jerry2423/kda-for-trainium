import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Prefetched double-buffer of the rhs limbs: build the NEXT chunk while matmul-ing.

    Forked from the promoted tmm_v3_mblk16_bf16_split. The promoted kernel builds one
    456-wide rhs limb set, matmuls its 16 subtiles, then rebuilds for the next chunk --
    a single limb buffer, build and matmul strictly serialized per chunk. This variant
    holds TWO fixed rhs-limb buffer sets (A and B) per M-block and runs an explicit
    build-ahead / prefetch schedule so each chunk's limb build is ISSUED BEFORE the
    previous chunk's matmul, giving the compiler the chance to overlap the (Vec/Scalar)
    limb build with the (PE) matmul and shave the residual PE stall at chunk boundaries.

    Prefetch schedule (primed alternation, exactly the double-buffer form):
      prime:            build A <- chunk 0
      for p in 0..10:   build B <- chunk 2p+1      (prefetch, overlaps the A matmul)
                        matmul/store A (chunk 2p)
                        build A <- chunk 2p+2       (prefetch, overlaps the B matmul)
                        matmul/store B (chunk 2p+1)
      final pair p=11:  build B <- chunk 23
                        matmul/store A (chunk 22)   (A holds 22 from p=10's build-ahead)
                        matmul/store B (chunk 23)
    The 24 chunks are emitted as a Python-unrolled straight-line sequence (NOT an
    nl.sequential_range, which would serialize iterations and DENY the compiler the
    cross-iteration pipelining that makes the prefetch possible -- see
    BL-20260709-dma-batching-regresses-pipeline). Straight-line source lets the compiler
    reorder by true data dependencies: each `build (next)` is independent of the
    preceding `matmul (current)`, so it can slot alongside it; the two fixed buffers A/B
    carry the WAR/WAW hazards that make it a genuine ping-pong.

    The cost this pays -- and the thing being measured -- is SBUF residency: two live
    rhs-limb sets are 2x(rhs_hi + rhs_lo) = 2x2x[128,16,456] bf16 = 58368 B/partition
    (~= 57 KB), i.e. +28.5 KB over tmm_v3's single set, on top of the 128 KB/partition
    resident lhs limbs + transient fp32 build scratch. Screen only (verify.py --fast +
    dump_metrics --fast vs a same-session tmm_v3 anchor); adopt only if wall beats v3 by
    > max(band, 3%) AND rel-L2 == 4.4515e-6 AND hbm_read <= 229 MB AND no spill
    (hbm_write 179.3 MB, psum 768). Expected reject: the tmm_v3 limb build is already
    fully hidden (Vec 13.7% / Scl 6.7% / GpSimd 14.2% / DMA 22.5% all << PE 98%), so
    there is no exposed prologue for the prefetch to hide, and the extra resident set
    enlarges the live working set enough to induce rhs input re-fetch.

    Numeric method is UNCHANGED from tmm_v3 (bit-exact reschedule): lhs_hi=bf16(lhs),
    lhs_lo=bf16(lhs-lhs_hi); rhs_hi=bf16(rhs), rhs_lo=bf16(rhs-rhs_hi); accumulate
    hi@hi + hi@lo + lo@hi in one fp32 PSUM bank, dropping lo@lo. rel-L2 stays 4.4515e-6.

    Shapes: lhs (K,M)=(2048,4096) -> v1 (128,16,4096); rhs (K,N)=(2048,10944) -> v2
    (128,16,10944); out (M,N)=(4096,10944) -> v3 (32,128,10944). N_CHUNK=456 (exact
    divisor, 456*24), <=512 (one fp32 PSUM bank) => no tail masking anywhere.
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
    N_PAIRS = N_CHUNKS // 2         # 12 double-buffered chunk pairs

    ki = nl.arange(K_IN)[:, None]        # partition index (contraction lane k_in)
    im = nl.arange(M_BLK_ROWS)[None, :]  # 2048-wide free index (m-rows of the block)
    inc = nl.arange(N_CHUNK)[None, :]    # 456-wide free index (n-slice)
    i128 = nl.arange(128)[None, :]       # 128-wide free index (subtile / output)

    out = nl.ndarray((32, 128, N), dtype=np.float32, buffer=nl.shared_hbm)

    def build_rhs_limbs(rhs_hi, rhs_lo, c):
        """Build the bf16 hi/lo limbs of rhs chunk c into the given buffer set."""
        for kt in nl.affine_range(K_TILES):
            rhs_f = nl.load(v2[ki, kt, N_CHUNK * c + inc], dtype=np.float32)
            rhs_hi[ki, kt, inc] = nl.copy(rhs_f[ki, inc], dtype=nl.bfloat16)
            rhs_lo[ki, kt, inc] = nisa.tensor_tensor(
                rhs_f[ki, inc], rhs_hi[ki, kt, inc], op=nl.subtract)

    def matmul_chunk(lhs_hi, lhs_lo, rhs_hi, rhs_lo, mb, c):
        """3-product bf16 matmul + store for all 16 subtiles of rhs chunk c."""
        for s in nl.affine_range(M_BLK):
            acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                acc[nl.arange(128)[:, None], inc] += nisa.nc_matmul(
                    lhs_hi[ki, kt, 128 * s + i128], rhs_hi[ki, kt, inc])
                acc[nl.arange(128)[:, None], inc] += nisa.nc_matmul(
                    lhs_hi[ki, kt, 128 * s + i128], rhs_lo[ki, kt, inc])
                acc[nl.arange(128)[:, None], inc] += nisa.nc_matmul(
                    lhs_lo[ki, kt, 128 * s + i128], rhs_hi[ki, kt, inc])
            out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
            out_sb[nl.arange(128)[:, None], inc] = nl.copy(
                acc[nl.arange(128)[:, None], inc], dtype=np.float32)
            nl.store(
                out[M_BLK * mb + s, nl.arange(128)[:, None], N_CHUNK * c + inc],
                value=out_sb[nl.arange(128)[:, None], inc])

    for mb in nl.affine_range(M_BLOCKS):
        # Resident bf16 limbs of the lhs block for these 2048 m-rows, built once/block.
        lhs_hi = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        lhs_lo = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            lhs_f = nl.load(v1[ki, kt, M_BLK_ROWS * mb + im], dtype=np.float32)
            lhs_hi[ki, kt, im] = nl.copy(lhs_f[ki, im], dtype=nl.bfloat16)
            lhs_lo[ki, kt, im] = nisa.tensor_tensor(
                lhs_f[ki, im], lhs_hi[ki, kt, im], op=nl.subtract)

        # Two FIXED rhs-limb buffer sets (A and B), reused across the whole N sweep.
        # Both live for the entire M-block -> the +28.5 KB/part double-buffer footprint.
        rhs_hi_a = nl.ndarray((par_dim(K_IN), K_TILES, N_CHUNK),
                              dtype=nl.bfloat16, buffer=nl.sbuf)
        rhs_lo_a = nl.ndarray((par_dim(K_IN), K_TILES, N_CHUNK),
                              dtype=nl.bfloat16, buffer=nl.sbuf)
        rhs_hi_b = nl.ndarray((par_dim(K_IN), K_TILES, N_CHUNK),
                              dtype=nl.bfloat16, buffer=nl.sbuf)
        rhs_lo_b = nl.ndarray((par_dim(K_IN), K_TILES, N_CHUNK),
                              dtype=nl.bfloat16, buffer=nl.sbuf)

        # Prime set A with chunk 0, then run the primed-alternation prefetch schedule.
        # The 24 chunks are emitted straight-line (Python-unrolled) so the compiler can
        # overlap each build-ahead with the preceding chunk's matmul.
        build_rhs_limbs(rhs_hi_a, rhs_lo_a, 0)
        for p in range(N_PAIRS - 1):            # p = 0..10 -> chunks (2p, 2p+1)
            build_rhs_limbs(rhs_hi_b, rhs_lo_b, 2 * p + 1)      # prefetch odd chunk
            matmul_chunk(lhs_hi, lhs_lo, rhs_hi_a, rhs_lo_a, mb, 2 * p)
            build_rhs_limbs(rhs_hi_a, rhs_lo_a, 2 * p + 2)      # prefetch next even chunk
            matmul_chunk(lhs_hi, lhs_lo, rhs_hi_b, rhs_lo_b, mb, 2 * p + 1)
        # Final pair (p = 11): chunk 22 already in A (built as 2*10+2), build 23 into B.
        last_even = 2 * (N_PAIRS - 1)           # 22
        build_rhs_limbs(rhs_hi_b, rhs_lo_b, last_even + 1)      # chunk 23
        matmul_chunk(lhs_hi, lhs_lo, rhs_hi_a, rhs_lo_a, mb, last_even)      # chunk 22
        matmul_chunk(lhs_hi, lhs_lo, rhs_hi_b, rhs_lo_b, mb, last_even + 1)  # chunk 23

    return out
