import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Wider rhs-limb build chunk (912) to halve the limb-build prologue count.

    Screen forked from the promoted tmm_v3_mblk16_bf16_split, which is PE-bound at 98%
    and within 3.76% of the 1.385x hard arithmetic ceiling; the only residual slack is
    the ~2% PE-idle gap. This variant attacks the limb-build PROLOGUE component of that
    gap: tmm_v3 rebuilds the rhs bf16 limbs 24 times per M-block (once per 456-wide
    chunk); this variant builds them over a 912-wide chunk, so there are only 12
    rhs-limb prologues per M-block (halved). Fewer, larger prologues => fewer PE-idle
    prologue bubbles IF the prologue was the idle source.

    CLEAN ISOLATION (why this is not a repeat of the rejected v4/v5): the MATMUL moving
    width stays 456 -- the proven-good, exact-divisor, mask-free width. Within each
    912-wide chunk the 16 subtiles are matmul'd in TWO 456-wide sub-chunks (912 = 456*2
    exactly), each into its own rotated [128,456] fp32 PSUM bank exactly as tmm_v3.
    So the PE column count, PSUM-bank usage, and matmul_instruction_count are BYTE-
    IDENTICAL to tmm_v3 (2*12*2*16*16*3 = 36864); the ONLY changed variable is the
    rhs-limb-build granularity (12 builds of [128,16,912] vs 24 of [128,16,456]). This
    is a pure prologue-count experiment, NOT the chunk-WIDTH change (N_CHUNK 456->512
    + tail) that tmm_v4/v5 already measured-rejected (PE-column-bound, +1.12%/+4.2%).

    Expected (strong prior): REJECT. The kernel is PE-column-bound, and tmm_v3's DMA is
    already fully hidden (22%) with the limb build (Vec 13.7% / Scl 6.7% / GpSimd 14.2%)
    well under PE 98% -- if the prologue is already hidden by the affine_range software
    pipeline, halving its count changes nothing on the wall; and the wider 912-wide rhs
    limbs enlarge the resident working set (2x[128,16,912] bf16 ~= 57 KB/part vs 28.5),
    pushing peak SBUF from ~168 to ~190+ KB/part -- risking a pipeline-constraining
    live-set enlargement (the bmm cross-batch / silu ping-pong anti-lever class) or
    spill. Screen only; adopt only if wall beats a same-session v3 anchor by
    > max(band, 3%) AND rel-L2 == 4.4515e-6 AND hbm_read <= 229 MB AND no spill.

    Numeric method is UNCHANGED from tmm_v3 (bit-exact reschedule): lhs_hi=bf16(lhs),
    lhs_lo=bf16(lhs-lhs_hi); rhs_hi=bf16(rhs), rhs_lo=bf16(rhs-rhs_hi); accumulate
    hi@hi + hi@lo + lo@hi in one fp32 PSUM bank, dropping lo@lo. rel-L2 stays 4.4515e-6.

    Shapes: lhs (K,M)=(2048,4096) -> v1 (128,16,4096); rhs (K,N)=(2048,10944) -> v2
    (128,16,10944); out (M,N)=(4096,10944) -> v3 (32,128,10944). N_CHUNK=912 (build) =
    exact divisor of N (912*12), split into 2 matmul sub-chunks of 456 each (456<=512,
    one fp32 PSUM bank) => no tail masking anywhere.
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
    N_BUILD = 912                   # rhs-limb build chunk width (exact divisor: 912*12)
    N_BUILDS = N // N_BUILD         # 12 rhs-limb prologues per M-block (was 24 at 456)
    N_SUB = 456                     # matmul moving width (proven-good, <= 512 PSUM)
    N_SUBS = N_BUILD // N_SUB       # 2 matmul sub-chunks per 912-wide build chunk

    ki = nl.arange(K_IN)[:, None]        # partition index (contraction lane k_in)
    im = nl.arange(M_BLK_ROWS)[None, :]  # 2048-wide free index (m-rows of the block)
    inb = nl.arange(N_BUILD)[None, :]    # 912-wide free index (rhs-limb build slice)
    ins = nl.arange(N_SUB)[None, :]      # 456-wide free index (matmul n-slice)
    i128 = nl.arange(128)[None, :]       # 128-wide free index (subtile / output)

    out = nl.ndarray((32, 128, N), dtype=np.float32, buffer=nl.shared_hbm)

    for mb in nl.affine_range(M_BLOCKS):
        # Resident bf16 limbs of the lhs block for these 2048 m-rows: [k_in=128, kt=16,
        # 2048], built once per block. Two bf16 limbs = 128 KB/partition (unchanged).
        lhs_hi = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        lhs_lo = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            lhs_f = nl.load(
                v1[ki, kt, M_BLK_ROWS * mb + im], dtype=np.float32)
            lhs_hi[ki, kt, im] = nl.copy(lhs_f[ki, im], dtype=nl.bfloat16)
            # lhs_lo = bf16(lhs - lhs_hi): exact fp32 residual, downcast to bf16 dest.
            lhs_lo[ki, kt, im] = nisa.tensor_tensor(
                lhs_f[ki, im], lhs_hi[ki, kt, im], op=nl.subtract)

        for c in nl.affine_range(N_BUILDS):
            # Build the rhs bf16 limbs over a 912-wide chunk: HALVED prologue count
            # (12/block vs 24). 2x[128,16,912] bf16 ~= 57 KB/partition.
            rhs_hi = nl.ndarray((par_dim(K_IN), K_TILES, N_BUILD),
                                dtype=nl.bfloat16, buffer=nl.sbuf)
            rhs_lo = nl.ndarray((par_dim(K_IN), K_TILES, N_BUILD),
                                dtype=nl.bfloat16, buffer=nl.sbuf)
            for kt in nl.affine_range(K_TILES):
                rhs_f = nl.load(
                    v2[ki, kt, N_BUILD * c + inb], dtype=np.float32)
                rhs_hi[ki, kt, inb] = nl.copy(rhs_f[ki, inb], dtype=nl.bfloat16)
                # rhs_lo = bf16(rhs - rhs_hi): fused residual-subtract into bf16 dest.
                rhs_lo[ki, kt, inb] = nisa.tensor_tensor(
                    rhs_f[ki, inb], rhs_hi[ki, kt, inb], op=nl.subtract)

            # Matmul in 2 sub-chunks of 456 (identical PE work / PSUM usage to tmm_v3).
            for sc in nl.affine_range(N_SUBS):
                for s in nl.affine_range(M_BLK):
                    acc = nl.zeros((par_dim(128), N_SUB), dtype=np.float32,
                                   buffer=nl.psum)
                    for kt in nl.affine_range(K_TILES):
                        # lhs_hi @ rhs_hi
                        acc[nl.arange(128)[:, None], ins] += nisa.nc_matmul(
                            lhs_hi[ki, kt, 128 * s + i128],
                            rhs_hi[ki, kt, N_SUB * sc + ins])
                        # lhs_hi @ rhs_lo
                        acc[nl.arange(128)[:, None], ins] += nisa.nc_matmul(
                            lhs_hi[ki, kt, 128 * s + i128],
                            rhs_lo[ki, kt, N_SUB * sc + ins])
                        # lhs_lo @ rhs_hi   (dropping lhs_lo @ rhs_lo, negligible term)
                        acc[nl.arange(128)[:, None], ins] += nisa.nc_matmul(
                            lhs_lo[ki, kt, 128 * s + i128],
                            rhs_hi[ki, kt, N_SUB * sc + ins])

                    out_sb = nl.ndarray((par_dim(128), N_SUB), dtype=np.float32,
                                        buffer=nl.sbuf)
                    out_sb[nl.arange(128)[:, None], ins] = nl.copy(
                        acc[nl.arange(128)[:, None], ins],
                        dtype=np.float32)
                    # logical n = 912*c + 456*sc + nj; logical row m = mb*2048+s*128+mi.
                    nl.store(
                        out[M_BLK * mb + s, nl.arange(128)[:, None],
                            N_BUILD * c + N_SUB * sc + ins],
                        value=out_sb[nl.arange(128)[:, None], ins])

    return out
