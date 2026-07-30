# silu_v3_s8 — finer free-axis tiling (s=8) BRACKET PROBE for the fused-SiLU kernel.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (silu_M4096_N7168_0.py) = 1.022441 ms; parent silu_v1 = 0.3009 ms.
#
# This is the single bounded bracket probe run because s=7 (1024-wide) screened
# monotone-best (latency still decreasing at the finest divisor tested). s=8 is the next
# exact divisor of 7168 = 2^10 * 7: chunk width 7168/8 = 896 elems (3.5 KB/partition),
# giving 32*8 = 256 flat pipeline iterations. Its job is to bracket the turn -- confirm
# whether latency-vs-s has a minimum at/near s=7 or is still falling. After this one probe
# the sweep stops (a single bounded probe past the finest monotone-best point, not an
# unbounded chain of finer divisors).
#
# The row-major (128, 32, 7168) input is memory-identical to a (128, 256, 896) view, so a
# flat nl.affine_range(256) reads exactly the same bytes as v1 in shorter contiguous
# bursts -- one deep pipeline, not 32 refilled depth-8 pipelines. Mask-free and
# rectangular (896 exact). HBM traffic unchanged: read-once/write-once = 234.88 MB
# (117 + 117 MB); this is a scheduling / pipeline-depth lever, never a traffic one. Two
# live SBUF tiles of [128, 896] fp32 = 2 * 3.5 KB = 7 KB/partition, far under budget.
#
# Per-partition burst width = 896 elems = 3.5 KB (fp32). Realized loop form: FLAT
# affine_range(256) over the reshaped (128, 256, 896) view (one deep pipeline).

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1):
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    P, MID, F, S = 128, 32, 7168, 8
    CH = F // S          # 896, exact (7168 = 2^10 * 7)
    ITERS = MID * S      # 256 -- depth of the single flat pipeline

    # Output has the same (128, 32, 7168) shape as v1 (guaranteed-correct reshape-back).
    v2 = nl.ndarray((P, MID, F), dtype=np.float32, buffer=nl.shared_hbm)

    # Pure-stride views: (128, 32, 7168) is row-major identical to (128, 256, 896).
    # No copy, no DMA -- just a finer partitioning of the same contiguous free block.
    v1r = v1.reshape((P, ITERS, CH))
    v2r = v2.reshape((P, ITERS, CH))

    p_ix = nl.arange(P)[:, None]
    f_ix = nl.arange(CH)[None, :]

    # One flat affine_range over all 256 chunks: the compiler pipelines DMA against the
    # fused SiLU across the whole depth-256 loop (finer than v1's depth-32).
    for j in nl.affine_range(ITERS):
        # Load one 896-wide chunk: [128(par), 896(free)] HBM -> SBUF, one short burst.
        x_tile = nl.ndarray((par_dim(P), CH), dtype=np.float32, buffer=nl.sbuf)
        x_tile[p_ix, f_ix] = nl.load(v1r[p_ix, j, f_ix], dtype=np.float32)

        # Fused SiLU on the Scalar Engine: y = x * sigmoid(x), one instruction.
        y_tile = nl.ndarray((par_dim(P), CH), dtype=np.float32, buffer=nl.sbuf)
        y_tile[p_ix, f_ix] = nisa.activation(
            op=nl.silu, data=x_tile[p_ix, f_ix], dtype=np.float32, mask=None)

        # Store the chunk back: [128, 896] SBUF -> HBM, same [p, j, c] mapping.
        nl.store(v2r[p_ix, j, f_ix], value=y_tile[p_ix, f_ix], mask=None)

    return v2
