# silu_v3_s7 — finer free-axis tiling (s=7) of the fused-SiLU streaming kernel.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (silu_M4096_N7168_0.py) = 1.022441 ms; parent silu_v1 = 0.3009 ms.
#
# Idea: split each 7168-wide middle slice into s=7 exact sub-chunks of width 7168/7=1024,
# so the compiler sees 32*7 = 224 independent iterations to software-pipeline (a DEEPER,
# finer pipeline than v1's 32). The row-major (128, 32, 7168) input is memory-identical
# to a (128, 224, 1024) view, so reshaping it and walking a single flat
# nl.affine_range(224) reads exactly the same bytes as v1 -- just in shorter contiguous
# bursts. This is a SINGLE deep pipeline (flat index), not 32 refilled depth-7 pipelines.
#
# s=7 is the finest divisor tested here: 1024-wide chunks (4 KB/partition), the natural
# 2^10 sub-block of 7168 = 2^10 * 7. All chunk widths are exact, so the kernel stays
# mask-free and rectangular -- no edge/tail handling introduced. HBM traffic is unchanged
# from v1: read-once/write-once = 2*4096*7168*4 B = 234.88 MB (117 + 117 MB). This is
# purely a scheduling / pipeline-depth lever, never a traffic one. Two live SBUF tiles of
# [128, 1024] fp32 = 2 * 4 KB = 8 KB/partition, far under budget.
#
# Per-partition burst width = 1024 elems = 4 KB (fp32). Realized loop form: FLAT
# affine_range(224) over the reshaped (128, 224, 1024) view (one deep pipeline).

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

    P, MID, F, S = 128, 32, 7168, 7
    CH = F // S          # 1024, exact (7168 = 2^10 * 7)
    ITERS = MID * S      # 224 -- depth of the single flat pipeline

    # Output has the same (128, 32, 7168) shape as v1 (guaranteed-correct reshape-back).
    v2 = nl.ndarray((P, MID, F), dtype=np.float32, buffer=nl.shared_hbm)

    # Pure-stride views: (128, 32, 7168) is row-major identical to (128, 224, 1024).
    # No copy, no DMA -- just a finer partitioning of the same contiguous free block.
    v1r = v1.reshape((P, ITERS, CH))
    v2r = v2.reshape((P, ITERS, CH))

    p_ix = nl.arange(P)[:, None]
    f_ix = nl.arange(CH)[None, :]

    # One flat affine_range over all 224 chunks: the compiler pipelines DMA against the
    # fused SiLU across the whole depth-224 loop (finer than v1's depth-32).
    for j in nl.affine_range(ITERS):
        # Load one 1024-wide chunk: [128(par), 1024(free)] HBM -> SBUF, one short burst.
        x_tile = nl.ndarray((par_dim(P), CH), dtype=np.float32, buffer=nl.sbuf)
        x_tile[p_ix, f_ix] = nl.load(v1r[p_ix, j, f_ix], dtype=np.float32)

        # Fused SiLU on the Scalar Engine: y = x * sigmoid(x), one instruction.
        y_tile = nl.ndarray((par_dim(P), CH), dtype=np.float32, buffer=nl.sbuf)
        y_tile[p_ix, f_ix] = nisa.activation(
            op=nl.silu, data=x_tile[p_ix, f_ix], dtype=np.float32, mask=None)

        # Store the chunk back: [128, 1024] SBUF -> HBM, same [p, j, c] mapping.
        nl.store(v2r[p_ix, j, f_ix], value=y_tile[p_ix, f_ix], mask=None)

    return v2
