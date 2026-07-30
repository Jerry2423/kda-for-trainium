# silu_v3_s7_nested — loop-form confirmation for the s=7 finer-tiling winner.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (silu_M4096_N7168_0.py) = 1.022441 ms; parent silu_v1 = 0.3009 ms.
#
# This is the flat-vs-nested confirmation run for the best-screening finer variant (s=7,
# 1024-wide, screen 0.2938 ms). It expresses the SAME 32*7 = 224-chunk iteration space as
# the flat runs/silu_v3_s7.py, but as a NESTED nl.affine_range(32) x nl.affine_range(7)
# over the original (128, 32, 7168) tensor (middle slice m, sub-chunk c of width 1024),
# with no reshape. Both loops are nl.affine_range, so the compiler is free to fuse/pipeline
# across the full 224-iteration space rather than refilling per outer step. The purpose is
# to confirm whether the flat single-index form and the nested form pipeline equally deep;
# the deeper-pipelining (lower-latency) form is the one kept for the promotion gate.
#
# Chunk c covers free columns [c*1024, (c+1)*1024); combined with middle slice m this walks
# exactly the same bytes as the flat (128, 224, 1024) view, contiguous per (m,c). Mask-free
# and rectangular (1024 exact). HBM traffic unchanged: 234.88 MB (117 + 117 MB), a
# scheduling lever only. Two live SBUF tiles of [128, 1024] fp32 = 8 KB/partition.
#
# Per-partition burst width = 1024 elems = 4 KB (fp32). Realized loop form: NESTED
# affine_range(32) x affine_range(7) over (128, 32, 7168) (the flat-vs-nested comparison).

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

    v2 = nl.ndarray((P, MID, F), dtype=np.float32, buffer=nl.shared_hbm)

    p_ix = nl.arange(P)[:, None]
    f_ix = nl.arange(CH)[None, :]

    # Nested affine_range over middle slice (32) x sub-chunk (7) = 224 iterations. Both
    # loops are affine_range so the compiler may pipeline across the full nested space.
    for m in nl.affine_range(MID):
        for c in nl.affine_range(S):
            # Free-axis start c*1024 is an affine function of the inner loop var.
            x_tile = nl.ndarray((par_dim(P), CH), dtype=np.float32, buffer=nl.sbuf)
            x_tile[p_ix, f_ix] = nl.load(
                v1[p_ix, m, c * CH + f_ix], dtype=np.float32)

            y_tile = nl.ndarray((par_dim(P), CH), dtype=np.float32, buffer=nl.sbuf)
            y_tile[p_ix, f_ix] = nisa.activation(
                op=nl.silu, data=x_tile[p_ix, f_ix], dtype=np.float32, mask=None)

            nl.store(v2[p_ix, m, c * CH + f_ix], value=y_tile[p_ix, f_ix], mask=None)

    return v2
