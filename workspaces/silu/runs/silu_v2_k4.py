# silu_v2_k4 — multi-slice DMA batching (k=4) with in-place SBUF compute, for SiLU.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (silu_M4096_N7168_0.py) = 1.022441 ms; parent silu_v1 = 0.3009 ms.
#
# K=4 divides 32 exactly (8 groups, no tail). At K=4 two separate x/y SBUF tiles would
# need 2 * 4 * 28KB = 224KB/partition, over the ~208KB budget -- so this variant writes
# the fused SiLU result back INTO the load buffer (in-place: activation dst aliases its
# data tile), leaving only ONE live [128,4,7168] tile = 112KB/partition. SiLU is a pure
# per-lane elementwise map (dst[i] = op(data[i])), so aliasing is architecturally safe;
# the NKI docs do not explicitly guarantee src==dst for nisa.activation, so correctness
# is confirmed empirically by verify.py's relative-L2 gate on all five seeds. Traffic is
# unchanged (read-once/write-once = 234.88 MB); the in-place aliasing is SBUF-internal
# and never touches HBM (v2 is a fresh shared_hbm ndarray, distinct from input v1).

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

    P, MID, F, K = 128, 32, 7168, 4
    v2 = nl.ndarray((P, MID, F), dtype=np.float32, buffer=nl.shared_hbm)

    p_ix = nl.arange(P)[:, None, None]
    k_ix = nl.arange(K)[None, :, None]
    f_ix = nl.arange(F)[None, None, :]

    # Grouped middle axis: 32 / 4 = 8 iterations, each one wide burst.
    for g in nl.affine_range(MID // K):
        # Load K contiguous middle-axis slices as one [128(par), K, 7168] slab.
        x_tile = nl.ndarray((par_dim(P), K, F), dtype=np.float32, buffer=nl.sbuf)
        x_tile[p_ix, k_ix, f_ix] = nl.load(
            v1[p_ix, g * K + k_ix, f_ix], dtype=np.float32)

        # Fused SiLU in-place: the activation writes back into x_tile (dst == data).
        # One live SBUF tile (112KB/partition) instead of two, which is what lets K=4
        # fit the 208KB budget. Validated by the L2 gate, not a documented promise.
        x_tile[p_ix, k_ix, f_ix] = nisa.activation(
            op=nl.silu, data=x_tile[p_ix, k_ix, f_ix], dtype=np.float32, mask=None)

        # Store the wide [128, K, 7168] slab back, same [p, m, f] mapping.
        nl.store(v2[p_ix, g * K + k_ix, f_ix],
                 value=x_tile[p_ix, k_ix, f_ix], mask=None)

    return v2
