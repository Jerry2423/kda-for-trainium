# silu_v2_k2 — multi-slice DMA batching (k=2) for elementwise SiLU (x * sigmoid(x)).
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (silu_M4096_N7168_0.py) = 1.022441 ms; parent silu_v1 = 0.3009 ms.
#
# Idea vs silu_v1: process K=2 contiguous middle-axis slices per iteration as ONE
# wider transfer. The tiled layout (128, 32, 7168) = [p, m, f] is row-major (middle
# stride 7168, free stride 1), so v1[:, g*K:(g+1)*K, :] is a contiguous [128, K*7168]
# per-partition slab. One [128, K, 7168] load HBM->SBUF, one fused SiLU over the
# K*7168 free elements (the Scalar engine flattens the free axes), one [128, K, 7168]
# store SBUF->HBM. This cuts the loop from 32 load/store pairs (k=1) to 16, amortizing
# per-DMA issue/semaphore overhead into fewer, larger bursts. Traffic is unchanged
# (still read-once/write-once = 234.88 MB), so this can only shrink the ~3% DMA-issue
# bubble, never remove traffic. Separate x/y SBUF tiles: 2 * K*28KB = 112KB/partition.

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

    P, MID, F, K = 128, 32, 7168, 2
    v2 = nl.ndarray((P, MID, F), dtype=np.float32, buffer=nl.shared_hbm)

    p_ix = nl.arange(P)[:, None, None]
    k_ix = nl.arange(K)[None, :, None]
    f_ix = nl.arange(F)[None, None, :]

    # Grouped middle axis: 32 / 2 = 16 iterations, each one wider burst. Iterations
    # are independent, so nl.affine_range lets the compiler pipeline DMA with compute.
    for g in nl.affine_range(MID // K):
        # Load K contiguous middle-axis slices as one [128(par), K, 7168] slab.
        x_tile = nl.ndarray((par_dim(P), K, F), dtype=np.float32, buffer=nl.sbuf)
        x_tile[p_ix, k_ix, f_ix] = nl.load(
            v1[p_ix, g * K + k_ix, f_ix], dtype=np.float32)

        # Fused SiLU on the Scalar Engine over all K*7168 free elements, one op.
        y_tile = nl.ndarray((par_dim(P), K, F), dtype=np.float32, buffer=nl.sbuf)
        y_tile[p_ix, k_ix, f_ix] = nisa.activation(
            op=nl.silu, data=x_tile[p_ix, k_ix, f_ix], dtype=np.float32, mask=None)

        # Store back the wider [128, K, 7168] slab, same [p, m, f] mapping.
        nl.store(v2[p_ix, g * K + k_ix, f_ix],
                 value=y_tile[p_ix, k_ix, f_ix], mask=None)

    return v2
