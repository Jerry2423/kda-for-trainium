# silu_v1 — first correct fp32 NKI kernel for elementwise SiLU (x * sigmoid(x)).
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (silu_M4096_N7168_0.py) = 1.022441 ms.
#
# Structure: one nl.affine_range(32) over the middle axis; each iteration loads a
# full-width [128, 7168] slice HBM->SBUF, applies the fused SiLU activation on the
# Scalar Engine (a single op = x * sigmoid(x), computed internally in fp32), and
# stores it back SBUF->HBM. Two live SBUF tiles per iteration (x_tile, y_tile),
# no inner free-dim loop, no masking (128*32 = 4096 = M and 7168 = N are exact).
# This is HBM-bandwidth-bound by design (Scalar compute floor is well under the
# HBM read+write floor), leaving DMA/compute overlap as the future tuning lever.

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

    P, MID, F = 128, 32, 7168
    v2 = nl.ndarray((P, MID, F), dtype=np.float32, buffer=nl.shared_hbm)

    # Partition axis = dim 0 (size 128); the free axis (size 7168) is handled in a
    # single activation call. Iterations over the middle axis are independent, so
    # nl.affine_range lets the compiler pipeline DMA with compute.
    for i0 in nl.affine_range(MID):
        # Load one middle-axis slice: [128(par), 7168(free)] HBM -> SBUF.
        x_tile = nl.ndarray((par_dim(P), F), dtype=np.float32, buffer=nl.sbuf)
        x_tile[nl.arange(P)[:, None], nl.arange(F)[None, :]] = nl.load(
            v1[nl.arange(P)[:, None], i0, nl.arange(F)[None, :]], dtype=np.float32)

        # Fused SiLU on the Scalar Engine: y = x * sigmoid(x), one instruction.
        y_tile = nl.ndarray((par_dim(P), F), dtype=np.float32, buffer=nl.sbuf)
        y_tile[nl.arange(P)[:, None], nl.arange(F)[None, :]] = nisa.activation(
            op=nl.silu,
            data=x_tile[nl.arange(P)[:, None], nl.arange(F)[None, :]],
            dtype=np.float32, mask=None)

        # Store back: [128(par), 7168(free)] SBUF -> HBM, same [p, m, f] mapping.
        nl.store(v2[nl.arange(P)[:, None], i0, nl.arange(F)[None, :]],
                 value=y_tile[nl.arange(P)[:, None], nl.arange(F)[None, :]],
                 mask=None)

    return v2
