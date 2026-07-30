# silu_v2_k3 — multi-slice DMA batching (k=3, exact-divisor tail) for SiLU.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (silu_M4096_N7168_0.py) = 1.022441 ms; parent silu_v1 = 0.3009 ms.
#
# Same idea as silu_v2_k2 but K=3. 32 does not divide by 3, so use an exact-divisor
# tail (NO masking): 10 full groups of 3 middle-axis slices (slices 0..29) plus one
# explicit tail group of 2 (slices 30..31). Each group is one contiguous
# [128, K, 7168] load HBM->SBUF, one fused SiLU over the K*7168 free elements, one
# [128, K, 7168] store SBUF->HBM. Both group shapes are exact (3 and 2), so no mask
# is needed anywhere. Separate x/y SBUF tiles: 2 * 3 * 28KB = 168KB/partition (fits
# the 208KB budget). Traffic is unchanged (read-once/write-once = 234.88 MB).

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

    P, MID, F, K = 128, 32, 7168, 3
    N_FULL = MID // K        # 10 full groups of 3
    TAIL = MID - N_FULL * K  # 2 leftover slices (exact, no mask)
    v2 = nl.ndarray((P, MID, F), dtype=np.float32, buffer=nl.shared_hbm)

    p3 = nl.arange(P)[:, None, None]
    k3 = nl.arange(K)[None, :, None]
    f3 = nl.arange(F)[None, None, :]

    # Main sweep: 10 groups of K=3 contiguous middle-axis slices, one wide burst each.
    for g in nl.affine_range(N_FULL):
        x_tile = nl.ndarray((par_dim(P), K, F), dtype=np.float32, buffer=nl.sbuf)
        x_tile[p3, k3, f3] = nl.load(v1[p3, g * K + k3, f3], dtype=np.float32)

        y_tile = nl.ndarray((par_dim(P), K, F), dtype=np.float32, buffer=nl.sbuf)
        y_tile[p3, k3, f3] = nisa.activation(
            op=nl.silu, data=x_tile[p3, k3, f3], dtype=np.float32, mask=None)

        nl.store(v2[p3, g * K + k3, f3], value=y_tile[p3, k3, f3], mask=None)

    # Exact tail group of TAIL=2 slices (slices 30..31), same fused pattern, no mask.
    pt = nl.arange(P)[:, None, None]
    kt = nl.arange(TAIL)[None, :, None]
    ft = nl.arange(F)[None, None, :]
    base = N_FULL * K

    xt = nl.ndarray((par_dim(P), TAIL, F), dtype=np.float32, buffer=nl.sbuf)
    xt[pt, kt, ft] = nl.load(v1[pt, base + kt, ft], dtype=np.float32)

    yt = nl.ndarray((par_dim(P), TAIL, F), dtype=np.float32, buffer=nl.sbuf)
    yt[pt, kt, ft] = nisa.activation(
        op=nl.silu, data=xt[pt, kt, ft], dtype=np.float32, mask=None)

    nl.store(v2[pt, base + kt, ft], value=yt[pt, kt, ft], mask=None)

    return v2
