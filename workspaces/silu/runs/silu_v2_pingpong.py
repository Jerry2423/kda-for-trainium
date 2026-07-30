# silu_v2_pingpong — explicit two-buffer (ping-pong) manual DMA prefetch for SiLU.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (silu_M4096_N7168_0.py) = 1.022441 ms; parent silu_v1 = 0.3009 ms.
#
# Idea: keep silu_v1's k=1 full-width [128,7168] tiling, but manually double-buffer
# the loads. Pre-allocate a 2-deep load buffer and, while computing slice i out of one
# half, prefetch slice i+1 into the other half. The DMA engine runs asynchronously, so
# the prefetch load overlaps the current slice's Scalar activation; the two halves keep
# the in-flight load from clobbering the tile under compute. Uses nl.sequential_range so
# the manual prefetch schedule is honored (the overlap comes from async DMA, not from
# compiler loop-pipelining). A prologue load (slice 0) and an epilogue compute (last
# slice) keep every slice read EXACTLY once -> HBM stays at the 234.88 MB floor.
#
# Expectation: ~0 gain. silu_v1 already uses nl.affine_range, which licenses the compiler
# to software-pipeline DMA against compute, and v1 already sits at the overlapped one-way
# estimate (0.3009 ~= 0.319 ms), not the serialized 0.638 ms. So explicit manual
# prefetching is most likely redundant with the compiler's auto-pipeline; it is worth
# keeping only if it measurably beats v1's steady-state latency beyond run-to-run jitter.

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

    p_ix = nl.arange(P)[:, None]
    f_ix = nl.arange(F)[None, :]

    # Two-deep ping-pong load buffer: leading axis (size 2) selects the half; the
    # partition axis is P, the free axis is F. 2 * 7168 * 4 = 56KB/partition.
    xbuf = nl.ndarray((2, par_dim(P), F), dtype=np.float32, buffer=nl.sbuf)

    # Prologue: prefetch slice 0 into half 0.
    xbuf[0, p_ix, f_ix] = nl.load(v1[p_ix, 0, f_ix], dtype=np.float32)

    # Steady state: for slice i, prefetch slice i+1 into the other half, then compute
    # and store slice i. Runs MID-1 times so the prefetch target i+1 is always valid.
    for i in nl.sequential_range(MID - 1):
        cur = i % 2
        nxt = (i + 1) % 2
        xbuf[nxt, p_ix, f_ix] = nl.load(v1[p_ix, i + 1, f_ix], dtype=np.float32)

        y_tile = nl.ndarray((par_dim(P), F), dtype=np.float32, buffer=nl.sbuf)
        y_tile[p_ix, f_ix] = nisa.activation(
            op=nl.silu, data=xbuf[cur, p_ix, f_ix], dtype=np.float32, mask=None)
        nl.store(v2[p_ix, i, f_ix], value=y_tile[p_ix, f_ix], mask=None)

    # Epilogue: compute + store the last slice (already prefetched in the final loop iter).
    last = MID - 1
    cur_last = last % 2
    y_last = nl.ndarray((par_dim(P), F), dtype=np.float32, buffer=nl.sbuf)
    y_last[p_ix, f_ix] = nisa.activation(
        op=nl.silu, data=xbuf[cur_last, p_ix, f_ix], dtype=np.float32, mask=None)
    nl.store(v2[p_ix, last, f_ix], value=y_last[p_ix, f_ix], mask=None)

    return v2
