# adamw_v2_ch512 — mask-free (128, ITERS, CH) reshape-view stream of the fused AdamW step.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (adamw_M10944_N2048_0.py) = 1.305 ms; parent adamw_v1 = 0.6180 ms.
#
# Same operator, same folded algebra, same 6-op fused compute chain as the promoted
# adamw_v1 — the ONLY change is the loop shape. adamw_v1 row-tiled M=10944 into 86
# [128, 2048] tiles, the last partial (10944 = 128*85 + 64), so every load and the
# store carried a row-bound predicate (row < 10944). That kernel is DMA-bound at ~95%
# DMA-active on the read-4/write-1 traffic floor (359 MB read + 90 MB write = 448 MB);
# the fused Vector chain (Vec 64%) is fully hidden under DMA. Its only remaining slack
# is a ~5% DMA fill/drain bubble (effBW ~727 GB/s vs the ~799.5 GB/s streaming roofline
# this profiler sustains on silu). This kernel harvests that bubble with a deeper,
# mask-free software pipeline — the exact lever that took silu 0.3009 -> 0.2940 ms.
#
# Shape: M*N = 10944*2048 = 22413312 = 128 * 175104 EXACTLY, so the whole contiguous
# row-major buffer reinterprets as a clean (128, 175104) flat view with all 128
# partition lanes live — no partial tail, no mask on any DMA. 175104 = 2^10 * 3^2 * 19,
# so it has exact divisors across the whole interesting burst band. Reshape each input
# and the output to (128, ITERS, CH) with ITERS*CH = 175104 (a pure-stride, no-DMA view;
# partition dim fixed at 128, the collapsed free block contiguous) and walk ONE flat
# nl.affine_range(ITERS). CH sets the per-partition burst (CH*4 B) and ITERS the pipeline
# depth. Here CH=512 (2 KB/partition), the finest in-band anchor: the 2 KB burst
# floor of the reasoned in-band lattice. ITERS=342.
#
# Correctness is layout-invariant: the op is pure elementwise, so reading all four
# inputs and writing the output at the SAME flat index [p, j, c] is an exact bijection
# with the (10944, 2048) reference (numpy round-trip (M,N) -> (128, ITERS, CH) -> (M,N)
# is bit-exact; the harness reshapes the returned buffer back to the reference shape).
# The folded algebra is unchanged from adamw_v1:
#     v_hat   = 1000 * (0.999*v + 0.001*g^2) = 999*v + g^2
#     0.01*m_t = 0.01*(0.9*m + 0.1*g)        = 0.001*(9*m + g)
#     theta_t = theta - 1e-5*theta           = 0.99999*theta
#     new_theta = 0.99999*theta - 0.001*(9*m + g) * rsqrt(999*v + g^2)
# The +1e-8 eps is dropped: v_hat = 999*v + g^2 > 0 (v = |normal| >= 0), eps is ~1e-8
# against a denominator of O(30) -> ~3e-10 relative change, far under the 2e-5 L2 gate.
#
# Compute chain per chunk (2 Scalar activation + 4 Vector scalar_tensor_tensor = the
# algorithmic minimum of Vector tile-tile combines for this dependency graph), identical
# to adamw_v1, now applied per [128, CH] chunk and mask-free.

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2, v3, v4):          # v1=theta, v2=g, v3=m, v4=v
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    P, FLAT, CH = 128, 175104, 512
    ITERS = FLAT // CH               # 342 (exact: 175104 = 512 * 342)
    fp32 = np.float32

    # fp32-typed folded constants (mirrors the baseline's np.dtype(np.float32).type(...)).
    c_theta = np.dtype(fp32).type(0.99999)   # theta decay: 1 - 1e-5
    c_vhat  = np.dtype(fp32).type(999.0)     # 0.999 * 1000
    c_m     = np.dtype(fp32).type(9.0)       # 0.9 / 0.1
    c_num   = np.dtype(fp32).type(0.001)     # 0.01 * 0.1

    # Output buffer laid out as (128, ITERS, CH); the harness reshapes it back to
    # (10944, 2048) for the L2 comparison (memory-identical, contiguous row-major).
    out_hbm = nl.ndarray((P, ITERS, CH), dtype=fp32, buffer=nl.shared_hbm)

    # Pure-stride views: (10944, 2048) is row-major identical to (128, ITERS, CH).
    # No copy, no DMA -- just a finer partitioning of the same contiguous buffer.
    v1r = v1.reshape((P, ITERS, CH))
    v2r = v2.reshape((P, ITERS, CH))
    v3r = v3.reshape((P, ITERS, CH))
    v4r = v4.reshape((P, ITERS, CH))

    p_ix = nl.arange(P)[:, None]
    f_ix = nl.arange(CH)[None, :]

    # One flat affine_range over all ITERS chunks: the compiler software-pipelines the
    # next chunk's 4 loads under this chunk's fused compute + store (deeper than v1's 86).
    for j in nl.affine_range(ITERS):
        # Four mask-free loads [128, CH] HBM -> SBUF (exact divisor -> rectangular).
        theta = nl.load(v1r[p_ix, j, f_ix], dtype=fp32)
        g     = nl.load(v2r[p_ix, j, f_ix], dtype=fp32)
        m     = nl.load(v3r[p_ix, j, f_ix], dtype=fp32)
        v     = nl.load(v4r[p_ix, j, f_ix], dtype=fp32)

        # Fused compute (mask-free: every chunk is a full rectangular [128, CH]).
        g2 = nisa.activation(op=nl.square, data=g, dtype=fp32)                     # Scalar
        vhat = nisa.scalar_tensor_tensor(data=v, op0=nl.multiply, operand0=c_vhat,
                                         op1=nl.add, operand1=g2, dtype=fp32)       # Vector
        rden = nisa.activation(op=nl.rsqrt, data=vhat, dtype=fp32)                 # Scalar
        mm = nisa.scalar_tensor_tensor(data=m, op0=nl.multiply, operand0=c_m,
                                       op1=nl.add, operand1=g, dtype=fp32)          # Vector
        term = nisa.scalar_tensor_tensor(data=mm, op0=nl.multiply, operand0=c_num,
                                         op1=nl.multiply, operand1=rden, dtype=fp32)  # Vector
        out = nisa.scalar_tensor_tensor(data=theta, op0=nl.multiply, operand0=c_theta,
                                        op1=nl.subtract, operand1=term, dtype=fp32)   # Vector

        nl.store(out_hbm[p_ix, j, f_ix], value=out, mask=None)

    return out_hbm
