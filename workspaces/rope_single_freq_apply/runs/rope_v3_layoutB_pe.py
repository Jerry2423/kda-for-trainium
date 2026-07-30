# rope_v3_layoutB_pe — layout-B (128-partition packed) RoPE apply, hybrid build:
# the half-swap+negate is built on the PE (Tensor) engine via a 0/+-1 permutation
# matmul; the cos/sin broadcasts stay on the Scalar engine.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency = 1.1418 ms. Parent = rope_v1.py (layout A, 0.9445 ms).
# Sibling = rope_v2_layoutB_scalar_w512.py (all-Scalar build, 0.7337 ms, exact).
#
# Operator (cross-half rotary embedding, elementwise over the free axis):
#   x0 = x_in[:64]      x1 = x_in[64:]
#   out0 = x0*cos - x1*sin   -> output rows  0:64
#   out1 = x0*sin + x1*cos   -> output rows 64:128
#
# Layout-B packed algebra (sign baked into the swapped operand so the final
# combine is a single 128-partition add):
#   A          = [x0; x1]     (natural 128-partition load, 1 DMA)
#   cos_pack   = [cos; cos]    (Scalar broadcast, SBUF)
#   sin_pack   = [sin; sin]    (Scalar broadcast, SBUF)
#   x_swap_neg = [-x1; +x0]    (PERMUTATION MATMUL on the PE engine -> PSUM)
#   M1  = A          * cos_pack   -> [ x0*cos ;  x1*cos ]   (both SBUF)
#   M2  = x_swap_neg * sin_pack   -> [-x1*sin ;  x0*sin ]   (PSUM * SBUF)
#   out = M1 + M2                 -> [out0; out1]
#
# Why this variant exists: the promoted sibling builds x_swap_neg on the Scalar
# engine, which pushes Scalar active to ~92% at W=512. This hybrid moves that one
# build onto the otherwise-idle PE (Tensor) engine, testing whether spreading the
# build across PE + Scalar lowers the co-limiter. The half-swap+negate is a dense
# 0/+-1 permutation: nc_matmul(stationary=swap_t, moving=A) = swap_t.T @ A, and
# swap_t is chosen so swap_t.T = [[0,-I],[I,0]], giving [-x1; +x0]. This is genuine
# [128,<=512] streaming matmul work on the PE (moving free <= 512 -> W=512), not a
# free copy, so its benefit hinges on hiding on the idle PE.
#
# Placement rule: nc_matmul writes PSUM, and nisa.tensor_tensor forbids BOTH
# operands in PSUM. So M2 pairs the PSUM x_swap_neg with the SBUF sin_pack, and M1
# pairs two SBUF operands -- at most one PSUM operand per tensor_tensor.
#
# CORRECTNESS CAVEAT: fp32 nc_matmul on trn2 is not documented bit-exact (it may
# decompose to tf32/bf16 internally; only the accumulation is fp32). Even a 0/+-1
# permutation may round the mantissa of A on the way in. This kernel therefore
# MUST clear the full 5-seed relative-L2 gate before its latency is trusted -- a
# nonzero rel-L2 (vs the sibling's exact 0.0) is the expected signal of that
# decomposition, and if it exceeds 2e-5 the candidate is rejected on correctness.
#
# HBM stays at the read-once/write-once floor (402.65 MB): x once (natural
# [128,W] load), cos/sin once each, out once. The permutation matrix is a compiled
# constant loaded to SBUF once (not HBM traffic per tile). W=512 divides S=2^18.

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim


def _swap_neg_matrix(d_head, half_d):
    # stationary operand for nc_matmul; nc_matmul(stationary, moving) =
    # stationary.T @ moving. We want (stationary.T @ A) = [-x1; +x0], i.e.
    # stationary.T = [[0, -I], [I, 0]] -> stationary = [[0, I], [-I, 0]]:
    #   stationary[i, half+i] = +1   (feeds +x0 into output rows half:d_head)
    #   stationary[half+i, i] = -1   (feeds -x1 into output rows 0:half)
    m = np.zeros((d_head, d_head), dtype=np.float32)
    for i in range(half_d):
        m[i, half_d + i] = 1.0
        m[half_d + i, i] = -1.0
    return m


@nki.jit
def kernel(x_in, cos, sin):
    d_head, S = x_in.shape
    half_d = d_head // 2
    assert d_head <= 128
    assert tuple(cos.shape) == (half_d, S)
    assert cos.shape == sin.shape
    assert x_in.dtype == cos.dtype == sin.dtype

    # nc_matmul moving free dim <= 512 on trn2, so the tile width is capped at 512.
    tile_width = 512
    assert S % tile_width == 0

    out = nl.ndarray((d_head, S), dtype=x_in.dtype, buffer=nl.shared_hbm)

    i_pf = nl.arange(d_head)[:, None]     # full 128-partition index
    i_ph = nl.arange(half_d)[:, None]     # half 64-partition index
    i_f = nl.arange(tile_width)[None, :]  # free indices
    i_d = nl.arange(d_head)[None, :]      # 128 free index for the permutation matrix

    # Load the 0/+-1 permutation once into SBUF (reused for every tile). It is a
    # compiled constant -> no per-tile HBM traffic.
    swap_const = nl.shared_constant(_swap_neg_matrix(d_head, half_d))
    swap_t = nl.ndarray((par_dim(d_head), d_head), dtype=x_in.dtype, buffer=nl.sbuf)
    swap_t[i_pf, i_d] = nl.load(swap_const[i_pf, i_d])

    for j in nl.affine_range(S // tile_width):
        base = j * tile_width

        # A = [x0; x1] on all 128 partitions -- one natural DMA.
        a = nl.ndarray((par_dim(d_head), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        a[i_pf, i_f] = nl.load(x_in[i_pf, base + i_f])

        # cos_pack = [cos; cos] and sin_pack = [sin; sin]: Scalar-engine broadcasts.
        cos_pack = nl.ndarray((par_dim(d_head), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        cos_pack[i_ph, i_f] = nl.load(cos[i_ph, base + i_f])
        cos_pack[half_d + i_ph, i_f] = nisa.activation(
            op=nl.copy, data=cos_pack[i_ph, i_f], scale=1.0, dtype=x_in.dtype)
        sin_pack = nl.ndarray((par_dim(d_head), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        sin_pack[i_ph, i_f] = nl.load(sin[i_ph, base + i_f])
        sin_pack[half_d + i_ph, i_f] = nisa.activation(
            op=nl.copy, data=sin_pack[i_ph, i_f], scale=1.0, dtype=x_in.dtype)

        # x_swap_neg = [-x1; +x0] via a permutation matmul on the PE engine -> PSUM.
        # nc_matmul(swap_t, a) = swap_t.T @ a = [[0,-I],[I,0]] @ [x0; x1] = [-x1; +x0].
        x_swap_neg = nl.ndarray((par_dim(d_head), tile_width), dtype=x_in.dtype, buffer=nl.psum)
        x_swap_neg[i_pf, i_f] = nisa.nc_matmul(swap_t[i_pf, i_d], a[i_pf, i_f])

        # Three packed tensor_tensor over [128, W]. M2 keeps its PSUM operand paired
        # with an SBUF operand (sin_pack); M1 is SBUF*SBUF -- never both in PSUM.
        m1 = nisa.tensor_tensor(a[i_pf, i_f], cos_pack[i_pf, i_f], nl.multiply)           # [ x0*cos ;  x1*cos ]
        m2 = nisa.tensor_tensor(x_swap_neg[i_pf, i_f], sin_pack[i_pf, i_f], nl.multiply)  # [-x1*sin ;  x0*sin ]
        out_tile = nl.ndarray((par_dim(d_head), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        out_tile[i_pf, i_f] = nisa.tensor_tensor(m1, m2, nl.add)  # [out0; out1]

        nl.store(out[i_pf, base + i_f], value=out_tile[i_pf, i_f])

    return out
