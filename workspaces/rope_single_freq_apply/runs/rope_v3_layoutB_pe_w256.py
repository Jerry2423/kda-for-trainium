# rope_v3_layoutB_pe_w256 — finer free-axis tile width (W=256) on the promoted
# layout-B PE/hybrid RoPE apply. IDENTICAL algebra and dtype to
# rope_v3_layoutB_pe.py; the ONLY change is tile_width 512 -> 256, which just
# deepens the affine_range pipeline (S//W = 1024 iterations instead of 512).
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency = 1.1418 ms. Parent = rope_v3_layoutB_pe.py (W=512, 0.696 ms).
#
# Why finer W: the parent is DMA-co-limited (DMA-active 98.8%, effBW 578 GB/s =
# 74% of the 781 GB/s single-core streaming roofline) with HBM at the
# read-once/write-once floor. A deeper affine_range pipeline amortizes the fixed
# DMA fill/drain bubble over more steady-state steps, which can raise effBW toward
# the roofline. The counter-force: per-tile issue/descriptor cost (and one
# nc_matmul permutation per tile) grows as bursts shrink; below ~a few KB/partition
# that overhead can overtake the pipeline-depth gain and regress. W=256 = 1 KB/
# partition burst (W=512 was 2 KB). The measurement decides which force wins.
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
# Placement rule: nc_matmul writes PSUM, and nisa.tensor_tensor forbids BOTH
# operands in PSUM. So M2 pairs the PSUM x_swap_neg with the SBUF sin_pack, and M1
# pairs two SBUF operands -- at most one PSUM operand per tensor_tensor.
#
# CORRECTNESS: only the tile count changes vs the parent, so this is
# arithmetic-preserving -- expect per-seed rel-L2 = 0.0 on all of [0,21,42,63,84]
# (any nonzero would signal a compiler-introduced change). fp32 nc_matmul on trn2
# is not documented bit-exact, but the parent's 0/+-1 permutation was empirically
# exact at W=512; W=256 must clear the full 5-seed gate before its latency is
# trusted (rel-L2 > 2e-5 => rejected on correctness).
#
# HBM stays at the read-once/write-once floor (402.65 MB) + a one-time swap_const;
# finer W only shrinks per-tile SBUF (no spill risk). W=256 divides S=2^18.

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

    # Finer free-axis tile width (nc_matmul moving free dim <= 512; 256 <= 512).
    tile_width = 256
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
