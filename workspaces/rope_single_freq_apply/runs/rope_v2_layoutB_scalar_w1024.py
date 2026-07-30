# rope_v2_layoutB_scalar — layout-B (128-partition packed) RoPE apply, exact fp32.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (rope_single_freq_apply_B1_H64_N4096_D128_0.py) = 1.1418 ms.
# Parent = rope_v1.py (layout A, 64-partition, 0.9445 ms, VECTOR-BOUND).
#
# Operator (cross-half rotary embedding, elementwise over the free axis):
#   x0 = x_in[:64]      x1 = x_in[64:]
#   out0 = x0*cos - x1*sin   -> output rows  0:64
#   out1 = x0*sin + x1*cos   -> output rows 64:128
#
# The predecessor 64-partition kernel is VECTOR-BOUND: it runs SIX tensor_tensor passes over
# [64, W] tiles (only 64 of 128 vector lanes), and those six passes co-limit the
# wall clock (Vec 91.6%; effBW 427 GB/s « the 781 GB/s roofline, so NOT
# bandwidth-bound). tensor_tensor latency is per-free-element and independent of
# partition count, so packing BOTH output halves onto all 128 partitions lets
# THREE [128, W] passes do the work of six [64, W] passes.
#
# Layout B packed algebra (the sign is baked into x_swap_neg so the final combine
# is a single 128-partition add):
#   A         = [x0; x1]      (natural 128-partition load, 1 DMA)
#   cos_pack  = [cos; cos]     (broadcast cos 64 -> 128 partitions)
#   sin_pack  = [sin; sin]     (broadcast sin 64 -> 128 partitions)
#   x_swap_neg= [-x1; +x0]     (swap the two 64-partition halves, negate the top)
#   M1  = A          * cos_pack   -> [ x0*cos ;  x1*cos ]
#   M2  = x_swap_neg * sin_pack   -> [-x1*sin ;  x0*sin ]
#   out = M1 + M2                 -> [ x0*cos - x1*sin ; x1*cos + x0*sin ] == [out0; out1]
#
# The cross-partition builds (64->128 broadcast, 64<->64 half-swap) are the whole
# bet: trn2 compute engines are partition-locked for >64 partitions, but Vector,
# Scalar and GpSimd all support CROSS-HALF movement (src 0:64 <-> dst 64:128) for
# a <=64-partition op. We route every build onto the SCALAR engine via
# nisa.activation(op=nl.copy, scale=+-1): Scalar is idle here (0.18%), whereas the
# Vector engine is the bound one (routing the copies there would be 3 packed + 4
# builds ~= 7xS vector-equiv, WORSE than layout A's 6xS). The negate FUSES into the
# top-half move (scale=-1), an exact fp32 sign flip. No SBUF->SBUF DMA is used
# (DGE is disabled by the NKIBench compiler flags, and DMA is already co-limiting).
#
# Exactness: the builds are pure data movement + an IEEE sign flip, and fp32 makes
# a + (-b) == a - b and (-x1)*sin == -(x1*sin) bit-identically, so this reproduces
# layout A's arithmetic -> rel-L2 expected 0.0 on all seeds.
#
# HBM stays at the read-once/write-once floor (402.65 MB): x is read once as the
# natural [128, W] load (128 MiB), cos/sin once each (64 MiB each), out written once
# as one [128, W] store (128 MiB). The 64->128 broadcast is an on-chip Scalar copy,
# NOT a second HBM read of cos/sin. W=2048 divides S=2^18 exactly (S/W=128 iters),
# so every tile is rectangular and mask-free.

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(x_in, cos, sin):
    d_head, S = x_in.shape
    half_d = d_head // 2
    assert d_head <= 128
    assert tuple(cos.shape) == (half_d, S)
    assert cos.shape == sin.shape
    assert x_in.dtype == cos.dtype == sin.dtype

    # Free-axis tile width: a power of two dividing S = 2^18 -> mask-free tiles.
    tile_width = 1024
    assert S % tile_width == 0

    out = nl.ndarray((d_head, S), dtype=x_in.dtype, buffer=nl.shared_hbm)

    i_pf = nl.arange(d_head)[:, None]     # 0..d_head-1   full 128-partition index
    i_ph = nl.arange(half_d)[:, None]     # 0..half_d-1   half 64-partition index
    i_f = nl.arange(tile_width)[None, :]  # 0..tile_width-1 free indices

    for j in nl.affine_range(S // tile_width):
        base = j * tile_width

        # A = [x0; x1] on all 128 partitions -- one natural DMA (no realign copy).
        a = nl.ndarray((par_dim(d_head), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        a[i_pf, i_f] = nl.load(x_in[i_pf, base + i_f])

        # cos_pack = [cos; cos]: load cos into the top half, then broadcast it to the
        # bottom half with a cross-half Scalar copy (no second HBM read of cos).
        cos_pack = nl.ndarray((par_dim(d_head), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        cos_pack[i_ph, i_f] = nl.load(cos[i_ph, base + i_f])
        cos_pack[half_d + i_ph, i_f] = nisa.activation(
            op=nl.copy, data=cos_pack[i_ph, i_f], scale=1.0, dtype=x_in.dtype)

        # sin_pack = [sin; sin]: same broadcast on the Scalar engine.
        sin_pack = nl.ndarray((par_dim(d_head), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        sin_pack[i_ph, i_f] = nl.load(sin[i_ph, base + i_f])
        sin_pack[half_d + i_ph, i_f] = nisa.activation(
            op=nl.copy, data=sin_pack[i_ph, i_f], scale=1.0, dtype=x_in.dtype)

        # x_swap_neg = [-x1; +x0]: swap the two halves of A and negate the top half,
        # all on the Scalar engine. Baking the sign here makes the final combine a
        # single add over all 128 partitions (a partition-dependent add/sub would be
        # two ops, defeating the 3-pass packing).
        x_swap_neg = nl.ndarray((par_dim(d_head), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        x_swap_neg[i_ph, i_f] = nisa.activation(          # top    = -x1  (from rows 64:128)
            op=nl.copy, data=a[half_d + i_ph, i_f], scale=-1.0, dtype=x_in.dtype)
        x_swap_neg[half_d + i_ph, i_f] = nisa.activation(  # bottom = +x0  (from rows 0:64)
            op=nl.copy, data=a[i_ph, i_f], scale=1.0, dtype=x_in.dtype)

        # Three packed tensor_tensor passes over [128, W] (was six over [64, W]).
        m1 = nisa.tensor_tensor(a[i_pf, i_f], cos_pack[i_pf, i_f], nl.multiply)          # [ x0*cos ;  x1*cos ]
        m2 = nisa.tensor_tensor(x_swap_neg[i_pf, i_f], sin_pack[i_pf, i_f], nl.multiply)  # [-x1*sin ;  x0*sin ]
        out_tile = nl.ndarray((par_dim(d_head), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        out_tile[i_pf, i_f] = nisa.tensor_tensor(m1, m2, nl.add)  # [x0*cos - x1*sin ; x1*cos + x0*sin]

        # Single [128, W] store -> output rows 0:128 in one DMA.
        nl.store(out[i_pf, base + i_f], value=out_tile[i_pf, i_f])

    return out
