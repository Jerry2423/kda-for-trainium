# rope_v1 — first correct fp32 NKI kernel for single-frequency RoPE apply.
# Scored via verify.py on the remote Trainium profiler (trn2, single-core).
# Baseline latency (rope_single_freq_apply_B1_H64_N4096_D128_0.py) = 1.1418 ms.
#
# Operator (cross-half rotary embedding, elementwise over the free axis):
#   x0 = x_in[:64]      x1 = x_in[64:]
#   out0 = x0*cos - x1*sin   -> output rows  0:64
#   out1 = x0*sin + x1*cos   -> output rows 64:128
#
# Layout A (64-partition, no copy): each half of x is loaded into its OWN base-0
# [64, W] SBUF tile. nl.load returns a fresh tile based at partition 0 regardless
# of the HBM row offset, so x1 (HBM rows 64:128) lands aligned with x0 at base 0
# and the six nisa.tensor_tensor passes see both operands at the same partition
# base — no nl.copy realignment (which the baseline needs because it loads all
# 128 rows of x into one tile and slices x1 off partition base 64).
#
# The free axis S=262144 does not fit one partition (1 MB > ~208 KB usable), so it
# is tiled into width-W chunks and streamed with a single nl.affine_range: the
# iterations are independent, letting the compiler software-pipeline DMA against
# compute. W=2048 divides S exactly (S/W = 128 iters), so every tile is
# rectangular and mask-free with no tail handling. Four loads + two stores per
# tile keep HBM traffic at the read-once/write-once floor (402.65 MB total).
# out0 is stored before out1's products are computed so its temporaries can be
# freed/reused, keeping the live SBUF tile count down.

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
    tile_width = 4096
    assert S % tile_width == 0

    out = nl.ndarray((d_head, S), dtype=x_in.dtype, buffer=nl.shared_hbm)

    i_p = nl.arange(half_d)[:, None]     # 0..half_d-1 partition indices (base 0)
    i_f = nl.arange(tile_width)[None, :]  # 0..tile_width-1 free indices

    for j in nl.affine_range(S // tile_width):
        base = j * tile_width

        # Load each operand into its own base-0 [half_d, tile_width] tile. Loading
        # x_in rows 64:128 into a fresh tile lands them at partition base 0, so no
        # nl.copy realignment is needed before tensor_tensor.
        x0 = nl.ndarray((par_dim(half_d), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        x1 = nl.ndarray((par_dim(half_d), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        c = nl.ndarray((par_dim(half_d), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        s = nl.ndarray((par_dim(half_d), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        x0[i_p, i_f] = nl.load(x_in[i_p, base + i_f])
        x1[i_p, i_f] = nl.load(x_in[half_d + i_p, base + i_f])
        c[i_p, i_f] = nl.load(cos[i_p, base + i_f])
        s[i_p, i_f] = nl.load(sin[i_p, base + i_f])

        # out0 = x0*cos - x1*sin  -> output rows 0:64. Stored early so e_cos/o_sin
        # can be freed before the out1 products are computed.
        e_cos = nisa.tensor_tensor(x0[i_p, i_f], c[i_p, i_f], nl.multiply)
        o_sin = nisa.tensor_tensor(x1[i_p, i_f], s[i_p, i_f], nl.multiply)
        out0 = nl.ndarray((par_dim(half_d), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        out0[i_p, i_f] = nisa.tensor_tensor(e_cos, o_sin, nl.subtract)
        nl.store(out[i_p, base + i_f], value=out0[i_p, i_f])

        # out1 = x0*sin + x1*cos  -> output rows 64:128 (partition-offset store).
        e_sin = nisa.tensor_tensor(x0[i_p, i_f], s[i_p, i_f], nl.multiply)
        o_cos = nisa.tensor_tensor(x1[i_p, i_f], c[i_p, i_f], nl.multiply)
        out1 = nl.ndarray((par_dim(half_d), tile_width), dtype=x_in.dtype, buffer=nl.sbuf)
        out1[i_p, i_f] = nisa.tensor_tensor(o_cos, e_sin, nl.add)
        nl.store(out[half_d + i_p, base + i_f], value=out1[i_p, i_f])

    return out
