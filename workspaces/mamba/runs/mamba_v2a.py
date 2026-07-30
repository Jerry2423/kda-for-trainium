import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np


@nki.jit
def kernel(delta, u, a, b, c):
    """Mamba selective-scan (SSM recurrence) — sequence-tiled, seq loop innermost.

    Inputs:
        delta: (C=256, M=7168) fp32
        u:     (C=256, M=7168) fp32
        a:     (C=256, S=16)   fp32
        b:     (S=16,  M=7168) fp32
        c:     (S=16,  M=7168) fp32
    Output:
        out:   (C=256, M=7168) fp32

    Fallback sequence-tiled structure: channels outer, states next, and the
    sequence tiles iterated innermost over a fully-unrolled static range. A
    per-state scan_init [128,1] carries the previous tile's last scan column
    forward. delta/u are loaded whole-M once per channel tile (reused across
    all states and seq tiles) while the per-tile working buffers stay narrow,
    so the whole-M spill is avoided.

    Note: the sequence carry loop must NOT be affine_range (it carries a
    loop-dependent initial state); a full static unroll keeps the carry correct.
    """
    channels, seq_len = delta.shape
    _, state_size = a.shape

    assert channels % 128 == 0

    channel_psize = nl.tile_size.pmax  # 128
    n_channel_tile = channels // channel_psize

    seq_tile_size = 512
    assert seq_len % seq_tile_size == 0
    n_seq_tile = seq_len // seq_tile_size

    output = nl.ndarray((channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)

    for i_channel_tile in nl.affine_range(n_channel_tile):
        channel_start = i_channel_tile * channel_psize

        # Output accumulator (sum over states), whole-M for this channel tile
        scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)

        # Load delta and u once per channel tile (reused across states and seq tiles)
        delta_i = nl.load(delta[channel_start:channel_start + channel_psize, 0:seq_len])
        u_i = nl.load(u[channel_start:channel_start + channel_psize, 0:seq_len])

        for i_state in nl.affine_range(state_size):
            A_i = nl.load(a[channel_start:channel_start + channel_psize, i_state])

            # Carried last scan column for this state (reset per state)
            scan_init = nl.zeros((channel_psize, 1), dtype=delta_i.dtype)

            for i_seq_tile in nl.static_range(n_seq_tile):
                seq_start = i_seq_tile * seq_tile_size
                seq_end = seq_start + seq_tile_size

                # deltaA = exp(a[c,s] * delta[c,m]) via Scalar Engine
                deltaA = nisa.activation(
                    op=nl.exp,
                    data=delta_i[0:channel_psize, seq_start:seq_end],
                    scale=A_i)

                # deltaBu = (delta * u) * b[s, :]  (b broadcast over channels)
                B_i = nl.load(b[i_state:i_state + 1, seq_start:seq_end])
                deltaU = nisa.tensor_tensor(
                    delta_i[0:channel_psize, seq_start:seq_end],
                    u_i[0:channel_psize, seq_start:seq_end],
                    op=nl.multiply)
                B_i_bcast = B_i.broadcast_to((channel_psize, seq_tile_size))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

                # First-order linear scan; initial = carried last column
                scan_res = nki.isa.tensor_tensor_scan(
                    deltaA, deltaBu, initial=scan_init,
                    op0=np.multiply, op1=np.add)
                # Carry this tile's last column forward; [...] preserves the [128,1] shape
                scan_init[...] = scan_res[0:channel_psize, seq_tile_size - 1]

                # scanC = scan_res * c[s, :]  (c broadcast over channels); accumulate
                C_i = nl.load(c[i_state:i_state + 1, seq_start:seq_end])
                C_i_bcast = C_i.broadcast_to((channel_psize, seq_tile_size))
                scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)
                scanC_accum[0:channel_psize, seq_start:seq_end] += scanC

        # Store the completed channel tile
        nl.store(output[channel_start:channel_start + channel_psize, 0:seq_len],
                 value=scanC_accum[0:channel_psize, 0:seq_len])

    return output
