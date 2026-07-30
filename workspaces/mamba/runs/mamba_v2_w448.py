import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np


@nki.jit
def kernel(delta, u, a, b, c):
    """Mamba selective-scan (SSM recurrence) — sequence-tiled, channels/state inner.

    Inputs:
        delta: (C=256, M=7168) fp32
        u:     (C=256, M=7168) fp32
        a:     (C=256, S=16)   fp32
        b:     (S=16,  M=7168) fp32
        c:     (S=16,  M=7168) fp32
    Output:
        out:   (C=256, M=7168) fp32

    The first-order linear recurrence state_i = deltaA_i * state_{i-1} + deltaBu_i
    is computed by nki.isa.tensor_tensor_scan (Vector Engine primitive).

    The sequence axis M is processed in tiles of seq_tile_size, iterated over a
    sequential (loop-carried) range: the scan's last column is carried forward
    as the next tile's initial state. This bounds the live SBUF working set to
    one seq-tile wide, eliminating the whole-M spill that made the untiled
    channels-outer kernel slower than the baseline.
    """
    channels, seq_len = delta.shape
    _, state_size = a.shape

    assert channels % 128 == 0

    channel_psize = nl.tile_size.pmax  # 128
    n_channel_tile = channels // channel_psize

    seq_tile_size = 448
    assert seq_len % seq_tile_size == 0
    n_seq_tile = seq_len // seq_tile_size

    output = nl.ndarray((channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)

    # Scan state carried across sequence tiles: last scan column per (channel tile, state)
    scan_state = nl.zeros((n_channel_tile, nl.par_dim(channel_psize), state_size),
                          dtype=delta.dtype)

    # Sequential over sequence tiles (loop-carried scan dependency)
    for i_seq_tile in nl.sequential_range(n_seq_tile):
        seq_start = i_seq_tile * seq_tile_size
        seq_end = seq_start + seq_tile_size

        # Output accumulator for THIS sequence tile only (sum over states)
        scanC_accum = nl.zeros((n_channel_tile, nl.par_dim(channel_psize), seq_tile_size),
                               dtype=delta.dtype)

        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            # Load delta and u once per (seq tile, channel tile)
            delta_i = nl.load(delta[channel_start:channel_start + channel_psize,
                                    seq_start:seq_end])
            u_i = nl.load(u[channel_start:channel_start + channel_psize,
                            seq_start:seq_end])

            for i_state in nl.affine_range(state_size):
                # Per-partition scalar: a[c, s] scales the exp argument
                A_i = nl.load(a[channel_start:channel_start + channel_psize, i_state])

                # deltaA = exp(a[c,s] * delta[c,m]) via Scalar Engine
                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)

                # deltaBu = (delta * u) * b[s, :]  (b broadcast over channels)
                B_i = nl.load(b[i_state:i_state + 1, seq_start:seq_end])
                deltaU = nisa.tensor_tensor(delta_i, u_i, op=nl.multiply)
                B_i_bcast = B_i.broadcast_to((channel_psize, seq_tile_size))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

                # First-order linear scan; initial = carried last column for tiles > 0
                initial_state = (scan_state[i_channel_tile, 0:channel_psize, i_state]
                                 if i_seq_tile > 0 else 0)
                scan_res = nki.isa.tensor_tensor_scan(
                    deltaA, deltaBu, initial=initial_state,
                    op0=np.multiply, op1=np.add)

                # Carry this tile's last column forward as the next tile's initial state
                if i_seq_tile < n_seq_tile - 1:
                    scan_state[i_channel_tile, 0:channel_psize, i_state:i_state + 1] = \
                        scan_res[:, seq_tile_size - 1:seq_tile_size]

                # scanC = scan_res * c[s, :]  (c broadcast over channels); accumulate
                C_i = nl.load(c[i_state:i_state + 1, seq_start:seq_end])
                C_i_bcast = C_i.broadcast_to((channel_psize, seq_tile_size))
                scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)
                scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_tile_size] += scanC

        # Store the completed sequence-tile slice for each channel tile
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            nl.store(output[channel_start:channel_start + channel_psize,
                            seq_start:seq_end],
                     value=scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_tile_size])

    return output
