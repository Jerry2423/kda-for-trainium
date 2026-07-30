import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np


@nki.jit
def kernel(delta, u, a, b, c):
    """Mamba selective-scan (SSM recurrence) — shared b/c broadcast + hoisted deltaU.

    Inputs:
        delta: (C=256, M=7168) fp32
        u:     (C=256, M=7168) fp32
        a:     (C=256, S=16)   fp32
        b:     (S=16,  M=7168) fp32
        c:     (S=16,  M=7168) fp32
    Output:
        out:   (C=256, M=7168) fp32

    Same recurrence and sequence tiling as the promoted seq-tile-outer kernel,
    with two orthogonal reductions of redundant work:
      1. The channel-independent b[s,:]/c[s,:] partition broadcasts are
         materialized once per state (state-outer, channel-inner) and reused
         across both channel tiles, halving the partition-broadcast matmuls.
      2. deltaU = delta * u is state-independent, so it is computed once per
         (seq tile, channel tile) and reused across all states, instead of
         being recomputed in every state iteration.

    Per (channel tile), the sum over states is accumulated in state order
    0..S-1 into its own buffer, so the reduction order is unchanged.
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

    # Scan state carried across sequence tiles: last scan column per (channel tile, state)
    scan_state = nl.zeros((n_channel_tile, nl.par_dim(channel_psize), state_size),
                          dtype=delta.dtype)

    for i_seq_tile in nl.sequential_range(n_seq_tile):
        seq_start = i_seq_tile * seq_tile_size
        seq_end = seq_start + seq_tile_size

        scanC_accum = nl.zeros((n_channel_tile, nl.par_dim(channel_psize), seq_tile_size),
                               dtype=delta.dtype)

        # Load delta/u once per (seq tile, channel tile) and precompute the
        # state-independent deltaU = delta * u once, reused across all states.
        delta_ct = nl.ndarray((n_channel_tile, nl.par_dim(channel_psize), seq_tile_size),
                              dtype=delta.dtype)
        deltaU_ct = nl.ndarray((n_channel_tile, nl.par_dim(channel_psize), seq_tile_size),
                               dtype=delta.dtype)
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            delta_ct[i_channel_tile] = nl.load(
                delta[channel_start:channel_start + channel_psize, seq_start:seq_end])
            u_i = nl.load(u[channel_start:channel_start + channel_psize, seq_start:seq_end])
            deltaU_ct[i_channel_tile] = nisa.tensor_tensor(
                delta_ct[i_channel_tile], u_i, op=nl.multiply)

        for i_state in nl.affine_range(state_size):
            # Materialize the channel-independent b/c broadcasts once per state,
            # shared across both channel tiles.
            B_i = nl.load(b[i_state:i_state + 1, seq_start:seq_end])
            C_i = nl.load(c[i_state:i_state + 1, seq_start:seq_end])
            B_bcast = nl.ndarray((nl.par_dim(channel_psize), seq_tile_size),
                                 dtype=delta.dtype)
            C_bcast = nl.ndarray((nl.par_dim(channel_psize), seq_tile_size),
                                 dtype=delta.dtype)
            B_bcast[...] = B_i.broadcast_to((channel_psize, seq_tile_size))
            C_bcast[...] = C_i.broadcast_to((channel_psize, seq_tile_size))

            for i_channel_tile in nl.affine_range(n_channel_tile):
                channel_start = i_channel_tile * channel_psize
                A_i = nl.load(a[channel_start:channel_start + channel_psize, i_state])

                deltaA = nisa.activation(op=nl.exp, data=delta_ct[i_channel_tile],
                                         scale=A_i)
                deltaBu = nisa.tensor_tensor(deltaU_ct[i_channel_tile], B_bcast,
                                             op=nl.multiply)

                initial_state = (scan_state[i_channel_tile, 0:channel_psize, i_state]
                                 if i_seq_tile > 0 else 0)
                scan_res = nki.isa.tensor_tensor_scan(
                    deltaA, deltaBu, initial=initial_state,
                    op0=np.multiply, op1=np.add)

                if i_seq_tile < n_seq_tile - 1:
                    scan_state[i_channel_tile, 0:channel_psize, i_state:i_state + 1] = \
                        scan_res[:, seq_tile_size - 1:seq_tile_size]

                scanC = nisa.tensor_tensor(scan_res, C_bcast, op=nl.multiply)
                scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_tile_size] += scanC

        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            nl.store(output[channel_start:channel_start + channel_psize,
                            seq_start:seq_end],
                     value=scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_tile_size])

    return output
