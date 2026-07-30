import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np


@nki.jit
def kernel(delta, u, a, b, c):
    """Mamba selective-scan (SSM recurrence) — channels-outer, state-inner.

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

    Loop order: channel tiles outer, states inner — delta/u loaded once per
    channel tile (not 16x as in the baseline).
    """
    channels, seq_len = delta.shape
    _, state_size = a.shape

    assert channels % 128 == 0

    channel_psize = nl.tile_size.pmax  # 128
    n_channel_tile = channels // channel_psize

    output = nl.ndarray((channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)

    for i_channel_tile in nl.affine_range(n_channel_tile):
        channel_start = i_channel_tile * channel_psize

        # Accumulator for the output (sum over states)
        scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len),
                               dtype=delta.dtype)

        # Load delta and u ONCE per channel tile (not per state)
        delta_i = nl.load(delta[channel_start:channel_start + channel_psize,
                                0:seq_len])
        u_i = nl.load(u[channel_start:channel_start + channel_psize,
                        0:seq_len])

        for i_state in nl.affine_range(state_size):
            # Per-partition scalar: a[c, s] scales the exp argument
            A_i = nl.load(a[channel_start:channel_start + channel_psize,
                            i_state])

            # deltaA = exp(a[c,s] * delta[c,m]) via Scalar Engine
            deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)

            # deltaU = delta[c,m] * u[c,m]
            deltaU = nisa.tensor_tensor(delta_i, u_i, op=nl.multiply)

            # Broadcast b[s, :] over channels
            B_i = nl.load(b[i_state:i_state + 1, 0:seq_len])
            B_i_bcast = B_i.broadcast_to((channel_psize, seq_len))
            deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

            # First-order linear scan: state_i = deltaA_i * state_{i-1} + deltaBu_i
            scan_res = nki.isa.tensor_tensor_scan(
                deltaA, deltaBu, initial=0,
                op0=np.multiply, op1=np.add)

            # Broadcast c[s, :] over channels and multiply
            C_i = nl.load(c[i_state:i_state + 1, 0:seq_len])
            C_i_bcast = C_i.broadcast_to((channel_psize, seq_len))
            scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

            # Accumulate over states
            scanC_accum += scanC

        # Store the completed channel tile
        nl.store(output[channel_start:channel_start + channel_psize,
                        0:seq_len],
                 value=scanC_accum)

    return output
