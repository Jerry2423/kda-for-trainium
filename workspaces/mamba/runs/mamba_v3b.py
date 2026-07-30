import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np


@nki.jit
def kernel(delta, u, a, b, c):
    """Mamba selective-scan (SSM recurrence) — sequence-tiled, DMA-broadcast b/c.

    Inputs:
        delta: (C=256, M=7168) fp32
        u:     (C=256, M=7168) fp32
        a:     (C=256, S=16)   fp32
        b:     (S=16,  M=7168) fp32
        c:     (S=16,  M=7168) fp32
    Output:
        out:   (C=256, M=7168) fp32

    Identical recurrence, sequence tiling and loop structure as the promoted
    seq-tile-outer kernel. The only change is the mechanism that materializes
    the b[s,:]/c[s,:] partition broadcast: instead of loading a [1, seq_tile]
    row into SBUF and letting the compiler broadcast it across the 128 channel
    partitions on the Tensor engine (nc_matmul(ones, row)), the row is loaded
    directly from HBM into a [128, seq_tile] SBUF tile via a partition-stride-0
    broadcast DMA (every partition reads the same HBM row). This spends idle DMA
    bandwidth instead of the saturated Tensor engine; the broadcast values are
    identical, so the result is unchanged.
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

        # Partition / free index vectors for the stride-0 broadcast access pattern.
        # p_idx * 0 forces partition stride 0 so every partition reads the same HBM row.
        p_idx = nl.arange(channel_psize)[:, None]
        f_idx = nl.arange(seq_tile_size)[None, :]

        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            delta_i = nl.load(delta[channel_start:channel_start + channel_psize,
                                    seq_start:seq_end])
            u_i = nl.load(u[channel_start:channel_start + channel_psize,
                            seq_start:seq_end])

            for i_state in nl.affine_range(state_size):
                A_i = nl.load(a[channel_start:channel_start + channel_psize, i_state])

                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)

                # Partition-stride-0 broadcast DMA: load b[s,:] into all 128 partitions
                # directly from HBM (idle DMA) instead of a PE partition broadcast.
                B_bcast = nl.ndarray((nl.par_dim(channel_psize), seq_tile_size),
                                     dtype=delta.dtype)
                nisa.dma_copy(dst=B_bcast[p_idx, f_idx],
                              src=b[i_state + p_idx * 0, seq_start + f_idx])
                deltaU = nisa.tensor_tensor(delta_i, u_i, op=nl.multiply)
                deltaBu = nisa.tensor_tensor(deltaU, B_bcast, op=nl.multiply)

                initial_state = (scan_state[i_channel_tile, 0:channel_psize, i_state]
                                 if i_seq_tile > 0 else 0)
                scan_res = nki.isa.tensor_tensor_scan(
                    deltaA, deltaBu, initial=initial_state,
                    op0=np.multiply, op1=np.add)

                if i_seq_tile < n_seq_tile - 1:
                    scan_state[i_channel_tile, 0:channel_psize, i_state:i_state + 1] = \
                        scan_res[:, seq_tile_size - 1:seq_tile_size]

                C_bcast = nl.ndarray((nl.par_dim(channel_psize), seq_tile_size),
                                     dtype=delta.dtype)
                nisa.dma_copy(dst=C_bcast[p_idx, f_idx],
                              src=c[i_state + p_idx * 0, seq_start + f_idx])
                scanC = nisa.tensor_tensor(scan_res, C_bcast, op=nl.multiply)
                scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_tile_size] += scanC

        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            nl.store(output[channel_start:channel_start + channel_psize,
                            seq_start:seq_end],
                     value=scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_tile_size])

    return output
