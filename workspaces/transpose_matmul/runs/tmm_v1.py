import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """First correct fp32 transpose-matmul kernel: out = lhs^T @ rhs.

    Shapes (as scored by NKIBench):
      lhs (K,M)=(2048,4096) K-major -> reshaped to v1 (128,16,4096)
      rhs (K,N)=(2048,10944)        -> reshaped to v2 (128,16,10944)
      out (M,N)=(4096,10944)        -> v3 (32,128,10944)
    MACs = M*N*K = 9.17e10.

    No explicit transpose stage. The reshape (K,.)->(128,16,.) maps the flat
    contraction index as k = k_in*16 + kt, so K sits on the PARTITION axis of
    both v1 and v2. nisa.nc_matmul(stationary, moving) computes stationary.T @
    moving with the contraction on the partition axis of both operands, which is
    exactly lhs^T @ rhs here: feeding stationary = v1[:, kt, m0:m0+128]
    (= [k_in=128, m_sub=128]) and moving = v2[:, kt, n0:n0+456]
    (= [k_in=128, n=456]) yields [m_sub=128, 456], accumulated over the 16 kt
    tiles to reconstruct the full K=2048 contraction (k_in*16 covers 0..2047).
    So the transposed-lhs operand is a free byproduct of the input layout.

    Loop structure: M-block-outer streaming GEMM. M (4096 = 32 tiles of 128) is
    split into 4 blocks of 1024 rows (8 subtiles each). Per block the lhs slice
    is loaded once and stays resident; rhs is streamed in N-chunks of 456 and
    reused across the 8 subtiles, so rhs is re-read 4x total (once per block),
    not 32x (which would go DMA-bound). N_CHUNK=456 is an exact divisor of
    N=10944 (456*24) and <= 512 (one fp32 PSUM bank), so every tile is full-size
    and there is NO tail-masking arithmetic anywhere -- removing the single
    largest correctness-bug surface (the NKIBench baseline uses N_CHUNK=1024 with
    mask=...>=0 tail bounds; this kernel deliberately replaces that).

    Output-index correspondence: subtile s of block mb writes out[8*mb+s, mi, n],
    and v3 (32,128,10944) reshapes to (4096,10944) so the logical output row of
    partition mi in that tile is m = mb*1024 + s*128 + mi -- matching
    out[m, n] = sum_k lhs[k, m] * rhs[k, n].

    Resident SBUF/partition: lhs_blk [128,16,1024] fp32 = 64 KB + rhs_chunk
    [128,16,456] fp32 ~= 28.5 KB (+ small out_sb temp) ~= 93 KB, well under the
    192 KB budget even with rhs double-buffering. PSUM: one 456-wide bank live
    per subtile (+ compiler rotation), far under the 8 banks. No spill; HBM stays
    near the once-lhs / ~4x-rhs / once-out floor.

    fp32 end to end: every load, the PSUM accumulator, the copy, and the store
    are np.float32; no dtype cast anywhere (bf16 compute split is a later phase).
    """
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    M = 4096
    K = 2048
    N = 10944
    K_IN = 128                      # contraction lanes on the partition axis
    K_TILES = 16                    # kt tiles; K_IN * K_TILES = 2048 = K
    M_BLK = 8                       # m-subtiles per block (each subtile 128 rows)
    M_BLOCKS = 4                    # M / (M_BLK*128) = 4096 / 1024
    M_BLK_ROWS = M_BLK * 128        # 1024 rows per block
    N_CHUNK = 456                   # exact divisor of N, <= 512 fp32 PSUM width
    N_CHUNKS = N // N_CHUNK         # 24; 456 * 24 = 10944 exactly

    out = nl.ndarray((32, 128, N), dtype=np.float32, buffer=nl.shared_hbm)

    for mb in nl.affine_range(M_BLOCKS):
        # Resident lhs block for these 1024 m-rows: [k_in=128, kt=16, 1024].
        # Loaded once per block, read-only afterward (64 KB/partition).
        lhs_blk = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                             dtype=np.float32, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            lhs_blk[nl.arange(K_IN)[:, None], kt, nl.arange(M_BLK_ROWS)[None, :]] = nl.load(
                v1[nl.arange(K_IN)[:, None], kt,
                   M_BLK_ROWS * mb + nl.arange(M_BLK_ROWS)[None, :]],
                dtype=np.float32)

        for c in nl.affine_range(N_CHUNKS):
            # rhs chunk, all 16 kt for this n-slice: [k_in=128, kt=16, 456].
            # Loaded once per (mb,c), reused across the 8 subtiles below.
            rhs_chunk = nl.ndarray((par_dim(K_IN), K_TILES, N_CHUNK),
                                   dtype=np.float32, buffer=nl.sbuf)
            for kt in nl.affine_range(K_TILES):
                rhs_chunk[nl.arange(K_IN)[:, None], kt, nl.arange(N_CHUNK)[None, :]] = nl.load(
                    v2[nl.arange(K_IN)[:, None], kt,
                       N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)

            for s in nl.affine_range(M_BLK):
                # Zero-initialized PSUM accumulator; Tensor-Engine accumulation
                # over the 16 kt tiles reconstructs the full K=2048 contraction.
                acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32,
                               buffer=nl.psum)
                for kt in nl.affine_range(K_TILES):
                    acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                        lhs_blk[nl.arange(K_IN)[:, None], kt,
                                128 * s + nl.arange(128)[None, :]],   # stationary [k_in,128]
                        rhs_chunk[nl.arange(K_IN)[:, None], kt,
                                  nl.arange(N_CHUNK)[None, :]])        # moving     [k_in,456]

                out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                    buffer=nl.sbuf)
                out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.copy(
                    acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)
                # out[8*mb+s, mi, 456*c+nj] -> logical row m = mb*1024 + s*128 + mi.
                nl.store(
                    out[8 * mb + s, nl.arange(128)[:, None],
                        N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                    value=out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

    return out
