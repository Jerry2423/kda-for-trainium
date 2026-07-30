import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Batched fp32 matmul out[b] = lhs[b] @ rhs[b], b in 0..15.

    Shapes: v1 lhs (16,4096,64)=(B,M,K); v2 rhs (16,64,4096)=(B,K,N);
    out (16,4096,4096)=(B,M,N). K=64 <= 128 so the whole contraction is one
    Tensor-Engine pass per output tile (single nc_matmul, no K-accumulation).

    Batch-blocked schedule (batch_block = 4). Where bmm_v2 processes one batch at
    a time — transpose that batch's 32 m-tiles into a resident pack, then stream its
    256 main matmuls — this variant hoists the transpose-pack and rhs load to cover a
    block of batch_block adjacent batches, then runs one deeper main-matmul stream over
    all batch_block*256 sites with no transpose interleaved. The transpose->matmul
    Pass-1/Pass-2 transition therefore occurs B/batch_block times instead of B, giving
    the compiler a longer independent-matmul run across the batch boundary to hide the
    per-instruction schedule gap. This is a pure schedule change: identical single-pass
    K=64 math to bmm_v2 (transpose-before-use is exact, so packing all m-tiles of the
    block up front then matmul is bit-identical to per-batch), same total matmul count.

    Resident SBUF per block: rhs 64x(batch_block*4096)x4B + packed transpose
    64x(batch_block*4096)x4B = batch_block*32 KB/partition (128 KB at batch_block=4),
    well within the ~208 KB budget (no spill; HBM read/write stay at the once-each
    floor 34 MB / 1074 MB).
    """
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    B = 16
    M = 4096
    K = 64
    N = 4096
    M_TILES = 32                    # 128-row m-tiles per batch
    BATCH_BLK = 4                   # batches processed per block
    N_BLOCKS = B // BATCH_BLK       # 4 blocks
    N_CHUNK = 512
    WIDE = 1024
    N_WIDE = N // WIDE              # 4 wide store groups
    G = WIDE // N_CHUNK             # 2 PSUM banks feed one wide store

    out = nl.ndarray((B, M, N), dtype=np.float32, buffer=nl.shared_hbm)

    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]],
        dtype=np.float32)

    for blk in nl.affine_range(N_BLOCKS):
        # Load the block's rhs tiles resident: [BATCH_BLK, 64, 4096].
        rhs_sb = nl.ndarray((BATCH_BLK, par_dim(K), N), dtype=np.float32,
                            buffer=nl.sbuf)
        for bb in nl.affine_range(BATCH_BLK):
            rhs_sb[bb, nl.arange(K)[:, None], nl.arange(N)[None, :]] = nl.load(
                v2[BATCH_BLK * blk + bb, nl.arange(K)[:, None], nl.arange(N)[None, :]],
                dtype=np.float32)

        # Pass 1: transpose every m-tile of every batch in the block into one
        # resident pack [BATCH_BLK, 64, 32*128=4096].
        lhs_t_pack = nl.ndarray((BATCH_BLK, par_dim(K), M_TILES * 128),
                                dtype=np.float32, buffer=nl.sbuf)
        for bb in nl.affine_range(BATCH_BLK):
            b = BATCH_BLK * blk + bb
            for s in nl.affine_range(M_TILES):
                lhs_sb = nl.ndarray((par_dim(128), K), dtype=np.float32,
                                    buffer=nl.sbuf)
                lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]] = nl.load(
                    v1[b, 128 * s + nl.arange(128)[:, None], nl.arange(K)[None, :]],
                    dtype=np.float32)
                psum_t = nl.ndarray((par_dim(K), 128), dtype=np.float32,
                                    buffer=nl.psum)
                psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                    lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]],
                    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    is_transpose=True, is_moving_onezero=True)
                lhs_t_pack[bb, nl.arange(K)[:, None], 128 * s + nl.arange(128)[None, :]] = nl.copy(
                    psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]],
                    dtype=np.float32)

        # Pass 2: one deep main-matmul stream over all BATCH_BLK*256 sites, no
        # transpose interleaved. 1024-wide coalesced stores.
        for bb in nl.affine_range(BATCH_BLK):
            b = BATCH_BLK * blk + bb
            for s in nl.affine_range(M_TILES):
                for w in nl.affine_range(N_WIDE):
                    acc = nl.ndarray((G, par_dim(128), N_CHUNK), dtype=np.float32,
                                     buffer=nl.psum)
                    for j in nl.affine_range(G):
                        c = G * w + j
                        acc[j, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.nc_matmul(
                            lhs_t_pack[bb, nl.arange(K)[:, None], 128 * s + nl.arange(128)[None, :]],
                            rhs_sb[bb, nl.arange(K)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])
                    out_sb = nl.ndarray((par_dim(128), WIDE), dtype=np.float32,
                                        buffer=nl.sbuf)
                    for j in nl.affine_range(G):
                        out_sb[nl.arange(128)[:, None], N_CHUNK * j + nl.arange(N_CHUNK)[None, :]] = nl.copy(
                            acc[j, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                            dtype=np.float32)
                    nl.store(
                        out[b, 128 * s + nl.arange(128)[:, None],
                            WIDE * w + nl.arange(WIDE)[None, :]],
                        value=out_sb[nl.arange(128)[:, None], nl.arange(WIDE)[None, :]])

    return out
