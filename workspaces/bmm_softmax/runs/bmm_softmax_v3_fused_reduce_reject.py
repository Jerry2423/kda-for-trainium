import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Batched fp32 matmul + row-softmax  out[b] = softmax_N(lhs[b] @ rhs[b]).

    Shapes (natural batched layout from the NKIBench reference's reshape-only
    transform_to_nki_inputs):
        v1: (16, 4096, 64) = [batch, m, k]   (lhs)
        v2: (16, 64, 4096) = [batch, k, n]   (rhs)
    Output:
        out: (16, 4096, 4096) = [batch, m, n]   (reshapes row-major to the ref)

    The reference is, per (b, m) row:
        x      = lhs[b] @ rhs[b]                    # scores over N
        max_x  = max_n x                            # row max over N
        exp_x  = exp(x - max_x)                     # max-shifted, overflow-safe
        sum_x  = sum_n exp_x
        out    = exp_x / sum_x                      # softmax over N (axis=2)

    Matmul core (mirrors the solved sibling bmm): nc_matmul(stationary, moving) =
    stationary.T @ moving, contraction dim k on the PARTITION axis of BOTH operands,
    both operands in SBUF. K=64 <= 128 so the whole contraction is ONE Tensor-Engine
    pass per score tile (no accumulation over K-tiles).
      * moving = rhs tile [k(par)=64, n(free)]. v2[b] is already [k, n], loads directly.
      * stationary must be [k(par)=64, m_in(free)=128]; a loaded lhs tile is
        [m_in(par)=128, k(free)=64], so transpose it once via the identity
        nc_matmul(is_transpose=True) idiom -> [k=64, m_in=128].

    Schedule (two-phase transpose-all, ported from the pure sibling bmm_v2): the lhs
    transpose is SEPARATED from the main matmul into two passes over a whole m-block.
    First pass: all M_SUB subtiles of the block are transposed up front into a resident
    packed [k=64, M_SUB*128] SBUF buffer. Second pass: the main matmuls run with NO
    transpose interleaved at the head of each burst, each subtile immediately followed by
    its softmax epilogue + store, so the compiler feeds the Tensor Engine a long
    uninterrupted matmul stream and hides the softmax Vector/Scalar work behind it.
    Pure schedule change: identical single-pass K=64 math (transpose-before-use is exact).

    Softmax epilogue (fused): the max-shifted softmax over the N free axis is computed
    with two Scalar/Vector fusions relative to the plain form:
      * tensor_reduce(nl.max, negate=True) writes -row_max DIRECTLY, folding the separate
        neg_max = row_max * -1 step into the max-reduce.
      * activation(op=exp, bias=neg_max, reduce_op=nl.add, reduce_res=row_sum,
        reduce_cmd=reset_reduce) produces exp_t AND the row sum sum_n exp_t in ONE Scalar
        pass -- the Scalar Engine reduces the activated data along the free axis into its
        accumulator registers and evicts them into row_sum, removing the standalone
        4096-wide tensor_reduce(add) Vector pass. reduce_cmd=reset_reduce zeroes the
        accumulators before this tile's reduction so the sum is exactly this row's terms.
    Softmax Vector passes therefore drop 3 -> 2 (max-reduce, normalize); the row sum moves
    onto the Scalar exp pass at the cost of one small accumulator-eviction instruction.
    fp32 throughout, same math (same set of exp terms summed in fp32); no softmax op reads
    or writes a PSUM tile (all reduce/activation/elementwise ops touch SBUF score/exp/out
    tiles; PSUM banks hold only matmul/transpose results, copied to SBUF immediately).

    SBUF budget per partition (M_SUB=32): lhs_t_pack 16 KB + rhs 16 KB + score 16 KB +
    exp_t 16 KB + out_t 16 KB + identity 0.5 KB ~= 80.5 KB of the ~208 KB trn2 usable,
    so no spill; HBM read/write stay at the read-once/write-once floor.
    """
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    B = 16             # batches
    M = 4096
    K = 64             # contraction depth (single Tensor-Engine pass, <= 128)
    N = 4096
    M_TILES = 32       # 4096 / 128
    M_SUB = 32         # m-subtiles transposed/streamed per m-block (whole batch)
    M_BLOCKS = M_TILES // M_SUB   # 1 (whole batch is one block)
    N_CHUNK = 512      # one fp32 PSUM bank in the free dim
    N_CHUNKS = N // N_CHUNK   # 8

    out = nl.ndarray((B, M, N), dtype=np.float32, buffer=nl.shared_hbm)

    # 128x128 identity in SBUF, used as the moving operand to transpose lhs tiles
    # on the Tensor Engine (is_transpose=True). Loaded once, reused for all tiles.
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]],
        dtype=np.float32)

    for b in nl.affine_range(B):
        # rhs[b] resident as [k(par)=64, n(free)=4096]; sliced per n-chunk below.
        rhs_sb = nl.ndarray((par_dim(K), N), dtype=np.float32, buffer=nl.sbuf)
        rhs_sb[nl.arange(K)[:, None], nl.arange(N)[None, :]] = nl.load(
            v2[b, nl.arange(K)[:, None], nl.arange(N)[None, :]], dtype=np.float32)

        for mblk in nl.affine_range(M_BLOCKS):
            # First pass: transpose all M_SUB subtiles of this block up front into a
            # resident packed [k=64, M_SUB*128] SBUF buffer, no matmul interleaved.
            lhs_t_pack = nl.ndarray((par_dim(K), M_SUB * 128), dtype=np.float32,
                                    buffer=nl.sbuf)
            for s in nl.affine_range(M_SUB):
                mt = M_SUB * mblk + s
                lhs_sb = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
                lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]] = nl.load(
                    v1[b, 128 * mt + nl.arange(128)[:, None], nl.arange(K)[None, :]],
                    dtype=np.float32)
                # Transpose lhs tile -> PSUM [k(par)=64, m_in(free)=128], copy to pack.
                # is_moving_onezero marks the identity (all ones/zeros) as a perf hint.
                psum_t = nl.ndarray((par_dim(K), 128), dtype=np.float32, buffer=nl.psum)
                psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                    lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]],
                    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    is_transpose=True, is_moving_onezero=True)
                lhs_t_pack[nl.arange(K)[:, None], 128 * s + nl.arange(128)[None, :]] = nl.copy(
                    psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]],
                    dtype=np.float32)

            # Second pass: main matmuls + fused softmax epilogue, no transpose
            # interleaved. score/exp_t/out_t live inside this loop so subtile s's
            # softmax can overlap subtile s+1's matmul burst.
            for s in nl.affine_range(M_SUB):
                mt = M_SUB * mblk + s

                # Build the full score row [m_in(par)=128, n(free)=4096] in SBUF from
                # 8 single-pass K=64 matmuls, each landing in one [128, 512] PSUM bank.
                score = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                for c in nl.affine_range(N_CHUNKS):
                    acc = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                     buffer=nl.psum)
                    acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.nc_matmul(
                        lhs_t_pack[nl.arange(K)[:, None], 128 * s + nl.arange(128)[None, :]],
                        rhs_sb[nl.arange(K)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])
                    score[nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]] = nl.copy(
                        acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                        dtype=np.float32)

                # Fused softmax over the N free axis (4096-wide), entirely in SBUF, fp32.
                # neg_max[128,1] = -max_n score : the max-reduce writes -row_max directly
                # via negate=True (folds the separate row_max * -1 step).
                neg_max = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                neg_max[nl.arange(128)[:, None], 0] = nisa.tensor_reduce(
                    nl.max, data=score[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    axis=[1], negate=True, dtype=np.float32)
                # exp_t[128,4096] = exp(score - row_max) AND row_sum[128,1] = sum_n exp_t
                # in one Scalar pass: the exp activation reduces its output along the free
                # axis (reduce_op=nl.add) into the accumulator registers and evicts them
                # into row_sum. reduce_cmd=reset_reduce zeroes the accumulators first so
                # the sum is exactly this row's exp terms. This removes the standalone
                # 4096-wide tensor_reduce(add) Vector pass.
                exp_t = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                row_sum = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nisa.activation(
                    op=nl.exp, data=score[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    bias=neg_max[nl.arange(128)[:, None], 0], scale=1.0,
                    reduce_op=nl.add, reduce_res=row_sum[nl.arange(128)[:, None], 0],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce, dtype=np.float32)
                # recip[128,1] = 1 / row_sum
                recip = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                recip[nl.arange(128)[:, None], 0] = nisa.reciprocal(
                    data=row_sum[nl.arange(128)[:, None], 0], dtype=np.float32)
                # out_t[128,4096] = exp_t * recip  (per-row [128,1] scale over the free axis)
                out_t = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                out_t[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nisa.tensor_scalar(
                    data=exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    op0=nl.multiply, operand0=recip[nl.arange(128)[:, None], 0],
                    dtype=np.float32)

                # One 4096-wide store: out[b, 128*mt:+128, :] (partition=m, free=n).
                nl.store(
                    out[b, 128 * mt + nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    value=out_t[nl.arange(128)[:, None], nl.arange(N)[None, :]])

    return out
