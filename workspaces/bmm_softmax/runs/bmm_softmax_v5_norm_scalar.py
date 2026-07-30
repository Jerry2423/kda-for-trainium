import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """bmm_softmax variant: softmax normalize moved to the Scalar engine.

    Identical to the promoted kernel bmm_softmax_v4 (two-phase transpose-all, max-negate
    fold, M_SUB=16, compiler-placed score copies) EXCEPT the final softmax normalize
    out_t = exp_t * recip runs on the Scalar engine via activation(op=nl.copy,
    scale=recip[128,1]) instead of the Vector tensor_scalar(op0=multiply). This frees
    the Vector engine (which also carries the max and row-sum reductions) at the cost
    of Scalar. Bit-exact per-row scale, same fp32 math -> rel-L2 stays 2.5683307869e-6.

    See runs/bmm_softmax_v4.py for the full two-phase transpose-all schedule + fused-softmax
    docstring; only the normalize-engine line differs here.
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
    M_SUB = 16         # m-subtiles transposed/streamed per m-block
    M_BLOCKS = M_TILES // M_SUB   # 2 blocks/batch (shallower stream; WITHIN one batch)
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
                # exp_t[128,4096] = exp(score - row_max) = activation(exp, score, bias=neg_max)
                # (negate=True above already folded the neg_max step; the row sum stays a
                # SEPARATE explicit Vector tensor_reduce(add) below -- folding it into this
                # activation via reduce_op/reduce_res made the compiler recompute the whole
                # matmul+score stream twice per inference, so it is deliberately not used.)
                exp_t = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nisa.activation(
                    op=nl.exp, data=score[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    bias=neg_max[nl.arange(128)[:, None], 0], scale=1.0, dtype=np.float32)
                # row_sum[128,1] = sum_n exp_t  (explicit free-axis add)
                row_sum = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                row_sum[nl.arange(128)[:, None], 0] = nisa.tensor_reduce(
                    nl.add, data=exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    axis=[1], dtype=np.float32)
                # recip[128,1] = 1 / row_sum
                recip = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                recip[nl.arange(128)[:, None], 0] = nisa.reciprocal(
                    data=row_sum[nl.arange(128)[:, None], 0], dtype=np.float32)
                # out_t[128,4096] = exp_t * recip  (per-row [128,1] scale over the free
                # axis), placed on the SCALAR engine via activation(op=nl.copy, scale=recip)
                # instead of the Vector tensor_scalar, to move the normalize off the
                # Vector engine that also carries the two mandatory reductions. Bit-exact
                # per-row scale (same fp32 multiply), so rel-L2 stays 2.5683307869e-6.
                out_t = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                out_t[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nisa.activation(
                    op=nl.copy, data=exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    scale=recip[nl.arange(128)[:, None], 0], dtype=np.float32)

                # One 4096-wide store: out[b, 128*mt:+128, :] (partition=m, free=n).
                nl.store(
                    out[b, 128 * mt + nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    value=out_t[nl.arange(128)[:, None], nl.arange(N)[None, :]])

    return out
