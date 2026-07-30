import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """bmm_softmax with the 8 PSUM->SBUF score-drain copies steered to the Vector engine.

    Identical to the promoted kernel bmm_softmax_v4 (two-phase transpose-all schedule,
    max-negate fold, M_SUB=16, full-row fused softmax) EXCEPT the 8 per-subtile score
    copies that drain each [128,512] matmul PSUM bank into the resident score[128,4096]
    SBUF tile are placed explicitly on the Vector engine via nisa.tensor_copy(engine=...)
    instead of leaving the engine to the compiler (nl.copy).

    Rationale: the copies free the 8 PSUM banks for the next subtile's matmuls; if they
    serialize on a loaded engine the drain gates the matmul stream. Routing all copies to
    Vector is the mirror of the all-Scalar variant -- it loads the same Vector engine that
    also carries the two mandatory reductions (max, row-sum), the expected-worst placement.
    Pure engine reassignment: bit-exact fp32 copy, same math as v4, so rel-L2 must stay
    2.5683307869e-6 on every seed.
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
    M_TILES = 32
    M_SUB = 16
    M_BLOCKS = M_TILES // M_SUB
    N_CHUNK = 512
    N_CHUNKS = N // N_CHUNK

    out = nl.ndarray((B, M, N), dtype=np.float32, buffer=nl.shared_hbm)

    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]],
        dtype=np.float32)

    for b in nl.affine_range(B):
        rhs_sb = nl.ndarray((par_dim(K), N), dtype=np.float32, buffer=nl.sbuf)
        rhs_sb[nl.arange(K)[:, None], nl.arange(N)[None, :]] = nl.load(
            v2[b, nl.arange(K)[:, None], nl.arange(N)[None, :]], dtype=np.float32)

        for mblk in nl.affine_range(M_BLOCKS):
            lhs_t_pack = nl.ndarray((par_dim(K), M_SUB * 128), dtype=np.float32,
                                    buffer=nl.sbuf)
            for s in nl.affine_range(M_SUB):
                mt = M_SUB * mblk + s
                lhs_sb = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
                lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]] = nl.load(
                    v1[b, 128 * mt + nl.arange(128)[:, None], nl.arange(K)[None, :]],
                    dtype=np.float32)
                psum_t = nl.ndarray((par_dim(K), 128), dtype=np.float32, buffer=nl.psum)
                psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                    lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]],
                    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    is_transpose=True, is_moving_onezero=True)
                lhs_t_pack[nl.arange(K)[:, None], 128 * s + nl.arange(128)[None, :]] = nl.copy(
                    psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]],
                    dtype=np.float32)

            for s in nl.affine_range(M_SUB):
                mt = M_SUB * mblk + s

                score = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                for c in nl.affine_range(N_CHUNKS):
                    acc = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                     buffer=nl.psum)
                    acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.nc_matmul(
                        lhs_t_pack[nl.arange(K)[:, None], 128 * s + nl.arange(128)[None, :]],
                        rhs_sb[nl.arange(K)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])
                    # Drain this PSUM bank to SBUF on the Vector engine (explicit steer).
                    score[nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]] = nisa.tensor_copy(
                        acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                        engine=nki.isa.engine.vector, dtype=np.float32)

                neg_max = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                neg_max[nl.arange(128)[:, None], 0] = nisa.tensor_reduce(
                    nl.max, data=score[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    axis=[1], negate=True, dtype=np.float32)
                exp_t = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nisa.activation(
                    op=nl.exp, data=score[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    bias=neg_max[nl.arange(128)[:, None], 0], scale=1.0, dtype=np.float32)
                row_sum = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                row_sum[nl.arange(128)[:, None], 0] = nisa.tensor_reduce(
                    nl.add, data=exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    axis=[1], dtype=np.float32)
                recip = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                recip[nl.arange(128)[:, None], 0] = nisa.reciprocal(
                    data=row_sum[nl.arange(128)[:, None], 0], dtype=np.float32)
                out_t = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                out_t[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nisa.tensor_scalar(
                    data=exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    op0=nl.multiply, operand0=recip[nl.arange(128)[:, None], 0],
                    dtype=np.float32)

                nl.store(
                    out[b, 128 * mt + nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    value=out_t[nl.arange(128)[:, None], nl.arange(N)[None, :]])

    return out
