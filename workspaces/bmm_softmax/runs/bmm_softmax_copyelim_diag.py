import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """DIAGNOSTIC ONLY -- NOT a production candidate, NEVER promotable.

    Copy-elimination probe for the fused bmm_softmax back-pressure hypothesis. This
    clones the promoted kernel bmm_softmax_v4 but REMOVES the 8 PSUM->SBUF score-drain
    copies: the score row is left resident in a single [128, 4096] PSUM tile (all 8
    banks), and the softmax max-reduce + exp read that PSUM tile DIRECTLY. This holds
    all 8 PSUM banks alive through the 4096-wide max and exp, so the next m-subtile's
    8 matmuls cannot reuse the banks until exp drains them.

    If TRUE PE-active INFLATES vs v4, the paper rejection of copy-elimination is
    validated and the "the copies are the fast bank drain, not waste" back-pressure
    story stands. If PE-active does NOT inflate, the drain is not gating the matmul
    and the back-pressure premise is downgraded (feeds the terminal-vs-rebalance decision).

    Same fp32 math as v4 (same set of exp terms summed in the same order), so the
    on-device rel-L2 must stay bit-identical (2.5683307869e-6). This file is named
    *_copyelim_diag so it can never be mistaken for a production kernel.
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
    N_CHUNKS = N // N_CHUNK   # 8 banks

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

                # Score row kept RESIDENT in one [128, 4096] PSUM tile (8 banks); each
                # single-pass K=64 matmul writes its own 512-wide bank slice. No drain
                # copy to SBUF -- the softmax reduce/exp read PSUM directly below.
                score_psum = nl.ndarray((par_dim(128), N), dtype=np.float32,
                                        buffer=nl.psum)
                for c in nl.affine_range(N_CHUNKS):
                    score_psum[nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]] = nisa.nc_matmul(
                        lhs_t_pack[nl.arange(K)[:, None], 128 * s + nl.arange(128)[None, :]],
                        rhs_sb[nl.arange(K)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])

                # max-shifted softmax reading the PSUM score tile DIRECTLY (the probe).
                neg_max = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                neg_max[nl.arange(128)[:, None], 0] = nisa.tensor_reduce(
                    nl.max, data=score_psum[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    axis=[1], negate=True, dtype=np.float32)
                exp_t = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nisa.activation(
                    op=nl.exp, data=score_psum[nl.arange(128)[:, None], nl.arange(N)[None, :]],
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
