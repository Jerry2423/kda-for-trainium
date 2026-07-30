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
        nc_matmul(is_transpose=True) idiom -> [k=64, m_in=128], copy to SBUF.

    Fusion: process one m-tile at a time. Its full score row is only [128, 4096]
    fp32 = 16 KB/partition, trivially SBUF-resident -- build it from 8 n-chunks of
    512, then softmax it IN PLACE over the free axis and store the normalized row.
    Scores never touch HBM. Every softmax reduce/activation/elementwise op runs on
    the SBUF score/exp tiles (free width 4096 is fine in SBUF); the [128, 512] PSUM
    banks only ever hold matmul/transpose results and are copied to SBUF immediately
    (the 512-fp32 PSUM free cap never applies to a softmax op).
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

        for mt in nl.affine_range(M_TILES):
            # Load lhs tile [m_in(par)=128, k(free)=64] from HBM into SBUF.
            lhs_sb = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
            lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]] = nl.load(
                v1[b, 128 * mt + nl.arange(128)[:, None], nl.arange(K)[None, :]],
                dtype=np.float32)

            # Transpose lhs tile -> PSUM [k(par)=64, m_in(free)=128], copy to SBUF.
            # is_moving_onezero marks the identity (all ones/zeros) as a perf hint.
            psum_t = nl.ndarray((par_dim(K), 128), dtype=np.float32, buffer=nl.psum)
            psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]],
                identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                is_transpose=True, is_moving_onezero=True)
            lhs_t = nl.ndarray((par_dim(K), 128), dtype=np.float32, buffer=nl.sbuf)
            lhs_t[nl.arange(K)[:, None], nl.arange(128)[None, :]] = nl.copy(
                psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]],
                dtype=np.float32)

            # Build the full score row [m_in(par)=128, n(free)=4096] in SBUF from
            # 8 single-pass K=64 matmuls, each landing in one [128, 512] PSUM bank.
            score = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
            for c in nl.affine_range(N_CHUNKS):
                acc = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                 buffer=nl.psum)
                acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.nc_matmul(
                    lhs_t[nl.arange(K)[:, None], nl.arange(128)[None, :]],
                    rhs_sb[nl.arange(K)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])
                score[nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]] = nl.copy(
                    acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)

            # Fused softmax over the N free axis (4096-wide), entirely in SBUF, fp32.
            # row_max[128,1] = max_n score  (reduce the free axis = reference axis 2)
            row_max = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
            row_max[nl.arange(128)[:, None], 0] = nisa.tensor_reduce(
                nl.max, data=score[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                axis=[1], dtype=np.float32)
            # neg_max[128,1] = -row_max  (the per-row bias added inside exp)
            neg_max = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
            neg_max[nl.arange(128)[:, None], 0] = nisa.tensor_scalar(
                data=row_max[nl.arange(128)[:, None], 0],
                op0=nl.multiply, operand0=np.float32(-1.0), dtype=np.float32)
            # exp_t[128,4096] = exp(score - row_max) = activation(exp, score, bias=neg_max)
            exp_t = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
            exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nisa.activation(
                op=nl.exp, data=score[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                bias=neg_max[nl.arange(128)[:, None], 0], scale=1.0, dtype=np.float32)
            # row_sum[128,1] = sum_n exp_t  (free-axis add)
            row_sum = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
            row_sum[nl.arange(128)[:, None], 0] = nisa.tensor_reduce(
                nl.add, data=exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                axis=[1], dtype=np.float32)
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

            # One 4096-wide store: out[b, 128*mt:+128, :] (partition=m, free=n), row-major.
            nl.store(
                out[b, 128 * mt + nl.arange(128)[:, None], nl.arange(N)[None, :]],
                value=out_t[nl.arange(128)[:, None], nl.arange(N)[None, :]])

    return out
