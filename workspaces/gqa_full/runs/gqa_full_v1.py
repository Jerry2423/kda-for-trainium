import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2, v3):
    """First correct fp32 NKI kernel for gqa_full: grouped-query full (non-causal)
    softmax attention. B=1, N=4096, QH=16, KH=8, n_rep=QH/KH=2, D=128.

    Per query head qh (its kv head is kh = qh // 2, since the reference does
    xk = np.repeat(k, n_rep=2, axis=head)):
        S = q_h @ k_h.T / sqrt(D)      # [N_q, N_k] scores
        A = softmax_over_Nk(S)         # row-softmax over the KEY axis
        O = A @ v_h                    # [N_q, D] context

    So this is a per-head bmm_softmax (scores + row-softmax over N) followed by a
    second matmul (context). The softmax epilogue is reused verbatim from the
    promoted sibling bmm_softmax_v4, with a 1/sqrt(D) scale added.

    Tiled layout (transform_to_nki_inputs is reshape-only; index maps verified by a
    standalone numpy reconstruction against the reference at rel-L2 2.43e-7):
      q  v1 = (32, 128, 16, 128)       = [t_q, p, qh, d],    n_q = 128*t_q + p
      k  v2 = (1, 8, 4, 128, 8, 128)   = [0, a, b, c, kh, d], n_k = 128*(4*a+b) + c
      v  v3 = (1, 32, 128, 1024)       = [0, t_v, p, kh*128+d], n_v = 128*t_v + p
      out v4 = (1, 8, 2, 32, 128, 128) = [0, kh, grp, t_q, pos, d]
                                         -> ref[0, qh=2*kh+grp, n=128*t_q+pos, d]
    The key axis n_k of the scores/attn matches the value axis n_v of v -- both are
    absolute sequence position n -- so the context matmul iterates n subtiles 0..31
    uniformly: k-columns built at offset 128*(4*a+b) and v-subtiles at t_v both index
    position n. The context loop therefore uses j = 4*a + b flattening.

    Fusion (per m_q tile, scores never touch HBM): build one query tile's full
    [128, 4096] score row in SBUF from 8 single-pass D=128 matmuls, softmax it in
    place over the 4096-wide key free axis, then consume it immediately in the
    32-step context matmul, and discard. One score row is only 16 KB/partition, so
    it stays resident with room to spare.

    nc_matmul(stationary, moving) = stationary.T @ moving, contracting on the
    PARTITION axis of both operands; both operands and the transpose identity must
    live in SBUF, so every matmul/transpose PSUM result is copied to SBUF before use.
    fp32 throughout; full-width (not chunked) softmax in the reference op order.
    """
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    N = 4096
    KH = 8             # kv heads
    N_REP = 2          # query groups per kv head (QH // KH); qh = 2*kh + grp
    D = 128            # head dim == contraction depth of matmul 1
    T = 32             # sequence tiles of 128 (N // 128); t_q, t_v, and the 32 j subtiles
    N_CHUNK = 512      # one fp32 PSUM bank in the score free dim
    N_CHUNKS = N // N_CHUNK   # 8
    scale = np.float32(1.0 / np.sqrt(D))   # 1/sqrt(128) = 0.08838835...

    out = nl.ndarray((1, 8, 2, 32, 128, 128), dtype=np.float32, buffer=nl.shared_hbm)

    # 128x128 identity in SBUF, used as the moving operand to transpose tiles on the
    # Tensor Engine (is_transpose=True). Loaded once, reused for every transpose.
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32)

    for kh in nl.affine_range(KH):
        # --- per-head shared operands, reused across the 2 query groups and all 32 t_q ---
        # k_t[d(par)=128, n_k(free)=4096]: transpose the 32 k subtiles once per head.
        # k native subtile v2[0,a,b,:,kh,:] is [c=n_k_sub(par)=128, d(free)=128];
        # transpose -> [d, n_k_sub], stored at column offset 128*(4*a+b).
        k_t = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        for a in nl.affine_range(8):
            for b in nl.affine_range(4):
                k_sub = nl.ndarray((par_dim(128), D), dtype=np.float32, buffer=nl.sbuf)
                k_sub[nl.arange(128)[:, None], nl.arange(D)[None, :]] = nl.load(
                    v2[0, a, b, nl.arange(128)[:, None], kh, nl.arange(D)[None, :]],
                    dtype=np.float32)
                k_t_ps = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
                k_t_ps[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                    k_sub[nl.arange(128)[:, None], nl.arange(D)[None, :]],
                    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    is_transpose=True, is_moving_onezero=True)
                k_t[nl.arange(128)[:, None], 128 * (4 * a + b) + nl.arange(128)[None, :]] = nl.copy(
                    k_t_ps[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32)

        # v_sb[p=n_v_sub(par)=128, n_v(free)=4096]: NO transpose -- v native
        # v3[0,t_v,:,kh*128:+128] is already [p=n_v_sub, d], the required moving layout.
        v_sb = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        for t_v in nl.affine_range(T):
            v_sb[nl.arange(128)[:, None], 128 * t_v + nl.arange(128)[None, :]] = nl.load(
                v3[0, t_v, nl.arange(128)[:, None], kh * 128 + nl.arange(128)[None, :]],
                dtype=np.float32)

        for grp in nl.affine_range(N_REP):
            qh = 2 * kh + grp
            for t_q in nl.affine_range(T):
                # --- scores: S[m_q, n_k] = sum_d q_h[m_q,d] * k_h[n_k,d] (contract d) ---
                # q native v1[t_q,:,qh,:] is [p=m_q(par)=128, d(free)=128]; transpose
                # -> q_t[d(par)=128, m_q(free)=128] so d is on the contraction partition.
                q_sb = nl.ndarray((par_dim(128), D), dtype=np.float32, buffer=nl.sbuf)
                q_sb[nl.arange(128)[:, None], nl.arange(D)[None, :]] = nl.load(
                    v1[t_q, nl.arange(128)[:, None], qh, nl.arange(D)[None, :]], dtype=np.float32)
                q_t_ps = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
                q_t_ps[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                    q_sb[nl.arange(128)[:, None], nl.arange(D)[None, :]],
                    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    is_transpose=True, is_moving_onezero=True)
                q_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
                q_t[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                    q_t_ps[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32)

                # Build the full score row [m_q(par)=128, n_k(free)=4096] in SBUF from
                # 8 single-pass D=128 matmuls, each landing in one [128,512] PSUM bank.
                score = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                for c in nl.affine_range(N_CHUNKS):
                    acc = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
                    acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.nc_matmul(
                        q_t[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        k_t[nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])
                    score[nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]] = nl.copy(
                        acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]], dtype=np.float32)

                # --- softmax over the 4096-wide key free axis, in SBUF, fp32 ---
                # Reference op order: scale the scores full-width first, then the
                # max-shifted softmax (max reduce with negate=True writes -row_max
                # directly, folding the separate *-1 step).
                score[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nisa.tensor_scalar(
                    data=score[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    op0=nl.multiply, operand0=scale, dtype=np.float32)
                neg_max = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                neg_max[nl.arange(128)[:, None], 0] = nisa.tensor_reduce(
                    nl.max, data=score[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    axis=[1], negate=True, dtype=np.float32)
                exp_t = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nisa.activation(
                    op=nl.exp, data=score[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    bias=neg_max[nl.arange(128)[:, None], 0], scale=1.0, dtype=np.float32)
                # Explicit Vector row sum (NOT fused into activation reduce_res -- the
                # fused form measured +75% wall on the sibling by recomputing the stream).
                row_sum = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                row_sum[nl.arange(128)[:, None], 0] = nisa.tensor_reduce(
                    nl.add, data=exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    axis=[1], dtype=np.float32)
                recip = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                recip[nl.arange(128)[:, None], 0] = nisa.reciprocal(
                    data=row_sum[nl.arange(128)[:, None], 0], dtype=np.float32)
                attn = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                attn[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nisa.tensor_scalar(
                    data=exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                    op0=nl.multiply, operand0=recip[nl.arange(128)[:, None], 0], dtype=np.float32)

                # --- context: O[m_q,d] = sum_{n_k} A[m_q,n_k] * v_h[n_k,d] (contract n_k) ---
                # attn is [m_q(par)=128, n_k(free)=4096]; for each of 32 n_k subtiles,
                # transpose attn[:,128*j:+128] -> A_t[n_k_sub=128, m_q=128] (contraction
                # n_k on partition), and accumulate A_t.T @ v_sb subtile j into one
                # zero-initialized [128,128] PSUM bank over j = 0..31 before copy/store.
                o_psum = nl.zeros((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
                for j in nl.affine_range(T):
                    a_t_ps = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
                    a_t_ps[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                        attn[nl.arange(128)[:, None], 128 * j + nl.arange(128)[None, :]],
                        identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        is_transpose=True, is_moving_onezero=True)
                    a_t = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
                    a_t[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                        a_t_ps[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32)
                    o_psum[nl.arange(128)[:, None], nl.arange(128)[None, :]] += nisa.nc_matmul(
                        a_t[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        v_sb[nl.arange(128)[:, None], 128 * j + nl.arange(128)[None, :]])

                o_sb = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
                o_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                    o_psum[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32)
                nl.store(
                    out[0, kh, grp, t_q, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    value=o_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]])

    return out
