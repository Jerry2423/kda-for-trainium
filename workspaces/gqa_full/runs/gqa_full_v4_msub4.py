import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim

# Query-tile block width processed together before draining (WITHIN the per-(kh,grp)
# reuse group). M_SUB=1 reproduces the combined-fold base gqa_full_v3c bit-for-bit;
# larger M_SUB lets the compiler overlap one block's softmax epilogue against another
# block's long context matmul stream. Swept in {1,2,4} (+8 as an SBUF-spill probe).
M_SUB = 4


@nki.jit
def kernel(v1, v2, v3):
    """Query-tile-batched two-phase gqa_full (grouped-query full softmax attention).

    Same math as gqa_full_v1 (B=1, N=4096, QH=16, KH=8, n_rep=2, D=128, fp32) and the
    same two softmax folds as gqa_full_v3c (scale-fold into activation(scale=);
    deferred normalization, context matmul on the unnormalized exp then a 128-wide
    O*recip). The only change vs v3c is that the 32 query tiles of each (kh,grp) reuse
    group are processed in blocks of M_SUB tiles:

      * build pass (per block): for each of the M_SUB query tiles, build its score row,
        run the softmax epilogue, and transpose all 32 exp subtiles into that tile's
        slot of a resident a_t_bank ([128, M_SUB*N]); keep its [128,1] reciprocal;
      * stream pass (per block): stream all M_SUB*32 context matmuls back-to-back, each
        query tile accumulating into its own zero-init o_psum, then scale by its
        reciprocal and store.

    Batching M_SUB tiles concatenates their context matmul streams into one longer
    uninterrupted PE run, so the compiler can overlap one block's softmax/transpose
    Vec/Scl work against another block's matmul stream. Stays WITHIN the per-(kh,grp)
    reuse group (never blocks across kv heads or query groups). Pure reschedule: the
    per-tile arithmetic and the j accumulation order are unchanged, so correctness
    matches v3c. fp32 throughout; scores never touch HBM.
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
    BLOCKS = T // M_SUB       # query-tile blocks per (kh,grp)
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

        # v_sb[p=n_v_sub(par)=128, n_v(free)=4096]: NO transpose -- v native is already
        # the moving layout.
        v_sb = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
        for t_v in nl.affine_range(T):
            v_sb[nl.arange(128)[:, None], 128 * t_v + nl.arange(128)[None, :]] = nl.load(
                v3[0, t_v, nl.arange(128)[:, None], kh * 128 + nl.arange(128)[None, :]],
                dtype=np.float32)

        for grp in nl.affine_range(N_REP):
            qh = 2 * kh + grp
            for blk in nl.affine_range(BLOCKS):
                # Resident per-block state: one transpose bank slot + one reciprocal per
                # query tile in the block. Only these survive from the build pass to the stream pass.
                a_t_bank = nl.ndarray((par_dim(128), M_SUB * N), dtype=np.float32, buffer=nl.sbuf)
                recip = nl.ndarray((par_dim(128), M_SUB), dtype=np.float32, buffer=nl.sbuf)

                # --- build pass: score + softmax + transpose-all, per query tile in block ---
                for mm in nl.affine_range(M_SUB):
                    t_q = blk * M_SUB + mm
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

                    # score row [m_q, n_k=4096] from 8 single-pass D=128 matmuls.
                    score = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                    for c in nl.affine_range(N_CHUNKS):
                        acc = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
                        acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.nc_matmul(
                            q_t[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                            k_t[nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])
                        score[nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]] = nl.copy(
                            acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]], dtype=np.float32)

                    # softmax over the 4096-wide key axis, fp32; 1/sqrt(D) folded into
                    # activation(scale=), row max on unscaled score, scaled [128,1] bias.
                    neg_max = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                    neg_max[nl.arange(128)[:, None], 0] = nisa.tensor_reduce(
                        nl.max, data=score[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                        axis=[1], negate=True, dtype=np.float32)
                    scaled_neg_max = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                    scaled_neg_max[nl.arange(128)[:, None], 0] = nisa.tensor_scalar(
                        data=neg_max[nl.arange(128)[:, None], 0],
                        op0=nl.multiply, operand0=scale, dtype=np.float32)
                    exp_t = nl.ndarray((par_dim(128), N), dtype=np.float32, buffer=nl.sbuf)
                    exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]] = nisa.activation(
                        op=nl.exp, data=score[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                        bias=scaled_neg_max[nl.arange(128)[:, None], 0], scale=scale, dtype=np.float32)
                    row_sum = nl.ndarray((par_dim(128), 1), dtype=np.float32, buffer=nl.sbuf)
                    row_sum[nl.arange(128)[:, None], 0] = nisa.tensor_reduce(
                        nl.add, data=exp_t[nl.arange(128)[:, None], nl.arange(N)[None, :]],
                        axis=[1], dtype=np.float32)
                    recip[nl.arange(128)[:, None], mm] = nisa.reciprocal(
                        data=row_sum[nl.arange(128)[:, None], 0], dtype=np.float32)

                    # transpose all 32 exp subtiles into this tile's a_t_bank slot.
                    for j in nl.affine_range(T):
                        a_t_ps = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
                        a_t_ps[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                            exp_t[nl.arange(128)[:, None], 128 * j + nl.arange(128)[None, :]],
                            identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                            is_transpose=True, is_moving_onezero=True)
                        a_t_bank[nl.arange(128)[:, None], mm * N + 128 * j + nl.arange(128)[None, :]] = nl.copy(
                            a_t_ps[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32)

                # --- stream pass: stream all M_SUB*32 context matmuls, then normalize+store ---
                for mm in nl.affine_range(M_SUB):
                    t_q = blk * M_SUB + mm
                    o_psum = nl.zeros((par_dim(128), 128), dtype=np.float32, buffer=nl.psum)
                    for j in nl.affine_range(T):
                        o_psum[nl.arange(128)[:, None], nl.arange(128)[None, :]] += nisa.nc_matmul(
                            a_t_bank[nl.arange(128)[:, None], mm * N + 128 * j + nl.arange(128)[None, :]],
                            v_sb[nl.arange(128)[:, None], 128 * j + nl.arange(128)[None, :]])
                    o_sb = nl.ndarray((par_dim(128), 128), dtype=np.float32, buffer=nl.sbuf)
                    o_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.tensor_scalar(
                        data=o_psum[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        op0=nl.multiply, operand0=recip[nl.arange(128)[:, None], mm], dtype=np.float32)
                    nl.store(
                        out[0, kh, grp, t_q, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                        value=o_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]])

    return out
