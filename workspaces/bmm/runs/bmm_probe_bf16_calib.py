import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """bmm_v1 with ONLY the main-matmul operands in bf16 (fp32-rate reference).

    Identical to runs/bmm_v1.py except the two main-matmul operands (the resident
    rhs tile and the transposed lhs tile) are bf16, single product. The lhs
    transpose stays a fp32 identity-matmul and the PSUM accumulation + output stay
    fp32. Plain bf16 mantissa error (~4e-3) is orders above the 2e-5 L2 gate, so
    this does NOT meet the correctness gate; it exists only to measure the
    fp32/bf16 latency (matmul-instruction) ratio — the hard fp32 PE emulation
    pass-multiple — which decides whether a 3-product compensated bf16x2 split
    could be net faster (ratio > 3) or would regress (ratio ~2). Record-only,
    NOT a correctness candidate.
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
    N_CHUNK = 512
    N_CHUNKS = N // N_CHUNK   # 8

    out = nl.ndarray((B, M, N), dtype=np.float32, buffer=nl.shared_hbm)

    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]],
        dtype=np.float32)

    for b in nl.affine_range(B):
        # rhs[b] resident as bf16 (main-matmul moving operand).
        rhs_sb = nl.ndarray((par_dim(K), N), dtype=nl.bfloat16, buffer=nl.sbuf)
        rhs_sb[nl.arange(K)[:, None], nl.arange(N)[None, :]] = nl.load(
            v2[b, nl.arange(K)[:, None], nl.arange(N)[None, :]], dtype=nl.bfloat16)

        for mt in nl.affine_range(M_TILES):
            lhs_sb = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
            lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]] = nl.load(
                v1[b, 128 * mt + nl.arange(128)[:, None], nl.arange(K)[None, :]],
                dtype=np.float32)

            # Transpose stays a fp32 identity-matmul, then downcast to bf16.
            psum_t = nl.ndarray((par_dim(K), 128), dtype=np.float32, buffer=nl.psum)
            psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]],
                identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                is_transpose=True, is_moving_onezero=True)
            lhs_t = nl.ndarray((par_dim(K), 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            lhs_t[nl.arange(K)[:, None], nl.arange(128)[None, :]] = nl.copy(
                psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]],
                dtype=nl.bfloat16)

            for c in nl.affine_range(N_CHUNKS):
                # Single-pass bf16 main matmul (plain bf16, NOT the 3-product split):
                # nc_matmul(stationary=lhs_t [k,m_in] bf16, moving=rhs chunk [k,512]
                # bf16) -> fp32 PSUM [m_in=128, n=512].
                acc = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                 buffer=nl.psum)
                acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.nc_matmul(
                    lhs_t[nl.arange(K)[:, None], nl.arange(128)[None, :]],
                    rhs_sb[nl.arange(K)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])

                out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                    buffer=nl.sbuf)
                out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.copy(
                    acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)
                nl.store(
                    out[b, 128 * mt + nl.arange(128)[:, None],
                        N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                    value=out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

    return out
