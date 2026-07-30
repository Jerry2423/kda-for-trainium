import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Batched fp32 matmul out[b] = lhs[b] @ rhs[b] — multi-bank allocation control.

    Identical math and issue ORDER to bmm_v1 (single-pass K=64, hoisted identity
    transpose, 8 n-chunks of 512 per m-tile). The ONLY change vs v1: the 8 n-chunks
    are processed in GROUPS of 2, and each group pre-declares a 2-bank PSUM
    accumulator `acc = (2, par_dim(128), 512)` up front — but the per-chunk work
    still runs in v1's original order within the group:

        matmul -> acc[0];  copy acc[0] -> store;   (chunk 2g)
        matmul -> acc[1];  copy acc[1] -> store;   (chunk 2g+1)

    This is a CONTROL that isolates the *allocation scope* (declaring a multi-bank
    tensor across a group) from the *scheduling* (issuing all matmuls before any
    copy/store drains). If this control moves latency vs v1, the win is allocation
    scope; if it is within noise vs v1 and only the issue-before-drain sibling moves,
    the win is scheduling. Pure fp32; correctness identical to v1 (same single '='
    write per output tile, no K-accumulation reorder).
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
    G = 2                      # PSUM banks pre-declared per n-chunk group
    N_GROUPS = N_CHUNKS // G   # 4

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

        for mt in nl.affine_range(M_TILES):
            lhs_sb = nl.ndarray((par_dim(128), K), dtype=np.float32, buffer=nl.sbuf)
            lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]] = nl.load(
                v1[b, 128 * mt + nl.arange(128)[:, None], nl.arange(K)[None, :]],
                dtype=np.float32)

            psum_t = nl.ndarray((par_dim(K), 128), dtype=np.float32, buffer=nl.psum)
            psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                lhs_sb[nl.arange(128)[:, None], nl.arange(K)[None, :]],
                identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                is_transpose=True, is_moving_onezero=True)
            lhs_t = nl.ndarray((par_dim(K), 128), dtype=np.float32, buffer=nl.sbuf)
            lhs_t[nl.arange(K)[:, None], nl.arange(128)[None, :]] = nl.copy(
                psum_t[nl.arange(K)[:, None], nl.arange(128)[None, :]],
                dtype=np.float32)

            for g in nl.affine_range(N_GROUPS):
                # Pre-declare a G-bank accumulator for this group (allocation scope),
                # but keep v1's per-chunk issue order: matmul, then its copy+store,
                # before the next chunk's matmul.
                acc = nl.ndarray((G, par_dim(128), N_CHUNK), dtype=np.float32,
                                 buffer=nl.psum)
                for j in nl.affine_range(G):
                    c = G * g + j
                    acc[j, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.nc_matmul(
                        lhs_t[nl.arange(K)[:, None], nl.arange(128)[None, :]],
                        rhs_sb[nl.arange(K)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])

                    out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                        buffer=nl.sbuf)
                    out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.copy(
                        acc[j, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                        dtype=np.float32)
                    nl.store(
                        out[b, 128 * mt + nl.arange(128)[:, None],
                            N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                        value=out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

    return out
