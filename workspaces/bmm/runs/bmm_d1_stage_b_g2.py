import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Batched fp32 matmul out[b] = lhs[b] @ rhs[b] — PSUM-bank pipelining (G=2).

    Same math as bmm_v1 (single-pass K=64, hoisted identity transpose, 8 n-chunks of
    512 per m-tile), but the 8 n-chunks are processed in groups of G=2 with
    ISSUE-BEFORE-DRAIN scheduling: within each group, both `nc_matmul`s are issued
    into two DISTINCT pre-declared PSUM banks acc[0], acc[1] FIRST, then both PSUM->
    SBUF copies + stores drain:

        matmul -> acc[0];  matmul -> acc[1];        (issue both)
        copy acc[0] -> store;  copy acc[1] -> store; (drain both)

    This lets the compiler pipeline matmul(g,1) while copy/store(g,0) drains, and
    matmul(g+1,0) while copy/store(g,1) drains — mirroring the NKIBench baseline's
    multi-bank v10 discipline and the dense-matmul sibling's multi-bank `acc` idiom,
    which reach a much lower per-matmul stall. Pure fp32; correctness identical to v1
    (each output tile is still a single '=' write of one K=64 nc_matmul; no
    K-accumulation reorder — only the copy/store DRAIN is deferred past the next
    matmul ISSUE).

    Bank budget: G=2 output banks + 1 transpose bank = 3 <= 8 physical PSUM banks.
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
    G = 2                      # distinct PSUM banks issued before draining
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
                acc = nl.ndarray((G, par_dim(128), N_CHUNK), dtype=np.float32,
                                 buffer=nl.psum)
                # Issue all G matmuls into distinct banks first ...
                for j in nl.affine_range(G):
                    c = G * g + j
                    acc[j, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.nc_matmul(
                        lhs_t[nl.arange(K)[:, None], nl.arange(128)[None, :]],
                        rhs_sb[nl.arange(K)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])
                # ... then drain all G copies + stores.
                for j in nl.affine_range(G):
                    c = G * g + j
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
