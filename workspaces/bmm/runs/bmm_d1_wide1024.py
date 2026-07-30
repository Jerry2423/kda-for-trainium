import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Batched fp32 matmul out[b] = lhs[b] @ rhs[b] — wide 1024 store granularity.

    Same math as bmm_v1 (single-pass K=64, hoisted identity transpose), but the
    output DRAIN is coalesced to 1024-wide stores, mirroring the NKIBench baseline's
    store granularity (its v11 is [128,1024], stored once per pair of 512 n-chunks).
    A fp32 PSUM bank holds at most 512 free elements, so each 1024-wide output tile is
    still produced by TWO [128,512] `nc_matmul`s into two distinct PSUM banks, then
    both are copied into ONE [128,1024] SBUF tile and stored with a SINGLE 1024-wide
    `nl.store`. This halves the store-instruction count (4096 -> 2048) relative to
    v1's 512-wide stores.

    Multi-bank PSUM pipelining alone (issue-before-drain into distinct banks) was
    measured to be a no-op here (the affine_range compiler already pipelines v1's
    single rotating bank), so this variant isolates the OUTPUT-drain granularity as
    the remaining pure-fp32 structural lever that the baseline uses and v1 does not.

    Pure fp32; correctness identical to v1 (each 512-wide half is a single '=' write
    of one K=64 nc_matmul; no K-accumulation reorder; the two halves are laid out
    contiguously in n, so the wide store writes the same values to the same output
    positions).

    Bank budget: 2 output banks + 1 transpose bank = 3 <= 8 physical PSUM banks.
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
    WIDE = 1024                    # coalesced store width
    N_WIDE = N // WIDE             # 4 wide groups per m-tile
    G = WIDE // N_CHUNK            # 2 PSUM banks feed one wide store

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

            for w in nl.affine_range(N_WIDE):
                acc = nl.ndarray((G, par_dim(128), N_CHUNK), dtype=np.float32,
                                 buffer=nl.psum)
                for j in nl.affine_range(G):
                    c = G * w + j
                    acc[j, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nisa.nc_matmul(
                        lhs_t[nl.arange(K)[:, None], nl.arange(128)[None, :]],
                        rhs_sb[nl.arange(K)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]])

                # Coalesce the G=2 PSUM halves into ONE wide SBUF tile, store once.
                out_sb = nl.ndarray((par_dim(128), WIDE), dtype=np.float32,
                                    buffer=nl.sbuf)
                for j in nl.affine_range(G):
                    out_sb[nl.arange(128)[:, None], N_CHUNK * j + nl.arange(N_CHUNK)[None, :]] = nl.copy(
                        acc[j, nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                        dtype=np.float32)
                nl.store(
                    out[b, 128 * mt + nl.arange(128)[:, None],
                        WIDE * w + nl.arange(WIDE)[None, :]],
                    value=out_sb[nl.arange(128)[:, None], nl.arange(WIDE)[None, :]])

    return out
