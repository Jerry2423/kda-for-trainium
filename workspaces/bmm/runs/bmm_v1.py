import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Batched fp32 matmul  out[b] = lhs[b] @ rhs[b]  for b in 0..15.

    Shapes (natural batched layout from the NKIBench reference's reshape-only
    transform_to_nki_inputs):
        v1: (16, 4096, 64) = [batch, m, k]   (lhs)
        v2: (16, 64, 4096) = [batch, k, n]   (rhs)
    Output:
        out: (16, 4096, 4096) = [batch, m, n]   (reshapes row-major to the ref)

    The Tensor Engine's nc_matmul(stationary, moving) = stationary.T @ moving,
    and requires the contraction dim (k) on the PARTITION axis of BOTH operands.
    Both operands must live in SBUF (nc_matmul cannot read HBM), so every tile is
    nl.load-ed into SBUF first.

    K=64 <= 128, so the whole contraction fits in ONE Tensor-Engine pass: each
    output tile is produced by a single nc_matmul (no accumulation over K-tiles).

      * moving = rhs tile [k(par)=64, n(free)]. v2[b] is already [k, n], so an rhs
        tile loads directly with k on the partition axis -- no transpose.
      * stationary must be [k(par)=64, m_in(free)=128] so stationary.T @ moving =
        [m_in, k] @ [k, n] = [m_in, n]. But v1[b] is [m, k], so a loaded lhs tile
        is [m_in(par)=128, k(free)=64]; transpose it to [k(par)=64, m_in(free)=128]
        via the identity nc_matmul(is_transpose=True) idiom, then copy to SBUF.

    Loop order is batch-outer: rhs[b] ([64, 4096] = 16 KB/partition) is loaded once
    per batch and sliced per n-chunk; each of the 32 m-tiles transposes its lhs tile
    once, then streams 8 n-chunks of width 512 (one fp32 PSUM bank) into a [128, 512]
    output tile stored into the 3D output.
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

            for c in nl.affine_range(N_CHUNKS):
                # Single-pass K=64 matmul: nc_matmul(stationary=lhs_t [k,m_in],
                #   moving=rhs chunk [k,512]) = [m_in, k] @ [k, 512] = [m_in, 512].
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
                # Output tile is [m_in(par), n(free)] -> store into
                # out[b, 128*mt : 128*mt+128, 512*c : 512*c+512] (partition axis = m,
                # free axis = n), matching out[b, m, n] row-major.
                nl.store(
                    out[b, 128 * mt + nl.arange(128)[:, None],
                        N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                    value=out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

    return out
