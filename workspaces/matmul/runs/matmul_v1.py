import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Dense fp32 GEMM  out = lhs @ rhs  (M=4096, K=5120, N=12288) in tiled layout.

    Inputs (tiled by the NKIBench reference's transform_to_nki_inputs):
        v1: (32, 128, 40, 128) = [m_tile, m_in, k_tile, k_in]  (lhs)
        v2: (40, 128, 12288)   = [k_tile, k_in, n]             (rhs)
    Output:
        v3: (32, 128, 12288)   = [m_tile, m_in, n]             (out)

    The Tensor Engine's nc_matmul(stationary, moving) = stationary.T @ moving,
    and requires the contraction dim (k_in) on the PARTITION axis of both
    operands. Both operands must live in SBUF (nc_matmul cannot read HBM), so
    every lhs/rhs tile is nl.load-ed into SBUF first. rhs tiles are
    [k_in(par), n(free)] -> used directly as the moving operand. lhs tiles are
    [m_in(par), k_in(free)] -> k_in is on the free axis, so each is transposed to
    [k_in(par), m_in(free)] via the identity nc_matmul(is_transpose=True) idiom
    before use as the stationary operand.

    Loop order is M-outer: for each of the 32 M-tiles we transpose that tile's 40
    lhs sub-tiles once, then stream all 24 N-chunks (width 512 = one fp32 PSUM
    bank), accumulating over the 40 K-tiles into a [m_in, 512] PSUM tile.
    """
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    M_TILES = 32       # 4096 / 128
    K_TILES = 40       # 5120 / 128
    N = 12288
    N_CHUNK = 512      # one fp32 PSUM bank in the free dim
    N_CHUNKS = N // N_CHUNK   # 24

    out = nl.ndarray((32, 128, 12288), dtype=np.float32, buffer=nl.shared_hbm)

    # 128x128 identity in SBUF, used as the moving operand to transpose lhs tiles
    # on the Tensor Engine (is_transpose=True). Loaded once, reused for all tiles.
    identity_const = nl.shared_constant(np.identity(128, dtype=np.float32))
    identity_local = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                buffer=nl.sbuf)
    identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
        identity_const[nl.arange(128)[:, None], nl.arange(128)[None, :]],
        dtype=np.float32)

    for mt in nl.affine_range(M_TILES):
        # Transposed lhs for this M-tile: lhs_t[kt] = [k_in(par)=128, m_in(free)=128].
        # kt is a LEADING index dim; par_dim(128) is the partition axis (mirrors
        # the baseline's v7/v9 SBUF shapes, proven to compile).
        lhs_t = nl.ndarray((K_TILES, par_dim(128), 128), dtype=np.float32,
                           buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            # Load lhs tile from HBM into SBUF: [m_in(par)=128, k_in(free)=128].
            lhs_sb = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                buffer=nl.sbuf)
            lhs_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(
                v1[mt, nl.arange(128)[:, None], kt, nl.arange(128)[None, :]],
                dtype=np.float32)
            # Transpose -> PSUM [k_in(par), m_in(free)], then copy to SBUF.
            # is_moving_onezero marks the identity (all ones/zeros) as a perf hint.
            psum_t = nl.ndarray((par_dim(128), 128), dtype=np.float32,
                                buffer=nl.psum)
            psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(
                lhs_sb[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                identity_local[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                is_transpose=True, is_moving_onezero=True)
            lhs_t[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(
                psum_t[nl.arange(128)[:, None], nl.arange(128)[None, :]],
                dtype=np.float32)

        for c in nl.affine_range(N_CHUNKS):
            # Accumulate ALL 40 K-tiles into one PSUM tile before eviction.
            acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32,
                           buffer=nl.psum)
            for kt in nl.affine_range(K_TILES):
                # Load rhs tile from HBM into SBUF: [k_in(par)=128, 512(free)].
                rhs_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                    buffer=nl.sbuf)
                rhs_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.load(
                    v2[kt, nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                    dtype=np.float32)
                # nc_matmul(stationary=lhs_t[kt] [k_in,m_in], moving=rhs_sb [k_in,512])
                #   = stationary.T @ moving = [m_in,k_in] @ [k_in,512] = [m_in,512]
                acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] += nisa.nc_matmul(
                    lhs_t[kt, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                    rhs_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

            out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                buffer=nl.sbuf)
            out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]] = nl.copy(
                acc[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]],
                dtype=np.float32)
            # Output tile is [m_in(par), n(free)] -> store into v3[mt, :, n0:n0+512]
            # (partition axis = m_in, free axis = n), matching v3[mt,mi,n]=out[mt*128+mi,n].
            nl.store(
                out[mt, nl.arange(128)[:, None], N_CHUNK * c + nl.arange(N_CHUNK)[None, :]],
                value=out_sb[nl.arange(128)[:, None], nl.arange(N_CHUNK)[None, :]])

    return out
