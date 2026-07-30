import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """bf16-split transpose-matmul, 2048-row M-block, 512-wide N chunks with a MASKED tail.

    A wider-N-chunk screen composed on tmm_v3_mblk16_bf16_split, using the classic masked-tail
    tiling instead of an exact divisor: N=10944 is tiled as 22 uniform 512-wide chunks (22*512 =
    11264 >= 10944), and the final chunk's 320 out-of-range columns (10944..11263) are masked off
    via valid_n = 512*c + inc < N on the rhs load and the output store. This is the tail-mask
    arithmetic the fp32 tmm_v1 deliberately removed by choosing the exact divisor 456; it is
    reintroduced here ONLY to measure whether fewer, wider matmuls (22 chunk-groups vs 24 at
    N_CHUNK=456 -> matmul_instruction_count 33792 vs 36864, -8.3%) buy any wall time. Since each
    real output column is computed exactly once either way, the PE COLUMN work is unchanged; the
    only possible gain is fixed per-matmul issue/fill overhead, which is measured, not assumed.

    Mask handling (mirrors the NKIBench baseline idiom mask = (N-1) - global_col >= 0):
      * rhs load is masked so out-of-range columns load as zero; the bf16 limbs are then built
        over the full 512 columns, so the zero tail columns contribute exactly zero through all
        three matmul products (no separate tail code path -- one uniform 512-wide body).
      * the output store is masked so the 320 invalid columns of the last chunk cannot write out
        of range (out is [.,.,10944]).
    Everything else is byte-for-byte tmm_v3_mblk16: compensated bf16x2 3-product split (lhs_hi/
    lhs_lo resident per 2048-row block, rhs_hi/rhs_lo per chunk; hi@hi + hi@lo + lo@hi in one fp32
    PSUM bank, dropping lo@lo; low limb produced directly by the residual subtract into a bf16
    destination). Layout: reshape (K,.)->(128,16,.) puts K on the partition axis so
    nc_matmul(stationary, moving) = stationary.T @ moving computes lhs^T @ rhs directly.
    Shapes: lhs->v1 (128,16,4096); rhs->v2 (128,16,10944); out->v3 (32,128,10944).
    """
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim

    M = 4096
    K = 2048
    N = 10944
    K_IN = 128                            # contraction lanes on the partition axis
    K_TILES = 16                          # kt tiles; K_IN * K_TILES = 2048 = K
    M_BLK = 16                            # m-subtiles per block (each subtile 128 rows)
    M_BLOCKS = 2                          # M / (M_BLK*128) = 4096 / 2048
    M_BLK_ROWS = M_BLK * 128              # 2048 rows per block
    N_CHUNK = 512                         # full moving width (one fp32 PSUM bank)
    N_CHUNKS = (N + N_CHUNK - 1) // N_CHUNK  # 22 = ceil(10944/512); last chunk has 192 real cols

    ki = nl.arange(K_IN)[:, None]         # partition index (contraction lane k_in)
    im = nl.arange(M_BLK_ROWS)[None, :]   # 2048-wide free index (m-rows of the block)
    inc = nl.arange(N_CHUNK)[None, :]     # 512-wide free index (n-slice, incl. masked tail)
    i128 = nl.arange(128)[None, :]        # 128-wide free index (subtile / output)

    out = nl.ndarray((32, 128, N), dtype=np.float32, buffer=nl.shared_hbm)

    for mb in nl.affine_range(M_BLOCKS):
        # Resident bf16 limbs of the lhs block for these 2048 m-rows: [k_in=128, kt=16, 2048],
        # built once per block. Two bf16 limbs = 128 KB/partition.
        lhs_hi = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        lhs_lo = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            lhs_f = nl.load(v1[ki, kt, M_BLK_ROWS * mb + im], dtype=np.float32)
            lhs_hi[ki, kt, im] = nl.copy(lhs_f[ki, im], dtype=nl.bfloat16)
            # lhs_lo = bf16(lhs - lhs_hi): exact fp32 residual, downcast to bf16 destination.
            lhs_lo[ki, kt, im] = nisa.tensor_tensor(
                lhs_f[ki, im], lhs_hi[ki, kt, im], op=nl.subtract)

        for c in nl.affine_range(N_CHUNKS):
            # valid_n[j] = (512*c + j) < N  == (N-1) - (512*c + j) >= 0. Masks the 320
            # out-of-range columns of the final chunk (c=21: cols 10752..11263, of which
            # 10944..11263 are invalid). Full chunks (c<21) are entirely valid.
            valid_n = (N - 1) - (N_CHUNK * c + inc) >= 0

            rhs_hi = nl.ndarray((par_dim(K_IN), K_TILES, N_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
            rhs_lo = nl.ndarray((par_dim(K_IN), K_TILES, N_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
            for kt in nl.affine_range(K_TILES):
                # Masked load: out-of-range columns come in as zero, so their bf16 limbs are zero
                # and contribute exactly zero through all three matmul products.
                rhs_f = nl.load(v2[ki, kt, N_CHUNK * c + inc], dtype=np.float32, mask=valid_n)
                rhs_hi[ki, kt, inc] = nl.copy(rhs_f[ki, inc], dtype=nl.bfloat16)
                # rhs_lo = bf16(rhs - rhs_hi): fused residual-subtract into a bf16 destination.
                rhs_lo[ki, kt, inc] = nisa.tensor_tensor(
                    rhs_f[ki, inc], rhs_hi[ki, kt, inc], op=nl.subtract)

            for s in nl.affine_range(M_BLK):
                acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.psum)
                for kt in nl.affine_range(K_TILES):
                    # lhs_hi @ rhs_hi
                    acc[nl.arange(128)[:, None], inc] += nisa.nc_matmul(
                        lhs_hi[ki, kt, 128 * s + i128], rhs_hi[ki, kt, inc])
                    # lhs_hi @ rhs_lo
                    acc[nl.arange(128)[:, None], inc] += nisa.nc_matmul(
                        lhs_hi[ki, kt, 128 * s + i128], rhs_lo[ki, kt, inc])
                    # lhs_lo @ rhs_hi   (dropping lhs_lo @ rhs_lo, the negligible cross term)
                    acc[nl.arange(128)[:, None], inc] += nisa.nc_matmul(
                        lhs_lo[ki, kt, 128 * s + i128], rhs_hi[ki, kt, inc])

                out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
                out_sb[nl.arange(128)[:, None], inc] = nl.copy(
                    acc[nl.arange(128)[:, None], inc], dtype=np.float32)
                # Masked store: the 320 invalid columns of the last chunk are not written
                # (out is [.,.,N]); out[16*mb+s, mi, 512*c+nj] -> logical row mb*2048+s*128+mi.
                nl.store(
                    out[M_BLK * mb + s, nl.arange(128)[:, None], N_CHUNK * c + inc],
                    value=out_sb[nl.arange(128)[:, None], inc], mask=valid_n)

    return out
