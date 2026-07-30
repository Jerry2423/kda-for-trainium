import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


@nki.jit
def kernel(v1, v2):
    """Compensated bf16-split transpose-matmul: out = lhs^T @ rhs (M=4096,K=2048,N=10944).

    The trn2 PE array is bf16-native; a correct fp32 GEMM runs multiple internal passes
    and is capped near ~50% MFU by that rate penalty (fp32 tmm_v1: PE=99.8%, MFU=49.4%,
    4.7274ms, emulating fp32 at 2.0 matmul-instructions/site). This kernel does the matmul
    in bf16 arithmetic while recovering ~16 effective mantissa bits with a two-limb
    compensated split, aiming to clear the relative-L2 gate (2e-5) at bf16-class matmul
    speed. Everything OUTSIDE the matmul -- the loop nest, constants, the fp32 PSUM
    accumulation, and the copy+store epilogue -- is byte-for-byte tmm_v1; the precision
    loss is confined to the matmul.

    Each fp32 operand is split into a high and low bfloat16 limb. The split order is
    PINNED and auditable (round-to-nearest-even via nl.copy(dtype=bfloat16); the residual
    is an fp32 tensor_tensor subtract, exact for these O(1) magnitudes):
        lhs (fp32) -> lhs_hi = bf16(lhs);  lhs_lo = bf16(lhs - lhs_hi)   # per m-block, resident
        rhs (fp32) -> rhs_hi = bf16(rhs);  rhs_lo = bf16(rhs - rhs_hi)   # per (mb,c), reused over 8 subtiles
    Three bf16 products are accumulated in fp32 PSUM in the FIXED order hi@hi, hi@lo,
    lo@hi, dropping the negligible lhs_lo@rhs_lo cross term (offline sim: 3-product worst
    4.453e-6 vs 4-product 3.492e-6 -- the dropped term is ~1e-6, swamped even by tmm_v1's
    tiny 3.99e-7 fp32 floor):
        lhs^T @ rhs  ~=  lhs_hi^T@rhs_hi + lhs_hi^T@rhs_lo + lhs_lo^T@rhs_hi

    This op is a PURE GEMM -- the simplest bf16x2-family member: no residual add, no
    RMSNorm, no g scale, and NO explicit transpose (the sibling matmul_add_rmsnorm had to
    transpose x per tile before splitting; here lhs arrives K-on-partition from the
    NKIBench reshape, so both limbs are built directly from the loaded fp32 tiles -- no
    PSUM round-trip, no transpose scratch). So the bf16 error enters ONLY the matmul and
    flows straight to the output; there is no norm self-cancellation and no composite
    path. tmm_v1's on-device fp32 floor is 3.99e-7 (the pure-GEMM regime, NOT the
    add_rmsnorm family's ~1.46e-5), and the device rel-L2 combines the fp32 floor and the
    bf16 error in quadrature -- with a 3.99e-7 floor the bf16 term dominates, so the
    predicted on-device rel-L2 ~= the offline bf16 number (4.45e-6), ~4.5x under the gate.

    Layout (unchanged from tmm_v1): the reshape (K,.)->(128,16,.) maps flat
    k = k_in*16 + kt, so K sits on the PARTITION axis of both v1 and v2, and
    nisa.nc_matmul(stationary, moving) = stationary.T @ moving computes lhs^T @ rhs
    directly -- the transposed-lhs operand is a free byproduct of the input layout.
    Shapes (as scored by NKIBench):
      lhs (K,M)=(2048,4096) K-major -> reshaped to v1 (128,16,4096)
      rhs (K,N)=(2048,10944)        -> reshaped to v2 (128,16,10944)
      out (M,N)=(4096,10944)        -> v3 (32,128,10944)

    Loop structure (byte-for-byte tmm_v1): M-block-outer streaming GEMM. M (4096 = 32
    tiles of 128) is split into 4 blocks of 1024 rows (8 subtiles each). Per block the
    lhs limbs are built once and stay resident; rhs limbs are built per N-chunk of 456
    and reused across the 8 subtiles, so rhs is re-read 4x total (once per block), not
    32x. N_CHUNK=456 is an exact divisor of N=10944 (456*24) and <=512 (one fp32 PSUM
    bank), so every tile is full-size and there is NO tail-masking arithmetic anywhere.

    Memory: two bf16 limbs occupy exactly the same bytes as one fp32 tile (2*2-byte bf16
    == one 4-byte fp32), so lhs_hi+lhs_lo = 2x[128,16,1024] bf16 = 64 KB/partition (SAME
    as v1's fp32 lhs_blk, dropped) and rhs_hi+rhs_lo = 2x[128,16,456] bf16 ~= 28.5
    KB/partition (SAME as v1's fp32 rhs_chunk). Transient fp32 build scratch is per-tile
    (freed across iterations). HBM is unchanged vs v1 (~392 MB read): the limbs are built
    on-chip from the same fp32 HBM loads, no extra reads. matmul_instruction_count rises
    24576 -> ~36864 (2.0 -> 3.0 instr/site); the win, if any, is that each new instr is a
    bf16 pass, not the ~1.8x fp32 emulation rate -- MEASURED against a same-session v1
    anchor, not assumed (the same 2.0/site count lost on swiglu but won on
    matmul_add_rmsnorm).

    fp32 accumulator, copy, and store are unchanged from tmm_v1; only the operand dtype
    path (bf16 limbs + 3 products) differs.
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
    K_IN = 128                      # contraction lanes on the partition axis
    K_TILES = 16                    # kt tiles; K_IN * K_TILES = 2048 = K
    M_BLK = 8                       # m-subtiles per block (each subtile 128 rows)
    M_BLOCKS = 4                    # M / (M_BLK*128) = 4096 / 1024
    M_BLK_ROWS = M_BLK * 128        # 1024 rows per block
    N_CHUNK = 456                   # exact divisor of N, <= 512 fp32 PSUM width
    N_CHUNKS = N // N_CHUNK         # 24; 456 * 24 = 10944 exactly

    ki = nl.arange(K_IN)[:, None]        # partition index (contraction lane k_in)
    im = nl.arange(M_BLK_ROWS)[None, :]  # 1024-wide free index (m-rows of the block)
    inc = nl.arange(N_CHUNK)[None, :]    # 456-wide free index (n-slice)
    i128 = nl.arange(128)[None, :]       # 128-wide free index (subtile / output)

    out = nl.ndarray((32, 128, N), dtype=np.float32, buffer=nl.shared_hbm)

    for mb in nl.affine_range(M_BLOCKS):
        # Resident bf16 limbs of the lhs block for these 1024 m-rows: [k_in=128, kt=16,
        # 1024], built once per block from the fp32 loads and read-only afterward. Two
        # bf16 limbs = 2x32 KB = 64 KB/partition = the same bytes as v1's fp32 lhs_blk.
        lhs_hi = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        lhs_lo = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            # Transient fp32 lhs tile (reused across kt); the limbs are built from it.
            lhs_f = nl.load(
                v1[ki, kt, M_BLK_ROWS * mb + im], dtype=np.float32)
            # lhs_hi = bf16(lhs)  (round-to-nearest-even cast)
            lhs_hi[ki, kt, im] = nl.copy(lhs_f[ki, im], dtype=nl.bfloat16)
            # residual = lhs - lhs_hi (fp32; exact for O(1) magnitudes), then lhs_lo = bf16(residual)
            lhs_res = nl.ndarray((par_dim(K_IN), M_BLK_ROWS), dtype=np.float32, buffer=nl.sbuf)
            lhs_res[ki, im] = nisa.tensor_tensor(
                lhs_f[ki, im], lhs_hi[ki, kt, im], op=nl.subtract)
            lhs_lo[ki, kt, im] = nl.copy(lhs_res[ki, im], dtype=nl.bfloat16)

        for c in nl.affine_range(N_CHUNKS):
            # bf16 limbs of the rhs chunk, all 16 kt for this n-slice: [k_in=128, kt=16,
            # 456]. Built once per (mb,c), reused across the 8 subtiles below. Two bf16
            # limbs ~= 2x14.25 KB = 28.5 KB/partition = the same bytes as v1's fp32 rhs_chunk.
            rhs_hi = nl.ndarray((par_dim(K_IN), K_TILES, N_CHUNK),
                                dtype=nl.bfloat16, buffer=nl.sbuf)
            rhs_lo = nl.ndarray((par_dim(K_IN), K_TILES, N_CHUNK),
                                dtype=nl.bfloat16, buffer=nl.sbuf)
            for kt in nl.affine_range(K_TILES):
                rhs_f = nl.load(
                    v2[ki, kt, N_CHUNK * c + inc], dtype=np.float32)
                # rhs_hi = bf16(rhs)
                rhs_hi[ki, kt, inc] = nl.copy(rhs_f[ki, inc], dtype=nl.bfloat16)
                # residual = rhs - rhs_hi (fp32), then rhs_lo = bf16(residual)
                rhs_res = nl.ndarray((par_dim(K_IN), N_CHUNK), dtype=np.float32, buffer=nl.sbuf)
                rhs_res[ki, inc] = nisa.tensor_tensor(
                    rhs_f[ki, inc], rhs_hi[ki, kt, inc], op=nl.subtract)
                rhs_lo[ki, kt, inc] = nl.copy(rhs_res[ki, inc], dtype=nl.bfloat16)

            for s in nl.affine_range(M_BLK):
                # Zero-initialized PSUM accumulator; Tensor-Engine accumulation over the
                # 16 kt tiles reconstructs the full K=2048 contraction. Per kt, three bf16
                # products in the FIXED order hi@hi, hi@lo, lo@hi (dropping lo@lo) land in
                # the same fp32 PSUM bank.
                acc = nl.zeros((par_dim(128), N_CHUNK), dtype=np.float32,
                               buffer=nl.psum)
                for kt in nl.affine_range(K_TILES):
                    # lhs_hi @ rhs_hi
                    acc[nl.arange(128)[:, None], inc] += nisa.nc_matmul(
                        lhs_hi[ki, kt, 128 * s + i128],   # stationary [k_in,128]
                        rhs_hi[ki, kt, inc])               # moving     [k_in,456]
                    # lhs_hi @ rhs_lo
                    acc[nl.arange(128)[:, None], inc] += nisa.nc_matmul(
                        lhs_hi[ki, kt, 128 * s + i128],
                        rhs_lo[ki, kt, inc])
                    # lhs_lo @ rhs_hi   (dropping lhs_lo @ rhs_lo, the negligible cross term)
                    acc[nl.arange(128)[:, None], inc] += nisa.nc_matmul(
                        lhs_lo[ki, kt, 128 * s + i128],
                        rhs_hi[ki, kt, inc])

                out_sb = nl.ndarray((par_dim(128), N_CHUNK), dtype=np.float32,
                                    buffer=nl.sbuf)
                out_sb[nl.arange(128)[:, None], inc] = nl.copy(
                    acc[nl.arange(128)[:, None], inc],
                    dtype=np.float32)
                # out[8*mb+s, mi, 456*c+nj] -> logical row m = mb*1024 + s*128 + mi.
                nl.store(
                    out[8 * mb + s, nl.arange(128)[:, None],
                        N_CHUNK * c + inc],
                    value=out_sb[nl.arange(128)[:, None], inc])

    return out
