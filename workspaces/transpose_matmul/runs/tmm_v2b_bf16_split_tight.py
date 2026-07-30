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

    Tightened limb build over tmm_v2_bf16_split: the low bf16 limb is produced DIRECTLY by
    the residual subtract writing into a bf16 destination (the tensor_tensor computes the
    exact fp32 residual internally and downcasts to bf16 at no extra cost), so there is NO
    separate fp32 residual buffer and NO extra copy. This was BUILT AS A DIAGNOSTIC to test the
    hypothesis that tmm_v2's +14.2% HBM-read growth came from the separate fp32 residual scratch
    inflating the limb-build working set and inducing compiler re-fetch.

    MEASURED RESULT (diagnostic, not promoted): this variant is BYTE-IDENTICAL to tmm_v2 on
    every metric (hbm_read 447832064, hbm_write 179306496, psum_read_sbuf_write_count 768,
    matmul_instruction_count 36864, vector/scalar instruction counts). The compiler already casts
    the tensor_tensor result to the destination dtype at no cost, so fusing the lo-limb is a NO-OP
    -- it does NOT reduce reads. This FALSIFIES the residual-buffer hypothesis: tmm_v2's read growth
    is rhs INPUT re-fetch under the M_BLK=8 resident-limb working set, and the fix is raising M_BLK
    to cut the rhs re-read multiple (tmm_v3_mblk16 at M_BLK=16 reads 229 MB, well below the floor).

    Numerically identical to tmm_v2: the low limb is bf16(lhs - lhs_hi) either way (the
    subtract accumulates in fp32, then rounds to bf16 round-to-nearest-even on the result).
    Everything else -- loop nest, constants, the 3-product accumulation, the fp32 PSUM
    accumulation, and the copy+store epilogue -- is byte-for-byte tmm_v1.

    Numeric method (unchanged from tmm_v2): each fp32 operand is split into a high and low
    bfloat16 limb (PINNED order); three bf16 products accumulate in fp32 PSUM in the FIXED
    order hi@hi, hi@lo, lo@hi, dropping the negligible lo@lo cross term:
        lhs (fp32) -> lhs_hi = bf16(lhs);  lhs_lo = bf16(lhs - lhs_hi)   # per m-block, resident
        rhs (fp32) -> rhs_hi = bf16(rhs);  rhs_lo = bf16(rhs - rhs_hi)   # per (mb,c), reused over 8 subtiles
        lhs^T @ rhs  ~=  lhs_hi^T@rhs_hi + lhs_hi^T@rhs_lo + lhs_lo^T@rhs_hi
    Offline sim: worst 3-product rel-L2 4.453e-6, 4-product 3.492e-6 (dropped lo@lo ~1e-6).
    On-device rel-L2 combines the fp32 floor (tmm_v1: 3.99e-7, pure-GEMM regime) and the bf16
    error in quadrature; with a 3.99e-7 floor the bf16 term dominates, so predicted on-device
    ~= the offline bf16 number (4.45e-6), ~4.5x under the 2e-5 gate.

    Layout (unchanged from tmm_v1): the reshape (K,.)->(128,16,.) maps flat k = k_in*16 + kt,
    so K sits on the PARTITION axis of both v1 and v2, and nisa.nc_matmul(stationary, moving)
    = stationary.T @ moving computes lhs^T @ rhs directly. Shapes:
      lhs (K,M)=(2048,4096) -> v1 (128,16,4096); rhs (K,N)=(2048,10944) -> v2 (128,16,10944);
      out (M,N)=(4096,10944) -> v3 (32,128,10944).

    Loop structure (byte-for-byte tmm_v1): M-block-outer streaming GEMM, 4 blocks of 1024
    rows (8 subtiles each). lhs limbs built once per block, resident; rhs limbs built per
    N-chunk of 456 and reused across the 8 subtiles (rhs re-read 4x, not 32x). N_CHUNK=456 =
    exact divisor of N (456*24), <=512 (one fp32 PSUM bank) => no tail masking anywhere.

    Memory: two bf16 limbs occupy the same bytes as one fp32 tile (2*2-byte bf16 == one 4-byte
    fp32). lhs_hi+lhs_lo = 2x[128,16,1024] bf16 = 64 KB/part (= v1's fp32 lhs_blk); rhs_hi+
    rhs_lo = 2x[128,16,456] bf16 ~= 28.5 KB/part (= v1's fp32 rhs_chunk). The fused-lo build
    keeps only ONE transient fp32 tile live per limb build (the loaded operand), no separate
    residual buffer -- but this did NOT change the measured HBM reads (still 447.8 MB, identical
    to tmm_v2), confirming the read growth is rhs re-fetch under the M_BLK=8 working set, not the
    residual scratch. See tmm_v3_mblk16_bf16_split for the actual fix (M_BLK=16 -> 229 MB).
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
        # Resident bf16 limbs of the lhs block for these 1024 m-rows: [k_in=128, kt=16, 1024],
        # built once per block from the fp32 loads. Two bf16 limbs = 64 KB/part = same bytes
        # as v1's fp32 lhs_blk.
        lhs_hi = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        lhs_lo = nl.ndarray((par_dim(K_IN), K_TILES, M_BLK_ROWS),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in nl.affine_range(K_TILES):
            # Transient fp32 lhs tile (the only fp32 scratch for this build; two consumers).
            lhs_f = nl.load(
                v1[ki, kt, M_BLK_ROWS * mb + im], dtype=np.float32)
            # lhs_hi = bf16(lhs)  (round-to-nearest-even cast)
            lhs_hi[ki, kt, im] = nl.copy(lhs_f[ki, im], dtype=nl.bfloat16)
            # lhs_lo = bf16(lhs - lhs_hi): the subtract computes the exact fp32 residual and
            # downcasts to the bf16 destination directly (no separate fp32 residual buffer).
            lhs_lo[ki, kt, im] = nisa.tensor_tensor(
                lhs_f[ki, im], lhs_hi[ki, kt, im], op=nl.subtract)

        for c in nl.affine_range(N_CHUNKS):
            # bf16 limbs of the rhs chunk, all 16 kt for this n-slice: [k_in=128, kt=16, 456],
            # built once per (mb,c), reused across the 8 subtiles. Two bf16 limbs ~= 28.5
            # KB/part = same bytes as v1's fp32 rhs_chunk.
            rhs_hi = nl.ndarray((par_dim(K_IN), K_TILES, N_CHUNK),
                                dtype=nl.bfloat16, buffer=nl.sbuf)
            rhs_lo = nl.ndarray((par_dim(K_IN), K_TILES, N_CHUNK),
                                dtype=nl.bfloat16, buffer=nl.sbuf)
            for kt in nl.affine_range(K_TILES):
                rhs_f = nl.load(
                    v2[ki, kt, N_CHUNK * c + inc], dtype=np.float32)
                # rhs_hi = bf16(rhs)
                rhs_hi[ki, kt, inc] = nl.copy(rhs_f[ki, inc], dtype=nl.bfloat16)
                # rhs_lo = bf16(rhs - rhs_hi): fused residual-subtract into a bf16 destination.
                rhs_lo[ki, kt, inc] = nisa.tensor_tensor(
                    rhs_f[ki, inc], rhs_hi[ki, kt, inc], op=nl.subtract)

            for s in nl.affine_range(M_BLK):
                # Zero-initialized PSUM accumulator; Tensor-Engine accumulation over the 16 kt
                # tiles reconstructs the full K=2048 contraction. Per kt, three bf16 products
                # in the FIXED order hi@hi, hi@lo, lo@hi (dropping lo@lo) land in one fp32 bank.
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
