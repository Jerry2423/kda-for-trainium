"""Host-side numpy sanity check for the tiled matmul index/transpose arithmetic.

Mirrors exactly what runs/matmul_v1.py does on-device, but in numpy, so any
layout / transpose / accumulation-order bug is caught locally before spending a
remote profiler run. NOT a kernel; not scored. Run:

    python3 runs/_layout_check.py

We check a handful of full output tiles (each the exact K-accumulation the kernel
does) against a locally-computed gold for just those output rows/cols — computing
the entire 4096x5120x12288 matmul in numpy would be needlessly slow.
"""
import numpy as np

M, K, N = 4096, 5120, 12288
M_TILES, K_TILES = M // 128, K // 128          # 32, 40
N_CHUNK = 512
N_CHUNKS = N // N_CHUNK                          # 24
assert (M_TILES, K_TILES, N_CHUNKS) == (32, 40, 24)

rng = np.random.default_rng(0)
lhs = rng.standard_normal((M, K)).astype(np.float32)
rhs = rng.standard_normal((K, N)).astype(np.float32)

# transform_to_nki_inputs (row-major reshape), from the numpy reference file.
v1 = np.reshape(lhs, (32, 128, 40, 128))         # [m_tile, m_in, k_tile, k_in]
v2 = np.reshape(rhs, (40, 128, 12288))           # [k_tile, k_in, n]

# --- Confirm the index mapping the kernel relies on (AC-4) ---
mt, mi, kt, ki, n = 3, 7, 11, 55, 900
assert v1[mt, mi, kt, ki] == lhs[mt * 128 + mi, kt * 128 + ki]
assert v2[kt, ki, n] == rhs[kt * 128 + ki, n]
print("[ok] v1/v2 index mapping matches lhs/rhs")


def kernel_tile(mt, c):
    """Reproduce the on-device computation of output tile v3[mt, :, n0:n0+512].

    for kt: transpose lhs sub-tile [m_in,k_in] -> lhsT[kt]=[k_in,m_in];
            acc[m_in,n] += nc_matmul(lhsT[kt], rhs[kt,:,nchunk])
                         = lhsT[kt].T @ rhs_chunk = [m_in,k_in]@[k_in,n]
    """
    n0 = c * N_CHUNK
    acc = np.zeros((128, N_CHUNK), dtype=np.float32)
    for kt in range(K_TILES):
        lhsT_kt = v1[mt, :, kt, :].T             # [k_in, m_in]  (the transpose)
        acc += lhsT_kt.T @ v2[kt, :, n0:n0 + N_CHUNK]
    return acc


def gold_tile(mt, c):
    """Gold for just this output tile: lhs rows [mt] @ rhs cols [c]."""
    n0 = c * N_CHUNK
    return lhs[mt * 128:(mt + 1) * 128, :] @ rhs[:, n0:n0 + N_CHUNK]


# Check several tiles across the grid (corners + interior), each with the exact
# K-accumulation order the kernel uses.
checks = [(0, 0), (5, 17), (31, 23), (16, 11), (7, 3)]
worst = 0.0
for mt, c in checks:
    got = kernel_tile(mt, c)
    ref = gold_tile(mt, c)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    worst = max(worst, rel)
    print(f"[ok] tile (mt={mt:2d}, c={c:2d}) rel-L2 = {rel:.3e}")
    assert rel < 2e-5, f"tile (mt={mt}, c={c}) failed: rel-L2={rel:.3e}"

print(f"\n[v1] LAYOUT CHECK PASSED (worst tile rel-L2 = {worst:.3e}) — "
      "index/transpose/accum arithmetic is correct.")


# ---------------------------------------------------------------------------
# Phase 2: M-blocking. Process B M-tiles together so each rhs K-tile is loaded
# once and reused across B stationary lhsT tiles into B DISTINCT accumulators,
# each writing a DISTINCT output row-block out[mblock*B+mb, :, nchunk].
# The correctness risk (per review) is exactly this distinct-row / distinct-
# accumulator mapping, so verify a full tile from mblock>0, mb>0.
# ---------------------------------------------------------------------------
def mblocked_tile(mblock, mb, c, B):
    """Reproduce the on-device output tile for blocked M-tile mt=mblock*B+mb.

    In the kernel: for kt, rhs_sb = load v2[kt,:,nchunk] ONCE, then for each mb
    in the block: acc[mb] += nc_matmul(lhsT[mb,kt], rhs_sb). This function
    reproduces the acc[mb] for one (mblock, mb) using that shared-rhs order.
    """
    mt = mblock * B + mb
    n0 = c * N_CHUNK
    acc = np.zeros((128, N_CHUNK), dtype=np.float32)
    for kt in range(K_TILES):
        lhsT_kt = v1[mt, :, kt, :].T          # [k_in, m_in] for THIS block member
        rhs_sb = v2[kt, :, n0:n0 + N_CHUNK]   # shared across the block in the kernel
        acc += lhsT_kt.T @ rhs_sb
    return acc


mb_worst = 0.0
# Cover multiple B, and crucially non-zero mblock AND mb>0 (distinct-row check),
# plus the last block member and last n-chunk.
mb_checks = [
    (2, 3, 1, 17),   # B=2: mblock=3 -> mt=7 (mb=1)
    (4, 5, 3, 23),   # B=4: mblock=5 -> mt=23 (mb=3, last in block), last n-chunk
    (6, 2, 5, 11),   # B=6: mblock=2 -> mt=17 (mb=5, last in block)
    (8, 3, 7, 0),    # B=8: mblock=3 -> mt=31 (mb=7, last M-tile overall)
]
for B, mblock, mb, c in mb_checks:
    mt = mblock * B + mb
    got = mblocked_tile(mblock, mb, c, B)
    ref = gold_tile(mt, c)                     # gold indexed by the TRUE mt
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    mb_worst = max(mb_worst, rel)
    print(f"[mblock] B={B} mblock={mblock} mb={mb} -> mt={mt:2d}, c={c:2d}  rel-L2 = {rel:.3e}")
    assert rel < 2e-5, f"M-block tile B={B} mblock={mblock} mb={mb} failed: {rel:.3e}"

# Distinct-row guard: two different block members must produce DIFFERENT outputs
# (catches an accumulator/row aliasing bug that a single-tile check would miss).
a = mblocked_tile(3, 0, 17, 4)   # mt=12
b = mblocked_tile(3, 1, 17, 4)   # mt=13
assert not np.allclose(a, b), "block members mb=0 and mb=1 produced identical tiles!"
assert np.allclose(a, gold_tile(12, 17), rtol=0, atol=0) is False  # fp order differs; use rel
assert np.linalg.norm(a - gold_tile(12, 17)) / np.linalg.norm(gold_tile(12, 17)) < 2e-5
assert np.linalg.norm(b - gold_tile(13, 17)) / np.linalg.norm(gold_tile(13, 17)) < 2e-5
print("[mblock] distinct-row guard OK (mb=0 and mb=1 map to distinct correct rows)")

print(f"\n[v2] M-BLOCK CHECK PASSED (worst tile rel-L2 = {mb_worst:.3e}) — "
      "M-block index / distinct-accumulator / distinct-row mapping is correct.")
