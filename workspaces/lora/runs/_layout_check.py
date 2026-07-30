"""Host-side numpy sanity check for the tiled lora index/transpose/accumulation.

Mirrors exactly what runs/lora_v1.py does on-device, but in numpy, so any layout /
transpose / accumulation-order bug is caught locally before spending a remote
profiler run. NOT a kernel; not scored. Run:

    python3 runs/_layout_check.py

The operator is  out = x @ w + (x @ a) @ b  (fp32):
    x (M=4096, K=5120), w (K=5120, N=12288), a (K=5120, R=128), b (R=128, N=12288).

Tiled layout from the reference transform_to_nki_inputs (row-major reshape):
    v1 (x):   (8, 4, 128, 40, 128) = [m_hi, m_lo, m_in, k_tile, k_in]
              row m = (m_hi*4 + m_lo)*128 + m_in, col k = k_tile*128 + k_in.
    v2 (w):   (40, 128, 12288)     = [k_tile, k_in, n]        (contraction on partition)
    v3 (a):   (40, 128, 128)       = [k_tile, k_in, r]        (contraction on partition)
    v4 (b):   (128, 12288)         = [r, n]                   (contraction on partition)
    v5 (out): (8, 4, 128, 96, 128) = [m_hi, m_lo, m_in, n_tile, n_in]
              row m = (m_hi*4 + m_lo)*128 + m_in, col n = n_tile*128 + n_in.

The M-tile index in v1/v5 is a 2-LEVEL (m_hi, m_lo) pair (8x4=32 M-tiles), unlike the
sibling matmul whose x/out were a flat (32,128,...). A swapped (m_hi, m_lo) silently
scrambles rows, so we check tiles with nonzero m_hi AND nonzero m_lo explicitly.

We check a handful of full output tiles (each the exact transpose + K-accumulation +
fused low-rank the kernel does) against a locally-computed gold for just those output
rows/cols — computing the entire 4096x5120x12288 matmul in numpy would be needlessly slow.

Two store mappings are covered. The width-128 kernel (lora_v1) computes one output n_tile
of width 128 per chunk and stores it directly to v5[m_hi,m_lo,:,nt,:]. The widened
kernels (lora_v2_mblk4 / lora_v3_bf16_split) widen the base GEMM to N_CHUNK=512, so each
512-wide computed tile for chunk c (columns [512c, 512c+512)) is stored as 4 sub-tile
writes out_sb[:, 128j:128j+128] -> v5[m_hi,m_lo,:,4c+j,:] for j in 0..3 (N_CHUNKS=24, so
4c+j spans 0..95). That 4-sub-tile decomposition is the one indexing difference vs the
sibling matmul (whose out N axis was flat), so it is verified numerically below, along
with a wrong-mapping negative control.
"""
import numpy as np

M, N, K, R = 4096, 12288, 5120, 128
M_HI, M_LO = 8, 4
M_TILES = M_HI * M_LO                            # 32
K_TILES = K // 128                               # 40
N_CHUNK = 128                                    # one output n_tile per chunk
N_TILES = N // N_CHUNK                            # 96
assert (M_TILES, K_TILES, N_TILES) == (32, 40, 96)

rng = np.random.default_rng(0)
x = rng.standard_normal((M, K)).astype(np.float32)
w = rng.standard_normal((K, N)).astype(np.float32)
a = rng.standard_normal((K, R)).astype(np.float32)
b = rng.standard_normal((R, N)).astype(np.float32)

# transform_to_nki_inputs (row-major reshape), from the numpy reference file.
v1 = np.reshape(x, (8, 4, 128, 40, 128))         # [m_hi, m_lo, m_in, k_tile, k_in]
v2 = np.reshape(w, (40, 128, 12288))             # [k_tile, k_in, n]
v3 = np.reshape(a, (40, 128, 128))               # [k_tile, k_in, r]
v4 = b                                           # [r, n]  (already (128, 12288))


# --- Index mapping checks: confirm the mappings the kernel relies on ---
# Include nonzero m_hi AND nonzero m_lo (the 2-level M-index), nonzero kt, nonzero r.
mh, ml, mi, kt, ki, r, n = 1, 3, 7, 11, 55, 90, 900
assert v1[mh, ml, mi, kt, ki] == x[(mh * 4 + ml) * 128 + mi, kt * 128 + ki]
assert v2[kt, ki, n] == w[kt * 128 + ki, n]
assert v3[kt, ki, r] == a[kt * 128 + ki, r]
assert v4[r, n] == b[r, n]
print("[ok] v1(x)/v2(w)/v3(a)/v4(b) index mappings match (incl. nonzero m_hi & m_lo)")


def down_proj_tT(mt):
    """Reproduce the on-device down-projection tT = (x_tile @ a)^T = [R, m_in].

    In the kernel: lhs_t[kt] = v1[m_hi,m_lo,:,kt,:].T = [k_in, m_in]; then
        tT += nc_matmul(stationary=a_local[kt] [k_in,R], moving=lhs_t[kt] [k_in,m_in])
            = a_local[kt].T @ lhs_t[kt] = [R,k_in] @ [k_in,m_in] = [R, m_in],
    accumulated over the 40 K-tiles. So tT = sum_kt (a[kt].T @ x[kt].T) = (x @ a).T.
    """
    m_hi, m_lo = mt // 4, mt % 4
    tT = np.zeros((R, 128), dtype=np.float32)
    for kt in range(K_TILES):
        lhs_t_kt = v1[m_hi, m_lo, :, kt, :].T    # [k_in, m_in]  (the identity transpose)
        a_kt = v3[kt, :, :]                      # [k_in, R]
        tT += a_kt.T @ lhs_t_kt                  # [R, k_in] @ [k_in, m_in] = [R, m_in]
    return tT


def base_tile(mt, nt):
    """Reproduce the on-device BASE GEMM contribution to v5[m_hi,m_lo,:,nt,:] = [m_in,n_in].

    acc[m_in,n] += nc_matmul(stationary=lhs_t[kt] [k_in,m_in], moving=w[kt,:,nt])
                 = lhs_t[kt].T @ w_chunk = [m_in,k_in] @ [k_in,n] = [m_in,n]
    """
    m_hi, m_lo = mt // 4, mt % 4
    n0 = nt * N_CHUNK
    acc = np.zeros((128, N_CHUNK), dtype=np.float32)
    for kt in range(K_TILES):
        lhs_t_kt = v1[m_hi, m_lo, :, kt, :].T    # [k_in, m_in]
        acc += lhs_t_kt.T @ v2[kt, :, n0:n0 + N_CHUNK]
    return acc


def kernel_tile(mt, nt):
    """Reproduce the on-device fused output tile: base x@w + fused low-rank (x@a)@b.

    fused: acc += nc_matmul(stationary=tT [r,m_in], moving=b[:,nt]) = tT.T @ b_chunk
                = (x@a) @ b_chunk = [m_in, n]  -> same layout, same PSUM bank.
    """
    n0 = nt * N_CHUNK
    tT = down_proj_tT(mt)                         # [R, m_in]
    return base_tile(mt, nt) + tT.T @ v4[:, n0:n0 + N_CHUNK]


def gold_tile(mt, nt):
    """Gold for just this output tile: full lora out rows [mt] @ cols [nt]."""
    n0 = nt * N_CHUNK
    rows = slice(mt * 128, (mt + 1) * 128)
    cols = slice(n0, n0 + N_CHUNK)
    xr = x[rows, :]                               # [128, K]
    return xr @ w[:, cols] + (xr @ a) @ b[:, cols]


# --- Down-projection identity: tT == (x@a).T for a sampled M-tile ---
mt_s = 7                                          # m_hi=1, m_lo=3 (both nonzero)
tT_got = down_proj_tT(mt_s)
xa_ref = (x[mt_s * 128:(mt_s + 1) * 128, :] @ a).T   # (x_tile @ a).T = [R, m_in]
rel_tT = np.linalg.norm(tT_got - xa_ref) / np.linalg.norm(xa_ref)
print(f"[ok] tT down-projection identity: mt={mt_s} (m_hi=1,m_lo=3) rel-L2 = {rel_tT:.3e}")
assert rel_tT < 2e-5, f"tT != (x@a).T for mt={mt_s}: rel-L2={rel_tT:.3e}"


# --- Fused accumulation checks: acc == x@w + (x@a)@b on sampled tiles ---
# Cover corners + interior; crucially at least one with nonzero m_hi, nonzero m_lo,
# nonzero kt (all K-tiles participate), nonzero r (all R participate), and a
# last/near-last n_tile (tail nt) to catch 2-level-M and tail mistakes.
checks = [
    (0, 0),            # first tile
    (7, 95),           # m_hi=1, m_lo=3 (both nonzero), LAST n_tile (tail)
    (31, 95),          # last M-tile (m_hi=7, m_lo=3), last n_tile
    (13, 40),          # m_hi=3, m_lo=1, interior
    (18, 71),          # m_hi=4, m_lo=2, interior
]
worst = 0.0
for mt, nt in checks:
    got = kernel_tile(mt, nt)
    ref = gold_tile(mt, nt)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    worst = max(worst, rel)
    m_hi, m_lo = mt // 4, mt % 4
    print(f"[ok] fused tile (mt={mt:2d} [m_hi={m_hi},m_lo={m_lo}], nt={nt:2d}) rel-L2 = {rel:.3e}")
    assert rel < 2e-5, f"fused tile (mt={mt}, nt={nt}) failed: rel-L2={rel:.3e}"


# --- Distinct-row / 2-level-M guard (catches a swapped m_hi/m_lo or flat-M bug) ---
# mt=7 (m_hi=1,m_lo=3) and mt=13 (m_hi=3,m_lo=1) are the SWAP of each other's levels;
# they must map to DIFFERENT output rows and each match its own gold.
a7, a13 = kernel_tile(7, 40), kernel_tile(13, 40)
g7, g13 = gold_tile(7, 40), gold_tile(13, 40)
assert not np.allclose(a7, a13), "mt=7 and mt=13 (level-swapped) produced identical tiles!"
assert np.linalg.norm(a7 - g7) / np.linalg.norm(g7) < 2e-5
assert np.linalg.norm(a13 - g13) / np.linalg.norm(g13) < 2e-5
print("[ok] 2-level-M distinct-row guard OK (mt=7 and mt=13 map to distinct correct rows)")


# --- Base-only negative control: base-GEMM-only (residual omitted) must FAIL the gate ---
base = base_tile(7, 40)                            # base x@w without the low-rank residual
rel_base = np.linalg.norm(base - g7) / np.linalg.norm(g7)
assert rel_base > 2e-5, "base-GEMM-only unexpectedly within gate — residual not exercised!"
print(f"[ok] negative control: base-only (residual dropped) rel-L2 = {rel_base:.3e} > 2e-5 (correctly fails)")


# ======================================================================================
# N_CHUNK=512 widened base GEMM store mapping: 4-sub-tile store.
# ======================================================================================
# The widened kernels compute one 512-wide fused tile per chunk c and store it as 4
# sub-tile writes  out_sb[:, 128j:128j+128] -> v5[m_hi, m_lo, :, 4c+j, :]  for j in 0..3.
# We reproduce that store and confirm each written n_tile matches its own gold, so a
# swapped sub-tile order or a wrong 4c+j base would be caught here, before remote spend.
N_CHUNK_WIDE = 512
N_CHUNKS_WIDE = N // N_CHUNK_WIDE                  # 24
SUBTILES = N_CHUNK_WIDE // 128                     # 4 n_tiles per 512-wide chunk
assert (N_CHUNKS_WIDE, SUBTILES) == (24, 4)


def chunk_wide(mt, c):
    """Reproduce the 512-wide fused tile the widened kernel computes for chunk c.

    Same math as kernel_tile but over the full 512-wide column block [512c, 512c+512):
    base x@w K-accumulated + fused low-rank (x@a)@b, giving [m_in, 512].
    """
    m_hi, m_lo = mt // 4, mt % 4
    n0 = c * N_CHUNK_WIDE
    acc = np.zeros((128, N_CHUNK_WIDE), dtype=np.float32)
    for kt in range(K_TILES):
        lhs_t_kt = v1[m_hi, m_lo, :, kt, :].T        # [k_in, m_in]
        acc += lhs_t_kt.T @ v2[kt, :, n0:n0 + N_CHUNK_WIDE]   # base x@w
    tT = down_proj_tT(mt)                            # [R, m_in]
    acc += tT.T @ v4[:, n0:n0 + N_CHUNK_WIDE]        # fused low-rank (x@a)@b
    return acc


def store_512_to_v5(mt, c, wide_tile, out_v5):
    """The kernel's 4-sub-tile store: out_sb[:,128j:128j+128] -> v5[m_hi,m_lo,:,4c+j,:]."""
    m_hi, m_lo = mt // 4, mt % 4
    for j in range(SUBTILES):
        out_v5[m_hi, m_lo, :, SUBTILES * c + j, :] = wide_tile[:, 128 * j:128 * j + 128]


# Sample (m_hi, m_lo, c) with nonzero m_hi, all m_lo, and an N-chunk boundary (last c=23,
# covering n_tiles 92..95). A fresh v5 buffer is filled by the modeled store, then every
# written n_tile is compared against the width-128 gold tile for that (mt, 4c+j).
v5_store = np.zeros((8, 4, 128, N_TILES, 128), dtype=np.float32)
wide_checks = [
    (0, 0),            # first tile, first chunk (n_tiles 0..3)
    (7, 23),           # m_hi=1, m_lo=3 (both nonzero), LAST chunk (n_tiles 92..95)
    (13, 10),          # m_hi=3, m_lo=1, interior chunk (n_tiles 40..43)
    (18, 17),          # m_hi=4, m_lo=2, interior chunk (n_tiles 68..71)
    (31, 23),          # last M-tile (m_hi=7, m_lo=3), last chunk
]
worst_wide = 0.0
for mt, c in wide_checks:
    store_512_to_v5(mt, c, chunk_wide(mt, c), v5_store)
    m_hi, m_lo = mt // 4, mt % 4
    for j in range(SUBTILES):
        nt = SUBTILES * c + j
        got = v5_store[m_hi, m_lo, :, nt, :]         # what the 4-sub-tile store wrote
        ref = gold_tile(mt, nt)                      # width-128 gold for (mt, nt)
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        worst_wide = max(worst_wide, rel)
        assert rel < 2e-5, (
            f"512-wide 4-sub-tile store (mt={mt} [m_hi={m_hi},m_lo={m_lo}], c={c}, "
            f"j={j} -> nt={nt}) failed: rel-L2={rel:.3e}")
    print(f"[ok] 512-wide store (mt={mt:2d} [m_hi={m_hi},m_lo={m_lo}], c={c:2d} "
          f"-> n_tiles {SUBTILES*c}..{SUBTILES*c+3}) all 4 sub-tiles match gold")
print(f"[ok] N_CHUNK=512 4-sub-tile store mapping OK (worst sub-tile rel-L2 = {worst_wide:.3e})")


# --- Negative control: a WRONG sub-tile mapping must diverge from gold ---
# Reverse the sub-tile order (j -> 3-j) inside the chunk: this is the kind of mistake
# that would scramble the 512-wide store. It must produce a mismatched n_tile.
mt_bad, c_bad = 7, 23                                # last chunk, nonzero m_hi & m_lo
wide_bad = chunk_wide(mt_bad, c_bad)
bad_hits = 0
for j in range(SUBTILES):
    nt = SUBTILES * c_bad + j
    wrong = wide_bad[:, 128 * (SUBTILES - 1 - j):128 * (SUBTILES - 1 - j) + 128]  # j -> 3-j
    ref = gold_tile(mt_bad, nt)
    if np.linalg.norm(wrong - ref) / np.linalg.norm(ref) > 2e-5:
        bad_hits += 1
assert bad_hits > 0, "reversed sub-tile order unexpectedly matched gold — store check is blind!"
print(f"[ok] negative control: reversed 4-sub-tile order diverges from gold on "
      f"{bad_hits}/{SUBTILES} sub-tiles (mapping check is discriminating)")


# ======================================================================================
# Weight-fold identity: out = x@w + (x@a)@b == x@(w + a@b) == x@w'
# ======================================================================================
# The weight-fold kernel (lora_v4_fold) uses the LoRA algebraic identity to
# materialize w' = w + a@b once to fp32 HBM -- a (K,N) tensor in the SAME [k_tile,k_in,n]
# tiled layout as w (v2) -- then computes a pure base GEMM out = x@w' (the lora_v3_bf16_split
# bf16x2 3-product split with the down-proj / up-proj / resident-a machinery deleted; the
# fold absorbed the low-rank term into the weights). Its prologue builds each w' K-tile as
#   ab_chunk = aT[kt].T @ b_chunk = a[kt] @ b[:, cols] = [k_in,R] @ [R,width] = [k_in,width]
#   w'[kt][:, cols] = v2[kt][:, cols] + ab_chunk   (fp32 add, stored fp32 to HBM)
# We check the fp32 fold identity on sampled tiles so a fold mistake (wrong sign, a not
# transposed, dropped low-rank, wrong k-tile slice) is caught locally before any spend.


def w_prime_tile(kt, n0, width):
    """Reproduce the on-device fold prologue for one w' K-tile column block.

    aT[kt] = v3[kt].T = [R, k_in]; ab_chunk = aT[kt].T @ b_chunk = a[kt] @ b[:, cols]
        = [k_in, R] @ [R, width] = [k_in, width]; w'[kt][:, cols] = v2[kt][:, cols] + ab.
    """
    a_kt = v3[kt, :, :]                        # [k_in, R]  (== a[kt*128:(kt+1)*128, :])
    b_cols = v4[:, n0:n0 + width]              # [R, width]
    return v2[kt, :, n0:n0 + width] + a_kt @ b_cols     # [k_in, width]


def folded_out_tile(mt, nt, ab_sign=1):
    """Reproduce the on-device folded output tile: out = x @ w' K-accumulated over K-tiles.

    acc[m_in,n] += lhs_t[kt].T @ w'[kt][:, cols] = [m_in,k_in] @ [k_in,width] = [m_in,width].
    No separate low-rank term -- the fold absorbed (x@a)@b into w'. ab_sign=-1 is the
    wrong-fold negative control (w' = w - a@b), which must diverge from the gold.
    """
    m_hi, m_lo = mt // 4, mt % 4
    n0 = nt * N_CHUNK
    acc = np.zeros((128, N_CHUNK), dtype=np.float32)
    for kt in range(K_TILES):
        lhs_t_kt = v1[m_hi, m_lo, :, kt, :].T                # [k_in, m_in]
        ab = v3[kt, :, :] @ v4[:, n0:n0 + N_CHUNK]           # a[kt] @ b[:, cols]
        wp = v2[kt, :, n0:n0 + N_CHUNK] + ab_sign * ab       # w'[kt][:, cols]
        acc += lhs_t_kt.T @ wp
    return acc


# --- Fold weight identity: the tiled w' reconstruction matches the global (w + a@b) slice.
# A transposed a, a wrong k-tile slice, or a swapped b would be caught here (the tiled
# v2[kt]/v3[kt]/v4 indices must reconstruct exactly w[rows,cols] + a[rows,:]@b[:,cols]).
kt_s, n0_s, wq = 11, 512, 512                    # nonzero k-tile, interior n-block
wp_got = w_prime_tile(kt_s, n0_s, wq)                                     # [k_in, wq]
rows = slice(kt_s * 128, (kt_s + 1) * 128)
cols = slice(n0_s, n0_s + wq)
wp_ref = w[rows, cols] + a[rows, :] @ b[:, cols]                          # global w + a@b
rel_wp = np.linalg.norm(wp_got - wp_ref) / np.linalg.norm(wp_ref)
assert rel_wp < 2e-5, f"w' fold-weight identity failed for kt={kt_s}: rel-L2={rel_wp:.3e}"
print(f"[ok] w' fold-weight identity: kt={kt_s} n0={n0_s} -> tiled (v2[kt]+v3[kt]@v4) "
      f"matches global (w+a@b) slice, rel-L2 = {rel_wp:.3e}")


# --- Fold output identity: x@w' == x@w + (x@a)@b (== gold) on sampled tiles ---
# Reuse the same (mt, nt) sample coverage as the fused-accumulation `checks` above:
# corners + interior with nonzero m_hi, nonzero m_lo, nonzero k-tiles, a tail n_tile.
worst_fold = 0.0
for mt, nt in checks:
    got = folded_out_tile(mt, nt)                # x @ (w + a@b)
    ref = gold_tile(mt, nt)                      # x@w + (x@a)@b
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    worst_fold = max(worst_fold, rel)
    m_hi, m_lo = mt // 4, mt % 4
    print(f"[ok] fold identity (mt={mt:2d} [m_hi={m_hi},m_lo={m_lo}], nt={nt:2d}) "
          f"x@(w+a@b) == x@w+(x@a)@b rel-L2 = {rel:.3e}")
    assert rel < 2e-5, f"fold identity (mt={mt}, nt={nt}) failed: rel-L2={rel:.3e}"


# --- Negative control: a WRONG fold (w' = w - a@b, sign flipped) must diverge from gold.
# The low-rank term dominates the output magnitude (~99.6% of L2), so flipping its sign is
# a gross error the check must catch -- confirms the fold identity above is discriminating.
mt_wf, nt_wf = 7, 40                              # nonzero m_hi & m_lo, interior
wrong_fold = folded_out_tile(mt_wf, nt_wf, ab_sign=-1)     # w' = w - a@b
gold_wf = gold_tile(mt_wf, nt_wf)
rel_wf = np.linalg.norm(wrong_fold - gold_wf) / np.linalg.norm(gold_wf)
assert rel_wf > 2e-5, "sign-flipped fold (w' = w - a@b) unexpectedly within gate — fold check is blind!"
print(f"[ok] negative control: sign-flipped fold (w' = w - a@b) rel-L2 = {rel_wf:.3e} "
      f"> 2e-5 (correctly fails)")


print(f"\n[layout] CHECK PASSED (worst fused tile rel-L2 = {worst:.3e}, worst 512-wide "
      f"sub-tile rel-L2 = {worst_wide:.3e}, worst fold-identity rel-L2 = {worst_fold:.3e}) "
      "— 2-level M-index / a,b mappings / tT down-projection / fused accumulation / "
      "N_CHUNK=512 4-sub-tile store / weight-fold identity are correct.")
