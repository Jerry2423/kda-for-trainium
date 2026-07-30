# add_rmsnorm_matmul — Phase 2 draft (profile-driven optimization)

## 0. TL;DR

Phase-1 `add_rmsnorm_matmul_v1` is **PE-bound at the fp32 systolic floor** (PE=94%,
MFU=44%, 0.4953 ms, 3.754x). The trn2 PE array is bf16-native; a correct fp32 GEMM
runs multiple internal bf16 passes, capping MFU near ~44%. That floor is the whole
game — DMA (24%), Vec (19%), Scl (14%) are all comfortably hidden under it.

The **only lever above the fp32 PE floor** is the sibling `rmsnorm_matmul`'s proven
Phase-3 win, transferred here: a **compensated bf16x2 split-matmul** (each fp32
operand → two bf16 limbs, 3 bf16 products in fp32 PSUM, drop the negligible lo·lo
term). On the *identical* M/N/K matmul the sibling measured **1.066x → 1.363x
(+28%, −21.8%, 1.279x)**. Applied here that projects **3.754x → ~4.8x** (≈0.387 ms).

I have already de-risked the one real concern — correctness margin — with a
**zero-remote-spend offline numpy sim** (`runs/offline_bf16_split_sim.py`,
evidence in `profile/add_rmsnorm_matmul_offline_bf16_split_sim.txt`):

| quantity | rel-L2 | vs 2e-5 gate |
|---|---|---|
| fp32 control vs reference (validates seed/draw/eps model) | 4.82e-7 | — |
| **bf16x2 3-product, WORST over 7 draws + both g-placements** | **4.45e-6** | **~4.5x under** |
| bf16x2 4-product (keeps lo·lo, fallback) | 3.48e-6 | ~5.7x under |
| plain bf16 (rejected route, scale check) | 2.3e-3 | 117x over |
| on-device fp32 v1 (reference datum) | 1.46e-5 | 1.37x under |

The bf16x2 error (4.45e-6) is not only ~4.5x under the gate, it is **~3.3x below
even the on-device fp32 v1's own 1.46e-5** — the compensation over-recovers relative
to fp32-on-a-bf16-array. This authorizes ONE remote bf16x2 attempt.

Plan: implement a small fp32 refactor first (**v2**: g-into-w fold + inv_rms
post-scale eviction) as the clean, correctness-preserving base and same-session
control, then build the **v3 bf16x2 split** on it. Everything else the sibling
already closed; I list those as record-only / do-not-explore.

---

## 1. Starting point — the Phase-1 kernel and its profile

`runs/add_rmsnorm_matmul_v1.py` (PROMOTED, 3.754x, full-5-seed PASS rel-L2 1.46e-5):

- Raw-2D self-slicing (this case's `transform_to_nki_inputs` is identity).
- M-outer over 32 tiles. Per M-tile: load `x`,`z` → `a = x+z` → fused SBUF RMSNorm
  (`square` → full-1024 free-axis `tensor_reduce` → `mean_eps = sumsq·(1/K)+eps`
  two-op `tensor_scalar` → `rsqrt`) → inline per-row `[128,1]` `inv_rms` scale →
  inline per-K `g` broadcast multiply → identity-matmul transpose of the 8
  `[128,128]` K-sub-tiles → `nc_matmul` K-accumulate into `[128,512]` fp32 PSUM over
  4 N-chunks → copy → store.
- **All of `w` resident** (8×`[128,2048]`, 64 KB/part, loaded once) — this was the
  Phase-1 win over the baseline's ~256 MB of in-loop `w` reloads.

Profiler digest (AC-5): **PE=94% MFU=44% Vec=19% Scl=14% DMA=24% HBMrd=42MB HBMwr=34MB.**

### Bottleneck read
- **PE-bound at the fp32 floor.** MFU=44% ≈ the sibling `rmsnorm_matmul_v1`'s 46%.
  fp32 matmul runs several internal bf16 passes on the bf16-native trn2 PE array, so
  ~44–46% MFU is a *structural rate penalty*, not schedulable slack.
- **DMA=24% (42 MB read)** — a single pass over `x`(16) + `z`(16) + `w`(8) + tiny
  `g`/identity. Higher than the sibling's 25 MB only because of the residual `z`
  read; still far under the PE wall. HBMwr=34 MB ≈ the 32 MB output floor. **Not a
  concern** — no HBM lever needed.
- **Vec=19% / Scl=14%** — RMSNorm + the inline `g`/`inv_rms` scales, fully hidden
  under PE. Small, but the two inline scales are exactly what the fp32 refactor (v2)
  moves off the per-M-tile critical path.

Conclusion: to go faster we must **cut PE time**, and the only correctness-viable way
to cut PE time on this shape is to run the matmul in bf16 arithmetic with
compensation. Micro-rearranging fp32 work cannot beat the fp32 rate penalty.

---

## 2. Directions enumerated, ranked (benefit vs risk)

### D1 — fp32 refactor: g-into-w fold + inv_rms post-scale eviction  *(ENABLER, low risk)*

**Idea.** Rewrite the algebra as `out[m,n] = inv_rms[m] · ( a[m,:] @ w'[:,n] )` with
`a = x+z` and `w'[k,n] = g[k]·w[k,n]`:

- **`g`-into-`w`:** `g` is indexed by the contraction column `k`, so it does NOT
  commute past the matmul — but it can be *folded into the resident weight once*.
  `w_sb[kt]` is `[k_in(par), n(free)]` and `g[kt·128 + k_in]` varies along its
  **partition** axis, so `w'[kt] = tensor_scalar(w_sb[kt], multiply, g_col[kt])`
  with a per-partition `[128,1]` `g` column — a natural per-partition scale, applied
  **8 times at load** instead of v1's **32× `[128,K]` free-axis activation multiply**.
- **`inv_rms` post-scale:** `inv_rms[m]` is per-row, commutes with the matmul, so
  apply it at PSUM→SBUF eviction via `tensor_scalar(acc, multiply, inv_rms_col)` —
  this *replaces* the `nl.copy` v1 already does, so it is free, and it removes v1's
  inline `norm = a·inv_rms` `[128,K]` `tensor_scalar` (32× `[128,1024]`).

**Result base.** Per M-tile: load `x`,`z` → `a=x+z` → RMSNorm reduction only
(`square`, `reduce`, `mean_eps` two-op, `rsqrt` → `inv_rms[128,1]`) → transpose the 8
RAW-`a` K-sub-tiles → matmul `a @ w'` → post-scale by `inv_rms` at eviction → store.

**eps handling (unchanged from v1, deliberately).** Keep `mean_eps = sumsq·(1/K)+eps`
as a two-op `tensor_scalar` then `rsqrt(mean_eps)`. (The sibling folded `1/K` into the
`rsqrt` scale because it had no `eps`; here `eps` is a runtime scalar, and the two-op
form avoids a runtime-scalar `bias`-tile portability question for `[128,1]`-negligible
cost.)

**Expected latency.** Within-noise vs v1 (still fp32, still PE-bound at 94%). The
sibling's analogous `v2_postscale` was +0.38% (within noise). **D1 is not a
promotion candidate on its own** — its value is (a) the clean base for the bf16 diff,
(b) a same-session fp32 control, (c) a guaranteed pure-fp32 fallback that also
shrinks Vec/Scl. Keep v1 promoted unless v2 beats it out-of-noise.

**Correctness.** fp32 throughout; algebraically identical to v1 up to fp32
reassociation. The offline `fp32_control` (which uses exactly this g-into-w +
post-scale commutation) reproduces the reference to **4.82e-7** across all seeds →
the commutation is sound. Verify full-5-seed on device.

**Risk:** low. Every primitive (`tensor_scalar` per-partition scale, `tensor_scalar`
reading PSUM at eviction) is used in NKIBench baselines and the sibling `v2`/`v4`.

### D2 — compensated bf16x2 split-matmul  *(THE win; medium risk; offline-gated GREEN)*

**Idea.** Build on D1's base. Split each fp32 operand into two bf16 limbs
(round-to-nearest-even, the cast the Scalar/Vector engines apply):
```
a_hi  = bf16(a),         a_lo  = bf16(a  - a_hi)     # per transposed activation sub-tile
w'_hi = bf16(w'),        w'_lo = bf16(w' - w'_hi)    # per resident weight tile, split ONCE
```
Accumulate **3 bf16 products** in fp32 PSUM, dropping the negligible `a_lo·w'_lo`:
```
a @ w'  ~=  a_hi@w'_hi + a_hi@w'_lo + a_lo@w'_hi
out[m,n] = inv_rms[m] · (that sum)                    # post-scale at eviction (from D1)
```
- **Split placement:** transpose RAW `a` (exact fp32 identity matmul) → `aT` fp32 →
  split into `aT_hi`,`aT_lo` bf16. Splitting after the transpose is identical to
  before (the transpose is exact, bf16 rounding is element-wise) and costs **one**
  transpose, not two.
- **`g` folded into `w'` BEFORE the split** (D1). Offline sim: g-into-w (worst
  4.440e-6) is marginally *more* accurate than g-on-activation (4.451e-6) **and**
  cheaper (split once on resident `w'`, no per-tile `g` multiply). RMSNorm reduction
  stays fp32; `inv_rms` post-scaled at eviction.
- **Memory:** `w'_hi`,`w'_lo` are 2× bf16 `[128,2048]`×8 = 32+32 = 64 KB/part
  (same total as v1's fp32 `w`); `aT_hi`,`aT_lo` are bf16 `[128,128]`×8, tiny. Fits.
- **HBM unchanged (~42 MB):** limbs are built on-chip from the same fp32 HBM loads.

**Expected latency.** The sibling measured **1.279x (−21.8%)** on the identical
M/N/K matmul (0.4716 → 0.3687 ms, twice). Framing: fp32-on-bf16-array ≈ 4 internal
bf16 passes; the compensated split does 3 explicit passes → ~3/4 PE passes → matmul
ceiling ~1.33x, ~1.28x end-to-end after limb-split/cast overhead. Projected here:
**0.4953 / 1.279 ≈ 0.387 ms → 1.859287 / 0.387 ≈ 4.8x** (3.754x → ~4.8x).

**Correctness — de-risked offline (the key gate before any remote spend).**
`runs/offline_bf16_split_sim.py` reproduces the EXACT scored input (seed 42, draw
order `x→w→z→g`, `eps=1e-5` no-draw) and the NKIBench reference, then models the
idealized 3-product split (numpy RNE limbs, exact fp32 accumulation ≥ HW accuracy):
- fp32 control → reference: **4.82e-7** (model validated).
- **bf16x2 3-product WORST over {42,0,21,63,84,123,2024} × {g-into-w, g-on-act}:
  4.45e-6** — ~4.5x under the 2e-5 gate, ~3.3x below v1's on-device 1.46e-5.
- The rel-L2 over the K=1024 dot product averages quasi-independent per-limb
  rounding, so it lands ~4.5e-6, far under the naive per-element 2^-16≈1.5e-5 bound
  (same mechanism the sibling documented).

**Promotion gate (HARD, both required):** full-5-seed on-device `l2_norm_passed`
AND p50 latency beats v1 out-of-noise (>1.8% band). Idealized sim is a green light,
not a HW guarantee — the on-device 5-seed run still decides.

**Risk:** medium — a lower-precision arithmetic change. Mitigated by (a) the offline
7-draw sim, (b) an exact sibling precedent on identical shapes, (c) v1 retained as
the pure-fp32 fallback.

### D3 — 4-product bf16x2 (keeps a_lo·w'_lo)  *(accuracy-repair FALLBACK, gated)*

Only if D2's **on-device** rel-L2 comes back marginal (say > ~1.5e-5, which the
offline sim makes unlikely). Adds a 4th bf16 pass (~+8% PE, erodes the win) for
offline 3.48e-6 vs 3-product 4.45e-6 — a small accuracy gain not worth the pass
unless needed. Held as a documented fallback, not explored proactively.

### D4 — fp32 loop reorder / stationary-reuse  *(CLOSED by sibling — record-only)*

Sibling `v3_stationary_reuse` (kt-outer/c-inner, 4 live PSUM banks to amortize the
`[128,128]` stationary fills) measured **+0.19% within-noise**, PE=97% unchanged —
v1's `affine_range` loops already give the compiler fill-optimal scheduling freedom.
No within-fp32 micro-lever exists. **Do not explore.**

### D5 — off-PE transpose  *(CLOSED by sibling — record-only)*

All routes closed in the sibling on this remote:
- `nisa.dma_transpose` fp32 → **INELIGIBLE** (2-byte-dtype only; SFKVectorizer
  INTERNAL_ERROR / exit 70).
- `nc_transpose(engine=vector)` fp32 → **+2.08% REGRESS** (Vec 7→90%, fp32 Vector
  transpose ~30× the PE identity-matmul).
- `nl.load_transpose2d` → correct but **within-noise** (PE stays 97%): decisive proof
  the transpose is already fully hidden under the PE-bound matmul.

The identity-matmul transpose stays. **Do not explore.**

### Also N/A / rejected
- **Plain bf16 (no compensation):** offline 2.3e-3 — fails the gate by 117x. Rejected.
- **M-blocking:** N/A — `w'` already fully resident, nothing to amortize by blocking M.
- **N_CHUNK=512:** already at the one-fp32-PSUM-bank optimum; all dims divide evenly
  (no edge tiles). No tiling freedom to exploit.

---

## 3. Execution plan (≤5 iters/direction; ~2 candidates + gated fallbacks)

1. **v2 (D1, fp32 base):** `runs/add_rmsnorm_matmul_v2_postscale.py`. g-into-w fold +
   inv_rms post-scale eviction. Verify full-5-seed PASS; record latency (expect
   within-noise vs v1). Same-session control. Do not promote unless it beats v1
   out-of-noise; keep as pure-fp32 fallback.
2. **v3 (D2, bf16x2 split):** `runs/add_rmsnorm_matmul_v3_bf16_split.py`, built on v2.
   The offline gate is already PASS (4.45e-6). Run `--fast` first (seed 42), then the
   FULL 5-seed measurement (drop `--fast`) twice for latency stability. Promote iff
   full-5-seed `l2_norm_passed` AND latency beats v1 out-of-noise (target ~0.387 ms,
   ~4.8x). Capture profiler digest (expect PE≈96%, MFU≈45%, Scl↑ from limb casts,
   HBM unchanged).
3. **Fallback (D3) only if v3 marginal on HW:** switch to the 4-product split. Do not
   pursue proactively.
4. **If v3 fails/regresses:** re-confirm the fp32 floor, keep v1/v2 promoted, and
   record the negative datum (mirrors the sibling Phase-2 floor-confirmation form).

## 4. Evidence to record
- `benchmark.csv`: one row per perf-relevant candidate (v2, v3) + the gated decisions
  (offline-gate authorization; D3-skip if not needed).
- `candidates.jsonl`: DAG nodes v2→v1, v3→v2, with metrics; offline-sim node already
  appended (`add_rmsnorm_matmul_offline_bf16_split_sim`, parent v1).
- `profile/`: bf16x2 before/after digest; offline-sim output already saved.

## 5. Correctness invariants (never regress)
- fp32 RMSNorm reduction (`square`/`reduce`/`mean_eps`/`rsqrt`); eps added AFTER the
  `/K` mean, matching the reference exactly.
- Post-scale `inv_rms` and g-into-`w'` are exact commutations (fp32 control 4.82e-7).
- Every promotion gated on **full 5-seed** `l2_norm_passed`, not `--fast` alone.
- **CAVEAT (from sibling):** the adapter fixes seed 42 for all 5 profiler seeds, so
  on-device "5-seed PASS" is weak on *input* diversity; the offline 7-draw sim
  mitigates. Keep v1 as the pure-fp32 fallback in case a future evaluator uses
  distinct per-seed inputs.
