# lora — Phase 3 draft (regime / shape specialization)

## Where phase 2 left us (ground truth)

| kernel | precision | latency (full 5-seed) | speedup | worst rel-L2 | HBMrd | role |
|--------|-----------|-----------------------|---------|--------------|-------|------|
| lora_v1 (phase 1) | fp32 | 38.3562 ms | 0.382x | 4.874e-7 | 7813 MB | superseded |
| lora_v2_mblk4 (D1) | fp32 | 14.8385 ms | 0.988x | 4.874e-7 | 2150 MB | **fp32 fallback** |
| **lora_v3_bf16_split (D2)** | bf16x2 base | **11.3034 ms** | **1.297x** | 6.240e-7 | 2112 MB | **PROMOTED** |

`out = x@w + (x@a)@b`  (M=4096, K=5120, N=12288, R=128, fp32). Baseline 14.6645 ms.

D2 is the compensated bf16x2 3-product split of the **base GEMM only** on top of the fp32
M-block (B=4 == m_lo, N_CHUNK=512); the down-projection `x@a` and the fused up-projection
`(x@a)@b` stay fp32, fused into the base PSUM bank with **no HBM round-trip** for the
intermediate.

## Where the time goes (D2 profile — `profile/lora_v3_bf16_split_digest.txt`)

- **Cleanly PE-bound at the base-GEMM systolic floor.** Wall 11.3034 ms; TRUE PE-active
  **10.7342 ms** (PE 94.97%); DMA idle at **47.58%**; HBMrd 2112 MB, HBMwr 201 MB
  byte-identical (no spill). The PE-idle bubble is `11.3034 − 10.7342 = 0.569 ms = 5.0%`
  of wall.
- **Matmul-instruction decomposition** (97536 total, verified against the profiler count
  within 2048 = the compiler's transpose bookkeeping):
  - base `x@w` bf16x2 3-product: **92160 = 94.5%** of matmul instructions
  - low-rank tail (down-proj 1280 + up-proj 768): **2048 = 2.10%**
  - x identity-transpose (shared with the base): **1280 = 1.31%**
- **The base GEMM is bit-identical to the sibling `matmul` operator** (M4096/K5120/N12288).
  matmul phase-3 established that base is within a few % of its hard arithmetic ceiling
  (matmul_v3_bf16_split 1.274x; the bf16x2 3-product count is a proven numeric floor there).
  lora reaches a *higher* speedup (1.297x) on the same base because the lora baseline
  (14.66 ms) is slower than the matmul baseline (13.58 ms) — the fused fp32 low-rank tail
  is cheap on the numerator and the composite dilutes the split error 11.4x.
- **The shape is edge-free.** M=4096=32·128, K=5120=40·128, N=12288=24·512=96·128, R=128 (one
  tile). Every tile is full; there is no ragged / edge-tile regime to specialize (unlike a
  streaming op — this is the bmm / transpose_matmul situation, not silu / rmsnorm_matmul).

**Phase-3 thesis.** lora is PE-bound at the base-GEMM floor with an edge-free shape and the
dominant lever (bf16x2 product count) already at its proven numeric floor. The prompt's
stated win — *fuse the low-rank result into the base matmul's output accumulation without an
extra HBM round-trip* — is **already realized in D2** (PSUM-fused fp32 tail, HBMwr
byte-identical). So the phase-3 question is narrow: **is there any restructuring that beats
1.297x, or is D2 the finalize?** Exactly one lever is genuinely untested on this op — the
canonical LoRA weight-fold — and it is worth one gated remote run; the rest model-reject
against the profile and sibling precedent.

---

## E1 (BUILD + measure) — weight-fold `w' = w + a@b`, then `out = x@w'` bf16x2

**The idea.** Use the LoRA algebraic identity `x@w + (x@a)@b = x@(w + a@b)`. Materialize
`w' = w + a@b` once (a `(K,N)=(5120,12288)` fp32 tensor), then the main loop is D2 with the
down-proj / up-proj / `tT` / resident-`a` machinery **deleted** — a pure `x@w'` bf16x2
3-product GEMM, i.e. literally the sibling `matmul_v3` kernel on folded weights.

**Why it is worth measuring (not pre-rejecting).** The tail costs
`11.3034 − 10.656 = 0.647 ms` of wall over the identical base GEMM (sibling matmul_v3 =
10.656 ms). The fold removes that tail. Its optimistic ceiling is the sibling wall against
the lora baseline: `14.6645 / 10.656 = 1.376x`. The cost it pays back:
- **a@b materialization**: 40 a-transposes + 960 fp32 matmuls (`a[kt].T @ b`, moving 512),
  ~0.2 ms PE — roughly cancels the removed down+up PE (0.225 ms). PE ≈ unchanged.
- **HBM**: read `w` once (252 MB) + write `w'` once (252 MB) on top of the main `w'` stream
  (2016 MB over 8 M-blocks, = D2's `w` stream). Net **+504 MB DMA**. D2's DMA is 47.6%
  active (≈5.9 ms idle head-room per inf), so +0.65 ms of DMA can hide under the 10.7 ms PE.

So the fold ∈ **[1.30x (prologue exposed, wash) … 1.38x (prologue hidden)]** — genuinely
uncertain, dominated by whether the `w'` materialization overlaps the main GEMM's PE. This
is precisely a phase-3 shape/structure question that only a measurement settles (matches the
tmm / bmm discipline of building the top lever, not projecting it).

**Numeric safety — the fold LOSES the 11.4x dilution, so it must be re-gated offline first.**
D2 keeps the low-rank fp32, so the split error (4.453e-6 in isolation) is diluted 11.4x to
3.93e-7. The fold routes the **entire** output (including the 99.6%-dominant low-rank part)
through one bf16x2 GEMM `x@w'`, so its rel-L2 is the **undiluted pure-GEMM value ≈ 4.45e-6**
(offline route [B]). Predicted device quadrature `sqrt(4.874e-7² + 4.45e-6²) ≈ 4.48e-6` —
still **~4.5x under the 2e-5 gate**, same margin as every sibling bf16x2 GEMM. Safe, but the
margin drops from 32x to 4.5x, so it MUST be gated before spend.
- **Pre-registered offline gate (no remote spend):** extend `runs/offline_lora_bf16_split_sim.py`
  with a `composite_fold_bf16x2` route = `mm_bf16x2_3prod(x, (w + a@b))` scored against the
  fp32 reference; authorize the remote run only if worst-over-seeds < 1.3e-5 (the existing
  AUTHORIZE_BELOW), keeping the fail-closed independent-reference control.
- **Pre-registered SBUF/HBM budget:** the main loop's resident set is *smaller* than D2 (no
  resident `a` 20 KB, no `tT`) → fits comfortably. The `w'` write is intentional
  materialization, **not** a spill signature; the AC-4 discipline for E1 is *"the wall must
  drop"*, not *"read stays at 2112 MB"* (unlike E3, which must hold the read floor).

**Considered-and-rejected fold variant (record, do not build):** folding `a@b` into the
per-chunk `w`-load in SBUF (avoiding the HBM `w'` write) either forces N-chunk outermost
(blows up x-transpose / lhs-limb residency) or recomputes `a@b` per M-block (8x redundant
PE). The clean fold is the HBM-materialized `w'` prologue + a D2-minus-tail main loop.

**Deliverable:** `runs/lora_v4_fold.py` (parent `lora_v3_bf16_split`). Extend
`runs/_layout_check.py` with a fold identity check (`x@(w+a@b) == x@w + (x@a)@b`, host numpy).

**Promote / reject criterion (pre-registered):** run offline gate → if it authorizes, one
full 5-seed remote run. PROMOTE `lora_v4_fold` iff it PASSES all 5 seeds AND its p50 beats
D2's 11.3034 ms **beyond same-session noise** (interleaved A/B, non-overlapping bands, per
BL-20260709). Otherwise FINALIZE D2 and record E1 as a measured reject (the extra HBM
round-trip the prompt warns against does not pay for the tail removal).

---

## E2 (MODEL-REJECT) — bf16x2 split of the up-projection `(x@a)@b`

Offline route [B'] (base + up-proj split) = **4.438e-6**, safe. But the up-proj is **768
instructions = 0.79%** of the 97536 total; a bf16x2 3-product split cuts at most ~17% of its
PE = **0.13% of total PE**, an order of magnitude below the ~1.3% measurement noise, while
adding `tT` / `b` limb builds (more Vec/Scl). Rejected on the profile. Moot under E1 (the
fold removes the up-proj entirely). Datum recorded; no remote spend.

## E3 (MODEL-REJECT) — double-buffer lhs limbs to close the 5% PE-idle bubble

The 0.569 ms bubble is the per-`m_hi`-block prologue (transpose + down-proj + limb build)
that cannot overlap that block's own N-loop (the N-loop consumes the limbs). Closing it
needs cross-block overlap → **double-buffered lhs limbs**.
- **Pre-registered offline SBUF gate:** 2× lhs limbs (B=4) = 160 KB + a_local 20 KB + tT
  2 KB + per-chunk w transients ~12 KB = **~194 KB/partition > the 192 KB trn2 limit** →
  predicted **hard spill**. This is the `tmm_v7_dbuf_rhs` read-floor-break signature.
- **Sibling precedent (measured, not projected):** the identical double-buffer lever was
  BUILT and measured-rejected on two siblings — `tmm_v7_dbuf_rhs` engaged its overlap
  (~0.2% PE dip) but broke the AC-4 read floor with a write-spill; `bmm` phase-3 found
  cross-block blocking a monotone anti-lever (enlarged live set constrains the affine_range
  pipeline). The ceiling here is only the 5% idle gap.
Rejected on the offline SBUF gate + sibling measured precedent. Build only as a contingency
**iff** E1 finalizes to D2 AND a re-derived SBUF budget shows real headroom (it does not).

## Settled regimes (no phase-3 action)

- **N_CHUNK = 512** is fixed: 512 = one fp32 PSUM bank (max moving-free width); 1024 is
  illegal, smaller only raises the matmul-site count (tmm / swiglu finding).
- **M-block B = 4** is settled by D4 (sibling matmul: B=2 0.983x under-amortized, B=8 0.968x,
  B=16 0.519x pressure; B=4 == m_lo is lora's natural arithmetic-free block).
- **Edge tiles: none** — all dimensions are exact tile multiples.

---

## Acceptance criteria (phase 3)

- **AC-1 (hard gate):** the finalized kernel PASSES all 5 seeds `[0,21,42,63,84]` at rel-L2
  < 2e-5. D2 already holds (6.240e-7). E1 must PASS if built.
- **AC-2 (offline no-spend gate for E1):** the extended offline sim authorizes the fold
  (`composite_fold_bf16x2` worst < 1.3e-5) with the fail-closed independent-reference
  control intact, BEFORE any remote run for E1.
- **AC-3 (promotion discipline):** any promotion over D2 must beat 11.3034 ms in a
  same-session interleaved A/B with non-overlapping noise bands (full 5-seed p50 is the gate
  metric, not `--fast`).
- **AC-4 (no unintended spill):** E1's HBM write beyond the 201 MB output + intentional 252 MB
  `w'` materialization, or any read beyond the modeled fold band, is a spill → reject that
  variant. E3's SBUF budget is pre-gated (predicted spill → not built).
- **AC-5 (seed caveat, honest):** the on-device "5 seeds" all draw seed-42 inputs (adapter
  reseeds `np.random.seed(42)`); the true distinct-input numeric margin is the offline sim's
  diversity draws. Report both.

## Evidence plan

- `benchmark.csv`: one row per measured candidate (E1 if built; a D2 same-session re-confirm
  control).
- `candidates.jsonl`: E1 node (parent `lora_v3_bf16_split`) if built; E2 / E3 model-reject
  nodes with the numeric rejection basis.
- `profile/`: `lora_v4_fold_digest.{txt,md}` if E1 is measured; otherwise a phase-3
  D2-reconfirm digest.
- `runs/offline_lora_bf16_split_sim.py`: extended with the fold route (AC-2 gate).
- `runs/_layout_check.py`: extended with the fold algebraic-identity check.
- `docs/phase3-exit-decision.md`: the finalize decision.

## Expected outcome

Most likely **FINALIZE `lora_v3_bf16_split` at 1.297x** (keep `lora_v2_mblk4` 0.988x as the
fp32 fallback): lora is PE-bound at the base-GEMM floor, the shape is edge-free, and the
prompt's intended low-rank fusion is already realized without an HBM round-trip. The one
measured build (E1 weight-fold) definitively tests the canonical LoRA trick + the prompt's
hint; if the `w'` materialization overlaps the main GEMM's idle DMA it could reach ~1.35x
and PROMOTE, otherwise it confirms the extra round-trip does not pay and D2 is the finalize.
E2 / E3 model-reject against the profile, the offline SBUF gate, and the tmm / bmm / matmul
sibling measured precedents.
