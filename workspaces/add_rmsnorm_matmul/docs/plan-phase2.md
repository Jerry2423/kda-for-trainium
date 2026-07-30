# add_rmsnorm_matmul — Phase 2 Plan: Break the fp32 PE Floor with a Compensated bf16x2 Split-Matmul

## Goal Description

Phase-1 `add_rmsnorm_matmul_v1` is correct and PE-bound at the fp32 systolic floor
(PE=94%, MFU=44%, 0.4953 ms, 3.754x over the 1.859287 ms baseline; full-5-seed PASS at
rel-L2 1.4635e-5). On the bf16-native trn2 PE array a correct fp32 GEMM runs multiple
internal bf16 passes, capping MFU near ~44%. DMA (24%), Vec (19%), Scl (14%) are all
hidden under that floor, so the ONLY lever that cuts wall-clock is cutting PE time — and
the only correctness-viable way to do that on this shape is to run the matmul in bf16
arithmetic with a compensated two-limb split.

This phase transfers the sibling `rmsnorm_matmul`'s already-PROMOTED Phase-3 win — a
compensated **bf16x2 split-matmul** (each fp32 operand → two bf16 limbs; accumulate three
bf16 products `hi@hi + hi@lo + lo@hi` in fp32 PSUM; drop the negligible `lo@lo` term) — to
this near-identical M/N/K matmul. The sibling measured 0.4716 → 0.3688 ms (1.279x; 1.066x →
1.363x over its baseline) with full-5-seed PASS. Correctness is already de-risked here by a
zero-remote-spend offline numpy simulation (`runs/offline_bf16_split_sim.py`): the idealized
3-product split's WORST relative-L2 is 4.451e-6 across 7 distinct input draws × both `g`
placements — ~4.5x under the 2e-5 gate and ~3.3x below v1's own on-device 1.46e-5.

The work is two candidates, built in order:
- **v2** — a pure-fp32 refactor (`g`-into-`w'` fold + `inv_rms` post-scale eviction) that is
  the clean base for the bf16 diff, a same-session fp32 control, and a retained pure-fp32
  fallback. Expected within-noise vs v1; not a promotion candidate on its own.
- **v3** — the compensated bf16x2 split built on v2. This is the intended promotion. The
  directional expectation is ~0.387 ms / ~4.8x, but promotion depends only on measured
  correctness and a measured out-of-noise latency win, not on hitting that number.

The kernel's raw-2D I/O and exact signature `kernel(x_tensor, w_tensor, eps, z_tensor,
g_tensor)` are preserved throughout. This plan implements ONLY `add_rmsnorm_matmul`; it does
not touch the benchmark definition or any other operator.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. Scoring command (run from inside `workspaces/add_rmsnorm_matmul/`):
`python3 ../../verify.py
--op add_rmsnorm_matmul --candidate runs/<file>.py` (add `--fast` for the seed-42 quick
check; drop it for the full 5-seed gate).

- AC-1: **v2 (`runs/add_rmsnorm_matmul_v2_postscale.py`) is a correct pure-fp32 refactor.**
  It rewrites the algebra as `out[m,n] = inv_rms[m] · ( a[m,:] @ w'[:,n] )` with `a = x+z`
  and `w'[k,n] = g[k]·w[k,n]`, folding `g` into resident `w` once via a per-partition
  `[128,1]` `tensor_scalar` on each `[k_in, n]` weight tile, and applying `inv_rms` as a
  `tensor_scalar` post-scale reading the PSUM accumulator directly at PSUM→SBUF eviction
  (replacing v1's `nl.copy`). The RMSNorm reduction stays fully fp32 and `eps` is added
  AFTER the `/K` mean (two-op `mean_eps = sumsq·(1/K) + eps`, then `rsqrt`). This is
  algebraically equivalent to v1 and accepted by the measured L2 gate (fp32 rounding/order
  may differ; the offline `fp32_control` using exactly this commutation matched the
  reference to 4.82e-7, confirming the equivalence is sound).
  - Positive Tests (expected to PASS):
    - Full-5-seed `l2_norm_passed = True` at seeds `[0,21,42,63,84]`.
    - Measured rel-L2 no worse than v1's 1.4635e-5 by more than a fixed absolute margin
      of 5e-6 (i.e. rel-L2 ≤ ~2.0e-5 is a hard fail regardless; the expectation is that v2
      lands at or below v1's value since it is fp32 throughout). See DEC-1.
    - Measured p50 latency within noise of v1 (same-session), consistent with the sibling's
      analogous `v2_postscale` at +0.38%.
  - Negative Tests (expected to FAIL):
    - Any seed with `l2_norm_passed = False`.
    - `g` folded onto the wrong axis (e.g. broadcast along the N/free axis of `w` instead of
      the k_in/partition axis) — this changes the math and must fail the L2 gate.
    - `eps` added before or scaled by `1/K` (i.e. `rsqrt((sumsq+eps)/K)`) — deviates from the
      reference and must fail or degrade L2.
  - AC-1.1: v2 is NOT promoted unless it beats v1 out-of-noise (see AC-3 band); it is retained
    in `runs/` as the pure-fp32 fallback and same-session control regardless of promotion.
    - Positive: v2 file is kept and recorded even when its latency is within-noise of v1.
    - Negative: deleting v2 after v3 passes, or promoting v2 on a within-noise latency delta.

- AC-2: **v3 (`runs/add_rmsnorm_matmul_v3_bf16_split.py`) is a correct compensated bf16x2
  split-matmul built on v2.** Split order is PINNED and auditable: for weights, compute
  `w' = g·w` in fp32 FIRST, then `w'_hi = bf16(w')`, `w'_lo = bf16(w' − w'_hi)` (matches the
  offline `bf16x2_g_into_w` that was gated); for the activation, transpose RAW `a` via the
  exact fp32 identity matmul → `aT` fp32, then `aT_hi = bf16(aT)`, `aT_lo = bf16(aT − aT_hi)`.
  Accumulate the 3 bf16 products `aT_hi@w'_hi + aT_hi@w'_lo + aT_lo@w'_hi` in fp32 PSUM
  (dropping `aT_lo@w'_lo`), and apply `inv_rms` as the post-scale at eviction. `w'_hi`/`w'_lo`
  are split ONCE on the resident weight before the M-loop; `aT_hi`/`aT_lo` are split per
  transposed sub-tile.
  - Positive Tests (expected to PASS):
    - Full-5-seed on-device `l2_norm_passed = True` at seeds `[0,21,42,63,84]`.
    - `--fast` (seed 42) `l2_norm_passed = True` as the quick pre-check before the full run.
    - Measured on-device rel-L2 comfortably under the 2e-5 gate (offline predicts ~4.5e-6);
      the completed offline 7-draw × 2-placement gate (worst 4.451e-6) is the pre-authorization
      and the diverse-input evidence that mitigates the fixed-seed-42 adapter caveat.
  - Negative Tests (expected to FAIL):
    - Plain single-limb bf16 matmul (no compensation) — offline 2.3e-3, ~117x over the gate.
    - Split order reversed (e.g. splitting `g` and `w` separately then multiplying bf16 limbs,
      or splitting before folding `g`) — invalidates the offline authorization; must be
      rejected in favor of the pinned `w' fp32 → hi → residual → lo` order.
    - Dropping `aT_hi@w'_lo` or `aT_lo@w'_hi` (a 2-product split) — loses a compensation term
      and should push rel-L2 toward the plain-bf16 regime.

- AC-3: **v3 latency beats v1 out-of-noise.** Promotion requires the measured p50 latency to
  beat the v1 promoted datum by more than the noise band (>1.8%; see DEC-2), measured with a
  FULL 5-seed run (not `--fast`) repeated at least twice for stability, compared against BOTH
  the v1 promoted row (0.4953 ms) and a same-session v1/v2 rerun as the noise anchor. The
  directional expectation is ~0.387 ms (~4.8x) but the exact figure is not itself an acceptance
  requirement.
  - Positive Tests (expected to PASS):
    - Two full-5-seed measurements of v3 both beat the same-session v1 anchor by >1.8%.
    - No large p95/variance regression relative to v1 across the repeated runs.
  - Negative Tests (expected to FAIL):
    - A single `--fast` measurement used as the promotion basis.
    - A within-noise or regressed latency promoted anyway.

- AC-4: **v3 profiler digest is captured and the "HBM unchanged" assumption is validated.**
  Record PE%, MFU%, Vec%, Scl%, DMA%, HBMrd, HBMwr for v3 and compare HBMrd/HBMwr against v1's
  42 MB / 34 MB (limbs are built on-chip from the same fp32 HBM loads, so HBM should be
  materially unchanged); also compare PE/MFU against v1 and, where useful, the sibling promoted
  split to separate op-specific overhead (residual `z`, `g`-fold) from expected split behavior.
  - Positive Tests (expected to PASS):
    - A digest is written under `profile/` for v3 with all seven metrics.
    - HBMrd/HBMwr within a small margin of v1's 42 MB / 34 MB.
  - Negative Tests (expected to FAIL):
    - Promotion recorded with no profiler digest.
    - A material HBMrd/DMA rise (limb spill/reload) left unflagged in the evidence.

- AC-5: **Correctness invariants are never regressed** (shared by v2 and v3): fp32 RMSNorm
  reduction (`square`/`reduce`/`mean_eps`/`rsqrt`); `eps` added AFTER the `/K` mean, matching
  the reference exactly; the raw-2D I/O and exact signature `kernel(x_tensor, w_tensor, eps,
  z_tensor, g_tensor)` preserved; `inv_rms` post-scale and `g`-into-`w'` treated as
  algebraically-equivalent commutations accepted by the measured L2 gate (fp32 control 4.82e-7).
  - Positive Tests (expected to PASS):
    - Both kernels accept `(x_tensor, w_tensor, eps, z_tensor, g_tensor)` and return `(M, N)`.
    - `eps` is consumed as a runtime scalar added after the `/K` mean.
  - Negative Tests (expected to FAIL):
    - Changing the signature, tiling the I/O differently, or hard-coding `eps`.
    - Performing the RMSNorm reduction in bf16.

- AC-6: **Fallbacks and closed directions are respected.** D3 (4-product split keeping
  `aT_lo@w'_lo`, ~+8% PE) is pursued ONLY if v3's on-device rel-L2 comes back marginal
  (> ~1.5e-5); it is not explored proactively. D4 (fp32 loop reorder / stationary-reuse) and D5
  (off-PE transpose: `dma_transpose` fp32 INELIGIBLE, `nc_transpose(engine=vector)` fp32 +2.08%
  regress, `nl.load_transpose2d` within-noise) are CLOSED by the sibling and are record-only.
  Plain-bf16, M-blocking (w' already resident), and N_CHUNK≠512 (one-fp32-PSUM-bank optimum)
  are rejected.
  - Positive Tests (expected to PASS):
    - D3 attempted only after a marginal v3 rel-L2 datum, and the D3-skip is recorded when v3
      is comfortably under the gate.
    - The identity-matmul transpose is retained.
  - Negative Tests (expected to FAIL):
    - Proactively implementing the 4-product split before v3 is measured.
    - Reopening D4/D5 or re-testing plain-bf16/M-blocking/N_CHUNK≠512 without new profiling
      evidence that contradicts the sibling results.

- AC-7: **Evidence is recorded per the KDA workflow.** `benchmark.csv` gets one row per
  perf-relevant candidate (v2, v3) plus the gated decisions (offline-gate authorization; D3
  skip-or-run); `candidates.jsonl` gets DAG nodes v2→v1 and v3→v2 with metrics (the offline-sim
  node `add_rmsnorm_matmul_offline_bf16_split_sim` is already appended); `profile/` gets the
  bf16x2 before/after digest (offline-sim output already saved). The recorded v3 evidence must
  state the actual split order (`w' fp32 → hi → residual → lo`; `aT fp32 → hi → residual → lo`)
  so it is auditable.
  - Positive Tests (expected to PASS):
    - `benchmark.csv` and `candidates.jsonl` contain the v2 and v3 rows/nodes with parent links.
    - Per-seed rel-L2 and latency are recorded (not only worst/mean).
  - Negative Tests (expected to FAIL):
    - A promoted candidate with no `benchmark.csv` row or no `candidates.jsonl` node.
    - Evidence that omits the split order or the DAG parent link.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
Two new tracked kernel sources — `runs/add_rmsnorm_matmul_v2_postscale.py` (fp32 refactor:
`g`-into-`w'` fold + `inv_rms` post-scale eviction) and `runs/add_rmsnorm_matmul_v3_bf16_split.py`
(compensated 3-product bf16x2 split built on v2) — each verified full-5-seed on device, with a
v3 profiler digest, all evidence recorded in `benchmark.csv` / `candidates.jsonl` / `profile/`,
and the gated D3 4-product fallback implemented ONLY if v3's on-device rel-L2 is marginal. v3 is
promoted if and only if it clears both the full-5-seed L2 gate and the out-of-noise latency band.

### Lower Bound (Minimum Acceptable Scope)
The v3 compensated bf16x2 split kernel, verified full-5-seed on device and promoted if it clears
both gates, with its `benchmark.csv` row, `candidates.jsonl` node (parent v2, or v1 if v2 is
skipped), and profiler digest recorded. v2 is strongly preferred as the clean base / control /
fallback and is the default path, but the irreducible deliverable is a correct, gated v3 plus its
evidence. If v3 fails or regresses, the minimum is a recorded negative datum that re-confirms the
fp32 floor with v1 (and v2, if built) retained.

### Allowed Choices
- Can use: the sibling `rmsnorm_matmul_v2_postscale.py` and `rmsnorm_matmul_v4_bf16_split.py` as
  structural templates; `nisa.tensor_scalar` (per-partition scale; PSUM-source eviction),
  `nisa.tensor_tensor` (subtract for the limb residual), `nl.copy` with `dtype=nl.bfloat16` (the
  RNE fp32→bf16 cast), `nisa.nc_matmul` (identity-transpose and the bf16 products), `nisa.activation`
  (`square`, `rsqrt`), `nisa.tensor_reduce`; all-`w'`-resident SBUF layout; M-outer loop;
  N_CHUNK=512; the two-op `mean_eps` `eps` form.
- Cannot use: editing `../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}` or hand-tuning
  a baseline; changing the kernel signature or raw-2D I/O; performing the RMSNorm reduction or the
  `eps`/`rsqrt` in reduced precision; plain single-limb bf16 for the main matmul; reopening the
  closed D4/D5 directions or re-testing M-blocking / N_CHUNK≠512 without contradicting profiling
  evidence; promoting on `--fast` alone or on a within-noise latency delta.

> **Note on Deterministic Designs**: The draft specifies a highly deterministic design — the
> algorithm (3-product compensated bf16x2 with the pinned split order, `g`-into-`w'` fold,
> `inv_rms` post-scale, fp32 RMSNorm, two-op `eps`) is fixed by the offline gate and the sibling
> precedent. The upper and lower bounds therefore nearly converge on "correct, gated v3 + evidence";
> the main allowed latitude is whether v2 is built as a separate artifact (default: yes) and whether
> the D3 fallback is triggered (only on a marginal v3 rel-L2).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions,
> not prescriptive requirements.

### Conceptual Approach

v2 (fp32 refactor), per M-tile, mirrors v1 minus the two inline activation scales:
```
a       = x + z                                  # residual add, [128, K] fp32
sumsq   = reduce_add( square(a) )                # fp32 RMSNorm reduction over K
mean_eps= sumsq * (1/K) + eps                    # two-op tensor_scalar (eps AFTER /K)
inv_rms = rsqrt(mean_eps)                         # [128,1]
# w' = g * w folded ONCE before the M-loop:  w'[kt] = tensor_scalar(w_sb[kt], multiply, g_col[kt])
#   where g_col[kt] is the [128,1] slice of g over the k_in partition axis of tile kt.
aT[kt]  = identity_matmul_transpose(a[:, 128*kt : 128*kt+128])   # exact fp32, [k_in, m_in]
acc     = sum_kt  nc_matmul(aT[kt], w'[kt][:, n_chunk])          # fp32 PSUM [128, 512]
out     = tensor_scalar(acc, multiply, inv_rms)  # post-scale at PSUM->SBUF eviction
```

v3 (bf16x2 split) builds on v2. Split `w'` once before the M-loop and each `aT` sub-tile:
```
# once, on resident w':      w'_hi = bf16(w'),  w'_res = w' - w'_hi (fp32 tensor_tensor),  w'_lo = bf16(w'_res)
# per transposed sub-tile:   aT_hi = bf16(aT),  aT_res = aT - aT_hi,                        aT_lo = bf16(aT_res)
acc = sum_kt [ nc_matmul(aT_hi[kt], w'_hi[kt]) + nc_matmul(aT_hi[kt], w'_lo[kt]) + nc_matmul(aT_lo[kt], w'_hi[kt]) ]
out = tensor_scalar(acc, multiply, inv_rms)      # post-scale eviction (drop aT_lo@w'_lo)
```
The three deltas from the sibling `rmsnorm_matmul_v4` are: (1) the residual add `a = x+z` before
the norm; (2) the `g`-into-`w'` fold (one extra per-partition `[128,1]` `tensor_scalar` on each
resident weight tile, applied before the split); (3) `+eps` inside the rsqrt (kept as the two-op
`mean_eps` rather than folding `1/K` into the `rsqrt` scale, because `eps` is a runtime scalar).

SBUF feasibility (budget 192 KB/partition): `w'_hi` + `w'_lo` bf16 = 32 + 32 = 64 KB/part (equal to
v1's fp32 `w`); `aT_hi`/`aT_lo` bf16 `[8,128,128]` are ~2 KB/part each; plus the transient fp32 `w'`
during the split and the RMSNorm intermediates. The sibling v4 (same layout minus the `g`-fold) fit
comfortably, so this fits.

### Relevant References
- `runs/add_rmsnorm_matmul_v1.py` — the promoted Phase-1 fp32 kernel; the starting structure.
- `runs/offline_bf16_split_sim.py` + `profile/add_rmsnorm_matmul_offline_bf16_split_sim.txt` — the
  completed offline gate (pins seed 42, draw order `x→w→z→g`, `eps=1e-5`; worst 4.451e-6).
- `../rmsnorm_matmul/runs/rmsnorm_matmul_v2_postscale.py` — fp32 post-scale-eviction template for v2.
- `../rmsnorm_matmul/runs/rmsnorm_matmul_v4_bf16_split.py` — the PROMOTED 3-product bf16x2 template
  for v3 (limb construction, split order, 3-product accumulation).
- `../rmsnorm_matmul/benchmark.csv` — sibling latency evidence (v4: 0.4716→0.3688 ms, 1.363x).
- `../../AccelOpt/NKIBench/reference/add_rmsnorm_matmul_M4096_N2048_K1024_numpy_1.py` — the numpy
  reference defining the exact `eps`-after-mean, draw order, and I/O identity transform.
- `../../verify.py` — the remote profiler gate (`l2_norm_passed` across seeds; p50 speedup).

## Dependencies and Sequence

### Milestones
1. **v2 — fp32 refactor base / control / fallback**: implement the `g`-into-`w'` fold and `inv_rms`
   post-scale eviction on the v1 structure.
   - Phase A: write `runs/add_rmsnorm_matmul_v2_postscale.py`; confirm signature and invariants (AC-5).
   - Phase B: verify full-5-seed on device (AC-1); record latency vs a same-session v1 anchor; keep
     as fallback (AC-1.1); record evidence (AC-7).
2. **v3 — compensated bf16x2 split (the intended promotion)**: build on v2.
   - Phase A: write `runs/add_rmsnorm_matmul_v3_bf16_split.py` with the pinned split order and
     3-product accumulation (AC-2); confirm the `g`-broadcast axis and split order (AC-2 negative tests).
   - Phase B: `--fast` seed-42 pre-check, then FULL 5-seed L2 gate (AC-2), repeated ≥2× for latency
     stability against the v1 anchor and the v1 promoted row (AC-3).
   - Phase C: capture the profiler digest and validate HBM-unchanged (AC-4); record all evidence (AC-7);
     promote iff both gates clear.
3. **Gated fallback / negative-datum handling** (conditional):
   - Step 1: if v3's on-device rel-L2 is marginal (> ~1.5e-5), implement the D3 4-product split (AC-6);
     otherwise record the D3-skip decision.
   - Step 2: if v3 fails or regresses, record the negative datum re-confirming the fp32 floor; keep
     v1/v2 promoted (AC-6, lower bound).

Dependencies: v3 depends on v2 (v2 is its structural base and same-session control). The D3 fallback
depends on a measured-marginal v3 rel-L2. The offline gate (already PASS) is a prerequisite that is
satisfied. D4/D5 are closed and are not on the critical path.

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Implement `runs/add_rmsnorm_matmul_v2_postscale.py`: `g`-into-`w'` fold (per-partition `[128,1]` scale on each resident weight tile) + `inv_rms` post-scale eviction; fp32 RMSNorm; two-op `eps`; preserve signature/raw-2D I/O | AC-1, AC-5 | coding | - |
| task2 | Verify v2 full-5-seed on device; measure latency vs same-session v1 anchor; retain as pure-fp32 fallback; record `benchmark.csv` row + `candidates.jsonl` node (v2→v1) | AC-1, AC-1.1, AC-7 | coding | task1 |
| task3 | Implement `runs/add_rmsnorm_matmul_v3_bf16_split.py` on the v2 base: pinned split order (`w' fp32→hi→residual→lo`; `aT fp32→hi→residual→lo`), 3-product `hi@hi+hi@lo+lo@hi` fp32-PSUM accumulation, drop `lo@lo`, `inv_rms` post-scale | AC-2, AC-5 | coding | task2 |
| task4 | Verify v3: `--fast` seed-42 pre-check, then FULL 5-seed L2 gate; confirm `g`-broadcast axis and split order via the AC-2 negative tests | AC-2 | coding | task3 |
| task5 | Measure v3 latency FULL 5-seed ≥2× vs v1 promoted row and same-session anchor; require >1.8% out-of-noise win with no p95/variance regression | AC-3 | coding | task4 |
| task6 | Capture v3 profiler digest (PE/MFU/Vec/Scl/DMA/HBMrd/HBMwr); validate HBM-unchanged vs v1 42/34 MB; flag any material DMA/HBM rise | AC-4 | coding | task5 |
| task7 | Record all v3 evidence: `benchmark.csv` row, `candidates.jsonl` node (v3→v2) with per-seed rel-L2/latency and the audited split order, `profile/` digest; promote iff both gates clear | AC-7 | coding | task6 |
| task8 | Decision point: if v3 rel-L2 marginal (>~1.5e-5) implement D3 4-product split; else record D3-skip. If v3 fails/regresses, record the fp32-floor negative datum | AC-6 | coding | task7 |
| task9 | (Optional) Cross-check the v3 profiler digest against the sibling promoted split and the theoretical PE floor to separate op-specific overhead (`z`, `g`-fold) from expected split behavior | AC-4 | analyze | task6 |

## Claude-Codex Deliberation

### Agreements
- The compensated bf16x2 split is the correct primary lever given the sibling precedent and this op's
  passing offline gate; micro-rearranging fp32 work cannot beat the fp32 PE rate penalty.
- v2 as a pure-fp32 refactor is the right base / same-session control / retained fallback, and folding
  `g` into `w'` BEFORE the split is the correct order to match the passing offline `bf16x2_g_into_w`.
- Necessary invariants: fp32 RMSNorm reduction, `eps` after `/K`, raw-2D I/O + exact signature.
- Full-5-seed on-device promotion gate plus repeated timing is appropriate; `--fast` alone is insufficient.
- Deferring the 4-product D3 unless v3's on-device rel-L2 is marginal is justified by the offline margin;
  D4/D5 stay closed.
- The SBUF footprint fits (limb `w'` footprint equals v1's fp32 `w`; sibling v4 already fit).

### Resolved Disagreements
- "Exact fp32 commutations" (Codex DISAGREE): folding `g` into `w'` and moving `inv_rms` to post-PSUM
  scaling are algebraically equivalent but IEEE fp32 rounding/order can differ. **Resolution:** the plan
  now states "algebraically equivalent; accepted by the measured L2 gate" (AC-1, AC-5) rather than "exact",
  while noting the offline `fp32_control` matched the reference to 4.82e-7.
- AC-1 tolerance underspecified (Codex REQUIRED_CHANGE): **Resolution:** AC-1 now sets a concrete bound —
  full-5-seed PASS and rel-L2 no worse than v1 by more than a fixed 5e-6 absolute margin (and never above
  the 2e-5 gate); v2 remains control/fallback and is not a promotion candidate unless faster (AC-1.1, AC-3).
- Directional target treated as an AC (Codex DISAGREE): **Resolution:** the ~0.387 ms / ~4.8x figure is
  explicitly demoted to a directional expectation in the Goal and AC-3; promotion depends only on measured
  correctness and a measured out-of-noise latency win.
- v3 comparison baseline (Codex REQUIRED_CHANGE): **Resolution:** AC-3 compares v3 against BOTH the v1
  promoted row (0.4953 ms) and a same-session v1/v2 rerun anchor, same harness/seeds/mode.
- `g`-broadcast-axis correctness (Codex REQUIRED_CHANGE): **Resolution:** added as an explicit AC-1/AC-2
  negative test (wrong-axis `g` must fail the L2 gate).
- Split-order auditability (Codex REQUIRED_CHANGE): **Resolution:** AC-2 pins the order and AC-7 requires
  the recorded evidence to state it (`w' fp32→hi→residual→lo`; `aT fp32→hi→residual→lo`).
- Per-seed reporting, digest-vs-sibling cross-check (Codex OPTIONAL_IMPROVEMENTS): adopted into AC-7
  (per-seed rel-L2/latency) and task9 (analyze cross-check).

### Convergence Status
- Final Status: `converged` (Phase-3 Codex first-pass + one Phase-5 convergence round; all REQUIRED_CHANGES
  incorporated; no high-impact DISAGREE remains; two low-stakes threshold/artifact questions carried to
  Pending User Decisions).

## Pending User Decisions

- DEC-1: v2 retention — must v2 be kept as a separate `runs/` fallback artifact even if v3 passes
  immediately, or may it remain only as a measured same-session control row?
  - Claude Position: Keep `runs/add_rmsnorm_matmul_v2_postscale.py` as a tracked pure-fp32 fallback
    regardless of v3 — it is cheap insurance against a future evaluator that uses distinct per-seed inputs
    (the fixed-seed-42 adapter caveat), and it is the clean fp32 base the bf16 diff is read against.
  - Codex Position: Reasonable either way; flagged as a genuine choice — v2 could remain only as a measured
    control row rather than a retained artifact if minimizing tracked files is preferred.
  - Tradeoff Summary: Retaining v2 costs one tracked file and one verified run but preserves a pure-fp32
    fallback and a clean control; dropping it saves the run/file but loses the fallback and the isolation of
    "refactor noise vs bf16 win". Default (recommended): retain v2.
  - Decision Status: PENDING

- DEC-2: latency out-of-noise threshold — is `>1.8%` over v1 sufficient to promote v3, or should a larger
  practical margin be required because v1 is already PE-bound and profiler noise may be session-dependent?
  - Claude Position: `>1.8%` is sufficient given the sibling measured a far larger −21.8% on the identical
    matmul and the win is expected to be ~+28%; require the win to hold across ≥2 full-5-seed runs against a
    same-session anchor so session noise is controlled.
  - Codex Position: `>1.8%` may be tight given session-dependent remote noise; consider requiring a larger
    practical margin, or explicitly confirm `>1.8%` is acceptable.
  - Tradeoff Summary: A tighter band (1.8%) promotes real-but-modest wins but risks promoting on session
    noise; a wider band is safer but could reject a genuine win if the transfer underperforms the sibling.
    Given the expected ~+28% headroom, 1.8% with a repeated-run requirement is low-risk; revisit only if v3
    lands unexpectedly close to v1.
  - Decision Status: PENDING

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone",
  "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `w_prime_hi`, `a_transposed_lo`,
  `inv_rms`, `mean_eps`), matching the existing NKI kernel style in `runs/` and the sibling workspace.

--- Original Design Draft Start ---

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

--- Original Design Draft End ---
