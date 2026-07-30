# lora — Phase 3 Plan (regime / shape specialization; E1 weight-fold gated build, else finalize D2)

## Goal Description

Close out phase 3 of the `lora` NKI/Trainium kernel — the fused low-rank residual
`out = x@w + (x@a)@b` (M=4096, K=5120, N=12288, R=128, fp32; baseline 14.6645 ms). Phase 2
already promoted `lora_v3_bf16_split` (D2) at **1.297x / 11.3034 ms** (compensated bf16x2
3-product split of the base GEMM only, low-rank kept fp32 and fused into the base PSUM bank
with no HBM round-trip) and kept `lora_v2_mblk4` (D1) at **0.988x** as the fp32 fallback.

Phase 3 answers one narrow question: **is there any restructuring that beats D2's 1.297x, or
is D2 the finalize?** The operator is cleanly PE-bound at the base-GEMM systolic floor, the
shape is edge-free (every dimension an exact tile multiple — no ragged regime to specialize),
and the dominant lever (the bf16x2 3-product count) is already at the numeric floor proven on
the shape-identical sibling `matmul`. Exactly one lever is genuinely untested on this op and
worth a single gated remote run: the canonical **LoRA weight-fold** (E1). Two further levers
(E2, E3) are model-rejected against the profile, an offline SBUF budget, and measured sibling
precedent.

Concretely, phase 3 must:
1. Build and offline-gate **E1** (`runs/lora_v4_fold.py`): the algebraic weight-fold
   `x@w + (x@a)@b = x@(w + a@b)` — materialize `w' = w + a@b` once to HBM (fp32), then run a
   pure `x@w'` bf16x2 3-product GEMM (D2 with the down-proj / up-proj / a-transpose /
   resident-`a` machinery deleted). It removes D2's ~0.647 ms low-rank tail but pays back
   `a@b` materialization PE and extra HBM traffic; the net is genuinely uncertain and settled
   only by measurement.
2. Record **E2** (bf16x2 split of the up-projection) and **E3** (double-buffer the lhs limbs
   to close the 5% PE-idle bubble) as model-based rejects with their quantitative rejection
   basis; do not build them (E3 only as a strictly-gated contingency that its own SBUF budget
   already predicts will not open).
3. **Finalize**: promote E1 over D2 only if it clears a pre-registered numeric + latency +
   HBM gate; otherwise finalize `lora_v3_bf16_split` at 1.297x and keep `lora_v2_mblk4` as the
   fp32 fallback. Write the exit decision.

## Acceptance Criteria

Following TDD philosophy, each criterion lists positive tests (expected to PASS when the
criterion is met) and negative tests (expected to FAIL / be rejected when the implementation
is working correctly). All rel-L2 figures are the NKIBench relative-L2 over the flattened
output; the correctness gate is `< 2e-5`.

- **AC-1 (correctness hard gate):** the finalized kernel PASSES all 5 profiler seeds
  `[0,21,42,63,84]` at rel-L2 `< 2e-5` (via `verify.py`, which gates on `l2_norm_passed`).
  D2 already holds (worst 6.240e-7); E1, if built and run, must PASS.
  - Positive Tests:
    - `verify.py --op lora --candidate runs/lora_v3_bf16_split.py` reports PASS on all 5 seeds
      (the finalize-default path).
    - If E1 is measured: `verify.py --op lora --candidate runs/lora_v4_fold.py` reports PASS
      on all 5 seeds with worst rel-L2 in the modeled fold band (predicted device quadrature
      `sqrt(4.874e-7² + 4.45e-6²) ≈ 4.48e-6`, still ~4.5x under gate).
  - Negative Tests:
    - Any candidate whose worst-of-5-seeds rel-L2 is `≥ 2e-5` is rejected (not finalizable).
    - A folded kernel that stores `w'` as bf16 in HBM (routing the 99.6%-dominant low-rank
      term through a single weak rounding) would blow past ~2.3e-3 and FAIL — this must not be
      the implementation.

- **AC-2 (offline no-spend gate for E1, fail-closed):** before ANY remote run for E1, the
  extended offline sim (`runs/offline_lora_bf16_split_sim.py`) authorizes the fold with the
  fail-closed independent-reference control intact.
  - AC-2.1 (absolute authorize cap): the folded route worst-over-seeds
    (`composite_fold_bf16x2` = `mm_bf16x2_3prod(x, fp32(w + a@b))` scored vs the fp32
    reference) is `< 1.3e-5` (the existing pre-registered `AUTHORIZE_BELOW`).
  - AC-2.2 (model-consistency band): the folded worst is also `< 8e-6`. The model predicts
    ~4.45e-6; a result in `[8e-6, 1.3e-5)` means the numeric model is wrong → HALT and
    investigate, do NOT spend even though it is under the absolute cap.
  - AC-2.3 (fp32 reassociation control): a reported `fold_fp32_control` line
    `rel_l2(fp32(x@(w + a@b)), x@w + (x@a)@b)` isolates the fp32 reassociation error from the
    bf16 rounding. Expected ~fp32-floor level (~1e-7), confirming the folded-route error is
    bf16-dominated (matches the pure-GEMM sibling), not a reassociation artifact.
  - AC-2.4 (fail-closed control preserved): the independent-reference draw control still
    raises (not `assert`, survives `python -O`) if the NKIBench reference is unreachable or
    the draw model mismatches, so the gate can never authorize on an unvalidated input model.
  - Positive Tests:
    - Running the extended sim prints the `composite_fold_bf16x2` gate datum, the diversity
      draws, `fold_fp32_control`, and an authorize/HALT verdict; on the expected inputs it
      authorizes with folded worst ~4.45e-6 (`< 8e-6 < 1.3e-5`).
    - Pointing `$NKIBENCH_ROOT` at the checkout keeps the independent control at ~0.
  - Negative Tests:
    - A folded worst `≥ 1.3e-5` prints "does NOT authorize" and blocks the remote run.
    - A folded worst in `[8e-6, 1.3e-5)` triggers the model-consistency HALT (no spend).
    - Removing / breaking the reference module makes the sim RAISE (fail closed), never
      authorize.

- **AC-3 (promotion discipline over D2):** any promotion over D2 must beat it on the full
  5-seed p50 (not `--fast`) in a same-session interleaved A/B, by a margin outside both
  session drift and a meaningful-improvement floor.
  - AC-3.1 (drift bracketing): the D2 anchor is measured BOTH before and after E1 in the same
    session; `band = |D2_before_p50 − D2_after_p50|`.
  - AC-3.2 (unambiguous comparator): PROMOTE E1 iff `E1_p50 < min(D2_before_p50,
    D2_after_p50) − max(band, 0.34 ms)` — i.e. E1 must beat the *faster* of the two D2
    brackets by more than `max(band, 3% of 11.3034 ms = 0.34 ms)`.
  - Positive Tests:
    - An E1 whose p50 (e.g. ≤ ~10.9 ms) sits below the faster D2 bracket by > max(band, 0.34
      ms) with non-overlapping bands PROMOTES.
    - A same-session D2-vs-D2 control shows E1's win is not session drift (band reported).
  - Negative Tests:
    - An E1 within `max(band, 0.34 ms)` of the faster D2 bracket (a wash) is NOT promoted →
      finalize D2, record E1 as a measured reject.
    - A promotion argued from a `--fast` number, or from the historical 11.3034 ms alone
      without a fresh same-session bracket, is rejected as not meeting the discipline.

- **AC-4 (no unintended spill; quantitative HBM bands, read and write tracked separately):**
  E1's measured HBM stays within pre-registered bands; anything materially above is a spill →
  reject that variant. E3 is pre-gated on SBUF (predicted spill → not built).
  - AC-4.1 (write band): HBM write ≈ **453 MB** = 201 MB output + 252 MB intentional `w'`
    materialization. Allowance: `≤ 500 MB` (≈ +10% for compiler bookkeeping). A write beyond
    ~500 MB is an unmodeled spill → reject.
  - AC-4.2 (read band): HBM read ≈ **2361 MB** = `w'` main stream ~2016 MB + `w` prologue
    read 252 MB + `x` ~84 MB + `b` prologue ~6 MB + `a` ~3 MB. Note the fold REMOVES D2's
    repeated per-`m_hi` `b` reloads (~44 MB). Allowance: `≤ 2600 MB` (≈ +10%). A read beyond
    ~2600 MB is an unmodeled spill / re-fetch → reject.
  - AC-4.3 (E3 SBUF pre-gate): the E3 double-buffer resident set (~194 KB/partition) exceeds
    the 192 KB trn2 SBUF limit before compiler temporaries → predicted hard spill → E3 is not
    built unless a re-derived budget shows real headroom below 192 KB with margin.
  - Positive Tests:
    - E1's profiler `HBMwr_MB ≤ 500` and `HBMrd_MB ≤ 2600`, both within band.
    - D2's finalize digest confirms HBMwr 201 MB / HBMrd 2112 MB (byte-identical, no spill).
  - Negative Tests:
    - An E1 with `HBMwr_MB > 500` (e.g. spilling `w'` or accumulators) is rejected.
    - An E1 with `HBMrd_MB > 2600` (e.g. re-fetching `w'` or an SBUF-spill re-read) is
      rejected.
    - Building E3 while its SBUF budget is over 192 KB is rejected by the pre-gate.

- **AC-5 (honest seed caveat):** the report states that all 5 on-device "seeds" draw the
  seed-42 input (the adapter reseeds `np.random.seed(42)` before every draw), so the true
  distinct-input numeric margin is the offline sim's diversity draws `[0,21,63,84]`; both the
  on-device 5-seed PASS and the offline diversity worst are reported, never conflated.
  - Positive Tests:
    - `candidates.jsonl` / the exit decision report both the on-device worst rel-L2 and the
      offline diversity worst for the finalized kernel.
  - Negative Tests:
    - A report that presents the on-device 5-seed PASS as if it were 5 distinct-input draws
      (omitting the seed-42 reuse caveat) is incorrect.

- **AC-6 (evidence completeness):** the phase-3 evidence set is written and internally
  consistent (see the Evidence plan): `benchmark.csv` rows, `candidates.jsonl` nodes with a
  correct parent DAG, `profile/` digests, the extended sim and layout check, and the exit
  decision doc.
  - Positive Tests:
    - `benchmark.csv` has a row per measured candidate (E1 if built) plus a same-session D2
      re-confirm control row; `candidates.jsonl` has the E1 node (parent `lora_v3_bf16_split`)
      if built and E2 / E3 model-reject nodes with numeric rejection basis; a
      `docs/phase3-exit-decision.md` records the finalize.
    - `runs/_layout_check.py` and `runs/offline_lora_bf16_split_sim.py` run clean.
  - Negative Tests:
    - An exit decision that promotes E1 without a `benchmark.csv` row and a `profile/`
      digest backing the latency claim is incomplete.
    - A `candidates.jsonl` E1 node whose parent is not `lora_v3_bf16_split` breaks the DAG.

## Path Boundaries

Phase 3 is a highly deterministic, tightly-scoped decision task: the phase-2 draft and the
sibling precedents fix almost every choice. The bounds are narrow by design.

### Upper Bound (Maximum Acceptable Scope)
Build exactly one new measured kernel, `runs/lora_v4_fold.py` (the HBM-materialized weight
fold), extend `runs/offline_lora_bf16_split_sim.py` with the `composite_fold_bf16x2` +
`fold_fp32_control` routes and `runs/_layout_check.py` with the fold algebraic-identity check,
run the AC-2 offline gate, and — only if it authorizes — spend one full 5-seed same-session
interleaved A/B remote run (D2-bracketed). Record E2 and E3 as model-reject nodes with their
quantitative basis. Write `benchmark.csv` / `candidates.jsonl` / `profile/` evidence and
`docs/phase3-exit-decision.md`. Finalize the winner and keep `lora_v2_mblk4` as the fp32
fallback.

### Lower Bound (Minimum Acceptable Scope)
If the AC-2 offline gate does NOT authorize E1 (folded worst `≥ 1.3e-5`, or in the
`[8e-6,1.3e-5)` model-inconsistency band, or the fail-closed control raises), spend no remote
run: finalize `lora_v3_bf16_split` at 1.297x (keep `lora_v2_mblk4` as fp32 fallback), record
E1 as a numeric no-spend reject alongside the E2 / E3 model-rejects, run a single same-session
D2 re-confirm for the finalize digest, and write the exit decision. This still satisfies every
AC (E1's build + gate is done; the remote spend is correctly withheld).

### Allowed Choices
- **Can use:** the HBM-materialized `w'` fold (a `shared_hbm` scratch tensor inside the kernel,
  exactly how `out` is allocated); the sibling `matmul_v3` base-GEMM recipe on folded weights;
  `a`-transpose via the existing identity-transpose idiom; the D2 M-block (B=4), N_CHUNK=512,
  4-sub-tile store, and 3-product bf16 split structure unchanged; the existing offline-sim and
  layout-check harness extended in place.
- **Cannot use:** storing `w'` as bf16 in HBM (must be fp32 until the main-GEMM consumption
  point splits it into limbs); folding `a@b` into the per-chunk `w`-load in SBUF (forces
  N-chunk outermost — blows up x-transpose / lhs-limb residency — or recomputes `a@b` per
  M-block at 8x redundant PE); persistent cross-invocation `w'` precompute (the kernel is
  per-invocation `kernel(v1,v2,v3,v4)→out`; there is no amortization surface); any promotion
  from a `--fast` number or the historical 11.3034 ms without a fresh same-session bracket;
  editing the NKIBench benchmark definition (`kernels`, `reference`, `seeds`, `summary.json`).
- **Fixed by prior phases (no phase-3 action):** N_CHUNK=512 (one fp32 PSUM bank; 1024 illegal;
  smaller raises the matmul-site count); M-block B=4 (== `m_lo`, the natural arithmetic-free
  block; sibling D4 sweep: B=2 0.983x, B=8 0.968x, B=16 0.519x); no edge tiles (all dims exact
  tile multiples).

> **Note on Deterministic Designs**: This plan is deliberately deterministic. E1's structure
> (HBM-materialized fp32 `w'` prologue + D2-minus-tail main loop), the offline gate thresholds,
> and the promotion protocol are all fixed per the draft + sibling precedent. The only true
> degree of freedom is the binary spend/no-spend outcome of the AC-2 gate and the
> promote/finalize outcome of the AC-3 A/B — both decided by measurement, not implementer
> choice.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

**E1 — weight-fold kernel (`runs/lora_v4_fold.py`), two phases inside one kernel invocation:**

```
# Prologue (SEPARATE pass; fully materializes w' to HBM before any main-GEMM read):
allocate w_prime = shared_hbm (K=5120, N=12288) fp32          # intentional materialization
resident a as aT[kt] = [R(par)=128, k_in=128]                 # a transposed: R on partition
for kt in 0..39:                                              # 40 K-tiles
  for c in 0..23:                                             # 24 N-chunks of 512
    ab_chunk[512] = aT[kt].T @ b_chunk[c]        # [k_in,R]@[R,512] = [k_in,512], fp32 PSUM
    w_chunk = load w[kt, :, c*512 : c*512+512]   # fp32
    w_prime[kt, :, c] = w_chunk + ab_chunk       # fp32 add, STORE fp32 to HBM
# 40 a-transposes + 40*24 = 960 fp32 moving-512 matmuls (~0.2 ms PE, additive on PE-bound op)

# Main loop = sibling matmul_v3 (D2 base GEMM) on w' — down-proj/up-proj/resident-a DELETED:
for m_hi in 0..7:                                # B=4 M-block == m_hi, members == m_lo
  build resident bf16 lhs limbs (lhs_hi, lhs_lo) per member per K-tile  # as D2
  for c in 0..23:                                # N_CHUNK=512
    for kt in 0..39:
      load w'[kt, :, c] fp32; split -> w'_hi, w'_lo (bf16)   # w' is fp32 in HBM, split here
      for m_lo in 0..3:
        acc[m_lo] += lhs_hi@w'_hi + lhs_hi@w'_lo + lhs_lo@w'_hi   # 3-product; drop lo@lo
    store acc as 4 sub-tiles -> out[m_hi, m_lo, :, 4c+j, :]
return out
```

Key structural guards (learned from D2 + Codex review):
- **RAW ordering:** the prologue fully writes `w'` to HBM before the main loop's first `w'`
  read. Do not preload or speculatively consume any `w'` tile before its fp32 store completes.
- **`w'` is fp32 in HBM**, split into bf16 limbs only at the main-GEMM consumption point
  (identical to how D2 splits `w`). Storing `w'` as bf16 routes the 99.6%-dominant low-rank
  term through one weak rounding → catastrophic (~2.3e-3, fails the gate).
- **Prologue PSUM/SBUF is separate from the main loop:** prologue peak ≈ 1 transpose bank + 1
  `a@b` accumulator `[128,512]`; main loop peak ≈ 4 base-acc banks — never interleaved, so
  ≤ 4-5 of 8 PSUM banks in either phase. The main loop's resident set is *smaller* than D2 (no
  resident `a`, no `tT`), so SBUF fits comfortably.

**Latency intuition (why it is a genuine coin-flip):** E1 removes D2's ~0.647 ms low-rank tail
(sibling `matmul_v3` base wall 10.656 ms vs D2 11.3034 ms) but the `a@b` materialization
(~0.2 ms PE) is *additive* on a PE-bound op (it cannot hide under the main GEMM's PE work), and
+~250 MB net HBM write must fit under DMA idle. Realistic band ≈ **[1.30x wash … ~1.35x
prologue-hidden]**; the 14.6645/10.656 = 1.376x figure is an unreachable theoretical bound (it
assumes zero materialization cost). Expected gain over D2 ≈ 0–4% (0–0.45 ms) → wash is the
likely outcome, which is why the whole point is to *measure* it under a strict gate.

**E2 / E3 (model-reject, do not build):**
- E2 (split the up-projection): up-proj is 768 / 97536 = 0.79% of matmul instructions;
  splitting cuts ~0.13% of total PE, an order below the ~1.3% measurement noise, while adding
  `tT` / `b` limb builds. Moot under E1 (the fold removes the up-proj entirely).
- E3 (double-buffer lhs limbs): closing the 5% (0.569 ms) prologue bubble needs cross-block
  overlap → 2× lhs limbs ≈ 194 KB/partition > 192 KB → predicted hard spill (the
  `tmm_v7_dbuf_rhs` read-floor-break signature). The identical lever was BUILT and
  measured-rejected on two siblings (`tmm_v7_dbuf_rhs` broke the read floor with a write-spill;
  `bmm` phase-3 found cross-block blocking a monotone anti-lever).

### Relevant References
- `runs/lora_v3_bf16_split.py` — D2, the incumbent; E1's main loop is this minus the
  down-proj / up-proj / a-transpose / resident-`a` machinery, reading `w'` instead of `w`.
- `runs/lora_v2_mblk4.py` — D1, the fp32 fallback; the fp32 M-block structure E1 inherits.
- `runs/offline_lora_bf16_split_sim.py` — the AC-2 gate to extend (add `composite_fold_bf16x2`
  and `fold_fp32_control`; the existing `mm_bf16x2_3prod`, `draw_inputs`, independent-reference
  control, and `AUTHORIZE_BELOW=1.3e-5` are reused).
- `runs/_layout_check.py` — the host layout check to extend with the fold algebraic identity
  `x@(w+a@b) == x@w+(x@a)@b` (fp32, sampled tiles).
- `runs/dump_metrics.py` — per-engine / HBM metric dump for the same-session A/B and digests.
- `docs/phase2-exit-decision.md`, `docs/draft-phase3.md` — the phase-2 ground truth and the
  phase-3 thesis this plan formalizes.
- `../../verify.py` — the correctness + latency harness (`--fast` for direction, full 5-seed
  for the promotion metric); gates on `l2_norm_passed`.
- Sibling precedents: `../matmul` (the shape-identical base GEMM; 3-product is the numeric
  floor; B=4 D4 sweep), `../transpose_matmul` (`tmm_v7_dbuf_rhs` measured double-buffer reject;
  the `max(band,3%)` promotion bar), `../bmm` (cross-block blocking anti-lever;
  `BL-20260709-fast-vs-full-run-latency`).

## Dependencies and Sequence

### Milestones

1. **Offline gate + host checks for E1 (no remote spend).**
   - Phase A: extend `runs/_layout_check.py` with the fp32 fold identity check
     (`x@(w+a@b) == x@w+(x@a)@b` on sampled tiles, plus a wrong-fold negative control). Depends
     on nothing; must pass before building the kernel.
   - Phase B: extend `runs/offline_lora_bf16_split_sim.py` with `composite_fold_bf16x2`
     (= `mm_bf16x2_3prod(x, fp32(w+a@b))` vs the fp32 reference, over seed 42 + diversity
     draws) and a `fold_fp32_control` line, keeping the fail-closed independent-reference
     control. Run it → produce the AC-2 authorize / HALT / no-authorize verdict.

2. **E1 kernel build (contingent on Milestone 1 passing the host identity check).**
   - Phase A: write `runs/lora_v4_fold.py` — the fp32 `w'` prologue (separate pass, RAW to
     HBM) + the D2-minus-tail main loop reading `w'`. Parent `lora_v3_bf16_split`.
   - Phase B: local compile / trace sanity (kernel builds, `w'` scratch is fp32 dtype).

3. **Remote A/B (contingent on Milestone 1 Phase B authorizing AND Milestone 2 building).**
   - Step 1: same-session interleaved A/B — D2 anchor before, E1, D2 anchor after; full 5-seed
     p50, plus `runs/dump_metrics.py` for TRUE PE-active / HBMrd / HBMwr / psum counters.
   - Step 2: apply the AC-3 comparator (`max(band, 0.34 ms)`) and AC-4 HBM bands.

4. **Finalize + evidence.**
   - Step 1: promote E1 iff it clears AC-1 + AC-3 + AC-4; else finalize D2. Keep
     `lora_v2_mblk4` as fp32 fallback regardless.
   - Step 2: write `benchmark.csv` rows, `candidates.jsonl` nodes (E1 measured or no-spend
     reject; E2 / E3 model-reject with basis), `profile/` digest, and
     `docs/phase3-exit-decision.md`.

Dependency summary: Milestone 1 gates Milestone 2 (host identity must pass) and Milestone 3
(AC-2 must authorize). Milestone 3 gates the promote branch of Milestone 4. E2 / E3 rejects and
the D2 finalize path have no remote dependency and can be recorded regardless of the E1 outcome.

## Task Breakdown

Each task carries exactly one routing tag (`coding` = implemented by Claude, `analyze` =
executed via Codex `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Extend `runs/_layout_check.py` with the fp32 fold algebraic-identity check `x@(w+a@b) == x@w+(x@a)@b` on sampled tiles + a wrong-fold negative control | AC-2.3, AC-6 | coding | - |
| task2 | Extend `runs/offline_lora_bf16_split_sim.py` with `composite_fold_bf16x2` and `fold_fp32_control` routes; keep the fail-closed independent-reference control; run and capture the AC-2 verdict | AC-2 | coding | task1 |
| task3 | Independent numeric review of the extended offline sim (dilution loss, quadrature ~4.48e-6, the 8e-6 model-consistency band, fail-closed control intact) before authorizing spend | AC-2 | analyze | task2 |
| task4 | Write `runs/lora_v4_fold.py`: fp32 `w'` HBM prologue (separate RAW pass) + D2-minus-tail bf16x2 main loop reading `w'`; enforce the `w'`-fp32 and no-speculative-consume guards | AC-1, AC-4 | coding | task2 |
| task5 | If AC-2 authorizes: run the same-session interleaved A/B (D2 bracketed before+after E1, full 5-seed p50) + `dump_metrics.py`; apply the AC-3 comparator and AC-4 HBM bands | AC-1, AC-3, AC-4 | coding | task4 |
| task6 | Record E2 (up-proj split) and E3 (double-buffer lhs limbs) as model-reject `candidates.jsonl` nodes with the quantitative rejection basis; do not build | AC-4.3, AC-6 | coding | - |
| task7 | Finalize decision (promote E1 or finalize D2; keep `lora_v2_mblk4` fp32 fallback); write `benchmark.csv`, `candidates.jsonl` nodes, `profile/` digest, `docs/phase3-exit-decision.md` with the AC-5 seed caveat | AC-1, AC-3, AC-5, AC-6 | coding | task5, task6 |

## Claude-Codex Deliberation

### Agreements
- The HBM-materialized fp32 `w'` fold is the cleanest legal E1 structure; an SBUF-only fold
  either recomputes `a@b` ~8x per M-block or flips the loop order and rebuilds x limbs ~24x —
  both rejected. Persistent cross-invocation precompute is impossible for a per-invocation
  `kernel(v1,v2,v3,v4)→out`.
- E1 is a marginal, likely-wash experiment (expected 0–4% over D2) but is worth exactly one
  measured run *because* it is treated as a measured reject unless it clears a strict gate;
  the 1.376x figure is an unreachable bound (a@b PE is additive on a PE-bound op → realistic
  ceiling ~1.35x).
- The offline gate must isolate the fp32 reassociation error (`fold_fp32_control`) from the
  bf16 rounding rather than assuming the folded route equals the sibling pure-GEMM 4.45e-6.
- `w'` must be fp32 in HBM until the main-GEMM consumption point; storing it as bf16 routes the
  dominant low-rank term through one weak rounding and fails the gate.
- AC-4 must track HBM read and write separately with concrete bands; the fold removes D2's
  repeated `b` reloads (read increase < the +252 MB naive estimate) but the `w'` write increase
  is real.
- E2 and E3 remain sound model-rejects (E2: 0.13% of total PE, below noise; E3: ~194 KB/part >
  192 KB, plus measured sibling double-buffer rejects).

### Resolved Disagreements
- **E1 optimistic ceiling (1.376x):** Codex flagged it as too high because `a@b` PE cannot hide
  under the main GEMM's PE on a PE-bound op. Resolution: demote 1.376x to an unreachable
  theoretical bound; the realistic band is [1.30x … ~1.35x], expected gain 0–4%. Rationale:
  additive prologue PE is correct for a PE-bound op; this reframes E1 as a coin-flip and
  hardens the "likely finalize D2" expectation.
- **Offline spend trigger too loose at 1.3e-5:** Codex noted that if the model predicts
  ~4.45e-6, an offline result near 1.0e-5 means the model is wrong. Resolution: keep 1.3e-5 as
  the absolute fail-closed cap (AC-2.1) AND add a model-consistency band `< 8e-6` (AC-2.2) that
  HALTs on `[8e-6, 1.3e-5)` even though it is under the cap.
- **A/B comparator ambiguity:** Codex required a single unambiguous D2 target. Resolution
  (AC-3.2): E1 must beat the *faster* of `D2_before_p50` / `D2_after_p50` by
  `> max(band, 0.34 ms)` (3% of 11.3034 ms), with `band = |D2_before − D2_after|`. Adopts the
  sibling `tmm` `max(band, 3%)` bar and Codex's "beat the faster bracket" conservative rule.
- **AC-4 "materially above" too subjective:** Resolution — pre-register concrete bands + a +10%
  bookkeeping allowance (write ≤ 500 MB, read ≤ 2600 MB), tracked per direction.
- **`w'` dtype and RAW ordering risk:** Resolution — explicit correctness guards (fp32 `w'`,
  no speculative consume before the fp32 store, separate prologue pass) plus a host fp32
  fold-identity check (task1) and an on-device `w'`-dtype assertion.

### Convergence Status
- Rounds executed: 1 (Codex first-pass analysis + 1 convergence pass).
- Final Status: `converged` — the second Codex pass returned substantive AGREE, no DISAGREE,
  and only spec-wording REQUIRED_CHANGES (A/B comparator + numeric HBM tolerance), both folded
  into AC-3.2 / AC-4. Codex explicitly reported "No remaining user decision."

## Pending User Decisions

- DEC-1: Spend the one E1 remote run if the offline gate authorizes, given the plan's own
  expectation is a wash (0–4% over D2)?
  - Claude Position: Yes — spend it if and only if AC-2 authorizes (folded worst `< 8e-6`).
    E1 definitively tests the canonical LoRA weight-fold + the phase-1 prompt's stated hint on
    the real hardware; a single gated, D2-bracketed A/B is cheap insurance against leaving a
    ~1.35x on the table, and a measured reject is itself first-class phase-3 evidence (the
    "extra HBM round-trip does not pay" datum). If AC-2 does not authorize, spend nothing.
  - Codex Position: N/A — open question. Codex agrees E1 is "worth measuring only because the
    plan treats it as a measured reject unless it clears a strict bar"; it raised no objection
    to spending under the strict gate and reported no remaining user decision.
  - Tradeoff Summary: One remote 5-seed A/B (+ D2 brackets) of profiler cost to convert a
    modeled coin-flip into a measured fact, vs saving that spend by finalizing D2 on the model
    alone (E1's host build + offline gate would still be recorded as a no-spend datum). The
    offline AC-2 gate already prevents any spend on a numerically-unsafe fold; DEC-1 only
    governs the authorized-but-marginal case.
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific workflow terminology such as
  "AC-", "Milestone", "Phase", "Step", "task1", "E1/E2/E3", or "DEC-". These are plan-document
  markers only.
- Use descriptive, domain-appropriate names in the kernel and helpers (e.g. `w_prime`,
  `ab_chunk`, `fold_prologue`, `composite_fold_bf16x2`, `fold_fp32_control`) that describe the
  math, not the plan structure.
- Match the existing repository idiom: the offline sim's route-function naming
  (`composite_*`), the layout check's `[ok]` assertion style, and the D2 kernel's tiling /
  transpose / limb-build conventions.
- Candidate `.py` sources under `runs/` are tracked; keep them self-documenting (the sibling
  kernels carry a docstring explaining the structure and the numeric argument).

--- Original Design Draft Start ---

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

--- Original Design Draft End ---
