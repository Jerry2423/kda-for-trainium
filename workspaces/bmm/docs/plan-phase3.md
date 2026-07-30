# bmm Phase 3 — Batch-Axis Stream-Depth Reschedule (fp32, gated on TRUE PE-active)

## Goal Description

Determine whether the ~13.3% residual PE bubble on `bmm_v2` — the phase-2 promotion at
1.253x (2.0352 ms), whose TRUE PE-active (2.0116 ms) sits 13.3% above the trn2 fp32 PE
floor (1.775 ms) — is recoverable by a **pure-fp32 reschedule that deepens the
independent-matmul stream across the batch axis**, the one structural dimension phase 2
left serial. `bmm` is the NKIBench batched matmul `out[b] = lhs[b] @ rhs[b]`, `b in 0..15`:
`lhs (16,4096,64)=(B,M,K)`, `rhs (16,64,4096)=(B,K,N)` fp32 → `out (16,4096,4096)`;
`B=16, M=4096, K=64, N=4096`; baseline **2.550 ms**.

This phase deliberately establishes with measurement, before spending remote runs, that
**classic shape specialization has no surface here**, then tests exactly one lever
(batch-axis stream depth) two ways:

- **No edge tiles.** Every axis divides cleanly: `M=4096=32·128`, `N=4096=8·512=4·1024`,
  `K=64≤128` (single pass), `B=16`. No ragged remainder tile exists to special-case.
- **Tiles are already maximal.** The main matmul is `[K=64]×[64,512]→[128,512]`; the moving
  free dim 512 is the hard PSUM-bank wall on trn2 (one `nc_matmul` writes one bank = 512 fp32
  elems/partition; 2048/4096 width is trn3-only). No wider tile exists.
- **The K=64 split is fixed and cost-free.** trn2 matmul latency is proportional to the
  destination free dim (512) ONLY, independent of K and M, so a half-full contraction axis
  costs nothing extra; and K cannot be packed across batches because `out[b]` are
  block-diagonal (stacking two batches on a 128-row contraction would sum their products —
  numerically wrong; closed in phase 2).

The bottleneck is therefore **PE scheduling across the batch axis**, not tile shape. Because
the outer batch loop is already `nl.affine_range(B)`, the compiler may *already* pipeline
across batch boundaries — in which case an explicit reschedule is a no-op (phase 2 proved
multi-bank/issue-order/wide-store reschedules all compiled to measured no-ops). This phase is
therefore run as a **hypothesis test**, not a presumed win: promote only a candidate that
beats `bmm_v2` out-of-noise on TRUE PE-active; if none does, the terminal result is "no
Phase-3-tested pure-fp32 batch-stream-depth schedule recovered the residual; keep `bmm_v2`."

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. Scoring uses the KDA loop:
`python3 ../../verify.py --op bmm --candidate runs/<file>.py [--fast]`,
with the same-session `runs/dump_metrics.py` idiom for TRUE-counter reads.

- AC-1: **Round-0 re-anchor (measure-first; near-zero remote risk).** Re-measure `bmm_v2`
  same-session via `dump_metrics` and confirm the PE-bound premise the whole phase rests on,
  then record the structural attribution before any code change. Round-0 does NOT count
  against the ≤5 optimization-iteration budget.
  - Positive Tests (expected to PASS):
    - A fresh same-session `bmm_v2` run reports TRUE PE-active ≈ 2.012 ms (`tensor_engine_active_time / metric-window`) and `tensor_engine_active_time_percent` ≈ 98–99%.
    - The recorded DMA-active **time** (≈ 1.434 ms) is strictly **below** TRUE PE-active, confirming PE-bound with DMA hidden.
    - `matmul_instruction_count == 8704`; per-matmul PE-active ≈ 0.2311 us is recorded against the pure main-matmul floor 0.2133 us (isolating the 0.0178 us/instr schedulable gap on the main matmuls).
    - Evidence lands under `profile/` and the PE-bound framing is stated as the phase premise.
  - Negative Tests (expected to FAIL):
    - Deciding PE-bound-vs-DMA-bound from the coarse `verify.py` DMA% (which swung 74/100/67/1% on the *same* kernel) instead of the counter-level DMA-active time.
    - Skipping the re-anchor and reusing stale phase-2 numbers from a different session.
  - AC-1.1: **D3 stays closed (record-only, no new probe).** State that a compensated
    3-product bf16x2 main matmul costs 3.0 passes > the measured fp32/bf16 ratio of 2.0 and
    would raise PE-active on a PE-bound kernel; D3 remains SKIPPED.
    - Positive: The plan/notes explicitly record D3 as CLOSED with the 3.0 > 2.0 cost-gate rationale (offline rel-L2 4.44e-6 would pass accuracy but the cost gate fails; both are required).
    - Negative: Spending any remote iteration building a bf16x2 candidate.

- AC-2: **D1 — multi-batch blocking (PRIMARY, pure fp32).** A candidate that blocks `B_BLK`
  batches together: transpose the `B_BLK·32` m-tiles into one resident pack, load `B_BLK`
  `rhs` tiles resident, then stream `B_BLK·256` main-matmul sites with no transpose
  interleaved, so the Pass-1/Pass-2 transition happens `16/B_BLK` times instead of 16.
  - Positive Tests (expected to PASS):
    - Sweep `B_BLK ∈ {2, 4}` (both divide 16). Each variant compiles, runs, and preserves `matmul_instruction_count == 8704` and HBM at the 34 MB read / 1074 MB write floor with no spill.
    - The best variant clears the AC-6 promotion gate (out-of-noise TRUE PE-active drop confirmed by a non-overlapping same-session A/B bracket) and full 5-seed rel-L2 PASS.
  - Negative Tests (expected to FAIL):
    - A variant whose resident footprint spills (HBM read rises above the 34 MB floor) is rejected.
    - `B_BLK=8` is NOT run (nominal `8·32·4B` pack + `8·16 KB` rhs ≈ 256 KB/part > ~208 KB/part usable ⇒ would spill); `B_BLK=6` is NOT run (does not divide 16 ⇒ tail-batch handling this phase does not build).
    - A variant that changes the single-pass K=64 math or transpose-before-use ordering (any rel-L2 drift off ≈ 1.83e-7) is rejected as an indexing bug, not promoted.
  - AC-2.1: **`B_BLK` scaling is the fixed-vs-scaling discriminator.** Use the sweep to
    classify the residual, but do not over-trust a single flat point.
    - Positive: If `B_BLK=2` recovers roughly half of what `B_BLK=4` recovers (monotone TRUE PE-active drop), conclude the batch-boundary cost is FIXED per-transition and D1 is a real lever.
    - Positive: If `B_BLK=2` is flat in TRUE PE-active but compiled/ran sanely (no spill, `matmul_instruction_count == 8704`, 5-seed PASS, no regression), still run `B_BLK=4` — a threshold effect may need a deeper stream before it moves.
    - Negative: Early-stopping the sweep after a flat `B_BLK=2` is only allowed when `B_BLK=2` is invalid (spills, changes instruction counts unexpectedly, fails correctness, or clearly regresses); otherwise `B_BLK=4` must be run before concluding D1 is dead.

- AC-3: **D2 — cross-batch double-buffering (ALT; only if D1 lands as a no-op).** Explicit
  ping-pong of two `(rhs, lhs_t_pack)` buffer sets: prefetch batch `b+1` (DMA-load `rhs[b+1]`,
  transpose `lhs[b+1]`'s pack into the alternate buffer) while batch `b`'s main stream runs
  (64 KB/part resident, fits < 208 KB/part).
  - Positive Tests (expected to PASS):
    - D2 is attempted only after D1 is measured to be a no-op; it preserves `matmul_instruction_count == 8704`, HBM at floor, and full 5-seed rel-L2 PASS.
    - Any promoted D2 candidate shows an out-of-noise TRUE PE-active drop (AC-6), i.e. it demonstrably reduces a PE bubble, not just overlaps DMA.
  - Negative Tests (expected to FAIL):
    - Spending iterations on both D1 and D2 after the first already landed a promotion.
    - Promoting a D2 candidate on a coarse DMA% improvement alone while TRUE PE-active is flat (DMA is already hidden, so DMA overlap cannot lower a PE-bound wall clock).
    - Exceeding the ≤2-iteration D2 budget.

- AC-4: **D3 — precision (bf16x2): CLOSED, record-only.** The only lever touching the
  1.775 ms PE floor is precision, and it is closed.
  - Positive Tests (expected to PASS): The plan records D3 as SKIPPED with the measured-ratio rationale (see AC-1.1); no bf16x2 kernel is built.
  - Negative Tests (expected to FAIL): Building or promoting any reduced-precision main-matmul candidate, or a bf16 output (the 2e-5 gate bans bf16 output).

- AC-5: **Correctness invariance + fallback.** Every promoted candidate is a pure reschedule.
  - Positive Tests (expected to PASS):
    - Full 5-seed rel-L2 PASS on seeds `[0, 21, 42, 63, 84]`, staying ≈ 1.83e-7; the max rel-L2 over all seeds is recorded (not just pass/fail).
    - `matmul_instruction_count == 8704`, single-pass K=64 preserved, transpose-before-use exact.
    - `bmm_v2` and `bmm_v1` are retained as fallbacks; the NKIBench baseline/reference are never edited.
  - Negative Tests (expected to FAIL):
    - Any rel-L2 drift off ≈ 1.83e-7 (signals an indexing/ordering bug) — reject, do not promote.
    - Deleting or overwriting `bmm_v2`/`bmm_v1`, or editing `../../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}`.

- AC-6: **Promotion / rejection gates (decide on TRUE PE-active, never coarse PE%/DMA%).**
  - Positive Tests (expected to PASS) — PROMOTE iff ALL hold:
    - Full 5-seed rel-L2 PASS (AC-5); `matmul_instruction_count == 8704`; HBM at 34/1074 MB floor, no spill.
    - TRUE PE-active drops **out-of-noise** vs a same-session `bmm_v2` anchor — "out-of-noise" = the drop exceeds the ~1.5–2% jitter band AND a same-session interleaved A/B bracket is **non-overlapping** (the exact evidence standard by which `bmm_v2` itself was promoted); counters (Vec/Scl/psum-copy/HBM) remain invariant.
    - p50 latency also wins out-of-band (promotion requires BOTH the primary TRUE-PE-active gate and the p50 gate).
  - Negative Tests (expected to FAIL) — REJECT / do-not-promote:
    - A candidate whose TRUE PE-active is **within same-session bracketed noise with invariant counters** is a compiler no-op → reject immediately; do not chase its latency noise. (For a pure reschedule `matmul_instruction_count` stays 8704 by design, so the no-op tell is TRUE PE-active + bracket overlap, NOT the instruction count.)
    - p50 improves but TRUE PE-active is flat → REJECT (latency noise).
    - TRUE PE-active improves but p50 does not → do NOT promote yet; KEEP as non-promoted evidence and repeat-confirm before any promotion.
    - `--fast` (seed-42) screens are filters only, never promotion evidence — promotion requires a full 5-seed run.

- AC-7: **Bookkeeping and discipline (≤5 optimization iterations total).**
  - Positive Tests (expected to PASS):
    - Each perf change appended to `benchmark.csv`; each candidate to `candidates.jsonl` with parent links (DAG); profiling evidence under `profile/`.
    - Per-candidate attribution recorded (schedule win / compiler no-op), with consistent terminology: source-level `nc_matmul` **sites** (256 main per batch) vs hardware **`matmul_instruction_count`** (512 main PE instructions per batch × 16 = 8192, + 512 transpose = 8704).
    - The same-session `bmm_v2` noise anchor is re-run before each comparison.
  - Negative Tests (expected to FAIL):
    - Exceeding 5 optimization iterations (Round-0 re-anchor excluded from this count).
    - Writing kernels outside `runs/`, or hand-tuning a baseline.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices for this
narrowly-scoped, measurement-gated phase.

### Upper Bound (Maximum Acceptable Scope)
The full D1 `B_BLK ∈ {2, 4}` sweep is run and, only if D1 is measured to be a compiler
no-op, one D2 cross-batch double-buffer variant (≤2 iterations) is attempted — every variant
gated on TRUE PE-active, with the single best out-of-noise candidate promoted as `bmm_v3`
over `bmm_v2`, full 5-seed PASS, HBM at floor, `bmm_v2`/`bmm_v1` kept as fallbacks, and
complete `benchmark.csv` / `candidates.jsonl` / `profile/` evidence with per-decision
attribution. Total ≤5 optimization iterations.

### Lower Bound (Minimum Acceptable Scope)
The Round-0 re-anchor confirms (or refutes) the PE-bound premise and D3-closed statement,
and at least one D1 `B_BLK=2` screen is run. If that screen is within same-session bracketed
noise with invariant counters (compiler no-op), the honest terminal result is recorded — "no
Phase-3-tested pure-fp32 batch-stream-depth schedule recovered the residual; `bmm_v2` is kept
as the promotion" — with the evidence booked. Establishing this negative result rigorously is
itself an acceptable completion of the phase.

### Allowed Choices
- Can use: pure-fp32 reschedules of the existing single-pass-K=64 two-phase structure
  (multi-batch blocking, cross-batch double-buffering / operand ping-pong, sub-block
  batch-interleave variants); `nl.affine_range` loop restructuring; resident SBUF packing up
  to the no-spill budget; the `dump_metrics` same-session TRUE-counter idiom; `--fast` screens
  as filters.
- Cannot use: reduced precision on the main matmul (bf16x2 / D3) or bf16 output (2e-5 gate
  bans bf16 out); wider moving free dim > 512 (trn2 PSUM-bank wall, trn3-only); K-packing two
  batches onto 128 contraction partitions (block-diagonal ⇒ numerically wrong); off-PE
  `load_transpose2d` (phase-2 counter-verified no-op — do not re-probe); store-direct-from-PSUM
  (no PSUM→HBM DMA path on trn2); editing the NKIBench baseline/reference or hand-tuning a
  baseline; `B_BLK ∈ {6, 8}` (tail / spill, excluded upfront).

> **Note on Deterministic Design**: This phase is highly constrained by measurement. The
> lever (batch-axis stream depth) and the fp32/single-pass-K=64/exact-transpose math are
> fixed; the only genuine choice is `B_BLK` and the D1-vs-D2 route, both resolved by the
> TRUE-PE-active gate rather than by preference. The upper and lower bounds converge quickly
> if the first screen shows a compiler no-op.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach
`bmm_v2` is two-phase *per batch*: for each of 16 batches it loads `rhs[b]`, transposes all
32 m-tiles into a resident `[64,4096]` pack (Pass 1), then streams 256 main matmuls (Pass 2).
Phase 2 showed deepening Pass 2 within a batch cut the per-matmul stall monotonically
(M_SUB 8→16→32 → 1.253x), but Pass 2 is re-primed 16 times and each batch head re-enters the
serial transpose→copy Pass 1 before the matmul stream refills the PE. The residual ~0.237 ms
is hypothesized to live at these 16 batch-boundary transitions.

D1 (literal) hoists the `rhs` load + transpose pack to cover `B_BLK` batches, then runs one
longer main stream, so the transition occurs `16/B_BLK` times:

```
for blk in affine_range(16 // B_BLK):
    # Pass 1: load rhs and transpose all m-tiles for the B_BLK batches in this block
    for bb in range(B_BLK):
        load rhs[blk*B_BLK + bb] resident
        transpose that batch's 32 m-tiles into the shared pack
    # Pass 2: one deep main-matmul stream across all B_BLK batches, no transpose interleaved
    for bb in range(B_BLK):
        for each of that batch's 256 main-matmul sites: nc_matmul -> copy -> 1024-wide store
```

Because the transpose floor is only 0.027 ms of the 0.237 ms residual, D1 must recover
**dependency/issue-latency gaps** at the batch head, not transpose work itself — reducing the
transition count is the mechanism, not removing transposes. A viable alternative if literal
blocking bloats live ranges (Codex suggestion): a **sub-block batch-interleave** — transpose
one m-tile across several batches, then stream, giving the compiler cross-batch PE
independence with a shorter pack live-range and less SBUF pressure than a full `B_BLK=4`
block. D2 is the explicit-overlap fallback (ping-pong two operand buffer sets) if D1's static
blocking is absorbed by the compiler.

### Relevant References
- `runs/bmm_v2.py` — the phase-2 promotion / start point (two-phase per-batch, M_SUB=32).
- `runs/bmm_d1_twophase.py`, `runs/bmm_d1_twophase_m16.py` — the phase-2 M-block trajectory (M8→M16→M32) proving stream depth is the lever.
- `runs/dump_metrics.py` — same-session TRUE-counter idiom (`tensor_engine_active_time_ns`, `matmul_instruction_count`, Vec/Scl/psum-copy, HBM bytes, DMA-active time).
- `profile/bmm_phase2_d2_dump_metrics.txt` — the counter evidence the PE-bound reframe rests on (PE-active 2.0116 ms > DMA-active 1.434 ms).
- `runs/bmm_v1.py` — pure-fp32 correctness-base fallback (0.663x).
- `../../AccelOpt/NKIBench/reference/` — numpy reference (read-only; never edit).
- `../../verify.py` — the correctness/perf gate (rel-L2 over 5 seeds; `--fast` = seed 42).

## Dependencies and Sequence

### Milestones
1. **M1 — Round-0 re-anchor (premise confirmation).**
   - Phase A: Re-run `bmm_v2` same-session via `dump_metrics`; confirm TRUE PE-active ≈ 2.012 ms and DMA-active time < PE-active (PE-bound).
   - Phase B: Record the structural attribution (0.0178 us/instr main-matmul gap) and the D3-closed statement. Gate: if the re-anchor refutes PE-bound, revisit the phase premise before building.
2. **M2 — D1 multi-batch blocking (primary lever).**
   - Phase A: Implement `B_BLK=2`; `--fast` + `dump_metrics` screen; check no-spill and counter invariants.
   - Phase B: Per AC-2.1, run `B_BLK=4` unless `B_BLK=2` is invalid/spills/regresses; promote the best out-of-noise variant on a full 5-seed run with a non-overlapping A/B bracket.
3. **M3 — D2 cross-batch double-buffer (conditional fallback).**
   - Step 1: Only if D1 is a measured no-op, implement the operand ping-pong.
   - Step 2: Gate on an out-of-noise TRUE-PE-active drop; ≤2 iterations; do not run if D1 already promoted.
4. **M4 — Terminal bookkeeping and attribution.**
   - Step 1: Append `benchmark.csv` rows and `candidates.jsonl` DAG entries; save `profile/` evidence.
   - Step 2: Record the per-decision attribution (schedule win vs compiler no-op) and, if nothing moved, the terminal "keep `bmm_v2`" conclusion.

Dependencies: M1 gates M2 (premise must hold). M2 gates M3 (D2 runs only if D1 is a no-op).
M4 depends on all prior milestones. D3 is closed and depends on nothing (record-only).

## Task Breakdown

Each task includes exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Round-0 re-anchor: re-run `bmm_v2` same-session via `dump_metrics`; confirm TRUE PE-active ≈ 2.012 ms and DMA-active time < PE-active; record `matmul_instruction_count==8704`, per-matmul 0.2311 us vs floor 0.2133 us; save to `profile/` | AC-1 | coding | - |
| task2 | Record-only D3-closed statement (3.0 > 2.0 pass ratio on a PE-bound kernel); no bf16x2 build | AC-1.1, AC-4 | coding | task1 |
| task3 | Analyze Round-0 evidence: confirm/refute PE-bound premise and batch-boundary localization; bound the fixed-vs-scaling expectation for the `B_BLK` sweep | AC-1, AC-2.1 | analyze | task1 |
| task4 | Implement D1 `B_BLK=2`; `--fast` + `dump_metrics` screen; verify no-spill (HBM 34/1074 MB) and `matmul_instruction_count==8704` | AC-2, AC-5 | coding | task3 |
| task5 | Per AC-2.1, implement/screen D1 `B_BLK=4` unless `B_BLK=2` is invalid/spills/regresses; classify residual as fixed-per-transition vs threshold vs compiler-floor | AC-2, AC-2.1, AC-5 | coding | task4 |
| task6 | Promote best D1 variant on a full 5-seed run with a non-overlapping same-session A/B bracket, out-of-noise TRUE PE-active + p50; else record D1 as a compiler no-op | AC-2, AC-5, AC-6, AC-7 | coding | task5 |
| task7 | D2 cross-batch double-buffer (ONLY if D1 is a measured no-op): implement operand ping-pong; gate on out-of-noise TRUE-PE-active drop; ≤2 iterations | AC-3, AC-5, AC-6 | coding | task6 |
| task8 | Terminal bookkeeping: `benchmark.csv` rows, `candidates.jsonl` DAG, `profile/` evidence, per-decision attribution; record terminal "keep `bmm_v2`" if nothing moved | AC-7 | coding | task6, task7 |

## Claude-Codex Deliberation

### Agreements
- The PE-bound reframe is correct and well-supported: the same-session counter dump shows
  DMA-active 1.434 ms (70.46%) strictly below TRUE PE-active 2.0116 ms (98.84%), so DMA is
  hidden and PE is the true bound — the phase-2 "DMA-bound at 100%" note was the jittery coarse
  proxy, not the counter truth.
- D3 (bf16x2 precision) is correctly CLOSED: the measured fp32/bf16 pass ratio is 2.0, so a
  3-product compensated split (3.0 passes) raises PE-active on a PE-bound kernel; the cost gate
  fails even though the offline accuracy sim (4.44e-6) would pass.
- Classic shape specialization has no surface (no edge tiles, tiles maximal at the 512
  PSUM-bank wall, K-split fixed and cost-free), so batch-axis stream depth is the only lever.
- Decisions must rest on TRUE PE-active + p50 + `matmul_instruction_count`, never the coarse
  PE%/DMA% (which swung 1–100% on identical kernels); a compiler-no-op reschedule is rejected.
- Promotion requires full 5-seed rel-L2 PASS, HBM at floor, and a non-overlapping same-session
  A/B bracket — the exact standard by which `bmm_v2` was promoted.

### Resolved Disagreements
- **Is D1 likely a real lever or another compiler no-op?** Codex (first pass): D1 is
  plausible but low-confidence — phase 2 showed the `affine_range` compiler already pipelines
  aggressively, and the outer batch loop is already `affine_range`, so the compiler may already
  cross batch boundaries. Resolution: run D1 explicitly as a **hypothesis test**, not a
  presumed win; add AC-2.1 as the fixed-vs-scaling discriminator; keep "keep `bmm_v2`" as a
  fully acceptable terminal outcome (Lower Bound).
- **Transpose-cost framing.** Codex: the transpose floor is only 0.027 ms of the 0.237 ms
  residual, so D1 recovers dependency/issue gaps, not transpose work. Resolution: Feasibility
  Hints and AC-2.1 now state the mechanism is transition-count reduction, not transpose
  removal.
- **Early-stop after a flat `B_BLK=2`.** Claude (candidate plan): stop the sweep if `B_BLK=2`
  is flat. Codex (second pass): too aggressive — a threshold effect may need the deeper
  `B_BLK=4` stream. Resolution (Claude accepts Codex): AC-2.1 now runs `B_BLK=4` even after a
  flat `B_BLK=2`, hard-stopping only when `B_BLK=2` is invalid/spills/regresses/fails
  correctness.
- **"Byte-identical PE-active" wording.** Codex (second pass): imprecise, since IR/assembly
  diff is unavailable in this harness and PE-active is measured with jitter. Resolution: AC-6
  now says "TRUE PE-active within same-session bracketed noise with invariant counters," and
  counter-identity is the no-op proxy (the same method phase 2 used).
- **Terminal-conclusion overreach.** Codex (second pass): "no remaining schedulable structure"
  overstates. Resolution: softened everywhere to "no Phase-3-tested pure-fp32
  batch-stream-depth schedule recovered the residual; keep `bmm_v2`."
- **`B_BLK` selection.** Codex (first pass): state whether `B_BLK` must divide 16; `B_BLK=8`
  nominal footprint exceeds usable SBUF. Resolution: sweep restricted to `{2, 4}` (both divide
  16); `B_BLK=8` (spill) and `B_BLK=6` (tail) excluded upfront in AC-2.
- **Rollback rules for p50-vs-PE-active disagreement.** Codex (both passes): specify. AC-6 now
  encodes: p50-only win → reject; PE-active-only win → keep as non-promoted evidence,
  repeat-confirm.

### Convergence Status
- Final Status: `converged`
- Rounds executed: 2 (Codex first-pass analysis + 1 convergence review). The second Codex pass
  returned no blocking `DISAGREE`; its three `REQUIRED_CHANGES` (run `B_BLK=4` after a flat
  `B_BLK=2`; replace "byte-identical" wording; soften the terminal conclusion) were all
  accepted and applied above.

## Pending User Decisions

_No unresolved Claude/Codex disagreements remain._ All of Codex's first-pass
`QUESTIONS_FOR_USER` were resolved during Phase 4–5 refinement, and the single second-pass
`UNRESOLVED` item was resolved by Claude accepting Codex's recommendation. The items below are
recorded for transparency with their resolutions; none are `PENDING`.

- DEC-1: Promotion gate — TRUE PE-active only, or TRUE PE-active AND end-to-end p50?
  - Claude Position: require both, TRUE PE-active as primary gate.
  - Codex Position: require both, TRUE PE-active as primary gate (agreed).
  - Tradeoff Summary: p50 alone is too jittery on this op; TRUE PE-active alone can move without a wall-clock win on a DMA-hidden kernel. Requiring both prevents false wins in either direction.
  - Decision Status: RESOLVED — AC-6 requires both.
- DEC-2: Is IR/assembly (byte-level) comparison available to detect compiler no-ops?
  - Claude Position: Not available — this harness measures only on the remote profiler.
  - Codex Position: If available, use it before spending full runs; otherwise use counter-identity.
  - Tradeoff Summary: IR diff would be the strongest no-op tell, but is absent; counter-identity (matmul/Vec/Scl/psum-copy + TRUE PE-active bracket) is the established phase-2 proxy.
  - Decision Status: RESOLVED — AC-6 uses counter-identity as the no-op proxy.
- DEC-3: Spend an iteration on `B_BLK=4` after a flat `B_BLK=2`?
  - Claude Position (revised): Yes — accept Codex's view; a threshold effect may need the deeper stream, and it is the only primary lever.
  - Codex Position: Yes; early-stop after a flat `B_BLK=2` is too aggressive for convergence.
  - Tradeoff Summary: Costs at most one extra iteration within the ≤5 budget; the downside of missing a threshold win outweighs the cost.
  - Decision Status: RESOLVED — AC-2.1 runs `B_BLK=4` unless `B_BLK=2` is invalid/spills/regresses.
- DEC-4: Does the Round-0 re-anchor count against the ≤5-iteration budget?
  - Claude Position: No — it is a measurement re-anchor, not an optimization iteration.
  - Codex Position: Clarify explicitly.
  - Tradeoff Summary: Counting it would waste 1/5 of the budget on a control measurement.
  - Decision Status: RESOLVED — AC-1 and AC-7 state Round-0 is excluded from the ≤5 count.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Phase", "Step", "D1/D2/D3", or similar workflow/plan markers.
- These terms are for plan documentation only, not for the resulting kernel source.
- Use descriptive, domain-appropriate naming in code (e.g. `batch_block`, `pack_transposes`,
  `main_matmul_stream`, `rhs_resident`) instead.
- Follow the existing `runs/bmm_v2.py` idiom: `@nki.jit`, `nl.affine_range`, resident SBUF
  packing, single-pass K=64 `nc_matmul`, 1024-wide coalesced stores; keep the kernel's
  docstring factual about the schedule change and the pure-fp32 / bit-identical-math property.

--- Original Design Draft Start ---

# bmm — Phase 3 draft: regime / shape specialization

**Operator:** `bmm` (NKIBench case 2). Batched matmul `out[b] = lhs[b] @ rhs[b]`,
`b in 0..15`. `lhs (16,4096,64)=(B,M,K)`, `rhs (16,64,4096)=(B,K,N)` fp32 →
`out (16,4096,4096)`. **B=16, M=4096, K=64, N=4096.** Baseline **2.550 ms**.

**Start point:** `runs/bmm_v2.py` = the phase-2 promotion, **1.253x (2.0352 ms)**, full
5-seed L2 PASS (rel-L2 1.83e-7). Two-phase per-batch structure: transpose all 32 lhs
m-tiles of a batch up front into a resident `[64,4096]` pack, then all 256 main matmuls
with 1024-wide coalesced stores. Pure fp32; `bmm_v1` (0.663x) retained as fallback.

---

## 1. The phase-3 question, answered honestly up front

Phase 3 asks: *analyze where time goes across the tensor's structure and specialize only
where the measured win justifies the complexity (tile-size regimes, partition/free splits,
edge tiles).* For bmm the honest answer is that **classic shape specialization has almost
no surface**, and I want to establish that with numbers before spending remote runs, so the
phase does not chase a dead lever:

- **No edge tiles.** Every axis divides cleanly: `M=4096=32·128`, `N=4096=8·512=4·1024`,
  `K=64≤128` (single pass), `B=16`. There is no ragged remainder tile to special-case — the
  usual "specialize the edge" regime split does not exist here.
- **Tiles are already maximal and cannot be widened.** The main matmul is
  `[K=64]×[64,512]→[128,512]`. The moving free dim 512 is the **hard PSUM-bank wall on
  trn2** (one `nc_matmul` writes one bank = 512 fp32 elems/partition; the 2048/4096 width
  is trn3-only — confirmed in the NKI API doc). The stationary free dim 128 fills the PE
  columns. So there is no "bigger tile" regime to switch into.
- **The K=64 partition/free split is fixed and cheap.** K=64 fills only 64 of 128 PE
  partition rows — but on trn2 the matmul cost is `elements_per_partition·100/freq`,
  i.e. **proportional to the dst free dim (512) ONLY**, independent of K and M. A half-full
  contraction axis therefore costs *nothing extra* in time; there is no partition-split
  regime that recovers it. And K cannot be packed: `out[b]` are block-diagonal, so stacking
  two batches onto a 128-row contraction would *sum* two batches' products — numerically
  wrong (already closed in phase 2).

So the phase-3 "structure" to analyze is **not tile shape** — it is the **schedule across
the batch axis**, the one structural dimension phase 2 stopped short of. Phase 2 proved the
lever is *stream depth* (deepening the independent-matmul run cut per-matmul stall
monotonically, M_SUB 8→16→32). Phase 3 extends that same lever past the batch-loop boundary.

## 2. The corrected bottleneck (this reframes what phase 2 recorded)

Phase 2's promotion note and the memory say "bmm_v2 goes DMA-bound at 100% (1074 MB write
floor), pure-fp32 headroom exhausted." **The authoritative same-session counters refute
that.** The coarse `verify.py` DMA% is jittery — across the four D2-bracket runs it read
74% / 100% / 67% / **1%** for the *same kernel*. The counter-level truth
(`profile/bmm_phase2_d2_dump_metrics.txt`, same-session, divided by the metric window):

| signal | bmm_v2 | note |
|---|---|---|
| TRUE PE-active / inf | **2.0116 ms** | tensor_engine_active_time / window |
| tensor_engine_active_time_percent | **98.84%** | PE is the bound |
| dma_active_time / inf | **1.434 ms** (70.46%) | *below* PE-active — DMA is hidden |
| matmul_instruction_count | 8704 | 8192 main + 512 transpose |
| per-matmul PE-active | 0.2311 us | |

⇒ **bmm_v2 is PE-BOUND, not write-bound.** The 1074 MB write DMA (~1.434 ms of active
time) sits *under* the 2.012 ms of PE-active time and is hidden. This matters because it
means there **is** still headroom on the binding engine, and the phase-3 lever is PE
scheduling — not the (already-hidden, at-floor) DMA.

### The theoretical PE floor (trn2 instruction-cost model)

Matmul latency on trn2 = `dst_free_elems · 100 / 240` ns (freq 2.40 GHz), free-dim-only:

| instruction | count | dst free | ns/instr | total |
|---|---|---|---|---|
| main matmul `[128,512]` | 8192 (4096 sites × 2 fp32 passes) | 512 | 213.3 | **1.748 ms** |
| identity transpose `[64,128]` | 512 | 128 | 53.3 | **0.027 ms** |
| **PE floor** | | | | **1.775 ms** |

- **PE floor ⇒ ceiling 2.550 / 1.775 = 1.437x.** This is the hard fp32 wall.
- **DMA write floor** ≈ 1073.7 MB / 368 GB/s ≈ 1.375 ms `<` PE floor 1.775 ms — confirms
  fp32 bmm is PE-bound end-to-end (DMA can never become the true binder at fp32).
- **bmm_v2 at 2.0116 ms is 13.3% above the PE floor** (`2.0116/1.775`). That ~0.237 ms
  excess is residual **per-instruction schedule bubble** — the phase-3 target. Closing it
  fully → ~1.79 ms → ~1.42x; closing half → ~1.89 ms → ~1.35x.

### Where the residual bubble most likely lives: the batch boundaries

bmm_v2 is *two-phase per batch*: for each of 16 batches it (1) loads `rhs[b]`, (2)
transposes all 32 m-tiles into the pack (Pass 1, transpose→copy chain), then (3) streams
256 main matmuls (Pass 2). Phase 2 showed deepening Pass 2 cuts the stall — but Pass 2 is
re-primed **16 times**, once per batch, and each batch head re-enters the serial
transpose→copy Pass 1 before its matmul stream can refill the PE. The 16 batch-boundary
transitions (Pass-1 transpose burst not yet overlapped with the *previous* batch's Pass-2
tail, plus the `rhs[b]` load and pack rebuild) are the natural place for the 13% residual.
This is the exact structural gap the phase-2 stream-deepening lever did not reach — it
deepened *within* a batch but left the batch boundary serial. **Round 0 must confirm this
localization before I build anything.**

## 3. Round 0 — measurements before any code change (near-zero remote risk)

Re-use the `runs/dump_metrics.py` idiom (reads TRUE `tensor_engine_active_time_ns` +
`matmul_instruction_count`, not the jittery PE%/DMA% proxy). All same-session vs a fresh
bmm_v2 anchor.

1. **Re-anchor bmm_v2 counters** — confirm TRUE PE-active ≈ 2.012 ms, matmul_instr 8704,
   and record the DMA-active *time* (not %) to nail down that PE-active `>` DMA-active
   (PE-bound). This is the fact the whole phase rests on; verify it fresh.
2. **Per-batch attribution proxy** — there is no per-region timeline from the profiler, so
   attribute structurally: the PE floor arithmetic above already isolates the residual to
   0.237 ms / 8704 instr = 0.0272 us/instr of average bubble. Cross-check by comparing
   bmm_v2's per-matmul 0.2311 us against the pure main-matmul floor 0.2133 us
   (213.3 ns) — the **0.0178 us/instr gap on the main matmuls alone** is the schedulable
   residual; the transpose adds the rest. If a candidate drives per-matmul toward 0.2133,
   it is closing the real bubble.
3. **(record-only) confirm D3 stays closed** — no new probe needed; phase-2 measured
   fp32/bf16 pass ratio = 2.0, so a 3-product bf16x2 main matmul costs 3.0 passes `>` 2.0
   and *raises* PE-active on a PE-bound kernel (would regress, exactly like swiglu's 2-pass
   all-3 split at 0.409x). D3 remains SKIPPED; the PE floor above is the fp32 wall and the
   only way past it (precision) is closed. State this so the phase does not relitigate it.

## 4. Optimization directions, ranked by expected benefit × confidence

### D1 — multi-batch blocking (PRIMARY; extends the proven phase-2 lever across batches)
Phase 2's winning lever was *stream depth*. bmm_v2 blocks one batch at a time (32 m-tiles).
D1 blocks **`B_BLK` batches together**: transpose the `B_BLK·32` m-tiles of `B_BLK` batches
into one resident pack, load `B_BLK` rhs tiles resident, then stream `B_BLK·256` main
matmuls with no transpose interleaved — so the Pass-1/Pass-2 transition happens `16/B_BLK`
times instead of 16, amortizing the batch-boundary bubble over a deeper stream.
- **SBUF budget (the binding constraint):** per batch the pack is `64×4096×4B = 16 KB/part`
  and rhs is `64×4096×4B = 16 KB/part` = 32 KB/part/batch. trn2 usable ≈ 208 KB/part ⇒
  `B_BLK ≤ 6` comfortably (192 KB), `B_BLK=4` is the safe sweet spot (128 KB, room for
  double-buffering the output SBUF tiles). Sweep `B_BLK ∈ {2, 4}` (and 8 only if SBUF
  fits without spill — watch HBMrd staying at the 34 MB floor).
- **Expected:** if the 13% residual is the batch boundary, this recovers most of it →
  ~1.85–1.90 ms → **~1.34–1.38x**. Pure fp32, bit-identical math (same 8704 instr, pure
  reschedule — like the M-block sweep).
- **Risk / kill-criterion:** phase 2 proved the `affine_range` compiler *already*
  software-pipelines aggressively and flattened every multi-bank/issue-order reschedule to
  a byte-identical no-op. The compiler may already pipeline across the batch `affine_range`,
  making D1 a no-op too. **Screen with `--fast` + `dump_metrics` first**; promote only if
  TRUE PE-active *drops* out-of-noise AND HBM stays at the 34/1074 MB floor (no spill from
  the larger resident footprint). ≤3 iterations (B_BLK sweep).

### D2 — cross-batch double-buffering of the resident operands (ALT; if D1 is a no-op)
If D1's static blocking is absorbed by the compiler, try the explicit ping-pong from the
`bc877398` / `3c7e053b` precedents: pre-allocate **two** `(rhs, lhs_t_pack)` buffer sets,
prefetch batch 0, then while batch `b`'s 256 matmuls stream, DMA-load `rhs[b+1]` and
transpose `lhs[b+1]`'s pack into the alternate buffer. This overlaps the Pass-1 transpose
burst and rhs load of batch `b+1` with batch `b`'s Pass-2 matmul stream — directly attacking
the batch-boundary serialization that D1 attacks statically.
- **Cost:** doubles the resident footprint to 64 KB/part (still fits, `< 208`).
- **Expected:** same target as D1 (~1.34–1.38x) via an explicit rather than
  compiler-inferred overlap. Prefer whichever of D1/D2 the counters show actually moves
  TRUE PE-active; they are two routes to the same batch-boundary bubble, so **do not spend
  iterations on both if the first lands.** ≤2 iterations.

### D3 — precision (bf16x2): CLOSED, record-only
The only lever that touches the 1.775 ms PE floor is precision, and it is closed: measured
fp32/bf16 pass ratio 2.0 ⇒ a compensated 3-product bf16x2 costs 3.0 passes `> 2.0`, *raising*
PE-active on a PE-bound kernel. bmm is a single raw matmul (no swiglu-style "split only the
cheap GEMM, keep the others fp32" rescue). SKIPPED; do not build. (Offline rel-L2 4.44e-6
would pass the accuracy gate, but the cost gate fails and both are required.)

### Closed / not-pursued (record-only, do not spend iterations)
- **Wider matmul tile** (moving free > 512): trn2 PSUM-bank wall, infeasible (trn3-only).
- **K-packing two batches onto 128 partitions:** numerically wrong (block-diagonal → sums
  batches). Closed in phase 2.
- **off-PE `load_transpose2d` (phase-2 D2):** counter-verified no-op (compiles to the same
  on-engine transpose, matmul_instr 8704 unchanged). Do not re-probe.
- **DMA store-burst / ping-pong output / bf16 output:** DMA is hidden (70% active `<` PE
  99%) and at the write floor; output dtype is the final result (2e-5 gate bans bf16 out).
  All dead — the bound is PE, not DMA.
- **store-direct-from-PSUM to drop the 4224 PSUM→SBUF copies:** architecturally impossible
  (no PSUM→HBM DMA path on trn2; HBM stores must source from SBUF). And the copies are
  already hidden behind matmul. Closed.

## 5. Method & discipline (per direction, ≤5 iterations total)
- **Noise anchor:** re-run bmm_v2 same-session as the control before each comparison
  (siblings saw ~0.02–0.5% jitter on this op; treat a ~1.5–2% band as noise). The coarse
  DMA% is NOT a decision metric here (it swung 1–100% on identical kernels) — decide on
  **TRUE PE-active (ms) + p50 latency + matmul_instruction_count**, not PE%/DMA%.
- **Screen then confirm:** `--fast` (seed 42) + `dump_metrics` to screen a B_BLK/ping-pong
  variant; promote only on a **full 5-seed** run (drop `--fast`) with an out-of-band p50
  win. A candidate whose TRUE PE-active is byte-identical to bmm_v2 is a compiler no-op →
  reject immediately (like the phase-2 multi-bank family), do not chase its latency noise.
- **Correctness invariant:** every promoted candidate is a pure reschedule (same 8704 instr,
  single-pass K=64, transpose-before-use is exact) ⇒ rel-L2 must stay 1.83e-7; any drift
  means an indexing bug. Keep bmm_v2 (and bmm_v1) as fallbacks.
- **Bookkeeping:** append each perf change to `benchmark.csv`; each candidate to
  `candidates.jsonl` with parent links (DAG); evidence under `profile/`. Kernels in `runs/`;
  never edit the baseline/reference.

## 6. Expected trajectory
`bmm_v2 1.253x → D1 multi-batch blocking (or D2 cross-batch double-buffer) ~1.34–1.38x` if
the batch-boundary bubble is real and not already compiler-pipelined; **hard fp32 ceiling
1.437x** (the 1.775 ms PE floor). Realistic promote target **~1.30–1.35x**. If both D1 and
D2 come back byte-identical (compiler already crosses the batch loop), the honest phase-3
conclusion is that bmm_v2 is within 13% of the fp32 PE floor with no remaining schedulable
structure — record that as the terminal result and keep bmm_v2. The whole phase is one
lever (batch-axis stream depth) tested two ways, gated hard on TRUE PE-active moving.

--- Original Design Draft End ---
