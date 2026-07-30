# matmul Phase 3 — Regime/Shape Specialization (fp32 PE-floor confirmation + safe micro-wins)

## Goal Description

Starting from the best correct kernel `runs/matmul_v2_b4.py` (M-blocked B=4, 1.017x
/ 13.3517 ms, PE=100%, MFU=49%), determine how much latency headroom actually
remains and capture only wins that measurably clear profiler noise. The Phase-3
analysis (profile/matmul_phase3_analysis.txt) finds the kernel is already at ~98% of
the fp32 PE floor: the trn2 PE array is bf16-native and fp32 matmul runs at ~2 passes
(half rate), so the true fp32 PE floor is ~13.1 ms and MFU~49% is a structural fp32
ceiling (the 2e-5 gate forbids bf16/tf32). This is a single fixed shape with all
tiles full, so there is no shape regime / edge tile to specialize. The realistic
goal is therefore to (a) **validate the fp32-floor claim with an on-device
calibration** rather than assert it, (b) attempt a few sub-1% low-risk levers and
keep any that beat B=4 above a noise threshold, and (c) never regress correctness.

Framing note (per Codex): the ~1.036x "ceiling" is a **score-vs-baseline** figure
(baseline/fp32-floor = 13.5785/13.1); the **incremental** headroom over B=4 is only
~0.8–1.9%, and most is unreachable. Phase 3 succeeds if it either finds a measured
win or documents the floor with evidence — both are acceptable exits.

## Acceptance Criteria

- AC-1: Correctness is never regressed — the promoted candidate passes relative-L2
  `< 2e-5` (fp32) on all five seeds `[0,21,42,63,84]` via full (non-`--fast`)
  `verify.py`.
  - Positive Tests: full `verify.py` reports `correct: 1/1`, all per-seed
    `l2_norm_passed` true.
  - Negative Tests: any candidate failing a seed, or downcasting an operand to
    bf16/tf32, is rejected; the promoted kernel stays whichever correct candidate
    is fastest (B=4 if nothing beats it).

- AC-2: The Phase-3 result is **≥ the Phase-2 best** on a stable full-run p50. A new
  candidate is promoted over B=4 only if it beats B=4 by a margin above profiler
  noise (target **≥ 0.3–0.5%**); otherwise B=4 remains the promoted kernel.
  - **Contemporaneous control (per Codex):** B=4 (`matmul_v2_b4.py`) must be re-run
    in the SAME measurement session as the candidates — do not compare against the
    historical 13.3517 ms alone. Take repeated/interleaved B=4 full runs, record the
    p50 and the run-to-run spread (the noise band), and judge every candidate against
    that fresh control.
  - Positive Tests: promoted candidate's full-run p50 ≥ the contemporaneous B=4 p50;
    if a new candidate is promoted, its improvement over the B=4 control exceeds the
    measured noise band.
  - Negative Tests: a candidate whose apparent win is within the measured noise band
    (e.g. < 0.3%) is NOT promoted; a `--fast`-only improvement that does not survive
    the full run is NOT promoted (Phase-2 lesson).

- AC-3: The fp32 PE-floor conclusion is **backed by on-device evidence**, not just
  the cost model. A small calibration establishes the fp32 vs bf16/tf32 `nc_matmul`
  rate on this machine for the tile shape/count in use.
  - Positive Tests: a recorded calibration (or a documented equivalent from the
    existing profiler metrics) shows fp32 matmul time ≈ 2× a bf16/tf32 matmul of the
    same shape, consistent with the ~13.1 ms fp32 floor and the 49% MFU.
  - Negative Tests: if calibration shows fp32 is NOT ~2× (i.e. the floor claim is
    wrong), the "near-optimal" conclusion is revised and the freed headroom is
    pursued — the floor claim must not be asserted without this check.

- AC-4: Every candidate (kept or rejected) is recorded with full profiler evidence
  — latency, speedup, MFU, PE-active, DMA-active, HBMrd, HBMwr — in `benchmark.csv`
  and `candidates.jsonl` (parent DAG rooted at `matmul_v2_b4`), with per-direction
  digests under `profile/`.
  - Positive Tests: each perf-relevant candidate has a benchmark.csv row + a
    candidates.jsonl node with a complete `metrics` block; profile/ holds the digest.
  - Negative Tests: promoting or rejecting a candidate without its full metric row
    is incomplete (the Phase-2 AC-3 gap must not recur).

- AC-5: The kernel stays a single `@nki.jit def kernel(v1, v2)` with the fixed tiled
  I/O contract, fp32 throughout, no harness/reference/baseline edits; each direction
  ends in an explicit keep/revise/reject decision within ≤5 iterations.
  - Positive Tests: adapter runs each candidate end-to-end; the evidence trail shows
    per-direction verdicts.
  - Negative Tests: signature/shape change, NKIBench edits, or an abandoned
    direction with no recorded verdict violate the contract.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A correct `kernel(v1, v2)` that is the empirical fastest among: the current B=4;
a variant that steers the PSUM→SBUF eviction / store copy to an idle engine
(ScalarE/VectorE) to remove any eviction bubble at PE=100%; and a re-measured B=8 /
B=16 (both divide 32) under the corrected trn2 SBUF budget. Plus an on-device
calibration validating the fp32-floor claim, and the full analysis documented. All
correctness invariants from Phase 1/2 hold.

### Lower Bound (Minimum Acceptable Scope)
Keep `matmul_v2_b4.py` as the promoted result (≥ 1.017x, all seeds pass), with the
fp32-floor analysis and calibration recorded as the Phase-3 deliverable — an honest
"already near the fp32 floor; further gain needs a precision change the gate forbids"
conclusion. No new kernel is required if none beats B=4 above noise.

### Allowed Choices
- Can use: engine steering for the eviction/store copy (`nisa.tensor_copy(engine=)`);
  B ∈ {4,8,16} (divisors of 32); a bounded DVE/off-PE transpose experiment; a small
  calibration kernel/measurement. All fp32.
- Cannot use: bf16/tf32/fp8 on the numeric path; `dma_transpose` for the fp32
  transpose (needs 2-byte dtype — ineligible); moving free > 512; masking (no
  remainders); NKIBench edits; changing the kernel signature or the fixed input
  layout (the caller contract is fixed — no pretransposed/packed lhs).

> **Note on Determinism**: The math (fp32 GEMM), the fixed single shape, and the
> fixed I/O contract leave almost no specialization surface. The genuine variables
> are the eviction-copy engine, the M-block factor B, and whether an off-PE transpose
> overlaps. Upper and lower bounds nearly coincide — the expected outcome is "confirm
> B=4 at the fp32 floor," with any micro-win a bonus.

## Feasibility Hints and Suggestions

> **Note**: Reference only — conceptual, not prescriptive.

### Conceptual Approach

Validate before optimizing (Codex): the whole phase hinges on the fp32-floor claim,
so establish it on-device first. A minimal calibration: time a handful of
`nc_matmul`s of the exact tile shape (`stat [128,128]`, `moving [128,512]`) in fp32
vs bf16 (correctness aside — this is a timing probe, not a scored kernel), and a
transpose-only probe. If fp32 ≈ 2× bf16, the ~13.1 ms floor and 49% MFU are
confirmed and B=4 is near-optimal. If not, revisit.

Then the two low-risk levers on B=4:
- D1 (eviction/store engine): at PE=100%, steer the `nl.copy`/store off the Tensor
  Engine's path (`nisa.tensor_copy(engine=nisa.scalar_engine or vector_engine)`) so
  the PSUM→SBUF drain overlaps instead of serializing. Only helps if eviction is on
  the critical path — verify in the profiler, keep only if > 1.017x above noise.
- D2 (B re-measure): re-run B=8 and B=16 full-run under the corrected SBUF budget
  (trn2 = 224 KB, 208 KB usable — not the 192 KB used when B=8 was first rejected).
  Note SBUF must count ALL live storage (lhsT workspace, rhs tile, output staging,
  temporaries), and B=8/16 may still be PSUM-bank/lifetime limited even if SBUF fits;
  lower HBMrd will NOT help if DMA is already hidden under PE. Try B=8 before B=16.

Transpose (D3): bounded experiment only — an off-PE (DVE) transpose of the 128×128
lhs tiles targets PE-cycle removal, but the transpose is 0.5% of runtime and DVE is
slower per tile, so the best case is tiny. `dma_transpose` is ineligible (fp32 = 4B).
The transpose is structurally unavoidable (lhs has k on the free axis). Time-box it.

### Relevant References
- `runs/matmul_v2_b4.py` — promoted Phase-2 kernel to evolve; `runs/matmul_v2_b8.py`
  exists (rejected, re-measure).
- `profile/matmul_phase3_analysis.txt` — the fp32-floor derivation this plan rests on.
- `.claude/skills/kernel-cost-analysis` — cost model + trn2 constants
  (Matmul cost, PE freq 240, SBUF 224 KB, PSUM 8 banks).
- `.claude/skills/kernel-optimization-kb` — `bc877398` (copy engine steering),
  transpose-microkernel selector (dma_transpose needs 2-byte dtype).
- `verify.py` — scoring + `summary_metrics` digest.

## Dependencies and Sequence

### Milestones
1. Validate the floor:
   - Phase A: on-device fp32-vs-bf16 (+ transpose-only) `nc_matmul` calibration;
     record. If it refutes the ~2× claim, re-open headroom.
2. Safe micro-wins (only if calibration confirms near-floor):
   - Step 1: D1 eviction/store engine steering on B=4; `--fast` then full; keep if
     > 1.017x above noise.
   - Step 2: D2 re-measure B=8, then B=16, full-run; keep fastest B.
   - Step 3: D3 time-boxed off-PE transpose experiment; almost certainly reject.
3. Promote + document:
   - Step 1: full 5-seed on the fastest correct candidate; confirm ≥ 1.017x.
   - Step 2: record all candidates + the floor analysis; if none beats B=4, promote
     B=4 with the fp32-floor evidence as the Phase-3 result.

Dependencies: Milestone 2 depends on 1 (don't chase micro-wins if the floor claim is
wrong). D1/D2/D3 are independent; each ends with a recorded verdict.

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task0 | Establish the contemporaneous B=4 control: re-run `matmul_v2_b4.py` full 5-seed several times in this session, record p50 + run-to-run spread (the noise band) that AC-2 judges candidates against | AC-2, AC-4 | coding | - |
| task1 | On-device calibration: time fp32 vs bf16 `nc_matmul` of the [128,128]x[128,512] tile (+ transpose-only probe) to validate the ~2x fp32 rate / ~13.1ms floor. Probe must write a live output (not be optimized away), use the same `--logical-nc-config=1`/single-core, comparable tile shape+count, a stated warmup/exclusion policy, and enough repetitions to support the "≈2x" claim; record in profile/ | AC-3 | coding | - |
| task2 | D1: variant of B=4 that steers the PSUM->SBUF eviction/store copy to an idle engine — try BOTH `scalar_engine` and `vector_engine` (one, then the other) via `nisa.tensor_copy(engine=)`; score --fast then full 5-seed; keep only if it beats the task0 B=4 control above the noise band | AC-2, AC-4 | coding | task0, task1 |
| task3 | D2: re-measure B=8 then B=16 full-run under corrected SBUF budget (count ALL live storage, not nominal); record metrics; a compile failure / SBUF-or-PSUM overflow is a VALID rejected outcome (record it); keep fastest B that beats the control above noise | AC-2, AC-4, AC-5 | coding | task0, task1 |
| task4 | D3 (time-boxed): bounded off-PE/DVE transpose experiment; measure PE-cycle change; almost certainly reject with evidence | AC-2, AC-4, AC-5 | coding | task1 |
| task5 | Full 5-seed on the fastest correct candidate; promote (parent=matmul_v2_b4); if none beats the B=4 control above noise, promote B=4 with fp32-floor evidence | AC-1, AC-2, AC-4 | coding | task2, task3, task4 |
| task6 | (Optional) Codex review: sanity-check the calibration interpretation (task1) + the promoted kernel's correctness | AC-1, AC-3 | analyze | task1, task5 |

## Claude-Codex Deliberation

### Agreements
- The kernel is near the fp32 PE floor; fp32 runs ~2x on the bf16-native array, so
  MFU~49% is a structural ceiling, not unused capacity — but this must be VALIDATED
  on-device (calibration), not just asserted.
- Real incremental headroom over B=4 is ~0.8–1.9%; the ~1.036x figure is score-vs-
  baseline, not additional gain.
- fp32 forbids bf16/tf32; `dma_transpose` is ineligible for fp32; the lhs transpose
  is structurally unavoidable; wider tiles are at hardware caps; no shape regime
  exists (single full shape).
- Promote only above a noise threshold; full 5-seed, not `--fast`, decides.

### Resolved Disagreements
- Transpose offload: draft called it "reject"; Codex wants a **bounded experiment**
  (it directly removes PE cycles, even if the upside is tiny). Resolution: keep it as
  a time-boxed experiment (task4), expected reject-with-evidence — not pre-rejected.
- Floor asserted vs measured: Codex flagged the floor was model-only. Resolution:
  added AC-3 + task1 (on-device fp32/bf16 calibration) so the claim is evidence-based.
- Ceiling wording: corrected ~1.036x to "score-vs-baseline ceiling; ~0.8–1.9%
  incremental," to avoid overstating headroom.

### Convergence Round 1 (second Codex pass, reviewing candidate plan v1)
Codex: "reasonable and mostly complete… correctly treats Phase 3 as validation plus
bounded micro-optimization." No substantive disagreement. Two `REQUIRED_CHANGES`,
both applied:
1. **Contemporaneous B=4 control / noise protocol** — AC-2's "above noise" needs B=4
   re-run in the same session (not vs the stale 13.3517 ms). → added `task0`
   (repeated/interleaved B=4 runs → p50 + noise band) and rewrote AC-2 to judge
   candidates against that fresh control.
2. **Tighten task1 calibration** — probe must write a live output (not be optimized
   away), use the same single-core/`--logical-nc-config=1`, comparable tile
   shape+count, a warmup/exclusion policy, and enough reps. → task1 updated.
`OPTIONAL_IMPROVEMENTS` folded in: D1 tries both `scalar_engine` and `vector_engine`
(task2); task6 now depends on task1+task5 (so it can review the calibration + the
promoted kernel); D2 records a compile/SBUF/PSUM overflow as a valid rejected
outcome (task3).

Convergence matrix (round 1):
| Topic | Claude (v1) | Codex | Resolution |
|---|---|---|---|
| noise baseline | vs historical 13.3517 ms | needs same-session control | resolved (task0 + AC-2 rewrite) |
| calibration rigor | "time fp32 vs bf16" | must not be optimized away; same flags; reps | resolved (task1 tightened) |
| D1 engine choice | "ScalarE/VectorE" | specify both, one then other | resolved (task2) |
| B=8/16 infeasibility | re-measure | record compile/SBUF fail as valid reject | resolved (task3) |
| PE=100% semantic | PENDING (low impact) | agrees mitigation sufficient | deferred (measured directly by task2/3) |

Round 2 not required: no `REQUIRED_CHANGES` remain; Codex's only UNRESOLVED (the
PE=100% metric semantic) is low-impact and mitigated by measuring candidate effects
directly.

### Convergence Status
- Final Status: `converged` (Codex first-pass + one convergence round; first-pass
  CORE_RISKS → wording fixes + AC-3 calibration, MISSING_REQUIREMENTS → noise
  threshold (AC-2) + full metric rows (AC-4), ALTERNATIVE_DIRECTIONS →
  calibration-first + bounded transpose + B=8-before-B=16; round-1 REQUIRED_CHANGES
  all applied. Only low-impact unresolved item is the PE-occupancy metric semantic,
  mitigated by direct measurement.)

## Pending User Decisions

Codex's `QUESTIONS_FOR_USER` are resolved from the harness; recorded for traceability:

- DEC-1: Input layout fixed, or can the caller provide pretransposed/packed lhs?
  - Resolution: **Fixed.** The NKIBench tiled contract (`v1 (32,128,40,128)`) is set
    by the benchmark; the kernel cannot change how inputs arrive. So the lhs
    transpose stays unavoidable. Status: `Resolved — layout fixed`.
- DEC-2: Single-core mandatory, or only for this phase?
  - Resolution: **Mandatory** — NKIBench scores single-core (`--logical-nc-config=1`).
    Status: `Resolved — single-core scoring`.
- DEC-3: Is ~1.036x an absolute score ceiling vs baseline (not incremental)?
  - Resolution: **Yes** — it is baseline/fp32-floor. Incremental over B=4 is ~1-2%.
    Corrected in the plan. Status: `Resolved`.
- DEC-4: Does profiler `PE=100%` mean wall-clock or active-cycle occupancy?
  - Claude Position: Unresolved from docs — the `summary_metrics` field semantics
    aren't fully specified. Status: `PENDING (low impact)`. Mitigation: D1 tests for
    recoverable eviction bubbles regardless; if PE=100% is active-cycle (hiding issue
    bubbles), D1/D2 could still recover small gains, which is exactly what task2/task3
    measure. Does not block the plan.

## Implementation Notes

### Code Style Requirements
- No plan-terminology (`AC-`, `Milestone`, `Step`, `Phase`, `D1/D2`, task IDs) in
  kernel code/comments; use domain names.
- fp32 on the numeric path; single `@nki.jit def kernel(v1, v2)`; comment the
  eviction-copy engine choice and the store axis mapping.
- Full 5-seed before promoting; record complete metric rows (no AC-3-style gaps).

--- Original Design Draft Start ---

# matmul Phase 3 — regime/shape specialization draft

## Starting point (best correct kernel)

`runs/matmul_v2_b4.py` (Phase 2): M-blocked B=4 fp32 GEMM. **1.017x (13.3517 ms)**,
all 5 seeds pass. Profiler: **PE=100%, MFU=49%, DMA=31%, HBMrd=2097 MB**. Fully
PE-bound.

## The central Phase-3 finding: the kernel is at the fp32 PE floor

Phase 3's premise is "specialize where the measured win justifies the complexity."
The analysis (profile/matmul_phase3_analysis.txt, grounded in the cost model +
knowledgebase) shows there is **very little to specialize**, because we are already
at the hardware floor for this precision:

- The trn2 PE array is **bf16-native**; **fp32 matmul runs at ~2 passes** (half rate).
  The naive cost-model floor (6.62 ms) is a *bf16-equivalent*; the true **fp32 PE
  floor ≈ 13.1 ms**. B=4 at 13.35 ms is **~98% of that floor**.
- MFU is measured vs the bf16 peak, so a correct fp32 GEMM is **capped near ~50% MFU
  by construction**. Our 49% is essentially the ceiling — not an inefficiency to fix.
- The 2e-5 L2 gate forbids bf16/tf32, so the fp32 floor is binding.
- This is a **single fixed shape with all tiles full** (32·128=4096, 40·128=5120,
  24·512=12288 — no remainders). There is **no shape regime / edge tile to
  specialize** — every tile is identical.

Absolute conceivable ceiling (floor, zero transpose, zero overhead) ≈ **1.036x**;
current 1.017x → **< 3% total headroom, most unreachable.**

## Directions (ranked by measured-win / complexity)

Per the user's decision, run small **low-risk** attempts, keep only measured wins,
and otherwise document the floor. No direction below is expected to exceed a few %.

### D1 — PSUM→SBUF eviction / store engine choice  [LOW value, low risk]
At PE=100%, the PSUM→SBUF copies (`nl.copy`) and stores may now sit on/near the
critical path. Try steering the copy to an idle engine (ScalarE or VectorE via
`nisa.tensor_copy(engine=...)`) so it doesn't contend with the Tensor Engine or
serialize eviction. Precedent: `bc877398` (move a copy from VectorE to ScalarE to
rebalance). Expected: 0–1%. Keep only if it beats 1.017x full-run.

### D2 — re-confirm B with the CORRECTED SBUF budget  [LOW value, low risk]
Phase 2 rejected B=8 partly on a mis-stated SBUF budget (used trn1's 192 KB; trn2 is
actually **224 KB / 208 KB usable**). B=8's full-run regression was likely PSUM-bank
exhaustion (all 8 banks) + scheduling, not an SBUF spill — but re-measure B=8 (and
B=16, which also divides 32) once with the correct understanding to confirm B=4 is
truly best. Expected: confirm B=4, or a marginal shift. Cheap to check.

### D3 — transpose scheduling  [VERY LOW value]
The 1280 lhs-transpose matmuls are 0.5% of runtime. `dma_transpose` is **ineligible**
(needs 2-byte dtype; fp32 is 4 bytes). DVE transpose of 128×128 is slower than the
current TensorE transpose. The transpose is **structurally unavoidable** (lhs has k
on the free axis; nc_matmul needs k on partition). So there is nothing worth doing
here beyond confirming the transpose already overlaps. Effectively **reject**.

### Rejected outright
- bf16/tf32 downcast — breaks 2e-5 (this is *why* the floor is 13.1 ms).
- Wider tiles — already at hardware caps (stationary 128, moving 512, contraction 128).
- Shape-regime split / edge-tile specialization — no regimes exist (single full shape).
- bf16x3 / split-fp32 emulation — the user chose the safe path; 3 bf16 passes would
  likely be slower than fp32's 2 passes and hitting 2e-5 is uncertain. Out of scope.

## Plan of attack (≤5 iters per direction)

1. D1: try the eviction/store engine tweak on B=4; `--fast` read, then full 5-seed if
   promising. Keep only if > 1.017x.
2. D2: re-measure B=8 and B=16 full-run for a clean comparison; keep the fastest B.
3. Record every candidate (kept/rejected) in benchmark.csv + candidates.jsonl
   (parent = matmul_v2_b4) + profile/. Never regress correctness (all 5 seeds).
4. Whatever the outcome, the Phase-3 deliverable includes the fp32-floor analysis so
   the near-optimality is documented, not just asserted.

## Target

Realistic: **hold ≥ 1.017x**, capture any measured win up to the ~1.036x ceiling.
Honest exit: if no candidate beats B=4, promote B=4 and report the fp32-floor
evidence explaining why further gain needs a precision change the gate forbids.

## Correctness / evidence contract (unchanged)
- fp32 throughout; all 5 seeds `[0,21,42,63,84]` pass relative-L2 `< 2e-5`.
- Single `@nki.jit def kernel(v1,v2)`; candidates in `runs/`; never edit baseline/reference.
- Parent DAG in `candidates.jsonl`; per-direction profiling under `profile/`; full
  5-seed (not just --fast) before promoting (Phase-2 lesson: --fast can mis-rank).

--- Original Design Draft End ---
