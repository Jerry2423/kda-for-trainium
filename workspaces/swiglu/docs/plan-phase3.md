# SwiGLU Phase 3 — bf16x2 Floor-Breaker (`swiglu_v2`) + Conditional M-Block Regime (`swiglu_v3_mblock`)

## Goal Description

Break the fp32 Tensor-Engine emulation floor that pins the only correct SwiGLU kernel
(`runs/swiglu_v1.py`) at **0.939×** (2.2079 ms, *slower* than the 2.0742 ms baseline),
and ship a kernel that beats the baseline.

The fused SwiGLU op is three chained GEMMs plus a SiLU gate, all fp32, at
M=4096, K=1024, N=3072:

```
up   = x @ w_up            # (M, N)
gate = x @ w_gate          # (M, N)
h    = SiLU(gate) * up     # elementwise on (M, N)
out  = h @ w_down          # (M, K)
```

`swiglu_v1` is measured **PE-bound at the trn2 fp32-emulation floor** (MFU=44%, PE=95%,
Vec=7%, DMA=49%, HBMrd=607MB, HBMwr=17MB, rel-L2=6.36e-7). The three GEMMs are ~95% of
PE work (§2.1 of the draft), and the trn2 Tensor Engine is bf16-native: fp32 matmul is
emulated at ~44% MFU. The single lever that moves a 95%-GEMM, fp32-floored kernel is to
stop paying the fp32 emulation tax.

**Part A (must-do, the floor-breaker):** build `runs/swiglu_v2.py` = `swiglu_v1` + a
compensated **bf16x2** (two-limb) 3-product split on **all three GEMMs**. Each fp32
operand becomes two bf16 limbs; three bf16 products accumulate in fp32 PSUM
(`a@b ≈ a_hi@b_hi + a_hi@b_lo + a_lo@b_hi`, dropping the negligible `lo⊗lo` term).
Transposes stay **exact fp32** (limbs are split *after* the transpose); the fused
`nl.silu` + `nl.multiply` stay fp32. An offline numpy sim (`runs/offline_bf16_split_sim.py`,
zero remote spend) that draws **real distinct per-seed inputs** proves worst-case
rel-L2 = **7.72e-6** (2.6× under the 2e-5 gate), essentially seed-independent
(7.706e-6…7.722e-6). Projected latency **~1.13–1.20×**.

**Part B (conditional, measured not assumed):** build `runs/swiglu_v3_mblock.py` — an
M-tile-block regime (process B M-tiles per weight stream so each fp32 weight loads once
and is reused B×, cutting weight-DMA volume ~B×) — **only if** `swiglu_v2`'s re-profile
shows weight-DMA exposed off "hidden." bf16x2 does not change HBM traffic, so if
`swiglu_v2` still hides DMA with comfortable margin, M-blocking is ≤1.05× polish not worth
the complexity. If not triggered, record a documented stop-at-v2 decision.

Keep `swiglu_v1` unchanged as the fp32 correctness fallback; never regress below it.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. On-device runs use:
`python3 ../../verify.py --op swiglu --candidate runs/<file>.py` (add `--fast` for quick iteration; drop it for the promoting run).

- AC-1: `swiglu_v2` clears the correctness gate on-device.
  - Positive Tests (expected to PASS):
    - Remote full run reports `l2_norm_passed=True` with rel-L2 < 2e-5 (expected ~7.7e-6).
    - The re-run offline sim (`runs/offline_bf16_split_sim.py`) shows all-3-bf16x2 worst
      rel-L2 ≤ 1.5e-5 across the distinct-seed set [42,0,21,63,84] (it currently reports 7.72e-6).
  - Negative Tests (expected to FAIL / be rejected):
    - Any seed reports rel-L2 ≥ 2e-5 on-device → do NOT promote; walk the fallback ladder
      (AC-6 / §5) and record the failing measurement.
    - Promoting a kernel whose on-device run reports `l2_norm_passed=False`.

- AC-2: `swiglu_v2` beats the baseline on latency.
  - Positive Tests (expected to PASS):
    - Remote full-run latency < 2.0742 ms (speedup > 1.0×); target ~1.73–1.84 ms
      (~1.13–1.20×).
    - MFU rises clearly above `swiglu_v1`'s 44%.
  - Negative Tests (expected to FAIL / be rejected):
    - Latency ≥ 2.0742 ms (speedup ≤ 1.0×) → `swiglu_v2` is NOT promotable over baseline.
      This is a **documented failed candidate**, still recorded in `benchmark.csv` /
      `candidates.jsonl` with its profile; `swiglu_v1` remains the fp32 fallback and the
      cause is diagnosed (chunk-size / limb-rebuild placement per AC-4, then the ladder).
    - Silently discarding a correct-but-slower `swiglu_v2` without recording its evidence.

- AC-3: `swiglu_v2` implements the bf16x2 split structurally as specified.
  - Positive Tests (expected to PASS):
    - All limbs are `nl.bfloat16`; PSUM accumulation, the fused `nl.silu`, the
      `nl.multiply`, and **both** identity-matmul transposes stay fp32.
    - `x` is transposed once into fp32 `xT` (8 sub-tiles), then split into `xT_hi`/`xT_lo`
      **once** and the same limbs are shared by BOTH the up and gate GEMMs (not recomputed
      per GEMM). `h` is transposed once into fp32 `hT` (24 sub-tiles), then split into
      `hT_hi`/`hT_lo` once.
    - Limb construction uses the round-to-nearest-even cast `nl.copy(dtype=nl.bfloat16)`
      for the hi limb, `nisa.tensor_tensor(op=nl.subtract)` for the exact fp32 residual,
      and `nl.copy(dtype=nl.bfloat16)` for the lo limb — the exact idiom the offline sim's
      `to_bf16_rne` models and that `add_rmsnorm_matmul_v3_bf16_split.py` uses.
    - Each GEMM issues exactly the 3 products `hi@hi + hi@lo + lo@hi` into a fp32 PSUM
      accumulator.
  - Negative Tests (expected to FAIL / be rejected):
    - Recomputing `xT` limbs separately for up and gate.
    - Splitting operands into limbs *before* (rather than after) the transpose.
    - Accumulating products in a bf16 PSUM bank, or emitting the `lo@lo` cross term.

- AC-4: `swiglu_v2` introduces no new HBM spill, and the per-chunk weight-limb rebuild is
  characterized against the PE budget.
  - Positive Tests (expected to PASS):
    - `HBMwr` stays ≈ output-only (~17 MB, matching `swiglu_v1`) → confirms `h` and all
      limb / transpose buffers remain resident (no spill).
    - The profile record for `swiglu_v2` includes **Vec/VE-active time** alongside PE-active,
      and the write-up interprets whether the limb-rebuild Vec work stays hidden under PE
      (diagnostic evidence: aggregate Vec% and any observable PE-wait / DMA-stall, plus the
      before/after latency delta — NOT a single pass/fail counter, since aggregate
      utilization does not by itself prove critical-path overlap).
    - The chunk-size choice is validated: confirm whether CHUNK=512 remains the default
      under bf16x2 (it is one fp32 PSUM bank; the three bf16 products still accumulate into
      that fp32 bank) and record SBUF/PSUM pressure + latency. If `swiglu_v2` misses AC-2 or
      spills, chunk size is a first-line diagnostic to sweep.
  - Negative Tests (expected to FAIL / be rejected):
    - `HBMwr` climbs materially above ~17 MB (indicates an `h` / limb / transpose spill) →
      diagnose and eliminate the spill before promoting.
    - Promoting `swiglu_v2` while the profile evidence omits Vec/VE-active time (the
      limb-rebuild cost is then uncharacterized).

- AC-5: The numerical gate is honest about what the on-device run exercises.
  - Positive Tests (expected to PASS):
    - `runs/offline_bf16_split_sim.py` is re-run and reused as the true multi-seed
      (distinct-seed) numerical gate; its worst rel-L2 is recorded.
    - The `candidates.jsonl` entry for `swiglu_v2` records BOTH the offline distinct-seed
      worst rel-L2 AND the on-device rel-L2, and explicitly flags the seed-42×5 adapter
      caveat (the on-device "5 seeds" all draw identical seed-42 inputs).
  - Negative Tests (expected to FAIL / be rejected):
    - Presenting the on-device run as 5-distinct-seed proof of correctness.
    - Omitting the offline distinct-seed evidence from the promotion record.

- AC-6: Part B (`swiglu_v3_mblock`) is attempted only on a measured trigger, with the
  trigger defined numerically before running v3.
  - Positive Tests (expected to PASS):
    - Before any v3 remote run, a numeric DMA trigger is written down (e.g. "trigger v3 iff
      DMA-active ≥ ~0.85× PE-active on the v2 profile, or an observable PE-wait/DMA-stall
      appears"), derived from `swiglu_v2`'s actual profile (§2.2 predicts the hidden margin
      shrinks from ~1.0 ms to ~0.65 ms).
    - If triggered: build `swiglu_v3_mblock` with B=2 first, then B=4, keeping the better
      (PSUM limit: `up_acc`+`gate_acc` = 2 banks / M-tile → B ≤ 4; per-M-tile activation
      state ≈ 28 KB/part × B stays within budget at B=4 ≈ 112 KB/part). The weight-limb
      rebuild is hoisted out of the M-loop (built once per chunk, reused across the B
      M-tiles), which also amortizes the limb-construction Vec ops.
    - If NOT triggered: record a documented decision to stop at `swiglu_v2`, with the
      DMA-margin number that justified stopping.
  - Negative Tests (expected to FAIL / be rejected):
    - Building `swiglu_v3_mblock` without first recording the measured v2 DMA-margin trigger.
    - Choosing B > 4 (exceeds the 8-bank PSUM budget for up+gate accumulation).
    - Attempting R2 edge/ragged-tile masking (all dims are exact 128-multiples → NO-OP,
      documented) or shipping an off-PE transpose (`dma_transpose` fp32-ineligible;
      `nc_transpose` vector-only) — both are recorded rejects, not implemented.

- AC-7: Evidence and the candidate DAG are complete for every scored candidate.
  - Positive Tests (expected to PASS):
    - `benchmark.csv` gains one row per scored candidate (including any correct-but-slower or
      failed candidate).
    - `candidates.jsonl` gains entries with the parent DAG (v2 ← v1; v3 ← v2 if built), each
      carrying structure, dtypes, metrics, rel-L2 (offline + on-device), and verdict.
    - Profile evidence for each scored candidate is saved under `profile/`.
  - Negative Tests (expected to FAIL / be rejected):
    - A scored candidate with no `benchmark.csv` row or no `candidates.jsonl` entry.
    - A `candidates.jsonl` entry missing its parent link.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. This
draft specifies a highly deterministic design (the bf16x2 3-product split idiom, transposes
exact fp32, limbs rebuilt per streamed weight chunk, the fallback-ladder order); the bounds
below reflect that narrow constraint.

### Upper Bound (Maximum Acceptable Scope)
`swiglu_v2` (all-3-GEMMs bf16x2, correct and > 1.0×) **plus** `swiglu_v3_mblock` (M-block
with B tuned over {2,4}, keeping the better) — built only because `swiglu_v2`'s measured
profile showed weight-DMA exposed. Both carry full profile evidence (latency, MFU, PE, Vec,
DMA, HBM read/write, offline + on-device rel-L2) and `candidates.jsonl` DAG rows. R2/R3/R4
are recorded investigate-and-close decisions with their supporting arithmetic.

### Lower Bound (Minimum Acceptable Scope)
`swiglu_v2` (all-3-GEMMs bf16x2) that clears the L2 gate on-device and lands > 1.0× over
the baseline, with its profile + evidence recorded, `swiglu_v1` kept as the fp32 fallback,
and Part B **documented-skipped** because the measured v2 profile showed DMA still hidden
with comfortable margin. If v2's numerics or speed surprise, the minimum acceptable outcome
is the best offline-PASS rung of the fallback ladder that beats baseline, or — if none beats
baseline — a documented decision to keep `swiglu_v1` with the failure evidence recorded.

### Allowed Choices
- Can use: the compensated bf16x2 3-product split (fixed idiom); per-streamed-chunk on-chip
  weight-limb rebuild (weights are too big to hold bf16 limbs resident, unlike the sibling);
  hoisting/reusing `xT` limbs across up+gate and `hT` limbs for down; sweeping CHUNK size
  and B ∈ {2,4} as diagnostics; the numerical fallback ladder (all-3 → up+down/gate-fp32 →
  only-down → fp32 v1); conditional per-GEMM precision ablations **only** if all-3 fails
  correctness, fails speed, or profiler attribution is ambiguous.
- Cannot use: hand-tuning the baseline; editing `../../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}`;
  fixing the adapter seed bug within this phase (out of scope — separate infra change);
  promoting any kernel that fails the L2 gate or is slower than `swiglu_v1`; bf16 PSUM
  accumulation; plain single-limb bf16 GEMMs (offline FAIL at 4.08e-3); splitting operands
  before the transpose; off-PE transposes (`dma_transpose`/`nc_transpose` — precedent reject).

> **Note on Deterministic Design**: The core technique (bf16x2 all-3-GEMMs) is fixed by the
> draft and sibling precedent; the upper/lower bounds converge on `swiglu_v2` and differ only
> on the *conditional* Part B, which is itself gated on a measured trigger. The genuine
> degrees of freedom are CHUNK size, B, limb-rebuild placement, and — only under numerical
> surprise — which GEMMs stay fp32.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

`swiglu_v2` is a minimal diff from `swiglu_v1`: same loop nest and layout, with limb splits
inserted. Per 128-row M-tile:

1. Load `x` `[m_in,1024]`; transpose once (exact fp32 identity-matmul) → 8 fp32 `xT`
   sub-tiles; split each into `xT_hi`/`xT_lo` (bf16) **once**, shared by up and gate.
2. up / gate, per N-chunk of 512, K-accum over 8 K-tiles into two fp32 PSUM banks:
   - Load the fp32 `w_up`/`w_gate` chunk `[k_in,512]`; build limbs on-chip:
     `w_hi = nl.copy(w, bf16)`; `w_res = nisa.tensor_tensor(w, w_hi, subtract)`;
     `w_lo = nl.copy(w_res, bf16)`.
   - Issue the 3 bf16 products `xT_hi@w_hi + xT_hi@w_lo + xT_lo@w_hi` into the accumulator.
3. PSUM→SBUF copy; fused `nl.silu(gate)`; `nl.multiply` → resident `h_sbuf[128,3072]` fp32
   (identical to v1, no spill).
4. Transpose `h` → 24 fp32 `hT` sub-tiles; split into `hT_hi`/`hT_lo` (bf16).
5. down, per K-out chunk of 512, N-accum over 24 N-tiles: load fp32 `w_down` chunk, build
   its limbs, issue the 3 bf16 products; copy; store.

SBUF fit (≈ 40 KB/partition of the ~200 KB budget): `h_sbuf` fp32 12 KB + `hT_hi/hT_lo` bf16
12 KB + `xT_hi/xT_lo` bf16 4 KB + streamed weight-chunk limbs ~1 KB + fp32 transients ~10 KB.
Weights are streamed (B=1) so only one chunk's limbs are live at a time; the bf16 limbs of
the whole weight are NOT held resident (unlike `add_rmsnorm_matmul_v3`, whose 2048-wide
weight fit) — swiglu's three weights are too big, hence per-chunk rebuild.

`swiglu_v3_mblock` (only if triggered): wrap B M-tiles inside one weight-chunk stream so the
fp32 weight (and its rebuilt limbs) is loaded/built once per chunk and reused across B
M-tiles. PSUM budget caps B ≤ 4; per-M-tile activation state ~28 KB/part scales to ~112
KB/part at B=4. Try B=2 then B=4, keep the better.

### Relevant References
- `runs/swiglu_v1.py` — the fp32 base; v2 is this structure + limb splits (loop nest unchanged).
- `runs/offline_bf16_split_sim.py` — zero-spend distinct-seed numerical gate; reproduces the
  exact seed-42 draw, models the bf16x2 split on each GEMM, reports worst rel-L2 per candidate.
- `profile/swiglu_offline_bf16x2_sim.txt` — the decisive offline evidence (all-3 = 7.72e-6 PASS).
- `profile/swiglu_v1_full5.txt` — v1's on-device metrics (the before-picture).
- `../add_rmsnorm_matmul/runs/add_rmsnorm_matmul_v3_bf16_split.py` — the sibling that proves
  the exact limb-construction idiom on this remote (split-after-exact-transpose; 3-product fp32-PSUM accumulation).
- `../matmul/runs/matmul_v2_b4.py` — the M-block (B=4) precedent for Part B.
- `docs/draft-phase2.md` / `docs/plan-phase2.md` — the full bf16x2 design + cost-model accounting.
- `verify.py` — gates on `l2_norm_passed` across seeds [0,21,42,63,84]; `--fast` = 1 seed / fewer iters.

## Dependencies and Sequence

### Milestones
1. Part A — `swiglu_v2` (bf16x2 all-3-GEMMs), the must-do floor-breaker.
   - Phase A: Re-run the offline sim to reconfirm the all-3 rel-L2 (~7.72e-6) before spending remote budget.
   - Phase B: Implement `swiglu_v2` = v1 + limb splits (AC-3 structure); `--fast` verify (correctness + latency sanity).
   - Phase C: Full run + profile; confirm AC-1 (rel-L2 PASS), AC-2 (>1.0×, MFU up), AC-4 (no new spill, Vec characterized, CHUNK validated); record AC-5 / AC-7 evidence.
   - Phase D (only on surprise): walk the §5 fallback ladder / sweep CHUNK / hoist limb rebuild.
2. Part B — `swiglu_v3_mblock` (M-block), conditional on the Milestone-1 profile.
   - Step 1: From v2's profile, write down the numeric DMA-margin trigger (AC-6). If DMA stays hidden with comfortable margin → record stop-at-v2 and finish.
   - Step 2 (only if triggered): build B=2, then B=4; verify each; keep the better; record evidence.
   - Step 3: Record R2 (NO-OP), R3 (transpose cost re-check on the v2 profile), R4 (precision ladder) as investigate-and-close decisions.

Dependencies: Part A depends on nothing new (base = `swiglu_v1` + existing offline sim). Part B
depends entirely on Part A's measured profile — it is not started until v2 is correct and its
DMA exposure is measured. R3's re-check depends on v2's actual (post-bf16x2) transpose share.

## Task Breakdown

Each task includes exactly one routing tag: `coding` (Claude) or `analyze` (Codex via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Re-run `runs/offline_bf16_split_sim.py`; reconfirm all-3-bf16x2 worst rel-L2 ≤ 1.5e-5 over distinct seeds | AC-1, AC-5 | coding | - |
| task2 | Implement `runs/swiglu_v2.py` = `swiglu_v1` + bf16x2 3-product split on all three GEMMs (xT/hT split once after exact fp32 transpose; per-chunk weight-limb rebuild; SiLU+multiply fp32) | AC-3 | coding | task1 |
| task3 | `--fast` verify `swiglu_v2` (quick correctness + latency sanity) | AC-1, AC-2 | coding | task2 |
| task4 | Full-run verify + profile `swiglu_v2`; capture latency, MFU, PE, **Vec/VE-active**, DMA, HBM r/w, on-device rel-L2 | AC-1, AC-2, AC-4 | coding | task3 |
| task5 | Interpret v2 profile: is limb-rebuild Vec hidden under PE? is CHUNK=512 still right? any new spill? (diagnostic reasoning over the counters) | AC-4 | analyze | task4 |
| task6 | Record `swiglu_v2` in `benchmark.csv` + `candidates.jsonl` (parent v1; offline + on-device rel-L2; seed-42×5 caveat; verdict) and save profile evidence | AC-5, AC-7 | coding | task4 |
| task7 | Write down the numeric Part-B DMA-margin trigger from v2's profile; decide trigger/stop | AC-6 | coding | task4 |
| task8 | If triggered: implement `runs/swiglu_v3_mblock.py` (weight-limb rebuild hoisted out of the B-M-tile loop), B=2 then B=4 | AC-6 | coding | task7 |
| task9 | If v3 built: verify + profile B=2 and B=4, keep the better; record in `benchmark.csv` + `candidates.jsonl` (parent v2) | AC-6, AC-7 | coding | task8 |
| task10 | Record R2 (NO-OP), R3 (transpose cost re-check on v2 profile), R4 (precision ladder) as documented investigate-and-close decisions | AC-6 | coding | task4 |
| task11 | (Only on numerical/speed surprise) walk the §5 fallback ladder and/or sweep CHUNK; record each attempt as a candidate | AC-1, AC-2 | coding | task4 |

## Claude-Codex Deliberation

### Agreements
- Building all-3-GEMMs bf16x2 first (not only-down first) is the right primary move — the
  objective is to escape fp32 emulation, and the offline worst rel-L2 (7.72e-6) is 2.6× under
  the gate.
- Keeping transposes, SiLU, multiply, and PSUM accumulation fp32 is correctly scoped and
  matches the validated numerical evidence.
- Splitting one fp32 `xT` once and sharing `xT_hi`/`xT_lo` across up+gate (not recomputing per
  GEMM) is a structural requirement, not an optimization.
- The numerical fallback ladder is ordered correctly for retreat.
- Part B (M-blocking) must be conditional on measured DMA exposure, not assumed; B=2 before B=4.
- Recording the seed-42×5 adapter caveat is necessary; fixing the adapter is correctly out of scope.
- `swiglu_v1` stays unchanged as the fp32 fallback; a correct-but-slower v2 is a documented
  candidate, not a silent discard.

### Resolved Disagreements
- **AC-4 "VE-hidden" as a hard gate (Codex DISAGREE → resolved):** Codex argued that aggregate
  Vec-active vs PE-active does not prove critical-path overlap, so "Vec% > PE% ⇒ reject" is too
  blunt. Resolution: AC-4 now treats Vec/VE-active as **diagnostic evidence** (must be recorded
  and interpreted) rather than a single pass/fail counter; the hard, testable part of AC-4 is the
  HBM-spill check (`HBMwr` ≈ 17 MB). Rationale: a kernel can be faster with well-overlapped vector
  work, so the promotion signal is the end-to-end latency (AC-2) plus no-spill, with Vec analysis
  explaining *why*.
- **Mandatory per-GEMM ablation candidates (Codex first-pass → resolved):** Resolution: ablations
  are **conditional**, triggered only if all-3 fails correctness, fails speed, or profiler
  attribution is ambiguous (task11 / §5 ladder). Rationale: all-3 passing correctness + speed is
  sufficient; forcing three candidates would burn the ≤5-iter Part-A budget for no attribution need.
- **CHUNK=512 assumed optimal (Codex → resolved):** Resolution: added an explicit CHUNK-validation
  point in AC-4/task5 — confirm 512 is still right under bf16x2 (it is one fp32 PSUM bank) and make
  chunk size a first-line diagnostic if v2 misses AC-2 or spills. Rationale: bf16x2 triples matmul
  issue count and adds limb buffers, so the fp32-era chunk choice deserves a recheck even though the
  fp32 PSUM bank width is unchanged.
- **Part B trigger vaguely "approaching/exceeding" (Codex → resolved):** Resolution: AC-6/task7 now
  require writing the numeric DMA-margin trigger *before* running v3, derived from v2's actual profile.
- **Failed candidates missing from evidence (Codex → resolved):** Resolution: AC-2/AC-7 now require a
  correct-but-slower or failing v2 to be recorded in `benchmark.csv` / `candidates.jsonl` with its profile.

### Convergence Status
- Final Status: `converged` (two Codex passes; all REQUIRED_CHANGES folded in as refinements; no
  high-impact disagreement remains — the core structure was agreed on both passes). Two residual items
  are budget-spending judgment calls carried to the user (see Pending User Decisions), not technical
  disagreements.

## Pending User Decisions

- DEC-1: If `swiglu_v2` is **correct but not faster** than the baseline, should the remaining Part-A
  iterations (≤5) be spent on the precision/CHUNK ablations to try to recover a win, or should the loop
  stop and keep `swiglu_v1` as the documented fp32 fallback?
  - Claude Position: Spend up to ~2 of the remaining iters on the highest-leverage diagnostics first
    (CHUNK sweep + hoisting/placement of the limb rebuild, since those directly address the most likely
    cause — limb-rebuild Vec eroding the PE win), then stop and keep v1 if still ≤1.0×. Correctness is
    already assured by the offline sim, so the risk is purely performance.
  - Codex Position: Needs a user decision — a correct-but-slower v2 could either trigger ablation within
    the 5 iters or simply stop as "not promotable."
  - Tradeoff Summary: Spending iters risks burning remote budget for a possibly-small win; stopping early
    risks leaving a real win (CHUNK / limb placement) on the table. Default (if no decision): follow the
    Claude Position — bounded diagnostic spend, then stop.
  - Decision Status: `PENDING`

- DEC-2: If `swiglu_v2` **beats the baseline but DMA is only moderately exposed** (near but not clearly
  past the trigger threshold), should Part B (`swiglu_v3_mblock`, ≤2 iters) still be attempted?
  - Claude Position: Only attempt Part B if the measured DMA-margin clearly crosses the pre-registered
    trigger (AC-6); a "moderate/ambiguous" reading defaults to a documented stop-at-v2, since M-blocking
    is ≤1.05× polish when DMA is still largely hidden and it adds real SBUF/PSUM complexity.
  - Codex Position: Depends on how aggressively the user wants to spend the 2 remote iters; needs a user decision.
  - Tradeoff Summary: Attempting Part B on a marginal trigger spends 2 remote iters for a likely-small win;
    skipping it may forgo a modest gain. Default (if no decision): follow the Claude Position — require a
    clear trigger crossing, else stop at v2.
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Phase", "Step", "task<N>", or similar workflow markers. These belong in this plan
  document only, not in the kernel sources.
- Use descriptive, domain-appropriate naming in code (e.g. `xT_hi`, `w_lo`, `up_acc`, `h_sbuf`,
  `M_BLOCK`) — matching the style already established in `swiglu_v1.py` and the sibling bf16x2 kernels.
- Preserve `swiglu_v1.py` unchanged; new work lands in `runs/swiglu_v2.py` and (conditionally)
  `runs/swiglu_v3_mblock.py`.

--- Original Design Draft Start ---

# swiglu — Phase 3 draft (regime / shape specialization)

## 0. TL;DR — and a critical status correction

**The phase-2 loop never delivered its primary kernel.** The RLCR loop derailed:
it retired a stale phase-1 loop, Codex surfaced a real *harness-seed* gap (all five
NKIBench "seeds" draw identical seed-42 inputs — see §6), and the agent stopped to
ask for direction. The driver committed only `draft-phase2.md` + `plan-phase2.md`
and marked `phase2.done`. **`swiglu_v2.py` (the compensated bf16x2 split) was never
built.** The only kernel that exists is `swiglu_v1` at **0.939x — still slower than
the 2.0742 ms baseline.**

Consequently the highest-value, still-undone work is exactly the lever phase 2
*defined but never implemented* — and it is also the lever this phase-3 prompt and
the progress memory both name as the phase-3 floor-breaker:

> **Part A (must-do, the floor-breaker): `swiglu_v2` = `swiglu_v1` + compensated
> bf16x2 3-product split on all three GEMMs.** The trn2 PE is bf16-native and
> emulates fp32 at ~44% MFU; two-limb bf16 arithmetic (recovering ~16 mantissa
> bits) runs at bf16 speed. An **offline numpy sim (zero remote spend)** already
> proves all-3-GEMMs bf16x2 clears the gate with margin — worst rel-L2 **7.72e-6 «
> 2e-5** over seeds [42,0,21,63,84], even with error compounding across the three
> chained GEMMs + the SiLU. The idiom is proven on this exact remote (rmsnorm_matmul
> 1.28×, add_rmsnorm_matmul 4.632×). Projected **~1.13–1.20×** (2.208 ms → ~1.73–1.84 ms).

> **Part B (the genuine phase-3 regime specialization, layered on v2): M-tile-block
> regime, gated on the post-v2 profile.** Once bf16x2 unblocks the PE floor, PE-active
> drops ~20% while weight-DMA is unchanged; if the re-profile shows DMA climbing off
> "hidden," specialize the M-tile regime (process B M-tiles per weight stream to
> amortize weight DMA B×; the matmul sibling found B=4 optimal). This is the "specialize
> only where the *measured* win justifies the complexity" mandate. Two important
> **negative** regime findings are pre-established (§4): the shapes are exact
> 128-multiples so there are **no edge/ragged tiles to specialize**, and the 512-wide
> free chunk is already the fp32-PSUM-bank optimum.

Do **not** do Part A and skip Part B, and do **not** invert them: without Part A,
phase 3 ships a "shape-specialized" kernel that is still stuck at 0.939× (below
baseline) — a failure. Part B only earns its complexity if v2's profile shows the
DMA wall; otherwise it is a documented, measured no-op.

---

## 1. Starting point

- **Best correct kernel:** `runs/swiglu_v1.py`, PROMOTED, fp32 throughout.
  Full-run rel-L2 = **6.36e-7** (≈31× under the 2e-5 gate). Latency **2.2079 ms →
  0.939×**. (Caveat: the "full 5-seed" run is seed-42 ×5 — see §6.)
- **Measured metrics (`profile/swiglu_v1_full5.txt`):**
  `MFU=44%  PE=95%  Vec=7%  Scl=4%  DMA=49%  HBMrd=607MB  HBMwr=17MB`.
  → PE-bound at the trn2 fp32-emulation floor; DMA (~1.08 ms) hidden under ~2.10 ms
  of PE-active time; `HBMwr≈output-only` confirms no h spill.
- **Structure (v1):** M-outer, B=1. Per 128-row M-tile: load x `[m_in,1024]`,
  transpose **once** into 8 shared fp32 `xT` sub-tiles (identity-matmul), consumed as
  the stationary operand by **both** up and gate; up/gate over 6 N-chunks of 512
  (K-accum 8 tiles into two fp32 PSUM banks); PSUM→SBUF copy → fused `nl.silu` →
  `nl.multiply` into a **resident** `h_sbuf[128,3072]` (no spill); transpose h into 24
  fp32 `hT` sub-tiles; down over 2 K-out chunks of 512 (N-accum 24 tiles) → store.
  Weights streamed from HBM.
- **Assets already on disk (from phase 2, reusable now):**
  - `runs/offline_bf16_split_sim.py` — the zero-spend numerical gate (verified to
    reproduce today).
  - `profile/swiglu_offline_bf16x2_sim.txt` — its decisive evidence.
  - `docs/draft-phase2.md` / `docs/plan-phase2.md` — the full bf16x2 design and
    cost-model accounting (Part A is essentially executing that plan).

---

## 2. Where time goes (the phase-3 "structure" analysis)

### 2.1 PE cost accounting (per 128-row M-tile, trn2 fp32, from the cost model)

| Work | # matmuls | moving | element-cycles | share |
|---|---|---|---|---|
| up GEMM   | 48 (6 N-chunks × 8 K) | 512 | 24576 | 31.6% |
| gate GEMM | 48                    | 512 | 24576 | 31.6% |
| down GEMM | 48 (2 Kout × 24 N)    | 512 | 24576 | 31.6% |
| x-transpose | 8  | 128 | 1024 | 1.3% |
| h-transpose | 24 | 128 | 3072 | 3.9% |
| **three GEMMs** | | | **73728** | **94.7%** |
| **all transposes** | | | **4096** | **5.3%** |

**The three GEMMs are ~95% of PE work.** The only lever that moves a 95%-GEMM,
PE-bound, fp32-floored kernel is *not paying the fp32 emulation tax* → bf16x2. This is
why Part A precedes any structural regime work.

### 2.2 How the cost balance shifts *after* bf16x2 (this is the phase-3 pivot)

bf16x2 replaces each fp32 GEMM matmul (internally multiple bf16 passes at ~44% MFU)
with **3 explicit bf16 products** at native bf16 rate. The transposes stay **exact
fp32** (we split into limbs *after* the transpose). So post-v2:

- GEMM element-cycles fall by the fp32→bf16 rate ratio (~1.2–1.28× empirically from
  siblings), but the transposes do **not** — their *relative* share roughly doubles
  (5.3% → ~9–11% of the now-smaller PE total). This makes the h-transpose worth a
  fresh look in Part B (§4.3), though the phase-2 cost model still predicts reject.
- Weight-DMA is **unchanged**: bf16x2 must load the fp32 weights to build the lo limb
  (`w - bf16(w)`), so `HBMrd≈607MB` and DMA-active ≈1.08 ms hold. With PE-active
  projected ~1.73 ms, DMA **stays hidden** — but the margin shrinks from ~1.0 ms to
  ~0.65 ms. **This shrinking margin is the trigger condition for the M-block regime
  (§4.1)** — hence "gated on the post-v2 profile," not assumed.

---

## 3. Part A — the floor-breaker: `swiglu_v2` (bf16x2, all-3-GEMMs)

### 3.1 The technique (sibling-proven; idiom lifted from `add_rmsnorm_matmul_v3`)

Each fp32 operand → two bf16 limbs; three bf16 products accumulate in fp32 PSUM
(drop the negligible lo⊗lo term):

```
a_hi = bf16(a),  a_lo = bf16(a - a_hi)        # round-to-nearest-even (nl.copy dtype=bf16)
b_hi = bf16(b),  b_lo = bf16(b - b_hi)
a @ b  ≈  a_hi@b_hi + a_hi@b_lo + a_lo@b_hi     # fp32 PSUM accumulation
```

Applied to all three swiglu GEMMs:
- **up:**   `xT_hi/xT_lo` (stationary, shared) ⊗ `w_up_hi/w_up_lo` (moving)
- **gate:** `xT_hi/xT_lo` (stationary, **same limbs as up** — split once, reuse) ⊗ `w_gate_hi/w_gate_lo`
- **down:** `hT_hi/hT_lo` (stationary) ⊗ `w_down_hi/w_down_lo` (moving)

The identity-matmul transposes stay **exact fp32**; split into limbs *afterward*
(splitting after an exact transpose == splitting before — proven on
add_rmsnorm_matmul v3). The fused `nl.silu` + `nl.multiply` producing `h_sbuf`
stay **fp32**, identical to v1.

### 3.2 The decisive, zero-spend numerical gate (already on disk)

`runs/offline_bf16_split_sim.py` reproduces the exact seed-42 input draw, computes the
fp32 reference, and models the bf16x2 split on each GEMM. Re-verified today
(`profile/swiglu_offline_bf16x2_sim.txt`), worst rel-L2 over seeds [42,0,21,63,84]:

| Variant | worst rel-L2 | verdict |
|---|---|---|
| **all 3 GEMMs bf16x2** | **7.72e-6** | **PASS** (2.6× under gate) |
| up+down bf16x2, gate fp32 | 6.30e-6 | PASS |
| up+gate bf16x2, down fp32 | 6.32e-6 | PASS |
| only down bf16x2 | 4.45e-6 | PASS |
| all 3 **plain bf16** (reject) | 4.08e-3 | **FAIL** (200× over) |

**Compounding is benign.** Error grows monotonically as more GEMMs go bf16x2
(4.4e-6 → 6.3e-6 → 7.7e-6) but the all-3 case is still 2.6× under gate. It is also
essentially **seed-independent** (7.706e-6…7.722e-6 across all five seeds) — a fact
that matters directly for the harness-seed caveat (§6).

**On-device prediction.** Per the add_rmsnorm_matmul precedent, on-device rel-L2 ≈
offline-bf16x2-error ⊕ fp32-emulation-floor in quadrature. Here the fp32 floor is
v1's measured **6.36e-7**, negligible next to 7.72e-6, so predicted on-device
≈ `sqrt(7.72e-6² + 6.4e-7²) ≈ 7.7e-6` — unlike add_rmsnorm_matmul, whose 1.46e-5 floor
dominated. Comfortable, but **confirm on the full run before promoting**.

### 3.3 Implementation sketch — minimal diff from v1

Structurally v1 with limb splits inserted; loop nest and layout unchanged.

1. **Setup:** identity `[128,128]` fp32 loaded once (exact transposes), as v1.
2. **Per M-tile:**
   - Load x `[m_in,1024]`; transpose once → 8 fp32 `xT` sub-tiles; split each into
     `xT_hi/xT_lo` (bf16). **Shared by up and gate.**
   - **up / gate**, per N-chunk (512), K-accum over 8 K-tiles into two fp32 PSUM banks:
     load fp32 `w_up`/`w_gate` chunk `[k_in,512]`, build `w_*_hi/w_*_lo` (bf16) on-chip
     (`hi = copy(w, bf16)`; `res = tensor_tensor(w, hi, subtract)`; `lo = copy(res, bf16)`),
     issue the **3 bf16 products** (`xT_hi@w_hi + xT_hi@w_lo + xT_lo@w_hi`) into the accumulator.
   - PSUM→SBUF copy; fused `nl.silu(gate)`; `nl.multiply` → `h_sbuf[128,3072]` fp32
     resident (identical to v1).
   - Transpose h → 24 fp32 `hT` sub-tiles; split into `hT_hi/hT_lo` (bf16).
   - **down**, per K-out chunk (512), N-accum over 24 N-tiles: load fp32 `w_down` chunk
     `[n_in,512]`, build `w_down_hi/w_down_lo`, issue the **3 bf16 products**; copy; store.
3. **dtypes:** all limbs `nl.bfloat16`; all PSUM accumulation + SiLU/multiply + all
   transposes stay fp32.

**SBUF fit (well within the ~200 KB/partition budget):** `h_sbuf` fp32 12 KB +
`hT_hi/hT_lo` bf16 12 KB + `xT_hi/xT_lo` bf16 4 KB + streamed weight-chunk limbs
(~1 KB) + fp32 transients (~10 KB) ≈ 40 KB/partition. Weights are streamed (B=1),
so only one chunk's limbs are live at a time. The bf16 limbs of the whole weight are
**not** held resident (unlike add_rmsnorm_matmul, whose 2048-wide weight fit) —
swiglu's three weights are too big, so we rebuild limbs per streamed chunk.

### 3.4 Iterations (≤5)
1. Build `swiglu_v2` = v1 + all-3 limb split; `--fast` verify (correctness + latency).
2. Full run + profile: confirm rel-L2 ≈7.7e-6 PASS, MFU rises (44% → ~55–70%),
   PE drops, DMA stays hidden. Record before/after in `benchmark.csv` + `candidates.jsonl`.
3. If rel-L2 surprises, walk the §5 fallback ladder.
4–5. Spare for a limb-order / eviction-fold micro-tune, then hand off to Part B.

---

## 4. Part B — regime / shape specialization (gated on v2's profile)

### 4.1 R1 — M-tile-block regime (B M-tiles per weight stream)  ★ PRIMARY (conditional)
- **What:** process B M-tiles inside one weight-chunk stream so each fp32 weight is
  loaded once and reused across B M-tiles → weight-DMA volume drops ~B×. This is the
  `matmul_v2_b4` lever (matmul sibling: B=4 → 1.017×).
- **Why conditional:** bf16x2 does not change HBM traffic, so if v2's re-profile still
  shows DMA hidden with comfortable margin, M-blocking is ≤1.05× polish and **not worth
  the complexity**. It earns its place only if v2 shows DMA climbing toward the PE wall
  (§2.2 predicts the margin shrinks from ~1.0 ms to ~0.65 ms — plausibly still hidden).
  **Decision is measured, not assumed.**
- **Constraints:** PSUM is the limiter — up_acc + gate_acc = 2 banks/M-tile, so B M-tiles
  need 2B of 8 banks during the up/gate phase → **B ≤ 4**. Per-M-tile activation state
  (`h_sbuf` 12 KB + `hT` limbs 12 KB + `xT` limbs 4 KB ≈ 28 KB) × B: B=4 → ~112 KB/part,
  still within budget. The weight-limb rebuild moves out of the M-loop (built once per
  chunk, reused across the B M-tiles) — which *also* amortizes the limb-construction Vec
  ops, a second-order bonus.
- **Iterations (≤2, only if triggered):** B=2, then B=4; keep the better.

### 4.2 R2 — edge / ragged-tile regime  ✗ NO-OP (documented negative)
- M=4096=32×128, K=1024=8×128, N=3072=24×128 are **all exact multiples of 128**. There
  are **no ragged partition or free edges** anywhere in the kernel — every tile is a full
  128×512 or 128×128. So the classic edge-tile specialization the phase-3 prompt lists
  **does not apply** to this shape. Record as investigated-and-closed; do not implement
  masking/predication.

### 4.3 R3 — free-chunk & transpose-layout regimes  ✗ likely REJECT (re-checked post-bf16x2)
- **Free chunk = 512** is one fp32 PSUM bank (accumulation stays fp32 even under bf16x2),
  so 512 remains the free-dim optimum; no regime to specialize.
- **h-transpose elimination via layout swap (D3 from phase 2):** the phase-2 cost model
  rejected this at fp32 ratios (turning up/gate into 24 moving=128 matmuls costs
  +30720 ec vs the 6144 ec h-transpose saved). Post-bf16x2 the transpose share ~doubles
  (§2.2), so **re-run the cost arithmetic once on v2's actual profile** before final
  reject — but the expectation is it still loses (the extra small-matmul fill overhead
  dwarfs the transpose even at bf16 rates). Off-PE transpose (dma_transpose / nc_transpose)
  stays a **precedent reject** (dma_transpose fp32-ineligible; nc_transpose vector [32,32]).

### 4.4 R4 — mixed-precision GEMM regime (the fallback ladder as a specialization)
- If on-device numerics surprise (unlikely; §3.2), specialize *which* GEMMs are bf16x2 vs
  fp32 per the §5 ladder — a precision-regime specialization that trades a little speed for
  margin. All rungs are offline-PASS.

---

## 5. Numerical safety — the fallback ladder

If v2's on-device full run exceeds the gate (unexpected: 7.7e-6 offline + negligible
fp32 floor), step down the offline-gated ladder (each rung already PASS in the sim):

1. **all 3 bf16x2** — 7.7e-6 (target).
2. **up+down bf16x2, gate fp32** — 6.3e-6 (gate feeds the SiLU → most error-sensitive;
   natural first retreat).
3. **only down bf16x2** — 4.4e-6 (down is 1/3 of PE MACs; smallest, safest win).
4. **fp32 (v1)** — the correctness floor. Never regress below v1's PASS.

Limb construction uses round-to-nearest-even (`nl.copy(dtype=nl.bfloat16)`), the exact
cast the sim models; the residual `a - a_hi` is exact in fp32 for these O(1) normals.

---

## 6. Known caveat — the harness-seed gap (do NOT block on it)

Codex correctly flagged during phase 2 that `adapter/nkibench_case.py` reseeds
`np.random.seed(42)` before **every** input draw (`DEFAULT_INPUT_SEED = 42`), so the
profiler's `multi_seed_seeds=[0,21,42,63,84]` all draw **identical** inputs — the
on-device gate is effectively seed-42 ×5, not five distinct seeds.

**Why this does not block phase 3:** for *this specific bf16x2 change* the seed-diversity
question is already answered off-device — the offline sim (§3.2) draws **real per-seed
inputs** for all five seeds and shows the error is essentially seed-independent
(7.706e-6…7.722e-6, a 0.2% spread). With Gaussian inputs contracted over K=1024 / N=3072,
the relative-L2 concentrates hard, so seed 42 is representative. The offline sim *is* the
multi-seed evidence.

**Scope decision:** fixing the shared adapter (one profiler call per seed) modifies
tooling every op depends on and spends ~5× remote budget; it is **out of scope for a
kernel-optimization phase** and belongs in a separate infra change with user direction.
Phase 3 will (a) rely on the offline multi-seed sim as the numerical gate, (b) run the
standard on-device full run for the latency/PE-metric evidence, and (c) flag this caveat
in `candidates.jsonl` so the promotion evidence is honest about what the on-device gate
actually exercised. Do not silently treat the on-device run as 5-distinct-seed proof.

---

## 7. Deliverables and success criteria

- **Deliverables:**
  - `runs/swiglu_v2.py` — bf16x2 all-3-GEMMs (Part A).
  - `runs/swiglu_v3_mblock.py` — **only if** R1 is triggered by v2's profile and
    measured to win (Part B); otherwise a documented decision to stop at v2.
  - Reuse the existing `runs/offline_bf16_split_sim.py` + `profile/swiglu_offline_bf16x2_sim.txt`.
  - `benchmark.csv` rows + `candidates.jsonl` entries (parent DAG: v2←v1, v3←v2),
    profile evidence under `profile/`.
- **Score command:**
  `python3
  ../../verify.py --op swiglu --candidate runs/<file>.py --fast` (drop `--fast` for the
  promoting run).
- **Success:** on-device full-run PASS (rel-L2 < 2e-5; expect ~7.7e-6) **and** speedup
  > 1.0× — projected **~1.13–1.20×** from v2 alone; a further ~1.0–1.05× if R1 lands.
  Keep v1 as the fp32 fallback.
- **Evidence to capture:** before/after latency; MFU (expect a clear rise from 44%);
  PE/DMA %; HBM bytes; the R1 trigger decision (with the DMA-margin number); the R3
  transpose cost re-check; the §6 seed caveat.
- **Iteration budget:** Part A ≤5 iters (primary, must-do); Part B / R1 ≤2 iters (only if
  v2's profile shows DMA climbing off "hidden"); R2/R3/R4 are recorded decisions, not
  explored unless triggered.
</content>
</invoke>

--- Original Design Draft End ---
