# swiglu Phase 2 — Break the fp32 PE Floor with a Compensated bf16x2 3-Product Split

## Goal Description

Produce `runs/swiglu_v2.py`: a correct NKI kernel for the fused SwiGLU feed-forward
op (M=4096, K=1024, N=3072) that replaces `swiglu_v1`'s emulated-fp32 matmuls with a
**compensated bf16x2 3-product split** on the three GEMMs (up, gate, down), while
keeping fp32 PSUM accumulation and exact fp32 identity-matmul transposes. The purpose
is to move the kernel off the trn2 fp32-emulation PE floor (where `v1` sits at
MFU≈44%, PE≈95%) by running the matmul arithmetic at bf16-native speed and recovering
~16 mantissa bits through two-limb compensation.

Success is defined by the NKIBench remote profiler: a full-5-seed relative-L2 PASS
(< 2e-5, over seeds `[0,21,42,63,84]`) **and** a latency below the reference baseline
of **2.0742 ms** (i.e. `speedup = 2.0742 / candidate_latency > 1.0x`, per
`verify.py`). The offline numpy sim already proves the numerical feasibility
(all-3-GEMMs bf16x2 worst rel-L2 = 7.72e-6, ≈2.6× under the gate). `swiglu_v1`
(PROMOTED, fp32, 0.939x, rel-L2 6.36e-7) is retained as the correctness fallback and
is never regressed.

The one structural difference from the resident-weight siblings that the plan must
respect: `swiglu` **streams weights (B=1)** because the three weights (288 KB/part
fp32) do not fit resident. Consequently the bf16 **weight-limb construction re-runs
on-chip for every M-tile (32×)** — a Vector/Scalar tax the siblings never paid. The
plan treats M-blocking (D2) as the primary mitigation for that tax, not merely as DMA
insurance.

## Acceptance Criteria

- AC-1: `runs/swiglu_v2.py` computes the fused SwiGLU op correctly under the NKIBench
  relative-L2 gate across the full seed set.
  - Positive Tests (expected to PASS):
    - `verify.py --op swiglu --candidate runs/swiglu_v2.py` (no `--fast`) reports
      `l2_norm_passed=True` for all seeds `[0,21,42,63,84]` with `relative_l2_error < 2e-5`.
    - The `--fast` (seed 42) smoke run passes before the full-5-seed run is spent.
  - Negative Tests (expected to FAIL):
    - A variant that accumulates the three bf16 products in a bf16 (not fp32) PSUM/buffer
      fails the gate (mirrors the offline "plain bf16" = 4.08e-3 FAIL, ≈200× over gate).
    - A variant that keeps only the `hi@hi` product (single-limb bf16, no compensation)
      fails the gate.
  - AC-1.1: The on-device rel-L2 is confirmed by the full-5-seed run before any
    promotion — the offline 7.72e-6 is a pre-check, not the promotion evidence.
    - Positive: recorded on-device worst-seed rel-L2 is reported in `candidates.jsonl`
      and is `< 2e-5`.
    - Negative: promoting on the strength of the `--fast` (seed 42) run alone, or on
      the offline sim alone, is rejected.

- AC-2: The bf16x2 split is implemented per the pinned arithmetic contract, preserving
  the fp32 accumulation and exact transposes.
  - Positive Tests (expected to PASS):
    - Each fp32 operand is split as `hi = bf16(v)`, `residual = v - hi` (fp32 subtract),
      `lo = bf16(residual)` (round-to-nearest-even via `nl.copy(dtype=nl.bfloat16)`).
    - Each GEMM accumulates exactly three bf16 products — `a_hi@b_hi + a_hi@b_lo +
      a_lo@b_hi` — into an fp32 PSUM bank; the `lo@lo` cross term is dropped.
    - The x-transpose (8 sub-tiles/M-tile) and h-transpose (24 sub-tiles/M-tile) remain
      exact fp32 identity matmuls; limbs are split from the fp32 transposed result
      afterward.
  - Negative Tests (expected to FAIL):
    - Splitting a limb before an inexact/bf16 transpose (breaks the "split-after-exact-
      transpose ≡ split-before" equivalence).
    - Introducing a fourth `lo@lo` product (measured-reject on the add_rmsnorm sibling:
      +cost for <2% accuracy) or reordering limbs so accumulation loses the hi term.

- AC-3: The intended PE mechanism is realized and confirmed on the profile — MFU rises
  off the fp32-emulation floor and the added weight-limb-construction work does not
  dominate.
  - Positive Tests (expected to PASS):
    - The D1 profile shows MFU rising clearly above `v1`'s 44% (target band ~55–70%)
      with a corresponding drop in PE-active time.
    - Non-PE engines do not become the wall: Vector/Scalar active time growth does not
      erase the PE-active-time reduction, and DMA active time stays hidden under
      PE-active time (or is addressed by D2).
  - Negative Tests (expected to FAIL):
    - MFU stays pinned near 44% (indicates the matmuls are still emulated fp32, i.e. the
      split did not lower to bf16 matmuls).
    - Vector/Scalar active time grows enough that candidate latency does not beat
      baseline despite an MFU improvement (weight-split tax exposed at B=1) — this is a
      trigger for D2, not an acceptable end state.

- AC-4: No memory regression — the kernel introduces no SBUF spill and no unexpected
  HBM traffic from limb construction.
  - Positive Tests (expected to PASS):
    - `HBMwr` stays ≈ output-only (~16–17 MB) as in `v1` (no h spill); `HBMrd` stays
      ≈ `v1`'s ~607 MB (limbs are built on-chip from streamed fp32 weights; bf16x2 does
      not change HBM traffic).
    - A peak-liveness SBUF/PSUM budget is written and shows the working set fits the
      ~208 KB/partition SBUF budget and ≤ 8 PSUM banks at all times.
  - Negative Tests (expected to FAIL):
    - The profile shows SBUF spill or HBM read/write inflating beyond `v1`'s footprint
      (weight limbs spilling to HBM ⇒ D1 as-built is dead and must be re-tiled or fall
      back to a selective variant).

- AC-5: Every scored candidate is recorded as evidence with parent DAG links, and the
  benchmark definition is never modified.
  - Positive Tests (expected to PASS):
    - Each perf change appends a row to `benchmark.csv`; each candidate appends a
      `candidates.jsonl` entry with `parent` = `swiglu_v1` (and further parents forming
      the DAG for D2/selective variants).
    - `runs/offline_bf16_split_sim.py` and `profile/swiglu_offline_bf16x2_sim.txt`
      remain the offline gate of record.
  - Negative Tests (expected to FAIL):
    - Any edit to `../../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}` or to
      a baseline kernel.
    - A promoted candidate lacking a `candidates.jsonl` entry or a `benchmark.csv` row.

- AC-6: The rejected directions are recorded with their evidence and are not
  implemented or re-probed.
  - Positive Tests (expected to PASS):
    - D3 (h-transpose elimination via layout swap) and D4 (off-PE transpose) appear in
      the plan/evidence as costed/precedent rejects with their reasoning.
  - Negative Tests (expected to FAIL):
    - Spending remote budget re-probing `dma_transpose`/`nc_transpose` for fp32, or
      implementing the layout-swap rewrite.

## Path Boundaries

> **Note on Deterministic Designs**: The draft specifies a highly deterministic design
> (bf16x2 3-product split, all three GEMMs, offline-gated). The bounds below are
> therefore narrow; the primary variation is the D1↔D2↔selective-variant ladder driven
> by the on-device profile, not a free choice of technique.

### Upper Bound (Maximum Acceptable Scope)
`runs/swiglu_v2.py` implements the bf16x2 3-product split on all three GEMMs with fp32
PSUM accumulation and exact fp32 transposes, passes the full-5-seed gate, and clears
the baseline. If the B=1 profile shows the weight-split tax exposed, the work extends
to M-blocking (D2, B=2 then B=4) as the primary mitigation, keeping the better
configuration; and, if the profile or correctness dictates, to a selective-variant
(up+gate, or only-down) that trades split coverage for a smaller tax or more margin.
All directions carry before/after latency + profile evidence and DAG-linked candidate
rows. D3/D4 are recorded rejects.

### Lower Bound (Minimum Acceptable Scope)
`runs/swiglu_v2.py` implements the offline-gated split on at least the direction that
achieves a full-5-seed PASS **and** a latency below the 2.0742 ms baseline (the primary
target is all-3; a selective variant is acceptable if it is what clears both gates),
recorded in `benchmark.csv` / `candidates.jsonl` with parent `swiglu_v1`. If no
bf16x2 variant clears the baseline after exhausting the D1/D2 ladder within budget,
`swiglu_v1` remains the PROMOTED kernel and the negative result (with profile evidence
of why) is recorded.

### Allowed Choices
- Can use: bf16x2 3-product split on any subset of the three GEMMs (all-3 primary; the
  up+gate / up+down / only-down variants are the offline-gated ladder); M-blocking with
  B ∈ {1,2,4} bounded by PSUM banks and SBUF peak liveness; the existing exact
  identity-matmul transpose idiom; fused `nisa.activation(op=nl.silu)` + `nl.multiply`
  as in `v1`.
- Cannot use: plain single-limb bf16 or bf16 accumulation (fails the gate); a `lo@lo`
  fourth product (measured-reject); prepacked/pre-split weight inputs or any change to
  the fixed NKIBench kernel signature `(v1,v2,v3,v4)->v5`; off-PE transpose for fp32
  (D4); the layout-swap h-transpose elimination (D3); editing the benchmark definition
  or hand-tuning a baseline.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach
Structurally `swiglu_v2` is `swiglu_v1` with limb splits inserted; the loop nest and
layout are unchanged (keeps the diff small and reviewable). Per M-tile:

1. Load `x[m_in,1024]`; transpose ONCE into 8 fp32 `xT` sub-tiles (exact identity
   matmul, `is_transpose=True`), then split each into `xT_hi/xT_lo` (bf16). These limbs
   are **shared by up and gate** (as `v1` shares the fp32 xT).
2. up / gate, per N-chunk (512), K-accum over 8 K-tiles into two fp32 PSUM banks: load
   fp32 `w_up`/`w_gate` chunk `[k_in,512]`, build `w_*_hi/w_*_lo` (bf16) on-chip, issue
   the 3 bf16 products (`xT_hi@w_hi + xT_hi@w_lo + xT_lo@w_hi`) into the accumulator.
3. PSUM→SBUF copy; fused `nl.silu(gate)`; `nl.multiply` → resident `h_sbuf[128,3072]`
   fp32 (identical to `v1`).
4. Transpose h into 24 fp32 `hT` sub-tiles; split into `hT_hi/hT_lo` (bf16).
5. down, per K-out chunk (512), N-accum over 24 N-tiles: load fp32 `w_down` chunk
   `[n_in,512]`, build `w_down_hi/w_down_lo`, issue the 3 bf16 products; copy; store.

Split contract (pinned): `hi = bf16(v)`, `residual = v - hi` (fp32 subtract), `lo =
bf16(residual)`; accumulate `hi@hi + hi@lo + lo@hi` in fp32 PSUM, drop `lo@lo`, never
accumulate in bf16.

The B=1 tax to watch: weight-limb construction (copy→bf16 hi, subtract residual,
copy→bf16 lo) runs for every weight chunk on every M-tile (~302M weight element-splits
per invocation) — the new Vector/Scalar work absent from `v1` (Vec=7%). If the D1
profile shows this exposed, M-blocking (D2) amortizes both weight HBM reads and the
limb construction B-fold; selective variants (fewer split GEMMs) reduce the tax at the
cost of split coverage.

### Relevant References
- `runs/swiglu_v1.py` — the fp32 base; `swiglu_v2` is a limb-split of this exact loop nest.
- `runs/offline_bf16_split_sim.py` + `profile/swiglu_offline_bf16x2_sim.txt` — the
  offline numerical gate (all-3 = 7.72e-6; the fallback-ladder margins).
- `profile/swiglu_v1_full5.txt` — the `v1` baseline metrics (MFU=44% PE=95% DMA=49%
  Vec=7% HBMrd=607MB HBMwr=17MB) the D1 profile is compared against.
- `../add_rmsnorm_matmul/runs/add_rmsnorm_matmul_v3_bf16_split.py` — the proven bf16x2
  3-product idiom on this remote (limb construction, 3-product accumulation, exact
  transpose then split), with the caveat that its weights are RESIDENT (limbs built once).
- `../rmsnorm_matmul/runs/rmsnorm_matmul_v4_bf16_split.py` — the 1.066x→1.363x precedent.
- `kernel-cost-analysis` — the theoretical PE floor to compare the D1 profile against.

## Dependencies and Sequence

### Milestones
1. D1 — bf16x2 3-product on all three GEMMs (PRIMARY):
   - Phase A: write the peak-liveness SBUF/PSUM budget (limbs, accumulators, h
     residency, hT limbs, output PSUM, transient residuals; account for any
     double-buffering). Confirm it fits before coding.
   - Phase B: build `swiglu_v2` from `v1` per the split contract; `--fast` (seed 42)
     verify for correctness + a first latency read.
   - Phase C: full-5-seed verify + profile; record on-device worst-seed rel-L2, MFU,
     PE/Vec/Scl/DMA active time and %, HBMrd/HBMwr; confirm AC-3 and AC-4.
   - Phase D: if rel-L2 exceeds the gate, walk the correctness ladder (Milestone 3).
2. D2 — M-blocking (mitigation, triggered by the D1 profile):
   - Depends on D1 producing a correct kernel whose profile shows the weight-split tax
     exposed or DMA climbing (per the numeric triggers in Task Breakdown).
   - Step 1: B=2 (bounded by 2B PSUM banks for up/gate accumulators and by SBUF peak
     liveness, which B also multiplies for xT/h_sbuf/hT/limbs — verify, do not assume
     B≤4 from PSUM alone).
   - Step 2: B=4 if B=2 helps and liveness permits; keep the better.
3. Selective-variant ladder (correctness and/or performance fallback):
   - Depends on the D1/D2 profile or a correctness miss: up+gate (shares x-limbs,
     splits 2/3 of weight volume) or only-down (splits 1/3, lowest tax); each is
     already an offline PASS. Rank by expected tax/benefit before implementing.
4. Rejects recorded (no work): D3 (layout-swap, +30720 element-cycles) and D4 (off-PE
   transpose, fp32-ineligible / [32,32]-limited) stay documented, not implemented.

<Dependencies are structural: D2 and the selective variants are conditioned on D1's
on-device profile and correctness result, not on a schedule.>

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Write the peak-liveness SBUF/PSUM budget for D1 (all-3 bf16x2, B=1): x/h limbs, w limbs per chunk, up/gate accumulators, h resident, hT limbs, output PSUM, transient residuals, double-buffering; confirm ≤208 KB/part and ≤8 PSUM banks | AC-4 | analyze | - |
| task2 | Build `runs/swiglu_v2.py` = `v1` + bf16x2 3-product split on all 3 GEMMs, per the pinned split contract; exact fp32 transposes then split; fp32 PSUM accumulation, drop `lo@lo` | AC-2 | coding | task1 |
| task3 | `--fast` (seed 42) verify of `swiglu_v2` for correctness + first latency read | AC-1 | coding | task2 |
| task4 | Full-5-seed verify + profile; record on-device rel-L2, MFU, PE/Vec/Scl/DMA active time+%, HBMrd/HBMwr into `benchmark.csv` + `candidates.jsonl` (parent `swiglu_v1`) | AC-1, AC-3, AC-4, AC-5 | coding | task3 |
| task5 | Interpret the D1 profile against the numeric D2 triggers (MFU rise, Vec/Scl active-time growth vs PE-active-time drop, DMA hidden-vs-exposed); decide whether D2/selective is required | AC-3 | analyze | task4 |
| task6 | If triggered: M-blocking D2 (B=2 then B=4), bounded by PSUM banks and re-verified SBUF peak liveness; verify + profile each; keep the better; DAG-link candidates | AC-3, AC-4, AC-5 | coding | task5 |
| task7 | If correctness surprises or the tax dominates: implement a selective offline-gated variant (up+gate or only-down), ranked by tax/benefit; verify + profile; DAG-link | AC-1, AC-3, AC-5 | coding | task5 |
| task8 | Record D3/D4 as costed/precedent rejects in the phase-2 evidence; do not implement or re-probe | AC-6 | coding | - |

## Claude-Codex Deliberation

### Agreements
- The bf16x2 3-product split is the correct and only floor-breaking lever; it is
  offline-gated PASS (all-3 = 7.72e-6, ≈2.6× under the 2e-5 gate) and proven on two
  siblings on this exact remote.
- D2 (M-blocking) is correctly reframed as the **primary mitigation for repeated
  weight-limb construction** under B=1 streaming, not merely DMA insurance.
- The correctness ladder and "never promote without a full-5-seed PASS" are sound;
  `--fast` (seed 42) first, full seeds only for candidates with plausible latency.
- Rejecting D3 (h-transpose elimination) and D4 (off-PE transpose) remains correct.
- A peak-liveness SBUF/PSUM budget and a profile check for no limb spills are required
  before/with D1.

### Resolved Disagreements
- Streamed-weight tax vs sibling transfer: Codex flagged that sibling speedups
  (resident weights, limbs built once) may not transfer because `swiglu` streams
  weights and re-splits them 32×. Resolved by elevating D2 to the primary tax
  mitigation and adding AC-3's explicit "Vec/Scl must not dominate" test — the expected
  ~1.13–1.20x is now framed as contingent on the tax staying hidden, verified on the
  profile, not assumed.
- "Quadrature" error model: Codex noted correlated GEMM error through the SiLU
  nonlinearity can exceed a quadrature estimate. Resolved by making AC-1.1 require the
  on-device full-5-seed rel-L2 as the promotion evidence (the offline 7.72e-6 and the
  quadrature note are pre-checks only).
- Qualitative D2 triggers: Codex required numeric/comparative triggers. Resolved by
  restating the D2 trigger in Task Breakdown as (a) MFU rose but latency does not beat
  baseline by the chosen margin, (b) Vec/Scl active-time growth erases the PE-active-time
  reduction, or (c) DMA active time becomes latency-correlated — measured as active
  time, not just %.
- HBM byte reconciliation: naive streamed volume (~1.2 GB) vs measured HBMrd=607 MB is
  a profiler unique-bytes/reuse artifact; AC-4 records it as a sanity note (bf16x2 does
  not change HBM traffic), not a blocker.
- Profile percentages are not additive: resolved by having AC-3/AC-4 compare **active
  time** deltas (PE-active shrinks; Vec/Scl/DMA active vs the new PE-active), since
  shrinking PE time makes raw percentages misleading.

### Convergence Status
- Rounds executed: 1 (Codex first-pass analysis + 1 convergence review).
- Final Status: `converged` — no remaining technical disagreements after incorporating
  Codex's REQUIRED_CHANGES; the only open items are the two user decisions below.

## Pending User Decisions

- DEC-1: Promotion margin for `swiglu_v2`.
  - Claude Position: Promote on any full-5-seed rel-L2 PASS (< 2e-5) with latency below
    the 2.0742 ms baseline (the draft's stated success). The expected ~1.13–1.20x is far
    outside profiler noise, so a margin rule only bites in the unlikely marginal case.
  - Codex Position: Promote only with a comfort margin — rel-L2 < 1.5e-5 **and** latency
    below baseline by ≥1% (optionally a repeat timing) — to avoid promoting profiler
    noise near 1.0x.
  - Tradeoff Summary: The strict `> 1.0x` rule is simplest and matches the draft; a
    margin rule is safer only if the win lands within ~1% of baseline (unexpected given
    the projection). Recommendation: adopt Codex's margin **only as a tie-breaker** — if
    latency is within ~1% of baseline, require a repeat run before promoting; otherwise
    promote on PASS + `> 1.0x`.
  - Decision Status: `PENDING`

- DEC-2: Is D2 (M-blocking) mandatory before falling back to `v1`?
  - Claude Position: D2 is triggered by the D1 profile (the draft's conditional framing),
    but if all-3 bf16x2 passes correctness and misses the latency target specifically
    because the B=1 weight-split tax is exposed, at least B=2 should be tried before
    declaring `v1` the stop — this is the primary mitigation for the identified tax.
  - Codex Position: Yes — make D2 mandatory after an all-3 correctness PASS if the B=1
    latency misses the promotion margin and the profile indicates the weight-split tax
    or exposed non-PE time; stopping at `v1` without trying B=2 leaves the primary
    mitigation untested.
  - Tradeoff Summary: Positions nearly coincide; the only question is remote-profiling
    budget for the extra D2 runs (task6, ≤2 iters). Recommendation: make D2 (B=2)
    mandatory in exactly the "correct-but-slow-due-to-exposed-tax" case, budget
    permitting; otherwise record the negative result and keep `v1`.
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Phase", "Step", "D1/D2", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `w_up_hi`,
  `w_up_lo`, `xT_hi`, `hT_lo`, `up_acc`), matching the naming already used in
  `swiglu_v1.py` and the sibling bf16-split kernels.

--- Original Design Draft Start ---

# swiglu — Phase 2 draft (profile-driven optimization)

## 0. TL;DR

Phase-1 `swiglu_v1` is a correct fp32 kernel at **0.939x** (2.2079 ms vs the
2.0742 ms baseline). The profile says it is **PE-bound at the trn2 fp32-emulation
floor** (PE=95%, MFU=44%, DMA=49% hidden). Every "fusion" lever the phase-2 prompt
lists — share x across up/gate, fuse the SiLU gate, keep the (M,N) intermediate in
SBUF, no HBM spill — is **already implemented in v1**. The cost model confirms the
transposes are only ~5% of PE work and that removing the h-transpose by a layout
swap actually *loses*. So the only lever that moves the compute floor is the one the
PE array's own arithmetic dictates:

> **Primary Phase-2 direction: compensated bf16x2 3-product split on all three
> GEMMs.** The trn2 PE is bf16-native and emulates fp32 at ~44% MFU; doing the
> arithmetic in two-limb bf16 (recovering ~16 mantissa bits) runs at bf16 speed.
> An **offline numpy sim (zero remote spend)** proves all-3-GEMMs bf16x2 clears the
> gate with margin (**worst rel-L2 = 7.7e-6 « 2e-5**, over seeds [42,0,21,63,84]),
> even with error compounding across the 3 chained GEMMs + the SiLU nonlinearity.

Expected: ~1.2–1.28x on the compute floor (sibling-proven: rmsnorm_matmul
1.066x→1.363x = 1.28x; add_rmsnorm_matmul ~1.2x from the same split). Secondary
lever (M-blocking) is an *enabler/insurance* — measured and applied only if the
post-bf16x2 profile shows DMA climbing off "hidden." Transpose-elimination and
off-PE transpose are **costed/precedent REJECTS** (below).

---

## 1. Starting point and the Phase-2 mandate

- **Best correct kernel:** `runs/swiglu_v1.py`, PROMOTED, full-5-seed PASS,
  rel-L2 = **6.36e-7** (≈31× under the 2e-5 gate — the fp32 floor here is
  remarkably low; see §5). Latency **2.2079 ms → 0.939x**.
- **Measured metrics (full-5-seed, `profile/swiglu_v1_full5.txt`):**
  `MFU=44%  PE=95%  Vec=7%  Scl=4%  DMA=49%  HBMrd=607MB  HBMwr=17MB`.
- Phase-2 goal (from the prompt): identify the real bottleneck, enumerate
  directions, rank by benefit-vs-risk, explore each ≤5 iterations, keep
  before/after latency + profiling evidence, never regress correctness.

---

## 2. Profile-driven bottleneck read

**The kernel is PE-bound, and the PE is stuck at the trn2 fp32-emulation floor.**

- `PE=95%` — the Tensor Engine is busy essentially all the time.
- `MFU=44%` — but it only achieves 44% of *peak* FLOPs. On trn2 the systolic array
  is **bf16-native**; a correct fp32 matmul is emulated with multiple internal bf16
  passes, capping MFU at ~44–46%. This exact signature (`PE≈95%, MFU≈44%`) recurs on
  every fp32 sibling (matmul, rmsnorm_matmul, add_rmsnorm_matmul). It is not a
  scheduling defect — it is the price of fp32 arithmetic on this hardware.
- `DMA=49%` — DMA is active ~1.08 ms, comfortably **hidden** under the ~2.10 ms of
  PE-active time. `HBMwr=17MB ≈ output-only (16MB)` confirms v1 does **not** spill h
  (unlike the baseline's `_spill_163`/`_reload_166`, ~+100MB traffic).
- `Vec=7%, Scl=4%` — the fused SiLU + the multiply are trivial; fully hidden.

**Consequence for direction-picking.** Because the kernel is PE-bound with DMA
hidden, *reducing DMA cannot speed it up* — the phase-1 memory's guess that
"M-blocking to amortize weight DMA" is the phase-2 win is **wrong for a PE-bound
kernel**. The only way below ~2.1 ms is to make the PE do the same math in fewer
cycles, which means **not paying the fp32 emulation tax**.

### 2.1 Cost-model accounting of PE work (per 128-row M-tile, trn2)

Using the instruction cost model (Formula A: Matmul latency ∝ moving-free elements;
`kernel-cost-analysis`), per M-tile v1 issues:

| Work | # matmuls | moving | element-cycles | share |
|---|---|---|---|---|
| up GEMM   | 48 (6 N-chunks × 8 K) | 512 | 24576 | 31.6% |
| gate GEMM | 48                    | 512 | 24576 | 31.6% |
| down GEMM | 48 (2 Kout × 24 N)    | 512 | 24576 | 31.6% |
| x-transpose | 8  | 128 | 1024 | 1.3% |
| h-transpose | 24 | 128 | 3072 | 3.9% |
| **useful GEMMs** | | | **73728** | **94.7%** |
| **all transposes** | | | **4096** | **5.3%** |

**The three GEMMs are ~95% of PE work; the transposes are ~5%.** Attacking the
GEMMs (bf16x2) dominates any transpose optimization by an order of magnitude.

---

## 3. Why the prompt's "fusion" directions are already spent

The phase-2 prompt lists candidate directions; v1 already realizes each:

| Prompt direction | Status in v1 |
|---|---|
| share the single x load across up+gate | **Done** — x transposed **once** per M-tile into 8 shared xT sub-tiles, consumed as the stationary operand by both up and gate. |
| fuse SiLU gate + multiply into down staging | **Done** — one `nisa.activation(op=nl.silu)` + one `nl.multiply` produce h resident. |
| keep (M,N) intermediate in SBUF | **Done** — `h_sbuf[128,3072]` (12 KB/part) stays resident; no spill (`HBMwr=17MB`). |
| tile K/N to keep PSUM banks full (free ≤ 512) | **Done** — up/gate over 6 N-chunks of 512; down over 2 K-out chunks of 512; K-accum in fp32 PSUM banks. |

So there is **no cheap fusion win left**; the remaining PE cost is intrinsic
arithmetic. This is the honest reason v1 (0.939x, less work) is essentially tied
with the baseline (which spills h and re-transposes x): **both are pinned at the
fp32 PE floor (~2.1 ms).** The floor is the enemy, and only bf16x2 moves it.

---

## 4. The lever: compensated bf16x2 3-product split (offline-GATED)

### 4.1 The technique (sibling-proven)

Each fp32 operand is kept as two bf16 limbs; three bf16 products accumulate in fp32
PSUM (the negligible lo⊗lo cross term is dropped):

```
a_hi = bf16(a),  a_lo = bf16(a - a_hi)        # round-to-nearest-even, ~16 mantissa bits
b_hi = bf16(b),  b_lo = bf16(b - b_hi)
a @ b  ≈  a_hi@b_hi + a_hi@b_lo + a_lo@b_hi     # fp32 accumulation
```

Applied to all three swiglu GEMMs (stationary = the transposed activation limbs,
moving = the weight limbs):

- **up:**   xT_hi/xT_lo (stationary, shared) ⊗ w_up_hi/w_up_lo (moving)
- **gate:** xT_hi/xT_lo (stationary, **same limbs as up**) ⊗ w_gate_hi/w_gate_lo
- **down:** hT_hi/hT_lo (stationary) ⊗ w_down_hi/w_down_lo (moving)

The fp32 identity-matmul transposes stay **exact fp32**; we split the fp32 result
into limbs afterward (splitting after an exact transpose == splitting before —
proven on add_rmsnorm_matmul v3).

### 4.2 Offline numerical gate — the decisive, zero-spend evidence

`runs/offline_bf16_split_sim.py` reproduces the exact scored input (seed-42 draw of
x, w_up, w_down, w_gate in reference order), computes the fp32 reference, and models
the bf16x2 split on each GEMM. Result (`profile/swiglu_offline_bf16x2_sim.txt`),
worst rel-L2 over seeds [42,0,21,63,84]:

| Variant | worst rel-L2 | verdict |
|---|---|---|
| **all 3 GEMMs bf16x2** | **7.72e-6** | **PASS** (2.6× under gate) |
| up+down bf16x2, gate fp32 | 6.30e-6 | PASS |
| up+gate bf16x2, down fp32 | 6.32e-6 | PASS |
| only down bf16x2 | 4.45e-6 | PASS |
| all 3 **plain bf16** (reject) | 4.08e-3 | **FAIL** |

**Key finding — compounding is benign.** The worry (phase-1 memory) was that bf16x2
error compounds across 3 chained GEMMs + the SiLU. It does grow monotonically
(4.4e-6 → 6.3e-6 → 7.7e-6 as more GEMMs go bf16x2), but even the all-3 case is
**2.6× under the gate**. Plain single-limb bf16 fails by 200×, confirming the split
is what makes it feasible. **The aggressive all-3 variant is the target**; the
partial variants are a ready fallback ladder (§6) if the device surprises.

### 4.3 Expected speedup and DMA headroom

- **Compute:** siblings give the empirical multiplier for fp32→bf16x2-3product:
  rmsnorm_matmul **1.28x** (1.066→1.363), add_rmsnorm_matmul **~1.2x**. Applying
  ~1.2–1.28x to v1's 0.939x ⇒ **~1.13–1.20x** projected (2.208 ms → ~1.73–1.84 ms).
- **DMA stays hidden.** bf16x2 must load fp32 weights (the lo limb needs
  `w - bf16(w)`), so `HBMrd` is unchanged (~607MB) and DMA-active stays ~1.08 ms.
  Projected PE-active ~1.73 ms > 1.08 ms ⇒ **DMA remains hidden** — the bf16x2 win
  is real, not a DMA mirage. (If a later push drops PE below ~1.1 ms, DMA becomes
  the wall — that is exactly when M-blocking, §5.2, earns its place.)
- **SBUF fits.** Weights are streamed, so only a single chunk's limbs are live
  (tiny); xT limbs (bf16, ~4 KB), hT limbs (bf16, ~12 KB), h_sbuf (fp32, 12 KB) all
  fit the 208 KB/part budget with room to spare.

---

## 5. Ranked directions (benefit vs risk)

### D1 — bf16x2 3-product on all three GEMMs  ★ PRIMARY
- **Benefit:** high (~1.2–1.28x, breaks the fp32 floor). **Risk:** low — offline-
  gated PASS (7.7e-6), primitive proven on two siblings on this exact remote.
- **Correctness caveat (from add_rmsnorm_matmul):** on-device rel-L2 can combine the
  offline bf16x2 error with the fp32-emulation floor **in quadrature**. Here the
  fp32 floor is v1's measured **6.36e-7** — negligible next to 7.7e-6 — so predicted
  on-device ≈ `sqrt(7.7e-6² + 6.4e-7²) ≈ 7.7e-6`, still 2.6× under gate. (Contrast
  add_rmsnorm, whose fp32 floor was 1.46e-5 and dominated.) Robust, but **must be
  confirmed with the full 5-seed run before promoting.**
- **Iterations (≤5):** (1) build swiglu_v2 = v1 + limb split on all 3 GEMMs; verify
  `--fast`. (2) full-5-seed verify + profile; confirm MFU rises (~44%→~55–70%) and
  DMA stays hidden. (3) if rel-L2 surprises, walk the §6 fallback ladder. (4–5)
  spare for a limb-order / eviction-fold micro-tune.

### D2 — M-blocking (B M-tiles per weight stream)  ◑ SECONDARY / ENABLER
- **Benefit:** recovers the ~5% PE-idle (95%→~100%) and cuts weight-DMA volume ~B×,
  keeping DMA hidden after D1 speeds the PE up. **Alone (fp32) it is ~≤1.05x** — a
  polish, not a floor-breaker (DMA is already hidden). **Risk:** medium — PSUM
  pressure: up_acc+gate_acc = 2 banks/M-tile, so B M-tiles need 2B of 8 banks (B≤4);
  h_sbuf grows to B×12 KB. Only pursue if D1's profile shows DMA% climbing toward
  the PE wall. **Decision gated on the post-D1 profile, not assumed.**
- **Iterations (≤2, only if triggered):** try B=2 then B=4; keep the better.

### D3 — eliminate the h-transpose via layout swap  ✗ REJECT (cost-model)
- Idea: emit up/gate directly in `[n_in, m]` layout so the down GEMM consumes it
  without the 24 h-transposes. **Cost model says this loses:** the h-transpose costs
  ~6144 element-cycles/M-tile, but emitting up/gate transposed turns each into 24
  small (moving=128) matmuls, adding ~2×(fill overhead) ≈ +30720 element-cycles.
  **Net +30720 (a loss).** The h-transpose is only 3.9% of PE work; not worth a
  structural rewrite. Record the cost math; do not implement.

### D4 — off-PE transpose (dma_transpose / nc_transpose)  ✗ REJECT (precedent)
- Siblings closed these: `dma_transpose` is **documented fp32-ineligible** (needs a
  2-byte dtype); `nc_transpose(engine=vector)` is limited to [32,32] (each [128,128]
  → 16 sub-transposes → far more Vector ops). Transposes are 5% of PE anyway.
  Record as investigated-and-closed; do not re-probe.

---

## 6. Numerical safety — the fallback ladder

If the on-device full-5-seed rel-L2 for all-3-bf16x2 exceeds the gate (unexpected,
given 7.7e-6 offline + negligible fp32 floor), step down the offline-gated ladder,
each variant already PASS in the sim, trading a little speed for margin:

1. **all 3 bf16x2** — 7.7e-6 (target).
2. **up+down bf16x2, gate fp32** — 6.3e-6 (gate feeds the SiLU → most error-
   sensitive; keeping it fp32 is the natural first retreat).
3. **only down bf16x2** — 4.4e-6 (down is 1/3 of PE MACs; smallest, safest win).
4. **fp32 (v1)** — always the correctness floor to fall back to.

All limb construction uses round-to-nearest-even (`nl.copy(dtype=nl.bfloat16)`), the
exact cast the sim models; the residual `a - a_hi` is exact in fp32 for these O(1)
normals. Never regress below v1's PASS.

---

## 7. Implementation sketch — `swiglu_v2` (bf16x2, all-3)

Structurally v1 with limb splits inserted; the loop nest and layout are unchanged.

1. **Setup:** identity [128,128] loaded once (fp32, for exact transposes), as in v1.
2. **Per M-tile:**
   - Load x tile `[m_in,1024]`; transpose once into 8 fp32 xT sub-tiles (exact
     identity matmul, `is_transpose=True`), then split each into
     `xT_hi/xT_lo` (bf16). These limbs are **shared by up and gate**.
   - **up / gate**, per N-chunk (512), K-accum over 8 K-tiles into two fp32 PSUM
     banks: load fp32 `w_up`/`w_gate` chunk `[k_in,512]`, build `w_*_hi/w_*_lo`
     (bf16) on-chip, issue the **3 bf16 products** (`xT_hi@w_hi + xT_hi@w_lo +
     xT_lo@w_hi`) into the accumulator.
   - PSUM→SBUF copy; fused `nl.silu(gate)`; `nl.multiply` → `h_sbuf[128,3072]` fp32
     resident (identical to v1).
   - Transpose h into 24 fp32 hT sub-tiles; split into `hT_hi/hT_lo` (bf16).
   - **down**, per K-out chunk (512), N-accum over 24 N-tiles: load fp32 `w_down`
     chunk `[n_in,512]`, build `w_down_hi/w_down_lo`, issue the **3 bf16 products**
     into the accumulator; copy; store to `v5`.
3. **dtypes:** all limbs `nl.bfloat16`; all PSUM accumulation and the SiLU/multiply
   stay fp32; transposes stay fp32.

Weights are still streamed (B=1) — the split changes arithmetic, not the loop
structure, keeping the diff from v1 small and reviewable.

---

## 8. Deliverable and success criteria

- **Deliverable:** `runs/swiglu_v2.py` (bf16x2 all-3-GEMMs), the offline sim
  (`runs/offline_bf16_split_sim.py`, already written) + its evidence
  (`profile/swiglu_offline_bf16x2_sim.txt`), a `benchmark.csv` row and a
  `candidates.jsonl` entry with parent `swiglu_v1`.
- **Score:** `python3
  ../../verify.py --op swiglu --candidate runs/swiglu_v2.py --fast` (then drop
  `--fast` for the promoting 5-seed run).
- **Success:** full-5-seed PASS (rel-L2 < 2e-5; expect ~7.7e-6) **and** speedup
  > 1.0x (projected ~1.13–1.20x). Keep v1 as the fp32 fallback.
- **Evidence to capture:** before/after latency, MFU (expect a clear rise from 44%),
  PE/DMA %, HBM bytes; the fallback-ladder decision if triggered.
- **Iteration budget:** D1 ≤5 iters (primary); D2 ≤2 iters (only if the post-D1
  profile shows DMA climbing off "hidden"); D3/D4 are recorded rejects, not explored.

--- Original Design Draft End ---
