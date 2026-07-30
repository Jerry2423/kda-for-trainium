# adamw Phase 3 — Complete the Bimodal Burst-Width Resonance Sweep (Regime / Shape Specialization)

## Goal Description

Close the one open question phase 2 left for the `adamw` operator (fused AdamW optimizer
step, M=10944, N=2048, fp32, pure elementwise): **is `CH=1216` the global peak of a
bimodal DMA-saturation resonance curve, or only a local optimum?**

The phase-2 winner `runs/adamw_v2_ch1216.py` (**2.330x, 0.5601 ms**) is a mask-free
`(128, ITERS, CH)` reshape-view stream that is **DMA-bound at ~99%** on an **immovable
448 MB traffic floor** (HBMrd 359 MB + HBMwr 90 MB), sitting at ~0.9973× of the
silu-anchored streaming roofline (~799.5 GB/s). Because `M·N = 22413312 = 128 · 175104`
**exactly** and `175104 = 2¹⁰·3²·19`, the reshape-view is **perfectly rectangular**:
zero edge tiles, zero masks, all 128 partition lanes live. The classic phase-3 levers
(edge-tile specialization, partition/free-split regimes, mixed tile-size regimes) therefore
have **no surface**; "regime specialization" collapses to the burst-width `CH` sweep.

Phase 2 tested only 4 of the 12 in-band `CH` divisors and stopped after one adjacent
bracket, under a bounded-sweep rule (`BL-20260709-finer-tiling-harvests-dma-bubble`) whose
stopping condition assumes a **smooth unimodal** curve. adamw's curve is explicitly
**bimodal** (two 99% lobes at `CH=1024` and `CH=1216` straddle an 88% trough at `CH=1152`),
which violates that premise. Phase 3's substantive contribution is to **sweep the untested
in-band divisors** and reach one of two equally valid, successful outcomes:

- **(a) Terminal closure (expected):** confirm `adamw_v2_ch1216` (2.330x) as the
  **in-band lattice-complete peak** and close the operator as terminal, with `adamw_v1`
  (2.112x) retained as the documented fp32 fallback; or
- **(b) Promotion (unlikely):** if a screened width is **materially** faster than a fresh
  same-session `ch1216` anchor (beyond measured profiler jitter) at DMA saturation, promote
  it via the interleaved 5-seed A/B/A/B/A protocol.

The fp32 traffic-floor kernel stays the default. This is a **pure scheduling** phase — every
screen keeps the 448 MB traffic floor and is byte-identical to `ch1216` except the `CH`/`ITERS`
constants — so there is effectively **zero correctness risk** (rel-L2 stays 3.42e-8 « the
2e-5 gate, layout-invariant).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. Correctness is NKIBench's relative-L2 gate
(`verify.py`, rel_tol 2e-5) across seeds `[0,21,42,63,84]`; `--fast` uses seed 42 only.

- AC-1: **Phase-3 structural surface is documented as burst-width-only.** The phase-3
  artifact records that adamw's reshape-view is edge-free and mask-free (`CH | 175104`
  exactly, `ITERS·CH = 175104`) with all 128 partition lanes live, so the classic phase-3
  levers — edge-tile / partial-tail specialization, partition/free-split regimes,
  mixed tile-size regimes — have **no surface**, and "regime specialization" reduces to the
  burst-width `CH` sweep.
  - Positive Tests (expected to PASS):
    - The artifact states the divisor-exactness proof (`128 · 175104 = 22413312 = M·N`;
      `175104 = 2¹⁰·3²·19`) and concludes every chunk is a full `[128, CH]` rectangle.
    - The artifact lists edge-tile spec, partition/free split, and mixed tile-size regimes
      as disarmed with the "no surface" reason for each.
  - Negative Tests (expected to FAIL):
    - A phase-3 direction that proposes edge-tile or partial-tail specialization work
      (there is no partial tail to specialize).
    - A phase-3 direction that proposes splitting the partition dimension below 128 lanes
      (that only underutilizes DMA).

- AC-2: **Untested in-band `CH` divisors are screened, changing only `CH`/`ITERS`.** Each
  screened candidate is produced by reusing the exact `adamw_v2_ch1216.py` kernel with only
  the `CH` constant changed (and `ITERS = 175104 // CH`), plus its header comment; the fused
  6-op compute chain, folded algebra, load/store structure, and dtypes are byte-for-byte
  unchanged. Screens run via `verify.py --op adamw --candidate runs/<file>.py --fast`.
  - AC-2.1: The **in-band lattice** is defined as `CH ∈ {512, 576, 608, 684, 768, 912,
    1024, 1152, 1216, 1368, 1536, 1824}` (all exact divisors of 175104 with per-partition
    burst `CH·4` in `[2 KB, 8 KB]`). The screens cover the untested members of this set
    (`512, 576, 608, 684, 768, 912, 1368, 1824`), at the breadth resolved in DEC-1.
    - Positive: every screened `CH` is an exact divisor of 175104 with `ITERS·CH = 175104`.
    - Negative: a screen with `ITERS·CH ≠ 175104` (non-exact divisor) — rejected as it would
      require a mask/edge and break the byte-identical invariant.
  - AC-2.2: **Architecture-anchor priority.** `CH=1368` (`ITERS=128`, loop trip count equals
    the partition count) and `CH=912` (`ITERS=192`, the gap below the 1024 lobe) are screened
    first as the most likely locations of a missed co-lobe, followed by `CH=768` (3.00 KB
    round-KB anchor).
    - Positive: the first screens include `CH=1368`, `CH=912`, `CH=768`.
    - Negative: skipping the two gap probes adjacent to the known 99% lobes.
  - Positive Tests (expected to PASS):
    - A screened kernel `diff`s against `adamw_v2_ch1216.py` in only the `CH`/`ITERS`
      constant lines and the header comment.
    - Each screen records `CH, ITERS, burst KB, latency_ms, speedup, DMA%, Vec%, Scl%,
      HBMrd_MB, HBMwr_MB` (and rel-L2 status) in `benchmark.csv` / `candidates.jsonl`.
  - Negative Tests (expected to FAIL):
    - A screen that alters the compute chain, folded constants, dtype, or load/store shape
      (no longer a pure `CH` change → confounds the resonance signal).

- AC-3: **Screens are judged against a fresh, bracketed same-session `ch1216` anchor.**
  Every screen is compared against `ch1216` re-measured `--fast` **in the same session**,
  not against the frozen historical 0.5601 ms, because sub-percent deltas are within remote
  profiler drift. The `ch1216` anchor is measured more than once and bracketed around the
  screens (e.g. anchor → screens → anchor) so anchor drift is observable.
  - Positive Tests (expected to PASS):
    - The session re-measures `ch1216 --fast` at least twice and reports each screen's delta
      versus a nearby anchor.
    - The observed anchor-to-anchor spread is recorded as the session jitter estimate.
  - Negative Tests (expected to FAIL):
    - Claiming a screen "wins" using only the historical 0.5601 ms constant as the baseline.
    - A single lone `ch1216` anchor with no repeat (jitter cannot be estimated).

- AC-4: **Traffic floor invariant holds at every `CH`.** Every screened candidate reports
  HBMrd ≈ 359 MB and HBMwr ≈ 90 MB (≈ 448 MB total, 4-read/1-write), within profiler
  rounding. The sweep is a pure scheduling lever and must never touch HBM byte counts.
  - Positive Tests (expected to PASS):
    - Every screen holds the 448 MB floor (read-4/write-1) within profiler rounding.
  - Negative Tests (expected to FAIL):
    - A `CH` whose HBMrd/HBMwr departs materially from 359/90 MB → flagged as an accidental
      extra pass or SBUF spill, disqualifying that width regardless of latency.

- AC-5: **"Material" is defined by measured jitter, not by the roofline headroom.** A screen
  is a **material** candidate (D2 trigger) only if it beats the nearby same-session `ch1216`
  `--fast` anchor by **more than the measured session jitter** `J_measured` (the observed
  anchor-to-anchor spread from AC-3), **and** its HBM traffic floor holds (AC-4). DMA% is
  used as a **supporting/diagnostic** signal (a material win is expected to coincide with
  DMA at or above the incumbent's, ~99%), **not** as a hard reject on its own: a candidate
  that is clearly faster with an invariant traffic floor is not rejected solely because its
  displayed DMA% rounds to, e.g., 98.8% instead of 99%.
  - Positive Tests (expected to PASS):
    - A width beating the nearby anchor by more than `J_measured` with the floor intact
      triggers D2 (AC-6).
    - A width within `J_measured` of the anchor is recorded as a **tie** and does **not**
      trigger promotion (incumbent wins ties).
  - Negative Tests (expected to FAIL):
    - Triggering promotion on a within-jitter "win".
    - Rejecting a clearly-faster, traffic-floor-intact candidate solely because DMA% displays
      one rounding step below 99%.
    - Defining materiality as "beyond the ~0.3% roofline headroom" (self-contradictory: a
      candidate cannot be both meaningfully faster than the incumbent and beyond an asserted
      hard ceiling — materiality is a measured-noise question, the roofline is only the
      *expectation* that a material win is unlikely).

- AC-6: **Conditional promote-test (only on a material candidate).** A material candidate
  from AC-5 is promote-tested with the phase-2 protocol: interleaved **full 5-seed
  A/B/A/B/A** (A = `ch1216`, B = candidate), with statistics defined as: `Abar`, `Bbar` =
  means of the A and B latencies; four gates:
  1. `Bbar < Abar − J_promote`, where `J_promote = max(0.0002 ms, observed A-spread this run)`
     (the promote noise band is the larger of the historical floor and the actual A-spread);
  2. every B < max(A);
  3. traffic floor intact (HBMrd 359 + HBMwr 90 = 448 MB on **every** B — no accidental extra pass);
  4. 5-seed L2 gate PASS on every run.
  Promote iff all four gates pass; otherwise `adamw_v2_ch1216` stays the winner.
  - Positive Tests (expected to PASS):
    - A candidate passing all four gates is promoted and recorded with the full A/B ledger.
  - Negative Tests (expected to FAIL):
    - Promoting when any single gate fails (e.g. `Bbar` within `J_promote` of `Abar`, or one
      B ≥ max(A), or traffic floor broken on any B, or any seed L2 fail).

- AC-7: **Terminal closure is a first-class successful outcome.** If no screened width is
  material (AC-5), the phase-3 result is recorded as a **success**: `adamw_v2_ch1216`
  confirmed as the **in-band lattice-complete peak**, operator closed as terminal, with the
  raw profiler rows for every screened `CH` preserved in the evidence and `adamw_v1`
  (2.112x) retained as the fp32 fallback.
  - Positive Tests (expected to PASS):
    - A no-new-material-win sweep is documented as a successful terminal closure, with the
      full screened-`CH` result table preserved.
  - Negative Tests (expected to FAIL):
    - Treating "no new promotion" as a phase-3 failure.
    - Forcing a within-jitter marginal candidate to be promoted just to produce a "win".
    - Claiming an unqualified "global peak" without stating the closure is scoped to the
      justified in-band lattice (see DEC-2 for the out-of-band justification).

- AC-8: **Correctness is preserved and finally re-verified on 5 seeds.** rel-L2 stays far
  under 2e-5 at every screened `CH` (layout-invariant: pure elementwise op, exact
  `(M,N) ↔ (128, ITERS, CH)` reshape round-trip). The final candidate — whether `ch1216`
  is confirmed or a new lobe is promoted — passes the **full 5-seed** gate `[0,21,42,63,84]`
  before the phase closes.
  - Positive Tests (expected to PASS):
    - The final candidate passes `verify.py` on all 5 seeds (`l2_norm_passed` per seed).
  - Negative Tests (expected to FAIL):
    - Any seed failing the rel-L2 gate on the final candidate.
    - Promoting a candidate validated only on `--fast` (seed 42) without the 5-seed run.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
Screen the **full untested in-band `CH` lattice** (`512, 576, 608, 684, 768, 912, 1368,
1824` — all 8 untested exact divisors of 175104 with burst 2–8 KB/partition) as `--fast`
runs, each byte-identical to `adamw_v2_ch1216.py` except the `CH`/`ITERS` constants and
header, judged against bracketed same-session `ch1216` anchors with a measured jitter band;
if (unlikely) a **material** faster 99%-lobe width surfaces, promote it via the interleaved
5-seed A/B/A/B/A protocol with all four gates; otherwise record `adamw_v2_ch1216` as the
in-band lattice-complete terminal peak with all screened rows preserved. `adamw_v1` remains
the fp32 fallback throughout.

### Lower Bound (Minimum Acceptable Scope)
Screen at least the three architecture-anchor / gap-probe widths — `CH=1368` (ITERS=128 =
partition count), `CH=912` (gap below the 1024 lobe), and `CH=768` (3.00 KB anchor) — the
locations most likely to hide a missed co-lobe, each byte-identical to `ch1216` except the
`CH`/`ITERS` constants, judged against a fresh same-session `ch1216` `--fast` anchor with
the traffic floor confirmed; conclude the bimodal question over those probes and record the
outcome (confirm `ch1216` terminal, or trigger D2 on a material win). This is the draft's
bounded-sweep floor.

### Allowed Choices
- Can use: changing only the `CH` (and derived `ITERS = 175104 // CH`) constant of the
  promoted `adamw_v2_ch1216.py` kernel; the reshape-view `(128, ITERS, CH)` streaming shape;
  `nl.affine_range` for the flat iteration; the existing 6-op fused chain (2 Scalar
  `activation` + 4 Vector `scalar_tensor_tensor`) and folded algebra, unchanged; `verify.py`
  `--fast` for screens and the full 5-seed gate for promotion; the interleaved A/B/A/B/A
  promote protocol.
- Cannot use: any `CH` that is not an exact divisor of 175104 (would reintroduce a
  mask/edge); changes to the compute chain, folded constants, dtype, or load/store shape
  during the sweep (confounds the resonance signal); bf16 / mixed-precision or any traffic-cut
  scheme (no reduction to average error under 2e-5, and the HBM tensors are fp32-owned);
  edge-tile / partial-tail specialization (no edges exist); partition/free-split below 128
  lanes; manual double-buffer / ping-pong via `sequential_range` (silu-precedent regression,
  low expected value at DMA=99%); folded-chain algebra-scheduling reorder (reserved, disarmed
  — see Feasibility Hints).

> **Note on Deterministic Designs**: This draft is highly deterministic — the only free
> variable is the `CH` divisor and the sweep breadth. Upper and lower bounds differ only in
> **how many** untested divisors are screened (the DEC-1 breadth question); the *method*
> (byte-identical `CH`-only screens, same-session bracketed anchors, jitter-defined
> materiality, 5-seed promote gate, incumbent wins ties) is fixed per this specification.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

1. **Generate a screen kernel per untested `CH`.** Copy `runs/adamw_v2_ch1216.py` to
   `runs/adamw_v2_ch<N>.py`, change the single line `P, FLAT, CH = 128, 175104, 1216` to the
   new `CH`, and update the header comment's `CH`/`ITERS`/burst annotation. `ITERS` is
   derived (`FLAT // CH`) and asserted exact. Nothing else changes — confirm with `diff`.
   (Precedent: the phase-2 `adamw_v2_ch1024/1152/1536.py` variants already differ from
   `ch1216.py` only in these lines.)

2. **Screen order (resonance-mapping, per AC-2.2):** `CH=1368` (ITERS=128) and `CH=912`
   (ITERS=192) first — the gap/anchor probes adjacent to the known 99% lobes — then `CH=768`
   (3.00 KB). Extend to the finer anchors (`684, 608, 576, 512`) and the wide anchor (`1824`)
   per the DEC-1 breadth decision.

3. **Anchor and measure:** interleave/bracket fresh `ch1216 --fast` anchors around the
   screens (anchor → probes → anchor). For each screen record `CH, ITERS, burst KB, latency,
   speedup, DMA%, Vec%, Scl%, HBMrd, HBMwr`. Latency tracks DMA% ~1:1 (phase-2 evidence), so
   DMA% is the fast diagnostic — but the **decision** signal is latency-vs-anchor plus the
   traffic floor (AC-5).

4. **Classify each screen:** *tie* (within `J_measured` of the anchor → keep 1216, per
   AC-5), *regression* (slower → reject, maps the trough), or *material* (faster beyond
   `J_measured` with floor intact → D2).

5. **If a material candidate appears (D2):** run the interleaved 5-seed A/B/A/B/A promote
   test (AC-6). Promote iff all four gates pass.

6. **Close:** if no material win, write the phase-3 exit decision documenting the in-band
   lattice-complete confirmation of `ch1216` (AC-7); else document the promotion. Record
   evidence in `benchmark.csv`, `candidates.jsonl` (DAG parent links), and a
   `docs/phase3-exit-decision.md` sibling of `docs/phase2-exit-decision.md`.

**Disarmed lever (reserved, do NOT implement unless all `CH` probes fail AND a profiler hint
changes):** reordering the folded compute chain (`g² → 999v+g² → rsqrt → multiply` has a
serial dependency; `9m+g` is independent). Expected upside is low (Vec 72% is already hidden
under DMA 99%) and it carries codegen/correctness risk — it is documented only for
completeness, not scheduled.

### Relevant References
- `runs/adamw_v2_ch1216.py` — the promoted phase-2 kernel; the exact template each screen
  copies (change only `CH`/`ITERS` + header).
- `runs/adamw_v2_ch1024.py`, `runs/adamw_v2_ch1152.py`, `runs/adamw_v2_ch1536.py` — phase-2
  screen variants; concrete precedent that only the `CH`/`ITERS` lines differ.
- `runs/adamw_v1.py` — the masked row-tile phase-1 kernel; the documented fp32 fallback.
- `docs/phase2-exit-decision.md` — the promote-test protocol and the bimodal-curve finding
  this phase completes.
- `../../AccelOpt/NKIBench/reference/adamw_M10944_N2048_numpy_1.py` — the numpy reference
  (never edited); source of the folded algebra and the `v = |normal| ≥ 0` fact that licenses
  dropping eps.
- `verify.py` (repo root) — scoring; `--fast` = seed 42, full gate = 5 seeds `[0,21,42,63,84]`
  at rel_tol 2e-5, gated on per-seed `l2_norm_passed`.
- `benchmark.csv`, `candidates.jsonl` — evidence rows / candidate DAG for this workspace.

## Dependencies and Sequence

### Milestones
1. **Structural framing (AC-1).** Record the edge-free / mask-free / 128-lanes-live structural
   argument and the reduction of "regime specialization" to the burst-width sweep.
   - Phase A: State the divisor-exactness proof and the disarmed classic levers.
   - Phase B: Fix the in-band lattice set and the out-of-band justification (DEC-2).
2. **Burst-width sweep (AC-2, AC-3, AC-4, AC-8).** Screen the untested in-band `CH` divisors.
   - Step 1: Resolve DEC-1 (sweep breadth) — determines how many divisors are screened.
   - Step 2: Generate `CH`-only screen kernels for the chosen widths; `diff`-confirm
     byte-identity to `ch1216`.
   - Step 3: Run bracketed `ch1216 --fast` anchors + screens; record all metrics; confirm the
     448 MB traffic floor and rel-L2 at each width.
3. **Materiality classification (AC-5).** Compute `J_measured` from the anchor spread; classify
   each screen as tie / regression / material.
4. **Conditional promotion (AC-6, AC-8).** Only if a material candidate exists: interleaved
   5-seed A/B/A/B/A promote-test with the four gates; final 5-seed correctness re-verify.
5. **Closure (AC-7).** Write the phase-3 exit decision: either in-band lattice-complete
   terminal confirmation of `ch1216`, or the promotion record. Preserve all screened rows.

Dependency notes: Milestone 2.Step 1 (DEC-1) gates the breadth of 2.Step 2. Milestone 4 is
reached only when Milestone 3 finds a material candidate; otherwise the sequence goes
2 → 3 → 5 (terminal closure). Milestone 1 can proceed in parallel with 2 (it is documentation
of already-established structure).

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Document the edge-free / mask-free / 128-lanes-live structural argument and the reduction of regime specialization to the burst-width sweep; list the disarmed classic levers | AC-1 | coding | - |
| task2 | Fix the in-band `CH` lattice set (12 divisors) and record the out-of-band exclusion justification (DEC-2) | AC-1, AC-2.1 | coding | task1 |
| task3 | Resolve DEC-1 (sweep breadth) with the user; set the list of `CH` widths to screen | AC-2 | coding | task2 |
| task4 | Generate `CH`-only screen kernels (`runs/adamw_v2_ch<N>.py`) for the chosen widths; `diff`-confirm byte-identity to `ch1216` except `CH`/`ITERS`/header | AC-2, AC-2.1 | coding | task3 |
| task5 | Run bracketed `ch1216 --fast` anchors + `--fast` screens; record CH/ITERS/burst/latency/speedup/DMA%/Vec%/Scl%/HBMrd/HBMwr/rel-L2; confirm 448 MB floor per width | AC-2.2, AC-3, AC-4, AC-8 | coding | task4 |
| task6 | Compute `J_measured` from anchor spread; classify each screen as tie / regression / material | AC-5 | coding | task5 |
| task7 | If a material candidate exists: interleaved 5-seed A/B/A/B/A promote-test with the four gates + final 5-seed correctness re-verify | AC-6, AC-8 | coding | task6 |
| task8 | Write `docs/phase3-exit-decision.md`: in-band lattice-complete terminal confirmation of `ch1216` (or the promotion record); preserve all screened rows; update `benchmark.csv` / `candidates.jsonl` | AC-7 | coding | task6, task7 |

## Claude-Codex Deliberation

### Agreements
- For a pure-elementwise fp32 traffic-floor kernel already at DMA 99%, burst-width (`CH`)
  resonance is the only high-EV phase-3 surface; edge-tile and partition/free-split levers
  are correctly disarmed (`ITERS·CH = 175104` exact, no masks, all 128 lanes live).
- Same-session anchoring is necessary; the historical 0.5601 ms constant must not be used to
  claim sub-percent wins.
- The promotion bar is appropriately conservative: traffic-floor invariant, 5-seed
  correctness, incumbent wins ties.
- Terminal closure with no new promotion is a valid, successful phase-3 outcome — *provided*
  the closure claim is scoped to the justified in-band lattice.
- bf16 / mixed precision stays out of scope (no reduction to average quantization error under
  the 2e-5 gate; HBM tensors are fp32-owned so bf16 cannot cut bytes it doesn't own).
- Manual double-buffer / ping-pong is low-expected-value after affine-range DMA=99% (framed as
  low-EV given the silu precedent, not as a universal prohibition).

### Resolved Disagreements
- **Materiality definition (Codex first-pass + round 2):** The draft's D2 trigger ("materially
  below 0.5601 ms, beyond the ~0.3% roofline headroom") is self-contradictory — a candidate
  cannot be both meaningfully faster than the incumbent *and* beyond an asserted hard ceiling.
  **Resolution:** materiality is defined by **measured same-session jitter** `J_measured`
  (AC-5), not by the roofline headroom; the roofline is reframed as the *expectation* that a
  material win is unlikely, not the threshold that defines one. Adopted.
- **Anchor baseline (Codex):** compare screens to a **fresh, bracketed** same-session `ch1216`
  anchor rather than the frozen historical number (AC-3). Adopted.
- **Jitter / `J` definition (Codex):** the historical `J=0.0002 ms` is too tight to be the sole
  materiality margin at this latency. **Resolution:** screen materiality uses `J_measured`
  (observed anchor spread); the promote gate uses `J_promote = max(0.0002 ms, observed A-spread)`
  (AC-5, AC-6). Adopted.
- **DMA% gate strictness (Codex UNRESOLVED → resolved by Claude):** a strict `DMA ≥ 99%`
  reject could wrongly kill a clearly-faster, traffic-floor-intact candidate whose DMA% merely
  rounds to 98.8%. **Resolution (materially safer):** DMA% is a **supporting/diagnostic** signal;
  the decision signals are latency-vs-anchor and the traffic floor (AC-5). Adopted — recorded as
  resolved rather than carried to the user because one option is clearly safer.
- **Closure-claim scope (Codex):** "global peak" overclaims — the sweep covers only the in-band
  divisor lattice. **Resolution:** the terminal claim is "**in-band** lattice-complete peak" with
  an explicit out-of-band exclusion justification (AC-7, DEC-2). Adopted.
- **Promotion statistics (Codex):** define `Abar`/`Bbar` as means, keep "every B < max(A)", and
  define `J_promote` explicitly (AC-6). Adopted.

### Convergence Status
- Final Status: `converged` (no high-impact Claude/Codex disagreement remains; all
  `REQUIRED_CHANGES` from the round-2 review are incorporated). Both carried decisions have now
  been **RESOLVED by the author** (2026-07-12): DEC-1 = exhaustive 8 (overrides the draft's
  cap≈5), DEC-2 = keep the reasoned in-band scope. No open decisions remain — proceed directly.

## Pending User Decisions

> **RESOLVED (2026-07-12, author call).** Both decisions below are settled; nothing here
> blocks execution. Run the sweep at the resolved breadth/scope directly.
> - **DEC-1 → EXHAUSTIVE 8.** Screen all 8 untested in-band divisors
>   (`512, 576, 608, 684, 768, 912, 1368, 1824`), anchors `1368, 912, 768` first. This
>   **overrides the original draft's "cap ≈ 5"** — the bimodal curve makes full in-band
>   coverage worth the ~8 cheap `--fast` screens; the closure claim becomes
>   "in-band lattice-complete".
> - **DEC-2 → KEEP REASONED IN-BAND SCOPE.** Scope the closure to `CH ∈ [512, 1824]`
>   (burst 2–7 KB/partition) with the burst-efficiency exclusion justification; do NOT
>   probe out-of-band anchors. Label the result "in-band lattice-complete".

- DEC-1: **Sweep breadth — exhaustive 8 untested divisors vs the draft's bounded ~5.**
  - Claude Position: Lean **exhaustive** (all 8 untested in-band divisors: `512, 576, 608, 684,
    768, 912, 1368, 1824`). `--fast` screens are cheap, the curve is explicitly bimodal so a
    bounded probe can miss a lobe, and full coverage is what upgrades the claim from "local
    optimum" to "in-band lattice-complete". Falls back gracefully: the three anchor/gap probes
    (`1368, 912, 768`) run first regardless, so a bounded run is a strict prefix.
  - Codex Position: **Exhaustive** (strong). "Use the exhaustive 8 untested widths because it is
    cheap and materially improves the closure claim"; bounded probing is weaker on a bimodal
    curve.
  - Tradeoff Summary: Exhaustive costs ~8 cheap `--fast` runs (plus bracketing anchors) and
    definitively closes the bimodal question; bounded (~5, the draft's explicit "do not chase all
    8 … cap ≈ 5") is cheaper but risks leaving a lobe untested and can only claim a probe-bounded
    optimum. **This conflicts with the original draft's explicit cap**, so it needs the author's
    call. Recommendation: exhaustive (both reviewers agree; low marginal cost).
  - Decision Status: `RESOLVED` — **EXHAUSTIVE 8** (author, 2026-07-12; overrides draft cap≈5)

- DEC-2: **Out-of-band divisor exclusion — confirm the in-band `[512, 2048]` (2–8 KB/partition)
    boundary.**
  - Claude Position: Exclude `CH < 512` (burst < 2 KB/partition underfills the DMA descriptor
    cadence and only lengthens the loop) and `CH > 1824` (burst > 7 KB gives `ITERS < 96`, too
    shallow a pipeline; the measured wide side already declines at `CH=1536`). Scope the closure
    claim to this in-band lattice.
  - Codex Position: The claim must either "cite prior exclusion of out-of-band widths or rename
    the result to 'in-band lattice-complete'." (Codex did not dispute the band itself, only the
    need to justify it.)
  - Tradeoff Summary: The in-band boundary is a reasoned burst-efficiency argument, not a proof;
    screening a couple of out-of-band anchors (e.g. `CH=456` at 1.78 KB, `CH=2432` at 9.5 KB — if
    exact divisors) would harden the boundary at small extra cost. Default is to keep the reasoned
    in-band scope and label the closure "in-band lattice-complete". Confirm whether that scope is
    acceptable or the boundary should itself be probed.
  - Decision Status: `RESOLVED` — **KEEP REASONED IN-BAND SCOPE** (author, 2026-07-12)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. the kernel header already
  describes the reshape-view shape, folded algebra, and `CH`/`ITERS` burst annotation — keep
  that domain-level style).
- Each screen kernel is a byte-identical copy of `adamw_v2_ch1216.py` differing only in the
  `CH`/`ITERS` constants and the header comment's `CH`/`ITERS`/burst annotation.

## Output File Convention

This plan is the main output file (`docs/plan-phase3.md`). The project's
`alternative_plan_language` is empty, so **no translated language variant is written**.

--- Original Design Draft Start ---

# adamw (M10944 N2048, fp32) — Phase 3 implementation draft (regime / shape specialization)

## Starting point

Phase-2 winner **`runs/adamw_v2_ch1216.py`** — the mask-free reshape-view stream:
- **0.5601 ms, 2.330x** over baseline (full 5-seed PASS; `--fast` 0.5614 ms / 2.325x).
- Structure: reshape the contiguous `(10944, 2048)` buffer to a **pure-stride, no-DMA**
  `(128, ITERS=144, CH=1216)` view and walk one flat `nl.affine_range(144)`; per chunk
  4 **mask-free** loads → the 6-op fused chain (2 Scalar `activation` square/rsqrt +
  4 Vector `scalar_tensor_tensor`) → 1 mask-free store. Folded algebra
  `new_theta = 0.99999·theta − 0.001·(9m+g)·rsqrt(999v+g²)` (eps dropped, `v_hat>0`),
  byte-for-byte unchanged from `adamw_v1`.
- Profiler digest: **DMA 99%, Vec 72%, Scl 34%, PE 0%**; HBMrd 359 MB, HBMwr 90 MB.
- `adamw_v1` (2.112x, masked row tiles) kept as the documented fp32 fallback.

Phase 2 established the headline fact that governs this entire phase: the kernel is
**DMA-bound at the achieved streaming roofline**, on an **immovable traffic floor**.

## Where the time goes: at the roofline, on an immovable floor

Read the promoted numbers as a roofline (this is the phase-3 diagnosis, not a new claim):

| metric | ch1216 | meaning |
|---|---|---|
| DMA active | **99%** | the sole constraint; saturated |
| Vec active | 72% | #2 engine, **hidden** under DMA (0.72·0.5601 = 0.40 ms « 0.99·0.5601 = 0.55 ms) |
| Scl active | 34% | the two nonlinearities (square, rsqrt), off Vector |
| PE / MFU | 0% | no matmul (irrelevant) |
| HBMrd | 359 MB | 4 × 89.66 MB = **read-once** for theta,g,m,v (no re-fetch) |
| HBMwr | 90 MB | 1 × 89.66 MB = **write-once** for new_theta (no spill) |

- **Traffic floor = 448.3 MB** (4R + 1W). At silu's achieved roofline on this trn2
  profiler (~799.5 GB/s) that floor is **0.5616 ms → a 2.324x hard ceiling**. The
  measured 0.5601 ms is **0.9973× of that floor** — effBW 799.6–801.6 GB/s, i.e.
  *dead-on the roofline*. There is ≈**0.3 % headroom** left, and it is a hardware
  ceiling, not a schedulable bubble.
- **There is no traffic lever.** All four inputs are genuinely read (every element
  feeds the update), the output is written once, and the HBM tensors are fp32 supplied
  by the harness. bf16 would neither cut HBM bytes (the tensors *are* fp32 in HBM) nor
  survive the 2e-5 L2 gate on a pure elementwise op with **no reduction** to average the
  error down — the opposite of the rmsnorm/matmul siblings where K-averaging licensed a
  bf16x2 split. Traffic is pinned at 448 MB at every tiling phase 2 measured.

## Phase-3 lens: the reshape-view is shape-homogeneous → no edge/regime surface

Phase 3's mandate is "analyze where time goes across the tensor's **structure** and
specialize only where a measured win justifies the complexity (tile-size regimes,
partition/free splits, edge tiles)." The decisive structural finding for adamw:

`M·N = 10944·2048 = 22413312 = 128 · 175104` **exactly**, and `175104 = 2¹⁰·3²·19`, so
`CH=1216 | 175104` **exactly** (ITERS=144). The reshape-view therefore homogenizes the
problem into a **perfectly rectangular** `(128, ITERS, CH)` stream:

- **Zero edge tiles** — every chunk is a full `[128, CH]` rectangle; there is no partial
  tail to specialize (contrast `adamw_v1`, which carried a `row<10944` predicate on
  the 64-valid-row last tile).
- **Zero masks** — no DMA in the whole kernel is predicated.
- **All 128 partition lanes live** — the partition dim is already at hardware max; a
  partition/free-split regime cannot help (fewer partitions only underutilizes DMA).
- **Every tile byte-identical in structure** — there is nothing heterogeneous to
  regime-specialize.

**Conclusion:** the classic phase-3 levers — edge-tile specialization, partition/free
split regimes, mixed tile-size regimes — have **no surface** on adamw. The *only*
"regime" axis that exists is the burst width **CH**, which was already the phase-2
lever. "Regime specialization" collapses to the burst-width sweep. This is the honest
terminal structural statement (cf. bmm/transpose_matmul phase 3: "shape edge-free →
the remaining question is numerical/scheduling, not tile geometry").

## The one open question phase 2 left: is CH=1216 the *global* resonance peak?

Phase 2 documented that adamw's latency-vs-burst-width curve is **non-monotone /
bimodal** — distinct from silu's smooth unimodal turn — with latency tracking DMA% 1:1:

| CH | ITERS | burst/part | latency | speedup | DMA% | role (phase 2) |
|----|-------|-----------|---------|---------|------|----------------|
| 1024 | 171 | 4.00 KB | 0.5921 | 2.204x | **99** | 99% lobe |
| 1152 | 152 | 4.50 KB | 0.6834 | 1.910x | 88 | trough between the lobes |
| **1216** | **144** | **4.75 KB** | **0.5614** | **2.325x** | **99** | **99% lobe (best)** |
| 1536 | 114 | 6.00 KB | 0.6515 | 2.003x | 91 | wider trough |

Phase 2 declared 1216 the "interior optimum bracketed below on both sides" and **stopped
the sweep after one adjacent bracket** (1152), citing the bounded-sweep rule
`BL-20260709-finer-tiling-harvests-dma-bubble`. **That rule's stopping condition assumes
a smooth unimodal curve** — one bracket on each side proves a peak only when the curve is
monotone away from it. adamw's curve is explicitly *bimodal*: two 99% lobes (1024, 1216)
straddle an 88% trough (1152). On a resonant curve, a single adjacent bracket does **not**
rule out a *third* 99% lobe elsewhere in the band. Only **4 of the 12 in-band divisors**
were tested, so 1216 is proven a *local* optimum but **not the global** saturation peak.

Full in-band divisor lattice (`175104`, burst 2–8 KB/partition), phase-2 coverage marked:

| CH | ITERS | burst/part | phase-2 status |
|----|-------|-----------|----------------|
| 512 | 342 | 2.000 KB | untested (fine anchor) |
| 576 | 304 | 2.250 KB | untested |
| 608 | 288 | 2.375 KB | untested |
| 684 | 256 | 2.672 KB | untested |
| 768 | 228 | 3.000 KB | untested (finer gap probe) |
| 912 | 192 | 3.562 KB | **untested — gap between 768 and the 1024 lobe** |
| 1024 | 171 | 4.000 KB | 99% lobe (2.204x) |
| 1152 | 152 | 4.500 KB | 88% trough (1.910x) |
| **1216** | **144** | **4.750 KB** | **99% lobe — BEST (2.325x)** |
| 1368 | 128 | 5.344 KB | **untested — gap between the 1216 lobe and the 1536 trough** |
| 1536 | 114 | 6.000 KB | 91% trough (2.003x) |
| 1824 | 96 | 7.125 KB | untested (wide anchor) |

## Direction D1 (PRIMARY): complete the burst-band divisor sweep — screen for a co-equal or higher 99% lobe

**What:** screen the untested in-band divisors as `--fast` (seed 42) runs, reusing the
exact `adamw_v2_ch1216.py` kernel with only the `CH` constant changed (ITERS = 175104//CH,
both exact). Watch **DMA% and latency** — the phase-2 evidence shows latency tracks DMA%
1:1, so DMA% is the fast discriminator. Success = a width that screens **below 0.5601 ms
at DMA ≥ 99 %**.

**Order (bounded, resonance-mapping):**
1. `CH=912` (ITERS=192) and `CH=1368` (ITERS=128) — the two **gap probes** immediately
   adjacent to the known 99% lobes; most likely location of a missed co-lobe.
2. `CH=768` (ITERS=228) — the next finer round-KB anchor (3.0 KB), maps the finer trend.
3. Extend **only if a trend emerges**: `CH=684/608` if the finer side trends up toward
   99%; `CH=1824` if the wider side unexpectedly recovers. Do **not** chase all 8 —
   stop as soon as the resonance shape is mapped and no width beats 1216 (cap ≈ 5 screens).

**Why this is disciplined, not a rule violation:** the bounded-sweep rule stops on a
*smooth* curve after one bracket; adamw's is *bimodal* (phase 2 said so explicitly), which
violates the rule's premise. Completing the lattice is the correct phase-3 diligence for a
resonance. Traffic is pinned at 448 MB at every CH (a pure **scheduling** lever), so there
is **zero correctness risk** — rel-L2 stays 3.42e-8 (layout-invariant, pure elementwise,
`(M,N)↔(128,ITERS,CH)` bit-exact round-trip) at every width.

**Honest expected outcome:** the kernel is already at **0.9973× of the roofline**, so the
realistic result is that **1216 is confirmed as the global peak** (or a second lobe ties it
within noise). A *materially* faster width is unlikely — 799.5 GB/s is a hardware ceiling,
not a bubble, and another 99% lobe would land within measurement noise of 0.56 ms, not
below it. The value of D1 is to **close the bimodal question rigorously** (turn "local
optimum" into "global peak, lattice-complete"), not to expect a new win. If a lobe ties
1216, keep 1216 (incumbent wins ties; no promote for a within-noise delta).

## Direction D2 (CONDITIONAL): promote-test only a *material* new lobe

**Trigger:** a screened width lands **materially below** 0.5601 ms (beyond the ~0.3 %
roofline headroom, i.e. an out-of-noise delta) **and** at DMA ≥ 99 %.

**Protocol:** the phase-2 promote-test — interleaved **full 5-seed A/B/A/B/A**
(A = ch1216, B = candidate), with the four gates:
1. `Bbar < Abar − J` (J = 0.0002 ms noise band);
2. every B < max(A);
3. traffic floor intact (HBMrd 359 + HBMwr 90 = 448 MB on every B — no accidental extra pass);
4. 5-seed L2 gate PASS.

Promote only if all four pass. Otherwise `adamw_v2_ch1216` stays the winner.

## Directions NOT taken (documented, with the trigger that stays disarmed)

- **Traffic cut (bf16 read/store, fused mega-load, dropping an input).** No surface:
  tensors are fp32 in HBM (bf16 cannot cut bytes it doesn't own), all four inputs are
  read-once and genuinely used, output written once, no re-fetch/spill. A pure elementwise
  op has no reduction to average bf16 error down under the 2e-5 gate. Traffic is at the
  read-4/write-1 floor and stays there.
- **Edge-tile / partial-tail specialization.** No edges exist — CH | 175104 exactly, every
  tile is a full `[128, CH]` rectangle (the whole point of the reshape-view vs v1's masked
  tail).
- **Partition/free-split regime.** Partition dim is already at the 128-lane hardware max;
  splitting it only underutilizes DMA.
- **Manual double-buffer / ping-pong (`sequential_range`).** Pre-rejected by silu on this
  profiler (`BL-20260709-dma-batching-regresses-pipeline`): denying `affine_range`'s free
  cross-iteration pipelining regresses ~2×. DMA is already 99% saturated — no bubble to
  ping-pong into.
- **Compute-chain rebalance (D3 in phase 2).** Vec 72% sits comfortably under DMA 99% —
  the 4-Vector chain is fully hidden. Rebalancing an already-hidden engine cannot move the
  DMA-bound wall clock.

## Correctness plan

Every D1 screen is byte-identical to the promoted kernel except the `CH`/`ITERS`
constants, both exact divisors of 175104 — so correctness is layout-invariant and unchanged
(rel-L2 3.42e-8 « 2e-5). The **final candidate** (whether 1216 is confirmed or a material
new lobe is promoted) is validated on the **full 5-seed gate** `[0,21,42,63,84]` before any
promotion, per the phase-2 protocol.

## Bottom line (anticipated)

adamw is a memory-bound elementwise op sitting **at the DMA streaming roofline** on an
**immovable 448 MB traffic floor**, expressed as a **shape-homogeneous, edge-free** stream.
There is no traffic lever, no edge/partition/regime surface, and no schedulable bubble left
(DMA 99 %, effBW = roofline). Phase 3's substantive contribution is to **complete the
bimodal-resonance divisor sweep** left bounded in phase 2 — most likely **confirming
`adamw_v2_ch1216` (2.330x) as the global, lattice-complete peak** and closing the operator
as terminal, with `adamw_v1` (2.112x) retained as the fp32 fallback. If (unlikely) a
materially faster 99% lobe surfaces, promote it via the 5-seed interleaved A/B protocol.

--- Original Design Draft End ---
