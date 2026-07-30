# silu (M4096 N7168, fp32) — Phase 3 Plan: Shape / Regime Specialization

## Goal Description

Take the promoted SiLU kernel `runs/silu_v1.py` (0.3009 ms, **3.398x** over the
1.022441 ms baseline; DMA=97% active, at the achieved single-core streaming HBM
roofline) and complete the shape/regime-specialization phase honestly. There are two
deliverables, and either terminal outcome is legitimate:

1. **Primary analysis (no code):** rigorously establish, with evidence, that the
   classic shape-specialization levers (edge tiles, ragged-tail masks, partition/free
   regime splits, data-dependent/hot-region branches) have **no target** on this
   operator, because the tiled input `(128, 32, 7168)` fp32 is **exact-rectangular**
   (`128·32 = 4096 = M` and `7168 = 2¹⁰·7 = N`, no ragged partition tile, no ragged
   free tail, no mask), the work is **data-uniform** (SiLU is a single elementwise map;
   every element costs identically on the Scalar engine, no reduction/reuse/branch), and
   the **partition dim is pinned** at the 128-lane hardware maximum. This mirrors phase
   2's roofline confirmation: documenting that a whole family of levers does not apply is
   the honest primary result.

2. **Primary experiment (the one untested regime):** explore **finer** free-axis tiling
   — the mirror image of phase 2's *wider* burst sweep (D1 k=2/3/4, which regressed
   monotonically). Split each 7168-wide middle slice into `s` exact sub-chunks of width
   `7168/s`, producing `32·s` pipeline iterations instead of 32. The hypothesis is that a
   finer software pipeline amortizes the fixed fill/drain bubble (measured ~3% ≈ 1/32)
   over more steady-state steps, nudging latency toward the fixed 0.2919 ms DMA-transfer
   wall; the counter-force is that shorter DMA bursts mean more descriptors / more
   issue overhead per byte. The outcome is **genuinely uncertain and must be measured.**

Promote a finer variant **only if** it clears a same-session full-5-seed noise band
against v1. Correctness must never regress: relative-L2 `< 2e-5` on all five seeds
`[0,21,42,63,84]`, fp32 in/out throughout. If nothing clears the band, **keep v1
unchanged** — an explicitly legitimate terminal outcome, exactly as in phase 2. Because
v1 is at the streaming roofline, the best conceivable win is **≤3%** (0.3009 → ~0.29 ms).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. The correctness gate is NKIBench relative-L2 (not allclose);
`verify.py`'s `l2_norm_passed` is authoritative. `--fast` = 1 seed (42), warmup=3,
iters=20 and is **screen-only**, never a promotion decision.

- AC-1: **Correctness is never regressed.** Every promotion candidate passes the FULL
  5-seed `[0,21,42,63,84]` relative-L2 gate; fp32 in/out is preserved everywhere.
  - Positive Tests (expected to PASS):
    - `verify.py --op silu --candidate runs/silu_v3_s<k>.py` reports `l2_norm_passed`
      true for all five seeds on any variant proposed for promotion.
    - The input and output HBM tensors are fp32; the Scalar activation computes in fp32.
  - Negative Tests (expected to FAIL):
    - A variant that introduces a bf16/tf32/fp16 operand, intermediate, or store
      anywhere is rejected (it violates the fp32 contract and fails the 2e-5 gate).
    - Promoting a candidate that passed only `--fast` (1 seed) without the full 5-seed
      run is rejected.

- AC-2: **The "no shape-specialization target" analysis is delivered (D-A).** A written
  analysis at `docs/phase3-shape-analysis.md` (with a `profile/` digest) proving the
  irregular-shape levers do not apply on this operator.
  - Positive Tests (expected to PASS):
    - The write-up states the exact-divisibility arithmetic (`128·32 = 4096 = M`,
      `7168 = 2¹⁰·7 = N`), the data-uniformity argument (single elementwise map, no
      reduction/reuse/hot region), and the pinned-partition argument (`par_dim = 128`
      is the hardware max; smaller wastes lanes, larger is illegal).
    - It explicitly enumerates each classic phase-3 lever (edge tiles, tail masks,
      partition/free regime splits, data-regime branches) and states, per lever, that it
      has no target here.
  - Negative Tests (expected to FAIL):
    - Adding a masked-tail or edge-tile code path for a shape that has **no** tail/edge
      is rejected: it is complexity for zero benefit and a review-rejectable
      overstatement of the target.
    - A write-up that merely asserts "no target" without the divisibility/uniformity/
      pinned-partition evidence is insufficient.

- AC-3: **Finer free-axis tile-width sweep (D-B).** Sweep `s ∈ {2, 4, 7}` — all exact
  divisors of `7168 = 2¹⁰·7`, so every chunk width (`3584`, `1792`, `1024`) is exact and
  the kernel stays **mask-free** and rectangular. Each variant splits each 7168 slice
  into `s` chunks and iterates over `32·s` steps. Candidates: `runs/silu_v3_s2.py`,
  `runs/silu_v3_s4.py`, `runs/silu_v3_s7.py`.
  - Positive Tests (expected to PASS):
    - Each candidate compiles cleanly (no fallback, no SBUF spill, no legality warning),
      is correct on the fast seed, and records latency, DMA%, Scl%, and HBM.
    - Every candidate's HBM counters stay **exactly** at `117 + 117 MB` (this is a
      scheduling lever, never a traffic one).
    - Each candidate's row is written to `benchmark.csv` and node to `candidates.jsonl`.
  - Negative Tests (expected to FAIL):
    - A variant whose HBM traffic rises above the `117 + 117 MB` floor is a **bug**, not
      a regime, and is rejected.
    - Continuing to probe finer `s` after latency has already turned upward (past the
      observed minimum) is rejected — stop at the minimum.
  - AC-3.1: **The finer iteration space realizes ONE deep software pipeline.** A finer
    candidate is only a valid test of the hypothesis if the compiler builds a single
    depth-`32·s` pipeline; a form that refills/drains per outer step (e.g. `32` shallow
    `depth-s` pipelines) creates 32 mini-bubbles and measures an implementation artifact
    instead of the intended finer pipeline.
    - Positive: the candidate expresses the iteration space so it pipelines as one deep
      loop — either a flat `nl.affine_range(32·s)` over a `(128, 32·s, 7168/s)` index
      view (memory-identical to the row-major `(128, 32, 7168)` input), or a nested
      `nl.affine_range(32) × nl.affine_range(s)` **if** it demonstrably pipelines as
      deep. The realized form (flat vs nested) is recorded in the evidence.
    - Negative: a candidate that only pipelines the inner `s` loop (refilling per outer
      middle-axis step) is rejected as not testing the stated hypothesis.
  - AC-3.2: **Explicit sweep stop / bracket rule.** Stop the sweep at the observed
    latency minimum. If `s = 7` (finest tested; 1024-wide, 4 KB/partition) screens
    monotone-best (latency still decreasing at the finest point), run **exactly one**
    bounded bracket probe at the next exact divisor to confirm the turn — see DEC-2 for
    the chosen probe. Otherwise stop.
    - Positive: after a monotone-best `s = 7`, one bracket probe (default `s = 8`,
      896-wide) is run to bracket the turn, and the sweep then stops.
    - Negative: running an unbounded chain of finer divisors, or declaring a minimum
      without either observing an upward turn or running the single bracket probe, is
      rejected.

- AC-4: **Same-session, pre-declared screen (D-B screening).** Each finer candidate's
  `--fast` latency is judged against a **freshly re-measured** v1 `--fast` in the **same
  profiler session**, never against the stale historical 0.3011 ms number. The
  fast-screen jitter band `Jf` is **pre-declared before any candidate is observed** (from
  repeated same-session v1 `--fast` measurements, e.g. an A,A,A or A/B/A screen with
  B = v1) and is never re-chosen after seeing candidate results.
  - Positive Tests (expected to PASS):
    - A candidate advances to the full promotion gate (AC-5) iff its same-session
      `--fast` latency clears the same-session v1 fast median under the DEC-1 policy
      relative to the pre-declared `Jf`.
    - `Jf` is fixed and recorded before candidate screening begins.
  - Negative Tests (expected to FAIL):
    - Comparing a candidate's `--fast` to a stale/historical v1 number rather than a
      same-session v1 measurement is rejected.
    - Choosing or widening `Jf` after seeing candidate results is rejected.

- AC-5: **Promotion gate (same-session interleaved full 5-seed).** To evaluate a
  candidate `B` against v1 (`A`), run one same-session interleaved **full** sequence
  `A0, B0, A1, B1, A2` (all five seeds, warmup=10, iters=100, p50). Define jitter
  `J = max(|A1 − A0|, |A2 − A1|)`, `Abar = median(A0, A1, A2)`, `Bbar = median(B0, B1)`.
  - Positive Tests (expected to PASS):
    - Promote `B` iff `Bbar < Abar − J` **AND** both `B0` and `B1` individually beat
      `max(A0, A1, A2)` **AND** `B`'s HBM counters stay at `117 + 117 MB` **AND** all
      five seeds pass AC-1.
    - The promotion formula is fixed in this plan before the sweep is run.
  - Negative Tests (expected to FAIL):
    - Promoting on a `--fast` result, on a single full run, on a within-`J` tie, or on a
      candidate whose HBM traffic rose above the floor is rejected.
    - A candidate that ties v1 within the noise band **keeps v1** (not promoted).

- AC-6: **Complexity must be earned.** The finer variant adds an inner loop / larger
  iteration count; it must earn its place with a measured, noise-band-clearing latency
  win.
  - Positive Tests (expected to PASS):
    - A finer variant is promoted only when its latency clears the AC-5 gate.
    - When no variant clears the gate, the strictly simpler v1 is kept and this is
      recorded as a legitimate terminal outcome.
  - Negative Tests (expected to FAIL):
    - Promoting a variant because its DMA% rose while its latency did **not** clear the
      AC-5 gate is rejected (DMA% is a diagnostic, not the promotion metric).
    - Keeping added iteration-count complexity that does not beat v1 outside the noise
      band is rejected.

- AC-7: **Evidence is complete and traceable.** New rows in `benchmark.csv`; new nodes
  in `candidates.jsonl` parented off `silu_v1` as a DAG (including first-class rejection
  nodes for any swept `s` that regresses); `profile/` digests; candidate `.py` sources
  under `runs/`.
  - Positive Tests (expected to PASS):
    - `candidates.jsonl` contains one node per swept `s` (and any bracket probe), each
      with a rejection-reason enum value from `{correctness_fail, compile_fallback,
      traffic_increase, latency_regress, within_noise, complexity_not_earned}` when
      applicable, and each records the realized loop form (flat vs nested) and the
      per-partition burst width/bytes.
    - A `profile/` DMA-efficiency digest records latency, DMA%, HBM bytes, and whether
      the effective/active bandwidth drops as the burst size shrinks; a separate digest
      accompanies the D-A shape analysis.
  - Negative Tests (expected to FAIL):
    - A promoted or rejected candidate with no `benchmark.csv` row, no `candidates.jsonl`
      node, or no `profile/` digest is rejected as untraceable.
    - Recording only that HBM stayed at floor, without the effective-bandwidth /
      descriptor-overhead evidence, is insufficient to explain a finer-tiling result.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.
This phase is highly deterministic: the operator, layout, dtype, correctness gate, and
traffic floor are all fixed. The only genuine branch point is whether a **finer** (never
wider) free-axis tiling candidate clears a numeric same-session noise band. The bounds
differ mainly in *how far* the optional D-B sweep and its bracket/loop-form probes are
carried.

### Upper Bound (Maximum Acceptable Scope)

The full bounded exploration is carried out: the D-A shape-analysis write-up with a
profile digest; the D-B finer sweep over `s ∈ {2, 4, 7}` (each realized as one deep
`32·s` pipeline per AC-3.1); one AC-3.2 bracket probe (default `s = 8`) iff `s = 7`
screens monotone-best; and, for the best-screening `s > 1`, one AC-3.1 confirmation run
comparing the flat vs nested loop form and keeping the deeper-pipelining one. Any
candidate that clears the same-session AC-5 gate is promoted with its `s`, loop form,
latency, and speedup recorded; all rejected `s` values are captured as first-class
rejection nodes. HBM stays at `117 + 117 MB` throughout.

### Lower Bound (Minimum Acceptable Scope)

The D-A analysis is delivered (it is the honest primary result and requires no code), and
the D-B sweep over `s ∈ {2, 4, 7}` is `--fast`-screened against a same-session v1 with the
pre-declared jitter band. If no `s` beats v1 in the screen, no full gate run is needed and
**v1 is kept unchanged**, with the screen results and the "no promotable finer regime"
conclusion recorded as evidence. This satisfies all acceptance criteria.

### Allowed Choices

- Can use: `nl.affine_range` over `32·s` (flat, over a reshaped/index-view input) or a
  nested `affine_range(32) × affine_range(s)` if it pipelines as one deep loop; exact
  chunk widths `7168/s` for `s ∈ {2, 4, 7}` (and `s = 8` only as the AC-3.2 bracket
  probe); `nisa.activation(op=nl.silu)` fused on the Scalar engine; in-place activation
  (`dst == data`) only if a finer variant shows SBUF pressure (it should not — every
  finer tile pair is ≤ 56 KB/partition, well under budget).
- Cannot use: any traffic-changing transform (bf16/tf32/fp16, dtype tricks, recompute);
  wider k-batching / multi-slice DMA coalescing (phase-2 D1, settled regression);
  explicit ping-pong / `nl.sequential_range` manual prefetch (phase-2 D2, redundant);
  per-kernel `dge_mode` toggling (`--disable-dge` is globally forced by the harness);
  changing the activation formula (sigmoid+multiply, exp-exact) — DMA-bound so it cannot
  help latency, and fused `nl.silu` is already L2-accurate; masked-tail/edge code paths
  (no tail/edge exists).

> **Note on Deterministic Designs**: This draft is highly deterministic — the operator,
> layout, dtype, correctness gate, traffic floor, and the settled non-goals are all
> fixed. The upper and lower bounds converge except in how far the bounded D-B sweep and
> its optional bracket/loop-form probes are carried, and "keep v1 unchanged" is a fixed,
> pre-declared legitimate terminal outcome.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

The finer-tiling candidate keeps v1's access pattern (contiguous along the free axis
within each partition) but splits each 7168-wide slice into `s` shorter contiguous runs,
so the compiler sees `32·s` independent iterations to pipeline. One possible realization
(flat form, `s` chunks):

```
P, MID, F = 128, 32, 7168
CH = F // s                       # exact: 3584 (s=2), 1792 (s=4), 1024 (s=7)
# Row-major (128, 32, 7168) is memory-identical to a (128, 32*s, F/s) view, so a flat
# affine_range(32*s) over the finer view walks the same bytes as v1, just in shorter runs.
for t in nl.affine_range(MID * s):        # single deep pipeline, depth 32*s
    x_tile = load([128, CH] slice for linear index t)   # HBM -> SBUF, one short burst
    y_tile = nisa.activation(op=nl.silu, data=x_tile)    # fused SiLU, Scalar engine
    store(y_tile back to the same [128, CH] slice)       # SBUF -> HBM
# Two live SBUF tiles of [128, CH] (<= 56 KB/partition total) — SBUF is a non-constraint.
```

The nested alternative (`affine_range(32) × affine_range(s)`) is only acceptable if it
demonstrably pipelines as one depth-`32·s` loop rather than refilling per outer step; the
AC-3.1 confirmation run for the best-screening `s` settles flat-vs-nested. The bubble
proxy `1/iters` motivates the direction (s=1→3.12%, s=2→1.56%, s=4→0.78%, s=7→0.45%), but
whether the real bubble tracks it — versus being dominated by DMA descriptor/issue
overhead as bursts shrink — is exactly the open question the measurement answers.

### Relevant References

- `runs/silu_v1.py` — the promoted parent kernel (flat `affine_range(32)`, full-width
  `[128,7168]` load → fused `nl.silu` → store; the finer sweep's parent in the DAG).
- `runs/silu_v2_k4.py` — the phase-2 *wider* variant; shows the multi-dim index-view
  tiling idiom (`p_ix/k_ix/f_ix` with `nl.arange`) and in-place activation, directly
  reusable for expressing the finer view.
- `docs/phase2-roofline-confirmation.md`, `docs/phase2-exit-decision.md` — the roofline
  and the same-session interleaved noise-band methodology (phase 2's AC-8) that this
  plan reuses as AC-5.
- `../../verify.py` — `--fast` (1 seed 42 / warmup 3 / iters 20) vs full (5 seeds /
  warmup 10 / iters 100); `l2_norm_passed` gate.
- `benchmark.csv`, `candidates.jsonl`, `profile/` — evidence sinks (candidate `.py`
  sources under `runs/` are tracked; other artifacts git-ignored).

## Dependencies and Sequence

### Milestones

1. **Milestone 1 — D-A shape-specialization analysis (independent, no code).**
   - Phase A: assemble the exact-divisibility arithmetic, data-uniformity, and
     pinned-partition evidence from the tiled `(128, 32, 7168)` shape.
   - Phase B: write `docs/phase3-shape-analysis.md` + a `profile/` digest; enumerate each
     classic phase-3 lever and its "no target" status (satisfies AC-2).

2. **Milestone 2 — Pre-declare the D-B screen band.**
   - Step 1: run same-session v1 `--fast` repeatedly (A,A,A or A/B/A with B=v1) and fix
     the jitter band `Jf` before any candidate exists (satisfies AC-4 pre-declaration).

3. **Milestone 3 — D-B finer sweep and screening.** Depends on Milestone 2.
   - Step 1: implement `runs/silu_v3_s2.py`, `silu_v3_s4.py`, `silu_v3_s7.py`, each
     realizing one deep `32·s` pipeline (AC-3, AC-3.1).
   - Step 2: `--fast`-screen each against same-session v1 under the DEC-1 policy; verify
     HBM stays at `117 + 117 MB`; stop at the observed latency minimum (AC-3, AC-3.2).
   - Step 3: iff `s = 7` is monotone-best, run one AC-3.2 bracket probe (default `s = 8`).

4. **Milestone 4 — Loop-form confirmation (conditional).** Depends on Milestone 3.
   - Step 1: for the best-screening `s > 1`, one confirmation run comparing flat vs
     nested loop form; keep the deeper-pipelining one (AC-3.1). Skip if s=1 (v1) is best.

5. **Milestone 5 — Promotion gate and evidence.** Depends on Milestone 3 (and 4 if run).
   - Step 1: for any candidate that beats same-session v1 in the screen, run the AC-5
     interleaved full `A0,B0,A1,B1,A2` sequence and apply the numeric promotion rule.
   - Step 2: promote the winner (if any) or keep v1; write all `benchmark.csv` rows,
     `candidates.jsonl` nodes (with rejection-reason enum + realized loop form + burst
     width/bytes), and `profile/` DMA-efficiency digests (satisfies AC-6, AC-7).

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Write `docs/phase3-shape-analysis.md` (exact-divisibility, data-uniformity, pinned-partition; enumerate each shape lever as "no target") + `profile/` digest | AC-2 | coding | - |
| task2 | Pre-declare the D-B fast-screen jitter band `Jf` from repeated same-session v1 `--fast` runs; record it before any candidate is screened | AC-4 | coding | - |
| task3 | Implement `runs/silu_v3_s2.py`, `silu_v3_s4.py`, `silu_v3_s7.py`, each realizing one deep `32·s` pipeline; `--fast`-screen each against same-session v1; verify HBM at `117+117 MB`; stop at the latency minimum | AC-3, AC-3.1, AC-3.2, AC-4 | coding | task2 |
| task4 | Iff `s=7` screens monotone-best, implement + `--fast`-screen one bracket probe (default `s=8`, per DEC-2) to bracket the turn | AC-3.2 | coding | task3 |
| task5 | Conditional loop-form probe: for the best-screening `s>1`, one confirmation run comparing flat vs nested `affine_range` form; keep the deeper-pipelining one; skip if s=1 best | AC-3.1 | coding | task3 |
| task6 | For any candidate beating same-session v1 in the screen, run the AC-5 interleaved full `A0,B0,A1,B1,A2` sequence and apply the numeric promotion rule | AC-1, AC-5, AC-6 | coding | task3, task4, task5 |
| task7 | Write all `benchmark.csv` rows and `candidates.jsonl` nodes (parented off `silu_v1`, with rejection-reason enum, realized loop form, burst width/bytes) + `profile/` DMA-efficiency digest | AC-7 | coding | task3, task4, task5, task6 |
| task8 | Independent review of the D-B finer-tiling result and the "keep v1 vs promote" decision against the roofline / descriptor-overhead reasoning | AC-6 | analyze | task6 |

## Claude-Codex Deliberation

### Agreements
- D-A is correctly scoped: exact-rectangular shape, no tails, data-uniform work, pinned
  partition count, and no masked-edge code path is the honest primary deliverable.
- D-B (finer free-axis tiling) is the right — and only — remaining regime: a scheduling
  experiment that holds HBM traffic at the `117 + 117 MB` floor.
- The `s ∈ {2, 4, 7}` set is reasonable (all exact divisors of `7168 = 2¹⁰·7`,
  guaranteeing mask-free chunks).
- Same-session v1 re-measurement and the interleaved full-5-seed promotion gate are
  appropriate given the tiny (~3%) remaining DMA-bubble ceiling.
- "Keep v1 unchanged" is a valid, pre-declared terminal outcome and must remain explicit.
- The settled phase-1/2 non-goals (wider k-batching, ping-pong, dge_mode, bf16/traffic
  reduction, activation-formula changes) are not re-litigated.

### Resolved Disagreements
- **Loop-form realization is central, not a conditional afterthought (Codex first pass →
  AC-3.1).** Codex flagged that a nested `affine_range(32) × affine_range(s)` could
  refill/drain per outer iteration (32 mini-bubbles) or pipeline only the inner loop,
  which would measure an artifact rather than the finer pipeline. **Resolution:** AC-3.1
  now *requires* each finer candidate to realize one deep `32·s` pipeline (flat view
  preferred; nested allowed only if it demonstrably pipelines as deep), and the flat-vs-
  nested comparison is elevated from an optional probe to a first-class correctness-of-
  experiment requirement recorded in evidence.
- **Same-session, pre-declared screening (Codex → AC-4).** Codex noted that screening a
  candidate's `--fast` against the stale historical v1 number is unreliable when the
  ceiling is ~3%, and that the jitter band must be fixed before results are seen.
  **Resolution:** AC-4 now mandates a freshly re-measured same-session v1 and a
  pre-declared jitter band `Jf` (from repeated same-session v1 `--fast` runs), never
  re-chosen after the fact.
- **HBM-at-floor is necessary but not sufficient (Codex → AC-7).** Codex noted shorter
  bursts can hold byte-traffic constant while dropping effective bandwidth via descriptor/
  issue overhead. **Resolution:** AC-7 now requires a DMA-efficiency digest capturing
  whether active bandwidth drops as burst size shrinks, plus per-partition burst
  width/bytes, so a finer-tiling result is explained, not just tabulated.
- **Explicit sweep stop / bracket rule (Codex → AC-3.2).** With sparse points `{2,4,7}`,
  "stop at the minimum" only detects a turn if latency worsens before `s = 7`.
  **Resolution:** AC-3.2 adds one bounded bracket probe (default `s = 8`) iff `s = 7`
  screens monotone-best, mirroring phase 2's conditional `k ∈ {5,6,7}` probe.

### Convergence Status
- Final Status: `converged` (2 convergence rounds; Codex round 1 = "largely converged, no
  substantive disagreement", round 2 = "Converged. REQUIRED_CHANGES: none"). The two
  remaining items are user decisions, not disagreements — Claude and Codex share the
  recommended answer on both.

## Pending User Decisions

- DEC-1: **Fast-screen near-tie policy.** How should a finer candidate whose same-session
  `--fast` latency falls within the pre-declared jitter band `Jf` of v1 be handled at the
  screen stage?
  - Claude Position: **Permissive** — advance candidates within `Jf` to the full AC-5
    gate; because the ceiling is only ~3%, a true 1-2% win could otherwise be lost to
    screen jitter, and the full interleaved gate is the real filter.
  - Codex Position: Permissive-within-the-declared-band (same recommendation); strict
    fast-screening can wrongly discard a real 1-2% win near the roofline.
  - Tradeoff Summary: Permissive spends one extra full-gate run per near-tie candidate
    but avoids discarding a real small win; strict is cheaper but risks a false negative
    against a ~3% ceiling. Both reviewers recommend permissive.
  - Decision Status: `PENDING`

- DEC-2: **`s`-sweep lower burst-size bound.** Is `s = 7` (1024-wide, 4 KB/partition) the
  intended hard lower burst-size bound, or should one extra exact divisor be tested if
  `s = 7` screens monotone-best?
  - Claude Position: Run **exactly one** bounded bracket probe at `s = 8` (896-wide,
    3.5 KB/partition) iff `s = 7` is monotone-best, to bracket the turn; otherwise stop.
  - Codex Position: Allow one probe, prefer `s = 8` first because it is the nearest exact
    divisor and changes less than jumping to `s = 14` (same recommendation).
  - Tradeoff Summary: One probe costs a single extra screen but confirms whether the
    latency-vs-`s` curve has actually turned; declaring `s = 7` a hard bound saves that
    run but leaves the turn unconfirmed. Both reviewers recommend the single `s = 8`
    probe.
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. name candidates by
  their `s` value: `silu_v3_s2`, `silu_v3_s4`, `silu_v3_s7`).
- Kernel comments should describe the finer-tiling structure (chunk width, iteration
  count, single-deep-pipeline intent, HBM-at-floor invariant) in domain terms, not by
  referencing this plan's criteria.

--- Original Design Draft Start ---

# silu (M4096 N7168, fp32) — Phase 3 implementation draft (regime / shape specialization)

## Goal

Phase 3 is *shape / regime specialization*: analyze where time goes across the
tensor's structure and specialize **only where the measured win justifies the added
complexity** (tile-size regimes, partition/free splits, edge tiles). Starting from
the promoted kernel `runs/silu_v1.py` (0.3009 ms, **3.398x** over the 1.022441 ms
baseline), the honest phase-3 job is (a) to establish rigorously that the *classic*
shape-specialization levers have **no target** on this operator, and (b) to explore
the one tile-size regime phase 2 never tested — the **finer** free-axis direction —
and promote it only if it clears the same-session noise band on a full 5-seed run.

Never regress correctness (relative-L2 < 2e-5 on all five seeds `[0,21,42,63,84]`).

## Where phase 1–2 left us (the measured starting point)

`runs/silu_v1.py`: one `nl.affine_range(32)` over the middle axis; each iteration
loads a full-width `[128, 7168]` fp32 slice HBM→SBUF, applies one fused
`nisa.activation(op=nl.silu)` on the Scalar engine, stores `[128, 7168]` SBUF→HBM.
Two live SBUF tiles (x_tile, y_tile), no inner free-dim loop, mask-free.

Profiler digest (`profile/silu_v1.txt`, full 5-seed):

| latency | speedup | MFU | PE | Vec | Scl | **DMA** | HBMrd | HBMwr |
|---------|---------|-----|----|-----|-----|---------|-------|-------|
| 0.3009 ms | 3.398x | 0% | 1% | 1% | 34% | **97%** | 117 MB | 117 MB |

Phase 2's terminal finding (`docs/phase2-roofline-confirmation.md`,
`docs/phase2-exit-decision.md`): v1 sits at the **achieved single-core streaming HBM
roofline**. Traffic is exactly the read-once/write-once floor
(`2·4096·7168·4 B = 234.88 MB` = measured 117+117 MB), so there is **no traffic left
to remove** and **no multiplicative headroom**. The only physically-available slack
is a ~3% (~9 µs) DMA-issue/fill-drain bubble. Phase 2 swept the *wider*-burst
direction (D1 k-batching k=2/3/4) — **monotone regression** (DMA 97→85→71%,
coarser pipeline) — and rejected ping-pong (D2, redundant with `affine_range`),
dge_mode (D4, globally `--disable-dge`), and bf16/traffic-reduction (D5, fp32 gate).
All of those are settled; **phase 3 must not re-litigate them** (see NON-GOALS).

## Structural analysis — is there ANY specialization target?

Phase-3's archetype levers are for *heterogeneous* work: irregular shapes with edge
tiles, ragged tails needing masks, or data regimes worth branching on. I checked each
against this operator's actual structure and traffic.

### 1. The shape is EXACT-rectangular — no edge/tail/mask regime

The tiled input is `(128, 32, 7168)` fp32:
- Partition axis `128·32 = 4096 = M` **exactly** (the 32 middle slices each fill all
  128 partitions; no ragged partition tile).
- Free axis `7168 = 2¹⁰·7 = N` **exactly** (no ragged free tail).

So there is **no edge tile, no tail, and no mask anywhere** in v1 (its comment already
notes "mask-free (128·32 = 4096 = M and 7168 = N are exact)"). The phase-3 lever
"specialize the edge tiles / handle the ragged tail differently" has **literally no
target here** — fabricating a masked-tail code path for a shape with no tail would add
complexity for zero benefit and would be rejected on review. This is the phase-3
analogue of phase-2's roofline finding: **the primary deliverable is documenting that
the irregular-shape levers do not apply**, with the exact-divisibility arithmetic as
evidence.

### 2. The work is data-UNIFORM — no hot-region / data-dependent regime

SiLU `y = x/(1+e^-x)` is a single elementwise map: every one of the 4096·7168 elements
costs identically on the Scalar engine, independent of value. There is no reduction,
no reuse, no data-dependent branch, and no "hot" sub-region of the tensor that would
justify a value-specialized or region-specialized code path. So the "specialize where
the work concentrates" lever also has **no target**.

### 3. Partition/free split is already optimal

v1 uses `par_dim = 128` (the hardware maximum partitions) and puts the entire 7168
free axis in one activation call. A smaller partition dim would waste partition lanes
(more iterations over the same bytes); a larger one is illegal (>128). So the
**partition** regime is pinned at the optimum. The only re-tiling degree of freedom
left is the **free-axis tile width**, addressed next.

## The one untested regime: FINER free-axis tiling (the opposite of phase-2's D1)

Phase 2 swept the free/middle burst **wider** (k middle-slices per DMA: k=2/3/4) and
found monotone regression — fewer `affine_range` iterations → coarser
compiler software-pipeline → the prologue/epilogue DMA bubble becomes *relatively
larger*. The mirror-image direction — **finer** tiles, i.e. *more* iterations — was
never tested, and it is the natural phase-3 "tile-size regime" lever.

**Idea.** Split each 7168-wide middle slice into `s` exact sub-chunks of width
`7168/s`, giving `32·s` total pipeline iterations instead of 32. All chunk widths are
exact (7168 = 2¹⁰·7), so this stays **mask-free** and rectangular — no edge handling
introduced. Access pattern is identical to v1 (contiguous along the free axis within
each partition), just split into more, shorter DMA bursts.

**Why it might help.** The measured ~3% DMA-idle bubble ≈ 1/32 — the cost of filling
and draining one iteration of a 32-deep pipeline. More iterations make the
software-pipeline *finer*, so the fixed fill/drain overhead is amortized over more
steady-state steps and the relative bubble shrinks toward the 0.292 ms DMA transfer
wall:

| s | chunk width | iters (32·s) | 2 live tiles (KB/part) | 1/iters (bubble proxy) |
|---|-------------|--------------|------------------------|------------------------|
| 1 (v1) | 7168 | 32 | 56.0 | 3.12% |
| 2 | 3584 | 64 | 28.0 | 1.56% |
| 4 | 1792 | 128 | 14.0 | 0.78% |
| 7 | 1024 | 224 | 8.0 | 0.45% |

If (and only if) the bubble tracks 1/iters, the DMA transfer wall
(`0.97·0.3009 = 0.2919 ms`) plus a shrinking bubble predicts ~0.2965 ms (s=2) →
~0.2942 ms (s=4). **SBUF is a non-constraint** (every finer tile pair ≤56 KB « the
budget — unlike phase-2's k=4 at 224 KB that forced in-place), so finer tiling is
"free" to try.

**Why it might NOT help (the honest counter-force).** Finer chunks mean more, shorter
contiguous DMA runs per partition → more DMA descriptors / issue overhead per byte.
This is the same per-burst-overhead mechanism, just pushed the other way from D1. So
the outcome is **genuinely uncertain and must be measured**: finer pipeline (+) vs
more descriptors (−). Phase 2 established that *wider* loses; it did **not** establish
what *finer* does. That is exactly the open question phase 3 answers.

**Ceiling / expectation.** The 0.2919 ms DMA-active transfer wall is fixed by the
traffic floor, so the best conceivable win is **≤3%** (0.3009 → ~0.29 ms). Realistic
outcome: single-digit-% or zero. Per the roofline, **"keep v1 unchanged" is an
explicitly legitimate terminal outcome** — as it was in phase 2.

## Directions, ranked by benefit/risk

### D-A (rank 1, PRIMARY analysis) — Document that shape-specialization has no target
Not a code change: the rigorous, evidence-backed statement (with exact-divisibility
arithmetic, data-uniformity, and pinned partition dim) that the classic phase-3 levers
(edge tiles, tail masks, partition/free regime splits, data-regime branches) have **no
target** on this exact-rectangular, data-uniform operator. This is the honest primary
deliverable, mirroring phase 2's roofline confirmation. Written to
`docs/phase3-shape-analysis.md` with a `profile/` digest.

### D-B (rank 2, PRIMARY experiment) — Finer free-axis tile-width sweep
Sweep `s ∈ {2, 4, 7}` (all exact divisors → mask-free): split each 7168 slice into
`s` chunks of `7168/s`, iterate `nl.affine_range` over `32·s` steps. Candidates
`runs/silu_v3_s2.py`, `silu_v3_s4.py`, `silu_v3_s7.py`. `--fast` screen each; record
latency, DMA%, Scl%, and HBM (**must stay at 117+117 MB** — this is a scheduling lever,
never a traffic one). Only a variant that **beats v1 in the `--fast` screen** advances
to the full-gate promotion run (mirror of phase-2 AC-3: a screen regression/tie is not
promoted). Expected: monotone one way; stop the sweep when latency turns (don't probe
past the observed minimum).

### D-C (rank 3, CONDITIONAL) — Loop-structure probe for the best-screening s
Only if some `s>1` screens as best: test whether expressing the iteration space as a
single flat `nl.affine_range(32*s)` over a `[128, 32*s, 7168/s]` view vs the nested
`affine_range(32) × affine_range(s)` changes how aggressively the compiler pipelines
(one candidate, e.g. `silu_v3_s{best}_flat.py`). One confirmation run; keep the
better-screening form only if it clears the gate. Skip entirely if s=1 (v1) screens
best.

### NON-GOALS (settled in phase 1–2 — do NOT re-litigate)
- **Wider k-batching** (phase-2 D1 k=2/3/4): monotone regression, rejected.
- **Explicit ping-pong / `sequential_range`** (D2): redundant with `affine_range`
  auto-pipelining, regressed.
- **dge_mode** (D4): `--disable-dge` is globally forced by the harness; no lever.
- **bf16 / tf32 / any traffic reduction** (D5): fp32 in/out contract fixes the 234.88 MB
  floor and any reduced-precision operand fails the 2e-5 rel-L2 gate.
- **Changing the activation formula** (sigmoid+mul, exp-exact): DMA-bound, so it cannot
  help latency, and the fused `nl.silu` LUT is already L2-accurate on all 5 seeds.

## Acceptance criteria

- **AC-1 (correctness).** Never regress rel-L2 < 2e-5. Any promotion candidate must
  pass the FULL 5-seed `[0,21,42,63,84]` run (not just `--fast`). No bf16/tf32/fp16
  introduced anywhere; fp32 in/out throughout. `verify.py`'s `l2_norm_passed` gate is
  authoritative.
- **AC-2 (primary analysis, D-A).** Deliver the "no shape-specialization target"
  write-up with exact-divisibility arithmetic (128·32 = 4096 = M, 7168 = N),
  data-uniformity, and the pinned-partition argument. **Negative test:** do NOT add a
  masked-tail or edge-tile code path for a shape that has no tail/edge — that would be
  complexity for zero benefit and is a review-rejectable overstatement of the target.
- **AC-3 (finer sweep, D-B).** Sweep `s ∈ {2,4,7}`, all mask-free. `--fast` is
  **screen-only**; a variant advances to the full promotion gate **iff** it beats v1's
  `--fast` latency beyond obvious jitter. Record every candidate's latency, DMA%, and
  HBM in `benchmark.csv` / `candidates.jsonl`; HBM must remain at 117+117 MB (a
  variant that inflates HBM traffic is a bug, not a regime). Stop the sweep at the
  observed latency minimum (don't probe finer past a turn).
- **AC-4 (promotion gate).** Promote a finer variant only if it clears the same-session
  noise band on the full 5-seed run (interleaved A0,B0,A1,B1,A2 vs v1, per the
  fast-vs-full-run lesson). A within-noise tie **keeps v1** (the simpler kernel wins
  ties — AC-5).
- **AC-5 (complexity justification).** The finer variant adds an inner loop / larger
  iteration count. It must **earn** its place with a measured, noise-band-clearing win;
  absent that, keep the strictly simpler v1. "Specialize only where the measured win
  justifies the added complexity" is the phase-3 mandate.
- **AC-6 (loop-structure probe, D-C).** Conditional on some s>1 screening best; at most
  one confirmation run; otherwise skip.
- **AC-7 (evidence).** New rows in `benchmark.csv`; new nodes in `candidates.jsonl`
  parented off `silu_v1` as a DAG (including first-class rejection nodes for any swept
  s that regresses); a `profile/` digest per major direction (finer-sweep table +
  shape-analysis). Candidate `.py` sources under `runs/`.

## Expected outcome (stated up front, honestly)

Given v1 is at the achieved streaming roofline with only a ~3% bubble available, the
most likely phase-3 result is **either** a small (<3%) finer-tiling win that clears the
noise band and is promoted, **or** "keep v1 unchanged" plus the rigorous
shape-specialization-has-no-target analysis. Both are legitimate terminal outcomes;
the discipline is to let the measured full-5-seed number decide and never promote
added complexity that does not beat v1 outside the noise band.

--- Original Design Draft End ---
