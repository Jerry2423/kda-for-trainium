# bmm_softmax — Phase 3 Plan: Fused-Softmax Epilogue Engine Scheduling (or terminal-optimum proof)

## Goal Description

Improve the fused `out[b] = softmax_N(lhs[b] @ rhs[b])` kernel past the phase-2 promotion
`runs/bmm_softmax_v4.py` (**1.946x, 3.7460 ms p50**, fp32, on-device rel-L2 `2.5683307869e-6`,
`matmul_instruction_count` 8704, `psum_read_sbuf_write_count` 4224, HBM 33.6/1073.7 MB at floor,
spill 0) **as a bit-exact pure engine/schedule reassignment** — OR, if the fused epilogue is
already at the compiler's engine-balanced optimum, prove that with controlled same-session
measurement and terminate keeping v4. Either outcome is a valid, honest phase exit.

Operator `bmm_softmax` (NKIBench case 2): `lhs (16,4096,64)=(B,M,K)`, `rhs (16,64,4096)=(B,K,N)`
fp32 → `out (16,4096,4096)`. **B=16, M=4096, K=64, N=4096.** Baseline **7.290 ms**.

**Classic shape specialization is explicitly out of surface here, with numbers on the record so the
phase does not chase a dead lever** (mirrors the solved sibling `bmm`):
- **No edge tiles.** Every axis divides cleanly: `M=4096=32·128`, `N=4096=8·512`, `K=64≤128`
  (single Tensor-Engine pass), `B=16`. No ragged remainder to special-case.
- **Tiles are already maximal.** The main matmul is `[K=64]×[64,512]→[128,512]`; the moving free
  dim 512 is the hard PSUM-bank wall on trn2 (one `nc_matmul` writes one bank = 512 fp32
  elems/partition; 2048/4096-wide is trn3-only). The stationary free dim 128 fills the PE columns.
  The softmax reduces over the full resident N=4096 row → no N-tiling regime either.
- **The K=64 partition/free split is fixed and cheap.** trn2 matmul cost is `dst_free_elems·100/freq`
  — proportional to the moving free dim (512) only, independent of K; a half-full contraction costs
  nothing extra. K cannot be packed across batches (`out[b]` block-diagonal — closed in `bmm`).

So the phase-3 "structure" to analyze is the **engine schedule of the fused softmax epilogue relative
to the matmul stream**, the one structural dimension phase 2 stopped short of. Phase 2 tuned
within-batch stream depth (`M_SUB`); phase 3 attacks the **engine placement** of the epilogue.

**Central hypothesis (to be tested, not assumed):** the fused kernel's matmul work is byte-identical
to pure `bmm` (8704 instr, same tiles), yet its TRUE PE-active is materially higher; the draft
attributes this to the softmax epilogue back-pressuring the matmul — the 8 PSUM→SBUF score copies
must drain (freeing the 8 PSUM banks) before the next 8 matmuls can reuse the banks, and those copies
plus the two reductions (Vector) and the `exp` (Scalar), both engines near ~70%, throttle the drain.
The lever is to drain the banks faster by spreading the epilogue across Vector **and** Scalar.
**This causal story is the primary risk the plan must falsify or confirm in round 0 (see AC-3),
because the profiler exposes only summary counters — no semaphore/timeline trace is available here.**

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic
verification. "Out-of-noise" is defined concretely in AC-8 (same-session interleaved bracket, ~1.5–2%
band); every perf claim below inherits that definition.

- AC-1: **Correctness invariance (HARD gate).** Every candidate is a pure engine/schedule
  reassignment of the same fp32 math (same set of `exp` terms summed in the same order), so its
  output must be numerically identical to v4 for each seed.
  - Positive Tests (expected to PASS):
    - Full `verify.py` (no `--fast`) reports `l2_norm_passed=True` on **all** seeds `[0,21,42,63,84]`.
    - Each seed's on-device `relative_l2_error` equals the same-session v4 value for that seed
      (v4 currently reports `2.5683307869e-6` on every seed; a candidate must match its per-seed v4
      figure, i.e. bit-identical output for the same seed input).
  - Negative Tests (expected to FAIL / cause rejection):
    - Any seed with `l2_norm_passed=False`.
    - A candidate whose per-seed `relative_l2_error` differs from same-session v4 for that seed
      (any drift ⇒ an indexing/placement/engine-numeric bug — reject, do not promote).
  - AC-1.1: A candidate is promoted only after the **full 5-seed** run passes; a `--fast` (seed-42)
    pass alone is a screen, never a promotion basis.

- AC-2: **Purity guards (HARD).** A pure reschedule must not change the work performed.
  - Positive Tests: candidate keeps `matmul_instruction_count`=8704, `psum_read_sbuf_write_count`=4224,
    `hbm_read_bytes`≈33.6 MB, `hbm_write_bytes`≈1073.7 MB (read-once/write-once floor), and
    `spill_save_bytes`=`spill_reload_bytes`=0.
  - Negative Tests: any deviation in these counters (e.g. matmul count ≠ 8704, HBM above floor, or
    non-zero spill) means the change was NOT a pure reschedule — investigate and reject as a
    correctness/architecture regression, exactly as the phase-2 `reduce_res` fusion was rejected when
    it doubled `matmul_instruction_count` 8704→17408.

- AC-3: **Round-0 controlled measurement & causal disambiguation (BEFORE any code change; near-zero
  remote risk).** Establishes the fact the phase rests on and disambiguates the causal story using
  only summary metrics (no timeline available).
  - Positive Tests:
    - Same-session re-anchor of v4: TRUE PE-active ≈ 3.371 ms, per-matmul PE-active ≈ 0.387 µs,
      `matmul_instruction_count`=8704, HBM 33.6/1073.7 MB, spill 0 — matches the digest.
    - **Depth-matched control:** the same-depth pure-`bmm` `M_SUB=16` anchor
      (`../bmm/runs/bmm_d1_twophase_m16.py`) is measured in the **same session**, so the softmax
      PE-active gap is compared depth-to-depth (M16-vs-M16), NOT the depth-confounded v4-M16-vs-
      pure-bmm-M32 "+68%" headline. Report the depth-matched gap explicitly.
    - The **static engine assignment** of each epilogue op (the 8 score copies + the normalize) is
      recorded from the per-engine instruction counts + a one-shot engine-annotated dump, so any
      rebalance is measured against the real current placement, not an assumed one. (Recorded as
      static placement, not a claim about dynamic scheduling/overlap, which summary metrics cannot
      prove.)
  - Negative Tests:
    - Proceeding to D1 without a same-session v4 anchor AND the depth-matched pure-bmm-M16 anchor.
    - Treating the +68% (M16-vs-M32) figure as the controlled gap.
  - AC-3.1: **Causal-premise check.** If the depth-matched (M16-vs-M16) PE-active gap is small AND
    the AC-6b copy-elimination diagnostic does not materially inflate PE-active, the back-pressure
    premise is downgraded and the phase leans toward the AC-7 terminal outcome rather than forcing a
    D1 promotion.

- AC-4: **D1 — epilogue engine rebalancing (PRIMARY).** Relieve the matmul↔softmax back-pressure by
  spreading the PSUM-drain/normalize across Vector and Scalar, as **separate levers** for clean
  attribution.
  - AC-4.1: **Copy placement first (normalize held at current).** Reassign the engine of the 8
    PSUM→SBUF score copies via `nisa.tensor_copy(dst, src, engine=...)` (NOT `nl.copy`, so engine
    placement is the actual experiment). Predeclared variants only (no open-ended search): (i) all
    copies on current engine [baseline], (ii) all copies on Scalar, (iii) all copies on Vector,
    (iv) 4 Vec / 4 Scalar alternating by chunk index; extend to 2/6 or 6/2 **only if** the 4/4 split
    shows out-of-noise signal.
  - AC-4.2: **Normalize placement second (on the winning copy assignment).** Try the normalize on
    Scalar via `nisa.activation(op=nl.copy, data=exp_t, scale=recip[P,1])` vs the current Vector
    `tensor_scalar(op0=multiply, operand0=recip)`.
  - Positive Tests (promotion): a variant improves **wall p50 out-of-noise** (AC-8) versus a
    same-session v4 bracket, while satisfying AC-1 and AC-2, and its per-engine actives corroborate
    the mechanism (TRUE PE-active drop and/or Vector-active drop without a wall-erasing Scalar tail).
  - Negative Tests (reject): a variant that (a) leaves TRUE PE-active within precision of v4 AND wall
    p50 flat/worse (compiler no-op), or (b) raises TRUE PE-active with no wall win (anti-lever), or
    (c) drops Vector-active but raises Scalar-active enough to expose a new tail that erases the wall
    improvement (moved the bottleneck, per Codex AC-7). See KILL-CRITERION.

- AC-5: **D2 — `M_SUB` re-sweep (SECONDARY, contingent).** Only if D1 lands a promotable win AND the
  purity guards (AC-2) remain stable on the winner: re-sweep `M_SUB ∈ {16, 32}` on the winning engine
  assignment, within one batch only (batch loop stays `affine_range(B)`; never cross-batch).
  - Positive Tests: a `M_SUB` point beats the D1 winner's same-session wall bracket out-of-noise while
    keeping AC-1/AC-2. `M_SUB=8` is added only if D1 materially changed the engine balance (phase-2
    already found M8 < M16).
  - Negative Tests: promoting a `M_SUB` change that is within noise or that breaks a purity guard;
    running D2 when D1 did not land (skip D2 entirely in that case).

- AC-6: **Closed levers — record-only, do NOT build for production, do NOT spend iterations.**
  - Positive Test: the exit decision records each as closed with its one-line reason; no production
    kernel is built for any of them.
  - Negative Test (would be a violation): building/tuning any of — bf16x2 3-product matmul split
    (fp32/bf16 pass ratio 2.0 < 3.0 ⇒ split *raises* PE; `[[BL-20260710-bf16x2-loses-when-fp32-
    emulates-in-2-passes]]`); bf16 `exp`/softmax (~1e-2 rel error over N=4096 ≫ the 2e-5 gate);
    cross-batch blocking / cross-batch double-buffer (measured anti-lever in `bmm`;
    `[[BL-20260710-cross-batch-blocking-is-an-antilever-on-affine-range]]`); GpSimd normalize/copies
    (API-infeasible: `tensor_scalar(engine=gpsimd)` is rsqrt-only and GpSimd cannot access PSUM);
    fused `activation(reduce_op=add, reduce_res=)` exp+row-sum (measured +75% wall via whole-stream
    2× recompute); copy-elimination **as a production kernel**; wider matmul tile / narrower
    N_CHUNK / K-packing / DMA store-burst / bf16 output.
  - AC-6b: **DIAGNOSTIC EXCEPTION (one probe, counts against budget).** Exactly one `--fast`
    copy-elimination probe (reduce/exp directly from PSUM) IS allowed to test the central causal claim
    — it removes the score-copy drain entirely. If it inflates TRUE PE-active, the paper rejection is
    validated and the back-pressure story stands; if it does NOT, D1's mechanistic model is revised
    (feeds AC-3.1). The probe file MUST be named/annotated diagnostic-only (e.g.
    `*_copyelim_diag.py`) so it can never be accidentally promoted; it is NOT a production candidate
    regardless of result.

- AC-7: **Terminal / no-op is an acceptable outcome (HARD honesty gate).** If every D1 variant is a
  compiler no-op (TRUE PE-active within precision of v4 AND wall flat), an anti-lever, or a
  bottleneck-mover (AC-4 negative (c)), record v4 as the fused kernel's engine-balanced optimum with
  no remaining schedulable structure, keep v4, and write the exit decision. This mirrors `bmm`'s
  phase-3 conclusion (all reschedules were byte-identical no-ops).
  - Positive Test: `docs/phase3-exit-decision.md` states the terminal finding with the round-0
    depth-matched evidence and the per-variant screen results; v1 and v4 both retained.
  - Negative Test: promoting any candidate whose only "win" is within noise, or declaring failure/
    forcing a change when the honest result is "already optimal."

- AC-8: **Measurement protocol & noise rule (applies to every candidate).**
  - Decide on **TRUE per-matmul PE-active (PE-active/8704) + p50 latency**, NOT the coarse
    PE%/DMA% proxy (jitter 1–100% on identical kernels in the siblings).
  - **Bracket:** measure each candidate in a same-session interleaved bracket against v4
    (v4→candidate→v4, or an alternating v4/candidate sequence), reporting p50 (and, where available,
    min/p90 or the spread) for both. Re-anchor v4 same-session before each comparison.
  - **Out-of-noise = non-overlapping same-session brackets, treating a ~1.5–2% band as noise**
    (the convention used throughout the sibling `benchmark.csv` files; e.g. v4's M16 {3.7461,3.7462}
    vs M32 {3.7719,3.7735} were non-overlapping). A ≥~2% wall p50 gap that is bracket-non-overlapping
    is the promotion signal.
  - Positive Test: a promoted candidate's bracket does not overlap v4's and the gap exceeds the noise
    band. Negative Test: promoting on a single unbracketed `--fast` delta, or on overlapping brackets.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
Round 0 fully executed (v4 same-session re-anchor + depth-matched pure-bmm-M16 anchor + static
epilogue engine-placement record + the one AC-6b copy-elimination diagnostic); D1 copy-placement
swept over the predeclared variants (AC-4.1) with normalize held, then normalize-placement swept on
the winner (AC-4.2); if a D1 win lands, D2 `M_SUB ∈ {16,32}` re-sweep on the winner (AC-5); the best
bit-exact correct candidate promoted with a full 5-seed run and full evidence in `benchmark.csv` /
`candidates.jsonl` / `profile/`; `docs/phase3-exit-decision.md` written with keep/revise/reject per
direction and the before/after evidence; `[[kda-bmm-softmax-progress]]` memory updated. All within
≤5 optimization iterations.

### Lower Bound (Minimum Acceptable Scope)
Round 0 executed and recorded; the D1 copy-placement variants screened with `--fast` + `dump_metrics`;
the honest verdict reached and written to `docs/phase3-exit-decision.md` — **including the terminal
"v4 is already engine-balanced, no promotion" outcome (AC-7) if the measurements support it.** v1 and
v4 both retained as fallbacks. No candidate is promoted unless it clears AC-1, AC-2, and AC-8.

### Allowed Choices
- **Can use:** `nisa.tensor_copy(dst, src, engine=nki.isa.engine.scalar|vector)` for PSUM→SBUF score
  copies; `nisa.activation(op=nl.copy, scale=recip[P,1])` (Scalar) or `nisa.tensor_scalar(op0=multiply,
  operand0=recip)` (Vector) for the normalize; per-chunk alternating engine assignment; the existing
  two-phase transpose-all schedule and `M_SUB` blocking within one batch; `tensor_reduce(nl.max,
  negate=True)` and explicit `tensor_reduce(nl.add)` (unchanged from v4).
- **Cannot use:** bf16 / tf32 / bf16x2 anywhere in matmul or softmax; bf16 output; any op that holds a
  PSUM bank alive through the 4096-wide `max`/`exp` (copies must drain PSUM to SBUF immediately, as in
  v4); fused `activation(reduce_res=)` row-sum (measured anti-lever); GpSimd for copies or the
  normalize (API-infeasible); cross-batch blocking or cross-batch double-buffer; removing the
  max-reduce or the normalize pass; widening the matmul tile past the 512-wide PSUM-bank wall.
- This is a **near-deterministic design**: the goal, math, tile shapes, and closed levers are fixed;
  the only open choice is the epilogue engine placement (AC-4) and the contingent `M_SUB` point
  (AC-5). Upper and lower bounds converge on "measure honestly; promote only a bracket-clean bit-exact
  win, else prove terminal."

## Feasibility Hints and Suggestions

> **Note**: Reference only — conceptual, not prescriptive.

### Conceptual Approach
1. **Round 0 (analyze + measure, no production code):** re-anchor v4 same-session via
   `runs/dump_metrics.py`; measure `../bmm/runs/bmm_d1_twophase_m16.py` same-session for the
   depth-matched control; record the static engine of each of the 8 score copies + the normalize from
   the per-engine counts + an engine-annotated dump; run the one AC-6b `--fast` copy-elimination
   diagnostic. Decide from AC-3.1 whether the back-pressure premise holds.
2. **D1 copy placement (AC-4.1):** clone v4 → swap the 8 `nl.copy` score drains for
   `nisa.tensor_copy(engine=...)` under the predeclared variants; `--fast` + `dump_metrics` screen
   each; keep the one that moves wall p50 out-of-noise (bracket vs v4). The 4/4-split-vs-all-Vec
   comparison itself empirically answers whether Vector and Scalar drain PSUM in parallel on trn2 (if
   4/4 does not beat all-Vec, parallel drain is not happening — do not assume it).
3. **D1 normalize placement (AC-4.2):** on the winning copy assignment, try normalize on Scalar vs
   Vector; keep the better bracket.
4. **D2 (AC-5), only if D1 landed:** re-sweep `M_SUB ∈ {16,32}` on the winner.
5. **Promote** the best bit-exact correct candidate with a full 5-seed run, or **record terminal**
   (AC-7). Write the exit decision either way.

Pseudocode sketch of the D1 copy-placement change (per n-chunk `c`, replacing v4's `nl.copy` drain):
```
# v4 today:  score[:, c*512:+512] = nl.copy(acc)              # engine chosen by compiler
# D1 variant: pick engine per chunk index to split the drain across two engines
eng = nki.isa.engine.scalar if (c % 2 == 1) else nki.isa.engine.vector   # variant (iv), 4/4
score[:, c*512:+512] = nisa.tensor_copy(acc, engine=eng)                 # bit-exact fp32 copy
```

### Relevant References
- `runs/bmm_softmax_v4.py` — phase-3 base kernel (the file each D1 variant clones).
- `runs/bmm_softmax_v1.py` — fp32 fallback / DAG root.
- `runs/dump_metrics.py` — TRUE PE-active + per-engine actives + counts + HBM + per-seed rel-L2.
- `../bmm/runs/bmm_d1_twophase_m16.py` — the depth-matched (M16) pure-bmm control anchor (AC-3).
- `docs/phase2-exit-decision.md`, `docs/phase2-bottleneck-evidence.md` — how v4 was reached; closed levers.
- `profile/bmm_softmax_v4_digest.txt` — the anchor counters.
- `verify.py` (repo root) — the 5-seed relative-L2 correctness gate.

## Dependencies and Sequence

### Milestones
1. **Round 0 — controlled anchors & causal check** (blocks everything):
   - Phase A: same-session v4 re-anchor + depth-matched pure-bmm-M16 anchor; report the depth-matched
     PE-active gap.
   - Phase B: record the static engine placement of the 8 score copies + normalize.
   - Phase C: the one AC-6b `--fast` copy-elimination diagnostic; apply AC-3.1 to decide whether to
     pursue D1 or lean terminal.
2. **D1 — epilogue engine rebalance** (depends on Milestone 1):
   - Phase A: copy-placement sweep (AC-4.1), normalize held.
   - Phase B: normalize-placement sweep on the winner (AC-4.2).
3. **D2 — `M_SUB` re-sweep** (depends on Milestone 2 landing a promotable win, AC-5).
4. **Exit** (depends on 2, and 3 if it ran): promote the best bit-exact candidate with a full 5-seed
   run OR record the terminal AC-7 outcome; write `docs/phase3-exit-decision.md`; update
   `[[kda-bmm-softmax-progress]]`.

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Round-0 same-session re-anchor of v4 + depth-matched pure-bmm-M16 anchor; report the M16-vs-M16 PE-active gap | AC-3 | coding | - |
| task2 | Record the static engine placement of the 8 score copies + normalize (per-engine counts + engine-annotated dump) | AC-3 | coding | task1 |
| task3 | One `--fast` copy-elimination diagnostic probe (`*_copyelim_diag.py`); interpret via AC-3.1 whether back-pressure premise holds | AC-6b, AC-3.1 | coding | task2 |
| task4 | Assess round-0 evidence: is the depth-matched gap + diagnostic consistent with the back-pressure story, or does it favor terminal? | AC-3.1, AC-7 | analyze | task3 |
| task5 | D1 copy-placement sweep (predeclared variants) via `nisa.tensor_copy(engine=...)`, normalize held; `--fast`+`dump_metrics` screen each; bracket the survivor vs v4 | AC-4.1, AC-8 | coding | task4 |
| task6 | D1 normalize-placement sweep (Scalar vs Vector) on the winning copy assignment; bracket vs v4 | AC-4.2, AC-8 | coding | task5 |
| task7 | Verify purity guards + full 5-seed correctness on any promotable D1 candidate | AC-1, AC-2 | coding | task6 |
| task8 | D2 `M_SUB ∈ {16,32}` re-sweep on the D1 winner — ONLY if D1 landed and purity held | AC-5 | coding | task7 |
| task9 | Write `docs/phase3-exit-decision.md` (keep/revise/reject per direction, before/after evidence, terminal-if-applicable); record all candidates in `benchmark.csv`/`candidates.jsonl`/`profile/`; update `[[kda-bmm-softmax-progress]]` | AC-6, AC-7 | coding | task7, task8 |

## Claude-Codex Deliberation

### Agreements
- Classic shape specialization has no surface (clean-dividing axes, tiles at the 512-wide PSUM-bank
  wall, full-row resident softmax) — phase 3 is correctly reframed to the epilogue engine schedule.
- Round 0 must measure a **same-depth** pure-bmm-M16 control (`../bmm/runs/bmm_d1_twophase_m16.py`),
  not rest on the depth-confounded v4-M16-vs-pure-bmm-M32 "+68%" headline.
- D1 must split **copy placement** from **normalize placement** as separate levers (not the bundled
  variant A), so attribution is clean; use `nisa.tensor_copy(engine=...)`, not `nl.copy`.
- Purity guards must include `psum_read_sbuf_write_count`=4224, HBM floor, and spill=0 — not only
  `matmul_instruction_count`.
- One `--fast` copy-elimination diagnostic (AC-6b) is worth the budget to test the central causal
  claim directly; it is diagnostic-only and never promotable.
- The terminal "v4 is already engine-balanced" outcome (AC-7) is an acceptable, honest phase exit.
- **DEC-1:** promotion is gated by **wall p50 out-of-noise**; TRUE PE-active is corroborating
  mechanism evidence, not the gate. A PE-active-only improvement with flat wall is NOT promoted.
- **DEC-2:** allow exactly one `--fast` diagnostic copy-elimination probe, counted against the budget.

### Resolved Disagreements
- **Depth confound in the "+68%" headline:** Codex flagged that v4 (`M_SUB=16`) vs pure-bmm-`M_SUB=32`
  overstates the softmax-specific gap (a same-depth M16 comparator makes the controlled gap far
  smaller). *Resolution:* AC-3(b) adds the same-session depth-matched pure-bmm-M16 anchor; the plan no
  longer rests on +68%. The draft's headline is retained in the appended draft as the original
  hypothesis but is explicitly downgraded to "directional, depth-confounded" in the plan.
- **Profiler semantics ("stalls counted as PE-active"):** Codex noted `tensor_engine_active_time_ns`
  may be a busy counter that *excludes* semaphore waits, which would invalidate the "stalls inflate
  PE-active" premise; and no Neuron Explorer timeline is available in this harness. *Resolution:*
  AC-3.1 disambiguates via the depth-matched gap + the AC-6b diagnostic rather than a timeline; if
  both point away from back-pressure, the phase leans terminal (AC-7). Named the top core risk.
- **Kill-criterion too strict:** Codex argued byte-identical PE-active should not auto-reject if wall
  p50 improves (reduced non-PE tail / overlap). *Resolution:* KILL-CRITERION and AC-4 negative tests
  now reject a PE no-op *only when wall is also flat/worse*; a wall win with flat PE-active is
  investigated, not auto-rejected. DEC-1 keeps wall as the gate.
- **"Byte-identical" applied to nanosecond measurements:** Codex noted active-ns are measurements, not
  bytes. *Resolution:* the plan says "within profiler/count precision" for ns/actives and reserves
  bit-identical for output/rel-L2 and instruction counts.
- **AC-1 over-stated a single rel-L2 scalar:** Codex noted different seeds may legitimately differ.
  *Resolution:* AC-1 now requires **per-seed** equality to same-session v4 for that seed (v4 happens
  to report the same figure on all 5 seeds, but the invariant is per-seed match, not a global
  constant).
- **"Attribute" the epilogue engine placement was too strong:** *Resolution:* AC-3 records **static
  engine assignment** (from counts + dump), explicitly not a claim about dynamic scheduling/overlap
  that summary metrics cannot prove.
- **Search creep in D1 variants:** *Resolution:* AC-4.1 predeclares the copy-placement variants
  (all-current / all-Scalar / all-Vector / 4-4; 2-6 or 6-2 only on signal).
- **Budget wording ambiguity:** *Resolution:* AC-6b and the KILL-CRITERION state the AC-6b diagnostic
  **counts** toward the ≤5 optimization iterations; round-0 re-anchors and closed-lever records do not.

### Convergence Status
- Final Status: `converged` — round 1 produced agreement on direction and both DEC positions; all of
  Codex's required changes were mechanical refinements with no opposing position and were adopted in
  full. No high-impact disagreement remains.

## Pending User Decisions

- DEC-1: **Promotion gate — wall p50 out-of-noise vs require both wall and PE-active to move.**
  - Claude Position: wall p50 out-of-noise (AC-8 bracket) is the gate; TRUE PE-active is corroborating
    mechanism evidence; a PE-active-only win with flat wall is NOT promoted.
  - Codex Position: agrees with Claude's position.
  - Tradeoff Summary: wall is the user-visible score; gating on PE-active alone risks promoting a
    mechanism artifact that does not move the score. No opposing position; adopted as the plan's rule.
  - Decision Status: `RESOLVED — wall p50 out-of-noise is the gate, PE-active corroborates` (both
    sides agree; recorded for user awareness, revise if the user disagrees).

- DEC-2: **Allow the one `--fast` diagnostic copy-elimination probe (AC-6b)?**
  - Claude Position: yes — one `--fast` run to test the central back-pressure claim; diagnostic-only,
    never promotable; counts against the ≤5 budget.
  - Codex Position: agrees.
  - Tradeoff Summary: costs one iteration but directly falsifies/confirms the phase's premise; without
    it, a null D1 result is ambiguous between "no back-pressure" and "back-pressure the reschedule
    couldn't reach." No opposing position; adopted.
  - Decision Status: `RESOLVED — allowed, counts against budget` (revise if the user disagrees).

- DEC-3: **Concrete noise threshold / bracket rule for "out-of-noise."**
  - Claude Position: adopt the sibling convention already in the repo — same-session interleaved
    bracket, ~1.5–2% band treated as noise, non-overlapping brackets required, with a ≥~2% wall p50
    gap as the promotion signal (AC-8).
  - Codex Position: agrees the plan is directionally correct but asked that the exact rule be made
    concrete before convergence.
  - Tradeoff Summary: a tighter band risks calling real wins noise; a looser one risks promoting
    jitter. The ~1.5–2% non-overlapping-bracket convention is what every sibling `benchmark.csv` used
    and is now written into AC-8.
  - Decision Status: `RESOLVED in-plan via AC-8 (~1.5–2% band, non-overlapping same-session brackets)`
    — surfaced here for explicit user awareness; the user may tighten/loosen the band.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Phase", "Step", "DEC-", "D1/D2", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting kernel code.
- Use descriptive, domain-appropriate naming in code (e.g. `score_copy_engine`, `normalize_on_scalar`,
  `bmm_softmax_v5_copysplit.py`, `bmm_softmax_copyelim_diag.py`) instead.

### Measurement & Evidence Discipline (from the draft §7 and the deliberation)
From `workspaces/bmm_softmax/`:
```bash
python3 \
    ../../verify.py --op bmm_softmax --candidate runs/<file>.py --fast   # gate first
python3 \
    runs/dump_metrics.py runs/<file>.py --fast                            # engine/PE-active screen
# then drop --fast on both for the promotion measurement (full 5-seed)
```
Decide on **TRUE per-matmul PE-active + p50 latency**, NOT coarse PE%/DMA% (jitter 1–100% on identical
kernels in the siblings). Re-anchor v4 same-session before each comparison; ~1.5–2% band = noise,
non-overlapping brackets = out-of-noise (AC-8). Capture a metrics table per direction diffing vs v4:
wall p50 (+ spread), TRUE PE-active ns, Vector-active ns, Scalar-active ns, `matmul_instruction_count`
(must stay 8704), `psum_read_sbuf_write_count` (4224), HBM read/write (floor), spill (0), and per-seed
rel-L2. Log every perf change in `benchmark.csv`; each candidate in `candidates.jsonl` with parent
links (DAG root `bmm_softmax_v1`, phase-3 base `bmm_softmax_v4`); evidence under `profile/`.

### KILL-CRITERION (governs every D1/D2 candidate)
Screen every variant with `--fast` + `dump_metrics` first. **Reject** a variant when it is a compiler
no-op *and* wall is flat (TRUE PE-active within precision of v4 AND wall p50 flat), or an anti-lever
(PE-active rises with no wall win), or a bottleneck-mover (Vector-active drops but a Scalar-active
rise exposes a new tail that erases the wall win). **Investigate** (do not auto-reject) a variant whose
wall p50 improves out-of-noise even with flat PE-active. ≤5 optimization iterations total; the AC-6b
diagnostic probe counts toward that budget; round-0 re-anchors and closed-lever records do not.

### Correctness Guardrails (never regress)
- fp32 throughout matmul and softmax; no bf16/tf32.
- Max-shifted softmax preserved: `exp(score − row_max)` then divide by the row sum, reduction over the
  N free axis (reference axis 2). Every candidate is a pure engine reassignment / schedule change —
  same set of fp32 `exp` terms summed in the same order ⇒ per-seed rel-L2 must match same-session v4
  (bit-identical). Any drift = an indexing/placement bug → reject.
- No softmax reduce/activation/elementwise op on a PSUM tile that would hold banks — copies drain PSUM
  to SBUF immediately, as in v4.
- Any new SBUF buffer must be budget-checked (v4 uses ~72.5 KB/partition of ~208 KB usable; stay well
  under to keep spill=0 and HBM at floor).
- Every candidate: `--fast` (seed 42) pre-check + `dump_metrics`, then full 5-seed `verify.py` before
  any promotion; require `l2_norm_passed=True` on all seeds `[0,21,42,63,84]`.

## Output File Convention

This plan is the main output file `docs/plan-phase3.md`. `alternative_plan_language` resolves to empty
(disabled) in the merged Humanize config, so **no translated variant is written**.

--- Original Design Draft Start ---

# bmm_softmax — Phase 3 draft: regime / shape specialization

**Operator:** `bmm_softmax` (NKIBench case 2). `out[b] = softmax_N(lhs[b] @ rhs[b])`,
`b in 0..15`. `lhs (16,4096,64)=(B,M,K)`, `rhs (16,64,4096)=(B,K,N)` fp32 →
`out (16,4096,4096)`. **B=16, M=4096, K=64, N=4096.** Baseline **7.290 ms**.

**Start point:** `runs/bmm_softmax_v4.py` = the phase-2 promotion, **1.946x (3.7460 ms)**,
full 5-seed L2 PASS, on-device rel-L2 **2.5683307869e-6** (~7.8x under the 2e-5 gate).
Structure: sibling `bmm_v2` two-phase transpose-all schedule + max-negate fold + `M_SUB=16`
within-batch depth. Pure fp32; `bmm_softmax_v1` (1.585x) retained as the fp32 fallback / DAG root.

---

## 1. The phase-3 question, answered honestly up front

Phase 3 asks: *analyze where time goes across the tensor's structure and specialize only where
the measured win justifies the complexity (tile-size regimes, partition/free splits, edge tiles).*
As in the solved sibling `bmm`, the honest answer is that **classic shape specialization has no
surface here**, and I want that on the record with numbers so the phase does not chase a dead lever:

- **No edge tiles.** Every axis divides cleanly: `M=4096=32·128`, `N=4096=8·512`, `K=64≤128`
  (single Tensor-Engine pass), `B=16`. There is no ragged remainder to special-case — the usual
  "specialize the edge" regime split does not exist.
- **Tiles are already maximal and cannot be widened.** The main matmul is `[K=64]×[64,512]→[128,512]`.
  The moving free dim 512 is the **hard PSUM-bank wall on trn2** (one `nc_matmul` writes one bank =
  512 fp32 elems/partition; 2048/4096-wide is trn3-only). The stationary free dim 128 fills the PE
  columns. No "bigger tile" regime to switch into. The softmax reduces over the **full N=4096 row**,
  which already lives in one resident SBUF tile — no N-tiling regime either.
- **The K=64 partition/free split is fixed and cheap.** K=64 fills 64 of 128 PE partition rows, but
  the trn2 matmul cost is `dst_free_elems·100/freq` — proportional to the moving free dim (512) ONLY,
  independent of K. A half-full contraction costs nothing extra; there is no partition-split regime
  that recovers it. K cannot be packed across batches (`out[b]` are block-diagonal — closed in `bmm`).

So the phase-3 "structure" to analyze is **not tile shape** — it is the **engine schedule of the
fused softmax epilogue relative to the matmul stream**, the one structural dimension phase 2 stopped
short of. Phase 2 tuned *within-batch stream depth* (`M_SUB`); phase 3 attacks the *engine placement*
of the epilogue that is inflating the Tensor Engine.

## 2. The corrected bottleneck — softmax back-pressures the matmul (the reframe that drives the phase)

Phase-2 exit called the kernel "PE-bound, pure-fp32 schedule levers exhausted." That is true but
under-specified. The authoritative same-session counters (`profile/bmm_softmax_v4_digest.txt`,
divided by the metric window) versus the pure-`bmm` sibling that runs the **byte-identical matmul**:

| signal | bmm_softmax_v4 (fused) | pure bmm_v2 (same 8704 matmuls) | note |
|---|---|---|---|
| p50 wall | **3.746 ms** | 2.036 ms | |
| TRUE PE-active / inf | **3.371 ms** (89.99%) | **2.011 ms** (98.9%) | **identical matmul, +1.36 ms** |
| per-matmul PE-active | **0.387 µs** | 0.231 µs | **+0.156 µs (+67%)** |
| Vec-active / inf | 2.687 ms (71.74%) | ~0.9 ms (20%) | +1.8 ms of softmax Vec |
| Scalar-active / inf | 2.627 ms (70.14%) | ~0.6 ms (14%) | +2.0 ms of softmax Scalar |
| GpSimd-active / inf | 0.190 ms (5.06%) | — | **nearly idle** |
| DMA-active / inf | 1.400 ms (37.38%) | 1.43 ms | hidden, at floor |
| matmul_instruction_count | 8704 | 8704 | **identical work** |
| psum_read_sbuf_write_count | 4224 | 4224 | identical |
| HBM read / write | 33.6 / 1073.7 MB | 34 / 1074 MB | at read-once/write-once floor, spill=0 |

**The single fact that drives phase 3:** the matmul workload is *byte-identical* to pure `bmm`
(8704 instr, same tiles), yet its **PE-active is inflated from 2.011 ms → 3.371 ms (+68%)**. The
matmul is not doing more work — it is **stalling**. And the exposed tail is only
`wall − PE-active = 3.746 − 3.371 = 0.375 ms`. So the prize is **not** the 0.375 ms exposed tail —
it is the **1.36 ms of PE inflation**, if it can be relieved.

### Where the inflation comes from — the PSUM-drain / Vector-contention chain

The schedule is: per m-subtile, 8 matmuls each land a `[128,512]` result in **one of the 8 PSUM
banks**, then each bank is copied out to the resident `score[128,4096]` SBUF tile (the
`psum_read_sbuf_write` copies), then the softmax runs on `score`. The copies exist **structurally**:
`score` must reach SBUF so the 8 PSUM banks free up for the *next* subtile's 8 matmuls — there is no
PSUM double-buffer (only 8 banks, all in use), and `score` cannot stay in PSUM because `exp` needs it
*after* the full-row `max`, by which time the banks are needed again. (Copy-elimination — reduce/exp
directly from PSUM — was considered and rejected on paper: it holds all 8 banks alive through the
expensive 4096-wide `max`+`exp`, which *lengthens* the bank-free critical path and *worsens*
pipelining. The copies are the fast drain, not waste.)

The chain that inflates PE: the copies and the softmax reductions (`max`, `sum`) both want the
**Vector engine**; the `exp` wants the **Scalar engine**; both are at ~70%. When a bank's copy is
queued behind softmax Vector/Scalar work, that PSUM bank stays occupied, so the next matmul that
targets it **stalls** — and the profiler counts the stall as PE-active. In pure `bmm`, Vector sits at
~20% idle so the copies drain instantly and per-matmul PE-active is 0.231 µs; here the loaded
Vec/Scalar engines throttle the drain and per-matmul PE-active rises to 0.387 µs. **The lever is to
drain the PSUM banks faster by spreading the epilogue across engines so the matmul stops waiting.**

This is a *real, mechanistic* contention that pure `bmm` lacked — `bmm`'s phase-3 PE bubble (13%
above floor) had no contention source (Vec idle), which is exactly why `bmm`'s reschedules were
no-ops. Here Vec+Scalar are both hot with a drain that competes with the matmul, so there is a
genuine mechanism to attack. (The no-op risk still applies — see the kill-criterion — but the
premise is stronger than `bmm`'s was.)

### Theoretical floor (for calibration, not a hard prediction)

Pure-fp32 PE floor (from `bmm`, free-dim-only cost model, 2.4 GHz): 8192 main passes @ 512-wide
(213.3 ns) + 512 transposes @ 128-wide (53.3 ns) = **1.775 ms**. The softmax `exp` is Scalar-only and
irreducible at ~512×4096-wide ≈ 1.75 ms; the two reductions (`max`,`sum`) are **Vector-only**
(`tensor_reduce` has no `engine=` arg — API-confirmed) and also irreducible. So the wall floor is
`max(PE, Vector-mandatory, Scalar-mandatory)` once the *movable* work (copies, normalize) is packed
into the gaps. **Caveat:** the per-op cost-model numbers do **not** cleanly reconcile with v4's
measured instruction counts (the compiler lowers/places the 4096-wide ops and the copies in ways the
simple model doesn't capture), so I will **not** hard-predict a floor — I will *measure* the current
engine placement in round 0 and treat the arithmetic as directional only. Best realistic PE target =
the pure-`bmm` hidden floor **2.011 ms** (softmax fully overlapped, zero back-pressure); if reached
with today's 0.375 ms tail, wall ≈ 2.4 ms → **~3.0x**. That is the optimistic ceiling, not a promise.

## 3. Round 0 — measurements before any code change (near-zero remote risk)

Re-use `runs/dump_metrics.py` (reads TRUE `tensor_engine_active_time_ns` + per-engine actives +
instruction counts, not the jittery PE%/DMA% proxy). All same-session vs a fresh v4 anchor.

1. **Re-anchor v4 counters** — confirm TRUE PE-active ≈ 3.371 ms, per-matmul PE-active ≈ 0.387 µs,
   matmul_instr 8704, HBM 33.6/1073.7 MB (spill 0). This is the fact the phase rests on; verify fresh.
2. **Attribute the current engine placement of the epilogue.** The compiler already chose engines for
   the 8 score copies and the normalize (v4's counts — Vec 3472 / Scalar 4628 instr — suggest some
   copies are *already* off Vector). Establish *which engine each epilogue op currently lands on*
   before touching it, so a rebalance is measured against the real starting placement, not an assumed
   one. (No new probe if the counts + a one-shot engine-annotated dump settle it.)
3. **(record-only) confirm the precision lever stays closed.** Phase-2/`bmm` measured fp32/bf16 pass
   ratio 2.0, so a 3-product bf16x2 main matmul costs 3.0 passes > 2.0 and *raises* PE on a PE-bound
   kernel; bf16 `exp`/softmax over N=4096 ≈ 1e-2 rel error, blowing the 2e-5 gate. Both closed; state
   it so the phase does not relitigate.

## 4. Optimization directions, ranked by expected benefit × confidence

### D1 — Epilogue engine rebalancing to relieve PSUM-drain back-pressure (PRIMARY)

**Hypothesis.** Per subtile, 8 `[128,512]` PSUM→SBUF score copies must drain before the next 8
matmuls reuse the banks. If those copies serialize on one loaded engine, the drain gates the matmul
and inflates PE-active. Spreading the drain (and the normalize) across Vector **and** Scalar frees the
banks faster → per-matmul PE-active drops toward the 0.231 µs pure-`bmm` floor.

**Feasibility (API-confirmed, `/nki-api-reference`):**
- `nisa.tensor_copy(dst, src, engine=nki.isa.engine.scalar)` moves a PSUM→SBUF copy to the Scalar
  engine (GpSimd is **not** allowed — it cannot read PSUM). Bit-exact fp32 copy on trn2 Scalar.
- The normalize can run on Scalar via `nisa.activation(op=nl.copy, data=exp_t, scale=recip[P,1])`
  (per-partition `[P,1]` scale is legal), or stay on Vector as `tensor_scalar(op0=multiply,
  operand0=recip)`.
- Direct precedents: `fd27f7ef` (`attention_tkg`) moved a hot `tensor_copy` to ScalarE to relieve
  VectorE; `63e18e33` (`attention_cte`) moved a normalize the *other* way (Scalar→Vector) because
  ScalarE was the bottleneck; `597cf19e` (`mlp_tkg`) *alternates* engines per iteration to consume
  two engines' bandwidth. → the choice is **profile-driven balancing**, not a fixed direction, which
  is why D1 is a small sweep, not a single edit.

**Variants to sweep (all bit-exact — pure engine reassignment, same fp32 math):**
- **A — split the 8 score copies 4 Vec / 4 Scalar** (alternate by chunk index, per `597cf19e`),
  normalize on Scalar. Halves the *serial* drain latency across two engines while keeping the two
  mandatory reductions on Vector.
- **B — all 8 copies on Scalar, normalize on Vector.** Frees Vector for the reductions.
- **C — all 8 copies on Vector, normalize on Scalar.** The mirror.
- (Round 0 tells us the current placement; only sweep the variants that differ from it.)

**Decision metric:** promote the variant with the lowest **TRUE per-matmul PE-active** (out-of-noise),
with HBM staying at the 33.6/1073.7 MB floor and matmul_instr = 8704. Wall p50 is the tie-break.

**Expected:** if drain contention is the cause and a split halves it, per-matmul PE-active
0.387 → ~0.30 µs, PE-active 3.37 → ~2.6 ms, wall ~2.9 ms → **~2.5x**. Numbers are hypotheses; the
profile gates them.

**Kill-criterion (inherited from `bmm` phase 3).** The `affine_range` compiler already pipelines
aggressively and flattened many `bmm` reschedules to byte-identical no-ops; it may already have chosen
a near-optimal engine placement (round-0 step 2 will show how close). **Screen every variant with
`--fast` + `dump_metrics` first; reject immediately any variant whose TRUE PE-active is byte-identical
to v4 (compiler no-op) or rises (anti-lever), exactly as `bmm` rejected its cross-batch reschedules.**
≤3 iterations (the copy-split + at most two engine-assignment points).

### D2 — `M_SUB` re-sweep on the winning engine assignment (SECONDARY, contingent)

Phase 2 found the interior optimum `M_SUB=16` *given v4's engine placement*. A faster-draining
epilogue (D1) changes the matmul↔softmax overlap, so the optimal stream depth may shift — a deeper
stream (`M_SUB=32`) could become viable again once the drain no longer gates it. **Only if D1 lands**,
re-sweep `M_SUB ∈ {16, 32}` on the winning assignment (≤2 iterations, reuses D1's kernel, within one
batch only — never cross-batch). Low-medium value; a within-winner tie-break, not a phase driver.

## 5. Closed / not-pursued (record-only — do NOT spend iterations)

- **bf16x2 3-product matmul split.** fp32/bf16 pass ratio 2.0 < 3.0 ⇒ split *raises* PE on a PE-bound
  kernel. `[[BL-20260710-bf16x2-loses-when-fp32-emulates-in-2-passes]]`. Closed.
- **bf16 `exp`/softmax.** ~1e-2 rel error over N=4096 » the 2e-5 gate (current margin only 7.8x). Closed.
- **Cross-batch blocking / cross-batch double-buffer.** Measured **anti-lever** in `bmm` phase 3
  (per-matmul stall 0.231→0.296(B2)→0.332(B4) µs monotone regression — the batch boundary is a helpful
  `affine_range` pipeline reset). `[[BL-20260710-cross-batch-blocking-is-an-antilever-on-affine-range]]`.
  D1/D2 stay **within one batch**. Closed.
- **GpSimd normalize / GpSimd copies (recruiting the idle 5% engine).** **API-infeasible**, not just
  precondition-false: `tensor_scalar(engine=gpsimd)` is **rsqrt-only** (no general `[P,1]` multiply),
  and GpSimd **cannot access PSUM** (so it cannot do the score copies). This *upgrades* phase-2's
  "GpSimd precondition false" to a hard infeasibility — the idle GpSimd is genuinely unusable for this
  epilogue. Do not build; do not re-probe.
- **Fused `activation(reduce_op=add, reduce_res=)` exp+row-sum.** Phase-2 measured reject: the
  `reduce_res` accumulator side-effect triggered a **whole-stream 2× recompute** (matmul 8704→17408,
  +75% wall). `profile/bmm_softmax_d2_compare.md`. Closed — keep the explicit `tensor_reduce(add)`.
- **Removing the max-reduce or the normalize pass.** Both required (overflow-safe max-shift; softmax
  normalization) and already minimal. Keep.
- **Copy-elimination (reduce/exp directly from PSUM).** Rejected on paper (§2): holds all 8 banks
  through the 4096-wide `max`+`exp`, lengthening the bank-free critical path — worsens pipelining on a
  PE-bound kernel. Do not build.
- **Wider matmul tile / narrower N_CHUNK / K-packing / DMA store-burst / bf16 output.** All closed in
  `bmm` (PSUM-bank wall; block-diagonal; DMA hidden at floor; 2e-5 bans bf16 out). Not re-probed.

## 6. Correctness guardrails (never regress)

- fp32 throughout matmul and softmax; no bf16/tf32.
- Max-shifted softmax preserved: `exp(score − row_max)` then divide by the row sum, reduction over the
  **N free axis** (reference axis 2). Every D1/D2 candidate is a **pure engine reassignment / schedule
  change** — same set of fp32 exp terms summed in the same order ⇒ rel-L2 must stay **2.5683307869e-6**
  (bit-identical). Any drift = an indexing/placement bug, reject.
- No softmax reduce/activation/elementwise op on a PSUM tile that would hold banks (copies drain PSUM
  to SBUF immediately, as in v4).
- Every candidate: `--fast` (seed 42) pre-check + `dump_metrics`, then **full 5-seed** `verify.py`
  before any promotion; require `l2_norm_passed=True` on all seeds `[0,21,42,63,84]`.

## 7. Measurement protocol (per candidate)

From `workspaces/bmm_softmax/`:
```bash
python3 \
    ../../verify.py --op bmm_softmax --candidate runs/<file>.py --fast   # gate first
python3 \
    runs/dump_metrics.py runs/<file>.py --fast                            # engine/PE-active screen
# then drop --fast on both for the promotion measurement
```
Decide on **TRUE per-matmul PE-active (ms) + p50 latency**, NOT coarse PE%/DMA% (jitter 1–100% on
identical kernels in the siblings). Re-anchor v4 same-session before each comparison; treat a ~1.5–2%
band as noise. Capture the digest per direction; diff vs v4 on: per-matmul PE-active, per-engine
active ms, matmul_instr (must stay 8704), psum copies, HBM (must stay at floor), rel-L2 (must stay
2.5683e-6). Log every perf change in `benchmark.csv`; each candidate in `candidates.jsonl` with parent
links (DAG root `bmm_softmax_v1`, phase-3 base `bmm_softmax_v4`); evidence under `profile/`.

## 8. Expected trajectory & exit

- `bmm_softmax_v4 1.946x → D1 epilogue engine rebalance ~2.2–2.6x` **if** the PSUM-drain contention is
  real and not already compiler-balanced; optimistic ceiling ~3.0x (PE-active → the pure-`bmm`
  2.011 ms hidden floor with today's 0.375 ms tail). D2 `M_SUB` re-sweep is a within-winner tie-break.
- **If D1 comes back byte-identical** (compiler already placed the epilogue optimally, like `bmm`'s
  reschedules), the honest phase-3 conclusion is that v4 is at the fused kernel's engine-balanced
  optimum with no remaining schedulable structure — record that as terminal and keep v4. The whole
  phase is one mechanistic lever (relieve the softmax→PSUM-drain→matmul back-pressure by engine
  balancing), gated hard on TRUE per-matmul PE-active moving out of noise.
- **Promote** the best correct candidate; **keep `bmm_softmax_v1`** (fp32 fallback) and `bmm_softmax_v4`
  as fallbacks. Write `docs/phase3-exit-decision.md` with keep/revise/reject per direction and the
  before/after evidence, then update `[[kda-bmm-softmax-progress]]`. ≤5 optimization iterations
  (round-0 re-anchor + closed-lever records excluded from the budget).

--- Original Design Draft End ---
