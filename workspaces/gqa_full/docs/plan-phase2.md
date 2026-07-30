# gqa_full — Phase 2 Plan (profile-driven latency optimization)

## Goal Description

Reduce the on-device p50 latency of the `gqa_full` kernel below the current best
(`runs/gqa_full_v1.py`, 10.6579 ms / 1.462x over the 15.579 ms baseline) **without
regressing correctness** (relative-L2 < 2e-5 across seeds `[0,21,42,63,84]`), by
attacking the profiled bind: **serialization**, i.e. the 5.532 ms exposed tail
(`wall − TRUE PE-active = 10.658 − 5.125`). The largest single engine is PE at
5.125 ms/inf (48.1%), Vec 4.321 ms (40.5%) and Scl 4.505 ms (42.3%) co-limit, DMA
is at the read-once/write-once HBM floor (3.9%, no spill). `sum(PE+Vec+Scl) =
13.95 ms` overlaps down to only 3.29 ms today; a schedule that overlaps to the max
engine (~5.1 ms) is the primary lever, not shrinking any one engine.

The operator is grouped-query full (non-causal) softmax attention: B=1, N=4096,
QH=16, KH=8, n_rep=QH/KH=2, D=128, fp32. Per query head `qh` (kv head `kh=qh//2`):
`S = q_h @ k_h.T / sqrt(D)` → `A = softmax_over_key_axis(S)` → `O = A @ v_h`. The
phase-1 base fuses this per head (scores never touch HBM) with a per-tile softmax
epilogue and a 32-step context matmul.

Explore the ranked directions below for **at most five iterations each**, with
before/after `verify.py` latency + full `summary_metrics` evidence justifying
keep / revise / reject. Keep `gqa_full_v1` as the fp32 fallback (DAG root). The
phase may legitimately end keeping v1 if no direction produces a promotable win
(mirroring the sibling `bmm_softmax` phase-3 terminal outcome) — the deliverable
is an evidence-backed decision, not a forced promotion.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. Correctness is measured on-device by `verify.py`
(per-seed `l2_norm_passed`, the authoritative gate), **not** allclose.

- AC-1: **Correctness never regresses.** Every candidate kept or promoted passes
  the relative-L2 gate on all seeds `[0,21,42,63,84]`.
  - Positive Tests (expected to PASS):
    - `verify.py --op gqa_full --candidate runs/<file>.py` (full run, no `--fast`)
      reports PASS with every per-seed `l2_norm_passed=True` and per-seed
      `relative_l2_error < 2e-5` before any promotion.
    - `--fast` (1 seed) may be used for early iteration/rejection only.
  - Negative Tests (expected to FAIL / be rejected):
    - Any seed reports `l2_norm_passed=False` or `relative_l2_error >= 2e-5`.
    - A candidate promoted on `--fast` evidence alone (without a full 5-seed pass).
  - AC-1.1: **Pure-reschedule purity check (diagnostic, not a hard gate).** For a
    candidate claimed to be a pure schedule change (D1a/D1b), on-device
    `relative_l2_error` is expected to equal v1's `2.874266e-6` (precedent:
    `bmm_softmax_v2` was bit-for-bit). This is a **diagnostic**: a deviation flags
    an unintended algorithmic/accumulation-order change to investigate; promotion
    still gates on AC-1's `< 2e-5`, not on exact equality.
    - Positive: pure-reschedule candidate's per-seed rel-L2 == 2.874266e-6 (matches
      v1 per-seed); or, if it differs, a direct candidate-vs-v1 output diff explains
      it as benign fp32 reassociation and rel-L2 stays `< 2e-5`.
    - Negative: rel-L2 moves materially off 2.874266e-6 with no benign explanation
      (indicates the transform is not the intended pure reschedule).

- AC-2: **D1a — tile-local two-phase context loop** (M_SUB=1). For a single
  `(kh,grp,t_q)` query tile, first transpose all 32 `A` subtiles into resident SBUF
  `A_t` tiles, then stream the 32 context matmuls back-to-back into the single
  zero-initialized `o_psum[128,128]` accumulator **in the same `j` order** as v1, so
  the transpose copies drain in parallel with an uninterrupted matmul stream instead
  of a per-`j` `transpose→copy→matmul` serial chain.
  - Positive Tests (expected to PASS):
    - Wall p50 drops below v1's (re-baselined in the same profiler batch) while
      **absolute PE work stays flat**: `matmul_instruction_count` and
      `tensor_engine_instruction_count` unchanged vs v1, `TRUE PE-active` ms within
      profiler noise of 5.125 ms (a tail-hiding win, not a PE collapse).
    - HBM stays at the read-once/write-once floor (HBMrd ~67.2 MB, HBMwr ~33.6 MB,
      DMA ~4%, no spill); rel-L2 satisfies AC-1 (and AC-1.1 diagnostic).
  - Negative Tests (expected to FAIL / be rejected):
    - **Absolute** PE work rises materially (`matmul_instruction_count` or
      `tensor_engine_instruction_count` increases, or `TRUE PE-active` rises beyond
      noise) — note: PE *utilization percent* rising while wall drops is EXPECTED and
      is NOT a rejection reason.
    - HBMrd/HBMwr rise above the floor or DMA% climbs (spill), or wall is
      flat/regressed vs re-baselined v1, or AC-1 fails.

- AC-3: **D1b — small M_SUB query-tile batching**, WITHIN the natural per-`(kh,grp)`
  reuse group. Process `M_SUB` query tiles together (transpose their `A_t`, then
  stream), sweeping `M_SUB ∈ {1, 2, 4}` with `M_SUB=8` kept only as a
  confirming-spill/regression upper probe. Corrected SBUF accounting (see AC-6 and
  Codex resolution): the draft's `{8,16,32}` sweep and "32·16 KB = 512 KB" figure
  conflated one `A_t` subtile with the full attention row tile; the realistic
  no-spill window is small M_SUB. Run this sweep **under the D2 live-set** (D2 frees
  the attn buffer, changing which M_SUB fit) — see Dependencies.
  - Positive Tests (expected to PASS):
    - An interior `M_SUB` beats `M_SUB=1` (AC-2 winner) on wall with no spill (HBM at
      floor), absolute PE work flat, AC-1 held.
  - Negative Tests (expected to FAIL / be rejected):
    - Blocking spans kv-heads or query groups (cross-group/cross-batch blocking — a
      proven anti-lever, see `kda-bmm-progress`), which must never be attempted.
    - Any `M_SUB` spills (HBMrd above ~67 MB floor / DMA climbs), or wall regresses
      monotonically with `M_SUB` (then keep the best no-spill point, e.g. M_SUB=1).

- AC-4: **D2 — scale-fold + defer-normalize** (Vec reduction), built under the D1
  winner and also tested standalone-in-pieces. (i) Fold `1/sqrt(D)` into
  `activation(scale=)` with `bias = scale·(−max_unscaled)`, removing the 4096-wide
  `score*scale` `tensor_scalar` per tile. (ii) Defer normalization past the context
  matmul: run the context matmul on the **unnormalized** `exp` tile, then scale the
  `[128,128]` output `O` by the `[128,1]` reciprocal — turning the 4096-wide
  `attn = exp·recip` into a 128-wide `O·recip` (`(exp/sum)@v == (exp@v)/sum`, a
  positive per-row scalar pulled out of the contraction). `recip` still comes from
  the full-width row-sum. Test **scale-fold alone**, **defer-normalize alone**, then
  **combined**, isolating a numerical vs scheduling regression if one appears.
  - Positive Tests (expected to PASS):
    - Vec-active drops vs the D1 winner; rel-L2 stays `< 2e-5` (offline check gave
      5.36e-7 for the combined fold; expect ~2.9e-6 on-device); wall improves or is
      neutral (within the win bar); HBM stays at floor. Dynamic-range note recorded:
      deferred `O = exp@v` with `exp ∈ [0,1]` and row-sum up to ~4096 is fp32-safe.
  - Negative Tests (expected to FAIL / be rejected):
    - rel-L2 approaches or crosses 2e-5 (a sign/scale placement error in
      `activation(scale=,bias=)` — note v1's own `activation(op=nl.exp, bias=neg_max,
      scale=1.0)` already exercises the `exp(scale·data+bias)` semantics, so the
      folded form must match empirically).
    - Wall regresses after repeated measurement, or HBM rises above the floor.

- AC-5: **D3 — bf16x2 3-product split on the score matmul only** (GATED, ranked
  last). The score matmul has moving `[128,512]` (the moving-512 regime where
  sibling GEMMs' compensated bf16x2 split won); the context matmul moving is
  `[128,128]` (weak candidate — do not split it). Attempt **only if** the post-D1/D2
  profile shows PE is the wall-limiter (PE-active ≳ Vec-active and ≳ Scl-active).
  Correctness is **not** inferred from the sibling GEMM quadrature floor (softmax
  exponentiates score perturbations) — it must be measured on-device.
  - Positive Tests (expected to PASS):
    - Gate open (PE the bind post-D1/D2) AND both `TRUE PE-active` and wall drop AND
      5-seed rel-L2 `< 2e-5` measured on-device.
  - Negative Tests (expected to FAIL / be rejected):
    - The gate is closed (PE not the wall-limiter) and D3 is attempted anyway.
    - PE-active drops but wall is flat (score matmul is not the wall driver) → reject.
    - Measured 5-seed rel-L2 `>= 2e-5` → reject; keep the D1/D2 winner and v1 as fp32
      fallbacks.

- AC-6: **SBUF / PSUM accounting is explicit BEFORE each candidate is run** (not
  only before promotion). For each of D1-only, D2-only, and D1+D2, document the
  per-partition live set separately: shared `k_t` (16 KB), shared `v_sb` (16 KB),
  `q_t`, the softmax row buffers (`score`/`exp`/`attn` ~16 KB each, fp32
  `[128,4096]`), the `A_t` destination (`M_SUB · 32 · 512 B`), `max`/`row_sum`/
  `recip` vectors, `o_sb`, copy temporaries, and the `o_psum` PSUM bank count. PSUM
  banks live must be `<= 8`.
  - Positive Tests (expected to PASS):
    - The written accounting for the chosen `M_SUB`/transpose-depth predicts total
      SBUF within budget and PSUM `<= 8` banks, and the run confirms no spill (HBM at
      floor, DMA ~4%).
  - Negative Tests (expected to FAIL / be rejected):
    - A candidate is run without a prior live-set table, or the run spills (HBM above
      floor) while the table claimed it would fit (accounting error to correct).

- AC-7: **Evidence hygiene.** Every candidate is measured under the same profiler
  configuration, parent, and metric window as v1; full `summary_metrics` are
  recorded (PE/Vec/Scl/DMA %, MFU, HBMrd/HBMwr, `matmul_instruction_count`,
  `tensor_engine_instruction_count`); within-noise candidates are re-measured;
  v1 is re-baselined in the same batch before declaring a win.
  - Positive Tests (expected to PASS):
    - Each perf change appended to `benchmark.csv`; each candidate to
      `candidates.jsonl` with parent links (DAG under `gqa_full_v1`); profiling
      artifacts under `profile/`; the promoted candidate keeps v1 as fallback.
  - Negative Tests (expected to FAIL / be rejected):
    - A win declared from a single measurement within profiler noise, or against a
      differently-configured/older v1 baseline, or with `summary_metrics` missing.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.
Phase 2 is scoped to the **fixed shape** `B=1, N=4096, QH=16, KH=8, D=128` (shape
specialization is phase 3).

### Upper Bound (Maximum Acceptable Scope)
All four levers explored to their evidence-based conclusion: D1a tile-local
two-phase promoted; D1b M_SUB `{1,2,4}` (+8 probe) swept under the D2 live-set and
the interior optimum promoted; D2 scale-fold + defer-normalize stacked under the D1
winner; D3 score-only bf16x2 attempted iff its PE-bind gate opens — each with full
`summary_metrics` before/after evidence, SBUF/PSUM accounting, and 5-seed
correctness, the fastest correct candidate promoted, `gqa_full_v1` retained as the
fp32 fallback, and every lesson (including measured rejects) recorded to
`benchmark.csv` / `candidates.jsonl` / `profile/`.

### Lower Bound (Minimum Acceptable Scope)
The primary lever D1a (tile-local two-phase context loop) is implemented, scored
full-5-seed, and either promoted (if it beats v1 beyond the win bar with no spill
and no correctness regression) or kept-v1-with-evidence (if it washes/regresses),
with the round-0 bottleneck decision and D1a result documented. D1b/D2/D3 may be
truncated if D1a already lands a promotable win and remaining budget is better spent
verifying it, provided the truncation is recorded with rationale.

### Allowed Choices
- Can use: pure reschedule / two-phase transpose-all restructuring of the context
  loop; small `M_SUB` query-tile batching **within** the per-`(kh,grp)` reuse group;
  folding `1/sqrt(D)` into `activation(scale=)`; deferring softmax normalization past
  the context matmul; score-matmul-only bf16x2 3-product split (gated on the PE bind
  + measured rel-L2); any change that keeps HBM at the read-once/write-once floor.
- Cannot use: cross-kv-head or cross-query-group blocking (measured anti-lever,
  `kda-bmm-progress`); fused `activation` `reduce_res`/`reduce_cmd` row-sum (measured
  +75% anti-lever via score-stream recompute, `kda-bmm-softmax-progress`);
  online/flash chunked softmax (adds Vec rescaling, wrong direction while Vec
  co-limits — the score row already fits SBUF at 16 KB/partition); eliminating the q
  or A transposes (fundamental to softmax-over-free-axis + the two matmul contraction
  layouts — D1 hides them, does not remove them); any change that raises HBM traffic
  above the floor or forces a spill; changing the operator's fixed shape.

> **Note on Deterministic Designs**: The draft ranks a fixed set of directions
> (D1/D2/D3) with a prescribed iteration budget and gating logic. The bounds above
> reflect that: the *set* of allowed levers is fixed, but which lands as the promoted
> kernel is discovered empirically (the profiler decides), so upper and lower bounds
> differ by how far the ranked exploration is carried, not by which techniques are
> permitted.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are
> conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

D1a (tile-local two-phase), per `(kh,grp,t_q)`, replacing v1's interleaved context
loop (`for j in 32: transpose→copy→ o_psum += A_t·v`):

```
# Phase A — transpose-all: fill a resident A_t bank for this query tile
for j in range(32):
    a_t_ps = nc_matmul(attn[:, 128*j:+128], identity, is_transpose=True)  # PE
    a_t[j] = copy(a_t_ps)                                                  # Vec/Scl drains
# Phase B — matmul-stream: uninterrupted PE stream into one accumulator
o_psum = zeros([128,128])
for j in range(32):
    o_psum += nc_matmul(a_t[j], v_sb[:, 128*j:+128])   # same j order as v1
o_sb = copy(o_psum); store(...)
```

The `a_t` bank for one tile is 32 × `[128,128]` fp32 = 32 × 512 B = 16 KB/partition.
D1b generalises Phase A/B across `M_SUB` query tiles held together (each adds its own
16 KB softmax buffer + 16 KB `a_t` bank), which is why the no-spill window is small.

D2, layered on the winner: `activation(op=nl.exp, data=score, bias=scaled_neg_max,
scale=1/sqrt(D))` (removes the separate `score*scale`), keep the context matmul on
the unnormalized `exp`, and finish with `O = tensor_scalar(o_sb, *recip[128,1])`
(128-wide instead of 4096-wide). `recip` still comes from the full-width row-sum of
`exp`.

D3, iff PE is the bind after D1/D2: apply the compensated bf16x2 3-product split to
the score matmul (moving `[128,512]`) only; leave the context matmul and all
transposes fp32; measure 5-seed rel-L2 on-device.

### Relevant References
- `runs/gqa_full_v1.py` — the phase-1 base; the context loop to restructure is the
  `for j in nl.affine_range(T)` block; the softmax epilogue is the `tensor_scalar
  *scale → tensor_reduce max negate → activation exp → tensor_reduce add → reciprocal
  → tensor_scalar *recip` chain.
- `profile/gqa_full_phase2_bottleneck_evidence.txt`, `profile/gqa_full_v1_digest.txt`
  — round-0 evidence (exposed tail, transpose-site census, numeric-equivalence check).
- `runs/dump_metrics.py` — the gqa-adapted digest helper for full `summary_metrics`.
- `../../verify.py` — the authoritative 5-seed rel-L2 gate.
- Sibling precedents (memory): `kda-bmm-softmax-progress` (two-phase tail-hiding win,
  interior M_SUB, fused-reduce anti-lever), `kda-bmm-progress` (cross-batch/group
  blocking anti-lever), `kda-matmul-progress`/`kda-transpose-matmul-progress`
  (bf16x2 split on moving-512).

## Dependencies and Sequence

### Milestones
1. **M0 — Round-0 bottleneck confirmation (already captured).** The serialization
   diagnosis and transpose census in `profile/gqa_full_phase2_bottleneck_evidence.txt`
   are the entry state; re-baseline v1 in the current profiler batch before any win
   is declared (AC-7).
2. **M1 — D1a tile-local two-phase (PRIMARY).**
   - Phase A: implement the two-phase restructure at M_SUB=1; write SBUF/PSUM
     accounting (AC-6) first.
   - Phase B: score full-5-seed + full `summary_metrics`; keep/revise/reject per AC-2
     and AC-1/AC-1.1.
3. **M2 — D2 scale-fold + defer-normalize.** Built under the D1a winner (or under v1
   if D1a washed). Test scale-fold alone, defer-normalize alone, then combined (AC-4).
   This precedes the M_SUB sweep because D2 frees the attn buffer and changes the
   feasible M_SUB set (Codex REQUIRED_CHANGE, resolved).
4. **M3 — D1b M_SUB sweep** `{1,2,4}` (+8 probe), under the D2 live-set, within the
   reuse group (AC-3, AC-6). Pick the interior optimum.
5. **M4 — D3 bf16x2 score split (GATED).** Only if the post-D1/D2 profile shows PE is
   the wall-limiter (AC-5). At most 2 iterations.
6. **M5 — Promote + record.** Promote the fastest correct candidate; keep v1 as the
   fp32 fallback; finalize `benchmark.csv` / `candidates.jsonl` / `profile/` (AC-7).

Dependencies: M1 depends on M0. M2 depends on M1 (stacks under the D1 winner, or
falls back to v1 if D1a washed). M3 depends on M2 (M_SUB feasibility is measured
under D2's live-set). M4 depends on the M1–M3 profile opening the PE-bind gate. M5
depends on all explored directions being scored. Each milestone is bounded by the
≤5-iteration budget; a direction may terminate early on a measured reject.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Re-baseline `gqa_full_v1` in the current profiler batch/window; confirm round-0 exposed-tail + transpose census as the entry state | AC-7 | coding | - |
| task2 | Write the D1a SBUF/PSUM live-set accounting (M_SUB=1) before coding | AC-6 | coding | task1 |
| task3 | Implement D1a tile-local two-phase context loop (transpose-all → matmul-stream, same `j` order) | AC-2 | coding | task2 |
| task4 | Score D1a full-5-seed + full `summary_metrics`; keep/revise/reject vs AC-2, AC-1, AC-1.1 | AC-2, AC-1 | coding | task3 |
| task5 | Implement + score D2 in pieces (scale-fold alone, defer-normalize alone, combined) under the D1a winner; record dynamic-range note | AC-4 | coding | task4 |
| task6 | Write per-M_SUB SBUF/PSUM accounting under the D2 live-set, then sweep D1b M_SUB `{1,2,4}` (+8 probe) within the reuse group | AC-3, AC-6 | coding | task5 |
| task7 | Evaluate the PE-bind gate on the post-D1/D2 profile; if open, implement + score D3 score-only bf16x2 (≤2 iters), else record gate closed | AC-5 | coding | task6 |
| task8 | Independent review of the D1a purity claim and the D2 defer-normalize numerics (accumulation-order / dynamic-range) | AC-1.1, AC-4 | analyze | task4 |
| task9 | Promote the fastest correct candidate, keep v1 as fp32 fallback, finalize benchmark.csv / candidates.jsonl / profile/ | AC-7 | coding | task7 |

## Claude-Codex Deliberation

Two Codex passes ran (first-pass critique of the draft, then a convergence review of
Claude's candidate plan). Model: the configured Codex review model, effort `high`.

### Agreements
- The bind is serialization (exposed tail), not a single engine; D1 two-phase
  transpose-all is the correct primary lever (sibling `bmm_softmax` precedent).
- D2's algebra `(exp/sum)@v == (exp@v)/sum` is exact-arithmetic-sound; must be gated
  empirically on rel-L2, not on "op order preserved".
- D3 (bf16x2 score split) is correctly demoted and gated; its correctness cannot be
  inferred from the sibling GEMM quadrature floor and must be measured on-device.
- Keep `gqa_full_v1` as the fp32 fallback; require full 5-seed `verify.py` before any
  promotion; stay within the per-`(kh,grp)` reuse group (no cross-group blocking); no
  fused `activation reduce_res`; no online/flash softmax.
- The A_t/SBUF accounting correction and the D1a-vs-D1b split are sound; the M_SUB
  sweep should be `{1,2,4}` with `8` only as a spill probe.

### Resolved Disagreements
- **SBUF arithmetic (Codex first pass, CORE_RISK):** the draft's "M_SUB=32 A_t tiles =
  32·16 KB = 512 KB/partition" conflated one `A_t` subtile (`[128,128]` fp32 = 512 B)
  with the full `[128,4096]` attention row tile (16 KB). Resolution: split D1 into
  D1a (tile-local, M_SUB=1, ~+16 KB, feasible) and D1b (small M_SUB batching); the
  512 KB pressure only arises from batching query tiles, so the realistic sweep is
  `{1,2,4}` (+8 probe), not `{8,16,32}`. AC-6 now requires an explicit live-set table
  before each run.
- **"Bit-identical" claim (both passes):** reframed as expected-then-verified. Pure
  reschedule is *expected* to reproduce v1's per-seed 2.874266e-6 (precedent
  `bmm_softmax_v2`), but this is a **diagnostic** (AC-1.1) — promotion gates on the
  actual `< 2e-5` (AC-1). A deviation triggers a candidate-vs-v1 output diff, not an
  automatic reject.
- **PE metric interpretation (Codex round 2, REQUIRED_CHANGE):** AC-2 rejects on
  *absolute* PE work rising (`matmul_instruction_count` / `tensor_engine_instruction_count`
  / `TRUE PE-active` ms), NOT on PE *utilization percent*, which is expected to rise
  when a tail-hiding win drops wall while PE work is flat.
- **Iteration order (Codex round 2, REQUIRED_CHANGE + UNRESOLVED):** Codex accepted
  D1a-first but required D2 be tested before/alongside the D1b M_SUB sweep (D2 frees
  the attn buffer and changes feasible M_SUB). Resolution: order is **D1a → D2 → D1b
  → D3(gated)**; the M_SUB sweep runs under the D2 live-set. The residual "what
  profiler noise threshold defines a win" is carried to DEC-1.
- **D2 dynamic range (Codex first pass, TECHNICAL_GAP):** deferred `O = exp@v` with
  `exp ∈ [0,1]` and row-sum up to ~4096 has larger pre-reciprocal magnitude than
  normalized `O`; verified fp32-safe and required to be recorded (AC-4).
- **`activation(scale=,bias=)` semantics (Codex first pass, TECHNICAL_GAP):** confirmed
  by v1's own passing usage `activation(op=nl.exp, data=score, bias=neg_max,
  scale=1.0)` = `exp(1.0·score + neg_max)` at rel-L2 2.874e-6, so the folded form
  `exp(scale·score + scale·(−max))` is semantically validated; still gated
  empirically per AC-4.

### Convergence Status
- Final Status: `converged` (Codex round 2 produced only refinements, all adopted;
  no high-impact DISAGREE survives; the two open items are user-preference decisions,
  not technical blockers, recorded below).

## Pending User Decisions

- DEC-1: **Profiler-noise "win" threshold and within-noise promotion policy.**
  - Claude Position: use a **3% p50-wall** bar (matching sibling `lora`/`tmm`
    practice); re-measure anything within noise; do not promote a within-noise
    candidate unless it is also strictly simpler than its parent.
  - Codex Position: `3% p50` is a reasonable default; re-measure within-noise
    candidates before declaring a win.
  - Tradeoff Summary: a tighter bar risks discarding a real small win under profiler
    jitter; a looser bar risks promoting noise. 3% p50 with re-measurement is the
    established sibling convention.
  - Decision Status: `PENDING` (proceeding with the 3% p50 default unless overridden).

- DEC-2: **Iteration order D1-first vs D2-first.**
  - Claude Position: **D1a → D2 → D1b → D3** — D1a is the primary lever; D2 is placed
    before the D1b M_SUB sweep so the sweep runs under D2's freed live-set.
  - Codex Position: D1a-first is acceptable; D2 must be tested before/alongside the
    D1b M_SUB sweep (not only after a D1 regression) — satisfied by the resolved order.
  - Tradeoff Summary: D2-first-entirely is cheaper to measure but evaluates Vec
    reduction against the un-overlapped schedule; the resolved order gets D1a's
    overlap first while still feeding D2's live-set into the M_SUB sweep.
  - Decision Status: `PENDING` (proceeding with D1a → D2 → D1b → D3 unless overridden).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Step", "Phase", "D1/D2/D3", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `a_t_bank`,
  `two_phase context loop`, `scaled_neg_max`, `deferred normalization`).

### Verification Commands
```bash
# fast (1 seed) during iteration:
python3 \
    ../../verify.py --op gqa_full --candidate runs/<kernel>.py --fast
# full 5-seed before promotion (drop --fast); full metrics:
python3 \
    runs/dump_metrics.py --op gqa_full --candidate runs/<kernel>.py
```

--- Original Design Draft Start ---

# gqa_full — Phase 2 Draft (profile-driven optimization)

## Starting point

Best correct kernel = `runs/gqa_full_v1.py` (DAG root, phase-1 promotion):
**1.462x / 10.6579 ms** over the 15.579 ms baseline, full 5-seed L2 PASS,
on-device rel-L2 **2.874e-6** (identical on every seed, ~7x under the 2e-5 gate).
fp32 throughout; per-head fusion (per-head bmm_softmax scores + full-row softmax
over the 4096-wide key axis, then a 32-step context matmul), scores never touch
HBM. Traffic is already at the read-once/write-once floor (HBMrd 67.2 MB == q+k+v,
HBMwr 33.6 MB == out, DMA 3.9%, no spill).

Phase-2 goal: minimize on-device latency without regressing correctness. Explore
each ranked direction for AT MOST five iterations, with before/after `verify.py`
latency + profiling evidence justifying keep / revise / reject.

## Round-0 bottleneck (evidence in `profile/gqa_full_phase2_bottleneck_evidence.txt`)

Per-inference, from `profile/gqa_full_v1_digest.txt` (metric window 2.0 inf):

| engine | active/inf | % |
|---|---|---|
| wall (p50) | 10.658 ms | — |
| **PE (TRUE)** | 5.125 ms | 48.1% |
| Vec | 4.321 ms | 40.5% |
| Scl | 4.505 ms | 42.3% |
| DMA | 0.415 ms | 3.9% (HBM floor, no spill) |

**The bind is SERIALIZATION, not any single engine.** The largest engine is PE at
5.125 ms, but wall is 10.658 ms — the **exposed tail (wall − PE) = 5.532 ms** (52%
of wall runs outside the PE stream). `sum(PE+Vec+Scl) = 13.95 ms` vs wall 10.66 ms
⇒ only 3.29 ms overlaps today. A schedule that overlaps down to the max engine
(5.125 ms) would be ~2.08x faster → up to ~3.0x total. **The primary lever is
overlap/schedule, not shrinking one engine.** This is the exact shape the sibling
`bmm_softmax` hit in phase 2: two-phase transpose-all cut its exposed tail
0.957→0.460 ms for 1.585→1.946x **without collapsing PE** — a tail-hiding win.

**Transposes are ~half the Tensor-Engine work** (the burden `bmm_softmax` lacked):
`tensor_engine_instruction_count 116385` vs `matmul_instruction_count 58112`.
Per-inference nc_matmul **sites** (512 tiles = kh8·grp2·tq32):

- real matmuls: score 8/tile = 4096; context 32/tile = 16384 → **20480**
- transposes: q 1/tile = 512; **A_t 32/tile = 16384**; k 256 → **17152** (46% of sites)

PE-cycle proxy (free-dim weighted): score 33% / context 33% / A_t-transpose 33%.
The **A_t transpose (16384 sites, the single largest PE class)** is the per-tile
32× attention transpose *inside* the context loop, run today as a serial
`transpose(PE) → copy(Vec/Scl) → matmul(PE)` chain 32× per tile — this interleave
is what keeps the PE stream shallow and holds the Vec/Scl copies in the exposed
tail.

## Ranked directions (benefit vs risk)

### D1 — Two-phase transpose-all in the context loop (+ M-block / M_SUB sweep) — PRIMARY

The promoted `bmm_softmax_v4` / `bmm_v2` lever, adapted to gqa's context matmul.
Today, per `(kh,grp,t_q)` the context loop does `for j in 32: transpose A[:,j] →
copy → o_psum += A_t·v[:,j]`, a PE→Vec→PE serial chain that also stalls the score
matmuls of the next tile. Restructure into two phases so the PE stream runs deep:

1. **Transpose-all first:** for the M-block of query tiles, transpose all 32 A
   subtiles (and q) up front into resident SBUF `A_t` tiles.
2. **Matmul-stream second:** then issue the 32 context matmuls back-to-back into
   the accumulating PSUM bank, so the transpose copies drain in parallel with a
   long uninterrupted matmul stream instead of gating it.

Sweep the M-block width **M_SUB ∈ {8, 16, 32}** query tiles processed together
before draining, exactly as `bmm_softmax_v4`. The sibling lesson
([[BL-20260711-heavy-epilogue-shifts-twophase-msub-optimum-interior]]): a **heavy
per-tile epilogue (full-row softmax) shifts the optimum to an INTERIOR M_SUB**
(bmm_softmax won at M16, not the whole-group M32). gqa's epilogue is heavier still
(softmax **plus** the 32× A_t transpose), so expect the interior optimum ≤ 16 —
test 16 first, then 8, then 32. Stay **within** the natural per-`(kh,grp)` reuse
group; never block across kv-heads (cross-batch/cross-group blocking is a proven
anti-lever, see [[kda-bmm-progress]] cross-batch-blocking-antilever).

- **Expected outcome:** tail-hiding win (wall drops toward PE-active ≈ 5 ms,
  PE-active roughly flat), NOT a PE collapse. Predict ~1.7–2.0x.
- **Risk:** medium. Pure reschedule → must stay **bit-identical** (rel-L2
  2.874e-6, matmul_instruction_count 58112, psum count, HBM floor all unchanged).
  SBUF: M_SUB=32 A_t tiles = 32·16 KB = 512 KB/partition would blow the budget —
  so M_SUB is *also* SBUF-bounded; the score/attn resident tile is 16 KB and a
  handful of A_t live tiles fit, but a full 32-wide A_t hold does not. This is a
  second reason the optimum is interior; verify no spill (DMA stays ~4%) at each
  M_SUB.
- **Iterations (≤5):** (1) two-phase at M_SUB=16 vs v1 baseline; (2) M_SUB=8;
  (3) M_SUB=32 (expect regress / SBUF pressure); pick the interior optimum;
  (4–5) reserve for a revise if the first cut regresses or spills.

### D2 — Scale-fold + defer-normalize (cheap, stackable, low-risk Vec reduction)

Two numerically-verified folds that shrink the co-limiting Vector engine
(offline check, seed 0, rel-L2 vs the v1 path = **5.36e-7** ≪ 2e-5 gate, below the
v1 device floor 2.874e-6; positive scale preserves row-argmax so the max-shift
stays valid):

1. **Fold `1/sqrt(D)` into `activation(scale=)`** and scale `neg_max` by the same
   factor (bias = `scale·(−max_unscaled)`), removing the full-width
   `score*scale` `tensor_scalar` (a 4096-wide Vec op every tile). This is the
   phase-1 draft's noted lever #2.
2. **Defer normalization past the context matmul:** run the context matmul on the
   **unnormalized** `exp` tile, then scale the small `[128,128]` output `O` by the
   `[128,1]` reciprocal. This turns the 4096-wide `attn = exp·recip` normalize
   (per tile) into a 128-wide `O·recip` — a **32× smaller** Vec op per tile —
   because `(exp/sum) @ v == (exp @ v) / sum` (per-row scalar pulls out of the
   contraction). `recip` is still computed from the full-width row-sum.

- **Expected outcome:** directly cuts Vec-active (the 40.5%/4.32 ms co-limiter);
  compounds with D1's overlap. Small standalone win; meaningful stacked.
- **Risk:** low. Numerically verified; still fp32; reference op order preserved up
  to the associativity of a positive per-row scalar. Build UNDER the D1 winner.
- **Iterations:** 1 (fold both, measure). Keep only if rel-L2 stays < 2e-5 (expect
  ~2.9e-6, unchanged) and wall improves or is neutral.

### D3 — bf16x2 3-product split on the score matmul (GATED, ranked last)

`matmul_instruction_count / real-GEMM-site = 58112/20480 = 2.84` (fp32 emulation
pass-multiple). The **score** matmul has moving `[128,512]` — the moving-512
regime where sibling GEMMs ran fp32 at ~1.8–1.95×/instr and the compensated
bf16x2 3-product split WON (matmul, rmsnorm_matmul, tmm). The **context** matmul
moving is `[128,128]` (small, weak split candidate — skip it).

BUT: PE is only 48% busy, and the split leaves the **46% transpose sites** in
fp32. Cutting PE moves the wall only AFTER D1+D2 have closed the Vec/Scl exposed
tail and PE has become the true bind. So D3 is **contingent**, gated on:
(a) post-D1/D2 profile showing PE ≥ ~Vec/Scl and PE the wall-limiter; and
(b) rel-L2 headroom — v1 is at 2.87e-6, only ~7× under 2e-5; the score-only split
adds ~4.45e-6 in quadrature (sibling floor), still passing, but confirm on-device.

- **Expected outcome:** uncertain; likely model/measured-reject given PE is not yet
  the bind. Explore ≤2 iterations only if the gate opens.
- **Risk:** high (correctness headroom + may not touch the wall). Keep v1 (and the
  D1/D2 winner) as fp32 fallback if it fails or washes.

### Not pursued (note-only)

- **Flash-style online softmax over n_k chunks:** its memory benefit is already
  captured (scores never hit HBM; DMA at 3.9%, no spill). Online softmax would
  ADD per-chunk running-max rescaling passes on the Vector engine — the WRONG
  direction while Vec co-limits. Only revisit if a future shape forces the score
  row out of SBUF (not the case here: 16 KB/partition fits).
- **Further transpose elimination:** the two transposes (q for scores, A for
  context) are fundamental to softmax-over-free-axis + the two matmul contraction
  layouts; D1 hides them rather than removing them.

## Success criteria / exit

- Correctness never regresses: every seed `[0,21,42,63,84]` passes rel-L2 < 2e-5
  (full run, not just `--fast`, before any promotion). Record on-device rel-L2.
- Promote the fastest candidate that holds correctness; keep `gqa_full_v1` as the
  fp32 fallback in the DAG. Log every perf change to `benchmark.csv`, every
  candidate to `candidates.jsonl` (parent links), profiling evidence to `profile/`.
- Target: close the 5.5 ms exposed tail via D1 (+D2), landing meaningfully above
  1.462x (aspirationally toward the ~2x that full PE/Vec overlap allows); treat
  D3 as upside only if PE becomes the bind.

## Validate / score

```bash
# fast (1 seed) during iteration:
python3 \
    ../../verify.py --op gqa_full --candidate runs/<kernel>.py --fast
# full 5-seed before promotion (drop --fast); full metrics:
python3 \
    runs/dump_metrics.py --op gqa_full --candidate runs/<kernel>.py
```

--- Original Design Draft End ---
