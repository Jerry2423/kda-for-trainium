# bmm_softmax — Phase 2: Profile-Driven Compute-Engine Optimization

## Goal Description

Beat the phase-1 promoted kernel `runs/bmm_softmax_v1.py` (**1.585x**, 4.5995 ms p50,
rel-L2 2.5683e-6) on the remote Trainium profiler **without changing the numerical
result** — a pure schedule / epilogue-fusion optimization of the fused
`out[b] = softmax_N(lhs[b] @ rhs[b])` kernel.

The measured bottleneck is fixed and drives the whole phase: v1 is **PE-bound** with a
matmul that is byte-identical to the pure sibling `bmm_v1` (TRUE PE-active 3.6424 ms ≈
3.6552 ms; `matmul_instruction_count` 8704 identical), and traffic is already at the
read-once / write-once floor with **zero spill** (HBM read 33.6 MB = input floor, write
1073.7 MB = output floor). There is **no DMA/spill surface** — the win must come from the
compute engines (PE, Vector, Scalar). The exposed tail is `wall − PE-active = 0.957 ms`
of softmax Vector/Scalar work not covered by the matmul, and the full softmax Vector/Scalar
stack (Vec 2.73 ms / 59.37%, Scalar 2.71 ms / 59.00%) is currently *hidden* under the
3.64 ms PE bind — so shrinking PE re-exposes it, and shrinking softmax then pays off.

Two complementary levers, ranked, each explored for ≤5 iterations:
- **D1 (PRIMARY):** port the sibling `bmm_v2` two-phase *transpose-all-then-matmul*
  schedule into the fused kernel to cut PE-active toward ~2.0 ms (the same matmul core
  went 3.66 → 2.01 ms in pure `bmm`).
- **D2 (COMPLEMENTARY):** fuse `exp`+row-sum into one Scalar `activation` pass and fold the
  max-negate into the max-reduce, dropping softmax Vector passes 3 → 2 — the lever that pays
  off once D1 re-exposes Vector as the bind.
- **D3 (CHEAP):** conditional `M_SUB ∈ {8,16,32}` schedule-depth sweep, only if D1 leaves
  Vector/Scalar exposed or spills.

`bmm_softmax_v1` is retained as the simple fp32 fallback regardless of outcome.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. All measurements come from `verify.py` (5-seed correctness
gate) and the remote-profiler digest (`runs/dump_metrics.py`) run from
`workspaces/bmm_softmax/`.

- AC-1: **Correctness is never regressed.** Every candidate stays fp32 throughout the
  matmul and softmax (no bf16, no tf32), preserves the max-shifted softmax
  (`exp(score − row_max)` then divide by the row sum, reduction over the N free axis =
  reference axis 2), and passes the NKIBench relative-L2 gate on all seeds `[0,21,42,63,84]`.
  - Positive Tests (expected to PASS):
    - `verify.py --op bmm_softmax --candidate runs/<file>.py` reports `l2_norm_passed=True`
      for all 5 seeds with worst on-device rel-L2 well under 2e-5 (v1 baseline 2.5683e-6).
    - No softmax reduce / activation / elementwise op reads or writes a PSUM tile; PSUM banks
      hold only matmul/transpose results, copied to SBUF before any softmax op.
  - Negative Tests (expected to FAIL / be rejected):
    - Any candidate whose worst-seed rel-L2 ≥ 2e-5, or any seed with `l2_norm_passed=False`,
      is rejected and not promoted.
    - A candidate that casts the matmul or any softmax stage to bf16/tf32 is rejected on the
      fp32 guardrail even if it happens to pass rel-L2.
- AC-2: **D1 ports the two-phase transpose-all schedule as a pure-schedule change.** Per
  batch, all 32 lhs subtiles are identity-transposed up front into a resident
  `lhs_t_pack[64, 32*128]` SBUF buffer; then the per-subtile main matmuls run against the
  resident rhs with no transpose interleaved, each subtile immediately followed by its fused
  softmax epilogue and 4096-wide store, with `score`/`exp_t`/`out_t` allocated inside the
  `affine_range` subtile loop so the compiler is free to overlap subtile `s`'s softmax with
  subtile `s+1`'s matmul.
  - Positive Tests (expected to PASS):
    - `matmul_instruction_count` stays **8704** (matmul math untouched, as in `bmm_v1→bmm_v2`).
    - `spill == 0`; HBM read ≈ 33.6 MB and HBM write ≈ 1073.7 MB (still at the floor).
    - rel-L2 stays ≈ 2.57e-6 (bit-identical matmul: transpose-before-use is exact).
  - Negative Tests (expected to FAIL / be rejected):
    - `matmul_instruction_count ≠ 8704` is a **blocking investigation item** — the candidate
      is not promotable until it is proven the matmul work/math was not changed (an
      explainable purity guard, not an automatic silent pass).
    - `spill > 0` or HBM read/write materially above the floor ⇒ SBUF-pressure regression;
      apply the AC-2.1 fallback ordering before considering the candidate.
  - AC-2.1: **SBUF pressure stays within budget or is handled by a defined fallback.** The
    single-iteration live set (pack 16 + rhs 16 + score 16 + exp_t 16 + out_t 16 + identity
    0.5 ≈ 80.5 KB/partition) and even a pipelined double-buffered epilogue
    (~128.5 KB/partition) fit under the ~208 KB trn2 usable SBUF; the simple sum is a
    hypothesis, not proof, since compiler liveness, bank conflicts, hidden temporaries, and
    transpose staging can invalidate it — `spill==0` is the real evidence.
    - Positive: profiler shows `spill==0` and HBM at floor for the chosen `M_SUB`.
    - Negative: if the compiler spills, fall back in this order — (1) smaller `M_SUB`
      (16, then 8), then (2) in-place softmax on fewer epilogue buffers — until `spill==0`.
- AC-3: **D1's outcome is classified from profiler metrics before any promotion decision.**
  Overlap of the heavier softmax epilogue under the next subtile's matmul is treated as a
  *hypothesis to measure*, not a property. Each D1 candidate is placed in exactly one bucket
  using wall, TRUE PE-active, Vec/Scalar active, `spill`, and HBM:
  (a) **win** — wall beats v1 by more than profiler noise AND PE-active dropped materially
  toward ~2.0 ms;
  (b) **PE-dropped-but-softmax-exposed** — PE-active dropped but wall barely moved because
  Vector/Scalar is now the bind ⇒ proceed to D2 (and/or D3);
  (c) **D1-failed** — PE-active ≈ unchanged ⇒ transpose-all did not deepen the stream;
  (d) **regressed** — wall up or `spill>0` ⇒ SBUF pressure, apply AC-2.1.
  - Positive Tests (expected to PASS):
    - The exit-decision doc records the bucket for each D1 candidate with the supporting
      metric deltas vs v1.
    - Promotion happens only from bucket (a), on a p50-wall margin over v1 (4.5995 ms) that
      exceeds profiler run-to-run variance, judged on repeated profiler runs (not a single
      best run). See DEC-1 for the concrete margin.
  - Negative Tests (expected to FAIL / be rejected):
    - Promoting a candidate on a single best-of-N run whose margin is within measured
      profiler noise is rejected.
    - Declaring D1 a "win" without the PE-active drop (bucket (b)/(c) dressed up as (a)) is
      rejected.
- AC-4: **D2 fuses the softmax epilogue on top of D1, attributed independently.** Apply
  `nisa.tensor_reduce(op=nl.max, negate=True)` to write `−row_max` directly (folding the
  separate `neg_max` op) and `nisa.activation(op=nl.exp, bias=neg_max, reduce_op=nl.add,
  reduce_res=row_sum)` to produce `exp_t` and the row sum in one Scalar pass, removing the
  standalone 4096-wide `tensor_reduce(add)` Vector pass. Softmax Vector passes go 3 → 2
  (max-reduce, normalize). Both NKI APIs are verified present in the installed neuronxcc
  wheel (`activation` accepts `bias` + `reduce_op=nl.add` + `reduce_res`; `tensor_reduce`
  accepts `negate=True`); reading `reduce_regs` into `reduce_res` costs one small extra
  eviction instruction (a net Vector win, not literally free).
  - Positive Tests (expected to PASS):
    - Measured Vector active time / Vector instruction count drops vs the D1-only kernel,
      without a Scalar increase that offsets it.
    - Correctness is attributed: run **D1-only** full 5-seed verify first, THEN **D1+D2**
      full 5-seed verify; rel-L2 stays well under 2e-5.
  - Negative Tests (expected to FAIL / be rejected):
    - If D1+D2 shows any rel-L2 drift toward the gate, fall back to the explicit
      `tensor_reduce(add)` row-sum (keep D1) — the fused `reduce_res` is rejected for that
      candidate.
    - A D2 candidate that does not reduce Vector work (no measurable pass drop) is not
      promoted over D1-only.
- AC-5: **D3 M-block sweep is conditional and shares D1's kernel.** Port `M_SUB=32` first
  (whole-batch, `bmm_v2`'s optimum). Only if D1 leaves Vector/Scalar exposed (bucket (b)) or
  spills (bucket (d)) do we sweep `M_SUB ∈ {8, 16, 32}` (reusing the D1 kernel), on the
  hypothesis that a shallower stream may pipeline the heavier softmax epilogue better. The
  sweep stays **within one batch** — never across the batch axis.
  - Positive Tests (expected to PASS):
    - Each swept `M_SUB` keeps `matmul_instruction_count==8704`, `spill==0`, HBM at floor,
      and rel-L2 < 2e-5; the best is chosen on measured wall.
    - The sweep runs only when justified by D1's bucket, not unconditionally.
  - Negative Tests (expected to FAIL / be rejected):
    - Any `M_SUB` grouping that crosses the batch axis (cross-batch blocking) is rejected on
      inherited measured evidence (anti-lever in `bmm` phase 3).
    - Running the full sweep when D1 already landed a clean bucket-(a) win (no exposure, no
      spill) is out of scope for this phase.
- AC-6: **The optional GpSimd-normalize lever is gated by a feasibility check, not built
  blindly.** Moving the normalize `tensor_scalar(*recip)` off Vector onto GpSimd is
  attempted **only** after D1+D2, **only** if Vector is still the bind, and **only** after an
  analyze-gate confirms the API path: the installed wheel documents
  `tensor_scalar(engine=gpsimd)` as *"only allowed for rsqrt"*, and the normalize is a
  multiply — so this lever is likely infeasible as originally worded and must be either
  reformulated (e.g. a bit-exact GpSimd path via `tensor_tensor(op=multiply)` with a
  broadcastable operand) or dropped.
  - Positive Tests (expected to PASS):
    - If pursued, a bit-exact GpSimd normalize keeps rel-L2 ≈ unchanged and measurably moves
      the 4096-wide normalize pass off the Vector critical path.
  - Negative Tests (expected to FAIL / be rejected):
    - `tensor_scalar(engine=gpsimd)` on a multiply is rejected as infeasible per the wheel's
      rsqrt-only constraint; do not ship a candidate that relies on it.
    - Building this lever before the profile shows Vector is still the bind after D1+D2 is
      out of scope.
- AC-7: **Inherited measured rejects are not built.** The following are excluded on prior
  measured evidence, not re-litigated:
  - Positive Tests (expected to PASS):
    - The exit-decision doc lists each reject with its evidence link and no candidate
      implements it.
  - Negative Tests (expected to FAIL / be rejected):
    - A **bf16x2 3-product matmul split** (rejected: the byte-identical `bmm` core has an
      fp32/bf16 pass-ratio of 2.0, so the split *raises* PE).
    - **bf16 `exp`/softmax** (rejected: ~1e-2 rel error over N=4096 blows past the 2e-5 gate).
    - **Cross-batch blocking or cross-batch double-buffer** (rejected: measured anti-lever;
      the batch boundary is a helpful reset).
    - **Removing the max-reduce or the normalize pass** (rejected: both required and already
      minimal after D2).
- AC-8: **Evidence and exit decision are recorded; v1 is kept as fallback.** Every perf
  change is logged in `benchmark.csv`; every candidate is recorded in `candidates.jsonl`
  with parent links as a DAG rooted at `bmm_softmax_v1`; profiling digests are kept under
  `profile/`; and `docs/phase2-exit-decision.md` records keep/revise/reject per direction
  with before/after evidence, followed by an update to the `[[kda-bmm-softmax-progress]]`
  memory.
  - Positive Tests (expected to PASS):
    - `docs/phase2-exit-decision.md` exists with a per-direction verdict and the promoted
      candidate's full metric diff vs v1.
    - `bmm_softmax_v1.py` remains present and referenced as the fp32 fallback.
  - Negative Tests (expected to FAIL / be rejected):
    - Promoting a candidate without a corresponding `benchmark.csv` row + `candidates.jsonl`
      entry + `profile/` digest is rejected as unrecorded.
    - Deleting or hand-tuning `bmm_softmax_v1.py` (the fallback / DAG root) is rejected.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. This
draft is a highly deterministic, evidence-gated optimization plan; the boundaries are
narrow by design.

### Upper Bound (Maximum Acceptable Scope)
Implement D1 (two-phase transpose-all schedule ported into the fused kernel) and, when the
profile shows Vector re-exposed, D2 (fused `exp`+row-sum with folded max-negate), plus a
conditional D3 `M_SUB ∈ {8,16,32}` sweep and — only behind its feasibility gate — the
optional GpSimd normalize rebalance. Each direction is explored for ≤5 iterations, every
candidate 5-seed-verified and profiled, and the best correct candidate is promoted with a
full exit-decision writeup. All changes remain pure schedule / epilogue-fusion:
`matmul_instruction_count` stays 8704, `spill==0`, HBM at the floor, fp32 throughout.

### Lower Bound (Minimum Acceptable Scope)
Implement and measure D1 alone (transpose-all port), classify its profiler bucket, and — if
it is a clean bucket-(a) win beating v1 by more than profiler noise — promote it with the
exit-decision doc and memory update, keeping `bmm_softmax_v1` as the fallback. If D1 does
not produce a promotable win, the minimum acceptable outcome is a recorded, evidence-backed
exit decision (keep v1, document why the levers did not pay off) — a measured negative
result is a valid phase outcome.

### Allowed Choices
- Can use: the sibling `bmm_v2` two-phase transpose-all schedule; `affine_range` free
  pipelining within one batch; `nisa.activation` with `bias` + `reduce_op=nl.add` +
  `reduce_res`; `nisa.tensor_reduce(negate=True)`; in-place softmax buffer reuse; `M_SUB`
  values in `{8,16,32}`; a feasibility-gated, bit-exact GpSimd normalize path.
- Cannot use: bf16/tf32 anywhere in matmul or softmax; a bf16x2 matmul split; bf16 `exp`;
  cross-batch blocking or cross-batch double-buffering; removing the max-reduce or normalize
  pass; any softmax op on a PSUM tile; `tensor_scalar(engine=gpsimd)` on a multiply (wheel
  rsqrt-only); editing the NKIBench benchmark definition or hand-tuning the baseline.

> **Note on Deterministic Designs**: The draft fixes the directions, their ranking, the
> correctness guardrails, and the measurement protocol. The upper and lower bounds differ
> only in how far down the D1→D2→D3 chain the profile justifies going; every step is gated
> by measured evidence, so there is little free choice beyond which levers the profiler
> tells us to pursue.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

D1 — restructure `bmm_softmax_v1`'s per-batch loop from `bmm_v1`'s interleaved schedule to
`bmm_v2`'s two-phase schedule, keeping the softmax epilogue per-m-tile (all 32 rows at once
would be 512 KB/partition — impossible):

```
for b in affine_range(B):                       # 16 batches, hard scheduling boundary
    load rhs[b] resident -> rhs_sb[64, 4096]
    # Phase A: transpose ALL 32 lhs subtiles up front, no matmul interleaved
    for s in affine_range(32):
        load lhs[b, 128*s:+128, :64] -> lhs_sb[128, 64]
        psum_t = nc_matmul(lhs_sb, identity, is_transpose=True, is_moving_onezero=True)
        lhs_t_pack[:, 128*s:+128] = copy(psum_t)     # resident [64, 32*128]
    # Phase B: main matmuls + fused softmax epilogue, no transpose interleaved
    for s in affine_range(32):
        score[128, 4096]  <- 8 single-pass K=64 nc_matmuls (lhs_t_pack[:,s], rhs chunks)
        row_max = tensor_reduce(max, score, negate=True)        # D2: -row_max directly
        exp_t, row_sum = activation(exp, score, bias=row_max,   # D2: fused exp + row-sum
                                    reduce_op=add, reduce_res=row_sum)
        recip = reciprocal(row_sum)
        out_t = tensor_scalar(exp_t, multiply, recip)           # normalize
        store out[b, 128*s:+128, :]                             # 4096-wide
        # score/exp_t/out_t declared INSIDE this loop => compiler may overlap
        # subtile s's Vec/Scalar epilogue with subtile s+1's PE matmul burst
```

D1 alone = the same epilogue as v1 but the new schedule (no `negate`/fused-reduce yet).
D2 = swap in the `negate=True` max-reduce and the fused `activation` reduce. D3 = wrap the
subtile loop in an `M_BLOCKS` outer loop and sweep `M_SUB`. The GpSimd lever, if feasible,
retargets only the final `tensor_scalar` normalize.

### Relevant References
- `workspaces/bmm_softmax/runs/bmm_softmax_v1.py` — phase-1 fused kernel (start-of-phase base).
- `workspaces/bmm/runs/bmm_v2.py` — the two-phase transpose-all schedule to port (PE 3.66→2.01 ms).
- `workspaces/bmm/runs/bmm_v1.py` — the interleaved schedule v1 currently mirrors.
- `workspaces/bmm_softmax/docs/phase2-bottleneck-evidence.md` — the measured PE-bound + exposed-softmax diagnosis.
- `workspaces/bmm_softmax/profile/bmm_softmax_v1_digest.txt` — v1 profiler digest (the diff baseline).
- `workspaces/bmm_softmax/runs/dump_metrics.py` — digest generator.
- NKI wheel `nki/isa/_activation.py`, `nki/isa/_tensor_ops.py` — verified `activation`
  reduce and `tensor_reduce(negate=)` signatures; `tensor_scalar` engine rsqrt-only note.
- Skill `kernel-cost-analysis` — theoretical per-engine floor to compare against.

## Dependencies and Sequence

### Milestones
1. **D1 — port the two-phase transpose-all schedule (PRIMARY).**
   - Phase A: transpose all 32 lhs subtiles per batch into `lhs_t_pack[64, 32*128]`.
   - Phase B: per-subtile 8×K=64 matmuls → v1's softmax epilogue → 4096-wide store,
     epilogue buffers inside the `affine_range` loop.
   - Gate: `--fast` correctness, then full 5-seed verify + profiler digest; classify the
     bucket (AC-3); if spilled, apply the AC-2.1 fallback ordering.
2. **D2 — fuse the softmax epilogue (COMPLEMENTARY).** Depends on D1.
   - Fold max-negate (`tensor_reduce(negate=True)`) and fuse `exp`+row-sum
     (`activation(reduce_op=add, reduce_res=)`); Vector passes 3 → 2.
   - Gate: D1-only verify first, then D1+D2 verify for attribution; measure Vector drop.
3. **D3 — conditional M-block sweep (CHEAP).** Depends on D1; runs only if D1 is bucket
   (b)/(d). Sweep `M_SUB ∈ {8,16,32}` within one batch.
4. **Optional — GpSimd normalize rebalance.** Depends on D1+D2 leaving Vector as the bind
   AND passing the AC-6 feasibility analyze-gate; otherwise dropped.
5. **Exit — promote + record.** Promote the best correct bucket-(a) candidate (or keep v1
   with a documented negative result); write `docs/phase2-exit-decision.md`; update
   `[[kda-bmm-softmax-progress]]`.

Dependencies: D2, D3, and the optional lever all build on D1's kernel; the optional lever
additionally depends on D2. Each batch remains a hard scheduling boundary (no cross-batch
work) throughout. The correctness gate (AC-1) and evidence recording (AC-8) apply to every
candidate at every milestone.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Port `bmm_v2`'s two-phase transpose-all schedule into the fused kernel (`M_SUB=32`), keeping v1's exact softmax epilogue; epilogue buffers inside the `affine_range` subtile loop | AC-2 | coding | - |
| task2 | `--fast` then full 5-seed `verify.py` on the D1 candidate; capture profiler digest; diff vs v1 | AC-1, AC-2 | coding | task1 |
| task3 | Classify the D1 candidate into a profiler bucket (win / PE-dropped-exposed / failed / regressed) from wall, PE-active, Vec/Scalar, spill, HBM | AC-3 | analyze | task2 |
| task4 | If D1 spills (bucket d): apply the fallback ordering — smaller `M_SUB` (16→8), then in-place softmax — until `spill==0` | AC-2.1 | coding | task3 |
| task5 | Apply D2 fusions on top of D1 (`tensor_reduce(negate=True)` + `activation(bias, reduce_op=add, reduce_res)`); D1-only verify then D1+D2 verify; measure Vector drop | AC-4 | coding | task3 |
| task6 | If D1 is bucket (b)/(d): sweep `M_SUB ∈ {8,16,32}` on the D1 kernel; verify + profile each; pick best wall | AC-5 | coding | task3 |
| task7 | Feasibility analyze-gate for the optional GpSimd normalize (wheel rsqrt-only constraint); reformulate or drop | AC-6 | analyze | task5 |
| task8 | If task7 passes and Vector is still the bind: implement the bit-exact GpSimd normalize; verify + profile | AC-6 | coding | task7 |
| task9 | Confirm no rejected direction was built (bf16x2 split, bf16 exp, cross-batch blocking, pass removal) | AC-7 | analyze | task5 |
| task10 | Promote the best correct bucket-(a) candidate (or keep v1 with documented negative result); log `benchmark.csv`, `candidates.jsonl` DAG, `profile/` digests | AC-3, AC-8 | coding | task3, task5, task6 |
| task11 | Write `docs/phase2-exit-decision.md` (keep/revise/reject per direction, before/after evidence) and update `[[kda-bmm-softmax-progress]]` | AC-8 | coding | task10 |

## Claude-Codex Deliberation

### Agreements
- D1 (port the `bmm_v2` two-phase transpose-all schedule) is the correct primary lever: it
  targets the measured PE-active bottleneck with a sibling schedule already proven on the
  identical matmul core.
- D2 is feasible and useful but **not free**: fusing `exp`+row-sum removes a full 4096-wide
  Vector add-reduce pass at the cost of one small eviction instruction (verified against the
  wheel's `activation` docstring).
- The optional GpSimd reciprocal-normalize lever must be feasibility-gated, not built
  blindly (wheel documents `tensor_scalar(engine=gpsimd)` as rsqrt-only).
- D1 overlap of the heavier softmax epilogue under the next subtile's matmul is a
  **hypothesis to measure**, not a property; classify each D1 candidate into explicit
  profiler buckets before deciding.
- Hard invariants for every candidate: `spill==0`, HBM at floor, fp32 softmax, max-shift
  preserved, full 5-seed rel-L2 < 2e-5 before promotion.
- Attribute correctness by running D1-only verify, then D1+D2 verify separately.
- The ~1.75 ms "hard floor" is a directional exit heuristic (derived from overlapping,
  non-additive active-time percentages), not a proven floor or a gate.

### Resolved Disagreements
- **`matmul_instruction_count==8704` — absolute invariant vs. explainable guard.** Claude
  initially proposed any delta disqualifies the candidate; Codex argued that is too absolute
  and a delta should trigger a blocking investigation with proof-of-equivalence rather than
  automatic rejection. **Resolution (both agree):** it must stay 8704; any delta is a
  *blocking investigation item* requiring proof the matmul math/work was not changed before
  the candidate is promotable (explainable purity guard). Reflected in AC-2.
- **D2 API "no additional cost" claim.** The draft said the fused row-sum is free; Codex
  flagged the reduction-order/precision risk. **Resolution:** verified against the wheel —
  the fused reduce costs one small eviction instruction (net Vector win), and D2 is
  attributed with a D1-only-then-D1+D2 verify plus an explicit-reduce fallback. Reflected in
  AC-4.
- **GpSimd normalize feasibility.** Draft assumed `tensor_scalar(engine=gpsimd)` works;
  wheel source shows it is rsqrt-only. **Resolution:** demoted behind a mandatory
  feasibility analyze-gate; reformulate via `tensor_tensor(multiply)` or drop. Reflected in
  AC-6.
- **SBUF budget certainty.** Claude cited ~80.5 KB (single) / ~128.5 KB (double-buffered)
  per partition; Codex noted the simple sum can be invalidated by compiler liveness/bank
  conflicts. **Resolution:** the budget is a hypothesis; `spill==0` is the real evidence,
  with a defined fallback ordering (smaller `M_SUB` → in-place softmax). Reflected in AC-2.1.
- **Promotion threshold.** Codex required a concrete "safe margin." **Resolution:** promote
  only from bucket (a) on a p50-wall margin over v1 exceeding profiler run-to-run variance,
  judged on repeated runs, not a single best run; the exact numeric margin is DEC-1.

### Convergence Status
- Final Status: `converged` (2 convergence rounds; both round-2 REQUIRED_CHANGES incorporated
  into AC-2 and AC-3/DEC-1; no high-impact disagreements remain).

## Pending User Decisions

- DEC-1: **Concrete promotion margin over v1.**
  - Claude Position: Promote a D1/D2/D3 candidate only if its p50 wall beats v1's 4.5995 ms
    by a margin clearly exceeding profiler run-to-run variance — recommend requiring a
    robust ≥ ~3% wall improvement (≈ ≥1.63x over the 7.290 ms baseline) confirmed on repeated
    runs; a genuine D1 win targets the much larger ~2.0–2.5x regime, so this floor is easily
    cleared by a real win and just guards against promoting noise.
  - Codex Position: Requires a numeric margin chosen *before* benchmarking, measured across
    repeated profiler runs rather than a single best run; recommends not promoting on
    single-run best-of-N within noise. Does not fix the exact number.
  - Tradeoff Summary: A tighter margin risks promoting profiler noise; a looser margin risks
    rejecting a small-but-real win. Given D1's expected large PE move, a ~3% robust-margin
    default is safe. User may override the exact percentage.
  - Decision Status: `PENDING` (default ~3% robust wall-margin applied if not overridden)
- DEC-2: **Hard performance target beyond "beat 1.585x."**
  - Claude Position: Treat all speedup numbers in the draft (~2.0–2.5x for D1, ~2.7–3.6x for
    D1+D2, ~1.75 ms floor) as **directional optimization trends, not hard requirements** —
    the draft itself says "Numbers are hypotheses; the profile gates them." The only hard
    gates are correctness (rel-L2 < 2e-5) and beating v1 by DEC-1's margin to promote.
  - Codex Position: Asks whether the bar is simply beating 1.585x or a harder target
    (>2.0x / >2.5x) is required for the phase to "succeed."
  - Tradeoff Summary: If the user sets a hard ≥2.0x target, a D1-only win below it would not
    close the phase; if directional (recommended), any promotable win closes it and a
    documented negative result is also a valid exit.
  - Decision Status: `PENDING` (default: directional trends; hard gates are correctness +
    beat-v1)
- DEC-3: **Exit condition around the ~1.75 ms Scalar-`exp` heuristic.**
  - Claude Position: The ~1.75 ms figure is a directional heuristic, not a gate; stop when
    the D1→D2→(D3) chain stops producing profiler-justified wins within the ≤5-iteration
    budget per direction, regardless of whether wall reaches ~1.75 ms.
  - Codex Position: Asks whether landing around 2.0–2.3 ms is an acceptable exit or the plan
    should keep searching toward ~1.75 ms.
  - Tradeoff Summary: Chasing the last few tenths of a millisecond past a clear win may spend
    iterations for little gain; the iteration budget and bucket classification bound the
    search naturally.
  - Decision Status: `PENDING` (default: exit on exhausted profiler-justified wins within the
    iteration budget, not on hitting ~1.75 ms)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Step", "Phase", "D1/D2/D3", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `lhs_t_pack`, `score`,
  `row_max`, `row_sum`, `M_SUB`), matching the existing `bmm_softmax_v1.py` /
  `bmm_v2.py` style.
- Candidate `.py` sources under `runs/` are tracked in git; other `runs/` artifacts and all
  of `profile/` are git-ignored. Do not edit the NKIBench benchmark definition or hand-tune
  the baseline.

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# bmm_softmax — Phase 2 Draft (profile-driven optimization)

Start-of-phase base: **`runs/bmm_softmax_v1.py`** — the phase-1 fused kernel, PROMOTED
at **1.585x** (4.5995 ms) over the 7.290 ms baseline, full 5-seed L2 PASS, on-device
rel-L2 **2.5683e-6** (~7.8x under the 2e-5 gate). This draft ranks the phase-2
levers, each explored for ≤5 iterations, and states the correctness guardrails and
measurement protocol. It does NOT change the numerical result.

---

## 1. Measured bottleneck (from phase-1 evidence, already collected)

Remote-profiler digest (`profile/bmm_softmax_v1_digest.txt`, full 5-seed, per-inference
after the 2.0x window normalization):

| metric | v1 (fused) | pure bmm_v1 (ref) | NKIBench baseline |
|---|---|---|---|
| p50 wall | **4.5995 ms** | 3.8477 ms | 7.3020 ms |
| TRUE PE-active/inf | **3.6424 ms** (79%) | 3.6552 ms (95%) | 3.7345 ms |
| Vec-active/inf | 2.73 ms (59.37%) | ~0.9 ms (20%) | 46.70% |
| Scalar-active/inf | 2.71 ms (59.00%) | ~0.6 ms (14%) | 51.02% |
| DMA-active/inf | 1.4083 ms (30.62%) | — | 3.4246 ms |
| matmul_instruction_count | **8704** | 8704 | 10240 |
| HBM read / write | 33.6 MB / 1073.7 MB | 34 / 1074 MB | 700.5 / 1740.6 MB |
| spill | **0** | 0 | ~1.33 GB round-trip |

**Two facts drive the whole phase (both already established, `docs/phase2-bottleneck-evidence.md`):**

1. **The kernel is PE-bound and the matmul is byte-identical to pure `bmm_v1`.**
   TRUE PE-active 3.6424 ms ≈ pure bmm_v1 3.6552 ms; matmul_instruction_count 8704
   identical. The fusion did NOT change the matmul workload — it runs `bmm_v1`'s
   *per-m-tile transpose→matmul* schedule. The exposed tail is
   `wall − PE-active = 4.5995 − 3.6424 = **0.957 ms**` of softmax Vec/Scalar work
   not covered by the matmul.

2. **Traffic is at the read-once/write-once floor, no spill.** HBMwr == output floor
   (1073.7 MB), HBMrd == input floor (33.6 MB), DMA-active 1.41 ms fully overlapped.
   Phase 2 has **no DMA/spill surface** — the win must come from the compute engines
   (PE, Vec, Scalar), not from moving bytes.

**The prize (why phase 2 has real headroom, not just 0.96 ms):** the softmax Vec/Scalar
stack (Vec 2.73, Scalar 2.71 ms) is *currently hidden* under the 3.64 ms PE bind — only
0.96 ms leaks out. The sibling `bmm` proved the *same* matmul core can run its PE-active
from 3.66 ms → **2.01 ms** by a pure schedule change (`bmm_v2` two-phase transpose-all,
per-matmul stall 0.420→0.231 µs, `[[kda-bmm-progress]]`). If we port that schedule, PE
drops toward ~2.0 ms, which **exposes the softmax as the new bind**. So the two levers
are complementary: one cuts PE (and re-exposes softmax), the other shrinks softmax.

---

## 2. Ranked directions

### D1 — Port the `bmm_v2` two-phase transpose-all schedule (PRIMARY; highest value, lowest risk)

**Hypothesis.** v1 runs `bmm_v1`'s schedule: per m-tile it does `transpose → copy → 8
matmuls`, so a serial `transpose→copy` dependency sits at the head of every matmul burst.
`bmm_v2` separated these: transpose **all 32 m-subtiles of the batch up front** into a
resident `[k=64, 32*128=4096]` SBUF pack, then run all main matmuls with **no transpose
interleaved**. On the identical matmul core this cut PE-active 3.66→2.01 ms (**1.89x
kernel-over-kernel**, `bmm_v1`→`bmm_v2`) because the long uninterrupted matmul stream lets
the compiler hide every PSUM→SBUF copy behind the next matmul.

**Port into the fused kernel.** Keep softmax per-m-tile (a full `[128,4096]` score row is
16 KB/part, resident; all 32 rows at once would be 512 KB/part — impossible, so softmax
must stay per-tile). Structure per batch `b`:
- **Phase A:** load + identity-transpose all 32 lhs subtiles into `lhs_t_pack[64, 32*128]`.
- **Phase B:** for each subtile `s` (in `affine_range`): 8 single-pass K=64 matmuls build
  `score[128,4096]`, then the fused softmax epilogue, then the 4096-wide store. `score`,
  `exp_t`, `out_t` stay allocated *inside* the `affine_range(32)` loop so the compiler can
  overlap subtile `s`'s softmax (Vec/Scalar) with subtile `s+1`'s matmul burst (PE) — the
  same free-pipelining `affine_range` gave `bmm_v2`.

**SBUF budget (verified):** pack 16 KB + rhs 16 KB + score 16 KB + exp_t 16 KB + out_t 16 KB
+ identity 0.5 KB ≈ **80.5 KB/partition** of the 208 KB trn2 usable — 127 KB headroom, no
spill. (Can drop toward 48 KB by doing softmax in-place on one buffer; not required.)

**Correctness risk: ~zero.** Transpose-before-use is exact, so transposing all subtiles up
front then matmul is *bit-identical* to interleaving — the matmul math is unchanged. The
softmax epilogue is byte-for-byte the phase-1 code. This is a pure schedule change.

**Expected benefit.** PE-active 3.64→~2.0 ms. Wall becomes bounded by `max(PE~2.0,
Vec, Scalar)` + residual gaps → expect a decisive move past 1.585x toward ~2.0–2.5x even
before D2. **Open question (the thing to measure):** in `bmm_v2` the epilogue between
matmul bursts was a light store; here it is the heavier softmax (Vec+Scalar). Whether
`affine_range` still hides subtile `s`'s softmax under subtile `s+1`'s matmul is exactly
what the profile must confirm.

**Iterations (≤5):** (1) port transpose-all + per-subtile softmax, `--fast` correctness
+ measure; (2) if softmax does NOT overlap (Vec/Scalar exposed between bursts), try
in-place softmax to shrink the live set; (3–5) reserved for the D3 M-block sweep below,
which shares this schedule.

### D2 — Fuse `exp`+row-sum and fold the max-negate (COMPLEMENTARY; needed once D1 exposes Vec)

**Hypothesis.** v1's softmax runs **3 full-width Vec passes** (max-reduce, add-reduce for
the row sum, normalize) + 1 Scalar pass (`exp`) + a tiny `neg_max`. Two of these collapse
for free (confirmed against `nki-api-reference`):
- `nisa.tensor_reduce(op=nl.max, negate=True)` writes `−row_max` directly, at no extra
  cost — kills the separate `neg_max = tensor_scalar(*−1)` op.
- `nisa.activation(op=nl.exp, bias=neg_max, reduce_op=nl.add, reduce_res=row_sum)`
  computes `exp_t` **and** the row sum in the *same* Scalar pass — the docs state the
  fused free-axis reduce is "no additional performance cost" beyond reading the
  accumulator out. This **removes the entire 4096-wide `tensor_reduce(add)` Vector pass.**

Net: softmax Vec load drops from 3 full-width passes → **2** (max-reduce, normalize),
Scalar unchanged. This is the lever that pays off *after* D1 re-exposes Vec as the bind.

**Correctness risk: low.** Same math; the fused sum accumulates the same `exp` values in
fp32. Reduction ordering is the caller's responsibility per docs but the accumulation is
the identical set of terms → rel-L2 expected to stay ~2.6e-6, far under 2e-5. Gate on the
full 5-seed run before promoting.

**Iterations (≤5):** (1) apply both fusions on top of D1, `--fast` + measure Vec drop;
(2) if the fused `reduce_res` shows any rel-L2 drift, fall back to the explicit add-reduce
(keep D1). Precedent: attention epilogues fuse exp/sum this way
(`[[kda-*]]`; knowledgebase `scheduling-and-pipelining.md` §5, `compute-fusion.md` §1).

### D3 — M-block / schedule-depth sweep (CHEAP; subsumed by D1 or a small tweak)

`bmm_v2` swept `M_SUB` and found whole-batch (`M_SUB=32`) optimal (stall
0.420→0.396(8)→0.340(16)→0.231(32) µs). Port `M_SUB=32` first. BUT here the epilogue
between bursts is the heavier softmax — a *smaller* M-block might pipeline better if the
softmax can't hide under a 32-deep stream. If D1 leaves Vec/Scalar exposed, sweep
`M_SUB ∈ {8, 16, 32}` (2 extra iterations, reuses D1's kernel). Low-medium value.

---

## 3. Explicit rejects (inherited measured evidence — do NOT build)

- **bf16x2 matmul split (3-product).** The matmul core is byte-identical to `bmm`, whose
  bf16 calib probe measured an fp32/bf16 pass-ratio of **2.0** (need >3). At ratio 2.0 the
  3-product bf16x2 split *raises* PE (it emulates fp32 in ~2 passes already), exactly the
  swiglu/`bmm` reject. `[[kda-bmm-progress]]`, `[[BL-20260710-bf16x2-loses-when-fp32-emulates-in-2-passes]]`.
  No build.
- **bf16 `exp`/softmax to halve the Scalar pass.** bf16 carries ~3 decimal digits (~1e-2
  rel error); softmax over N=4096 would blow past the 2e-5 gate (current margin only 7.8x).
  Reject on numerics.
- **Cross-batch blocking / cross-batch double-buffer.** Measured **ANTI-LEVER** in `bmm`
  phase 3: blocking adjacent batches regresses monotonically (stall 0.231→0.296→0.332 µs)
  — the enlarged cross-batch live set constrains the `affine_range` pipeline; the batch
  boundary is a *helpful* reset, not a bubble. `[[kda-bmm-progress]]`,
  `[[BL-20260710-cross-batch-blocking-is-an-antilever-on-affine-range]]`. The D1/D3
  schedule depth stays **within one batch** (`M_SUB ≤ 32`), never across the batch axis.
- **Removing the max-reduce or the normalize pass.** Both are required (overflow-safe
  max-shift; softmax normalization) and already minimal (1 pass each after D2). Keep.

## 4. Optional lower-priority lever (only if D1+D2 leave Vec exposed)

- **Engine rebalancing — move the normalize `tensor_scalar(*recip)` off Vector.**
  `tensor_scalar` runs on Vector/Scalar/**GpSimd**; GpSimd sits at ~4% idle. If after
  D1+D2 the profile still shows Vec as the bind (2 passes), pinning the normalize to
  GpSimd (`engine=`) takes a full 4096-wide pass off the Vec critical path. Cheap, bit-
  exact; test only if warranted by the profile. (Knowledgebase `dma-and-engines.md`:
  VectorE↔ScalarE↔GpSimd offload.)

---

## 5. Correctness guardrails (never regress)

- fp32 throughout the matmul and softmax; no bf16, no tf32.
- Max-shifted softmax preserved: `exp(score − row_max)` (overflow-safe), then divide by
  the row sum. Reduction over the **N free axis** (reference axis 2).
- No softmax reduce/activation/elementwise op on a PSUM tile — PSUM banks hold only
  matmul/transpose, copied to SBUF immediately (as in v1).
- Every candidate: `--fast` pre-check, then **full 5-seed** `verify.py` before any
  promotion; require `l2_norm_passed=True` on all seeds `[0,21,42,63,84]` and record the
  worst on-device rel-L2 (must stay « 2e-5).

## 6. Measurement protocol (per candidate)

From `workspaces/bmm_softmax/`:
```bash
python3 \
    ../../verify.py --op bmm_softmax --candidate runs/<file>.py --fast   # gate first
# then drop --fast for the promotion measurement
```
For each direction capture the digest (`runs/dump_metrics.py`) and diff vs v1 on:
TRUE PE-active/inf, Vec/Scalar-active, matmul_instruction_count (should stay 8704 for
D1/D3; unchanged by D2), HBM read/write (must stay at the floor), psum copies, and wall.
Keep evidence under `profile/`; log every perf change in `benchmark.csv`; record each
candidate in `candidates.jsonl` with parent links (DAG root = `bmm_softmax_v1`).

## 7. Expected outcome & exit

- **Primary target:** decisively beat 1.585x. D1 alone (PE 3.64→~2.0 ms) should reach
  ~2.0–2.5x; D1+D2 (Vec 3→2 passes, softmax exposed but smaller) should approach the
  compute floor set by `max(PE~2.0 ms, Scalar exp ~1.75 ms, Vec~1.8 ms)` ≈ ~2.0–2.5 ms
  → roughly **2.7–3.6x** over baseline. Numbers are hypotheses; the profile gates them.
- **Hard floor on trn2:** the Scalar `exp` pass (~1.75 ms theoretical for 512 × 4096-wide
  passes) is irreducible here — the Vector-fused `nisa.exponential` that would move it is
  NeuronCore-v4-only. So do not expect to beat ~1.75 ms wall regardless of schedule.
- **Promote** the best correct candidate; **keep `bmm_softmax_v1`** as the simple fp32
  fallback. Write `docs/phase2-exit-decision.md` with keep/revise/reject per direction and
  the before/after evidence, then update `[[kda-bmm-softmax-progress]]`.

--- Original Design Draft End ---
