# silu (M4096 N7168, fp32) — Phase 2: Roofline Confirmation + DMA-Bubble Harvest

## Goal Description

Take the promoted phase-1 kernel `runs/silu_v1.py` (0.3009 ms, 3.398x over the
1.022441 ms baseline) and run one bounded, profile-driven optimization pass. The
central, verified finding that shapes the whole effort: **v1 already sits at the
achieved single-core streaming roofline** for this elementwise fp32 access pattern
and compiler schedule. So this phase has two deliverables, in priority order:

1. **Primary — rigorously confirm the roofline with evidence**, not assertion:
   traffic is exactly the read-once/write-once minimum (`2 · 4096 · 7168 · 4 B =
   234.88 MB`, which equals the measured `HBMrd 117 MB + HBMwr 117 MB`), the
   effective aggregate HBM bandwidth is `234.88 MB / 0.3009 ms ≈ 781 GB/s`, and the
   profiler shows `DMA = 97 %` active (only ~3 %, ~9 µs, of DMA-idle bubble).

2. **Secondary — harvest the ~3 % DMA-issue bubble only if it clears a same-session
   noise band.** Enumerate the levers, rank them by benefit/risk, and explore each
   for **at most five iterations**, collecting before/after latency (`verify.py`) and
   per-engine + HBM-traffic evidence to justify keep / revise / reject. **Never
   regress correctness** (relative-L2 `< 2e-5` on all five seeds `[0, 21, 42, 63,
   84]`).

A legitimate terminal outcome of this phase is **"v1 unchanged plus a documented
roofline confirmation."** There is no multiplicative headroom to chase: the fp32
in/out contract fixes the 234.88 MB traffic floor, so the only physically available
slack is the DMA-issue/scheduling bubble, and that is single-digit-% at most.

### Verified harness facts (ground truth for this plan)

These were confirmed against the harness during planning and are not re-litigated:

- `verify.py` prints HBM counters as **decimal MB** (`bytes / 1e6`); `4096 · 7168 ·
  4 = 117,440,512 B = 117.44 MB` per direction, `234.88 MB` total — matching the
  measured `HBMrd 117 MB + HBMwr 117 MB` exactly.
- The scoring harness compiles with `--disable-dge --logical-nc-config=1`
  (hardcoded in `adapter/nkibench_case.py`, `NKIBENCH_COMPILER_FLAGS`), so **hardware
  DGE is already off globally** — this confirms D4's premise before any iteration.
- The output `v2` is a **fresh** `nl.shared_hbm` ndarray, distinct from input `v1`,
  so there is **no HBM input/output aliasing hazard**; D3's in-place aliasing concern
  is purely SBUF-internal (`y_tile` aliasing `x_tile` within one unary activation).
- `verify.py --fast` = seed 42, `warmup=3`, `iters=20` (a **screen**). The full run
  = all five seeds `[0, 21, 42, 63, 84]`, `warmup=10`, `iters=100` (the **decision**).
  Latency is p50 on-device.
- Layout `(128, 32, 7168)` is row-major: for a fixed partition `p`, `v1[:, i0:i0+k,
  :]` is a **contiguous** `[128, k·7168]` per-partition span (middle stride 7168, free
  stride 1).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. The `verify.py` invocation used throughout is:

```
python3 \
    ../../verify.py --op silu --candidate runs/<file>.py [--fast]
```

- AC-1: Correctness never regresses. Every **promoted** candidate passes NKIBench's
  relative-L2 gate (`||v_k − v_r||_2 < 2e-5 · ||v_r||_2`, fp32) on all five seeds
  `[0, 21, 42, 63, 84]` on the **full** run (not `--fast`).
  - Positive Tests (expected to PASS):
    - The full run (no `--fast`) exits `0`, reports `correct: 1/1` and a `geomean
      speedup` line, and every per-seed `l2_norm_passed` is true across all five
      seeds. The machine-checkable signal is `verify.py` **exit code 0** with all
      per-seed `l2_norm_passed` true (strings read from actual output, not assumed).
  - Negative Tests (expected to FAIL / block promotion):
    - Any candidate promoted on a `--fast` (single-seed) run alone, without a passing
      full five-seed run, is rejected.
    - A candidate that introduces a `bf16`/`tf32`/`fp16` operand, intermediate, or
      store anywhere fails the `2e-5` gate and must not be promoted.
    - A D3 in-place variant whose SBUF `src == dst` aliasing corrupts output surfaces
      as a per-seed `l2_norm_passed` false and is rejected.

- AC-2: The roofline finding is documented with evidence so "no multiplicative
  headroom" is defensible, not merely asserted.
  - Positive Tests (expected to PASS):
    - A written record (in `profile/` and/or the phase-2 handoff notes) states: total
      traffic `= 234.88 MB` and equals the measured `HBMrd + HBMwr`; effective
      aggregate BW `≈ 781 GB/s`; `DMA ≈ 97 %` active. It explicitly frames this as the
      **achieved single-core streaming roofline** for this access pattern and compiler
      schedule.
    - The record notes that the cost model's `368 GB/s` is a **per-unidirectional-
      stream** figure, so its serialized `234.88 MB / 368 GB/s ≈ 0.638 ms` prediction
      is ~2x conservative because real HBM overlaps read and write (measured
      `0.3009 ms` ≈ the conservative overlapped one-way estimate `117.44 MB / 368 GB/s
      ≈ 0.319 ms`, and is in fact already below it).
  - Negative Tests (expected to FAIL / be rejected in review):
    - Claiming the measured `781 GB/s` is the global Trainium HBM-fabric mathematical
      maximum (rather than the achieved streaming roofline for this kernel) is an
      overstatement and is rejected.
    - Asserting "no headroom" without the traffic-floor and per-engine numbers is
      incomplete.

- AC-3: D1 — wider DMA bursts via multi-slice batching — is swept over `k ∈ {1, 2,
  3, 4}` and the result is recorded. Each `k` variant processes `k` contiguous
  middle-axis slices as **one** `[128, k·7168]` load → **one** fused
  `nisa.activation(op=nl.silu)` over the `k·7168` free dim → **one** `[128, k·7168]`
  store, keeping the exact `(128, 32, 7168)` output mapping (`32` is divisible by `1,
  2, 4`; `k = 3` needs an exact-divisor tail scheme since `32 / 3` is not integral).
  - Positive Tests (expected to PASS):
    - For each swept `k`, a `--fast` screen is run first; any variant whose `--fast`
      latency beats v1 gets a full five-seed run. `k`, latency, speedup, and HBM
      counters are recorded in `benchmark.csv` / `candidates.jsonl` / `profile/`.
    - A `k > 1` variant is **promoted only if** it passes AC-8's same-session noise
      band **and** its HBM counters stay at `117.44 + 117.44 MB` (no extra traffic)
      **and** all five seeds pass AC-1.
  - AC-3.1: The `k` upper bound is made **empirical**, not merely asserted. The
    committed sweep is `k ∈ {1, 2, 3, 4}`; the documented rationale is (a) the single
    fused activation spans `k·7168` free elements and `k = 4 → 28672` is the largest
    `k·7168` under the commonly-cited **soft** `~32767` Scalar activation free-dim
    limit (this constant was **not** found documented in the installed compiler
    artifact during phase 1, so it is treated as soft), `k = 5 → 35840` would exceed
    it and force an inner free-dim tile; and (b) double-buffered pipeline residency:
    even with D3 in-place (`k` slices live) `2 · k · 28 KB ≤ 208 KB ⇒ k ≤ 3` for full
    overlap, so `k = 4` already trades some overlap for fewer DMAs.
    - Positive: **Iff** `k = 4` is still the monotone-best at the ceiling (latency
      still decreasing at `k = 4`), a cheap `--fast` screen of `k ∈ {5, 6, 7}`
      (in-place) is run to make the bound empirical. Any promising `k > 4` result MUST
      re-enter the **full** AC-8 same-session promotion gate before it can replace v1.
    - Negative: A compile failure or an internally-tiled activation at `k ≥ 5`
      empirically confirms the soft bound and closes the sweep; asserting `k = 4` is
      the maximum without either this probe or a documented hard constraint is rejected.
  - Negative Tests (expected to FAIL):
    - A D1 variant that emits `k` independent per-slice DMAs (same per-slice traffic,
      no descriptor coalescing) rather than one wider burst shows no issue-overhead
      reduction and is not a win; this is checked against the profiler evidence.
    - A `k > 1` variant that merely ties v1 within the noise band is not promoted.

- AC-4: D2 — explicit ping-pong double-buffering — is tested with **exactly one**
  confirmation run (`k = 1`, two-buffer manual prefetch) and the result recorded.
  - Positive Tests (expected to PASS):
    - One `silu_v2_pingpong` candidate is scored against v1 and recorded. It is kept
      only if it clears AC-8's noise band; otherwise the digest records it as
      **redundant with the compiler's `affine_range` software-pipeline** (v1 already
      sits at the overlapped estimate, not the serialized `0.638 ms`).
  - Negative Tests (expected to FAIL):
    - Spending more than one iteration re-tuning ping-pong once it shows ~0 gain, or
      promoting it on a within-noise tie, is rejected.

- AC-5: D3 — in-place compute (`y_tile` aliases `x_tile`) — is **not** a standalone
  candidate; it is folded into the `k = 4` D1 variant as the SBUF enabler (`k = 1..3`
  use separate load+store buffers, `2k ≤ 7 ⇒ k ≤ 3`; `k = 4` needs in-place to fit
  208 KB/partition), and is validated empirically before any in-place promotion.
  - Positive Tests (expected to PASS):
    - The in-place `k = 4` variant passes all five seeds (AC-1); compiles with **no**
      fallback/legality warning; profiler HBM counters remain **exactly** at
      `117.44 + 117.44 MB`; and there are no added SBUF spills vs the separate-buffer
      variant.
  - Negative Tests (expected to FAIL):
    - Assuming `nisa.activation` `src == dst` is safe without the empirical
      correctness + profile check is rejected; if the compiler does not document
      `src == dst` support, D3 is treated as **experimental and correctness/profile-
      gated**. An aliasing hazard manifests as an L2 failure that `verify.py` catches.

- AC-6: D4 and D5 are rejected with recorded reasoning, without spending iterations.
  - Positive Tests (expected to PASS):
    - D4 (`dge_mode`) is rejected on the **confirmed** `--disable-dge` premise
      (hardware DGE already off), recorded in the phase-2 notes; it is attempted at
      most once **only if** D1/D2 leave an unexplained bubble.
    - D5 (bf16 / any traffic reduction) is rejected outright: the fp32 gate mandates
      the `234.88 MB` read-once/write-once floor; there is no dtype/fusion/recompute
      trick left. (Recorded so phase 3 does not re-litigate it.)
  - Negative Tests (expected to FAIL):
    - Spending a full iteration on D4 before D1/D2 reveal an unexplained bubble, or
      introducing bf16 anywhere, is rejected.

- AC-7: Evidence is complete for every candidate and every promotion decision.
  - Positive Tests (expected to PASS):
    - Each candidate is a node in `candidates.jsonl` (DAG parented off `silu_v1`),
      each performance-relevant change is a row in `benchmark.csv`
      (`timestamp,op,candidate,parent,passed,latency_ms,speedup,notes`), and each
      major direction has a `profile/` digest **including HBM traffic counters**, not
      just latency.
    - The candidate `.py` sources under `runs/` are committed (tracked), per the
      repo's evidence rules.
  - Negative Tests (expected to FAIL):
    - A promotion with a missing `benchmark.csv` row, missing `candidates.jsonl` node,
      or a `profile/` digest lacking HBM counters is incomplete and rejected.

- AC-8: Noise discipline gates every promotion with a numeric, pre-declared rule.
  - Positive Tests (expected to PASS):
    - To evaluate a `k > 1` (or any) candidate `B` against v1 (`A`), one same-session
      interleaved **full** sequence is run: `A0, B0, A1, B1, A2` (all five seeds,
      `warmup=10`, `iters=100`, p50). Define baseline jitter `J = max(|A1 − A0|,
      |A2 − A1|)`; let `Abar = median(A0, A1, A2)` and `Bbar = median(B0, B1)` (the
      two-point median is the arithmetic mean of `B0` and `B1`). **Promote `B` only
      if** `Bbar < Abar − J` **and** both `B0` and `B1` individually beat `max(A0, A1,
      A2)` **and** `B`'s HBM counters stay at `117.44 + 117.44 MB` **and** all five
      seeds pass AC-1. Otherwise keep v1.
    - This formula is fixed in the plan **before** the sweep; `--fast` is screen-only
      and never a promotion decision.
  - Negative Tests (expected to FAIL):
    - Promoting on a `--fast` result, on a single full run, on a within-`J` tie, or
      on a candidate whose HBM traffic rose above the floor, is rejected.
    - A candidate that ties v1 within the noise band keeps v1 (not promoted).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.
This phase is highly deterministic: the operator, layout, dtype, and correctness
gate are fixed, and the only genuine branch point is whether a DMA-bubble-harvest
candidate clears a numeric noise band. So the bounds are narrow and differ mainly in
*how far the optional D1/D2/D3 exploration is carried*.

### Upper Bound (Maximum Acceptable Scope)

The full bounded exploration is carried out: the D1 sweep over `k ∈ {1, 2, 3, 4}`
(with the D3 in-place enabler used to reach `k = 4`, and the empirical `k ∈ {5, 6,
7}` `--fast` probe iff `k = 4` is still monotone-best), the single D2 ping-pong
confirmation run, D4/D5 rejected with recorded reasoning, and — if any candidate
clears the AC-8 noise band — the best is promoted with `k`, latency, speedup, and
HBM counters recorded. The roofline confirmation (AC-2) is documented with full
evidence. This completes the phase without over-engineering — no dtype tricks, no
layout changes, no sub-noise chasing.

### Lower Bound (Minimum Acceptable Scope)

`silu_v1` remains the phase-2 result, **unchanged**, accompanied by a rigorous,
evidence-backed roofline confirmation (AC-2) and the recorded reasoning that D4/D5
offer nothing (AC-6). This is a legitimate terminal outcome: if the D1 sweep and the
D2 run all tie v1 within the noise band, the honest, correct conclusion is that v1
is at the achieved streaming roofline. Correctness is never regressed.

### Allowed Choices

- Can use:
  - D1 multi-slice batching: one contiguous `[128, k·7168]` load → one fused
    `nisa.activation(op=nl.silu)` → one store, `k ∈ {1, 2, 3, 4}` (and the empirical
    `k ∈ {5, 6, 7}` probe under AC-3.1's condition).
  - D2 explicit ping-pong double-buffering (one confirmation run only).
  - D3 in-place SBUF compute (`y_tile` aliases `x_tile`) as the enabler for `k = 4`,
    correctness/profile-gated per AC-5.
  - Diagnostics: an identity-copy / load-store-only run with the same tiling to
    isolate the pure DMA-issue floor from activation overlap **if** an unexplained
    bubble remains after D1/D2; a compiler-visible static unroll of 2–3 middle slices
    as an alternative to reshaping to `[128, k·7168]`.
  - `nl.affine_range` over the (grouped) middle axis; exact-divisor tail handling for
    `k = 3`.
- Cannot use:
  - `bf16` / `tf32` / `fp16` anywhere, or any dtype change (fp32 in/out is mandated;
    D5 rejected).
  - Any layout change or transpose (elementwise ⇒ layout is correctness-neutral; keep
    `(128, 32, 7168)` so harness reconciliation is untouched; batching slabs
    contiguous middle-axis slices, it does not re-layout).
  - Masking / partial-tile logic where dimensions are exact (`k ∈ {1, 2, 4}` divide
    `32`; only `k = 3` needs an exact-divisor tail, no mask).
  - `dge_mode` tuning as a primary lever (D4 neutralized by the hardcoded
    `--disable-dge`).
  - Promotion on a `--fast` result, a single full run, or a within-noise tie.

> **Note on Deterministic Designs**: The operator, layout, dtype, and correctness
> gate are fixed by the draft and the harness, so the bounds are narrow. The only
> real branch point is the numeric AC-8 noise-band test; a "no promotion, v1 stays"
> result is an explicitly acceptable lower bound, not a failure.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are
> conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

D1 `k`-batched skeleton (one wider burst per grouped iteration):

```python
@nki.jit
def kernel(v1):                         # v1: (128, 32, 7168) fp32
    P, MID, F, K = 128, 32, 7168, k     # k in {1,2,3,4}
    v2 = nl.ndarray((P, MID, F), dtype=np.float32, buffer=nl.shared_hbm)
    for g in nl.affine_range(MID // K):                 # grouped middle-axis
        # one contiguous [128, K*7168] slab: middle stride 7168, free stride 1
        x_tile = nl.load(v1[:, g*K:(g+1)*K, :])         # [128, K, 7168] -> [128, K*7168]
        y_tile = nisa.activation(op=nl.silu, data=x_tile)  # one Scalar op over K*7168
        nl.store(v2[:, g*K:(g+1)*K, :], value=y_tile)   # one wider store
    return v2
```

- `k = 3` does not divide `32`; use an exact-divisor tail (e.g. group `[3,3,...,2]`
  or handle the final short group explicitly) — no mask, exact shapes.
- D3 in-place for `k = 4`: let the activation write into the load buffer
  (`y_tile` aliases `x_tile`) so only `k` slices are live (`k · 28 KB ≤ 208 KB`),
  which is what lets `k = 4` fit; validate per AC-5.
- D2 ping-pong: pre-allocate two `[128, 7168]` buffer sets and manually prefetch
  group `i + 1` while computing group `i`; expected ~0 because `affine_range` already
  licenses the compiler to software-pipeline.

Noise-band evaluation (AC-8), pseudocode:

```
run A0, B0, A1, B1, A2   # same session, full 5-seed, warmup=10 iters=100, p50
J    = max(|A1 - A0|, |A2 - A1|)
Abar = median(A0, A1, A2)
Bbar = (B0 + B1) / 2      # two-point median = mean
promote = (Bbar < Abar - J) and (B0 < max(A0,A1,A2)) and (B1 < max(A0,A1,A2))
          and HBM_counters(B) == (117.44 MB read + 117.44 MB write)
          and all_five_seeds_pass_L2(B)
```

### Relevant References

- `workspaces/silu/docs/draft-phase2.md` — the source draft (preserved below).
- `workspaces/silu/runs/silu_v1.py` — the phase-1 promoted kernel (the starting
  point; `affine_range(32)`, full-width `[128,7168]` load → fused `nl.silu` → store).
- `workspaces/silu/profile/silu_v1.txt` — the phase-1 per-engine digest (DMA=97%,
  Scl=34%, HBMrd=HBMwr=117MB) that grounds the roofline analysis.
- `workspaces/silu/docs/plan-phase1.md` — house style for AC structure and the
  measured-vs-floor / LUT-accuracy findings carried into this phase.
- `adapter/nkibench_case.py` — `NKIBENCH_COMPILER_FLAGS` (`--disable-dge
  --logical-nc-config=1`), the confirmation of D4's premise.
- `../../verify.py` — the correctness/latency harness (rel-L2 gate; prints
  MFU/PE/Vec/Scl/DMA/HBMrd/HBMwr as decimal MB; `--fast` = seed 42).
- `../AccelOpt/NKIBench/reference/silu_M4096_N7168_numpy_0.py` — numpy reference
  (`x / (1 + np.exp(-x))`) and the `(128,32,7168)` reshape contract.
- Skill `kernel-cost-analysis` — theoretical per-engine floor (Scalar
  compute ≈ 0.191 ms, HBM one-way ≈ 0.319 ms) to compare against the profiler.
- Skill `kernel-optimization-kb` — precedents for DMA coalescing
  (`d3cbeffd`) and double-buffering (`bc877398`, `3c7e053b`).

## Dependencies and Sequence

### Milestones

1. Roofline confirmation (primary, independent of any candidate):
   - Phase A: From `profile/silu_v1.txt`, assemble the evidence for AC-2 — traffic
     floor `= 234.88 MB = HBMrd + HBMwr`, effective BW `≈ 781 GB/s`, `DMA ≈ 97 %`,
     and the per-unidirectional-stream framing of the model's `368 GB/s`.
   - Phase B: Write it up as the phase-2 handoff, framed as the **achieved streaming
     roofline** (not the global HBM-fabric maximum).

2. D1 burst-batching sweep (secondary; the one lever with a physical basis):
   - Step 1: Implement and `--fast`-screen `silu_v2_k2` (`k = 2`, separate buffers).
   - Step 2: Implement and `--fast`-screen `silu_v2_k3` (`k = 3`, exact-divisor tail).
   - Step 3: If the sweep is still improving, implement `silu_v2_k4` with the D3
     in-place enabler; `--fast`-screen, then validate in-place per AC-5.
   - Step 4: For any variant that beats v1 in the screen, run the AC-8 same-session
     interleaved full sequence and apply the numeric promotion rule.
   - Step 5 (conditional): iff `k = 4` is monotone-best, `--fast`-probe `k ∈ {5, 6,
     7}` in-place to make the upper bound empirical; any winner re-enters the full
     AC-8 gate.

3. D2 ping-pong confirmation (secondary; exactly one run):
   - Step 1: Implement `silu_v2_pingpong` (`k = 1`, manual two-buffer prefetch).
   - Step 2: Run AC-8's evaluation vs v1; keep only if it clears the band, else record
     it as redundant with the compiler's `affine_range` pipeline.

4. D4/D5 rejection + evidence close-out:
   - Step 1: Record D4 rejected on the confirmed `--disable-dge` premise (attempt once
     only if an unexplained bubble remains after Milestones 2–3).
   - Step 2: Record D5 (bf16) rejected outright on the fp32 traffic-floor argument.
   - Step 3: Ensure every candidate has its `benchmark.csv` row, `candidates.jsonl`
     DAG node (parented off `silu_v1`), and `profile/` digest with HBM counters; state
     the final result (promoted variant or "v1 unchanged + roofline confirmed").

Dependencies: Milestone 1 depends only on the phase-1 digest and can proceed
immediately. Milestone 2 Step 3 (`k = 4`) depends on the D3 in-place validation;
Milestone 2 Step 5 depends on `k = 4` being monotone-best. Every promotion depends on
Milestone 1's evidence context and on the AC-8 same-session sequence. Milestone 4
depends on Milestones 2–3 to know whether an unexplained bubble remains for D4.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Assemble and write the roofline-confirmation evidence (234.88 MB traffic floor = measured HBMrd+HBMwr; ≈781 GB/s effective BW; DMA≈97%; per-unidirectional-stream framing of the model's 368 GB/s) as the phase-2 handoff, framed as achieved streaming roofline | AC-2 | coding | - |
| task2 | Implement `runs/silu_v2_k2.py` (k=2, one `[128,2·7168]` load → fused `nl.silu` → store, separate buffers); `--fast`-screen | AC-3 | coding | - |
| task3 | Implement `runs/silu_v2_k3.py` (k=3, exact-divisor tail, no mask); `--fast`-screen | AC-3 | coding | task2 |
| task4 | Implement `runs/silu_v2_k4.py` with D3 in-place (`y_tile` aliases `x_tile`); `--fast`-screen and validate in-place per AC-5 (5-seed L2, no fallback, HBM at floor, no extra spills) | AC-3, AC-5 | coding | task3 |
| task5 | For any k-variant beating v1 in the screen, run the AC-8 same-session interleaved full sequence `A0,B0,A1,B1,A2` and apply the numeric promotion rule | AC-3, AC-8 | coding | task2, task3, task4 |
| task6 | Conditional empirical bound probe: iff k=4 is monotone-best, `--fast`-screen k∈{5,6,7} in-place; a compile failure / internal tiling confirms the soft 32767 free-dim bound; any winner re-enters the full AC-8 gate | AC-3.1 | coding | task4, task5 |
| task7 | Implement `runs/silu_v2_pingpong.py` (k=1 manual two-buffer prefetch); run AC-8 evaluation vs v1; keep only if it clears the band, else record redundant-with-compiler-pipeline | AC-4, AC-8 | coding | - |
| task8 | Record D4 rejected on the confirmed `--disable-dge` premise (attempt once only if an unexplained bubble remains) and D5 (bf16) rejected on the fp32 traffic-floor argument | AC-6 | coding | task5, task7 |
| task9 | Independent cost/bottleneck check: confirm the Scalar compute floor (≈0.191 ms) stays hidden under DMA even at k=4, and sanity-check the 781 GB/s vs the model's 368 GB/s per-stream figure | AC-2 | analyze | task1 |
| task10 | Close out evidence: every candidate has a `benchmark.csv` row, a `candidates.jsonl` DAG node (parent = `silu_v1`), and a `profile/` digest **with HBM counters**; state the final result (promoted variant or "v1 unchanged + roofline confirmed") | AC-7 | coding | task5, task6, task7, task8 |

## Claude-Codex Deliberation

### Agreements
- The HBM traffic floor is correctly framed: exactly one fp32 read + one fp32 write,
  `234.88 MB`, with no masking/layout/dtype escape hatch; `781 GB/s` effective is the
  achieved streaming roofline, not the global HBM-fabric maximum.
- Rejecting D4 is correct now that `--disable-dge` is confirmed hardcoded in
  `adapter/nkibench_case.py`; rejecting D5 (bf16) is correct under the fp32 gate.
- D3's in-place concern is SBUF-local only — there is no HBM aliasing hazard because
  `v2` is a fresh `shared_hbm` distinct from `v1`.
- D2 as a single confirmation run is reasonable; if `affine_range` pipelining already
  does the work, repeated ping-pong variants are noise-chasing.
- Promotion must require full five-seed runs, not `--fast`; `--fast` is screen-only.
- The D1 contiguity claim is correct: `(128,32,7168)` row-major makes `v1[:, i0:i0+k,
  :]` a contiguous `[128, k·7168]` per-partition span.

### Resolved Disagreements
- **Noise band was underspecified (Codex REQUIRED_CHANGE).** Codex flagged that
  "clears noise" and a bare `A/B/B/A` were too subjective. **Resolved** — AC-8 now
  fixes a numeric rule *before* the sweep: interleaved `A0,B0,A1,B1,A2` full runs,
  jitter `J = max(|A1−A0|, |A2−A1|)`, promote iff `Bbar < Abar − J` and both `B` runs
  beat `max(A0,A1,A2)` and HBM stays at the floor and all seeds pass. (Two-point
  median `Bbar` = mean of `B0,B1`, stated explicitly per Codex's wording tightening.)
- **D1 `k` upper bound was asserted, not proven (Codex REQUIRED_CHANGE).**
  **Resolved** — AC-3.1 documents the bound rationale (soft `~32767` free-dim limit;
  double-buffer residency) *and* makes it empirical: iff `k = 4` is monotone-best, a
  `--fast` probe of `k ∈ {5, 6, 7}` runs, and any `k > 4` winner must re-enter the
  full AC-8 gate before replacing v1 (Codex's second wording tightening).
- **D3 in-place safety needed stronger validation (Codex REQUIRED_CHANGE).**
  **Resolved** — AC-5 requires 5-seed L2, compile with no fallback, HBM counters at
  the exact floor, and no extra SBUF spills; if `src == dst` is not documented for the
  activation, D3 is treated as experimental and correctness/profile-gated.
- **"0.319 ms floor" wording (Codex OPTIONAL, adopted).** Reworded to "cost-model
  conservative one-way estimate" since measured `0.3009 ms` is already below it.
- **DMA=97% overstatement risk (Codex CORE_RISK, adopted).** AC-2 explicitly frames
  the result as the achieved single-core streaming roofline, not the global Trainium
  HBM-fabric mathematical maximum.

### Convergence Status
- Final Status: `converged` (round 2 of the Claude↔Codex loop; Codex returned
  `AGREE`, `CONVERGENCE: no REQUIRED_CHANGES remain`, with only two wording
  tightenings — both folded into AC-8 and AC-3.1 above).

## Pending User Decisions

None. All Codex `QUESTIONS_FOR_USER` and `REQUIRED_CHANGES` were resolved during
convergence and are recorded here for traceability (no item remains `PENDING`):

- **Distinct vs aliased HBM in/out buffers** (Codex QUESTION): resolved by
  inspection — `v2` is a fresh `nl.shared_hbm` ndarray, so in/out are not aliased and
  D3 is a purely SBUF-internal concern.
- **Profiler counter units** (Codex QUESTION): resolved by inspection — `verify.py`
  prints decimal MB (`bytes / 1e6`), so the `234.88 MB` arithmetic is exact.
- **Whether NKI exposes DMA descriptor counts / lowered IR here** (Codex QUESTION):
  the remote profiler returns per-engine %, MFU, and HBM bytes, not descriptor counts;
  the plan records DMA command count / schedule evidence only on a best-effort basis
  and otherwise relies on active-% + HBM-MB + latency.
- **Quantitative-metric classification** (draft-answered): the `2e-5` rel-L2 gate on
  all five seeds is a **hard requirement**; the latency figures (0.3009 → ~0.29 ms
  single-digit-% target) are an **optimization trend/direction**, promoted only if
  the AC-8 noise band is cleared. No separate user confirmation needed — the draft
  pre-classifies these and explicitly allows "v1 unchanged" as a valid outcome.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `x_tile`,
  `y_tile`, `g`/group index, `K`), matching `silu_v1.py`'s idiom.
- fp32 end-to-end with explicit `np.float32`; no implicit narrowing anywhere.
- Candidate filenames follow the house convention (`runs/silu_v2_k2.py`,
  `runs/silu_v2_k3.py`, `runs/silu_v2_k4.py`, `runs/silu_v2_pingpong.py`).

--- Original Design Draft Start ---

# silu (M4096 N7168, fp32) — Phase 2 implementation draft (profile-driven optimization)

## Goal

Starting from the promoted phase-1 kernel `runs/silu_v1.py` (0.3009 ms, 3.398x over
the 1.022441 ms baseline), use the profiler to find the *real* remaining bottleneck,
enumerate optimization directions, rank them by expected benefit vs risk, and explore
each for **at most five iterations** — collecting before/after latency (`verify.py`)
and per-engine evidence to justify keep / revise / reject. Never regress correctness
(relative-L2 < 2e-5 on all five seeds).

## Where phase 1 left us (the measured starting point)

`runs/silu_v1.py`: one `nl.affine_range(32)` over the middle axis; each iteration
loads a full-width `[128, 7168]` fp32 slice HBM→SBUF, applies one fused
`nisa.activation(op=nl.silu)` on the Scalar engine, stores `[128, 7168]` SBUF→HBM.
Two live SBUF tiles (x_tile, y_tile), no inner free-dim loop, mask-free.

Profiler digest (`profile/silu_v1.txt`, full 5-seed):

| latency | speedup | MFU | PE | Vec | **Scl** | **DMA** | HBMrd | HBMwr |
|---------|---------|-----|----|----|---------|---------|-------|-------|
| 0.3009 ms | 3.398x | 0% | 1% | 1% | 34% | **97%** | 117 MB | 117 MB |

Correct 1/1 on all five seeds; the fused `nl.silu` LUT is L2-accurate under the
2e-5 gate, so the phase-1 correctness ladder never had to descend to
sigmoid+multiply or the exp-exact chain. **No accuracy concern carries into phase 2.**

## Bottleneck analysis — v1 is already on the HBM roofline

This is the central finding of phase-2 investigation, and it reframes the whole
phase. I reconstructed the roofline two independent ways:

**1. Traffic is fixed and minimal.** SiLU is a pure elementwise map. The fp32 in/out
contract forces exactly read-once + write-once:
`2 * 4096 * 7168 * 4 B = 234.9 MB` total HBM traffic. v1's profiler numbers
(HBMrd 117 MB + HBMwr 117 MB = 234.9 MB) are **exactly** this floor — there are
zero redundant SBUF passes, zero recompute. There is no traffic left to remove.

**2. v1 is at the measured bandwidth ceiling.** Effective aggregate HBM bandwidth
implied by v1: `234.9 MB / 0.3009 ms = 781 GB/s`. Two cross-checks:
   - The cost model's conservative trn2 figure is 368 GB/s ⇒ serialized floor
     `234.9 MB / 368 GB/s = 0.638 ms`. v1 (0.3009 ms) is **2.1x below** that, which
     means the model's 368 GB/s is ~2x conservative for this streaming pattern
     (read and write overlap on the real HBM channels; the model serializes them).
   - Cost-model *compute* floor (fused silu, Scalar cpe=1, freq=120 MHz-scaled,
     free=7168, x32 iters) ≈ **0.191 ms** — comfortably hidden under DMA (Scl=34% «
     DMA=97%). And the per-direction overlapped DMA floor
     (`117 MB / 368 GB/s = 0.319 ms`) essentially equals the measured 0.3009 ms.
   - **DMA=97% active** is the direct confirmation: the DMA subsystem is saturated.
     Only ~3% (~9 µs) of the runtime is DMA-idle bubble.

**Conclusion: there is no multiplicative headroom.** AccelOpt's "~1.67x here" was
relative to the *baseline* (which does 4 vector passes through 5 SBUF buffers);
v1 already captured that and more (3.398x) by collapsing to read-once → fused-silu
→ write-once. The only slack that physically remains is the ~3% DMA-idle bubble.
Phase-2 realistic ceiling: **single-digit-% latency (0.3009 → ~0.29 ms), or zero.**
The honest primary deliverable of phase 2 is a *rigorous confirmation that we are at
the roofline*, plus harvesting the DMA-issue bubble **only if the gain clears the
same-session noise band** (per the fast-vs-full-run latency lesson).

## Optimization directions — enumerated, ranked by benefit/risk

The one lever with any physical basis is **reducing DMA-issue / scheduling overhead**
so the ~3% idle bubble shrinks. Everything that would give a *multiple* (less
traffic) is blocked by the fp32 contract. Ranked:

### D1 (rank 1, primary) — Wider DMA bursts via multi-slice batching

**Idea.** Process `k` middle-axis slices per iteration as **one contiguous** transfer
instead of `k` separate ones. Because the tiled layout is `(128, 32, 7168)` = `[p, m, f]`
with middle-axis stride 7168 and free stride 1, `v1[:, i0:i0+k, :]` is a *contiguous*
`[128, k, 7168] = [128, k*7168]` slab per partition. So one `nl.load` of
`[128, k*7168]`, one `nisa.activation(op=nl.silu)` over the `k*7168` free dim (well
under the ~32767 Scalar free-dim limit for k≤4: `4*7168 = 28672`), one `nl.store`.

**Why it could help.** Cuts the DMA op count from 32 loads + 32 stores (k=1) to
`32/k` each. Each DMA carries a fixed issue/semaphore cost (`semaphore_start` = 1300 ns
in the model); fewer, larger bursts amortize that fixed cost and can shrink the ~3%
idle. Precedent: `d3cbeffd` [legacy] gather/scatter "coalescing DMA copy to improve
performance"; the general wider-burst pattern.

**Tension to measure (this is why it's a sweep, not a fixed choice).** `affine_range`
overlaps DMA with compute *across iterations*; with `k` large there are fewer
iterations (k=4 → 8 iters), so the pipeline is coarser and prologue/epilogue bubbles
are relatively larger. So there's a sweet spot: bigger bursts cut per-DMA overhead
but coarsen the pipeline. **Sweep k ∈ {1, 2, 3, 4}** and keep the best.

**SBUF budget check (trn2, 208 KB usable/partition, 28 KB per slice).** Distinct
load+store buffers need `2k` slices live ⇒ `2k ≤ 7` ⇒ **k ≤ 3** with separate
x/y buffers; k=4 needs in-place (D3) or single-buffered store. This bounds the sweep
and couples D1 with D2/D3.

**Risk.** Low — correctness-neutral (elementwise; same math, just wider tiles;
128*32=4096 and 7168 stay exact, still mask-free). Main risk is *no* gain (bubble was
never issue-bound) or a small regression at large k (coarser pipeline). Cheap to test.

### D2 (rank 2) — Explicit double-buffering / ping-pong SBUF

**Idea.** Pre-allocate two SBUF buffer sets and manually prefetch slice i+1 while
computing slice i (precedent `bc877398`, `3c7e053b`).

**Why it is rank 2, not rank 1.** v1 *already* uses `nl.affine_range`, which licenses
the compiler to software-pipeline DMA against compute — and the evidence says it is
doing so: v1 sits at the overlapped floor (0.3009 ≈ 0.319 ms one-way), not the
serialized 0.638 ms. So explicit ping-pong is most likely **redundant** here. Worth
**exactly one** confirmation run: if the compiler's auto-pipeline is already optimal,
manual ping-pong yields ~0 and may even hurt (extra SBUF pressure competes with D1's
batch width, per the budget check above). Keep only if it clears the noise band.

**Risk.** Low correctness risk; real risk is wasted SBUF that shrinks the feasible
batch width. Test D2 as a small standalone check, then only combine with D1 if it won.

### D3 (rank 3, enabler) — In-place compute (write silu output into the load buffer)

**Idea.** For an elementwise op, `y_tile` can alias `x_tile` (activation reads and
writes the same SBUF region). Halves live SBUF residency from `2k` to `k` slices,
which **unlocks larger batches** for D1 (e.g. k=4, or k=6–7 single-buffered) without
blowing the 208 KB budget.

**Why rank 3.** It removes no HBM traffic (still read-once/write-once) and no compute,
so on its own it changes nothing measurable. Its only value is as an **enabler** for
a wider D1 sweep. Test only in combination with D1 if the k≤3 sweep suggests bigger
bursts are still improving at the SBUF ceiling.

**Risk.** Low, but must confirm `nisa.activation` supports in-place src==dst without a
read/write hazard on the Scalar engine; verify correctness on the full 5 seeds (an
in-place aliasing bug would show as an L2 failure, which `verify.py` catches).

### D4 (rank 4, likely rejected) — DMA descriptor mode (`dge_mode`)

Precedent `d1124a76` sets `dge_mode=none` for static contiguous DMAs to move
descriptor generation off the critical path. **But the scoring harness compiles with
`--disable-dge`** (from the acceptance contract), so hardware DGE is already off
globally — this lever is very likely **neutralized before we touch it**. Record the
reasoning; try at most one run only if D1/D2 leave an unexplained bubble, otherwise
reject without spending an iteration.

### D5 (rejected outright) — bf16 / traffic reduction

The only way to get a *multiple* is to move fewer bytes. The gate mandates fp32 in
and fp32 out, and 234.9 MB is the read-once/write-once minimum for that contract.
There is no dtype trick, no fusion, no recompute-avoidance left. **Reject** — and this
is precisely *why* no multiplicative phase-2 win exists. (Documented so phase 3, if it
specializes shapes, doesn't re-litigate this.)

## Experiment plan (fits the "≤5 iterations per direction" budget)

All candidates parented off `silu_v1` in `candidates.jsonl`; each perf change recorded
in `benchmark.csv`; profiler digests kept under `profile/`.

1. **D1 sweep (primary, up to 4 iters):** `silu_v2_k2`, `silu_v2_k3`, and — if D3
   confirms in-place is safe — `silu_v2_k4`. For each: `verify.py --fast` first to
   screen, then full 5-seed on any that beats v1. Record k vs latency; keep the best.
2. **D2 (1 iter):** `silu_v2_pingpong` (k=1 explicit two-buffer prefetch). Compare to
   v1. Expected ~0; keep only if it clears the noise band. If it wins, retest the D1
   sweep with ping-pong layered on (SBUF permitting).
3. **D3** is not a standalone candidate — it's folded into the k=4 D1 variant as the
   enabler. Its "test" is that the in-place k=4 kernel passes the full 5-seed L2 gate.
4. **D4** only if an unexplained bubble remains after 1–2; else reject on the
   `--disable-dge` reasoning without an iteration.

**Noise discipline (mandatory).** v1's win is real and large; any phase-2 delta is
tiny (single-digit %). Before promoting *anything*, establish a same-session noise
band: run v1 and the candidate back-to-back on the full 5-seed measurement and only
promote if the improvement exceeds run-to-run jitter. A candidate that merely ties v1
within noise is **not** promoted — v1 stays. This follows the recorded fast-vs-full
latency lesson: `--fast` (seed 42, low iters) is a screen, not a decision.

## Acceptance for phase 2

1. Correctness never regresses: every promoted candidate passes relative-L2 < 2e-5 on
   all five seeds (full run, not `--fast`).
2. The DMA-roofline finding is documented with evidence (traffic = 234.9 MB floor;
   effective BW ≈ 781 GB/s; DMA=97% saturated) so the "no multiplicative headroom"
   conclusion is defensible, not asserted.
3. If a burst-coalescing (D1) variant clears the same-session noise band, promote it
   and record k, latency, and the new speedup; otherwise keep `silu_v1` as the phase-2
   result and record that it is confirmed at the roofline.
4. Every candidate in `candidates.jsonl` (DAG parented off `silu_v1`); every perf
   change in `benchmark.csv`; profiler evidence under `profile/`.

## What phase 2 deliberately does NOT do

- **No new math / no dtype change.** fp32 in/out is fixed; the fused `nl.silu` LUT is
  already L2-accurate. No sigmoid+multiply, no exp-exact, no bf16 — none would help
  (compute is hidden; traffic is fixed) and bf16 would risk the gate.
- **No layout change / no transpose.** Elementwise ⇒ layout is correctness-neutral;
  keep `(128, 32, 7168)` so the harness reconciliation is untouched. Any batching
  slabs contiguous middle-axis slices, not a re-layout.
- **No chasing sub-noise deltas.** If the roofline analysis holds (it does on the
  measured numbers), phase 2 may correctly conclude with v1 unchanged plus a
  documented roofline confirmation. Reordering instructions to fight a 3% bubble that
  is within noise is not a win.

--- Original Design Draft End ---
