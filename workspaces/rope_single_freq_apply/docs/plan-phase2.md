# rope_single_freq_apply — Phase 2: Profile-Driven Latency Optimization (Layout B Packing)

## Goal Description

Start from the promoted phase-1 kernel (`runs/rope_v1.py`, layout A, 64-partition,
no-copy, **0.9445 ms, 1.209x** over the 1.1418 ms NKIBench baseline) and cut on-device
latency **without ever regressing correctness** (rel-L2 must stay `< 2e-5`; today it is
exactly `0.0` on all five seeds `[0, 21, 42, 63, 84]`).

The lever is chosen from the **measured phase-1 profiler verdict**, not a fresh guess.
That verdict (`docs/phase1-bottleneck-digest.md`) is: **VECTOR-BOUND, co-limited with
DMA-active, HBM at the read-once/write-once floor** — Vec active 91.6%, DMA active 93.5%,
MBU only 29.9%, effective BW ~427 GB/s (well below the ~781 GB/s single-core streaming
roofline the silu campaign measured on this same profiler), HBM traffic exactly
402.65 MB (268.44 read + 134.22 write). The pure-DMA ceiling at 781 GB/s is
`402.65 MB / 781 GB/s = 0.516 ms`; we measure 0.944 ms, so **~0.43 ms is vector time not
hidden under DMA**. Two hard constraints follow and gate every direction:

1. **HBM is already at the floor** (0% over). There is no traffic to remove, and any
   *added* HBM read (a second read of `cos`/`sin`, or reloading `x`) pushes above floor
   and costs DMA-active time on an already-93.5%-busy engine → forbidden.
2. **The prize is the ~0.43 ms of unhidden vector time.** The only way to shrink it is
   **fewer / wider vector passes**. The realistic floor is the ~0.516 ms DMA ceiling →
   best-case latency ~0.55–0.65 ms → ~1.75–2.0x total.

The primary direction is **layout B**: because `tensor_tensor` op latency is per-free-
element and independent of partition count (128 SIMD lanes run in parallel), packing both
RoPE output halves onto all 128 partitions lets **3 `tensor_tensor` passes do the work of
6**, roughly halving the co-limiting vector term. `x_in` is *already* `[x0; x1]` on 128
partitions in HBM, so the natural load costs one DMA (cheaper than layout A's two loads).
The crux and the whole bet: the required **cross-partition** builds (`Aswap±`, `Ccos`,
`Csin`) must land on an **idle** engine (PE / DMA / Scalar) and **hide** under the floor,
rather than bouncing back onto the Vector engine and netting nothing.

Explore each ranked direction for **at most five design iterations across the D1
realizations**, collecting before/after `verify.py` latency + the per-engine profiler
digest to justify keep / revise / reject. Full-seed validation runs and threshold reruns
are **not** counted against that design-iteration budget (see AC-5). The deliverable is
the best correct kernel promoted (with `benchmark.csv` / `candidates.jsonl` / `profile/`
evidence) plus a short verdict on whether layout B beat layout A and which build-engine
realization won — steering phase-3 shape specialization.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: **Correctness preserved on all five seeds.** Every promoted candidate passes the
  NKIBench relative-L2 gate (`l2_norm_passed == true`, rel-L2 `< 2e-5`) on all of
  `[0, 21, 42, 63, 84]` via `verify.py --op rope_single_freq_apply`. Exact-preserving
  variants (D1b DMA/ScalarE build, D2 `W`-sweep) are expected to report rel-L2 `= 0.0`
  (IEEE fp32 negation is a sign-bit flip, so `a + (−b) ≡ a − b` and `(−x1)·sin ≡
  −(x1·sin)` bit-identically; pure data movement does not perturb the arithmetic).
  Bitwise equality is **not** required — the gate is rel-L2, not allclose. NaN / infinity
  / signed-zero / subnormal handling is **out of scope**: NKIBench inputs are finite
  `np.random.normal` draws.
  - Positive Tests (expected to PASS):
    - Full 5-seed `verify.py` (non-`--fast`) reports `l2_norm_passed` true for every seed on the promoted candidate.
    - D1b and D2 candidates report per-seed rel-L2 `= 0.0` (exact); any exact-path candidate with nonzero rel-L2 is a red flag whose arithmetic ordering is diffed before trusting it.
  - Negative Tests (expected to FAIL):
    - A layout-B candidate whose packed algebra is wrong (e.g. sign not baked into `Aswap±`, or halves not swapped) fails the gate on at least one seed.
    - A candidate scored on only the single `--fast` seed 42 is NOT eligible for promotion; the full 5-seed gate is mandatory before any promote.

  - AC-1.1: **PE-matmul variants gated at full fidelity before promotion.** Any candidate
    that builds `Aswap±`, `Ccos`, or `Csin` via a PE (`nc_matmul`) permutation (D1a, D1c)
    MUST pass the FULL 5-seed gate — not just the `--fast` seed-42 screen — before it may
    be promoted, because fp32 matmul may decompose internally (bf16/tf32) and perturb the
    result.
    - Positive: a D1a/D1c candidate is promoted only after a recorded full 5-seed pass.
    - Negative: a D1a/D1c candidate promoted on the strength of a `--fast`-only run, or with any seed failing, is rejected.

- AC-2: **HBM traffic stays at the read-once/write-once floor.** No candidate raises HBM
  traffic materially above the 402.65 MB floor (HBMrd ≈ 268.44 MB, HBMwr ≈ 134.22 MB).
  "Floor-preserving" means no *systematic* increase beyond profiler noise, judged on the
  **read and write components separately**, not the total alone. The 64→128 broadcast of
  `cos`/`sin` and the half-swap of `x` must be SBUF-resident (DMA) or PE-permutation —
  never a second HBM read. SBUF/PSUM spills that surface as extra HBM read/write are a
  rejection.
  - Positive Tests (expected to PASS):
    - Measured `HBMrd_MB` and `HBMwr_MB` each sit within a small tolerance (≈ ±10%, consistent with phase-1 AC-2) of 268.44 / 134.22 MB, with the read/write split intact.
    - The layout-B natural load reads `x`, `cos`, `sin` exactly once and writes `out` exactly once per full pass.
  - Negative Tests (expected to FAIL):
    - A candidate that broadcasts `cos`/`sin` to 128 partitions by re-reading them from HBM (≈2× the trig reads) is rejected.
    - A candidate whose measured HBM read or write is repeatably above floor (e.g. from PSUM/SBUF spill) is rejected even if latency looks favorable.

- AC-3: **Layout-B packed algebra is correct and minimal (3 `tensor_tensor`).** The packed
  compute is `M1 = A ⊙ Ccos`, `M2 = Aswap± ⊙ Csin`, `out = M1 + M2`, where `A = [x0; x1]`
  (natural 128-partition load), `Aswap± = [−x1; +x0]` (halves swapped, top half negated),
  `Ccos = [cos; cos]`, `Csin = [sin; sin]`. The sign is baked into `Aswap±` so the final
  combine is a **single `add` across all 128 partitions** (a partition-dependent
  add-top/sub-bottom is not one `tensor_tensor`). This yields exactly `[out0; out1] =
  [x0·cos − x1·sin; x0·sin + x1·cos]`.
  - Positive Tests (expected to PASS):
    - The kernel realizes exactly three `nisa.tensor_tensor` passes over `[128, W]` (two multiplies + one add), verifiable by inspection.
    - The output rows `0:64` equal `x0·cos − x1·sin` and rows `64:128` equal `x0·sin + x1·cos`, matching the numpy oracle.
  - Negative Tests (expected to FAIL):
    - Baking the sign into a `[−sin; +sin]` `Csin` instead of `Aswap±` (forcing a partition-dependent final combine that is not a single `tensor_tensor`) is rejected.
    - A realization that needs a partition realignment before the final `add` (breaking the 3-pass count) is not the layout-B target and is documented as such.

- AC-4: **The packing win must MATERIALIZE as measured evidence, not merely relocate the
  cost.** A promoted layout-B candidate shows **Vec% dropping materially toward ~half**
  while PE% / DMA% / Scl% absorb the cross-partition build, HBM stays at floor (AC-2), and
  **latency falls**. Trading 6 vector passes for 3 vector passes + 3 movement ops with
  net-zero (or worse) latency is a **documented reject**, not a promote — and the evidence
  records which engine the build landed on and the post-build Vec/DMA/PE/Scl %.
  - Positive Tests (expected to PASS):
    - The promoted candidate's digest shows a materially lower `Vec_pct` than layout A's 91.6% AND a lower latency than 0.9445 ms, with HBM at floor.
    - The candidate notes attribute the vector reduction to packing (3 vs 6 passes) and name the engine(s) that absorbed the build.
  - Negative Tests (expected to FAIL):
    - A candidate whose `Vec_pct` drops but whose latency does not improve (the build did not hide — it serialized or bounced back onto Vector) is recorded as a reject with the per-engine reason, not promoted.
    - Declaring a layout-B win from latency alone, without the per-engine digest showing where the vector time went, fails this criterion.

- AC-5: **Build-engine exploration with an explicit keep/revise/reject rule.** Explore the
  D1 realizations in ascending correctness risk, within **≤5 design iterations across
  D1a/D1b/D1c** (full-seed validation runs, threshold reruns, and the D2 `W`-sweep are
  separate and NOT counted against this cap):
    - **D1b first** (lowest correctness risk): build `Ccos`/`Csin`/`Aswap±` via SBUF→SBUF
      DMA broadcast/swap, negate the top half on ScalarE (`activation(copy, scale=−1)`),
      all-SBUF (no PSUM constraint). Arithmetically exact.
    - **D1a / D1c** (escalate only if D1b's added DMA-active eats the win): move the builds
      onto the idle **PE** (permutation matmul → PSUM) and Scalar. For D1c hybrid, place
      `M1 = A(SBUF) ⊙ Ccos(SBUF)` and `M2 = Aswap±(PSUM) ⊙ Csin(SBUF)` so at most one
      operand per `tensor_tensor` is in PSUM (the "not both operands in PSUM" rule) with
      zero extra copies.
    Each D1 variant is itself a **full `verify.py` candidate kernel** whose per-engine
    digest reveals whether the build hid — there are no separate isolated-primitive
    microbenchmarks. Keep/revise/reject on **stabilized p50** (per AC-8's measurement
    protocol):
    - **Keep & promote** if p50 `< 0.80 ms` (> 1.42x) with all 5 seeds passing (stretch
      success `< 0.65 ms`, ~1.75x).
    - **Revise** (try the next D1 variant) if p50 is `0.80–0.90 ms`: packing helped but the
      build did not fully hide — move it to a more idle engine.
    - **Reject, keep layout A** if p50 `≥ 0.90 ms` after exhausting D1a/b/c, or if the gate
      fails on every exact-preserving variant. Record *why* (which engine, post Vec/DMA/PE
      %) so phase 3 does not re-tread it.
  - Positive Tests (expected to PASS):
    - D1b is attempted before any PE-path variant; each variant's decision cites its stabilized p50 against the 0.80 / 0.90 ms thresholds.
    - The recorded evidence names the build engine per variant and the resulting per-engine digest.
  - Negative Tests (expected to FAIL):
    - Jumping straight to the PE path (D1a) without first trying the exact D1b path, absent a compile/feasibility reason, is out of the planned order.
    - Promoting a variant on a single unstable profiler run near a threshold, without the AC-8 rerun, fails this criterion.

- AC-6: **Finer/coarser free-axis `W` sweep (D2), exact and cheap.** Sweep legal
  power-of-two tile widths `W ∈ {512, 1024, 2048, 4096}` (each divides `S = 2^18` →
  mask-free; `1536` is excluded because it does not divide `S`). `W = 4096` is used only
  if an SBUF/compile headroom precheck clears it. The best `W` must be **revalidated on
  the actually promoted layout** — layout B doubles the partition count and live-tile
  count, so its viable/optimal `W` may differ from (and be smaller than) layout A's; a
  layout-A `W`-sweep may serve as an early baseline-refiner but its winner does not
  automatically transfer. This is a near-zero-risk, arithmetic-preserving probe (silu
  precedent found finer tiles win by amortizing the pipeline fill/drain bubble; the
  effBW 427 « 781 gap and 30% MBU here hint at bubble headroom), but the digest rates it
  **secondary** to D1.
  - Positive Tests (expected to PASS):
    - A `W` that lowers latency with all 5 seeds passing (rel-L2 = 0.0) on the promoted layout is adopted.
    - Every swept `W` divides `S` exactly (no masking / tail emitted).
    - `W = 4096` is only adopted after a recorded SBUF/compile headroom check on the target layout.
  - Negative Tests (expected to FAIL):
    - A `W` that does not divide `S` (requiring a masked tail) is not used.
    - Assuming layout A's best `W` is optimal for layout B without revalidating on layout B fails this criterion.
    - A `W` change that raises latency or fails a seed is recorded and dropped, not promoted.

- AC-7: **Evidence-rejected directions (D3) stated, not iterated.** Two directions are
  rejected up front on evidence and consume ~0 iterations:
    - **bf16 / lower-precision downcast — REJECT.** RoPE is pure elementwise with no
      reduction to average away rounding (unlike the rmsnorm_matmul case where K-averaging
      rescued compensated-bf16); bf16 elementwise error ≈ 2⁻⁸ ≈ 4e-3 » the 2e-5 gate. fp32
      is mandatory.
    - **Explicit ping-pong / wider burst-batching — REJECT (precedent).** The silu campaign
      found wider burst-batching + manual ping-pong regressed; `affine_range` already
      builds the software pipeline. At most one confirmatory probe if D1/D2 both stall.
  - Positive Tests (expected to PASS):
    - The plan/evidence records these rejections with their rationale; no full iteration budget is spent implementing them.
  - Negative Tests (expected to FAIL):
    - Spending design iterations building a bf16 kernel (which will fail the 2e-5 gate) instead of documenting the rejection fails this criterion.

- AC-8: **Measurement protocol + evidence integrity.** Each candidate is scored with the
  recorded protocol and recorded with full evidence:
    - **Protocol:** `--fast` (seed 42) is the iteration screen; the full 5-seed run is
      mandatory before any promote. Latency is compared on **stabilized p50** — a candidate
      landing within ~3% of a decision threshold (0.80 / 0.90 ms) or near a promotion tie
      is **re-profiled** (≥2 profiler runs) and, where available, min/p90 or spread is
      recorded to distinguish a real win from profiler/compile-cache noise.
    - **Evidence:** one `benchmark.csv` row and one `candidates.jsonl` entry per candidate,
      parented to `rope_v1` (DAG), following the existing workspace schema; the profiler
      digest per direction under `profile/`; the candidate `.py` under `runs/`; NO edits to
      `../../AccelOpt/NKIBench/`. Each entry records the realization path label, the build
      engine, the post-metrics (Vec/DMA/PE/Scl/MBU %, HBMrd/HBMwr/total), the exact
      `verify.py` command, and — for rejects — a "reason rejected" note (correctness fail /
      HBM regression / no-latency-win / compile-or-resource fail / unstable timing). The
      final promoted entry records the **phase-3 steering verdict** (did layout B beat
      layout A; which build-engine realization won).
  - Positive Tests (expected to PASS):
    - After scoring, `benchmark.csv` gains one row per candidate and `candidates.jsonl` one JSON line per candidate, each `parent`-linked to `rope_v1`, with the metrics digest and path label.
    - A near-threshold candidate has ≥2 recorded profiler runs before its keep/reject decision.
    - `git status` shows new/edited files only under `workspaces/rope_single_freq_apply/runs/`, `profile/`, `benchmark.csv`, `candidates.jsonl` — never under `../../AccelOpt/NKIBench/`.
  - Negative Tests (expected to FAIL):
    - Any edit to `../../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}` fails the workspace contract.
    - A `candidates.jsonl` entry with no `parent` link (broken DAG), or a promotion from a single unstable near-threshold run, fails this criterion.
    - A rejected variant recorded with no per-engine digest and no reason-rejected note fails this criterion.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
All ranked directions explored to their evidence-bounded limits: the D2 `W`-sweep as an
early exact baseline-refiner; layout B implemented and profiled across the D1b (exact
DMA/ScalarE), D1a (PE permutation → PSUM), and D1c (hybrid) build-engine realizations
within the ≤5-design-iteration cap; the best correct variant's `W` revalidated on its own
layout; a single confirmatory ping-pong/burst probe only if D1 and D2 both stall. Each
candidate carries a full per-engine profiler digest, stabilized (re-profiled)
near-threshold measurements, and a "reason rejected" note where applicable. The promoted
kernel clears p50 `< 0.80 ms` (ideally approaching the ~0.516 ms DMA ceiling), passes all
5 seeds, holds HBM at the 402.65 MB floor, and is recorded with a phase-3 steering verdict
naming the winning build engine.

### Lower Bound (Minimum Acceptable Scope)
The exact D1b layout-B variant and the D2 `W`-sweep are implemented and scored on the full
5-seed gate, each recorded with a per-engine digest and parented evidence row. If no
layout-B realization hides its build well enough to beat layout A (all land `≥ 0.90 ms`),
the **outcome is still acceptable**: layout A (`rope_v1.py`, 0.9445 ms) remains the
promoted kernel, and phase 2 delivers the recorded reason (which engine the build landed
on, post Vec/DMA/PE %) plus any exact `W` improvement, steering phase 3 away from the dead
lever. Correctness (rel-L2 `< 2e-5` on all seeds) and HBM-floor preservation are hard
floors that every recorded candidate must respect.

### Allowed Choices
- Can use: the natural 128-partition `A = [x0; x1]` load; `nl.load` / `nl.store` with
  explicit predeclared `nl.ndarray((par_dim(P), W))` SBUF tiles and `nl.arange` indexing
  (repo convention); `nisa.tensor_tensor` with `nl.multiply` / `nl.add`; `nl.affine_range`
  for the free-axis loop; SBUF→SBUF DMA broadcast/swap (D1b), PE `nc_matmul` permutation
  → PSUM (D1a/D1c), and ScalarE `activation(copy, scale=−1)` for the exact negate; legal
  power-of-two `W ∈ {512, 1024, 2048, 4096}` dividing `S`; the PSUM-placement discipline
  that keeps at most one `tensor_tensor` operand in PSUM.
- Must verify before use: the fp32 exactness of any PE-matmul build path (full 5-seed gate
  before trusting its latency — AC-1.1); an SBUF/compile headroom precheck before `W =
  4096` on the target layout; that each cross-partition build actually lands on PE / DMA /
  Scalar and hides (read the per-engine digest — AC-4), not that it does so by assumption.
- Cannot use: any dtype other than fp32 (bf16 fails the gate — AC-7); a second HBM read of
  `x` / `cos` / `sin` or any change that raises HBM above the 402.65 MB floor (AC-2);
  `nc_stream_shuffle` for the 64↔64 half-swap (it runs on the Vector engine we are
  unloading and only shuffles within 32-partition quadrants); masked / tail tiles (choose
  `W` dividing `S`); baking the sign anywhere but `Aswap±` if it breaks the single-`add`
  final combine (AC-3); editing the NKIBench baseline/reference/seeds/`summary.json`;
  hand-tuning a baseline.

> **Note on Deterministic Designs**: The draft specifies a highly deterministic algebra
> (fixed layout-B packed identity, fixed op count, fixed correctness gate, fixed HBM floor)
> but leaves a genuine open choice in the **build-engine realization** (D1a vs D1b vs D1c)
> whose winner is decided empirically by the profiler. The path boundaries are therefore
> narrow on the math/dtype/traffic/evidence axes and deliberately exploratory on the
> build-engine axis within the ≤5-iteration cap. The fixed problem instance (`S = 262144`,
> `D = 128`, fp32, identity `transform_to_nki_inputs`, single trn2 core) is not to be
> generalized in phase 2.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Layout B packed compute — one possible realization (D1b, the exact, all-SBUF path tried
first):

```
P      = 128
half   = 64
W      = 2048                       # power-of-two dividing S = 2^18; revalidate on layout B
out = nl.ndarray((128, S), dtype=fp32, buffer=nl.shared_hbm)

for j in nl.affine_range(S // W):                     # independent iters -> compiler pipelines DMA vs compute
    base = j * W
    # 1 DMA for x (natural [x0; x1] on 128 partitions), 1 each for cos/sin (64 partitions)
    A  = load x_in[0:128,  base:base+W]               # [128, W] = [x0; x1]
    c  = load cos[0:64,    base:base+W]               # [64,  W]
    s  = load sin[0:64,    base:base+W]               # [64,  W]

    # cross-partition BUILD (must hide on an idle engine, no extra HBM):
    Ccos    = broadcast c 64->128 partitions          # [cos; cos]   (SBUF->SBUF DMA)
    Csin    = broadcast s 64->128 partitions           # [sin; sin]   (SBUF->SBUF DMA)
    Aswap   = swap the two 64-partition halves of A    # [x1; x0]     (SBUF->SBUF DMA)
    Aswap_pm= negate top half of Aswap on ScalarE      # [-x1; +x0]   (activation copy scale=-1, exact)

    # 3 tensor_tensor over [128, W]  (was 6 over [64, W] in layout A):
    M1  = tensor_tensor(A,        Ccos, multiply)      # [ x0*cos ;  x1*cos ]
    M2  = tensor_tensor(Aswap_pm, Csin, multiply)      # [-x1*sin ;  x0*sin ]
    out_tile = tensor_tensor(M1, M2, add)              # [ x0*cos - x1*sin ; x1*cos + x0*sin ] == [out0; out1]
    store out[0:128, base:base+W] = out_tile           # single [128, W] DMA
```

For D1a/D1c, the three builds become 0/±1 permutation matmuls on the idle PE engine:
`Ccos = [I; I]·cos`, `Csin = [I; I]·sin`, `Aswap± = S±·A` with `S± = [[0, −I], [I, 0]]`,
outputs in PSUM. Because a dense 0/±1 permutation matmul contracts over 128 and produces
`[128, W]`, it is **real W-streaming matmul work**, not a free copy — its win depends on
that work *hiding* on the otherwise-idle PE. Place operands so at most one `tensor_tensor`
input is in PSUM (D1c: `M1 = A(SBUF) ⊙ Ccos(SBUF)`, `M2 = Aswap±(PSUM) ⊙ Csin(SBUF)`).
Verify the fp32 gate on any PE path before trusting its latency.

The decisive signal for D1 (AC-4): **Vec% dropping toward half while HBM_total stays at
floor and PE%/DMA%/Scl% absorb the build** — that is the packing win materializing rather
than merely relocating. If Vec% drops but latency does not, the build serialized or
bounced back onto Vector; record which engine and revise.

### Relevant References
- `docs/phase1-bottleneck-digest.md` — the measured phase-1 verdict (VECTOR-BOUND, HBM at floor) that selects the layout-B lever; the source of truth for phase 2.
- `runs/rope_v1.py` — the promoted phase-1 layout-A kernel (parent of all phase-2 candidates); the fallback if no layout-B realization hides.
- `../../AccelOpt/NKIBench/kernels/rope_single_freq_apply_B1_H64_N4096_D128_0.py` — NKIBench baseline (1.1418 ms); its `nl.copy` realign is exactly what packing avoids; do not edit.
- `../../AccelOpt/NKIBench/reference/rope_single_freq_apply_B1_H64_N4096_D128_numpy_1.py` — numpy oracle; identity `transform_to_nki_inputs`; the exact op sequence and shapes to reproduce.
- `../../verify.py` — scores a candidate; gates on `l2_norm_passed`; prints the profiler digest / `summary_metrics`. Run with `python3 ../../verify.py --op rope_single_freq_apply --candidate runs/<file>.py [--fast]`.
- `benchmark.csv` / `candidates.jsonl` — the evidence-row / JSON schema to mirror (parent = `rope_v1`).
- Skill `nki-api-reference` / `nki-concept-docs` — for confirming SBUF→SBUF DMA broadcast/swap semantics, the `nc_stream_shuffle` quadrant restriction, `nc_matmul` fp32 decomposition behavior, and the `tensor_tensor` PSUM-operand rule.
- Skill `kernel-cost-analysis` — theoretical per-engine cost of the 3-pass vs 6-pass vector floor and the PE permutation-matmul cost (context only; the measured digest is the source of truth — the cost model over-predicts vector for this op).
- Skill `kernel-optimization-kb` — precedents for SBUF partition broadcast (`TensorView.broadcast()` on the DMA path, GpSimd SBUF→SBUF `dma_engine.gpsimd_dma`) and PE permutation.
- Memory `[[kda-rope-progress]]` — phase-1 verdict and the layout-B lever framing.
- Memory `[[kda-silu-progress]]` — finer-`W` precedent (finer wins when a DMA bubble exists) and the ping-pong/burst-batching regression precedent (D3).
- Memory `[[kda-rmsnorm-matmul-progress]]` — why compensated-bf16 worked there (K-averaging) and does NOT transfer to RoPE (no reduction) — the D3 bf16 rejection rationale.

## Dependencies and Sequence

### Milestones
1. **D2 `W`-sweep on layout A (early exact baseline-refiner).** Independent of the
   layout-B work; can run first because it is arithmetic-preserving and cheap.
   - Phase A: SBUF/compile headroom precheck for `W = 4096`; screen legal `W ∈ {512, 1024,
     2048, 4096}` with `--fast`.
   - Phase B: full 5-seed on any `W` that lowers latency; record the geometry sensitivity
     (this establishes whether the ~0.43 ms gap is tile-geometry sensitive before layout B).
2. **D1b — exact all-SBUF layout B.** (Depends on the packed algebra AC-3.)
   - Step 1: implement the natural `A` load + SBUF→SBUF DMA broadcast/swap + ScalarE
     negate + 3 `tensor_tensor`; `--fast` screen for compile + correctness.
   - Step 2: full 5-seed gate; read the digest — did Vec% drop toward half, HBM stay at
     floor, latency fall below 0.80 ms? Apply the keep/revise/reject rule (AC-5).
3. **D1a / D1c — PE / hybrid build.** (Depends on Milestone 2 outcome: only if D1b's added
   DMA-active eats the win.)
   - Step 1: move the builds onto idle PE (→ PSUM) with the one-PSUM-operand placement;
     `--fast` screen.
   - Step 2: FULL 5-seed gate (AC-1.1) before trusting latency; read the per-engine digest.
4. **Select, revalidate `W`, and record.** (Depends on Milestones 1–3.)
   - Step 1: pick the lowest-p50 candidate passing all 5 seeds; re-profile near-threshold /
     tie candidates to clear noise (AC-8); break near-noise ties toward the simpler/exact
     kernel (see DEC-1).
   - Step 2: revalidate the best `W` on the promoted layout (AC-6).
   - Step 3: record `benchmark.csv` + `candidates.jsonl` (parent = `rope_v1`) + `profile/`
     digest + the phase-3 steering verdict; confirm no NKIBench edits.

Dependency summary: the D2 `W`-sweep (M1) runs independently/early; layout B proceeds
D1b (M2) → conditionally D1a/D1c (M3) → selection + `W`-revalidation + evidence (M4). If
every layout-B realization lands `≥ 0.90 ms`, layout A stays promoted and M4 records the
reason and any exact `W` gain (lower-bound-acceptable outcome).

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Confirm NKI primitive semantics for the cross-partition build: SBUF→SBUF DMA 64→128 partition broadcast + 64↔64 half-swap, ScalarE exact negate, `nc_matmul` fp32 decomposition behavior, and the `tensor_tensor` "not both operands in PSUM" rule (via `nki-api-reference` / `kernel-optimization-kb`) | AC-2, AC-3, AC-1.1 | analyze | - |
| task2 | D2: SBUF/compile headroom precheck for `W=4096`; `W`-sweep `{512,1024,2048,4096}` on layout A (`runs/rope_v1.py`), `--fast` screen then full 5-seed on any improver; record geometry sensitivity | AC-6, AC-8 | coding | - |
| task3 | D1b: implement exact all-SBUF layout B (`runs/rope_v2_layoutB_dma.py`) — natural `A=[x0;x1]` load, SBUF→SBUF DMA broadcast of `cos`/`sin` and half-swap of `x`, ScalarE negate of top half, 3 `nisa.tensor_tensor` over `[128,W]`, single `[128,W]` store | AC-3, AC-2, AC-1 | coding | task1 |
| task4 | Score D1b: `--fast` screen then full 5-seed `verify.py`; read the per-engine digest; confirm Vec% dropped, HBM at floor; apply keep/revise/reject (AC-5); record evidence + reason if rejected | AC-1, AC-4, AC-5, AC-8 | coding | task3 |
| task5 | D1a/D1c (only if task4 revises): implement PE-permutation / hybrid build (`runs/rope_v3_layoutB_pe.py`) with one-PSUM-operand placement (`M1=A(SBUF)⊙Ccos(SBUF)`, `M2=Aswap±(PSUM)⊙Csin(SBUF)`); FULL 5-seed gate before trusting latency | AC-1.1, AC-3, AC-5 | coding | task4 |
| task6 | Score D1a/D1c: full 5-seed gate + per-engine digest; verify no PSUM/SBUF spill to HBM; apply keep/revise/reject; record evidence + reason | AC-1.1, AC-2, AC-4, AC-5, AC-8 | coding | task5 |
| task7 | Select the lowest-p50 correct candidate; re-profile near-threshold/tie candidates (≥2 runs) to clear noise; revalidate best `W` on the promoted layout; break near-noise ties toward the simpler/exact kernel per DEC-1 | AC-5, AC-6, AC-8 | coding | task4, task6 |
| task8 | Record final evidence: `benchmark.csv` rows + `candidates.jsonl` entries (parent=`rope_v1`) with path labels, build engine, post-metrics, `verify.py` command, and reason-rejected notes; write the phase-3 steering verdict (did layout B beat layout A, which build-engine won); confirm no `../../AccelOpt/NKIBench/` edits | AC-8 | coding | task7 |
| task9 | (Optional) Cost-model cross-check of the 3-pass vector floor and the PE permutation-matmul cost via `kernel-cost-analysis`, for phase-3 context (measured digest remains the source of truth) | AC-4 | analyze | task1 |

## Claude-Codex Deliberation

### Agreements
- The primary lever is correctly tied to the measured phase-1 bottleneck: cut the vector
  `tensor_tensor` passes 6 → 3 while preserving the HBM read/write floor.
- AC-1/AC-1.1 split is sound: exact-preserving variants (D1b, D2) target rel-L2 = 0.0;
  PE-based variants (D1a/D1c) require the full 5-seed gate before promotion.
- AC-2/AC-4 are necessary: layout B only counts if it lowers *exposed* vector time without
  creating hidden HBM traffic or merely relocating the cost to DMA/PE — proven by the
  per-engine digest, not latency alone.
- AC-3 captures the packed RoPE algebra correctly, including sign-baking into `Aswap±` so
  the final combine is one `add` over all 128 partitions.
- D1b-first sequencing is right (least numerical risk before PE machinery); rejecting bf16
  and explicit ping-pong/burst-batching up front (D3/AC-7) is well-grounded.
- The evidence requirements (parented rows, per-engine digest, path labels, phase-3 verdict)
  are correct and sufficient for phase-3 steering.

### Resolved Disagreements
- **Iteration budget ambiguity (Codex DISAGREE/REQUIRED_CHANGE, accepted):** Codex objected
  that a strict "≤5 iters total" could starve required full-seed validation, threshold
  reruns, and the `W`-sweep. Resolution: the ≤5 cap now applies to **D1 design iterations
  across D1a/b/c only**; full-seed validation, near-threshold reruns, and the D2 `W`-sweep
  are explicitly separate and uncounted (AC-5, Goal).
- **HBM floor absolutism (Codex REQUIRED_CHANGE, accepted):** "≤ 402.65 MB" was too
  absolute for a noisy profiler. Resolution: AC-2 now defines floor-preserving as no
  *systematic* increase beyond noise, judged on the **read and write components separately**
  (≈ ±10%, matching phase-1 AC-2), and rejects repeatable extra traffic (e.g. from spill).
- **`W` transfer assumption (Codex REQUIRED_CHANGE, accepted):** applying layout A's best
  `W` to layout B was unsafe because layout B doubles partition/live-tile pressure.
  Resolution: AC-6 requires the best `W` to be **revalidated on the promoted layout**; the
  layout-A sweep is only an early baseline-refiner whose winner does not auto-transfer.
- **`W`-sweep should also go coarser (Codex TECHNICAL_GAP, accepted):** the sweep now spans
  `{512, 1024, 2048, 4096}` (not only finer), since the ~0.43 ms gap could be loop/fill-
  drain overhead that a coarser `W` reduces, guarded by an SBUF/compile precheck for 4096.
- **Promotion-by-p50 under-specified (Codex DISAGREE/REQUIRED_CHANGE, accepted):** near-
  threshold promotion from a single noisy run was under-guarded. Resolution: AC-8 adds a
  measurement protocol — stabilized p50, ≥2 re-profiles within ~3% of a threshold or a tie,
  optional min/p90/spread recording.
- **D1a is real matmul work, not a free copy (Codex CORE_RISK/TECHNICAL_GAP, accepted):** a
  dense 0/±1 permutation matmul contracts over 128 and streams `[128,W]`, so it is genuine
  PE work whose benefit hinges on hiding on idle PE. Folded into AC-4 and the Feasibility
  section; task1 confirms whether the compiler recognizes the 0/1 sparsity before over-
  investing in D1a.
- **PSUM/SBUF spill must be checked, not just HBM bytes (Codex MISSING_REQUIREMENT,
  accepted):** AC-2 now rejects candidates whose spills surface as extra HBM read/write,
  and task6 explicitly verifies no PSUM/SBUF spill to HBM on the PE path.
- **Reason-rejected + version/command provenance (Codex MISSING_REQUIREMENT, accepted):**
  AC-8 requires a "reason rejected" note per failed variant and recording the exact
  `verify.py` command with each candidate.
- **Scope invariants (Codex MISSING_REQUIREMENT, accepted):** the fixed instance
  (`S=262144`, `D=128`, fp32, identity transform, single trn2 core) is stated in the Path
  Boundaries as not-to-be-generalized in phase 2.
- **Correctness-question resolutions (Codex QUESTIONS_FOR_USER, resolved from the benchmark
  contract):** bitwise equality is NOT required (gate is rel-L2 < 2e-5); NaN/inf/signed-
  zero/subnormal are OUT OF SCOPE (NKIBench inputs are finite `np.random.normal`); each D1
  variant is itself a full correct-kernel probe, so no separate isolated-primitive remote
  microbenchmarks are needed. Folded into AC-1 and AC-5.

### Convergence Status
- Final Status: `converged`
- Rounds executed: 1 (the second-pass Codex review returned no high-impact `DISAGREE` and
  no blocking `REQUIRED_CHANGES` beyond the clarifications folded in above; one genuine
  policy question — the modest-PE-win-vs-complexity tradeoff — is carried to DEC-1).

## Pending User Decisions

- DEC-1: **Complexity/fragility tie-break when layout B wins only modestly.** If a
  PE-permutation layout-B variant (D1a/D1c) wins only modestly (e.g. ~0.82 ms) but carries
  fragile PE machinery and fp32-decomposition risk, while a simpler exact kernel (layout A,
  D1b, or D2-tuned) sits close behind (e.g. ~0.86 ms), which is promoted?
  - Claude Position: promote the lowest correct p50, but only when the win is **repeatable
    outside the profiler noise band** (per AC-8 reruns); if the PE win is within ~3–5% of a
    simpler exact candidate, prefer the simpler/more-robust kernel and record the fragile
    one as evidence. This keeps the loop honest without discarding a real, repeatable win.
  - Codex Position: matches Claude's position with one refinement — promote the fragile PE
    version only if it beats the simpler exact candidate by **> ~3–5% repeatably**;
    otherwise prefer the simpler exact kernel.
  - Tradeoff Summary: promoting the raw lowest p50 maximizes the recorded speedup but can
    enshrine a brittle kernel whose advantage is within noise and whose PE path may behave
    differently across compiler versions; preferring the simpler exact kernel on near-noise
    ties trades a sub-percent of speed for robustness and phase-3 maintainability. Claude
    and Codex agree on the *mechanism* (repeatable > ~3–5% margin); the open item is the
    exact threshold and whether the user wants raw-speed-max or robustness-preferred as the
    default. Recommendation: robustness-preferred with a ~3–5% repeatable-margin gate, as
    written in AC-8 and Milestone 4.
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Step", "Phase", "D1a/D1b/D1c", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `x0`, `x1`, `cos`,
  `sin`, `A`, `cos_stacked`, `x_swap_neg`, `out0`, `out1`, `tile_width`), matching the
  baseline, `rope_v1.py`, and silu-precedent style.

--- Original Design Draft Start ---

# rope_single_freq_apply — Phase 2 implementation draft (profile-driven optimization)

## Goal

Start from the promoted phase-1 kernel (`runs/rope_v1.py`, layout A, **0.9445 ms,
1.209x** over the 1.1418 ms baseline) and cut on-device latency **without ever
regressing correctness** (rel-L2 must stay `< 2e-5`, and today is exactly `0.0`
on all five seeds). Use the phase-1 profiler verdict — not a fresh guess — to pick
the lever, then explore each ranked direction for **at most five iterations**,
collecting before/after `verify.py` latency + the profiler engine digest to justify
keep / revise / reject.

## Phase-1 verdict recap (the source of truth — see `docs/phase1-bottleneck-digest.md`)

| metric | value | reading |
|--------|-------|---------|
| latency p50 | 0.9445 ms | current best (layout A, `W=2048`) |
| Vec active % | **91.6%** | six `tensor_tensor` passes on **64 of 128 lanes** |
| DMA active % | **93.5%** (sw-dma 99.6%) | co-limiter, but *not* bandwidth-bound |
| MBU % | 29.9% | HBM fabric bandwidth only 30% used |
| HBMrd / HBMwr / total | 268.44 / 134.22 / **402.65 MB** | **exactly the read-once/write-once floor** |
| eff BW | ~427 GB/s | « ~781 GB/s silu streaming roofline |

**Verdict: VECTOR-BOUND, co-limited with DMA-active, HBM at floor.** The pure-DMA
ceiling at 781 GB/s is `402.65 MB / 781 GB/s = 0.516 ms`; we measure 0.944 ms. The
**~0.43 ms gap is vector time that is not hidden under DMA** — the six vector passes
co-limit the wall clock. Two hard constraints fall out of this and gate every
direction below:

1. **HBM is already at the floor** (0% over). No traffic to remove; and any *added*
   HBM read (e.g. a second read of `cos`/`sin`, or reloading `x` twice) pushes above
   floor and costs DMA-active time on an already-93.5%-busy engine → forbidden.
2. **The prize is the ~0.43 ms of unhidden vector time.** The only way to shrink it
   is to do **fewer / wider vector passes**. The floor we are chasing is the
   ~0.516 ms DMA ceiling → best-case latency ~0.55–0.65 ms → **~1.75–2.0x** total.

## Why "fewer vector passes" means 128-partition packing (instruction-selection check)

`out0 = x0·cos − x1·sin`, `out1 = x0·sin + x1·cos` is **4 products + 2 combines = 6
`tensor_tensor` passes** in the 64-partition layout, and that is **minimal for 64
partitions**:

- No fused multiply-add collapses it. `cos`/`sin` are **full `[64, W]` tiles** (they
  vary along the free axis), *not* per-partition `[P,1]` scalars — so
  `nisa.tensor_scalar` (2 ops, but operands must be scalar/`[P,1]`) and
  `nisa.scalar_tensor_tensor` (`(data op0 scalar) op1 tile`) do **not** apply, and
  there is no `accumulate` flag on `tensor_tensor`. (Confirmed against
  `api-nki-isa-tensor.md` / `api-nki-isa-misc.md`.)
- `tensor_tensor` cost is **per free-element and independent of partition count** (128
  SIMD lanes run in parallel), so a `[128, W]` pass costs the same wall-clock as a
  `[64, W]` pass. Packing both output halves onto all 128 lanes therefore lets
  **3 passes do the work of 6**, roughly halving the co-limiting vector term.

This is the phase-1 digest's designated lever and the primary phase-2 direction.

### The packed-compute algebra (layout B)

`x_in` is **already** `[x0; x1]` on 128 partitions in HBM (`A`). Build:

- `A       = [x0; x1]`  (natural load, 1 DMA — no copy, cheaper than layout A's two loads)
- `Aswap±  = [−x1; +x0]` (swap the two 64-partition halves **and negate the top half**)
- `Ccos    = [cos; cos]` (broadcast `cos` 64→128 partitions)
- `Csin    = [sin; sin]` (broadcast `sin` 64→128 partitions)

Then **3 `tensor_tensor` over `[128, W]`**:

```
M1  = A     ⊙ Ccos      # [ x0·cos ;  x1·cos ]
M2  = Aswap± ⊙ Csin     # [-x1·sin ;  x0·sin ]
out = M1 + M2           # [ x0·cos − x1·sin ; x1·cos + x0·sin ]  ✓  == [out0; out1]
```

Baking the sign into `Aswap±` (rather than a `[−sin;+sin]` `Csin`) makes the final
combine a **single `add` across all 128 partitions** — a partition-dependent
add-top/sub-bottom is *not* one `tensor_tensor`, so the sign must live in exactly one
operand. Store `out` as one `[128, W]` DMA. Loads (x + cos + sin) and the store stay
**exactly at the HBM floor** — no extra HBM traffic.

**Exactness:** IEEE fp32 makes `a + (−b) ≡ a − b` and `(−x1)·sin ≡ −(x1·sin)`
bit-identically (negation is a sign-bit flip), so the packed order reproduces layout
A's arithmetic → rel-L2 should stay `0.0`. **Caveat to verify (see risks):** if the
broadcast/swap is built via a PE matmul, `nc_matmul` on fp32 may decompose internally
(bf16/tf32) and perturb the result — must re-check all 5 seeds.

### The crux: which engine builds `Aswap±`, `Ccos`, `Csin` — and does it hide?

These three are **cross-partition** data moves (replicate 64→128; swap halves). The
whole bet is that they land on an **idle** engine and hide under the DMA/vector floor.
Established fact (checked against the API + optimization knowledgebase): the only
engines that move data across partitions are **DMA**, **PE (matmul)**, and
**`nc_stream_shuffle`** — and:

- **`nc_stream_shuffle` is ruled out**: it runs on the **Vector Engine** (the engine
  we are trying to unload) and only shuffles **within 32-partition quadrants**, so it
  cannot express a 64↔64 half-swap (`api-nki-isa-misc.md`).
- **Vector/Scalar copies cannot cross partitions** — they are partition-locked. So a
  plain `nl.copy`/ScalarE activation can only *negate/scale in place*, not replicate
  or swap partitions.

That leaves two realizations to try (this is what the ≤5 iterations explore):

- **D1a — PE permutation matmul (idle engine, PE=0.2%).** `Ccos = [I;I]·cos`,
  `Csin = [I;I]·sin`, `Aswap± = S±·A` with `S± = [[0,−I],[I,0]]` are all 0/±1
  permutation matmuls on the **totally idle** Tensor engine → PSUM. Pro: uses dead
  silicon, so it can hide fully. Cons: (i) matmul output is PSUM and
  `tensor_tensor` forbids **both** operands in PSUM, so at most one packed operand per
  TT may be PSUM (needs careful SBUF/PSUM placement, maybe one extra copy);
  (ii) **fp32-matmul exactness risk** — must verify the gate still passes; (iii) three
  W-streaming matmuls are not literally free even when PE is idle.
- **D1b — SBUF→SBUF DMA broadcast/swap (no HBM traffic).** Replicate/swap partitions
  via SBUF-resident DMA (precedent: `TensorView.broadcast()` on the DMA path,
  `5f08e8cb`; GpSimd SBUF→SBUF `dma_engine.gpsimd_dma`, `dma-and-engines.md`). The
  negate (top-half sign) is exact on **ScalarE** (`activation(copy, scale=−1)`, idle).
  Pro: pure data movement → **arithmetically exact**, all-SBUF (no PSUM constraint).
  Con: adds **DMA-active** time on the already-93.5%-busy DMA engine — but MBU is only
  30%, so there is fabric-bandwidth headroom; the question the profiler answers is
  whether the added SBUF↔SBUF *active* time hides under the compute.
- **D1c — hybrid**: broadcast `Ccos`/`Csin` via DMA→SBUF (exact), build `Aswap±` via
  PE→PSUM, negate on ScalarE. Placement `M1 = A(SBUF)⊙Ccos(SBUF)`,
  `M2 = Aswap±(PSUM)⊙Csin(SBUF)` satisfies the "not both PSUM" rule with zero extra
  copies. Balances load across DMA + PE + Scalar.

## Ranked directions (benefit vs risk)

| # | direction | expected benefit | risk | iters |
|---|-----------|------------------|------|-------|
| **D1** | **Layout B: 128-partition packing (6→3 vector passes)** | **high** — toward the 0.516 ms DMA floor, up to ~1.75–2.0x | **high** — cross-partition build must hide on an idle engine; PE path has fp32-exactness + PSUM constraints | ≤5 (across D1a/b/c variants) |
| D2 | Finer free-axis `W` sweep on the best kernel | low-med — harvests DMA fill/drain bubble; effBW 427«781 hints at a pipelining gap | low — mask-free power-of-two `W`, no arithmetic change | ≤2 |
| D3 | Rejected-by-evidence (documented, ~0 iters) | — | — | 0–1 |

### D1 — Layout B (primary). Plan per iteration

1. **D1b first (lowest correctness risk).** Implement DMA/ScalarE broadcast+swap, all
   SBUF, keep `W=2048`. Score `--fast` (seed 42) then full 5-seed. Read the digest:
   did **Vec%** drop toward ~half and did **latency** fall below 0.80 ms?
2. **D1a / D1c** if D1b's DMA-active additions eat the win: move the builds onto the
   idle **PE** (and Scalar for negate). Verify the fp32 gate on the PE path *before*
   trusting latency.
3. Keep the variant with the lowest latency **that still passes all 5 seeds**.

**Keep / revise / reject rule for D1:**
- **Keep & promote** if latency `< 0.80 ms` (> 1.42x; clear win over 1.209x) with all
  seeds passing. Stretch success `< 0.65 ms` (~1.75x).
- **Revise** (try the next D1 variant) if `0.80–0.90 ms`: the packing helped but the
  build didn't fully hide — try moving the build to a more idle engine.
- **Reject, keep layout A** if `≥ 0.90 ms` after exhausting D1a/b/c, or if the gate
  fails on every exact-preserving variant. Record *why* (which engine the build landed
  on, Vec%/DMA%/PE% after) so phase 3 doesn't re-tread it.

### D2 — Finer free-axis `W` (secondary, cheap)

Sweep `W ∈ {1024, 512, 1536}` on whichever kernel D1 promotes (mask-free needs `W`
dividing `S = 2^18`; 1536 does not — restrict to powers of two `{1024, 512}` unless a
padded-tail variant is warranted). The silu campaign `[[kda-silu-progress]]` found
**finer wins** (optimum ~4 KB/partition burst) by amortizing the pipeline fill/drain
bubble. Here the digest calls this **secondary** — DMA is co-saturated in *active
time* at the floor, so the ceiling is small — but effBW 427 GB/s « 781 and MBU 30%
leave room for a bubble, and it is a near-zero-risk probe. **Keep** any `W` that
lowers latency with 5 seeds passing; else record the sweep and drop.

### D3 — Rejected by evidence (state, don't burn iterations)

- **bf16 / lower-precision downcast — REJECT.** RoPE is pure elementwise with **no
  reduction to average away rounding** (unlike `[[kda-rmsnorm-matmul-progress]]` where
  K-averaging rescued compensated-bf16). bf16 elementwise error ≈ 2⁻⁸ ≈ 4e-3 »» the
  2e-5 gate → fails. fp32 is mandatory.
- **Explicit ping-pong / wider burst-batching — REJECT (precedent).** The silu
  campaign found wider burst-batching + manual ping-pong **regressed**; `affine_range`
  already builds the software pipeline. At most one confirmatory probe if D1/D2 stall.

## Correctness guardrails (every candidate)

- rel-L2 stays `< 2e-5`; today it is `0.0` exact — treat any nonzero rel-L2 as a red
  flag and diff the arithmetic ordering. The **PE-matmul path (D1a)** is the one place
  packing can perturb fp32 → **always run the full 5-seed gate, not just `--fast`,
  before promoting a PE-path candidate.**
- Never raise HBM traffic above the 402.65 MB floor. Confirm `HBM_total_MB` in the
  digest is unchanged (broadcast/swap must be SBUF-resident or PE-permutation, never a
  re-read of `x`/`cos`/`sin` from HBM).
- Record every perf change in `benchmark.csv`, every candidate in `candidates.jsonl`
  (parent = `rope_v1`), and keep each direction's profiler digest under `profile/`.

## Measurement protocol

For each candidate, from `workspaces/rope_single_freq_apply/`:

```bash
# fast probe (seed 42) during iteration
python3 \
    ../../verify.py --op rope_single_freq_apply --candidate runs/<file>.py --fast
# full 5-seed + higher-iter latency before any promote
python3 \
    ../../verify.py --op rope_single_freq_apply --candidate runs/<file>.py
```

Read from the printed digest / `summary_metrics`: **latency p50, Vec%, DMA%, Scl%,
PE%, MBU%, HBMrd/HBMwr/total**. The decisive signal for D1 is **Vec% dropping toward
half while HBM_total stays at floor and PE%/DMA% absorb the build** — that is the
packing win materializing rather than just relocating.

## Risks / watch-items

- **The build may not hide (D1's core risk).** If the swap/broadcast lands back on the
  Vector engine (e.g. compiler routes a copy there) or serializes the DMA, we trade 3
  vector passes for 3 movement ops and net nothing. Mitigation: pin the build to
  PE/DMA/Scalar explicitly and read the per-engine digest, not just latency.
- **PSUM "not both operands" rule** forces careful SBUF/PSUM placement on the PE path
  (D1a/c); a mis-placement forces an extra copy that can erase the win.
- **fp32 PE-matmul may not be bit-exact** → gate risk on D1a; prefer the exact
  DMA/Scalar path (D1b) unless PE demonstrably wins on latency *and* passes 5 seeds.
- **Small absolute headroom.** The hard floor is 0.516 ms (2.21x max). Set
  expectations: a solid but sub-2x win is the realistic target; don't over-invest
  iterations past a clear plateau — bank the best correct kernel and move on.

## Deliverable

The best correct kernel promoted (with `benchmark.csv` / `candidates.jsonl` /
`profile/` evidence), plus a short verdict noting whether layout B beat layout A and
which build-engine realization won — steering phase-3 shape specialization.

--- Original Design Draft End ---
