# rope_single_freq_apply — Phase 1: First Correct NKI Kernel + Measured Bottleneck Digest

## Goal Description

Deliver the **first correct** NKI kernel for the single-frequency RoPE apply on
AWS Trainium (trn2, single core) and, in the same pass, record a **measured
profiler bottleneck digest** that decides the phase-2 direction.

The kernel must pass NKIBench's relative-L2 correctness gate
(`||v_k − v_r||_2 < 2e-5 · ||v_r||_2`, fp32) on **all five seeds `[0, 21, 42, 63, 84]`**
for the fixed problem instance `x_in (128, 262144)` fp32, `cos`/`sin` (64, 262144) fp32,
output `(128, 262144)` fp32. `transform_to_nki_inputs` is the identity, so the kernel
consumes the natural 2D tensors directly; the signature is
`@nki.jit def kernel(x_in, cos, sin)` returning one `shared_hbm` output of shape `(128, S)`.

The reference math (verified in numpy to rel-L2 = 0.0) is:
`x0 = x[:64]`, `x1 = x[64:]`, `out0 = x0*cos − x1*sin`, `out1 = x0*sin + x1*cos`,
`out = concat([out0, out1], axis=0)`.

This is **correctness-first**: a clean, fully understood kernel over speed. The default
implementation is **layout A** — a 64-partition, no-copy structure that loops over
free-axis (S) chunks of width `W = 2048` with `nl.affine_range`, does the six
`nisa.tensor_tensor` passes on base-0-aligned operands, and reads/writes HBM
once (the 402.65 MB traffic floor). The second, equally important deliverable is the
profiler digest (Vec/Scl/DMA %, HBM read/write bytes, effective bandwidth, latency)
interpreted into an explicit **vector-bound vs DMA/scheduling-bound verdict** that
sets the phase-2 lever (layout B 128-partition packing if vector-bound; finer `W`
if DMA/scheduling-bound). Layout B is **explicitly out of scope for phase 1**.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: **Correctness across all five seeds.** The kernel passes the NKIBench
  relative-L2 gate (`l2_norm_passed == true`) on every seed in `[0, 21, 42, 63, 84]`
  via `verify.py --op rope_single_freq_apply`, run at full (non-`--fast`) fidelity
  before the candidate is recorded as the phase-1 baseline.
  - Positive Tests (expected to PASS):
    - Full 5-seed `verify.py` run reports `l2_norm_passed` true for all of `[0,21,42,63,84]`.
    - The recorded rel-L2 values are well under `2e-5` (order 1e-6 or smaller, as this is exact fp32 arithmetic with no approximation), giving margin against seed-to-seed jitter.
  - Negative Tests (expected to FAIL):
    - A kernel that computes `out1 = x0*cos + x1*sin` (swapped coefficients) fails the gate on at least one seed.
    - A kernel that writes `out1` to output rows `0:64` instead of `64:128` (half-swap) fails the gate.
    - A kernel that passes only the single `--fast` seed 42 but is not run through the full 5-seed gate is NOT eligible to be recorded as the phase-1 baseline.

- AC-2: **HBM traffic at the read-once/write-once floor.** The kernel issues 4 loads
  (`x0, x1, cos, sin`) + 2 stores (`out0, out1`) per tile with no redundant reads;
  measured HBM traffic sits at the theoretical payload floor within normal profiler
  variation. Theoretical payload: `HBMrd ≈ 268.44 MB` (x 128 MB + cos 64 MB + sin 64 MB),
  `HBMwr ≈ 134.22 MB` (out 128 MB), total `≈ 402.65 MB`. (MB = 2^20 bytes; 128·262144·4 B = 128 MiB.)
  - Positive Tests (expected to PASS):
    - Measured `HBMrd_MB` + `HBMwr_MB` from `summary_metrics` is within a small tolerance (≈ ±10%) of 402.65 MB, and the read/write split matches the ~268/~134 MB expectation.
    - Each HBM row of `x_in`, `cos`, `sin` is read exactly once per full pass; the output is written exactly once.
  - Negative Tests (expected to FAIL):
    - A kernel whose measured HBM traffic is materially above ~402.65 MB (e.g. re-loading `cos`/`sin` per output half, ~2× reads) is flagged and investigated before recording.
    - A kernel that loads all 128 partitions of `x` and then also separately re-loads either half fails the read-once check.

- AC-3: **Layout, op count, and mask-free tiling.** The compute is exactly six
  `nisa.tensor_tensor` ops (4 multiplies + one subtract + one add); both operands of
  every `tensor_tensor` share the same SBUF base partition (base-0 aligned); and the
  free-axis tile width `W` is a power of two dividing `S = 2^18` so tiles are
  rectangular and mask-free with no tail handling.
  - Positive Tests (expected to PASS):
    - The kernel source contains exactly six `nisa.tensor_tensor` calls realizing `x0*cos`, `x1*sin`, `out0 = (x0*cos)−(x1*sin)`, `x0*sin`, `x1*cos`, `out1 = (x1*cos)+(x0*sin)`.
    - `W` divides `S` exactly (e.g. `W ∈ {1024, 2048, 4096}`, `S/W` integer), so no masking is emitted and there is no tail iteration.
  - Negative Tests (expected to FAIL):
    - A layout that feeds `tensor_tensor` two operands at different partition bases (e.g. `x0` at base 0 and `x1` at base 64) without realignment is rejected by the compiler / produces wrong results.
    - A `W` that does not divide `S` (requiring a masked tail) is out of scope for phase 1 and not used.

- AC-4: **No-copy path verified, or documented fallback taken.** The default layout-A
  no-copy form (explicit predeclared `nl.ndarray((par_dim(64), W))` tiles; load
  `x_in[64+p, col]` into `x1`'s base-0 tile; partition-offset store to `out[64:128, col]`
  via explicit `nl.arange` indexing) is proven to compile and be correct on the fast
  seed-42 screen **before** the full run. If it fails to compile or is incorrect, the
  kernel falls back to the baseline-style 128-partition `x` load + **SBUF-only**
  `nl.copy` realignment of `x1` to base 0 (still six `tensor_tensor`, still read-once/
  write-once — the copy introduces no extra HBM reads). The candidate notes state
  which path was taken.
  - Positive Tests (expected to PASS):
    - `--fast` (seed 42) screen: the no-copy form compiles under `--disable-dge --logical-nc-config=1` and passes rel-L2; evidence records path label `layout_a_no_copy`.
    - If no-copy fails, the copy-realign fallback compiles and passes the same screen; evidence records path label `layout_a_copy_realign_fallback` and the fallback preserves the AC-2 floor (SBUF-only copy, no extra HBM reads).
  - Negative Tests (expected to FAIL):
    - Recording the candidate without stating which path (`layout_a_no_copy` vs `layout_a_copy_realign_fallback`) was used is incomplete evidence.
    - A fallback that realigns `x1` by issuing a second HBM load (instead of an SBUF copy) violates AC-2 and is rejected.

- AC-5: **Profiler digest recorded with an explicit phase-2 verdict.** For the promoted
  candidate, the `summary_metrics` digest (`Vec_pct`, `Scl_pct`, `DMA_pct`,
  `HBMrd_MB`, `HBMwr_MB`, plus latency and — where available — effective bandwidth) is
  recorded, alongside the theoretical bytes for anchoring, and interpreted into an
  explicit verdict: **vector-bound** (high `Vec_pct` with near-floor HBM traffic →
  phase-2 lever is layout B) vs **DMA/scheduling-bound** (`DMA_pct`-dominated or poor
  effective bandwidth despite floor traffic → phase-2 lever is finer `W` / scheduling).
  - Positive Tests (expected to PASS):
    - `candidates.jsonl` entry contains a `metrics` object with `Vec_pct`, `Scl_pct`, `DMA_pct`, `HBMrd_MB`, `HBMwr_MB`, and the notes state the bound verdict and the resulting phase-2 lever.
    - The verdict is derived from measured numbers read together (percentages + latency + bytes + bandwidth), not from a single percentage in isolation and not from the theoretical vector cost model (which is known to over-predict for this op).
  - Negative Tests (expected to FAIL):
    - Recording only pass/fail and latency, with no per-engine digest, fails this criterion.
    - Declaring "vector-bound" or "DMA-bound" without citing the measured metrics that support it fails this criterion.

- AC-6: **Evidence integrity.** The candidate is recorded with a `benchmark.csv` row and
  a `candidates.jsonl` entry whose `parent` links to the NKIBench baseline
  (`rope_single_freq_apply_B1_H64_N4096_D128_0.py`), following the existing silu-workspace
  schema. The candidate `.py` source lives under `runs/`; the NKIBench baseline and
  reference are never edited. Per-seed correctness and rel-L2 values are recorded, not
  just the aggregate pass flag.
  - Positive Tests (expected to PASS):
    - After scoring, `benchmark.csv` gains one row (`timestamp,op,candidate,parent,passed,latency_ms,speedup,notes`) and `candidates.jsonl` gains one JSON line matching the silu schema (`id, op, candidate, parent, passed, verdict, seeds, latency_ms, baseline_latency_ms, speedup, structure, metrics, notes`).
    - `git status` shows the new kernel under `workspaces/rope_single_freq_apply/runs/` and no modifications under `../AccelOpt/NKIBench/`.
  - Negative Tests (expected to FAIL):
    - Any edit to `../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}` fails the workspace contract.
    - A candidates.jsonl entry with no `parent` link (broken DAG) fails this criterion.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A layout-A, 64-partition, no-copy kernel with `W = 2048`: four explicit base-0 input
tiles (`x0, x1, cos, sin`), six `nisa.tensor_tensor` passes with early store of `out0`
and temp-buffer reuse to keep live SBUF comfortable, partition-offset store of `out1`
to rows `64:128`, all inside one `nl.affine_range(S // W)` loop. It reads/writes HBM
exactly once (402.65 MB floor), passes the full 5-seed rel-L2 gate, and is recorded
with a complete profiler digest and an explicit phase-2 bottleneck verdict. Optionally
cross-checked against the theoretical per-engine cost model for context (not as a gate).

### Lower Bound (Minimum Acceptable Scope)
Any correct NKI kernel that passes the full 5-seed rel-L2 gate for this instance —
including the baseline-style 128-partition `x` load + SBUF-only `nl.copy` realign
(six `tensor_tensor`, read-once/write-once) — recorded with a `benchmark.csv` row, a
`candidates.jsonl` entry parented to the baseline, and a profiler digest with a bound
verdict. Correctness plus the recorded measured digest is the floor; the no-copy
refinement is a quality target, not a gate.

### Allowed Choices
- Can use: `nl.load` / `nl.store` with explicit predeclared `nl.ndarray((par_dim(P), W))`
  SBUF tiles and `nl.arange` indexing (the established repo convention — baseline and
  silu precedent both use it); `nisa.tensor_tensor` with `nl.multiply` / `nl.add` /
  `nl.subtract`; `nl.affine_range` for the free-axis loop; free-axis tile width
  `W ∈ {1024, 2048, 4096}` (default 2048), each a power of two dividing `S`; an
  SBUF-only `nl.copy` realign as the documented fallback if no-copy slicing fails to
  compile; a compile+fast-seed screen as the no-copy semantic probe.
- Must verify before use: `W = 4096` requires an SBUF/compile headroom check before
  adoption (it is the riskiest against live-tile pressure once temp lifetime and any
  pipeline overlap are counted); prefer `W = 2048` unless a check clears 4096.
- Cannot use: implementing the layout-B 128-partition packing (deferred to phase 2);
  any dtype other than fp32; masked/tail tiles (choose `W` that divides `S`); redundant
  HBM reads (e.g. re-loading `cos`/`sin` per output half, or a second HBM load to
  realign `x1`); editing the NKIBench baseline, reference, seeds, or `summary.json`;
  hand-tuning a baseline.

> **Note on Deterministic Designs**: The draft specifies a highly deterministic design
> (fixed op sequence, fixed layout-A structure, fixed correctness gate). The path
> boundaries are correspondingly narrow: the upper and lower bounds differ only in
> whether the no-copy refinement lands vs the copy-realign fallback, and in digest
> completeness. Everything else (math, dtype, floor traffic, evidence schema) is fixed
> per the draft and the NKIBench contract.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Layout A (64-partition, no-copy) — one possible realization:

```
P_half = 64
W      = 2048                 # power of two dividing S = 2^18; S/W = 128 iters, mask-free
out = nl.ndarray((128, S), dtype=fp32, buffer=nl.shared_hbm)

for j in nl.affine_range(S // W):            # independent iterations -> compiler pipelines DMA vs compute
    base = j * W
    # Predeclare base-0 tiles explicitly (do NOT rely on slice-shorthand for partition
    # normalization). Load the upper half of x into its OWN base-0 [64, W] tile.
    x0 = nl.ndarray((par_dim(P_half), W), dtype=fp32, buffer=nl.sbuf)   # x_in rows  0:64
    x1 = nl.ndarray((par_dim(P_half), W), dtype=fp32, buffer=nl.sbuf)   # x_in rows 64:128 -> lands at base 0
    c  = nl.ndarray((par_dim(P_half), W), dtype=fp32, buffer=nl.sbuf)
    s  = nl.ndarray((par_dim(P_half), W), dtype=fp32, buffer=nl.sbuf)
    x0[...] = nl.load(x_in[nl.arange(P_half)[:,None],        base + nl.arange(W)[None,:]])
    x1[...] = nl.load(x_in[P_half + nl.arange(P_half)[:,None], base + nl.arange(W)[None,:]])
    c[...]  = nl.load(cos[nl.arange(P_half)[:,None],        base + nl.arange(W)[None,:]])
    s[...]  = nl.load(sin[nl.arange(P_half)[:,None],        base + nl.arange(W)[None,:]])

    e_cos = nisa.tensor_tensor(x0, c, nl.multiply)     # x0*cos
    o_sin = nisa.tensor_tensor(x1, s, nl.multiply)     # x1*sin
    out0  = nisa.tensor_tensor(e_cos, o_sin, nl.subtract)
    nl.store(out[nl.arange(P_half)[:,None], base + nl.arange(W)[None,:]], out0)  # store early, free temps
    e_sin = nisa.tensor_tensor(x0, s, nl.multiply)     # x0*sin
    o_cos = nisa.tensor_tensor(x1, c, nl.multiply)     # x1*cos
    out1  = nisa.tensor_tensor(o_cos, e_sin, nl.add)
    nl.store(out[P_half + nl.arange(P_half)[:,None], base + nl.arange(W)[None,:]], out1)  # rows 64:128
```

Sequencing within the loop: computing/storing `out0` before `out1` lets the two
multiply temps (`e_cos`, `o_sin`) be freed/reused before the `out1` temps, keeping the
live tile count (and thus SBUF pressure) down — useful headroom for any compiler
pipeline overlap.

The no-copy claim (`nl.load(x_in[64:128, ...])` landing at SBUF partition base 0, and a
partition-offset store to `out[64:128, ...]`) is treated as an **empirical hypothesis
proven by a fast compile+seed-42 screen**, not an assumption. If it does not compile,
fall back to a 128-partition `x` load and an SBUF-only `nl.copy` of the lower half to
base 0 (exactly the baseline's realign), which keeps read-once/write-once.

### Relevant References
- `../../AccelOpt/NKIBench/reference/rope_single_freq_apply_B1_H64_N4096_D128_numpy_1.py` — numpy oracle; `transform_to_nki_inputs` is identity; defines the exact op sequence and shapes.
- `../../AccelOpt/NKIBench/kernels/rope_single_freq_apply_B1_H64_N4096_D128_0.py` — baseline kernel (1.1418 ms): 128-partition `x` load, `nl.copy` realign of the lower half, six `nisa.tensor_tensor`; the copy is exactly what layout A removes; also the fallback template.
- `workspaces/silu/runs/silu_v1.py` — precedent for the `nl.affine_range` + explicit `nl.ndarray((par_dim(P), F))` + `nl.arange` tile-and-stream idiom used here.
- `workspaces/silu/benchmark.csv` and `workspaces/silu/candidates.jsonl` — the exact evidence-row / JSON schema to mirror (columns and keys).
- `../../verify.py` — scores a candidate; gates on `l2_norm_passed`; prints the profiler digest / `summary_metrics`. Run with `python3 ../../verify.py --op rope_single_freq_apply --candidate runs/<file>.py [--fast]`.
- `../../baselines.json` — cached baseline entry (1.1418 ms) for `rope_single_freq_apply[case=1]`.
- Memory `[[kda-silu-progress]]` — silu precedent: finer free-axis tiling won when DMA-bound (optimum ~4 KB/partition burst); wider bursts / ping-pong regressed. Informs the phase-2 finer-`W` lever if this op turns out DMA-bound.
- Memory `[[kda-rope-progress]]` — rope-specific notes: cross-half, six vector passes, identity transform, layout A vs layout B framing.

## Dependencies and Sequence

### Milestones
1. **Correct layout-A kernel (no-copy default).**
   - Phase A: Write the layout-A kernel (`runs/rope_v1.py`) — explicit base-0 tiles,
     six `tensor_tensor`, `W = 2048`, `nl.affine_range(S // W)`, partition-offset store.
   - Phase B: Fast (`--fast`, seed 42) compile+correctness screen of the no-copy form.
     If it fails, switch to the SBUF-only copy-realign fallback and re-screen.
2. **Full correctness gate.** (Depends on Milestone 1.)
   - Step 1: Run the full 5-seed `verify.py` and confirm `l2_norm_passed` on all of
     `[0, 21, 42, 63, 84]`; capture per-seed rel-L2 values.
3. **Measured bottleneck digest + phase-2 verdict.** (Depends on Milestone 2.)
   - Step 1: Read `summary_metrics` (Vec/Scl/DMA %, HBMrd/HBMwr, latency, effective BW).
   - Step 2: Confirm HBM traffic sits at the ~402.65 MB floor (AC-2).
   - Step 3: Derive and record the vector-bound vs DMA/scheduling-bound verdict and the
     resulting phase-2 lever.
4. **Evidence recording.** (Depends on Milestones 2–3.)
   - Step 1: Append the `benchmark.csv` row and the `candidates.jsonl` entry (parent =
     baseline), including the path label, metrics, and verdict.
   - Step 2 (optional): Cross-check against the theoretical per-engine cost floor for
     context.

Dependency summary: kernel source → fast screen (→ fallback if needed) → full 5-seed
gate → profiler digest/verdict → evidence rows. The optional cost cross-check depends
only on the kernel source and can run alongside the digest interpretation.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Write layout-A no-copy kernel `runs/rope_v1.py`: explicit base-0 `par_dim(64)` tiles for `x0/x1/cos/sin`, six `nisa.tensor_tensor` (early `out0` store + temp reuse), partition-offset store of `out1` to rows 64:128, `nl.affine_range(S//W)`, `W=2048`; assert shapes/dtype | AC-3, AC-2 | coding | - |
| task2 | Fast `--fast` (seed 42) screen: prove the no-copy partition-offset load/store form compiles under `--disable-dge --logical-nc-config=1` and is correct; record path label `layout_a_no_copy` | AC-4 | coding | task1 |
| task3 | If task2 fails, implement the SBUF-only `nl.copy` realign fallback (128-partition `x` load + copy lower half to base 0; no extra HBM reads) and re-screen; record label `layout_a_copy_realign_fallback` | AC-4, AC-2 | coding | task2 |
| task4 | Run the full 5-seed `verify.py` (no `--fast`); confirm `l2_norm_passed` on `[0,21,42,63,84]`; capture per-seed rel-L2 | AC-1 | coding | task2 |
| task5 | Read `summary_metrics`; confirm HBM traffic at ~402.65 MB floor; derive vector-bound vs DMA/scheduling-bound verdict + phase-2 lever | AC-2, AC-5 | coding | task4 |
| task6 | Record evidence: `benchmark.csv` row + `candidates.jsonl` entry (parent = baseline), with path label, metrics digest, and verdict; verify no edits under `../AccelOpt/NKIBench/` | AC-5, AC-6 | coding | task5 |
| task7 | (Optional) Cross-check the six-`tensor_tensor` layout-A cost against the theoretical per-engine floor via `kernel-cost-analysis` for phase-2 context | AC-5 | analyze | task1 |

## Claude-Codex Deliberation

### Agreements
- Phase 1 stays scoped to layout A (first correct no-copy kernel) plus a measured
  profiler digest; layout B is deferred to phase 2, conditioned on the measured verdict.
- The no-copy partition-offset load/store form is an empirical risk best handled by an
  explicit fast compile+seed-42 screen, with a documented SBUF-only copy-realign fallback.
- SBUF tiles should be **predeclared explicitly** (`nl.ndarray((par_dim(64), W))` + load
  into them) rather than relying on slice-shorthand for partition-base normalization.
- The theoretical vector cost model over-predicts for this op (it already exceeds the
  measured baseline that does exactly these six ops), so the measured `summary_metrics`
  digest — read as a whole — is the source of truth for the phase-2 verdict, not the
  cost model and not any single percentage.
- `W = 2048` is a comfortable phase-1 default; `W = 4096` needs an SBUF/compile check;
  `W` must divide `S` (power of two) to stay mask-free.
- Evidence must record per-seed correctness and rel-L2 values (not just the pass flag),
  the path label, the full metrics digest, and the theoretical bytes for anchoring.

### Resolved Disagreements
- **AC-2 exactness (Codex DISAGREE, accepted):** Codex objected to requiring measured
  HBM traffic to equal *exactly* 402.65 MB, since profiler counters carry rounding /
  metadata / tool-specific accounting. Resolution: AC-2 now states the theoretical
  payload floor (HBMrd ≈ 268.44 MB, HBMwr ≈ 134.22 MB, total ≈ 402.65 MB) as an expected
  target **with tolerance (≈ ±10%)** and a read/write-split sanity check, investigating
  only if materially above floor.
- **Fallback must preserve the floor (Codex REQUIRED_CHANGE, accepted):** the
  `nl.copy` realign fallback is constrained to be **SBUF-only** and must not introduce a
  second HBM read of `x1` — encoded in AC-4 and the Path Boundaries.
- **W=4096 risk (Codex REQUIRED_CHANGE, accepted):** an SBUF/compile headroom check is
  now required before adopting `W = 4096`; default remains `W = 2048`.
- **Explicit phase-2 verdict threshold (Codex OPTIONAL, accepted):** AC-5 now spells out
  the decision rule — high `Vec_pct` with near-floor HBM traffic → layout B; `DMA_pct`-
  dominated or poor effective bandwidth despite floor traffic → finer `W` / scheduling.
- **Evidence path label + theoretical bytes (Codex OPTIONAL, accepted):** the candidate
  records `layout_a_no_copy` vs `layout_a_copy_realign_fallback` and the theoretical
  bytes next to the measured bytes (AC-4, AC-5).
- **`affine_range` semantics (Codex CORE_RISK, accepted):** the plan no longer treats
  the loop keyword as a guarantee of pipelining; iterations are independent, but
  pipelining is verified from profiler behavior, not assumed. Wording adjusted in the
  approach and Milestone 3.
- **Loop-construct / API style (Codex QUESTION, resolved by convention):** use the
  repo's `nl.load`/`nl.store` + explicit `par_dim` tile idiom (baseline + silu precedent),
  not `nisa.dma_copy`; recorded in Allowed Choices.
- **Separate semantic-probe candidate (Codex QUESTION, resolved):** the `--fast` seed-42
  screen serves as the semantic probe; no separately recorded probe candidate is needed.

### Convergence Status
- Final Status: `converged`
- Rounds executed: 1 (second-pass Codex review returned no high-impact `DISAGREE` and no
  blocking `REQUIRED_CHANGES` beyond the clarifications folded in above).

## Pending User Decisions

- DEC-1: Phase-1 speed policy — is a correct-but-slower kernel acceptable as the phase-1
  measurement baseline, or must the phase-1 candidate avoid a clear regression versus the
  cached 1.1418 ms baseline?
  - Claude Position: Accept any correct kernel that passes the 5-seed gate as the phase-1
    measurement baseline; phase 1 is explicitly correctness-and-measurement-first (the
    draft sets no hard speedup target). The layout-A no-copy form should already meet or
    beat the baseline by removing its `nl.copy`, but a regression should not block phase 1
    from producing the steering digest.
  - Codex Position: N/A — open question raised in Codex's `QUESTIONS_FOR_USER`; no opposing
    technical position.
  - Tradeoff Summary: Treating "no regression vs baseline" as a hard phase-1 gate risks
    blocking the measurement deliverable if the no-copy form happens to tie/regress;
    treating phase 1 as purely correctness+measurement keeps the loop moving and defers
    all speed work to phase 2, at the cost of possibly recording a non-improving phase-1
    candidate. Recommendation: keep it a soft target (record the digest regardless; note
    any regression), which is how AC-1/AC-5 are currently written.
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `x0`, `x1`, `cos`,
  `sin`, `out0`, `out1`, `tile_width`), matching the baseline and silu-precedent style.

--- Original Design Draft Start ---

# rope_single_freq_apply (D128, B*H*N=262144, fp32) — Phase 1 implementation draft

## Goal

Produce the **first correct** NKI kernel for the single-frequency RoPE apply,
passing NKIBench's relative-L2 gate (`||v_k - v_r||_2 < 2e-5 * ||v_r||_2`, fp32)
on all five seeds `[0, 21, 42, 63, 84]`. Correctness-first: a clean, fully
understood kernel over speed. But pick a loop/compute structure that is already
reasonable (minimal DMA, no redundant copies) and, crucially, **use phase 1 to
measure the real bottleneck** — because unlike the silu case this op is *not*
obviously HBM-bound (see the hardware grounding below).

Baseline latency (`rope_single_freq_apply_B1_H64_N4096_D128_0.py`) = **1.1418 ms**
(cached in `baselines.json`), measured through the same profiler path.

## What the operator is

Rotary position embedding, applied elementwise with a **cross-half interaction**.
Split `x` into two halves over the head dim `D=128` (`half=64`):

```
x0 = x[:64, :]        # lower half of D
x1 = x[64:, :]        # upper half of D
out0 = x0 * cos - x1 * sin      # -> output rows   0:64
out1 = x0 * sin + x1 * cos      # -> output rows  64:128
out  = concat([out0, out1], axis=0)   # (128, S)
```

Each output element depends on the **co-located column** of both halves plus the
co-located `cos`/`sin`. There is no reduction and no matmul — but it is *not* a
pure single-input elementwise map like silu: every output column mixes `x0` and
`x1`, so the kernel does **four tensor×tensor products + two tensor±tensor
combines = six vector passes** per element. That distinction drives the whole
bottleneck story below.

## Tiled layout (verified in numpy)

`transform_to_nki_inputs(inputs)` returns `inputs` unchanged — **it is the
identity**. So, unlike silu (which reshapes to a 3D `(128,32,7168)` tiled view),
the kernel here consumes the *natural* 2D tensors directly:

- `x_in`: `(128, 262144)` = `(D, B*H*N)` fp32 — partition axis = `D` (=128), free axis = `S` (=262144)
- `cos`, `sin`: `(64, 262144)` = `(D/2, S)` fp32 — partition axis = 64, free axis = `S`
- output: `(128, 262144)` fp32, same layout as `x_in`

`transform_nki_outputs(k_res, ref)` just wraps `(k_res,)`. So the kernel signature
matches the baseline exactly:

```python
@nki.jit
def kernel(x_in, cos, sin):   # returns one shared_hbm out of shape (128, S)
```

**Verified in numpy** (seed 42, the adapter's fixed input seed): `transform_to_nki_inputs`
is identity, and the op sequence above reproduces `forward(...)` with **rel-L2 =
0.0 (exact match)**. So the math is settled before we write a line of NKI; the
only remaining content is (a) the partition layout, (b) the free-axis tiling, and
(c) keeping loads/stores/compute lean.

### SBUF budget forces a free-axis (S) tile

Per-partition data for the whole `S=262144` free axis is `262144 * 4 B = 1 MB`,
far over the ~208 KB usable SBUF/partition (trn2). So we tile the **free axis**
`S` into chunks of width `W` and loop. `S = 2^18`, so any power-of-two `W`
divides it exactly → **mask-free, rectangular tiles, no tail handling**. Live
tiles per iteration are ~8 `[64, W]` fp32 buffers (`x0, x1, c, s` + products/
combines); budget:

| `W`  | iters `S/W` | 8 tiles `[64,W]` | % usable | fits |
|------|-------------|------------------|----------|------|
| 2048 | 128         | 64 KB/part       | 30%      | yes  |
| 4096 | 64          | 128 KB/part      | 54%      | yes (room for a phase-2 double buffer) |
| 8192 | 32          | 256 KB/part      | 108%     | no   |

Phase-1 default: **`W = 2048`** (128 pipeline iterations, comfortable headroom).
This is a *starting point*, not a tuned value — free-axis tile width is an
explicit phase-2/3 lever (the silu campaign found **finer wins**, optimum ~4 KB/
partition ≈ `W=1024` for fp32; see `[[kda-silu-progress]]`).

## Hardware grounding: cost model + bottleneck (trn2, single core)

**HBM traffic floor (hard lower bound — bytes that must move):**

```
read  : x 128 MB + cos 64 MB + sin 64 MB = 256 MB
write : out 128 MB
total : 402.65 MB
```

At ~800 GB/s effective HBM BW (measured on this profiler for the silu streaming
kernel), the HBM floor = **0.503 ms** → a **2.27× ceiling** vs the 1.1418 ms
baseline *if* we ever became fully DMA-bound. This is the one number I trust as a
hard floor.

**Vector-engine cost (the reason this op differs from silu):** six
`tensor_tensor` passes, each over `S = 262144` free elements/partition. Per the
cost model (Formula A/B, trn2 Vector @ 0.96 GHz):

- both operands in SBUF → 2 cyc/elem → `6 * 2 * 262144 * 100/96` = **3.28 ms**
- optimistic 1 cyc/elem → **1.64 ms**

Both estimates are **at or above the measured 1.1418 ms baseline** — which does
exactly these six `tensor_tensor` ops. So the theoretical vector model
*over-predicts* the real device here (real trn2 fp32 vector throughput evidently
beats the naive 2-cyc formula). **Takeaway: the cost model is not a reliable
floor for this op; the profiler's measured `Vec%` vs `DMA%` digest is the source
of truth.** Two consequences:

1. Phase 1 must **read the profiler `summary_metrics`** (Vec / Scl / DMA % +
   HBMrd/HBMwr) on the first correct kernel to learn whether we are vector-bound
   or DMA/scheduling-bound. That verdict, not an assumption, sets phase-2's
   direction. (Silu was ~97% DMA on a single cheap Scalar op; rope with 6 vector
   passes may sit much higher on `Vec%`.)
2. If the profiler shows we are **vector-bound**, the dominant lever is the
   **packed 128-partition layout** described next; if **DMA/scheduling-bound**,
   the lever is finer free-axis tiling (silu precedent).

**Instruction selection note:** the products are genuine tensor×tensor
(`x0 * cos`), so `tensor_scalar` / `scalar_tensor_tensor` (the cheaper 1-cyc
paths) do **not** apply — one operand would have to be a scalar. No fused
"a*b − c*d" primitive exists. So six `tensor_tensor` is the minimum for the
64-partition layout; the only way to cut vector work is to pack onto 128
partitions (halving the op count), below.

## Phase-1 kernel structure (layout A — 64-partition, no copy)

Loop over free-axis chunks; each iteration is fully independent → `nl.affine_range`
so the compiler software-pipelines DMA against compute (the silu lesson: let
`affine_range` build one deep pipeline).

```
for j in nl.affine_range(S // W):          # 128 independent iterations
    cols = j*W : (j+1)*W
    x0 = load x_in[0:64,   cols]           # [64, W] -> SBUF partition 0
    x1 = load x_in[64:128, cols]           # [64, W] -> SBUF partition 0 (fresh tile!)
    c  = load cos[:, cols]                 # [64, W]
    s  = load sin[:, cols]                 # [64, W]

    e_cos = tensor_tensor(x0, c, multiply)     # x0*cos
    o_sin = tensor_tensor(x1, s, multiply)     # x1*sin
    out0  = tensor_tensor(e_cos, o_sin, subtract)   # x0*cos - x1*sin
    e_sin = tensor_tensor(x0, s, multiply)     # x0*sin
    o_cos = tensor_tensor(x1, c, multiply)     # x1*cos
    out1  = tensor_tensor(o_cos, e_sin, add)        # x1*cos + x0*sin

    store out[0:64,   cols] = out0
    store out[64:128, cols] = out1
```

**Why this is cleaner than the baseline (which needs an `nl.copy`):** the baseline
loads all of `x_in` as a 128-partition tile and slices `x1` from partitions
`64:128`, so its lower operand sits at partition base 64 and must be `nl.copy`-ed
to base 0 before `tensor_tensor` (which requires both operands at the same base
partition). By instead **loading `x1 = x_in[64:128, cols]` into its own fresh
`[64,W]` tile**, the destination lands at partition 0 (confirmed via the NKI docs:
`nl.load` returns a new SBUF tile based at partition 0 regardless of the HBM row
offset). So `x0, x1, c, s` are all base-0 and aligned — **six `tensor_tensor`
with zero copies**. Storing `out1` back to HBM rows `64:128` is the mirror image
and is a supported partition-offset store.

- 4 loads + 2 stores per tile → total HBM traffic = the 402.65 MB floor exactly
  (read-once/write-once); no redundant reads.
- Uses only 64 of 128 vector lanes. This does **not** waste vector wall-clock in
  layout A (op latency depends on free-dim size, not partition count), but it *is*
  the slack that layout B exploits.

## Phase-2/3 levers to preview (set up, not implement now)

- **Layout B — packed 128-partition compute (the big vector lever).** Because
  vector op latency is per-free-element and *independent of partition count*,
  packing both output halves onto all 128 partitions lets 3 ops do the work of 6:
  `t1 = x * cos_stacked` (`[128,W]`), `t2 = x_swap * sin_stacked` (`[128,W]`),
  `out = t1 ± t2` — where `x_swap` is `x` with its two halves swapped and the
  lower half negated, and `cos_stacked`/`sin_stacked` are `cos`/`sin` broadcast to
  128 partitions. This **halves vector time** but adds a cross-partition
  swap/negate and either a partition-broadcast copy or a second `cos`/`sin` DMA
  (which would raise HBM traffic). Whether it wins depends entirely on the phase-1
  `Vec%` vs `DMA%` verdict — hence measuring first.
- **Finer free-axis tiling** (`W` → 1024 or below). The silu campaign showed finer
  chunks amortize the pipeline fill/drain bubble when DMA-bound (optimum ~4 KB/
  partition burst). Cheap to sweep once we know the bottleneck.
- **In-place / buffer reuse** to shrink live SBUF and enable deeper double
  buffering (only if scheduling-bound).

## Correctness plan

1. Numpy oracle (already done): identity transform + op sequence → rel-L2 0.0.
2. First candidate = layout A above, `W=2048`. Score with
   `verify.py --op rope_single_freq_apply --candidate runs/<file>.py --fast`
   (seed 42), then the full 5-seed run before recording as the phase-1 baseline.
3. Record the profiler digest (Vec/Scl/DMA %, HBMrd/HBMwr) in `benchmark.csv` and
   `candidates.jsonl` — this is the phase-1 deliverable that steers phase 2.

## Risks / watch-items

- **Partition-offset load/store semantics** — the whole no-copy design rests on
  `nl.load(x_in[64:128, :])` landing at partition 0 and `nl.store(out[64:128,:])`
  writing rows 64:128. Confirmed in the NKI docs; if the compiler rejects it,
  fall back to the baseline's 128-partition-load + `nl.copy` realign (still 6
  `tensor_tensor`, one extra copy).
- **`tensor_tensor` operand memory spaces** — both operands in SBUF is the 2-cyc
  path; that's inherent to a two-tensor multiply and matches the baseline.
- **Bottleneck surprise** — the explicit hypothesis is that rope may be
  *vector-bound* (6 passes) where silu was DMA-bound (1 pass). Phase 1 exists to
  confirm/refute this before committing phase-2 effort to the wrong lever.

--- Original Design Draft End ---
