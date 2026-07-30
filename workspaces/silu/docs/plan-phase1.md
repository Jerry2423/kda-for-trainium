# silu (M4096 N7168, fp32) — Phase 1: First Correct HBM-Bound Kernel

## Goal Description

Produce the **first correct** NKI kernel for elementwise SiLU / swish
`y = x * sigmoid(x) = x / (1 + exp(-x))` over an fp32 `(4096, 7168)` tensor,
reshaped by NKIBench to `v1 (128, 32, 7168)` = `[partition=128, middle=32, free=7168]`,
returning a `shared_hbm` `v2` of the same shape.

The kernel must pass NKIBench's relative-L2 correctness gate
(`||v_k - v_r||_2 < 2e-5 * ||v_r||_2`, per seed) on all five seeds `[0, 21, 42, 63, 84]`,
measured through `verify.py` on the remote profiler. **Correctness is the sole
promotion gate.** Latency is recorded as evidence and is *expected but not required*
to land well below the 1.022 ms baseline, toward the ~0.638 ms HBM read+write floor.

The structure must be clean and already HBM-bound-shaped — a single middle-axis
loop of `load → compute → store` over full-width `[128, 7168]` tiles — so that
phase 2 tunes DMA/compute overlap rather than rewriting the kernel. It must NOT
reproduce the baseline's pathological 5-buffer / 4-vector-pass shape.

The compute sequence is chosen by a **correctness-first decision ladder**: try the
fused single-Scalar-op form first, and descend only on an actual L2 failure. The
rung that first passes on all five seeds becomes the phase-1 kernel, and the
LUT-accuracy finding is recorded for phase 2/3.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. The `verify.py` invocation used throughout is:

```
python3 ../../verify.py --op silu --candidate runs/<file>.py [--fast]
```

(There is no local NKI/Trainium install on this host; kernels compile and run only
on the remote profiler, so a `--fast` run — seed 42, `warmup=3`, `iters=20` — is the
cheapest available combined compile-and-correctness check. There is no separate
local compile gate.)

- AC-1: The promoted kernel passes NKIBench's relative-L2 correctness gate on all
  five seeds `[0, 21, 42, 63, 84]` (`||v_k - v_r||_2 < 2e-5 * ||v_r||_2`, fp32),
  measured through `verify.py` on the remote profiler.
  - Positive Tests (expected to PASS):
    - `verify.py --op silu --candidate runs/<file>.py --fast` reports the op line
      as `PASS` (seed 42) with a scored latency printed.
    - The full run (no `--fast`) exits `0` and its summary reports every seed
      passing: it prints `correct: 1/1` and a `geomean speedup` line, and every
      per-seed `l2_norm_passed` is true across `[0, 21, 42, 63, 84]`. The
      machine-checkable signal is `verify.py` **exit code 0** with all per-seed
      `l2_norm_passed` true (the exact printed strings are read from actual
      `verify.py` output, not assumed).
  - Negative Tests (expected to FAIL when the kernel is wrong):
    - A kernel that mis-maps the middle axis (e.g. treats `v1[:, i0, :]` as if the
      middle index were a partition or free coordinate) fails the L2 gate.
    - A kernel that transposes or otherwise mis-orients the output relative to
      `v2[p, m, f] == y[p*32 + m, f]` fails the L2 gate.
    - A kernel that downcasts any operand, intermediate, or the store to
      `bf16`/`tf32` fails the `2e-5` L2 gate.
  - AC-1.1 (diagnostic guidance, not a hard gate): passing the official `2e-5`
    gate on all seeds is the requirement. The mean relative L2 is also read as a
    sanity signal; a result that merely squeaks under `2e-5` (e.g. `1.9e-5`) is
    treated as suspect and investigated (typically by scoring the next rung down
    the ladder for comparison) before promotion.

- AC-2: The kernel matches the baseline's signature and I/O contract.
  - Positive Tests (expected to PASS):
    - `def kernel(v1)` accepts `v1` of shape `(128, 32, 7168)`, dtype fp32, and
      returns a single `v2 = nl.ndarray((128, 32, 7168), dtype=np.float32,
      buffer=nl.shared_hbm)`; the adapter reshapes `v2` back to `(4096, 7168)`
      without error.
  - Negative Tests (expected to FAIL):
    - Returning a tensor of a different shape, a non-`shared_hbm` buffer, or a
      wrong dtype causes the adapter/gate to reject the candidate.

- AC-3: The kernel is fp32 end-to-end with explicit fp32 constants.
  - Positive Tests (expected to PASS):
    - Every load, activation, arithmetic op, intermediate buffer, and store is
      fp32; any scalar constants used by the fallback rungs (e.g. `-1.0`, `1.0`)
      are fp32 and any `scale` argument passed to `nisa.activation` is fp32.
  - Negative Tests (expected to FAIL):
    - Introducing a `bf16`/`tf32`/`fp16` operand, constant, or store anywhere in
      the compute chain fails the `2e-5` L2 gate.

- AC-4: The index mapping is exact and mask-free in the primary candidate.
  - Positive Tests (expected to PASS):
    - Output obeys `v2[p, m, f] == y[p*32 + m, f]`; because SiLU is elementwise,
      the layout is correctness-neutral and the same op is applied to every
      element. Dimensions are exact (`128 * 32 = 4096 = M`, `7168 = N`), so the
      primary candidate contains **no** masking or partial-tile logic.
  - Negative Tests (expected to FAIL):
    - Unnecessary masking / partial-tile logic in the primary candidate is
      disallowed (there are no partial tiles to guard). A transposed or otherwise
      mis-mapped index fails the L2 gate.
    - Exception for forced fallback tiling: if a compile/legality/SBUF failure of
      the full-width form forces narrower tiling, exact-divisor tiles that need no
      mask are preferred; any mask that is introduced must be justified in the
      candidate notes and must preserve the exact output shape and `[p, m, f]`
      mapping.

- AC-5: The promoted kernel has a clean, HBM-bound-shaped, single-loop structure.
  - Positive Tests (expected to PASS):
    - The promoted (primary) candidate uses exactly one `nl.affine_range` over the
      middle axis (`32`), with a body of `load [128, 7168] → compute → store
      [128, 7168]` and **no** inner free-dim loop (the `7168` free axis is handled
      in one activation call), holding at most a small, fixed number of live SBUF
      buffers (≤ 2 for v1, ≤ 3 for v2, ≤ 5 for v3).
    - `affine_range` is used because the middle-axis iterations are independent
      (no loop-carried dependency), letting the compiler pipeline DMA and compute.
  - Negative Tests (expected to FAIL / be rejected in review):
    - A promoted candidate that adds a nested inner free-dim loop where a
      full-width activation suffices, or uses `sequential_range` without a real
      loop-carried dependency, is rejected as not HBM-bound-shaped.
    - v1 and v2 must **not** replicate the baseline's 4-vector-pass / 5-buffer
      shape. v3 (exp-exact) is the correctness fallback and **may** use the
      `exp → add → reciprocal → multiply` chain; it is exempt from the
      not-4-pass rule and is judged only on (a) being reached only after a real
      L2 failure of both v1 and v2, and (b) preserving the exact full-shape
      `[p, m, f]` mapping.
    - Fallback narrower tiling (exact-divisor free or middle tiles, no shape
      remap, no partial tiles) is permitted **only** after a documented
      compile/legality/SBUF failure of the full-width form; it is not a default.

- AC-6: The decision ladder is followed and full evidence is recorded.
  - Positive Tests (expected to PASS):
    - v1 (fused `nl.silu`) is implemented and scored first. The ladder descends to
      v2 (`nl.sigmoid` + `nl.multiply`), then v3 (exp-exact baseline math), **only**
      on an actual rel-L2 failure of the rung above.
    - The rung that first passes on all five seeds is recorded as the phase-1
      kernel, and the LUT-accuracy finding (which activation tables pass the gate)
      is written down for phase 2/3.
    - Evidence is complete: a row appended to `benchmark.csv`
      (`timestamp,op,candidate,parent,passed,latency_ms,speedup,notes`), a node
      appended to `candidates.jsonl` (parent link forming the DAG — the root
      candidate's parent is the baseline kernel filename), and a per-engine
      profiler digest saved under `profile/`
      (`## verify.py --fast` block + `## verify.py FULL` block + a bottleneck read).
    - For every rung that is *tried*, its `candidates.jsonl` verdict distinguishes
      a compile/legality failure from a numerical rel-L2 failure, so the phase-2
      handoff records **why** a rung was rejected, not merely that it was.
  - Negative Tests (expected to FAIL):
    - Descending the ladder without an observed L2 failure of the higher rung, or
      promoting a rung whose full 5-seed run was never recorded, is rejected.
    - A promotion with a missing `benchmark.csv` row, missing `candidates.jsonl`
      node, or missing `profile/` digest is incomplete and rejected.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.
This design is highly deterministic — the operator, layout, dtype, tiling shape, and
compute-sequence ladder are fixed by the draft — so the bounds are narrow and differ
mainly in *how far down the ladder* correctness forces the implementation.

### Upper Bound (Maximum Acceptable Scope)

v1 (fused `nl.silu`) passes rel-L2 on all five seeds and is promoted: a single
Scalar-engine op per tile, one `nl.affine_range(32)` over the middle axis, full-width
`[128, 7168]` tiles, two live SBUF buffers, evidence recorded (benchmark.csv row +
candidates.jsonl node + profile digest), and the LUT-accuracy finding noted for
phase 2. This completes the goal with the minimal instruction count and without
over-engineering — no double buffering, no batching, no dtype tricks.

### Lower Bound (Minimum Acceptable Scope)

Whichever rung of the ladder (v1, then v2, then v3) **first** passes rel-L2 on all
five seeds, recorded with complete evidence. v3 (exp-exact baseline math) is
guaranteed-correct by construction and is the exit guarantee that phase 1 ends with a
correct kernel regardless of LUT behavior. Even if the promoted rung only ties or
modestly beats the 1.022 ms baseline, it is a valid phase-1 kernel because correctness
— not speedup — is the promotion gate.

### Allowed Choices

- Can use:
  - Compute sequence per the ladder: v1 `nisa.activation(op=nl.silu, ...)`;
    v2 `nisa.activation(op=nl.sigmoid, ...)` + `nl.multiply`;
    v3 `nisa.activation(op=nl.exp, scale=-1.0)` + `nisa.tensor_scalar(add 1.0)` +
    `nisa.reciprocal` + `nl.multiply` (baseline math).
  - `nl.affine_range` over the middle axis (independent iterations).
  - Full-width `[128, 7168]` tiles as the primary shape; exact-divisor narrower
    free/middle tiling **only** as a documented fallback if a rung exceeds SBUF or
    fails legality at full width (correctness-neutral; no shape remap, no masks).
- Cannot use:
  - `bf16` / `tf32` / `fp16` anywhere (fp32 is mandated by the gate).
  - Any layout change or transpose (elementwise ⇒ layout is correctness-neutral;
    keep the baseline `(128, 32, 7168)` shape so harness reconciliation is untouched).
  - Double buffering / ping-pong SBUF, multi-slice SBUF batching, or tile-width
    parameterization/sweeping (all deferred to phase 2/3).
  - Masking / partial-tile logic in the primary candidate (dims are exact).

> **Note on Deterministic Designs**: The draft fixes the operator, layout, dtype,
> primary tiling, and the ordered compute ladder, so the upper and lower bounds differ
> only in which ladder rung correctness forces. "Allowed Choices" is intentionally
> narrow per the draft specification; the only genuine branch point is an *observed*
> rel-L2 failure, which deterministically selects the next rung.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Primary candidate (v1) skeleton — one middle-axis loop, one Scalar op per tile:

```python
@nki.jit
def kernel(v1):                       # v1: (128, 32, 7168) fp32
    P, MID, F = 128, 32, 7168
    v2 = nl.ndarray((P, MID, F), dtype=np.float32, buffer=nl.shared_hbm)
    for i0 in nl.affine_range(MID):                        # 32 middle-axis slices
        x_tile = nl.load(v1[:, i0, :])                     # [128, 7168] HBM->SBUF
        y_tile = nisa.activation(op=nl.silu, data=x_tile)  # Scalar: x*sigmoid(x)
        nl.store(v2[:, i0, :], value=y_tile)               # [128, 7168] SBUF->HBM
    return v2
```

Pin the exact index-expression spellings to the baseline's addressing idiom
(`nl.arange(128)[:, None]`, `nl.arange(7168)[None, :]`) during implementation so the
partition axis is unambiguously `128` and the free axis is `7168`.

Fallback rungs, reached only on an observed rel-L2 failure of the rung above:

```python
# v2 — isolates the sigmoid LUT (separate table from silu)
sig_tile = nisa.activation(op=nl.sigmoid, data=x_tile)     # Scalar
y_tile   = nl.multiply(x_tile, sig_tile)                   # Vector, tensor_tensor

# v3 — exp-exact = baseline math, passes by construction
e     = nisa.activation(op=nl.exp, data=x_tile, scale=-1.0, bias=0.0)  # exp(-x)
denom = nisa.tensor_scalar(data=e, op0=nl.add, operand0=1.0)           # 1 + exp(-x)
recip = nisa.reciprocal(data=denom)
y     = nl.multiply(x_tile, recip)
```

SBUF budget (trn2, ~208 KB/partition usable): `[128, 7168]` fp32 = 28 KB/partition, so
v1 ≈ 56 KB (2 buffers), v2 ≈ 84 KB (3), v3 ≈ 140 KB (5) — all fit at full width before
compiler temporaries, so the ladder does not require narrower tiling for SBUF reasons.

Hardware grounding (from `kernel-cost-analysis`, trn2): HBM floor to read
117 MB + write 117 MB fp32 ≈ `235 MB / 368 GB/s = 0.638 ms`; fused-SiLU Scalar compute
floor ≈ 0.191 ms ⇒ **HBM-bandwidth bound** with zero Vector time. The cost model
over-prices `reciprocal` (cpe=26), so do **not** infer "reciprocal is the bottleneck"
from theory — trust measured-vs-floor (1.022 vs 0.638 ms) and the profiler's per-engine
numbers instead.

### Relevant References

- `workspaces/silu/docs/draft-phase1.md` — the source draft (preserved below).
- `../AccelOpt/NKIBench/kernels/silu_M4096_N7168_0.py` — the baseline kernel (4-pass
  Vector chain, `i0=32 × i1=7 (1024-wide) × i2=2 (512-wide)`, 5 SBUF buffers, 1.022 ms).
- `../AccelOpt/NKIBench/reference/silu_M4096_N7168_numpy_0.py` — numpy reference
  (`x / (1 + np.exp(-x))`) and the `transform_to_nki_inputs` / `transform_nki_outputs`
  reshape contract that fixes `v2[p, m, f] == y[p*32 + m, f]`.
- `../../verify.py` — the correctness/latency harness (rel-L2 gate; prints
  MFU/PE/Vec/Scl/DMA/HBMrd/HBMwr; `--fast` = seed 42 only).
- `baselines.json` — cached baseline latency `silu[case=2] = 1.022441 ms`.
- `workspaces/matmul/docs/plan-phase1.md` and `workspaces/rmsnorm_matmul/` evidence —
  house style for AC structure, `benchmark.csv` / `candidates.jsonl` / `profile/`.
- Skill `kernel-cost-analysis` — theoretical per-engine floor to compare
  against the profiler's measured numbers.

## Dependencies and Sequence

### Milestones

1. Correct primary kernel (v1):
   - Phase A: Write `runs/silu_v1.py` with the full-width `nl.affine_range(32)` /
     fused `nl.silu` skeleton, pinning exact index spellings and fp32 dtype.
   - Phase B: Score with `--fast` (seed 42) as the cheap compile+correctness check;
     if it PASSes, run the full 5-seed pass.
2. Ladder descent (only if a rung fails rel-L2):
   - Step 1: On a v1 L2 failure, implement `runs/silu_v2_sigmoid_mul.py`
     (`nl.sigmoid` + `nl.multiply`) and score it.
   - Step 2: On a v2 L2 failure, implement `runs/silu_v3_exp_exact.py` (baseline
     math), which passes by construction, and score it.
   - Each rung is a small, independent edit; record whether a failure was a
     compile/legality failure or a numerical rel-L2 failure.
3. Promotion + evidence + handoff:
   - Step 1: Promote the first rung that passes all five seeds.
   - Step 2: Append the `benchmark.csv` row and the `candidates.jsonl` DAG node,
     and save the `profile/` per-engine digest.
   - Step 3: Record the LUT-accuracy finding (which activation tables passed) and
     the measured-vs-floor gap as the phase-2 handoff.

Dependencies: Milestone 2 depends on an observed failure in Milestone 1 (and each
ladder step depends on the failure of the step above); Milestone 3 depends on some rung
passing the full 5-seed run in Milestone 1 or 2. Phase 2 (DMA/compute overlap toward
the 0.638 ms floor) depends on the phase-1 profiler digest.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Write `runs/silu_v1.py`: `nl.affine_range(32)` over the middle axis, load `[128,7168]` → `nisa.activation(op=nl.silu)` → store, fp32 end-to-end, exact index spellings, no mask, ≤2 live SBUF buffers | AC-2, AC-3, AC-4, AC-5 | coding | - |
| task2 | Score v1: `verify.py --fast` (seed 42) as compile+correctness check, then full 5-seed run; capture PASS/FAIL and per-engine metrics | AC-1 | coding | task1 |
| task3 | If v1 fails rel-L2: write `runs/silu_v2_sigmoid_mul.py` (`nl.sigmoid` + `nl.multiply`, ≤3 buffers) and score it | AC-1, AC-5, AC-6 | coding | task2 |
| task4 | If v2 fails rel-L2: write `runs/silu_v3_exp_exact.py` (exp→add→reciprocal→multiply baseline math, ≤5 buffers) and score it | AC-1, AC-5, AC-6 | coding | task3 |
| task5 | Promote the first passing rung; append `benchmark.csv` row + `candidates.jsonl` DAG node (parent = baseline filename) + save `profile/` digest; verdicts distinguish legality vs L2 failure | AC-6 | coding | task2 |
| task6 | Record the LUT-accuracy finding and measured-vs-HBM-floor gap as the phase-2 handoff note in the profile digest | AC-6 | coding | task5 |

## Claude-Codex Deliberation

### Agreements
- The decision ladder is correct: fused `nl.silu` first, then `sigmoid * x`, then the
  exact `exp/add/reciprocal/multiply` chain only on an observed correctness failure.
- Correctness (rel-L2 on all five seeds) as the sole phase-1 promotion gate is
  appropriate; latency is recorded as evidence, not gated.
- All required APIs are present in the installed compiler artifact
  (`nl.silu`, `nl.sigmoid`, `nl.exp`, `nisa.activation`, `nisa.reciprocal`,
  `nisa.tensor_scalar`, `nl.multiply`, `nl.affine_range`), and `nl.affine_range` is the
  correct iterator for the independent-iteration `load → compute → store` body.
- All three rungs fit SBUF at full `[128, 7168]` width (v1≈56 KB, v2≈84 KB, v3≈140 KB
  per partition, under ~208 KB usable), so the ladder does not need narrower tiling for
  SBUF reasons.
- The evidence schema (benchmark.csv columns, candidates.jsonl DAG node, profile digest)
  matches the house conventions used by the matmul / rmsnorm_matmul workspaces.

### Resolved Disagreements
- AC-5 rigidity vs fallback tiling: Codex flagged that "exactly one `affine_range`, no
  inner loop" contradicted the later allowance of fallback tiling. **Resolved** — the
  single-loop / full-width shape is now a requirement of the *promoted (primary)*
  candidate only; exact-divisor fallback tiling is explicitly permitted after a
  documented compile/legality/SBUF failure of the full-width form.
- v3 "not 4-pass" conflict: Codex noted v3 is legitimately a 4-pass chain because it *is*
  the exp-exact fallback. **Resolved** — the not-4-pass rule now targets v1/v2 only; v3
  is exempt and judged solely on being reached after a real L2 failure and preserving
  the exact `[p, m, f]` mapping.
- AC-4 masking absoluteness: Codex argued "masking present fails" was too absolute for a
  forced-fallback path. **Resolved** — unnecessary masking is disallowed in the primary
  candidate, but a forced fallback may use exact-divisor tiles (preferably mask-free);
  any introduced mask must be justified and preserve the exact output shape.
- Hardcoded evidence string: Codex cautioned against hardcoding `correct: 1/1`.
  **Resolved** — AC-1's machine-checkable signal is `verify.py` exit code 0 with all
  per-seed `l2_norm_passed` true; the printed strings are read from actual output.
  (`verify.py` does print `correct: 1/1` and a `geomean speedup` line on success.)
- Failure-cause granularity (Codex optional, adopted): AC-6 now requires each tried
  rung's `candidates.jsonl` verdict to distinguish a compile/legality failure from a
  numerical rel-L2 failure, improving the phase-2 handoff.
- Draft's "7168 < 32767 activation limit": investigation of the installed compiler
  artifact found **no** documented activation free-dim limit constant (no `32767`), but
  also nothing indicating `7168` is illegal, and whether the tile issues as one
  instruction or is internally tiled does not affect correctness. **Resolved** — the
  plan does not rely on the specific `32767` figure; the "single instruction" claim is
  softened to a latency-interpretation note, not a correctness assumption.

### Convergence Status
- Final Status: `converged` (round 2 of the Claude↔Codex loop; Codex returned
  `REQUIRED_CHANGES: none`, `DISAGREE: none`, `CONVERGENCE: converged`).

## Pending User Decisions

None. All Codex `QUESTIONS_FOR_USER` and round-1 `UNRESOLVED` items were resolved during
convergence and are recorded here for traceability (no item remains `PENDING`):

- Quantitative-metric classification (draft-answered): the draft author explicitly
  states phase 1 is "judged on correctness, not the speedup." Therefore the `2e-5`
  rel-L2 gate across all five seeds is a **hard requirement**, while the latency figures
  (0.638 ms HBM floor; ~0.61 ms / ~1.67× direction) are an **optimization
  trend/expectation**, not a phase-1 gate. No separate confirmation was needed because
  the draft pre-classifies these.
- `nl.silu` LUT accuracy vs the reference: resolved-by-design — the ladder scores v1
  empirically on the full 5-seed run and deterministically descends on failure. No user
  decision required.
- Whether `[128, 7168]` emits one activation instruction or is compiler-internally
  tiled: irrelevant to correctness; affects only latency interpretation and is noted in
  the profile digest. No user decision required.
- Whether phase 1 must beat baseline latency: resolved per the draft — correctness is
  the sole gate; the plan adds no speedup gate.
- Input distribution: the reference uses `np.random.normal(0, 1)` fp32 (ordinary
  values), so there are no NaN/inf/extreme cases to special-case.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `x_tile`, `y_tile`,
  `sig_tile`), matching the baseline kernel's idiom.
- fp32 constants must be written explicitly (e.g. `np.float32`-typed `1.0` / `-1.0`) so
  no implicit narrowing occurs; any `scale` passed to `nisa.activation` must be fp32.

--- Original Design Draft Start ---

# silu (M4096 N7168, fp32) — Phase 1 implementation draft

## Goal

Produce the **first correct** NKI kernel for elementwise SiLU / swish
`y = x / (1 + exp(-x)) = x * sigmoid(x)` over a `(4096, 7168)` fp32 tensor,
passing NKIBench's relative-L2 gate (`||v_k - v_r|| < 2e-5 * ||v_r||`) on all
five seeds `[0,21,42,63,84]`. Prefer a clean, understood, correct kernel over
speed; leave aggressive tuning to phase 2/3. But choose a loop/compute structure
that is already reasonable (minimal DMA, compute hidden under DMA) so we don't
start from the baseline's pathological 5-buffer / 4-vector-pass shape.

## What the operator is

Pure elementwise activation. No reduction, no matmul, no fusion. Every output
element depends only on the co-located input element:

```
y[i,j] = x[i,j] * sigmoid(x[i,j]),   sigmoid(t) = 1 / (1 + exp(-t))
```

The only real content of the kernel is: **(a) the tiled layout**, **(b) which
instruction sequence computes silu**, and **(c) keeping the pass HBM-bound.**

## Tiled layout (from the numpy reference, verified in numpy)

`transform_to_nki_inputs` reshapes the natural (row-major) input:

- `x (4096, 7168)` -> `v1 (128, 32, 7168)` = `[p, m, f]`
  - `v1[p, m, f] == x[p*32 + m, f]`  (**verified**: 20000 random probes match)

Output `v2 (128, 32, 7168)` = `[p, m, f]`, reshaped back to `(4096, 7168)` by
`transform_nki_outputs`, so `v2[p, m, f] == y[p*32 + m, f]`. The signature
therefore matches the baseline: `def kernel(v1)` returning a `shared_hbm` v2 of
the same shape.

Note the layout: the **partition axis is dim 0 (size 128)**, the middle axis
(size 32) and the free axis (size 7168) are both per-partition. Because silu is
purely elementwise, the layout is irrelevant to *correctness* — we just apply the
same op to every element — so no transpose, no cross-partition traffic. All the
128s / 32 / 7168 are exact (128*32 = 4096 = M, 7168 = N), so **no masking or
partial tiles anywhere.**

### SBUF budget forces a middle-axis tile

Per-partition data = `32 * 7168 = 229376` fp32 elements = **896 KB/partition**,
which far exceeds the ~208 KB usable SBUF per partition (trn2). So we cannot hold
a partition's whole row in SBUF; we tile the middle (32) axis.

Natural phase-1 tiling: **loop `i0 in range(32)`**, each iteration operating on
`v1[:, i0, :] = [128, 7168]` = **28 KB/partition**. That fits SBUF with huge
headroom (leaves room for the phase-2 double buffer), and `7168` is within the
Scalar-engine activation free-dim limit (well under 32767), so a single
activation instruction covers the whole 7168-wide slice — no inner free-dim loop
needed. This gives the minimal instruction count: **32 loads + 32 stores** of
`[128, 7168]`, each partition reading a contiguous 7168-run.

## Hardware grounding: cost model + bottleneck (trn2)

Numbers from `kernel-cost-analysis` (trn2: Scalar 1.20 GHz, Vector
0.96 GHz, HBM aggregate 368 GB/s = 16 x 23 GB/s):

- **HBM floor** — read 117 MB + write 117 MB fp32: `235 MB / 368 GB/s = 0.638 ms`.
  (Per-slice Formula-E: load `[128,7168]` = `7168*4*ceil(128/16)/23 ≈ 9.97 us`,
  x32 = 319 us load + 319 us store = 638 us.)
- **Compute floor, fused silu** — Scalar activation, cpe=1, free=7168:
  `1*7168*100/120 = 5.97 us/slice` x32 = **0.191 ms**. Comfortably **under** the
  0.638 ms DMA floor -> the fused-silu kernel is **HBM-bandwidth bound**, using
  zero Vector-engine time.
- **Baseline measured = 1.022 ms**; AccelOpt's reported ~1.67x = **0.612 ms**,
  which is essentially the 0.638 ms HBM floor. **All the headroom is in becoming
  DMA-bound** — i.e. removing extra SBUF passes / instruction overhead and
  overlapping compute with DMA. The baseline leaves ~1.6x on the table by doing
  four Vector passes (exp + add + reciprocal + multiply) through five separate
  large SBUF buffers.

Caveat on the model: the cost model prices `reciprocal` at cpe=26, which would
put the baseline's Vector chain at ~6.9 ms — far above its measured 1.02 ms. So
the model over-states reciprocal, and we should **not** claim "reciprocal is the
bottleneck" from theory alone. The trustworthy signal is measured-vs-HBM-floor
(1.02 vs 0.64 ms), and the real per-engine numbers the profiler returns
(`verify.py` prints MFU / Vec / Scl / DMA / HBM). Phase 1 doesn't need to win
this; it needs a correct kernel whose structure is already HBM-bound-shaped so
phase 2 tunes DMA overlap, not a rewrite.

## Compute-sequence choice: a correctness-first decision ladder

The crux is which instruction sequence computes silu, trading op-count (speed)
against LUT-approximation risk under the 2e-5 rel-L2 gate. The reference computes
`x/(1+exp(-x))` in fp32 (numpy exp ~0.5 ULP), i.e. effectively fp32-exact. The
**baseline uses the hardware `exp` LUT and passes the gate**, which proves this
target's activation LUTs are L2-accurate for this function class. Three rungs,
best-and-simplest first:

**v1 (primary) — fused `nl.silu`, single Scalar instruction.**
```python
y_tile = nisa.activation(op=nl.silu, data=x_tile)   # = x * sigmoid(x)
```
Simplest possible kernel: `load -> silu -> store`. One Scalar op, zero Vector
ops, HBM-bound at the floor, one intermediate buffer. `nl.silu` is a first-class
op specifier added in NKI 2.21 (release notes; confirmed in
`nki-api-reference`), computing exactly `x*sigmoid(x)` internally in fp32. The
**only** risk is that the silu LUT is a distinct table from `exp` and its
approximation error could exceed 2e-5 rel-L2. Given the exp LUT passes, this is
*likely* fine — but it must be scored, not assumed.

**v2 (fallback if v1 fails L2) — `nl.sigmoid` activation + `nl.multiply`.**
```python
sig_tile = nisa.activation(op=nl.sigmoid, data=x_tile)   # Scalar
y_tile   = nl.multiply(x_tile, sig_tile)                 # Vector, tensor_tensor
```
Two ops (1 Scalar + 1 Vector), still drops the baseline's `add` and expensive
`reciprocal`. Exactly equals `x*sigmoid(x)`. Isolates the sigmoid LUT (separate
from silu), so if v1 failed on a silu-specific table, this may pass. Vector
`multiply` (cpe=2, both-SBUF) is ~478 us total, still under the 638 us DMA floor,
so it stays HBM-bound.

**v3 (guaranteed-correct safety net) — exp-exact = baseline math.**
```python
e     = nisa.activation(op=nl.exp, data=x_tile, scale=-1.0, bias=0.0)  # exp(-x)
denom = nisa.tensor_scalar(data=e, op0=nl.add, operand0=1.0)           # 1 + exp(-x)
recip = nisa.reciprocal(data=denom)
y     = nl.multiply(x_tile, recip)
```
This is the accepted baseline's exact sequence (hardware `exp` + exact arith), so
it **passes the gate by construction** — the exit guarantee that phase 1 ends
with a correct kernel regardless of LUT behavior. Slower (4 ops), but only used
if both LUT rungs fail.

**Plan:** implement and score v1 first (expected pass). Only descend the ladder
on an actual L2 failure; each rung is a small, independent edit the RLCR loop can
score. Record which rung passes as the phase-1 kernel and note the LUT-accuracy
finding for phase 2/3.

## Kernel skeleton (v1)

```python
@nki.jit
def kernel(v1):                      # v1: (128, 32, 7168) fp32
    P, MID, F = 128, 32, 7168
    v2 = nl.ndarray((P, MID, F), dtype=np.float32, buffer=nl.shared_hbm)
    for i0 in nl.affine_range(MID):                        # 32 middle-axis slices
        x_tile = nl.load(v1[:, i0, :])                     # [128, 7168] HBM->SBUF
        y_tile = nisa.activation(op=nl.silu, data=x_tile)  # Scalar: x*sigmoid(x)
        nl.store(v2[:, i0, :], value=y_tile)               # [128, 7168] SBUF->HBM
    return v2
```
(Exact index-expression form — `nl.arange(128)[:,None]` etc. — to match the
baseline's addressing idiom; the plan step will pin the precise API spellings.)

## What phase 1 deliberately does NOT do (hooks for phase 2/3)

- **No double buffering yet.** v1 is serialized load->compute->store per slice.
  Phase 2's main lever is overlapping DMA with compute (ping-pong SBUF buffers,
  `affine_range` software pipelining) to actually hit the 0.638 ms HBM floor.
- **No multi-slice batching.** ~208 KB / 28 KB ≈ 7 slices could share one SBUF
  residency to amortize loop overhead; deferred to phase 2 once we see the real
  DMA-vs-overhead split from the profiler.
- **No dtype tricks.** fp32 in/out is mandated by the gate; the compute is
  Scalar-only and already under the DMA floor, so there's nothing to gain from
  bf16 here (unlike a compute-bound op) — and it would risk the L2 gate.
- **No layout change.** Elementwise means layout is correctness-neutral; keep the
  baseline's `(128, 32, 7168)` shape so the harness reconciliation is untouched.

## Acceptance for phase 1

1. Passes relative-L2 on all five seeds (run without `--fast` before promoting).
2. Clean, single-loop structure; every tile understood; no masking.
3. Record the run in `benchmark.csv` and the candidate DAG in `candidates.jsonl`;
   keep the profiler's per-engine digest under `profile/` (it drives phase 2).
   Expectation: v1 already lands well below 1.022 ms (toward the ~0.61 ms floor)
   since it removes three of the baseline's four passes — but phase 1 is judged
   on correctness, not the speedup.

--- Original Design Draft End ---
