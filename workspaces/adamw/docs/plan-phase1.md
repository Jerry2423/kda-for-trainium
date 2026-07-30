# AdamW (M10944×N2048, fp32) — Phase 1: First Correct Fused Kernel

## Goal Description

Produce the **first correct** NKI kernel for the fused **AdamW optimizer step** and
land it as `runs/adamw_v1.py`. The kernel takes four `(10944, 2048)` fp32 HBM tensors
(`theta, g, m, v`) and returns one `(10944, 2048)` fp32 output (`new_theta`), passing
NKIBench's relative-L2 correctness gate
(`||v_kernel − v_ref||_2 < 2e-5 · ||v_ref||_2`) on all five seeds `[0, 21, 42, 63, 84]`.

The reference computation (from `AccelOpt/NKIBench/reference/adamw_M10944_N2048_numpy_1.py`)
is a pure elementwise update — no reduction, no matmul, no cross-partition traffic:

```
theta_t   = theta - 1e-5 * theta
m_t       = 0.9 * m + 0.1 * g
v_t       = 0.999 * v + 0.001 * g * g
v_hat     = v_t * 1000
new_theta = theta_t - 0.01 * m_t / (sqrt(v_hat) + 1e-8)
```

The priority is a **clean, understood, correct** kernel that already has a reasonable
loop/compute structure (minimal DMA, single-op-per-tile free axis, minimal Vector-engine
pressure) so it does not inherit the auto-generated baseline's pathological 20-buffer,
heavily-fragmented shape. Aggressive tuning (DMA/compute overlap, Vector rebalance, shape
specialization) is explicitly deferred to Phase 2 and Phase 3.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. All tests are executed through the repo's verification path
(from inside `workspaces/adamw/`):

```
python3 \
    ../../verify.py --op adamw --candidate runs/adamw_v1.py --fast
```

- AC-1: The kernel passes NKIBench's relative-L2 correctness gate on all five seeds.
  - Positive Tests (expected to PASS):
    - `verify.py --op adamw --candidate runs/adamw_v1.py` reports `l2_norm_passed = true`
      for every seed in `[0, 21, 42, 63, 84]` (full run, after a `--fast` sanity pass).
    - A standalone numpy replay of the chosen folded algebra reproduces the reference to
      worst-case rel-L2 on the order of ~3.4e-8 across the five seeds (≈580× under the
      2e-5 gate), confirming the algebra itself is not the failure source if hardware
      passes.
  - Negative Tests (expected to FAIL):
    - Swapping any two of the input-argument roles (e.g. treating `v3` as `v` and `v4`
      as `m`) causes `l2_norm_passed = false`.
    - Reversing the final subtraction so the kernel computes `term − 0.99999·theta`
      (i.e. `reverse1=True` on the last op) causes `l2_norm_passed = false`.

- AC-2: The input/output contract exactly matches the benchmark baseline's signature and
  argument mapping.
  - AC-2.1: Signature is `def kernel(v1, v2, v3, v4)` with the verified mapping
    `v1 = theta`, `v2 = g`, `v3 = m`, `v4 = v` (read directly off the baseline's
    `nl.load` sources in `AccelOpt/NKIBench/kernels/adamw_M10944_N2048_0.py`).
    - Positive: the kernel's four `nl.load` calls read `v1→theta`, `v2→g`, `v3→m`,
      `v4→v`; the correctness gate passes (couples with AC-1).
    - Negative: any other permutation of the four loads fails AC-1's gate.
  - AC-2.2: Output is a fresh `nl.shared_hbm` `(10944, 2048)` fp32 `ndarray` that the
    function returns; the harness's identity `transform_nki_outputs` reshape accepts it.
    - Positive: `verify.py` accepts the returned tensor shape/dtype without a reshape or
      dtype error.
    - Negative: returning a `(2048, 10944)` (transposed) or non-fp32 output is rejected.

- AC-3: The kernel compiles and runs cleanly in the NKIBench harness (no compile error,
  no runtime invalid-value abort) — a hard Phase-1 gate independent of latency.
  - Positive Tests (expected to PASS):
    - `verify.py` completes without a compilation exception or a runtime error for the
      candidate.
    - The masked-tail structure produces a correct final 64-row tile (the
      `10944 = 128·85 + 64` partial tile), verified by AC-1's gate passing (the tail is
      where correctness is most at risk).
  - Negative Tests (expected to FAIL):
    - A single-instruction free axis wider than the hardware per-instruction limit fails
      to compile (guards against assuming `[128,2048]` is legal without checking).
    - An off-by-one tail predicate that stores rows `>= 10944` or drops valid rows fails
      AC-1's gate.

- AC-4: The kernel uses a clean fused compute structure with minimal Vector-engine
  pressure — structurally distinct from the fragmented baseline.
  - AC-4.1: The compute chain uses at most the algorithmic minimum of tile–tile combines
    on the Vector engine for this dependency graph (four `scalar_tensor_tensor`), with the
    two nonlinearities (`square`, `rsqrt`) offloaded to the Scalar engine via
    `nisa.activation`. (The `square` op MAY fall back to a Vector `g*g` per the fallback
    ladder without violating Phase-1 correctness.)
    - Positive: the primary implementation issues 2 Scalar `activation` + 4 Vector
      `scalar_tensor_tensor` ops per tile; correctness passes.
    - Negative: reproducing the baseline's ~15-op, 20-buffer, `[128,512]`-fragmented,
      `reciprocal`-based chain does not satisfy the "clean fused structure" intent (even
      though it is also correct).
  - AC-4.2: SBUF live-tile footprint stays well within budget (≈80 KB/partition for ~10
    live `[128,2048]` fp32 tiles at 8 KB/partition each, under the ~192–208 KB usable
    SBUF), leaving room for later double-buffering.
    - Positive: the kernel compiles without an SBUF over-allocation error.
    - Negative: forcing all 86 tiles resident simultaneously (baseline-style
      `(86, 2, ...)` giant buffers) would blow the budget and is not done.

- AC-5 (SOFT — direction, not a hard gate): Latency is near the baseline. Per the draft's
  explicit framing ("prefer a clean, understood, correct kernel over speed"), Phase-1
  success does NOT require a speedup; a candidate that passes AC-1..AC-4 is accepted even
  if it is at or modestly above baseline. The soft expectation is roughly ≤ 1.3× the
  baseline `1.305 ms` (i.e. no major regression); an actual speedup is welcome evidence
  but is a Phase-2/3 objective.
  - Positive Tests (expected to PASS):
    - The measured latency is recorded in `benchmark.csv` with `speedup = baseline/latency`
      whatever its value; a result in the ~1.0–1.3× baseline band is expected.
  - Negative Tests (expected to FAIL — but only as a signal, never blocking Phase-1
    acceptance if AC-1..AC-4 hold):
    - A latency dramatically worse than baseline (e.g. > 2×) is flagged as a regression to
      investigate before promoting, even though it does not by itself fail the correctness
      gate.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. The
draft specifies a highly deterministic design; the bounds are correspondingly narrow.

### Upper Bound (Maximum Acceptable Scope)
A single-pass, row-tiled fused kernel implementing the primary design: `[128, 2048]` tiles
over 86 iterations (last partial), the folded algebra
`new_theta = 0.99999·theta − 0.001·(9m + g)·rsqrt(999v + g²)`, computed as 2 Scalar
`activation` (`square`, `rsqrt`) + 4 Vector `scalar_tensor_tensor` ops, with
`nl.affine_range` to permit compiler pipelining, masked loads/stores for the tail, fp32
throughout, passing all five seeds. Includes recording the candidate in `benchmark.csv` +
`candidates.jsonl` and saving a profiler digest under `profile/`. It does NOT add
Phase-2/3 work (explicit double-buffering, burst tuning, engine rebalancing, or the
`P=96×114` shape specialization).

### Lower Bound (Minimum Acceptable Scope)
A correct kernel that passes the relative-L2 gate on all five seeds via any of the
correctness-preserving fallback rungs — e.g. `[128, 512]` sub-tiles instead of a single
`[128, 2048]` free-axis op, `sqrt + reciprocal` (with `+1e-8`) instead of `rsqrt`, or a
Vector `g*g` instead of `activation(square)`. The chain remains materially cleaner than
the fragmented 20-buffer baseline. Latency is recorded but not required to beat baseline.

### Allowed Choices
- Can use: `nisa.activation` (with `op=nl.square`, `op=nl.rsqrt`, `op=nl.sqrt`),
  `nisa.scalar_tensor_tensor`, `nisa.tensor_scalar`, `nl.multiply`/`nl.add`/`nl.subtract`,
  `nisa.reciprocal`, `nl.load`/`nl.store` with mask predicates, `nl.affine_range`,
  `nl.shared_hbm`/`nl.sbuf` buffers, `par_dim(128)` partition tiling, fp32
  (`np.float32`-typed) scalar constants. May choose the free-axis tile width
  (`2048` primary, `512` fallback) and the eps handling (`rsqrt` no-eps primary,
  `sqrt → +1e-8 → reciprocal` fallback).
- Cannot use: any change to the benchmark definition
  (`AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}`); a transpose or
  cross-partition reduction (the op is purely elementwise); bf16/fp16 intermediates that
  would lower precision below the fp32 reference; hand-tuning the baseline; writing outside
  `workspaces/adamw/runs/` and `workspaces/adamw/profile/`.

> **Note on Deterministic Design**: The draft fixes the algebra, the op sequence, the arg
> mapping, and the tiling. The upper and lower bounds differ only in which
> correctness-preserving fallback rung is used, not in the target behavior. Latency
> optimization is deliberately out of the Phase-1 bounds.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

Primary kernel skeleton (pseudocode — descriptive names, no workflow markers in the actual
source):

```python
@nki.jit
def kernel(v1, v2, v3, v4):          # v1=theta, v2=g, v3=m, v4=v
    P, N, T = 128, 2048, 86
    out_hbm = nl.ndarray((10944, 2048), dtype=np.float32, buffer=nl.shared_hbm)
    for i0 in nl.affine_range(T):    # affine_range → compiler may pipeline DMA w/ compute
        rows   = 128 * i0 + nl.arange(128)[:, None]
        m_pred = (-128 * i0 - nl.arange(128)[:, None] + 10943 >= 0)   # row < 10944
        theta = nl.load(v1[rows, nl.arange(2048)], mask=m_pred)
        g     = nl.load(v2[rows, nl.arange(2048)], mask=m_pred)
        m     = nl.load(v3[rows, nl.arange(2048)], mask=m_pred)
        v     = nl.load(v4[rows, nl.arange(2048)], mask=m_pred)
        # 6-op fused chain (compute unmasked — padding rows produced but never stored)
        g2   = nisa.activation(op=nl.square, data=g)                          # Scalar : g²
        vhat = nisa.scalar_tensor_tensor(v, nl.multiply, 999.0, nl.add, g2)   # Vector : 999v + g²
        rden = nisa.activation(op=nl.rsqrt, data=vhat)                        # Scalar : 1/sqrt(vhat)
        mm   = nisa.scalar_tensor_tensor(m,     nl.multiply, 9.0,     nl.add,      g)     # 9m + g
        term = nisa.scalar_tensor_tensor(mm,    nl.multiply, 0.001,   nl.multiply, rden)  # 0.001·mm·rden
        out  = nisa.scalar_tensor_tensor(theta, nl.multiply, 0.99999, nl.subtract, term) # 0.99999·theta − term
        nl.store(out_hbm[rows, nl.arange(2048)], value=out, mask=m_pred)
    return out_hbm
```

Verified API semantics that make the chain correct (confirmed against the NKI docs):
- `nisa.scalar_tensor_tensor(data, op0, operand0, op1, operand1, reverse0=False,
  reverse1=False)` computes `(data op0 operand0) op1 operand1` on the **Vector** engine,
  where `operand0` is a compile-time scalar (or `[N,1]` vector) and `operand1` is a **full
  same-shape tile**; math is done in fp32. The only non-commutative op in the chain is the
  final `subtract`: with default `reverse1=False` it computes
  `temp − operand1 = 0.99999·theta − term`, which is the intended order.
- `nisa.activation(op, data, bias=None, scale=1.0)` computes `op(scale·data + bias)` on the
  **Scalar** engine, defaults `scale=1.0, bias=None`, and accepts `op=nl.square`,
  `op=nl.rsqrt`, `op=nl.sqrt`. So `activation(op=nl.square, data=g) = g²` and
  `activation(op=nl.rsqrt, data=vhat) = 1/sqrt(vhat)` are exact.

Algebraic folding (verified in numpy, worst rel-L2 ~3.4e-8 across all five seeds):
- `v_hat = 1000·(0.999·v + 0.001·g²) = 999·v + g²`
- `0.01·m_t = 0.01·(0.9·m + 0.1·g) = 0.001·(9·m + g)`
- `theta_t = theta − 1e-5·theta = 0.99999·theta`
- eps: `1/sqrt(v_hat)` drops the `+1e-8` (since `v_hat = 999v + g² > 0` because
  `v = |normal| ≥ 0`, and eps ≈ 1e-8 vs a denominator of O(30) → ~3e-10 relative change,
  far under the gate).

Fallback ladder (each rung is correctness-preserving; step down only if the rung above
fails):
1. If hardware `rsqrt` precision fails the 2e-5 gate → use `activation(op=nl.sqrt)` then
   `nisa.reciprocal`, or keep the `+1e-8` eps. **Note:** exact reference parity requires
   `sqrt(v_hat) → add 1e-8 → reciprocal`; do NOT fold eps as `activation(..., bias=1e-8)`,
   because that would compute `rsqrt(v_hat + 1e-8)` (eps applied *before* the root), a
   different expression. Any eps fallback must be re-verified by the L2 gate.
2. If the single `[128, 2048]` free-axis op fails compile or SBUF allocation → split `N`
   into `[128, 512]` (or `[128, 1024]`) sub-tiles inside the tile loop.
3. If `activation(op=nl.square)` is unavailable or slower → compute `g²` with a Vector
   `nl.multiply(g, g)` (5 Vector ops total; slightly worse Vector floor, still correct).

Tail handling (86 tiles, last partial with 64 valid rows, `10944 = 128·85 + 64`):
- Mask every `nl.load` and the final `nl.store` with `row < 10944`
  (`-128·i0 − arange(128)[:,None] + 10943 >= 0`, copied verbatim from the correct baseline).
- Leave the 6-op compute **unmasked**: padding rows produce garbage (possibly NaN from
  `rsqrt` of negative garbage) but are never stored, so they are harmless — this matches
  the successful silu/mamba precedents in this repo. If any harness-level NaN diagnostic
  objects, the mitigation on file is to either mask the compute too (as the baseline does)
  or supply benign masked-load defaults (`theta=g=m=0, v=1`, keeping `v_hat > 0`).

### Relevant References
- `AccelOpt/NKIBench/reference/adamw_M10944_N2048_numpy_1.py` — numpy reference + the
  identity `transform_to_nki_inputs`/`transform_nki_outputs` reshapes and `get_inputs`
  (note `v = abs(normal) ≥ 0`, so `v_hat > 0`).
- `AccelOpt/NKIBench/kernels/adamw_M10944_N2048_0.py` — the auto-generated baseline
  (`kernel(v1,v2,v3,v4)`, 20 SBUF buffers, `[128,512]` inner tiles, `sqrt + reciprocal`,
  masked tail predicate). Source of the verified arg mapping and the tail predicate.
- `verify.py` (invoked via `../../verify.py`) — the correctness/latency harness; gates on
  `l2_norm_passed`.
- `baselines.json` — `adamw[case=2]` baseline latency `1.305 ms`.
- Prior-op memory: silu (DMA-bound, finer free-axis tiling won), mamba (static unroll),
  rmsnorm_matmul (fp32-floor sensitivity) — precedents for the deferred Phase-2/3 levers.

### Hardware grounding (trn2, for context only)
Per `[128, 2048]` fp32 tile (from `kernel-cost-analysis`):
- DMA floor: model-conservative ~1.225 ms (368 GB/s), but measured HBM ~781 GB/s →
  ~0.574 ms real DMA ceiling.
- Vector floor: 4 `scalar_tensor_tensor` ≈ 0.734 ms.
- Scalar floor: 2 `activation` ≈ 0.294 ms.
- Baseline measured = 1.305 ms.

**Phase-1 finding (sets up later phases):** adamw is NOT trivially DMA-bound like silu — at
real HBM bandwidth the Vector floor (0.734 ms) is comparable to / above the DMA floor
(0.574 ms), so reducing/rebalancing Vector pressure and guaranteeing DMA/compute overlap is
the real lever. This is a Phase-2/3 concern; Phase-1 only needs correctness at ~baseline
latency and should trust measured-vs-floor over raw theory (the model over-prices the
Vector chain; the baseline's 1.3 ms confirms this).

## Dependencies and Sequence

### Milestones
1. Implement the primary kernel: write `runs/adamw_v1.py` per the skeleton above with the
   folded algebra, the 6-op fused chain, `affine_range(86)`, masked loads/stores, and fp32
   constants.
   - Phase A: confirm the arg mapping and tail predicate against the baseline (done in
     planning; re-check in code).
   - Phase B: implement the 6-op chain with the confirmed `scalar_tensor_tensor` /
     `activation` semantics.
2. Verify correctness: score with `--fast`, then the full five-seed run.
   - Step 1: `--fast` sanity pass.
   - Step 2: full `[0,21,42,63,84]` run; require `l2_norm_passed` on all seeds.
   - Step 3: on failure, invoke `kernel-accuracy-debugging` before guessing — likely suspects
     are the arg mapping (`v1..v4`), a tail predicate off-by-one, or an operand/reverse
     ordering error. Then walk the fallback ladder (eps/rsqrt → tile width → square op) if a
     precision or legality issue is diagnosed.
3. Record evidence: append the perf result to `benchmark.csv`; append the candidate to
   `candidates.jsonl` (parent = baseline, as a DAG); save the profiler digest under
   `profile/`.

Dependency structure: Milestone 2 depends on Milestone 1; Milestone 3 depends on a passing
Milestone 2. The fallback ladder is entered only if Milestone 2 Step 2 fails, and loops
back into a revised Milestone 1.

## Task Breakdown

Each task includes exactly one routing tag (`coding` = implemented by Claude, `analyze` =
executed via Codex / analysis skill).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Confirm in-code the arg mapping (`v1=theta,v2=g,v3=m,v4=v`) and tail predicate (`row<10944`) against the baseline kernel source. | AC-2, AC-3 | coding | - |
| task2 | Implement `runs/adamw_v1.py`: `[128,2048]` row tiling over `affine_range(86)`, masked loads/stores, fp32 constants. | AC-3, AC-4.2 | coding | task1 |
| task3 | Implement the 6-op fused chain (2 `activation` square/rsqrt + 4 `scalar_tensor_tensor`) with the verified operand/reverse ordering for the folded algebra. | AC-1, AC-4.1 | coding | task2 |
| task4 | Score with `verify.py --fast`, then the full five-seed run; require `l2_norm_passed` on `[0,21,42,63,84]`. | AC-1, AC-3 | coding | task3 |
| task5 | If the gate fails, diagnose via `kernel-accuracy-debugging` and apply the correctness-preserving fallback ladder (eps/rsqrt → `[128,512]` tiles → Vector `g*g`); re-verify. | AC-1 | coding | task4 |
| task6 | Record the result in `benchmark.csv`, the candidate in `candidates.jsonl` (parent=baseline), and save the profiler digest under `profile/`; note measured latency vs the `1.305 ms` baseline. | AC-5 | coding | task4 |
| task7 | (Optional) Sanity-replay the folded algebra in numpy across the five seeds to confirm the ~3.4e-8 margin independently of hardware, isolating algebra vs hardware if the gate fails. | AC-1 | analyze | - |

## Claude-Codex Deliberation

### Agreements
- The primary 6-op folded-algebra chain is reasonable and correct for Phase 1.
- The confirmed `scalar_tensor_tensor` operand order resolves the main correctness risk
  (final subtraction computes `0.99999·theta − term` with default `reverse1=False`).
- `activation(op=nl.square)` and `activation(op=nl.rsqrt)` support the planned `g²` and
  `1/sqrt(v_hat)` exactly; `sqrt` is available for the fallback.
- Dropping the `+1e-8` eps is acceptable given the numpy margin and `v = |normal| ≥ 0`
  (so `v_hat > 0`), pending the on-hardware L2 gate.
- `[128,2048]` row tiling with masked loads/stores and unmasked compute is a reasonable,
  precedented pattern for this harness; SBUF sizing (~80 KB/partition) is not a blocker.
- The fallback ladder is appropriately Phase-1 (compile/correctness-oriented, not premature
  optimization).
- Acceptance framing is correct: full five-seed relative-L2 pass + clean compile/run are
  hard gates; latency is a soft direction.

### Resolved Disagreements
- **eps fallback exactness (Codex minor DISAGREE, accepted):** keeping `+1e-8` is only exact
  as `sqrt(v_hat) → add 1e-8 → reciprocal`. Applying eps via `activation(..., bias=1e-8)`
  would compute `rsqrt(v_hat + 1e-8)` — a *different* expression (eps inside the root).
  Resolution: the plan's fallback rung 1 now explicitly specifies the `sqrt → +eps →
  reciprocal` form and requires L2 re-verification of any eps variant. Rationale: matches
  the numpy reference exactly and avoids a subtle non-parity.
- **Single-tile legality (Codex first-pass CORE_RISK):** "`2048 < 32767`" does not by itself
  prove the `[128,2048]` op compiles. Resolution: legality is treated as an AC-3 compile
  gate with a `[128,512]` fallback rung, not an assumption. Rationale: cheap to fall back,
  and the baseline already demonstrates `[128,512]` works.
- **Tail-row NaN safety (Codex CORE_RISK):** unmasked compute could feed garbage/negative
  padding into `rsqrt`. Resolution: kept unmasked (matches silu/mamba precedent) since
  unstored rows are never read; documented mitigation (mask compute, or benign load
  defaults `theta=g=m=0, v=1`) if a harness NaN diagnostic ever objects.

### Convergence Status
- Final Status: `converged` (1 convergence round; no `REQUIRED_CHANGES` and no `UNRESOLVED`
  after resolving the eps-fallback wording).

## Pending User Decisions

- DEC-1: Is the ~1.0–1.3× baseline latency a hard Phase-1 requirement or a soft direction?
  - Claude Position: Soft direction. The draft explicitly states "prefer a clean,
    understood, correct kernel over speed" and defers tuning to Phase 2/3; correctness +
    clean compile/run are the only hard gates. Encoded as AC-5 (soft) with AC-1..AC-4 hard.
  - Codex Position: Agrees — "I would require correctness plus no major regression, not a
    hard speedup... treating actual speedup as welcome evidence rather than a requirement."
  - Tradeoff Summary: Treating latency as a hard gate would risk rejecting a correct, clean
    kernel over a Phase-2/3 concern; treating it as soft matches the draft's stated intent.
    Both Claude and Codex — and the draft itself — agree it is soft; recorded here only so
    the user can override if a hard latency gate is actually desired.
  - Decision Status: Resolved from draft as SOFT (AC-5). Override if a hard latency gate is
    intended.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Phase", "Step", "task1", or similar workflow markers.
- These terms are for this plan document only, not for the resulting kernel source.
- Use descriptive, domain-appropriate naming in code (e.g. `theta`, `g`, `m`, `v`, `g2`,
  `vhat`, `rden`, `mm`, `term`, `out`), mirroring the reference's semantics.
- Type fp32 scalar constants explicitly as `np.float32` (mirroring the baseline's
  `np.dtype(np.float32).type(...)`).

--- Original Design Draft Start ---

# adamw (M10944 N2048, fp32) — Phase 1 implementation draft

## Goal

Produce the **first correct** NKI kernel for the fused **AdamW optimizer step**
over four `(10944, 2048)` fp32 tensors (`theta, g, m, v`) → one `(10944, 2048)`
output `new_theta`, passing NKIBench's relative-L2 gate
(`||v_k - v_r||_2 < 2e-5 * ||v_r||_2`) on all five seeds `[0,21,42,63,84]`.
Prefer a clean, understood, correct kernel over speed; leave aggressive tuning to
phase 2/3. But choose a loop/compute structure that is already reasonable (minimal
DMA, single-op-per-tile free axis, minimal Vector-engine pressure) so we don't
inherit the baseline's pathological 20-buffer / heavily-fragmented shape.

## What the operator is

Pure **elementwise** update. No reduction, no matmul, no cross-partition traffic.
Every output element depends only on the four co-located input elements. The numpy
reference (`../../AccelOpt/NKIBench/reference/adamw_M10944_N2048_numpy_1.py`):

```
theta_t     = theta - 1e-5 * theta
m_t         = 0.9 * m + 0.1 * g
v_t         = 0.999 * v + 0.001 * g * g
v_hat       = v_t * 1000
new_theta_t = theta_t - 0.01 * m_t / (sqrt(v_hat) + 1e-8)
```

The only real content of the kernel is: **(a) the tiled layout**, **(b) which
fused instruction sequence computes the update in the fewest Vector ops**, and
**(c) keeping the pass close to HBM-bound.**

## Tiled layout — an IDENTITY reshape (verified)

`transform_to_nki_inputs` reshapes each `(10944, 2048)` input to `(10944, 2048)`
— i.e. a **no-op identity reshape**. So the tiled inputs the kernel receives are
exactly the natural 2D arrays, indexed `[row, col]`. This matches the baseline,
whose signature is `def kernel(v1, v2, v3, v4)` with each `vN` a 2D
`(10944, 2048)` HBM tensor. **Input arg order (read off the baseline's `nl.load`
calls):**

| arg | tensor  | reference role |
|-----|---------|----------------|
| v1  | `theta` | parameters     |
| v2  | `g`     | gradient       |
| v3  | `m`     | 1st moment     |
| v4  | `v`     | 2nd moment     |

Output is a fresh `shared_hbm` `(10944, 2048)` fp32 tensor, reshaped back to
`(10944, 2048)` by `transform_nki_outputs` — again identity. Signature therefore
matches the baseline exactly.

Because the op is purely elementwise, layout is irrelevant to *correctness* — we
just apply the same update to every element. No transpose, no cross-partition
work.

### Row-tiling into 128-partition blocks (with a masked tail)

Partition dim ≤ 128, so we tile the row axis (`M = 10944`) into blocks of 128
partitions and keep the full `N = 2048` free axis in one instruction (2048 <
32767 activation free-dim limit, and well within Vector limits — no inner
free-dim loop). `10944 = 128 * 85 + 64`, so we need **86 tiles**, the **last one
partial (64 valid rows)**. This mirrors the baseline's tiling choice.

- **Tail handling:** mask every `nl.load` and the final `nl.store` with the
  row-bound predicate `-128*i0 - arange(128)[:,None] + 10943 >= 0` (exactly the
  baseline's predicate). Compute ops on the padding rows produce garbage but are
  **never stored**, so they need no mask — keeping the compute clean.
- **No-mask alternative (noted for later):** `10944 = 96 * 114` and `96 ≤ 128`,
  so partition-dim `P = 96` gives 114 exact tiles with zero masking. DMA byte
  volume is identical (Formula-E per-stream cost `≈ 245 µs` either way — 128-part
  ×86 vs 96-part ×114 differ by <1%), so this only removes the (nearly free) mask
  predicate at the cost of more loop iterations. Not worth it for phase 1; keep
  the well-understood 128×86 masked shape.

Per-partition SBUF for a `[128, 2048]` fp32 tile = `2048*4 = 8 KB`. Even holding
~10 such tiles live (≈80 KB/partition) sits comfortably under the ~208 KB usable
SBUF, leaving room for the phase-2 double buffer. **No middle-axis tiling needed**
(contrast silu, whose 32×7168 row forced a middle tile).

## Algebraic simplification (fold the constants) — verified in numpy

The reference multiplies then divides by 1000; fold it so the denominator's `sqrt`
sees the pre-scaled value and the numerator's `0.01` collapses into the `m_t`
coefficients. Two identities:

- `v_hat = 1000 * (0.999*v + 0.001*g²) = 999*v + g²`
  → the `*1000` disappears; `0.999*1000 = 999`, `0.001*1000 = 1`.
- `0.01 * m_t = 0.01 * (0.9*m + 0.1*g) = 0.009*m + 0.001*g = 0.001 * (9*m + g)`
- `theta_t = theta - 1e-5*theta = 0.99999 * theta`

Two eps-handling options, **both numerically identical** and both pass:

- **eps outside sqrt** (matches reference exactly): `sqrt(v_hat) + 1e-8`, then
  reciprocal.
- **eps dropped / rsqrt** (chosen): `1/sqrt(v_hat)`. Since `v_hat = 999*v + g² > 0`
  (`v = |normal| ≥ 0`) and the `1e-8` eps is `~1e-8` vs a denominator of O(30),
  dropping it changes the result by ~3e-10 relative — far below the gate.

**Numpy verification** (worst rel-L2 across all 5 seeds, fp32):

| formulation                     | worst rel-L2 | gate 2e-5 |
|---------------------------------|--------------|-----------|
| full simplified (eps outside)   | 3.42e-08     | PASS      |
| rsqrt, no eps                   | 3.42e-08     | PASS      |
| **6-op `scalar_tensor_tensor` chain (chosen)** | **3.43e-08** | **PASS** |

Margin is ~580× under the gate, so the simplification is safe with huge headroom.

## Chosen instruction sequence — 6 ops/tile (2 Scalar + 4 Vector)

Each op processes one `[128, 2048]` tile. Using the fused NKI primitives
(signatures confirmed via `nki-api-reference`):

- `nisa.activation(op, data, bias, scale)` computes `op(scale*data + bias)` on the
  **Scalar** engine.
- `nisa.scalar_tensor_tensor(data, op0, operand0, op1, operand1)` computes
  `(data op0 operand0) op1 operand1` on the **Vector** engine, where `operand0` is
  a **scalar** (its pre-scale is free, ≈ one `tensor_tensor` latency) and
  `operand1` is a **full tile**. This is the workhorse: it folds each constant
  pre-multiply into the tile-tile combine for free.

```
g2   = activation(op=square,  data=g)                              # Scalar : g²
vhat = scalar_tensor_tensor(v,  mult, 999.0,   add,   g2)          # Vector : 999v + g²
rden = activation(op=rsqrt,   data=vhat)                           # Scalar : 1/sqrt(vhat)
mm   = scalar_tensor_tensor(m,  mult, 9.0,     add,   g)           # Vector : 9m + g
term = scalar_tensor_tensor(mm, mult, 0.001,   mult,  rden)        # Vector : (0.001·mm)·rden
out  = scalar_tensor_tensor(theta, mult, 0.99999, subtract, term)  # Vector : 0.99999·theta − term
store(out)
```

`0.001 * (9m + g) = 0.009m + 0.001g = 0.01 * m_t` ✓, and multiplying by `rden =
1/sqrt(v_hat)` gives `0.01*m_t / sqrt(v_hat)` ✓. Final `stt` yields
`0.99999*theta − term` ✓.

Why this split:
- The two `sqrt`/`square` nonlinearities go on the **Scalar** engine
  (`activation`), keeping them **off** the Vector engine.
- The **four inherent tile-tile combines** (`999v+g²`, `9m+g`, `mm·rden`,
  `theta−term`) must live on the Vector engine (`tensor_tensor` family);
  `scalar_tensor_tensor` fuses each one's scalar pre-multiply for free, so **4 is
  the algorithmic minimum** of Vector ops for this dependency graph. Contrast the
  baseline, which spends ~11 Vector/Scalar ops through 20 SBUF buffers.

## Hardware grounding: cost model + bottleneck (trn2)

Per `[128, 2048]` fp32 tile, from `kernel-cost-analysis`
(trn2: Vector 0.96 GHz, Scalar 1.20 GHz, DMA 16×23 GB/s):

- **DMA (Formula E, HBM↔SBUF):** load `[128,2048]` = `8192 B/part · ceil(128/16)/23
  ≈ 2849 ns`. 4 loads + 1 store = `5 · 2849 ≈ 14.2 µs/tile` × 86 = **1.225 ms**
  model DMA-issue floor (= 448 MB / 368 GB/s aggregate). But measured HBM on this
  harness runs ~**781 GB/s** (2× the model's conservative 368) → **~0.574 ms**
  real DMA ceiling.
- **Vector floor:** 4 `scalar_tensor_tensor` (Formula A, cpe=1) `= 4 · 2048·100/96
  ≈ 8533 ns/tile` × 86 = **0.734 ms**.
- **Scalar floor:** 2 `activation` (cpe=1) `= 2 · 2048·100/120 ≈ 3413 ns/tile` × 86
  = **0.294 ms**.
- **Baseline measured = 1.305 ms.**

**Bottleneck read:** by the conservative model, DMA-issue (1.225 ms) dominates and
Vector (0.734 ms) hides under it — so a clean fused single-pass kernel should land
near the baseline immediately and the win is becoming truly DMA-bound. **But** if
real HBM ≈ 781 GB/s, the DMA floor drops to ~0.574 ms and the **Vector floor
(0.734 ms) becomes the true bottleneck**. This is the phase-1 finding that sets up
later phases: **adamw is NOT trivially DMA-bound like silu** — its 4 tile-tile
Vector ops are comparable to the real DMA floor, so reducing/rebalancing Vector
pressure (and guaranteeing DMA/compute overlap) is the real lever. (Caveat: the
model prices `reciprocal` at cpe=26; we avoid it entirely by using `rsqrt` on the
Scalar engine, and the baseline's measured 1.3 ms already shows the model
overstates the true Vector chain cost — trust measured-vs-floor over raw theory.)

Phase-1 target is simply **correctness at ≈ baseline latency (~1.0–1.3x)** with a
clean fused single pass. The overlap / Vector-rebalance / shape-specialization
levers below are explicitly deferred.

## Kernel structure (phase 1)

```python
@nki.jit
def kernel(v1, v2, v3, v4):          # v1=theta, v2=g, v3=m, v4=v
    P, N, T = 128, 2048, 86
    out_hbm = nl.ndarray((10944, 2048), dtype=fp32, buffer=nl.shared_hbm)
    for i0 in nl.affine_range(T):    # affine_range → compiler pipelines DMA w/ compute
        rows = 128*i0 + arange(128)[:, None]
        m_pred = (-128*i0 - arange(128)[:, None] + 10943 >= 0)   # tail mask
        # 4 masked loads [128,2048] HBM→SBUF
        theta = load(v1[rows, arange(2048)], mask=m_pred)
        g     = load(v2[rows, arange(2048)], mask=m_pred)
        m     = load(v3[rows, arange(2048)], mask=m_pred)
        v     = load(v4[rows, arange(2048)], mask=m_pred)
        # 6-op fused chain (unmasked — padding rows never stored)
        g2   = activation(op=square, data=g)
        vhat = scalar_tensor_tensor(g2? ...)      # see sequence above
        rden = activation(op=rsqrt, data=vhat)
        mm   = scalar_tensor_tensor(m,  mult, 9.0,     add,  g)
        term = scalar_tensor_tensor(mm, mult, 0.001,   mult, rden)
        out  = scalar_tensor_tensor(theta, mult, 0.99999, subtract, term)
        store(out_hbm[rows, arange(2048)], out, mask=m_pred)
    return out_hbm
```

- `nl.affine_range(86)` (not `sequential_range`): iterations are independent, so
  the compiler is free to pipeline the next tile's loads under this tile's
  compute/store.
- SBUF tiles: 4 loaded (`theta,g,m,v`) + up to 5 intermediates (`g2, vhat, rden,
  mm, term`) + `out`. ≈ 80 KB/partition live, room for double-buffering. Phase 1
  keeps them distinct/named for clarity; buffer reuse is a phase-2 knob.
- fp32 scalar constants typed as `np.float32` (mirroring the baseline's
  `np.dtype(np.float32).type(...)`).

## Correctness plan

1. Implement as `runs/adamw_v1.py`.
2. Score with `--fast` first, then full 5-seed before recording:
   ```
   python3 \
       ../../verify.py --op adamw --candidate runs/adamw_v1.py --fast
   ```
3. Gate is `l2_norm_passed` across seeds `[0,21,42,63,84]` — trust `verify.py`.
4. If the L2 gate fails, invoke `kernel-accuracy-debugging` (likely suspects:
   wrong input arg mapping `v1..v4`, a mask predicate off-by-one on the tail, or
   an `scalar_tensor_tensor` operand/reverse-flag ordering error) before guessing.
5. Record the result in `benchmark.csv` and the candidate in `candidates.jsonl`
   (parent = baseline). Save the profiler digest under `profile/`.

## Risks / open questions

- **`scalar_tensor_tensor` operand roles:** `operand0` must be the scalar and
  `operand1` the full tile; confirm the `reverse0/reverse1` defaults give
  `(data op0 scalar) op1 tile` (subtract must be `theta_scaled − term`, not the
  reverse). Verified against the numpy model above.
- **`activation(op=square)` availability** on the Scalar engine (vs a Vector
  `tensor_tensor(g,g)`). If `square` is unavailable/slower, fall back to a Vector
  `g*g`, which shifts one op onto the Vector engine (5 Vector ops) — still
  correct, slightly worse Vector floor; a phase-2 concern only.
- **Tail mask correctness** on the 64-row last tile — the single highest-risk
  correctness item; the predicate is copied verbatim from the (correct) baseline.

## Deferred to phase 2 / 3 (explicitly out of scope for phase 1)

- **DMA/compute overlap & Vector rebalance** (the real lever): confirm the fused
  pass is DMA-bound in *measurement*; if Vector-bound, move an op to Scalar/GpSimd
  or restructure to cut a tile-tile combine.
- **Double-buffering / burst-size tuning** on the load stream (cf. silu phase 3:
  finer free-axis tiling won; here the free axis is a single 2048 already).
- **Shape specialization** (phase 3): e.g. the `P=96 × 114` no-mask tiling, or
  wider/narrower row tiles, static unrolling (cf. mamba v5).

--- Original Design Draft End ---
