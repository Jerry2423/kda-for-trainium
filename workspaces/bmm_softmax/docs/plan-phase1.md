# bmm_softmax Phase 1 — First Correct fp32 Fused NKI Kernel

## Goal Description

Produce the first CORRECT NKI/Trainium kernel for the `bmm_softmax` operator (NKIBench
case 2): a batched matmul `x[b] = lhs[b] @ rhs[b]` for `b in 0..15`, followed by a
row-softmax over the N axis, in fp32. Shapes:
`lhs v1 (16,4096,64)=(B,M,K)`, `rhs v2 (16,64,4096)=(B,K,N)`,
`out (16,4096,4096)=(B,M,N)` (B=16, M=4096, K=64, N=4096). The reference computes, per
`(b,m)` row: `max_x = max_n x`, `exp_x = exp(x - max_x)`, `sum_exp = sum_n exp_x`,
`out = exp_x / sum_exp` (softmax over `axis=2`).

The kernel must pass NKIBench's relative-L2 correctness gate (`rel_tol = 2e-5`) across all
five seeds `[0,21,42,63,84]`, validated by `verify.py` (which gates on `l2_norm_passed`,
not allclose). Phase 1 prioritizes a clean, fully-understood, correct kernel over speed;
latency and profiler metrics are captured as evidence only, with NO phase-1 performance
floor. The matmul core is reused from the solved sibling `bmm` (identical
`(B16,M4096,K64,N4096)` GEMM, promoted 5-seed-correct); only the epilogue changes from a
plain store to a fused row-softmax.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. AC-1 through AC-5 are phase-1 correctness/structure gates;
AC-6 is promotion evidence (not a correctness gate).

- AC-1: The kernel traces and compiles under the NKIBench harness for `bmm_softmax`.
  - Positive Tests (expected to PASS):
    - `verify.py --op bmm_softmax --candidate runs/bmm_softmax_v1.py --fast` reaches the
      correctness stage (kernel traced, compiled, and ran on the remote profiler) without
      a trace/compile/codegen error.
    - The single `@nki.jit def kernel(v1, v2)` entry point with module-level + in-function
      NKI imports (the traced convention used by the baseline and the `bmm` sibling) is
      accepted by the tracer.
  - Negative Tests (expected to FAIL):
    - A kernel that places a softmax reduce/activation/elementwise op on a PSUM tile
      whose free width exceeds the PSUM cap (512 fp32) fails to compile and is rejected.
    - A kernel with a shape/`is_moving_onezero` mismatch in the identity-transpose call,
      or a malformed `activation`/`tensor_scalar`/`tensor_reduce` arg form, fails to trace.
- AC-2: The kernel passes the NKIBench relative-L2 correctness gate across all five seeds
  via a full (non-fast) `verify.py` run. This is the SOLE phase-1 correctness gate; it is
  decided on relative-L2 only (`< 2e-5`), never on allclose, and latency variance never
  rejects a candidate.
  - Positive Tests (expected to PASS):
    - `verify.py --op bmm_softmax --candidate runs/bmm_softmax_v1.py` (no `--fast`) reports
      `l2_norm_passed=True` for every seed in `[0,21,42,63,84]`.
    - The single-seed `--fast` gate passes first as a cheap pre-check.
  - Negative Tests (expected to FAIL):
    - A kernel that downcasts the matmul or softmax to bf16/tf32, reduces over the wrong
      axis (M instead of N), omits the max-shift, or mis-maps the output row-major order
      produces `l2_norm_passed=False` on at least one seed and is rejected.
    - A kernel that skips subtracting the row max and overflows `exp` (producing `inf`/`nan`)
      fails the gate.
  - AC-2.1: If the full-row (4096-wide) softmax path fails AC-2 for ANY reason — trace
    error, compile/codegen error, remote runtime error, `l2_norm_passed=False` on any seed,
    or unstable/nondeterministic result — the implementation falls back to the proven
    chunked (512-wide + `loop_reduce`) softmax form (see Allowed Choices) and re-runs AC-2.
    The fallback trigger is correctness-gated, not compile-only.
    - Positive: When full-row passes 5-seed L2, no chunked fallback run is required; the
      full-row candidate is the phase-1 result.
    - Negative: Declaring phase-1 done on a full-row kernel that failed any seed, without
      exercising the chunked fallback, is rejected.
- AC-3: The contraction is a faithful single-pass fp32 matmul: K=64 contracted in ONE
  `nc_matmul` per score tile, with no K-accumulation loop and no lower-precision path.
  - Positive Tests (expected to PASS):
    - Each `(b, mt, c)` score chunk is produced by exactly one main `nc_matmul` with K=64
      on the partition axis of both operands, orientation `[m_in=128(par), n=512(free)]`
      (inherited verbatim from the passing `bmm_v1`).
    - The rhs tile loads directly as `[k=64(par), n(free)]` (no transpose); the lhs tile
      `[m=128(par), k=64(free)]` is transposed via the identity idiom to
      `[k=64(par), m=128(free)]` (`is_transpose=True`, `is_moving_onezero=True`) and copied
      to SBUF before use as the stationary operand.
    - Static review confirms: no `+=` accumulation over K-tiles, and no explicit bf16/tf32
      cast or approximate mode anywhere (matmul or softmax).
  - Negative Tests (expected to FAIL):
    - Source containing a K-tile loop with `+=` accumulation, or any bf16/tf32 cast, is
      rejected on review.
    - Feeding lhs with K on the free axis as the stationary operand, or a matmul whose
      partition axis is not the K contraction, produces wrong results or fails to trace.
- AC-4: The fused softmax epilogue is mathematically faithful to the reference and reduces
  over the correct axis, entirely in SBUF (never on a PSUM tile).
  - Positive Tests (expected to PASS):
    - The row tile is `[m_in=128(par), n=4096(free)]`; the max and sum reductions are over
      the FREE axis (`axis=[1]`) = reference `axis=2` (N).
    - The epilogue computes, per row: `row_max` (free-axis max), `exp = exp(score - row_max)`
      (via `activation(op=nl.exp, bias=neg_max, scale=1.0)` with a `[128,1]` per-partition
      bias), `row_sum` (free-axis add over `exp`), `recip = reciprocal(row_sum)`, and
      `out = exp * recip` (via `tensor_scalar(op0=nl.multiply, operand0=recip[128,1])`).
    - Every softmax reduce/activation/elementwise op reads and writes SBUF tiles only; the
      `[128,512]` PSUM banks hold matmul/transpose results and are copied out to SBUF
      immediately.
  - Negative Tests (expected to FAIL):
    - Reducing over the partition axis (M), or omitting the `- row_max` shift, changes the
      result and fails AC-2.
    - Running a `tensor_reduce`/`activation` directly on a `[128,4096]` PSUM tile (exceeds
      the 512 fp32 PSUM free cap) fails to compile.
- AC-5: The kernel output reshapes to the reference shape `(16,4096,4096)` row-major with
  full, non-overlapping coverage (16 batches × 32 M-tiles × 8 N-chunks).
  - Positive Tests (expected to PASS):
    - The returned tensor — direct 3D `(16,4096,4096)` OR the 4D fallback
      `(16,32,128,4096)` — reshapes to `(16,4096,4096)` via `transform_nki_outputs` and
      matches the reference softmax within the L2 gate.
    - Tile union is exact: `32·128 = 4096` rows and `8·512 = 4096` cols per batch, no
      gap/overlap; the store maps `out[b, 128*mt+p, n]` (partition→m, free→n), row-major.
  - Negative Tests (expected to FAIL):
    - An output whose tiling leaves a gap or overlap, or whose row-major order differs from
      the reference, fails the L2 gate.
- AC-6 (promotion evidence, not a correctness gate): After AC-2 passes, phase-1 bookkeeping
  is recorded so phase 2 starts from evidence. Required for promotion/documentation, not for
  correctness acceptance.
  - Positive Tests (expected to PASS):
    - After a full run, `benchmark.csv` has the `bmm_softmax_v1` perf row, `candidates.jsonl`
      has the DAG root entry (parent = `baseline:bmm_softmax_B16_K64_M4096_N4096_0.py`), and
      the `profile/` digest holds MFU / PE% / Vec% / Scl% / DMA% / HBMrd / HBMwr.
    - The recorded notes state which softmax path passed (`full_row_4096` vs
      `chunked_512_loop_reduce`), for phase-2 reasoning.
  - Negative Tests (expected to FAIL):
    - A candidate documented as promoted with no `benchmark.csv` / `candidates.jsonl` /
      `profile/` evidence is treated as incomplete for promotion (it does not retroactively
      fail AC-2).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A clean single-file kernel `runs/bmm_softmax_v1.py` with a `@nki.jit def kernel(v1, v2)`
entry point that: loads a 128×128 identity into SBUF once; per batch loads `rhs[b]` resident
as `[64(par),4096(free)]` (16 KB/partition); per M-tile loads and identity-transposes the
lhs tile once, streams 8 N-chunks of width 512 (single-pass K=64 `nc_matmul`) copied into a
resident `score[128,4096]` SBUF tile; then runs a FULL-ROW (4096-wide) fused softmax
epilogue (`tensor_reduce(max)` → `neg_max` → `activation(exp, bias=neg_max)` →
`tensor_reduce(add)` → `reciprocal` → `tensor_scalar(*recip)`) entirely in SBUF, reusing the
`score`/`exp` storage in place where the compiler allows; issues one 4096-wide store per
M-tile into a direct 3D output `(16,4096,4096)`; captures the full profiler metrics digest
into `profile/`; and records the `benchmark.csv` row and `candidates.jsonl` DAG root. No
performance optimization (no two-phase transpose-all, no `activation` fused row-sum, no
ping-pong buffering, no bf16) is done in phase 1 — those are explicitly phase-2/phase-3.

### Lower Bound (Minimum Acceptable Scope)
Any single fp32 kernel that passes the 5-seed relative-L2 gate for `bmm_softmax` and is
recorded as the DAG root — including the documented fallbacks: the chunked
(512-wide + `loop_reduce`) softmax form (the baseline's proven pattern applied per-M-tile to
the resident score) when the full-row 4096-wide path fails AC-2; the 4D output shape
`(16,32,128,4096)` when the direct-3D partition-offset store fails to trace; and per-chunk /
1024-wide rhs loads when full rhs residency is the actual trace/resource issue.

### Allowed Choices
- Can use: fp32 throughout via the repo's established NKI path; the proven identity-transpose
  idiom (`is_transpose=True`, `is_moving_onezero=True`); `N_CHUNK=512` for the main matmul;
  the full-row (4096-wide) softmax as PRIMARY OR the chunked (512-wide + `nl.loop_reduce`)
  softmax as the correctness-gated fallback; `neg_max` built either as a separate
  `tensor_scalar(*-1)` or folded into the max reduce via `negate=True`; either direct 3D
  output `(16,4096,4096)` or the 4D fallback `(16,32,128,4096)`; rhs resident `[64,4096]` OR
  per-chunk / 1024-wide rhs loads; in-place reuse of the `score`/`exp` SBUF tile; module-level
  + in-function NKI imports (the traced convention).
- Cannot use: bf16/tf32 or any explicit lower-precision cast / approximate mode anywhere
  (matmul or softmax); a K-split accumulation loop; any softmax reduce/activation/elementwise
  op on a PSUM tile (or a single matmul with PSUM free width > 512 fp32); the `activation`
  fused row-sum (`reduce_op=nl.add`) shortcut in phase 1 (kept separate for clarity; it is a
  phase-2 lever); the two-phase transpose-all-up-front schedule (`bmm_v2`) in phase 1
  (phase-2 lever); hand-tuning of the baseline; edits to any
  `../../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}` file.

> **Note on Deterministic Designs**: The math mapping (transpose orientation, K=64 single
> pass, N=512 chunk, output tiling, softmax over the N free axis, max-shift for overflow
> safety) is fixed per the draft — those bounds converge. The genuinely open choices are:
> (a) full-row 4096-wide softmax vs chunked 512-wide softmax (full-row primary, chunked as a
> correctness-gated fallback per AC-2.1); (b) output shape (3D primary vs 4D fallback); and
> (c) rhs residency granularity — (b) and (c) gated purely by whether the tracer accepts the
> cleaner form.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach
```
identity_local = load 128x128 identity into SBUF        # once, reused for all transposes
for b in affine_range(16):                              # batch
    rhs_sb = load v2[b, 0:64, 0:4096] -> [64(par), 4096(free)]   # once per batch (16 KB/part)
    for mt in affine_range(32):                                  # 4096/128 M-tiles
        lhs_sb = load v1[b, mt*128:+128, 0:64] -> [128(par), 64(free)]
        lhs_t_psum = nc_matmul(lhs_sb, identity_local, is_transpose=True, is_moving_onezero=True)
                     -> PSUM [64(par),128(free)]
        lhs_t = copy(lhs_t_psum) -> SBUF [64(par),128(free)]

        # build the full score row [128, 4096] in SBUF (8 chunks of 512)
        score = SBUF [128(par), 4096(free)]                      # 16 KB/part
        for c in affine_range(8):                                # 4096/512 N-chunks
            acc = nc_matmul(lhs_t, rhs_sb[:, c*512:+512]) -> PSUM [128(par),512(free)]
            score[:, c*512:+512] = copy(acc)                     # PSUM -> SBUF

        # PRIMARY: full-row fused softmax over the free axis (N=4096), all in SBUF, fp32
        row_max = tensor_reduce(nl.max, score, axis=[1])         -> [128,1]
        neg_max = tensor_scalar(row_max, op0=nl.multiply, operand0=-1.0)  # or reduce negate=True
        exp_t   = activation(op=nl.exp, data=score, bias=neg_max[128,1], scale=1.0)  -> [128,4096]
        row_sum = tensor_reduce(nl.add, exp_t, axis=[1])         -> [128,1]
        recip   = reciprocal(row_sum)                            -> [128,1]
        out_t   = tensor_scalar(exp_t, op0=nl.multiply, operand0=recip[128,1])  -> [128,4096]
        store out[b, mt*128:+128, :] = out_t                     # one 4096-wide store

        # FALLBACK (only if the full-row path fails AC-2): chunked 512 + loop_reduce
        #   per-512-chunk tensor_reduce(max) + loop_reduce -> global row_max
        #   neg_max = -row_max
        #   per-chunk activation(exp, bias=neg_max) into exp chunks; tensor_reduce(add)
        #     + loop_reduce -> global row_sum
        #   recip = reciprocal(row_sum)
        #   per-chunk tensor_scalar(* recip); store
        # (this is literally the baseline's proven epilogue applied to the resident score)
return out   # (16,4096,4096); reshapes row-major to the reference shape
```
All slices become affine index expressions (`nl.arange(64)` for the K partition,
`nl.arange(128)` for M partition/free after transpose, `nl.arange(512)` for the N chunk,
`nl.arange(4096)` for the full free axis). Allocations carry `par_dim`: transpose PSUM/SBUF
tiles as `par_dim(64)`; main PSUM, `score`, `exp_t`, `out_t` as `par_dim(128)`; `[128,1]`
reduction results as `par_dim(128)`.

### Relevant References
- `../../AccelOpt/NKIBench/kernels/bmm_softmax_B16_K64_M4096_N4096_0.py` — the baseline
  itself is the CANONICAL proof of every softmax primitive at 512-wide: `tensor_reduce(nl.max,
  axis=[1])`, `nl.loop_reduce` to combine chunk partials, `tensor_scalar(op0=nl.maximum,
  operand0=-3.4e38, op1=nl.multiply, operand1=-1.0)` for `neg_max`, `activation(op=nl.exp,
  bias=[128,1], scale=1.0)`, `reciprocal([128,1])`, and `tensor_scalar(op0=nl.multiply,
  operand0=[128,1])` for the per-row divide. This is the exact fallback pattern.
- `../bmm/runs/bmm_v1.py` — the matmul core to mirror verbatim: identity-transpose (line 84),
  rhs-once-per-batch resident load (line 70), single-pass K=64 main matmul into `[128,512]`
  PSUM (line 98), direct 3D output store (line 110).
- `../bmm/runs/bmm_v2.py` — the promoted two-phase transpose-all schedule (1.253x); the
  phase-2 lever, NOT used in phase 1.
- `../rmsnorm_matmul/runs/rmsnorm_matmul_v1.py` (lines 87–105) — proves free-axis
  `tensor_reduce(nl.add, axis=[1])` and `tensor_scalar(op0=nl.multiply, operand0=[128,1])`
  broadcast-multiply at 1024-wide, 5-seed correct.
- `../../AccelOpt/NKIBench/reference/bmm_softmax_B16_K64_M4096_N4096_numpy_1.py` — reference:
  reshape-only `transform_to_nki_inputs`, row-major `transform_nki_outputs`.
- `verify.py` — correctness gate (`l2_norm_passed` across seeds); `--fast` runs a single seed.

## Dependencies and Sequence

### Milestones
1. Kernel implementation: write `runs/bmm_softmax_v1.py` (identity load → batch loop →
   M-tile transpose → 8 N-chunk matmuls into a resident `score[128,4096]` → full-row fused
   softmax → one 4096-wide store). Depends on the matmul mapping (reused from `bmm_v1`) and
   the softmax primitive forms (proven by the baseline + rmsnorm siblings) being fixed.
   - Phase A: identity load + batch/M loop; the 8-chunk matmul into the resident `score` tile
     (direct 3D output).
   - Phase B: full-row softmax epilogue on the SBUF `score` tile; static-review checkpoint —
     confirm reduce is over the FREE axis, `bias`/`operand0` are `[128,1]`, no softmax op on
     PSUM, no `+=` K loop, no dtype cast, exactly one main `nc_matmul` per `(b,mt,c)`.
2. Fast correctness gate: `verify.py --fast` (1 seed). If the full-row path fails to trace,
   compile, or run, OR is numerically off, apply the fallbacks. Depends on Milestone 1.
   - Step 1: if the full-row 4096-wide softmax fails AC-2 for any reason, switch the softmax
     epilogue to the chunked (512-wide + `loop_reduce`) form (the baseline's proven pattern).
   - Step 2: if the direct 3D partition-offset store fails to trace, switch the output to the
     4D baseline shape `(16,32,128,4096)`.
   - Step 3: only if rhs residency is the actual trace/resource issue, drop to 1024-wide or
     per-chunk rhs loads.
3. Full correctness measurement: `verify.py` without `--fast` (5 seeds); capture the metrics
   digest. Depends on Milestone 2 passing.
4. Bookkeeping (promotion evidence): append the `benchmark.csv` row, the `candidates.jsonl`
   DAG root (parent = `baseline:bmm_softmax_B16_K64_M4096_N4096_0.py`), record which softmax
   path passed (`full_row_4096` vs `chunked_512_loop_reduce`), and write the `profile/`
   digest. Depends on Milestone 3.

## Task Breakdown

Each task includes exactly one routing tag (`coding` = implemented by Claude, `analyze` =
via Codex `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Implement `runs/bmm_softmax_v1.py`: mirror `bmm_v1` through the matmul (identity transpose, K=64 single pass, 8 N-chunks of 512 into a resident `score[128,4096]`, direct 3D output) | AC-3, AC-5 | coding | - |
| task2 | Add the full-row (4096-wide) fused softmax epilogue on the SBUF `score` tile (`tensor_reduce(max)` → `neg_max` → `activation(exp, bias)` → `tensor_reduce(add)` → `reciprocal` → `tensor_scalar(*recip)`), all in SBUF | AC-4 | coding | task1 |
| task3 | Static-review checkpoint: reduce over FREE axis, `bias`/`operand0` are `[128,1]`, no softmax op on PSUM, no `+=` K loop, no dtype cast, one main `nc_matmul` per `(b,mt,c)`, affine index forms | AC-3, AC-4 | coding | task2 |
| task4 | Run fast (1-seed) `verify.py` gate; on ANY AC-2 failure (trace/compile/runtime/L2/instability) of the full-row path, switch the softmax to the chunked 512+`loop_reduce` fallback and re-run; then apply output-shape / rhs-granularity fallbacks in order if needed | AC-1, AC-2.1, AC-5 | coding | task3 |
| task5 | Run full 5-seed `verify.py`; confirm `l2_norm_passed` on all seeds | AC-2 | coding | task4 |
| task6 | Capture the profiler metrics digest into `profile/`; append `benchmark.csv` + `candidates.jsonl` root; record which softmax path passed | AC-6 | coding | task5 |
| task7 | After the digest is captured, characterize the measured bottleneck (PE% vs DMA% vs Scl/Vec%, HBMrd/HBMwr vs the read-once/write-once floor) as phase-2 evidence, and note whether the baseline-spill diagnosis is confirmed or refuted (evidence only, not a phase-1 gate) | AC-6 | analyze | task5 |

## Claude-Codex Deliberation

### Agreements
- The math mapping is correct: K=64 on the partition axis of both operands, identity-transpose
  lhs `[128,64] → [64,128]`, rhs used directly as `[64,512]` chunks, main matmul → `[128,512]`,
  covering `16 × 32 × 8` tiles; softmax reduces over the N free axis (`axis=[1]` = ref
  `axis=2`); max-shift is softmax-invariant and prevents `exp` overflow.
- Phase 1 is correctness-first: no performance floor; latency and metrics are evidence only.
  The 2e-5 relative-L2 gate (across seeds `[0,21,42,63,84]`) is the sole correctness gate,
  decided by `verify.py` on `l2_norm_passed`, not allclose.
- fp32 only; no bf16/tf32 or approximate mode; no K-accumulation loop; PSUM main free width
  capped at 512; NO softmax reduce/activation/elementwise op on a PSUM tile.
- Every softmax primitive the plan needs (free-axis `tensor_reduce(max/add)`, `[128,1]`
  `activation(exp, bias=...)`, `[128,1]` `tensor_scalar(*...)`, `reciprocal`) is already
  PROVEN on-device — by the `bmm_softmax` baseline itself at 512-wide and by the
  `rmsnorm_matmul` siblings at 1024-wide — so the original "API/broadcast shape" risk is
  retired. The only residual risk is running these ops at 4096-wide in one instruction.
- Full-row 4096-wide softmax as PRIMARY is acceptable given no phase-1 perf floor; the chunked
  512+`loop_reduce` form is the correctness-gated fallback (Codex explicitly ruled full-row
  primary "reasonable").

### Resolved Disagreements
- 4096-wide vector-op risk (Codex first-pass #1 concern): Claude showed via repo evidence that
  all softmax primitives + `[128,1]` broadcasts are proven (baseline 512-wide, rmsnorm
  1024-wide), narrowing the open risk to the single 4096-wide width. Resolution: keep full-row
  as primary, with the chunked 512+`loop_reduce` form (the baseline's own proven epilogue) as
  an explicit fallback. Codex AGREED.
- Fallback trigger scope (Codex second-pass REQUIRED_CHANGE): the fallback must fire on ANY
  AC-2 failure — trace error, compile/codegen error, remote runtime error, `l2_norm_passed=False`
  on any seed, or unstable/nondeterministic result — not only on a trace/compile failure, since
  the full-row path could compile and still be numerically off. Resolution: AC-2.1 written as a
  correctness-gated (not compile-only) fallback trigger. Applied.
- Numeric-safety phrasing (Codex first-pass): the "expected rel-L2 << 2e-5" claim was softened
  from an a-priori assertion to "expected (all fp32: 1.83e-7 matmul floor + fp32 exp/sum/div),
  CONFIRMED by the 5-seed run", because NKI `exp`/`reciprocal` accuracy and softmax sensitivity
  are only fully settled by the on-device gate. Applied.
- Baseline-slow (score-spill) diagnosis: retained as phase-2 perf MOTIVATION only, explicitly
  labeled UNVERIFIED until profiled (task7); it is NOT a phase-1 correctness claim (Codex
  cautioned against anchoring phase-1 on unverified perf diagnosis). Applied.
- Path choice full-row-vs-chunked as PRIMARY (Codex QUESTIONS_FOR_USER #1): resolved during
  convergence to full-row-primary + chunked-fallback (no perf floor makes either primary
  acceptable; trying the cleaner/simpler full-row first is fine when the fallback is
  correctness-gated). No user decision needed.
- Two-phase transpose-all (`bmm_v2`) and `activation` fused row-sum: both explicitly deferred
  to phase 2 and placed in "Cannot use" for phase 1, keeping the phase-1 kernel simplest.

### Convergence Status
- Final Status: `converged` (1 convergence round; the single REQUIRED_CHANGE — correctness-gated
  fallback trigger — was applied, both optional improvements adopted, no high-impact disagreement
  remaining).

## Pending User Decisions

None. Codex's three first-pass `QUESTIONS_FOR_USER` were all substantively resolved during
convergence: (1) full-row-vs-chunked primary → full-row primary with a correctness-gated chunked
fallback (AC-2.1); (2) a slower two-pass fallback's acceptability → the chunked in-SBUF fallback
covers this without a separate store-scores pass, and is acceptable given no phase-1 perf floor;
(3) canonical repo softmax idioms → the `bmm_softmax` baseline itself (512-wide) plus the
`rmsnorm_matmul` siblings (1024-wide) are the canonical examples, now cited in Relevant
References. The only quantitative threshold (relative-L2 `< 2e-5` across the five seeds) is a
benchmark-fixed HARD gate, not a user-tunable target — `verify.py` is the correctness source of
truth per CLAUDE.md.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code (batch / m-tile / n-chunk, `lhs_t`,
  `rhs_sb`, `score`, `row_max`, `neg_max`, `exp_t`, `row_sum`, `recip`, `out_t`,
  `identity_local`), matching the baseline, `bmm_v1`, and `rmsnorm_matmul_v1` conventions.

--- Original Design Draft Start ---

# bmm_softmax — Phase 1 Draft (first correct fused NKI kernel)

## Goal

Produce the first **correct** NKI kernel for `bmm_softmax` (NKIBench case 2): a
batched matmul followed by a row-softmax over the N axis. Prioritize a clean,
obviously-correct kernel that passes the relative-L2 gate across all five seeds
`[0,21,42,63,84]`; speed is phase-2/3 work. It must beat nothing to be a valid
phase-1 base, but the fused design below should already clear the baseline by a
wide margin (see "Why this beats the baseline").

## Operator

- Shapes/dtype: `lhs v1 (16,4096,64)=(B,M,K)`, `rhs v2 (16,64,4096)=(B,K,N)`,
  `out (16,4096,4096)=(B,M,N)`, fp32. `B=16, M=4096, K=64, N=4096`.
- Reference (`bmm_softmax_..._numpy_1.py`):
  ```python
  x       = lhs @ rhs                       # (B,M,N)
  max_x   = np.max(x, axis=2, keepdims=True)   # row max over N
  exp_x   = np.exp(x - max_x)
  sum_exp = np.sum(exp_x, axis=2, keepdims=True)
  out     = exp_x / sum_exp                  # softmax over N, per (b,m) row
  ```
  So softmax is over the **N=4096 axis**, independently for each of the `B*M`
  rows. `transform_to_nki_inputs` is reshape-only; `transform_nki_outputs`
  reshapes our result to the ref shape, so returning `(16,4096,4096)` is an
  identity reshape (this is what the sibling `bmm_v1` did and it passed L2).

## The matmul core is a solved sibling

`bmm` (NKIBench case 2, workspace `../bmm/`) has the **identical** GEMM
`(B16,M4096,K64,N4096)` and a promoted, 5-seed-correct kernel. Reuse its core
verbatim; only the epilogue changes (softmax instead of a plain store).

Facts inherited from the `bmm` core (see `../bmm/benchmark.csv`):
- `nc_matmul(stationary, moving) = stationary.T @ moving`; contraction dim `k`
  must be on the **partition** axis of both operands, both operands in SBUF.
- `K=64 <= 128` ⇒ the whole contraction is **one** Tensor-Engine pass per output
  tile (single `nc_matmul`, no K-accumulation loop).
- `moving = rhs[b]` tile `[k=64(par), n(free)]` loads directly (v2[b] is `[k,n]`).
- `stationary` must be `[k=64(par), m_in=128(free)]`; a loaded lhs tile is
  `[m_in=128(par), k=64(free)]`, so transpose it once via the identity
  `nc_matmul(is_transpose=True, is_moving_onezero=True)` idiom → `[k=64, m_in=128]`,
  copy to SBUF.
- fp32 `nc_matmul` on this core measured rel-L2 `1.83e-7` (fp32 emulation floor),
  far under the `2e-5` gate. Adding fp32 softmax vector ops keeps us at that floor.

## Why the baseline is slow (and what we fix)

Baseline latency is **7.29ms**, ~2.9x the pure-`bmm` baseline (2.55ms), despite
computing the same 1GB output. The baseline materializes essentially the entire
`(B,M,N)` score matrix in SBUF (`v11 ≈ [4,8,16,4,2,128,512]`) and does a chunked
online max/sum across it — that resident set is ~1GB and **spills to HBM**, so
the scores round-trip through HBM twice (write scores, read back for exp/divide)
on top of the 1GB output write.

**Fix = fusion.** Process one m-tile at a time. Its full score row is only
`[128, 4096] fp32 = 16 KB/partition`, trivially resident. Compute the row,
softmax it in place, store the normalized row, discard. Scores never touch HBM;
HBM traffic drops to the once-each floor (read lhs+rhs ≈ 34 MB, write out ≈ 1074 MB).

## Kernel plan (phase-1: `bmm_softmax_v1`)

Batch-outer, mirror `bmm_v1`'s structure exactly through the matmul, then replace
the plain store with a **full-row fused softmax epilogue** over the 4096 free axis.

```
out = ndarray((16,4096,4096), fp32, shared_hbm)
identity_local[128,128] = load(shared_constant(I128))     # once, for the transpose

for b in affine_range(16):
    rhs_sb[64, 4096] = load(v2[b])                         # resident, 16 KB/part
    for mt in affine_range(32):                            # 32 = 4096/128 m-tiles
        lhs_sb[128, 64]  = load(v1[b, 128*mt:.., :])
        psum_t[64,128]   = nc_matmul(lhs_sb, identity_local,
                                     is_transpose=True, is_moving_onezero=True)
        lhs_t[64,128]    = copy(psum_t)                    # stationary [k, m_in]

        # --- build the full score row [128, 4096] in SBUF (8 chunks of 512) ---
        score[128, 4096]                                   # 16 KB/part
        for c in affine_range(8):                          # 8 = 4096/512
            acc[128,512] = nc_matmul(lhs_t, rhs_sb[:, 512*c:512*c+512])  # 1 PSUM bank
            score[:, 512*c:512*c+512] = copy(acc)          # PSUM -> SBUF

        # --- fused softmax over the free axis (N=4096), all in SBUF, fp32 ---
        row_max[128,1] = tensor_reduce(max, score, axis=free)
        neg_max[128,1] = tensor_scalar(row_max, mul=-1.0)  # bias = -row_max
        exp_t[128,4096] = activation(exp, score, bias=neg_max, scale=1.0)  # exp(score - row_max)
        row_sum[128,1]  = tensor_reduce(add, exp_t, axis=free)
        recip[128,1]    = reciprocal(row_sum)
        out_t[128,4096] = tensor_scalar(exp_t, mul=recip)  # per-row multiply
        store(out[b, 128*mt:.., :], out_t)                 # one 4096-wide store
```

### Why full-row (not the baseline's chunked online softmax)

Because the full row (16 KB/part) is trivially resident, we do **not** need the
flash-style online max/sum with per-chunk rescaling that the baseline uses. One
`tensor_reduce(max)`, one `activation(exp, bias=-max)`, one `tensor_reduce(add)`,
one `reciprocal`, one `tensor_scalar(*recip)` — each a single instruction over the
whole 4096-wide axis (SBUF allows up to 32767 free elements; the 512 PSUM cap does
not apply since we reduce the SBUF score tile, not PSUM). This is both simpler and
strictly fewer vector passes than the online scheme.

### Correctness reasoning

- **Math matches the reference step-for-step**: row max over N, subtract, exp,
  row sum over N, divide. Softmax is invariant to the max shift; subtracting the
  row max both matches the reference numerics and prevents any exp overflow
  (scores ~ N(0,64), std≈8; `exp(score-max) <= 1`, sum <= 4096, all fp32-safe).
- **Axis is correct**: each m-tile row tile is `[m_in=128(par), n=4096(free)]`;
  reducing the free axis is reducing over N = reference `axis=2`. Store maps
  `out[b, 128*mt+p, n]` (partition→m, free→n), row-major = ref `(B,M,N)`.
- **fp32 throughout**: matmul at the emulation floor (`1.83e-7` on the sibling)
  plus fp32 exp/sum/divide ⇒ expected rel-L2 « `2e-5`. No bf16 anywhere in phase 1.
- **`activation` semantics** (confirmed via NKI API docs): computes
  `op(data*scale + bias)`; `bias` may be a `[128,1]` per-partition vector on
  NeuronCore-v3 (trn2) — exactly the per-row `-row_max`. The baseline uses this
  same `bias=[128,1]` pattern, so it is a supported idiom on this target.

### Resident SBUF budget (per partition, fp32) — no spill

`rhs_sb` 16 KB + `score` 16 KB + `exp_t` 16 KB + `lhs_sb`/`lhs_t` < 1 KB ≈ **~48 KB**
of the 192 KB budget. (`out_t` can reuse `exp_t` in place via the `tensor_scalar`
dst to save 16 KB if the compiler wants it; not required.) Well clear of spill, so
HBM stays at the read-once/write-once floor.

## Risks / things to verify during the RLCR loop

1. **`activation` fused-reduce vs separate reduce.** The API also supports fusing
   the row-sum into the `exp` `activation` (`reduce_op=nl.add, reduce_res=...`) in
   one Scalar-Engine pass. Phase-1 keeps the sum as a **separate** `tensor_reduce`
   for maximum clarity/verifiability; the fusion is a phase-2 lever (saves one
   Vector pass), noted but not taken now.
2. **`tensor_scalar` with a `[128,1]` vector operand** for the per-row multiply by
   `recip` (and the `*-1` for `neg_max`). The baseline uses `tensor_scalar` with a
   `[128,1]` `operand0` (`v16`) for exactly the final per-row multiply, so the
   pattern is supported. Confirm the API arg form on first compile.
3. **Output declaration `(16,4096,4096)`** returned directly (vs the baseline's
   `(16,32,128,4096)`). `bmm_v1` used the flat `(B,M,N)` form and passed the gate;
   `transform_nki_outputs` reshapes to ref shape (identity here).
4. **PSUM free-axis cap.** Keep every reduce/activation on the **SBUF** `score`/
   `exp_t` tiles (4096 allowed), never on a PSUM tile (512 cap). The 8 chunk
   matmuls each land in a `[128,512]` PSUM bank and are copied out immediately.

## Acceptance for phase 1

- 5-seed relative-L2 PASS (`< 2e-5`) via
  `python3 ../../verify.py --op bmm_softmax --candidate runs/bmm_softmax_v1.py` (drop `--fast` for the promote measurement).
- Record the perf row in `benchmark.csv` and the candidate in `candidates.jsonl`
  (parent = `baseline:bmm_softmax_B16_K64_M4096_N4096_0.py`), with the profiler
  digest (PE/Vec/Scl/DMA %, MFU, HBM rd/wr) so phase 2 knows the bottleneck engine.

## Phase-2 / phase-3 outlook (not implemented now)

- **Phase 2 (profile-driven):** adopt the sibling's proven **two-phase transpose
  M-block=32** schedule (`bmm_v2`, 1.253x on pure bmm) to remove per-tile
  transpose→matmul serialization; fuse the row-sum into the `exp` activation to
  drop a Vector pass. Bottleneck will likely be DMA (1 GB output write) or the
  matmul PE time — read the digest to decide. Softmax vector work (exp over 1 GB
  of scores) is new pressure on Scalar/Vector vs pure bmm; watch Scl/Vec %.
- **Phase 3 (shape specialization):** N=4096 is exactly 8×512 (PSUM-bank) and
  M=4096 exactly 32×128 — edge-free, like the `bmm` sibling, so classic shape
  specialization has little surface. Precision (bf16x2 for the matmul) is a
  possible lever but softmax is exp of scores, so any matmul error is amplified
  through exp — the `2e-5` gate is tighter here than for a plain GEMM; treat bf16
  with caution and gate on an offline rel-L2 sim first (as the sibling did).
```

--- Original Design Draft End ---
