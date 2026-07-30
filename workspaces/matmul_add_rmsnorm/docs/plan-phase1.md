# matmul_add_rmsnorm — Phase 1 Plan: First Correct NKI Kernel (fp32, weight-resident)

## Goal Description

Deliver the first **correct** NKI kernel for the fused operator `matmul_add_rmsnorm`
(NKIBench case 1) and, in the same kernel, land the single obvious, low-risk performance
win: making the weight matrix fully SBUF-resident so it is loaded once and reused, instead
of the baseline's reload-inside-the-M-loop pathology.

The operator computes, in fp32:

```
y   = x @ w + z                                  # dense GEMM, then residual add
rms = sqrt( mean_N(y^2) + eps )                  # row RMS reduced over the N (output) axis
out = y * g / rms                                # per-column g scale, per-row 1/rms
```

Shapes / dtype (all fp32): `x(M=4096, K=2048)`, `w(K=2048, N=2048)`, `z(M=4096, N=2048)`,
`g(N=2048,)`, `eps=1e-5` (python scalar), `out(M=4096, N=2048)`. Kernel signature is fixed:
`kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor)` with `eps` as the 3rd positional
argument. I/O is **raw 2D** — this case's `transform_to_nki_inputs` is the identity, so the
kernel receives the raw 2D tensors and slice-tiles itself, and returns a raw 2D `(4096,2048)`.

The NKIBench baseline (`matmul_add_rmsnorm_M4096_N2048_K2048_0.py`, latency **3.768493 ms**)
reloads all of `w` inside the M-loop: `32 (M-tiles) × 4 (N-chunks) × 16 (K-tiles) = 2048`
weight loads, streaming the 16 MB weight matrix ~32× ≈ **537 MB** of redundant HBM reads.
That reload traffic — not the compute — is the dominant cost, exactly the pathology the
sibling op `add_rmsnorm_matmul` fixed to get **3.754×** from weight-residency alone.

Phase 1 is **pure fp32, correctness-first**: no precision tricks, no loop restructuring beyond
weight residency. The kernel uses the explicit `nisa.nc_matmul` + identity-matmul transpose
idiom (proven by `matmul_v1`, `rmsnorm_matmul_v1`, `add_rmsnorm_matmul_v1`) rather than the
baseline's high-level `nl.matmul`, so the known Phase-2 win (compensated bf16x2 split-matmul)
is a clean extension rather than a rewrite.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. The authoritative correctness oracle is `verify.py`, which gates
on `l2_norm_passed` (relative-L2, `rel_tol=2e-5`) across seeds `[0,21,42,63,84]`.

- AC-1: The kernel passes the NKIBench relative-L2 correctness gate on **all five** seeds in
  fp32.
  - Positive Tests (expected to PASS):
    - `verify.py --op matmul_add_rmsnorm --candidate runs/matmul_add_rmsnorm_v1.py` (full
      5-seed) reports PASS with `l2_norm_passed` true for every seed `[0,21,42,63,84]`.
    - The `--fast` single-seed run passes before the full 5-seed run is attempted.
  - Negative Tests (expected to FAIL):
    - A variant that normalizes over K instead of N fails the gate.
    - A variant that adds `eps` before the `/N` mean (i.e. scales `eps` by `1/N`) fails or
      drifts near the tolerance boundary.
    - A variant that drops any seed and reports fewer than five per-seed results is not
      accepted as passing.

- AC-2: The kernel reproduces the reference math exactly (order and placement of every
  operation), all intermediates in fp32.
  - AC-2.1: The RMS reduction is over the **N free axis** of the assembled `[128, N]` matmul
    output (`tensor_reduce(axis=[1])`), NOT over K.
    - Positive: `sumsq` is a single reduction over the full `N=2048` row (the baseline already
      does exactly one `nl.sum(in_square, axis=[1])` over N and passes).
    - Negative: reducing per 512-wide N-chunk (four independent reductions without a combining
      pass) normalizes over 512 columns and fails.
  - AC-2.2: `eps` is added **after** the `1/N` mean: `mean_eps = sumsq*(1/N) + eps`, matching
    `np.mean(y**2, axis=-1) + eps`.
    - Positive: `eps` appears as an additive term applied to the already-averaged sum-of-squares.
    - Negative: `eps` folded into the `1/N` scale (`(sumsq+eps)*(1/N)` or `sumsq*(1/N+eps)`)
      fails.
  - AC-2.3: `g` is treated as a length-**N** free-axis scale, applied as a `[1,N]→[128,N]`
    broadcast multiply on the output, and is **never** folded into `w` (g does not sit on the
    contraction axis for this op).
    - Positive: `g` is loaded as `[1,N]` and broadcast across the partition (row) axis at the
      output multiply.
    - Negative: folding `g` into `w` before the GEMM changes `y = x@w + z` (and hence the RMS)
      and fails.
  - AC-2.4: The residual `z` is added to the GEMM output **before** the square/reduction and is
    the same `y` used in the final `y * g / rms`.
    - Positive: `y = (x@w) + z` is formed first; both the reduction input and the output scale
      read this `y`.
    - Negative: reducing only `x@w` and adding `z` afterward (or omitting `z` from the RMS
      input) fails.
  - AC-2.5: The full `y[128,N]` buffer stays live through the output scale; the square uses a
    **separate** `sq[128,N]` temp so `y` is not clobbered by the reduction path.
    - Positive: `sq = square(y)` writes a distinct buffer; `tensor_reduce` reads `sq`; the
      second chunk loop reads the original `y`.
    - Negative: writing the square in place over `y` corrupts the `y * inv_rms * g` output.

- AC-3: The weight-reload pathology is eliminated — `w` is loaded **once, outside the M-tile
  loop**, and reused across all 32 M-tiles; the profiler confirms HBM read traffic collapses
  toward the single-pass floor and latency drops materially.
  - Positive Tests (expected to PASS):
    - There is exactly one weight-load region for `w`, lexically outside the per-M-tile loop;
      no `nl.load` of `w` appears inside the M-loop.
    - The profiler `summary_metrics.hbm_read_bytes` is near the one-pass read floor
      ≈ `x(32 MB) + w(16 MB) + z(32 MB) ≈ 80 MB` (plus profiler/runtime overhead), i.e.
      dramatically below the baseline's ~537 MB weight-reload traffic.
    - Measured speedup (`baseline_latency / candidate_latency`) is materially greater than 1
      (large — see path boundaries; the exact multiple is a direction, not a hard gate).
  - Negative Tests (expected to FAIL / signal residency failure):
    - `hbm_read_bytes` remaining near the baseline's weight-reload magnitude indicates a spill
      or a reload was reintroduced → residency FAILED → fallback ladder (AC-5) triggers.
    - Any `nl.load` of `w` inside the M-tile loop is a residency regression.

- AC-4: The kernel is built on the explicit `nisa.nc_matmul` + identity-matmul transpose idiom
  (not high-level `nl.matmul`), so a later bf16x2 split is a localized change.
  - Positive Tests:
    - GEMM uses `nisa.nc_matmul(stationary=xT[kt], moving=w_sb[kt,:,chunk])`; the `x` sub-tiles
      are transposed via `nisa.nc_matmul(..., is_transpose=True, is_moving_onezero=True)` against
      a `[128,128]` identity.
    - `w` is used directly as the moving operand in its native `[k(par), n(free)]` layout.
  - Negative Tests:
    - Using `nl.matmul` (baseline high-level path) does not satisfy this criterion.

- AC-5: SBUF residency has a **pre-committed, bounded fallback ladder** — Phase 1 does not
  depend on a single tight layout and never degrades to open-ended debugging or baseline-style
  reload.
  - Positive Tests:
    - Primary path: `w` fully resident (`w_sb[16][128,N]` ≈ 128 KB/partition) with the working
      set fitting under the ~192 KB/partition target, verified by compile + AC-1 pass +
      AC-3 HBM-drop.
    - If the primary path fails to compile or HBM reads do not drop, the ladder is applied in
      order: (a) tighten buffer lifetimes / reuse buffers in place; (b) partial-`w` residency
      by N-block or K-block group that still materially cuts reload versus baseline.
    - Whichever residency path succeeds is recorded (AC-6) so later comparisons are unambiguous.
  - Negative Tests:
    - Falling back to baseline-style per-M-tile weight reloads is not an acceptable resolution.
    - Introducing bf16 / precision changes, off-PE transpose, or M-blocking as a "fix" violates
      the Phase-1 boundary.

- AC-6: Evidence is recorded for the accepted candidate.
  - Positive Tests:
    - A row is appended to `benchmark.csv` (op, candidate, parent, passed, latency_ms, speedup,
      notes).
    - An entry is appended to `candidates.jsonl` with a parent link (DAG) and a note of which
      residency path succeeded (full-`w` vs fallback).
    - A profiler digest is saved under `profile/`.
  - Negative Tests:
    - A passing candidate with no `benchmark.csv` / `candidates.jsonl` / `profile/` evidence is
      incomplete.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. This draft
specifies a largely deterministic design (fixed op semantics, fixed idiom, fixed dtype), so the
bounds are narrow and differ mainly in how SBUF pressure is resolved.

### Upper Bound (Maximum Acceptable Scope)
A single fp32 kernel `runs/matmul_add_rmsnorm_v1.py` that: loads `w` fully resident once
(outside the M-loop) via the explicit `nisa.nc_matmul` + identity-transpose idiom; per M-tile
transposes the 16 K-sub-tiles of `x`, streams all four 512-wide N-chunks accumulating over the
16 K-tiles into PSUM and writing `acc + z` into a full `[128,N]` SBUF `y` buffer, then performs
a single free-axis square+reduce over the full row, computes `inv_rms`, and applies
`y * inv_rms * g` in a second chunk loop before storing; passes all five seeds; drops HBM read
traffic to near the one-pass floor; and records benchmark/candidate/profile evidence. No
Phase-2/3 work included.

### Lower Bound (Minimum Acceptable Scope)
A correct fp32 kernel that passes all five seeds (AC-1, AC-2) and materially reduces weight-
reload HBM traffic versus baseline (AC-3) — even if, due to compiler liveness, it uses the
bounded fallback (partial-`w` residency by N/K-block group) rather than full residency. The
explicit `nc_matmul` idiom (AC-4) and evidence recording (AC-6) still hold.

### Allowed Choices
- Can use: full-`w` residency OR the bounded fallback ladder (tighten lifetimes → partial-`w`
  residency by N/K-block group); either buffer order for the output scale that is algebraically
  `y * g / rms` (e.g. `(y * inv_rms) * g_bcast`, validated by the 5-seed gate); `nisa.activation`
  for square/rsqrt with a `[128,1]` zero bias; `tensor_reduce` / `tensor_scalar` for the norm.
- Cannot use (deferred or closed): bf16 / any precision-splitting or approximate path (Phase 2);
  high-level `nl.matmul` for the GEMM; off-PE transpose — `dma_transpose` is fp32-ineligible and
  `nc_transpose` (vector) regressed, both CLOSED on siblings; M-blocking / loop reorder (Phase
  2/3, only if the profiler shows an exposed fill or DMA bubble); folding `g` into `w`.

> **Note on Deterministic Designs**: The operator semantics, dtype (fp32), transpose idiom, and
> API surface are fixed by the draft and the proven sibling. The only genuine implementation
> choice is how SBUF pressure is resolved (full vs fallback residency), which the fallback
> ladder in AC-5 makes deterministic rather than open-ended.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements. The proven sibling `add_rmsnorm_matmul_v1` is a
> near-exact structural template (it does norm→GEMM; this op does GEMM→add→norm, the mirror).

### Conceptual Approach

Preamble (once, outside the M-loop):
1. `bias_zero = zeros([par_dim(128), 1])` — portable `[128,1]` zero bias for the Scalar-Engine
   square / rsqrt activations.
2. `g_tile = load(g_tensor.reshape((1, N)))` → `[1, N]`, broadcast `[1,N]→[128,N]` at the output
   multiply (free-axis, exactly the baseline's `g_bcast`).
3. `identity_local[128,128]` from `nl.shared_constant(np.identity(128))` — the moving operand
   for the identity transpose.
4. **w fully resident:** `w_sb[kt] = [par_dim(128), N]` for `kt` in `0..15`, `nl.load` once
   (16 MB total, 128 KB/partition), used directly as the moving operand in native
   `[k(par), n(free)]` layout.

Per M-tile (`for mt in nl.affine_range(32)`):
1. `x_sb = load(x_tensor[mt*128 + ix, ik])` → `[128, K=2048]`.
2. Transpose the 16 K-sub-tiles: `xT[kt] = [k_in(par), m_in(free)=128]` via identity
   `nc_matmul(is_transpose=True, is_moving_onezero=True)` → PSUM → copy to SBUF.
3. **First chunk loop — GEMM + residual add into the full row:** for each of 4 N-chunks `c`:
   `acc = zeros([128,512], psum)`; for each of 16 K-tiles: `acc += nc_matmul(xT[kt],
   w_sb[kt, :, c*512:...])`; then load `z_tile[:, c*512:...]` and write
   `y[:, c*512:...] = acc + z_tile` into a full `[128, N]` SBUF buffer (fusing the residual add).
4. **Single fused RMSNorm over N (free axis), non-clobbering:**
   - `sq = activation(op=square, data=y, bias=bias_zero)` → separate `[128,N]` temp (leave `y`
     intact).
   - `sumsq = tensor_reduce(add, sq, axis=[1])` → `[128,1]` (one full-N reduce).
   - `mean_eps = tensor_scalar(sumsq, mul, 1/N, add, eps)` → `sum/N + eps`.
   - `inv_rms = activation(op=rsqrt, data=mean_eps, bias=bias_zero)` → `[128,1] = 1/rms`.
5. **Second chunk loop — output scale + store:** for each of 4 N-chunks `c`:
   `out_sb = (y[:, chunk] * inv_rms) * g_bcast[:, chunk]`; `nl.store(out[mt*128+ix, chunk],
   out_sb)`.

SBUF budget (per partition, ~192 KB target): `w` resident 128 + `x_sb` 8 + `xT` (16×[.,128]) 8
+ full-row `y` 8 + `sq` 8 + `g_tile`/`identity`/`[128,1]` scalars ≈ 1 (`z` loaded per chunk,
consumed into `y`, no full-`z` lifetime) ≈ **~161 KB/part**. PSUM (identity-transpose scratch,
`[128,512]` accumulator) is a **separate physical memory** (~16 KB/part, 8 banks) and does NOT
count against this SBUF weight budget. If compiler liveness/padding pushes over ~192 KB, apply
the AC-5 fallback ladder.

Residency evidence: this harness has **no** local BIR/compiler spill report (that path is out
of scope per CLAUDE.md) — the decisive evidence is the **remote profiler**'s
`summary_metrics.hbm_read_bytes` (surfaced by `verify.py`). A drop from ~537 MB toward ~80 MB
plus a large speedup confirms residency; reads staying near baseline weight traffic signal a
spill.

### Relevant References
- `../AccelOpt/NKIBench/kernels/matmul_add_rmsnorm_M4096_N2048_K2048_0.py` — the baseline
  (reload pathology; also the proven single full-N `nl.sum(..., axis=[1])` reduce).
- `../AccelOpt/NKIBench/reference/matmul_add_rmsnorm_M4096_N2048_K2048_numpy_1.py` — numpy
  oracle (`y=x@w+z`, `rms=sqrt(mean(y^2,axis=-1)+eps)`, `y*g/rms`; input draw order
  `x→w→eps→z→g`).
- `workspaces/add_rmsnorm_matmul/runs/add_rmsnorm_matmul_v1.py` — near-exact structural template
  (raw-2D I/O, fused norm, w-resident, `nc_matmul`+identity transpose, distinct `a_sb`/`sq`/
  `norm` buffers; got 3.754× on the mirror op).
- `workspaces/matmul/runs/matmul_v1.py` — the `nc_matmul` transpose idiom in isolation.
- `verify.py` — correctness oracle (`l2_norm_passed`) and profiler-metric digest
  (`mfu_estimated_percent`, `dma_active_time_percent`, `hbm_read_bytes`, `hbm_write_bytes`).
- Memory: `kda-add-rmsnorm-matmul-progress`, `kda-rmsnorm-matmul-progress`,
  `kda-matmul-progress`.

## Dependencies and Sequence

### Milestones
1. Kernel implementation (`runs/matmul_add_rmsnorm_v1.py`): depends on the reference/baseline
   read-through and the sibling template.
   - Phase A: preamble — constants, `bias_zero`, `g_tile`, `identity_local`, full-`w` load
     outside the M-loop.
   - Phase B: per-M-tile transpose of the 16 `x` K-sub-tiles.
   - Phase C: first chunk loop — GEMM over 16 K-tiles per N-chunk into PSUM, `acc + z` into the
     full `[128,N]` `y` buffer.
   - Phase D: single non-clobbering norm — `sq` temp → `tensor_reduce(axis=[1])` → `mean_eps`
     (`*1/N` then `+eps`) → `rsqrt`.
   - Phase E: second chunk loop — `y * inv_rms * g_bcast`, store.
2. Verification: depends on Milestone 1.
   - Phase A: `--fast` single-seed sanity pass.
   - Phase B: full 5-seed run (drop `--fast`).
   - Phase C: read the profiler digest — confirm `hbm_read_bytes` drop and speedup.
3. Residency resolution (only if Milestone 2 shows a spill / no HBM drop / compile failure):
   apply the AC-5 fallback ladder in order, re-verify.
4. Evidence recording: depends on a passing candidate — `benchmark.csv`, `candidates.jsonl`
   (with parent link + residency-path note), `profile/` digest.

<Dependencies are structural: verification depends on the kernel; residency resolution is
conditional on the profiler result; evidence recording depends on a passing run.>

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Write `runs/matmul_add_rmsnorm_v1.py`: preamble with full-`w` resident load outside the M-loop, `nc_matmul`+identity transpose idiom | AC-3, AC-4, AC-5 | coding | - |
| task2 | Implement per-M-tile: transpose 16 `x` K-sub-tiles; first chunk loop GEMM+`z` into full `[128,N]` `y` | AC-2.4, AC-4 | coding | task1 |
| task3 | Implement the single non-clobbering fused norm (separate `sq` temp, full-N reduce, eps-after-mean) and the second chunk loop output scale (`y*inv_rms*g`) + store | AC-2.1, AC-2.2, AC-2.3, AC-2.5 | coding | task2 |
| task4 | Verify: `--fast` then full 5-seed `verify.py`; confirm `l2_norm_passed` on all seeds | AC-1, AC-2 | coding | task3 |
| task5 | Read the profiler digest; confirm `hbm_read_bytes` near the ~80 MB one-pass floor and material speedup; if spill/no-drop/compile-fail, apply the AC-5 fallback ladder and re-verify | AC-3, AC-5 | coding | task4 |
| task6 | Record evidence: `benchmark.csv` row, `candidates.jsonl` entry (parent link + residency-path note), `profile/` digest | AC-6 | coding | task5 |
| task7 | (Conditional) If the 5-seed gate fails, invoke `kernel-accuracy-debugging` and diagnose the likely eps/mean-order or reduce-axis mistake (not precision) | AC-1, AC-2 | analyze | task4 |

## Claude-Codex Deliberation

### Agreements
- Full-`w` residency (load once outside the M-loop, reuse across all 32 M-tiles) is the correct,
  highest-value, lowest-risk Phase-1 win — it kills the baseline's ~537 MB reload traffic, the
  same pathology the mirror sibling fixed for 3.754×.
- The explicit `nisa.nc_matmul` + identity-matmul transpose idiom is the right foundation (not
  `nl.matmul`), so the Phase-2 bf16x2 split is a localized extension.
- Math fidelity is exactly pinned: reduce over the N free axis; `eps` added after the `1/N` mean;
  `g` as a length-N free-axis broadcast never folded into `w`; `z` added before the square; fp32
  throughout.
- PSUM is a separate physical memory from SBUF and must not be charged against the 128 KB/part
  weight budget; the profiler's `hbm_read_bytes` is the right (and only, in this harness)
  residency/spill evidence; a single full-`N=2048` free-axis reduce is already proven by the
  passing baseline.
- The Phase-1 fallback ladder is appropriately bounded (tighten lifetimes → partial-`w`
  residency), never drifting into bf16, off-PE transpose, loop reorder, or baseline-style reload.

### Resolved Disagreements
- Per-chunk vs full-row normalization (Round 1): Codex flagged that reducing inside each 512-wide
  N-chunk would normalize over 512 columns, not full N. Resolution: the per-M-tile structure was
  made explicit as two separate chunk loops — a first loop that GEMMs all four chunks and writes
  `acc + z` into the full `[128,N]` `y` buffer, then a **single** square+reduce over the full row,
  then a second loop that applies `y*inv_rms*g` and stores. No per-chunk normalization anywhere.
- `g_tile` omitted from the budget (Round 1): Resolution: `g_tile` (and identity / `[128,1]`
  scalars) are now itemized; budget ≈ 161 KB/part, still under ~192 KB.
- Concrete residency evidence (Round 1): Resolution: AC-3 now specifies the one-pass HBM read
  floor ≈ `x(32 MB)+w(16 MB)+z(32 MB) ≈ 80 MB` as the target, with reads staying near baseline
  weight traffic defined as a residency failure that triggers the fallback ladder.
- Non-clobbering square (Round 2): Codex noted that squaring in place would corrupt the `y`
  needed by the output scale. Resolution: the square writes a **separate** `sq[128,N]` temp
  (+8 KB/part budgeted); `tensor_reduce` reads `sq`; `y` stays live through the second chunk
  loop. This matches the proven sibling (distinct `a_sb`/`sq`/`norm` buffers).

### Convergence Status
- Final Status: `converged` (3 convergence rounds; Round 3 returned no `REQUIRED_CHANGES`, no
  `DISAGREE`, no `UNRESOLVED`).

## Pending User Decisions

All items below were substantively resolved during Phase 3–5 refinement (the draft itself
pre-answers them); none block auto-start. They are surfaced for visibility only.

- DEC-1: Is the "~3–4× speedup" a hard requirement or an optimization direction?
  - Claude Position: Optimization direction. The draft is explicit: "Phase 1 is graded on
    correctness, not the exact number"; success = full-5-seed PASS. AC-1/AC-2 are hard gates;
    AC-3's speedup is "materially > 1 / large", not a fixed multiple.
  - Codex Position: N/A - open question (Codex's `QUESTIONS_FOR_USER` asked whether a
    lower-than-ideal speedup is acceptable if full residency fails; the fallback ladder answers
    this — a correct kernel with material reload reduction is acceptable).
  - Tradeoff Summary: Treating the number as hard would over-constrain a correctness-first phase
    and conflict with the draft. Resolved as a direction; the hard gate is the rel-L2 5-seed pass.
  - Decision Status: Resolved per draft — speedup is a direction, not a hard gate.

- DEC-2: Is `192 KB/partition` the usable SBUF budget after compiler-reserved space, or only the
  architectural maximum?
  - Claude Position: Treat ~192 KB as a target, not a guarantee; the budget (~161 KB/part) leaves
    headroom, and AC-5's bounded fallback ladder handles the case where real usable SBUF is lower.
  - Codex Position: N/A - open question (Codex asked for confirmed usable SBUF; unknowable in this
    harness without local compiler reports, which are out of scope).
  - Tradeoff Summary: The decisive signal is empirical — compile success + `hbm_read_bytes` drop.
    The fallback ladder makes a tighter-than-expected budget a bounded, deterministic path rather
    than a blocker.
  - Decision Status: Resolved by design — empirical (profiler) verification + AC-5 fallback ladder.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `w_sb`, `xT`, `rmsnorm_in`/
  `y`, `sq`, `inv_rms`, `g_bcast`), matching the proven sibling's style.

--- Original Design Draft Start ---

# matmul_add_rmsnorm — Phase 1 draft (first correct NKI kernel)

## 1. Operator and contract

**Op:** `matmul_add_rmsnorm`, NKIBench case `1`. Fused dense GEMM → residual-add → RMSNorm.

**Reference** (`AccelOpt/NKIBench/reference/matmul_add_rmsnorm_M4096_N2048_K2048_numpy_1.py`):

```python
def forward(x, w, eps, z, g):
    y   = np.matmul(x, w) + z                               # GEMM then residual add
    rms = np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True) + eps)  # row RMS over N (axis=-1)
    return y * g / rms                                      # per-col g, per-row 1/rms
```

**Shapes / dtype (all fp32):**
- `x`: `(M=4096, K=2048)`
- `w`: `(K=2048, N=2048)`
- `z`: `(M=4096, N=2048)`   — residual, added to the GEMM output
- `g`: `(N=2048,)`          — learned scale along the **output** dim N
- `eps`: python float scalar (1e-5)
- output: `(M=4096, N=2048)`

**Signature (matches baseline):** `def kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor)`
(eps is the 3rd positional arg, a runtime python scalar).

**I/O layout — RAW 2D, NOT pre-tiled.** This case's `transform_to_nki_inputs` is the
IDENTITY (`return inputs`), so the kernel receives the raw 2D tensors above and returns a
raw 2D `(4096, 2048)`; the harness's `transform_nki_outputs` reshapes to the ref shape
(already 2D → identity). The kernel slice-tiles itself (`x_tensor[i*128 + ix, ik]`), exactly
like the NKIBench baseline and the sibling `add_rmsnorm_matmul` (which is also raw-2D I/O).

**Correctness gate:** relative-L2 `||v_k - v_r||_2 < 2e-5 * ||v_r||_2`, fp32, across seeds
`[0,21,42,63,84]` (`verify.py` gates on `l2_norm_passed`). Phase 1 stays **pure fp32** —
no precision tricks; correctness first.

**Score:** `baseline_latency / candidate_latency`, p50 on-device, single core,
`--disable-dge --logical-nc-config=1`. Baseline latency = **3.768493 ms** (baselines.json).

## 2. This op is the MIRROR of the rmsnorm siblings (matmul-first)

The two prior fused ops — `rmsnorm_matmul` and `add_rmsnorm_matmul` — do **norm → GEMM**.
This op does **GEMM → add → norm**, the reverse. That single reordering changes three things,
all of which make this op *structurally simpler* than the siblings:

| aspect | siblings (norm→GEMM) | this op (GEMM→add→norm) |
|---|---|---|
| reduction axis of the norm | over **K** (contraction) | over **N** (the GEMM **output** free axis) |
| does the norm need a transpose? | norm result is on the m-partition, matmul needs k-partition → **transpose the activation** | matmul output is already `[m_in(par), n(free)]`; norm reduces the **free** axis → **no norm transpose** |
| `g` placement | length **K** = contraction axis → does NOT commute past the matmul, must fold into w or apply on activation | length **N** = free axis of the output → trivial `[1,N]→[128,N]` broadcast multiply, exactly like the baseline |
| residual `z` | `x+z` **before** norm, shape `(M,K)` | `+z` **after** GEMM, shape `(M,N)`, added to the matmul result |

So the *only* transpose required is `x → xT` for the Tensor Engine (contraction-on-partition),
identical to the promoted `matmul_v1` / `add_rmsnorm_matmul_v1` idiom. The norm is a clean
free-axis reduce over the natural matmul output layout — the *easy* direction. This also means
the fully-fused SBUF structure the baseline already uses is the right shape; Phase 1's job is
to keep it and **kill the weight reload**.

## 3. Why the baseline is slow — the dominant Phase-1 win

The NKIBench baseline (`kernels/matmul_add_rmsnorm_M4096_N2048_K2048_0.py`) loads **all of w
inside the M-loop**:

```python
for i in range(M//128):            # 32 M-tiles
    for n in range(N//512):        # 4 N-chunks
        for k in range(K//128):    # 16 K-tiles
            w_tile = nl.load(w_tensor[k*128:(k+1)*128, n*512:(n+1)*512])  # reloaded 32*4*16 = 2048x
            res_psum += nl.matmul(x_tiles[:, k*128:...], w_tile)
```

That is **2048 weight loads** streaming the full 16 MB weight matrix **32 times ≈ 537 MB** of
redundant HBM reads. That reload traffic (not the compute) is why the baseline is 3.768 ms —
and it is the same pathology the siblings had (`add_rmsnorm_matmul`: baseline 1.859 ms →
w-resident v1 0.495 ms = **3.754x** from the reload fix alone).

**w is 16 MB = 128 KB/partition** (16 K-tiles × 2048 × 4 B). SBUF budget is ~192 KB/partition,
and the per-M-tile working set is ~40 KB/partition (x, xT, the assembled matmul output, z, sq —
each ≤ `[128,2048]` fp32 = 8 KB/part). **128 + 40 = ~168 KB < 192 KB → w fully resident is
feasible** (tighter than the siblings, whose K=1024 gave a 64 KB weight; here K=2048 doubles it,
so budget headroom is only ~24 KB — flag as a risk, see §7). Loading w **once** and reusing it
across all 32 M-tiles is the single biggest, lowest-risk Phase-1 win, and matches the proven
sibling structure.

## 4. Phase-1 kernel design (v1, fp32, w-resident, explicit nc_matmul)

Use the **explicit `nisa.nc_matmul` + identity-matmul transpose** idiom (proven on this remote
by `matmul_v1`, `rmsnorm_matmul_v1`, `add_rmsnorm_matmul_v1`), NOT the baseline's high-level
`nl.matmul`. Reason: the known Phase-2 win across every sibling is the **compensated bf16x2
split-matmul**, which requires operating on the transposed limbs directly through `nc_matmul`;
starting from the explicit idiom makes Phase 2 a clean extension rather than a rewrite.

**Constants:** `M_TILES=32`, `K_TILES=16` (2048/128), `N=2048`, `N_CHUNK=512` (one fp32 PSUM
bank), `N_CHUNKS=4`, `INV_N = 1/N`.

**Tensor-engine mapping.** `nc_matmul(stationary, moving) = stationary.T @ moving`, contraction
(k_in) on the **partition** axis of both, both in SBUF. We want `out[m,n] = sum_k x[m,k]·w[k,n]`:
- `w` is `[k(par), n(free)]` in HBM already → load directly as the **moving** operand.
- `x` is `[m(par), k(free)]` → k is on the free axis → transpose each `[128,128]` K-sub-tile to
  `xT[kt] = [k_in(par), m_in(free)=128]` via the identity `nc_matmul(is_transpose=True,
  is_moving_onezero=True)` idiom, then use as the **stationary** operand.
- product → `[m_in(par), n(free)]` — the natural layout for the row-wise (over-N) norm.

**Preamble (once):**
1. `bias_zero = zeros([par_dim(128),1])` for Scalar-Engine activations (square, rsqrt) — a
   `[128,1]` bias is portable across NeuronCore generations (scalar bias needs v3+).
2. `g_tile = nl.load(g_tensor.reshape((1,N)))` → `[1,N]`, broadcast `[1,N]→[128,N]` at use
   (free-axis, exactly the baseline's `g_bcast`).
3. `identity_local[128,128]` from `nl.shared_constant(np.identity(128))` — the moving operand
   for the transpose, loaded once.
4. **w fully resident:** `w_sb[kt] = [par_dim(128), N]` for kt in 0..15, `nl.load` once
   (16 MB total, 128 KB/part).

**Per-M-tile loop (`for mt in nl.affine_range(32)`):**
1. Load `x_sb = x_tensor[mt*128+ix, ik]` → `[128, K=2048]`.
2. Transpose the 16 K-sub-tiles: `xT[kt] = [k_in, m_in=128]` via identity nc_matmul → PSUM →
   copy to SBUF (mirrors `matmul_v1`).
3. **GEMM, assemble the full row into SBUF** (RMSNorm needs the whole N-row before reducing):
   `rmsnorm_in = [128, N]`; for each of 4 N-chunks `c`: `acc = zeros([128,512], psum)`; for
   each of 16 K-tiles: `acc += nc_matmul(xT[kt], w_sb[kt, :, c*512:...])`; then
   `rmsnorm_in[:, c*512:...] = copy(acc)`.
4. **Residual add:** `y = rmsnorm_in + z_tile`, where `z_tile = z_tensor[mt*128+ix, iy]`
   (`[128,N]`, free-axis `tensor_tensor` add — matches reference `y = matmul + z`).
5. **Fused RMSNorm over N (free axis), entirely in SBUF:**
   - `sq = activation(op=square, data=y, bias=bias_zero)` → `[128,N]`
   - `sumsq = tensor_reduce(add, sq, axis=[1])` → `[128,1]` (single full-N free-axis reduce)
   - `mean_eps = tensor_scalar(sumsq, mul, INV_N, add, eps)` → `sum/N + eps` (eps added AFTER
     the mean, NOT scaled by 1/N — matches `np.mean(...) + eps`)
   - `inv_rms = activation(op=rsqrt, data=mean_eps, bias=bias_zero)` → `[128,1] = 1/rms`
6. **Output scale:** `out = y * g * inv_rms`. Apply as `tmp = y * inv_rms` (per-row `[128,1]`
   `tensor_scalar`) then `out_sb = tmp * g_bcast` (per-col `[1,N]→[128,N]` `nl.multiply`), or
   the equivalent baseline order (`y * inv_rms` then `* g_bcast`). Both reproduce
   `y * g / rms` (associativity of scalar multiplies; validated by the fp32 control below).
7. `nl.store(out[mt*128+ix, iy], out_sb)`.

Return the `(M,N)` `nl.shared_hbm` output.

## 5. Correctness notes (must-match details)

- **eps placement:** `mean(y²) + eps`, eps added *after* the `/N` mean (baseline does
  `mean = square_sum / N; mean = mean + eps`). Do NOT fold eps into the `1/N` scale.
- **Reduce axis:** the norm is over **N** (`axis=-1`), which is the free axis of the assembled
  `[128,N]` tile → `tensor_reduce(axis=[1])`. (Contrast the siblings, which reduced over K.)
- **g is length N** (free/output axis) → `[1,N]` broadcast, NEVER folded into w (it does not sit
  on the contraction axis here). This is *simpler* than the sibling `add_rmsnorm_matmul`, whose
  g was length-K and needed a fold.
- **fp32 throughout** — matmul in fp32 PSUM, norm in fp32. The 2e-5 rel-L2 gate is tight even
  for pure fp32 (siblings measured on-device fp32 rel-L2 ~1.46e-5, only ~1.37x under the gate,
  because trn2 emulates fp32 matmul in multiple bf16 passes). Phase 1 must not add any extra
  precision loss; keep every intermediate fp32.
- **Input draw order** (for later offline sims, not needed to code v1): `get_inputs` draws
  `x → w → eps → z → g`; eps is a non-random 1e-5.

## 6. SBUF budget (per partition, 128 partitions)

| buffer | shape/part | KB/part |
|---|---|---|
| w resident (16 K-tiles) | 16 × 2048 fp32 | 128.0 |
| x_sb | 2048 fp32 | 8.0 |
| xT (16 × [.,128]) | 16 × 128 fp32 | 8.0 |
| rmsnorm_in / y / sq (reused) | 2048 fp32 each | ~8–24 |
| z_tile | 2048 fp32 | 8.0 |
| g_tile, identity, [128,1] scalars | small | <2 |
| **total** | | **~168–184 KB** |

Under the ~192 KB/part budget, but with less headroom than the siblings (their w was 64 KB).
If it does not fit / spills, the fallback is to **reuse buffers aggressively** (compute `sq`
in place over `y`, don't keep both `rmsnorm_in` and `y`) before considering partial-w or
M-blocking. Correctness does not depend on w being fully resident — it's a perf choice — so a
spill would only cost speed, not correctness.

## 7. What Phase 1 does NOT do (defer)

- **No bf16x2 split** (the sibling Phase-2/3 win, +28% there). Phase 1 is pure fp32.
- **No g-into-w fold / post-scale eviction refactor** — g is free-axis here so the fold is a
  no-op; the inv_rms/g eviction-fold micro-opts are a Phase-2 concern.
- **No off-PE transpose exploration** (dma_transpose fp32-ineligible, nc_transpose(vector)
  regressed — both CLOSED in sibling phase-2; do not re-explore).
- **No M-blocking / loop reorder** — Phase-2/3 levers, only if the profiler shows an exposed
  fill or DMA bubble.

## 8. Risks / open questions

- **SBUF headroom (~24 KB):** w-resident is feasible but tight (§6). If the compiler spills,
  reuse the `y`/`sq`/`rmsnorm_in` buffers in place; correctness is unaffected.
- **fp32 rel-L2 margin is thin** (~1.37x under gate on the siblings). If v1 fails the gate,
  invoke `kernel-accuracy-debugging`; the likely culprit would be an eps/mean-order or
  reduce-axis mistake, not precision (pure fp32 should pass as it did on all siblings).
- **512 identity transposes** (16/M-tile × 32) live on the PE alongside the matmul. On the
  siblings these were fully hidden under the PE-bound matmul; expected here too, but confirm
  in the Phase-2 profiler digest rather than assuming.

## 9. Deliverable & verification

- Kernel: `runs/matmul_add_rmsnorm_v1.py`, single `@nki.jit def kernel(x_tensor, w_tensor, eps,
  z_tensor, g_tensor)`, structure per §4.
- Score (from `workspaces/matmul_add_rmsnorm/`):
  ```bash
  python3 \
      ../../verify.py --op matmul_add_rmsnorm --candidate runs/matmul_add_rmsnorm_v1.py --fast
  ```
  then full 5-seed (drop `--fast`) before recording.
- Record the perf change in `benchmark.csv`, the candidate (with parent link) in
  `candidates.jsonl`, and the profiler digest under `profile/`.
- **Phase-1 success = full-5-seed PASS.** Expected speedup: large (killing 537 MB of reload
  traffic), plausibly in the ~3–4x range by analogy to `add_rmsnorm_matmul_v1` (3.754x) scaled
  for this op's 2× matmul work (K=2048 vs 1024), but Phase 1 is graded on correctness, not the
  exact number.

See sibling evidence: `workspaces/add_rmsnorm_matmul/` (raw-2D I/O + fused-norm template),
`workspaces/matmul/runs/matmul_v1.py` (nc_matmul transpose idiom), memory
`kda-add-rmsnorm-matmul-progress`, `kda-rmsnorm-matmul-progress`, `kda-matmul-progress`.

--- Original Design Draft End ---
