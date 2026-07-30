# add_rmsnorm_matmul — Phase 1: First Correct w-Resident fp32 NKI Kernel

## Goal Description

Implement `runs/add_rmsnorm_matmul_v1.py`: the first correct, fp32, weight-resident NKI
kernel for the fused `add_rmsnorm_matmul` operator (NKIBench case 2). The kernel computes,
in one pass, the residual add `a = x + z`, RMSNorm over the K axis with `+eps`, the per-K
learned scale `g`, and the dense GEMM against `w`, returning a raw 2D `(4096, 2048)` output.

The single dominant win over the 1.859 ms NKIBench baseline is **making `w` fully resident**:
the baseline reloads the entire 8 MB weight matrix inside its M-loop (`32 × 4 × 8 = 1024`
weight loads ≈ 256 MB of redundant HBM reads), while `w` is only 64 KB/partition and fits in
SBUF (budget ~192 KB/partition). Loading `w` once before the M-loop and reusing it across all
32 M-tiles eliminates that redundant traffic. This structure is directly adapted from the
promoted sibling kernel `workspaces/rmsnorm_matmul/runs/rmsnorm_matmul_v1.py`, with three
deltas folded in: the residual add, the `+eps` term, and the per-K scale `g`.

The critical structural difference from the sibling: `add_rmsnorm_matmul`'s
`transform_to_nki_inputs` is the **identity**, so the kernel receives **raw 2D** tensors and
must return raw 2D — it slice-tiles itself (`x_tensor[mt*128 + ix, iy]`,
`w_tensor[kt*128:(kt+1)*128, :]`) exactly as the NKIBench baseline already does, rather than
consuming pre-tiled 3D inputs like the sibling.

Phase 1's scope is **one clean, correct, w-resident fp32 kernel** that passes all 5 correctness
seeds. The known further levers — folding `g` into `w`, applying `inv_rms` as a post-scale at
PSUM eviction, and the compensated bf16x2 split-matmul — are explicitly **deferred to Phases
2 and 3**.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic
verification. Verification is via `verify.py` against the remote Trainium profiler.

- AC-1: **Correctness across all 5 seeds.** The full (non-`--fast`) `verify.py` run reports
  `l2_norm_passed` for every seed in `[0,21,42,63,84]` (relative-L2 `< 2e-5 * ||ref||_2`).
  - Positive Tests (expected to PASS):
    - `verify.py --op add_rmsnorm_matmul --candidate runs/add_rmsnorm_matmul_v1.py` (full) reports 5/5 seeds `l2_norm_passed = true`.
    - `--fast` (seed 42) reports `l2_norm_passed = true` as the quick pre-check.
  - Negative Tests (expected to be REJECTED by static/math review OR to fail verification):
    - A kernel that applies `g` AFTER the matmul (`g` does not commute past the GEMM) is mathematically wrong → rejected by static/math review; if run, fails the L2 gate.
    - A kernel that scales `eps` by `1/K` — i.e. computes `rsqrt((sumsq + eps)/K)` instead of `rsqrt(sumsq/K + eps)` — is wrong per the reference → rejected by static/math review. Note: omitting `eps` entirely (`eps = 1e-5` is tiny relative to the mean of squares of unit-normal data) may still pass the relative-L2 gate on random inputs, so `eps` correctness is enforced by static/math review, not by an expected seed failure.
    - A kernel using plain bf16 in the norm reduction or the matmul is out of Phase-1 scope → rejected by static review (a plain-bf16 GEMM would also fail the L2 gate; compensated bf16x2 is a Phase-3 lever, not permitted here).
- AC-2: **Raw 2D I/O contract.** The kernel signature is
  `kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor)` and it returns a raw 2D `(4096, 2048)`
  tensor allocated as `nl.ndarray((4096, 2048), dtype=fp32, buffer=nl.shared_hbm)`.
  - Positive Tests:
    - Output tensor has shape `(4096, 2048)` and is accepted as-is by the identity `transform_nki_outputs`.
    - Loads/stores use raw 2D affine indexing (`x_tensor[mt*128 + ix, iy]`, `w_tensor[kt*128:(kt+1)*128, :]`, `out[mt*128 + ix, c*512 + iz]`) — the same pattern the baseline compiles successfully.
  - Negative Tests:
    - Returning a 3D `(32, 128, 2048)` tensor (sibling-style) — although `transform_nki_outputs` would reshape-pass it, this violates the Phase-1 raw-2D layout contract → rejected in code review.
    - Using sibling-style pre-tiled 3D input indexing (`w[kt, ik, j]`, `x[mt, mi, k]`) is incorrect for this op's identity transform → rejected in review / fails to match input shapes.
- AC-3: **`w` is resident (loaded exactly once, before the M-loop).** No `nl.load(w_tensor...)`
  appears inside the `mt` loop.
  - Positive Tests:
    - Static read of the kernel source shows all 8 `w` tile loads occur before `for mt in ...`; zero `w` loads inside the M-loop (source-review checklist item).
    - Profiler HBM read volume is consistent with a single pass over each input — roughly `w 8 MB + x 16 MB + z 16 MB + tiny (g, identity const)` — i.e. **not** the baseline's ~256 MB+ of redundant `w` traffic. Record HBMwr as well (output write floor ~32 MB).
  - Negative Tests:
    - Any `nl.load(w_tensor...)` inside the `mt` loop fails AC-3 (this is precisely the baseline anti-pattern being removed).
    - Profiler HBMrd on the order of the baseline's ~256 MB indicates `w` is being reloaded → fails AC-3.
- AC-4: **Speedup over baseline.** Candidate p50 latency beats the baseline (1.859287 ms) by
  more than measurement noise.
  - Positive Tests:
    - Candidate p50 latency `<` baseline out of the ~2.5% noise band, recorded in `benchmark.csv` with `speedup = 1.859287 / candidate_latency > 1.0`.
    - A speedup `>= 1.5x` is treated as confirmation that the w-resident plan is delivering the expected weight-traffic reduction (see DEC-1; not a hard gate).
  - Negative Tests:
    - Candidate latency within noise of — or slower than — the baseline indicates the w-resident restructuring did not take effect (e.g. residency not held, spill) → does not satisfy AC-4.
- AC-5: **Bottleneck digest recorded for Phase-2 planning.** The profiler digest
  (PE%, MFU, Vec%, Scl%, DMA%, HBMrd, HBMwr) is captured in `benchmark.csv` and
  `candidates.jsonl` (with parent `add_rmsnorm_matmul_M4096_N2048_K1024_0.py`).
  - Positive Tests:
    - `benchmark.csv` row and `candidates.jsonl` entry both contain the per-engine digest; expectation is PE-bound (~97% PE, MFU ~46%, matching the sibling's fp32 systolic floor).
    - The recorded digest is sufficient to decide the Phase-2 opener (g-into-w fold + inv_rms post-scale eviction).
  - Negative Tests:
    - A recorded candidate lacking the per-engine digest fails AC-5 (cannot drive Phase-2 decisions).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. This draft
specifies a highly deterministic Phase-1 design (adapt the proven sibling structure to raw-2D
I/O with three well-defined deltas), so the bounds are narrow and largely converge.

### Upper Bound (Maximum Acceptable Scope)
A single fp32 kernel `runs/add_rmsnorm_matmul_v1.py` that: loads `w` fully resident once
(8 tiles of `[128, 2048]`); loads `g` once and the `[128,128]` identity once; per M-tile loads
`x` and `z`, computes `a = x + z`, the fused SBUF RMSNorm (`square → full-K reduce → mean → +eps
→ rsqrt`), the inline per-row `inv_rms` multiply, the inline per-K `g` multiply, the
identity-`nc_matmul` transpose of the 8 normalized `[128,128]` K-sub-tiles, and the fp32 PSUM
K-accumulated GEMM over 4 N-chunks of 512; stores raw 2D output. Passes all 5 seeds and is
recorded with its full profiler digest. This is the complete Phase-1 goal — no more.

### Lower Bound (Minimum Acceptable Scope)
The same kernel meeting AC-1 (5-seed correctness), AC-2 (raw-2D I/O), AC-3 (`w` resident, zero
in-loop `w` loads), and AC-4 (beats baseline out of noise), recorded with at least the core
profiler digest (AC-5). Correctness and the w-resident structure are the non-negotiable core;
a modest-but-real speedup that clears the noise band satisfies the minimum even if it lands
below the 1.5x confirmation mark.

### Allowed Choices
- Can use: fp32 throughout; the sibling `rmsnorm_matmul_v1` structure adapted to raw-2D
  self-slicing; `nisa.activation` (square, rsqrt) with a `[128,1]` zero-bias tile;
  `nisa.tensor_reduce`; `nisa.tensor_scalar`; `g_tile.broadcast_to((128,K))` (or
  `nl.broadcast_to`); the identity-`nc_matmul` transpose idiom
  (`is_transpose=True, is_moving_onezero=True`); `nisa.nc_matmul` with fp32 PSUM accumulate;
  `N_CHUNK = 512` (one fp32 PSUM bank). The SAFE normalization spelling
  (`mean = sumsq*(1/K); mean_eps = mean + eps; inv_rms = rsqrt(mean_eps)`) is the default.
- Can use (conditionally): the folded normalization `rsqrt(sumsq*(1/K) + eps)` via a runtime
  `eps` post-scale bias, **only if** that NKI spelling is proven to compile on this remote;
  otherwise the SAFE add form is used.
- Cannot use (deferred to later phases, out of Phase-1 scope): folding `g` into resident `w`;
  applying `inv_rms` as a post-scale at PSUM eviction; any bf16 (plain or compensated bf16x2)
  in the norm or the matmul; `dma_transpose` or `nc_transpose(engine=vector)` for the transpose
  (the sibling proved `dma_transpose` is fp32-ineligible and vector `nc_transpose` regresses).

> **Note on Deterministic Design**: The draft prescribes a fixed structure with only one
> genuinely open numeric choice (the SAFE-vs-folded `eps` spelling, resolved above in favor of
> SAFE unless the folded form is compile-proven). Upper and lower bounds therefore nearly
> coincide; the "Allowed Choices" above enumerate the fixed, draft-specified toolset.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

Constants: `M=4096, K=1024, N=2048`; `M_TILES=32` (4096/128), `K_TILES=8` (1024/128),
`N_CHUNK=512` (one fp32 PSUM bank in the free dim), `N_CHUNKS=4`. All dims divide evenly → no
edge tiles.

Setup (once, before the M-loop):
- `out = nl.ndarray((4096, 2048), dtype=fp32, buffer=nl.shared_hbm)`.
- `bias_zero` — a `[128,1]` zero bias tile for the Scalar-Engine activations (portable form).
- Load `g` once as `[1, K]` (`g_tensor.reshape((1,K))`), as the baseline does.
- Load the `[128,128]` identity constant into SBUF once (moving operand for the transpose idiom).
- Load `w` fully resident: `w_sb[kt] = [k_in(par)=128, n=2048]` for `kt in 0..7` from
  `w_tensor[kt*128:(kt+1)*128, :]` (8·128·2048·4B = 8 MB = 64 KB/partition).

Per M-tile `mt in affine_range(32)`:
1. Load `x_tile`, `z_tile` = `[128, 1024]` from `x_tensor[mt*128 + ix, iy]`, `z_tensor[mt*128 + ix, iy]`.
2. `a = nl.add(x_tile, z_tile)` → `[128,1024]` in SBUF.
3. Fused RMSNorm over K on `a` (SAFE spelling, mirrors the baseline exactly):
   - `sq = activation(op=nl.square, data=a, bias=bias_zero, scale=1.0)` → `[128,1024]` (fp32).
   - `sumsq = tensor_reduce(nl.add, sq, axis=[1])` → `[128,1]` (single full-1024 free reduce).
   - `mean = sumsq * (1.0/K)`  (or `sumsq / K`).
   - `mean_eps = nl.add(mean, eps)`  (runtime python-float `eps`; baseline-proven).
   - `inv_rms = nl.rsqrt(mean_eps)` → `[128,1]`.
   - **Do not** compute `rsqrt((sumsq + eps)/K)` — that scales `eps` by `1/K` and is wrong.
     The folded `rsqrt(sumsq*(1/K) + eps)` is mathematically correct but only permitted if the
     runtime-`eps`-as-post-scale-bias spelling is proven to compile; default to the SAFE form.
4. `norm = tensor_scalar(a, op0=nl.multiply, operand0=inv_rms)` → `[128,1024]` per-row scale.
5. Apply `g` inline: `g_bcast = g_tile.broadcast_to((128,K))`; `y = nl.multiply(norm, g_bcast)`
   → `[128,1024]` (free-axis multiply, obviously correct; fold-into-`w` deferred to Phase 2).
6. Transpose the 8 normalized `[128,128]` K-sub-tiles of `y` to `yT[kt] = [k_in(par), m_in(free)]`
   via the identity `nc_matmul` idiom (`nisa.nc_matmul(y[:,kt*128:...], identity,
   is_transpose=True, is_moving_onezero=True)` → PSUM → copy to SBUF). Needed because
   `nc_matmul(stationary, moving) = stationary.T @ moving` requires the contraction dim (k_in)
   on the partition axis of both operands.
7. Matmul: for `c in affine_range(4)`: `acc = zeros([128,512], psum)`; for `kt in affine_range(8)`:
   `acc += nc_matmul(stationary=yT[kt] [k_in,m_in], moving=w_sb[kt][:, 512c:512c+512])` = `[m_in,512]`.
   Then `out_sb = copy(acc)` and `nl.store(out[mt*128 + ix, c*512 + iz], out_sb)`.

SBUF budget: `w_sb` 64 KB/part + `a`/`sq`/`norm`/`y` (~4 KB each) + `yT` (8·128·4B = 4 KB) +
identity + small vectors ≪ 192 KB/part. Comfortable.

Handling `g` (the algebraic reason for the inline placement):
`out[m,n] = inv_rms[m] · sum_k ( a[m,k] · g[k] · w[k,n] )`. `inv_rms[m]` is per-row (commutes
with the matmul — a Phase-2 post-scale-eviction lever). `g[k]` is per-contraction-column and
does **not** commute past the matmul; Phase 1 applies it inline on the activation (broadcast
`[1,K]→[128,K]` along the partition axis, then free-axis multiply). Folding `g` into resident
`w` (`w'[k,n] = g[k]·w[k,n]`, a per-partition `[128,1]` scale done 8× at load instead of 32× on
the activation) is the Phase-2 opener — measure first.

### Relevant References
- `workspaces/rmsnorm_matmul/runs/rmsnorm_matmul_v1.py` — the promoted sibling kernel; direct
  structural template (w-resident load, fused SBUF RMSNorm, identity-transpose, K-accum PSUM).
  Note it takes **pre-tiled 3D** inputs; adapt to raw-2D self-slicing here.
- `AccelOpt/NKIBench/kernels/add_rmsnorm_matmul_M4096_N2048_K1024_0.py` — the NKIBench baseline;
  proves raw-2D affine self-slicing compiles and shows the runtime-`eps` `nl.add(mean, eps)`
  path, the `g_tile.broadcast_to((TILE_M, K))` idiom, and the in-loop `w`-reload anti-pattern
  being removed.
- `AccelOpt/NKIBench/reference/add_rmsnorm_matmul_M4096_N2048_K1024_numpy_1.py` — the numpy
  reference (`forward`, `get_inputs`, identity `transform_to_nki_inputs`/`transform_nki_outputs`).
- `verify.py` — scoring/gating harness (`--op`, `--candidate`, `--fast`; `l2_norm_passed` gate;
  `--fast` = seed 42 warmup3/iters20, full = seeds `[0,21,42,63,84]` warmup10/iters100).
- `workspaces/rmsnorm_matmul/benchmark.csv` — sibling evidence: v1 PE=97%/MFU=46%, the fp32
  systolic floor; dma_transpose fp32-ineligible; vector nc_transpose regresses; bf16x2 → 1.363x.

## Dependencies and Sequence

### Milestones
1. **Kernel implementation** (adapt sibling to raw-2D + three deltas):
   - Phase A: Set up the once-only resident state (`out`, `bias_zero`, `g`, identity, `w_sb`).
   - Phase B: Implement the per-M-tile body — residual add, fused RMSNorm (SAFE `+eps`), inline
     `inv_rms` scale, inline `g` multiply, identity-transpose, fp32 PSUM K-accumulated GEMM,
     raw-2D store.
2. **Correctness verification** (depends on Milestone 1):
   - Step 1: `--fast` (seed 42) pre-check for `l2_norm_passed`.
   - Step 2: full 5-seed run; all of `[0,21,42,63,84]` must pass (AC-1).
3. **Performance measurement and recording** (depends on Milestone 2 passing):
   - Step 1: full-iteration p50 latency; compute speedup vs 1.859287 ms (AC-4).
   - Step 2: capture the profiler bottleneck digest (PE/MFU/Vec/Scl/DMA/HBMrd/HBMwr) (AC-5).
   - Step 3: record the row in `benchmark.csv` and the node in `candidates.jsonl` (parent
     `add_rmsnorm_matmul_M4096_N2048_K1024_0.py`); confirm PE-bound to seed Phase-2 planning.

Dependency summary: implementation → correctness gate → performance/digest recording. Nothing
downstream is recorded until the 5-seed correctness gate passes.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Set up once-only resident state: `out` (raw 2D shared_hbm), `bias_zero`, `g` as `[1,K]`, `[128,128]` identity, and `w_sb[kt]=[128,2048]` loaded from `w_tensor[kt*128:(kt+1)*128,:]` before the M-loop | AC-2, AC-3 | coding | - |
| task2 | Implement per-M-tile body: load `x`/`z` (raw 2D), `a=x+z`, fused RMSNorm with SAFE `+eps` spelling, inline `inv_rms` multiply, inline `g_bcast` multiply | AC-1 | coding | task1 |
| task3 | Implement identity-`nc_matmul` transpose of the 8 normalized `[128,128]` K-sub-tiles and the fp32 PSUM K-accumulated GEMM over 4 N-chunks of 512, then raw-2D store to `out[mt*128+ix, c*512+iz]` | AC-1, AC-2 | coding | task2 |
| task4 | Static/math self-review checklist: `g` before matmul, `+eps` after `/K` mean, fp32 throughout, zero `nl.load(w_tensor...)` inside the `mt` loop, output shape `(4096,2048)` | AC-1, AC-2, AC-3 | coding | task3 |
| task5 | `--fast` (seed 42) then full 5-seed `verify.py`; confirm `l2_norm_passed` on all of `[0,21,42,63,84]` | AC-1 | coding | task4 |
| task6 | Full-iteration p50 measurement; compute speedup vs 1.859287 ms; capture profiler digest (PE/MFU/Vec/Scl/DMA/HBMrd/HBMwr); record `benchmark.csv` + `candidates.jsonl` | AC-4, AC-5 | coding | task5 |
| task7 | If correctness fails or the realized speedup is unexpectedly small (well below the w-resident expectation), analyze the failure/profiler digest for root cause (e.g. residency not held, SBUF spill, eps/g placement) | AC-1, AC-3, AC-4 | analyze | task6 |

## Claude-Codex Deliberation

### Agreements
- The Phase-1 plan is reasonable and appropriately scoped: fp32 throughout, raw-2D I/O,
  `w` loaded once before the M-loop, M-outer tiling, identity-`nc_matmul` transpose, fp32 PSUM
  accumulation, `N_CHUNK=512`.
- `g` must be applied on the K/contraction axis before the matmul (or folded into `w`);
  applying `g` after the matmul is wrong. Deferring the `g`-into-`w` fold to Phase 2 is correct.
- The w-resident restructuring is the right Phase-1 win; the SBUF budget (~64 KB/part for `w`,
  ample headroom for temporaries) is plausible.
- Full 5-seed `l2_norm_passed` plus latency + profiler-digest recording is the right promotion
  gate; keeping bf16, post-scale eviction, and g-folding out of scope is appropriate here.
- Correctness of `eps` and `g` placement is enforced by static/math review, not by assuming the
  L2 gate will necessarily catch every omission.

### Resolved Disagreements
- **`eps` / `1/K` ordering**: Claude's v1 stated "folding `1/K` into the rsqrt scale is wrong"
  too broadly. Codex clarified: `rsqrt((sumsq + eps)/K)` is wrong (scales `eps` by `1/K`), but
  `rsqrt(sumsq/K + eps)` is mathematically correct and only unavailable if the runtime-`eps`
  post-scale-bias spelling does not compile. **Resolution**: adopt the SAFE add-form
  (`mean = sumsq/K; mean_eps = mean + eps; inv_rms = rsqrt(mean_eps)`), which mirrors the
  baseline and is compile-proven; document the folded form as a conditional allowed choice.
- **Negative-test framing**: Claude's v1 asserted omitting `eps` or using bf16 "fails ≥1 seed".
  Codex noted `eps = 1e-5` is tiny relative to the mean-of-squares of unit-normal data, so
  omission may still pass relative-L2; plain bf16 likely fails but is better excluded by scope.
  **Resolution**: AC-1 negatives are reworded to "rejected by static/math review OR fails
  verification", with an explicit note that `eps` correctness is a review-enforced invariant.
- **AC-3 HBM read floor**: Claude's v1 said "~8 MB". Codex noted the realistic read floor is
  `w 8 MB + x 16 MB + z 16 MB + tiny`, with the key check being "not ~256 MB+ redundant `w`
  traffic". **Resolution**: AC-3 now states the realistic per-input single-pass floor and adds
  the HBMwr (~32 MB) recording.

### Convergence Status
- Final Status: `converged` (Round 2 returned `REQUIRED_CHANGES: NONE`, `DISAGREE: NONE`,
  `UNRESOLVED: NONE` after all Round-1 required changes were integrated). Two convergence rounds
  executed.

## Pending User Decisions

- DEC-1: **Phase-1 promotion / success threshold.** The draft describes "a large speedup"
  without a hard numeric bar.
  - Claude Position: Record `add_rmsnorm_matmul_v1` as the Phase-1 kernel if it passes the full
    5 seeds AND beats the 1.859287 ms baseline out of the ~2.5% noise band; treat `>= 1.5x` as
    confirmation that the w-resident plan worked (given the ~256 MB → single-pass weight-traffic
    reduction), but do not impose a hard minimum beyond beating baseline.
  - Codex Position: Agrees — promote Phase 1 on full 5-seed correctness + an out-of-noise win;
    treat `>= 1.5x` as confirmation of the intended w-resident win, not as a hard correctness
    gate. (Codex separately floated `1.2x` as a soft "working" indicator and `1.4x+` as "plan
    working" in its first pass; both converge on the same policy.)
  - Tradeoff Summary: A hard high bar (e.g. "must hit 1.5x") risks rejecting a correct kernel
    that still captured a real win but landed lower due to the fp32 systolic floor / PE-bound
    ceiling. A too-loose bar risks recording noise as progress. The recommended policy —
    5-seed PASS + out-of-noise win, with 1.5x as a confidence marker — balances both and matches
    how the sibling task recorded its kernels.
  - Decision Status: `PENDING` (recommended default: adopt Claude's/Codex's converged policy).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `w_sb`, `inv_rms`, `g_bcast`,
  `yT`, `acc`), consistent with the sibling `rmsnorm_matmul_v1` kernel's style.

--- Original Design Draft Start ---

# add_rmsnorm_matmul — Phase 1 draft (first correct NKI kernel)

## 1. Operator and contract

**Op:** `add_rmsnorm_matmul`, NKIBench case `2`. Fused residual-add + RMSNorm + dense GEMM.

**Reference computation** (`AccelOpt/NKIBench/reference/add_rmsnorm_matmul_M4096_N2048_K1024_numpy_1.py`):

```python
def forward(x, w, eps, z, g):
    y = x + z                                  # residual add
    t = np.sum(np.square(y)/K, axis=-1, keepdims=True)   # mean of squares over K
    t = (t + eps)
    y = y / np.sqrt(t)                          # RMSNorm (per-row scale)
    y = y * g                                   # per-K learned scale (g is length K)
    return np.matmul(y, w)                      # dense GEMM
```

**Shapes / dtype (all fp32):**
- `x`, `z`: `(M=4096, K=1024)`
- `w`: `(K=1024, N=2048)`
- `g`: `(K=1024,)`  — learned scale along the contraction dim
- `eps`: python float scalar (1e-5)
- output: `(M=4096, N=2048)`

**Signature (matches baseline):** `def kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor)`.

**I/O layout — RAW 2D, NOT pre-tiled.** This case's `transform_to_nki_inputs` is the
IDENTITY (returns inputs unchanged), so the kernel receives the raw 2D tensors above and
must return a raw 2D `(4096, 2048)`. This differs from the sibling `rmsnorm_matmul`, whose
reference pre-reshaped inputs to 3D `(32,128,1024)` / `(8,128,2048)`. So this kernel slice-
tiles itself (`x_tensor[i*128 + ix, iy]`), exactly like the NKIBench baseline does.

**Correctness gate:** relative-L2 `||v_k - v_r||_2 < 2e-5 * ||v_r||_2`, fp32, across seeds
`[0,21,42,63,84]`. (`verify.py` gates on `l2_norm_passed`.)

**Score:** `baseline_latency / candidate_latency`, p50 on-device, single core,
`--disable-dge --logical-nc-config=1`. Baseline latency = **1.859287 ms** (baselines.json).

## 2. Why the baseline is slow — the dominant Phase-1 win

The NKIBench baseline (`kernels/add_rmsnorm_matmul_M4096_N2048_K1024_0.py`) loads **all of w
inside the M-loop**:

```python
for i in range(32):            # M-tiles
    ...RMSNorm on this M-tile...
    for n in range(4):         # N-chunks
        for k in range(8):     # K-tiles
            w_tile = nl.load(w_tensor[k*128:(k+1)*128, n*512:(n+1)*512])   # <-- reloaded 32*4*8 = 1024x
            res_psum += nl.matmul(rmsnorm_out_tile[:, k*128:...], w_tile)
```

That is **1024 weight loads** streaming the full 8 MB weight matrix **32 times** ≈ **256 MB**
of redundant HBM reads. This is why the baseline latency (1.859 ms) is ~3.7× the sibling
`rmsnorm_matmul` baseline (0.503 ms) despite near-identical arithmetic.

**w is only 8 MB = 64 KB/partition** (SBUF budget ~192 KB/partition). Loading it **fully
resident once** and reusing it across all 32 M-tiles is the single biggest, lowest-risk win,
and it is exactly the structure my promoted sibling `rmsnorm_matmul_v1` used (1.066x there,
where the baseline was already w-efficient). Here the baseline is w-*inefficient*, so the
same structure should recover a large multiple. This is the Phase-1 kernel's core.

## 3. Relationship to the sibling `rmsnorm_matmul` (high-confidence reuse)

This op is `rmsnorm_matmul` plus a residual add and a per-K learned scale. My promoted
sibling kernel `workspaces/rmsnorm_matmul/runs/rmsnorm_matmul_v1.py` is directly adaptable.
Three deltas to fold in:

1. **Residual add** `a = x + z` before the norm. One extra `nl.load(z)` per M-tile and one
   `nl.add` (or fold into the square activation's input). Then everything (`square`, reduce,
   norm, matmul) runs on `a` instead of raw `x`.
2. **`+ eps` inside the rsqrt**: `inv_rms = rsqrt(mean_k(a^2) + eps)`. The baseline does
   `mean = square_sum / K; mean = nl.add(mean, eps); rms_reciprocal = nl.rsqrt(mean)`.
3. **Per-K learned scale `g`**: `y = normalized * g`, where `g` has length K (varies along
   the contraction axis). Handling in §5.

Everything else — w-resident load, per-row fused RMSNorm, identity-matmul transpose of the
normalized activation to put the contraction dim on the partition axis, K-accumulate into a
`[128,512]` fp32 PSUM bank, N in 4 chunks of 512 — carries over verbatim. Off-PE transpose
routes and the fp32-rate ceiling were already fully explored in the sibling's phases 2-3;
Phase 1 here only needs a clean, correct, w-resident kernel.

## 4. Tiling plan (M-outer, w-resident) — the Phase-1 kernel

Constants: `M=4096, K=1024, N=2048`; `M_TILES=32` (4096/128), `K_TILES=8` (1024/128),
`N_CHUNK=512` (one fp32 PSUM bank in the free dim), `N_CHUNKS=4`. All dims divide evenly →
no edge tiles.

Setup (once):
- `out = nl.ndarray((4096, 2048), dtype=fp32, buffer=nl.shared_hbm)` — 2D, matching the
  identity output transform.
- Load `g` once into SBUF as `[1, K]` (`g_tensor.reshape((1,K))`), like the baseline.
- Load identity `[128,128]` const into SBUF once (moving operand for the transpose idiom).
- Load `w` fully resident: `w_sb[kt] = [k_in(par)=128, n=2048]` for `kt in 0..7`
  (8·128·2048·4B = 8 MB = 64 KB/partition), from `w_tensor[kt*128:(kt+1)*128, :]`.

Per M-tile `mt in affine_range(32)`:
1. Load `x_tile`, `z_tile` = `[128, 1024]` from `x_tensor[mt*128 + ix, iy]`,
   `z_tensor[mt*128 + ix, iy]`.
2. `a = nl.add(x_tile, z_tile)`  → `[128,1024]` in SBUF.
3. **Fused RMSNorm over K** on `a`:
   - `sq = activation(op=nl.square, data=a, bias=bias_zero, scale=1.0)` (Scalar Engine, fp32).
   - `sumsq = tensor_reduce(nl.add, sq, axis=[1])` → `[128,1]` (single full-1024 free reduce).
   - `mean = sumsq / K`; `mean = nl.add(mean, eps)`; `inv_rms = nl.rsqrt(mean)` → `[128,1]`.
     (Mirror the baseline's runtime-`eps` path to avoid the v2/v3 scalar-bias portability
     question. Optionally fold `1/K` into the rsqrt `scale` as the sibling did, keeping the
     `+eps` as a separate add on `sumsq` — verify equivalence; keep the simple form if unsure.)
   - `norm = tensor_scalar(a, op0=nl.multiply, operand0=inv_rms)` → `[128,1024]` per-row scale.
4. **Apply g** (per-K scale) — see §5. Phase-1 choice: `g_bcast = g_tile.broadcast_to((128,K))`
   then `y = nl.multiply(norm, g_bcast)` → `[128,1024]` (baseline's approach, free-axis, safe).
5. **Transpose** the 8 normalized `[128,128]` K-sub-tiles of `y` to `yT[kt] = [k_in(par),
   m_in(free)]` via the identity nc_matmul idiom (`nisa.nc_matmul(y[:,kt*128:...],
   identity, is_transpose=True, is_moving_onezero=True)` → PSUM → copy to SBUF). Needed because
   `nc_matmul(stationary, moving) = stationary.T @ moving` requires the contraction dim (k_in)
   on the partition axis of both operands.
6. **Matmul**: for `c in affine_range(4)`: `acc = zeros([128,512], psum)`; for `kt in
   affine_range(8)`: `acc += nc_matmul(stationary=yT[kt] [k_in,m_in], moving=w_sb[kt]
   [k_in, 512c:512c+512])` = `[m_in,512]`. Then `out_sb = copy(acc)` and
   `nl.store(out[mt*128 + ix, c*512 + iz], out_sb)`.

SBUF budget check: w_sb 64 KB/part + a/sq/y/norm (~4 KB each) + yT (8·128·4B=4 KB) +
identity + small vectors ≪ 192 KB/part. Fine.

## 5. Handling `g` — decision for Phase 1 vs. a Phase-2 lever

`out[m,n] = inv_rms[m] · sum_k ( a[m,k] · g[k] · w[k,n] )`.

- `inv_rms[m]` is **per-row** (indexed by m, the output partition) → it commutes with the
  matmul (a scalar per output row). The sibling proved a **post-scale eviction fold** (apply
  it via `tensor_scalar` reading PSUM at eviction) is exact to ~4.8e-7. That's a Phase-2
  micro-lever; **Phase 1 applies it inline** on the normalized activation (simplest, correct).
- `g[k]` is **per-contraction-column** (indexed by k) → it does **NOT** commute past the
  matmul. Two correct placements:
  - **(Phase-1) on the activation**: `y = norm * g_bcast`, `g_bcast` = broadcast of the
    `[1,K]` g-row to `[128,K]` along the partition axis (baseline's approach). Free-axis
    multiply, obviously correct.
  - **(Phase-2 lever) fold g into resident w once**: `w'[k,n] = g[k]·w[k,n]`. Since w_sb tiles
    are `[k_in(par), n]`, g is a per-partition `[128,1]` scale → one `tensor_scalar` per w-tile
    at load time, done 8× total instead of 32× on the activation. Combined with inv_rms
    post-scale eviction, the per-M-tile inner work drops to add + norm + transpose + matmul.
    Defer to Phase 2 (measure first).

**Phase 1 = inline `g_bcast` multiply** for a clean, obviously-correct baseline. Record the
g-into-w + post-scale-eviction fold as the Phase-2 opener.

## 6. Primitives (all proven on this remote in the sibling task)

- `nisa.activation(op=nl.square, ...)` and `nisa.activation(op=nl.rsqrt, ...)` — Scalar Engine,
  fp32; `output = op(data*scale + bias)`. Use a `[128,1]` zero-bias tile for portability.
- `nisa.tensor_reduce(nl.add, ..., axis=[1])` — full-K free-axis reduce → `[128,1]`.
- `nisa.tensor_scalar(data, op0=nl.multiply, operand0=inv_rms[128,1])` — per-row free-axis
  broadcast scale.
- `nl.broadcast_to(g_tile, (128,K))` — `[1,K]`→`[128,K]` partition broadcast for the g multiply
  (or use baseline's `g_tile.broadcast_to((128,K))`).
- `nisa.nc_matmul(..., is_transpose=True, is_moving_onezero=True)` with a `[128,128]` identity
  — transpose idiom (PROVEN; the sibling confirmed dma_transpose is fp32-ineligible and
  nc_transpose(vector) regresses, so the identity-matmul transpose is the right Phase-1 choice).
- `nisa.nc_matmul(stationary, moving)` = `stationary.T @ moving`, fp32 accumulate in PSUM.

All are the exact ops used in the promoted `rmsnorm_matmul_v1`.

## 7. Correctness risks & mitigations

- **eps placement**: must be added AFTER the `/K` mean and BEFORE the sqrt/rsqrt
  (`rsqrt(mean + eps)`). Mirror the baseline's `nl.add(mean, eps)` exactly.
- **g axis**: g is length-K = the contraction axis. On the activation it broadcasts along the
  free (K) axis of `[128,K]` — matches `g_bcast[128,K]`. If ever folded into w it is a
  per-partition `[128,1]` scale (k_in on w's partition axis). Do not confuse with a per-row scale.
- **Draw order / seeds**: the profiler fixes seed 42 for the input draw (adapter note); the
  reference draws `x, w, eps, z, g` in that order in `get_inputs`. The kernel is agnostic to
  draw order (it just consumes the tiled args), so this is only a correctness-of-comparison
  concern the harness already handles.
- **Output layout**: return raw 2D `(4096,2048)`; `transform_nki_outputs` reshapes to ref
  shape (identity here). Store with `out[mt*128 + ix, c*512 + iz]` indexing.
- Validate on `--fast` (seed 42) first, then full 5-seed before recording. Expect PE-bound,
  correct on all seeds (fp32 throughout; no precision shortcuts in Phase 1).

## 8. Deliverable & how it's scored

- Kernel file: `runs/add_rmsnorm_matmul_v1.py`.
- Validate/score (from `workspaces/add_rmsnorm_matmul/`):
  ```bash
  python3 \
      ../../verify.py --op add_rmsnorm_matmul --candidate runs/add_rmsnorm_matmul_v1.py --fast
  # then drop --fast for the full 5-seed / higher-iter measurement before recording.
  ```
- Record the perf change in `benchmark.csv`; add the candidate to `candidates.jsonl` with
  parent `add_rmsnorm_matmul_M4096_N2048_K1024_0.py`; read the profiler digest (PE/MFU/Vec/
  Scl/DMA/HBM) to confirm the bottleneck for Phase 2.

## 9. Expected outcome

First correct fp32 kernel passing all 5 seeds, with a large speedup over the 1.859 ms
baseline driven almost entirely by making w resident (eliminating ~256 MB of redundant weight
reloads). The kernel should land PE-bound (fp32 systolic floor, MFU ~46% like the sibling),
setting up Phase 2 (g-into-w fold + inv_rms post-scale eviction) and the Phase-3 compensated
bf16x2 split (the sibling's +28% surprise win, gated on the 2e-5 L2 tolerance holding under
K-averaging — which it did there at ~4.5e-6).

--- Original Design Draft End ---
