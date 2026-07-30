# bmm Phase 1 — First Correct fp32 NKI Kernel

## Goal Description
Produce the first CORRECT NKI/Trainium kernel for the `bmm` operator (NKIBench case 2):
batched matmul `out[b] = lhs[b] @ rhs[b]` for `b in 0..15`, fp32, with
`lhs (16,4096,64)=(B,M,K)`, `rhs (16,64,4096)=(B,K,N)`, `out (16,4096,4096)=(B,M,N)`
(B=16, M=4096, K=64, N=4096). The kernel must pass NKIBench's relative-L2 correctness
gate (rel_tol=2e-5) across all five seeds `[0,21,42,63,84]`, validated by `verify.py`.
Phase 1 prioritizes a clean, fully-understood, correct kernel over speed; performance is
captured as evidence only, with NO phase-1 performance floor.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: The kernel passes the NKIBench relative-L2 correctness gate for bmm across all
  five seeds via a full (non-fast) `verify.py` run. This is the sole phase-1 correctness gate.
  - Positive Tests (expected to PASS):
    - `verify.py --op bmm --candidate runs/bmm_v1.py` (no `--fast`) reports `l2_norm_passed=True`
      for every seed in `[0,21,42,63,84]`.
    - The single-seed `--fast` gate passes first as a cheap pre-check.
  - Negative Tests (expected to FAIL):
    - A kernel that downcasts to bf16/tf32, drops or mis-tiles K terms, or mis-maps the
      output produces `l2_norm_passed=False` on at least one seed and is rejected.
    - A kernel that fails to trace/compile never reaches a passing gate.
- AC-2: The contraction is a faithful single-pass fp32 matmul: K=64 contracted in ONE
  `nc_matmul` per output tile, with no K-accumulation loop and no lower-precision path.
  - Positive Tests (expected to PASS):
    - Each `(b, mt, c)` output tile is produced by exactly one main `nc_matmul` with K=64 on
      the partition axis of both operands.
    - Static review confirms: no `+=` accumulation over K-tiles, and the implementation uses the
      repo's established fp32 NKI path with no explicit lower-precision cast or approximate mode.
  - Negative Tests (expected to FAIL):
    - Source containing a K-tile loop with `+=` accumulation, or any bf16/tf32 cast, is rejected
      on review.
- AC-3: The layout / transpose / matmul mapping is correct for this shape.
  - Positive Tests (expected to PASS):
    - The rhs tile loads directly as `[k=64(par), n(free)]` (no transpose).
    - The lhs tile `[m=128(par), k=64(free)]` is transposed via the identity idiom to
      `[k=64(par), m=128(free)]` (`is_transpose=True`, `is_moving_onezero=True`) and copied to SBUF.
    - The main `nc_matmul(stationary=lhs_t[64,128], moving=rhs[64,512])` yields `[m=128(par), n=512(free)]`.
  - Negative Tests (expected to FAIL):
    - Feeding lhs with K on the free axis as the stationary operand, or a matmul whose partition
      axis is not the K contraction, produces wrong results or fails to trace.
- AC-4: The kernel output reshapes to the reference shape `(16,4096,4096)` row-major with full,
  non-overlapping coverage (32 M-tiles × 8 N-chunks × 16 batches).
  - Positive Tests (expected to PASS):
    - The returned tensor — direct 3D `(16,4096,4096)` OR the 4D fallback `(16,32,128,4096)` —
      reshapes to `(16,4096,4096)` and matches `np.matmul(lhs, rhs)` within the L2 gate.
    - Tile union is exact: `32·128 = 4096` rows and `8·512 = 4096` cols per batch, no gap/overlap.
  - Negative Tests (expected to FAIL):
    - An output whose tiling leaves a gap or overlap, or whose row-major order differs from the
      reference, fails the L2 gate.
- AC-5 (promotion evidence, not a correctness gate): After AC-1 passes, phase-1 bookkeeping is
  recorded so phase 2 starts from evidence. This is required for promotion/documentation, not for
  correctness acceptance.
  - Positive Tests (expected to PASS):
    - After a full run, `benchmark.csv` has the `bmm_v1` perf row, `candidates.jsonl` has the DAG
      root entry (parent = `bmm_B16_M4096_K64_N4096_0.py`), and `profile/` holds the
      MFU/PE/Vec/DMA/HBMrd/HBMwr digest.
  - Negative Tests (expected to FAIL):
    - A candidate documented as promoted with no `benchmark.csv` / `candidates.jsonl` / `profile/`
      evidence is treated as incomplete for promotion (it does not retroactively fail AC-1).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A clean single-file kernel `runs/bmm_v1.py` with a `@nki.jit def kernel(v1, v2)` entry point that:
loads a 128×128 identity into SBUF once; per batch loads `rhs[b]` resident as `[64(par),4096(free)]`
(16 KB/partition); per M-tile loads and identity-transposes the lhs tile once; streams 8 N-chunks of
width 512 producing `[128,512]` PSUM tiles copied to SBUF and stored into a direct 3D output
`(16,4096,4096)`; captures the full profiler metrics digest into `profile/` as write-bound evidence;
and records the `benchmark.csv` row and `candidates.jsonl` DAG root. No performance optimization
(no store-burst fattening, no ping-pong buffering) is done in phase 1.

### Lower Bound (Minimum Acceptable Scope)
Any single fp32 kernel that passes the 5-seed relative-L2 gate for bmm and is recorded as the DAG
root — including the documented fallbacks (4D output `(16,32,128,4096)`; rhs loaded per-chunk or in
1024-wide blocks) when the direct-3D / full-resident form fails to trace.

### Allowed Choices
- Can use: fp32 throughout via the repo's established NKI path; the proven identity-transpose idiom;
  `N_CHUNK=512` for the main matmul; either direct 3D output `(16,4096,4096)` or the 4D fallback
  `(16,32,128,4096)`; rhs resident `[64,4096]` OR per-chunk / 1024-wide rhs loads; module-level +
  in-function NKI imports (the traced convention).
- Cannot use: bf16/tf32 or any explicit lower-precision cast / approximate mode anywhere; a K-split
  accumulation loop; a single matmul with PSUM free width > 512 fp32; hand-tuning of the baseline;
  edits to any `../../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}` file.

> **Note on Deterministic Designs**: The math mapping (transpose orientation, K=64 single pass,
> N=512 chunk, output tiling) is fixed per the draft — those bounds converge. The only genuinely open
> choices are output shape (3D primary vs 4D fallback) and rhs residency granularity, and both are
> gated purely by whether the tracer accepts the cleaner form.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions,
> not prescriptive requirements.

### Conceptual Approach
```
identity_local = load 128x128 identity into SBUF        # once, reused for all transposes
for b in affine_range(16):                              # batch
    rhs_sb = load v2[b, 0:64, 0:4096] -> [64(par), 4096(free)]     # once per batch (16 KB/part)
    for mt in affine_range(32):                                    # 4096/128 M-tiles
        lhs_sb = load v1[b, mt*128:+128, 0:64] -> [128(par), 64(free)]
        # transpose lhs tile on the PE:
        lhs_t_psum = nc_matmul(lhs_sb, identity_local, is_transpose=True, is_moving_onezero=True)
                     -> PSUM [64(par),128(free)]
        lhs_t = copy(lhs_t_psum) -> SBUF [64(par),128(free)]
        for c in affine_range(8):                                  # 4096/512 N-chunks
            acc = nc_matmul(lhs_t, rhs_sb[:, c*512:+512]) -> PSUM [128(par),512(free)]
            out_sb = copy(acc) -> SBUF [128,512]
            store out[b, mt*128:+128, c*512:+512] = out_sb
return out   # (16,4096,4096); reshapes row-major to the reference shape
```
All slices become affine index expressions (`nl.arange(64)` for the K partition, `nl.arange(128)` for
M partition/free after transpose, `nl.arange(512)` for the N chunk). Allocations carry `par_dim`:
transpose PSUM/SBUF tiles as `par_dim(64)`; main PSUM and output SBUF tiles as `par_dim(128)`.

### Relevant References
- `../../AccelOpt/NKIBench/kernels/bmm_B16_M4096_K64_N4096_0.py` — proven baseline: exact
  identity-transpose call (line 34), 4D output store (line 42), rhs-once-per-batch load (line 30).
- `workspaces/matmul/runs/matmul_v1.py` — sibling dense GEMM: identity-transpose idiom, `N_CHUNK=512`,
  and a strided 3D output store `out[mt, arange(128)[:,None], n0+arange(512)[None,:]]` — structurally
  the bmm 3D store, differing only by a partition-axis offset; proven to compile and pass the L2 gate.
- `../../AccelOpt/NKIBench/reference/bmm_B16_M4096_K64_N4096_numpy_1.py` — reference: reshape-only
  `transform_to_nki_inputs`, row-major `transform_nki_outputs`.
- `verify.py` — correctness gate (`l2_norm_passed` across seeds); `--fast` runs a single seed.

## Dependencies and Sequence

### Milestones
1. Kernel implementation: write `runs/bmm_v1.py` (identity load → batch loop → M-tile transpose →
   N-chunk matmul → store). Depends on the layout/transpose mapping being fixed.
   - Phase A: identity load + batch/M/N loop nest with direct 3D output.
   - Phase B: static review checkpoint — confirm no `+=`, no K loop, no dtype cast, exactly one main
     `nc_matmul` per `(b, mt, c)`, and affine index expressions for all loads/stores.
2. Fast correctness gate: `verify.py --fast` (1 seed). If it fails to trace, apply the documented
   fallbacks in order. Depends on Milestone 1.
   - Step 1: if the direct 3D partition-offset store fails to trace, switch the output to the 4D
     baseline shape `(16,32,128,4096)` first (closest to the proven baseline).
   - Step 2: only if the rhs-resident load is the actual trace/resource issue, drop to 1024-wide or
     per-chunk rhs loads.
3. Full correctness measurement: `verify.py` without `--fast` (5 seeds); capture the metrics digest.
   Depends on Milestone 2 passing.
4. Bookkeeping (promotion evidence): append the `benchmark.csv` row, the `candidates.jsonl` DAG root,
   and write the `profile/` digest. Depends on Milestone 3.

## Task Breakdown

Each task includes exactly one routing tag (`coding` = implemented by Claude, `analyze` = via Codex).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Implement `runs/bmm_v1.py` per the mapping (identity transpose, K=64 single pass, N=512 chunks, direct 3D output) | AC-2, AC-3 | coding | - |
| task2 | Static-review checkpoint: no `+=`, no K loop, no dtype cast, one main `nc_matmul` per `(b,mt,c)`, affine index forms | AC-2, AC-3 | coding | task1 |
| task3 | Run fast (1-seed) `verify.py` gate; on trace failure apply fallbacks in order (4D output first, then rhs granularity) | AC-1, AC-4 | coding | task2 |
| task4 | Run full 5-seed `verify.py`; confirm `l2_norm_passed` on all seeds | AC-1 | coding | task3 |
| task5 | Capture profiler metrics digest into `profile/`; append `benchmark.csv` + `candidates.jsonl` root | AC-5 | coding | task4 |
| task6 | Confirm the theoretical write-bound framing against measured HBMwr/DMA%/PE% (evidence for phase 2, not a phase-1 gate) | AC-5 | analyze | task4 |

## Claude-Codex Deliberation

### Agreements
- The math mapping is correct: K=64 on the partition axis of both operands, identity-transpose lhs
  `[128,64] → [64,128]`, rhs used directly as `[64,512]` chunks, main matmul → `[128,512]`, covering
  `16 × 32 × 8` tiles.
- Phase 1 is correctness-first: no performance floor; latency and metrics are evidence only.
- fp32 only; no bf16/tf32 or approximate mode; no K-accumulation loop; PSUM main free width capped at 512.
- Pseudocode slices must become affine index expressions with `par_dim` allocations.
- The 3D-primary / 4D-fallback output decision is acceptable given the proven sibling store.

### Resolved Disagreements
- Output shape (direct 3D vs baseline 4D): Claude keeps direct 3D `(16,4096,4096)` primary because
  `matmul_v1` proves a structurally identical strided 3D store (differing only by a partition-axis
  offset); Codex preferred baseline 4D for lowest risk. Resolution: 3D primary, 4D `(16,32,128,4096)`
  as an explicit gated fallback tried FIRST if the tracer rejects the partition-offset 3D store. Both
  reshape row-major identically.
- rhs residency (`[64,4096]` resident vs 1024-wide blocks): resident kept as primary (16 KB/part,
  capacity-safe); 1024-wide / per-chunk documented as a fallback, applied only if rhs residency is the
  actual trace/resource issue. The draft's "avoids 32× reload" framing is softened — the baseline also
  loads rhs once per batch, so the win here is simplicity, not reload elimination.
- "bit-faithful to NumPy" softened to "relative-L2 safe": correctness is argued against the 2e-5 gate,
  not bit-equivalence (Trainium fp32 matmul ordering need only pass the tolerance).
- AC-5 scope: moved out of the correctness acceptance gate and reworded as promotion evidence, so a
  correct kernel is never rejected for missing bookkeeping artifacts (Codex REQUIRED_CHANGE, accepted).
- DEC-1 (verifier seed behavior): resolved as non-blocking — `verify.py` is the source of truth per
  CLAUDE.md; single-pass fp32 has no seed-dependent accumulation order, so this is not phase-1 scope.
  If seed variation is later suspicious, note it as verifier-risk evidence, not a phase-1 gate.

### Convergence Status
- Final Status: `converged` (1 convergence round; all REQUIRED_CHANGES applied, no high-impact
  disagreement remaining).

## Pending User Decisions

None. (DEC-1 on verifier seed behavior was raised by Codex and resolved as non-blocking during
convergence — see Resolved Disagreements. `verify.py` is treated as the correctness source of truth
per CLAUDE.md.)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code (batch / m-tile / n-chunk, `lhs_t`, `rhs_sb`,
  `identity_local`, etc.), matching the baseline and `matmul_v1` conventions.

--- Original Design Draft Start ---

# bmm — Phase 1 draft: first correct fp32 NKI kernel

**Operator:** `bmm` (NKIBench case 2). Batched matmul `out[b] = lhs[b] @ rhs[b]`
for `b in 0..15`. Shapes/dtype: `lhs (16,4096,64)`, `rhs (16,64,4096)` fp32 →
`out (16,4096,4096)` fp32. **B=16, M=4096, K=64, N=4096.**

Baseline kernel: `../../AccelOpt/NKIBench/kernels/bmm_B16_M4096_K64_N4096_0.py`
(measured baseline latency **2.550 ms**, from `baselines.json`).
Numpy reference: `../../AccelOpt/NKIBench/reference/bmm_B16_M4096_K64_N4096_numpy_1.py`.

Phase-1 goal: produce the **first CORRECT** NKI kernel that passes the relative-L2
gate (`||v_k − v_r||₂ < 2e-5·||v_r||₂`, fp32) across all five seeds `[0,21,42,63,84]`.
Prioritize a clean, fully-understood kernel over speed.

---

## 1. Input/output layout contract (the one thing that differs from `matmul`)

The reference's `transform_to_nki_inputs` only **reshapes** the inputs — it does
**not** pre-tile them (unlike the dense-`matmul` case, whose inputs arrive already
split into `[m_tile,128,k_tile,128]`). So the kernel consumes **natural batched
layout**:

- `v1 = lhs = (16, 4096, 64) = (B, M, K)` — for batch `b`, `v1[b]` is `[M, K]`.
- `v2 = rhs = (16, 64, 4096) = (B, K, N)` — for batch `b`, `v2[b]` is `[K, N]`.

`transform_nki_outputs` reshapes the kernel result to `ref.shape = (16,4096,4096)`
in row-major order, so the kernel may return **`out = (16, 4096, 4096) = (B, M, N)`**
directly (row-major-contiguous; reshapes to the reference shape trivially). This is
cleaner than the baseline's `(16,32,128,4096)` and reshapes identically.

**Consequence of K=64:** the contraction depth is 64 ≤ 128, so the *entire* K
dimension fits in a single Tensor-Engine pass. **There is no K-accumulation loop**
— each output tile is produced by exactly one `nc_matmul` (no `+=` over K-tiles).
This makes the kernel markedly simpler than `matmul_v1` (which loops 40 K-tiles).

## 2. Tensor-Engine mechanics (how the matmul maps to hardware)

`nisa.nc_matmul(stationary, moving) = stationary.T @ moving`, and the **contraction
dim must be on the PARTITION axis of BOTH operands**. We want
`out[m,n] = Σ_k lhs[m,k]·rhs[k,n]`, contracting over K.

- **moving = rhs tile** `[k(par)=64, n(free)]`. `v2[b]` is `[K,N]` → K is already the
  leading (partition-mappable) axis, so an rhs tile loads **directly** as
  `[k=64(par), n(free)]`. No transpose needed.
- **stationary must be** `[k(par)=64, m_in(free)=128]` so that
  `stationary.T @ moving = [m_in,k] @ [k,n] = [m_in,n]` (output partition = m_in,
  free = n).
  But `v1[b]` is `[M,K]` → a loaded lhs tile is `[m_in=128(par), k=64(free)]`, with
  K on the **free** axis. So the lhs tile must be **transposed** to `[k, m_in]`.

**Transpose idiom (identical to the baseline and `matmul_v1`, proven to compile):**
`nc_matmul(stationary=lhs_sb[m_in=128(par), k=64(free)], moving=identity[128,128],
is_transpose=True)` → `[k=64(par), m_in=128(free)]` in PSUM. This is exactly the
baseline's line-34 pattern (`v7[…128,64]` → `v8[…64,128]`). Copy that PSUM tile to
SBUF to use as the stationary operand.

- **main matmul:** `nc_matmul(stationary=lhs_t[k=64(par), m_in=128(free)],
  moving=rhs_chunk[k=64(par), n=512(free)])` → `[m_in=128(par), n=512(free)]` PSUM
  tile = the output tile. Contraction K=64 on the partition of both. ✓

**Tile sizes.** M is tiled by 128 (partition limit) → 32 M-tiles. N is chunked by
**512** (the proven fp32 moving-free width — one matmul pass; used by both the
baseline `v10` and `matmul_v1`) → 8 N-chunks. K=64 is a single untiled contraction.

## 3. Loop nest (clean, correct, write-efficient-enough for phase 1)

```
identity_local = load 128×128 identity into SBUF        # once, reused for all transposes
for b in affine_range(16):                              # batch
    rhs_sb = load v2[b, 0:64, 0:4096]  -> [64(par), 4096(free)]   # once per batch (1 MB, 16 KB/part)
    for mt in affine_range(32):                         # M-tiles (4096/128)
        lhs_sb = load v1[b, mt*128:+128, 0:64] -> [128(par), 64(free)]
        # transpose lhs tile on the PE:
        lhs_t_psum = nc_matmul(lhs_sb, identity_local, is_transpose=True) -> [64(par),128(free)]
        lhs_t = copy(lhs_t_psum) -> SBUF [64(par),128(free)]
        for c in affine_range(8):                       # N-chunks (4096/512)
            acc = nc_matmul(lhs_t, rhs_sb[:, c*512:+512]) -> PSUM [128(par),512(free)]
            out_sb = copy(acc) -> SBUF [128,512]
            store out[b, mt*128:+128, c*512:+512] = out_sb
return out   # (16,4096,4096)
```

Why load `rhs[b]` **once per batch**: `rhs[b]` is `[64,4096]` = 1 MB (16 KB per
partition on 64 partitions — trivially within trn2's ~208 KB usable/partition).
Loading it once and slicing per N-chunk avoids the 32× reload that streaming it
inside the M-loop would incur. `lhs[b]` (`[4096,64]`) is streamed one 128-row tile
at a time (can't fit 4096 rows in 128 partitions). This keeps HBM **reads** minimal
(lhs 16.8 MB + rhs 16.8 MB, each read once).

## 4. Correctness reasoning

- Every `(b, mt, c)` computes `out[b, mt·128 : mt·128+128, c·512 : c·512+512]`
  exactly `= lhs[b][those 128 rows, :] @ rhs[b][:, those 512 cols]` — a full,
  exact fp32 contraction over all K=64 (single matmul, no partial sums to
  reconcile). Union over `(mt, c)` tiles the full `[4096,4096]` output with no
  overlap or gap (32·128 = 4096, 8·512 = 4096). Union over `b` covers all 16
  batches. ⇒ bit-faithful to `np.matmul(lhs, rhs)` up to fp32 matmul rounding.
- No accumulation-order divergence (K un-split), no bf16 anywhere → error is pure
  single-pass fp32 matmul rounding, far under the 2e-5 relative-L2 gate. (The
  `matmul` sibling passes 5-seed L2 with the *same* fp32 identity-transpose idiom
  and a *deeper* 40-way K accumulation, so K=64 single-pass is strictly safer.)

## 5. Bottleneck framing (for phases 2–3; not acted on in phase 1)

Theoretical floors on trn2 (from `kernel-cost-analysis` cost model):

| Component | Floor |
|---|---|
| PE main matmuls (dst free-elements 4096·32·16 ÷ 2.40 GHz) | 0.874 ms |
| PE lhs transposes (512 × 128 free ÷ 2.40 GHz) | 0.027 ms |
| **PE total** | **~0.90 ms** |
| **HBM output write** (1.074 GB fp32 @ ~781 GB/s) | **~1.375 ms** |

⇒ **This kernel is WRITE-BOUND, not PE-bound** — the opposite regime from the
dense `matmul` sibling (which was PE-bound and won via M-blocking to cut rhs
*reloads*). Here HBM reads are tiny (33 MB total) and rhs reloads are already
eliminated, so **M-blocking is not the lever**. The output is genuinely 1.074 GB of
fp32 and the L2 gate (2e-5) forbids a bf16 output, so **~1.375 ms is a near-hard
ceiling (~1.85× over baseline)**. The phase-2/3 levers will be: (a) **write-DMA
efficiency** — accumulate a full `[128, 4096]` M-tile row in SBUF and issue one
16 KB-contiguous store per M-tile instead of eight 2 KB stores, to fatten burst
size; (b) **overlap** compute/transpose behind the output DMA (ping-pong output
SBUF buffers). Phase 1 does **not** do these — it stores per 512-chunk for maximum
clarity; it just records the measured `HBMwr` / `DMA%` / `PE%` digest so phase 2
starts from evidence.

## 6. Phase-1 validation & bookkeeping

1. Write kernel to `runs/bmm_v1.py` with a single `@nki.jit def kernel(v1, v2)`
   entry point and the module-level + in-function NKI imports (matching the
   baseline / `matmul_v1` convention that is known to trace).
2. Fast gate first (1 seed):
   ```
   python3 \
       ../../verify.py --op bmm --candidate runs/bmm_v1.py --fast
   ```
3. On PASS, run the full 5-seed measurement (drop `--fast`) before recording, and
   capture the printed `metrics:` digest (`MFU/PE/Vec/DMA/HBMrd/HBMwr`) into
   `profile/` as the phase-1 bottleneck evidence (expect write-bound: high `DMA%`,
   `HBMwr ≈ 1074 MB`, modest `PE%`).
4. Append the perf row to `benchmark.csv` and the candidate (parent =
   `bmm_B16_M4096_K64_N4096_0.py`) to `candidates.jsonl` as the DAG root.

## 7. Risks / watch-items

- **3D HBM output store**: storing an SBUF `[128,512]` tile into
  `out[b, m_slice[:,None], n_slice[None,:]]` on a `(16,4096,4096)` tensor is a
  standard strided 2D access (partition stride = 4096 elems, free stride = 1);
  `matmul_v1` does the equivalent on a 3D output. If the tracer rejects the 3D
  form, fall back to the baseline's 4D output shape `(16,32,128,4096)`.
- **`is_transpose` operand shapes**: stationary `[128(par),64(free)]`, identity
  `[128,128]` → output `[64(par),128(free)]`. Mirror the baseline's exact index
  expressions to avoid a shape/`is_moving_onezero` mismatch.
- **PSUM width**: main-matmul output `[128,512]` fp32 = 512 elems/partition, one
  PSUM bank (2048) — safe. Do not widen a single matmul past 512 fp32 free.

--- Original Design Draft End ---
