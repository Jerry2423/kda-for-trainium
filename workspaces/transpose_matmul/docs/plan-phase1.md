# transpose_matmul — Phase 1 Plan: First Correct NKI Kernel (`runs/tmm_v1.py`)

## Goal Description

Deliver `runs/tmm_v1.py`: a fully correct fp32 NKI/Trainium kernel for `transpose_matmul` (`out = lhs^T @ rhs`, shapes lhs `(K,M)=(2048,4096)` K-major, rhs `(K,N)=(2048,10944)`, out `(M,N)=(4096,10944)`, ~9.17e10 MACs) that passes the NKIBench relative-L2 correctness gate across all five seeds `[0,21,42,63,84]` at roughly baseline latency (baseline = 4.849615 ms).

This is a **correctness-first** phase. The kernel exploits the fact that NKIBench's `transform_to_nki_inputs` reshapes both operands `(K,·) -> (128,16,·)`, placing the contraction dimension K on the partition axis of both `v1` (lhs) and `v2` (rhs). Because `nisa.nc_matmul(stationary, moving)` computes `stationary.T @ moving` with the contraction on the partition axis of both operands, `lhs^T @ rhs` is computed **with no explicit transpose stage** — the transposed-lhs operand is a free byproduct of the input layout. The design is a clean M-block-outer streaming GEMM using `N_CHUNK=456` (an exact divisor of N that is ≤ 512, the fp32 PSUM bank width) so that **every tile is full-size and no tail-masking arithmetic exists anywhere** — eliminating the single largest correctness-bug surface.

Speed is explicitly **not** the objective here; the real speedup levers (bf16×2 compute split, shape specialization) belong to later phases. Phase 1's job is a correct, clean, PE-bound baseline plus a saved profiler digest that becomes the round-0 bottleneck reference for phase 2.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. Correctness is judged by NKIBench's relative-L2 gate (`||v_k − v_r||₂ < 2e-5·||v_r||₂`), NOT by allclose or bit-identity to numpy — `verify.py` gates on `l2_norm_passed`.

- AC-1: The kernel passes the NKIBench relative-L2 correctness gate on the full seed set.
  - Positive Tests (expected to PASS):
    - `python3 ../../verify.py --op transpose_matmul --candidate runs/tmm_v1.py` (full run) reports `l2_norm_passed = true` for every seed in `[0,21,42,63,84]`.
    - The recorded per-seed relative-L2 values sit comfortably below the `2e-5` gate (a faithful reordered-K fp32 GEMM lands far inside the gate).
  - Negative Tests (expected to FAIL):
    - A variant that feeds `v1`/`v2` slices in the wrong operand roles (e.g. swapping stationary/moving so the contraction is not on the partition axis) fails the gate.
    - A variant that mis-maps the output index (e.g. writes tile `s` to `out[8*mb + s ± offset, ...]`) fails the gate.
  - AC-1.1: Correctness is defined by relative-L2, not bit-identity to numpy.
    - Positive: Recorded rel-L2 is > 0 but ≪ `2e-5` (reordered K-accumulation vs the numpy reference is expected and acceptable).
    - Negative: A criterion demanding literal bit-exact equality with `np.matmul(lhs.T, rhs)` would be rejected as an incorrect success definition (the accumulation order legitimately differs).

- AC-2: No-mask / no-out-of-bounds invariant — every tile is full-size.
  - Positive Tests (expected to PASS):
    - The generated kernel source contains **no** `mask=` tail arithmetic (no `... >= 0` bound expressions) on any `nl.load`, `nisa.nc_matmul`, `nl.copy`, or `nl.store`.
    - All tile extents divide their axes exactly: `M = 4096 = 128·32`, `K = 2048 = 128·16`, `N = 10944 = 456·24`; every load/store slice is fully in-bounds.
  - Negative Tests (expected to FAIL):
    - Choosing an `N_CHUNK` that does not divide `10944` (e.g. 512) without reintroducing a mask would produce an out-of-bounds slice on the final chunk — rejected.
    - Any masked partial-tile path reappearing in the kernel violates this criterion.

- AC-3: fp32 numeric path with correct PSUM accumulation semantics.
  - Positive Tests (expected to PASS):
    - All `nl.load`, PSUM accumulators, `nl.copy`, and `nl.store` use `np.float32`; there are no dtype casts (no bf16 or other narrowing anywhere in the kernel).
    - The per-`(mb, c, s)` PSUM accumulator is zero-initialized before its 16-tile `kt` loop (via `nl.zeros(...)` on the PSUM buffer, or an equivalent guaranteed-zero start), so no stale PSUM contents are reused.
    - The `acc += nisa.nc_matmul(...)` over the 16 `kt` tiles accumulates in PSUM (Tensor-Engine accumulation), producing the full K=2048 contraction, followed by exactly one PSUM→SBUF copy and one SBUF→HBM store per output tile.
  - Negative Tests (expected to FAIL):
    - A variant that omits the PSUM zero-initialization (reusing a rotated bank's stale values) fails the correctness gate on at least one seed.
    - A variant that copies each `kt` partial to SBUF and re-adds there (PSUM→SBUF round trips instead of PSUM accumulation) is rejected as not matching the design.

- AC-4: Resource budget respected — no spill, HBM traffic near the expected floor (evidence-based).
  - Positive Tests (expected to PASS):
    - Static budget holds: resident `lhs_blk` `[128,16,1024]` fp32 = 64 KB/partition + `rhs_chunk` `[128,16,456]` fp32 ≈ 28.5 KB/partition (+ small temps) ≈ 93 KB/partition, under the 192 KB SBUF budget even with rhs double-buffering; PSUM uses at most a few 456-wide banks (well under 8).
    - The saved profiler digest shows no evidence of SBUF/PSUM spill or unexpected memory amplification, and HBM read/write bytes are consistent with the once-lhs / ~4×-rhs / once-out model (rhs re-read factor ≈ 4). The static byte model (lhs ≈ 33.5 MB×1, rhs ≈ 89.6 MB×~4 ≈ 0.36 GB, out ≈ 179 MB×1) is recorded alongside the measured counters.
  - Negative Tests (expected to FAIL):
    - An m-tile-outer variant that re-reads rhs 32× (~2.9 GB) would push measured HBM traffic and DMA time far above the model and toward the PE floor — rejected as the phase-1 structure.
    - Profiler counters showing spill or HBM bytes materially above the ~4×-rhs model indicate the residency assumption broke and must be investigated before promotion.

- AC-5: Latency is at roughly baseline (correctness-first; exact promotion band is a user decision — see DEC-1).
  - Positive Tests (expected to PASS):
    - The full (non-`--fast`) run records a median latency, and that latency is at roughly baseline order (near 4.849615 ms), consistent with a compute-bound fp32 GEMM — i.e. not pathologically slow.
    - The recorded speedup vs baseline is reported in `benchmark.csv` regardless of value.
  - Negative Tests (expected to FAIL):
    - A candidate that is correct but many times slower than baseline (e.g. from 32× rhs re-reads becoming DMA-bound) is flagged and not promoted as the phase-1 baseline.
    - If DEC-1 resolves to a bounded band (e.g. ≤ 1.25× baseline), a candidate above that band fails promotion.

- AC-6: Evidence artifacts are produced (reproducibility deliverable of the KDA workflow, after correctness passes).
  - Positive Tests (expected to PASS):
    - `benchmark.csv` gains a row (`timestamp,op,candidate,parent,passed,latency_ms,speedup,notes`) for `tmm_v1`.
    - `candidates.jsonl` gains a DAG entry with `parent = baseline:transpose_matmul_M4096_K2048_N10944_0.py`, plus `seeds`, `latency_ms`, `baseline_latency_ms=4.849615`, `speedup`, `rel_l2_gate=2e-5`, and a `structure` note.
    - The profiler digest (MFU / PE / Vec / Scl / DMA / HBMrd / HBMwr, as printed by `verify.py`) is saved under `profile/`.
  - AC-6.1: The profiler digest is inspected to identify the dominant engine (see DEC-2 for whether PE-dominance is a hard gate or a recorded diagnostic).
    - Positive: The digest is read and the dominant engine is recorded; if PE is dominant it confirms the compute-bound hypothesis and sets the phase-2 lever (bf16×2).
    - Negative: Promoting `tmm_v1` as the phase-2 baseline **without** having saved and inspected the round-0 profiler digest is rejected (phase 2 has no bottleneck reference to work against).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A single clean `runs/tmm_v1.py` implementing the M-block-outer streaming GEMM exactly as specified (no transpose, `N_CHUNK=456`, single-bank PSUM accumulation over 16 `kt` tiles, `M_BLK=8`), fully documented with an in-code correctness/residency rationale comment, passing all five seeds, with `benchmark.csv` + `candidates.jsonl` + `profile/` digest recorded and the dominant engine identified. Optionally includes a recorded static HBM-byte model alongside the measured counters. This is the complete phase-1 deliverable without over-engineering (no bf16, no shape sweeps, no masking).

### Lower Bound (Minimum Acceptable Scope)
The same `runs/tmm_v1.py` M-block-outer no-transpose kernel passing all five seeds on the relative-L2 gate, with the `benchmark.csv` row, the `candidates.jsonl` DAG entry, and the saved `profile/` digest. This is the least that satisfies correctness (AC-1–AC-3), the resource/latency sanity (AC-4, AC-5), and the evidence requirement (AC-6).

### Allowed Choices
This is a highly **deterministic** design; the draft fixes the structure and constants. The upper and lower bounds nearly converge.
- Can use: `nisa.nc_matmul(stationary, moving)` with the K partition-axis layout; fp32 PSUM accumulation via `acc += nc_matmul(...)`; `nl.zeros`/`nl.ndarray` PSUM/SBUF buffers; `affine_range` loops; the documented constants `M_TILES=32, K_TILES=16, M_BLK=8, M_BLOCKS=4, N_CHUNK=456, N_CHUNKS=24`.
- Tunable (explicit knob, correctness-neutral): `M_BLK` — `8` is the phase-1 default (matches baseline granularity); `16` fits SBUF and would halve rhs traffic and is a documented follow-up experiment only after v1 correctness lands (not required for phase 1).
- Cannot use: any tail-masking arithmetic (`mask=... >= 0`); any dtype cast (bf16 or otherwise) — fp32 end to end; any explicit transpose / identity-matmul stage (the layout makes it unnecessary); an m-tile-outer structure that re-reads rhs 32× (would go DMA-bound).

> **Note on Deterministic Designs**: The draft specifies a fixed approach with essentially no open implementation choices, so the path boundaries are intentionally narrow. The only genuine knob (`M_BLK`) is pinned to `8` for phase 1. Promotion-policy questions that are not implementation choices are carried as user decisions (DEC-1, DEC-2), not as boundary options.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The layout insight (verified numerically: `reshape(128,16,·)[k_in,kt,·] == orig[k_in*16+kt,·]`, i.e. `k = k_in*16 + kt`) means K is already on the partition axis of both `v1` and `v2`, and `k_in(128) × kt(16) = 2048 = K` reconstructs the full contraction. One possible implementation path (from the draft):

```
out = nl.ndarray((32, 128, 10944), fp32, shared_hbm)   # = v3

for mb in affine_range(4):                       # 4 m-blocks of 1024 rows
    lhs_blk = sbuf[par_dim(128), 16, 1024]       # 64 KB/part, loaded once per mb
    for kt in affine_range(16):
        lhs_blk[:, kt, :] = load(v1[:128, kt, 1024*mb : 1024*mb+1024])

    for c in affine_range(24):                    # 24 n-chunks of 456
        rhs_chunk = sbuf[par_dim(128), 16, 456]   # ~28.5 KB/part, once per (mb,c)
        for kt in affine_range(16):
            rhs_chunk[:, kt, :] = load(v2[:128, kt, 456*c : 456*c+456])

        for s in affine_range(8):                 # 8 m-subtiles in the block
            acc = nl.zeros(par_dim(128), 456, psum)          # zero-init PSUM
            for kt in affine_range(16):
                acc += nc_matmul(lhs_blk[:, kt, 128*s:128*s+128],  # stationary [k_in,128]
                                 rhs_chunk[:, kt, :])               # moving     [k_in,456]
            out_sb = copy(acc -> sbuf)                        # [128, 456]
            store(out[8*mb + s, :128, 456*c : 456*c+456], out_sb)
```

The inner matmul yields `stationary.T @ moving = [m_sub=128, k_in] @ [k_in, 456] = [m_sub=128, 456]`, accumulated over 16 `kt` tiles. Output tile `[m_sub(par), n(free)]` stores into `out[8*mb+s, :, 456*c : 456*c+456]`, matching `v3[m_tile, m_in, n]`. The logical row of output partition `mi` in tile `s` of block `mb` is `m = mb*1024 + s*128 + mi` — this index correspondence should be explicitly checked (it is what the relative-L2 gate ultimately verifies).

M-block-outer (vs the dead-simple m-tile-outer) is chosen so rhs is re-read only 4× (~0.36 GB) instead of 32× (~2.9 GB → ~3.7 ms DMA, dangerously close to the PE floor), keeping the kernel comfortably PE-bound.

### Relevant References
- `workspaces/transpose_matmul/docs/draft-phase1.md` — the source design (preserved at the bottom of this plan).
- `../AccelOpt/NKIBench/reference/transpose_matmul_M4096_K2048_N10944_numpy_1.py` — numpy reference (`forward` = `np.matmul(lhs.T, rhs)`; `transform_to_nki_inputs` does the `(K,·)->(128,16,·)` reshape).
- `../AccelOpt/NKIBench/kernels/transpose_matmul_M4096_K2048_N10944_0.py` — the baseline kernel (uses `N_CHUNK=1024`/2×512 PSUM banks WITH mask tail arithmetic; the phase-1 design deliberately replaces this with exact-divisor `N_CHUNK=456` and zero masks).
- `workspaces/bmm/runs/bmm_v2.py` — sibling promoted fp32 GEMM confirming the real NKI idioms: `nc_matmul(stationary, moving) = stationary.T @ moving`, PSUM accumulation, `nl.copy` PSUM→SBUF, `nl.store`, resident-operand blocking, SBUF/PSUM budgeting.
- `workspaces/matmul/runs/matmul_v1.py` — the m-tile-outer contrast (transpose-per-tile; the design here avoids both the transpose and the 32× rhs re-read).
- `verify.py` (`--op transpose_matmul --candidate <path> [--fast]`) — correctness gate on `l2_norm_passed` + prints the profiler digest (MFU/PE/Vec/Scl/DMA/HBMrd/HBMwr) from the remote profiler's `summary_metrics`.
- `baselines.json` (`transpose_matmul[case=2]`) — baseline kernel + latency `4.849615 ms`.

## Dependencies and Sequence

### Milestones

1. Milestone A — Implement `runs/tmm_v1.py` (the correct no-transpose kernel).
   - Phase A: Encode constants and the `out = nl.ndarray((32,128,10944), fp32, shared_hbm)` output; set up the M-block-outer loop nest with resident `lhs_blk` load.
   - Phase B: Implement the per-chunk `rhs_chunk` load, the per-`s` zero-initialized PSUM accumulator, the 16-tile `kt` `nc_matmul` accumulation, and the single copy→store, with the output-index correspondence documented.

2. Milestone B — Verify correctness (depends on Milestone A).
   - Step 1: Iterate with `verify.py --fast` (1 seed) until `l2_norm_passed = true`; if a seed fails, diagnose via the accuracy-debugging methodology (layout/index/zero-init/accumulation) before changing code.
   - Step 2: Confirm the no-mask / no-cast invariants (AC-2, AC-3) by inspecting the generated kernel.

3. Milestone C — Promote and record evidence (depends on Milestone B passing).
   - Step 1: Run the full (non-`--fast`) 5-seed verification; record per-seed rel-L2 and median latency.
   - Step 2: Write the `benchmark.csv` row and the `candidates.jsonl` DAG entry (parent = baseline).
   - Step 3: Save the `profile/` digest and identify the dominant engine (confirm PE-dominance if present) as the round-0 bottleneck reference for phase 2.

Dependency summary: Milestone A → B → C is strictly sequential (correctness must pass before promotion/evidence). Within A, Phase A precedes Phase B. DEC-1 and DEC-2 gate only the *promotion* decision in Milestone C, not the implementation in A/B.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Implement the M-block-outer no-transpose GEMM skeleton in `runs/tmm_v1.py`: constants, `shared_hbm` output, resident `lhs_blk` load, `affine_range` loop nest | AC-2, AC-3 | coding | - |
| task2 | Implement the inner compute: per-`(mb,c)` `rhs_chunk` load, per-`s` zero-initialized PSUM accumulator, 16-tile `kt` `nc_matmul` PSUM accumulation, single PSUM→SBUF copy and SBUF→HBM store, with the `m = mb*1024 + s*128 + mi` output-index correspondence documented in-code | AC-1, AC-3 | coding | task1 |
| task3 | Verify no-mask / no-cast / single-copy invariants by inspecting the generated kernel source (no `mask=`, no dtype cast, no transpose/identity stage) | AC-2, AC-3 | coding | task2 |
| task4 | Iterate correctness with `verify.py --fast` until `l2_norm_passed=true`; diagnose any seed failure via the layout/index/zero-init/accumulation methodology | AC-1 | coding | task2 |
| task5 | Run full 5-seed verification; record per-seed rel-L2, median latency, and speedup | AC-1, AC-5 | coding | task4 |
| task6 | Write `benchmark.csv` row and `candidates.jsonl` DAG entry (parent = baseline); save `profile/` digest | AC-6 | coding | task5 |
| task7 | Inspect the profiler digest, record the dominant engine and static HBM-byte model, and confirm no spill / ~4×-rhs traffic; establish the round-0 bottleneck reference for phase 2 | AC-4, AC-6.1 | analyze | task6 |

## Claude-Codex Deliberation

### Agreements
- The no-transpose mapping is sound: the verified reshape contract `k = k_in*16 + kt` puts K on the partition axis of both operands, so `nc_matmul(stationary, moving) = stationary.T @ moving` computes `lhs^T @ rhs` with the full K=2048 contraction reconstructed by accumulating over the 16 `kt` tiles — no explicit transpose stage needed.
- Exact tiling removes tail masks on all three axes (`M=4096=32·128`, `K=2048=16·128`, `N=10944=24·456`); `N_CHUNK=456` (≤ 512 fp32 PSUM width, exact divisor of N) is a defensible correctness-first choice, and eliminating mask arithmetic removes the largest correctness-bug surface.
- Correctness-first is the right phase-1 stance; latency is secondary provided the kernel is not pathologically slow.
- `M_BLK=8` as default with `M_BLK=16` deferred to later tuning is reasonable.
- Full 5-seed relative-L2 must gate promotion; `--fast` (1 seed) is iteration-only.

### Resolved Disagreements
- **"Bit-exact fp32" claim (Codex first + second pass vs draft §5)**: Codex flagged the draft's "bit-exact fp32" / "~0 rel-L2" language as too strong — the K-accumulation order differs from numpy, so equality should not be assumed. Resolution: the plan defines correctness strictly as relative-L2 within the `2e-5` gate (AC-1, AC-1.1), expects a value "comfortably below `2e-5`" (not "~0"/bit-identical), and requires recording actual per-seed rel-L2. Rationale: matches how NKIBench actually gates (`l2_norm_passed`) and how prior matmul-family kernels passed with reordered K-accumulation.
- **AC-4 no-spill / HBM-traffic claim (Codex second pass)**: Codex objected that "no spill" and the 4×-rhs traffic model are not guaranteed by the static ~93 KB/partition estimate alone. Resolution: AC-4 is now evidence-based — it requires the saved profiler counters to corroborate no-spill and the ~4×-rhs model, and records the static byte model alongside the measured counters. Rationale: the harness measures HBM/DMA on the remote profiler, so the claim should be verified, not asserted.
- **Output-index verification (Codex second pass)**: Codex required explicitly verifying that `out[8*mb+s, :, 456*c:456*c+456]` corresponds to logical rows `m = mb*1024 + s*128 + mi`, not just tile-ordinal syntax. Resolution: added to task2 (documented in-code) and covered by AC-1's negative test on mis-mapped output indices.
- **No hidden staging (Codex second pass)**: Resolution — AC-2/AC-3 require the generated kernel to have no masks, no dtype casts, and no transpose/identity staging beyond the single required PSUM→SBUF→HBM output copy (task3).

### Convergence Status
- Rounds executed: Codex first-pass analysis (Phase 3) + 1 convergence round (Phase 5). No high-impact `DISAGREE` and no `REQUIRED_CHANGES` remained after the round (all required changes were folded into the ACs); the remaining Codex `UNRESOLVED` items are promotion-policy questions requiring human decisions, carried below.
- Final Status: `converged`

## Pending User Decisions

- DEC-1: Phase-1 latency acceptance policy (also confirms the draft's quantitative target "roughly baseline latency (≥ ~1.0×)").
  - Claude Position: Correctness-only for promotion, with latency **recorded** — accept `tmm_v1` as the phase-1 baseline as long as it passes all 5 seeds and is at roughly baseline order (not pathologically slow, e.g. not many-× slower from a DMA blowup). The draft's own target is "~baseline latency, not a speed win," so a hard speed band is not the point of phase 1.
  - Codex Position: Define the policy explicitly before promotion — either correctness-only with latency recorded, or correctness plus a loose bound such as ≤ 1.25× or ≤ 1.5× baseline; do not accept a pathologically slow kernel.
  - Tradeoff Summary: Correctness-only maximizes flexibility for phase 1 and matches the draft intent, but risks accepting a needlessly slow baseline; a loose bound (≤ 1.25–1.5× baseline) guards against that at the cost of possibly re-working structure in phase 1 rather than deferring to phase 2. Both agree a pathologically slow kernel must not be promoted. **Recommendation**: correctness-only with latency recorded, plus a soft sanity ceiling of ≤ 1.5× baseline (fail promotion only if grossly slower) — captures the draft intent while honoring Codex's "not pathologically slow" guard.
  - Decision Status: `PENDING`

- DEC-2: Is "PE is the dominant engine" a hard pass/fail gate for accepting `tmm_v1`, or a recorded diagnostic for phase 2?
  - Claude Position: Diagnostic, not a hard gate. Correctness (AC-1–AC-3) plus a saved, inspected profiler digest (AC-6.1) is what phase 1 must guarantee. If the profile shows a non-PE bottleneck, that is still valid round-0 evidence — it just changes the phase-2 starting lever.
  - Codex Position: PE-dominance may not be a strict acceptance criterion for first correctness; a correct kernel showing a DMA/SBUF bottleneck is still useful phase-1 evidence, though it changes the phase-2 base.
  - Tradeoff Summary: Treating PE-dominance as a hard gate would (incorrectly) fail a correct kernel whose measured bottleneck differs from the compute-bound hypothesis; treating it as a recorded diagnostic keeps phase 1 about correctness while still capturing the bottleneck for phase 2. Claude and Codex agree it should be diagnostic. **Recommendation**: diagnostic (AC-6.1 requires saving + inspecting the digest and recording the dominant engine; it does not require PE to be dominant).
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `lhs_blk`, `rhs_chunk`, `acc`, `M_BLK`, `N_CHUNK`).
- Repository-facing files, comments, and commit messages must be in English (per project CLAUDE.md).
- Do not edit the NKIBench benchmark definition (`../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}`); the candidate lives only in `workspaces/transpose_matmul/runs/tmm_v1.py`. Candidate `.py` sources under `runs/` are tracked; other `runs/` artifacts and all of `profile/` are git-ignored.

### Note on Codex Cross-Review
This plan incorporated a Codex first-pass analysis and one Codex convergence round. Cross-review confidence is normal (Codex was available for both passes).

--- Original Design Draft Start ---

# transpose_matmul — Phase 1 Draft (first correct NKI kernel)

## 1. Operator & shapes

`out = lhs^T @ rhs`, fp32.
- `lhs`: stored **(K, M) = (2048, 4096)** — i.e. K-major.
- `rhs`: **(K, N) = (2048, 10944)**.
- `out`: **(M, N) = (4096, 10944)**.
- MACs = M·N·K = 4096·10944·2048 = **9.17e10**.

NKIBench tiles the operands (partition dim ≤ 128) via `transform_to_nki_inputs`:
```
lhs (2048,4096) -> reshape (128, 16, 4096)   # v1
rhs (2048,10944) -> reshape (128, 16, 10944) # v2
out                                          # v3 = (32, 128, 10944)
```

## 2. The layout insight (the main lever, already used by the baseline)

For the reshape `(K, ·) -> (128, 16, ·)`, index maps as `k = k_in*16 + kt`
(verified numerically: `reshape(128,16,·)[k_in,kt,·] == orig[k_in*16+kt, ·]`).
So:

- `v1[k_in(128,par), kt(16), m(4096)]` — **K is on the partition axis**.
- `v2[k_in(128,par), kt(16), n(10944)]` — same.

The Tensor Engine's `nisa.nc_matmul(stationary, moving)` computes
`stationary.T @ moving` and requires the **contraction dim on the partition axis
of BOTH operands**. Here the contraction is K, and K is *already* on the
partition axis of both v1 and v2. Therefore:

> **No transpose is needed.** `lhs^T @ rhs` where lhs is (K,M) is exactly what
> `nc_matmul` computes when we feed `stationary = v1[:, kt, m0:m0+128]`
> (= `[k_in=128, m_sub=128]`) and `moving = v2[:, kt, n0:n0+W]`
> (= `[k_in=128, n=W]`): result `= [m_sub, n]`, accumulated over the 16 `kt`
> tiles. Full contraction = `k_in(128) × kt(16) = 2048 = K`. ✔

This contrasts with my prior `matmul_v1`/`bmm_v2`, where lhs arrived M-major and
needed an identity `nc_matmul(is_transpose=True)` per tile. Here that whole
transpose stage disappears — the transposed-lhs operand is a free byproduct of
the input layout.

## 3. Bottleneck: compute-bound fp32 GEMM

- Baseline latency = **4.849615 ms** (from `baselines.json`), 9.17e10 MACs
  ⇒ ~1.9e13 MAC/s effective — right at the fp32 PE floor observed in my
  matmul-family work (`[[kda-bmm-progress]]`, `[[kda-matmul-progress]]`).
- HBM traffic even with generous re-loads is far under the roofline: lhs 33.5 MB
  (1×) + rhs 89.6 MB (re-loaded a few ×) + out 179 MB ≈ 0.5–0.7 ms at ~781 GB/s
  (`[[kda-silu-progress]]` roofline). **DMA is not the wall.**
- ⇒ Phase 1 correctness cost ≈ baseline; the real speedup levers are compute
  (phase 2 bf16×2, phase 3 shape specialization). Phase 1 target: a clean,
  fully-correct kernel at roughly baseline latency (≥ ~1.0×), not a speed win.

## 4. Phase-1 kernel design (`runs/tmm_v1.py`)

Clean **M-block-outer streaming** GEMM, no transpose, single-bank PSUM tiles.

Constants:
```
M_TILES = 32   (4096/128)     K_TILES = 16   (contraction tiles)
M_BLK   = 8 tiles (=1024 m)   M_BLOCKS = 4
N       = 10944               N_CHUNK = 456   N_CHUNKS = 24   (456*24 = 10944 exactly)
```

**N_CHUNK = 456** is chosen deliberately: it is an exact divisor of N and ≤ 512
(one fp32 PSUM bank is 512 wide). This gives **zero tail masking anywhere** —
the single largest correctness-bug surface (the baseline's `mask=... >= 0`
arithmetic) is eliminated for the phase-1 correct baseline. The ~2% free-axis
under-fill vs 512 is a phase-2 concern, not a phase-1 one.

```
out = nl.ndarray((32, 128, 10944), fp32, shared_hbm)

for mb in affine_range(M_BLOCKS):                 # 4 m-blocks of 1024
    # Resident lhs block for these 1024 m-rows: [k_in=128, kt=16, 1024]
    #   (16*1024*4 = 64 KB/partition). Loaded ONCE per m-block; read-only after.
    lhs_blk = sbuf[par_dim(128), 16, 1024]
    for kt in affine_range(16):
        lhs_blk[:, kt, :] = load(v1[:128, kt, 1024*mb : 1024*mb+1024])

    for c in affine_range(N_CHUNKS):              # 24 n-chunks of 456
        # rhs chunk, all 16 kt for this n-slice: [k_in=128, kt=16, 456]
        #   (16*456*4 ≈ 28.5 KB/partition). Loaded ONCE per (mb,c), reused
        #   across the 8 m-subtiles below (so rhs is read 4× total, once per mb).
        rhs_chunk = sbuf[par_dim(128), 16, 456]
        for kt in affine_range(16):
            rhs_chunk[:, kt, :] = load(v2[:128, kt, 456*c : 456*c+456])

        for s in affine_range(M_BLK):             # 8 m-subtiles in the block
            acc = nl.zeros(par_dim(128), 456, psum)     # 1 PSUM bank
            for kt in affine_range(16):
                acc += nc_matmul(
                    lhs_blk[:, kt, 128*s : 128*s+128],  # stationary [k_in,128]
                    rhs_chunk[:, kt, :])                # moving     [k_in,456]
            out_sb = copy(acc -> sbuf)                  # [128, 456]
            store(out[8*mb + s, :128, 456*c : 456*c+456], out_sb)
```

Result of the inner matmul: `stationary.T @ moving = [m_sub=128, k_in] @ [k_in,
456] = [m_sub=128, 456]`, accumulated over the 16 `kt` tiles → the full K=2048
contraction. Output tile `[m_sub(par), n(free)]` stores directly into
`out[8*mb+s, :, n0:n0+456]`, matching `v3[m_tile, m_in, n]`.

**Resident SBUF budget/partition**: lhs_blk 64 KB + rhs_chunk ~28.5 KB (+ small
out_sb/load temps) ≈ 93 KB, well under 192 KB even with double-buffering of the
per-chunk rhs. **PSUM**: one 456-wide bank live per `s` (+ compiler rotation) —
far under the 8 banks. No spill; HBM stays near the once-lhs / 4×-rhs / once-out
floor.

**Why M-block-outer (not the dead-simple m-tile-outer of `matmul_v1`)**: reloading
rhs per m-tile would be 32× rhs traffic (~2.9 GB → ~3.7 ms DMA, dangerously close
to the PE floor). Blocking M into 4 resident blocks caps rhs re-reads at 4×
(~0.36 GB), keeping the kernel comfortably PE-bound. M_BLK is a correctness-neutral
knob (8 matches baseline granularity; 16 fits SBUF and would halve rhs traffic) —
left as an explicit tunable for phase 2/3.

## 5. Correctness argument

- Math is **bit-exact fp32**: same operands, same K-accumulation order semantics
  as a standard tiled GEMM; `nc_matmul` fp32 accumulate in PSUM, one copy to
  SBUF, one store. No dtype casts, no approximation.
- No masking ⇒ no partial-tile arithmetic to get wrong; every tile is full-size.
- Gate = NKIBench relative-L2 `||v_k − v_r||₂ < 2e-5·||v_r||₂` across seeds
  `[0,21,42,63,84]`. A faithful fp32 GEMM sits at ~0 rel-L2 (well inside 2e-5);
  `verify.py` gates on `l2_norm_passed` — trust it.

## 6. Evidence plan

Run from `workspaces/transpose_matmul/`:

1. **Correctness/score (fast)** while iterating:
   ```
   python3 \
       ../../verify.py --op transpose_matmul --candidate runs/tmm_v1.py --fast
   ```
2. **Promote**: re-run without `--fast` (full 5-seed / higher-iter) before recording.
3. Record the perf row in `benchmark.csv`
   (`timestamp,op,candidate,parent,passed,latency_ms,speedup,notes`).
4. Record the candidate in `candidates.jsonl` (DAG: `parent =
   baseline:transpose_matmul_M4096_K2048_N10944_0.py`), with `seeds`,
   `latency_ms`, `baseline_latency_ms=4.849615`, `speedup`, `rel_l2_gate=2e-5`,
   and a `structure` note.
5. Save the profiler digest (MFU / PE / Vec / Scl / DMA / HBMrd / HBMwr, printed
   by `verify.py`) under `profile/` — this is the round-0 bottleneck baseline
   that phase 2 works against. **Confirm PE ≈ dominant engine** (validates the
   compute-bound hypothesis before choosing the phase-2 lever).

## 7. Forward-looking (NOT phase 1 — for context only)

- **Phase 2 lever candidate: bf16×2 3-product split** on the matmul. My
  matmul-family results are split: it *won* big on `matmul_add_rmsnorm`
  (`[[kda-matmul-add-rmsnorm-progress]]`, +4.88×, because per-instruction fp32
  *rate* on a moving-512 GEMM dominates, not the emulation instruction *count*),
  but *lost* on swiglu (`[[kda-swiglu-progress]]`, fp32 emulates in ~2 passes).
  This kernel is a large moving-N (456–512) fp32 GEMM — the *favorable* case —
  so bf16×2 is the leading phase-2 hypothesis, to be MEASURED not assumed. The
  rel-L2 headroom (2e-5 gate; compensated-bf16 lands ~4.5e-6 offline, ~1.5e-5
  on-device in quadrature with the fp32 floor) must be re-checked on-device.
- **Phase 3 lever candidate: shape specialization** — M_BLK width (8→16),
  N_CHUNK (456 vs 512+mask), and stationary/moving reuse tuned to the measured
  PE-idle gap.

**Phase-1 deliverable: `runs/tmm_v1.py`, correct across all 5 seeds, ~baseline
latency, PE-bound confirmed in `profile/`.**

--- Original Design Draft End ---
