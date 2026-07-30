# transpose_matmul Phase 2 — Profile-Driven Optimization: Compensated bf16x2 3-Product Split

## Goal Description

Beat the promoted phase-1 kernel `tmm_v1` (1.026x, 4.7274 ms, full-5-seed relative-L2 PASS at 3.99e-7) on `transpose_matmul` (`out = lhs^T @ rhs`, lhs (K,M)=(2048,4096), rhs (K,N)=(2048,10944), out (M,N)=(4096,10944), fp32) by attacking the sole established bottleneck: the fp32 systolic PE floor. Round-0 profiling shows `tmm_v1` is PE-BOUND (PE active 99.80%, TRUE PE-active 4.718 ms ≈ p50 4.727 ms, MFU 49.41% = the structural bf16-native fp32 emulation cap), with DMA fully hidden (21.4%) and HBM exactly at the once-lhs/4×-rhs/once-out floor (392.2 MB read / 179.3 MB write, no spill), emulating fp32 at 2.0 matmul-instructions/site.

The primary lever (D1) is a **compensated bf16x2 3-product split** of both operands: run the matmul in bf16 arithmetic (recovering ~16 effective mantissa bits via two-limb high/low splits) while clearing the NKIBench 2e-5 relative-L2 gate at bf16-class matmul rate. The directional target is ~1.28x over baseline (the sibling GEMMs won ×1.245–1.279 kernel-over-kernel), but per the governing lesson [[BL-20260710-bf16x2-loses-when-fp32-emulates-in-2-passes]] the SPEED win does not transfer across operators — it depends on a per-op hardware quantity (per-instruction fp32 rate + limb residency) and **must be MEASURED on this operator**, not assumed. `tmm_v1` emulates at the same 2.0 matmul-instr/site count that LOST on swiglu's all-3 split yet WON on the `matmul_add_rmsnorm` sibling; both sign-deciding conditions here point to a win, but the outcome is measured.

Deliver a kernel promoted only when it beats a same-session `tmm_v1` anchor on wall-clock p50 outside the control-drift band, with full-5-seed correctness and HBM still at the no-spill floor. Wall-clock is the promotion authority; TRUE PE-active is the diagnostic that explains the outcome. If the split does not produce a real wall win, keep `tmm_v1` as the guaranteed pure-fp32 fallback and record the per-instruction-rate datum. Never regress correctness.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. Verification uses the workspace tooling: `verify.py --op transpose_matmul --candidate <path> [--fast]` (gates on `l2_norm_passed`) and `runs/dump_metrics.py` (surfaces `tensor_engine_active_time_ns`, `matmul_instruction_count`, per-engine instruction counts, `hbm_read_bytes`/`hbm_write_bytes`, `psum_read_sbuf_write_count`, the full latency distribution, and per-seed `relative_l2_error`). 

- AC-1: **Offline numeric gate (zero remote spend).** An offline numpy simulation `runs/offline_bf16_split_sim.py` reproduces the exact scored input and proves the compensated bf16x2 3-product split clears the 2e-5 relative-L2 gate with margin BEFORE any kernel is built.
  - Positive Tests (expected to PASS):
    - The sim draws inputs with `np.random.seed(seed)` then `lhs = normal(0,1,(K,M))`, `rhs = normal(0,1,(K,N))` in reference draw-order (adapter `DEFAULT_INPUT_SEED=42`), and its fp32 control `lhs.T @ rhs` reproduces the NKIBench numpy reference to ~1e-7 (validates seed / draw-order / dtype / formula).
    - Worst 3-product relative-L2 over seeds {42,0,21,63,84,123,2024} is comfortably below 2e-5 (predicted ≈4.5e-6, the pure-GEMM family regime), which authorizes building the D1 kernel.
    - The sim also reports the 4-product number (sizes the dropped `lo@lo` term) and plain single-limb bf16 (scale/calibration datum only; expected ~1e-3, not a candidate).
  - Negative Tests (expected to FAIL / be rejected when working correctly):
    - fp32 control deviating from the reference by more than ~1e-6 (indicates a seed / draw-order / dtype mismatch — the sim is invalid and must not gate).
    - Worst 3-product relative-L2 at or above ~1.3e-5 (the point where the predicted device quadrature could approach 2e-5) is NOT "comfortably below": it does not authorize the kernel; instead it records a precision-floor datum and triggers reconsideration (e.g. 4-product).
    - Plain single-limb bf16 passing the 2e-5 gate would signal a mis-scaled sim (bf16 alone is expected to fail).
- AC-2: **D1 kernel correctness (on-device, the scored gate).** `runs/tmm_v2_bf16_split.py` passes the NKIBench relative-L2 gate on the full 5-seed run.
  - Positive Tests (expected to PASS):
    - `verify.py --op transpose_matmul --candidate runs/tmm_v2_bf16_split.py` reports PASS (full 5-seed, `l2_norm_passed = True` every seed).
    - Every per-seed `relative_l2_error` (from `dump_metrics.py`) is below 2e-5, and the per-seed MAX is reported (not only the aggregate pass/fail).
    - Output contains only finite values (no NaN/Inf) on all seeds.
  - Negative Tests (expected to FAIL / be rejected):
    - Any seed with `l2_norm_passed = False` or `relative_l2_error ≥ 2e-5` → correctness regression, kernel rejected regardless of speed.
    - Any NaN/Inf in the output → rejected.
  - AC-2.1: **Quadrature corroboration (evidence, NOT a hard gate).** The backed-out bf16 term `sqrt(ondevice² − floor²)` (using `tmm_v1`'s measured floor 3.99e-7) is recorded in the digest and is expected to match the offline 3-product number (≈4.5e-6). This corroborates the error model but is NOT required equality — accumulation order, bf16 conversion semantics, and compiler scheduling can perturb it; the hard gate is AC-2's `relative_l2_error < 2e-5`.
    - Positive: recorded backed-out bf16 term is within the same order of magnitude as the offline sim number.
    - Negative: treating a quadrature mismatch as a correctness failure when AC-2's rel-L2 gate passes (mismatch is a documented diagnostic, not a rejection).
- AC-3: **D1 speed measurement (diagnostic; explains the wall-clock outcome).** A single-session `dump_metrics.py` run captures the same metric set for BOTH `tmm_v2` and a same-session `tmm_v1` anchor, so the wall-clock outcome (AC-5) is explainable.
  - Positive Tests (expected to PASS):
    - `matmul_instruction_count` for `tmm_v2` ≈ 36864 (3.0 instr/site, up from v1's 24576 at 2.0/site) — confirms the 3-product split lowered to the intended instruction shape.
    - The dump reports, for both kernels in the same session: TRUE PE-active/inf, total cycles / `neuroncore_cycle_count`, `matmul_instruction_count`, tensor/scalar/vector engine instruction counts (to catch limb-build `copy`/`subtract` pressure), `hbm_read_bytes`/`hbm_write_bytes`, `psum_read_sbuf_write_count`, and wall p50/p90.
    - The TRUE PE-active sign vs the v1 anchor is recorded and interpreted (a DROP is the sibling-win signature; a RISE is the swiglu-loss signature) as the explanation for AC-5's wall result.
  - Negative Tests (expected to FAIL / be flagged):
    - Missing a same-session v1 anchor in the same dump (cross-session comparison is invalid given remote variance) → AC-3 not satisfied.
    - A large rise in non-matmul instruction counts (`copy`/`subtract`/`tensor_tensor`) that eats the matmul saving, flagged as a latency-neutral-cause even if `matmul_instruction_count` matched the target.
- AC-4: **HBM stays at the no-spill floor.** The split builds limbs on-chip from the same fp32 loads, so HBM traffic must not grow.
  - Positive Tests (expected to PASS):
    - `hbm_read_bytes` ≈ 392 MB and `hbm_write_bytes` ≈ 179 MB for `tmm_v2` (near the v1 once-lhs/4×-rhs/once-out floor).
    - `psum_read_sbuf_write_count` shows one PSUM→SBUF drain per output tile (768) — no extra round trips, no spill signature.
  - Negative Tests (expected to FAIL / be rejected):
    - `hbm_read_bytes` materially above ~392 MB or the appearance of spill/reload bytes → SBUF pressure from holding hi/lo limbs; rejected as a spill regression even if correctness passes.
    - `psum_read_sbuf_write_count` above 768 → extra PSUM round trips from bank pressure; flagged.
- AC-5: **Promotion (wall-clock is the authority).** `tmm_v2` is promoted only on a real wall-clock win over a same-session `tmm_v1` anchor, outside the measured control-drift band.
  - Positive Tests (expected to PASS):
    - The control band is measured by running the `tmm_v1` anchor twice in the same session (bracket `v1a → tmm_v2 → v1b`); the drift band = `abs(p50(v1a) − p50(v1b))`.
    - `tmm_v2` full-5-seed p50 beats the (nearer/worse applicable) v1 anchor by more than `max(control-drift band, 3% fixed floor)` — a real win, not remote noise. On a genuine D1 win the ~1.28x target implies a double-digit-% margin, so this is easily cleared.
    - No per-seed latency outlier regression, latency distribution (p50/p90) stable, and the candidate is recorded in `benchmark.csv` + `candidates.jsonl` (DAG parent link `tmm_v1`) with evidence under `profile/`.
  - Negative Tests (expected to FAIL / be handled by disposition):
    - Correct with TRUE PE-active dropping but wall p50 inside the drift band (latency-neutral) → **ARCHIVE**, not promote (record as a branch; D2 may compose on it; document that another bottleneck absorbed the PE saving).
    - Wall p50 regressing beyond the band, OR any spill/correctness failure → **REJECT**; keep `tmm_v1` and record the per-instruction-rate datum (second confirming point after swiglu that the 2.0/site count can go either way).
    - Promoting on a <3% or within-band apparent win → rejected as chasing profiler noise.
    - Note: if the wall improves outside the band while TRUE PE-active RISES, still **PROMOTE** (wall-clock authority) and document that the PE-active thesis was wrong for this operator.
- AC-6: **fp32 fallback preserved.** `tmm_v1` is retained unchanged as the guaranteed pure-fp32 fallback regardless of D1 outcome (the 5 profiler seeds reuse the seed-42 input, so on-device input diversity is weak; the offline sim's distinct draws are the real margin evidence).
  - Positive Tests: `runs/tmm_v1.py` remains present and unmodified; if D1 is REJECTed, `tmm_v1` stays the promoted kernel of record.
  - Negative Tests: deleting or hand-tuning `tmm_v1`, or promoting `tmm_v2` in a way that loses the fp32 fallback path.
- AC-7: **Secondary levers gated on D1's post-profile.** D2/D3/D4 are only attempted under their stated conditions and only adopted on measured wins.
  - Positive Tests (expected to PASS):
    - D2 (N_CHUNK 456→512 + masked tail) is `--fast`-screened only after D1 lands, composed on the D1 kernel, and adopted only if it clears the same-session band AND stays correct (mask boundary checked with per-seed rel-L2 max and HBM-read sanity so a bad mask cannot silently add rhs traffic).
    - D3 (M_BLK 8→16 or 4) is `--fast`-screened only if D1 shifts the DMA balance off hidden; treated as a possible anti-lever (enlarged resident live set can constrain the affine_range pipeline per [[BL-20260710-cross-batch-blocking-is-an-antilever-on-affine-range]]); rejected on increased PE-active, spills, or worse pipeline occupancy even if HBM bytes drop.
    - D4 (4-product) is SKIPPED unless D1's on-device rel-L2 comes back marginal (near 2e-5); if built, recorded as a measured reject unless the backed-out bf16 term is itself near the gate.
  - Negative Tests (expected to FAIL / be rejected):
    - Adopting D2/D3 on an instruction-count improvement alone without a real wall-clock win outside the band, or with a correctness/HBM regression.
    - Building D4 when D1's margin was comfortable (numerically unnecessary; siblings measured ~+28% latency for a term swamped by the tiny fp32 floor).
    - Running D3 when DMA remained fully hidden (no basis to expect a PE-bound wall to move).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
The offline sim (AC-1) plus the D1 kernel `runs/tmm_v2_bf16_split.py` measured to a three-state disposition (PROMOTE / ARCHIVE / REJECT), AND — if and only if D1's post-profile justifies them — one screening pass each of D2 (N_CHUNK=512 + masked tail, composed on the D1 kernel) and D3 (M_BLK sweep), each `--fast`-screened and adopted only on a measured same-session wall win. All evidence recorded in `benchmark.csv`, `candidates.jsonl` (DAG parent links), and `profile/`. This completes the profile-driven optimization without over-engineering: it attacks the one real bottleneck (the fp32 PE floor) and only spends on secondary levers when the D1 profile shows a basis for them.

### Lower Bound (Minimum Acceptable Scope)
The offline sim (AC-1) and the D1 kernel measured against a same-session `tmm_v1` anchor with the three-state disposition applied, correctness never regressed, and `tmm_v1` preserved as the fp32 fallback. If D1 ARCHIVEs or REJECTs, that is a complete and acceptable outcome: the plan's value is the MEASURED sign on this operator (a second data point after swiglu on whether the 2.0-instr/site count wins or loses), and no secondary lever need be run. Even a REJECT that keeps `tmm_v1` at 1.026x satisfies the lower bound, provided the per-instruction-rate datum is recorded.

### Allowed Choices
- Can use: the compensated bf16x2 two-limb high/low split via `nl.copy(dtype=nl.bfloat16)` (round-to-nearest-even) and `nisa.tensor_tensor(..., op=nl.subtract)` (fp32 upcast) for the residual; a fixed 3-product accumulation `hi@hi + hi@lo + lo@hi` dropping `lo@lo`, accumulated in one fp32 PSUM bank; N_CHUNK=512 with masked tail (D2) and M_BLK ∈ {4, 16} (D3) as gated secondary screens; the workspace `verify.py` / `dump_metrics.py` tooling and same-session anchoring.
- Cannot use: any change to the `tmm_v1` loop nest / constants / PSUM-accumulation / copy+store epilogue beyond the operand dtype path for D1 (D1 is a localized matmul-only diff, NO enabler refactor); editing the NKIBench benchmark definition or hand-tuning a baseline; promoting on instruction-count or PE-active alone without a wall-clock win outside the control-drift band; deleting or modifying `tmm_v1`; allclose-based correctness judgment (the gate is relative-L2 via `l2_norm_passed`).

> **Note on Deterministic Design**: The draft pins the D1 kernel design tightly — the split order, product order, dropped term, tiling constants, and measurement protocol are fixed. The path boundaries reflect that: D1's structure is deterministic (upper and lower bound converge on "the localized matmul-only split, measured"), and the only genuine width is whether the D1 post-profile authorizes the D2/D3 secondary screens and whether the outcome is PROMOTE/ARCHIVE/REJECT — which is decided by measurement, not by implementation choice.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

D1 mirrors the sibling idiom `matmul_add_rmsnorm_v2_bf16_split.py`, but simpler: `transpose_matmul` has NO transpose stage (lhs arrives K-on-partition from the NKIBench reshape) and NO residual/RMSNorm/g epilogue, so the limbs are built directly from the loaded fp32 tiles with no PSUM round-trip or transpose scratch. Everything outside the matmul stays byte-for-byte `tmm_v1`.

Pinned split (round-to-nearest-even; residual upcasts to fp32):

```
# once per m-block, resident:
lhs_hi = bf16(lhs)              # nl.copy(dtype=nl.bfloat16)
lhs_lo = bf16(lhs - lhs_hi)     # residual via nisa.tensor_tensor(op=nl.subtract), then bf16 copy
# once per (mb, c), reused across the 8 subtiles:
rhs_hi = bf16(rhs)
rhs_lo = bf16(rhs - rhs_hi)
```

3 bf16 products into one fp32 PSUM bank, FIXED order, dropping `lhs_lo@rhs_lo`:

```
acc += nc_matmul(lhs_hi[kt, :, 128*s:], rhs_hi[kt])   # hi @ hi
acc += nc_matmul(lhs_hi[kt, :, 128*s:], rhs_lo[kt])   # hi @ lo
acc += nc_matmul(lhs_lo[kt, :, 128*s:], rhs_hi[kt])   # lo @ hi
```

Two bf16 limbs occupy exactly the same bytes as one fp32 tile (2×2-byte bf16 = one 4-byte fp32), so the resident lhs limbs (2×32 KB = 64 KB/part = same as v1's fp32 `lhs_blk`) and streamed rhs limbs (2×~14.25 KB ≈ 28.5 KB/part = same as v1's fp32 `rhs_chunk`) keep the v1 SBUF footprint; limbs are built on-chip from the same fp32 HBM loads so HBM stays at the floor. `matmul_instruction_count` rises 24576→~36864 (2.0→3.0/site) — the win, if any, is that each new instruction is a bf16 pass, not the ~1.8× fp32 emulation rate.

Measurement flow (the loss-lesson requires MEASURING the sign, not assuming it):
1. Offline sim → numeric gate (no remote spend).
2. Build `tmm_v2`; `verify.py --fast` for correctness + latency direction.
3. `dump_metrics.py` capturing `tmm_v2` AND a same-session `tmm_v1` anchor (bracket `v1a → tmm_v2 → v1b` for the drift band) → read TRUE PE-active sign + full metric set.
4. Apply the three-state disposition: PROMOTE only on a real wall win outside the band; ARCHIVE if correct + PE-active drops but wall is neutral; REJECT if incorrect/spills/regresses.
5. Confirm a KEEP on the full 5-seed run; then optionally screen D2/D3 if the D1 profile justifies them.

### Relevant References
- `workspaces/transpose_matmul/runs/tmm_v1.py` — the phase-1 kernel; D1 is a localized matmul-only diff on this loop nest/constants/epilogue.
- `workspaces/transpose_matmul/profile/tmm_v1_digest.md` and `profile/tmm_v1_dump.txt` — the round-0 bottleneck reference (PE-bound at the fp32 floor).
- `workspaces/matmul_add_rmsnorm/runs/matmul_add_rmsnorm_v2_bf16_split.py` — the sibling bf16-split idiom to mirror (the structural twin that WON at 2.0/site).
- `workspaces/matmul_add_rmsnorm/runs/offline_bf16_split_sim.py` — the offline sim to mirror (simplify: no z/g/norm; pure GEMM). Also `workspaces/rmsnorm_matmul/runs/offline_bf16_split_sim.py`, `workspaces/swiglu/runs/offline_bf16_split_sim.py` for the pure-GEMM family numbers.
- `workspaces/transpose_matmul/runs/dump_metrics.py` — surfaces TRUE PE-active, `matmul_instruction_count`, per-engine counts, HBM bytes, `psum_read_sbuf_write_count`, latency distribution, per-seed `relative_l2_error`.
- `verify.py` (repo root) — the correctness/latency gate (`l2_norm_passed`).
- BitLessons: [[BL-20260710-bf16x2-loses-when-fp32-emulates-in-2-passes]] (the measure-the-sign lesson), [[BL-20260709-compensated-bf16x2-split-beats-fp32-floor]] (quadrature error model), [[BL-20260709-fp32-pe-floor-calibration]] (MFU ~50% fp32 floor), [[BL-20260710-cross-batch-blocking-is-an-antilever-on-affine-range]] (D3 anti-lever risk), [[BL-20260709-fast-vs-full-run-latency]] (control-band methodology).

## Dependencies and Sequence

### Milestones
1. **Offline numeric gate** (no remote spend): build and run the offline sim; confirm the 3-product split clears 2e-5 with margin.
   - Phase A: write `runs/offline_bf16_split_sim.py` (mirror the sibling sim, simplified to a pure GEMM: no z/g/norm) with the exact seeded draw and the RNE two-limb split.
   - Phase B: run it; check fp32 control ≈ reference (~1e-7) and worst 3-product rel-L2 comfortably below 2e-5. This gate decides whether to build D1.
2. **D1 kernel build + measured disposition** (depends on Milestone 1 passing): implement and profile the split, then classify PROMOTE/ARCHIVE/REJECT.
   - Phase A: implement `runs/tmm_v2_bf16_split.py` as a localized matmul-only diff on `tmm_v1` (limb build + 3-product accumulation; epilogue unchanged).
   - Phase B: `verify.py --fast` for correctness + latency direction.
   - Phase C: `dump_metrics.py` capturing `tmm_v2` and a same-session `tmm_v1` anchor bracket (`v1a → tmm_v2 → v1b`); read TRUE PE-active sign, `matmul_instruction_count`, per-engine counts, HBM bytes, `psum_read_sbuf_write_count`, wall p50/p90.
   - Phase D: apply the three-state disposition against the measured control-drift band.
3. **Promotion confirmation** (depends on Milestone 2 = PROMOTE-candidate): full-5-seed run, confirm wall win outside the band and full-seed correctness; record in `benchmark.csv` + `candidates.jsonl` + `profile/`.
4. **Secondary levers (optional, gated on the D1 post-profile)**: only if the D1 profile provides a basis.
   - Step 1: D2 (N_CHUNK=512 + masked tail) `--fast` screen composed on the D1 kernel — only if D1 landed.
   - Step 2: D3 (M_BLK sweep) `--fast` screen — only if D1 shifted the DMA balance off hidden.
   - (D4 4-product is SKIPPED unless D1's on-device rel-L2 came back marginal.)

Dependencies: Milestone 2 depends on Milestone 1 (numeric gate authorizes the build). Milestone 3 depends on a PROMOTE-candidate from Milestone 2. Milestone 4 depends entirely on Milestone 2's post-profile (no unconditional secondary spend). `tmm_v1` remains the fallback throughout (AC-6). Total remote spend budgeted at ≤5 profiler iterations.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Write `runs/offline_bf16_split_sim.py`: exact seeded draw (`lhs=normal(0,1,(K,M))`, `rhs=normal(0,1,(K,N))`, seed 42), fp32 control vs reference, worst 3-product / 4-product / plain-bf16 rel-L2 over the seed sweep | AC-1 | coding | - |
| task2 | Run the offline sim; verify fp32 control ≈ reference (~1e-7) and worst 3-product rel-L2 comfortably below 2e-5 → numeric go/no-go for D1 | AC-1 | coding | task1 |
| task3 | Implement `runs/tmm_v2_bf16_split.py`: localized matmul-only diff on `tmm_v1` — bf16 limb build (pinned order), 3-product `hi@hi+hi@lo+lo@hi` into one fp32 PSUM bank, epilogue byte-for-byte v1 | AC-2, AC-4 | coding | task2 |
| task4 | `verify.py --fast` on `tmm_v2`: correctness (l2_norm_passed) + latency direction; check finite output | AC-2 | coding | task3 |
| task5 | `dump_metrics.py` capturing `tmm_v2` + same-session `tmm_v1` anchor bracket (`v1a → tmm_v2 → v1b`); record TRUE PE-active, matmul/per-engine instruction counts, HBM bytes, `psum_read_sbuf_write_count`, wall p50/p90, per-seed rel-L2 | AC-3, AC-4 | coding | task4 |
| task6 | Apply the three-state disposition (PROMOTE / ARCHIVE / REJECT) against the measured control-drift band; back out the quadrature bf16 term as corroboration | AC-2.1, AC-5 | coding | task5 |
| task7 | On a PROMOTE-candidate: full-5-seed run, confirm wall win outside `max(drift band, 3%)` and full-seed correctness; record in `benchmark.csv` + `candidates.jsonl` (parent `tmm_v1`) + `profile/`. On ARCHIVE/REJECT: record the disposition + per-instruction-rate datum, keep `tmm_v1` | AC-5, AC-6 | coding | task6 |
| task8 | (Optional) D2 N_CHUNK=512 + masked-tail `--fast` screen composed on the D1 kernel — only if D1 landed; adopt only on a measured wall win with correctness + HBM-read sanity | AC-7 | coding | task7 |
| task9 | (Optional) D3 M_BLK sweep `--fast` screen — only if D1 shifted the DMA balance; treat as possible anti-lever, reject on PE-active/spill/occupancy regression | AC-7 | coding | task7 |
| task10 | (Conditional) Independent analytic review of the offline sim's error budget and the D1 measured PE-active sign against the sibling/swiglu precedents, to sanity-check the disposition | AC-1, AC-3 | analyze | task6 |

## Claude-Codex Deliberation

### Agreements
- Same-session `tmm_v1` anchoring is essential; prior cross-op bf16x2 behavior is non-transferable and the TRUE PE-active sign must be measured on THIS operator.
- The offline numeric gate (AC-1) is the right first step — reproducing NKIBench seeding/shape before touching NKI avoids wasting remote cycles on a numerically doomed split.
- The 3-product split (dropping `lo@lo`) is the correct D1 candidate; 4-product should not be first-line (it gives up most of the instruction-reduction thesis for a term swamped by the tiny fp32 floor).
- HBM/no-spill guardrails (`hbm_read_bytes`/`hbm_write_bytes` near the floor, `psum_read_sbuf_write_count`) are the right Trainium-specific checks.
- Keeping `tmm_v1` as the fp32 fallback is correct regardless of D1 outcome.
- D2/D3 gating after D1 is reasonable; `N_CHUNK=512+mask` carries off-by-one and DMA-shape risk, and `M_BLK` can easily be an anti-lever while the kernel stays PE-bound.
- No separate artificial micro-probe: it would have different memory behavior and not be faithful; the full D1 kernel's `--fast` `dump_metrics` run IS the faithful probe, since D1 is a localized matmul-only diff on the already-correct `tmm_v1`.

### Resolved Disagreements
- **Promotion gate = wall-clock, not PE-active** (Round 1): Codex noted "KEEP if PE-active drops" is only a diagnostic, and "REJECT if PE-active rises" is too absolute if wall time improves. Claude v1 gated on the PE-active sign. **Resolution:** wall-clock p50 outside the control band is the sole promotion authority; TRUE PE-active is recorded to EXPLAIN the outcome. Adopted the three-state disposition (PROMOTE/ARCHIVE/REJECT). If wall improves with PE-active rising, PROMOTE and document the thesis was wrong for this op.
- **Latency-neutral-but-PE-active-drops → ARCHIVE** (Round 1, Codex UNRESOLVED item): Codex recommended archive, not promote. **Resolution:** ARCHIVE such a candidate (recorded as a branch in `candidates.jsonl`, available for D2 composition), do not promote; inspect what other bottleneck absorbed the PE saving before spending on D2/D3.
- **Promotion margin pinned numerically** (Rounds 1–2): Codex flagged "outside the control band" as too vague, and in Round 2 made convergence conditional on an explicit threshold. **Resolution:** measure the band via a same-session `v1a → tmm_v2 → v1b` bracket (band = `abs(p50(v1a) − p50(v1b))`) and require the v2 win to exceed `max(control-drift band, 3% fixed floor)` — the fixed floor prevents a tiny observed spread from promoting noise. A genuine ~1.28x win clears this by a wide margin.
- **AC-2 softened from equality to corroboration** (Round 1): Codex said "matches quadrature" is too strong as a gate. **Resolution:** the hard gate is on-device `relative_l2_error < 2e-5`; the backed-out bf16 term `sqrt(ondevice² − floor²)` ≈ offline number is recorded as corroborating evidence (AC-2.1), not required equality.
- **AC-3 broadened** (Round 1): `matmul_instruction_count` is necessary but not sufficient. **Resolution:** the same-session dump must also report TRUE PE-active, total cycles, tensor/scalar/vector instruction counts (catch limb-build `copy`/`subtract` pressure), HBM bytes, `psum_read_sbuf_write_count`, and wall p50/p90.
- **Measurable D1 failure modes added** (Round 1): **Resolution:** AC negative tests now cover SBUF spill from holding hi/lo limbs, non-matmul instruction-count growth, PSUM bank pressure, per-seed rel-L2 MAX, and a NaN/Inf finite check.
- **Micro-probe** (Rounds 1–2, Codex UNRESOLVED item): **Resolution:** no separate probe (faithfulness objection); the full D1 `--fast` run serves the purpose.

### Convergence Status
- Final Status: `converged` (Round 2: no DISAGREE, no UNRESOLVED; the single conditional REQUIRED_CHANGE — pin the promotion threshold as `max(v1↔v1 spread, fixed few-percent floor)` — is implemented in AC-5 as `max(control-drift band, 3%)`, and the optional `v1a → D1 → v1b` bracket is adopted).
- Convergence rounds executed: 2.

## Pending User Decisions

_No pending user decisions._ All Codex questions and disagreements were resolved during the convergence loop. The following defaults were adopted from the draft and the deliberation; each is an explicit, adjustable choice rather than an open question:

- The `2e-5` relative-L2 is NKIBench's fixed HARD correctness gate (not adjustable — it is the benchmark definition).
- The `~1.28x` speedup is an explicitly DIRECTIONAL target ("a MEASURED target, not a promise"), not a hard requirement; the hard promotion requirement is only "a real wall-clock win over the same-session `tmm_v1` anchor outside the control band."
- The promotion-margin fixed floor is set to **3%** (the `max(control-drift band, 3%)` rule). This is a documented default chosen to reject remote noise while easily clearing on a genuine double-digit-% win; adjust if the observed same-session `tmm_v1` control drift is routinely larger.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", "D1/D2/D3/D4", or similar workflow/plan markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `lhs_hi`/`lhs_lo`, `rhs_hi`/`rhs_lo`, `acc`), matching the existing `tmm_v1.py` and sibling bf16-split kernels' idiom and comment density.
- Kernel docstrings should explain the numeric method (compensated bf16x2 split, dropped `lo@lo`, quadrature error model) and the layout, as the sibling kernels do — that is domain documentation, not plan terminology.

--- Original Design Draft Start ---

# transpose_matmul — Phase 2 draft (profile-driven optimization)

Operator: `transpose_matmul` (NKIBench case 2). `out = lhs^T @ rhs`,
lhs (K,M)=(2048,4096) K-major, rhs (K,N)=(2048,10944), out (M,N)=(4096,10944),
fp32. MACs = M·N·K ≈ 9.17e10. Baseline 4.849615 ms.

Start point: **tmm_v1** (`runs/tmm_v1.py`), the promoted phase-1 kernel —
**1.026x (4.7274 ms)**, full-5-seed L2 PASS (rel-L2 3.99e-7). fp32 no-transpose
M-block-outer streaming GEMM: the NKIBench reshape (K,·)→(128,16,·) maps flat
k = k_in·16 + kt so K sits on the PARTITION axis of both operands, and
`nc_matmul(stationary, moving) = stationary.T @ moving` computes lhs^T @ rhs
directly — **no explicit transpose stage**. Round-0 profiler digest:
`profile/tmm_v1_digest.md`, raw dump `profile/tmm_v1_dump.txt`.

---

## 1. Round-0 bottleneck (established, not re-litigated)

tmm_v1 is **PE-BOUND at the fp32 systolic floor**:

| metric | value | reading |
|---|---|---|
| PE active % | **99.80%** | PE is essentially the entire wall clock |
| TRUE PE-active/inf | **4.718 ms** | ≈ p50 4.7277 ms → matmul IS the latency |
| MFU | **49.41%** | the fp32 floor: bf16-native array emulating fp32, capped ~50% by the bf16-peak MFU denominator — structural, not inefficiency ([[BL-20260709-fp32-pe-floor-calibration]]) |
| Vec / Scl / DMA % | 5.0 / 0.2 / 21.4 | all hidden well under PE; **DMA fully hidden** |
| HBMrd / HBMwr | 392.2 / 179.3 MB | EXACT once-lhs / 4×-rhs / once-out model; **no spill** |
| matmul_instruction_count | 24576 | over 12288 sites (4·24·8·16) = **2.0 instr/site** → fp32 emulates in 2.0 passes |

**Conclusion (DEC-2 diagnostic, already recorded): the bottleneck is COMPUTE —
the fp32 PE rate.** DMA is hidden and HBM is at the floor, so every
bandwidth/locality lever (M-block enlargement, N-tiling for DMA amortization,
double-buffering) can only touch already-hidden time and **cannot move a
PE-bound wall clock**. The one lever that attacks the actual bottleneck is
**lower matmul precision.** This exactly matches the phase-1 memory's phase-2
hand-off (lever = compute / fp32-PE floor).

---

## 2. Lever enumeration and ranking

Ranked by expected benefit vs risk. Only D1 attacks the bottleneck; D2–D4 are
PE-side micro-levers gated on D1's post-profile.

| # | lever | expected benefit | risk | priority |
|---|---|---|---|---|
| **D1** | **compensated bf16x2 3-product split** (both operands) | **HIGH (~1.25x kernel-over-kernel → ~1.28x over baseline)** | LOW-MED (numeric proven offline; **speed must be measured**) | **PRIMARY** |
| D2 | N_CHUNK 456→512 + masked tail | LOW-MED (fewer, wider matmuls; amplified 3× under the split) | LOW-MED (reintroduces the mask arithmetic phase-1 removed) | secondary, screen after D1 |
| D3 | M_BLK 8→16 (or 4) sweep | LOW (DMA already hidden) — possible **ANTI-lever** | MED (enlarged resident live set can constrain the affine_range pipeline, [[BL-20260710-cross-batch-blocking-is-an-antilever-on-affine-range]]) | screen `--fast` only if D1 shifts the DMA balance |
| D4 | 4-product split (keep lo@lo) | negligible numeric gain | HIGH latency (+~28% on siblings) | **SKIP** unless D1 comes back marginal |

### Why D1 is the right primary and why it is NOT assumable (the crux)

The compensated bf16x2 3-product split runs the matmul in bf16 arithmetic while
recovering ~16 effective mantissa bits, clearing the 2e-5 relative-L2 gate at
bf16-class matmul rate ([[BL-20260709-compensated-bf16x2-split-beats-fp32-floor]]).
It has been PROMOTED on three sibling GEMMs (rmsnorm_matmul 1.066x→1.363x;
add_rmsnorm_matmul 3.754x→4.632x; matmul_add_rmsnorm 3.920x→4.879x) and LOST on
one (swiglu all-3-GEMM 0.409x). The governing lesson
[[BL-20260710-bf16x2-loses-when-fp32-emulates-in-2-passes]] is explicit: **the
numeric margin transfers across ops (the offline sim proves it), but the SPEED
win does NOT — it depends on a per-op hardware quantity (per-instruction fp32
rate + limb residency) that MUST be measured first.**

tmm_v1 emulates at **2.0 matmul-instr/site — the same count that made swiglu's
all-3 split LOSE.** So the count alone does NOT license the split. But the two
conditions that decide the sign both point to a WIN here, and this op is the
**structural twin of `matmul_add_rmsnorm`'s GEMM (same M=4096, K=2048, dense
wide moving, 2.0/site), which WON** at that identical count:

1. **Per-instruction fp32 rate.** The winning siblings measured fp32 ≈ 1.8× the
   bf16 per-instruction rate on a dense moving-512 GEMM (add_rmsnorm_matmul:
   matmul instrs +44% yet TRUE PE-active **−23.4%**). tmm's moving operand is
   456-wide (same dense regime), so the split converts 2.0 fp32 passes (at
   ~1.8×) into 3.0 bf16 passes (at 1.0×) ≈ 3.6 → 3.0 bf16-equiv cost ≈ −17%
   PE-active predicted. The full-matmul phase-3 probe measured fp32/bf16 ≈ 3.62×
   end-to-end — an even wider gap. **Must measure the actual delta on this op.**
2. **Limb residency (no reload trap).** swiglu LOST partly because its weights do
   NOT fit resident, so bf16 limbs had to be rebuilt from re-loaded weights →
   DMA-bound. **Here, two bf16 limbs occupy exactly the same bytes as one fp32
   tile** (2×2-byte bf16 = one 4-byte fp32), so the resident lhs block and the
   streamed rhs chunk keep their phase-1 SBUF footprint, and **HBM stays at the
   floor** (limbs built on-chip from the same fp32 loads — no extra reads). Both
   swiglu-loss conditions are ABSENT.

**tmm is strictly cheaper to split than the winning sibling:** the sibling had
to transpose x per tile (fp32 identity matmul → PSUM → copy) before splitting
the activation; **tmm has no transpose at all** — lhs arrives K-on-partition, so
both limbs are built directly from the loaded fp32 tiles (no PSUM round-trip, no
transpose scratch). And there is no residual-add / RMSNorm / g epilogue to
expose. So tmm sits in the **best of both regimes**: the sibling's favorable
SPEED regime (resident limbs, dense wide moving, 2.0/site) **plus** the
swiglu/matmul favorable PRECISION regime (tiny fp32 floor — see §4).

Honest expectation: sibling kernel-over-kernel wins were ×1.245–1.279, so
1.026x → **≈1.28x over baseline**. But per the loss-lesson this is a MEASURED
target, not a promise; a RISE in TRUE PE-active would mean the split loses here
and the fp32 floor is terminal.

---

## 3. D1 kernel design (localized diff on tmm_v1, NO enabler refactor)

Mirror the sibling idiom `matmul_add_rmsnorm_v2_bf16_split.py`. The loop nest,
constants (M_BLK=8, N_CHUNK=456, 24 chunks, 16 kt), PSUM accumulation, and the
copy+store epilogue are **byte-for-byte v1**. Only the operand dtype path
changes; precision loss is confined to the matmul.

**Pinned, auditable split order** (round-to-nearest-even via `nl.copy(dtype=nl.bfloat16)`;
residual via `nisa.tensor_tensor(..., op=nl.subtract)` which upcasts to fp32):

```
lhs (fp32) -> lhs_hi = bf16(lhs);  lhs_lo = bf16(lhs - lhs_hi)   # once per m-block, resident
rhs (fp32) -> rhs_hi = bf16(rhs);  rhs_lo = bf16(rhs - rhs_hi)   # once per (mb, c), reused across 8 subtiles
```

**3 bf16 products in one fp32 PSUM bank, FIXED order, dropping lhs_lo@rhs_lo:**

```
acc += nc_matmul(lhs_hi[kt, :, 128*s:], rhs_hi[kt])   # hi @ hi
acc += nc_matmul(lhs_hi[kt, :, 128*s:], rhs_lo[kt])   # hi @ lo
acc += nc_matmul(lhs_lo[kt, :, 128*s:], rhs_hi[kt])   # lo @ hi
```

(The split is symmetric — hi@hi + hi@lo + lo@hi keeps ~16 mantissa bits; the
dropped lo@lo is ~1e-6, confirmed negligible by the 4-product offline check.)

**Structural changes vs v1:**
- After loading each fp32 lhs tile into a transient buffer, build `lhs_hi[kt]`,
  `lhs_lo[kt]` bf16 `[128,16,1024]` (2×32 KB = 64 KB/part = **same bytes as v1's
  fp32 lhs_blk**, which is dropped). Built once per m-block, resident.
- After loading each fp32 rhs tile, build `rhs_hi`, `rhs_lo` bf16 `[128,16,456]`
  (2×~14.25 KB ≈ 28.5 KB/part = **same bytes as v1's fp32 rhs_chunk**). Built
  once per (mb,c), reused across the 8 subtiles.
- Inner loop: 1 fp32 `nc_matmul` → 3 bf16 `nc_matmul` into the same PSUM bank.
- Epilogue (`nl.copy` PSUM→SBUF fp32, `nl.store`) unchanged.

**SBUF budget:** resident lhs limbs 64 KB + rhs limbs 28.5 KB + transient fp32
build scratch (lhs tile 64 KB freed after build, rhs tile ~28.5 KB) + out_sb
(1.8 KB) + PSUM banks — peak well under 192 KB, same argument as the sibling.
**HBM unchanged vs v1** (~392 MB read: limbs built from the same fp32 loads).

**matmul_instruction_count** will rise 24576 → ~36864 (2.0→3.0/site); the win is
that each new instr is a bf16 pass, not the 2.0× fp32 emulation. This is the
number the measurement protocol reads.

Deliverable: `runs/tmm_v2_bf16_split.py` (parent tmm_v1).

---

## 4. Correctness plan (offline-first, zero remote spend)

Build `runs/offline_bf16_split_sim.py` (mirror the sibling sim, simplified — no
z/g/norm; this op is a pure GEMM):
- Reproduce the exact scored input: `np.random.seed(seed)` then draw
  `lhs = normal(0,1,(K,M))`, `rhs = normal(0,1,(K,N))` in reference order
  (adapter DEFAULT_INPUT_SEED=42; note the adapter reseeds to 42 for every
  profiler draw, so the offline multi-seed sweep IS the real input-diversity
  evidence).
- fp32 control `lhs.T @ rhs` must reproduce the numpy reference to ~1e-7
  (validates seed / draw-order / dtype).
- Report worst 3-product rel-L2 over seeds {42,0,21,63,84,123,2024}; also the
  4-product number (to size the dropped lo@lo) and plain-bf16 (scale check,
  expect ~2e-3 FAIL).

**Predicted numbers** (from the pure-GEMM family — rmsnorm_matmul offline
4.455e-6, swiglu 7.7e-6): worst 3-product rel-L2 ≈ **4.5e-6**, comfortably below
the 2e-5 gate. The offline sim GATES: comfortably-below authorizes building the
kernel; at/above records the precision-floor datum instead.

**On-device rel-L2 = quadrature(fp32-floor, bf16-term)**
([[BL-20260709-compensated-bf16x2-split-beats-fp32-floor]]). tmm_v1's measured
fp32 floor is **3.99e-7 — tiny** (the pure-GEMM regime: swiglu 6.36e-7,
rmsnorm_matmul 4.8e-7; NOT the add_rmsnorm-family ~1.46e-5, which carries a
RMSNorm square-reduce feedback tmm lacks). So predicted on-device
≈ sqrt(3.99e-7² + 4.5e-6²) ≈ **4.5e-6 ≈ the offline number** — the bf16 term is
the whole story here, ~4.5× under the gate. Confirm on-device by backing out the
bf16 term (√(ondevice² − floor²)) and checking it matches the offline sim, per
the family protocol. **D4 (4-product) is predicted UNNECESSARY** and should be
SKIPPED (worst ≈4.5e-6 « the 1.8e-5 danger band the siblings used).

---

## 5. Measurement protocol (the loss-lesson requires it)

Per [[BL-20260710-bf16x2-loses-when-fp32-emulates-in-2-passes]], MEASURE the PE
delta before promoting — do not assume the sibling win transfers:

1. Offline sim first (§4). If it clears the gate → build the kernel.
2. `verify.py --fast` on tmm_v2 → correctness + latency direction.
3. `runs/dump_metrics.py --fast` on tmm_v2 AND a same-session tmm_v1 anchor →
   read **TRUE PE-active/inf** and **matmul_instruction_count** for both.
   Per-instruction rate = TRUE PE-active / matmul_instruction_count:
   - v1 anchor ≈ 4.718 ms / 24576 ≈ 0.192 µs/instr (fp32, 2.0/site).
   - **KEEP if v2 TRUE PE-active DROPS** (sibling signature: instrs +50%, but
     PE-active −~20%). **REJECT if it RISES** (swiglu signature: +18.6%) → the
     split loses on this op, the fp32 floor is terminal, keep v1, record the
     per-instruction-rate datum.
4. On a KEEP, confirm on the **full 5-seed run** and rank by stable p50 against a
   same-session tmm_v1 control band ([[BL-20260709-fast-vs-full-run-latency]]);
   promote only if outside the noise band.

Keep tmm_v1 as the guaranteed **pure-fp32 fallback** (the 5 profiler seeds reuse
the seed-42 input, so on-device input diversity is weak; the offline sim's
distinct draws are the real margin evidence).

---

## 6. Secondary levers (gated on D1's post-profile)

- **D2 — N_CHUNK 456→512 + masked tail.** 10944/512 = 21.375 → 21 full 512-wide
  chunks + 1 tail of 192, i.e. 22 chunks vs 24. Fewer, wider matmuls cut
  per-instruction issue overhead — amplified 3× under the split (each site is 3
  matmuls). Cost: reintroduces the `mask=…>=0` arithmetic phase-1 deliberately
  removed (the baseline's largest bug surface). Total PE columns pushed are ~the
  same either way, so the gain is only the fixed-overhead reduction — **screen
  with `--fast` after D1 lands**; adopt only if it clears the same-session band
  and stays correct (mask off-by-one is the risk).
- **D3 — M_BLK 8→16 (or 4).** Cuts rhs re-reads 4×→2×, but DMA is already hidden
  (21%), so it cannot help a PE-bound wall clock unless D1's limb-building shifts
  DMA off hidden. Enlarging M_BLK also enlarges the resident lhs-limb live set,
  which can **CONSTRAIN the affine_range software pipeline** (the bmm
  cross-batch-blocking anti-lever, [[BL-20260710-cross-batch-blocking-is-an-antilever-on-affine-range]]).
  Treat as a possible anti-lever — screen `--fast` only, do not assume monotone
  benefit.
- **D4 — 4-product.** SKIP (predicted numerically unnecessary, §4; siblings
  measured +~28% latency for a term swamped by the tiny fp32 floor). Build only
  if D1 surprises with a marginal on-device reading, and then record as a
  measured reject unless the backed-out bf16 term is itself near the gate.

---

## 7. Exit criteria / plan for the ≤5-iteration budget

- **Iter 0:** offline sim → numeric gate (no spend).
- **Iter 1:** build tmm_v2 (D1), `--fast` verify + dump_metrics vs same-session
  v1 anchor → measure the PE-active sign.
- **Iter 2:** if KEEP, full-5-seed confirm + control band → promote tmm_v2.
- **Iter 3 (optional):** D2 N_CHUNK=512 `--fast` screen, composed on tmm_v2.
- **Iter 4 (optional):** D3 M_BLK `--fast` screen only if D1 moved the DMA
  balance.

Success = a promoted kernel that beats tmm_v1's 1.026x (target ≈1.28x) with
full-5-seed L2 PASS, HBM still at the floor, and the PE-active drop
documented against the same-session anchor. Failure mode (split RISES PE-active)
= fp32 floor is terminal, keep tmm_v1, record the per-instruction-rate datum as
the second confirming data point (after swiglu) that the 2.0/site count can go
either way. Record every candidate in `benchmark.csv` + `candidates.jsonl` (DAG
parent links), profiling evidence under `profile/`. Never regress correctness.

--- Original Design Draft End ---
