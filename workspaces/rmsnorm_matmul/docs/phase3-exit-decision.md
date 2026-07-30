# rmsnorm_matmul (M4096 N2048 K1024, fp32) — Phase 3 Exit Decision

`out = rmsnorm(x) @ w`, RMSNorm over K. Fixed shape M=4096, N=2048, K=1024, fp32. Scored
single-core on the remote profiler; correctness = NKIBench relative-L2 < 2e-5 on seeds
`[0,21,42,63,84]` (`verify.py` gates on `l2_norm_passed` for all seeds).

## Outcome: the fp32 rate ceiling was BROKEN — new best kernel promoted

Phase 3 was scoped expecting a **floor-confirmation exit** (v1 stays at 1.066×). The two
within-fp32 levers landed exactly as predicted (within noise), but the offline-gated
precision-split swing **succeeded**, producing a large, out-of-noise, full-5-seed-correct win.

| Kernel | Latency (ms) | Speedup vs baseline (0.502647 ms) | vs v1 | Correctness | Decision |
|---|---|---|---|---|---|
| v1 (pure fp32 fused) — fallback | 0.4714 / 0.4716 | **1.066×** | — | full-5-seed PASS | kept as fp32 fallback |
| v2_postscale (eviction fold) | 0.4733 | 1.062× | +0.4% (noise) | full-5-seed PASS | enabler (base for P1/P3) |
| stationary-reuse reorder | 0.4723 | 1.064× | +0.19% (noise) | full-5-seed PASS | NOT promoted (within-noise closing datum) |
| **bf16×2 split — PROMOTED** | **0.3687 / 0.3688** | **1.363×** | **1.279× faster** | **full-5-seed PASS ×2** | **promoted (new best)** |

`runs/rmsnorm_matmul_v4_bf16_split.py` is the new best kernel at **1.363×**;
`runs/rmsnorm_matmul_v1.py` remains the guaranteed pure-fp32 fallback (1.066×).

## How each Phase-3 job resolved

**1. Shape-lever closure (AC-6):** every classic phase-3 lever is vacuous or already at its
constraint — recorded in `docs/shape-specialization-closure-phase3.md`. No edge/partial tiles
(all dims divide evenly), layout forced by `nc_matmul` (k on partition, `[m_in, n]` output),
N_CHUNK=512=psum_fmax already maximal, w already fully resident so M-blocking is vacuous, LNC2
out of contract.

**2a. Stationary-activation reuse reorder (P1):** reordered the main matmul to `kt`-outer /
`c`-inner with 4 live `[128,512]` fp32 PSUM banks, cutting stationary fills 4× (1024→256).
Numerically equivalent; full-5-seed PASS; **0.4723 ms = +0.19% vs control, within noise**, PE
unchanged at 97%. Decisive closing datum: the stationary fills were **already hidden** under the
PE-bound matmul (consistent with the phase-2 `load_transpose2d` result). NOT promoted (Codex
concurred: within-noise, keep v1). This confirms there is no exposed within-fp32 micro-lever.

**2b. Compensated bf16×2 split-matmul (P3):** the only lever that can move the fp32 rate ceiling.

- **Offline pre-check (AC-4, zero remote spend):** `runs/offline_bf16_split_sim.py` reproduced
  the exact scored input (seed 42) and the NKIBench reference; the fp32 control matched the
  reference to 4.84e-7 (validating seed/draw-order/dtype/formula). The idealized bf16×2 3-product
  worst rel-L2 across 7 distinct input draws was **4.455e-6** — comfortably below the 2e-5 gate and
  the 1.5e-5 spend threshold (~3.4× better than the plan's naive ~1.5e-5 estimate, because the
  relative-L2 over the K=1024 accumulation partially cancels per-limb rounding). This **authorized**
  one remote attempt (AC-4.2 positive; Codex concurred).
- **On-device result (AC-5, HARD gate):** `--fast` PASS, then two full-5-seed runs PASS at 0.3687
  and 0.3688 ms, with the same-session v1 control re-confirmed at 0.4716. **−21.8% vs control =
  1.279× faster than v1**, far out of noise. Clears BOTH AC-5 gates (full-5-seed PASS AND
  out-of-noise win). **PROMOTED.**
- **Design:** split each fp32 operand into two bf16 limbs (`x_hi=bf16(x)`, `x_lo=bf16(x−x_hi)` RNE;
  `w_hi`/`w_lo` the same, w split once resident), accumulate 3 bf16 products
  `x_hi@w_hi + x_hi@w_lo + x_lo@w_hi` in fp32 PSUM (drop `x_lo@w_lo` ≈ 1e-6), RMSNorm stays fp32,
  per-row `inv_rms` applied post-scale at eviction. bf16-rate matmul (3 passes at ~3.23× fp32 rate
  ≈ 0.93× fp32 compute) plus hidden limb overhead nets the measured 1.279× end-to-end.

**P2 (contingency):** skipped by measurement — P1 surfaced no exposed Vec/Scl bubble (Vec 7% / Scl
17%, both hidden), so no norm-fold rebalance was warranted.

## Why the plan's "fp32 floor is the ceiling" expectation was beaten

The plan (and the sibling `matmul` task) treated the 2e-5 gate as effectively forbidding a
lower-precision matmul, because plain bf16 (~4e-3 error) fails by orders of magnitude and the naive
compensated-bf16 error estimate (~1.5e-5) sat razor-thin against the gate. Two things made the
difference here: (a) the **two-limb compensation** recovers ~16 effective mantissa bits, and (b) the
**relative-L2 metric over a K=1024 dot-product** averages/cancels per-limb rounding, so the *actual*
error (4.5e-6) is ~4.5× under the gate rather than marginal. The offline sim measured this before
spending any remote iteration, and the on-device full-5-seed gate confirmed it holds on hardware.

## Residual risk (recorded, not blocking)

The adapter fixes `np.random.seed(42)` for every profiler seed (`adapter/nkibench_case.py:47`), so
the on-device "5-seed PASS" is a repeat of the same input, weak on *input* diversity. This is
mitigated by the offline sim exercising 7 genuinely distinct input draws (all ~4.455e-6, ~4.5× under
the gate). If a future evaluator used genuinely distinct per-seed inputs, the on-device evidence
would be overstated — the offline margin makes that acceptable, not zero-risk. v1 remains the
guaranteed pure-fp32 fallback if a stricter correctness regime is ever required.

## Evidence
- `benchmark.csv` / `candidates.jsonl` — v1 control, v2_postscale, stationary-reuse, offline-sim
  datum, bf16×2 split (promoted), P2-skipped, and both Codex review nodes (DAG with parent links).
- `profile/rmsnorm_matmul_v4_bf16_split.txt` — bf16×2 split profiler digest + interpretation.
- `profile/rmsnorm_matmul_offline_bf16_split_sim.txt` — offline sim full output.
- `docs/shape-specialization-closure-phase3.md` — AC-6 lever closure.
- `runs/rmsnorm_matmul_v4_bf16_split.py` — promoted kernel; `runs/rmsnorm_matmul_v1.py` — fp32 fallback.
