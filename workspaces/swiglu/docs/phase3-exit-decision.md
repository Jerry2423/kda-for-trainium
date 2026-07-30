# SwiGLU Phase 3 — Outcome: WIN at `swiglu_v3_mblock` B=4 (1.026×, first to beat baseline)

## TL;DR
After three candidates, phase 3 **beats the baseline**: `swiglu_v3_mblock` (B=4) =
**2.0219 ms / 1.026×**, correct (on-device rel-L2 = 4.476e-6 « the 2e-5 gate). It is the
first SwiGLU kernel to exceed the 2.074 ms baseline (v1 was 0.939×). The winning recipe is
**down-GEMM-only bf16x2 + M-tile-block (B=4)**:

1. **bf16x2 on all three GEMMs (`swiglu_v2`) LOSES (0.409×).** The profiler's absolute
   `matmul_instruction_count` shows a correct fp32 SwiGLU GEMM emulates in only ~2 bf16 passes
   here, so the 3-product split (3 passes) is +50% matmul instructions / +33% PE compute — it
   makes the PE do *more* work. (The plan's ≥3.6-pass assumption, from the matmul sibling, does
   not hold on this op.)
2. **bf16x2 on the down GEMM only (`swiglu_v2_only_down`) = 0.971×**, correct: its true PE-active
   (1.95 ms) is actually *below* v1's fp32 2.09 ms, but it is DMA-bound (DMA=100%, HBMrd 656 MB)
   because the fp32 w_down chunk is reloaded per M-tile (B=1) to rebuild its bf16 limbs on-chip.
3. **M-blocking that DMA-bound rung (B=4) amortizes the weight-DMA/limb-rebuild 4×** (HBMrd
   656→205 MB, DMA 100→25%), dropping latency to the ~1.92 ms PE floor → **1.026×, PROMOTED.**

**Terminal decision: PROMOTE `swiglu_v3_mblock` (B=4, 1.026×) as the phase-3 win; keep
`swiglu_v1` as the guaranteed-correct fp32 fallback.** Codex (high effort) R1 review correctly
pushed to build the M-block (I had wrongly deferred it); the win vindicates that push.

## The measured ground truth (from `runs/dump_metrics.py`, per-inference, window-normalized)
Profiler metric windows captured ~2.0 iterations (`total_time_ns / p50` ≈ 2.005), so raw
`active_time_ns` was divided by the window count. On-device rel-L2 is the per-seed
`relative_l2_error` from `multi_seed_correctness` (all five values byte-identical — seed-42×5
caveat below).

| kernel | p50 | speedup | TRUE PE-active | PE% | DMA% | HBMrd | matmul_instr | on-device rel-L2 |
|---|---|---|---|---|---|---|---|---|
| baseline | 2.074257 ms | 1.000× | — | — | — | — | — | — |
| `swiglu_v1` (fp32) | 2.2077 ms | 0.939× | 2.092 ms | 94.8% | 49% | 607 MB | 10240 | 6.36e-7 |
| `swiglu_v2` (all-3 bf16x2) | 5.0626 ms | 0.409× | 2.480 ms | 49.0% | 24% | 485 MB | 14848 | 7.708e-6 |
| `swiglu_v2_only_down` (down bf16x2, B=1) | 2.1372 ms | 0.971× | 1.947 ms | 91.1% | **100%** | 656 MB | 11776 | 4.476e-6 |
| `swiglu_v3_mblock` B=2 | 2.1875 ms | 0.948× | — | 91% | 31% | 314 MB | — | 4.476e-6 |
| **`swiglu_v3_mblock` B=4 (PROMOTED)** | **2.0219 ms** | **1.026×** | **1.917 ms** | 94.8% | 25% | 205 MB | 11776 | 4.476e-6 |

**All on-device rel-L2 match their offline-sim predictions** (v1 fp32 floor 6.36e-7; down-only
4.476e-6 ≈ offline 4.447e-6; all-3 7.708e-6 ≈ offline 7.722e-6) — bf16x2 behaves exactly as
modeled; correctness was never the issue.

## Why all-3 bf16x2 loses but down-only + M-block wins
**Instruction-count decomposition (static per-kernel, the airtight part):**
- **v1**: 10240 = 1024 transposes + 9216 GEMM instr over 4608 fp32 GEMM sites = **exactly 2.0
  matmul-instr / fp32 site** (MFU 44.4% ⇒ 1/0.444 = 2.25 pass-equiv).
- **v2 (all-3)**: 14848 = 1024 transposes + 13824 over 13824 bf16 sites = **exactly 3.0 / site**
  (the 3-product split). ⇒ **+50% matmul instructions / +33% PE compute** — the OPPOSITE of the
  matmul sibling (fp32 ≈ 3.6× bf16, where 3 products won at 1.28–1.36×). All-3's true PE-active
  rose to 2.480 ms (PE%=49, not the ~35 a serialization artifact would give — genuine PE work),
  AND its full 3-weight per-M-tile limb rebuild was exposed on Vec+Scl. 0.409×.
- **down-only**: 11776 matmul instr (only the down GEMM tripled; up/gate stay fp32 at 2/site).
  Its true PE-active (1.95 ms) is **below v1's** — the down-GEMM bf16x2 is PE-cheaper than fp32
  here (down is 1/3 of the GEMM MACs, and 3 bf16 passes < the fp32 emulation cost on that
  moving-width), and keeping up/gate fp32 avoids their limb rebuild. So the earlier "PE-active is
  monotone increasing in #bf16x2-GEMMs" claim was **wrong** — corrected by this measurement.
- **The real bottleneck story:** down-only's sub-baseline PE floor was masked by a DMA wall — at
  B=1 the fp32 w_down chunk is reloaded per M-tile to rebuild its bf16 limbs (HBMrd 607→656 MB,
  DMA 49→100%). **M-blocking (process B M-tiles per weight stream) is exactly the lever that
  amortizes that reload B×.** At B=4: HBMrd 205 MB, DMA back to 25% (hidden under PE), latency
  falls to the 1.92 ms PE floor = 1.026×. B=2 (2.188 ms/0.948×) only amortizes DMA 2× and is
  still PE-bound *before* the DMA fully hides, so B=4 wins.

## Part B (M-block `swiglu_v3_mblock`) — TRIGGERED and BUILT (AC-6)
- **Pre-registered numeric trigger:** build M-block iff a bf16x2 candidate shows weight-DMA
  exposed off "hidden" (DMA-active ≳ 0.85× PE-active) with a sub-baseline PE floor.
- **Measured trigger crossing:** `swiglu_v2_only_down` is DMA-bound (DMA=100%, normalized
  DMA/PE ≈ 1.1 » 0.85) AND its true PE floor (1.947 ms) is *below* the 2.074 ms baseline — so
  M-blocking (which amortizes exactly that exposed weight-DMA) had a credible sub-baseline
  target. (Note: the all-3 `swiglu_v2` did NOT trigger M-block — it is PE-bound at 2.480 ms >
  baseline, so amortizing its hidden DMA cannot help; M-block was correctly applied to the
  DMA-bound *down-only* rung, not the PE-bound all-3.)
- **Built B=2 then B=4, kept B=4.** PSUM budget honored: up/gate use 2B fp32 banks → B=4 = 8
  banks (the max). Per-M-tile activation state ≈ 28 KB/part × 4 = 112 KB/part, within budget
  (HBMwr stayed 17 MB = output-only, no spill). B=4 is the promoted win.
- **Promotion discipline:** `--fast` (1.025×) and full (1.026×) agreed; a same-session
  interleaved noise band confirmed it — B4 {2.0209, 2.0202, 2.0193} ms vs v1 {2.2053, 2.2068,
  2.2061} ms, NON-overlapping, jitter ~0.08%, B4 stably < 2.074 baseline on every run.

## R2 / R3 (AC-6 investigate-and-close)
- **R2 — edge/ragged-tile regime: NO-OP.** M=4096, K=1024, N=3072 are all exact 128-multiples;
  no ragged tiles. Not implemented (recorded).
- **R3 — free-chunk & transpose-layout: REJECT.** CHUNK=512 = one fp32 PSUM bank (the max
  moving-free width); **CHUNK=1024 is ILLEGAL** (exceeds it); a smaller CHUNK only raises the
  matmul-site count. So CHUNK=512 is fixed. Transposes (1024 matmul instr, 5.3% of PE) are part
  of the PE floor but not removable within budget: off-PE transpose is a precedent reject
  (`dma_transpose` fp32-ineligible/exit-70; `nc_transpose` vector-only regresses —
  [[BL-20260709-offpe-transpose-hidden-under-pe-floor]]); the D3 layout-swap loses on the cost
  model (+30720 ec to save 6144 ec). Recorded reject. (Note: R4 — the mixed-precision ladder —
  is no longer a "close" decision: down-only is not just a ladder rung, it is the parent of the
  promoted M-block win.)

## Seed caveat (AC-5) — honest, with the measured on-device rel-L2
The on-device "5 seeds" [0,21,42,63,84] all draw identical seed-42 inputs
(`adapter/nkibench_case.py` reseeds `np.random.seed(42)` before every draw) — confirmed: all
five per-seed `relative_l2_error` are byte-identical (v3_mblock_b4 = 4.475585e-6 on every seed).
The TRUE distinct-seed numerical evidence is the offline sim (down-only bf16x2 worst rel-L2 =
4.447e-6, essentially seed-independent); the on-device 4.476e-6 matches it. Because ONLY the down
GEMM is bf16x2, the promoted kernel is on the *smallest* (safest) error rung of the ladder, 4.5×
under the gate. (verify.py gates on `l2_norm_passed` = True on all seeds; the profiler's aggregate
`all_seeds_passed` flag reads False due to a stricter allclose/max_abs sub-check, NOT the NKIBench
L2 gate.) Fixing the shared adapter remains out of scope (a separate infra change).

## What this phase contributes
- **A promoted 1.026× kernel** (`swiglu_v3_mblock` B=4) — the first SwiGLU kernel to beat
  baseline, up from v1's 0.939×.
- **A reusable, non-obvious recipe** (BitLesson `BL-20260710-bf16x2-loses-when-fp32-emulates-in-2-passes`):
  when a compensated bf16x2 split does NOT help (because fp32 emulates in only ~2 passes on this
  op), it can still be a net win **on a single GEMM** if (a) that GEMM's bf16x2 is PE-cheaper than
  its fp32 emulation, and (b) the resulting per-tile fp32-weight-reload-for-limb-rebuild DMA wall
  is amortized by M-blocking. Measure the per-op fp32-emulation matmul-pass multiple and each
  rung's true PE-active + DMA% before deciding — do NOT reject the ladder by monotonicity
  projection (down-only breaks strict PE monotonicity).
- **A zero-throwaway-kernel diagnostic**: the profiler's `summary_metrics` returns absolute
  `tensor_engine_active_time_ns`, `matmul_instruction_count`, AND per-seed `relative_l2_error` —
  dump them (via `runs/dump_metrics.py`) instead of building a PE-probe kernel or trusting the
  5-percent digest verify.py prints. This is what turned "all-3 lost, stop" into "down-only has a
  sub-baseline PE floor hidden behind a DMA wall → M-block it."
