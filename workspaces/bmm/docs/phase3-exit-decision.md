# bmm Phase 3 — Exit Decision

**Terminal result: keep `bmm_v2` (1.253x, 2.0363 ms). No phase-3-tested pure-fp32
batch-stream-depth schedule recovered the ~13.3% residual PE bubble.**

This was run as a hypothesis test, and the hypothesis was **refuted with measurement**: the one
structural lever phase 2 left (batch-axis stream depth) does not help — it *hurts*.

## What the phase asked

bmm_v2's TRUE PE-active (2.0138 ms this session) sits +13.4% above the trn2 fp32 PE floor
(1.775 ms). Classic shape specialization has no surface (no edge tiles — every axis divides
cleanly; tiles maximal at the 512 fp32 PSUM-bank wall; K=64 single-pass fixed and cost-free), so
the only lever is PE scheduling across the batch axis — the one dimension phase 2 left serial.

## Round 0 — premise CONFIRMED (AC-1)

Fresh same-session full-run `dump_metrics` on `bmm_v2` (profile/bmm_phase3_round0_anchor.txt):

| signal | this session | verdict |
|---|---|---|
| TRUE PE-active / inf | 2.0138 ms (98.90%) | PE-bound |
| DMA-active / inf | 1.366 ms (raw 3.0245 / window 2.214) | < PE-active → DMA hidden |
| matmul_instruction_count | 8704 | == phase-2 |
| per-matmul PE-active | 0.2314 us (vs 0.2133 floor → 0.0181 us/instr gap) | schedulable residual |
| HBM read / write | 34 / 1074 MB | floor, no spill |
| rel-L2 (5 seeds) | 1.827975e-07 | == phase-2 |

Decided PE-bound on the **counter-level DMA-active time**, NOT the coarse DMA% (67% this
session — jittery, swung 1–100% on this kernel in phase 2). Codex (high effort) AGREED (high
confidence); the window-normalization (raw active_time_ns ÷ (total_time_ns/p50)) is sound.

## D1 — multi-batch blocking: measured ANTI-LEVER (AC-2, AC-2.1, AC-6)

Both variants are **bit-exact pure reschedules** (identical `matmul_instruction_count` 8704,
`psum_read_sbuf_write_count` 4224, HBM 34/1074 MB no spill, rel-L2 1.827975e-07 == seed 42) that
**regress monotonically**:

| kernel | p50 | TRUE PE-active/inf | per-matmul PE-active | vs bmm_v2 |
|---|---|---|---|---|
| bmm_v2 (anchor) | 2.0363 ms | 2.0138 ms | 0.2314 us | — |
| D1 B_BLK=2 (transition 8×) | 2.6099 ms | 2.5722 ms | 0.2955 us | **+28.2%** |
| D1 B_BLK=4 (transition 4×) | 2.9182 ms | 2.8871 ms | 0.3317 us | **+43.3%** |

**Classification (AC-2.1):** not fixed-per-transition (that would DROP the stall monotonically
toward the floor), not a threshold win (that would recover at B_BLK=4), but an **anti-lever** —
deepening the cross-batch resident stream monotonically raises the per-matmul stall.

**Mechanism:** `bmm_v2` allocates small `rhs_sb`/`lhs_t_pack` buffers freshly *inside* the
`affine_range(B)` loop, so the compiler rotates them per batch — the batch boundary is a
**helpful pipeline reset**, not an unhidden bubble. D1 collapses `B_BLK` batches into one
long-lived `[B_BLK, 64, 4096]` resident block, doubling/quadrupling buffer live-ranges and
**constraining** the software pipeline (the SBUF-pressure regression mode from
BL-20260709-fast-vs-full-run-latency, where the higher-resident B=8 lost the full run). The
0.237 ms residual is therefore **not** a harvestable batch-boundary bubble; it is spread across
the main matmuls, and bmm_v2's per-batch rotation is already near-optimal. This refutes the
batch-boundary hypothesis *more strongly* than a no-op would.

D1 sweep hard-stopped per AC-2.1 (clear monotone regression). Both rejected per AC-6 (out-of-noise
TRUE PE-active rise, invariant counters confirm pure reschedule).

## D2 — cross-batch double-buffering: SKIPPED (AC-3)

AC-3 gates D2 on "**only if D1 lands as a compiler no-op**." D1 did not land as a no-op — it
**regressed**, a stronger negative signal. D2 (two concurrent `(rhs, lhs_t_pack)` buffer sets +
hand-rolled cross-batch prefetch) would add ≥64 KB/part resident footprint (like B_BLK=2) and
deepen the exact cross-batch resident overlap D1 just proved harmful. Two independent negatives:

1. The monotone D1 regression **isolates** cross-batch resident lifetime as the thing that hurts;
   D2 deepens it.
2. BL-20260709-dma-batching-regresses-pipeline: hand-rolled ping-pong via `sequential_range`
   regressed ~2x on silu because it **denies** the compiler the cross-iteration pipelining
   `affine_range` gives for free.

Codex (high effort) adversarially reviewed the skip decision and **AGREED (~0.8 confidence)**:
building D2 "looks like chasing a contradicted hypothesis, not adversarial due diligence." The
only D2 that could win would need to preserve bmm_v2's exact per-batch affine matmul schedule AND
prove the alternate prefetch is off the hot path — which the enlarged-live-set failure mode makes
unlikely. Contrast swiglu (where a deferred lever had a *positive* sub-baseline PE-floor signal →
build it, and it won); here **every** signal is negative. D2 SKIPPED.

## D3 — precision (bf16x2): CLOSED, record-only (AC-1.1, AC-4)

A compensated 3-product bf16x2 main matmul costs 3.0 bf16 passes > the phase-2-measured fp32/bf16
pass ratio of 2.0, so it would **raise** matmul work +50% and TRUE PE-active on a PE-bound kernel
(the swiglu all-3 sign-flip, 0.409x). Offline rel-L2 4.44e-6 would pass the 2e-5 accuracy gate,
but **both** gates are required and the **cost** gate fails. No bf16x2 kernel built; bf16 output
banned (2e-5 gate). D3 SKIPPED.

## Discipline (AC-5, AC-7)

- **Correctness invariance:** every candidate is a pure reschedule — rel-L2 1.827975e-07 exact,
  `matmul_instruction_count` 8704, single-pass K=64, transpose-before-use exact.
- **Fallbacks retained:** `bmm_v2` (promoted, 1.253x) and `bmm_v1` (0.663x pure-fp32) kept;
  NKIBench baseline/reference never edited.
- **Iterations:** 2 optimization iterations used (D1 B_BLK=2, D1 B_BLK=4). Round-0 re-anchor and
  D3-record excluded per AC-1/AC-7. ≤5 budget respected (D2 skip and D3 close spent 0).
- **Evidence:** benchmark.csv + candidates.jsonl DAG rows for the re-anchor, both D1 screens, and
  the D2-skip node; profile/ holds the anchor, both screens, and the round-0 analysis.

## Bottom line

`bmm_v2` at 1.253x is within 13.4% of the hard fp32 PE floor (ceiling 1.437x). Phase 3
established rigorously that the residual is **not** a batch-boundary bubble — the batch loop's
per-batch buffer rotation is already the compiler's preferred schedule, and every tested
deepening of the cross-batch stream regresses. The pure-fp32 lever is exhausted; the only route
past the PE floor (precision) is cost-closed. **bmm_v2 remains the promotion.**
