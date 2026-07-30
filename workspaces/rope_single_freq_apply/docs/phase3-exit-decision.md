# rope_single_freq_apply — Phase 3 exit decision (regime / shape specialization)

## Outcome: AT ACHIEVABLE CEILING. Incumbent `rope_v3_layoutB_pe.py` (W=512, 0.696 ms, **1.641x**) is KEPT. No finer-W variant clears the promotion bar.

The one real regime surface — free-axis tile width finer than the phase-2 W=512 floor —
was swept (W=256, W=128) on the promoted PE/hybrid layout B. It found a genuine but
**sub-noise-band** optimum at W=256 (~0.95% faster) with W=128 confirming the turn. Below
the AC-3 >3-5% bar → not promoted. The kernel is DMA-co-limited at ~74% of the streaming
roofline with HBM at the byte floor; the residual gap is fixed software-DMA descriptor
overhead, not a tunable lever.

| kernel | build | tile W | latency p50 | vs baseline | effBW (% of 781) | verdict |
|--------|-------|--------|-------------|-------------|------------------|---------|
| baseline (NKIBench) | — | 512 | 1.1418 ms | 1.000x | — | — |
| `rope_v1` layout A | vector-bound | 2048 | 0.9445 ms | 1.209x | 427 GB/s (55%) | phase-1 |
| `rope_v2_layoutB_scalar_w512` | all-Scalar | 512 | 0.7337 ms | 1.557x | 549 GB/s (70%) | exact fallback |
| **`rope_v3_layoutB_pe`** | **PE + Scalar** | **512** | **0.696 ms** | **1.641x** | **580.6 GB/s (74.3%)** | **KEPT (record)** |
| `rope_v3_layoutB_pe_w256` | PE + Scalar | 256 | 0.6870 ms | 1.662x | 586.2 GB/s (75.1%) | finest-best, sub-band |
| `rope_v3_layoutB_pe_w128` | PE + Scalar | 128 | 0.6986 ms | 1.634x | 576.5 GB/s (73.8%) | the turn |
| `rope_v2_layoutB_scalar_w256` | all-Scalar | 256 | 0.7335 ms | 1.556x | — (Scl-bound) | D2 re-race, PE wins |

All finer-W candidates are **exact** (per-seed rel-L2 = 0.0; W=256 full 5-seed, W=128 fast
seed 42), HBM at the read-once/write-once floor. Incumbent 5-seed gate re-confirmed this
round: 5/5 exact, p50 0.6929 ms.

## Why no finer W wins (the mechanism, measured)

**D1 — the effBW turn.** effBW = total_bytes/latency rises to a W=256 optimum then falls:
74.3% (W512) → 75.1% (W256) → 73.8% (W128) of the 781 GB/s roofline. A deeper
`affine_range` pipeline amortizes the fixed DMA fill/drain bubble over more steady-state
steps (effBW rises); at very small bursts (0.5 KB/partition at W=128) per-tile issue +
per-tile `nc_matmul` overhead overtakes the pipeline-depth gain (effBW falls). Phase 2 had
already harvested the bulk of the bubble at W=512 (DMA-active 94% → 98%); W=256 harvests
only the last sliver (98% → 99%), worth ~0.95%. HBM byte-traffic is flat across the sweep;
DMA% (98-99%) is too coarse to show the turn — effBW is the resolving diagnostic.

The full-run interleaved bracket (A=W512, B=W256) is consistent (all 3 B-runs below all 3
A-runs) but tiny: A mean 0.6936, B mean 0.6870 → ~0.95%, INSIDE the ~3-5% DEC-1 noise band.
One bracket past the finest-best (W=128 regressed) confirms the turn → the sweep stops
(bounded, per BL-20260709-finer-tiling-harvests-dma-bubble). **AC-3 not met → keep incumbent.**

**D2 — engine placement re-raced at W=256.** PE build still beats all-Scalar by ~6.5%
(0.6859 vs 0.7335, non-overlapping) — outside the band, so DEC-1's tie-break toward the
simpler throttle-free build does NOT trigger. KEY finding: Scalar_w256 (0.7335) ≈
Scalar_w512 (0.7337) because the all-Scalar build is **Scl-bound (92%)**, not DMA-bound —
finer W harvests no bubble for it. **Only the DMA-co-limited PE build benefits from finer
W**, which cleanly isolates the finer-tiling mechanism. The PE power-throttle (§2b) persists
essentially unchanged at W=256 (`throttle_active_nc0` 724k ns ≈ 737k ns at W=512; total PE
work is identical) — it is a real-but-hidden cost, not a margin-flipper.

**D3 — DMA burst/queue shaping: investigated no-lever.** No queue-starvation or
fragmentation to exploit: `dma_queue_count=35` (compiler already spreads the 4 streams),
`input/output/weight_queue_bytes = 0`, DMA saturated ~99%. And §2a `dge_mode` is
re-confirmed dead — `--disable-dge` forces `DgeType=None` globally; the profile shows DMA is
~99.5% software-dynamic. The residual ~25% effBW gap to roofline is fixed software-DMA
descriptor overhead the kernel cannot tune.

## Acceptance criteria — final status

- **AC-1 (exactness):** PASS. Every candidate arithmetic-preserving; per-seed rel-L2 = 0.0
  (W=256 full 5-seed; W=128/Scalar_w256 fast seed 42). Incumbent 5-seed re-confirmed exact.
- **AC-2 (HBM at floor):** PASS. All candidates 268.5 + 134.2 = 402.72 MB (floor + 65 KB
  `swap_const`); no spill, no extra traffic at any W.
- **AC-3 (measured win justifies complexity):** the finest-best W=256 wins only ~0.95%
  (< the 3-5% bar) → **not promoted**; incumbent W=512 kept. The sweep is the evidence.
- **AC-4 (robustness tie-break):** N/A-triggered — PE beats Scalar by ~6.5% at W=256
  (outside the band), so the tie-break condition (within-band) never fires; PE kept.
- **AC-5 (honest ceiling):** DELIVERED. The finer-W sweep (turn at W=256/128) + the dead
  `dge_mode` (§2a) + the persistent-but-hidden PE-throttle (§2b) + no queue-starvation (D3)
  together explain why 0.696 ms / 74%-of-roofline is the achievable ceiling under
  `--disable-dge --logical-nc-config=1`.

## Final kernel of record

`runs/rope_v3_layoutB_pe.py` — layout B, PE/hybrid build, W=512. **0.696 ms, 1.641x over
the 1.1418 ms baseline** (1.358x over layout A), exact on all 5 seeds. DMA-co-limited at
effBW 580 GB/s = 74% of the 781 GB/s single-core streaming roofline, HBM at the byte floor.
`rope_v2_layoutB_scalar_w512` (0.7337 ms) remains the guaranteed-exact, throttle-free
fallback if a future compiler revision breaks the `nc_matmul` 0/±1 exactness.

Deliverables: finer-W kernel sources (`runs/rope_v3_layoutB_pe_w{256,128}.py`,
`runs/rope_v2_layoutB_scalar_w256.py`), `benchmark.csv` + `candidates.jsonl` rows,
`profile/rope_phase3_finer_wsweep_digest.md`, this exit decision.
