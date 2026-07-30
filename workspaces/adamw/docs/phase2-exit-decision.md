# adamw phase-2 exit decision — PROMOTE adamw_v2_ch1216 (2.330x)

## Outcome

**Promoted `runs/adamw_v2_ch1216.py` at 2.330x (0.5601 ms)**, +10.1% over the phase-1
winner `adamw_v1` (2.110x / 0.6180 ms). `adamw_v1` is kept as the documented fallback.

The D1 lever fired exactly as the plan (and the silu phase-3 precedent) predicted: a
**mask-free `(128, ITERS, CH)` reshape-view stream** with a bounded CH burst-width sweep
harvested the ~5% DMA-idle bubble. effBW rose **726 → 799.6 GB/s** — dead-on silu's
achieved streaming roofline (~799.5) and the plan's 2.33x ceiling — with **HBM traffic
unchanged at 448 MB** (359 rd + 90 wr). This is a pure scheduling win, never a traffic one.

## What was done (D1, PRIMARY)

Ported `adamw_v1` to the reshape-view shape: `M*N = 10944*2048 = 22413312 = 128 * 175104`
exactly, so the contiguous row-major buffer reinterprets as a clean `(128, 175104)` flat
view with all 128 partition lanes live — **no partial tail, no mask on any DMA** (v1 had a
`10944 = 128*85 + 64` partial-tail predicate on every load/store). Reshape each input and
the output to `(128, ITERS, CH)` (a pure-stride, no-DMA view) and walk one flat
`nl.affine_range(ITERS)`. The folded algebra and the 6-op fused chain (2 Scalar + 4 Vector)
are **byte-for-byte unchanged** from v1 — only the loop shape changes.

Correctness is layout-invariant (pure elementwise op): numpy confirms the
`(M,N) → (128,ITERS,CH) → (M,N)` round-trip is bit-exact and the folded algebra over the
reshaped view is **rel-L2 = 3.42e-8** vs the reference (~580× under the 2e-5 gate). Every
scored candidate passed the 5-seed L2 gate.

## The sweep (4 D1 candidates, ≤ 5 budget)

`--fast` screen (seed 42), same-session v1 `--fast` reference = 0.6185 ms / 2.110x / DMA 95%:

| CH | ITERS | burst/part | latency | speedup | DMA% | Vec% | effBW | role |
|----|-------|-----------|---------|---------|------|------|-------|------|
| v1 | 86 tiles | 8.00 KB | 0.6185 | 2.110x | 95 | 64 | 725 | parent |
| 1024 | 171 | 4.00 KB | 0.5921 | 2.204x | **99** | 69 | 757 | anchor (silu 2¹⁰ optimum) |
| 1152 | 152 | 4.50 KB | 0.6834 | 1.910x | 88 | 59 | 656 | bracket probe |
| **1216** | **144** | **4.75 KB** | **0.5614** | **2.325x** | **99** | 72 | **798** | **sweep-best** |
| 1536 | 114 | 6.00 KB | 0.6515 | 2.003x | 91 | 60 | 688 | wider probe |

**CH=1216 is an interior optimum, bracketed BELOW on both sides**: finer neighbours 1024
(2.204x) and 1152 (1.910x) are both lower, and the wider neighbour 1536 (2.003x) is lower.
Traffic is pinned at 448 MB (359+90) at **every** CH — the sweep never touches HBM bytes.
Per the bounded-sweep rule (`BL-20260709-finer-tiling-harvests-dma-bubble`), one bracket
probe (1152) confirmed the turn and the sweep STOPPED — no further divisors chased.

## Promote-test (interleaved full 5-seed A/B/A/B/A; A=v1, B=ch1216)

| run | kernel | latency | speedup | DMA | HBMrd | HBMwr | L2 |
|-----|--------|---------|---------|-----|-------|-------|-----|
| A0 | v1 | 0.6166 | 2.117x | 95% | 359 | 90 | PASS |
| B0 | ch1216 | 0.5601 | 2.330x | 99% | 359 | 90 | PASS |
| A1 | v1 | 0.6184 | 2.110x | 95% | 359 | 90 | PASS |
| B1 | ch1216 | 0.5612 | 2.325x | 99% | 359 | 90 | PASS |
| A2 | v1 | 0.6169 | 2.116x | 95% | 359 | 90 | PASS |

- `Abar = 0.61730 ms` (max 0.6184, spread 0.0018); `Bbar = 0.56065 ms`.
- **GATE1** `Bbar < Abar − J` (J=0.0002): 0.56065 < 0.61710 → PASS, delta **0.0566 ms =
  283× J = 31.5× the A-spread** (unambiguous, far outside any noise).
- **GATE2** every B < max(A): 0.5601, 0.5612 both < 0.6184 → PASS.
- **GATE3** traffic floor intact: HBMrd 359 MB + HBMwr 90 MB = 448 MB on every B → PASS
  (no accidental extra pass).
- **GATE4** 5-seed L2 gate: PASS on all 5 runs.

All four gates pass → **PROMOTE**.

## Why 1216 and not the silu-style smooth optimum (finding)

Unlike silu's smooth unimodal latency-vs-fineness turn, adamw's curve is **non-monotone /
bimodal**: latency tracks DMA% almost 1:1 across the four independent `--fast` runs
(CH=1024 and 1216 both software-pipeline to **DMA-saturation 99%** and win ~0.56–0.59 ms;
CH=1152, 1536 stall at 88–91% and lose ~0.65–0.68 ms). The tight latency↔DMA% coupling
across independent runs rules out noise. The likely cause is that adamw moves **5 DMA
streams per iteration** (4 loads + 1 store) vs silu's 2, so DMA-queue saturation is
**burst-width-specific** — certain CH widths resonate the descriptor cadence to keep the
DMA engine saturated, others leave it fill/drain-starved. The bounded sweep still isolates
a cleanly bracketed peak at CH=1216; going finer (1024, 1152) or wider (1536) both regress.

## Directions NOT taken (correctly)

- **D2 (manual double-buffer / ping-pong)** — NOT fired. Trigger was "D1 plateaus below
  ~2.25x with a residual DMA bubble the sweep can't close." D1 landed at **2.330x with
  DMA=99%** (bubble fully harvested, at the roofline) — the trigger never armed. Also
  pre-rejected: silu already showed ping-pong (`sequential_range`) regresses ~2× on this
  profiler by denying `affine_range`'s free cross-iteration pipelining
  (`BL-20260709-dma-batching-regresses-pipeline`).
- **D3 (compute-chain rebalance)** — NOT fired. Trigger was "the finer end lifts Vec%
  toward DMA%." At the CH=1216 optimum Vec=72% sits comfortably under DMA=99% (the 4-Vector
  chain stays hidden); no rebalance needed. (Vec% only *rose* on the DMA-saturated widths
  because more of the wall clock is productive DMA — it never threatened to become the
  limiter.)

Both remain queued, correctly out of scope — firing either would have been an unforced
error against a kernel already at the achieved streaming roofline.

## Bottom line

adamw phase 2 lands the mask-free reshape-view finer-tiling win the plan predicted, at the
**top** of its realistic band (2.15–2.30x) and effectively **at the 2.33x ceiling**
(effBW 799.6 GB/s = the streaming roofline). The kernel is DMA-bound at 99% on the
unchanged 448 MB traffic floor — there is no remaining lever. `adamw_v2_ch1216` is the
phase-2 winner; `adamw_v1` is the documented fp32 fallback.
