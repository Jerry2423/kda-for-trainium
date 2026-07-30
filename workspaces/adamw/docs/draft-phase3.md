# adamw (M10944 N2048, fp32) — Phase 3 implementation draft (regime / shape specialization)

## Starting point

Phase-2 winner **`runs/adamw_v2_ch1216.py`** — the mask-free reshape-view stream:
- **0.5601 ms, 2.330x** over baseline (full 5-seed PASS; `--fast` 0.5614 ms / 2.325x).
- Structure: reshape the contiguous `(10944, 2048)` buffer to a **pure-stride, no-DMA**
  `(128, ITERS=144, CH=1216)` view and walk one flat `nl.affine_range(144)`; per chunk
  4 **mask-free** loads → the 6-op fused chain (2 Scalar `activation` square/rsqrt +
  4 Vector `scalar_tensor_tensor`) → 1 mask-free store. Folded algebra
  `new_theta = 0.99999·theta − 0.001·(9m+g)·rsqrt(999v+g²)` (eps dropped, `v_hat>0`),
  byte-for-byte unchanged from `adamw_v1`.
- Profiler digest: **DMA 99%, Vec 72%, Scl 34%, PE 0%**; HBMrd 359 MB, HBMwr 90 MB.
- `adamw_v1` (2.112x, masked row tiles) kept as the documented fp32 fallback.

Phase 2 established the headline fact that governs this entire phase: the kernel is
**DMA-bound at the achieved streaming roofline**, on an **immovable traffic floor**.

## Where the time goes: at the roofline, on an immovable floor

Read the promoted numbers as a roofline (this is the phase-3 diagnosis, not a new claim):

| metric | ch1216 | meaning |
|---|---|---|
| DMA active | **99%** | the sole constraint; saturated |
| Vec active | 72% | #2 engine, **hidden** under DMA (0.72·0.5601 = 0.40 ms « 0.99·0.5601 = 0.55 ms) |
| Scl active | 34% | the two nonlinearities (square, rsqrt), off Vector |
| PE / MFU | 0% | no matmul (irrelevant) |
| HBMrd | 359 MB | 4 × 89.66 MB = **read-once** for theta,g,m,v (no re-fetch) |
| HBMwr | 90 MB | 1 × 89.66 MB = **write-once** for new_theta (no spill) |

- **Traffic floor = 448.3 MB** (4R + 1W). At silu's achieved roofline on this trn2
  profiler (~799.5 GB/s) that floor is **0.5616 ms → a 2.324x hard ceiling**. The
  measured 0.5601 ms is **0.9973× of that floor** — effBW 799.6–801.6 GB/s, i.e.
  *dead-on the roofline*. There is ≈**0.3 % headroom** left, and it is a hardware
  ceiling, not a schedulable bubble.
- **There is no traffic lever.** All four inputs are genuinely read (every element
  feeds the update), the output is written once, and the HBM tensors are fp32 supplied
  by the harness. bf16 would neither cut HBM bytes (the tensors *are* fp32 in HBM) nor
  survive the 2e-5 L2 gate on a pure elementwise op with **no reduction** to average the
  error down — the opposite of the rmsnorm/matmul siblings where K-averaging licensed a
  bf16x2 split. Traffic is pinned at 448 MB at every tiling phase 2 measured.

## Phase-3 lens: the reshape-view is shape-homogeneous → no edge/regime surface

Phase 3's mandate is "analyze where time goes across the tensor's **structure** and
specialize only where a measured win justifies the complexity (tile-size regimes,
partition/free splits, edge tiles)." The decisive structural finding for adamw:

`M·N = 10944·2048 = 22413312 = 128 · 175104` **exactly**, and `175104 = 2¹⁰·3²·19`, so
`CH=1216 | 175104` **exactly** (ITERS=144). The reshape-view therefore homogenizes the
problem into a **perfectly rectangular** `(128, ITERS, CH)` stream:

- **Zero edge tiles** — every chunk is a full `[128, CH]` rectangle; there is no partial
  tail to specialize (contrast `adamw_v1`, which carried a `row<10944` predicate on
  the 64-valid-row last tile).
- **Zero masks** — no DMA in the whole kernel is predicated.
- **All 128 partition lanes live** — the partition dim is already at hardware max; a
  partition/free-split regime cannot help (fewer partitions only underutilizes DMA).
- **Every tile byte-identical in structure** — there is nothing heterogeneous to
  regime-specialize.

**Conclusion:** the classic phase-3 levers — edge-tile specialization, partition/free
split regimes, mixed tile-size regimes — have **no surface** on adamw. The *only*
"regime" axis that exists is the burst width **CH**, which was already the phase-2
lever. "Regime specialization" collapses to the burst-width sweep. This is the honest
terminal structural statement (cf. bmm/transpose_matmul phase 3: "shape edge-free →
the remaining question is numerical/scheduling, not tile geometry").

## The one open question phase 2 left: is CH=1216 the *global* resonance peak?

Phase 2 documented that adamw's latency-vs-burst-width curve is **non-monotone /
bimodal** — distinct from silu's smooth unimodal turn — with latency tracking DMA% 1:1:

| CH | ITERS | burst/part | latency | speedup | DMA% | role (phase 2) |
|----|-------|-----------|---------|---------|------|----------------|
| 1024 | 171 | 4.00 KB | 0.5921 | 2.204x | **99** | 99% lobe |
| 1152 | 152 | 4.50 KB | 0.6834 | 1.910x | 88 | trough between the lobes |
| **1216** | **144** | **4.75 KB** | **0.5614** | **2.325x** | **99** | **99% lobe (best)** |
| 1536 | 114 | 6.00 KB | 0.6515 | 2.003x | 91 | wider trough |

Phase 2 declared 1216 the "interior optimum bracketed below on both sides" and **stopped
the sweep after one adjacent bracket** (1152), citing the bounded-sweep rule
`BL-20260709-finer-tiling-harvests-dma-bubble`. **That rule's stopping condition assumes
a smooth unimodal curve** — one bracket on each side proves a peak only when the curve is
monotone away from it. adamw's curve is explicitly *bimodal*: two 99% lobes (1024, 1216)
straddle an 88% trough (1152). On a resonant curve, a single adjacent bracket does **not**
rule out a *third* 99% lobe elsewhere in the band. Only **4 of the 12 in-band divisors**
were tested, so 1216 is proven a *local* optimum but **not the global** saturation peak.

Full in-band divisor lattice (`175104`, burst 2–8 KB/partition), phase-2 coverage marked:

| CH | ITERS | burst/part | phase-2 status |
|----|-------|-----------|----------------|
| 512 | 342 | 2.000 KB | untested (fine anchor) |
| 576 | 304 | 2.250 KB | untested |
| 608 | 288 | 2.375 KB | untested |
| 684 | 256 | 2.672 KB | untested |
| 768 | 228 | 3.000 KB | untested (finer gap probe) |
| 912 | 192 | 3.562 KB | **untested — gap between 768 and the 1024 lobe** |
| 1024 | 171 | 4.000 KB | 99% lobe (2.204x) |
| 1152 | 152 | 4.500 KB | 88% trough (1.910x) |
| **1216** | **144** | **4.750 KB** | **99% lobe — BEST (2.325x)** |
| 1368 | 128 | 5.344 KB | **untested — gap between the 1216 lobe and the 1536 trough** |
| 1536 | 114 | 6.000 KB | 91% trough (2.003x) |
| 1824 | 96 | 7.125 KB | untested (wide anchor) |

## Direction D1 (PRIMARY): complete the burst-band divisor sweep — screen for a co-equal or higher 99% lobe

**What:** screen the untested in-band divisors as `--fast` (seed 42) runs, reusing the
exact `adamw_v2_ch1216.py` kernel with only the `CH` constant changed (ITERS = 175104//CH,
both exact). Watch **DMA% and latency** — the phase-2 evidence shows latency tracks DMA%
1:1, so DMA% is the fast discriminator. Success = a width that screens **below 0.5601 ms
at DMA ≥ 99 %**.

**Order (bounded, resonance-mapping):**
1. `CH=912` (ITERS=192) and `CH=1368` (ITERS=128) — the two **gap probes** immediately
   adjacent to the known 99% lobes; most likely location of a missed co-lobe.
2. `CH=768` (ITERS=228) — the next finer round-KB anchor (3.0 KB), maps the finer trend.
3. Extend **only if a trend emerges**: `CH=684/608` if the finer side trends up toward
   99%; `CH=1824` if the wider side unexpectedly recovers. Do **not** chase all 8 —
   stop as soon as the resonance shape is mapped and no width beats 1216 (cap ≈ 5 screens).

**Why this is disciplined, not a rule violation:** the bounded-sweep rule stops on a
*smooth* curve after one bracket; adamw's is *bimodal* (phase 2 said so explicitly), which
violates the rule's premise. Completing the lattice is the correct phase-3 diligence for a
resonance. Traffic is pinned at 448 MB at every CH (a pure **scheduling** lever), so there
is **zero correctness risk** — rel-L2 stays 3.42e-8 (layout-invariant, pure elementwise,
`(M,N)↔(128,ITERS,CH)` bit-exact round-trip) at every width.

**Honest expected outcome:** the kernel is already at **0.9973× of the roofline**, so the
realistic result is that **1216 is confirmed as the global peak** (or a second lobe ties it
within noise). A *materially* faster width is unlikely — 799.5 GB/s is a hardware ceiling,
not a bubble, and another 99% lobe would land within measurement noise of 0.56 ms, not
below it. The value of D1 is to **close the bimodal question rigorously** (turn "local
optimum" into "global peak, lattice-complete"), not to expect a new win. If a lobe ties
1216, keep 1216 (incumbent wins ties; no promote for a within-noise delta).

## Direction D2 (CONDITIONAL): promote-test only a *material* new lobe

**Trigger:** a screened width lands **materially below** 0.5601 ms (beyond the ~0.3 %
roofline headroom, i.e. an out-of-noise delta) **and** at DMA ≥ 99 %.

**Protocol:** the phase-2 promote-test — interleaved **full 5-seed A/B/A/B/A**
(A = ch1216, B = candidate), with the four gates:
1. `Bbar < Abar − J` (J = 0.0002 ms noise band);
2. every B < max(A);
3. traffic floor intact (HBMrd 359 + HBMwr 90 = 448 MB on every B — no accidental extra pass);
4. 5-seed L2 gate PASS.

Promote only if all four pass. Otherwise `adamw_v2_ch1216` stays the winner.

## Directions NOT taken (documented, with the trigger that stays disarmed)

- **Traffic cut (bf16 read/store, fused mega-load, dropping an input).** No surface:
  tensors are fp32 in HBM (bf16 cannot cut bytes it doesn't own), all four inputs are
  read-once and genuinely used, output written once, no re-fetch/spill. A pure elementwise
  op has no reduction to average bf16 error down under the 2e-5 gate. Traffic is at the
  read-4/write-1 floor and stays there.
- **Edge-tile / partial-tail specialization.** No edges exist — CH | 175104 exactly, every
  tile is a full `[128, CH]` rectangle (the whole point of the reshape-view vs v1's masked
  tail).
- **Partition/free-split regime.** Partition dim is already at the 128-lane hardware max;
  splitting it only underutilizes DMA.
- **Manual double-buffer / ping-pong (`sequential_range`).** Pre-rejected by silu on this
  profiler (`BL-20260709-dma-batching-regresses-pipeline`): denying `affine_range`'s free
  cross-iteration pipelining regresses ~2×. DMA is already 99% saturated — no bubble to
  ping-pong into.
- **Compute-chain rebalance (D3 in phase 2).** Vec 72% sits comfortably under DMA 99% —
  the 4-Vector chain is fully hidden. Rebalancing an already-hidden engine cannot move the
  DMA-bound wall clock.

## Correctness plan

Every D1 screen is byte-identical to the promoted kernel except the `CH`/`ITERS`
constants, both exact divisors of 175104 — so correctness is layout-invariant and unchanged
(rel-L2 3.42e-8 « 2e-5). The **final candidate** (whether 1216 is confirmed or a material
new lobe is promoted) is validated on the **full 5-seed gate** `[0,21,42,63,84]` before any
promotion, per the phase-2 protocol.

## Bottom line (anticipated)

adamw is a memory-bound elementwise op sitting **at the DMA streaming roofline** on an
**immovable 448 MB traffic floor**, expressed as a **shape-homogeneous, edge-free** stream.
There is no traffic lever, no edge/partition/regime surface, and no schedulable bubble left
(DMA 99 %, effBW = roofline). Phase 3's substantive contribution is to **complete the
bimodal-resonance divisor sweep** left bounded in phase 2 — most likely **confirming
`adamw_v2_ch1216` (2.330x) as the global, lattice-complete peak** and closing the operator
as terminal, with `adamw_v1` (2.112x) retained as the fp32 fallback. If (unlikely) a
materially faster 99% lobe surfaces, promote it via the 5-seed interleaved A/B protocol.
