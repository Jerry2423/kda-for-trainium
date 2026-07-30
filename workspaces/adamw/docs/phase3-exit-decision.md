# adamw phase-3 exit decision — PROMOTE adamw_v2_ch1824 (2.384x)

## Outcome

**Promoted `runs/adamw_v2_ch1824.py` at 2.384x (0.5474 ms)**, +2.6% over the phase-2 winner
`adamw_v2_ch1216` (2.330x / 0.5601 ms). `adamw_v2_ch1216` is kept as the documented
prior-best; `adamw_v1` (2.112x, masked row tiles) remains the fp32 fallback.

This is the plan's **low-probability outcome (b): promotion**, not the expected terminal
closure (a). Completing the bimodal burst-width sweep the phase-2 bounded rule left open
surfaced a **third 99% DMA-saturation lobe at the wide in-band edge** (`CH=1824`, ITERS=96,
7.125 KB/partition) that phase 2 never tested. It beats a fresh same-session `ch1216` anchor
by **9.5× the measured `--fast` jitter** and passes the full interleaved 5-seed A/B/A/B/A
promote-test on all four gates. The win is a **pure scheduling** win — the 448 MB traffic
floor is byte-identical at every `CH` — so correctness is layout-invariant and unchanged.

## Phase-3 structural surface: burst-width-only (AC-1)

adamw's reshape-view is **shape-homogeneous and edge-free**, so the classic phase-3 levers
have no surface. The decisive divisor-exactness proof:

- `M·N = 10944·2048 = 22413312 = 128 · 175104` **exactly**, and `175104 = 2¹⁰·3²·19`.
- Therefore every in-band `CH | 175104` yields `ITERS = 175104 / CH` exact, and each chunk
  is a **full `[128, CH]` rectangle** — zero edge tiles, zero masks, all 128 partition lanes
  live (contrast `adamw_v1`, which carried a `row < 10944` predicate on its 64-valid-row
  last tile).

Classic levers disarmed (each has **no surface**):

| Lever | Why disarmed |
|---|---|
| Edge-tile / partial-tail specialization | No edges exist — `CH \| 175104` exactly, every tile is a full `[128, CH]` rectangle; there is no partial tail to specialize. |
| Partition / free-split regime | Partition dim already at the 128-lane hardware max; splitting below 128 only underutilizes DMA. |
| Mixed tile-size regimes | Every tile is byte-identical in structure; nothing heterogeneous to regime-specialize. |

"Regime specialization" therefore collapses to the **burst-width `CH` sweep** — the one axis
that exists. This matches the bmm / transpose_matmul phase-3 terminal statement ("shape
edge-free → the remaining question is numerical/scheduling, not tile geometry").

## The bimodal-resonance question phase 2 left open

Phase 2 tested only **4 of the 12** in-band divisors (`1024, 1152, 1216, 1536`), found
`CH=1216` an interior optimum bracketed below on both sides, and **stopped after one adjacent
bracket** under `BL-20260709-finer-tiling-harvests-dma-bubble`. But that rule's stopping
condition assumes a **smooth unimodal** curve; adamw's curve is explicitly **bimodal** (two
99% lobes at 1024/1216 straddle an 88% trough at 1152). On a resonant curve, one adjacent
bracket does **not** rule out a third lobe elsewhere in the band. Phase 3 screened the
**8 untested in-band divisors** (DEC-1 = exhaustive) to settle it.

## The sweep (all screens byte-identical to ch1216 except CH/ITERS+header)

`--fast` screens (seed 42). Same-session `ch1216 --fast` anchors bracketed around the
screens (anchor → screens → anchor): **A0 0.5625, A1 0.5615, A2 0.5629, A3 0.5614 ms**
→ anchor mean **0.5621 ms**, **`J_measured` = 0.0015 ms** (anchor-to-anchor spread).

| CH | ITERS | burst/part | latency | speedup | DMA% | Vec% | Scl% | effBW GB/s | role vs anchor |
|----|-------|-----------|---------|---------|------|------|------|-----------|----------------|
| 512 | 342 | 2.000 KB | 0.7768 | 1.680 | 93 | 57 | 30 | 578 | trough (+38.2%) |
| 576 | 304 | 2.250 KB | 0.5839 | 2.235 | 99 | 76 | 40 | 769 | lower lobe (+3.9%) |
| 608 | 288 | 2.375 KB | 0.5860 | 2.227 | 99 | 74 | 40 | 766 | lower lobe (+4.3%) |
| 684 | 256 | 2.672 KB | 0.6390 | 2.042 | 99 | 67 | 34 | 702 | DMA-saturated dip (+13.7%) |
| 768 | 228 | 3.000 KB | 0.7517 | 1.736 | 89 | 56 | 28 | 597 | trough (+33.7%) |
| **912** | 192 | 3.562 KB | 0.5606 | 2.328 | 99 | 74 | 36 | 800 | **TIE (−0.3% = 1.0×J)** |
| 1024 | 171 | 4.000 KB | 0.5921* | 2.204 | 99 | 69 | — | 757 | phase-2 lobe |
| 1152 | 152 | 4.500 KB | 0.6834* | 1.910 | 88 | 59 | — | 656 | phase-2 trough |
| **1216** | 144 | 4.750 KB | 0.5621 (anchor) | 2.320 | 99 | 72 | 34 | 799 | **incumbent** |
| 1368 | 128 | 5.344 KB | 0.5825 | 2.240 | 99 | 69 | 32 | 770 | lower lobe (+3.6%) |
| 1536 | 114 | 6.000 KB | 0.6515* | 2.003 | 91 | 60 | — | 688 | phase-2 trough |
| **1824** | 96 | 7.125 KB | **0.5478** | **2.383** | 99 | 72 | 32 | **819** | **MATERIAL (−2.5% = 9.5×J)** |

`*` = phase-2 `--fast` figures (this session re-anchored `ch1216` only; the phase-2 widths
were not re-screened — their role in the curve is unchanged).

Out-of-band wide bracket (DEC-2 deviation, see below):

| CH | ITERS | burst/part | latency | speedup | DMA% | effBW GB/s | role |
|----|-------|-----------|---------|---------|------|-----------|------|
| 2304 | 76 | 9.000 KB | 0.6105 | 2.138 | 94 | 735 | wide edge turns over (+8.6%) |

**Every screen holds the 448 MB traffic floor** (HBMrd 359 MB + HBMwr 90 MB, 4-read/1-write)
within profiler rounding, including the OOB bracket (AC-4). rel-L2 is layout-invariant
(3.42e-8 « 2e-5) at every width (AC-8) — pure elementwise op, exact
`(M,N) ↔ (128,ITERS,CH)` reshape round-trip.

The curve is **tri-lobal**, not bi-lobal: three 99% DMA-saturation lobes (`576/608`,
`912`+`1024`+`1216`, and `1824`) separated by stall troughs (`512`, `768`, `1152`, `1536`).
The `1824` lobe at the wide edge is the highest — a resonance phase 2 could not have found by
bracketing around 1216.

## Materiality classification (AC-5)

`J_measured` = 0.0015 ms (the A0–A3 `--fast` anchor spread). Against the anchor mean 0.5621:

- **`CH=1824` — MATERIAL.** 0.5478 (reconfirm 0.5476) beats the anchor mean by 0.0143 ms
  = **9.5× `J_measured`**, floor intact, DMA=99%. Triggers the D2 promote-test.
- **`CH=912` — TIE.** 0.5606 is only 0.0015 ms (1.0× `J_measured`) under the anchor — a real
  99% co-lobe but within jitter; incumbent wins ties (no promotion).
- **All others — REGRESSION.** 1368/608/576 are lower lobes (+3.6–4.3%); 684 is a
  DMA-saturated-but-slower dip (+13.7%, a useful counterexample — see below); 512/768 are
  deep stall troughs (+33.7–38.2%).

DMA% is used only as a supporting/diagnostic signal, per AC-5. The `CH=684` point is the
proof this is correct: it shows **DMA=99% yet latency +13.7%** — 99% DMA-active does **not**
imply peak effective bandwidth (the descriptor cadence can keep the engine "busy" while still
leaving throughput on the table). Latency-vs-anchor plus the traffic floor is the decision
signal; DMA% alone would have mis-ranked 684 as a co-lobe.

## Promote-test (interleaved full 5-seed A/B/A/B/A; A=ch1216, B=ch1824) — AC-6

| run | kernel | latency | speedup | DMA | HBMrd | HBMwr | 5-seed L2 |
|-----|--------|---------|---------|-----|-------|-------|-----------|
| A0 | ch1216 | 0.5619 | 2.323x | 99% | 359 | 90 | PASS |
| B0 | ch1824 | 0.5474 | 2.384x | 99% | 359 | 90 | PASS |
| A1 | ch1216 | 0.5622 | 2.321x | 99% | 359 | 90 | PASS |
| B1 | ch1824 | 0.5474 | 2.384x | 99% | 359 | 90 | PASS |
| A2 | ch1216 | 0.5617 | 2.323x | 99% | 359 | 90 | PASS |

- `Abar = 0.56193 ms` (max 0.5622, A-spread 0.0005 ms); `Bbar = 0.54740 ms`.
- `J_promote = max(0.0002 ms, A-spread 0.0005) = 0.0005 ms`.
- **GATE1** `Bbar < Abar − J_promote`: 0.54740 < 0.56143 → PASS, delta **0.01453 ms =
  29.1× J_promote** (far outside noise).
- **GATE2** every B < max(A): 0.5474, 0.5474 both < 0.5622 → PASS.
- **GATE3** traffic floor intact: HBMrd 359 + HBMwr 90 = 448 MB on **every** B → PASS
  (no accidental extra pass, no spill).
- **GATE4** 5-seed L2 gate `[0,21,42,63,84]`: PASS on all five runs.

All four gates pass → **PROMOTE `adamw_v2_ch1824`**.

## Plan deviation: one out-of-band bracket (DEC-2), and why it was required

DEC-2 (author-resolved) said "keep the reasoned in-band scope `[512, 1824]`; do NOT probe
out-of-band anchors; label the result in-band lattice-complete." But the material winner
**landed at `CH=1824` — the widest in-band divisor, i.e. the boundary itself.** DEC-2's
exclusion rationale was "the measured wide side already declines at `CH=1536`" — that premise
is **falsified**: 1536 (0.6515) is a *trough*, and 1824 *recovers* to a higher lobe than the
incumbent. A maximum sitting on the sweep boundary is a **censored maximum** — you cannot call
it a bracketed peak without one probe past the edge. I therefore ran **one** out-of-band
bracket, `CH=2304` (9.0 KB, the immediate next wide divisor): 0.6105 ms (+8.6%, DMA falls to
94%). This **confirms the wide edge turns over just past 1824**, so 1824 is a true peak
bracketed below on **both** sides (1536 finer, 2304 wider), not a boundary artifact. The
probe is still a byte-identical `CH`-only screen at the 448 MB floor — zero correctness risk.

Consequence: the closure claim is scoped to what the evidence measures — `adamw_v2_ch1824`
is the promoted **in-band lattice winner, de-censored/bracketed on the wide side by the
single `CH=2304` OOB probe**. The whole in-band lattice plus that one out-of-band bracket
was screened, so the promoted lobe is bracketed below on both sides (1536 finer, 2304 wider)
by measured regressions. This single bracket is sufficient to de-censor the boundary win and
justify promotion; it is **not** a proof of a global optimum over all exact wider divisors —
further out-of-band widths (2432, 2736, 3072) remain out of scope and are not needed for the
phase-3 promotion decision (see "Directions NOT taken").

## The roofline was the wrong anchor (key finding)

Phase 2 asserted a hard ceiling of **799.5 GB/s** (the streaming roofline silu sustains on
this trn2 profiler) and read `ch1216` at 0.9973× of it — "at the roofline, ~0.3% headroom
left, a hardware ceiling not a schedulable bubble." **That anchor was wrong for adamw.** The
promoted `ch1824` runs at **effBW ≈ 819–820 GB/s**, ~2.6% **above** the silu roofline. The
cause: silu is a **balanced 2-stream** kernel (1 read + 1 write); adamw is a **read-heavy
5-stream** kernel (4 reads + 1 write, a 4:1 read:write ratio). HBM sustains higher aggregate
bandwidth for a read-dominated access pattern than for a balanced one, so borrowing silu's
achieved figure as adamw's ceiling under-counted the real roofline by a few percent. `ch1216`
was a strong lobe but **never at adamw's true ceiling** — there was a real, if small,
scheduling win still on the table, and the wide-edge resonance found it. This refines the
phase-2 lesson: a roofline anchored on a *different op's stream ratio* is an estimate, not a
proof, and a "material win beyond the asserted roofline headroom" (which the plan called
self-contradictory) is exactly what a wrong anchor produces — vindicating AC-5's decision to
define materiality by measured jitter, not by the roofline headroom.

## Directions NOT taken (correctly)

- **Traffic cut (bf16 read/store, dropping an input).** No surface: fp32-owned HBM tensors
  (bf16 cannot cut bytes it doesn't own), all four inputs read-once and genuinely used, output
  written once, no re-fetch/spill. A pure elementwise op has no reduction to average bf16
  error down under 2e-5. Traffic pinned at the 448 MB floor at every width.
- **Manual double-buffer / ping-pong (`sequential_range`).** Pre-rejected by silu on this
  profiler (`BL-20260709-dma-batching-regresses-pipeline`): denying `affine_range`'s free
  cross-iteration pipelining regresses ~2×. DMA is 99% saturated at the promoted width — no
  bubble to ping-pong into.
- **Folded compute-chain algebra-scheduling reorder (reserved, disarmed).** Vec=72% sits
  hidden under DMA=99%; rebalancing an already-hidden engine cannot move the DMA-bound wall
  clock. Its trigger ("all CH probes fail AND a profiler hint changes") never armed.
- **Further out-of-band widths (2432, 2736, 3072).** One bracket (2304) confirms the
  wide-edge turnover; more OOB probes are unnecessary and out of the (deviated) scope.

## Bottom line

adamw phase 3 completes the bimodal-resonance divisor sweep phase 2 left bounded, and — in
the plan's low-probability branch — **promotes a third, higher 99% DMA-saturation lobe at the
wide in-band edge**: `adamw_v2_ch1824` at **2.384x (0.5474 ms)**, +2.6% over `ch1216`. It is
the promoted **in-band lattice winner**, de-censored on the wide side by the single `CH=2304`
OOB bracket (not a proven global peak over unscreened wider divisors). The win is pure
scheduling (448 MB floor unchanged, rel-L2 layout-invariant), bracketed below on both sides
(1536 finer, 2304 wider), and passes the full four-gate 5-seed promote-test at 29× the
promote noise band. The finding that matters beyond the number: adamw's read-heavy
4:1 5-stream DMA sustains ~820 GB/s, **above** the silu-anchored 799.5 GB/s "roofline" —
so `ch1216` was never at adamw's true ceiling, and a roofline borrowed from a different
stream ratio is an estimate, not a proof. `adamw_v2_ch1216` is retained as the prior-best;
`adamw_v1` (2.112x) remains the documented fp32 fallback.
