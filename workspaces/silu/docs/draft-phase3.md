# silu (M4096 N7168, fp32) — Phase 3 implementation draft (regime / shape specialization)

## Goal

Phase 3 is *shape / regime specialization*: analyze where time goes across the
tensor's structure and specialize **only where the measured win justifies the added
complexity** (tile-size regimes, partition/free splits, edge tiles). Starting from
the promoted kernel `runs/silu_v1.py` (0.3009 ms, **3.398x** over the 1.022441 ms
baseline), the honest phase-3 job is (a) to establish rigorously that the *classic*
shape-specialization levers have **no target** on this operator, and (b) to explore
the one tile-size regime phase 2 never tested — the **finer** free-axis direction —
and promote it only if it clears the same-session noise band on a full 5-seed run.

Never regress correctness (relative-L2 < 2e-5 on all five seeds `[0,21,42,63,84]`).

## Where phase 1–2 left us (the measured starting point)

`runs/silu_v1.py`: one `nl.affine_range(32)` over the middle axis; each iteration
loads a full-width `[128, 7168]` fp32 slice HBM→SBUF, applies one fused
`nisa.activation(op=nl.silu)` on the Scalar engine, stores `[128, 7168]` SBUF→HBM.
Two live SBUF tiles (x_tile, y_tile), no inner free-dim loop, mask-free.

Profiler digest (`profile/silu_v1.txt`, full 5-seed):

| latency | speedup | MFU | PE | Vec | Scl | **DMA** | HBMrd | HBMwr |
|---------|---------|-----|----|-----|-----|---------|-------|-------|
| 0.3009 ms | 3.398x | 0% | 1% | 1% | 34% | **97%** | 117 MB | 117 MB |

Phase 2's terminal finding (`docs/phase2-roofline-confirmation.md`,
`docs/phase2-exit-decision.md`): v1 sits at the **achieved single-core streaming HBM
roofline**. Traffic is exactly the read-once/write-once floor
(`2·4096·7168·4 B = 234.88 MB` = measured 117+117 MB), so there is **no traffic left
to remove** and **no multiplicative headroom**. The only physically-available slack
is a ~3% (~9 µs) DMA-issue/fill-drain bubble. Phase 2 swept the *wider*-burst
direction (D1 k-batching k=2/3/4) — **monotone regression** (DMA 97→85→71%,
coarser pipeline) — and rejected ping-pong (D2, redundant with `affine_range`),
dge_mode (D4, globally `--disable-dge`), and bf16/traffic-reduction (D5, fp32 gate).
All of those are settled; **phase 3 must not re-litigate them** (see NON-GOALS).

## Structural analysis — is there ANY specialization target?

Phase-3's archetype levers are for *heterogeneous* work: irregular shapes with edge
tiles, ragged tails needing masks, or data regimes worth branching on. I checked each
against this operator's actual structure and traffic.

### 1. The shape is EXACT-rectangular — no edge/tail/mask regime

The tiled input is `(128, 32, 7168)` fp32:
- Partition axis `128·32 = 4096 = M` **exactly** (the 32 middle slices each fill all
  128 partitions; no ragged partition tile).
- Free axis `7168 = 2¹⁰·7 = N` **exactly** (no ragged free tail).

So there is **no edge tile, no tail, and no mask anywhere** in v1 (its comment already
notes "mask-free (128·32 = 4096 = M and 7168 = N are exact)"). The phase-3 lever
"specialize the edge tiles / handle the ragged tail differently" has **literally no
target here** — fabricating a masked-tail code path for a shape with no tail would add
complexity for zero benefit and would be rejected on review. This is the phase-3
analogue of phase-2's roofline finding: **the primary deliverable is documenting that
the irregular-shape levers do not apply**, with the exact-divisibility arithmetic as
evidence.

### 2. The work is data-UNIFORM — no hot-region / data-dependent regime

SiLU `y = x/(1+e^-x)` is a single elementwise map: every one of the 4096·7168 elements
costs identically on the Scalar engine, independent of value. There is no reduction,
no reuse, no data-dependent branch, and no "hot" sub-region of the tensor that would
justify a value-specialized or region-specialized code path. So the "specialize where
the work concentrates" lever also has **no target**.

### 3. Partition/free split is already optimal

v1 uses `par_dim = 128` (the hardware maximum partitions) and puts the entire 7168
free axis in one activation call. A smaller partition dim would waste partition lanes
(more iterations over the same bytes); a larger one is illegal (>128). So the
**partition** regime is pinned at the optimum. The only re-tiling degree of freedom
left is the **free-axis tile width**, addressed next.

## The one untested regime: FINER free-axis tiling (the opposite of phase-2's D1)

Phase 2 swept the free/middle burst **wider** (k middle-slices per DMA: k=2/3/4) and
found monotone regression — fewer `affine_range` iterations → coarser
compiler software-pipeline → the prologue/epilogue DMA bubble becomes *relatively
larger*. The mirror-image direction — **finer** tiles, i.e. *more* iterations — was
never tested, and it is the natural phase-3 "tile-size regime" lever.

**Idea.** Split each 7168-wide middle slice into `s` exact sub-chunks of width
`7168/s`, giving `32·s` total pipeline iterations instead of 32. All chunk widths are
exact (7168 = 2¹⁰·7), so this stays **mask-free** and rectangular — no edge handling
introduced. Access pattern is identical to v1 (contiguous along the free axis within
each partition), just split into more, shorter DMA bursts.

**Why it might help.** The measured ~3% DMA-idle bubble ≈ 1/32 — the cost of filling
and draining one iteration of a 32-deep pipeline. More iterations make the
software-pipeline *finer*, so the fixed fill/drain overhead is amortized over more
steady-state steps and the relative bubble shrinks toward the 0.292 ms DMA transfer
wall:

| s | chunk width | iters (32·s) | 2 live tiles (KB/part) | 1/iters (bubble proxy) |
|---|-------------|--------------|------------------------|------------------------|
| 1 (v1) | 7168 | 32 | 56.0 | 3.12% |
| 2 | 3584 | 64 | 28.0 | 1.56% |
| 4 | 1792 | 128 | 14.0 | 0.78% |
| 7 | 1024 | 224 | 8.0 | 0.45% |

If (and only if) the bubble tracks 1/iters, the DMA transfer wall
(`0.97·0.3009 = 0.2919 ms`) plus a shrinking bubble predicts ~0.2965 ms (s=2) →
~0.2942 ms (s=4). **SBUF is a non-constraint** (every finer tile pair ≤56 KB « the
budget — unlike phase-2's k=4 at 224 KB that forced in-place), so finer tiling is
"free" to try.

**Why it might NOT help (the honest counter-force).** Finer chunks mean more, shorter
contiguous DMA runs per partition → more DMA descriptors / issue overhead per byte.
This is the same per-burst-overhead mechanism, just pushed the other way from D1. So
the outcome is **genuinely uncertain and must be measured**: finer pipeline (+) vs
more descriptors (−). Phase 2 established that *wider* loses; it did **not** establish
what *finer* does. That is exactly the open question phase 3 answers.

**Ceiling / expectation.** The 0.2919 ms DMA-active transfer wall is fixed by the
traffic floor, so the best conceivable win is **≤3%** (0.3009 → ~0.29 ms). Realistic
outcome: single-digit-% or zero. Per the roofline, **"keep v1 unchanged" is an
explicitly legitimate terminal outcome** — as it was in phase 2.

## Directions, ranked by benefit/risk

### D-A (rank 1, PRIMARY analysis) — Document that shape-specialization has no target
Not a code change: the rigorous, evidence-backed statement (with exact-divisibility
arithmetic, data-uniformity, and pinned partition dim) that the classic phase-3 levers
(edge tiles, tail masks, partition/free regime splits, data-regime branches) have **no
target** on this exact-rectangular, data-uniform operator. This is the honest primary
deliverable, mirroring phase 2's roofline confirmation. Written to
`docs/phase3-shape-analysis.md` with a `profile/` digest.

### D-B (rank 2, PRIMARY experiment) — Finer free-axis tile-width sweep
Sweep `s ∈ {2, 4, 7}` (all exact divisors → mask-free): split each 7168 slice into
`s` chunks of `7168/s`, iterate `nl.affine_range` over `32·s` steps. Candidates
`runs/silu_v3_s2.py`, `silu_v3_s4.py`, `silu_v3_s7.py`. `--fast` screen each; record
latency, DMA%, Scl%, and HBM (**must stay at 117+117 MB** — this is a scheduling lever,
never a traffic one). Only a variant that **beats v1 in the `--fast` screen** advances
to the full-gate promotion run (mirror of phase-2 AC-3: a screen regression/tie is not
promoted). Expected: monotone one way; stop the sweep when latency turns (don't probe
past the observed minimum).

### D-C (rank 3, CONDITIONAL) — Loop-structure probe for the best-screening s
Only if some `s>1` screens as best: test whether expressing the iteration space as a
single flat `nl.affine_range(32*s)` over a `[128, 32*s, 7168/s]` view vs the nested
`affine_range(32) × affine_range(s)` changes how aggressively the compiler pipelines
(one candidate, e.g. `silu_v3_s{best}_flat.py`). One confirmation run; keep the
better-screening form only if it clears the gate. Skip entirely if s=1 (v1) screens
best.

### NON-GOALS (settled in phase 1–2 — do NOT re-litigate)
- **Wider k-batching** (phase-2 D1 k=2/3/4): monotone regression, rejected.
- **Explicit ping-pong / `sequential_range`** (D2): redundant with `affine_range`
  auto-pipelining, regressed.
- **dge_mode** (D4): `--disable-dge` is globally forced by the harness; no lever.
- **bf16 / tf32 / any traffic reduction** (D5): fp32 in/out contract fixes the 234.88 MB
  floor and any reduced-precision operand fails the 2e-5 rel-L2 gate.
- **Changing the activation formula** (sigmoid+mul, exp-exact): DMA-bound, so it cannot
  help latency, and the fused `nl.silu` LUT is already L2-accurate on all 5 seeds.

## Acceptance criteria

- **AC-1 (correctness).** Never regress rel-L2 < 2e-5. Any promotion candidate must
  pass the FULL 5-seed `[0,21,42,63,84]` run (not just `--fast`). No bf16/tf32/fp16
  introduced anywhere; fp32 in/out throughout. `verify.py`'s `l2_norm_passed` gate is
  authoritative.
- **AC-2 (primary analysis, D-A).** Deliver the "no shape-specialization target"
  write-up with exact-divisibility arithmetic (128·32 = 4096 = M, 7168 = N),
  data-uniformity, and the pinned-partition argument. **Negative test:** do NOT add a
  masked-tail or edge-tile code path for a shape that has no tail/edge — that would be
  complexity for zero benefit and is a review-rejectable overstatement of the target.
- **AC-3 (finer sweep, D-B).** Sweep `s ∈ {2,4,7}`, all mask-free. `--fast` is
  **screen-only**; a variant advances to the full promotion gate **iff** it beats v1's
  `--fast` latency beyond obvious jitter. Record every candidate's latency, DMA%, and
  HBM in `benchmark.csv` / `candidates.jsonl`; HBM must remain at 117+117 MB (a
  variant that inflates HBM traffic is a bug, not a regime). Stop the sweep at the
  observed latency minimum (don't probe finer past a turn).
- **AC-4 (promotion gate).** Promote a finer variant only if it clears the same-session
  noise band on the full 5-seed run (interleaved A0,B0,A1,B1,A2 vs v1, per the
  fast-vs-full-run lesson). A within-noise tie **keeps v1** (the simpler kernel wins
  ties — AC-5).
- **AC-5 (complexity justification).** The finer variant adds an inner loop / larger
  iteration count. It must **earn** its place with a measured, noise-band-clearing win;
  absent that, keep the strictly simpler v1. "Specialize only where the measured win
  justifies the added complexity" is the phase-3 mandate.
- **AC-6 (loop-structure probe, D-C).** Conditional on some s>1 screening best; at most
  one confirmation run; otherwise skip.
- **AC-7 (evidence).** New rows in `benchmark.csv`; new nodes in `candidates.jsonl`
  parented off `silu_v1` as a DAG (including first-class rejection nodes for any swept
  s that regresses); a `profile/` digest per major direction (finer-sweep table +
  shape-analysis). Candidate `.py` sources under `runs/`.

## Expected outcome (stated up front, honestly)

Given v1 is at the achieved streaming roofline with only a ~3% bubble available, the
most likely phase-3 result is **either** a small (<3%) finer-tiling win that clears the
noise band and is promoted, **or** "keep v1 unchanged" plus the rigorous
shape-specialization-has-no-target analysis. Both are legitimate terminal outcomes;
the discipline is to let the measured full-5-seed number decide and never promote
added complexity that does not beat v1 outside the noise band.
