# silu (M4096 N7168, fp32) — Phase-3 shape / regime specialization analysis

This is the **primary** phase-3 deliverable: a rigorous, evidence-backed statement that
the classic shape-specialization levers — edge tiles, ragged-tail masks, partition/free
regime splits, and data-dependent / hot-region branches — have **no target** on this
operator. It is the phase-3 analogue of phase-2's roofline confirmation: documenting that
a whole family of levers does not apply *is* the honest primary result, and it is backed
here by exact arithmetic, not assertion.

Grounding evidence: `profile/silu_v1.txt` (promoted parent, full 5-seed) and
`profile/phase3_shape_analysis.txt` (this analysis's digest).

| kernel | latency | speedup | MFU | PE | Vec | Scl | DMA | HBMrd | HBMwr |
|--------|---------|---------|-----|----|-----|-----|-----|-------|-------|
| silu_v1 | 0.3009 ms | 3.398x | 0% | 1% | 1% | 34% | **97%** | 117 MB | 117 MB |

## 0. What "shape / regime specialization" means, and why it usually has a target

Phase-3's archetype levers exist for **heterogeneous** work: shapes that do not tile
evenly (so some tiles are full and some are ragged "edge" tiles), free axes that leave a
short tail needing a predicated / masked partial store, partition-vs-free regimes where a
different tiling wins for tall-skinny vs short-fat blocks, or data where some region is
"hot" (larger, denser, more expensive) and worth a specialized branch. Each of those
levers pays off precisely when the work is *not* uniform. Below I check each against this
operator's **actual** tiled structure and traffic and show, per lever, that the
precondition for a payoff is absent.

The operator: NKIBench `silu_M4096_N7168_0` — elementwise `SiLU(x) = x / (1 + e^{-x})`
on an `(M, N) = (4096, 7168)` fp32 tensor. `transform_to_nki_inputs` reshapes it
row-major to `(128, 32, 7168)` before the kernel sees it (see
`../../AccelOpt/NKIBench/reference/silu_M4096_N7168_numpy_0.py`), so the kernel's tiled
input is exactly `(128, 32, 7168)` fp32.

## 1. The shape is EXACT-rectangular — no edge / tail / mask regime

The tiled input `(128, 32, 7168)` fp32 divides exactly on **both** relevant axes:

```
partition axis:  128 · 32 = 4096 = M      (exact)   — 32 middle slices, each fills all
                                                       128 partition lanes; no ragged
                                                       partition tile.
free axis:       7168 = 2^10 · 7 = N       (exact)   — the entire N fits with no leftover;
                                                       no ragged free tail.
```

`7168 = 1024 · 7 = 2^10 · 7`. Its divisors are `{1,2,4,7,8,14,16,28,32,56,64,112,128,
224,256,448,512,896,1024,1792,3584,7168}` — so *any* free-axis chunking by a divisor
`s ∈ {2,4,7,...}` produces exact chunk widths `7168/s ∈ {3584, 1792, 1024, ...}` with **no
remainder**. There is no width anywhere in the design space that leaves a partial chunk.

Consequences, per classic lever:

- **Edge tiles — NO TARGET.** An "edge tile" lever specializes the last (partial) tile
  along an axis that does not divide evenly. Here both axes divide evenly, so every tile
  is a full tile. There is no last-partial tile to specialize. v1's own comment already
  records this: *"mask-free (128·32 = 4096 = M and 7168 = N are exact)"*
  (`runs/silu_v1.py:9`). Fabricating an edge-tile code path for a shape with no edge is
  complexity for zero benefit — and is explicitly a review-rejectable overstatement of
  the target (AC-2 negative test).

- **Ragged-tail masks — NO TARGET.** A tail mask predicates the final partial store when
  the free axis leaves `N mod tile ≠ 0` lanes. Here `7168 mod (7168/s) = 0` for every
  divisor `s`, so the store is always a full, unmasked write. Both v1 and every finer
  candidate call `nl.store(..., mask=None)`. Adding a masked-tail path is complexity for
  zero benefit (AC-2 negative test). *(Contrast: phase-2's `k=3` batching produced a
  non-divisor group count `32/3` and had to add an explicit exact-divisor tail group —
  still mask-free, but only because the tail was handled as a second exact group. The
  finer divisors `{2,4,7}` do not even need that: `32·s` is always an integer.)*

## 2. The work is data-UNIFORM — no hot-region / data-dependent regime

`SiLU(x) = x · sigmoid(x)` is a **single elementwise map**: the output at each position
depends only on the input at that same position, and every one of the `4096 · 7168 =
29,360,128` elements costs **identically** on the Scalar Engine — one fused
`nisa.activation(op=nl.silu)` per element, independent of the element's value. There is:

- **no reduction** (no sum/max/norm collapsing an axis — unlike rmsnorm, which reduces
  along K and has a genuine partition/free tradeoff),
- **no reuse** (each input element is touched exactly once; no operand is re-read across
  output tiles — unlike a matmul, whose stationary operand is reused across the moving
  free axis),
- **no data-dependent branch** (the fused SiLU LUT/activation path is the same for all
  inputs; there is no `if x > c` fast/slow split),
- **no "hot" sub-region** (the tensor is `np.random.normal(0, 1)` — statistically
  homogeneous; no block is denser or more expensive than another).

So the "specialize where the work concentrates / branch on the data regime" lever has
**no target**: there is nowhere the work concentrates and nothing to branch on. A
value-specialized or region-specialized code path would compute the identical result at
identical cost with added complexity — rejected (AC-2 negative test, AC-6).

## 3. The partition regime is PINNED at the hardware optimum

v1 uses `par_dim = 128` — the Trainium hardware maximum number of partition lanes — and
places the entire `7168` free axis under the activation. Examine the two directions the
partition/free split could move:

- **Fewer partitions (`par_dim < 128`):** wastes partition lanes. The same `4096·7168`
  elements would need *more* iterations over the *same* bytes (e.g. `par_dim = 64` ⇒ 64
  middle slices instead of 32), strictly worse occupancy with no traffic change. No
  target.
- **More partitions (`par_dim > 128`):** illegal — 128 is the hardware lane count; a
  partition axis > 128 does not exist on the engine.

So the **partition** regime is at its unique optimum and pinned by hardware; it is not a
free variable. This is what leaves the **free-axis tile width** as the *only* remaining
re-tiling degree of freedom — and that single lever is exactly what the D-B finer sweep
(`docs/plan-phase3.md`, `runs/silu_v3_s*.py`) probes. Note that the free-axis tile-width
sweep is a **scheduling** lever (it changes the software-pipeline depth), *not* a
shape-specialization lever: it introduces no edge, no tail, no mask, and no data branch —
every chunk stays a full exact `7168/s`-wide rectangle. So even the one live degree of
freedom does not resurrect any of the shape-specialization levers dismissed above.

## 4. Lever-by-lever summary (the enumerated "no target" table)

| Classic phase-3 lever | Precondition for a payoff | Present here? | Verdict |
|-----------------------|---------------------------|---------------|---------|
| Edge-tile specialization | An axis that does not divide evenly (partial last tile) | No — `128·32=4096=M`, `7168=2¹⁰·7=N`, both exact | **No target** |
| Ragged-tail mask | `N mod tile ≠ 0` leftover lanes needing a predicated store | No — `7168 mod (7168/s)=0` for every divisor `s`; all stores `mask=None` | **No target** |
| Partition/free regime split | A different tiling wins for tall-skinny vs short-fat | No — `par_dim=128` is the pinned hw max; fewer wastes lanes, more is illegal | **No target** (pinned optimum) |
| Data-dependent / hot-region branch | Non-uniform work: reduction, reuse, value-dependent cost, or a hot block | No — single elementwise map, no reduction/reuse/branch, statistically homogeneous input | **No target** |
| *(free-axis tile-width re-tiling)* | *A pipeline-depth win from finer/coarser bursts* | *The one live DOF — but a **scheduling** lever, not shape specialization; introduces no edge/tail/mask/branch* | *Probed by D-B (`s∈{2,4,7}`); measured, not assumed* |

## 5. Consequence for phase 3

The classic shape-specialization family has **no target** on this exact-rectangular,
data-uniform, partition-pinned operator. Every irregular-shape lever's payoff
precondition (a ragged edge, a masked tail, a partition/free tradeoff, a hot region) is
provably absent by the arithmetic above. The only remaining re-tiling degree of freedom
is the free-axis **tile width**, which is a scheduling lever — not shape specialization —
and is measured (not assumed) by the D-B finer sweep. Because v1 is already at the
achieved single-core streaming HBM roofline (phase-2 finding: traffic == read-once/
write-once floor `234.88 MB`, DMA=97% active, ~781 GB/s effective), the best conceivable
D-B win is **≤3%** (0.3009 → ~0.29 ms), and **"keep v1 unchanged"** remains an explicitly
legitimate terminal outcome — exactly as in phase 2.
