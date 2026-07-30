# mamba (M7168 C256 S16, fp32) — Phase 2 implementation draft (profile-driven)

## Starting point and the phase-1 puzzle

Phase 1 (`runs/mamba_v1.py`, commit 719684c) produced the first correct kernel but
it is **0.832× — slower than the baseline** (1.5116 ms vs 1.258 ms). That regression
is the whole content of phase 2: the phase-1 "load delta/u once instead of 16×"
hygiene was supposed to help and instead hurt. Phase 1's own digest already named the
two suspects — `HBMrd=72MB` (4.5× the ~16 MB read-once floor → **compiler spilling**)
and `PE=52% / Vec=51%` co-limit (**not** the pure-Vector-bound story the phase-1 draft
hypothesized). Phase 2 confirms both with direct measurement and fixes them.

Correctness gate: relative-L2 `< 3e-5 · ||v_r||` on seeds `[0,21,42,63,84]`, fp32.
The math is settled and unchanged from phase 1 (numpy oracle rel-L2 4.08e-7); every
phase-2 candidate keeps the identical op sequence, only the **loop/tile structure**
changes. Never regress correctness.

## Measured evidence (the profile-driven core of this phase)

I profiled the AccelOpt `samples/nki/mamba_*.py` reference kernels through our own
`verify.py` path (seed 42, `--fast`) to get real per-engine breakdowns for the ~1.6×
target the prompt cites. Evidence saved under `profile/refs/` (gitignored copies +
`README.md`):

| kernel | structure | latency | speedup | PE | Vec | Scl | DMA | HBMrd |
|--------|-----------|---------|---------|----|----|----|----|-------|
| baseline | state-outer, delta/u reloaded 16× | 1.2583 ms | 1.000× | — | — | — | — | — |
| **our v1** (= AccelOpt `mamba_v2`) | channels-outer, load-once, **whole-M** scan | 1.5116 ms | 0.832× | 52% | 51% | 16% | 11% | **72 MB** |
| AccelOpt `mamba_v3` | seq-tile **inner** (`static_range` 512), `[128,1]` carry | 0.8367 ms | 1.504× | 93% | 82% | 22% | 13% | 16 MB |
| AccelOpt `mamba_optimized` | seq-tile **outer** (`sequential_range` 512), carried `scan_state` | **0.7823 ms** | **1.608×** | **99%** | 89% | 23% | 11% | 16 MB |

Three facts fall straight out of this table:

1. **Our v1 reproduces `mamba_v2` to the digit** (1.5116 ms, PE52/Vec51/72MB). The
   regression is real and understood, not measurement noise.

2. **One lever explains the entire 1.51 → 0.78 ms gap: sequence tiling** (chunk the
   M=7168 free axis into 512-wide tiles, carry the scan's last column forward as the
   next tile's `initial`). It changes *nothing* about the math and yet:
   - **kills the spill**: `HBMrd` 72 MB → **16 MB** (the read-once floor).
   - **unlocks pipelining**: PE/Vec 52/51% → **99/89%**.

3. **At 1.6× the bottleneck engine is the Tensor engine (PE=99%) — and mamba has no
   matmul.** This is the ceiling-raising lever for beating the reference (see §"Beyond
   1.6×").

### Why whole-M spills — SBUF arithmetic (trn2, 208 KB/partition usable)

A `[128, 7168]` fp32 tile is **28.0 KB/partition**. v1 holds live across the 16-state
loop: `delta_i`, `u_i`, `scanC_accum` (3 × 28 = 84 KB) **plus** the per-state
temporaries `deltaA`, `deltaBu`, `scan_res`, `scanC` (~4 × 28 = 112 KB) → **~196 KB
peak, right at the 208 KB usable limit.** The allocator has no slack for
double-buffering, so it spills intermediates to HBM → the 72 MB read traffic. Sequence
tiling at 512 shrinks every tile to **2.0 KB/partition**; the whole hoisted working set
is ~20–45 KB, leaving ample room for the compiler to double-buffer (which is what turns
PE/Vec from 51% into 89–99%).

### Why PE is busy at all — partition-broadcast is a hidden matmul

Confirmed via the NKI arch guide + the NKI kernel library source: `broadcast_to((128, M))`
of a `[1, M]` SBUF row **across the partition dimension** is lowered to
`nc_matmul(ones[128,1], row[1,M])` on the **Tensor engine** (result lands in PSUM).
mamba broadcasts `b[s,:]` and `c[s,:]` this way once per state. Those implicit matmuls
are the *only* PE work in the kernel, and they sit in a serial dependency with the
Vector multiplies that consume them — exactly the "PE≈Vec≈50%, neither saturated"
signature v1 shows, and the "PE=99%" ceiling `mamba_optimized` hits once pipelining
removes the stalls. This is what §"Beyond 1.6×" attacks.

## Ranked optimization directions

Ranked by expected benefit × confidence. Each is a ≤5-iteration exploration with
before/after `verify.py` latency + the profiler digest as the keep/revise/reject gate.

### D1 — Sequence tiling with carried scan state  ⭐ (primary; ~1.5–1.6× expected)

**What.** Chunk M=7168 into `seq_tile` (start 512; 7168/512 = 14, exact) tiles. Keep
channels-outer / state-inner and load `delta`/`u` once per channel tile (the phase-1
hygiene is *correct*, it was just starved of SBUF). Per state, scan each seq-tile with
`initial = scan_init` where `scan_init` is the previous tile's last column, a
per-partition `[128,1]` carried state. `nki.isa.tensor_tensor_scan` accepts a `[P,1]`
tile as `initial` (documented; = the `result[:,i-1]` fed into column 0).

**Two sub-variants, both measured on the references above — try both:**
- **D1a (= `mamba_v3` shape): seq-tile INNER, `static_range`.** State loop outside,
  seq-tile loop inside with a per-state `scan_init` carried by `static_range`
  (the AccelOpt comment warns `sequential_range` here gave *wrong* answers and worse
  perf; `static_range` fully unrolls the 14 tiles). Measured 1.504×.
- **D1b (= `mamba_optimized` shape): seq-tile OUTER, `sequential_range`.** Seq-tile
  loop outermost over a `[n_channel_tile, 128, S]` `scan_state` array; channels then
  states inside; store each seq-tile's output slice directly (no whole-M accumulator).
  Measured **1.608×** — the best reference. The outer `sequential_range` cleanly
  expresses the cross-tile carry and keeps the live accumulator at `seq_tile` width.

**Why it wins.** Kills the spill (28 KB → 2 KB tiles) and enables double-buffering.
**Risk.** Low — it's the exact structure of two measured, correctness-passing references;
same op sequence, same scan primitive. **Correctness watch:** the carried `initial`
introduces a loop-carried dependency, so the carry loop must be `sequential_range`
(D1b) or `static_range` (D1a), never `affine_range` — getting this wrong silently
corrupts the scan. Validate on seed 42 first, then full 5-seed before promoting.

### D2 — Tune the seq-tile width (cheap sweep on top of D1's winner)

**What.** Once D1's structure is chosen, sweep `seq_tile ∈ {256, 512, 1024}` (all
divide 7168: 28/14/7 tiles). 512 is AccelOpt's "magic number"; our silu task found
**finer** free-axis tiling beat wider (optimum ~4 KB/partition burst) — so 256 (1 KB)
is worth a shot, and 1024 (4 KB) brackets the other side. Pure knob turn, no structural
change.

**Why.** The optimum balances pipeline depth (more tiles = more overlap) against
per-tile fixed overhead + carry-dependency serialization. **Benefit** likely a few %.
**Risk** minimal. Rank second because it only refines D1.

### D3 — Move the b/c partition-broadcast OFF the Tensor engine  ⭐ (ceiling-raiser; beat 1.6×)

**What.** At 1.6× the kernel is **PE=99%-bound on the implicit broadcast matmuls**,
while **DMA sits at 11%**. Relocate the `b`/`c` partition-broadcast off the saturated
PE. Three costed options (from the API research), to try in order of confidence:

- **D3a — Hoist the broadcast out of the channel loop.** `b[s,:]` and `c[s,:]` are
  **channel-independent** — but v1/references re-broadcast them inside the 2-iteration
  channel loop. Broadcasting once per (seq-tile, state) and reusing across both channel
  tiles **halves** the broadcast matmuls (896 → 448, ≈ −50% PE work). Pure hoist,
  obviously correct, no new API. **Highest-confidence ceiling raiser.**
- **D3b — Partition-stride-0 DMA broadcast.** Load `b[s,:]`/`c[s,:]` directly into a
  `[128, seq_tile]` SBUF tile via a stride-0 partition access pattern
  (`src.ap(pattern=[[0,128],[1,seq_tile]])` → `dma_copy`), spending idle DMA bandwidth
  instead of PE. Real technique (the NKI kernel library `dma_broadcast`); caveat is extra
  descriptor traffic (128 partitions), but DMA has huge headroom here.
- **D3c — `nc_stream_shuffle` broadcast on VectorE.** Keeps it off PE at the cost of
  Vector cycles; only attractive if Vec has more slack than DMA after D1 (Vec is 89%,
  so likely *not* — rank D3c last).

**Why.** This is the only lever that can push past the 1.608× reference, because that
reference is PE-bound. **Risk** moderate: D3a is safe; D3b/D3c change how a tile is
materialized and need a correctness re-check + a digest read to confirm PE actually
drops without a new bottleneck appearing. Gated on D1 landing first.

### D4 — Hoist state-independent `deltaU = delta·u` out of the state loop (minor)

**What.** `deltaU = delta_i · u_i` does **not** depend on the state `s`, yet it is
recomputed inside the 16-state loop (1 of ~5 Vector passes/state). Compute it once per
(channel tile / seq-tile) and reuse. Removes ~20% of the Vector `tensor_tensor` work.

**Why / risk.** Correct and cheap. But it only helps if the kernel is Vector-bound
*after* D1+D3 — at PE=99% the Vector engine is the *second* bottleneck (89%), so this is
a follow-on, not a lead. It also adds one live `[128, seq_tile]` buffer (trivial at
2 KB). Rank low; revisit only if a post-D3 digest shows Vec back on top. **Note:** this
is the phase-1 draft's "biggest vector lever" — the measurement demotes it, which is
the point of a profile-driven phase.

## Plan of attack (≤5 iterations)

1. **Iter 1 — D1b** (`mamba_optimized` shape: seq-tile outer, `sequential_range`,
   carried `scan_state`, seq_tile=512). Expected ≈1.6×. Score seed-42 `--fast`; read
   the digest; if it reproduces PE≈99/Vec≈89/16MB, **promote** as the new phase-2 best.
   Fallback D1a (`static_range` inner) if the outer `sequential_range` misbehaves.
2. **Iter 2 — D3a** (hoist b/c broadcast out of the channel loop) on top of D1b. Target
   PE < 99%, latency < 0.78 ms. Keep only if the digest confirms PE drops.
3. **Iter 3 — D2** seq-tile sweep {256, 1024} against the D1b(+D3a) winner. Keep best.
4. **Iter 4 — D3b** (stride-0 DMA broadcast) *if* D3a left PE still on top; compare
   against the DMA-active digit.
5. **Iter 5 — D4** *only if* a digest shows Vector back as the bottleneck; else spend
   the iteration hardening the current best (full 5-seed run + shape-specialization
   preview for phase 3).

Record every candidate in `benchmark.csv` + `candidates.jsonl` (DAG parent links),
profiling evidence under `profile/`. Full 5-seed `verify.py` (drop `--fast`) before any
promotion. Phase-3 preview: S=16, C=256 (exactly 2 channel tiles), M=7168 are all fixed
— unroll the state loop and hard-tune the seq-tile width once the structure is locked.

## Risks / watch-items

- **Carry-dependency loop kind.** The carried `initial` makes the seq-tile loop
  loop-carried → must be `sequential_range`/`static_range`, never `affine_range`.
  Mis-choosing silently corrupts the scan (AccelOpt hit exactly this). First check on
  every D1 variant: seed-42 rel-L2 passes.
- **`scan_state` extraction.** Carrying "the last column" (`scan_res[:, seq_tile-1]`)
  into a `[128,1]` slice must land at partition-aligned `[P,1]`; verify against the
  reference indexing (`scan_init[...] = scan_res[0:128, seq_tile-1]`).
- **D3b descriptor cost.** Stride-0 DMA into `[128, seq_tile]` still emits ≥128
  descriptors; confirm the DMA digit stays well below saturation and that PE actually
  falls — otherwise revert to D3a.
- **Don't over-fit `--fast`.** All reference numbers above are seed-42/fast; confirm the
  ranking holds on the full 5-seed / higher-iter run before promoting, as other tasks
  showed fast-mode can mis-rank close candidates.
