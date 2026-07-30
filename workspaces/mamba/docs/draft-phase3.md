# mamba (M7168 C256 S16, fp32) — Phase 3 implementation draft (regime / shape specialization)

## Starting point — a blunt correction that reframes the whole phase

Phase 3 is nominally "shape specialization," but the honest state of the workspace
forces a two-part phase, and the draft must say so up front:

- **The only kernel that exists in `runs/` is `runs/mamba_v1.py` at 0.832× — SLOWER
  than the 1.258 ms baseline.** `candidates.jsonl` has exactly one node; `benchmark.csv`
  one perf row.
- **Phase 2 never landed a kernel.** Its draft (`docs/draft-phase2.md`) and plan
  (`docs/plan-phase2.md`) are excellent and committed, but the phase-2 RLCR loop was
  **blocked by a stale phase-1 lock** (see `logs/phase2.3-loop.log`: it held
  `plan-phase1.md`, `current_round=0`, never reviewed, never retired). So the
  profile-driven **sequence-tiling win that phase 2 designed (measured 1.5–1.6× on the
  reference kernels through our own harness) was never written to code.**

Therefore phase 3 cannot "specialize" a fast kernel — there isn't one yet. Phase 3
must **(A) first land the proven phase-2 lever (sequence tiling) as the new baseline**,
then **(B) specialize the now-fixed shapes on top of it.** Part A is not busywork or
re-litigation: it is the evidence-verified 1.6× structure that simply has no `runs/`
artifact. Part B is where the genuine phase-3 content lives.

Correctness gate is unchanged: relative-L2 `< 3e-5 · ||v_r||₂` on seeds
`[0,21,42,63,84]`, fp32. The numpy oracle for this op sequence is rel-L2 4.08e-7
(~75× inside the gate). Every phase-3 candidate keeps fp32 and the identical op
sequence (`activation(exp,scale=A)`, `tensor_tensor(delta,u)`, `tensor_tensor(·,b)`,
`tensor_tensor_scan`, `tensor_tensor(·,c)`, accumulate); only loop/tile structure,
broadcast *mechanism*, and op *fusion* change. Never regress correctness.

## The measured evidence phase 2 gathered (still the ground truth for phase 3)

Profiled through our `verify.py` path (seed 42, `--fast`), saved under `profile/refs/`:

| kernel | structure | latency | speedup | PE | Vec | Scl | DMA | HBMrd |
|--------|-----------|---------|---------|----|----|----|----|-------|
| baseline | state-outer, delta/u reloaded 16× | 1.2583 ms | 1.000× | — | — | — | — | — |
| **our v1** (= AccelOpt `mamba_v2`) | channels-outer, load-once, **whole-M** | 1.5116 ms | 0.832× | 52% | 51% | 16% | 11% | **72 MB** |
| `ref_v3` | seq-tile **inner** (`static_range` 512), `[128,1]` carry | 0.8367 ms | 1.504× | 93% | 82% | 22% | 13% | 16 MB |
| `ref_optimized` | seq-tile **outer** (`sequential_range` 512), `[n_ct,128,S]` `scan_state` | **0.7823 ms** | **1.608×** | **99%** | 89% | 23% | 11% | 16 MB |

Two facts drive phase 3:

1. **Sequence tiling is the floor** (0.832× → 1.6×): chunk M=7168 into 512-wide tiles,
   carry the scan's last column forward as the next tile's `initial`. It changes
   nothing about the math; it kills the SBUF spill (72 MB → 16 MB read-once floor) and
   lets the compiler pipeline PE against Vec (51/51% → 99/89%). Land this first.

2. **At 1.6× the bottleneck engine is the Tensor engine (PE=99%) — and mamba has no
   matmul.** The `b[s,:]`/`c[s,:]` partition-dim `broadcast_to((128, seq_tile))` is
   lowered to `nc_matmul(ones[1,128], row[1,seq_tile])` on the PE (confirmed:
   `trainium_inferentia2_arch.md` — "NKI invokes such matmul under the hood when
   `broadcast_input.broadcast_to((M, …))` is called"; and `fused_mamba.md` —
   "partition-dim broadcast often requires a separate instruction on TensorE"). Those
   broadcast matmuls are the *only* PE work, and they saturate it while **DMA sits idle
   at 11%.** This is the ceiling to attack in the specialization half.

## Where time goes across this tensor's structure (the phase-3 analysis)

The prompt asks to "analyze where time goes across the tensor's structure." The
structure is small and fully static, which is exactly why specialization pays:

- **Partition axis (channels):** C=256 = **exactly 2 tiles of 128** — no remainder, no
  edge tile. The 2-iteration channel loop is a compile-time constant.
- **Free axis (sequence):** M=7168 = 2¹⁰·7. Divides exactly by {128, 256, 448, 512,
  896, 1024, 1792} — every candidate seq-tile is clean; again **no edge tile.**
- **State axis:** S=16, fixed and tiny — the inner state loop is a compile-time constant.
- **Engine time at the 1.6× point:** PE 99% (b/c broadcast matmuls — pure overhead,
  not real compute), Vec 89% (the 4–5 `tensor_tensor` per state + the scan), Scalar 23%
  (the `exp`), **DMA 11% (idle).**

The structural hotspot is unambiguous: **the b/c partition-broadcast matmuls saturate
the PE.** There are *no* edge tiles to specialize (everything divides evenly), so the
phase-3 specialization is **not** "handle the ragged tile" — it is: pick the seq-tile
regime that minimizes PE pressure for the fixed M, relocate the broadcast off the PE,
and exploit the compile-time-constant S=16 / 2-channel loops.

## Ranked specialization directions

Ranked by expected benefit × confidence. Each is a ≤5-iteration probe gated on
`verify.py` latency + the profiler digest, correctness (5-seed) always mandatory.

### S0 — Land sequence tiling with carried scan state  ⭐ (FLOOR; must land; ~1.5–1.6×)

Carried over from the phase-2 plan (its D1), never coded. This is the phase-3 baseline;
everything else stacks on it.

- **S0-lead = D1b (mirror `profile/refs/ref_optimized.py`, the 1.608× ref):** seq-tile
  **outer** over `nl.sequential_range(n_seq_tile)`; a `[n_channel_tile, 128, S]`
  `scan_state` carried across tiles; channels then states inside; store each seq-tile's
  output slice (no whole-M accumulator). `initial = scan_state[i_ct,:,i_state]` for
  tiles > 0, else 0; after each scan, write `scan_res[:, seq_tile-1:seq_tile]` back into
  the `[i_ct, :, i_state]` slot.
- **S0-fallback = D1a (mirror `profile/refs/ref_v3.py`, 1.504×):** seq-tile **inner**
  over `nl.static_range(n_seq_tile)` (fully unrolled), per-state `scan_init` `[128,1]`
  carried as `scan_init[...] = scan_res[:, seq_tile-1]`.

**Correctness watch (from the AccelOpt source):** the carried `initial` is a
loop-carried dependency, so the carry loop MUST be `sequential_range` (D1b) or
`static_range` (D1a) — **never `affine_range`** (silently corrupts the scan; AccelOpt
hit this). `tensor_tensor_scan`'s `initial` accepts a `[P,1]` tile or a scalar 0
(confirmed in the API docs). Land D1b, measure seed-42 then 5-seed; keep D1a as the
guaranteed fallback if our exact build makes the outer `sequential_range` misbehave.
Promote the faster passing variant as `mamba_v2` (parent `mamba_v1`).

### S1 — Move the b/c broadcast OFF the Tensor engine  ⭐ (ceiling-raiser; beat 1.6×)

This is the one lever that can pass the 1.608× reference, because that reference is
PE=99%-bound on broadcast matmuls while DMA idles at 11%. Two sub-levers, safe → strong:

- **S1a — Hoist the broadcast out of the 2-channel loop (D3a).** `b[s,:]`/`c[s,:]` are
  channel-independent; with C fixed at exactly 2 tiles, broadcasting once per
  (seq-tile, state) and reusing across both channel tiles **halves** the broadcast
  matmuls (at seq_tile=512: 896 → 448). Pure hoist, obviously correct, no new API.
  **Caveat:** the compiler may already CSE the channel-independent broadcast, in which
  case S1a is a no-op — gate on measured latency, not intent.
- **S1b — Stride-0 DMA-load broadcast (D3b) — the real ceiling-raiser.** Instead of
  broadcasting a `[1, seq_tile]` SBUF row across partitions via `nc_matmul`, load
  `b[s,:]`/`c[s,:]` from HBM directly into a `[128, seq_tile]` SBUF tile with a
  **stride-0 partition access pattern** (every partition reads the same HBM row). This
  spends idle DMA bandwidth instead of saturated PE. **Direct precedent: `5f08e8cb`
  (`mlp_tkg_down_projection.py`)** replaced a `nc_stream_shuffle`/PE partition-broadcast
  of a `[1,H]` bias with exactly this — a `TensorView.broadcast(dim=0, size=T)` view fed
  to `dma_copy` (compiler lowers it to one broadcasted-DMA pattern). Implement via the
  broadcast-view (or `src.ap(pattern=[[0,128],[1,seq_tile]])`) + `dma_copy`, prefer
  `dge_mode=none` for the static contiguous load (precedent `d1124a76`).

**Why S1 is the phase-3 keystone.** It attacks the #1 measured bottleneck and is the
only direction with headroom below 0.78 ms. **Keep/drop metric (from the phase-2
deliberation): latency is primary**, PE-active-*time* (`latency × PE%`) is the
directional proxy (PE% alone is confounded — it can stay ~99% even if total PE work
halves), DMA% is the guardrail (S1b must not push DMA toward saturation). **Risk:** S1b
emits ≥128 descriptors per broadcast; confirm DMA stays well below saturation and PE
actually falls, else fall back to S1a. Correctness is trivially preserved — the
broadcast *values* are identical; only how the tile is materialized changes.

### S2 — Specialize the seq-tile regime for the fixed M=7168 (the "tile-size regime" lever)

This is the prompt's headline specialization. Once S0's structure is chosen, sweep the
exact divisors `seq_tile ∈ {256, 448, 512, 896, 1024}` (28/16/14/8/7 tiles). **The
regime tradeoff here is the OPPOSITE of the silu task's "finer wins":**

- silu was DMA-streaming (no scan, no broadcast) and preferred finer tiles (~4 KB burst).
- mamba is **PE-bound on broadcast matmuls whose count is `n_seq_tile × S × (channels or
  1 if hoisted)` — i.e. inversely proportional to seq_tile width.** Larger tiles ⇒ fewer
  broadcasts ⇒ less PE pressure. So the hypothesis is **wider wins** (896/1024), bounded
  by (a) SBUF/double-buffer slack — a `[128,1024]` fp32 tile is 4 KB/part, still tiny —
  and (b) the outer carry serialization (more tiles = more sequential carry points).
  512 was the reference's "magic number," but that was *before* attacking the broadcast;
  the optimal regime interacts with S1 (if S1b moves broadcasts to DMA, the DMA
  descriptor count then scales with `n_seq_tile`, flipping the pressure back toward
  wider tiles). Sweep S2 **after** S1 lands, and hard-code the winner for the fixed M.

Benefit likely a few %, but it's a pure knob and it's exactly the shape-specialization
the phase asks for. Risk minimal.

### S3 — Relieve the Vector engine (fuse + hoist) — only after PE drops

Once S1 moves the broadcast off PE, **Vec (89%) becomes the limiter.** Two moves:

- **S3a — Fuse `scanC = scan_res·c` + accumulate into one Vector op.** Today it's
  `tensor_tensor(scan_res, c_bcast, multiply)` then `scanC_accum += scanC` (two Vector
  passes). `nisa.scalar_tensor_tensor(dst, data=scan_res, op0=multiply,
  operand0=c_row/c_bcast, op1=add, operand1=scanC_accum)` does `(scan_res·c)+accum` in
  **one pass at `tensor_tensor` cost** (confirmed in API docs). Constraint: `data` and
  `operand1` can't both be in PSUM — both are SBUF here, fine. (If S1b makes c a
  free-dim-foldable operand this fuses even more cleanly.)
- **S3b — Hoist state-independent `deltaU = delta·u` out of the state loop (D4).**
  `deltaU` doesn't depend on `s`, yet it's recomputed in all 16 state iterations. Compute
  once per (channel-tile, seq-tile); removes ~20% of the Vector `tensor_tensor` work.
  Adds one live `[128, seq_tile]` buffer (2 KB — trivial).

Both are correctness-neutral (same math). Rank below S1/S2 because they only pay once PE
is no longer the ceiling — but a post-S1 digest showing Vec on top makes them the next
lever, and S3b is safe enough to fold in regardless.

### S4 — Static-unroll the compile-time-constant loops (cheap shape specialization)

S=16 and n_channel_tile=2 are compile-time constants. Replacing their `affine_range`
with `static_range` (full unroll) can give the compiler a longer, dependency-free
instruction window to double-buffer across — a classic fixed-shape specialization. Note
D1b already needs the *seq* loop to be `sequential_range` (carry); unrolling the **state
and channel** loops is independent of that and correctness-neutral. Cheap probe; keep
only on a measured latency drop.

## Plan of attack (≤5 iterations)

1. **Iter 1 — S0 (land the floor).** Implement D1b (seq_tile=512), score seed-42
   `--fast`, read digest; expect ≈0.78 ms / 16 MB / PE≈99 / Vec≈89. Also implement +
   measure D1a once. Promote the faster passing variant as `mamba_v2` after a **full
   5-seed** run. This alone takes us from 0.832× to ~1.6×.
2. **Iter 2 — S1 (off-PE broadcast).** S1a hoist first (safe); if the digest shows PE
   unchanged (compiler already CSE'd), go S1b (stride-0 DMA broadcast, `5f08e8cb`
   pattern). Keep only if latency drops below the S0 winner; PE-active-time proxy + DMA
   guardrail. Target < 0.78 ms. `mamba_v3`.
3. **Iter 3 — S2 (seq-tile regime sweep {256,448,896,1024} vs 512)** against the current
   winner. Hard-code the winning width for M=7168. `mamba_v4`.
4. **Iter 4 — S3 (Vec relief)** *if* the post-S1 digest shows Vec as the limiter:
   `scalar_tensor_tensor` fuse (S3a) + hoist `deltaU` (S3b). Else spend it on S4 unroll.
   `mamba_v5`.
5. **Iter 5 — Harden + final specialization lock.** Full 5-seed `verify.py` (no
   `--fast`) on the best candidate; ensure `benchmark.csv` / `candidates.jsonl` complete
   with DAG parent links and per-engine digests; try S4 unroll if not yet done. Report
   final speedup vs baseline.

Record every candidate in `benchmark.csv` + `candidates.jsonl` (DAG parent links),
profiling evidence under `profile/`. Kernel sources under `runs/` (tracked). Full 5-seed
before any promotion — never promote on `--fast` alone (other tasks showed fast-mode can
mis-rank close candidates).

## Correctness watch-items

- **Carry-loop kind.** The carried `initial` makes the seq-tile loop loop-carried → must
  be `sequential_range` (D1b) or `static_range` (D1a), never `affine_range`. First check
  on every S0 variant: seed-42 rel-L2 passes.
- **Carry extraction identity.** The carried state is exactly the previous tile's final
  scan column: D1a carries a `[128,1]` slice (not squeezed / not from the wrong axis);
  D1b writes state `s`'s final value into `[i_channel_tile, :, i_state]` without
  cross-mixing the 2 channel tiles or the 16 states. A corrupted carry exceeds 3e-5 given
  the 4.08e-7 oracle margin, so 5-seed pass ⇒ carry indexing is correct.
- **S1b broadcast values.** The stride-0 DMA broadcast must materialize the *same* row in
  every partition (partition stride 0, free stride 1). Verify rel-L2 stays ~1e-6, not
  merely < 3e-5.
- **Op sequence unchanged.** No precision reduction below fp32, no reordered
  non-associative math — a diff vs the S0 winner shows only structure / broadcast-mechanism
  / fusion changes.

## Risks / watch-items

- **S1a may be a no-op** (compiler CSE) — that's fine, it de-risks S1b; gate on latency.
- **S1b descriptor cost** — ≥128 descriptors/broadcast; if DMA climbs toward saturation
  without PE falling, revert to S1a. This is the one moderate-risk lever.
- **S2 regime interacts with S1** — the PE-vs-DMA pressure balance shifts once the
  broadcast moves engines, so sweep S2 *after* S1 is decided, not before.
- **Don't over-fit `--fast`** — confirm the ranking on the full 5-seed / higher-iter run
  before promoting.
- **Scope discipline** — S0 (the ~1.6× floor) is the phase's hard success target; S1–S4
  are evidence-gated specializations, kept only where the measured win justifies the
  added complexity, exactly as the prompt directs.
