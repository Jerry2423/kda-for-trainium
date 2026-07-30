# rope_single_freq_apply — Phase 3 draft (regime / shape specialization)

## 0. Starting point (phase-2 exit state)

| kernel | build | tile W | p50 | vs baseline | vs layout A | verdict |
|--------|-------|--------|-----|-------------|-------------|---------|
| baseline (NKIBench) | — | 512 | 1.1418 ms | 1.000x | — | — |
| `rope_v1` (layout A) | vector-bound | 2048 | 0.9445 ms | 1.209x | 1.000x | phase-1 promoted |
| `rope_v2_layoutB_scalar_w512` | all-Scalar | 512 | 0.7337 ms | 1.557x | 1.286x | **exact fallback** |
| **`rope_v3_layoutB_pe`** | **PE + Scalar** | **512** | **0.696 ms** | **1.641x** | **1.358x** | **PROMOTED** |

All layout-B candidates are **exact** (per-seed rel-L2 = 0.0 on all of `[0,21,42,63,84]`).
The promoted kernel is layout B (128-partition packed): `A=[x0;x1]` natural load, `cos_pack`/
`sin_pack` Scalar broadcasts, `x_swap_neg=[-x1;+x0]` via a 0/±1 permutation `nc_matmul` on
the PE engine, then 3 packed `tensor_tensor` over `[128,512]` and one store.

**Bottleneck (measured, `rope_v3`):** DMA-bound. `DMA-active 98.8% (0.689 ms)` is the sole
wall-clock limiter; all compute engines hide under it — `Vec 68% (0.475 ms)`, `PE 51%
(0.354 ms)`, `Scl 47% (0.329 ms)`. HBM traffic `402.72 MB` = the read-once/write-once floor
to the byte (+65 KB one-time `swap_const`). `effBW 578 GB/s = 74%` of the 781 GB/s single-core
streaming roofline (0.516 ms pure-DMA ceiling), `MBU ~40%`. The remaining headroom is a
**~0.17 ms DMA-active-vs-roofline gap**, not compute.

## 1. Phase-3 shape analysis — where does time go across the tensor's structure?

The phase-3 prompt asks me to specialize by tile-size regime / partition-free split / edge
tiles. I analyzed each against the actual shapes and most have **no surface** here:

- **Edge tiles / ragged remainder: NONE.** `S = B·H·N = 262144 = 2^18`, so *every* power-of-two
  tile width divides `S` exactly — all tiles are rectangular and mask-free at any W. There is no
  tail tile to specialize.
- **Partition split: NONE to gain.** `D = d_head = 128 = P_MAX` exactly; layout B already packs
  both output halves onto all 128 partitions (the phase-2 win). Partition usage is saturated.
- **Per-stream partition heterogeneity (structural, but not a byte lever):** the 4 DMA streams
  per tile have different partition footprints — `x`-load `[128,W]`, `cos`-load `[64,W]`,
  `sin`-load `[64,W]`, `out`-store `[128,W]`. This is inherent (`cos`/`sin` are `(64,S)` HBM
  tensors); it cannot be coalesced (three *distinct* HBM allocations) and is already at the byte
  floor. Not a specialization surface, just context for DMA-active.
- **Tile-size regime (the ONE real lever): free-axis width `W`.** Phase 2 swept
  `W ∈ {512,1024,2048,4096}` and found **W=512 wins** (monotone-ish improvement as W shrinks;
  W=1024 was a non-monotone dip). Phase 2 explicitly *stopped at W=512* as "the finest point in
  the plan's sweep contract"; **going finer (W=256, W=128) was declared out-of-contract and left
  for phase 3.** This is the direction phase 3 exists to test.

**Precedent that makes finer-W the primary bet:** the silu phase-3 SURPRISE (my own prior task)
was exactly this — for a DMA-bound streaming kernel, *finer* free-axis tiling WON
(`affine_range(224)` over a reshape view, "go finer not wider on affine_range streaming;
optimum ~4 KB/partition burst, s=8 bracket regresses"). rope is now in the same regime
(DMA-co-limited, 74% of roofline), and its phase-2 sweep already trends finer-is-better down to
the contract floor. W=512 = 2 KB/partition burst; W=256 = 1 KB, W=128 = 0.5 KB — untested.

## 2. Two new phase-3 findings from the profile (both change the plan)

### 2a. `dge_mode` is a DEAD lever under this harness (closes the top DMA precedent)
The knowledgebase's #1 DMA fix ("set `dge_mode=none` for static contiguous DMAs" —
`d1124a76`) does **not** apply: this harness compiles every kernel with
`--disable-dge --logical-nc-config=1`, and `--disable-dge` forces the backend to set
`DgeType=None` **globally**, nullifying any per-op `dge_mode=none/hwdge/swdge` request in the
measured NEFF. The profile confirms all DMA is already software-dynamic
(`software_dynamic_dma_active_time_percent = 0.994`), i.e. runtime-generated descriptors.
`dge_mode` is therefore not a usable lever — the descriptor-generation overhead riding inside
DMA-active is a fixed cost of the harness's compile flags, not something the kernel can tune.
This is why effBW plateaus at ~74% of roofline: the residual gap is software-DMA descriptor
overhead we cannot remove, plus the (already-harvested) fill/drain bubble.

### 2b. The PE-hybrid path incurs power-throttling the Scalar path does NOT
The `rope_v3` (PE) profile reports throttle activity that is **absent** on both `rope_v1`
(layout A) and `rope_v2_layoutB_scalar_w512` (the metrics read `None` there):

| metric (`rope_v3` PE) | value |
|---|---|
| `throttle_active_nc0_time_ns` | 737,342 (0.737 ms) |
| `throttle_activity_1_active_time_nc0_percent` | 0.522 @ util-limit 0.5 |
| `throttle_avg_util_limit_nc0_percent` | 0.722 |

Most-likely reading: the `nc_matmul` on the PE (Tensor) engine draws enough power to trip a
utilization-throttle for roughly half the run. The PE path **still wins** (0.696 < 0.734 ms), so
the throttle is not disqualifying — but it is a hidden cost the "PE hides under DMA" active-time
story misses, and it means the PE-vs-Scalar margin could shift with tile geometry. This is a
**flagged observation to re-verify at finer W**, not yet a confirmed defect.

## 3. Directions to try (measured, exact, in-scope)

### D1 — Finer free-axis tile-width sweep (PRIMARY)
Extend W below the phase-2 floor: **W ∈ {256, 128}**, on the promoted PE-hybrid layout B.
- **Exactness:** arithmetic-preserving — only the tile *count* changes (`affine_range(S//W)`),
  not the ops or dtype. Expect rel-L2 = 0.0. Both fit the PE cap (`nc_matmul` moving-free ≤ 512;
  256 and 128 ≤ 512) and only *shrink* SBUF vs W=512 (no spill risk).
- **Hypothesis:** deeper `affine_range` pipeline shrinks the relative DMA fill/drain bubble and
  raises DMA-active / effBW toward the 781 GB/s roofline (silu finer-wins precedent + rope's own
  finer-is-better phase-2 trend).
- **Risk:** per-tile overhead grows — W=256 → 1024 iters, W=128 → 2048 iters; more loop /
  descriptor / PE-issue overhead may dominate and *regress*. That is the measurement's job to
  decide.
- **Method:** `--fast` screen (seed 42) for W=256 and W=128; the best finer W clears the full
  5-seed gate + stabilized p50 (3 runs) before any promotion.

### D2 — Throttle-aware PE-vs-Scalar re-race at the winning finer W (SECONDARY)
Re-run BOTH build-engine realizations (PE-hybrid `rope_v3` and throttle-free all-Scalar
`rope_v2_layoutB_scalar`) at the D1-winning W, head-to-head with stabilized p50.
- **Why:** finer tiles shrink the per-tile PE matmul (W=128 → `[128,128]@[128,128]`), which may
  change both PE efficiency and the throttle penalty. If the throttle-free Scalar variant catches
  or beats PE at finer W, prefer it (DEC-1 already tie-breaks toward the simpler, robust path).
- **Exactness:** both are exact (pure data movement + IEEE sign flip / 0/±1 permutation).
- This is the "engine-placement regime" specialization — the one place the tensor's compute
  structure meets a hardware cost regime (power throttle).

### D3 — DMA burst/queue shaping (CONTINGENT — likely no clean lever, document either way)
Only if the D1/D2 digests show queue-starvation or burst-fragmentation evidence: investigate
whether the 4-stream (`x`/`cos`/`sin` read + `out` write) schedule can spread across more
parallel DMA queues or issue larger contiguous bursts. Under `--disable-dge` the compiler already
uses ~35 software-DMA queues and `dge_mode` is nullified (§2a), so I expect **no clean NKI knob**
— pursue only on evidence, else record as an investigated no-lever.

## 4. Directions rejected up front (with reasons)

- **`dge_mode=none/hwdge/swdge`** — DEAD under `--disable-dge` (§2a); the harness forces
  `DgeType=None` regardless. Closes knowledgebase DMA precedent #1.
- **bf16 / tf32 downcast (D3 in phase 2)** — DEAD (AC-7 carryover). RoPE is pure elementwise with
  no reduction to average away rounding; bf16 error ~4e-3 ≫ the 2e-5 gate. fp32 mandatory.
- **Manual ping-pong / wider-burst batching** — DEAD (AC-7, silu precedent). `affine_range`
  already builds the software pipeline; the phase-2 W-sweep showed *wider* W regresses.
- **Cutting HBM bytes / coalescing the 3 reads** — IMPOSSIBLE. Traffic is at the read-once/
  write-once floor to the byte; `x`, `cos`, `sin` are distinct HBM tensors.
- **Edge-tile / partition-split specialization** — NO SURFACE. `D=128=P_MAX`, `S=2^18` divisible
  by every power-of-two W → perfectly homogeneous, mask-free tiles.

## 5. Acceptance criteria (phase 3)

- **AC-1 Correctness.** Every candidate arithmetic-preserving → full 5-seed rel-L2 = 0.0 expected
  (any nonzero is a red flag for a compiler-introduced change and must be investigated). Screen
  `--fast` (seed 42); promote only on the full 5-seed gate.
- **AC-2 HBM at floor.** 402.65 MB (+65 KB `swap_const` for the PE variant) unchanged — finer W
  must not introduce spill or extra traffic.
- **AC-3 Measured win justifies complexity.** Promote a finer-W variant only if its stabilized
  p50 (3 runs) beats the current **0.696 ms by more than the ~3–5% DEC-1 noise band**. Otherwise
  keep `rope_v3` and record the finer-W sweep as evidence that W=512 is the floor.
- **AC-4 Robustness tie-break (DEC-1).** If two variants land within the noise band, prefer the
  throttle-free / simpler one (all-Scalar, no PE).
- **AC-5 Honest ceiling.** If no finer W wins, deliver the documented "at achievable ceiling"
  verdict — the finer-W sweep + throttle (§2b) + `dge_mode` (§2a) findings together explain why
  0.696 ms / 74%-of-roofline is near the limit under these compile flags. "Specialize only where
  the measured win justifies the added complexity" explicitly permits *no further specialization*
  as the answer.

## 6. Deliverables

- Finer-W candidate kernel `.py` sources under `runs/` (tracked).
- `benchmark.csv` row per perf change; `candidates.jsonl` DAG entries (parent = `rope_v3` for the
  PE finer-W variants, `rope_v2_layoutB_scalar_w512` for the Scalar finer-W variants).
- `profile/` digest for the finer-W sweep + the throttle / `dge_mode` analysis.
- `docs/phase3-exit-decision.md` with the final verdict.
- Final promoted kernel evaluated on the **full 5-seed correctness gate** with speedup vs
  baseline reported.

## 7. Expected outcome (honest)

Two plausible endings, both acceptable:
1. **Finer W wins** — W=256 (or 128) shaves the DMA bubble, pushing effBW past 578 GB/s and p50
   below 0.696 ms by > the noise band → promote it (and re-check PE-vs-Scalar per D2).
2. **W=512 is already the floor** — finer W regresses on per-tile overhead → keep `rope_v3`, and
   the phase-3 deliverable is the rigorous "at achievable ceiling" verdict (§2a software-DMA
   overhead + §2b throttle + the sweep proving 512 is the min). Given the kernel is DMA-bound at
   74% of roofline with byte traffic at the floor and `dge_mode` unavailable, ending 2 is a
   genuine possibility, not a failure.
