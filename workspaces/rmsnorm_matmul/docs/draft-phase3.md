# rmsnorm_matmul (M4096 N2048 K1024, fp32) — Phase 3 implementation draft

## Goal

Phase 3 = **regime / shape specialization**: analyze where time goes across the
tensor's structure and specialize *only* where a measured win justifies the added
complexity. Start from the promoted kernel (`runs/rmsnorm_matmul_v1.py`, **1.066x**,
full 5-seed PASS) and never regress correctness (NKIBench relative-L2 gate
`< 2e-5·||v_r||` on seeds `[0,21,42,63,84]`, fp32) or latency below v1.

The honest framing up front, so the phase is scoped correctly: **Phase 2 already
established that v1 sits at the fp32 systolic PE floor** with the only non-floor PE
work (the transpose) *fully hidden* under the matmul (`load_transpose2d` removed the
transpose entirely and landed within noise, PE pinned at 97%). Phase 3's job is to
(a) confirm there is no *shape-structure* lever left that Phase 2 didn't already
cover, and (b) run the one remaining swing that could beat the fp32 floor — a
precision-split matmul — as a gated calibration, since it is the only thing that can
move the structural ceiling.

## What "shape specialization" can and cannot buy here (the decisive analysis)

Phase 3 for other operators means: edge tiles, tile-size regimes, partition/free
splits per shape. I checked each against *this* op's fixed shape and found **the shape
offers essentially no specialization surface**:

| Specialization lever | Applies here? | Why |
|---|---|---|
| **Edge / partial tiles** | **No** | Every dim divides evenly: M=4096=32·128, K=1024=8·128, N=2048=4·512. There is **no ragged tile anywhere** — no masks, no remainder loop. Nothing to specialize. |
| **Tile-size regime (partition/free split)** | **No** | The layout is *forced*. `nc_matmul` needs the contraction dim (k_in) on the partition axis; the required output tile is `[m_in(par), n(free)]`. So m_in must be the stationary/partition dim and n the moving/free dim. You cannot swap m↔n without transposing the whole result back. |
| **N-chunk (moving free) width** | **Already optimal** | v1 uses N_CHUNK=512 = `psum_fmax` = exactly one PSUM bank — the documented per-matmul moving-free maximum (precedent `6288aaad`: "tile budget is psum_fmax, not pmax"). Larger is impossible; smaller wastes the array. |
| **M-blocking (sibling matmul's phase-2 win)** | **No** | That win removed *redundant w HBM reloads* by reusing a loaded w-tile across B output-row tiles. Here w is **already fully resident** (8 MB, loaded once) and DMA is only ~20% busy — there is no redundant traffic to block for. |
| **LNC2 sharding** | **Out of contract** | Scored single-core, `--logical-nc-config=1`. Not a lever on this harness. |

So the classic phase-3 specializations are all either **vacuous** (no edge tiles) or
**already at their constraint** (N=512=psum_fmax; w resident). This is itself a
finding worth recording, not a gap.

### Where the time actually goes (measured + modeled, reconciled)

```
v1: latency=0.4716ms  MFU=46%  PE=97%  Vec=15%  Scl=11%  DMA=20%  HBMrd=25MB HBMwr=34MB
```

- **bf16-peak floor** for this GEMM = 2·M·N·K / (128·128·2.4e9·2) = **218.5 µs**. This
  equals the cost model's "stream-only" PE floor exactly, and **218.5 / 471.6 = 46.3%
  = the measured MFU.** So MFU=46% is *literally* "measured latency vs the bf16-peak
  denominator" — it is a structural fp32 ceiling, not inefficiency.
- The 2.16× gap between measured (471.6 µs) and the bf16 floor (218.5 µs) is dominated
  by the **fp32 PE-rate penalty**: D4's same-kernel bf16 swap ran **3.23× faster**
  end-to-end. That penalty is unschedulable — it is the array being bf16-native.
- Vec 15% / Scl 11% / DMA 20%: RMSNorm and all data movement are **comfortably hidden**
  under the PE-bound matmul. Confirmed by D1 (post-scale fold moved the scale pass off
  the input with no latency change) and D2b (transpose fully removed → within noise).

**Conclusion:** the only two things that could move latency are (1) shaving any
*exposed* systolic weight-fill on the main matmul — a within-fp32 micro-lever — and
(2) breaking the fp32 rate ceiling itself, which requires lower precision and thus
must clear the 2e-5 gate. Everything else is hidden or already optimal.

## Candidate directions, ranked by benefit ÷ risk

### P1 — Main-matmul loop reorder to amortize the stationary weight-fill (PRIMARY; cheap, correctness-safe)

This is the **one main-matmul micro-lever Phase 2 never touched** — Phase 2 worked
entirely on the *transpose*, never on the GEMM's own schedule.

**Observation.** v1's matmul loop is `c`-outer (4 N-chunks), `kt`-inner (8 K-tiles):
```
for c in 4:            # N-chunk
    acc = zeros[128,512]
    for kt in 8:       # K-tile — stationary xT[kt] changes every call
        acc += nc_matmul(stationary=xT[kt], moving=w_sb[kt, c-slice])
```
The **stationary operand `xT[kt]` is reloaded into the systolic array on every one of
the 1024 `nc_matmul` calls.** A `[128,128]` stationary costs ~128 cycles to fill the
array vs 512 cycles to stream the moving tile. If those fills are *exposed*, they are
128/(128+512) ≈ 20% of the matmul, ~55 µs.

**Fix.** Reorder to `kt`-outer, `c`-inner with **4 live PSUM accumulators** (4 banks,
fits the 8-bank budget), so each stationary `xT[kt]` is loaded **once** and streamed
against all 4 N-chunks before moving on:
```
acc[0..3] = zeros[128,512]         # 4 live PSUM banks
for kt in 8:
    xt = xT[kt]                    # load stationary ONCE
    for c in 4:
        acc[c] += nc_matmul(stationary=xt, moving=w_sb[kt, c-slice])
```
This is the weight-stationary dataflow the knowledgebase documents for amortizing
weight load (`scheduling-and-pipelining`; `6288aaad` on PSUM-bank accumulator sizing).

- **Theoretical ceiling:** fill goes from 1024×128 to 256×128 cycles → saves ~41 µs =
  **~8.7% of latency** *if fills are currently fully exposed*.
- **Realistic expectation: likely within noise.** Two strong priors say the compiler
  already hides these fills: (i) v1's loops are `nl.affine_range` (unordered), so the
  scheduler already has freedom to pipeline fills behind the previous matmul's stream;
  (ii) D2b removed the *transpose's* fills+compute entirely and moved nothing — direct
  evidence that PE-side fills on this kernel are not on the critical path. So P1 is most
  likely a **within-noise confirmation**, not a win.
- **Why do it anyway:** it is *cheap* (loop reorder + 4 PSUM banks), **bit-identical**
  in accumulation order (same K-accumulation, just N streamed inside), and it closes the
  last structural question — "is any weight-fill exposed on the main matmul?" — with a
  measurement instead of an assumption. Correctness risk ≈ 0 (no arithmetic change).
- **Watch:** 4 live `[128,512]` fp32 PSUM banks = 4 of 8 banks — fine. Confirm the
  reorder doesn't spill SBUF (xT already resident) or serialize on PSUM-bank
  dependencies. Keep the D1 post-scale eviction fold (apply `inv_rms` via `tensor_scalar`
  reading PSUM directly) since it is already proven and removes the input-scale pass.
- **Iterations:** ≤2 (implement on the D1 base; `--fast` triage; full-5-seed if it beats
  v1 by > the ~1.8% noise band).

### P2 — Fold the RMSNorm scale onto w-load instead of per-tile (LOW value, only if P1 idle) 

A variant to consider only if P1 shows unexpected exposed Vector/Scalar time: since
Vec+Scl are ~26% and hidden, there is no expected win, but if the reorder surfaces a
norm bubble, fusing the square+reduce+rsqrt tighter (or computing sum-of-squares via a
`nc_matmul(x, x)`-style path that shares the PE) could rebalance. **Default: skip** —
the norm is measured-hidden; this is a contingency, not a planned iteration.

### P3 — bf16×2 compensated split-matmul (STRETCH; the ONLY lever above the fp32 floor; GATED)

The fp32 PE rate is the structural ceiling (MFU=46% = bf16-peak/measured). The *only*
way past it is to do the matmul in bf16 arithmetic while recovering enough precision to
clear the 2e-5 gate — a **split/compensated GEMM**:

- Split each fp32 operand into two bf16 limbs: `x ≈ x_hi + x_lo`, `w ≈ w_hi + w_lo`
  (`x_hi = bf16(x)`, `x_lo = bf16(x - x_hi)`; same for w).
- Accumulate 3 bf16 products into fp32 PSUM: `x_hi·w_hi + x_hi·w_lo + x_lo·w_hi`
  (drop the negligible `x_lo·w_lo`). Effective mantissa ≈ 2·8 = 16 bits → rel error
  ≈ 2^-16 ≈ **1.5e-5**, just under the **2e-5** gate.
- Speed: 3 bf16 matmul passes, but bf16 is ~3.23× faster than fp32 (D4), so the compute
  is ~3×/3.23× ≈ 0.93× of the fp32 matmul — plus the split/quantize overhead. Net
  ceiling maybe ~1.05–1.2× *if and only if* it passes correctness.

- **Risk: HIGH on correctness.** 1.5e-5 vs 2e-5 is razor-thin; the accumulation of 3
  bf16 rounding steps plus the RMSNorm's own error could push it over on some seed.
  The sibling matmul task explicitly flagged "compensated-bf16x3 ~1.21x optimistic,
  uncertain vs 2e-5" and treated the 2e-5 gate as forbidding it. So this is a **gated,
  one-shot calibration**, not a planned promotion:
  - Implement the 3-pass split; score **full 5-seed** (not just `--fast`) to read the
    *actual* rel-L2 margin on the worst seed, plus latency.
  - **Promote only if** it PASSES all 5 seeds *and* beats v1 out-of-noise. Otherwise
    record it as the definitive "fp32 precision floor cannot be beaten within the gate"
    datum (upgrading D4's single-precision-swap calibration to a real attempt) and stop.
- **Risk: API.** Needs bf16 casts (`nl.copy`/activation with `dtype=bf16`) and bf16
  `nc_matmul` operands — bf16 matmul is the *native* path so this is well-supported
  (D4 already compiled a bf16 matmul on this remote). The split subtraction is plain
  `tensor_tensor`. Lower API risk than Phase 2's transpose probes.
- **Iterations:** ≤2 (implement 3-pass; full-5-seed correctness+latency; decide).

### P4 — REJECT: plain bf16 / tf32 matmul
Already closed by D4 (bf16 mantissa ~4e-3 » 2e-5; fails correctness). Not re-run.

## Plan of attack (order)

1. **Re-anchor the noise band:** one same-session v1 full-5-seed control (as in Phase 2)
   so P1/P3 are compared against a fresh control, not the historical 0.4716.
2. **P1 (primary):** reorder the main matmul to kt-outer / 4-live-PSUM-accumulator
   weight-stationary form on the D1 post-scale base. `--fast` triage; if > noise band
   over control, full 5-seed. Record PE% / MFU before-after to see if any fill left the
   PE. Most-likely outcome: within-noise → keep whichever of {v1, P1} is simpler/faster.
3. **P3 (gated stretch):** implement the bf16×2 split matmul; **full-5-seed** correctness
   read first (the gate is the whole question), then latency. Promote only on PASS +
   out-of-noise win; else record as the precision-floor confirmation.
4. Keep the best correct kernel. Expected: if neither P1 nor P3 clears the noise band /
   gate, **v1 remains promoted** and Phase 3 exits as a *specialization-space closure*:
   documented evidence that the shape has no edge/tile-regime lever and the fp32 rate is
   the ceiling — mirroring the sibling matmul's phase-3 floor-confirmation.

Never promote a candidate that fails any seed or regresses below v1. Each direction gets
≤5 iterations; stop a direction early on a clear reject or a clean within-noise read.

## Correctness invariants (must hold; unchanged from v1 unless noted)

- Every dim divides evenly → **no masks / partial tiles** anywhere (this is also why
  edge-tile specialization is vacuous).
- Reduction stays a single full-1024 free-axis `tensor_reduce`; `inv_rms = rsqrt(sumsq·1/K)`
  (folded 1/K) — rel-L2 1.3e-7 role, identical to v1.
- **P1:** matmul accumulation stays **fp32 in PSUM**, same K-accumulation order (only the
  N-chunk loop moves inside the K loop) → **bit-identical** result to the D1 base. Output
  tile `[m_in(par), n(free)]` and per-row `inv_rms` broadcast at eviction unchanged.
- **P3 only:** accumulation becomes 3 bf16 products summed in fp32 PSUM. This *changes the
  numerics* (that's the point) → its correctness is re-established empirically on the full
  5-seed gate, and it is promoted *only* if the measured worst-seed rel-L2 < 2e-5. The
  fp32 v1/P1 path is untouched and remains the guaranteed fallback.

## Risks / things to watch

- **P1 within-noise (most likely).** Do not promote a < ~1.8% "win"; re-run the control
  and require margin. A clean within-noise read is a *valid closing result*, not a
  failure — it confirms fills are already hidden (consistent with D2b).
- **P3 correctness cliff.** The 1.5e-5 vs 2e-5 margin is thin and seed-dependent; always
  score **all 5 seeds** before believing a pass. RMSNorm error stacks on top of the
  matmul split error. If any seed fails, it's a reject, full stop.
- **PSUM pressure (P1).** 4 live `[128,512]` fp32 banks = 4/8 banks; P3's split may want
  more transient PSUM/SBUF for limbs — watch for spills in the profiler (sibling saw
  residency regressions when banks/buffers grew).
- **Noise band.** ~±1.8% run-to-run (sibling + Phase-2 observed). `--fast` for triage,
  **full 5-seed** before any promotion.
- **Older remote NKI.** P1 uses only proven primitives (nc_matmul, PSUM accumulate,
  tensor_scalar) — no API risk. P3's bf16 matmul path already compiled in D4.

## Phase-3 success criterion

A correct (full 5-seed PASS) kernel that beats v1's **1.066x** by more than the noise
band. Realistic outcomes, in order of likelihood:
1. **Floor-confirmation (most likely):** P1 within noise, P3 fails the gate → **v1 stays
   promoted (1.066x)**, now with the added evidence that (a) the shape has no edge/tile
   specialization surface, (b) the main-matmul weight-fills are already hidden, and (c)
   the fp32 rate is the hard ceiling that the 2e-5 gate forbids breaking. This is a
   complete, defensible phase-3 result — the same shape of outcome as the sibling matmul
   task, reached by measurement rather than assumption.
2. **Small P1 win:** if some weight-fill was exposed, the reorder lands ~1.07–1.09x →
   promote P1.
3. **P3 surprise:** if the bf16×2 split both passes all 5 seeds and beats v1 out-of-noise
   (~1.1–1.3x), promote it — the only path that breaks the fp32 floor. Treated as a
   high-value long shot, gated strictly on the correctness measurement.
