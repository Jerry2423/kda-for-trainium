# silu (M4096 N7168, fp32) — Phase 2 implementation draft (profile-driven optimization)

## Goal

Starting from the promoted phase-1 kernel `runs/silu_v1.py` (0.3009 ms, 3.398x over
the 1.022441 ms baseline), use the profiler to find the *real* remaining bottleneck,
enumerate optimization directions, rank them by expected benefit vs risk, and explore
each for **at most five iterations** — collecting before/after latency (`verify.py`)
and per-engine evidence to justify keep / revise / reject. Never regress correctness
(relative-L2 < 2e-5 on all five seeds).

## Where phase 1 left us (the measured starting point)

`runs/silu_v1.py`: one `nl.affine_range(32)` over the middle axis; each iteration
loads a full-width `[128, 7168]` fp32 slice HBM→SBUF, applies one fused
`nisa.activation(op=nl.silu)` on the Scalar engine, stores `[128, 7168]` SBUF→HBM.
Two live SBUF tiles (x_tile, y_tile), no inner free-dim loop, mask-free.

Profiler digest (`profile/silu_v1.txt`, full 5-seed):

| latency | speedup | MFU | PE | Vec | **Scl** | **DMA** | HBMrd | HBMwr |
|---------|---------|-----|----|----|---------|---------|-------|-------|
| 0.3009 ms | 3.398x | 0% | 1% | 1% | 34% | **97%** | 117 MB | 117 MB |

Correct 1/1 on all five seeds; the fused `nl.silu` LUT is L2-accurate under the
2e-5 gate, so the phase-1 correctness ladder never had to descend to
sigmoid+multiply or the exp-exact chain. **No accuracy concern carries into phase 2.**

## Bottleneck analysis — v1 is already on the HBM roofline

This is the central finding of phase-2 investigation, and it reframes the whole
phase. I reconstructed the roofline two independent ways:

**1. Traffic is fixed and minimal.** SiLU is a pure elementwise map. The fp32 in/out
contract forces exactly read-once + write-once:
`2 * 4096 * 7168 * 4 B = 234.9 MB` total HBM traffic. v1's profiler numbers
(HBMrd 117 MB + HBMwr 117 MB = 234.9 MB) are **exactly** this floor — there are
zero redundant SBUF passes, zero recompute. There is no traffic left to remove.

**2. v1 is at the measured bandwidth ceiling.** Effective aggregate HBM bandwidth
implied by v1: `234.9 MB / 0.3009 ms = 781 GB/s`. Two cross-checks:
   - The cost model's conservative trn2 figure is 368 GB/s ⇒ serialized floor
     `234.9 MB / 368 GB/s = 0.638 ms`. v1 (0.3009 ms) is **2.1x below** that, which
     means the model's 368 GB/s is ~2x conservative for this streaming pattern
     (read and write overlap on the real HBM channels; the model serializes them).
   - Cost-model *compute* floor (fused silu, Scalar cpe=1, freq=120 MHz-scaled,
     free=7168, x32 iters) ≈ **0.191 ms** — comfortably hidden under DMA (Scl=34% «
     DMA=97%). And the per-direction overlapped DMA floor
     (`117 MB / 368 GB/s = 0.319 ms`) essentially equals the measured 0.3009 ms.
   - **DMA=97% active** is the direct confirmation: the DMA subsystem is saturated.
     Only ~3% (~9 µs) of the runtime is DMA-idle bubble.

**Conclusion: there is no multiplicative headroom.** AccelOpt's "~1.67x here" was
relative to the *baseline* (which does 4 vector passes through 5 SBUF buffers);
v1 already captured that and more (3.398x) by collapsing to read-once → fused-silu
→ write-once. The only slack that physically remains is the ~3% DMA-idle bubble.
Phase-2 realistic ceiling: **single-digit-% latency (0.3009 → ~0.29 ms), or zero.**
The honest primary deliverable of phase 2 is a *rigorous confirmation that we are at
the roofline*, plus harvesting the DMA-issue bubble **only if the gain clears the
same-session noise band** (per the fast-vs-full-run latency lesson).

## Optimization directions — enumerated, ranked by benefit/risk

The one lever with any physical basis is **reducing DMA-issue / scheduling overhead**
so the ~3% idle bubble shrinks. Everything that would give a *multiple* (less
traffic) is blocked by the fp32 contract. Ranked:

### D1 (rank 1, primary) — Wider DMA bursts via multi-slice batching

**Idea.** Process `k` middle-axis slices per iteration as **one contiguous** transfer
instead of `k` separate ones. Because the tiled layout is `(128, 32, 7168)` = `[p, m, f]`
with middle-axis stride 7168 and free stride 1, `v1[:, i0:i0+k, :]` is a *contiguous*
`[128, k, 7168] = [128, k*7168]` slab per partition. So one `nl.load` of
`[128, k*7168]`, one `nisa.activation(op=nl.silu)` over the `k*7168` free dim (well
under the ~32767 Scalar free-dim limit for k≤4: `4*7168 = 28672`), one `nl.store`.

**Why it could help.** Cuts the DMA op count from 32 loads + 32 stores (k=1) to
`32/k` each. Each DMA carries a fixed issue/semaphore cost (`semaphore_start` = 1300 ns
in the model); fewer, larger bursts amortize that fixed cost and can shrink the ~3%
idle. Precedent: `d3cbeffd` [legacy] gather/scatter "coalescing DMA copy to improve
performance"; the general wider-burst pattern.

**Tension to measure (this is why it's a sweep, not a fixed choice).** `affine_range`
overlaps DMA with compute *across iterations*; with `k` large there are fewer
iterations (k=4 → 8 iters), so the pipeline is coarser and prologue/epilogue bubbles
are relatively larger. So there's a sweet spot: bigger bursts cut per-DMA overhead
but coarsen the pipeline. **Sweep k ∈ {1, 2, 3, 4}** and keep the best.

**SBUF budget check (trn2, 208 KB usable/partition, 28 KB per slice).** Distinct
load+store buffers need `2k` slices live ⇒ `2k ≤ 7` ⇒ **k ≤ 3** with separate
x/y buffers; k=4 needs in-place (D3) or single-buffered store. This bounds the sweep
and couples D1 with D2/D3.

**Risk.** Low — correctness-neutral (elementwise; same math, just wider tiles;
128*32=4096 and 7168 stay exact, still mask-free). Main risk is *no* gain (bubble was
never issue-bound) or a small regression at large k (coarser pipeline). Cheap to test.

### D2 (rank 2) — Explicit double-buffering / ping-pong SBUF

**Idea.** Pre-allocate two SBUF buffer sets and manually prefetch slice i+1 while
computing slice i (precedent `bc877398`, `3c7e053b`).

**Why it is rank 2, not rank 1.** v1 *already* uses `nl.affine_range`, which licenses
the compiler to software-pipeline DMA against compute — and the evidence says it is
doing so: v1 sits at the overlapped floor (0.3009 ≈ 0.319 ms one-way), not the
serialized 0.638 ms. So explicit ping-pong is most likely **redundant** here. Worth
**exactly one** confirmation run: if the compiler's auto-pipeline is already optimal,
manual ping-pong yields ~0 and may even hurt (extra SBUF pressure competes with D1's
batch width, per the budget check above). Keep only if it clears the noise band.

**Risk.** Low correctness risk; real risk is wasted SBUF that shrinks the feasible
batch width. Test D2 as a small standalone check, then only combine with D1 if it won.

### D3 (rank 3, enabler) — In-place compute (write silu output into the load buffer)

**Idea.** For an elementwise op, `y_tile` can alias `x_tile` (activation reads and
writes the same SBUF region). Halves live SBUF residency from `2k` to `k` slices,
which **unlocks larger batches** for D1 (e.g. k=4, or k=6–7 single-buffered) without
blowing the 208 KB budget.

**Why rank 3.** It removes no HBM traffic (still read-once/write-once) and no compute,
so on its own it changes nothing measurable. Its only value is as an **enabler** for
a wider D1 sweep. Test only in combination with D1 if the k≤3 sweep suggests bigger
bursts are still improving at the SBUF ceiling.

**Risk.** Low, but must confirm `nisa.activation` supports in-place src==dst without a
read/write hazard on the Scalar engine; verify correctness on the full 5 seeds (an
in-place aliasing bug would show as an L2 failure, which `verify.py` catches).

### D4 (rank 4, likely rejected) — DMA descriptor mode (`dge_mode`)

Precedent `d1124a76` sets `dge_mode=none` for static contiguous DMAs to move
descriptor generation off the critical path. **But the scoring harness compiles with
`--disable-dge`** (from the acceptance contract), so hardware DGE is already off
globally — this lever is very likely **neutralized before we touch it**. Record the
reasoning; try at most one run only if D1/D2 leave an unexplained bubble, otherwise
reject without spending an iteration.

### D5 (rejected outright) — bf16 / traffic reduction

The only way to get a *multiple* is to move fewer bytes. The gate mandates fp32 in
and fp32 out, and 234.9 MB is the read-once/write-once minimum for that contract.
There is no dtype trick, no fusion, no recompute-avoidance left. **Reject** — and this
is precisely *why* no multiplicative phase-2 win exists. (Documented so phase 3, if it
specializes shapes, doesn't re-litigate this.)

## Experiment plan (fits the "≤5 iterations per direction" budget)

All candidates parented off `silu_v1` in `candidates.jsonl`; each perf change recorded
in `benchmark.csv`; profiler digests kept under `profile/`.

1. **D1 sweep (primary, up to 4 iters):** `silu_v2_k2`, `silu_v2_k3`, and — if D3
   confirms in-place is safe — `silu_v2_k4`. For each: `verify.py --fast` first to
   screen, then full 5-seed on any that beats v1. Record k vs latency; keep the best.
2. **D2 (1 iter):** `silu_v2_pingpong` (k=1 explicit two-buffer prefetch). Compare to
   v1. Expected ~0; keep only if it clears the noise band. If it wins, retest the D1
   sweep with ping-pong layered on (SBUF permitting).
3. **D3** is not a standalone candidate — it's folded into the k=4 D1 variant as the
   enabler. Its "test" is that the in-place k=4 kernel passes the full 5-seed L2 gate.
4. **D4** only if an unexplained bubble remains after 1–2; else reject on the
   `--disable-dge` reasoning without an iteration.

**Noise discipline (mandatory).** v1's win is real and large; any phase-2 delta is
tiny (single-digit %). Before promoting *anything*, establish a same-session noise
band: run v1 and the candidate back-to-back on the full 5-seed measurement and only
promote if the improvement exceeds run-to-run jitter. A candidate that merely ties v1
within noise is **not** promoted — v1 stays. This follows the recorded fast-vs-full
latency lesson: `--fast` (seed 42, low iters) is a screen, not a decision.

## Acceptance for phase 2

1. Correctness never regresses: every promoted candidate passes relative-L2 < 2e-5 on
   all five seeds (full run, not `--fast`).
2. The DMA-roofline finding is documented with evidence (traffic = 234.9 MB floor;
   effective BW ≈ 781 GB/s; DMA=97% saturated) so the "no multiplicative headroom"
   conclusion is defensible, not asserted.
3. If a burst-coalescing (D1) variant clears the same-session noise band, promote it
   and record k, latency, and the new speedup; otherwise keep `silu_v1` as the phase-2
   result and record that it is confirmed at the roofline.
4. Every candidate in `candidates.jsonl` (DAG parented off `silu_v1`); every perf
   change in `benchmark.csv`; profiler evidence under `profile/`.

## What phase 2 deliberately does NOT do

- **No new math / no dtype change.** fp32 in/out is fixed; the fused `nl.silu` LUT is
  already L2-accurate. No sigmoid+multiply, no exp-exact, no bf16 — none would help
  (compute is hidden; traffic is fixed) and bf16 would risk the gate.
- **No layout change / no transpose.** Elementwise ⇒ layout is correctness-neutral;
  keep `(128, 32, 7168)` so the harness reconciliation is untouched. Any batching
  slabs contiguous middle-axis slices, not a re-layout.
- **No chasing sub-noise deltas.** If the roofline analysis holds (it does on the
  measured numbers), phase 2 may correctly conclude with v1 unchanged plus a
  documented roofline confirmation. Reordering instructions to fight a 3% bubble that
  is within noise is not a win.
