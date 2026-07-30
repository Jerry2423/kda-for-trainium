# bmm — Phase 3 draft: regime / shape specialization

**Operator:** `bmm` (NKIBench case 2). Batched matmul `out[b] = lhs[b] @ rhs[b]`,
`b in 0..15`. `lhs (16,4096,64)=(B,M,K)`, `rhs (16,64,4096)=(B,K,N)` fp32 →
`out (16,4096,4096)`. **B=16, M=4096, K=64, N=4096.** Baseline **2.550 ms**.

**Start point:** `runs/bmm_v2.py` = the phase-2 promotion, **1.253x (2.0352 ms)**, full
5-seed L2 PASS (rel-L2 1.83e-7). Two-phase per-batch structure: transpose all 32 lhs
m-tiles of a batch up front into a resident `[64,4096]` pack, then all 256 main matmuls
with 1024-wide coalesced stores. Pure fp32; `bmm_v1` (0.663x) retained as fallback.

---

## 1. The phase-3 question, answered honestly up front

Phase 3 asks: *analyze where time goes across the tensor's structure and specialize only
where the measured win justifies the complexity (tile-size regimes, partition/free splits,
edge tiles).* For bmm the honest answer is that **classic shape specialization has almost
no surface**, and I want to establish that with numbers before spending remote runs, so the
phase does not chase a dead lever:

- **No edge tiles.** Every axis divides cleanly: `M=4096=32·128`, `N=4096=8·512=4·1024`,
  `K=64≤128` (single pass), `B=16`. There is no ragged remainder tile to special-case — the
  usual "specialize the edge" regime split does not exist here.
- **Tiles are already maximal and cannot be widened.** The main matmul is
  `[K=64]×[64,512]→[128,512]`. The moving free dim 512 is the **hard PSUM-bank wall on
  trn2** (one `nc_matmul` writes one bank = 512 fp32 elems/partition; the 2048/4096 width
  is trn3-only — confirmed in the NKI API doc). The stationary free dim 128 fills the PE
  columns. So there is no "bigger tile" regime to switch into.
- **The K=64 partition/free split is fixed and cheap.** K=64 fills only 64 of 128 PE
  partition rows — but on trn2 the matmul cost is `elements_per_partition·100/freq`,
  i.e. **proportional to the dst free dim (512) ONLY**, independent of K and M. A half-full
  contraction axis therefore costs *nothing extra* in time; there is no partition-split
  regime that recovers it. And K cannot be packed: `out[b]` are block-diagonal, so stacking
  two batches onto a 128-row contraction would *sum* two batches' products — numerically
  wrong (already closed in phase 2).

So the phase-3 "structure" to analyze is **not tile shape** — it is the **schedule across
the batch axis**, the one structural dimension phase 2 stopped short of. Phase 2 proved the
lever is *stream depth* (deepening the independent-matmul run cut per-matmul stall
monotonically, M_SUB 8→16→32). Phase 3 extends that same lever past the batch-loop boundary.

## 2. The corrected bottleneck (this reframes what phase 2 recorded)

Phase 2's promotion note and the memory say "bmm_v2 goes DMA-bound at 100% (1074 MB write
floor), pure-fp32 headroom exhausted." **The authoritative same-session counters refute
that.** The coarse `verify.py` DMA% is jittery — across the four D2-bracket runs it read
74% / 100% / 67% / **1%** for the *same kernel*. The counter-level truth
(`profile/bmm_phase2_d2_dump_metrics.txt`, same-session, divided by the metric window):

| signal | bmm_v2 | note |
|---|---|---|
| TRUE PE-active / inf | **2.0116 ms** | tensor_engine_active_time / window |
| tensor_engine_active_time_percent | **98.84%** | PE is the bound |
| dma_active_time / inf | **1.434 ms** (70.46%) | *below* PE-active — DMA is hidden |
| matmul_instruction_count | 8704 | 8192 main + 512 transpose |
| per-matmul PE-active | 0.2311 us | |

⇒ **bmm_v2 is PE-BOUND, not write-bound.** The 1074 MB write DMA (~1.434 ms of active
time) sits *under* the 2.012 ms of PE-active time and is hidden. This matters because it
means there **is** still headroom on the binding engine, and the phase-3 lever is PE
scheduling — not the (already-hidden, at-floor) DMA.

### The theoretical PE floor (trn2 instruction-cost model)

Matmul latency on trn2 = `dst_free_elems · 100 / 240` ns (freq 2.40 GHz), free-dim-only:

| instruction | count | dst free | ns/instr | total |
|---|---|---|---|---|
| main matmul `[128,512]` | 8192 (4096 sites × 2 fp32 passes) | 512 | 213.3 | **1.748 ms** |
| identity transpose `[64,128]` | 512 | 128 | 53.3 | **0.027 ms** |
| **PE floor** | | | | **1.775 ms** |

- **PE floor ⇒ ceiling 2.550 / 1.775 = 1.437x.** This is the hard fp32 wall.
- **DMA write floor** ≈ 1073.7 MB / 368 GB/s ≈ 1.375 ms `<` PE floor 1.775 ms — confirms
  fp32 bmm is PE-bound end-to-end (DMA can never become the true binder at fp32).
- **bmm_v2 at 2.0116 ms is 13.3% above the PE floor** (`2.0116/1.775`). That ~0.237 ms
  excess is residual **per-instruction schedule bubble** — the phase-3 target. Closing it
  fully → ~1.79 ms → ~1.42x; closing half → ~1.89 ms → ~1.35x.

### Where the residual bubble most likely lives: the batch boundaries

bmm_v2 is *two-phase per batch*: for each of 16 batches it (1) loads `rhs[b]`, (2)
transposes all 32 m-tiles into the pack (Pass 1, transpose→copy chain), then (3) streams
256 main matmuls (Pass 2). Phase 2 showed deepening Pass 2 cuts the stall — but Pass 2 is
re-primed **16 times**, once per batch, and each batch head re-enters the serial
transpose→copy Pass 1 before its matmul stream can refill the PE. The 16 batch-boundary
transitions (Pass-1 transpose burst not yet overlapped with the *previous* batch's Pass-2
tail, plus the `rhs[b]` load and pack rebuild) are the natural place for the 13% residual.
This is the exact structural gap the phase-2 stream-deepening lever did not reach — it
deepened *within* a batch but left the batch boundary serial. **Round 0 must confirm this
localization before I build anything.**

## 3. Round 0 — measurements before any code change (near-zero remote risk)

Re-use the `runs/dump_metrics.py` idiom (reads TRUE `tensor_engine_active_time_ns` +
`matmul_instruction_count`, not the jittery PE%/DMA% proxy). All same-session vs a fresh
bmm_v2 anchor.

1. **Re-anchor bmm_v2 counters** — confirm TRUE PE-active ≈ 2.012 ms, matmul_instr 8704,
   and record the DMA-active *time* (not %) to nail down that PE-active `>` DMA-active
   (PE-bound). This is the fact the whole phase rests on; verify it fresh.
2. **Per-batch attribution proxy** — there is no per-region timeline from the profiler, so
   attribute structurally: the PE floor arithmetic above already isolates the residual to
   0.237 ms / 8704 instr = 0.0272 us/instr of average bubble. Cross-check by comparing
   bmm_v2's per-matmul 0.2311 us against the pure main-matmul floor 0.2133 us
   (213.3 ns) — the **0.0178 us/instr gap on the main matmuls alone** is the schedulable
   residual; the transpose adds the rest. If a candidate drives per-matmul toward 0.2133,
   it is closing the real bubble.
3. **(record-only) confirm D3 stays closed** — no new probe needed; phase-2 measured
   fp32/bf16 pass ratio = 2.0, so a 3-product bf16x2 main matmul costs 3.0 passes `>` 2.0
   and *raises* PE-active on a PE-bound kernel (would regress, exactly like swiglu's 2-pass
   all-3 split at 0.409x). D3 remains SKIPPED; the PE floor above is the fp32 wall and the
   only way past it (precision) is closed. State this so the phase does not relitigate it.

## 4. Optimization directions, ranked by expected benefit × confidence

### D1 — multi-batch blocking (PRIMARY; extends the proven phase-2 lever across batches)
Phase 2's winning lever was *stream depth*. bmm_v2 blocks one batch at a time (32 m-tiles).
D1 blocks **`B_BLK` batches together**: transpose the `B_BLK·32` m-tiles of `B_BLK` batches
into one resident pack, load `B_BLK` rhs tiles resident, then stream `B_BLK·256` main
matmuls with no transpose interleaved — so the Pass-1/Pass-2 transition happens `16/B_BLK`
times instead of 16, amortizing the batch-boundary bubble over a deeper stream.
- **SBUF budget (the binding constraint):** per batch the pack is `64×4096×4B = 16 KB/part`
  and rhs is `64×4096×4B = 16 KB/part` = 32 KB/part/batch. trn2 usable ≈ 208 KB/part ⇒
  `B_BLK ≤ 6` comfortably (192 KB), `B_BLK=4` is the safe sweet spot (128 KB, room for
  double-buffering the output SBUF tiles). Sweep `B_BLK ∈ {2, 4}` (and 8 only if SBUF
  fits without spill — watch HBMrd staying at the 34 MB floor).
- **Expected:** if the 13% residual is the batch boundary, this recovers most of it →
  ~1.85–1.90 ms → **~1.34–1.38x**. Pure fp32, bit-identical math (same 8704 instr, pure
  reschedule — like the M-block sweep).
- **Risk / kill-criterion:** phase 2 proved the `affine_range` compiler *already*
  software-pipelines aggressively and flattened every multi-bank/issue-order reschedule to
  a byte-identical no-op. The compiler may already pipeline across the batch `affine_range`,
  making D1 a no-op too. **Screen with `--fast` + `dump_metrics` first**; promote only if
  TRUE PE-active *drops* out-of-noise AND HBM stays at the 34/1074 MB floor (no spill from
  the larger resident footprint). ≤3 iterations (B_BLK sweep).

### D2 — cross-batch double-buffering of the resident operands (ALT; if D1 is a no-op)
If D1's static blocking is absorbed by the compiler, try the explicit ping-pong from the
`bc877398` / `3c7e053b` precedents: pre-allocate **two** `(rhs, lhs_t_pack)` buffer sets,
prefetch batch 0, then while batch `b`'s 256 matmuls stream, DMA-load `rhs[b+1]` and
transpose `lhs[b+1]`'s pack into the alternate buffer. This overlaps the Pass-1 transpose
burst and rhs load of batch `b+1` with batch `b`'s Pass-2 matmul stream — directly attacking
the batch-boundary serialization that D1 attacks statically.
- **Cost:** doubles the resident footprint to 64 KB/part (still fits, `< 208`).
- **Expected:** same target as D1 (~1.34–1.38x) via an explicit rather than
  compiler-inferred overlap. Prefer whichever of D1/D2 the counters show actually moves
  TRUE PE-active; they are two routes to the same batch-boundary bubble, so **do not spend
  iterations on both if the first lands.** ≤2 iterations.

### D3 — precision (bf16x2): CLOSED, record-only
The only lever that touches the 1.775 ms PE floor is precision, and it is closed: measured
fp32/bf16 pass ratio 2.0 ⇒ a compensated 3-product bf16x2 costs 3.0 passes `> 2.0`, *raising*
PE-active on a PE-bound kernel. bmm is a single raw matmul (no swiglu-style "split only the
cheap GEMM, keep the others fp32" rescue). SKIPPED; do not build. (Offline rel-L2 4.44e-6
would pass the accuracy gate, but the cost gate fails and both are required.)

### Closed / not-pursued (record-only, do not spend iterations)
- **Wider matmul tile** (moving free > 512): trn2 PSUM-bank wall, infeasible (trn3-only).
- **K-packing two batches onto 128 partitions:** numerically wrong (block-diagonal → sums
  batches). Closed in phase 2.
- **off-PE `load_transpose2d` (phase-2 D2):** counter-verified no-op (compiles to the same
  on-engine transpose, matmul_instr 8704 unchanged). Do not re-probe.
- **DMA store-burst / ping-pong output / bf16 output:** DMA is hidden (70% active `<` PE
  99%) and at the write floor; output dtype is the final result (2e-5 gate bans bf16 out).
  All dead — the bound is PE, not DMA.
- **store-direct-from-PSUM to drop the 4224 PSUM→SBUF copies:** architecturally impossible
  (no PSUM→HBM DMA path on trn2; HBM stores must source from SBUF). And the copies are
  already hidden behind matmul. Closed.

## 5. Method & discipline (per direction, ≤5 iterations total)
- **Noise anchor:** re-run bmm_v2 same-session as the control before each comparison
  (siblings saw ~0.02–0.5% jitter on this op; treat a ~1.5–2% band as noise). The coarse
  DMA% is NOT a decision metric here (it swung 1–100% on identical kernels) — decide on
  **TRUE PE-active (ms) + p50 latency + matmul_instruction_count**, not PE%/DMA%.
- **Screen then confirm:** `--fast` (seed 42) + `dump_metrics` to screen a B_BLK/ping-pong
  variant; promote only on a **full 5-seed** run (drop `--fast`) with an out-of-band p50
  win. A candidate whose TRUE PE-active is byte-identical to bmm_v2 is a compiler no-op →
  reject immediately (like the phase-2 multi-bank family), do not chase its latency noise.
- **Correctness invariant:** every promoted candidate is a pure reschedule (same 8704 instr,
  single-pass K=64, transpose-before-use is exact) ⇒ rel-L2 must stay 1.83e-7; any drift
  means an indexing bug. Keep bmm_v2 (and bmm_v1) as fallbacks.
- **Bookkeeping:** append each perf change to `benchmark.csv`; each candidate to
  `candidates.jsonl` with parent links (DAG); evidence under `profile/`. Kernels in `runs/`;
  never edit the baseline/reference.

## 6. Expected trajectory
`bmm_v2 1.253x → D1 multi-batch blocking (or D2 cross-batch double-buffer) ~1.34–1.38x` if
the batch-boundary bubble is real and not already compiler-pipelined; **hard fp32 ceiling
1.437x** (the 1.775 ms PE floor). Realistic promote target **~1.30–1.35x**. If both D1 and
D2 come back byte-identical (compiler already crosses the batch loop), the honest phase-3
conclusion is that bmm_v2 is within 13% of the fp32 PE floor with no remaining schedulable
structure — record that as the terminal result and keep bmm_v2. The whole phase is one
lever (batch-axis stream depth) tested two ways, gated hard on TRUE PE-active moving.
