# bmm_softmax — Phase 3 draft: regime / shape specialization

**Operator:** `bmm_softmax` (NKIBench case 2). `out[b] = softmax_N(lhs[b] @ rhs[b])`,
`b in 0..15`. `lhs (16,4096,64)=(B,M,K)`, `rhs (16,64,4096)=(B,K,N)` fp32 →
`out (16,4096,4096)`. **B=16, M=4096, K=64, N=4096.** Baseline **7.290 ms**.

**Start point:** `runs/bmm_softmax_v4.py` = the phase-2 promotion, **1.946x (3.7460 ms)**,
full 5-seed L2 PASS, on-device rel-L2 **2.5683307869e-6** (~7.8x under the 2e-5 gate).
Structure: sibling `bmm_v2` two-phase transpose-all schedule + max-negate fold + `M_SUB=16`
within-batch depth. Pure fp32; `bmm_softmax_v1` (1.585x) retained as the fp32 fallback / DAG root.

---

## 1. The phase-3 question, answered honestly up front

Phase 3 asks: *analyze where time goes across the tensor's structure and specialize only where
the measured win justifies the complexity (tile-size regimes, partition/free splits, edge tiles).*
As in the solved sibling `bmm`, the honest answer is that **classic shape specialization has no
surface here**, and I want that on the record with numbers so the phase does not chase a dead lever:

- **No edge tiles.** Every axis divides cleanly: `M=4096=32·128`, `N=4096=8·512`, `K=64≤128`
  (single Tensor-Engine pass), `B=16`. There is no ragged remainder to special-case — the usual
  "specialize the edge" regime split does not exist.
- **Tiles are already maximal and cannot be widened.** The main matmul is `[K=64]×[64,512]→[128,512]`.
  The moving free dim 512 is the **hard PSUM-bank wall on trn2** (one `nc_matmul` writes one bank =
  512 fp32 elems/partition; 2048/4096-wide is trn3-only). The stationary free dim 128 fills the PE
  columns. No "bigger tile" regime to switch into. The softmax reduces over the **full N=4096 row**,
  which already lives in one resident SBUF tile — no N-tiling regime either.
- **The K=64 partition/free split is fixed and cheap.** K=64 fills 64 of 128 PE partition rows, but
  the trn2 matmul cost is `dst_free_elems·100/freq` — proportional to the moving free dim (512) ONLY,
  independent of K. A half-full contraction costs nothing extra; there is no partition-split regime
  that recovers it. K cannot be packed across batches (`out[b]` are block-diagonal — closed in `bmm`).

So the phase-3 "structure" to analyze is **not tile shape** — it is the **engine schedule of the
fused softmax epilogue relative to the matmul stream**, the one structural dimension phase 2 stopped
short of. Phase 2 tuned *within-batch stream depth* (`M_SUB`); phase 3 attacks the *engine placement*
of the epilogue that is inflating the Tensor Engine.

## 2. The corrected bottleneck — softmax back-pressures the matmul (the reframe that drives the phase)

Phase-2 exit called the kernel "PE-bound, pure-fp32 schedule levers exhausted." That is true but
under-specified. The authoritative same-session counters (`profile/bmm_softmax_v4_digest.txt`,
divided by the metric window) versus the pure-`bmm` sibling that runs the **byte-identical matmul**:

| signal | bmm_softmax_v4 (fused) | pure bmm_v2 (same 8704 matmuls) | note |
|---|---|---|---|
| p50 wall | **3.746 ms** | 2.036 ms | |
| TRUE PE-active / inf | **3.371 ms** (89.99%) | **2.011 ms** (98.9%) | **identical matmul, +1.36 ms** |
| per-matmul PE-active | **0.387 µs** | 0.231 µs | **+0.156 µs (+67%)** |
| Vec-active / inf | 2.687 ms (71.74%) | ~0.9 ms (20%) | +1.8 ms of softmax Vec |
| Scalar-active / inf | 2.627 ms (70.14%) | ~0.6 ms (14%) | +2.0 ms of softmax Scalar |
| GpSimd-active / inf | 0.190 ms (5.06%) | — | **nearly idle** |
| DMA-active / inf | 1.400 ms (37.38%) | 1.43 ms | hidden, at floor |
| matmul_instruction_count | 8704 | 8704 | **identical work** |
| psum_read_sbuf_write_count | 4224 | 4224 | identical |
| HBM read / write | 33.6 / 1073.7 MB | 34 / 1074 MB | at read-once/write-once floor, spill=0 |

**The single fact that drives phase 3:** the matmul workload is *byte-identical* to pure `bmm`
(8704 instr, same tiles), yet its **PE-active is inflated from 2.011 ms → 3.371 ms (+68%)**. The
matmul is not doing more work — it is **stalling**. And the exposed tail is only
`wall − PE-active = 3.746 − 3.371 = 0.375 ms`. So the prize is **not** the 0.375 ms exposed tail —
it is the **1.36 ms of PE inflation**, if it can be relieved.

### Where the inflation comes from — the PSUM-drain / Vector-contention chain

The schedule is: per m-subtile, 8 matmuls each land a `[128,512]` result in **one of the 8 PSUM
banks**, then each bank is copied out to the resident `score[128,4096]` SBUF tile (the
`psum_read_sbuf_write` copies), then the softmax runs on `score`. The copies exist **structurally**:
`score` must reach SBUF so the 8 PSUM banks free up for the *next* subtile's 8 matmuls — there is no
PSUM double-buffer (only 8 banks, all in use), and `score` cannot stay in PSUM because `exp` needs it
*after* the full-row `max`, by which time the banks are needed again. (Copy-elimination — reduce/exp
directly from PSUM — was considered and rejected on paper: it holds all 8 banks alive through the
expensive 4096-wide `max`+`exp`, which *lengthens* the bank-free critical path and *worsens*
pipelining. The copies are the fast drain, not waste.)

The chain that inflates PE: the copies and the softmax reductions (`max`, `sum`) both want the
**Vector engine**; the `exp` wants the **Scalar engine**; both are at ~70%. When a bank's copy is
queued behind softmax Vector/Scalar work, that PSUM bank stays occupied, so the next matmul that
targets it **stalls** — and the profiler counts the stall as PE-active. In pure `bmm`, Vector sits at
~20% idle so the copies drain instantly and per-matmul PE-active is 0.231 µs; here the loaded
Vec/Scalar engines throttle the drain and per-matmul PE-active rises to 0.387 µs. **The lever is to
drain the PSUM banks faster by spreading the epilogue across engines so the matmul stops waiting.**

This is a *real, mechanistic* contention that pure `bmm` lacked — `bmm`'s phase-3 PE bubble (13%
above floor) had no contention source (Vec idle), which is exactly why `bmm`'s reschedules were
no-ops. Here Vec+Scalar are both hot with a drain that competes with the matmul, so there is a
genuine mechanism to attack. (The no-op risk still applies — see the kill-criterion — but the
premise is stronger than `bmm`'s was.)

### Theoretical floor (for calibration, not a hard prediction)

Pure-fp32 PE floor (from `bmm`, free-dim-only cost model, 2.4 GHz): 8192 main passes @ 512-wide
(213.3 ns) + 512 transposes @ 128-wide (53.3 ns) = **1.775 ms**. The softmax `exp` is Scalar-only and
irreducible at ~512×4096-wide ≈ 1.75 ms; the two reductions (`max`,`sum`) are **Vector-only**
(`tensor_reduce` has no `engine=` arg — API-confirmed) and also irreducible. So the wall floor is
`max(PE, Vector-mandatory, Scalar-mandatory)` once the *movable* work (copies, normalize) is packed
into the gaps. **Caveat:** the per-op cost-model numbers do **not** cleanly reconcile with v4's
measured instruction counts (the compiler lowers/places the 4096-wide ops and the copies in ways the
simple model doesn't capture), so I will **not** hard-predict a floor — I will *measure* the current
engine placement in round 0 and treat the arithmetic as directional only. Best realistic PE target =
the pure-`bmm` hidden floor **2.011 ms** (softmax fully overlapped, zero back-pressure); if reached
with today's 0.375 ms tail, wall ≈ 2.4 ms → **~3.0x**. That is the optimistic ceiling, not a promise.

## 3. Round 0 — measurements before any code change (near-zero remote risk)

Re-use `runs/dump_metrics.py` (reads TRUE `tensor_engine_active_time_ns` + per-engine actives +
instruction counts, not the jittery PE%/DMA% proxy). All same-session vs a fresh v4 anchor.

1. **Re-anchor v4 counters** — confirm TRUE PE-active ≈ 3.371 ms, per-matmul PE-active ≈ 0.387 µs,
   matmul_instr 8704, HBM 33.6/1073.7 MB (spill 0). This is the fact the phase rests on; verify fresh.
2. **Attribute the current engine placement of the epilogue.** The compiler already chose engines for
   the 8 score copies and the normalize (v4's counts — Vec 3472 / Scalar 4628 instr — suggest some
   copies are *already* off Vector). Establish *which engine each epilogue op currently lands on*
   before touching it, so a rebalance is measured against the real starting placement, not an assumed
   one. (No new probe if the counts + a one-shot engine-annotated dump settle it.)
3. **(record-only) confirm the precision lever stays closed.** Phase-2/`bmm` measured fp32/bf16 pass
   ratio 2.0, so a 3-product bf16x2 main matmul costs 3.0 passes > 2.0 and *raises* PE on a PE-bound
   kernel; bf16 `exp`/softmax over N=4096 ≈ 1e-2 rel error, blowing the 2e-5 gate. Both closed; state
   it so the phase does not relitigate.

## 4. Optimization directions, ranked by expected benefit × confidence

### D1 — Epilogue engine rebalancing to relieve PSUM-drain back-pressure (PRIMARY)

**Hypothesis.** Per subtile, 8 `[128,512]` PSUM→SBUF score copies must drain before the next 8
matmuls reuse the banks. If those copies serialize on one loaded engine, the drain gates the matmul
and inflates PE-active. Spreading the drain (and the normalize) across Vector **and** Scalar frees the
banks faster → per-matmul PE-active drops toward the 0.231 µs pure-`bmm` floor.

**Feasibility (API-confirmed, `/nki-api-reference`):**
- `nisa.tensor_copy(dst, src, engine=nki.isa.engine.scalar)` moves a PSUM→SBUF copy to the Scalar
  engine (GpSimd is **not** allowed — it cannot read PSUM). Bit-exact fp32 copy on trn2 Scalar.
- The normalize can run on Scalar via `nisa.activation(op=nl.copy, data=exp_t, scale=recip[P,1])`
  (per-partition `[P,1]` scale is legal), or stay on Vector as `tensor_scalar(op0=multiply,
  operand0=recip)`.
- Direct precedents: `fd27f7ef` (`attention_tkg`) moved a hot `tensor_copy` to ScalarE to relieve
  VectorE; `63e18e33` (`attention_cte`) moved a normalize the *other* way (Scalar→Vector) because
  ScalarE was the bottleneck; `597cf19e` (`mlp_tkg`) *alternates* engines per iteration to consume
  two engines' bandwidth. → the choice is **profile-driven balancing**, not a fixed direction, which
  is why D1 is a small sweep, not a single edit.

**Variants to sweep (all bit-exact — pure engine reassignment, same fp32 math):**
- **A — split the 8 score copies 4 Vec / 4 Scalar** (alternate by chunk index, per `597cf19e`),
  normalize on Scalar. Halves the *serial* drain latency across two engines while keeping the two
  mandatory reductions on Vector.
- **B — all 8 copies on Scalar, normalize on Vector.** Frees Vector for the reductions.
- **C — all 8 copies on Vector, normalize on Scalar.** The mirror.
- (Round 0 tells us the current placement; only sweep the variants that differ from it.)

**Decision metric:** promote the variant with the lowest **TRUE per-matmul PE-active** (out-of-noise),
with HBM staying at the 33.6/1073.7 MB floor and matmul_instr = 8704. Wall p50 is the tie-break.

**Expected:** if drain contention is the cause and a split halves it, per-matmul PE-active
0.387 → ~0.30 µs, PE-active 3.37 → ~2.6 ms, wall ~2.9 ms → **~2.5x**. Numbers are hypotheses; the
profile gates them.

**Kill-criterion (inherited from `bmm` phase 3).** The `affine_range` compiler already pipelines
aggressively and flattened many `bmm` reschedules to byte-identical no-ops; it may already have chosen
a near-optimal engine placement (round-0 step 2 will show how close). **Screen every variant with
`--fast` + `dump_metrics` first; reject immediately any variant whose TRUE PE-active is byte-identical
to v4 (compiler no-op) or rises (anti-lever), exactly as `bmm` rejected its cross-batch reschedules.**
≤3 iterations (the copy-split + at most two engine-assignment points).

### D2 — `M_SUB` re-sweep on the winning engine assignment (SECONDARY, contingent)

Phase 2 found the interior optimum `M_SUB=16` *given v4's engine placement*. A faster-draining
epilogue (D1) changes the matmul↔softmax overlap, so the optimal stream depth may shift — a deeper
stream (`M_SUB=32`) could become viable again once the drain no longer gates it. **Only if D1 lands**,
re-sweep `M_SUB ∈ {16, 32}` on the winning assignment (≤2 iterations, reuses D1's kernel, within one
batch only — never cross-batch). Low-medium value; a within-winner tie-break, not a phase driver.

## 5. Closed / not-pursued (record-only — do NOT spend iterations)

- **bf16x2 3-product matmul split.** fp32/bf16 pass ratio 2.0 < 3.0 ⇒ split *raises* PE on a PE-bound
  kernel. `[[BL-20260710-bf16x2-loses-when-fp32-emulates-in-2-passes]]`. Closed.
- **bf16 `exp`/softmax.** ~1e-2 rel error over N=4096 » the 2e-5 gate (current margin only 7.8x). Closed.
- **Cross-batch blocking / cross-batch double-buffer.** Measured **anti-lever** in `bmm` phase 3
  (per-matmul stall 0.231→0.296(B2)→0.332(B4) µs monotone regression — the batch boundary is a helpful
  `affine_range` pipeline reset). `[[BL-20260710-cross-batch-blocking-is-an-antilever-on-affine-range]]`.
  D1/D2 stay **within one batch**. Closed.
- **GpSimd normalize / GpSimd copies (recruiting the idle 5% engine).** **API-infeasible**, not just
  precondition-false: `tensor_scalar(engine=gpsimd)` is **rsqrt-only** (no general `[P,1]` multiply),
  and GpSimd **cannot access PSUM** (so it cannot do the score copies). This *upgrades* phase-2's
  "GpSimd precondition false" to a hard infeasibility — the idle GpSimd is genuinely unusable for this
  epilogue. Do not build; do not re-probe.
- **Fused `activation(reduce_op=add, reduce_res=)` exp+row-sum.** Phase-2 measured reject: the
  `reduce_res` accumulator side-effect triggered a **whole-stream 2× recompute** (matmul 8704→17408,
  +75% wall). `profile/bmm_softmax_d2_compare.md`. Closed — keep the explicit `tensor_reduce(add)`.
- **Removing the max-reduce or the normalize pass.** Both required (overflow-safe max-shift; softmax
  normalization) and already minimal. Keep.
- **Copy-elimination (reduce/exp directly from PSUM).** Rejected on paper (§2): holds all 8 banks
  through the 4096-wide `max`+`exp`, lengthening the bank-free critical path — worsens pipelining on a
  PE-bound kernel. Do not build.
- **Wider matmul tile / narrower N_CHUNK / K-packing / DMA store-burst / bf16 output.** All closed in
  `bmm` (PSUM-bank wall; block-diagonal; DMA hidden at floor; 2e-5 bans bf16 out). Not re-probed.

## 6. Correctness guardrails (never regress)

- fp32 throughout matmul and softmax; no bf16/tf32.
- Max-shifted softmax preserved: `exp(score − row_max)` then divide by the row sum, reduction over the
  **N free axis** (reference axis 2). Every D1/D2 candidate is a **pure engine reassignment / schedule
  change** — same set of fp32 exp terms summed in the same order ⇒ rel-L2 must stay **2.5683307869e-6**
  (bit-identical). Any drift = an indexing/placement bug, reject.
- No softmax reduce/activation/elementwise op on a PSUM tile that would hold banks (copies drain PSUM
  to SBUF immediately, as in v4).
- Every candidate: `--fast` (seed 42) pre-check + `dump_metrics`, then **full 5-seed** `verify.py`
  before any promotion; require `l2_norm_passed=True` on all seeds `[0,21,42,63,84]`.

## 7. Measurement protocol (per candidate)

From `workspaces/bmm_softmax/`:
```bash
python3 \
    ../../verify.py --op bmm_softmax --candidate runs/<file>.py --fast   # gate first
python3 \
    runs/dump_metrics.py runs/<file>.py --fast                            # engine/PE-active screen
# then drop --fast on both for the promotion measurement
```
Decide on **TRUE per-matmul PE-active (ms) + p50 latency**, NOT coarse PE%/DMA% (jitter 1–100% on
identical kernels in the siblings). Re-anchor v4 same-session before each comparison; treat a ~1.5–2%
band as noise. Capture the digest per direction; diff vs v4 on: per-matmul PE-active, per-engine
active ms, matmul_instr (must stay 8704), psum copies, HBM (must stay at floor), rel-L2 (must stay
2.5683e-6). Log every perf change in `benchmark.csv`; each candidate in `candidates.jsonl` with parent
links (DAG root `bmm_softmax_v1`, phase-3 base `bmm_softmax_v4`); evidence under `profile/`.

## 8. Expected trajectory & exit

- `bmm_softmax_v4 1.946x → D1 epilogue engine rebalance ~2.2–2.6x` **if** the PSUM-drain contention is
  real and not already compiler-balanced; optimistic ceiling ~3.0x (PE-active → the pure-`bmm`
  2.011 ms hidden floor with today's 0.375 ms tail). D2 `M_SUB` re-sweep is a within-winner tie-break.
- **If D1 comes back byte-identical** (compiler already placed the epilogue optimally, like `bmm`'s
  reschedules), the honest phase-3 conclusion is that v4 is at the fused kernel's engine-balanced
  optimum with no remaining schedulable structure — record that as terminal and keep v4. The whole
  phase is one mechanistic lever (relieve the softmax→PSUM-drain→matmul back-pressure by engine
  balancing), gated hard on TRUE per-matmul PE-active moving out of noise.
- **Promote** the best correct candidate; **keep `bmm_softmax_v1`** (fp32 fallback) and `bmm_softmax_v4`
  as fallbacks. Write `docs/phase3-exit-decision.md` with keep/revise/reject per direction and the
  before/after evidence, then update `[[kda-bmm-softmax-progress]]`. ≤5 optimization iterations
  (round-0 re-anchor + closed-lever records excluded from the budget).
