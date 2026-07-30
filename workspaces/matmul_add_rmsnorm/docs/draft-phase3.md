# matmul_add_rmsnorm — Phase 3 draft (regime / shape specialization)

## 0. TL;DR

The promoted `matmul_add_rmsnorm_v2_bf16_split` is **4.879x (0.7722 ms)**, full-5-seed
PASS, rel-L2 1.544749e-5 (1.30x under the 2e-5 gate). Phase 3 is regime/shape
specialization. Three things are true here at once:

1. **Every classic *shape* lever is closed identically to the siblings**
   (`add_rmsnorm_matmul`, `rmsnorm_matmul`) — same fixed contract, all dims divide
   evenly (M=4096=32·128, K=2048=16·128, N=2048=4·512): no edge tiles, `nc_matmul`
   forces the k-on-partition layout, N_CHUNK=512=`psum_fmax` is maximal, `w` limbs are
   fully resident so M-blocking is vacuous, LNC2 is out of the single-core contract.
   Documented once by reference in §2 / a short closure doc, mirroring the sibling's
   AC-6 closure.

2. **The sibling's *headline* phase-3 micro-lever is already spent here — measured, not
   inherited.** On `add_rmsnorm_matmul` the phase-3 primary was "split the transposed
   activation limbs directly from the transpose PSUM bank, dropping the intermediate
   fp32 `xT_f` copy" (its D1). That exact transform was **already built and measured in
   *this* task during phase 2** as `runs/matmul_add_rmsnorm_v2_psum_split.py`: a
   **byte-identical compiler no-op** (matmul/Vec/Scl/psum instruction counts all `==`,
   TRUE PE-active 0.7078→0.7079 ms, rel-L2 bit-exact 1.544749e-5, +0.08% latency within
   noise). neuronx-cc already copy-propagates the exact fp32 PSUM→SBUF copy. That
   surface is **closed by this op's own measurement**, so phase 3 does not re-litigate
   it.

3. **The one genuinely-untested lever is a GEMM loop reorder for stationary
   (weight-load) reuse.** The promoted v2 sits at **PE=91.66% (~64 µs idle of the
   772 µs wall)**. In v2 the GEMM is **N-chunk-outer**, so each transposed activation
   limb — `xT_hi[kt]` / `xT_lo[kt]`, which is the *stationary* operand loaded into the
   PE array (see §1.2) — is reloaded **once per N-chunk = 4× per M-tile**. Reordering to
   **K-tile-outer with 4 live PSUM banks**, grouping the matmuls by shared stationary
   limb, lengthens the stationary-reuse run from **2 → 8 consecutive matmuls** and cuts
   stationary loads **128 → 32 per M-tile**. This is the only lever that could touch the
   ~64 µs PE-idle without changing the arithmetic. It is new to this task (the sibling
   never tested it) and is *enabled by this op's larger K* (16 K-tiles, 2× the sibling's
   8), which doubles the accumulation depth over which reuse can be grouped.

**Expected outcome, stated honestly.** The prior is that this reorder is a **compiler
no-op or a small regression**, on two measured precedents: (a) `v2_psum_split` above
(this task) and (b) `bmm`'s phase-3 finding that multi-bank PSUM pipelining is a
compiler no-op *and* enlarging the live PSUM/resident working set **regresses**
monotonically as it constrains the `affine_range` software pipeline
(`cross-batch-blocking-antilever`). Holding 4 live [128,512] accumulators is the same
"enlarged live set" risk. But the reuse *structure* here is different from both (it
changes the stationary-reuse run length, which the compiler cannot manufacture from the
N-chunk-outer source without reordering across the whole chunk loop), and PE=91.66%
leaves a small but real idle to probe. So phase 3 **measures one reorder candidate** and
promotes it only on an out-of-noise win + full-5-seed PASS; otherwise it is recorded as
a floor-confirmation and v2 stays promoted. The realistic ceiling if *all* 64 µs idle
were recovered is ~0.708 ms → **~5.32x**; the honest expectation is at or near v2.

**No precision change is on the table.** The rel-L2 margin is 1.30x, the 4-product v2b
was already a decision-SKIP (offline moves rel-L2 only 4.454e-6→3.491e-6 for ~+25% PE;
sibling v3b MEASURED-REJECT +28%), and plain bf16 fails the gate 117×. Phase 3 optimizes
the *schedule around the fixed 3-product arithmetic*, not the arithmetic (D5 forbidden).

---

## 1. Starting point — the promoted kernel and its profile

### 1.1 Kernel and measured profile

`runs/matmul_add_rmsnorm_v2_bf16_split.py` (PROMOTED, 4.879x, 0.7722 ms, full-5-seed
PASS rel-L2 1.544749e-5). Structure per M-tile (32 tiles), all fp32 I/O:

1. transpose + limb-split of the 16 RAW-`x` K-sub-tiles: per sub-tile — identity
   `nc_matmul(is_transpose)` → `psum_t` fp32; `xT_f = copy(psum_t)`; `xT_hi = bf16(xT_f)`;
   `xT_res = xT_f − xT_hi`; `xT_lo = bf16(xT_res)`;
2. GEMM (N-chunk-outer): `for c in 4: acc=zeros[128,512] psum; for kt in 16: acc += 3
   products (xT_hi@w_hi + xT_hi@w_lo + xT_lo@w_hi)`; then `y[:,chunk] = acc + z_tile`
   (residual add before norm) into a full [128,2048] `y` SBUF buffer;
3. fused fp32 RMSNorm over N (`square` → full-2048 `tensor_reduce(axis=[1])` → two-op
   `mean_eps = sumsq·(1/N)+eps` → `rsqrt` → `inv_rms[128,1]`);
4. output scale full-width: `out = (y·inv_rms)·g_bcast` (2 ops + 1 store).

`w_hi`/`w_lo` (split once at load) are fully resident bf16 (128 KB/part total = same
bytes as v1's one fp32 w). HBM unchanged from v1 (84 MB read / 34 MB write) — limbs
built on-chip.

**Profiler digest (promoted v2, same-session control against v1):**

| metric | v1 fp32 | **v2 bf16x2 (PROMOTED)** | reading |
|---|---|---|---|
| p50 latency | 0.9608 ms | **0.7722 ms** | −19.6% |
| speedup | 3.920x | **4.879x** | |
| **PE %** | 96.19 | **91.66** | v2 slightly idle — the phase-3 surface |
| MFU % | 45.57 | 42.55 | bf16 rate |
| Vec % | 14.88 | 26.62 | limb subtracts (hidden) |
| Scl % | 9.07 | 13.24 | bf16 casts (hidden) |
| DMA % | 19.38 | 23.93 | hidden; HBM flat |
| HBMrd / HBMwr | 84 / 34 MB | **84 / 34 MB** | one-pass floor, IDENTICAL |
| matmul_instruction_count | 4616 | 6664 | 512 transp + 2048·2(fp32) → 512 + 2048·3(bf16) |
| vector_engine_instruction_count | 400 | 566 | limb subtracts |
| scalar_engine_instruction_count | 225 | 246 | bf16 casts |
| **TRUE PE-active/inf** | 0.9242 ms | **0.7078 ms** | −23.4% — the real win (phase 2) |
| psum_read_sbuf_write_count | 132 | 132 | PSUM pressure unchanged |

### 1.2 The PE-idle read (the whole phase-3 argument)

TRUE PE-active is **0.7078 ms** of the **0.7722 ms** wall → **~64 µs PE-idle (8.3%)**.
PE-active is the fixed floor: `2·M·N·K` at the bf16 systolic rate, run 3× for the
3-product compensated split. **Cutting PE-active further needs either fewer products
(fails the gate) or a lower-precision matmul (fails the gate)** — closed. Every non-PE
engine is well under 50% and HBM is at the one-pass floor, so the only latency left to
chase is the **64 µs PE-idle**, and the only precision-neutral way to chase it is to
schedule the *same* matmuls so the PE array stalls less.

**Which operand is the "weight load".** `nc_matmul(stationary, moving) = stationary.T @
moving`, contraction on the partition axis of both. The tile shapes force the roles: the
stationary operand's free dim must be ≤128 and the moving operand's ≤512. Here
`xT_hi[kt]`/`xT_lo[kt]` are [k_in=128, m_in=128] (free=128 → **stationary**, loaded into
the array) and `w_hi[kt,c]`/`w_lo[kt,c]` are [k_in=128, n=512] (free=512 → **moving**,
streamed). So the *activation transpose is the stationary/weight-loaded operand* and the
weight `w` streams. A stationary load costs ~`num_partitions`≈128 array cycles; it is
**skippable when consecutive matmuls reuse the same stationary**. The cost model
(`kernel-cost-analysis`, Formula A) charges Matmul as `dst_free·100/freq` and
does **not** bill the stationary load separately — i.e. it *assumes* the load pipelines
behind the previous matmul's moving stream. Whether that assumption holds at v2's reuse
pattern is exactly what D1 measures.

---

## 2. Shape-lever closure (identical to the siblings)

| Lever | Applies? | Reason (this op's fixed shape M4096 N2048 K2048) |
|---|---|---|
| **Edge / partial tiles** | **No — vacuous** | M=32·128, K=16·128, N=4·512 all divide evenly. No ragged tile, no remainder loop, no mask anywhere. Edge specialization needs an edge; there is none. |
| **Tile-size / partition-free regime** | **No — layout forced** | `nc_matmul` needs k_in on the partition axis of both operands and produces `[m_in(par), n(free)]`; m_in is forced onto the stationary/partition side, n onto the moving/free side. Swapping m↔n would require transposing the N=2048-wide result back — far larger than any tiling gain. Also the RMSNorm reduces over N; keeping N on the free axis makes it a cheap in-partition `tensor_reduce`. Forced. |
| **N-chunk (moving-free) width** | **No — already maximal** | N_CHUNK=512 = `psum_fmax` = one fp32 PSUM bank in the free dim (knowledgebase `6288aaad`: "tile budget is psum_fmax"). Larger exceeds a bank; smaller wastes systolic streaming width. Pinned. |
| **M-blocking** (the `matmul` task's phase-2 win) | **No — vacuous** | That win removed *redundant w HBM reloads* by reusing a loaded w-tile across output-row tiles. Here `w_hi`/`w_lo` are fully resident (128 KB/part, budget ~208 KB) loaded **once** before the M-loop; each `x`/`z` tile is read exactly once (HBMrd=84 MB = the x+w+z one-pass floor). There is no redundant HBM traffic to block for. (D1 below is *not* M-blocking — it does not touch HBM; it reschedules on-chip PSUM/stationary reuse.) |
| **LNC2 / multi-core sharding** | **No — out of contract** | Scored single-core (`--logical-nc-config=1`). Using LNC2 would change the scoring contract, not optimize within it. |

**K note vs the siblings.** This op's K=2048 (16 K-tiles) is 2× `add_rmsnorm_matmul`'s
K=1024 (8 K-tiles) — the only shape difference that could make a *scheduling* lever
behave differently here. It doubles the K-accumulation depth per PSUM bank and doubles
the number of distinct stationary limbs (32 vs 16 per M-tile), which is precisely what
gives D1 (§3) a longer reuse run to group. Everything else closes identically.

---

## 3. Directions enumerated, ranked

### D1 — GEMM loop reorder for stationary (weight-load) reuse  *(PRIMARY; measure)*

**Idea.** v2's GEMM is **N-chunk-outer**: for each of the 4 N-chunks it fully
accumulates all 16 K-tiles' 3 products into one [128,512] PSUM bank before moving to the
next chunk. Consequence: each stationary limb `xT_hi[kt]` is loaded into the array for
chunk 0, and **loaded again** for chunks 1/2/3 — 4 loads per limb per M-tile. Within a
chunk the reuse run is only 2 (P1→P2 share `xT_hi[kt]`, then P3 changes to `xT_lo[kt]`,
then the next kt changes again).

Reorder to **K-tile-outer with 4 live PSUM accumulators**, grouped by stationary limb:

```python
acc = [zeros[128,512] psum  for c in 4]          # 4 live banks, accumulate across all kt
for kt in 16:
    # hi-pass: xT_hi[kt] STATIONARY for 8 consecutive matmuls (P1,P2 over 4 chunks)
    for c in 4:
        acc[c] += xT_hi[kt] @ w_hi[kt,c]
        acc[c] += xT_hi[kt] @ w_lo[kt,c]
    # lo-pass: xT_lo[kt] STATIONARY for 4 consecutive matmuls (P3 over 4 chunks)
    for c in 4:
        acc[c] += xT_lo[kt] @ w_hi[kt,c]
for c in 4: y[:,chunk_c] = acc[c] + z_tile[c]     # residual add, then the fp32 norm epilogue (v2, unchanged)
```

Now `xT_hi[kt]` is stationary across **8** consecutive matmuls and `xT_lo[kt]` across
**4** — reuse runs of 8 and 4 instead of 2. Stationary loads drop from 128 → **32 per
M-tile** (16 kt × 2 limbs). If the array's weight-load is not fully hidden at v2's
reuse pattern, this recovers part of the 64 µs idle.

**PSUM feasibility.** 4 live [128,512] accumulators = 4 banks (512 ≤ 2048 elem/bank) +
1 bank for the [128,128] transpose = **5 of 8 banks**. Fits. `psum_read_sbuf_write_count`
should stay ~132 (same number of evictions; they just happen after the kt loop).

**Correctness (NOT bit-exact — re-gate required).** The 3 products and the RNE
split are unchanged, so the bf16 error is the same *class*. But the fp32 PSUM
accumulation **order** changes (hi-pass P1,P2 for all kt, then lo-pass P3 for all kt, vs
v2's per-kt P1,P2,P3 interleave). fp32 add is non-associative, so rel-L2 will move by
~ulp-level, expected to stay ≈1.5447e-5. **This re-opens the correctness gate** — must
run full-5-seed and confirm rel-L2 ≈ 1.54e-5 and PASS (unlike the sibling's D1 / this
op's v2_psum_split, which were bit-exact). If rel-L2 jumps materially, that is a real
scheduling/aliasing bug to investigate, not a precision tradeoff.

**Expected latency — measure; prior is no-op/small-regress.** Two measured precedents
say the compiler may already extract this and/or the enlarged live set may hurt:
- `v2_psum_split` (this task, phase 2): a source reschedule that the compiler had
  already applied → byte-identical no-op.
- `bmm` phase 3: multi-bank PSUM "issue-before-drain" pipelining was a **compiler
  no-op** (`affine_range` already pipelines the rotating bank), *and* enlarging the live
  resident/PSUM working set **regressed monotonically** (the
  `cross-batch-blocking-antilever` lesson) because it constrains the software pipeline.
  Holding 4 live accumulators across the full 16-tile K-loop is the same enlarged-live-set
  risk — here it may cost more than the stationary-reuse saves.
The reason it is still worth **one** datum: the reuse-run-length change (2→8) is a
structural property of the source loop order the compiler is unlikely to synthesize from
the N-chunk-outer form (it would have to reorder across the entire chunk loop), and this
op's 16 K-tiles give the longest reuse run in the family. Realistic expectation: within
noise of v2, or a small regress; optimistic ceiling ~5.0–5.3x if idle is genuinely
weight-load-bound. **Promote iff full-5-seed PASS AND p50 beats v2 out-of-noise (>1.8%
band).** Otherwise record the floor-confirmation; v2 stays promoted.

**Risk:** low-moderate. No new primitive, no dtype/algebra change; only the loop
structure and PSUM-bank count change. The one real risk (regression from 4 live banks)
is itself an informative measured datum.

### D2 — 2-bank / 2-chunk grouping variant  *(secondary; only if D1's profile points here)*

If D1 shows the 4-live-bank version regressed *specifically* on PSUM/pipeline pressure
(PE-idle up, not down) rather than the reorder being a pure no-op, try the intermediate:
group by stationary over **2 chunks at a time** (2 live banks + transpose = 3 banks),
i.e. an outer 2-iteration chunk-pair loop with kt-outer inside. This keeps a reuse run of
4 (hi-pass over 2 chunks) while halving the live PSUM set — testing whether the
regression is the enlarged live set (D2 recovers) or the reorder itself (D2 also no-op).
**Contingent, not proactive** — build only if D1's digest shows a specific PSUM/pipeline
bubble; otherwise skip and record why (mirrors the sibling's contingent-D2 discipline).

### D3 — PSUM-source activation-limb split  *(CLOSED — already measured in phase 2)*

The sibling's phase-3 primary (split `xT_hi`/`xT_lo` directly from the transpose PSUM
bank, dropping the fp32 `xT_f` copy). **Already built and measured here as
`runs/matmul_add_rmsnorm_v2_psum_split.py`** during phase 2: byte-identical compiler
no-op (all instruction counts `==`, TRUE PE-active 0.7078→0.7079 ms, rel-L2 bit-exact,
+0.08% within noise). neuronx-cc copy-propagates the exact fp32 PSUM→SBUF copy. **Do not
rebuild** — cite the phase-2 datum.

### D4 — off-PE transpose to remove the 512 transpose matmuls  *(CLOSED — record-only)*

Both siblings closed this: SBUF→SBUF `dma_transpose` of a [128,128] tile is infeasible
(hwdge needs `src.shape[0]==16`, swdge needs an HBM source — shape/memory block, not just
dtype), and `nc_transpose`(vector) lands in fp32 PSUM needing a re-cast and measured a
+2% regress. The 512 identity-matmul transposes (~27 µs of PE-active) are already hidden
under the PE-bound matmul. **Do not explore.**

### D5 — precision / product-count changes  *(FORBIDDEN this phase)*

3-product bf16 is pinned: margin 1.30x, v2b 4-product was a decision-SKIP (sibling v3b
MEASURED-REJECT +28% for a ~1.6% accuracy move swamped by the fp32 floor), plain bf16
fails 117×. Every phase-3 candidate keeps the exact 3-product arithmetic.

### D6 — split-before-transpose (wide limb ops)  *(CLOSED — record-only)*

Splitting `x` into limbs *before* transpose (3 wide [128,2048] ops instead of granular
per-sub-tile ops) **doubles the transpose PE work** (transpose `x_hi` and `x_lo`
separately → 32 transpose matmuls/M-tile instead of 16). Adds PE-active — the one thing
we cannot afford. Phase 2 already fixed "split *after* the transpose costs one transpose,
not two." **Do not implement.**

---

## 4. Execution plan (≤2 candidates; measure-first)

1. **Re-anchor (same-session control).** Re-measure v2 full-5-seed via
   `runs/dump_metrics.py` to pin this session's PE-active / PE% / idle and latency
   anchor (profiler jitter ~±1.8%). Confirm PE≈91.7%, HBM 84/34 MB, rel-L2 1.5447e-5.
2. **v3 (D1, stationary-reuse reorder):** `runs/matmul_add_rmsnorm_v3_stationary_reorder.py`,
   forked from v2. Change *only* the GEMM loop nest (K-tile-outer, 4 live PSUM banks,
   hi-pass/lo-pass grouping); epilogue (residual add + fp32 RMSNorm + full-width output
   scale) byte-for-byte v2. `--fast` PASS gate first, then **full 5-seed** twice for
   stability + the same-session v2 anchor. Capture the profiler digest (watch PE% and
   TRUE PE-active, and `psum_read_sbuf_write_count`).
   - **Promote iff** full-5-seed PASS AND p50 beats v2 out-of-noise (>1.8%).
   - **Otherwise** record the within-noise floor-confirmation (or the regression as a
     first-class negative datum, per `bmm`'s anti-lever discipline); v2 stays promoted.
3. **v4 (D2) only if D1's profile shows a specific PSUM/pipeline bubble.** Contingent,
   same gates. If D1 is a clean no-op, skip D2 and record why (proactive D2 forbidden).
4. **Close-out:** whichever of v2/v3(/v4) is fastest is the phase-3 (and task) result.
   Report speedup vs the 3.768493 ms baseline on the **full** 5-seed correctness gate.

## 5. Evidence to record

- `benchmark.csv`: one row per perf-relevant candidate (v3, and v4 if run), plus the
  D3(closed-in-phase-2)/D4/D5/D6 decisions as record-only notes.
- `candidates.jsonl`: DAG node v3→v2 (and v4→v3 if run) with metrics, `rel_l2`,
  `per_seed_rel_l2`, the non-bit-exact-reorder note, and the `per_seed_latency_ms=null`
  / `latency_scope` caveat carried from v2.
- `profile/`: v3 digest with the PE-idle before/after read (TRUE PE-active,
  `psum_read_sbuf_write_count`), and a short shape-closure note (or a reference to §2 /
  the sibling's closure doc) for AC-6.

## 6. Correctness invariants (never regress)

- fp32 residual add `y = x@w + z` **before** the norm; fp32 RMSNorm reduction over N
  (`square` / full-2048 `tensor_reduce(axis=[1])` / `mean_eps` / `rsqrt`); **eps added
  AFTER the `/N` mean**, matching the reference `np.mean(y**2,axis=-1)+eps`.
- `g` is on the OUTPUT free axis (N) → `[1,N]→[128,N]` broadcast multiply applied
  **after** the norm; **never folded into w** (folding would scale y before the norm and
  break `rms=sqrt(mean(y^2))`). `inv_rms` is **not** commuted out (norm reduces over N →
  the full [128,N] row must be assembled before `inv_rms` is known).
- The 3-product bf16 split and its PINNED split order (`w`→`w_hi`→`w_res`→`w_lo`;
  `xT`→`xT_hi`→`xT_res`→`xT_lo`; products `hi@hi + hi@lo + lo@hi`, drop `lo@lo`) are
  unchanged. D1 changes only the *order* of fp32 PSUM accumulation, so rel-L2 must stay
  ≈1.5447e-5 (not bit-exact; re-gate full-5-seed).
- Raw-2D I/O + exact signature `kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor)`;
  full-width [128,N] output store.
- Every promotion gated on **full 5-seed** `l2_norm_passed`, not `--fast` alone.
- **CAVEAT (carried from phase 2):** the adapter fixes seed 42 for all 5 profiler seeds,
  so on-device 5-seed PASS is a determinism/stability gate, weak on *input* diversity;
  the offline 7-draw sim (worst bf16-only 4.454e-6) covers input diversity. `v1`
  (3.920x, pure fp32) is retained as the guaranteed-correct fp32 fallback.
