# add_rmsnorm_matmul — Phase 3 draft (regime / shape specialization)

## 0. TL;DR

The promoted `add_rmsnorm_matmul_v3_bf16_split` is **4.632x (0.4013 ms)**, full-5-seed
PASS, rel-L2 1.528e-5 (1.31x under the 2e-5 gate). Phase 3 is regime/shape
specialization. Two things are true here at once:

1. **Every classic *shape* lever is closed identically to the sibling** `rmsnorm_matmul`
   (same fixed M=4096, N=2048, K=1024, fp32 I/O). No edge tiles (all dims divide
   evenly), `nc_matmul` forces the k-on-partition layout, N_CHUNK=512=`psum_fmax` is
   maximal, `w'` is fully resident so M-blocking is vacuous, LNC2 is out of the
   single-core contract. This is documented once, by reference, in
   `docs/shape-specialization-closure-phase3.md` (mirrors the sibling's AC-6 closure).

2. **The one genuine phase-3 surface is op-specific: a PE-idle gap.** The promoted v3
   sits at **PE=89% (≈44 µs idle of the 401 µs wall)**, whereas the sibling's
   *identical* bf16-split kernel reached **PE=96% (≈15 µs idle)**. The two have nearly
   identical **PE-active (~355 µs — it is the same 3-product bf16 matmul)**, so the
   ~29 µs of *extra* idle is this op's extra per-M-tile non-PE work not fully
   overlapping the matmul:
   - residual add `a = x+z` (a [128,1024] Vec op the sibling does not have),
   - the extra `z` read (+16 MB HBM → DMA 24→29% vs sibling's 25 MB),
   - and the **granular per-sub-tile activation limb-split** — 8 sub-tiles × 4
     Vec/Scalar ops = **32 small [128,128] ops per M-tile**, each paying the Vector
     engine's fixed ~268 ns `semaphore_start` + 161 ns `write_drain` on only ~133 ns
     of compute.

The **primary lever** is a *bit-exact* simplification of that limb-split (D1): split
the `aT` limbs **directly from the transpose PSUM**, dropping v3's intermediate fp32
`aT_f` SBUF copy. This cuts the per-M-tile split from **32 → 24 Vec/Scalar ops** with
**zero numeric change** (rel-L2 stays 1.528e-5 exactly) and **zero extra PE work**. If
the exposed non-PE time is real, this recovers part of the 44→15 µs gap
(≈**4.7–5.0x**); if it is already hidden (the sibling's phase-3 measurements suggest
transpose/fills were hidden *at PE=96%*), the result is a within-noise
floor-confirmation and v3 stays promoted. Either way it is a safe, well-scoped move.

**No precision change is on the table.** The rel-L2 margin is only 1.31x, D3
(4-product) was already MEASURED-REJECT (+28% for a 1.6% accuracy move hidden under the
fp32 hardware floor), and going the other way (fewer products / plain bf16) fails the
gate. So phase 3 optimizes the *schedule around the fixed 3-product arithmetic*, not the
arithmetic.

---

## 1. Starting point — the promoted kernel and its profile

`runs/add_rmsnorm_matmul_v3_bf16_split.py` (PROMOTED, 4.632x, full-5-seed PASS
rel-L2 1.528e-5). Per M-tile (32 tiles):

1. load `x`,`z` → `a = x+z` ([128,1024] fp32 `tensor_tensor`),
2. fused fp32 RMSNorm reduction (`square` → full-1024 `tensor_reduce` → two-op
   `mean_eps = sumsq·(1/K)+eps` → `rsqrt` → `inv_rms[128,1]`),
3. **transpose + limb-split of the 8 RAW-`a` K-sub-tiles**: per sub-tile — identity
   `nc_matmul(is_transpose)` → `psum_t` fp32; `aT_f = copy(psum_t)` fp32 (SBUF);
   `aT_hi = bf16(aT_f)`; `aT_res = aT_f − aT_hi` fp32; `aT_lo = bf16(aT_res)`,
4. main matmul: **3 bf16 products** (`aT_hi@w'_hi + aT_hi@w'_lo + aT_lo@w'_hi`) × 8
   K-tiles, accumulated in a [128,512] fp32 PSUM, over 4 N-chunks,
5. `inv_rms` post-scale at PSUM→SBUF eviction (`tensor_scalar` reading the accumulator)
   → store.

`w'_hi`/`w'_lo` (g folded in, split once) are fully resident bf16 (64 KB/part). HBM is
unchanged from v1 (42 MB read / 34 MB write) — limbs are built on-chip.

**Profiler digest (promoted v3):** MFU=41% PE=89% Vec=22% Scl=24% DMA=29%
HBMrd=42MB HBMwr=34MB.

### The PE-idle read (the whole phase-3 argument)

| kernel | wall (µs) | PE% | PE-active (µs) | **PE-idle (µs)** | HBMrd |
|---|---|---|---|---|---|
| v1 (fp32 fused) | 495.3 | 94% | 465.6 | 29.7 | 42 MB |
| **v3 (bf16-split, PROMOTED)** | 401.3 | **89%** | **357.2** | **44.1 (11%)** | 42 MB |
| sibling v4 (bf16-split, identical matmul) | 368.8 | **96%** | 354.0 | **14.8 (4%)** | 25 MB |

- The two bf16-split kernels have **the same matmul** (3 bf16 products, same K=1024,
  N=2048) and land within ~1% on PE-active (~355 µs). That is the real, unmovable
  floor: `2·M·N·K` at the bf16 systolic rate, run 3× ≈ the measured PE-active. **Cutting
  PE-active further requires either fewer products (fails the gate) or a lower-precision
  matmul (fails the gate).** Closed.
- The **difference** between the two is **PE-idle: 44 µs here vs 15 µs on the sibling.**
  That 29 µs gap is not the matmul — it is this op's extra per-M-tile non-PE work
  becoming *exposed* once the matmul got fast (fp32→bf16 dropped PE occupancy 97→89%:
  the denominator shrank, so the same fixed non-PE overhead shows as more idle).
- **Cost-model view of why the limb-split is the suspect.** Per M-tile the activation
  split issues 32 small [128,128] Vec/Scalar ops. On trn2 a [128,128] copy is ~133 ns
  of compute but carries ~268 ns Vector `semaphore_start` + 161 ns `write_drain`. Even
  with heavy cross-engine/cross-iteration overlap (which the measured 89% already
  reflects), fixed-overhead-dominated small ops are exactly the kind of work that leaves
  bubbles the wide sibling (no add, no z, PE=96%) never had. The residual add and z-read
  are single wide ops / DMA and are far likelier already hidden.

Conclusion: the only latency left to chase is the **PE-idle gap**, and the cheapest,
safest way to chase it is to **remove non-PE ops without touching the arithmetic**.

---

## 2. Directions enumerated, ranked (benefit vs risk)

### D1 — split `aT` limbs directly from the transpose PSUM  *(PRIMARY; bit-exact; low risk)*

**Idea.** v3's split reads the intermediate fp32 SBUF copy `aT_f`:
```
psum_t = nc_matmul(a_sub, identity, is_transpose=True)   # fp32 PSUM
aT_f   = copy(psum_t)                # fp32 PSUM->SBUF   (v3 line 151)  <-- REMOVE
aT_hi  = bf16(aT_f)                                       # v3 line 153
aT_res = aT_f - aT_hi                # fp32 tensor_tensor  v3 line 156
aT_lo  = bf16(aT_res)                                     # v3 line 158
```
Because `aT_f` is an *exact* fp32 copy of `psum_t`, every downstream read can source
`psum_t` directly:
```
aT_hi  = bf16(psum_t)                # bf16(psum_t)  == bf16(aT_f)   bit-for-bit
aT_res = psum_t - aT_hi              # (psum_t-aT_hi) == (aT_f-aT_hi) bit-for-bit
aT_lo  = bf16(aT_res)
```
This is **4 → 3 ops per sub-tile (32 → 24 per M-tile)**, and it is **bit-identical** —
`aT_hi`/`aT_lo` are the same bf16 values, so the 3-product matmul consumes the same
limbs and rel-L2 is exactly 1.528e-5. Reading `tensor_tensor`/`copy` directly from PSUM
is already proven in this kernel (the `inv_rms` post-scale reads the PSUM accumulator at
line 179; the eviction path is the same shape).

**Cost.** −8 ops/M-tile × 32 M-tiles = 256 fewer [128,128] Vec/Scalar ops overall, each
fixed-overhead-dominated. Zero extra PE. The only tradeoff is that `psum_t` stays live
one op longer (until `aT_res` is computed) before its PSUM bank frees — negligible: the
transpose uses a [128,128] (=128 elem/part) bank and there are 8 PSUM banks of 2048, so
there is no PSUM-pressure conflict with the [128,512] main accumulator.

**Expected latency.** Uncertain by design — this must be **measured**. Optimistic:
recovers part of the 44→15 µs idle gap → ~0.38–0.385 ms → **~4.8–5.0x**. Pessimistic
(sibling precedent: transpose and stationary fills were *already hidden* at PE=96%):
within-noise, PE stays ~89%, v3 remains promoted as a floor-confirmation. Both are
acceptable phase-3 exits; the change is cheap and risk-free.

**Correctness.** Bit-exact (argued above). Still gate on full-5-seed on device — expect
rel-L2 to reproduce 1.528e-5 identically; if it does not, that is a real bug to
investigate, not a precision tradeoff.

**Risk:** very low. No new primitive, no dtype change, no algebra change.

### D2 — reduce residual-add / RMSNorm-reduction exposure  *(secondary; measure only if D1 leaves idle)*

Only pursue if D1's profile still shows a Vec/Scalar bubble. Candidates, all
numerically neutral:
- **Per-sub-tile add fold:** compute `a` for sub-tile `kt` just before transposing it,
  so M-tile `t`'s adds overlap `t`'s transposes and `t−1`'s matmul. v3 already relies on
  `affine_range` to pipeline across M-tiles, so this is likely a no-op — measure before
  believing it.
- **Square-from-add fusion:** the `square` activation could read `a` immediately; no
  structural change expected. Record-only unless D1's digest points here.

D2 is contingent and not a headline candidate; it exists so the phase doesn't stop at D1
if D1 surfaces a specific remaining bubble.

### D3 — split-before-transpose (wide limb ops)  *(CLOSED — record-only)*

Splitting `a`'s limbs as **wide [128,1024] ops before** transposing (3 wide ops instead
of 24 small ops) sounds attractive but **doubles the transpose PE work**: each of `a_hi`
and `a_lo` must be transposed separately → 16 transpose `nc_matmul`s and 16 PSUM→SBUF
copies per M-tile vs v3's 8. That adds PE-active (the one thing we cannot afford) to save
Vec ops. Phase-2 already fixed "split *after* the transpose … costs **one** transpose,
not two." D1 keeps that single-transpose property while still shrinking the op count.
**Do not implement.**

### D4 — off-PE transpose to remove transpose PE work  *(CLOSED — record-only)*

Re-checked for the bf16-limb world (the sibling closed it for fp32 only): even with bf16
tiles, an SBUF→SBUF `dma_transpose` of a [128,128] tile is **still infeasible** — the
hardware-DGE path needs `src.shape[0]==16`, the software-DGE path needs the source in
HBM; the shape/memory constraints block it, not just the dtype. `nc_transpose(bf16)`
lands in fp32 PSUM and needs a re-cast (no net win, and the sibling measured the
vector-engine transpose as a +2% regress). The identity-matmul transpose stays. **Do
not explore.**

### D5 — precision / product-count changes  *(FORBIDDEN this phase)*

The 3-product bf16 arithmetic is pinned: margin is only 1.31x, D3-4-product was
MEASURED-REJECT (+28% for 1.6%), plain bf16 fails 117x. Phase 3 does **not** touch the
arithmetic — every candidate must reproduce rel-L2 1.528e-5 exactly.

### Also N/A (shape closure — see `docs/shape-specialization-closure-phase3.md`)
- **Edge / partial tiles:** none — M=32·128, K=8·128, N=4·512 all divide evenly.
- **Partition/free regime:** forced by `nc_matmul` (k on partition, `[m_in,n]` out).
- **N_CHUNK:** 512 = `psum_fmax`, already maximal.
- **M-blocking:** vacuous — `w'` fully resident, each `x`/`z` read once.
- **LNC2 / multi-core:** out of the single-core scoring contract.

---

## 3. Execution plan (≤3 candidates; bit-exact-first)

1. **v4 (D1, aT-split-from-PSUM):** `runs/add_rmsnorm_matmul_v4_psum_split.py`, forked
   from v3. Drop the `aT_f` fp32 copy; source `aT_hi`/`aT_lo` from `psum_t`. Verify
   full-5-seed PASS and confirm rel-L2 == 1.528e-5 (bit-exact). Run `--fast`, then the
   FULL 5-seed latency twice for stability, plus a same-session v3 anchor. Capture the
   profiler digest (watch PE% and Vec/Scl%).
   - **Promote iff** full-5-seed PASS AND p50 beats v3 out-of-noise (>1.8% band).
   - **Otherwise** record the within-noise floor-confirmation; v3 stays promoted.
2. **v5 (D2) only if D1 leaves a measurable Vec/Scalar bubble.** Contingent, not
   proactive. Same gates.
3. **Close-out:** whichever of v3/v4 is fastest is the phase-3 (and task) result. Report
   speedup vs the 1.859287 ms baseline on the **full** correctness gate.

## 4. Evidence to record
- `benchmark.csv`: one row per perf-relevant candidate (v4, and v5 if run), plus the
  D3/D4/D5-closed decisions as record-only rows/notes.
- `candidates.jsonl`: DAG node v4→v3 (and v5→v4 if run) with metrics, `rel_l2`,
  `per_seed_rel_l2`, the bit-exactness note, and the `per_seed_latency_ms=null` /
  `latency_scope` caveat carried from v2/v3.
- `profile/`: v4 digest + PE-idle before/after interpretation; a short shape-closure doc
  (or a reference to the sibling's) for AC-6.

## 5. Correctness invariants (never regress)
- fp32 RMSNorm reduction (`square`/`reduce`/`mean_eps`/`rsqrt`); eps added AFTER the `/K`
  mean, matching the reference exactly.
- g folded into `w'` on the k_in/partition axis; `inv_rms` applied post-scale at
  eviction — both exact commutations (offline fp32 control 4.82e-7).
- The 3-product bf16 split and its PINNED split order are unchanged; D1 only removes an
  exact intermediate copy, so rel-L2 must reproduce **1.528e-5** bit-for-bit.
- Every promotion gated on **full 5-seed** `l2_norm_passed`, not `--fast` alone.
- **CAVEAT (carried from phase 2):** the adapter fixes seed 42 for all 5 profiler seeds,
  so on-device 5-seed PASS is weak on *input* diversity; the offline 7-draw sim
  mitigates. v1 (3.754x) and v2 (3.898x) remain the pure-fp32 fallbacks.
