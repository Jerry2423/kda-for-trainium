# transpose_matmul — Phase 3 exit decision

**Verdict: FINALIZE `tmm_v3_mblk16_bf16_split` at 1.334x (3.6338 ms).**
`tmm_v1.py` (fp32, 1.026x) retained as the guaranteed pure-fp32 fallback.

Phase 3 is *shape / regime specialization*. The measured answer is that **the
dominant lever (product count) is a proven numeric floor and the shape is
edge-free**, so tmm_v3 is within 3.76% of the hard 1.385x arithmetic ceiling. Both
gap-only levers were **built and measured-rejected**: the wider limb-build chunk
(`tmm_v6_nchunk912`) spilled catastrophically, and the double-buffer / prefetch of the
rhs limbs (`tmm_v7_dbuf_rhs`) engaged its intended overlap (a ~0.2% PE-active dip) but
broke the AC-4 read floor (283 MB) with a small write-spill — a real-but-tiny wall gain
that cannot waive the immutable read gate. This mirrors the `bmm` phase-3 outcome ("at
the floor → finalize; every reschedule regressed"), not the `silu`/`rmsnorm_matmul`
outcome (a new lever appeared).

---

## 1. Headline: tmm_v3 is within 3.76% of the hard arithmetic ceiling (AC-1)

Same-session anchors this phase (`profile/tmm_phase3_v3a.txt`, `_v3b.txt`, both
`dump_metrics --fast`) reproduce the phase-2 numbers exactly:

| metric | v3a | v3b | reading |
|---|---|---|---|
| wall p50 | 3.6355 ms | 3.6359 ms | 1.334x (control band \|A−B\| = 0.011%) |
| TRUE PE-active/inf | 3.5630 ms | 3.5658 ms | the matmul IS the wall (band 0.08%) |
| PE active % | 98.01% | 98.07% | PE is essentially the entire wall |
| matmul_instruction_count | 36864 | 36864 | 3.0 instr/site over 12288 sites |
| hbm_read / write | 229.0 / 179.3 MB | 229.0 / 179.3 MB | below the v1 floor; no spill |
| psum_read_sbuf_write_count | 768 | 768 | one drain/tile, no spill |
| rel-L2 | 4.4515e-6 | 4.4515e-6 | ~4.5x under the 2e-5 gate |

### The cost-model floor (kernel-cost-analysis, trn2)

The Tensor Engine streams the moving operand two columns/cycle (bf16 double-pumped);
the stationary weight load pipelines behind the previous matmul's moving stream, so
a 456-wide moving fully hides the 128-cycle load. The floor is total-moving-columns
/ 2 / freq:

```
moving-col-cycles = (M/128 m-subtiles) · (K/128 kt) · (3 products) · N
                  = 32 · 16 · 3 · 10944 = 16,809,984
PE floor (bf16, 2 col/cyc, 2.40 GHz) = 16,809,984 / 2 / 2.40e9 = 3.502 ms
```

- **Ceiling speedup = 4.849615 / 3.502 = 1.385x** (hard).
- v3 PE-active 3.564 ms is **1.77% above the floor** (~1.7 ns residual bubble/instr;
  the weight load is confirmed fully hidden by the wide moving stream).
- v3 wall 3.6338 ms is **3.76% above the ceiling**; **1.96% (69.8 µs) is the PE-idle
  gap** (wall − PE-active), the remaining ~1.8% is the near-floor PE-active bubble.

The only two slivers are (a) the ~1.8% PE-active bubble (systolic warm-up per matmul,
essentially unrecoverable) and (b) the ~1.96% PE-idle gap. The dominant term
(PE-active) is set by the product count — a hard numeric floor (§3).

---

## 2. The shape is edge-free — nothing to specialize (AC-1)

| axis | size | tiling | remainder |
|---|---|---|---|
| M (rows) | 4096 | 32 × 128 subtiles, M_BLK=16 | **0** (exact) |
| K (contraction) | 2048 | 16 × 128 kt on partition | **0** (exact) |
| N (cols) | 10944 = 2⁶·3²·19 | 456 × 24 (≤512 PSUM) | **0** (exact divisor) |

- N_CHUNK=456 makes every tile full-size — **zero tail masking anywhere**; no edge
  tile exists to give a different regime.
- Phase 2 already swept the chunk-width knob and **measured-rejected both wider
  forms**: N_CHUNK=512 mask-free 192-tail (`tmm_v4`, wall +1.12%) and masked-tail
  (`tmm_v5`, +4.2%). The kernel is **PE-column-bound, not issue-overhead-bound** —
  fewer/wider matmuls cannot help (total PE columns pushed are invariant).
- M_BLK is pinned at 16 by the AC-4 read floor (M_BLK=8 → 448 MB read regression;
  M_BLK=32 → 256 KB/part resident bf16 limbs spills the ~208 KB SBUF).

The classic shape moves (edge-tile fast path, partition/free re-split, per-regime
tile size) are all no-ops or already-rejected. A "shape" phase on a
perfectly-divisible single-regime GEMM confirms no specialization is warranted.

---

## 3. Product count (3) is the numeric floor — PE-active cannot drop (AC-1)

PE-active scales **linearly** with bf16 products/site. Offline reproduction this
phase (`runs/offline_product_count_floor.py`, zero remote spend, full K=2048, N(0,1),
seeds {42,0,84}; fp32 control bit-exact 0.0), in `profile/tmm_phase3_product_count_floor.txt`:

| scheme | products | worst rel-L2 | vs 2e-5 gate |
|---|---|---|---|
| plain bf16 | 1 | 2.351e-3 | **FAIL ×118** |
| bf16 split lhs only | 2 | 1.662e-3 | **FAIL ×83** |
| bf16 split rhs only | 2 | 1.663e-3 | **FAIL ×83** |
| **bf16 3-product (current)** | **3** | **4.453e-6** | **PASS (4.5× under)** |
| plain fp16 | 1 | 2.939e-4 | **FAIL ×15** |
| fp16 split (drop one lo) | 2 | 2.078e-4 | **FAIL ×10** |
| fp16 3-product | 3 | 5.187e-7 | PASS (over-accurate) |

Any 2-product scheme leaves one operand at 7-bit (bf16) / 10-bit (fp16) mantissa,
whose rounding dominates 10–118× over the gate. **fp16 does not rescue a 2-product
scheme** (still 10× over) and its 5-bit exponent is a correctness risk for zero speed
gain (fp16 limbs cost the same PE cycles as bf16). **3 bf16 products is the numeric
floor; PE-active cannot drop; 1.385x is a hard ceiling.** 4-product is the wrong
direction (accuracy already 4.5× spare).

---

## 4. Gap-only lever screens (the only room left is the ~2% PE-idle gap)

### E1 — N_CHUNK=912 (halve the rhs limb-build prologue count): MEASURED REJECT (AC-2)

`runs/tmm_v6_nchunk912.py` (parent tmm_v3). **Clean isolation:** the limbs are built
over a 912-wide chunk (12 prologues/block vs 24) but the matmul moving width stays
456 (two 456-wide sub-chunks per 912 build), so matmul_instruction_count, PSUM usage,
and PE columns are **byte-identical to v3** (36864). The only changed variable is the
rhs-limb build granularity — NOT the chunk-width change v4/v5 already rejected.

Screen (`profile/tmm_phase3_v6_nchunk912.txt`, `dump_metrics --fast`):

| metric | v3 anchor | E1 (v6) | delta |
|---|---|---|---|
| wall p50 | 3.6355 ms | **7.3303 ms** | **+101.6%** |
| TRUE PE-active/inf | 3.5630 ms | **4.0458 ms** | **+13.5% (RISING → reject)** |
| hbm_read | 229.0 MB | **1764.7 MB** | **7.7× (AC-4 FAIL)** |
| hbm_write | 179.3 MB | **259.6 MB** | +45% → **SPILL (AC-4 FAIL)** |
| matmul_instr | 36864 | 36864 | identical (isolation confirmed) |
| rel-L2 | 4.4515e-6 | 4.4515e-6 | correct |

**Verdict: REJECT** on all three gates simultaneously — wall +101.6% (≫ band+3%),
PE-active RISING, and AC-4 read/spill both blown. Mechanism: the 912-wide rhs limbs
= 2×[128,16,912] bf16 ≈ 57 KB/part (vs v3's 28.5 KB); on top of the 128 KB/part
resident lhs limbs + transient fp32 build tiles, peak SBUF exceeds the ~192 KB
budget, so the compiler spills the bf16 limbs and re-fetches (hbm_write +80 MB spill
saves, hbm_read explodes 7.7×). This is the enlarged-resident-live-set anti-lever
(`bmm` cross-batch class) measured directly on this op. The "halved prologue count"
never mattered because v3's limb build is **already fully hidden** (Vec 13.7% / Scl
6.7% / GpSimd 14.2% / DMA 22.5%, all « PE 98%).

### E2 — double-buffer / prefetch rhs limbs: MEASURED REJECT (AC-3)

`runs/tmm_v7_dbuf_rhs.py` (parent tmm_v3). Holds **two FIXED resident rhs-limb buffer
sets (A / B)** per M-block and runs the faithful **build-ahead / prefetch** schedule:
prime A ← chunk 0, then for p=0..10 build B ← chunk 2p+1 (prefetch), matmul A(2p), build
A ← chunk 2p+2 (prefetch), matmul B(2p+1); final pair build B ← 23, matmul A(22), matmul
B(23). Emitted straight-line (Python-unrolled, NOT `sequential_range`) so each
build-ahead can overlap the *preceding* chunk's matmul — the genuine ping-pong.
M_BLK=16, N_CHUNK=456, 3 products, output layout unchanged — bit-exact reschedule. (This
is the faithful prefetch; an earlier round built both chunks of a pair before either
matmul — a two-set batch — which Codex review flagged as not the specified lever.)

Screen (`profile/tmm_phase3r2_v7_dbuf.txt`, `--fast`; same-session anchors
`tmm_phase3r2_v3a.txt`/`_v3b.txt`):

| metric | v3 anchor (a/b) | E2 (v7) | delta |
|---|---|---|---|
| wall p50 | 3.6356 / 3.6352 ms | **3.6256 ms** | −0.27% (≪ 3% adopt bar) |
| TRUE PE-active/inf | 3.5643 / 3.5632 ms | **3.5567 ms** | −0.2% (prefetch overlapped a sliver) |
| hbm_read | 229.0 MB | **283.4 MB** | **+23.8% (AC-4 FAIL)** |
| hbm_write | 179.3 MB | **183.0 MB** | **+2.1% → small write-spill (AC-4 FAIL)** |
| psum | 768 | 768 | unchanged |
| matmul_instr | 36864 | 36864 | identical (pure reschedule) |
| rel-L2 | 4.4515e-6 | 4.4515e-6 | correct |

**Verdict: REJECT** on the AC-4 gate. Unlike the batch version, the faithful prefetch
*did* engage — TRUE PE-active dips 0.2% and the wall dips 0.27% (both just outside the
0.011% control band), because two fixed buffers let the compiler start chunk c+1's build
during chunk c's matmul. But (a) v3's limb build was **already ~86% hidden** (Vec 13.7%
/ Scl 6.7% / GpSimd 14.2% / DMA 22.5%, all « PE 98%), so there is little exposed build to
recover — the 0.27% wall gain is inside the ~2% PE-idle-gap ceiling and **far below the
3% adopt threshold**; and (b) holding **both** limb sets resident for the whole N-sweep
enlarges the live working set enough to break the immutable AC-4 read floor
(hbm_read 283 MB, +23.8%) **and** tip a small write-spill (hbm_write 183 MB, +2.1%). This
is exactly the `tmm_v2` M_BLK=8 situation — a tiny wall gain cannot waive the immutable
read/spill gate. The double-buffer's intended mechanism works but is not worth its
memory cost: the enlarged-resident-live-set anti-lever, now with the prefetch actually
engaged.

Note: E2's read regression (283 MB) is *larger* than the earlier batch version's
(257 MB) because both fixed limb sets are live for the entire N-sweep, not just within a
pair; the wider-build screen (`tmm_v6`, 1765 MB catastrophic spill), the batch v7
(257 MB), and this faithful prefetch v7 (283 MB) are three points on the same
enlarged-resident-live-set anti-lever, differing only in how the compiler places the
extra footprint. All reject on AC-4.

E3 (engine placement for the limb-build subtract) and E4/E5 are SKIP: E3's target
(limb build) is already hidden and not the idle source; E4 (4-product) adds PE for
accuracy already 4.5× spare; E5 (edge-tile) is a no-op on an edge-free shape.

---

## 5. Disposition

- **PROMOTED (unchanged): `tmm_v3_mblk16_bf16_split` — 1.334x (3.6338 ms)**,
  full-5-seed L2 PASS (rel-L2 4.4515e-6), hbm_read 229 MB / write 179.3 MB / psum 768
  (no spill), PE 98%, within 3.76% of the 1.385x hard ceiling.
- **FALLBACK (retained): `tmm_v1.py` — 1.026x**, pure fp32, guaranteed-correct.
- **REJECTED (measured): `tmm_v6_nchunk912`** (wider limb-build chunk) — wall +101.6%,
  PE-active +13.5%, hbm_read 7.7×, spill. Enlarged-resident-live-set anti-lever.
- **REJECTED (measured): `tmm_v7_dbuf_rhs`** (double-buffer / prefetch rhs limbs,
  faithful build-ahead schedule) — the prefetch engaged (wall −0.27%, PE-active −0.2%)
  but the gain is ≪ the 3% adopt bar, and it breaks the AC-4 floor (hbm_read 283 MB,
  +23.8%; hbm_write 183 MB, small write-spill). Same enlarged-resident-live-set
  anti-lever; a tiny wall gain cannot waive the immutable read/spill gate.

**The phase-3 deliverable is the proof that tmm_v3 is near-optimal** (1.385x ceiling +
3-product numeric floor + edge-free shape), not a new speedup. 2 optimization
iterations used (wider limb-build chunk, double-buffer); both measured-rejected. Never
regressed correctness or the AC-4 read floor.
