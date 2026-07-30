# matmul Phase 3 — regime specialization: the compute-PRECISION regime

## TL;DR (what changed since the 2026-07-09 close)

Phase 3 was originally closed with "matmul_v2_b4 (1.017x) is at the fp32 PE floor;
the only way past it is lower precision, which is out of scope / uncertain vs the
2e-5 gate." **That "uncertain" has since been resolved to "gate-legal, large win"
by four sibling GEMMs** run after this task closed — and now by this op's own
zero-spend offline gate. The phase-3 win to capture is therefore not a *tile-shape*
regime (there are no edge tiles here) but the **compute-precision regime**: a
compensated **bf16x2 3-product split** of the GEMM, which trades the fp32 PE penalty
for 3 bf16-rate passes and clears 2e-5 comfortably.

- **Starting point:** `runs/matmul_v2_b4.py` (Phase 2), **1.017x (13.35 ms)**, PE=100%,
  MFU=49%, DMA=31%, HBMrd=2097 MB. Fully PE-bound at the *fp32* systolic rate.
- **New target:** `runs/matmul_v3_bf16_split.py` — a localized bf16x2-split diff on
  v2_b4. **Expected ~1.22–1.33x** (siblings landed −20% to −24% PE-active wall).
- v2_b4 is **retained as the guaranteed pure-fp32 fallback** (identical to every
  sibling: keep the fp32 kernel, promote the split on top).

## Why this reverses the old "fp32 floor is terminal" conclusion

The old draft's fp32-floor analysis is *correct and unchanged*: trn2's PE array is
bf16-native, fp32 emulates at ~2 passes, so a correct fp32 GEMM is capped near ~50%
MFU. What was wrong was treating that floor as *binding under the 2e-5 gate*. The
floor is only binding **if you must stay in fp32** — and the compensated split does
not. Every GEMM sibling proved it:

| op | fp32 floor | bf16x2 split | mechanism |
|---|---|---|---|
| **transpose_matmul** (pure dense `lhsᵀ@rhs`, moving-GEMM twin of matmul) | 1.026x | **1.334x** | 3-product split, both operands |
| **matmul_add_rmsnorm** (dense GEMM, moving-512, +epilogue) | 3.920x | **4.879x** | 3-product split; per-instr fp32 rate ~1.8x on moving-512 |
| rmsnorm_matmul | 1.066x | 1.363x | split |
| swiglu down-GEMM | (fp32) | 1.026x | split (down-GEMM only) |

`transpose_matmul` is the structural twin (a plain dense GEMM whose only difference
is that its inputs already arrive K-on-partition, so it skips matmul's lhs
transpose). It went 1.026x → 1.334x with exactly the split this draft proposes.

## Correctness is authorized — offline numeric gate (zero remote spend)

`runs/offline_bf16_split_sim.py` reproduces the adapter's seed-42 draw (and 4 more
seeds), computes the fp32 reference the NKIBench way, and scores an idealized bf16x2
3-product split. Result (`profile/matmul_phase3_bf16split_offline_gate.txt`):

```
fp32 CONTROL vs reference   rel-L2 = 0.000e+00   (seed/draw/dtype/formula bit-exact)
plain bf16 (rejected route)        = 2.350e-03   (~1e-3 scale check)
bf16x2 3-product (worst, 5 seeds)  = 4.454e-06
bf16x2 4-product (keeps lo@lo)     = 3.494e-06   (sizes the dropped term)
predicted device quadrature        = 4.474e-06   vs gate 2.0e-05   -> 4.5x under
```

This is the **pure-GEMM family** (no RMSNorm square-reduce feedback), so the bf16
error flows straight to the output — no composite/quadrature surprise like the
add_rmsnorm siblings. matmul_v2_b4's measured fp32 floor is 4.207e-7 (layout check),
sub-µ, so the device rel-L2 ≈ the bf16 term itself (~4.5e-6). K=5120 (vs the twin's
2048) is immaterial: the dropped lo@lo term is ~2⁻¹⁶ relative per product and the
rel-L2 is ~K-independent (measured identical 4.454e-6 across seeds).

## Expected speedup (from the measured per-instruction fp32 rate)

matmul's own calibration measured fp32/bf16-1-product = **3.62x** (13.35 ms vs the
3.69 ms bf16 probe). With fp32 at 2.0 instr/site and bf16 at 1.0, the per-instruction
fp32 rate is 3.62/2.0 ≈ **1.81x** — the moving-512 dense-GEMM regime, matching
matmul_add_rmsnorm's ~1.8x. The 3-product split runs 3.0 bf16-rate instr/site:

    split PE-active / fp32 PE-active  ≈  3.0 / 3.62  ≈  0.83   (−17% floor estimate)

Siblings measured a slightly larger real wall drop (−19.6% add_rmsnorm, −24.4%
transpose_matmul), because DMA that was marginal at fp32 becomes fully hidden once
PE-active shrinks. **Realistic target: 13.35 ms → ~10.3–11.0 ms = ~1.23–1.32x**
(baseline 13.578 ms), up from 1.017x. Honest floor: if only −17% converts, ~1.22x.

## Kernel design — localized diff on matmul_v2_b4 (D1, PRIMARY)

Keep v2_b4's entire structure — M-block **B=4**, N_CHUNK=512, K-accumulate 40 kt into
B distinct [128,512] fp32 PSUM banks, single copy+store epilogue. Change only the
operand precision and the matmul body:

1. **lhs transpose stays fp32, split AFTER transpose.** lhs arrives [m_in(par),
   k_in(free)]; the identity `nc_matmul(is_transpose=True)` idiom produces
   `lhs_t = [k_in(par), m_in(free)]` in fp32 PSUM (exact — 1 transpose/tile, count
   unchanged). Copy to SBUF fp32, then split element-wise:
   `lhs_hi = bf16(lhs_t)`, `lhs_lo = bf16(lhs_t − lhs_hi)` (residual subtract into a
   bf16 destination, no separate fp32 residual buffer). Store resident bf16 limbs
   `lhs_hi[mb,kt]`, `lhs_lo[mb,kt]` = [k_in=128, m_in=128]. *Do NOT split before the
   transpose* — that would double the transpose count for no gain.
2. **rhs split per chunk.** Load `rhs_f = [k_in,512]` fp32 once per (chunk, kt),
   `rhs_hi = bf16(rhs_f)`, `rhs_lo = bf16(rhs_f − rhs_hi)`.
3. **3-product accumulation** into the one fp32 PSUM bank per member, pinned order:
   `acc[mb] += lhs_hi@rhs_hi + lhs_hi@rhs_lo + lhs_lo@rhs_hi` (drop lo@lo). This is
   byte-for-byte the `tmm_v3_mblk16` body; the offline sim uses the identical order.

### SBUF / PSUM sizing (why this is *lower*-risk than the siblings)

Per-partition resident (128 partitions): `lhs_hi` + `lhs_lo` = 2 × [4,40,128] bf16 =
2 × 40 KB = **80 KB/part — exactly the bytes of v2_b4's fp32 `lhs_t`** (half the
dtype, twice the limbs). So the resident working set **does not grow**. Transients:
`rhs_hi/lo` ~1 KB each, `rhs_f`/`out_sb` ~2 KB, transpose scratch ~0.5 KB — peak
well under 90 KB/part vs the ~192 KB budget. Contrast transpose_matmul, which sat at
128 KB resident / 168 KB peak and had to fight AC-4 read-floor breaks; here the
headroom is large.

PSUM: 4 acc banks ([128,512] fp32 = 1 bank each) + 1 transpose bank = **5 of 8**.
B stays 4 — which matches the promoted fp32 kernel and keeps DMA hidden. B=8 is
**measured-rejected**, not bank-count-infeasible: `matmul_v2_b8.py` exists and ran
at 0.968x (enlarged lhsT residency + all PSUM banks hurt occupancy/schedule), and
the transpose PSUM bank is *transient* (used in the transpose-build loop before the
N-chunk accumulation loop), so it is not co-live with the acc banks. B=8 is rejected
on measured occupancy/schedule grounds and is risky under the split's added
register/schedule pressure — not proven bank-count-infeasible.

### The AC-4 read-floor gate (the transpose_matmul lesson)

The twin's first split candidate (`tmm_v2`, M_BLK=8) was correct **and** faster but
was **blocked** because its enlarged resident limbs made the compiler re-fetch ~15%
of rhs tiles (hbm_read 392→448 MB) — an AC-4 read-floor break that prose cannot
waive. Here the limb bytes equal the fp32 bytes and B is unchanged, so I *predict*
the floor holds — but this is an explicit gate, not an assumption:

> **D1 acceptance:** correct (5-seed rel-L2 < 2e-5, expect ≈4.47e-6) **AND** hbm_read
> stays ≈2097 MB, hbm_write ≈201 MB, psum-drain count flat (no spill) **AND** full
> 5-seed p50 beats 1.017x by more than the ±1.8% noise band. Any read-floor break →
> treat like tmm_v2 (do not promote on wall win alone; investigate before adopting).

## Directions, ranked

- **D1 — bf16x2 3-product split (PRIMARY).** As above. Expected ~1.23–1.32x.
  Offline-authorized; on-device re-gate required (not bit-exact vs fp32).
- **D2 — 4-product split (add lo@lo), CONTINGENT/record-only.** Offline 3.494e-6 vs
  3-product 4.454e-6 — marginally more accurate but +~25% PE work. The 3-product
  margin (4.5x under gate) needs no help, and both add_rmsnorm and transpose_matmul
  MEASURED-REJECTED the 4-product (worse for no correctness need). Build only if D1
  unexpectedly lands near the danger band (>1.3e-5), which the offline gate says it
  will not. Otherwise record-only.
- **D3 — M-block / DMA watch, CONTINGENT.** If D1's hbm_read breaks the floor or DMA
  becomes exposed (unlikely — limbs don't grow, DMA hidden at 31% and rises only to
  ~39% at the shorter wall), revisit. B=8 is measured-rejected (v2_b8 0.968x:
  occupancy/schedule pressure, not bank-count-infeasible — the transpose PSUM is
  transient), so any fix would be chunk-level, not a bigger B. No planned build.

## Why classic shape specialization is vacuous here (carried forward, still true)

This is a **single fixed shape with every tile full**: M=32·128, K=40·128,
N=24·512 — no remainders, no edge tiles, no regime to branch on. `nc_matmul` forces
k-on-partition (the lhs transpose is structural, not a choice); N_CHUNK=512 is the
fp32 PSUM-bank cap; stationary/contraction are at the 128 partition cap. There is no
*tile-shape* specialization to make. Phase 3's "specialize only where the measured
win justifies the complexity" therefore points at the **precision regime** (D1), the
one axis with real, sibling-proven headroom. The fp32 micro-levers explored in the
original round 0 (eviction-copy engine steering `nisa.tensor_copy(engine=)`,
off-PE transpose, B-sweep) remain measured-rejected/infeasible and are not revisited;
they were sub-noise even before the split reset the floor.

## Correctness / evidence contract

- fp32 I/O; internal bf16 limbs; all 5 seeds `[0,21,42,63,84]` pass relative-L2
  `< 2e-5`. Expected on-device rel-L2 ≈ 4.47e-6 (quadrature of the 4.21e-7 fp32 floor
  and the 4.454e-6 offline bf16 term; the bf16 term dominates).
- Single `@nki.jit def kernel(v1, v2)`; new candidate in `runs/matmul_v3_bf16_split.py`;
  never edit the baseline/reference; v2_b4 kept as fallback.
- Record the candidate in `benchmark.csv` + `candidates.jsonl` (parent = matmul_v2_b4);
  profiling digest under `profile/`. **Full 5-seed** (not `--fast`) before promoting —
  Phase-2 lesson: `--fast` mis-ranked B=8. Report the same-session v2_b4↔v3 A/B bracket
  (siblings' method) so the win is measured against a contemporaneous control, not the
  stale 13.35 ms number.

## Target and honest exit

Realistic: **~1.23–1.32x** (13.35 → ~10.3–11.0 ms), a step change over 1.017x, at a
correctness margin 4.5x inside the gate. If — against the offline evidence and four
siblings — the on-device split fails the AC-4 read floor and cannot be brought back
without regressing wall, promote nothing new and keep v2_b4 (1.017x) with the split's
measured-reject record. Given the offline gate PASS, the same-bytes resident footprint,
and the direct transpose_matmul precedent, promotion of the split is the expected
outcome.
