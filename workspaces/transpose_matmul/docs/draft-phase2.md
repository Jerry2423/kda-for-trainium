# transpose_matmul — Phase 2 draft (profile-driven optimization)

Operator: `transpose_matmul` (NKIBench case 2). `out = lhs^T @ rhs`,
lhs (K,M)=(2048,4096) K-major, rhs (K,N)=(2048,10944), out (M,N)=(4096,10944),
fp32. MACs = M·N·K ≈ 9.17e10. Baseline 4.849615 ms.

Start point: **tmm_v1** (`runs/tmm_v1.py`), the promoted phase-1 kernel —
**1.026x (4.7274 ms)**, full-5-seed L2 PASS (rel-L2 3.99e-7). fp32 no-transpose
M-block-outer streaming GEMM: the NKIBench reshape (K,·)→(128,16,·) maps flat
k = k_in·16 + kt so K sits on the PARTITION axis of both operands, and
`nc_matmul(stationary, moving) = stationary.T @ moving` computes lhs^T @ rhs
directly — **no explicit transpose stage**. Round-0 profiler digest:
`profile/tmm_v1_digest.md`, raw dump `profile/tmm_v1_dump.txt`.

---

## 1. Round-0 bottleneck (established, not re-litigated)

tmm_v1 is **PE-BOUND at the fp32 systolic floor**:

| metric | value | reading |
|---|---|---|
| PE active % | **99.80%** | PE is essentially the entire wall clock |
| TRUE PE-active/inf | **4.718 ms** | ≈ p50 4.7277 ms → matmul IS the latency |
| MFU | **49.41%** | the fp32 floor: bf16-native array emulating fp32, capped ~50% by the bf16-peak MFU denominator — structural, not inefficiency ([[BL-20260709-fp32-pe-floor-calibration]]) |
| Vec / Scl / DMA % | 5.0 / 0.2 / 21.4 | all hidden well under PE; **DMA fully hidden** |
| HBMrd / HBMwr | 392.2 / 179.3 MB | EXACT once-lhs / 4×-rhs / once-out model; **no spill** |
| matmul_instruction_count | 24576 | over 12288 sites (4·24·8·16) = **2.0 instr/site** → fp32 emulates in 2.0 passes |

**Conclusion (DEC-2 diagnostic, already recorded): the bottleneck is COMPUTE —
the fp32 PE rate.** DMA is hidden and HBM is at the floor, so every
bandwidth/locality lever (M-block enlargement, N-tiling for DMA amortization,
double-buffering) can only touch already-hidden time and **cannot move a
PE-bound wall clock**. The one lever that attacks the actual bottleneck is
**lower matmul precision.** This exactly matches the phase-1 memory's phase-2
hand-off (lever = compute / fp32-PE floor).

---

## 2. Lever enumeration and ranking

Ranked by expected benefit vs risk. Only D1 attacks the bottleneck; D2–D4 are
PE-side micro-levers gated on D1's post-profile.

| # | lever | expected benefit | risk | priority |
|---|---|---|---|---|
| **D1** | **compensated bf16x2 3-product split** (both operands) | **HIGH (~1.25x kernel-over-kernel → ~1.28x over baseline)** | LOW-MED (numeric proven offline; **speed must be measured**) | **PRIMARY** |
| D2 | N_CHUNK 456→512 + masked tail | LOW-MED (fewer, wider matmuls; amplified 3× under the split) | LOW-MED (reintroduces the mask arithmetic phase-1 removed) | secondary, screen after D1 |
| D3 | M_BLK 8→16 (or 4) sweep | LOW (DMA already hidden) — possible **ANTI-lever** | MED (enlarged resident live set can constrain the affine_range pipeline, [[BL-20260710-cross-batch-blocking-is-an-antilever-on-affine-range]]) | screen `--fast` only if D1 shifts the DMA balance |
| D4 | 4-product split (keep lo@lo) | negligible numeric gain | HIGH latency (+~28% on siblings) | **SKIP** unless D1 comes back marginal |

### Why D1 is the right primary and why it is NOT assumable (the crux)

The compensated bf16x2 3-product split runs the matmul in bf16 arithmetic while
recovering ~16 effective mantissa bits, clearing the 2e-5 relative-L2 gate at
bf16-class matmul rate ([[BL-20260709-compensated-bf16x2-split-beats-fp32-floor]]).
It has been PROMOTED on three sibling GEMMs (rmsnorm_matmul 1.066x→1.363x;
add_rmsnorm_matmul 3.754x→4.632x; matmul_add_rmsnorm 3.920x→4.879x) and LOST on
one (swiglu all-3-GEMM 0.409x). The governing lesson
[[BL-20260710-bf16x2-loses-when-fp32-emulates-in-2-passes]] is explicit: **the
numeric margin transfers across ops (the offline sim proves it), but the SPEED
win does NOT — it depends on a per-op hardware quantity (per-instruction fp32
rate + limb residency) that MUST be measured first.**

tmm_v1 emulates at **2.0 matmul-instr/site — the same count that made swiglu's
all-3 split LOSE.** So the count alone does NOT license the split. But the two
conditions that decide the sign both point to a WIN here, and this op is the
**structural twin of `matmul_add_rmsnorm`'s GEMM (same M=4096, K=2048, dense
wide moving, 2.0/site), which WON** at that identical count:

1. **Per-instruction fp32 rate.** The winning siblings measured fp32 ≈ 1.8× the
   bf16 per-instruction rate on a dense moving-512 GEMM (add_rmsnorm_matmul:
   matmul instrs +44% yet TRUE PE-active **−23.4%**). tmm's moving operand is
   456-wide (same dense regime), so the split converts 2.0 fp32 passes (at
   ~1.8×) into 3.0 bf16 passes (at 1.0×) ≈ 3.6 → 3.0 bf16-equiv cost ≈ −17%
   PE-active predicted. The full-matmul phase-3 probe measured fp32/bf16 ≈ 3.62×
   end-to-end — an even wider gap. **Must measure the actual delta on this op.**
2. **Limb residency (no reload trap).** swiglu LOST partly because its weights do
   NOT fit resident, so bf16 limbs had to be rebuilt from re-loaded weights →
   DMA-bound. **Here, two bf16 limbs occupy exactly the same bytes as one fp32
   tile** (2×2-byte bf16 = one 4-byte fp32), so the resident lhs block and the
   streamed rhs chunk keep their phase-1 SBUF footprint, and **HBM stays at the
   floor** (limbs built on-chip from the same fp32 loads — no extra reads). Both
   swiglu-loss conditions are ABSENT.

**tmm is strictly cheaper to split than the winning sibling:** the sibling had
to transpose x per tile (fp32 identity matmul → PSUM → copy) before splitting
the activation; **tmm has no transpose at all** — lhs arrives K-on-partition, so
both limbs are built directly from the loaded fp32 tiles (no PSUM round-trip, no
transpose scratch). And there is no residual-add / RMSNorm / g epilogue to
expose. So tmm sits in the **best of both regimes**: the sibling's favorable
SPEED regime (resident limbs, dense wide moving, 2.0/site) **plus** the
swiglu/matmul favorable PRECISION regime (tiny fp32 floor — see §4).

Honest expectation: sibling kernel-over-kernel wins were ×1.245–1.279, so
1.026x → **≈1.28x over baseline**. But per the loss-lesson this is a MEASURED
target, not a promise; a RISE in TRUE PE-active would mean the split loses here
and the fp32 floor is terminal.

---

## 3. D1 kernel design (localized diff on tmm_v1, NO enabler refactor)

Mirror the sibling idiom `matmul_add_rmsnorm_v2_bf16_split.py`. The loop nest,
constants (M_BLK=8, N_CHUNK=456, 24 chunks, 16 kt), PSUM accumulation, and the
copy+store epilogue are **byte-for-byte v1**. Only the operand dtype path
changes; precision loss is confined to the matmul.

**Pinned, auditable split order** (round-to-nearest-even via `nl.copy(dtype=nl.bfloat16)`;
residual via `nisa.tensor_tensor(..., op=nl.subtract)` which upcasts to fp32):

```
lhs (fp32) -> lhs_hi = bf16(lhs);  lhs_lo = bf16(lhs - lhs_hi)   # once per m-block, resident
rhs (fp32) -> rhs_hi = bf16(rhs);  rhs_lo = bf16(rhs - rhs_hi)   # once per (mb, c), reused across 8 subtiles
```

**3 bf16 products in one fp32 PSUM bank, FIXED order, dropping lhs_lo@rhs_lo:**

```
acc += nc_matmul(lhs_hi[kt, :, 128*s:], rhs_hi[kt])   # hi @ hi
acc += nc_matmul(lhs_hi[kt, :, 128*s:], rhs_lo[kt])   # hi @ lo
acc += nc_matmul(lhs_lo[kt, :, 128*s:], rhs_hi[kt])   # lo @ hi
```

(The split is symmetric — hi@hi + hi@lo + lo@hi keeps ~16 mantissa bits; the
dropped lo@lo is ~1e-6, confirmed negligible by the 4-product offline check.)

**Structural changes vs v1:**
- After loading each fp32 lhs tile into a transient buffer, build `lhs_hi[kt]`,
  `lhs_lo[kt]` bf16 `[128,16,1024]` (2×32 KB = 64 KB/part = **same bytes as v1's
  fp32 lhs_blk**, which is dropped). Built once per m-block, resident.
- After loading each fp32 rhs tile, build `rhs_hi`, `rhs_lo` bf16 `[128,16,456]`
  (2×~14.25 KB ≈ 28.5 KB/part = **same bytes as v1's fp32 rhs_chunk**). Built
  once per (mb,c), reused across the 8 subtiles.
- Inner loop: 1 fp32 `nc_matmul` → 3 bf16 `nc_matmul` into the same PSUM bank.
- Epilogue (`nl.copy` PSUM→SBUF fp32, `nl.store`) unchanged.

**SBUF budget:** resident lhs limbs 64 KB + rhs limbs 28.5 KB + transient fp32
build scratch (lhs tile 64 KB freed after build, rhs tile ~28.5 KB) + out_sb
(1.8 KB) + PSUM banks — peak well under 192 KB, same argument as the sibling.
**HBM unchanged vs v1** (~392 MB read: limbs built from the same fp32 loads).

**matmul_instruction_count** will rise 24576 → ~36864 (2.0→3.0/site); the win is
that each new instr is a bf16 pass, not the 2.0× fp32 emulation. This is the
number the measurement protocol reads.

Deliverable: `runs/tmm_v2_bf16_split.py` (parent tmm_v1).

---

## 4. Correctness plan (offline-first, zero remote spend)

Build `runs/offline_bf16_split_sim.py` (mirror the sibling sim, simplified — no
z/g/norm; this op is a pure GEMM):
- Reproduce the exact scored input: `np.random.seed(seed)` then draw
  `lhs = normal(0,1,(K,M))`, `rhs = normal(0,1,(K,N))` in reference order
  (adapter DEFAULT_INPUT_SEED=42; note the adapter reseeds to 42 for every
  profiler draw, so the offline multi-seed sweep IS the real input-diversity
  evidence).
- fp32 control `lhs.T @ rhs` must reproduce the numpy reference to ~1e-7
  (validates seed / draw-order / dtype).
- Report worst 3-product rel-L2 over seeds {42,0,21,63,84,123,2024}; also the
  4-product number (to size the dropped lo@lo) and plain-bf16 (scale check,
  expect ~2e-3 FAIL).

**Predicted numbers** (from the pure-GEMM family — rmsnorm_matmul offline
4.455e-6, swiglu 7.7e-6): worst 3-product rel-L2 ≈ **4.5e-6**, comfortably below
the 2e-5 gate. The offline sim GATES: comfortably-below authorizes building the
kernel; at/above records the precision-floor datum instead.

**On-device rel-L2 = quadrature(fp32-floor, bf16-term)**
([[BL-20260709-compensated-bf16x2-split-beats-fp32-floor]]). tmm_v1's measured
fp32 floor is **3.99e-7 — tiny** (the pure-GEMM regime: swiglu 6.36e-7,
rmsnorm_matmul 4.8e-7; NOT the add_rmsnorm-family ~1.46e-5, which carries a
RMSNorm square-reduce feedback tmm lacks). So predicted on-device
≈ sqrt(3.99e-7² + 4.5e-6²) ≈ **4.5e-6 ≈ the offline number** — the bf16 term is
the whole story here, ~4.5× under the gate. Confirm on-device by backing out the
bf16 term (√(ondevice² − floor²)) and checking it matches the offline sim, per
the family protocol. **D4 (4-product) is predicted UNNECESSARY** and should be
SKIPPED (worst ≈4.5e-6 « the 1.8e-5 danger band the siblings used).

---

## 5. Measurement protocol (the loss-lesson requires it)

Per [[BL-20260710-bf16x2-loses-when-fp32-emulates-in-2-passes]], MEASURE the PE
delta before promoting — do not assume the sibling win transfers:

1. Offline sim first (§4). If it clears the gate → build the kernel.
2. `verify.py --fast` on tmm_v2 → correctness + latency direction.
3. `runs/dump_metrics.py --fast` on tmm_v2 AND a same-session tmm_v1 anchor →
   read **TRUE PE-active/inf** and **matmul_instruction_count** for both.
   Per-instruction rate = TRUE PE-active / matmul_instruction_count:
   - v1 anchor ≈ 4.718 ms / 24576 ≈ 0.192 µs/instr (fp32, 2.0/site).
   - **KEEP if v2 TRUE PE-active DROPS** (sibling signature: instrs +50%, but
     PE-active −~20%). **REJECT if it RISES** (swiglu signature: +18.6%) → the
     split loses on this op, the fp32 floor is terminal, keep v1, record the
     per-instruction-rate datum.
4. On a KEEP, confirm on the **full 5-seed run** and rank by stable p50 against a
   same-session tmm_v1 control band ([[BL-20260709-fast-vs-full-run-latency]]);
   promote only if outside the noise band.

Keep tmm_v1 as the guaranteed **pure-fp32 fallback** (the 5 profiler seeds reuse
the seed-42 input, so on-device input diversity is weak; the offline sim's
distinct draws are the real margin evidence).

---

## 6. Secondary levers (gated on D1's post-profile)

- **D2 — N_CHUNK 456→512 + masked tail.** 10944/512 = 21.375 → 21 full 512-wide
  chunks + 1 tail of 192, i.e. 22 chunks vs 24. Fewer, wider matmuls cut
  per-instruction issue overhead — amplified 3× under the split (each site is 3
  matmuls). Cost: reintroduces the `mask=…>=0` arithmetic phase-1 deliberately
  removed (the baseline's largest bug surface). Total PE columns pushed are ~the
  same either way, so the gain is only the fixed-overhead reduction — **screen
  with `--fast` after D1 lands**; adopt only if it clears the same-session band
  and stays correct (mask off-by-one is the risk).
- **D3 — M_BLK 8→16 (or 4).** Cuts rhs re-reads 4×→2×, but DMA is already hidden
  (21%), so it cannot help a PE-bound wall clock unless D1's limb-building shifts
  DMA off hidden. Enlarging M_BLK also enlarges the resident lhs-limb live set,
  which can **CONSTRAIN the affine_range software pipeline** (the bmm
  cross-batch-blocking anti-lever, [[BL-20260710-cross-batch-blocking-is-an-antilever-on-affine-range]]).
  Treat as a possible anti-lever — screen `--fast` only, do not assume monotone
  benefit.
- **D4 — 4-product.** SKIP (predicted numerically unnecessary, §4; siblings
  measured +~28% latency for a term swamped by the tiny fp32 floor). Build only
  if D1 surprises with a marginal on-device reading, and then record as a
  measured reject unless the backed-out bf16 term is itself near the gate.

---

## 7. Exit criteria / plan for the ≤5-iteration budget

- **Iter 0:** offline sim → numeric gate (no spend).
- **Iter 1:** build tmm_v2 (D1), `--fast` verify + dump_metrics vs same-session
  v1 anchor → measure the PE-active sign.
- **Iter 2:** if KEEP, full-5-seed confirm + control band → promote tmm_v2.
- **Iter 3 (optional):** D2 N_CHUNK=512 `--fast` screen, composed on tmm_v2.
- **Iter 4 (optional):** D3 M_BLK `--fast` screen only if D1 moved the DMA
  balance.

Success = a promoted kernel that beats tmm_v1's 1.026x (target ≈1.28x) with
full-5-seed L2 PASS, HBM still at the floor, and the PE-active drop
documented against the same-session anchor. Failure mode (split RISES PE-active)
= fp32 floor is terminal, keep tmm_v1, record the per-instruction-rate datum as
the second confirming data point (after swiglu) that the 2.0/site count can go
either way. Record every candidate in `benchmark.csv` + `candidates.jsonl` (DAG
parent links), profiling evidence under `profile/`. Never regress correctness.
