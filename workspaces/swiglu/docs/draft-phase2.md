# swiglu — Phase 2 draft (profile-driven optimization)

## 0. TL;DR

Phase-1 `swiglu_v1` is a correct fp32 kernel at **0.939x** (2.2079 ms vs the
2.0742 ms baseline). The profile says it is **PE-bound at the trn2 fp32-emulation
floor** (PE=95%, MFU=44%, DMA=49% hidden). Every "fusion" lever the phase-2 prompt
lists — share x across up/gate, fuse the SiLU gate, keep the (M,N) intermediate in
SBUF, no HBM spill — is **already implemented in v1**. The cost model confirms the
transposes are only ~5% of PE work and that removing the h-transpose by a layout
swap actually *loses*. So the only lever that moves the compute floor is the one the
PE array's own arithmetic dictates:

> **Primary Phase-2 direction: compensated bf16x2 3-product split on all three
> GEMMs.** The trn2 PE is bf16-native and emulates fp32 at ~44% MFU; doing the
> arithmetic in two-limb bf16 (recovering ~16 mantissa bits) runs at bf16 speed.
> An **offline numpy sim (zero remote spend)** proves all-3-GEMMs bf16x2 clears the
> gate with margin (**worst rel-L2 = 7.7e-6 « 2e-5**, over seeds [42,0,21,63,84]),
> even with error compounding across the 3 chained GEMMs + the SiLU nonlinearity.

Expected: ~1.2–1.28x on the compute floor (sibling-proven: rmsnorm_matmul
1.066x→1.363x = 1.28x; add_rmsnorm_matmul ~1.2x from the same split). Secondary
lever (M-blocking) is an *enabler/insurance* — measured and applied only if the
post-bf16x2 profile shows DMA climbing off "hidden." Transpose-elimination and
off-PE transpose are **costed/precedent REJECTS** (below).

---

## 1. Starting point and the Phase-2 mandate

- **Best correct kernel:** `runs/swiglu_v1.py`, PROMOTED, full-5-seed PASS,
  rel-L2 = **6.36e-7** (≈31× under the 2e-5 gate — the fp32 floor here is
  remarkably low; see §5). Latency **2.2079 ms → 0.939x**.
- **Measured metrics (full-5-seed, `profile/swiglu_v1_full5.txt`):**
  `MFU=44%  PE=95%  Vec=7%  Scl=4%  DMA=49%  HBMrd=607MB  HBMwr=17MB`.
- Phase-2 goal (from the prompt): identify the real bottleneck, enumerate
  directions, rank by benefit-vs-risk, explore each ≤5 iterations, keep
  before/after latency + profiling evidence, never regress correctness.

---

## 2. Profile-driven bottleneck read

**The kernel is PE-bound, and the PE is stuck at the trn2 fp32-emulation floor.**

- `PE=95%` — the Tensor Engine is busy essentially all the time.
- `MFU=44%` — but it only achieves 44% of *peak* FLOPs. On trn2 the systolic array
  is **bf16-native**; a correct fp32 matmul is emulated with multiple internal bf16
  passes, capping MFU at ~44–46%. This exact signature (`PE≈95%, MFU≈44%`) recurs on
  every fp32 sibling (matmul, rmsnorm_matmul, add_rmsnorm_matmul). It is not a
  scheduling defect — it is the price of fp32 arithmetic on this hardware.
- `DMA=49%` — DMA is active ~1.08 ms, comfortably **hidden** under the ~2.10 ms of
  PE-active time. `HBMwr=17MB ≈ output-only (16MB)` confirms v1 does **not** spill h
  (unlike the baseline's `_spill_163`/`_reload_166`, ~+100MB traffic).
- `Vec=7%, Scl=4%` — the fused SiLU + the multiply are trivial; fully hidden.

**Consequence for direction-picking.** Because the kernel is PE-bound with DMA
hidden, *reducing DMA cannot speed it up* — the phase-1 memory's guess that
"M-blocking to amortize weight DMA" is the phase-2 win is **wrong for a PE-bound
kernel**. The only way below ~2.1 ms is to make the PE do the same math in fewer
cycles, which means **not paying the fp32 emulation tax**.

### 2.1 Cost-model accounting of PE work (per 128-row M-tile, trn2)

Using the instruction cost model (Formula A: Matmul latency ∝ moving-free elements;
`kernel-cost-analysis`), per M-tile v1 issues:

| Work | # matmuls | moving | element-cycles | share |
|---|---|---|---|---|
| up GEMM   | 48 (6 N-chunks × 8 K) | 512 | 24576 | 31.6% |
| gate GEMM | 48                    | 512 | 24576 | 31.6% |
| down GEMM | 48 (2 Kout × 24 N)    | 512 | 24576 | 31.6% |
| x-transpose | 8  | 128 | 1024 | 1.3% |
| h-transpose | 24 | 128 | 3072 | 3.9% |
| **useful GEMMs** | | | **73728** | **94.7%** |
| **all transposes** | | | **4096** | **5.3%** |

**The three GEMMs are ~95% of PE work; the transposes are ~5%.** Attacking the
GEMMs (bf16x2) dominates any transpose optimization by an order of magnitude.

---

## 3. Why the prompt's "fusion" directions are already spent

The phase-2 prompt lists candidate directions; v1 already realizes each:

| Prompt direction | Status in v1 |
|---|---|
| share the single x load across up+gate | **Done** — x transposed **once** per M-tile into 8 shared xT sub-tiles, consumed as the stationary operand by both up and gate. |
| fuse SiLU gate + multiply into down staging | **Done** — one `nisa.activation(op=nl.silu)` + one `nl.multiply` produce h resident. |
| keep (M,N) intermediate in SBUF | **Done** — `h_sbuf[128,3072]` (12 KB/part) stays resident; no spill (`HBMwr=17MB`). |
| tile K/N to keep PSUM banks full (free ≤ 512) | **Done** — up/gate over 6 N-chunks of 512; down over 2 K-out chunks of 512; K-accum in fp32 PSUM banks. |

So there is **no cheap fusion win left**; the remaining PE cost is intrinsic
arithmetic. This is the honest reason v1 (0.939x, less work) is essentially tied
with the baseline (which spills h and re-transposes x): **both are pinned at the
fp32 PE floor (~2.1 ms).** The floor is the enemy, and only bf16x2 moves it.

---

## 4. The lever: compensated bf16x2 3-product split (offline-GATED)

### 4.1 The technique (sibling-proven)

Each fp32 operand is kept as two bf16 limbs; three bf16 products accumulate in fp32
PSUM (the negligible lo⊗lo cross term is dropped):

```
a_hi = bf16(a),  a_lo = bf16(a - a_hi)        # round-to-nearest-even, ~16 mantissa bits
b_hi = bf16(b),  b_lo = bf16(b - b_hi)
a @ b  ≈  a_hi@b_hi + a_hi@b_lo + a_lo@b_hi     # fp32 accumulation
```

Applied to all three swiglu GEMMs (stationary = the transposed activation limbs,
moving = the weight limbs):

- **up:**   xT_hi/xT_lo (stationary, shared) ⊗ w_up_hi/w_up_lo (moving)
- **gate:** xT_hi/xT_lo (stationary, **same limbs as up**) ⊗ w_gate_hi/w_gate_lo
- **down:** hT_hi/hT_lo (stationary) ⊗ w_down_hi/w_down_lo (moving)

The fp32 identity-matmul transposes stay **exact fp32**; we split the fp32 result
into limbs afterward (splitting after an exact transpose == splitting before —
proven on add_rmsnorm_matmul v3).

### 4.2 Offline numerical gate — the decisive, zero-spend evidence

`runs/offline_bf16_split_sim.py` reproduces the exact scored input (seed-42 draw of
x, w_up, w_down, w_gate in reference order), computes the fp32 reference, and models
the bf16x2 split on each GEMM. Result (`profile/swiglu_offline_bf16x2_sim.txt`),
worst rel-L2 over seeds [42,0,21,63,84]:

| Variant | worst rel-L2 | verdict |
|---|---|---|
| **all 3 GEMMs bf16x2** | **7.72e-6** | **PASS** (2.6× under gate) |
| up+down bf16x2, gate fp32 | 6.30e-6 | PASS |
| up+gate bf16x2, down fp32 | 6.32e-6 | PASS |
| only down bf16x2 | 4.45e-6 | PASS |
| all 3 **plain bf16** (reject) | 4.08e-3 | **FAIL** |

**Key finding — compounding is benign.** The worry (phase-1 memory) was that bf16x2
error compounds across 3 chained GEMMs + the SiLU. It does grow monotonically
(4.4e-6 → 6.3e-6 → 7.7e-6 as more GEMMs go bf16x2), but even the all-3 case is
**2.6× under the gate**. Plain single-limb bf16 fails by 200×, confirming the split
is what makes it feasible. **The aggressive all-3 variant is the target**; the
partial variants are a ready fallback ladder (§6) if the device surprises.

### 4.3 Expected speedup and DMA headroom

- **Compute:** siblings give the empirical multiplier for fp32→bf16x2-3product:
  rmsnorm_matmul **1.28x** (1.066→1.363), add_rmsnorm_matmul **~1.2x**. Applying
  ~1.2–1.28x to v1's 0.939x ⇒ **~1.13–1.20x** projected (2.208 ms → ~1.73–1.84 ms).
- **DMA stays hidden.** bf16x2 must load fp32 weights (the lo limb needs
  `w - bf16(w)`), so `HBMrd` is unchanged (~607MB) and DMA-active stays ~1.08 ms.
  Projected PE-active ~1.73 ms > 1.08 ms ⇒ **DMA remains hidden** — the bf16x2 win
  is real, not a DMA mirage. (If a later push drops PE below ~1.1 ms, DMA becomes
  the wall — that is exactly when M-blocking, §5.2, earns its place.)
- **SBUF fits.** Weights are streamed, so only a single chunk's limbs are live
  (tiny); xT limbs (bf16, ~4 KB), hT limbs (bf16, ~12 KB), h_sbuf (fp32, 12 KB) all
  fit the 208 KB/part budget with room to spare.

---

## 5. Ranked directions (benefit vs risk)

### D1 — bf16x2 3-product on all three GEMMs  ★ PRIMARY
- **Benefit:** high (~1.2–1.28x, breaks the fp32 floor). **Risk:** low — offline-
  gated PASS (7.7e-6), primitive proven on two siblings on this exact remote.
- **Correctness caveat (from add_rmsnorm_matmul):** on-device rel-L2 can combine the
  offline bf16x2 error with the fp32-emulation floor **in quadrature**. Here the
  fp32 floor is v1's measured **6.36e-7** — negligible next to 7.7e-6 — so predicted
  on-device ≈ `sqrt(7.7e-6² + 6.4e-7²) ≈ 7.7e-6`, still 2.6× under gate. (Contrast
  add_rmsnorm, whose fp32 floor was 1.46e-5 and dominated.) Robust, but **must be
  confirmed with the full 5-seed run before promoting.**
- **Iterations (≤5):** (1) build swiglu_v2 = v1 + limb split on all 3 GEMMs; verify
  `--fast`. (2) full-5-seed verify + profile; confirm MFU rises (~44%→~55–70%) and
  DMA stays hidden. (3) if rel-L2 surprises, walk the §6 fallback ladder. (4–5)
  spare for a limb-order / eviction-fold micro-tune.

### D2 — M-blocking (B M-tiles per weight stream)  ◑ SECONDARY / ENABLER
- **Benefit:** recovers the ~5% PE-idle (95%→~100%) and cuts weight-DMA volume ~B×,
  keeping DMA hidden after D1 speeds the PE up. **Alone (fp32) it is ~≤1.05x** — a
  polish, not a floor-breaker (DMA is already hidden). **Risk:** medium — PSUM
  pressure: up_acc+gate_acc = 2 banks/M-tile, so B M-tiles need 2B of 8 banks (B≤4);
  h_sbuf grows to B×12 KB. Only pursue if D1's profile shows DMA% climbing toward
  the PE wall. **Decision gated on the post-D1 profile, not assumed.**
- **Iterations (≤2, only if triggered):** try B=2 then B=4; keep the better.

### D3 — eliminate the h-transpose via layout swap  ✗ REJECT (cost-model)
- Idea: emit up/gate directly in `[n_in, m]` layout so the down GEMM consumes it
  without the 24 h-transposes. **Cost model says this loses:** the h-transpose costs
  ~6144 element-cycles/M-tile, but emitting up/gate transposed turns each into 24
  small (moving=128) matmuls, adding ~2×(fill overhead) ≈ +30720 element-cycles.
  **Net +30720 (a loss).** The h-transpose is only 3.9% of PE work; not worth a
  structural rewrite. Record the cost math; do not implement.

### D4 — off-PE transpose (dma_transpose / nc_transpose)  ✗ REJECT (precedent)
- Siblings closed these: `dma_transpose` is **documented fp32-ineligible** (needs a
  2-byte dtype); `nc_transpose(engine=vector)` is limited to [32,32] (each [128,128]
  → 16 sub-transposes → far more Vector ops). Transposes are 5% of PE anyway.
  Record as investigated-and-closed; do not re-probe.

---

## 6. Numerical safety — the fallback ladder

If the on-device full-5-seed rel-L2 for all-3-bf16x2 exceeds the gate (unexpected,
given 7.7e-6 offline + negligible fp32 floor), step down the offline-gated ladder,
each variant already PASS in the sim, trading a little speed for margin:

1. **all 3 bf16x2** — 7.7e-6 (target).
2. **up+down bf16x2, gate fp32** — 6.3e-6 (gate feeds the SiLU → most error-
   sensitive; keeping it fp32 is the natural first retreat).
3. **only down bf16x2** — 4.4e-6 (down is 1/3 of PE MACs; smallest, safest win).
4. **fp32 (v1)** — always the correctness floor to fall back to.

All limb construction uses round-to-nearest-even (`nl.copy(dtype=nl.bfloat16)`), the
exact cast the sim models; the residual `a - a_hi` is exact in fp32 for these O(1)
normals. Never regress below v1's PASS.

---

## 7. Implementation sketch — `swiglu_v2` (bf16x2, all-3)

Structurally v1 with limb splits inserted; the loop nest and layout are unchanged.

1. **Setup:** identity [128,128] loaded once (fp32, for exact transposes), as in v1.
2. **Per M-tile:**
   - Load x tile `[m_in,1024]`; transpose once into 8 fp32 xT sub-tiles (exact
     identity matmul, `is_transpose=True`), then split each into
     `xT_hi/xT_lo` (bf16). These limbs are **shared by up and gate**.
   - **up / gate**, per N-chunk (512), K-accum over 8 K-tiles into two fp32 PSUM
     banks: load fp32 `w_up`/`w_gate` chunk `[k_in,512]`, build `w_*_hi/w_*_lo`
     (bf16) on-chip, issue the **3 bf16 products** (`xT_hi@w_hi + xT_hi@w_lo +
     xT_lo@w_hi`) into the accumulator.
   - PSUM→SBUF copy; fused `nl.silu(gate)`; `nl.multiply` → `h_sbuf[128,3072]` fp32
     resident (identical to v1).
   - Transpose h into 24 fp32 hT sub-tiles; split into `hT_hi/hT_lo` (bf16).
   - **down**, per K-out chunk (512), N-accum over 24 N-tiles: load fp32 `w_down`
     chunk `[n_in,512]`, build `w_down_hi/w_down_lo`, issue the **3 bf16 products**
     into the accumulator; copy; store to `v5`.
3. **dtypes:** all limbs `nl.bfloat16`; all PSUM accumulation and the SiLU/multiply
   stay fp32; transposes stay fp32.

Weights are still streamed (B=1) — the split changes arithmetic, not the loop
structure, keeping the diff from v1 small and reviewable.

---

## 8. Deliverable and success criteria

- **Deliverable:** `runs/swiglu_v2.py` (bf16x2 all-3-GEMMs), the offline sim
  (`runs/offline_bf16_split_sim.py`, already written) + its evidence
  (`profile/swiglu_offline_bf16x2_sim.txt`), a `benchmark.csv` row and a
  `candidates.jsonl` entry with parent `swiglu_v1`.
- **Score:** `python3
  ../../verify.py --op swiglu --candidate runs/swiglu_v2.py --fast` (then drop
  `--fast` for the promoting 5-seed run).
- **Success:** full-5-seed PASS (rel-L2 < 2e-5; expect ~7.7e-6) **and** speedup
  > 1.0x (projected ~1.13–1.20x). Keep v1 as the fp32 fallback.
- **Evidence to capture:** before/after latency, MFU (expect a clear rise from 44%),
  PE/DMA %, HBM bytes; the fallback-ladder decision if triggered.
- **Iteration budget:** D1 ≤5 iters (primary); D2 ≤2 iters (only if the post-D1
  profile shows DMA climbing off "hidden"); D3/D4 are recorded rejects, not explored.
