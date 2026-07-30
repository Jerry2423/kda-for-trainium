# swiglu — Phase 3 draft (regime / shape specialization)

## 0. TL;DR — and a critical status correction

**The phase-2 loop never delivered its primary kernel.** The RLCR loop derailed:
it retired a stale phase-1 loop, Codex surfaced a real *harness-seed* gap (all five
NKIBench "seeds" draw identical seed-42 inputs — see §6), and the agent stopped to
ask for direction. The driver committed only `draft-phase2.md` + `plan-phase2.md`
and marked `phase2.done`. **`swiglu_v2.py` (the compensated bf16x2 split) was never
built.** The only kernel that exists is `swiglu_v1` at **0.939x — still slower than
the 2.0742 ms baseline.**

Consequently the highest-value, still-undone work is exactly the lever phase 2
*defined but never implemented* — and it is also the lever this phase-3 prompt and
the progress memory both name as the phase-3 floor-breaker:

> **Part A (must-do, the floor-breaker): `swiglu_v2` = `swiglu_v1` + compensated
> bf16x2 3-product split on all three GEMMs.** The trn2 PE is bf16-native and
> emulates fp32 at ~44% MFU; two-limb bf16 arithmetic (recovering ~16 mantissa
> bits) runs at bf16 speed. An **offline numpy sim (zero remote spend)** already
> proves all-3-GEMMs bf16x2 clears the gate with margin — worst rel-L2 **7.72e-6 «
> 2e-5** over seeds [42,0,21,63,84], even with error compounding across the three
> chained GEMMs + the SiLU. The idiom is proven on this exact remote (rmsnorm_matmul
> 1.28×, add_rmsnorm_matmul 4.632×). Projected **~1.13–1.20×** (2.208 ms → ~1.73–1.84 ms).

> **Part B (the genuine phase-3 regime specialization, layered on v2): M-tile-block
> regime, gated on the post-v2 profile.** Once bf16x2 unblocks the PE floor, PE-active
> drops ~20% while weight-DMA is unchanged; if the re-profile shows DMA climbing off
> "hidden," specialize the M-tile regime (process B M-tiles per weight stream to
> amortize weight DMA B×; the matmul sibling found B=4 optimal). This is the "specialize
> only where the *measured* win justifies the complexity" mandate. Two important
> **negative** regime findings are pre-established (§4): the shapes are exact
> 128-multiples so there are **no edge/ragged tiles to specialize**, and the 512-wide
> free chunk is already the fp32-PSUM-bank optimum.

Do **not** do Part A and skip Part B, and do **not** invert them: without Part A,
phase 3 ships a "shape-specialized" kernel that is still stuck at 0.939× (below
baseline) — a failure. Part B only earns its complexity if v2's profile shows the
DMA wall; otherwise it is a documented, measured no-op.

---

## 1. Starting point

- **Best correct kernel:** `runs/swiglu_v1.py`, PROMOTED, fp32 throughout.
  Full-run rel-L2 = **6.36e-7** (≈31× under the 2e-5 gate). Latency **2.2079 ms →
  0.939×**. (Caveat: the "full 5-seed" run is seed-42 ×5 — see §6.)
- **Measured metrics (`profile/swiglu_v1_full5.txt`):**
  `MFU=44%  PE=95%  Vec=7%  Scl=4%  DMA=49%  HBMrd=607MB  HBMwr=17MB`.
  → PE-bound at the trn2 fp32-emulation floor; DMA (~1.08 ms) hidden under ~2.10 ms
  of PE-active time; `HBMwr≈output-only` confirms no h spill.
- **Structure (v1):** M-outer, B=1. Per 128-row M-tile: load x `[m_in,1024]`,
  transpose **once** into 8 shared fp32 `xT` sub-tiles (identity-matmul), consumed as
  the stationary operand by **both** up and gate; up/gate over 6 N-chunks of 512
  (K-accum 8 tiles into two fp32 PSUM banks); PSUM→SBUF copy → fused `nl.silu` →
  `nl.multiply` into a **resident** `h_sbuf[128,3072]` (no spill); transpose h into 24
  fp32 `hT` sub-tiles; down over 2 K-out chunks of 512 (N-accum 24 tiles) → store.
  Weights streamed from HBM.
- **Assets already on disk (from phase 2, reusable now):**
  - `runs/offline_bf16_split_sim.py` — the zero-spend numerical gate (verified to
    reproduce today).
  - `profile/swiglu_offline_bf16x2_sim.txt` — its decisive evidence.
  - `docs/draft-phase2.md` / `docs/plan-phase2.md` — the full bf16x2 design and
    cost-model accounting (Part A is essentially executing that plan).

---

## 2. Where time goes (the phase-3 "structure" analysis)

### 2.1 PE cost accounting (per 128-row M-tile, trn2 fp32, from the cost model)

| Work | # matmuls | moving | element-cycles | share |
|---|---|---|---|---|
| up GEMM   | 48 (6 N-chunks × 8 K) | 512 | 24576 | 31.6% |
| gate GEMM | 48                    | 512 | 24576 | 31.6% |
| down GEMM | 48 (2 Kout × 24 N)    | 512 | 24576 | 31.6% |
| x-transpose | 8  | 128 | 1024 | 1.3% |
| h-transpose | 24 | 128 | 3072 | 3.9% |
| **three GEMMs** | | | **73728** | **94.7%** |
| **all transposes** | | | **4096** | **5.3%** |

**The three GEMMs are ~95% of PE work.** The only lever that moves a 95%-GEMM,
PE-bound, fp32-floored kernel is *not paying the fp32 emulation tax* → bf16x2. This is
why Part A precedes any structural regime work.

### 2.2 How the cost balance shifts *after* bf16x2 (this is the phase-3 pivot)

bf16x2 replaces each fp32 GEMM matmul (internally multiple bf16 passes at ~44% MFU)
with **3 explicit bf16 products** at native bf16 rate. The transposes stay **exact
fp32** (we split into limbs *after* the transpose). So post-v2:

- GEMM element-cycles fall by the fp32→bf16 rate ratio (~1.2–1.28× empirically from
  siblings), but the transposes do **not** — their *relative* share roughly doubles
  (5.3% → ~9–11% of the now-smaller PE total). This makes the h-transpose worth a
  fresh look in Part B (§4.3), though the phase-2 cost model still predicts reject.
- Weight-DMA is **unchanged**: bf16x2 must load the fp32 weights to build the lo limb
  (`w - bf16(w)`), so `HBMrd≈607MB` and DMA-active ≈1.08 ms hold. With PE-active
  projected ~1.73 ms, DMA **stays hidden** — but the margin shrinks from ~1.0 ms to
  ~0.65 ms. **This shrinking margin is the trigger condition for the M-block regime
  (§4.1)** — hence "gated on the post-v2 profile," not assumed.

---

## 3. Part A — the floor-breaker: `swiglu_v2` (bf16x2, all-3-GEMMs)

### 3.1 The technique (sibling-proven; idiom lifted from `add_rmsnorm_matmul_v3`)

Each fp32 operand → two bf16 limbs; three bf16 products accumulate in fp32 PSUM
(drop the negligible lo⊗lo term):

```
a_hi = bf16(a),  a_lo = bf16(a - a_hi)        # round-to-nearest-even (nl.copy dtype=bf16)
b_hi = bf16(b),  b_lo = bf16(b - b_hi)
a @ b  ≈  a_hi@b_hi + a_hi@b_lo + a_lo@b_hi     # fp32 PSUM accumulation
```

Applied to all three swiglu GEMMs:
- **up:**   `xT_hi/xT_lo` (stationary, shared) ⊗ `w_up_hi/w_up_lo` (moving)
- **gate:** `xT_hi/xT_lo` (stationary, **same limbs as up** — split once, reuse) ⊗ `w_gate_hi/w_gate_lo`
- **down:** `hT_hi/hT_lo` (stationary) ⊗ `w_down_hi/w_down_lo` (moving)

The identity-matmul transposes stay **exact fp32**; split into limbs *afterward*
(splitting after an exact transpose == splitting before — proven on
add_rmsnorm_matmul v3). The fused `nl.silu` + `nl.multiply` producing `h_sbuf`
stay **fp32**, identical to v1.

### 3.2 The decisive, zero-spend numerical gate (already on disk)

`runs/offline_bf16_split_sim.py` reproduces the exact seed-42 input draw, computes the
fp32 reference, and models the bf16x2 split on each GEMM. Re-verified today
(`profile/swiglu_offline_bf16x2_sim.txt`), worst rel-L2 over seeds [42,0,21,63,84]:

| Variant | worst rel-L2 | verdict |
|---|---|---|
| **all 3 GEMMs bf16x2** | **7.72e-6** | **PASS** (2.6× under gate) |
| up+down bf16x2, gate fp32 | 6.30e-6 | PASS |
| up+gate bf16x2, down fp32 | 6.32e-6 | PASS |
| only down bf16x2 | 4.45e-6 | PASS |
| all 3 **plain bf16** (reject) | 4.08e-3 | **FAIL** (200× over) |

**Compounding is benign.** Error grows monotonically as more GEMMs go bf16x2
(4.4e-6 → 6.3e-6 → 7.7e-6) but the all-3 case is still 2.6× under gate. It is also
essentially **seed-independent** (7.706e-6…7.722e-6 across all five seeds) — a fact
that matters directly for the harness-seed caveat (§6).

**On-device prediction.** Per the add_rmsnorm_matmul precedent, on-device rel-L2 ≈
offline-bf16x2-error ⊕ fp32-emulation-floor in quadrature. Here the fp32 floor is
v1's measured **6.36e-7**, negligible next to 7.72e-6, so predicted on-device
≈ `sqrt(7.72e-6² + 6.4e-7²) ≈ 7.7e-6` — unlike add_rmsnorm_matmul, whose 1.46e-5 floor
dominated. Comfortable, but **confirm on the full run before promoting**.

### 3.3 Implementation sketch — minimal diff from v1

Structurally v1 with limb splits inserted; loop nest and layout unchanged.

1. **Setup:** identity `[128,128]` fp32 loaded once (exact transposes), as v1.
2. **Per M-tile:**
   - Load x `[m_in,1024]`; transpose once → 8 fp32 `xT` sub-tiles; split each into
     `xT_hi/xT_lo` (bf16). **Shared by up and gate.**
   - **up / gate**, per N-chunk (512), K-accum over 8 K-tiles into two fp32 PSUM banks:
     load fp32 `w_up`/`w_gate` chunk `[k_in,512]`, build `w_*_hi/w_*_lo` (bf16) on-chip
     (`hi = copy(w, bf16)`; `res = tensor_tensor(w, hi, subtract)`; `lo = copy(res, bf16)`),
     issue the **3 bf16 products** (`xT_hi@w_hi + xT_hi@w_lo + xT_lo@w_hi`) into the accumulator.
   - PSUM→SBUF copy; fused `nl.silu(gate)`; `nl.multiply` → `h_sbuf[128,3072]` fp32
     resident (identical to v1).
   - Transpose h → 24 fp32 `hT` sub-tiles; split into `hT_hi/hT_lo` (bf16).
   - **down**, per K-out chunk (512), N-accum over 24 N-tiles: load fp32 `w_down` chunk
     `[n_in,512]`, build `w_down_hi/w_down_lo`, issue the **3 bf16 products**; copy; store.
3. **dtypes:** all limbs `nl.bfloat16`; all PSUM accumulation + SiLU/multiply + all
   transposes stay fp32.

**SBUF fit (well within the ~200 KB/partition budget):** `h_sbuf` fp32 12 KB +
`hT_hi/hT_lo` bf16 12 KB + `xT_hi/xT_lo` bf16 4 KB + streamed weight-chunk limbs
(~1 KB) + fp32 transients (~10 KB) ≈ 40 KB/partition. Weights are streamed (B=1),
so only one chunk's limbs are live at a time. The bf16 limbs of the whole weight are
**not** held resident (unlike add_rmsnorm_matmul, whose 2048-wide weight fit) —
swiglu's three weights are too big, so we rebuild limbs per streamed chunk.

### 3.4 Iterations (≤5)
1. Build `swiglu_v2` = v1 + all-3 limb split; `--fast` verify (correctness + latency).
2. Full run + profile: confirm rel-L2 ≈7.7e-6 PASS, MFU rises (44% → ~55–70%),
   PE drops, DMA stays hidden. Record before/after in `benchmark.csv` + `candidates.jsonl`.
3. If rel-L2 surprises, walk the §5 fallback ladder.
4–5. Spare for a limb-order / eviction-fold micro-tune, then hand off to Part B.

---

## 4. Part B — regime / shape specialization (gated on v2's profile)

### 4.1 R1 — M-tile-block regime (B M-tiles per weight stream)  ★ PRIMARY (conditional)
- **What:** process B M-tiles inside one weight-chunk stream so each fp32 weight is
  loaded once and reused across B M-tiles → weight-DMA volume drops ~B×. This is the
  `matmul_v2_b4` lever (matmul sibling: B=4 → 1.017×).
- **Why conditional:** bf16x2 does not change HBM traffic, so if v2's re-profile still
  shows DMA hidden with comfortable margin, M-blocking is ≤1.05× polish and **not worth
  the complexity**. It earns its place only if v2 shows DMA climbing toward the PE wall
  (§2.2 predicts the margin shrinks from ~1.0 ms to ~0.65 ms — plausibly still hidden).
  **Decision is measured, not assumed.**
- **Constraints:** PSUM is the limiter — up_acc + gate_acc = 2 banks/M-tile, so B M-tiles
  need 2B of 8 banks during the up/gate phase → **B ≤ 4**. Per-M-tile activation state
  (`h_sbuf` 12 KB + `hT` limbs 12 KB + `xT` limbs 4 KB ≈ 28 KB) × B: B=4 → ~112 KB/part,
  still within budget. The weight-limb rebuild moves out of the M-loop (built once per
  chunk, reused across the B M-tiles) — which *also* amortizes the limb-construction Vec
  ops, a second-order bonus.
- **Iterations (≤2, only if triggered):** B=2, then B=4; keep the better.

### 4.2 R2 — edge / ragged-tile regime  ✗ NO-OP (documented negative)
- M=4096=32×128, K=1024=8×128, N=3072=24×128 are **all exact multiples of 128**. There
  are **no ragged partition or free edges** anywhere in the kernel — every tile is a full
  128×512 or 128×128. So the classic edge-tile specialization the phase-3 prompt lists
  **does not apply** to this shape. Record as investigated-and-closed; do not implement
  masking/predication.

### 4.3 R3 — free-chunk & transpose-layout regimes  ✗ likely REJECT (re-checked post-bf16x2)
- **Free chunk = 512** is one fp32 PSUM bank (accumulation stays fp32 even under bf16x2),
  so 512 remains the free-dim optimum; no regime to specialize.
- **h-transpose elimination via layout swap (D3 from phase 2):** the phase-2 cost model
  rejected this at fp32 ratios (turning up/gate into 24 moving=128 matmuls costs
  +30720 ec vs the 6144 ec h-transpose saved). Post-bf16x2 the transpose share ~doubles
  (§2.2), so **re-run the cost arithmetic once on v2's actual profile** before final
  reject — but the expectation is it still loses (the extra small-matmul fill overhead
  dwarfs the transpose even at bf16 rates). Off-PE transpose (dma_transpose / nc_transpose)
  stays a **precedent reject** (dma_transpose fp32-ineligible; nc_transpose vector [32,32]).

### 4.4 R4 — mixed-precision GEMM regime (the fallback ladder as a specialization)
- If on-device numerics surprise (unlikely; §3.2), specialize *which* GEMMs are bf16x2 vs
  fp32 per the §5 ladder — a precision-regime specialization that trades a little speed for
  margin. All rungs are offline-PASS.

---

## 5. Numerical safety — the fallback ladder

If v2's on-device full run exceeds the gate (unexpected: 7.7e-6 offline + negligible
fp32 floor), step down the offline-gated ladder (each rung already PASS in the sim):

1. **all 3 bf16x2** — 7.7e-6 (target).
2. **up+down bf16x2, gate fp32** — 6.3e-6 (gate feeds the SiLU → most error-sensitive;
   natural first retreat).
3. **only down bf16x2** — 4.4e-6 (down is 1/3 of PE MACs; smallest, safest win).
4. **fp32 (v1)** — the correctness floor. Never regress below v1's PASS.

Limb construction uses round-to-nearest-even (`nl.copy(dtype=nl.bfloat16)`), the exact
cast the sim models; the residual `a - a_hi` is exact in fp32 for these O(1) normals.

---

## 6. Known caveat — the harness-seed gap (do NOT block on it)

Codex correctly flagged during phase 2 that `adapter/nkibench_case.py` reseeds
`np.random.seed(42)` before **every** input draw (`DEFAULT_INPUT_SEED = 42`), so the
profiler's `multi_seed_seeds=[0,21,42,63,84]` all draw **identical** inputs — the
on-device gate is effectively seed-42 ×5, not five distinct seeds.

**Why this does not block phase 3:** for *this specific bf16x2 change* the seed-diversity
question is already answered off-device — the offline sim (§3.2) draws **real per-seed
inputs** for all five seeds and shows the error is essentially seed-independent
(7.706e-6…7.722e-6, a 0.2% spread). With Gaussian inputs contracted over K=1024 / N=3072,
the relative-L2 concentrates hard, so seed 42 is representative. The offline sim *is* the
multi-seed evidence.

**Scope decision:** fixing the shared adapter (one profiler call per seed) modifies
tooling every op depends on and spends ~5× remote budget; it is **out of scope for a
kernel-optimization phase** and belongs in a separate infra change with user direction.
Phase 3 will (a) rely on the offline multi-seed sim as the numerical gate, (b) run the
standard on-device full run for the latency/PE-metric evidence, and (c) flag this caveat
in `candidates.jsonl` so the promotion evidence is honest about what the on-device gate
actually exercised. Do not silently treat the on-device run as 5-distinct-seed proof.

---

## 7. Deliverables and success criteria

- **Deliverables:**
  - `runs/swiglu_v2.py` — bf16x2 all-3-GEMMs (Part A).
  - `runs/swiglu_v3_mblock.py` — **only if** R1 is triggered by v2's profile and
    measured to win (Part B); otherwise a documented decision to stop at v2.
  - Reuse the existing `runs/offline_bf16_split_sim.py` + `profile/swiglu_offline_bf16x2_sim.txt`.
  - `benchmark.csv` rows + `candidates.jsonl` entries (parent DAG: v2←v1, v3←v2),
    profile evidence under `profile/`.
- **Score command:**
  `python3
  ../../verify.py --op swiglu --candidate runs/<file>.py --fast` (drop `--fast` for the
  promoting run).
- **Success:** on-device full-run PASS (rel-L2 < 2e-5; expect ~7.7e-6) **and** speedup
  > 1.0× — projected **~1.13–1.20×** from v2 alone; a further ~1.0–1.05× if R1 lands.
  Keep v1 as the fp32 fallback.
- **Evidence to capture:** before/after latency; MFU (expect a clear rise from 44%);
  PE/DMA %; HBM bytes; the R1 trigger decision (with the DMA-margin number); the R3
  transpose cost re-check; the §6 seed caveat.
- **Iteration budget:** Part A ≤5 iters (primary, must-do); Part B / R1 ≤2 iters (only if
  v2's profile shows DMA climbing off "hidden"); R2/R3/R4 are recorded decisions, not
  explored unless triggered.
</content>
</invoke>
