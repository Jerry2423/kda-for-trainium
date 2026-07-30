# matmul Phase 3 — regime/shape specialization draft

## Starting point (best correct kernel)

`runs/matmul_v2_b4.py` (Phase 2): M-blocked B=4 fp32 GEMM. **1.017x (13.3517 ms)**,
all 5 seeds pass. Profiler: **PE=100%, MFU=49%, DMA=31%, HBMrd=2097 MB**. Fully
PE-bound.

## The central Phase-3 finding: the kernel is at the fp32 PE floor

Phase 3's premise is "specialize where the measured win justifies the complexity."
The analysis (profile/matmul_phase3_analysis.txt, grounded in the cost model +
knowledgebase) shows there is **very little to specialize**, because we are already
at the hardware floor for this precision:

- The trn2 PE array is **bf16-native**; **fp32 matmul runs at ~2 passes** (half rate).
  The naive cost-model floor (6.62 ms) is a *bf16-equivalent*; the true **fp32 PE
  floor ≈ 13.1 ms**. B=4 at 13.35 ms is **~98% of that floor**.
- MFU is measured vs the bf16 peak, so a correct fp32 GEMM is **capped near ~50% MFU
  by construction**. Our 49% is essentially the ceiling — not an inefficiency to fix.
- The 2e-5 L2 gate forbids bf16/tf32, so the fp32 floor is binding.
- This is a **single fixed shape with all tiles full** (32·128=4096, 40·128=5120,
  24·512=12288 — no remainders). There is **no shape regime / edge tile to
  specialize** — every tile is identical.

Absolute conceivable ceiling (floor, zero transpose, zero overhead) ≈ **1.036x**;
current 1.017x → **< 3% total headroom, most unreachable.**

## Directions (ranked by measured-win / complexity)

Per the user's decision, run small **low-risk** attempts, keep only measured wins,
and otherwise document the floor. No direction below is expected to exceed a few %.

### D1 — PSUM→SBUF eviction / store engine choice  [LOW value, low risk]
At PE=100%, the PSUM→SBUF copies (`nl.copy`) and stores may now sit on/near the
critical path. Try steering the copy to an idle engine (ScalarE or VectorE via
`nisa.tensor_copy(engine=...)`) so it doesn't contend with the Tensor Engine or
serialize eviction. Precedent: `bc877398` (move a copy from VectorE to ScalarE to
rebalance). Expected: 0–1%. Keep only if it beats 1.017x full-run.

### D2 — re-confirm B with the CORRECTED SBUF budget  [LOW value, low risk]
Phase 2 rejected B=8 partly on a mis-stated SBUF budget (used trn1's 192 KB; trn2 is
actually **224 KB / 208 KB usable**). B=8's full-run regression was likely PSUM-bank
exhaustion (all 8 banks) + scheduling, not an SBUF spill — but re-measure B=8 (and
B=16, which also divides 32) once with the correct understanding to confirm B=4 is
truly best. Expected: confirm B=4, or a marginal shift. Cheap to check.

### D3 — transpose scheduling  [VERY LOW value]
The 1280 lhs-transpose matmuls are 0.5% of runtime. `dma_transpose` is **ineligible**
(needs 2-byte dtype; fp32 is 4 bytes). DVE transpose of 128×128 is slower than the
current TensorE transpose. The transpose is **structurally unavoidable** (lhs has k
on the free axis; nc_matmul needs k on partition). So there is nothing worth doing
here beyond confirming the transpose already overlaps. Effectively **reject**.

### Rejected outright
- bf16/tf32 downcast — breaks 2e-5 (this is *why* the floor is 13.1 ms).
- Wider tiles — already at hardware caps (stationary 128, moving 512, contraction 128).
- Shape-regime split / edge-tile specialization — no regimes exist (single full shape).
- bf16x3 / split-fp32 emulation — the user chose the safe path; 3 bf16 passes would
  likely be slower than fp32's 2 passes and hitting 2e-5 is uncertain. Out of scope.

## Plan of attack (≤5 iters per direction)

1. D1: try the eviction/store engine tweak on B=4; `--fast` read, then full 5-seed if
   promising. Keep only if > 1.017x.
2. D2: re-measure B=8 and B=16 full-run for a clean comparison; keep the fastest B.
3. Record every candidate (kept/rejected) in benchmark.csv + candidates.jsonl
   (parent = matmul_v2_b4) + profile/. Never regress correctness (all 5 seeds).
4. Whatever the outcome, the Phase-3 deliverable includes the fp32-floor analysis so
   the near-optimality is documented, not just asserted.

## Target

Realistic: **hold ≥ 1.017x**, capture any measured win up to the ~1.036x ceiling.
Honest exit: if no candidate beats B=4, promote B=4 and report the fp32-floor
evidence explaining why further gain needs a precision change the gate forbids.

## Correctness / evidence contract (unchanged)
- fp32 throughout; all 5 seeds `[0,21,42,63,84]` pass relative-L2 `< 2e-5`.
- Single `@nki.jit def kernel(v1,v2)`; candidates in `runs/`; never edit baseline/reference.
- Parent DAG in `candidates.jsonl`; per-direction profiling under `profile/`; full
  5-seed (not just --fast) before promoting (Phase-2 lesson: --fast can mis-rank).
