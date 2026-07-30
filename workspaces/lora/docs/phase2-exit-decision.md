# lora — Phase 2 Exit Decision

## Outcome: PROMOTE `lora_v3_bf16_split` (1.297x); keep `lora_v2_mblk4` (0.988x) as fp32 fallback

The phase-2 arc took lora from **0.382x → 1.297x** (a 3.4x wall-clock improvement over the
phase-1 kernel) by porting the sibling matmul ladder to the shape-identical base GEMM and
grafting the cheap fused low-rank residual on top:

| kernel | precision | latency (full 5-seed) | speedup | worst rel-L2 | HBMrd | role |
|--------|-----------|-----------------------|---------|--------------|-------|------|
| lora_v1 (phase 1) | fp32 | 38.3562 ms | 0.382x | 4.874e-7 | 7813 MB | superseded |
| **lora_v2_mblk4 (D1)** | fp32 | 14.8385 ms | 0.988x | 4.874e-7 | 2150 MB | **fp32 fallback** |
| **lora_v3_bf16_split (D2)** | bf16x2 base | **11.3034 ms** | **1.297x** | 6.240e-7 | 2112 MB | **PROMOTED** |

## Why this is the stop

- **AC-1 (hard gate):** both scored kernels PASS all 5 seeds. D1 worst rel-L2 4.874e-7
  (fp32 floor, byte-identical to lora_v1). D2 worst rel-L2 6.240e-7 = the device quadrature
  `sqrt(fp32_floor^2 + composite_bf16^2)` = 6.261e-7 within 0.34% (backed-out bf16 3.897e-7
  == offline composite 3.930e-7 within 0.9%) — the split behaves exactly as the offline sim
  modeled. Both ~32x under 2e-5.
- **D1 collapsed the HBM traffic** 7813→2150 MB (3.6x) and flipped the op from DMA-co-
  saturated (89%) to PE-bound with DMA idle (33%), landing on the sibling matmul_v2_b4 fp32
  floor (1.017x, MFU 49%). HBM write byte-identical 201 MB, no spill (AC-2/AC-2.1).
- **D2 broke the fp32 floor** with the base-only bf16x2 3-product split: TRUE PE-active
  −27.4% (14.79→10.73 ms), latency −23.8% vs same-session D1, HBM read essentially unchanged
  (2112 MB — bf16 limbs built on-chip and held resident, no reload trap), no spill. Both
  swiglu-loss conditions absent (high per-instruction fp32 rate on the dense moving-512 base
  GEMM + resident limbs), so the split is a real PE reduction (BL-20260710).
- **Guards held before spend:** the offline sim (AC-4) authorized D2 (composite 3.930e-7 <
  1.3e-5) with a fail-closed independent-reference control (raises, survives `python -O` —
  Codex-verified); the extended host layout check (AC-5) covers the N_CHUNK=512 4-sub-tile
  store (worst 4.9e-7) plus a reversed-order negative control; Codex (high effort)
  independently confirmed the dilution (11.4x), quadrature, and fail-closed arguments.

## D3 / D4 — model-based rejects (DEC-2; no DMA/blocking gap to measure against)

D2 is cleanly PE-bound (latency 11.30 ms ≈ PE-active 10.73 ms, DMA idle 47.6%, no spill), so
neither contingency has a gap to close; both are recorded rejects (candidates.jsonl).

- **D3 (resident `b`):** `b` is 6.29 MB; the current load-once-per-N-chunk gives 50.3 MB of
  reload traffic, so making it resident saves 44.0 MB = **2.09% of D2's 2112 MB HBM read**.
  Removing 2.1% of already-hidden DMA on a PE-bound op cannot move the wall.
- **D4 (B∈{2,8}):** carries the sibling matmul finding on this identical base GEMM — B=4
  optimal, B=2 0.983x (under-amortized), B=8 0.968x, B=16 0.519x (SBUF/PSUM pressure). lora's
  2-level (m_hi,m_lo) index makes B=4 == m_lo the natural arithmetic-free block; B=8 needs 8
  PSUM acc banks (zero headroom for transpose/down-proj) and doubles D2's resident bf16 lhs
  limbs — exactly the pressure regime the sibling B=8 regressed on.

## Evidence
`benchmark.csv` (D1, D2 rows), `candidates.jsonl` (D1, D2, D3-reject, D4-reject; parent DAG
lora_v1→lora_v2_mblk4→lora_v3_bf16_split), `profile/lora_v2_mblk4_digest.{md,txt}`,
`profile/lora_v3_bf16_split_digest.{md,txt}`, `runs/offline_lora_bf16_split_sim.py`,
`runs/_layout_check.py` (extended), `runs/lora_v2_mblk4.py`, `runs/lora_v3_bf16_split.py`.
