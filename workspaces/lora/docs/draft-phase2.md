# lora — Phase 2 draft: profile-driven optimization

## Starting point

`runs/lora_v1.py` — the phase-1 promoted kernel. Full 5-seed PASS, worst rel-L2
**4.874e-7**, p50 **38.3562 ms / 0.382x** (slower than the 14.6645 ms baseline; phase 1
had no speed target). Structure: M-outer over the 2-level `(m_hi, m_lo)` index (8×4=32
tiles), identity-transpose shared `lhs_t`, `a` resident, down-projection `tT=(x@a)^T`
K-accumulated then fp32-copied to SBUF, per n-tile (96 of **width 128**) the base `x@w`
K-accumulated into one PSUM bank with the low-rank `(x@a)@b = tT.T@b` **fused into the
same bank** (no HBM round-trip) before one copy+store.

## Profile diagnosis (lora_v1, remote 5-seed) — `profile/lora_v1_digest.{md,txt}`

| metric | value | reading |
|--------|-------|---------|
| latency p50 | 38.3562 ms | 0.382x |
| MFU | **17.69%** | matmuls are far too narrow |
| PE active | 88.5% (TRUE PE-active/inf **33.94 ms**) | PE-bound; even perfect DMA leaves ~34 ms |
| DMA active | 89.1% | co-saturated by the reload traffic |
| Vec / Scl | 1.3% / 0.1% | epilogue is not the issue |
| HBM read | **7812.7 MB** | 22.7× the ~344 MB single-pass ideal |
| HBM write | 201.3 MB | one output pass (correct, no spill) |
| matmul_instruction_count | 255744 | inflated by N_CHUNK=128 |

Two root causes, **both already solved by the shape-identical sibling `matmul`**
(base GEMM `x@w` is M4096/N12288/K5120 — identical to the `matmul` case):

1. **`N_CHUNK=128` (no M-block).** The base GEMM issues `[128,128]@[128,128]` matmuls
   — MFU 18% vs the sibling `matmul_v1`'s 41% at `N_CHUNK=512`. This alone explains why
   lora_v1 (38 ms) is ~2.4× slower than `matmul_v1` (15.9 ms) for the *same* base GEMM.
   `N_CHUNK=512` cuts base matmuls 122880→30720 (4× fewer, 4× wider).
2. **No M-blocking → `w` re-streamed per M-tile (32×).** `HBMrd=7813MB ≈ 32×(240MB w +
   6MB b)`. The sibling's fix (M-block B=4, load each `w`/`b` K-tile once and reuse
   across the B members) took `matmul` 0.855x→1.017x and dropped its HBMrd 7584→2097 MB.

The sibling `matmul` ladder for this identical base GEMM: fp32 M-outer 0.855x →
**M-block B=4, N_CHUNK=512 = 1.017x** (the fp32 floor) → **bf16x2 3-product split =
1.274x**. lora should follow the same ladder; the fusion/down-projection is grafted on
top and is cheap (see below).

## Theoretical framing (MAC decomposition) — confirms PE-bound, base-dominated

| path | MACs | share |
|------|------|-------|
| base `x@w` | 2.577e11 | **96.6%** |
| down `x@a` | 2.68e9 | 1.0% |
| up `(x@a)@b` | 6.44e9 | 2.4% |
| **low-rank total** | — | **3.4%** |

The base GEMM is 96.6% of the compute; the fused low-rank path is a 3.4% tail. So the
whole phase-2 win is **making the base GEMM fast** — exactly the sibling matmul problem,
plus a cheap fused residual. `kernel-cost-analysis` and the sibling
calibration agree the trn2 PE array is bf16-native and emulates fp32 at ~2 passes, so a
correct fp32 base GEMM caps near ~50% MFU — the bf16x2 split is the lever past that.

## KEY lora-specific finding — the bf16x2 split is SAFER here (offline-verified)

The output magnitude is **dominated by the low-rank term**, not the base:

| term | element std | ‖·‖₂ | fraction of output L2 |
|------|-------------|------|-----------------------|
| base `x@w` | 71.6 | 4.49e4 | **8.8%** |
| low-rank `(x@a)@b` | 813 | 5.10e5 | **99.6%** |

Because `x@a` has variance ~K then `@b` sums over R=128, the low-rank output has
variance ~R·K and swamps the base's ~K by ≈√R = 11×. Offline numpy check (seed 42, the
one input the remote gate scores — full K/R, subset M/N; `/tmp` scratch, mirrors the
sibling `offline_bf16_split_sim.py` RNE method):

- **[A] base bf16x2 3-product + fp32 low-rank → composite rel-L2 = 3.9e-7** (51× under
  the 2e-5 gate). The base-only split error is 4.45e-6, but it is diluted 11× in the
  composite. This is the plan.
- [B] split base AND up-projection → 4.45e-6 (still 4.5× under, but no PE upside — the
  low-rank path is only 3.4% of MACs — so not worth the extra risk).
- [C] plain single-limb bf16 everywhere → 2.35e-3 (fails; the rejected route, scale check).

Conclusion: split ONLY the base GEMM; keep the down-projection `x@a` and the up-projection
`(x@a)@b` in fp32. The composite rel-L2 (3.9e-7) is *below even lora_v1's fp32 floor*
(4.87e-7) because the fp32 low-rank term dominates and the split touches only the small
base — no offline surprise expected on device. (The offline sim will be committed as a
`runs/offline_lora_bf16_split_sim.py` before D2 is scored, ported from the sibling with
the composite base+low-rank forward and the independent-reference fail-closed control.)

## Optimization directions (ranked by expected benefit vs risk)

### D1 — N_CHUNK=512 + M-block B=4, fp32 (the sibling matmul_v2_b4 recipe). PRIMARY.
Expected benefit **HIGH**, risk **LOW** (proven port of an identical base GEMM).

Port `matmul_v2_b4`'s block structure onto lora. The 2-level M-index makes this a
**natural, arithmetic-free** fit: the M-block IS `m_hi` (8 blocks), the B=4 members ARE
`m_lo` (0..3). No flat-index floor-div/mod, no `M_TILES % B` divisibility worry (it is
structurally 8×4). Per m_hi block:
- Build the shared `lhs_t[m_lo, kt] = [k_in, m_in]` for all 4 members (transpose once each).
- Down-projection `tT[m_lo] = (x@a)^T` per member (4 tiles `[R,128]`), fp32, K-accumulated
  then copied to SBUF — same as v1 but 4 members resident.
- Per N-chunk (24 of width 512): load each `w` K-tile `[k_in,512]` **once**, reuse across
  the 4 members into 4 distinct `[128,512]` fp32 PSUM banks (base `x@w`); then fuse the
  low-rank `tT[m_lo].T @ b_chunk` into each member's bank before copy+store.
- Store: the 512-chunk `c` covers output n_tiles `[4c, 4c+4)`; `v5`'s N axis is
  `[n_tile(96), n_in(128)]`, so store the 512-wide `out_sb` as **4 sub-tile writes**
  `v5[m_hi, m_lo, :, 4c+j, :]` for `j∈0..3` (mask-free; verified the index mapping
  numerically). This is the one lora-vs-matmul store difference (matmul's out N axis was
  flat `[..., n]` so it stored 512 contiguously; here the reshaped `(96,128)` N axis needs
  the 4-way split).

Predicted: HBMrd `w`×8 + `b`×8 + `x` + `a` ≈ **2150 MB** (~3.6× less than 7813 MB), MFU
into the ~45–49% band, latency into the sibling's fp32-floor neighborhood. Target: cross
1.0x (become the fp32 fallback). PSUM: 4 acc banks; transpose/tT use ≤2 banks in the
prior phase — peak 4–5 of 8. SBUF: fp32 `lhs_t` (4×40×128×4 = 80 KB/part) + `a` 20 KB +
`tT` 2 KB + transients ~6 KB ≈ 108 KB/part < ~192 KB. OK.

### D2 — bf16x2 3-product split on the BASE GEMM, on top of D1. PROMOTION CANDIDATE.
Expected benefit **HIGH** (sibling matmul got 1.274x; transpose_matmul 1.334x), risk
**MEDIUM**, fully guarded by the offline sim (predicted composite rel-L2 3.9e-7).

Port `matmul_v3_bf16_split`'s body: keep each fp32 base operand as two bf16 limbs and
accumulate three bf16-rate products in the fp32 PSUM bank, dropping the negligible lo@lo:
```
lhs_hi = bf16(lhs_t),  lhs_lo = bf16(lhs_t - lhs_hi)   # per member, resident
w_hi   = bf16(w_chunk), w_lo   = bf16(w_chunk - w_hi)  # per K-tile, transient
acc[m_lo] += lhs_hi@w_hi + lhs_hi@w_lo + lhs_lo@w_hi   # base x@w only
```
- Split `lhs_t` AFTER the fp32 identity transpose (transpose count unchanged). The two
  bf16 limbs are the same bytes as v2_b4's fp32 `lhs_t` (half dtype, twice the limbs), so
  the resident working set does not grow.
- The low limb is produced by the residual subtract into a bf16 destination
  (`nisa.tensor_tensor(..., op=nl.subtract)`), as in the sibling — no extra fp32 buffer.
- **Keep the down-projection `x@a` and the fused up-projection `(x@a)@b` in fp32**
  (only 3.4% of MACs; and they carry the 99.6%-dominant output magnitude, so fp32 there
  costs almost nothing and removes all doubt). The fused `tT.T @ b_chunk` matmul stays a
  single fp32 nc_matmul into the shared bank.

Predicted: PE-active drops ~24% (matmul's measured RAW PE-active on the same base GEMM),
MFU ~46%, HBMrd essentially unchanged (limbs built on-chip from the same loads), latency
past the fp32 floor toward ~1.2–1.3x. Guard: run the offline sim to reconfirm <1.3e-5
before spending a remote run; gate on the on-device 5-seed `l2_norm_passed`. PSUM: 4 acc
banks + 1 transient transpose bank = 5 of 8. SBUF: 2 bf16 limbs (80 KB) + a (20) + tT (2)
+ transients (6) ≈ 108 KB/part. OK. If D2 passes and beats D1, **PROMOTE D2, keep D1 as
the fp32 fallback** (the sibling pattern).

### D3 — resident `b` across all M-blocks. LOW value; measure only if DMA-bound after D2.
`b` is 6.3 MB, reused across all 8 blocks; making it resident saves only the `b`×8 = 50 MB
of the ~2150 MB D1 HBMrd (**2.3%**). Since the op is PE-bound (PE-active 34 ms ≫ the
read-limited floor), this is unlikely to move the wall. Resident `b` is 24·512·4 = 48
KB/part fp32, which competes with the bf16 limbs' 80 KB — tolerable but not free. Try at
most once, only if D1/D2 profiles show DMA re-emerging as the binding constraint;
otherwise reject on the model and record the datum.

### D4 — M-block factor sweep B∈{2,8} (i.e. re-nest the 2-level index). DEFER.
The sibling `matmul` swept B∈{2,4,8,16} and found **B=4 optimal** (B=8 regressed on
SBUF/PSUM pressure, B=16 0.519x). lora's 2-level index makes B=4 the natural block with
zero index arithmetic; B=2/8 would require re-nesting `m_hi`/`m_lo` across the block
boundary (more bug surface). Given the sibling's B=4 optimum, only explore if D1
disappoints; otherwise carry the sibling's finding and skip. (Kept within the ≤5-iteration
budget as a contingency, not a planned spend.)

## Plan of record (≤5 iterations per direction)

1. **D1**: implement `runs/lora_v2_mblk4.py` (N_CHUNK=512, B=4 = m_hi/m_lo). `--fast`
   screen → full 5-seed. Expect ~1.0–1.3x, HBMrd ~2.1 GB. Record `benchmark.csv` +
   `candidates.jsonl` (parent=lora_v1) + `profile/`. This is the fp32 fallback.
2. **D2**: commit `runs/offline_lora_bf16_split_sim.py` (composite forward, fail-closed
   independent-reference control), confirm <1.3e-5; then implement
   `runs/lora_v3_bf16_split.py` (base-only bf16x2 3-product on top of D1). `--fast` →
   full 5-seed, verify `l2_norm_passed` on all 5 + worst rel-L2 ~4e-7…4.5e-6. If it beats
   D1 → PROMOTE, keep D1 as fp32 fallback.
3. **D3/D4**: only if D1/D2 leave a clear DMA or blocking gap in the profile; otherwise
   record the model-based reject and stop. Never regress correctness (gate on 5-seed L2).

## Correctness guards / watch-items

- **2-level M-index as the block/member split** — `m_hi` is the block, `m_lo` (0..3) the
  member; store to `v5[m_hi, m_lo, :, n_tile, :]`. A swapped pair scrambles rows (the
  `_layout_check.py` mt=7 vs mt=13 guard already covers this; extend the host check to the
  512-wide 4-sub-tile store before D1).
- **512→4-sub-tile store mapping** — chunk `c` → n_tiles `4c..4c+3`; verified
  numerically. This is the only new indexing vs the sibling matmul.
- **`tT` PSUM→SBUF copy before the N-loop** — unchanged from v1; must survive across the
  24 N-chunks (PSUM banks are reused by `acc`).
- **bf16 split touches ONLY the base GEMM** — down-proj and fused low-rank stay fp32;
  offline sim reconfirms composite <1.3e-5 before any remote spend.
- **Full 5-seed before promoting** (the sibling lesson: `--fast` can mislead on
  SBUF/PSUM-pressure regimes; B=8 looked best on `--fast` but regressed on the full run).

## Validation

From `workspaces/lora/`:
```
python3 \
    ../../verify.py --op lora --candidate runs/<file>.py --fast   # drop --fast to promote
```
Gate on `l2_norm_passed` across all 5 seeds. Record each perf change in `benchmark.csv`,
each candidate in `candidates.jsonl` (parent links as a DAG), profiling evidence under
`profile/`.
