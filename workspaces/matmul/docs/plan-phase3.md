# matmul Phase 3 — Compute-Precision Regime: Compensated bf16x2 3-Product Split

## Goal Description

Break the `matmul` kernel past its fp32 PE floor by implementing a compensated
**bf16x2 3-product split** of the dense GEMM `out = lhs @ rhs`
(M=4096, K=5120, N=12288, fp32 I/O), delivered as a **new** kernel file
`runs/matmul_v3_bf16_split.py` that is a **localized diff** on the promoted Phase-2
kernel `runs/matmul_v2_b4.py`.

The Phase-2 kernel is fully PE-bound at the fp32 systolic rate (1.017x, ~13.35 ms,
PE=100%, MFU=49%, DMA=31%, HBMrd=2097 MB, HBMwr=201 MB). trn2's PE array is
bf16-native; fp32 emulates at ~2 passes, capping a correct fp32 GEMM near ~50% MFU.
That floor is only binding **while the kernel stays in fp32**. A compensated
bf16x2 split keeps each fp32 operand as two bf16 limbs (`hi = bf16(x)`,
`lo = bf16(x − hi)`) and accumulates three bf16-rate products in an fp32 PSUM bank
(`hi@hi + hi@lo + lo@hi`, dropping the negligible `lo@lo` term). The offline numeric
gate proves this clears the NKIBench 2e-5 relative-L2 correctness gate with a ~4.5x
margin, and four remote-proven sibling GEMMs — most directly the structural twin
`transpose_matmul` (1.026x → 1.334x) — establish the on-device speedup precedent.

The measurable objective: a full 5-seed p50 speedup of **~1.23–1.32x** (honest floor
~1.22x if only the −17% per-instruction estimate converts), replacing the fp32 floor
result, while holding the HBM read/write floor and passing all 5 correctness seeds.
`matmul_v2_b4.py` is **retained unchanged** as the guaranteed pure-fp32 fallback.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. Verification uses (from `workspaces/matmul/`):

```bash
python3 \
    ../../verify.py --op matmul --candidate runs/matmul_v3_bf16_split.py
```

(full 5-seed run; `verify.py` gates on `l2_norm_passed` and surfaces
MFU / DMA% / HBMrd / HBMwr / per-engine util in its metrics digest).

- **AC-1: Correctness — all 5 seeds pass the NKIBench relative-L2 gate.**
  Every seed in `[0, 21, 42, 63, 84]` satisfies `||v_k − v_r||_2 < 2e-5 · ||v_r||_2`
  in fp32. Expected worst on-device rel-L2 ≈ 4.47e-6 (the quadrature
  `sqrt(fp32_floor² + bf16²)` of the measured 4.207e-7 fp32 layout floor and the
  4.454e-6 offline bf16 term; the bf16 term dominates). This is the pure-GEMM family
  — no RMSNorm square-reduce feedback — so the bf16 error flows straight to the
  output with no composite surprise.
  - Positive Tests (expected to PASS):
    - `verify.py` reports `correct` (all 5 seeds `l2_norm_passed = true`).
    - The offline gate (`runs/offline_bf16_split_sim.py`) worst 5-seed 3-product
      rel-L2 is 4.454e-6, comfortably < 1.3e-5 (authorizes the build).
    - Per-seed rel-L2 values, when surfaced, are each below 1.0e-5 (margin check).
  - Negative Tests (expected to FAIL):
    - Any seed with rel-L2 ≥ 2e-5 → correctness FAIL → do not promote.
    - A plain single-limb bf16 GEMM (offline 2.35e-3) would fail the gate by ~100x
      and must not be the implemented route.
    - A limb construction that stores/uses `lo` as fp32 or reorders products away
      from the pinned `hi@hi + hi@lo + lo@hi` set changes the error budget and is
      rejected.

- **AC-2: Performance — full 5-seed p50 beats a contemporaneous v2_b4 control by
  more than the same-session noise band.**
  Promotion requires a **full 5-seed (not `--fast`)** same-session A/B bracket: run a
  contemporaneous `matmul_v2_b4` control and `matmul_v3_bf16_split` in the same
  session after warmup, discard cold-start first-iteration outliers, and require the
  v3 p50 to beat the v2_b4 control p50 by more than `max(observed same-session
  control drift, 1.8%)`. Expected v3 result ~1.23–1.32x (honest floor ~1.22x);
  v2_b4 control ~1.017–1.022x.
  - Positive Tests (expected to PASS):
    - Full 5-seed v3 p50 speedup ≥ ~1.22x and clears the control by more than the
      noise band in the same session.
    - PE-active wall shrinks (siblings measured −19.6% to −24.4%), consistent with 3
      bf16-rate products replacing 2 fp32-rate passes at the measured 3.62x
      fp32/bf16-1-product ratio (≈ 3.0/3.62 ≈ 0.83, a −17% floor estimate).
  - Negative Tests (expected to FAIL):
    - `--fast` used as the promotion measurement (Phase-2 lesson: `--fast` mis-ranked
      B=8) → invalid evidence; re-run full 5-seed.
    - v3 p50 within the noise band of the v2_b4 control, or slower → no promotion.
    - A cold-start first-iteration outlier (e.g. the ~15.33 ms first run vs stable
      ~13.05 ms p50 seen in the Phase-2 reconfirm) used as the reported number →
      invalid; use the post-warmup p50.

- **AC-3: HBM read/write floor gate — no re-fetch, no spill.**
  `hbm_read` stays within +2% of the v2_b4 floor (≤ ~2139 MB vs 2097 MB) and
  `hbm_write` within +2% of 201 MB (≤ ~205 MB). Because the resident limb bytes equal
  the former fp32 `lhs_t` bytes and B is unchanged (see AC-5), the floor is predicted
  to hold — but this is an explicit measured gate, not an assumption. SBUF spill
  manifests as `hbm_write` inflating above the ~201 MB floor (there is no separately
  printed psum-drain counter; spill is inferred from `hbm_write` staying at floor).
  - Positive Tests (expected to PASS):
    - `verify.py` metrics digest shows HBMrd ≈ 2097 MB (within +2%) and
      HBMwr ≈ 201 MB (within +2%).
    - Both v2_b4 control and v3 HBMrd/HBMwr recorded in the same evidence digest and
      compared numerically (record exact byte counts alongside rounded MB).
  - Negative Tests (expected to FAIL):
    - `hbm_read` rises like the twin's blocked `tmm_v2` (392 → 448 MB, +14%) → a
      read-floor break (the `transpose_matmul` lesson) → **do not promote on wall win
      alone**; investigate a read-neutral variant before adopting (this is a hard
      gate, not advisory).
    - `hbm_write` inflates above ~205 MB → SBUF spill → reject and diagnose before
      any promotion.

- **AC-4: Localized diff and preserved fp32 fallback.**
  `matmul_v3_bf16_split.py` is a **new** file; `matmul_v2_b4.py` is unchanged
  byte-for-byte and retained as the fallback. The v2_b4 structure is preserved:
  M-block B=4, N_CHUNK=512, K-accumulate 40 kt into B distinct [128,512] fp32 PSUM
  banks, single copy+store epilogue. Only the operand precision and the matmul body
  change. PSUM budget: 4 acc banks + 1 transpose bank = 5 of 8; B stays 4.
  - Positive Tests (expected to PASS):
    - `git status` / diff shows `matmul_v2_b4.py` unmodified and a new
      `matmul_v3_bf16_split.py` added.
    - Kernel keeps a single `@nki.jit def kernel(v1, v2)` entry matching the tiled
      baseline signature; the baseline/reference under `../AccelOpt/NKIBench/` is
      never edited.
    - PSUM allocation stays at 5 of 8 banks (4 acc + 1 transient transpose).
  - Negative Tests (expected to FAIL):
    - Any edit to `matmul_v2_b4.py`, the baseline, or the NKIBench reference → reject.
    - Raising B to 8 as part of this build → B=8 is measured-rejected
      (`matmul_v2_b8` ran at 0.968x: enlarged lhs residency + all PSUM banks hurt
      occupancy/schedule) and is risky under the split's added register/schedule
      pressure; it is not part of D1. (Note: the transpose PSUM bank is transient —
      used in the transpose-build loop before the N-chunk accumulation loop — so it
      is not co-live with the acc banks; B=8 is rejected on measured
      occupancy/schedule grounds, not proven bank-count-infeasible.)

- **AC-5: Correct compensated-split limb construction (numeric method fixed).**
  - AC-5.1: **lhs is split AFTER the fp32 transpose.** lhs arrives
    `[m_in(par), k_in(free)]`; the identity `nc_matmul(is_transpose=True)` idiom
    produces `lhs_t = [k_in(par), m_in(free)]` in fp32 PSUM (exact; 1 transpose/tile,
    count unchanged). The kernel must **not** retain a full resident fp32 `lhs_t`:
    copy one [128,128] tile at a time to bounded transient fp32 SBUF scratch, split
    it element-wise into the resident bf16 limbs, and free the fp32 scratch. Splitting
    *before* the transpose (which would double the transpose count) is rejected.
    - Positive: only two resident bf16 limbs `lhs_hi`, `lhs_lo` per member survive
      the staging path; transpose count equals v2_b4's.
    - Negative: a resident fp32 `lhs_t` kept alongside the bf16 limbs (resident set
      grows ~1.5x, AC-3 read floor at risk) → reject; splitting before transpose
      (transpose count doubles) → reject.
  - AC-5.2: **Residual-subtract limb build with no separate fp32 residual buffer.**
    `hi = bf16(x)`; `lo = bf16(x − hi)` computed as a residual subtract into a bf16
    destination (the fp32 residual is exact internally and downcast for free). Applies
    to both lhs (post-transpose) and rhs (`rhs_hi = bf16(rhs_f)`,
    `rhs_lo = bf16(rhs_f − rhs_hi)`, split per (chunk, kt)).
    - Positive: `lo` is a bf16-valued limb produced by the fused residual subtract.
    - Negative: keeping a separate fp32 residual buffer, or storing `lo` as fp32 →
      reject (deviates from the offline-modeled method and grows the working set).
  - AC-5.3: **3-product accumulation, pinned order.** Accumulate into one fp32 PSUM
    bank per member in the exact order `acc[mb] += lhs_hi@rhs_hi + lhs_hi@rhs_lo +
    lhs_lo@rhs_hi`, dropping `lo@lo`. This is the **same product set and pinned
    on-device product order** as `tmm_v3_mblk16`; the offline sim uses the identical
    order and predicts the rel-L2 error budget (worst 4.454e-6) but is a numerical
    model, **not** a bit-exact oracle — on-device correctness must still be measured
    on all 5 seeds (AC-1).
    - Positive: three `nc_matmul` products per (member, kt) in the pinned order, all
      accumulating into the member's fp32 PSUM bank.
    - Negative: including `lo@lo` (that is D2, record-only, not D1); reordering or
      dropping a different product → changes the error budget → reject for D1.

- **AC-6: Evidence and reproducibility.**
  Record the candidate in `benchmark.csv` (one perf row) and `candidates.jsonl`
  (parent = `matmul_v2_b4`, as a DAG link), with a profiling digest under `profile/`.
  The digest includes PE / MFU / DMA% / HBMrd / HBMwr for **both** the v2_b4 control
  and v3 from the same session, plus per-seed rel-L2 when available, and references
  the offline gate (`profile/matmul_phase3_bf16split_offline_gate.txt`).
  - Positive Tests (expected to PASS):
    - `benchmark.csv` gains a v3 row and `candidates.jsonl` gains a v3 entry with
      `parent = matmul_v2_b4`.
    - The evidence digest shows the same-session A/B metrics for control and v3,
      exact byte counts alongside rounded MB, and cites the offline gate result.
  - Negative Tests (expected to FAIL):
    - Promotion recorded without the same-session control bracket, or against the
      stale 13.35 ms number instead of a contemporaneous control → invalid evidence.
    - A resident fp32 `lhs_t` surviving the split staging path (code-inspection note
      or profile evidence contradicts AC-5.1) → evidence gate fails.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.
This is a **deterministic single-shape design**: M=32·128, K=40·128, N=24·512 — every
tile full, no remainders, no edge tiles, no tile-shape regime to branch on. The
approach is fixed per the draft and matches four remote-proven sibling GEMMs. Upper
and lower bounds therefore nearly converge: the single planned build is D1.

### Upper Bound (Maximum Acceptable Scope)
The implementation delivers `matmul_v3_bf16_split.py` (the D1 compensated bf16x2
3-product split as a localized diff on v2_b4), passes all 5 correctness seeds, holds
the HBM read/write floor, and is promoted on a full 5-seed same-session A/B bracket
showing ~1.23–1.32x — with complete evidence (benchmark.csv, candidates.jsonl,
profile/ digest for both control and v3). D2 (4-product) and D3 (DMA/M-block watch)
remain record-only / contingent and are built only if their explicit trigger fires
(D2 only if D1 rel-L2 unexpectedly > 1.3e-5; D3 only if D1 breaks the read floor or
exposes DMA).

### Lower Bound (Minimum Acceptable Scope)
The implementation delivers `matmul_v3_bf16_split.py` implementing exactly the D1
3-product split (AC-5), passing all 5 correctness seeds (AC-1) with the localized
diff and preserved fp32 fallback (AC-4). If — against the offline evidence and four
siblings — the on-device split fails the HBM read-floor gate (AC-3) and cannot be
brought back without regressing wall, **promote nothing new**: keep v2_b4 (1.017x)
and record the split's measured-reject datum (correctness, wall, HBM counters,
root-cause note) as the phase-3 evidence. A correctly-recorded negative result that
preserves the fp32 fallback still satisfies the minimum.

### Allowed Choices
- Can use: internal bf16 limbs with fp32 I/O; the compensated 3-product split with
  the pinned order `hi@hi + hi@lo + lo@hi`; the v2_b4 structure (B=4, N_CHUNK=512,
  fp32 PSUM accumulation, identity-transpose lhs, copy+store epilogue); bounded
  transient fp32 scratch for the post-transpose split; a cheap single-seed
  counter-probe run before the full 5-seed spend to catch a read-floor/spill
  regression early.
- Cannot use: editing `matmul_v2_b4.py`, the baseline, or the NKIBench reference;
  plain single-limb bf16 (fails the gate by ~100x); a resident fp32 `lhs_t` kept
  alongside the bf16 limbs; splitting before the transpose; raising B to 8; `--fast`
  as the promotion measurement; promoting on a wall win when the read floor breaks.

> **Note on Deterministic Designs**: The draft specifies a fixed method (the
> compensated bf16x2 3-product split with a pinned product order), so the path
> boundaries are deliberately narrow. Upper and lower bounds converge on the single D1
> build; "Allowed Choices" reflects that the numeric method is fixed per the draft and
> the sibling precedent, with the only real branch being the honest exit (promote the
> split, or preserve v2_b4 and record a measured-reject if the read floor breaks).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

Start from `runs/matmul_v2_b4.py` and change only the operand precision and the matmul
body. Conceptual sketch (illustrative, not prescriptive):

```
for mblk in affine_range(M_BLOCKS):            # M_TILES=32, B=4 -> 8 blocks
  # Resident bf16 limbs for this block's B members: [B, K_TILES, k_in=128, m_in=128]
  lhs_hi = bf16 ndarray(B, K_TILES, par_dim(128), 128)
  lhs_lo = bf16 ndarray(B, K_TILES, par_dim(128), 128)
  for mb in range(B):
    for kt in range(K_TILES):                  # K_TILES=40
      lhs_sb   = load v1[mblk*B+mb, :, kt, :]           # [m_in, k_in] fp32
      psum_t   = nc_matmul(lhs_sb, identity, is_transpose=True)   # -> [k_in, m_in] fp32 PSUM
      lhs_t_f  = copy(psum_t) into BOUNDED TRANSIENT fp32 SBUF     # one [128,128] tile
      lhs_hi[mb,kt] = bf16(lhs_t_f)
      lhs_lo[mb,kt] = bf16(lhs_t_f - lhs_hi[mb,kt])    # residual-subtract into bf16 dest
      # transient fp32 scratch freed here; no resident fp32 lhs_t

  for c in range(N_CHUNKS):                     # N_CHUNK=512, 24 chunks
    acc = zeros(B, par_dim(128), 512) fp32 PSUM  # B distinct acc banks
    for kt in range(K_TILES):
      rhs_f  = load v2[kt, :, 512*c : 512*c+512]        # [k_in, 512] fp32, loaded ONCE
      rhs_hi = bf16(rhs_f)
      rhs_lo = bf16(rhs_f - rhs_hi)                     # residual-subtract into bf16 dest
      for mb in range(B):
        acc[mb] += nc_matmul(lhs_hi[mb,kt], rhs_hi)     # pinned order
        acc[mb] += nc_matmul(lhs_hi[mb,kt], rhs_lo)
        acc[mb] += nc_matmul(lhs_lo[mb,kt], rhs_hi)     # drop lo@lo
    for mb in range(B):
      out_sb = copy(acc[mb]); store out[mblk*B+mb, :, 512*c : 512*c+512]
```

SBUF sizing (per partition, over the 128-partition axis): v2_b4's fp32
`lhs_t = (B=4, K_TILES=40, 128, 128)` holds `4·40·128 = 20480` fp32 elems/partition
= 80 KB/part. Each bf16 limb holds the same `4·40·128` elems × 2 B = 40 KB/part; two
limbs = 80 KB/part — **exactly the former fp32 `lhs_t` bytes** (half the dtype, twice
the limbs). The resident working set does not grow. Transients (`rhs_hi/lo` ~1 KB
each, `rhs_f`/`out_sb` ~2 KB, transpose scratch ~0.5 KB) keep peak well under
90 KB/part vs the ~192 KB budget — much more headroom than `transpose_matmul`, which
sat at 128 KB resident / 168 KB peak and had to fight read-floor breaks. PSUM: 4 acc
banks ([128,512] fp32 = 1 bank each) + 1 transient transpose bank = 5 of 8.

Suggested sequencing to de-risk remote spend: run the offline gate first (already
done — 4.454e-6), then optionally a cheap single-seed counter probe to confirm the
HBM read floor holds before the full 5-seed promotion measurement.

### Relevant References
- `runs/matmul_v2_b4.py` — the Phase-2 promoted kernel; the exact structural base for
  the localized diff (B=4, N_CHUNK=512, identity-transpose, fp32 PSUM accumulation).
- `runs/offline_bf16_split_sim.py` + `profile/matmul_phase3_bf16split_offline_gate.txt`
  — the zero-spend numeric gate (worst 5-seed 3-product rel-L2 = 4.454e-6).
- `../transpose_matmul/runs/tmm_v3_mblk16_bf16_split.py` — the structural twin's
  promoted split kernel; the reference for the pinned 3-product body and the
  residual-subtract-into-bf16 limb construction.
- `../transpose_matmul/runs/tmm_v2_bf16_split.py` — the twin's read-floor-blocked
  first split (M_BLK=8, 392→448 MB read); the negative exemplar for AC-3.
- `../matmul_add_rmsnorm/runs/matmul_add_rmsnorm_v2_bf16_split.py` — sibling
  3-product split (3.920x → 4.879x), the moving-512 dense-GEMM regime precedent.
- `../../verify.py` — gates on `l2_norm_passed`; surfaces MFU/DMA/HBMrd/HBMwr/
  per-engine util used by AC-2/AC-3.
- `../AccelOpt/NKIBench/reference/matmul_M4096_N12288_K5120_numpy_2.py` — the numpy
  reference whose `get_inputs()` draw (lhs then rhs, normal(0,1), fp32, per seed)
  matches the offline sim exactly (correctness-margin transfer verified).

## Dependencies and Sequence

### Milestones
1. **Numeric authorization (complete).**
   - Phase A: Offline 3-product gate produces worst 5-seed rel-L2 4.454e-6 vs 2e-5
     (4.5x under) and confirms the input draw matches the reference — done.
2. **Implement the D1 split kernel.**
   - Step 1: Copy `matmul_v2_b4.py` to `matmul_v3_bf16_split.py`; keep structure.
   - Step 2: Add the post-transpose lhs limb build (bounded transient fp32 scratch →
     resident bf16 `lhs_hi`/`lhs_lo`), per AC-5.1/5.2.
   - Step 3: Add per-chunk rhs limb build and replace the single fp32 `nc_matmul` with
     the pinned 3-product accumulation, per AC-5.3.
3. **On-device re-gate and evidence.** Depends on Milestone 2.
   - Step 1: (Optional) cheap single-seed counter probe to check the HBM read floor
     early (AC-3).
   - Step 2: Full 5-seed same-session A/B bracket vs a contemporaneous v2_b4 control;
     record correctness (AC-1), speedup (AC-2), HBM counters (AC-3).
   - Step 3: Write `benchmark.csv` row + `candidates.jsonl` entry (parent=matmul_v2_b4)
     + `profile/` digest (AC-6).
4. **Promotion decision (honest exit).** Depends on Milestone 3.
   - Promote v3 if AC-1 ∧ AC-2 ∧ AC-3 all hold; otherwise keep v2_b4 and record the
     measured-reject datum.
5. **Contingent-only (built only on trigger).**
   - D2 (4-product) only if D1 rel-L2 unexpectedly > 1.3e-5; D3 (chunk-level DMA
     rework) only if D1 breaks the read floor or exposes DMA. No planned build.

Dependency summary: Milestone 1 (done) authorizes Milestone 2; Milestone 2 produces
the artifact Milestone 3 measures; Milestone 3's counters drive the Milestone 4
promotion decision; Milestone 5 fires only on the specified triggers.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Confirm the offline numeric gate output (worst 5-seed 3-product rel-L2 4.454e-6 < 2e-5) and that the reference input draw matches the sim | AC-1 | analyze | - |
| task2 | Create `runs/matmul_v3_bf16_split.py` from v2_b4; keep B=4/N_CHUNK=512/PSUM structure; leave v2_b4 unchanged | AC-4 | coding | task1 |
| task3 | Implement the post-transpose lhs limb build (bounded transient fp32 scratch → resident bf16 `lhs_hi`/`lhs_lo`; residual-subtract into bf16; no resident fp32 lhs_t) | AC-5 | coding | task2 |
| task4 | Implement per-chunk rhs limb build + the pinned 3-product accumulation `hi@hi + hi@lo + lo@hi` (drop lo@lo) into the fp32 PSUM banks | AC-5 | coding | task3 |
| task5 | Run full 5-seed correctness + same-session A/B perf bracket vs a contemporaneous v2_b4 control (no `--fast` for promotion; discard cold-start outliers) | AC-1, AC-2 | coding | task4 |
| task6 | Read the profiler digest; verify HBMrd ≤ ~2139 MB and HBMwr ≤ ~205 MB (no re-fetch, no spill) for both control and v3 | AC-3 | coding | task5 |
| task7 | Record `benchmark.csv` row + `candidates.jsonl` entry (parent=matmul_v2_b4) + `profile/` digest with both-kernel metrics and per-seed rel-L2; cite the offline gate | AC-6 | coding | task6 |
| task8 | Promotion decision: promote v3 iff AC-1∧AC-2∧AC-3 hold; else keep v2_b4 and record the measured-reject datum | AC-2, AC-3 | coding | task7 |

## Claude-Codex Deliberation

### Agreements
- D1 (the compensated bf16x2 3-product split) is the correct primary — and only —
  planned build; the verified input draw removes the main correctness uncertainty and
  the offline worst-case (4.454e-6) gives a comfortable ~4.5x margin under 2e-5.
- v2_b4 is PE-bound at the fp32 floor; the structural twin `transpose_matmul`
  (1.026x → 1.334x) is strong precedent for the split win.
- The HBM read/write floor gates are well chosen and directly measurable via
  `verify.py`, with `hbm_write` inflation as the practical SBUF-spill signal.
- Keeping `matmul_v2_b4.py` unchanged and adding a new `matmul_v3_bf16_split.py` is
  the right fallback boundary.
- Splitting after the fp32 lhs transpose (not before) is correct — splitting before
  would double the transpose work for no gain.
- D2 (4-product) and D3 (DMA/M-block watch) are correctly contingent/record-only, not
  planned work.

### Resolved Disagreements
- **B=8 feasibility framing**: Codex objected that "B=8 is PSUM-bank-infeasible
  (8 acc + 1 transpose > 8 banks)" is overstated, since `matmul_v2_b8` exists and ran,
  and the transpose PSUM is transient (before the N-loop accumulators), so it is not
  co-live with the acc banks. Resolution: reworded to "B stays 4; B=8 is
  **measured-rejected** (v2_b8 = 0.968x: enlarged lhs residency + all PSUM banks hurt
  occupancy/schedule) and risky under the split's added pressure — not part of D1,"
  dropping the bank-count-infeasibility claim (AC-4).
- **"Byte-identical to the offline sim"**: Codex noted the offline sim is a numerical
  model, not a bit-exact oracle of the device kernel. Resolution: reworded to "same
  product **set** and pinned on-device product **order** as `tmm_v3_mblk16`; the sim
  predicts the rel-L2 error budget but on-device correctness must still be measured on
  all 5 seeds" (AC-5.3, AC-1).
- **SBUF same-bytes claim**: Codex flagged that "80 KB same as v2_b4" holds only if
  the fp32 `lhs_t` is actually replaced by the bf16 limbs, not kept alongside them.
  Resolution: AC-5.1 now explicitly forbids a resident fp32 `lhs_t` (build limbs from
  bounded transient fp32 scratch and free it); the same-bytes claim is stated as
  conditional on that rule. The per-partition arithmetic was verified
  (fp32 `lhs_t` = 80 KB/part; two bf16 limbs = 40 KB/part each = 80 KB/part).
- **Promotion noise band**: Codex noted a fixed ±1.8% band is under-specified given
  observed cold-start outliers. Resolution: AC-2 now defines promotion as a full
  5-seed same-session A/B bracket after warmup, threshold `> max(observed same-session
  control drift, 1.8%)`, with cold-start first-iteration outliers discarded.
- **Early counter probe**: Codex suggested a cheap single-seed counter run before the
  full 5-seed spend. Resolution: added as an allowed (optional) de-risking step in the
  Path Boundaries and Milestone 3.

### Convergence Status
- Final Status: `converged`
- Convergence matrix: Codex first-pass (Phase 3) raised 6 core risks + missing
  requirements; Claude candidate v1 (Phase 4) addressed them; Codex round 1 (Phase 5)
  returned no `UNRESOLVED` and 4 wording/protocol `REQUIRED_CHANGES`; Claude candidate
  v2 applied all 4; Codex round 2 returned `REQUIRED_CHANGES: None`, `UNRESOLVED: None`
  (converged). 2 convergence rounds executed. Two of Codex's `QUESTIONS_FOR_USER` were
  resolved by verification: the reference `get_inputs()` draw (lhs then rhs, normal(0,1),
  fp32, per seed) matches the offline sim exactly, and `verify.py` surfaces
  MFU/DMA/HBMrd/HBMwr/per-engine util so the read/write-floor gate is directly
  measurable. No pending user decisions remain.

## Pending User Decisions

None. All questions raised during Codex analysis and the convergence loop were
resolved during plan refinement (input-draw match verified; profiler metric surface
verified; read-floor threshold quantified at +2%; hard-vs-soft gate status set — 2e-5
is the hard NKIBench contract, ~1.23–1.32x is the directional target with an honest
exit, the HBM read floor is a hard gate; cheap-probe vs full-5-seed sequencing settled;
v2_b4 preserved byte-for-byte). No opposing Claude/Codex positions remain open.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `lhs_hi`, `lhs_lo`,
  `rhs_hi`, `rhs_lo`, `acc`), matching the naming already used in `matmul_v2_b4.py`
  and the sibling `tmm_v3_mblk16` kernel.
- All repository-facing files, comments, and commit messages are in English.

--- Original Design Draft Start ---

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
pressure — not proven bank-count-infeasible.

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

--- Original Design Draft End ---
