# lora — Phase 2 Plan: Profile-Driven Optimization (M-block + bf16x2 base split)

## Goal Description

Take the phase-1 lora kernel `runs/lora_v1.py` (correct fp32 `out = x@w + (x@a)@b`,
worst rel-L2 4.874e-7, but **0.382x** at 38.3562 ms — slower than the 14.6645 ms
baseline because phase 1 had no speed target) and make it fast without ever regressing
correctness. The profile is unambiguous: lora_v1 is **PE-bound** (PE 88.5%, TRUE
PE-active 33.94 ms) and **DMA co-saturated** (89.1%) with **MFU 17.69%** and
**HBM read 7812.7 MB** — 22.7× the ~344 MB single-pass ideal — because it re-streams all
of `w` (and `b`) from HBM once per M-tile across 32 M-tiles. Because the base GEMM `x@w`
(M4096/N12288/K5120) is **96.6% of the MACs** and is **shape-identical** to the sibling
`matmul` operator, the entire phase-2 win is "make the base GEMM fast, then graft the
cheap fused low-rank residual (3.4% of MACs) on top" — i.e. port the proven sibling
matmul ladder.

The plan follows the sibling matmul ladder for this identical base GEMM (verified
numbers): fp32 M-outer `matmul_v1` 0.855x (HBMrd 7584 MB, MFU 41%) →
**M-block B=4, N_CHUNK=512 `matmul_v2_b4` 1.017x** (HBMrd 2097 MB, MFU 49%, PE 100%) →
**compensated bf16x2 3-product split `matmul_v3_bf16_split` 1.274x** (HBMrd 2037 MB,
MFU 46%; on-device rel-L2 4.455e-6 == its offline bf16 term).

Two candidate kernels are produced:
- **D1 (`runs/lora_v2_mblk4.py`)** — fp32, N_CHUNK=512 with M-block B=4, the fp32
  fallback (port of `matmul_v2_b4` structure with lora's fused low-rank residual).
- **D2 (`runs/lora_v3_bf16_split.py`)** — the compensated bf16x2 3-product split applied
  to the **base GEMM only**, on top of D1 — the promotion candidate. The down-projection
  `x@a` and the fused up-projection `(x@a)@b` stay fp32 (only 3.4% of MACs and they carry
  99.6% of the output magnitude, so fp32 there is nearly free and removes all doubt).

D3 (resident `b`) and D4 (M-block factor sweep B∈{2,8}) are contingencies: pursue only if
the D1/D2 profile reveals a DMA or blocking gap; otherwise record a model-based reject
with rationale (carrying the sibling's B=4-optimal / B=8-regressed / B=16=0.519x finding).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. The **only hard correctness gate** is AC-1 (NKIBench
relative-L2 ≤ 2e-5 across all 5 seeds via `verify.py`'s `l2_norm_passed`). Speedup
targets (crossing 1.0x) are directional unless the user promotes them to hard in DEC-1.

- **AC-1: Correctness — every scored candidate PASSES the NKIBench relative-L2 gate on all 5 seeds.**
  Run `verify.py --op lora --candidate runs/<file>.py` (full, no `--fast`) before any
  promotion; gate on `l2_norm_passed == True` for seeds `[0, 21, 42, 63, 84]`. Report the
  worst per-seed rel-L2.
  - Positive Tests (expected to PASS):
    - D1 (`lora_v2_mblk4.py`) full 5-seed run: `l2_norm_passed` True on all 5, worst
      rel-L2 in the fp32-floor neighborhood (~4.9e-7, matching lora_v1 / the sibling
      fp32 regime).
    - D2 (`lora_v3_bf16_split.py`) full 5-seed run: `l2_norm_passed` True on all 5, worst
      rel-L2 ≈ the device quadrature `sqrt(fp32_floor² + diluted_bf16²)` ≈ **~6e-7**
      (fp32 floor ~4.9e-7 ⊕ diluted base-split ~3.9e-7) — comfortably under 2e-5.
  - Negative Tests (expected to FAIL / be rejected):
    - A base-GEMM-only variant with the low-rank residual omitted fails the gate (the
      `_layout_check.py` base-only negative control already asserts rel-L2 > 2e-5).
    - A plain single-limb bf16 kernel (no compensation) scores ≈ 2.35e-3 — rejected.
    - Any candidate with even one seed `l2_norm_passed == False` is not promotable,
      regardless of latency.

- **AC-2: D1 — fp32 M-block (B=4) + N_CHUNK=512 base GEMM with fused low-rank residual, HBM traffic collapsed.**
  Implement `runs/lora_v2_mblk4.py` porting `matmul_v2_b4`'s block structure onto lora's
  2-level M-index, where the M-block IS `m_hi` (8 blocks) and the B=4 members ARE
  `m_lo` (0..3) — a natural, arithmetic-free fit (no flat-index floor-div/mod, no
  `M_TILES % B` divisibility concern; it is structurally 8×4).
  - Positive Tests (expected to PASS):
    - AC-1 passes for D1 (full 5-seed).
    - Profile shows **HBM read collapsed** from 7813 MB toward the modeled ~2.1 GB band
      (`w`×8 + `b`×8 + `x` + `a`), i.e. a ≈3.6× reduction; **HBM write stays ≈ 201 MB**
      (one output pass, no spill).
    - MFU rises out of the 18% regime toward the sibling's ~45–49% band; latency drops
      well below lora_v1's 38.36 ms into the sibling fp32-floor neighborhood.
    - Each N-chunk loads every `w` K-tile `[k_in,512]` exactly **once** and reuses it
      across the 4 members into 4 distinct `[128,512]` fp32 PSUM banks; `b_chunk` is
      loaded **once per N-chunk** and reused across the 4 members (not re-loaded per
      member).
    - The 512-wide output tile is stored as 4 sub-tile writes
      `v5[m_hi, m_lo, :, 4c+j, :]` for `j∈0..3` (chunk `c` covers n_tiles `[4c, 4c+4)`).
  - Negative Tests (expected to FAIL / be rejected):
    - HBM read remaining near 7–8 GB (M-blocking not actually amortizing `w`) — reject.
    - HBM write materially above ~201 MB or HBM read ballooning far above the ~2.1 GB
      band (SBUF/PSUM spill) — reject, treat as a broken implementation of the design.
    - `b_chunk` re-loaded inside the member loop (avoidable DMA that weakens the HBM
      model) — reject in review.
  - AC-2.1: PSUM/SBUF budget holds — peak ≤ 8 PSUM banks (4 acc banks + ≤2 for
    transpose/`tT`), SBUF ≈ 108 KB/partition (fp32 `lhs_t` ~80 KB + `a` ~20 KB + `tT`
    ~2 KB + transients ~6 KB) < ~192 KB, confirmed by no-spill profile.
    - Positive: profiler shows no PSUM/SBUF spill traffic.
    - Negative: any spill signature in the profile fails this sub-criterion.

- **AC-3: D2 — compensated bf16x2 3-product split on the BASE GEMM ONLY, low-rank path fp32, correctness preserved.**
  Implement `runs/lora_v3_bf16_split.py` as a localized diff on D1, porting
  `matmul_v3_bf16_split`'s body onto the base `x@w`:
  `lhs_hi = bf16(lhs_t)`, `lhs_lo = bf16(lhs_t − lhs_hi)`, `w_hi = bf16(w_chunk)`,
  `w_lo = bf16(w_chunk − w_hi)`, then
  `acc[m_lo] += lhs_hi@w_hi + lhs_hi@w_lo + lhs_lo@w_hi` (drop the negligible `lo@lo`).
  The down-projection `x@a` and the fused up-projection `(x@a)@b` **stay fp32** (a single
  fp32 `nc_matmul` of `tT.T @ b_chunk` into the shared bank).
  - **AC-3.1 (HARD correctness-structure guard — down-projection operand lifetime):**
    Because the base split keeps only resident **bf16** limbs (no resident fp32 `lhs_t`),
    the fp32 down-projection must obtain its fp32 operand from the transient transpose
    scratch **before it is freed**. In the transpose/split loop, from the transient fp32
    `lhs_t_f = copy(psum_transpose)`: (a) accumulate the fp32 down-projection
    `tT_psum[m_lo] += nc_matmul(a_local[kt], lhs_t_f)`, then (b) build `lhs_hi`/`lhs_lo`
    bf16 limbs from the same `lhs_t_f`. No full resident fp32 `lhs_t` survives; SBUF stays
    ≈ 108 KB/partition (2 bf16 limbs ~80 KB, same bytes as D1's fp32 `lhs_t`).
    - Positive: `tT` (fp32) equals `(x@a)^T` for the sampled tiles (host check), and the
      device D2 passes AC-1.
    - Negative: a port that drops the fp32 `lhs_t` without folding the down-projection
      leaves `x@a` with no operand (compile/runtime failure or wrong result) — must be
      caught before any remote spend.
  - **AC-3.2 (HARD correctness-structure guard — `tT_psum` accumulated EXACTLY ONCE per M-block):**
    `tT_psum[m_lo]` is zeroed once per B=4 M-block and accumulated over the 40 K-tiles
    **in the M-block prologue, before the N-chunk loop**. The N-chunk loop only *reads*
    the completed `tT` (as `tT.T @ b_chunk`); it must never update `tT_psum` again.
    Folding the down-projection into a per-N-chunk path would re-accumulate it 24× (for
    N_CHUNK=512 over N=12288) and silently corrupt the output while looking structurally
    plausible.
    - Positive: the fused output tiles match gold in `_layout_check.py`; device passes AC-1.
    - Negative: `tT` accumulated inside the N-chunk loop (24× over-count) — the host
      layout check and/or the offline sim must reject it before remote spend.
  - Positive Tests (expected to PASS):
    - AC-4 offline guard authorizes the split (composite rel-L2 < 1.3e-5) BEFORE the
      first D2 remote run.
    - D2 passes AC-1 (full 5-seed), worst rel-L2 ≈ ~6e-7 (device quadrature), and its
      profile shows PE-active dropping (~24% on the base GEMM per the sibling) with HBM
      read essentially unchanged vs D1 (limbs built on-chip from the same loads).
  - Negative Tests (expected to FAIL / be rejected):
    - Splitting the down-projection or the up-projection as well (no PE upside — 3.4% of
      MACs — and unnecessary correctness risk) — out of scope for D2.
    - Keeping a resident fp32 `lhs_t` alongside the bf16 limbs (≈80+80 KB plus `a`/`tT`/
      transients approaches the ~192 KB/partition limit and risks a spill) — reject.
    - Promoting D2 on a `--fast` screen alone — must be a full 5-seed run (sibling lesson:
      `--fast` misled on B=8).
  - **Promotion rule:** promote D2 only if it passes AC-1 on a full 5-seed run AND beats
    D1 in same-session latency beyond noise; **keep D1 as the fp32 fallback** either way
    (the sibling pattern).

- **AC-4: Offline no-spend guard for D2 — committed BEFORE D2 is scored on the remote profiler.**
  Commit `runs/offline_lora_bf16_split_sim.py` (ported from the sibling
  `matmul/runs/offline_bf16_split_sim.py`) that computes the **full-shape composite
  forward** `x@w + (x@a)@b` at **seed 42** (the single input the remote gate scores — the
  adapter pins `np.random.seed(42)` for all 5 profiler seeds), splitting **only the base**
  `x@w` (3-product) and keeping the low-rank path fp32, and reports the composite NKIBench
  relative-L2 against the fp32 reference.
  - Positive Tests (expected to PASS):
    - The composite bf16x2 3-product rel-L2 (route [A]) is reported at ~3.9e-7 and the
      sim's verdict authorizes the split (composite < 1.3e-5).
    - An **independent, fail-closed reference control** loads the actual NKIBench
      reference module (via `NKIBENCH_ROOT`, matching `verify.py`) and **raises** (not
      `assert`, which `-O` would strip) if the reference is unreachable or the draw model
      diverges — the gate can never authorize on an unvalidated input.
    - Reports the base-only split error (route [B], ~4.45e-6) and the plain-single-limb
      bf16 control (route [C], ~2.35e-3, which must be far above the gate) for scale, plus
      input-diversity draws `[0,21,63,84]` reported separately as NOT-gated.
  - Negative Tests (expected to FAIL / be rejected):
    - If the composite rel-L2 came out ≥ 1.3e-5, the sim does NOT authorize D2 and the
      precision-floor datum is recorded instead of spending a remote run.
    - Running the sim without the reachable NKIBench reference raises and refuses to
      authorize (fail-closed), rather than silently scoring a wrong input.

- **AC-5: Host-side layout guard extended for the 512-wide 4-sub-tile store.**
  Extend `runs/_layout_check.py` (numpy, not scored) to cover the new N_CHUNK=512 store
  mapping — chunk `c` (columns `[512c, 512c+512)`) decomposed into sub-tiles
  `out_sb[:, 128j:128j+128] → v5[m_hi, m_lo, :, 4c+j, :]` for `j∈0..3`, with `N_CHUNKS=24`
  so `4c+j` spans `0..95` — while retaining the existing 2-level-M distinct-row guard
  (mt=7 `[m_hi=1,m_lo=3]` vs mt=13 `[m_hi=3,m_lo=1]` must map to distinct correct rows)
  and the base-only negative control.
  - Positive Tests (expected to PASS):
    - The reconstructed 512-wide fused tiles (base + fused low-rank) match a locally
      computed gold for sampled `(m_hi, m_lo, c)` including nonzero `m_hi`, all `m_lo`, and
      an N-chunk boundary (e.g. last chunk `c=23`, covering n_tiles 92..95).
    - The 2-level-M distinct-row guard and the base-only negative control still assert.
  - Negative Tests (expected to FAIL / be rejected):
    - A swapped `(m_hi, m_lo)` or wrong `4c+j` sub-tile mapping makes a checked tile
      diverge from gold — the check must assert and fail locally, before remote spend.

- **AC-6: Evidence trail — every scored change and every disposition recorded.**
  - Positive Tests (expected to PASS):
    - `benchmark.csv` gets one row per perf-affecting scored candidate (D1, D2), with
      latency, speedup, and notes.
    - `candidates.jsonl` gets one object per candidate with the parent DAG
      (`lora_v2_mblk4` parent `lora_v1`; `lora_v3_bf16_split` parent `lora_v2_mblk4`),
      per-seed rel-L2, worst rel-L2, gate, latency, and the profile metrics.
    - A `profile/` digest is written for each scored candidate.
    - D3 (resident `b`) and D4 (B-sweep {2,8}) are each dispositioned as either a measured
      candidate or a **recorded model-based reject with rationale** (carrying the sibling
      finding: matmul B=4 optimal, B=8 full-run regressed 0.968x, B=16 0.519x; resident
      `b` saves only ~2.3% of the D1 HBMrd on a PE-bound op).
  - Negative Tests (expected to FAIL / be rejected):
    - A scored candidate with no `benchmark.csv` / `candidates.jsonl` row, or a skipped
      D3/D4 with no recorded rationale — incomplete evidence.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. This
is a **highly deterministic design**: the draft prescribes a specific, evidence-backed
port of the sibling matmul ladder, so the bounds are narrow.

### Upper Bound (Maximum Acceptable Scope)
Both D1 (`lora_v2_mblk4.py`, fp32 fallback) and D2 (`lora_v3_bf16_split.py`, base-only
bf16x2 3-product split) implemented and scored full-5-seed; D2 promoted if it passes
AC-1 and beats D1 in same-session latency, with D1 retained as the fp32 fallback; the
offline sim (AC-4) and the extended host layout check (AC-5) committed; and D3/D4
dispositioned (measured only if the D1/D2 profile shows a DMA/blocking gap, otherwise a
recorded model-based reject). All within the ≤5-iterations-per-direction budget.

### Lower Bound (Minimum Acceptable Scope)
D1 (`lora_v2_mblk4.py`) implemented, passing AC-1 full-5-seed, with HBM read collapsed
toward the ~2.1 GB band and no spill, recorded as the fp32 fallback and a strict
improvement over lora_v1 (0.382x) — plus the AC-5 host layout guard for the new store
mapping. This is the minimum that satisfies "make the base GEMM fast" without the bf16
promotion; D2 is pursued next but D1 alone is a coherent, correct, evidence-backed stop.

### Allowed Choices
- **Can use:** the sibling matmul recipes as the structural template (`matmul_v2_b4` for
  D1, `matmul_v3_bf16_split` for D2); the 2-level `(m_hi, m_lo)` index as the natural
  B=4 block/member split; fp32 PSUM accumulation; the identity-transpose idiom; the
  compensated 3-product bf16 split (drop `lo@lo`) on the base GEMM only; `NCHUNK=256` as
  a *diagnostic* fallback if the 512-wide 4-sub-tile store shows codegen/pressure issues.
- **Cannot use:** splitting the down-projection or up-projection into bf16 (no PE upside,
  needless risk); a resident fp32 `lhs_t` alongside bf16 limbs in D2 (spill risk); plain
  single-limb bf16 anywhere (fails the gate at ~2.35e-3); promoting on a `--fast` screen
  alone; editing anything under `../../AccelOpt/NKIBench/`; hand-tuning a baseline.

> **Note on Deterministic Design:** the draft fixes the algorithm (port the sibling
> ladder). The upper and lower bounds differ only in how far down the ladder the phase
> reaches (D1 alone vs D1+D2 promoted); the *method* at each rung is prescribed, not a
> free choice.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

D1 (`lora_v2_mblk4.py`), per `m_hi` block (8 blocks), members `m_lo ∈ 0..3`:
```
for m_hi in affine_range(8):                       # M-block == m_hi
    # prologue: shared transposed lhs + fp32 down-proj, for all 4 members
    for m_lo in affine_range(4):
        for kt in affine_range(40):
            lhs_sb  = load x tile [m_in, k_in]
            lhs_t[m_lo,kt] = transpose -> [k_in, m_in]         # identity idiom
        tT_psum[m_lo] = zeros([R,128]); accumulate over kt:
            tT_psum[m_lo] += nc_matmul(a_local[kt], lhs_t[m_lo,kt])   # (x@a)^T
        tT[m_lo] = copy(tT_psum[m_lo]) -> fp32 SBUF                    # before N loop
    for c in affine_range(24):                     # N_CHUNK = 512
        acc = zeros([4,128,512])                    # 4 distinct fp32 PSUM banks
        for kt in affine_range(40):
            w_chunk = load v2[kt, :, 512c:512c+512]  # ONCE, reuse across 4 members
            for m_lo in affine_range(4):
                acc[m_lo] += nc_matmul(lhs_t[m_lo,kt], w_chunk)        # base x@w
        b_chunk = load v4[:, 512c:512c+512]          # ONCE, reuse across 4 members
        for m_lo in affine_range(4):
            acc[m_lo] += nc_matmul(tT[m_lo], b_chunk)                  # fused (x@a)@b
            out_sb = copy(acc[m_lo])
            for j in range(4):                       # 512-wide -> 4 sub-tile stores
                store out_sb[:,128j:128j+128] -> v5[m_hi,m_lo,:,4c+j,:]
```

D2 (`lora_v3_bf16_split.py`) is a localized diff on D1: in the prologue, from the
transient fp32 `lhs_t_f`, first accumulate `tT_psum[m_lo] += nc_matmul(a_local[kt],
lhs_t_f)` (fp32 down-proj, AC-3.1/AC-3.2), then build resident `lhs_hi/lhs_lo` bf16 limbs
and free `lhs_t_f`; in the N-chunk loop, split `w_chunk` into `w_hi/w_lo` per (c,kt) and
replace the single base matmul with the 3-product `lhs_hi@w_hi + lhs_hi@w_lo +
lhs_lo@w_hi`. The fused `tT[m_lo].T @ b_chunk` stays a single fp32 `nc_matmul`. The low
limb is produced by `nisa.tensor_tensor(..., op=nl.subtract)` into a bf16 destination (no
extra fp32 buffer), exactly as the sibling.

### Relevant References
- `runs/lora_v1.py` — the phase-1 kernel; the fused single-bank low-rank residual and the
  2-level (m_hi, m_lo) M-index to preserve.
- `../matmul/runs/matmul_v2_b4.py` — the fp32 M-block B=4, N_CHUNK=512 structure to port for D1.
- `../matmul/runs/matmul_v3_bf16_split.py` — the compensated bf16x2 3-product split body for D2
  (note: it keeps NO resident fp32 lhs_t — the source of the AC-3.1 down-proj fold).
- `../matmul/runs/offline_bf16_split_sim.py` — the offline sim to port for AC-4 (RNE limb
  construction, independent fail-closed reference control, quadrature prediction).
- `runs/_layout_check.py` — the host numpy layout check to extend for the 512-wide store (AC-5).
- `profile/lora_v1_digest.md` — the phase-1 profile (PE 88.5%, HBMrd 7813 MB) motivating the work.

## Dependencies and Sequence

### Milestones
1. **D1 — fp32 M-block fallback (`lora_v2_mblk4.py`).**
   - Phase A: extend `_layout_check.py` for the 512-wide 4-sub-tile store (AC-5); run it,
     confirm all guards pass.
   - Phase B: implement D1 (port `matmul_v2_b4` structure + lora fused residual);
     `--fast` screen, then full 5-seed (AC-1, AC-2); record `benchmark.csv` +
     `candidates.jsonl` (parent `lora_v1`) + `profile/` (AC-6). This becomes the fp32
     fallback.
2. **D2 — bf16x2 base-only split promotion candidate (`lora_v3_bf16_split.py`).**
   - Phase A: commit `runs/offline_lora_bf16_split_sim.py`; run it; confirm composite
     rel-L2 < 1.3e-5 (AC-4). **Gate D2 on this before any remote spend.**
   - Phase B: implement D2 as a localized diff on D1 with the AC-3.1 (down-proj fold) and
     AC-3.2 (`tT_psum` once-per-block) guards; extend `_layout_check.py` if needed to
     exercise the fold order; `--fast` then full 5-seed (AC-1, AC-3); record evidence.
   - Phase C: if D2 passes AC-1 and beats D1 in same-session latency beyond noise,
     PROMOTE D2 and keep D1 as the fp32 fallback.
3. **D3 / D4 — contingencies.**
   - Only if the D1/D2 profile shows a DMA or blocking gap: measure D3 (resident `b`)
     and/or D4 (B∈{2,8}). Otherwise record a model-based reject with rationale (AC-6),
     carrying the sibling finding.

Dependencies: AC-5 (layout guard) precedes D1 coding. D1 (AC-2) precedes D2 (AC-3). AC-4
(offline sim) precedes the first D2 remote run. AC-6 evidence accrues at every scored
step. AC-1 gates every promotion.

## Task Breakdown

Each task includes exactly one routing tag (`coding` = Claude, `analyze` = Codex).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Extend `runs/_layout_check.py` for the N_CHUNK=512 4-sub-tile store mapping; keep the 2-level-M distinct-row guard and base-only negative control; run and confirm all guards pass | AC-5 | coding | - |
| task2 | Implement `runs/lora_v2_mblk4.py` (fp32, B=4 = m_hi/m_lo, N_CHUNK=512; load `w` K-tile and `b_chunk` once per chunk, reuse across 4 members; fuse low-rank into each bank; 4-sub-tile store) | AC-2 | coding | task1 |
| task3 | Score D1: `--fast` screen then full 5-seed via `verify.py --op lora`; verify `l2_norm_passed` on all 5; record `benchmark.csv` + `candidates.jsonl` (parent `lora_v1`) + `profile/` digest | AC-1, AC-2, AC-6 | coding | task2 |
| task4 | Port `runs/offline_lora_bf16_split_sim.py` from the sibling: full-shape composite forward at seed 42, split base only, independent fail-closed reference control; run and confirm composite rel-L2 < 1.3e-5 | AC-4 | coding | - |
| task5 | Implement `runs/lora_v3_bf16_split.py` as a localized diff on D1: base-GEMM-only bf16x2 3-product split, down/up-projection fp32, with the down-proj-fold (AC-3.1) and `tT_psum`-once-per-block (AC-3.2) guards | AC-3 | coding | task3, task4 |
| task6 | Score D2: `--fast` then full 5-seed; verify `l2_norm_passed` on all 5 and worst rel-L2 ≈ device quadrature; record evidence (parent `lora_v2_mblk4`); PROMOTE if it beats D1 same-session, keep D1 as fp32 fallback | AC-1, AC-3, AC-6 | coding | task5 |
| task7 | Read the D1/D2 profile; if a DMA/blocking gap appears, measure D3 (resident `b`) and/or D4 (B∈{2,8}); otherwise record a model-based reject with rationale carrying the sibling finding | AC-6 | coding | task6 |
| task8 | Independent numeric review of the D2 offline-sim math and the composite-error / device-quadrature argument (confirm the base-only split error is diluted ~11× by the fp32-dominant low-rank output; sanity-check the ~6e-7 device prediction) | AC-3, AC-4 | analyze | task4 |

## Claude-Codex Deliberation

### Agreements
- D1 (B=4, N_CHUNK=512, `w` reuse across 4 members, 512-wide stores) is the right fp32
  fallback and matches the sibling matmul evidence and the HBM model.
- D2 (split base GEMM only, keep both low-rank projections fp32, require same-session
  latency evidence before promotion) is technically sound as a promotion candidate.
- The offline full-shape seed-42 guard, the extended host layout guard, and the
  evidence-trail requirements are appropriate and should gate the work.
- `b_chunk` must be loaded once per N-chunk and reused across the 4 members; loading it
  per member leaves avoidable traffic and weakens the HBM model.
- No-spill is a real acceptance signal: HBM write must stay ≈ 201 MB and HBM read near
  the modeled ~2.1 GB band; a spill means the kernel is broken relative to its design.

### Resolved Disagreements
- **D2 down-projection operand lifetime (Codex first-pass CORE_RISK):** the sibling
  `matmul_v3_bf16_split` keeps NO resident fp32 `lhs_t` — only bf16 limbs. A naive port
  would leave lora's fp32 down-projection `x@a` with no operand. **Resolution:** fold the
  fp32 down-proj accumulation into the transpose/split loop, consuming the transient fp32
  `lhs_t_f` before it is freed (AC-3.1). Keeps SBUF ≈ 108 KB/partition; avoids the
  reject-worthy "resident fp32 + bf16 limbs together" ≈192 KB spill risk.
- **D2 `tT_psum` loop placement (Codex second-pass REQUIRED_CHANGE):** the fold is only
  correct if `tT_psum` is accumulated exactly once per B=4 M-block over K in the prologue,
  before the N-chunk loop. Accumulating it inside the N-chunk loop would re-run the
  down-projection 24× and silently corrupt the output. **Resolution:** pinned as the hard
  guard AC-3.2 (prologue-only accumulation; N-loop only reads the completed `tT`).
- **"Composite error below the fp32 floor" framing (Codex CORE_RISK):** the draft notes
  the offline composite (3.9e-7) is below lora_v1's fp32 floor (4.87e-7). Codex correctly
  cautions this must not be read as "bf16 reduces error." **Resolution:** the plan gates
  purely on rel-L2 ≤ 2e-5 (AC-1) and predicts the **device** rel-L2 as the quadrature
  `sqrt(fp32_floor² + diluted_bf16²) ≈ ~6e-7` (the fp32 floor DOMINATES here, inverting
  the sibling matmul case where the bf16 term dominated a sub-1e-6 floor). The offline
  3.9e-7 is an idealized-composite datum, not a promise the device beats its own floor.
- **Offline validation scope (Codex QUESTION):** **Resolution:** the offline sim computes
  the FULL-shape composite forward at seed 42 (the single gated input), not a stratified
  subset — this matches the sibling `offline_bf16_split_sim.py` (which draws full shapes)
  and avoids tile-dependent error being missed.
- **`N_CHUNK=256` diagnostic (Codex ALTERNATIVE):** accepted as an allowed *diagnostic*
  fallback if the 512-wide 4-sub-tile store shows codegen/pressure issues, not as a
  planned rung.

### Convergence Status
- Final Status: `converged` (second Codex pass: no REQUIRED_CHANGES remain and no
  UNRESOLVED opposite opinions, conditional only on accepting the AC-3.2 loop-placement
  clarification, which is adopted). Two convergence rounds executed (Codex first-pass
  critique → candidate plan v1 → Codex second-pass review → this plan).

## Pending User Decisions

- **DEC-1: Is crossing 1.0x (beating the 14.6645 ms baseline) a HARD phase-2 success
  requirement, or a directional target?**
  - Claude Position: **Directional target.** The single hard gate is correctness (AC-1,
    rel-L2 ≤ 2e-5). Success = a correct candidate that strictly improves on lora_v1
    (0.382x); crossing 1.0x is the strong expected outcome (the sibling ladder reached
    1.017x fp32 and 1.274x with the bf16 split for this identical base GEMM) and D2 should
    land near the sibling's 1.2–1.3x, but the promotion decision is "same-session latency
    beats the predecessor," not an absolute 1.0x cliff.
  - Codex Position: Open question — "Is D1 required to beat baseline by any margin, or is
    any stable >1.0x acceptable?" (raised for explicit human decision).
  - Tradeoff Summary: Treating 1.0x as HARD would fail an otherwise-correct, materially
    faster-than-lora_v1 kernel that lands at, say, 0.95x — over-strict given phase-1 was
    0.382x. Treating it as directional keeps correctness as the only hard gate while still
    aiming for (and very likely hitting) the sibling's >1.0x neighborhood.
  - Decision Status: `PENDING`

- **DEC-2: Must D3 (resident `b`) and D4 (B-sweep {2,8}) each have at least one MEASURED
  candidate, or is a recorded model-based reject acceptable?**
  - Claude Position: **Model-based reject acceptable** (with recorded rationale), unless
    the D1/D2 profile shows a DMA or blocking gap — then measure. This matches the
    established repo convention (the sibling matmul recorded model/infeasibility rejects,
    e.g. B=6 infeasible, and D2/D3 not triggered) and the physics: the op is PE-bound, and
    resident `b` saves only ~2.3% of the D1 HBMrd, so a measured run is unlikely to move
    the wall.
  - Codex Position: Open question — "Are model-based rejects for D3/D4 acceptable in the
    evidence trail, or must each have at least one measured candidate?"
  - Tradeoff Summary: Requiring a measured D3/D4 spends remote runs on directions the
    model predicts are near-no-ops (rigor vs cost). A recorded model-reject preserves the
    evidence trail cheaply but rests on the model; the profile-gated escalation ("measure
    only if a gap appears") is the middle path.
  - Decision Status: `PENDING`

> Both pending decisions have a clear Claude recommendation and do NOT block starting the
> implementation (they affect the *promotion/stop* judgment and the *evidence rigor* for
> contingencies, not the D1/D2 build). They were surfaced by Codex for explicit human
> sign-off; resolve them at plan review or when starting the RLCR loop.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Phase", "task1", "DEC-", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting kernel/check sources.
- Use descriptive, domain-appropriate naming in code (e.g. `m_hi`/`m_lo`, `lhs_hi`,
  `w_lo`, `tT_psum`, `N_CHUNK`), matching the style of `lora_v1.py` and the sibling
  matmul kernels.

### Correctness Guards / Watch-Items (carried from the draft, sharpened by deliberation)
- **2-level M-index as block/member split:** `m_hi` is the block, `m_lo` (0..3) the
  member; store to `v5[m_hi, m_lo, :, n_tile, :]`. A swapped pair scrambles rows — the
  `_layout_check.py` mt=7 vs mt=13 guard covers this; extend it to the 512-wide store
  (AC-5) before D1.
- **512→4-sub-tile store mapping:** chunk `c` → n_tiles `4c..4c+3`; `out_sb[:, 128j:128j+128]
  → v5[m_hi,m_lo,:,4c+j,:]`, `j∈0..3`, `N_CHUNKS=24` so `4c+j∈0..95`. The one new indexing
  vs the sibling matmul (whose out N axis was flat) — verify numerically.
- **`tT` PSUM→SBUF copy before the N-loop:** unchanged from v1; must survive across the
  24 N-chunks (PSUM banks are reused by `acc`).
- **bf16 split touches ONLY the base GEMM** (D2): down-proj and fused low-rank stay fp32;
  the offline sim reconfirms composite < 1.3e-5 before any remote spend.
- **Full 5-seed before promoting:** `--fast` can mislead on SBUF/PSUM-pressure regimes
  (the sibling B=8 looked best on `--fast` but regressed on the full run).


--- Original Design Draft Start ---

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

--- Original Design Draft End ---
