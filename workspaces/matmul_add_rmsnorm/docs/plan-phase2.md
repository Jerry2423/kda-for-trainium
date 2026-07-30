# matmul_add_rmsnorm — Phase 2 Plan: Break the fp32 PE Floor with a Compensated bf16x2 Split-Matmul

## Goal Description

Phase-1 `matmul_add_rmsnorm_v1` is correct and PE-bound at the fp32 systolic floor
(PE=96%, MFU=46%, 0.9612 ms, 3.920x over the 3.768493 ms baseline; full-5-seed PASS). The
trn2 PE array is bf16-native, so a correct fp32 GEMM runs multiple internal bf16 passes and
caps MFU near ~46%. Every other engine is hidden under that floor (Vec 15%, Scl 9%, DMA 20%;
HBMrd 84 MB is already the ~80 MB single-pass floor), so the ONLY lever that cuts wall-clock
is cutting PE time — and the only correctness-viable way to do that on this shape is to run
the main matmul in bf16 arithmetic with a compensated two-limb split.

This phase transfers the two siblings' already-PROMOTED win — a compensated **bf16x2
split-matmul** (each fp32 operand → two bf16 limbs; accumulate three bf16 products
`hi@hi + hi@lo + lo@hi` in fp32 PSUM; drop the negligible `lo@lo` term) — to this op. The
siblings measured `rmsnorm_matmul` 1.066x → 1.363x (+28%) and `add_rmsnorm_matmul` 3.754x →
4.632x (+23%), both full-5-seed PASS.

The one NEW risk vs the siblings is real and is what this plan gates on carefully: this op is
GEMM → add → norm, so the bf16 matmul error lands in `y = x@w + z`, and `y` feeds **both**
`inv_rms[m] = 1/sqrt(mean_n(y²)+eps)` **and** the output numerator `y·g` — a composite norm
path no sibling sim ever exercised. It is de-risked by a zero-remote-spend offline numpy sim
(`runs/offline_bf16_split_sim.py`, already written and run): fp32 control rel-L2 = 0.000e+00
(model exact), plain bf16 = 2.350e-3 (fails the gate 117x), **bf16x2 3-product WORST over 7
seeds = 4.454e-6** (~4.5x under the 2e-5 gate), 4-product = 3.491e-6 (negligible improvement).
The composite norm path partly SELF-CANCELS (a coherent relative perturbation `δ` in `y`
scales the numerator by ~`δ` and `inv_rms` by ~`−δ`, so `out = y·inv_rms` is first-order
insensitive to a common-mode scaling of `y`). Predicted on-device rel-L2 combines the fp32
floor and the bf16 error in QUADRATURE — `sqrt(1.46e-5² + 4.454e-6²) = 1.526e-5` — the same
method that predicted 1.526e-5 and measured 1.528e-5 on the `add_rmsnorm_matmul` sibling.

The work is **one promotion candidate** plus a **costed correctness-rescue fallback**:
- **v2** (`runs/matmul_add_rmsnorm_v2_bf16_split.py`) — the compensated bf16x2 3-product split
  built DIRECTLY on v1. This op does NOT need the sibling's v2 enabler refactor (see the two
  op-specific simplifications below), so the v1→v2 diff is LOCALIZED to the matmul; the
  residual add, RMSNorm, and full-width output scale are byte-for-byte fp32 v1. Because there
  is no refactor, **v1 itself is the same-session fp32 control** — there is no refactor-noise
  to isolate. This is the intended promotion. Directional expectation ~0.75 ms / ~5.0x
  (~3 bf16 passes vs fp32's ~4), but promotion depends ONLY on measured correctness + a
  measured out-of-noise latency win, not on hitting that number.
- **v2b** (`runs/matmul_add_rmsnorm_v2b_bf16_split4.py`) — the 4-product split (adds the
  dropped `lo@lo` term). A CORRECTNESS RESCUE ONLY, built iff v2 fails or is marginal (see
  AC-5). It is NEVER a promotion candidate: the analogous sibling v3b measured +28% latency
  for a ~1.6% rel-L2 gain (a false repair, because the fp32 floor dominates).

Two op-specific simplifications vs the sibling bf16-split kernel, both KEPT from v1:
1. **`g` is NOT folded into `w`.** `g` is length-N on the OUTPUT free axis, applied AFTER the
   norm (`out = y·g/rms`). Folding `g[n]` into `w[k,n]` would scale `y` before the norm and
   change `rms = sqrt(mean(y²))` — algebraically wrong. (Contrast the sibling, whose per-K `g`
   folded cleanly into the resident weight.)
2. **`inv_rms` does NOT commute out to a post-scale eviction.** The norm reduces over N, so the
   ENTIRE `[128,N]` row `y` must be assembled in SBUF before `inv_rms` is known; it cannot be
   applied chunk-by-chunk at PSUM→SBUF eviction the way the sibling did (its norm reduced over
   K, independent of the matmul output). v1's structure — assemble full `y`, single full-N
   reduce, full-width output scale — is already correct and is KEPT.

The kernel's raw-2D I/O and exact signature `kernel(x_tensor, w_tensor, eps, z_tensor,
g_tensor)` are preserved throughout. This plan implements ONLY `matmul_add_rmsnorm`; it does
not touch the benchmark definition or any other operator.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. Scoring command (run from inside `workspaces/matmul_add_rmsnorm/`):
`python3 ../../verify.py
--op matmul_add_rmsnorm --candidate runs/<file>.py` (add `--fast` for the seed-42 quick check;
drop it for the full 5-seed gate). Numeric per-seed rel-L2 is read via `runs/rel_l2_probe.py`
(copied from the sibling; `verify.py` prints only the bool gate + latency).

- AC-1: **v2 (`runs/matmul_add_rmsnorm_v2_bf16_split.py`) is a correct compensated bf16x2
  3-product split built directly on v1.** The split order is PINNED and auditable. For the
  weight, once at load per K-tile: `w_hi = bf16(w_f)`, `w_res = w_f − w_hi` (fp32, exact for
  O(1) magnitudes), `w_lo = bf16(w_res)`; store `w_hi`, `w_lo` as resident `[16,128,N]` bf16.
  For the activation, per M-tile per K-sub-tile: transpose the RAW `x` sub-tile via the exact
  fp32 identity `nc_matmul` (unchanged from v1) to `xT_f`, THEN `xT_hi = bf16(xT_f)`,
  `xT_res = xT_f − xT_hi`, `xT_lo = bf16(xT_res)` — the split is applied to the TRANSPOSED
  fp32 value, matching the offline sim's element-wise split (splitting after the transpose is
  identical to before it because the transpose is exact and bf16 rounding is element-wise; v1
  already performs this identity fp32 transpose so any transpose rounding is already inside
  v1's fp32 floor). `bf16(.)` is `nl.copy(dtype=nl.bfloat16)`, round-to-nearest-even. Accumulate
  the three bf16 products `nc_matmul(xT_hi, w_hi) + nc_matmul(xT_hi, w_lo) + nc_matmul(xT_lo,
  w_hi)` into the fp32 PSUM bank, dropping `xT_lo@w_lo`. The fixed product summation order
  (hi@hi, then hi@lo, then lo@hi) is part of the pinned contract and must be preserved for
  auditability; accumulation is normal fp32 PSUM accumulation with no intermediate narrowing.
  - Positive Tests (expected to PASS):
    - Full-5-seed on-device `l2_norm_passed = True` at seeds `[0,21,42,63,84]`.
    - `--fast` (seed 42) `l2_norm_passed = True` as the quick pre-check before the full run.
    - Measured on-device rel-L2 under the 2e-5 gate on every seed (offline predicts a bf16-only
      term of 4.454e-6 and a device quadrature of ~1.53e-5); the completed offline 7-seed gate
      (worst 4.454e-6) is the pre-authorization for one remote attempt.
  - Negative Tests (expected to FAIL):
    - Plain single-limb bf16 matmul (no compensation) — offline 2.350e-3, ~117x over the gate.
    - Dropping `xT_hi@w_lo` or `xT_lo@w_hi` (a 2-product split) — loses a compensation term and
      pushes rel-L2 toward the plain-bf16 regime; must fail or badly degrade the L2 gate.
    - Splitting before the transpose, or reversing the pinned `fp32 → hi → residual → lo` order
      (e.g. building `w_lo` from a bf16-rounded residual of a bf16 value) — invalidates the
      offline authorization and must be rejected.
    - Performing the residual add, the RMSNorm reduction, `eps`/`rsqrt`, or the output scale in
      reduced precision — precision loss must stay confined to the matmul.

- AC-2: **v2 latency beats v1 out-of-noise.** Promotion requires the measured p50 latency to
  beat the v1 promoted datum by more than the noise band (>1.8%; see DEC-2), measured with a
  FULL 5-seed run (NOT `--fast`) repeated at least twice for stability (the ">=2×" is a run-count
  requirement, not a speedup target), compared against BOTH the v1 promoted row (0.9612 ms) and
  a same-session v1 rerun as the noise anchor. The directional expectation is ~0.75 ms (~5.0x)
  but the exact figure is NOT itself an acceptance requirement.
  - Positive Tests (expected to PASS):
    - Two full-5-seed measurements of v2 both beat the same-session v1 anchor by >1.8%.
    - No large p95/variance regression relative to v1 across the repeated runs.
  - Negative Tests (expected to FAIL):
    - A single `--fast` measurement used as the promotion basis.
    - A within-noise or regressed latency promoted anyway.
    - Interpreting ">=2×" as a required 2x speedup (it is a run-count requirement; the expected
      gain is ~+23–28% per the siblings, not 2x).

- AC-3: **The same-session fp32 floor is measured, not borrowed.** Because v2's diff is
  localized to the matmul and the fp32 epilogue is byte-for-byte v1, v1 IS the fp32 control:
  measure v1's ACTUAL per-seed rel-L2 and p50 latency THIS session (via `runs/rel_l2_probe.py`,
  full 5 seeds) to replace the borrowed sibling floor of 1.46e-5. This is a DIAGNOSTIC input to
  the margin estimate; it does not change the hard gate (AC-1's on-device rel-L2 < 2e-5 on every
  seed is the promotion gate regardless of the recomputed margin).
  - Positive Tests (expected to PASS):
    - v1's per-seed rel-L2 and latency are recorded this session as the control anchor.
    - If the measured v1 floor is materially above 1.46e-5, the predicted v2 quadrature margin
      is recomputed and the danger-band interpretation (AC-4) is re-checked before promoting.
  - Negative Tests (expected to FAIL):
    - Promoting v2 using only the borrowed 1.46e-5 floor when a same-session v1 rel-L2 is cheaply
      available.
    - Treating the recomputed margin as a substitute for the hard 2e-5 per-seed gate.

- AC-4: **v2 numeric per-seed rel-L2 is recorded, with an explicit danger band.** Record the
  numeric rel-L2 for every seed (not only worst/mean, and not only pass/fail) and the per-seed
  delta versus the same-session v1 floor. Define a DANGER BAND: a worst-seed rel-L2 ≥ 1.8e-5 is
  flagged as marginal even if technically < 2e-5 (the predicted 1.526e-5 quadrature and the
  pessimistic linear-sum bound of ~1.905e-5 mean the margin is real but thin).
  - Positive Tests (expected to PASS):
    - `candidates.jsonl` records per-seed rel-L2 and per-seed delta vs the same-session v1 floor.
    - A worst-seed rel-L2 in `[1.8e-5, 2e-5)` is explicitly labeled "marginal — v2b triggered".
  - Negative Tests (expected to FAIL):
    - Recording only a pass/fail bool or only the worst-seed value.
    - A worst-seed rel-L2 ≥ 1.8e-5 promoted without the marginal flag and the v2b decision.

- AC-5: **v2b (`runs/matmul_add_rmsnorm_v2b_bf16_split4.py`) is a correctness rescue only,
  gated on a true danger band.** Build v2b IFF v2 FAILS the 2e-5 gate on any seed OR is marginal
  (worst-seed rel-L2 ≥ 1.8e-5). At the predicted passing ~1.53e-5, v2b is NOT built and the skip
  decision is recorded. v2b is v2 plus ONLY the fourth product `nc_matmul(xT_lo, w_lo)` — the
  identical pinned split and order, one term added, nothing else changed. v2b is NEVER a
  promotion candidate (offline it moves rel-L2 4.454e-6 → 3.491e-6, ~22% of the bf16 term and
  ~1% of the quadrature, while adding a 4th matmul pass ~+25% PE time; the sibling v3b measured
  +28% latency for ~1.6% rel-L2 = MEASURED-REJECT).
  - Positive Tests (expected to PASS):
    - v2b is attempted only after a failing or marginal (≥1.8e-5) v2 rel-L2 datum.
    - When v2 is comfortably under the gate (< 1.8e-5), the v2b-skip decision is recorded as a
      negative datum.
    - If built, v2b changes exactly one variable (adds `xT_lo@w_lo`); the split order is
      otherwise identical to v2.
  - Negative Tests (expected to FAIL):
    - Proactively building v2b before v2 is measured, or building it when v2 passes < 1.8e-5.
    - Promoting v2b on latency (it is a correctness rescue, expected slower than v2).
    - A v2b that changes the split order or any term beyond adding `lo@lo`.

- AC-6: **v2 profiler digest is captured; HBM-unchanged and the K=2048 instruction/PSUM story
  are validated.** Record PE%, MFU%, Vec%, Scl%, DMA%, HBMrd, HBMwr for v2 and compare against
  v1 (0.9612 ms, PE 96, MFU 46, HBMrd 84 MB, HBMwr 34 MB). The bf16 limbs are built on-chip from
  the same fp32 HBM loads, so HBM should be materially unchanged. Confirm the 3-product shape
  (K=2048 → 16 K-tiles × 3 products = 48 `nc_matmul` issues per 512-wide N-chunk) does NOT
  become instruction-issue-bound and that PSUM pressure is unchanged (one `[128,512]` fp32
  accumulator bank per chunk, exactly as v1). If the digest shows PE idle (<~95%), apply the
  sibling's bit-exact "split `aT` limbs directly from PSUM" reschedule
  (`xT_hi = bf16(psum_t)`, `xT_lo = bf16(psum_t − xT_hi)`, dropping the fp32 `xT_f`) as a
  latency-only, zero-precision-change tweak that requires its own same-session comparison.
  - Positive Tests (expected to PASS):
    - A digest with all seven metrics is written under `profile/` for v2.
    - HBMrd/HBMwr within a small margin of v1's 84 MB / 34 MB.
    - PE utilization is characterized; if <~95% the reschedule tweak is applied and re-measured.
  - Negative Tests (expected to FAIL):
    - Promotion recorded with no profiler digest.
    - A material HBMrd/DMA rise (limb spill/reload) or an instruction-issue-bound PE left
      unflagged in the evidence.

- AC-7: **Correctness invariants are never regressed** (shared by v2 and v2b): the raw-2D I/O
  and exact signature `kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor)` preserved; `y = x@w
  + z` residual add done in fp32 BEFORE the norm; the fp32 RMSNorm reduction over the N free
  axis (`square`/`tensor_reduce`/`mean_eps`/`rsqrt`); `eps` added AFTER the `/N` mean (two-op
  `mean_eps = sumsq·(1/N) + eps`, matching the reference); `g` applied on the OUTPUT free axis
  as a `[1,N]→[128,N]` broadcast multiply, NEVER folded into `w`; the full-width `[128,N]`
  output store retained (phase-1 R1 measured it 6.4% faster than per-chunk).
  - Positive Tests (expected to PASS):
    - Both kernels accept `(x_tensor, w_tensor, eps, z_tensor, g_tensor)` and return `(M, N)`.
    - `eps` is consumed as a runtime scalar added after the `/N` mean; `g` stays on the output axis.
  - Negative Tests (expected to FAIL):
    - Changing the signature, tiling the I/O differently, or hard-coding `eps`.
    - Folding `g` into `w`, or performing the RMSNorm reduction / `eps` / `rsqrt` in bf16.
    - Reverting to a per-512-chunk output store (a recorded phase-1 measured-reject).

- AC-8: **Evidence is recorded per the KDA workflow.** `benchmark.csv` gets one row per
  perf-relevant candidate (v2, and v2b only if built) plus the gated decisions (offline-gate
  authorization already recorded; v2b build-or-skip); `candidates.jsonl` gets a DAG node v2→v1
  (and v2b→v2 if built) with per-seed rel-L2, per-seed latency, metrics, and the audited split
  order; `profile/` gets the before/after digest. The offline-sim node
  `matmul_add_rmsnorm_offline_bf16_split_sim` is already appended.
  - Positive Tests (expected to PASS):
    - `benchmark.csv` and `candidates.jsonl` contain the v2 row/node with the v2→v1 parent link.
    - Per-seed rel-L2 and latency are recorded (not only worst/mean); the split order is stated.
  - Negative Tests (expected to FAIL):
    - A promoted candidate with no `benchmark.csv` row or no `candidates.jsonl` node.
    - Evidence that omits the split order, the product summation order, or the DAG parent link.

- AC-9: **Closed directions are respected (record-only, not re-explored).** These are closed by
  the profile or by sibling precedent and are NOT to be re-tested without new contradicting
  evidence: off-PE transpose (`dma_transpose` fp32/bf16 SBUF→SBUF infeasible, `nc_transpose`
  (vector) regressed +2%, `load_transpose2d` hidden — and transposes are already hidden under
  the PE-bound matmul); output-store structure (phase-1 R1 full-width store 6.4% faster);
  M-blocking / loop reorder (w is fully resident, single HBM pass, no weight-reload bubble);
  N_CHUNK≠512 (512 = one fp32 PSUM bank = maximal moving-free width); `g`-into-`w` fold
  (algebraically wrong for this mirror op).
  - Positive Tests (expected to PASS):
    - The identity-matmul transpose, full-width output store, all-w-resident layout, and
      N_CHUNK=512 are retained.
    - Any decision to touch a closed direction cites new profiling evidence that contradicts the
      prior result.
  - Negative Tests (expected to FAIL):
    - Reopening off-PE transpose, output-store chunking, M-blocking, or N_CHUNK≠512 without new
      contradicting evidence.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
One new tracked kernel source `runs/matmul_add_rmsnorm_v2_bf16_split.py` (compensated 3-product
bf16x2 split built directly on v1, with the pinned split order and fixed product summation
order), verified full-5-seed on device with per-seed rel-L2 recorded, a same-session v1 fp32
control (rel-L2 + latency) measured as the floor/noise anchor, a v2 profiler digest validating
HBM-unchanged and the K=2048 3-product instruction/PSUM story, all evidence recorded in
`benchmark.csv` / `candidates.jsonl` / `profile/`, and the gated v2b 4-product rescue implemented
ONLY if v2 fails or is marginal (≥1.8e-5). v2 is promoted if and only if it clears both the
full-5-seed L2 gate and the out-of-noise latency band; if PE comes back idle (<~95%) the
bit-exact PSUM-reschedule tweak is applied and re-measured. v1 is retained as the pure-fp32
fallback.

### Lower Bound (Minimum Acceptable Scope)
The v2 compensated bf16x2 3-product split kernel, verified full-5-seed on device and promoted if
it clears both gates, with its `benchmark.csv` row, `candidates.jsonl` node (parent v1) recording
per-seed rel-L2 and the audited split order, and a v2 profiler digest. If v2 fails the L2 gate or
regresses on latency, the minimum is a recorded negative datum that re-confirms the fp32 floor,
with v1 retained as the promoted fallback (and v2b built only if v2 failed on correctness).

### Allowed Choices
- Can use: the sibling `add_rmsnorm_matmul_v3_bf16_split.py` and
  `rmsnorm_matmul_v4_bf16_split.py` as structural templates for the limb construction and
  3-product accumulation; `nl.copy(dtype=nl.bfloat16)` (the RNE fp32→bf16 cast),
  `nisa.tensor_tensor` (subtract for the limb residual), `nisa.nc_matmul` (identity-transpose
  and the bf16 products), `nisa.activation` (`square`, `rsqrt`), `nisa.tensor_reduce`,
  `nisa.tensor_scalar`; the all-w-resident SBUF layout (two bf16 limbs = same bytes as v1's one
  fp32 `w`); M-outer loop; N_CHUNK=512; the two-op `mean_eps` `eps` form; the full-width output
  store; optionally the bit-exact "split limbs directly from PSUM" reschedule if PE is idle.
- Cannot use: editing `../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}` or
  hand-tuning a baseline; changing the kernel signature or raw-2D I/O; folding `g` into `w`;
  performing the residual add, RMSNorm reduction, `eps`/`rsqrt`, or output scale in reduced
  precision; plain single-limb bf16 for the main matmul; a 2-product split (dropping a
  compensation term); reversing the pinned split order or the fixed product summation order;
  reopening the closed directions (off-PE transpose, output-store chunking, M-blocking,
  N_CHUNK≠512) without new contradicting evidence; promoting on `--fast` alone or on a
  within-noise latency delta; treating v2b as a promotion/latency candidate.

> **Note on Deterministic Designs**: The draft specifies a highly deterministic design — the
> algorithm (3-product compensated bf16x2 with the pinned split order and fixed product
> summation order, `g` on the output axis, `inv_rms` computed from the full assembled `y`, fp32
> RMSNorm, two-op `eps`, full-width output store) is fixed by the offline gate and the sibling
> precedent. The upper and lower bounds therefore nearly converge on "correct, gated v2 +
> evidence"; the main allowed latitude is whether the PSUM-reschedule tweak is applied (only on
> an idle-PE digest) and whether the v2b rescue is triggered (only on a failing/marginal v2).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions,
> not prescriptive requirements.

### Conceptual Approach

v2 is v1 with the matmul localized-diff only. Preamble (once, replacing v1's fp32 `w_sb`):
```
for kt in K_TILES(16):
    w_f      = load(w[kt])                       # [128, N] fp32
    w_hi[kt] = bf16(w_f)                          # round-to-nearest-even
    w_res    = w_f - w_hi[kt]                     # fp32 tensor_tensor (exact for O(1))
    w_lo[kt] = bf16(w_res)                        # resident [16,128,N] bf16 limbs
```
Per M-tile (transpose + split, then the 3-product matmul; the fp32 epilogue is byte-for-byte v1):
```
x_sb = load(x[mt])                               # [128, K] fp32
for kt in K_TILES(16):
    xT_f     = identity_matmul_transpose(x_sb[:, 128*kt:])   # exact fp32 [k_in, m_in]
    xT_hi[kt]= bf16(xT_f)
    xT_res   = xT_f - xT_hi[kt]                   # fp32
    xT_lo[kt]= bf16(xT_res)
for c in N_CHUNKS(4):                             # 512-wide, one fp32 PSUM bank
    acc = 0
    for kt in K_TILES(16):
        acc += nc_matmul(xT_hi[kt], w_hi[kt][:, chunk])   # hi @ hi
        acc += nc_matmul(xT_hi[kt], w_lo[kt][:, chunk])   # hi @ lo
        acc += nc_matmul(xT_lo[kt], w_hi[kt][:, chunk])   # lo @ hi   (drop lo@lo)
    y[:, chunk] = acc + z_tile                    # fp32 residual add before the norm
# --- byte-for-byte v1 fp32 epilogue over the full [128,N] y row ---
sumsq   = reduce_add( square(y) )                 # single full-N free-axis reduce
mean_eps= sumsq*(1/N) + eps                       # two-op tensor_scalar, eps AFTER /N
inv_rms = rsqrt(mean_eps)                         # [128,1]
out     = (y * inv_rms) * g_bcast                 # full-width [128,N], 2 ops + 1 store
```
The three deltas from v1 are: (1) `w` stored as two bf16 limbs instead of one fp32 tile;
(2) each transposed `xT` sub-tile split into two bf16 limbs; (3) the single fp32 `nc_matmul`
accumulation replaced by the 3-product bf16 accumulation. Everything below `y[:, chunk] = acc +
z_tile` is unchanged from v1.

v2b (only if triggered) = v2 with one added product inside the K-tile loop:
`acc += nc_matmul(xT_lo[kt], w_lo[kt][:, chunk])` (the `lo@lo` term). Nothing else changes.

SBUF feasibility (budget ~192 KB/partition): `w_hi` + `w_lo` bf16 `[16,128,N]` = 64 + 64 = 128
KB/part (identical to v1's one fp32 `w`); the M-loop transients (`x_sb`, `xT_hi/lo`, split
temps, `y`, `sq`, `z_tile`, `out_sb`) add ~43 KB for a ~171 KB M-loop peak; the preamble peak
(limbs + transient fp32 `w_f`/`w_res`) is ~145 KB. Both under budget. If the compiler ever
spills, compute the norm `sq` in place over `y`'s buffer before touching residency.

### Relevant References
- `runs/matmul_add_rmsnorm_v1.py` — the promoted Phase-1 fp32 kernel; the starting structure and
  the byte-for-byte fp32 epilogue that v2 keeps.
- `runs/offline_bf16_split_sim.py` + `profile/matmul_add_rmsnorm_offline_bf16_split_sim.txt` — the
  completed offline gate (pins seed 42, draw order `x→w→z→g`, `z=(M,N)`, `g=(N,)`, `eps=1e-5`;
  worst 3-product 4.454e-6; the composite norm path modeled exactly).
- `runs/rel_l2_probe.py` — copy from `../add_rmsnorm_matmul/runs/rel_l2_probe.py`; prints per-seed
  numeric rel-L2 from the profiler response (needed for AC-3/AC-4).
- `../add_rmsnorm_matmul/runs/add_rmsnorm_matmul_v3_bf16_split.py` — the PROMOTED 3-product bf16x2
  template (limb construction, pinned split order, 3-product accumulation).
- `../add_rmsnorm_matmul/docs/plan-phase2.md` — the sibling's phase-2 plan (quadrature method,
  danger-band and fallback discipline).
- `../rmsnorm_matmul/runs/rmsnorm_matmul_v4_bf16_split.py` — the original promoted split template.
- `../../AccelOpt/NKIBench/reference/matmul_add_rmsnorm_*numpy*.py` — the numpy reference defining
  the exact `eps`-after-mean, draw order, and I/O identity transform.
- `../../verify.py` — the remote profiler gate (`l2_norm_passed` across seeds; p50 speedup).

## Dependencies and Sequence

### Milestones
1. **Same-session fp32 floor + probe setup** (control anchor):
   - Phase A: copy `runs/rel_l2_probe.py` from the sibling workspace.
   - Phase B: measure v1's actual per-seed rel-L2 and full-5-seed p50 latency THIS session as the
     fp32 control floor and the promotion noise anchor (AC-3).
2. **v2 — compensated bf16x2 3-product split (the intended promotion)**: built directly on v1.
   - Phase A: write `runs/matmul_add_rmsnorm_v2_bf16_split.py` with the pinned split order and the
     fixed 3-product accumulation; keep the residual add, RMSNorm, and full-width output store
     byte-for-byte v1 (AC-1, AC-7).
   - Phase B: `--fast` seed-42 pre-check, then the FULL 5-seed L2 gate; record numeric per-seed
     rel-L2 and the per-seed delta vs the v1 floor; apply the danger band (AC-1, AC-4).
   - Phase C: measure v2 latency FULL 5-seed ≥2× vs the v1 promoted row and the same-session
     anchor, requiring a >1.8% out-of-noise win with no p95/variance regression (AC-2).
   - Phase D: capture the profiler digest; validate HBM-unchanged and the K=2048 3-product
     instruction/PSUM story; if PE idle (<~95%) apply the PSUM-reschedule tweak and re-measure
     (AC-6). Record all evidence and promote iff both gates clear (AC-8).
3. **Gated rescue / negative-datum handling** (conditional):
   - Step 1: if v2 fails the 2e-5 gate on any seed OR is marginal (worst ≥1.8e-5), implement the
     v2b 4-product rescue (one added `lo@lo` product, split order otherwise identical); otherwise
     record the v2b-skip decision (AC-5).
   - Step 2: if v2 fails or regresses on latency, record the negative datum re-confirming the
     fp32 floor; keep v1 promoted (lower bound).

Dependencies: v2 depends on the Milestone-1 v1 floor/anchor (its same-session control). The v2b
rescue depends on a failing or marginal v2 rel-L2. The offline gate (already PASS) is a satisfied
prerequisite. The closed directions (AC-9) are not on the critical path.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Copy `runs/rel_l2_probe.py` from `../add_rmsnorm_matmul/runs/` and adapt the `--op` default to `matmul_add_rmsnorm`; smoke-check it prints per-seed rel-L2 | AC-3 | coding | - |
| task2 | Measure v1 same-session fp32 control: full-5-seed per-seed rel-L2 (via probe) + p50 latency anchor; record as the floor and noise baseline | AC-3, AC-2 | coding | task1 |
| task3 | Implement `runs/matmul_add_rmsnorm_v2_bf16_split.py` on the v1 structure: resident `w_hi/w_lo` bf16 limbs (pinned `w_f→hi→residual→lo`), per-sub-tile `xT_hi/xT_lo` split on the transposed fp32 value, fixed 3-product `hi@hi+hi@lo+lo@hi` fp32-PSUM accumulation (drop `lo@lo`); residual add + RMSNorm + full-width output store byte-for-byte v1 | AC-1, AC-7 | coding | task2 |
| task4 | Verify v2: `--fast` seed-42 pre-check, then FULL 5-seed L2 gate; record numeric per-seed rel-L2 and per-seed delta vs the v1 floor; apply the ≥1.8e-5 danger band | AC-1, AC-4 | coding | task3 |
| task5 | Measure v2 latency FULL 5-seed ≥2× vs the v1 promoted row (0.9612 ms) and the same-session anchor; require >1.8% out-of-noise win, no p95/variance regression | AC-2 | coding | task4 |
| task6 | Capture v2 profiler digest (PE/MFU/Vec/Scl/DMA/HBMrd/HBMwr); validate HBM-unchanged vs v1 84/34 MB and that the 48-issue/N-chunk 3-product is not instruction-issue-bound (PSUM unchanged); if PE idle <~95% apply the bit-exact PSUM-reschedule tweak and re-measure | AC-6 | coding | task5 |
| task7 | Record all v2 evidence: `benchmark.csv` row, `candidates.jsonl` node (v2→v1) with per-seed rel-L2/latency and the audited split + product-summation order; `profile/` digest; promote iff both gates clear | AC-8 | coding | task6 |
| task8 | Decision point: if v2 fails the 2e-5 gate on any seed OR is marginal (worst ≥1.8e-5), implement `runs/matmul_add_rmsnorm_v2b_bf16_split4.py` (v2 + only the `lo@lo` product) and record it as a negative/rescue datum; else record the v2b-skip. If v2 fails/regresses on latency, record the fp32-floor negative datum and keep v1 promoted | AC-5 | coding | task7 |
| task9 | (Optional) Cross-check the v2 profiler digest against the sibling promoted split and the theoretical PE floor (`kernel-cost-analysis`) to separate op-specific overhead (`z` add, `g` broadcast, full-N norm) from expected 3-product split behavior; sanity-check the recomputed quadrature margin against the measured per-seed rel-L2 | AC-6, AC-4 | analyze | task6 |

## Claude-Codex Deliberation

### Agreements
- The compensated bf16x2 3-product split is the correct and only primary lever: v1 is PE-bound at
  the fp32 rate penalty, every other engine is hidden, and micro-rearranging fp32 work cannot beat
  the fp32 PE floor. Both siblings promoted the identical split.
- v2 as the single promotion candidate is right; v2b (4-product) should remain a correctness rescue
  only, never a latency/promotion candidate.
- Because v2's diff is localized to the matmul and the fp32 epilogue is byte-for-byte v1, v1 IS the
  same-session fp32 control — there is no refactor and thus no refactor-noise to isolate. (This op
  correctly skips the sibling's v2 enabler refactor: `g` is on the output axis and does not fold
  into `w`; `inv_rms` cannot commute to a PSUM-eviction post-scale because the norm reduces over N.)
- Necessary invariants: fp32 residual add before the norm, fp32 RMSNorm reduction over N, `eps`
  after the `/N` mean, `g` on the output free axis (never folded), raw-2D I/O + exact signature,
  full-width output store.
- Full-5-seed on-device promotion gate plus repeated timing is appropriate; `--fast` alone is
  insufficient.
- The SBUF footprint fits (limb `w` footprint equals v1's fp32 `w`; ~171 KB M-loop peak < ~192 KB).
- The offline sim's composite norm path (bf16 error feeding both `inv_rms` and the numerator) is the
  right model, and the partial self-cancellation is a genuine first-order effect; the measured
  4.454e-6 is the residual after it.

### Resolved Disagreements
- v2b trigger threshold (Codex first-pass CORE_RISK + REQUIRED_CHANGE): the draft's `>1.5e-5`
  trigger sits BELOW the predicted mean 1.526e-5, so it would fire even when v2 passes the 2e-5
  gate — forcing a known false repair. **Resolution:** AC-5 moves the trigger to a true danger
  band — build v2b IFF v2 fails the 2e-5 gate on any seed OR worst-seed rel-L2 ≥ 1.8e-5; at the
  predicted passing ~1.53e-5, v2b is NOT built and the skip is recorded.
- Borrowed fp32 floor (Codex CORE_RISK + MISSING_REQUIREMENT): the 1.46e-5 floor is borrowed from a
  sibling with K=1024; this op has K=2048 and reduces over N=2048, which can change rounding
  exposure. **Resolution:** AC-3 requires measuring v1's ACTUAL per-seed rel-L2 this session (v1 is
  the fp32 control) and recomputing the quadrature margin if the measured floor is materially
  higher; the hard 2e-5 per-seed gate remains the promotion gate regardless (Codex REQUIRED_CHANGE).
- Quadrature independence / thin margin (Codex CORE_RISK + TECHNICAL_GAP): quadrature assumes
  independent error sources, but both the fp32 floor and the bf16x2 error originate in matmul
  rounding and pass through the same RMSNorm. **Resolution:** AC-4 records numeric per-seed rel-L2
  and per-seed deltas, notes the pessimistic linear-sum bound (~1.905e-5), and applies the
  ≥1.8e-5 danger band so a margin-compressed seed is caught even under gate.
- Same-session control gap (Codex CORE_RISK #2): **Resolution:** AC-2/AC-3 require a same-session
  v1 rerun as both the latency noise anchor and the fp32 rel-L2 floor; v2 is compared against BOTH
  the historical 0.9612 ms row and the fresh anchor.
- Split-order / transpose exactness (Codex TECHNICAL_GAP): transpose-then-split differs from
  split-then-transpose only if the transpose is not exact fp32. **Resolution:** AC-1 pins the split
  on the TRANSPOSED fp32 value and notes v1 already performs this identity fp32 transpose, so any
  transpose rounding is already inside v1's measured fp32 floor.
- AC-2 ">=2×" wording (Codex round-1 REQUIRED_CHANGE): clarified to mean a run-count (≥2 full-5-seed
  measurements for stability), NOT a 2x speedup target; added as an AC-2 negative test.
- PSUM accumulation / product order (Codex round-1 REQUIRED_CHANGE): AC-1 and AC-8 pin the fixed
  product summation order (hi@hi, hi@lo, lo@hi) and require normal fp32 PSUM accumulation with no
  intermediate narrowing, recorded for auditability.
- Per-seed delta reporting + single-variable v2b (Codex round-1 OPTIONAL_IMPROVEMENTS): adopted into
  AC-4 (per-seed delta vs the same-session v1 floor) and AC-5 (v2b changes exactly one variable).

### Convergence Status
- Final Status: `converged` (Phase-3 Codex first-pass + one Phase-5 convergence round; all
  REQUIRED_CHANGES from both passes incorporated; no high-impact DISAGREE remains; two items —
  danger-band seed-expansion and the latency noise threshold — carried to Pending User Decisions).

## Pending User Decisions

- DEC-1: Danger-band seed expansion — if v2's worst-seed rel-L2 lands near the 1.8e-5 danger band
  (e.g. in `[1.7e-5, 1.9e-5]`), should the correctness evaluation expand beyond the 5 gate seeds
  `[0,21,42,63,84]` to more input draws before deciding promote-vs-rescue?
  - Claude Position: The 5 gate seeds ARE the NKIBench correctness gate, and the offline sim already
    tested 7 diverse draws (worst 4.454e-6), so the on-device gate + danger band is sufficient;
    expand seeds only if a gate seed actually lands in `[1.7e-5, 1.9e-5]`, treating it as a marginal
    trigger for both wider seeds and the v2b rescue.
  - Codex Position: Residual risk is empirical — device accumulation/order effects may not follow
    the offline quadrature model, and 1.8e-5 is close enough to the gate that seed expansion may be
    warranted if the worst seed lands near it.
  - Tradeoff Summary: Expanding seeds costs remote spend but hardens the correctness claim against a
    fixed-5-seed adapter caveat; not expanding saves spend and trusts the 7-draw offline gate plus
    the on-device 5-seed gate. Default (recommended): do not pre-expand; expand only on a
    near-danger-band worst seed. Low stakes — a diagnostic, not the hard gate.
  - Decision Status: PENDING

- DEC-2: Latency out-of-noise threshold — is `>1.8%` over v1 sufficient to promote v2, or should a
  larger practical margin be required because v1 is already PE-bound and remote p50 noise can be
  session-dependent?
  - Claude Position: `>1.8%` is sufficient given the siblings measured far larger wins (+23–28%) on
    the identical split and the expectation here is ~+25%; require the win to hold across ≥2
    full-5-seed runs against a same-session anchor so session noise is controlled.
  - Codex Position: `>1.8%` may be tight given session-dependent remote noise; consider a larger
    practical margin, or explicitly confirm `>1.8%` is acceptable.
  - Tradeoff Summary: A tighter band promotes real-but-modest wins but risks promoting on session
    noise; a wider band is safer but could reject a genuine win if the transfer underperforms the
    siblings. Given the expected ~+25% headroom, `>1.8%` with a repeated-run requirement is
    low-risk; revisit only if v2 lands unexpectedly close to v1. Default (recommended): `>1.8%`
    with the ≥2-run requirement.
  - Decision Status: PENDING

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `w_hi`, `w_lo`, `xT_hi`,
  `xT_lo`, `inv_rms`, `mean_eps`), matching the existing NKI kernel style in `runs/` and the
  sibling workspace.

--- Original Design Draft Start ---

# matmul_add_rmsnorm — Phase 2 draft (profile-driven optimization)

## 0. TL;DR

Phase-1 `matmul_add_rmsnorm_v1` is correct and **PE-bound at the fp32 systolic floor**
(PE=96%, MFU=46%, 0.9612 ms, 3.920x over the 3.768493 ms baseline; full-5-seed PASS).
Every other engine is hidden under that floor (Vec 15%, Scl 9%, DMA 20%; HBMrd 84 MB is
already the ~80 MB single-pass floor). On the bf16-native trn2 PE array a *correct* fp32
GEMM runs multiple internal bf16 passes, which is exactly why MFU is capped near ~46%.
**The only lever that cuts wall-clock is cutting PE time**, and the only correctness-viable
way to do that on this shape is the **compensated bf16x2 split-matmul** — the same win that
was PROMOTED on both siblings (`rmsnorm_matmul`: 1.066x→1.363x, +28%;
`add_rmsnorm_matmul`: 3.754x→4.632x, +23%).

The single new risk vs the siblings — the bf16 matmul error entering the norm path — is
already de-risked by a zero-remote-spend offline numpy sim (§3): worst bf16-only rel-L2 =
**4.454e-6** across 7 seeds, predicted device quadrature **1.526e-5** (~1.31x under the
2e-5 gate). SBUF fits (§5). Plan is **one** promotion candidate (v2, the bf16x2 split built
directly on v1) plus a costed accuracy-repair fallback (v2b, the 4-product split) if the
measured on-device rel-L2 crosses a marginal threshold.

## 1. Phase-1 baseline and the measured bottleneck

Promoted `runs/matmul_add_rmsnorm_v1.py` (0.9612 ms = 3.920x), full-5-seed PASS. Profiler
digest (from `benchmark.csv` / `candidates.jsonl`):

| engine | v1 | reading |
|---|---|---|
| **PE** | **96%** | saturated — the binding constraint |
| MFU | 46% | fp32 emulation rate penalty (bf16-native array runs fp32 as multi-pass) |
| Vec | 15% | RMSNorm reduce + output scale — hidden |
| Scl | 9% | square + rsqrt activations — hidden |
| DMA | 20% | HBMrd 84 MB (x 32 + w 16 + z 32 ≈ 80 MB one-pass floor), HBMwr 34 MB — hidden |

**Diagnosis: the kernel is PE-bound and the PE is running at the fp32 emulation rate.** Two
independent lines of evidence make this a *floor*, not a schedule artifact:
1. All non-PE engines are well under 50% and HBMrd is already at the single-pass floor
   (84 MB) — there is no memory-traffic or Vec/Scl headroom to reclaim; the earlier
   full-width-vs-chunked output-store A/B already confirmed the epilogue is not the binding
   engine (chunking it *raised* PE to 98% and regressed 6.4%).
2. Both siblings sat at the identical fp32 floor (PE 94–97%, MFU 44–46%) and both were only
   broken by dropping the matmul to bf16-class precision.

The theoretical picture (see `kernel-cost-analysis` for the cost model): a
single-pass bf16 matmul of this shape has a PE floor of roughly `M·N·(K/128)` dst-free
cycles ≈ the sibling's ~218 µs class scaled for K=2048; the fp32 emulation costs ~2.1x
that, which is what 0.9612 ms reflects. Cutting to a **3-product** bf16 split is ~3 passes
vs fp32's ~4 (trn2 emulates fp32 matmul in ~4 bf16 passes), i.e. a ~0.75x PE-time ratio →
the directional expectation is **~0.75 ms / ~5.0x**. Promotion depends only on a *measured*
out-of-noise win + full-5-seed PASS, not on hitting that number.

## 2. Why bf16x2 is the ONLY real lever (what is closed)

Enumerate and rank the directions; everything except the split is closed by the profile or
by sibling precedent:

| direction | expected benefit | verdict |
|---|---|---|
| **compensated bf16x2 split-matmul** | **cut PE time ~0.75x → ~5x** | **PRIMARY — the only lever that touches the binding engine** |
| off-PE transpose (dma_transpose / nc_transpose / load_transpose2d) | move the 512 identity transposes off PE | **CLOSED** on both siblings: dma_transpose is fp32/bf16 SBUF→SBUF infeasible (hwdge needs src.shape[0]==16, swdge needs HBM src), nc_transpose(vector) regressed +2%, load_transpose2d hidden. And the transposes are ALREADY hidden under the PE-bound matmul here (PE=96%), so there is no idle to reclaim. |
| output-store structure (full-width vs per-chunk) | Vec/Scl overlap | **CLOSED in phase-1 R1**: full-width store measured 6.4% faster; chunking the pure-SBUF epilogue only adds Vec/Scl+store ops. |
| M-blocking / loop reorder | fill/DMA overlap | **N/A**: w is fully resident (single HBM pass), so there is no weight-reload bubble to block against; DMA is 20% and hidden. |
| N_CHUNK sizing | PSUM bank utilization | **CLOSED**: 512 = one fp32 PSUM bank = maximal moving-free width; siblings confirmed optimal. |
| fp32 loop reorder / g-into-w fold | — | **N/A / no-op here** (see §4: g is free-axis, does NOT fold; inv_rms does NOT commute out). |

So Phase 2 is essentially one direction explored carefully, with a costed fallback — not a
scatter of micro-opts.

## 3. De-risking the bf16 split for THIS op (the mirror twist)

**This is the one place matmul_add_rmsnorm genuinely differs from the siblings**, and it must
be measured, not assumed. In the norm→GEMM siblings the bf16 error entered *only the matmul*;
`inv_rms` was computed from the *exact fp32 activation* and commuted out as a post-scale, so
the norm path was pristine. Here the op is **GEMM → add → norm**: the bf16 matmul error lands
in `y = x@w + z`, and `y` feeds **both** `inv_rms[m] = 1/sqrt(mean_n(y²)+eps)` **and** the
output numerator `y·g`. The error propagates through the norm — a path no sibling sim ever
exercised.

I built and ran an offline numpy sim (`runs/offline_bf16_split_sim.py`, zero remote spend;
evidence `profile/matmul_add_rmsnorm_offline_bf16_split_sim.txt`, candidates node
`matmul_add_rmsnorm_offline_bf16_split_sim`) that reproduces the EXACT scored input (seed 42,
draw order x→w→z→g, z=(M,N), g=(N,), eps=1e-5) and this composite epilogue:

| model | rel-L2 vs fp32 reference | meaning |
|---|---|---|
| fp32 control (exact matmul + norm+scale) | **0.000e+00** | seed/draw-order/dtype/eps/formula all match — model is exact |
| plain single-limb bf16 | 2.350e-3 | fails the gate 117x — confirms single-limb is out |
| **bf16x2 3-product** (drop lo@lo) | **4.452e-6** (seed 42); **4.454e-6 worst over 7 seeds** | ~4.5x under the 2e-5 gate; ~3.3x below the fp32 sibling's own on-device 1.46e-5 |
| bf16x2 4-product (keeps lo@lo) | 3.491e-6 | sizes the dropped cross term — negligible improvement |

**The composite norm path does NOT blow up the error.** In fact it partially self-cancels: a
coherent relative perturbation `δ` in `y` scales the numerator by ~`δ` and `inv_rms` by ~`−δ`,
so `out = y·inv_rms` is first-order insensitive to a common-mode scaling of `y`. The measured
4.454e-6 is the residual after that cancellation. The dropped `lo@lo` term is negligible
(3-product 4.454e-6 vs 4-product 3.491e-6), so **3-product is the right choice**.

**KEY calibration — QUADRATURE (learned from the sibling):** the offline number is the
*bf16-only* term. The on-device rel-L2 combines the hardware fp32 floor (present in v1 too —
trn2 emulates "fp32" with rounding the numpy sim can't see, ~1.46e-5 on the sibling) with the
bf16 error **in quadrature**:
`sqrt(1.46e-5² + 4.454e-6²) = 1.526e-5`. This is not a hypothesis — the sibling
`add_rmsnorm_matmul` predicted 1.526e-5 by the identical method and *measured 1.528e-5 on
device*. So the expected on-device rel-L2 here is **~1.53e-5, ~1.31x under the 2e-5 gate**:
comfortable, but the margin is real and thin enough to keep the 4-product repair costed (§6).

## 4. Two op-specific simplifications vs the sibling bf16-split kernel

The sibling `add_rmsnorm_matmul_v3` needed a **v2 enabler refactor** first (g-into-w' fold +
inv_rms post-scale eviction) because its g was per-K (contraction axis) and its inv_rms
commuted out. **Neither applies here**, so we skip the enabler and diff directly on v1:

1. **g is NOT folded into w.** g is length-N on the *output* free axis, applied *after* the
   norm (`out = y·g/rms`). Folding `g[n]` into `w[k,n]` would scale `y` *before* the norm and
   change `rms = sqrt(mean(y²))` — algebraically wrong. g stays exactly where v1 has it: a
   `[1,N]→[128,N]` broadcast multiply on the output. (Contrast the sibling, whose per-K g
   folded cleanly into the resident weight.)
2. **inv_rms does NOT commute out to a post-scale eviction.** The norm reduces over N, so the
   *entire* `[128,N]` row `y` must be assembled in SBUF before `inv_rms` is known — we cannot
   apply it at PSUM→SBUF eviction chunk-by-chunk the way the sibling did (its norm reduced
   over K, independent of the matmul output). v1's structure — assemble full y, then a single
   full-N reduce, then a full-width output scale — is already the correct shape and is *kept*.

So the v1→v2 diff is **localized to the matmul only**: split w into two bf16 limbs once at
load; split each transposed x sub-tile into two bf16 limbs; replace the single fp32
`nc_matmul` accumulation with the 3-product bf16 accumulation. The residual add, RMSNorm, and
output scale are byte-for-byte v1 (all fp32).

## 5. Candidate v2 design (compensated bf16x2 3-product split, built on v1)

`runs/matmul_add_rmsnorm_v2_bf16_split.py`. Same signature, raw-2D I/O, M-outer loop, and
fp32 epilogue as v1. Diffs:

**Pinned, auditable split order** (bf16(.) = `nl.copy(dtype=nl.bfloat16)`, round-to-nearest-even):
- Weight, once at load (replaces the fp32 `w_sb`): for each of 16 K-tiles,
  `w_f = load(w[kt])` (fp32) → `w_hi[kt] = bf16(w_f)` → `w_res = w_f − w_hi` (fp32, exact for
  O(1)) → `w_lo[kt] = bf16(w_res)`. Store `w_hi`, `w_lo` as resident `[16,128,N]` bf16 (32 KB
  each = 64 KB/part total, **identical bytes to v1's one fp32 w**).
- Activation, per M-tile per K-sub-tile (replaces the fp32 `xT`): transpose the RAW x
  sub-tile to `xT_f = [k_in, m_in]` via the exact fp32 identity `nc_matmul` (unchanged from
  v1) → `xT_hi[kt] = bf16(xT_f)` → `xT_res = xT_f − xT_hi` → `xT_lo[kt] = bf16(xT_res)`.
  Splitting after the transpose is identical to before it (transpose is exact, bf16 rounding
  is element-wise).

**Matmul (replaces v1's single fp32 accumulation):** per N-chunk `c` (4 chunks of 512), per
K-tile `kt` (16), accumulate three bf16 products into the fp32 PSUM bank:
```
acc += nc_matmul(xT_hi[kt], w_hi[kt, :, chunk])   # hi @ hi
acc += nc_matmul(xT_hi[kt], w_lo[kt, :, chunk])   # hi @ lo
acc += nc_matmul(xT_lo[kt], w_hi[kt, :, chunk])   # lo @ hi   (drop lo@lo)
```
Then, **exactly as v1**: `y[:, chunk] = acc + z_tile` (fp32 residual add before the norm),
and after all 4 chunks the fp32 RMSNorm (square → full-N reduce → `sumsq/N + eps` → rsqrt)
and the full-width output scale `out = y·inv_rms·g`.

**Loop-order note (de-risk aT-split cost):** v3 of the sibling split each `[128,128]` aT
sub-tile via an intermediate fp32 SBUF copy (`aT_f`) then two casts; the sibling's phase-3 D1
showed the compiler already elides the redundant copy and the sub-tile split work is hidden
under the PE-bound matmul, so I will write the straightforward per-sub-tile split and *check
the digest* rather than pre-optimizing it. If the digest shows PE idle (<~95%) I have the
sibling's bit-exact "split aT limbs directly from PSUM" reschedule (`aT_hi=bf16(psum_t)`,
`aT_lo=bf16(psum_t−aT_hi)`, dropping the fp32 `aT_f`) as a zero-precision-change tweak lever.

## 6. Accuracy-repair fallback v2b (4-product), costed, built only if triggered

Trigger (from the sibling's playbook): if v2's *measured on-device* rel-L2 exceeds **1.5e-5**
on any seed (the marginal threshold; expected ~1.53e-5 by §3 quadrature, so this may fire),
build `runs/matmul_add_rmsnorm_v2b_bf16_split4.py` = v2 + the fourth product
`acc += nc_matmul(xT_lo[kt], w_lo[kt, :, chunk])`. Offline this only moves rel-L2 4.454e-6 →
3.491e-6 (~22% of the bf16 term, ~1% of the quadrature) while adding a 4th matmul pass
(+~25% PE time). On the sibling the analogous v3b measured **+28% latency for a ~1.6% rel-L2
improvement → MEASURED-REJECT** (a false repair when the fp32 floor dominates). I expect the
same here and will keep v2b as a recorded negative datum unless v2 actually fails the gate
(in which case the 4-product is a genuine correctness necessity, not a false repair). This is
"build + measure, don't skip-by-model" per the sibling discipline.

## 7. SBUF budget (per partition, 128 partitions) — fits

| buffer | bytes/part | note |
|---|---|---|
| `w_hi` + `w_lo` (16 K-tiles, N=2048, bf16) | 64 + 64 = **128** KB | **same as v1's one fp32 w** (2 bf16 limbs = 1 fp32) |
| identity | <1 KB | |
| M-loop transients: x_sb (8) + xT_hi/lo (4+4) + aT split temps (~1) + y (8) + sq (8) + z_tile (2) + out_sb (8) | ~**43** KB | peak in the M-loop |
| **M-loop total** | **~171 KB** | < ~192 KB budget |
| preamble peak (w limbs + transient w_f + w_res) | ~145 KB | < budget |

Comfortable. If the compiler ever spills, reuse `sq` over `y`'s buffer (compute norm in place)
before touching residency — correctness is unaffected (residency is a perf choice).

## 8. Acceptance / how each candidate is judged

Score from `workspaces/matmul_add_rmsnorm/`:
```bash
python3 \
    ../../verify.py --op matmul_add_rmsnorm --candidate runs/<file>.py --fast   # seed-42 quick
# drop --fast for the full 5-seed gate before recording/promoting
```
Numeric per-seed rel-L2 (verify.py prints only the bool gate + latency): copy the sibling's
`runs/rel_l2_probe.py` into this workspace and run it to record the numeric rel-L2 (needed to
evaluate the §6 trigger).

- **v2 (bf16x2 3-product) — the intended promotion.** Promote iff: full-5-seed
  `l2_norm_passed = True` AND measured p50 is out-of-noise faster than a same-session v1
  control (record a fresh v1 run as the noise anchor; sibling jitter was <0.1%). Expect
  ~0.75 ms / ~5x and rel-L2 ~1.53e-5.
- **v2b (4-product) — fallback only.** Build+measure only if v2's measured rel-L2 crosses
  1.5e-5; keep as negative datum unless v2 fails the gate outright.
- **v1 retained as the pure-fp32 fallback** (DEC: keep a correct fp32 path if any bf16 seed
  is ever marginal).
- Never regress correctness; ≤5 iterations on the one direction.

## 9. Deliverables

- `runs/matmul_add_rmsnorm_v2_bf16_split.py` (promotion candidate), and `v2b_bf16_split4.py`
  only if triggered.
- `runs/offline_bf16_split_sim.py` (already written) + `runs/rel_l2_probe.py` (copy from
  sibling) as evidence helpers.
- Record each perf change in `benchmark.csv`; each candidate (with parent links) in
  `candidates.jsonl`; profiler digests + the offline-sim output under `profile/`.

See sibling evidence: `workspaces/add_rmsnorm_matmul/runs/add_rmsnorm_matmul_v3_bf16_split.py`
(the split-matmul template), `workspaces/add_rmsnorm_matmul/docs/plan-phase2.md`,
`workspaces/rmsnorm_matmul/runs/rmsnorm_matmul_v4_bf16_split.py`; memory
`kda-add-rmsnorm-matmul-progress`, `kda-rmsnorm-matmul-progress`, `kda-matmul-progress`.

--- Original Design Draft End ---
