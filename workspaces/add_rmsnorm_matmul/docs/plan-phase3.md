# add_rmsnorm_matmul — Phase 3 Plan: Chase the op-specific PE-idle gap with a bit-exact PSUM-source limb split

## Goal Description

The promoted `add_rmsnorm_matmul_v3_bf16_split` is **4.632x (0.4013 ms)**, full-5-seed
PASS, rel-L2 1.528e-5 (1.31x under the 2e-5 gate; MFU=41% PE=89% Vec=22% Scl=24% DMA=29%
HBMrd=42MB HBMwr=34MB). Phase 3 is regime/shape specialization. Two things are true at once:

1. **Every classic *shape* lever is closed identically to the sibling `rmsnorm_matmul`.**
   Same fixed M=4096, N=2048, K=1024, fp32 I/O. No edge tiles (all dims divide evenly),
   `nc_matmul` forces the k-on-partition layout, N_CHUNK=512=`psum_fmax` is maximal, `w'` is
   fully resident so M-blocking is vacuous, LNC2 is out of the single-core contract. This is
   documented once (mirroring the sibling's AC-6 closure).

2. **The one genuine phase-3 surface is op-specific: a PE-idle gap.** The promoted v3 sits at
   **PE=89% (≈44 µs idle of the 401 µs wall)**, whereas the sibling's *identical* bf16-split
   kernel reached **PE=96% (≈15 µs idle)**. The two have nearly identical **PE-active (~355 µs
   — the same 3-product bf16 matmul)**, so the ~29 µs of *extra* idle is this op's extra
   per-M-tile non-PE work not fully overlapping the matmul: the residual add `a = x+z` (a
   [128,1024] Vec op the sibling lacks), the extra `z` read (+16 MB HBM), and the **granular
   per-sub-tile activation limb-split** — 8 sub-tiles × 4 Vec/Scalar ops = **32 small
   [128,128] ops per M-tile**, each paying the Vector engine's fixed ~268 ns `semaphore_start`
   + 161 ns `write_drain` on only ~133 ns of compute.

The **primary lever (D1)** is a *bit-exact* simplification of that limb-split: split the `aT`
limbs **directly from the transpose PSUM** (`psum_t`), dropping v3's intermediate fp32 SBUF
copy `aT_f`. This cuts the per-M-tile activation split from **32 → 24 Vec/Scalar ops** with
**zero intended numeric change** (rel-L2 target reproduces 1.528e-5) and **zero extra PE
work**. If the exposed non-PE time is real, this recovers part of the 44→15 µs gap
(directionally ~4.7–5.0x); if it is already hidden (the sibling's phase-3 measurements suggest
transpose/fills were hidden *at PE=96%*), the result is a within-noise floor-confirmation and
v3 stays promoted. **Both outcomes are acceptable phase-3 exits.**

Because the "bit-exact" premise rests on trn2 compiler-lowering details — direct
`nl.copy(psum_t, dtype=bf16)` and `tensor_tensor(psum_t[fp32], aT_hi[bf16], subtract)` with a
PSUM-space first operand — those primitives are treated as **compile-gated and
semantically-probed** before any full-5-seed/profiling spend, never as assumptions. **No
precision change is on the table:** the rel-L2 margin is only 1.31x, the 4-product variant was
already MEASURED-REJECT (+28% latency for a 1.6% accuracy move hidden under the fp32 hardware
floor), and fewer products / plain bf16 fail the gate. Phase 3 optimizes the *schedule around
the fixed 3-product arithmetic*, not the arithmetic.

This plan implements ONLY `add_rmsnorm_matmul`; it does not touch the benchmark definition
(`../../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}`) or any other operator.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. Scoring command (run from inside `workspaces/add_rmsnorm_matmul/`):
`python3 ../../verify.py
--op add_rmsnorm_matmul --candidate runs/<file>.py` (add `--fast` for the seed-42 quick check;
drop it for the full 5-seed gate). "rel-L2" everywhere means the verifier-reported relative-L2
field, not rounded text copied through another layer.

- AC-1: **v4 (`runs/add_rmsnorm_matmul_v4_psum_split.py`) is a bit-exact D1 fork of v3 that
  sources the `aT` limbs directly from the transpose PSUM.** Forked from
  `runs/add_rmsnorm_matmul_v3_bf16_split.py`, it removes the intermediate fp32 SBUF copy `aT_f`
  and sources `aT_hi = bf16(psum_t)` and `aT_res = psum_t − aT_hi` (fp32 `tensor_tensor`)
  directly from the transpose PSUM bank, then `aT_lo = bf16(aT_res)`. This is **4 → 3 ops per
  sub-tile (32 → 24 per M-tile)**. Everything else is byte-for-byte v3: the 3 bf16 products
  (`aT_hi@w'_hi + aT_hi@w'_lo + aT_lo@w'_hi`, `aT_lo@w'_lo` dropped), the PINNED split order,
  `g` folded into resident `w' = g·w` in fp32 before its bf16 split, `inv_rms` post-scale at
  PSUM→SBUF eviction, fully-fp32 RMSNorm (`square`/`reduce`/`mean_eps`/`rsqrt`), `eps` added
  AFTER the `/K` mean, and the raw-2D I/O with the exact signature
  `kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor)`.
  - Positive Tests (expected to PASS):
    - The v4 file exists in `runs/`, and a diff against v3 shows ONLY the removal of the `aT_f`
      copy and the re-pointing of the `aT_hi`/`aT_res` reads to `psum_t`.
    - The per-M-tile transpose+split loop issues 3 Vec/Scalar ops per sub-tile, not 4.
  - Negative Tests (expected to FAIL):
    - Any change to the arithmetic: product count, split order, `g`-fold placement (must stay
      on the k_in/partition axis, folded into `w'` before the split — a free-axis broadcast
      changes the math), `inv_rms` placement, RMSNorm dtype, or the `eps`-after-mean form.
    - Retaining the `aT_f` fp32 copy (D1 not actually applied) but relabelling the file.

- AC-2: **Compile + semantic gate (runs BEFORE any full-5-seed or profiling spend).** v4
  compiles with no PSUM bank-allocation or spill error, and a single `--fast` (seed-42) run
  reproduces v3's per-seed rel-L2. This is the D1 semantic probe: direct
  `nl.copy(psum_t, dtype=bf16)` and `tensor_tensor(psum_t[fp32], aT_hi[bf16], subtract)` (a
  PSUM-space first operand) are **compile-gated primitives**, not assumptions. Reading one
  operand from PSUM is proven in this kernel (the `inv_rms` post-scale reads the PSUM
  accumulator); the second operand of `tensor_tensor` remains SBUF, so the known
  both-operands-in-PSUM restriction is not triggered.
  - Positive Tests (expected to PASS):
    - v4 compiles cleanly (no PSUM-bank/spill/allocation error).
    - `--fast` `l2_norm_passed = True` AND the verifier-reported rel-L2 matches v3's seed-42
      value (1.528161e-5) to printed precision.
  - Negative Tests (expected to FAIL / halt):
    - A compile, PSUM-bank, or spill failure → **D1 is codegen-infeasible**; record it CLOSED
      as a first-class negative datum (like the sibling's D4), and v3 stays promoted. Do NOT
      spend full-5-seed/profiling runs on an uncompilable v4.
    - rel-L2 drift beyond print noise on `--fast` → **INVESTIGATE**; do NOT proceed to
      promotion. A passing-but-drifted v4 is a bug to explain, not a precision tradeoff.

- AC-3: **v4 is bit-exact-correct on the full gate.** After AC-2 passes, a FULL 5-seed run has
  `l2_norm_passed = True` at seeds `[0,21,42,63,84]`, and each per-seed rel-L2 equals v3's
  1.528161e-5 within a small tolerance `|Δrel-L2| ≤ 1e-7` (the bit-exact expectation, since v4
  consumes the same bf16 limbs as v3). Any drift beyond that tolerance is a **bug to
  investigate**, not a precision tradeoff, and blocks promotion until explained.
  - Positive Tests (expected to PASS):
    - Full-5-seed `l2_norm_passed = True`, per-seed rel-L2 == 1.528161e-5 within 1e-7.
  - Negative Tests (expected to FAIL):
    - Any seed with `l2_norm_passed = False`.
    - Per-seed rel-L2 moving away from 1.528161e-5 by more than 1e-7 while still passing —
      surfaces a hidden lowering difference; must be investigated, not auto-promoted.

- AC-4: **v4 profiler digest captured; HBM/spill unchanged.** Record PE%, MFU%, Vec%, Scl%,
  DMA%, HBMrd, HBMwr for v4 under `profile/`, alongside the exact profiler command/config and
  the compiler/runtime version, and compare against v3 (0.4013 ms, PE=89%, 42MB/34MB) and the
  sibling. Because limbs are built on-chip, HBM must be materially unchanged.
  - Positive Tests (expected to PASS):
    - A digest with all seven metrics is written under `profile/`, plus the profiler
      command/config and version string.
    - HBMrd/HBMwr within a small margin of v3's 42 MB / 34 MB; no new SBUF/PSUM spill.
    - If v4 wins, the win correlates with reduced Vec/Scl% or reduced PE-idle **without** an
      increase in DMA/HBM or a drop in PE-active efficiency.
  - Negative Tests (expected to FAIL):
    - A promotion recorded with no profiler digest.
    - A material HBMrd/DMA rise or a new spill left unflagged in the evidence.

- AC-5: **Promotion is gated on measured correctness AND an out-of-noise latency win.** Promote
  v4 IFF full-5-seed `l2_norm_passed = True` (AC-3) AND its p50 beats a same-session v3 anchor
  by more than the noise band (**>1.8%**), measured over **≥2 full-5-seed v4 runs** bracketed
  by v3 anchors (a v3 anchor before and after the v4 runs) with no p90/variance regression.
  Otherwise, record the within-noise **floor-confirmation** and v3 stays promoted. The
  directional expectation is ~0.38–0.385 ms (~4.8–5.0x) but the exact figure is **not** an
  acceptance requirement.
  - Positive Tests (expected to PASS):
    - ≥2 full-5-seed v4 measurements both beat the same-session v3 anchor by >1.8%, with stable
      p90/variance → promote v4.
    - v4 lands within ±1.8% of the v3 anchor → record floor-confirmation, keep v3 promoted
      (also an acceptable exit).
  - Negative Tests (expected to FAIL):
    - A `--fast`-only measurement used as the promotion basis.
    - A within-noise or regressed latency promoted anyway.
    - Promotion declared without a same-session v3 anchor bracketing the v4 runs.

- AC-6: **Shape-specialization closure is documented (AC-6 mirror of the sibling).** A short
  `docs/shape-specialization-closure-phase3.md` (or an explicit by-reference note to the
  sibling `rmsnorm_matmul`'s closure) records that every classic phase-3 shape lever is vacuous
  or pinned: no edge/partial tiles (M=32·128, K=8·128, N=4·512 divide evenly), layout forced by
  `nc_matmul` (k on partition, `[m_in,n]` output), N_CHUNK=512=`psum_fmax` already maximal,
  M-blocking vacuous (`w'` fully resident, each `x`/`z` read once), LNC2 out of the single-core
  contract.
  - Positive Tests (expected to PASS):
    - The closure doc (or by-reference note) exists and covers all five levers.
  - Negative Tests (expected to FAIL):
    - A phase-3 shape lever claimed "open" without evidence, or the closure left undocumented.

- AC-7: **Closed directions are recorded as first-class negative data, not silently dropped.**
  `benchmark.csv`/`candidates.jsonl` carry record-only entries for: D3 split-before-transpose
  (doubles transpose PE work — 16 transpose `nc_matmul`s + 16 PSUM→SBUF copies per M-tile vs
  v3's 8 — adding the one thing we cannot afford; **do not implement**); D4 off-PE transpose
  (re-checked for the bf16-limb world: `dma_transpose` of a [128,128] tile is still infeasible —
  HW-DGE needs `src.shape[0]==16`, SW-DGE needs the source in HBM; `nc_transpose(bf16)` lands in
  fp32 PSUM needing a re-cast, and the sibling measured a vector-engine transpose as +2%; **do
  not explore**); D5 precision/product-count changes (**FORBIDDEN this phase** — 3-product bf16
  pinned, margin 1.31x, D3 4-product MEASURED-REJECT, plain bf16 fails ~117x).
  - Positive Tests (expected to PASS):
    - D3/D4/D5 each recorded with the closure rationale.
  - Negative Tests (expected to FAIL):
    - Implementing D3, D4, or D5, or dropping any of them without a recorded rationale.

- AC-8: **Correctness invariants are never regressed.** fp32 RMSNorm reduction
  (`square`/`reduce`/`mean_eps`/`rsqrt`), `eps` added AFTER the `/K` mean; `g` folded into
  `w' = g·w` on the k_in/partition axis in fp32 before the bf16 split; `inv_rms` applied
  post-scale at eviction (both exact commutations — offline fp32 control 4.82e-7); the 3-product
  bf16 split and its PINNED split order unchanged; the raw-2D I/O and exact signature preserved.
  Every promotion gated on **full 5-seed** `l2_norm_passed`, not `--fast` alone.
  - Positive Tests (expected to PASS):
    - v4 reproduces v3's rel-L2 1.528e-5 and preserves all invariants above.
  - Negative Tests (expected to FAIL):
    - `eps` added before/scaled by `1/K` (`rsqrt((sumsq+eps)/K)`); `g` broadcast on the N/free
      axis; RMSNorm reduction moved out of fp32; `inv_rms` folded pre-matmul incorrectly.
  - AC-8.1: **Carried input-diversity caveat (record-only, non-blocking).** The adapter fixes
    `np.random.seed(42)` for all 5 profiler seeds, so on-device 5-seed PASS is weak on *input*
    diversity; the offline 7-draw sim mitigates. v1 (3.754x) and v2 (3.898x) remain pure-fp32
    fallbacks. Fixing the adapter is OUT of scope (it is the NKIBench benchmark contract);
    promotion evidence is stated as being against the current contract.
    - Positive: the caveat is restated in the v4 evidence.
    - Negative: claiming genuine per-seed input diversity from the on-device 5-seed run.

- AC-9: **D2 is contingent, not proactive, and must be numerically neutral.** Pursue D2
  (reduce residual-add / RMSNorm-reduction exposure) as `v5` ONLY if D1's profiler digest still
  shows a measurable Vec/Scalar bubble. Any D2 candidate must be numerically neutral and gated
  exactly like v4 (AC-2/AC-3/AC-5). The "per-sub-tile add fold" is explicitly flagged: it is
  acceptable ONLY if it does not change the full-K fp32 RMSNorm reduction (a partial/streaming
  reduction changes fp32 summation order and is NOT numerically neutral).
  - Positive Tests (expected to PASS):
    - v5 is attempted only when D1 leaves a measured bubble, and it reproduces rel-L2 1.528e-5.
  - Negative Tests (expected to FAIL):
    - Building v5 proactively with no measured bubble.
    - A D2 variant that reorders the full-K fp32 reduction (changes the reduction math).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. This phase
is highly deterministic: the arithmetic is pinned and the single primary lever is fixed by the
draft, so the bounds are narrow.

### Upper Bound (Maximum Acceptable Scope)
Implement and measure D1 as `v4` (PSUM-source limb split), gated through the compile+semantic
probe (AC-2), the full-5-seed bit-exact gate (AC-3), and the bracketed same-session v3-anchored
latency comparison (AC-4/AC-5). If — and only if — D1's profile still shows a measurable
Vec/Scalar bubble, additionally implement and measure one numerically-neutral D2 variant as
`v5`. Document the shape-specialization closure (AC-6) and record D3/D4/D5 as first-class closed
directions (AC-7). Promote whichever of v3/v4(/v5) is fastest under the full correctness gate,
and report the speedup vs the 1.859287 ms baseline.

### Lower Bound (Minimum Acceptable Scope)
Implement D1 as `v4`, run the AC-2 compile+semantic probe, and record the outcome. If v4 is
codegen-infeasible (AC-2 negative), record it CLOSED and keep v3 promoted. If v4 compiles and
passes but lands within noise of v3, record the floor-confirmation and keep v3 promoted. Either
way, produce the AC-6 shape-closure doc/reference and the AC-7 closed-direction records. A
within-noise or codegen-infeasible result is a **complete, acceptable** phase-3 exit — a
successful floor confirmation, not a failure.

### Allowed Choices
- Can use: the existing bf16x2 3-product split verbatim; direct PSUM-source reads for the limb
  split (`nl.copy(psum_t, dtype=bf16)`, `tensor_tensor(psum_t, aT_hi, subtract)`) if they
  compile; the same identity-matmul transpose; the same `affine_range` M-loop pipelining; one
  contingent numerically-neutral D2 schedule tweak.
- Cannot use: any change to the 3-product bf16 arithmetic, product count, split order, dtypes,
  or `g`/`inv_rms`/`eps` placement (D5, FORBIDDEN); split-before-transpose (D3, doubles
  transpose PE); off-PE / `dma_transpose` / `nc_transpose` transpose replacements (D4,
  infeasible); any reduction-order-changing D2 variant; any edit to the benchmark definition or
  another operator; a `--fast`-only or within-noise promotion.

> **Note on Deterministic Design**: The draft pins the arithmetic and fixes D1 as the single
> primary lever, so the upper and lower bounds nearly converge — the only elective scope is the
> contingent D2, which is itself gated on a measurement. "Allowed Choices" reflects this narrow
> constraint: the schedule may change (bit-exactly), the arithmetic may not.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach
D1 removes v3's intermediate fp32 SBUF copy inside the per-sub-tile transpose+split loop. In v3
(`runs/add_rmsnorm_matmul_v3_bf16_split.py`, the `for kt in nl.affine_range(K_TILES)` block
around the transpose):

```
psum_t = nc_matmul(a_sub, identity, is_transpose=True)   # fp32 PSUM
aT_f   = copy(psum_t)                # fp32 PSUM->SBUF        <-- REMOVE
aT_hi  = bf16(aT_f)                                          # v3
aT_res = aT_f - aT_hi                # fp32 tensor_tensor     # v3
aT_lo  = bf16(aT_res)                                        # v3
```

Because `aT_f` is an *exact* fp32 copy of `psum_t`, every downstream read can source `psum_t`
directly, which is intended to be bit-identical:

```
psum_t = nc_matmul(a_sub, identity, is_transpose=True)   # fp32 PSUM (unchanged)
aT_hi  = bf16(psum_t)                # bf16(psum_t)   == bf16(aT_f)   (intended bit-for-bit)
aT_res = psum_t - aT_hi              # (psum_t-aT_hi) == (aT_f-aT_hi) (intended bit-for-bit)
aT_lo  = bf16(aT_res)
```

Net: 4 → 3 ops per sub-tile, 32 → 24 per M-tile, 256 fewer [128,128] Vec/Scalar ops overall,
each fixed-overhead-dominated; zero extra PE. `psum_t` stays live one op longer (until `aT_res`
is computed) — the transpose uses a [128,128] bank and there are 8 PSUM banks of 2048, so there
is no PSUM-pressure conflict with the [128,512] main accumulator, which is allocated in a later
loop. **The bit-identity is the *intent*, not a guarantee** — it depends on trn2 lowering
canonicalizing the PSUM read to the same fp32 value before the bf16 round. That is exactly why
AC-2 makes the `--fast` rel-L2 match a hard gate before any further spend.

### Relevant References
- `runs/add_rmsnorm_matmul_v3_bf16_split.py` — the PROMOTED base; fork v4 from it. The `aT_f`
  copy and the two downstream reads are the only lines D1 touches; the `inv_rms` post-scale
  already reads a PSUM accumulator directly (the proven precedent for one PSUM operand).
- `runs/add_rmsnorm_matmul_v3b_bf16_split4.py` — the D3 4-product MEASURED-REJECT datum (why D5
  is forbidden).
- `runs/offline_bf16_split_sim.py` — the offline pre-check (worst idealized rel-L2 4.451e-6;
  7-draw input diversity that mitigates the fixed-seed-42 caveat).
- `workspaces/rmsnorm_matmul/docs/shape-specialization-closure-phase3.md` — the sibling's AC-6
  closure to mirror or reference.
- `workspaces/rmsnorm_matmul/docs/phase3-exit-decision.md` — the sibling's phase-3 write-up
  showing transpose/stationary-fills were already hidden at PE=96% (the pessimistic-case
  precedent for D1).
- `../../verify.py` — the correctness/latency harness; `--fast` = seed-42 only, drop for the
  full 5-seed gate.

## Dependencies and Sequence

### Milestones
1. **v4 (D1, aT-split-from-PSUM) built and compile+semantic-gated.**
   - Phase A: fork `runs/add_rmsnorm_matmul_v4_psum_split.py` from v3; drop the `aT_f` copy;
     source `aT_hi`/`aT_res` from `psum_t` (AC-1).
   - Phase B: compile + `--fast` semantic probe; confirm rel-L2 matches 1.528161e-5 (AC-2). On
     compile/spill failure → record D1 CLOSED, keep v3, jump to Milestone 4.
2. **v4 correctness + latency measurement.**
   - Phase A: full-5-seed run; confirm per-seed rel-L2 == v3 within 1e-7 (AC-3).
   - Phase B: ≥2 full-5-seed latency runs bracketed by same-session v3 anchors; capture the
     profiler digest + command/config + version (AC-4). Apply the promotion gate (AC-5).
3. **v5 (D2) — contingent.** Only if Milestone 2's digest shows a measurable Vec/Scalar bubble:
   build one numerically-neutral D2 variant and gate it identically (AC-9). Skip otherwise.
4. **Close-out and evidence.** Document the shape-specialization closure (AC-6); record D3/D4/D5
   as closed negative data (AC-7); write `benchmark.csv` rows and `candidates.jsonl` DAG nodes
   (v4→v3, and v5→v4 if run) with metrics, `rel_l2`, `per_seed_rel_l2`, the bit-exactness note,
   and the `per_seed_latency_ms=null` / `latency_scope` caveat carried from v2/v3. Report the
   final speedup of the fastest of v3/v4(/v5) vs the 1.859287 ms baseline on the full gate.

Milestone 2 depends on Milestone 1 Phase B passing (AC-2). Milestone 3 depends on Milestone 2's
digest. Milestone 4 depends on the promotion decision from Milestone 2 (and 3 if run).

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Fork `runs/add_rmsnorm_matmul_v4_psum_split.py` from v3; remove the `aT_f` fp32 SBUF copy; source `aT_hi = bf16(psum_t)` and `aT_res = psum_t − aT_hi` directly from the transpose PSUM; keep all arithmetic byte-for-byte v3 | AC-1, AC-8 | coding | - |
| task2 | Compile v4 and run the `--fast` (seed-42) semantic probe; confirm no PSUM-bank/spill error and rel-L2 == 1.528161e-5; on failure record D1 CLOSED and stop before further spend | AC-2 | coding | task1 |
| task3 | Run v4 full-5-seed; confirm `l2_norm_passed` and per-seed rel-L2 == v3 within 1e-7; investigate any drift | AC-3, AC-8 | coding | task2 |
| task4 | Run ≥2 full-5-seed v4 latency runs bracketed by same-session v3 anchors; capture profiler digest + exact profiler command/config + compiler/runtime version; verify HBM/spill unchanged | AC-4 | coding | task3 |
| task5 | Apply the promotion gate: promote v4 iff full-5-seed PASS AND p50 beats the v3 anchor >1.8% with stable p90; else record floor-confirmation and keep v3 | AC-5 | coding | task4 |
| task6 | If (and only if) task4's digest shows a measurable Vec/Scalar bubble, design one numerically-neutral D2 schedule tweak (analysis of whether it changes the fp32 reduction order) | AC-9 | analyze | task4 |
| task7 | If task6 authorizes it, implement `runs/add_rmsnorm_matmul_v5_*.py` and gate it identically to v4 (compile+semantic, full-5-seed bit-exact, bracketed latency) | AC-9 | coding | task6 |
| task8 | Write `docs/shape-specialization-closure-phase3.md` (or a by-reference note to the sibling), covering all five shape levers | AC-6 | coding | - |
| task9 | Record D3/D4/D5 as first-class closed negative data in `benchmark.csv`/`candidates.jsonl` with closure rationales | AC-7 | coding | - |
| task10 | Write the v4 (and v5 if run) `benchmark.csv` rows and `candidates.jsonl` DAG nodes with metrics, per-seed rel-L2, bit-exactness note, and the carried `per_seed_latency_ms`/`latency_scope` caveat; report final speedup vs 1.859287 ms baseline | AC-4, AC-5, AC-8.1 | coding | task5 |

## Claude-Codex Deliberation

### Agreements
- D1 is the correct, tightly-scoped primary lever: it targets the stated op-specific PE-idle
  gap, preserves the pinned 3-product bf16 arithmetic, and cuts 32→24 Vec/Scalar ops per M-tile
  with zero extra PE work.
- The PSUM-direct limb split must be treated as a **compile-gated / codegen-gated probe**, not
  an assumption — direct `nl.copy(psum_t, dtype=bf16)` and PSUM-source `tensor_tensor` depend on
  trn2 lowering. Reading ONE operand from PSUM is valid by precedent (the `inv_rms` post-scale);
  both-operands-in-PSUM is the known-forbidden case and D1 does not trigger it (the second
  `tensor_tensor` operand stays in SBUF).
- Precision is pinned: no product-count/precision change this phase (D5 forbidden); D3 4-product
  is already a MEASURED-REJECT; the rel-L2 "excess" over the offline 4.45e-6 is the pre-existing
  fp32 hardware floor, not a v3 defect.
- Promotion must require a same-session v3 anchor, ≥2 full-5-seed runs, an out-of-noise (>1.8%)
  p50 win with stable p90, and a profiler digest correlating the win with reduced Vec/Scl% or
  PE-idle without an HBM/DMA rise. A `--fast`-only or within-noise promotion is rejected.
- A within-noise or codegen-infeasible D1 is an acceptable floor-confirmation exit; v3 stays
  promoted. The shape-lever closure and the D3/D4/D5 closed directions are recorded as
  first-class results.
- D2 is contingent on a measured bubble and must be numerically neutral; the "per-sub-tile add
  fold" is NOT numerically neutral if it changes the full-K fp32 reduction order.
- The fixed-seed-42 adapter caveat is real but out of scope (it is the benchmark contract);
  evidence is stated as against the current contract, mitigated by the offline 7-draw sim, with
  v1/v2 retained as pure-fp32 fallbacks.

### Resolved Disagreements
- **"Bit-exact" as a hard equality vs. a gated expectation** (Codex first-pass CORE_RISK): Codex
  warned that exact printed equality of rel-L2 is brittle and that direct PSUM sourcing could be
  numerically identical in intent yet not byte-identical due to lowering. Resolution: keep
  "bit-exact" as the *intent/expectation*, but (a) gate the `--fast` rel-L2 match in AC-2 before
  any spend, (b) express the AC-3 full-gate criterion as `|Δrel-L2| ≤ 1e-7` rather than exact
  string equality, and (c) block promotion and require investigation on any drift. Both sides
  agreed this is the right handling.
- **Whether a separate remote D0 microprobe is needed** (Codex first-pass ALTERNATIVE): folded
  the semantic probe into AC-2's `--fast` compile+rel-L2 gate rather than spending a separate
  remote run, keeping the remote budget bounded. Codex accepted.
- **PSUM liveness / bank-pressure casualness** (Codex first-pass TECHNICAL_GAP): the draft's bank
  argument was "too casual." Resolution: AC-2 and AC-4 explicitly gate on no PSUM-bank/spill
  error and no HBM/spill regression, turning the liveness claim into a measured check rather
  than an assertion.
- **D2 per-sub-tile add fold numerical neutrality** (Codex first-pass CORE_RISK): flagged as NOT
  obviously neutral because RMSNorm needs the full-K reduction before `inv_rms`. Resolution:
  AC-9 explicitly forbids any D2 variant that changes the full-K fp32 reduction order.

### Convergence Status
- Final Status: `converged` (Codex second-pass review returned no REQUIRED_CHANGES and no
  high-impact DISAGREE, explicitly stating the plan can be marked converged; three non-blocking
  OPTIONAL_IMPROVEMENTS — bracket v4 with two v3 anchors, define "rel-L2" as the verifier field,
  include the exact profiler command in the evidence — were folded into AC-2/AC-4/AC-5).

## Pending User Decisions

None. All four of Codex's first-pass `QUESTIONS_FOR_USER` were resolved during convergence and
the second Codex pass raised no unresolved items:
- Non-bit-identical-but-passing v4 → NOT auto-promotable; investigate (AC-3).
- One extra remote run for a D0 probe → folded into AC-2's `--fast` gate; no separate run.
- Fixed-seed adapter fix → out of scope (benchmark contract); evidence stated against the
  current contract (AC-8.1).
- Remote-spend budget → bounded: v4 = `--fast` + ≥2 full-5-seed + bracketing v3 anchors; v5
  contingent on a measured bubble.

Quantitative-metric classification (confirmed from the draft, no ambiguity remained): the 2e-5
rel-L2 gate and the >1.8% latency noise band are **hard gates**; the ~4.8–5.0x expected speedup
is an **optimization direction**, explicitly "uncertain by design — must be measured," and a
within-noise result is an acceptable exit.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Phase", "Step", "task1", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. the kernel file, its
  docstring, and its inline comments should read as a self-contained NKI kernel — the v3
  docstring style is a good model).

### Evidence and Repository Conventions
- Run the KDA loop from inside `workspaces/add_rmsnorm_matmul/`; evidence paths are
  workspace-relative.
- Candidate `.py` sources under `runs/` are tracked; other `runs/` artifacts and all of
  `profile/` are git-ignored. Never edit the benchmark definition or hand-tune a baseline.
- Every promotion is gated on the FULL 5-seed `l2_norm_passed`, never `--fast` alone.

--- Original Design Draft Start ---

# add_rmsnorm_matmul — Phase 3 draft (regime / shape specialization)

## 0. TL;DR

The promoted `add_rmsnorm_matmul_v3_bf16_split` is **4.632x (0.4013 ms)**, full-5-seed
PASS, rel-L2 1.528e-5 (1.31x under the 2e-5 gate). Phase 3 is regime/shape
specialization. Two things are true here at once:

1. **Every classic *shape* lever is closed identically to the sibling** `rmsnorm_matmul`
   (same fixed M=4096, N=2048, K=1024, fp32 I/O). No edge tiles (all dims divide
   evenly), `nc_matmul` forces the k-on-partition layout, N_CHUNK=512=`psum_fmax` is
   maximal, `w'` is fully resident so M-blocking is vacuous, LNC2 is out of the
   single-core contract. This is documented once, by reference, in
   `docs/shape-specialization-closure-phase3.md` (mirrors the sibling's AC-6 closure).

2. **The one genuine phase-3 surface is op-specific: a PE-idle gap.** The promoted v3
   sits at **PE=89% (≈44 µs idle of the 401 µs wall)**, whereas the sibling's
   *identical* bf16-split kernel reached **PE=96% (≈15 µs idle)**. The two have nearly
   identical **PE-active (~355 µs — it is the same 3-product bf16 matmul)**, so the
   ~29 µs of *extra* idle is this op's extra per-M-tile non-PE work not fully
   overlapping the matmul:
   - residual add `a = x+z` (a [128,1024] Vec op the sibling does not have),
   - the extra `z` read (+16 MB HBM → DMA 24→29% vs sibling's 25 MB),
   - and the **granular per-sub-tile activation limb-split** — 8 sub-tiles × 4
     Vec/Scalar ops = **32 small [128,128] ops per M-tile**, each paying the Vector
     engine's fixed ~268 ns `semaphore_start` + 161 ns `write_drain` on only ~133 ns
     of compute.

The **primary lever** is a *bit-exact* simplification of that limb-split (D1): split
the `aT` limbs **directly from the transpose PSUM**, dropping v3's intermediate fp32
`aT_f` SBUF copy. This cuts the per-M-tile split from **32 → 24 Vec/Scalar ops** with
**zero numeric change** (rel-L2 stays 1.528e-5 exactly) and **zero extra PE work**. If
the exposed non-PE time is real, this recovers part of the 44→15 µs gap
(≈**4.7–5.0x**); if it is already hidden (the sibling's phase-3 measurements suggest
transpose/fills were hidden *at PE=96%*), the result is a within-noise
floor-confirmation and v3 stays promoted. Either way it is a safe, well-scoped move.

**No precision change is on the table.** The rel-L2 margin is only 1.31x, D3
(4-product) was already MEASURED-REJECT (+28% for a 1.6% accuracy move hidden under the
fp32 hardware floor), and going the other way (fewer products / plain bf16) fails the
gate. So phase 3 optimizes the *schedule around the fixed 3-product arithmetic*, not the
arithmetic.

---

## 1. Starting point — the promoted kernel and its profile

`runs/add_rmsnorm_matmul_v3_bf16_split.py` (PROMOTED, 4.632x, full-5-seed PASS
rel-L2 1.528e-5). Per M-tile (32 tiles):

1. load `x`,`z` → `a = x+z` ([128,1024] fp32 `tensor_tensor`),
2. fused fp32 RMSNorm reduction (`square` → full-1024 `tensor_reduce` → two-op
   `mean_eps = sumsq·(1/K)+eps` → `rsqrt` → `inv_rms[128,1]`),
3. **transpose + limb-split of the 8 RAW-`a` K-sub-tiles**: per sub-tile — identity
   `nc_matmul(is_transpose)` → `psum_t` fp32; `aT_f = copy(psum_t)` fp32 (SBUF);
   `aT_hi = bf16(aT_f)`; `aT_res = aT_f − aT_hi` fp32; `aT_lo = bf16(aT_res)`,
4. main matmul: **3 bf16 products** (`aT_hi@w'_hi + aT_hi@w'_lo + aT_lo@w'_hi`) × 8
   K-tiles, accumulated in a [128,512] fp32 PSUM, over 4 N-chunks,
5. `inv_rms` post-scale at PSUM→SBUF eviction (`tensor_scalar` reading the accumulator)
   → store.

`w'_hi`/`w'_lo` (g folded in, split once) are fully resident bf16 (64 KB/part). HBM is
unchanged from v1 (42 MB read / 34 MB write) — limbs are built on-chip.

**Profiler digest (promoted v3):** MFU=41% PE=89% Vec=22% Scl=24% DMA=29%
HBMrd=42MB HBMwr=34MB.

### The PE-idle read (the whole phase-3 argument)

| kernel | wall (µs) | PE% | PE-active (µs) | **PE-idle (µs)** | HBMrd |
|---|---|---|---|---|---|
| v1 (fp32 fused) | 495.3 | 94% | 465.6 | 29.7 | 42 MB |
| **v3 (bf16-split, PROMOTED)** | 401.3 | **89%** | **357.2** | **44.1 (11%)** | 42 MB |
| sibling v4 (bf16-split, identical matmul) | 368.8 | **96%** | 354.0 | **14.8 (4%)** | 25 MB |

- The two bf16-split kernels have **the same matmul** (3 bf16 products, same K=1024,
  N=2048) and land within ~1% on PE-active (~355 µs). That is the real, unmovable
  floor: `2·M·N·K` at the bf16 systolic rate, run 3× ≈ the measured PE-active. **Cutting
  PE-active further requires either fewer products (fails the gate) or a lower-precision
  matmul (fails the gate).** Closed.
- The **difference** between the two is **PE-idle: 44 µs here vs 15 µs on the sibling.**
  That 29 µs gap is not the matmul — it is this op's extra per-M-tile non-PE work
  becoming *exposed* once the matmul got fast (fp32→bf16 dropped PE occupancy 97→89%:
  the denominator shrank, so the same fixed non-PE overhead shows as more idle).
- **Cost-model view of why the limb-split is the suspect.** Per M-tile the activation
  split issues 32 small [128,128] Vec/Scalar ops. On trn2 a [128,128] copy is ~133 ns
  of compute but carries ~268 ns Vector `semaphore_start` + 161 ns `write_drain`. Even
  with heavy cross-engine/cross-iteration overlap (which the measured 89% already
  reflects), fixed-overhead-dominated small ops are exactly the kind of work that leaves
  bubbles the wide sibling (no add, no z, PE=96%) never had. The residual add and z-read
  are single wide ops / DMA and are far likelier already hidden.

Conclusion: the only latency left to chase is the **PE-idle gap**, and the cheapest,
safest way to chase it is to **remove non-PE ops without touching the arithmetic**.

---

## 2. Directions enumerated, ranked (benefit vs risk)

### D1 — split `aT` limbs directly from the transpose PSUM  *(PRIMARY; bit-exact; low risk)*

**Idea.** v3's split reads the intermediate fp32 SBUF copy `aT_f`:
```
psum_t = nc_matmul(a_sub, identity, is_transpose=True)   # fp32 PSUM
aT_f   = copy(psum_t)                # fp32 PSUM->SBUF   (v3 line 151)  <-- REMOVE
aT_hi  = bf16(aT_f)                                       # v3 line 153
aT_res = aT_f - aT_hi                # fp32 tensor_tensor  v3 line 156
aT_lo  = bf16(aT_res)                                     # v3 line 158
```
Because `aT_f` is an *exact* fp32 copy of `psum_t`, every downstream read can source
`psum_t` directly:
```
aT_hi  = bf16(psum_t)                # bf16(psum_t)  == bf16(aT_f)   bit-for-bit
aT_res = psum_t - aT_hi              # (psum_t-aT_hi) == (aT_f-aT_hi) bit-for-bit
aT_lo  = bf16(aT_res)
```
This is **4 → 3 ops per sub-tile (32 → 24 per M-tile)**, and it is **bit-identical** —
`aT_hi`/`aT_lo` are the same bf16 values, so the 3-product matmul consumes the same
limbs and rel-L2 is exactly 1.528e-5. Reading `tensor_tensor`/`copy` directly from PSUM
is already proven in this kernel (the `inv_rms` post-scale reads the PSUM accumulator at
line 179; the eviction path is the same shape).

**Cost.** −8 ops/M-tile × 32 M-tiles = 256 fewer [128,128] Vec/Scalar ops overall, each
fixed-overhead-dominated. Zero extra PE. The only tradeoff is that `psum_t` stays live
one op longer (until `aT_res` is computed) before its PSUM bank frees — negligible: the
transpose uses a [128,128] (=128 elem/part) bank and there are 8 PSUM banks of 2048, so
there is no PSUM-pressure conflict with the [128,512] main accumulator.

**Expected latency.** Uncertain by design — this must be **measured**. Optimistic:
recovers part of the 44→15 µs idle gap → ~0.38–0.385 ms → **~4.8–5.0x**. Pessimistic
(sibling precedent: transpose and stationary fills were *already hidden* at PE=96%):
within-noise, PE stays ~89%, v3 remains promoted as a floor-confirmation. Both are
acceptable phase-3 exits; the change is cheap and risk-free.

**Correctness.** Bit-exact (argued above). Still gate on full-5-seed on device — expect
rel-L2 to reproduce 1.528e-5 identically; if it does not, that is a real bug to
investigate, not a precision tradeoff.

**Risk:** very low. No new primitive, no dtype change, no algebra change.

### D2 — reduce residual-add / RMSNorm-reduction exposure  *(secondary; measure only if D1 leaves idle)*

Only pursue if D1's profile still shows a Vec/Scalar bubble. Candidates, all
numerically neutral:
- **Per-sub-tile add fold:** compute `a` for sub-tile `kt` just before transposing it,
  so M-tile `t`'s adds overlap `t`'s transposes and `t−1`'s matmul. v3 already relies on
  `affine_range` to pipeline across M-tiles, so this is likely a no-op — measure before
  believing it.
- **Square-from-add fusion:** the `square` activation could read `a` immediately; no
  structural change expected. Record-only unless D1's digest points here.

D2 is contingent and not a headline candidate; it exists so the phase doesn't stop at D1
if D1 surfaces a specific remaining bubble.

### D3 — split-before-transpose (wide limb ops)  *(CLOSED — record-only)*

Splitting `a`'s limbs as **wide [128,1024] ops before** transposing (3 wide ops instead
of 24 small ops) sounds attractive but **doubles the transpose PE work**: each of `a_hi`
and `a_lo` must be transposed separately → 16 transpose `nc_matmul`s and 16 PSUM→SBUF
copies per M-tile vs v3's 8. That adds PE-active (the one thing we cannot afford) to save
Vec ops. Phase-2 already fixed "split *after* the transpose … costs **one** transpose,
not two." D1 keeps that single-transpose property while still shrinking the op count.
**Do not implement.**

### D4 — off-PE transpose to remove transpose PE work  *(CLOSED — record-only)*

Re-checked for the bf16-limb world (the sibling closed it for fp32 only): even with bf16
tiles, an SBUF→SBUF `dma_transpose` of a [128,128] tile is **still infeasible** — the
hardware-DGE path needs `src.shape[0]==16`, the software-DGE path needs the source in
HBM; the shape/memory constraints block it, not just the dtype. `nc_transpose(bf16)`
lands in fp32 PSUM and needs a re-cast (no net win, and the sibling measured the
vector-engine transpose as a +2% regress). The identity-matmul transpose stays. **Do
not explore.**

### D5 — precision / product-count changes  *(FORBIDDEN this phase)*

The 3-product bf16 arithmetic is pinned: margin is only 1.31x, D3-4-product was
MEASURED-REJECT (+28% for 1.6%), plain bf16 fails 117x. Phase 3 does **not** touch the
arithmetic — every candidate must reproduce rel-L2 1.528e-5 exactly.

### Also N/A (shape closure — see `docs/shape-specialization-closure-phase3.md`)
- **Edge / partial tiles:** none — M=32·128, K=8·128, N=4·512 all divide evenly.
- **Partition/free regime:** forced by `nc_matmul` (k on partition, `[m_in,n]` out).
- **N_CHUNK:** 512 = `psum_fmax`, already maximal.
- **M-blocking:** vacuous — `w'` fully resident, each `x`/`z` read once.
- **LNC2 / multi-core:** out of the single-core scoring contract.

---

## 3. Execution plan (≤3 candidates; bit-exact-first)

1. **v4 (D1, aT-split-from-PSUM):** `runs/add_rmsnorm_matmul_v4_psum_split.py`, forked
   from v3. Drop the `aT_f` fp32 copy; source `aT_hi`/`aT_lo` from `psum_t`. Verify
   full-5-seed PASS and confirm rel-L2 == 1.528e-5 (bit-exact). Run `--fast`, then the
   FULL 5-seed latency twice for stability, plus a same-session v3 anchor. Capture the
   profiler digest (watch PE% and Vec/Scl%).
   - **Promote iff** full-5-seed PASS AND p50 beats v3 out-of-noise (>1.8% band).
   - **Otherwise** record the within-noise floor-confirmation; v3 stays promoted.
2. **v5 (D2) only if D1 leaves a measurable Vec/Scalar bubble.** Contingent, not
   proactive. Same gates.
3. **Close-out:** whichever of v3/v4 is fastest is the phase-3 (and task) result. Report
   speedup vs the 1.859287 ms baseline on the **full** correctness gate.

## 4. Evidence to record
- `benchmark.csv`: one row per perf-relevant candidate (v4, and v5 if run), plus the
  D3/D4/D5-closed decisions as record-only rows/notes.
- `candidates.jsonl`: DAG node v4→v3 (and v5→v4 if run) with metrics, `rel_l2`,
  `per_seed_rel_l2`, the bit-exactness note, and the `per_seed_latency_ms=null` /
  `latency_scope` caveat carried from v2/v3.
- `profile/`: v4 digest + PE-idle before/after interpretation; a short shape-closure doc
  (or a reference to the sibling's) for AC-6.

## 5. Correctness invariants (never regress)
- fp32 RMSNorm reduction (`square`/`reduce`/`mean_eps`/`rsqrt`); eps added AFTER the `/K`
  mean, matching the reference exactly.
- g folded into `w'` on the k_in/partition axis; `inv_rms` applied post-scale at
  eviction — both exact commutations (offline fp32 control 4.82e-7).
- The 3-product bf16 split and its PINNED split order are unchanged; D1 only removes an
  exact intermediate copy, so rel-L2 must reproduce **1.528e-5** bit-for-bit.
- Every promotion gated on **full 5-seed** `l2_norm_passed`, not `--fast` alone.
- **CAVEAT (carried from phase 2):** the adapter fixes seed 42 for all 5 profiler seeds,
  so on-device 5-seed PASS is weak on *input* diversity; the offline 7-draw sim
  mitigates. v1 (3.754x) and v2 (3.898x) remain the pure-fp32 fallbacks.

--- Original Design Draft End ---
