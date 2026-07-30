# matmul_add_rmsnorm — Phase 3 Plan: Chase the ~64 µs PE-idle with a stationary-reuse GEMM loop reorder (measure-first, non-bit-exact)

## Goal Description

The promoted `matmul_add_rmsnorm_v2_bf16_split` is **4.879x (0.7722 ms)**, full-5-seed
PASS, rel-L2 1.544749e-5 (1.30x under the 2e-5 gate; PE=91.66% MFU=42.55% Vec=26.62%
Scl=13.24% DMA=23.93% HBMrd=84 MB HBMwr=34 MB; TRUE PE-active 0.7078 ms). Phase 3 is
regime/shape specialization. Three things are true at once:

1. **Every classic *shape* lever is closed identically to the siblings** (`add_rmsnorm_matmul`,
   `rmsnorm_matmul`). Same fixed M=4096, N=2048, K=2048, fp32 I/O. No edge tiles (M=32·128,
   K=16·128, N=4·512 all divide evenly), `nc_matmul` forces the k-on-partition layout,
   N_CHUNK=512=`psum_fmax` is maximal, `w` limbs are fully resident so M-blocking is vacuous,
   LNC2 is out of the single-core contract. Documented once (mirroring the sibling's closure).

2. **The sibling's headline phase-3 micro-lever is already spent here — measured, not inherited.**
   The sibling's phase-3 primary ("split the transposed activation limbs directly from the
   transpose PSUM bank, dropping the intermediate fp32 `xT_f` copy") was already built and
   measured in *this* task during phase 2 as `runs/matmul_add_rmsnorm_v2_psum_split.py`: a
   **byte-identical compiler no-op** (all instruction counts `==`, TRUE PE-active
   0.7078→0.7079 ms, rel-L2 bit-exact 1.544749e-5, +0.08% within noise). neuronx-cc already
   copy-propagates the exact fp32 PSUM→SBUF copy. That surface is closed by this op's own
   measurement.

3. **The one genuinely-untested lever is a GEMM loop reorder for stationary (weight-load)
   reuse.** v2's GEMM is **N-chunk-outer**, so each transposed activation limb `xT_hi[kt]` /
   `xT_lo[kt]` — the *stationary* operand loaded into the PE array — is reloaded **once per
   N-chunk = 4× per M-tile**, with an in-chunk stationary-reuse run of only 2. Reordering to
   **K-tile-outer with 4 live PSUM accumulators**, grouped by shared stationary limb, lengthens
   the stationary-reuse run from **2 → 8** consecutive matmuls (`xT_hi[kt]` across P1+P2 over 4
   chunks) and **→ 4** (`xT_lo[kt]` across P3 over 4 chunks), cutting stationary loads from
   **128 → 32 per M-tile**. This is the only precision-neutral lever that could touch the ~64 µs
   PE-idle (8.3% of the wall), and it is enabled by this op's larger K (16 K-tiles, 2× the
   sibling's 8), which doubles the accumulation depth over which reuse can be grouped.

**Stated honestly, the prior is that this reorder is a compiler no-op or a small regression**,
on two measured precedents: (a) `v2_psum_split` (this task) was a byte-identical no-op; (b)
`bmm`'s phase-3 finding that multi-bank PSUM pipelining is a compiler no-op *and* enlarging the
live PSUM/resident working set **regresses monotonically** as it constrains the `affine_range`
software pipeline (the `cross-batch-blocking-antilever` lesson). Holding 4 live [128,512]
accumulators is the same enlarged-live-set risk. The reason it is still worth **one** datum: the
reuse-run-length change (2→8) is a structural property of the source loop order that the
compiler cannot manufacture from the N-chunk-outer form without reordering across the whole
chunk loop, and PE=91.66% leaves a small but real idle to probe. So phase 3 **measures one
reorder candidate (D1)** and promotes it only on a full-5-seed PASS + an out-of-noise p50 win;
otherwise it is recorded as a floor-confirmation (or a first-class negative datum) and v2 stays
promoted. The realistic ceiling if *all* 64 µs idle were recovered is ~0.708 ms → **~5.32x**;
the honest expectation is at or near v2. **A within-noise floor-confirmation or a
codegen-infeasible result is a complete, acceptable phase-3 exit — not a failure.**

**No precision change is on the table.** The rel-L2 margin is 1.30x, the 4-product v2b was
already a decision-SKIP, and plain bf16 fails the gate 117×. Phase 3 optimizes the *schedule
around the fixed 3-product arithmetic*, not the arithmetic (D5 forbidden).

This plan implements ONLY `matmul_add_rmsnorm`; it does not touch the benchmark definition
(`../../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}`) or any other operator.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic
verification. Scoring command (run from inside `workspaces/matmul_add_rmsnorm/`):
`python3 ../../verify.py
--op matmul_add_rmsnorm --candidate runs/<file>.py` (add `--fast` for the seed-42 quick check;
drop it for the full 5-seed gate). Full profiler metrics come from `runs/dump_metrics.py`.
"rel-L2" everywhere means the verifier-reported relative-L2 field, not rounded text copied
through another layer.

- AC-1: **v3 (`runs/matmul_add_rmsnorm_v3_stationary_reorder.py`) is a GEMM-loop-reorder-only
  fork of v2.** Forked from `runs/matmul_add_rmsnorm_v2_bf16_split.py`, it changes ONLY the GEMM
  loop nest to K-tile-outer with **4 live [128,512] fp32 PSUM accumulators**, grouped by shared
  stationary limb: for each `kt`, a hi-pass runs `xT_hi[kt]` as the stationary operand across all
  4 chunks for both P1 (`xT_hi@w_hi`) and P2 (`xT_hi@w_lo`) — 8 consecutive matmuls — then a
  lo-pass runs `xT_lo[kt]` across all 4 chunks for P3 (`xT_lo@w_hi`) — 4 consecutive matmuls;
  after the `kt` loop, each accumulator adds its `z_tile` chunk and evicts to the `y` buffer.
  Everything else is byte-for-byte v2: the transpose + PINNED limb split (`w`→`w_hi`→`w_res`→
  `w_lo`; `xT`→`xT_hi`→`xT_res`→`xT_lo`), the 3 products (`xT_hi@w_hi + xT_hi@w_lo + xT_lo@w_hi`,
  `xT_lo@w_lo` dropped), the fp32 residual add before the norm, the fully-fp32 RMSNorm
  (`square`/full-N `tensor_reduce`/`mean_eps`/`rsqrt`), `eps` added AFTER the `/N` mean, `g`
  applied on the OUTPUT free axis after the norm, the full-width [128,N] output store, and the
  raw-2D I/O with the exact signature `kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor)`.
  - Positive Tests (expected to PASS):
    - The v3 file exists in `runs/`, and a diff against v2 shows changes confined to the GEMM
      loop nest and its PSUM-accumulator allocation (plus the post-`kt` residual-add/eviction).
    - The intra-`kt` matmul order is exactly hi-pass (P1,P2 over 4 chunks) then lo-pass (P3 over
      4 chunks), giving stationary-reuse runs of 8 and 4.
    - All 4 chunk accumulators are zeroed exactly once per M-tile **before** the `kt` loop and are
      not reused across M-tiles (each M-tile gets its own fresh set of 4 PSUM accumulators) — this
      forecloses the cross-chunk / cross-M-tile contamination bug mode.
    - `matmul_instruction_count` is expected to stay 6664 vs v2 — the *same* matmuls, reordered
      (see AC-4 for how a differing count is handled).
  - Negative Tests (expected to FAIL):
    - Any change to the arithmetic: product count, split order, dropped-term choice, `g`-fold
      placement (must stay on the N/free axis, applied after the norm — folding into `w` changes
      the math), `inv_rms` placement (must not commute out — the norm reduces over N), RMSNorm
      dtype, or the `eps`-after-mean form.
    - The reorder collapsing to <4 live accumulators, or an intra-`kt` order that does not
      produce the 8/4 reuse runs, while relabelling the file as the reorder.

- AC-2: **Compile + PSUM-allocation + semantic gate (runs BEFORE any full-5-seed or profiling
  spend).** v3 compiles with **no PSUM bank-allocation or spill error**, and a single `--fast`
  (seed-42) run passes with a verifier-reported rel-L2 within a small band of v2's seed-42 value.
  The intended live PSUM set is 4 [128,512] accumulator banks + 1 [128,128] transpose bank —
  an **estimated ~5 of 8 fp32 PSUM banks**, but this is a source-level estimate, NOT a guaranteed
  feasibility fact: `affine_range` software-pipelining, accumulator-init/cast temporaries, the
  post-`kt` eviction, and any implicit compiler buffers can raise actual liveness, and the
  transpose bank's liveness window relative to the GEMM is compiler-decided. Holding 4 live
  accumulators simultaneously across the full 16-tile `kt` loop is therefore a **compile-gated**
  property to be confirmed against the actual compiler allocation, not asserted. On success, the
  digest (AC-4) should confirm no new spill.
  - Positive Tests (expected to PASS):
    - v3 compiles cleanly (no PSUM-bank/spill/allocation error), and the actual compiler
      allocation/liveness is checked (no unexpected spill).
    - `--fast` `l2_norm_passed = True` AND the rel-L2 is within `|Δ| ≤ 1e-6` of v2's seed-42
      1.544749e-5 (a small ulp-level move from the changed fp32 accumulation order is expected;
      see AC-3).
  - Negative Tests (expected to FAIL / halt):
    - A compile, PSUM-bank, or spill failure → **D1 is codegen-infeasible**; record it CLOSED as
      a first-class negative datum (like the sibling's D4), keep v2 promoted, and do NOT spend
      full-5-seed/profiling runs on an uncompilable v3. **A compile failure produces no profiler
      digest, so it does NOT trigger the AC-9 D2 probe** (D2 is strictly profiler-contingent on a
      *compiling* D1 that shows a PSUM/pipeline bubble); a compile failure simply closes D1 and
      keeps v2 promoted. (The 4→2-bank move is a distinct enough schedule that a "shrink the live
      set" reaction to an allocation failure would be a new proactive candidate, which the draft's
      ≤2-candidate / no-proactive-D2 discipline forbids.)
    - rel-L2 drift beyond the `1e-6` band on `--fast` → **INVESTIGATE**; a materially different
      rel-L2 signals a scheduling/aliasing bug (wrong accumulator, cross-chunk contamination),
      not a precision tradeoff. Do NOT proceed to promotion.

- AC-3: **v3 is correct on the full gate, with a rel-L2 guard band.** After AC-2 passes, a FULL
  5-seed run has `l2_norm_passed = True` at seeds `[0,21,42,63,84]`, and each **per-seed** rel-L2
  stays `≈ 1.5447e-5` — specifically `≤ 1.8e-5` (the danger-band guard) and within `|Δ| ≤ 1e-6`
  of the **same-session v2 anchor's per-seed rel-L2**. The comparison reference is the
  same-session v2 anchor's per-seed values (from Milestone 1 / the AC-5 bracket), not a
  transcribed constant; the single value `1.544749e-5` is used as the reference **only because v2
  measured a UNIFORM per-seed rel-L2 of 1.544749e-5 across all 5 seeds** — a direct consequence of
  the benchmark contract (the adapter fixes `np.random.seed(42)` for all 5 profiler seeds, so the
  5 seeds share one input draw; see AC-8.1), which makes rel-L2 effectively seed-invariant under
  this contract. If a future v2 anchor shows any per-seed spread, compare v3 seed-by-seed against
  that anchor rather than against the constant. This is **not** a bit-exact expectation (v3
  changes the fp32 PSUM accumulation order — hi-pass all `kt` then lo-pass all `kt` vs v2's
  per-`kt` interleave — and fp32 add is non-associative), so a small ulp-level move is allowed; a
  **material** move (toward or past 1.8e-5) is a scheduling/aliasing bug to investigate, not a
  precision tradeoff, and blocks promotion until explained.
  - Positive Tests (expected to PASS):
    - Full-5-seed `l2_norm_passed = True`; every per-seed rel-L2 `≤ 1.8e-5` and within `1e-6` of
      the same-session v2 anchor's per-seed value (≈1.544749e-5).
  - Negative Tests (expected to FAIL):
    - Any seed with `l2_norm_passed = False`, or any per-seed rel-L2 `> 1.8e-5` (fails the guard
      band even if it clears the raw 2e-5 gate).
    - A per-seed rel-L2 moving away from the v2 anchor by more than `1e-6` while still passing —
      surfaces a hidden reorder/aliasing bug; must be investigated, not auto-promoted.

- AC-4: **v3 profiler digest captured; the reuse mechanism is verified, HBM/spill unchanged.**
  Record PE%, MFU%, Vec%, Scl%, DMA%, HBMrd, HBMwr, `matmul_instruction_count`, TRUE PE-active
  (`tensor_engine_active_time_ns`), and `psum_read_sbuf_write_count` for v3 under `profile/`,
  alongside the exact profiler command/config and the compiler/runtime version, compared against
  v2 (0.7722 ms, PE=91.66%, TRUE PE-active 0.7078 ms, 84/34 MB, psum_read 132). Because the limbs
  are built on-chip and the matmuls are only reordered, HBM and `matmul_instruction_count` should
  be materially unchanged. **The primary win mechanism is a drop in TRUE PE-idle (equivalently a
  rise in PE%) attributable to fewer stationary-operand reloads** — the profiler does not expose a
  direct "stationary load count", so PE-idle / `tensor_engine_active_time_ns` is the mechanism
  metric, and `matmul_instruction_count`, HBMrd/HBMwr, and `psum_read_sbuf_write_count` are
  **regression sentinels** (they must stay flat to prove the change was a pure reorder, but they
  are not themselves the reuse evidence). A genuine win must show the PE-idle mechanism, not
  measurement variance.
  - Positive Tests (expected to PASS):
    - A digest with all listed metrics is written under `profile/`, plus the profiler
      command/config and version string.
    - HBMrd/HBMwr within a small margin of 84 MB / 34 MB; no new SBUF/PSUM spill; and
      `matmul_instruction_count` stays 6664 (if it differs, the difference must be **explained** as
      a compiler bookkeeping/unrolling artifact with the arithmetic still provably the same 3
      products — it is not an automatic fail, but an unexplained change blocks promotion).
    - If v3 wins, the win correlates with a higher PE% / lower TRUE PE-idle (the stationary-reuse
      mechanism) with `psum_read_sbuf_write_count` flat (~132) or improved, and no DMA/HBM rise.
  - Negative Tests (expected to FAIL):
    - A promotion recorded with no profiler digest.
    - A latency change promoted while HBM or `psum_read` moved materially, or while
      `matmul_instruction_count` changed without an explanation that preserves the 3-product
      arithmetic (indicates the memory/arithmetic pattern changed, not a pure reorder); or a
      "win" with no corresponding PE-idle mechanism (i.e. pure variance) left unflagged.

- AC-5: **Promotion is gated on measured correctness AND an out-of-noise latency win.** Promote
  v3 IFF full-5-seed `l2_norm_passed = True` with the AC-3 guard band satisfied, AND its p50
  beats a same-session v2 anchor by more than the noise band (**>1.8%**), measured over **≥2
  full-5-seed v3 runs** bracketed by v2 anchors (a v2 anchor before and after the v3 runs) with
  **no p90/p99/variance regression**. Otherwise, record the within-noise **floor-confirmation**
  (or the regression as a first-class negative datum, per `bmm`'s anti-lever discipline) and v2
  stays promoted. The directional expectation is at-or-near v2, with an optimistic ceiling of
  ~5.0–5.32x if the idle is genuinely weight-load-bound; the exact figure is a **trend/direction,
  NOT** an acceptance requirement.
  - Positive Tests (expected to PASS):
    - ≥2 full-5-seed v3 measurements both beat the same-session v2 anchor by >1.8%, with stable
      p90/p99/variance → promote v3.
    - v3 lands within ±1.8% of the v2 anchor → record floor-confirmation, keep v2 promoted (also
      an acceptable exit).
  - Negative Tests (expected to FAIL):
    - A `--fast`-only measurement used as the promotion basis.
    - A within-noise or regressed latency promoted anyway, or a promotion declared without a
      same-session v2 anchor bracketing the v3 runs, or with a p90/p99/variance regression.

- AC-6: **Shape-specialization closure is documented (mirror of the sibling's AC-6).** A short
  `docs/shape-specialization-closure-phase3.md` (or an explicit by-reference note to §2 of the
  draft and the sibling `add_rmsnorm_matmul`/`rmsnorm_matmul` closures) records that every
  classic phase-3 shape lever is vacuous or pinned: no edge/partial tiles (M=32·128, K=16·128,
  N=4·512 divide evenly), layout forced by `nc_matmul` (k on partition, `[m_in,n]` output),
  N_CHUNK=512=`psum_fmax` already maximal, M-blocking vacuous (`w` limbs fully resident, each
  `x`/`z` read once at the 84 MB one-pass floor), LNC2 out of the single-core contract. It also
  records the **K-note**: this op's K=2048 (16 K-tiles) is 2× the sibling's K=1024 (8), doubling
  the accumulation depth and the count of distinct stationary limbs (32 vs 16 per M-tile) — the
  one shape difference that gives D1 its longer reuse run.
  - Positive Tests (expected to PASS):
    - The closure doc (or by-reference note) exists and covers all five levers plus the K-note.
  - Negative Tests (expected to FAIL):
    - A phase-3 shape lever claimed "open" without evidence, or the closure left undocumented.

- AC-7: **Closed directions are recorded as first-class negative data, not silently dropped.**
  `benchmark.csv`/`candidates.jsonl` carry record-only entries for: **D3** PSUM-source
  activation-limb split (already built + measured in phase 2 as
  `runs/matmul_add_rmsnorm_v2_psum_split.py` — byte-identical compiler no-op; cite the phase-2
  datum, **do not rebuild**); **D4** off-PE transpose (`dma_transpose` of a [128,128] tile
  infeasible — HW-DGE needs `src.shape[0]==16`, SW-DGE needs an HBM source; `nc_transpose`
  lands in fp32 PSUM needing a re-cast and measured +2% on the siblings; the 512 identity-matmul
  transposes are already hidden under the PE-bound matmul; **do not explore**); **D6**
  split-before-transpose (doubles the transpose PE work — transpose `x_hi` and `x_lo` separately
  → 32 transpose matmuls/M-tile instead of 16 — adding the one thing we cannot afford; **do not
  implement**); **D5** precision/product-count changes (**FORBIDDEN this phase** — 3-product bf16
  pinned, margin 1.30x, v2b 4-product decision-SKIP, plain bf16 fails ~117×).
  - Positive Tests (expected to PASS):
    - D3/D4/D5/D6 each recorded with the closure rationale (D3 citing the phase-2 no-op datum).
  - Negative Tests (expected to FAIL):
    - Implementing D3, D4, D5, or D6, or dropping any of them without a recorded rationale.

- AC-8: **Correctness invariants are never regressed.** fp32 residual add `y = x@w + z` **before**
  the norm; fp32 RMSNorm reduction over N (`square`/full-2048 `tensor_reduce(axis=[1])`/
  `mean_eps`/`rsqrt`), with `eps` added AFTER the `/N` mean (matching `np.mean(y**2,axis=-1)+eps`);
  `g` on the OUTPUT free axis (N), applied as a `[1,N]→[128,N]` broadcast multiply **after** the
  norm and **never folded into `w`**; `inv_rms` **not** commuted out (the norm reduces over N, so
  the full [128,N] row is assembled before `inv_rms` is known); the 3-product bf16 split and its
  PINNED split order unchanged; raw-2D I/O + the exact signature; full-width [128,N] output store.
  D1 changes ONLY the *order* of fp32 PSUM accumulation. Every promotion gated on **full 5-seed**
  `l2_norm_passed`, not `--fast` alone.
  - Positive Tests (expected to PASS):
    - v3 reproduces v2's rel-L2 ≈1.5447e-5 (within the AC-3 band) and preserves all invariants.
  - Negative Tests (expected to FAIL):
    - `eps` added before / scaled by `1/N` (`rsqrt((sumsq+eps)/N)`); `g` folded into `w` or moved
      onto the contraction axis; `inv_rms` commuted to a chunk-wise post-scale; the RMSNorm
      reduction moved out of fp32; the residual add moved after the norm.
  - AC-8.1: **Carried input-diversity caveat (record-only, non-blocking).** The adapter fixes
    `np.random.seed(42)` for all 5 profiler seeds, so on-device 5-seed PASS is a
    determinism/stability gate, weak on *input* diversity; the offline 7-draw sim (worst bf16-only
    4.454e-6) covers input diversity. `v1` (3.920x, pure fp32) is retained as the guaranteed-correct
    fp32 fallback. Fixing the adapter is OUT of scope (it is the NKIBench benchmark contract);
    promotion evidence is stated as being against the current contract.
    - Positive: the caveat is restated in the v3 evidence.
    - Negative: claiming genuine per-seed input diversity from the on-device 5-seed run.

- AC-9: **D2 (2-bank / 2-chunk grouping) is contingent, not proactive, and numerically re-gated.**
  Pursue D2 as `v4` (`runs/matmul_add_rmsnorm_v4_stationary_reorder_2bank.py`) ONLY if D1's
  profiler digest shows a **specific PSUM/pipeline bubble** — i.e. PE-idle went **up** (not down)
  and the digest attributes it to the enlarged 4-bank live set rather than the reorder being a
  pure no-op. D2 groups the stationary reuse over **2 chunks at a time** (2 live [128,512] banks +
  1 transpose = 3 banks), keeping a reuse run of 4 while halving the live PSUM set — testing
  whether the regression is the enlarged live set (D2 recovers) or the reorder itself (D2 also a
  no-op). It is gated exactly like v3 (AC-2/AC-3/AC-4/AC-5), including the non-bit-exact rel-L2
  re-gate.
  - Positive Tests (expected to PASS):
    - v4 is attempted only when D1's digest shows a specific PSUM/pipeline bubble, and it
      reproduces rel-L2 within the AC-3 band.
  - Negative Tests (expected to FAIL):
    - Building v4 proactively when D1 is a clean no-op (record why D2 was skipped instead).
    - A D2 variant that changes the arithmetic or the 3-product split (only the bank/chunk
      grouping and the fp32 accumulation order may change).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. This phase is
highly deterministic: the arithmetic is pinned and the single primary lever (D1) is fixed by the
draft, so the bounds are narrow. The one elective scope (D2) is itself gated on a measurement.

### Upper Bound (Maximum Acceptable Scope)
Implement and measure D1 as `v3` (K-tile-outer, 4 live [128,512] PSUM banks, hi-pass/lo-pass
stationary grouping), gated through the compile+PSUM-allocation+semantic probe (AC-2), the
full-5-seed guard-banded gate (AC-3), and the bracketed same-session v2-anchored latency
comparison (AC-4/AC-5). If — and only if — D1's profile shows a specific PSUM/pipeline bubble,
additionally implement and measure the one 2-bank D2 variant as `v4` (AC-9). Document the
shape-specialization closure (AC-6) and record D3/D4/D5/D6 as first-class closed directions
(AC-7). Promote whichever of v2/v3(/v4) is fastest under the full correctness gate, and report
the speedup vs the 3.768493 ms baseline.

### Lower Bound (Minimum Acceptable Scope)
Implement D1 as `v3`, run the AC-2 compile+PSUM-allocation+semantic probe, and record the
outcome. If v3 is codegen-infeasible (AC-2 negative — e.g. the 4-bank live set fails allocation),
record it CLOSED and keep v2 promoted. If v3 compiles and passes but lands within noise of v2,
record the floor-confirmation and keep v2 promoted. Either way, produce the AC-6 shape-closure
doc/reference and the AC-7 closed-direction records. A within-noise or codegen-infeasible result
is a **complete, acceptable** phase-3 exit — a successful floor confirmation, not a failure.

### Allowed Choices
- Can use: the existing bf16x2 3-product split verbatim; K-tile-outer loop reordering with 2 or 4
  live [128,512] PSUM accumulators; either `affine_range` or a sequential `range` for the
  accumulator-heavy K loop if one compiles/schedules better than the other (the loop *form* is a
  free implementation choice as long as the arithmetic and the hi-pass/lo-pass reuse structure
  are preserved); the same identity-matmul transpose and PINNED limb split; the same fused fp32
  RMSNorm epilogue and full-width output store.
- Cannot use: any change to the 3-product bf16 arithmetic, product count, split order, dropped
  term, or dtypes (D5, FORBIDDEN); `g` folded into `w` or moved to the contraction axis;
  `inv_rms` commuted to a chunk-wise post-scale; a partial/streaming RMSNorm reduction that
  changes the fp32 summation order; split-before-transpose (D6, doubles transpose PE); off-PE /
  `dma_transpose` / `nc_transpose` transpose replacements (D4, infeasible); rebuilding the D3
  PSUM-source split (already a measured no-op); any edit to the benchmark definition or another
  operator; a `--fast`-only or within-noise promotion.

> **Note on Deterministic Design**: The draft pins the arithmetic and fixes D1 as the single
> primary lever, so the upper and lower bounds nearly converge — the only elective scope is the
> contingent D2, gated on a measurement. "Allowed Choices" reflects this narrow constraint: the
> schedule (loop order, live-bank count, loop form) may change, and because that changes the fp32
> accumulation order the rel-L2 must be re-gated (non-bit-exact, unlike the sibling's bit-exact
> D1); the arithmetic may not change.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach
D1 changes ONLY the GEMM loop nest in `runs/matmul_add_rmsnorm_v2_bf16_split.py`. v2's GEMM is
N-chunk-outer (each chunk fully accumulates its 16 K-tiles × 3 products in one [128,512] PSUM
bank before the next chunk), so each stationary limb `xT_hi[kt]`/`xT_lo[kt]` is reloaded once per
chunk = 4× per M-tile, with an in-chunk reuse run of only 2:

```
for c in 4:                                  # N-chunk outer
    acc = zeros[128,512] psum
    for kt in 16:
        acc += xT_hi[kt] @ w_hi[kt,c]        # P1  (stationary xT_hi[kt])
        acc += xT_hi[kt] @ w_lo[kt,c]        # P2  (stationary xT_hi[kt])  reuse run = 2
        acc += xT_lo[kt] @ w_hi[kt,c]        # P3  (stationary changes to xT_lo[kt])
    y[:,chunk_c] = acc + z_tile[c]
```

Reorder to K-tile-outer with 4 live accumulators, grouped by shared stationary limb:

```
acc = [zeros[128,512] psum for c in 4]       # 4 live banks, accumulate across all kt
for kt in 16:
    for c in 4:                              # hi-pass: xT_hi[kt] stationary for 8 matmuls
        acc[c] += xT_hi[kt] @ w_hi[kt,c]     # P1
        acc[c] += xT_hi[kt] @ w_lo[kt,c]     # P2
    for c in 4:                              # lo-pass: xT_lo[kt] stationary for 4 matmuls
        acc[c] += xT_lo[kt] @ w_hi[kt,c]     # P3
for c in 4: y[:,chunk_c] = acc[c] + z_tile[c]   # residual add, then the fp32 norm epilogue (v2, unchanged)
```

Now `xT_hi[kt]` is stationary across **8** consecutive matmuls and `xT_lo[kt]` across **4**.
Stationary loads drop 128 → 32 per M-tile. **PSUM feasibility (source-level estimate,
compile-gated):** 4 live [128,512] banks (512 ≤ 2048 elem/bank) + 1 [128,128] transpose bank ≈ 5
of 8 at the source level — but actual liveness is compiler-decided (`affine_range` pipelining,
init/cast temporaries, and the post-`kt` eviction can raise it), so this is an estimate to be
confirmed against the real allocation, not a guaranteed fit. The 4 accumulators must remain
simultaneously live across the whole `kt` loop, which is exactly the enlarged-live-set risk the
`bmm` precedent flagged; AC-2 compile-gates this before any remote spend. **Loop form is a free
choice:** the pseudocode's `affine_range` may inflate the live set via software pipelining; a
sequential `range` on the accumulator loop is an allowed alternative if it compiles/schedules
cleaner (the arithmetic and the reuse structure are what must be preserved, not the loop
keyword). **Correctness is NOT bit-exact:** the fp32 PSUM accumulation order changes (hi-pass all
`kt`, then lo-pass all `kt`, vs v2's per-`kt` P1,P2,P3 interleave), so rel-L2 will move by
~ulp-level and must be re-gated full-5-seed (AC-3); a *material* move is a bug, not a precision
tradeoff.

### Relevant References
- `runs/matmul_add_rmsnorm_v2_bf16_split.py` — the PROMOTED base; fork v3 from it. Only the
  `for c in nl.affine_range(N_CHUNKS): ... for kt in nl.affine_range(K_TILES): ...` GEMM block and
  its PSUM-accumulator allocation change; the transpose+split loop and the RMSNorm epilogue are
  untouched.
- `runs/matmul_add_rmsnorm_v2_psum_split.py` — the D3 byte-identical compiler no-op datum (why D3
  is closed; the `cross-batch-blocking-antilever` / compiler-copy-propagation precedents).
- `runs/dump_metrics.py` — dumps the full profiler `summary_metrics` (TRUE PE-active,
  `matmul_instruction_count`, `psum_read_sbuf_write_count`, per-engine counts) that `verify.py`'s
  5 percentages omit; use it for the AC-4 mechanism check.
- `runs/offline_bf16_split_sim.py` — the offline pre-check (worst idealized bf16-only rel-L2
  4.454e-6; 7-draw input diversity mitigating the fixed-seed-42 caveat).
- `workspaces/add_rmsnorm_matmul/docs/shape-specialization-closure-phase3.md` — the sibling's AC-6
  closure to mirror or reference.
- `../../verify.py` — the correctness/latency harness; `--fast` = seed-42 only, drop for the full
  5-seed gate.

## Dependencies and Sequence

### Milestones
1. **Re-anchor v2 (same-session control).** Re-measure v2 full-5-seed via `runs/dump_metrics.py`
   to pin this session's PE-active / PE% / idle and latency anchor (profiler jitter ~±1.8%).
   Confirm PE≈91.7%, TRUE PE-active ≈0.7078 ms, HBM 84/34 MB, rel-L2 1.544749e-5.
2. **v3 (D1, stationary-reuse reorder) built and compile+semantic-gated.**
   - Phase A: fork `runs/matmul_add_rmsnorm_v3_stationary_reorder.py` from v2; change only the
     GEMM loop nest (K-tile-outer, 4 live PSUM banks, hi-pass/lo-pass grouping) and the post-`kt`
     residual add + eviction (AC-1).
   - Phase B: compile + PSUM-allocation + `--fast` semantic probe; confirm rel-L2 within `1e-6`
     of the v2 anchor (AC-2). On compile/spill failure → record D1 CLOSED, keep v2 promoted, and
     **do NOT run D2** (D2 is strictly profiler-contingent on a compiling D1; a compile failure
     yields no digest) — jump to Milestone 5; otherwise continue.
3. **v3 correctness + latency measurement.**
   - Phase A: full-5-seed run; confirm the AC-3 guard band (per-seed rel-L2 ≤ 1.8e-5 and within
     `1e-6` of v2).
   - Phase B: ≥2 full-5-seed latency runs bracketed by same-session v2 anchors; capture the
     profiler digest + command/config + version, and the mechanism read (PE% / TRUE PE-idle /
     `psum_read_sbuf_write_count`) (AC-4). Apply the promotion gate (AC-5).
4. **v4 (D2, 2-bank variant) — contingent.** Only if Milestone 3's digest shows a specific
   PSUM/pipeline bubble (PE-idle up, attributed to the 4-bank live set): build the one 2-bank
   variant and gate it identically (AC-9). Skip otherwise and record why.
5. **Close-out.** Document the AC-6 shape-closure and the AC-7 closed directions. Whichever of
   v2/v3(/v4) is fastest under the full 5-seed gate is the phase-3 (and task) result; report the
   speedup vs the 3.768493 ms baseline.

Dependencies: Milestone 1 anchors Milestones 3/5. Milestone 2 gates Milestone 3 (no remote spend
on an uncompilable v3). Milestone 4 depends on Milestone 3's digest. Milestone 5's closure docs
(AC-6/AC-7) have no code dependency and may proceed in parallel with the measurement milestones.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Re-anchor v2: full-5-seed re-measure via `dump_metrics.py`, pin PE%/TRUE PE-active/latency/rel-L2 | AC-4, AC-5 | coding | - |
| task2 | Fork `v3_stationary_reorder.py` from v2: K-tile-outer GEMM, 4 live PSUM banks, hi-pass/lo-pass grouping, post-`kt` residual add + eviction; arithmetic/epilogue byte-for-byte v2 | AC-1, AC-8 | coding | task1 |
| task3 | Compile + PSUM-allocation + `--fast` semantic probe; confirm rel-L2 within 1e-6 of 1.544749e-5; on compile/spill failure record D1 CLOSED | AC-2 | coding | task2 |
| task4 | Full-5-seed correctness with the guard band (per-seed rel-L2 ≤ 1.8e-5, within 1e-6 of v2) | AC-3, AC-8 | coding | task3 |
| task5 | ≥2 full-5-seed latency runs bracketed by same-session v2 anchors; capture profiler digest + mechanism read (PE%/PE-idle/psum_read); apply the promotion gate | AC-4, AC-5 | coding | task4 |
| task6 | (Contingent) build + gate the 2-bank D2 `v4` ONLY if task5's digest shows a specific PSUM/pipeline bubble; else record why skipped | AC-9 | coding | task5 |
| task7 | Write the shape-specialization closure doc (5 levers + K-note) or by-reference note | AC-6 | coding | - |
| task8 | Record D3/D4/D5/D6 closed directions in `benchmark.csv`/`candidates.jsonl` with rationale (D3 cites the phase-2 no-op datum; no v3 profiling dependency) | AC-7 | coding | task3 |
| task9 | Independent review of the v3 accumulation-order change: confirm rel-L2 is expected to stay ≈1.5447e-5 and the reorder cannot alias/contaminate accumulators | AC-1, AC-3 | analyze | task2 |

## Claude-Codex Deliberation

### Agreements
- The D1 reorder must be **measured, not assumed**: the prior (compiler no-op or small regression)
  is explicit, and a within-noise or codegen-infeasible outcome is an acceptable phase-3 exit.
- The 3-product bf16 arithmetic is **pinned**; D5 precision changes are forbidden this phase.
- The correctness re-gate is **not bit-exact** (unlike the sibling's D1): the fp32 accumulation
  order changes, so rel-L2 is re-measured full-5-seed with a guard band.
- Promotion requires an out-of-noise (>1.8%) p50 win over a same-session v2 anchor with no
  variance regression; a `--fast`-only or within-noise promotion is disallowed.
- A compile/PSUM-allocation failure is a first-class CLOSED datum, not a silent drop; remote
  spend is gated behind the AC-2 compile probe.

### Resolved Disagreements
- **D1-first vs 2-bank-D2-first (Codex QUESTIONS_FOR_USER Q1).** Codex asked whether to prioritize
  the lower-live-set 2-bank variant first. Resolution: keep the draft's ordering — D1 (4-bank) is
  the primary probe because it yields the maximum reuse run (8) and the strongest signal; D2 is
  contingent on D1 showing a *specific* PSUM/pipeline bubble (AC-9). Rationale: if D1 is a clean
  no-op, D2 adds no information; if D1 regresses on live-set pressure, D2 is the diagnostic that
  separates "enlarged live set" from "reorder itself". This preserves the ≤2-candidate budget.
- **rel-L2 acceptance margin (Codex Q3 / CANDIDATE_CRITERIA).** Codex flagged that passing near
  1.99e-5 should not promote. Resolution: adopt a hard **guard band** (per-seed rel-L2 ≤ 1.8e-5
  AND within 1e-6 of v2's 1.544749e-5) in AC-3, and treat any material move as an
  investigate-not-promote bug. Since v3 only reorders fp32 accumulation, rel-L2 is expected to
  stay ≈1.5447e-5; a jump toward the band is itself a bug signal.
- **Mechanism vs variance (Codex CORE_RISKS / CANDIDATE_CRITERIA).** Resolution: AC-4 requires a
  win to correlate with a measured PE-idle/PE% mechanism and with `matmul_instruction_count` /
  HBM / `psum_read` held flat, so a lucky-variance "win" cannot be promoted.
- **`affine_range` vs sequential loop for the 4-bank accumulator loop (Codex MISSING_REQUIREMENTS
  / TECHNICAL_GAPS).** Resolution: the loop *form* is an explicit Allowed Choice; AC-2 compile-gates
  whichever form is used, and the enlarged-live-set risk is called out as the primary regression
  mode. The reuse structure (hi-pass/lo-pass) and arithmetic are what must be preserved.
- **D2 after a D1 *compile* failure (Codex Round-1 UNRESOLVED / REQUIRED_CHANGES).** Codex found a
  contradiction: AC-2 originally let a compile failure fall back to D2, but AC-9 makes D2 strictly
  profiler-contingent (a compile failure yields no digest). Resolution: keep D2 **strictly
  profiler-contingent** — a D1 compile/allocation failure CLOSES D1 and keeps v2 promoted, with no
  D2 fallback. Rationale: the draft's discipline is ≤2 candidates with "no proactive D2"; a
  "shrink the live set after an allocation failure" reaction is a *new* proactive candidate, not
  the measurement-triggered D2 the draft authorizes. AC-2's negative test now says this explicitly.
- **PSUM "5 of 8 banks" framed as fact (Codex Round-1 REQUIRED_CHANGES / DISAGREE).** Resolution:
  AC-2 reworded to an *estimate* to be confirmed against the actual compiler allocation/liveness,
  noting `affine_range` pipelining / init-cast temporaries / eviction / implicit buffers can raise
  liveness — the compile gate is the real check, not the arithmetic count.
- **Fixed `1.544749e-5` rel-L2 reference brittleness (Codex Round-1 REQUIRED_CHANGES).**
  Resolution: AC-3 now compares against the **same-session v2 anchor's per-seed** rel-L2, using the
  constant only because v2 measured a UNIFORM per-seed value under the fixed-seed-42 contract
  (AC-8.1); if a future anchor shows spread, compare seed-by-seed.
- **`matmul_instruction_count` hard-equality too strict (Codex Round-1 REQUIRED_CHANGES /
  OPTIONAL).** Resolution: AC-4 demotes it from an automatic-fail to a "must-be-explained-if-different"
  sentinel (a compiler bookkeeping change that preserves the 3-product arithmetic is allowed; an
  *unexplained* change blocks promotion). PE-idle is named the primary mechanism; HBM / psum_read /
  matmul_count are regression sentinels.
- **Accumulator-init invariant (Codex Round-1 OPTIONAL).** Resolution: AC-1 now spells out that all
  4 accumulators are zeroed once per M-tile before the `kt` loop and never reused across M-tiles
  (closing the named cross-chunk/cross-M-tile contamination bug mode).
- **Doc close-out coupling (Codex Round-1 OPTIONAL).** Resolution: task8 (closed-direction records)
  now depends on task3, not task5 — documentation does not need v3 latency measurement.

### Convergence Status
- Final Status: `converged` (2 Codex rounds; Round-2 raised only clarifications, all resolved
  against the draft + prior-phase evidence with no remaining opposite opinions).

## Pending User Decisions

- (none) — the draft is prescriptive (arithmetic pinned; D1 primary, D2 contingent; speedup is an
  explicit trend, not a hard target), and every Codex `QUESTIONS_FOR_USER` item resolved cleanly
  against the draft + prior-phase evidence (see Resolved Disagreements). The quantitative values
  are all inherited from the benchmark contract / prior phases: the **2e-5 rel-L2 gate** and the
  **3.768493 ms baseline** are the fixed NKIBench scoring contract (hard); the **1.8e-5 danger
  band**, the **>1.8% latency noise band**, and the **1e-6 rel-L2 drift band** are established
  guard thresholds from phase 2 (hard gates for promotion); and the **~5.0–5.32x speedup ceiling**
  is an optimization **trend/direction**, explicitly NOT a hard requirement (a within-noise
  floor-confirmation is an acceptable exit).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Phase", "Step", "D1"/"D2", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. a comment explaining the
  K-tile-outer stationary-reuse grouping and the 4-live-PSUM-bank accumulation, not "D1").

--- Original Design Draft Start ---

# matmul_add_rmsnorm — Phase 3 draft (regime / shape specialization)

## 0. TL;DR

The promoted `matmul_add_rmsnorm_v2_bf16_split` is **4.879x (0.7722 ms)**, full-5-seed
PASS, rel-L2 1.544749e-5 (1.30x under the 2e-5 gate). Phase 3 is regime/shape
specialization. Three things are true here at once:

1. **Every classic *shape* lever is closed identically to the siblings**
   (`add_rmsnorm_matmul`, `rmsnorm_matmul`) — same fixed contract, all dims divide
   evenly (M=4096=32·128, K=2048=16·128, N=2048=4·512): no edge tiles, `nc_matmul`
   forces the k-on-partition layout, N_CHUNK=512=`psum_fmax` is maximal, `w` limbs are
   fully resident so M-blocking is vacuous, LNC2 is out of the single-core contract.
   Documented once by reference in §2 / a short closure doc, mirroring the sibling's
   AC-6 closure.

2. **The sibling's *headline* phase-3 micro-lever is already spent here — measured, not
   inherited.** On `add_rmsnorm_matmul` the phase-3 primary was "split the transposed
   activation limbs directly from the transpose PSUM bank, dropping the intermediate
   fp32 `xT_f` copy" (its D1). That exact transform was **already built and measured in
   *this* task during phase 2** as `runs/matmul_add_rmsnorm_v2_psum_split.py`: a
   **byte-identical compiler no-op** (matmul/Vec/Scl/psum instruction counts all `==`,
   TRUE PE-active 0.7078→0.7079 ms, rel-L2 bit-exact 1.544749e-5, +0.08% latency within
   noise). neuronx-cc already copy-propagates the exact fp32 PSUM→SBUF copy. That
   surface is **closed by this op's own measurement**, so phase 3 does not re-litigate
   it.

3. **The one genuinely-untested lever is a GEMM loop reorder for stationary
   (weight-load) reuse.** The promoted v2 sits at **PE=91.66% (~64 µs idle of the
   772 µs wall)**. In v2 the GEMM is **N-chunk-outer**, so each transposed activation
   limb — `xT_hi[kt]` / `xT_lo[kt]`, which is the *stationary* operand loaded into the
   PE array (see §1.2) — is reloaded **once per N-chunk = 4× per M-tile**. Reordering to
   **K-tile-outer with 4 live PSUM banks**, grouping the matmuls by shared stationary
   limb, lengthens the stationary-reuse run from **2 → 8 consecutive matmuls** and cuts
   stationary loads **128 → 32 per M-tile**. This is the only lever that could touch the
   ~64 µs PE-idle without changing the arithmetic. It is new to this task (the sibling
   never tested it) and is *enabled by this op's larger K* (16 K-tiles, 2× the sibling's
   8), which doubles the accumulation depth over which reuse can be grouped.

**Expected outcome, stated honestly.** The prior is that this reorder is a **compiler
no-op or a small regression**, on two measured precedents: (a) `v2_psum_split` above
(this task) and (b) `bmm`'s phase-3 finding that multi-bank PSUM pipelining is a
compiler no-op *and* enlarging the live PSUM/resident working set **regresses**
monotonically as it constrains the `affine_range` software pipeline
(`cross-batch-blocking-antilever`). Holding 4 live [128,512] accumulators is the same
"enlarged live set" risk. But the reuse *structure* here is different from both (it
changes the stationary-reuse run length, which the compiler cannot manufacture from the
N-chunk-outer source without reordering across the whole chunk loop), and PE=91.66%
leaves a small but real idle to probe. So phase 3 **measures one reorder candidate** and
promotes it only on an out-of-noise win + full-5-seed PASS; otherwise it is recorded as
a floor-confirmation and v2 stays promoted. The realistic ceiling if *all* 64 µs idle
were recovered is ~0.708 ms → **~5.32x**; the honest expectation is at or near v2.

**No precision change is on the table.** The rel-L2 margin is 1.30x, the 4-product v2b
was already a decision-SKIP (offline moves rel-L2 only 4.454e-6→3.491e-6 for ~+25% PE;
sibling v3b MEASURED-REJECT +28%), and plain bf16 fails the gate 117×. Phase 3 optimizes
the *schedule around the fixed 3-product arithmetic*, not the arithmetic (D5 forbidden).

---

## 1. Starting point — the promoted kernel and its profile

### 1.1 Kernel and measured profile

`runs/matmul_add_rmsnorm_v2_bf16_split.py` (PROMOTED, 4.879x, 0.7722 ms, full-5-seed
PASS rel-L2 1.544749e-5). Structure per M-tile (32 tiles), all fp32 I/O:

1. transpose + limb-split of the 16 RAW-`x` K-sub-tiles: per sub-tile — identity
   `nc_matmul(is_transpose)` → `psum_t` fp32; `xT_f = copy(psum_t)`; `xT_hi = bf16(xT_f)`;
   `xT_res = xT_f − xT_hi`; `xT_lo = bf16(xT_res)`;
2. GEMM (N-chunk-outer): `for c in 4: acc=zeros[128,512] psum; for kt in 16: acc += 3
   products (xT_hi@w_hi + xT_hi@w_lo + xT_lo@w_hi)`; then `y[:,chunk] = acc + z_tile`
   (residual add before norm) into a full [128,2048] `y` SBUF buffer;
3. fused fp32 RMSNorm over N (`square` → full-2048 `tensor_reduce(axis=[1])` → two-op
   `mean_eps = sumsq·(1/N)+eps` → `rsqrt` → `inv_rms[128,1]`);
4. output scale full-width: `out = (y·inv_rms)·g_bcast` (2 ops + 1 store).

`w_hi`/`w_lo` (split once at load) are fully resident bf16 (128 KB/part total = same
bytes as v1's one fp32 w). HBM unchanged from v1 (84 MB read / 34 MB write) — limbs
built on-chip.

**Profiler digest (promoted v2, same-session control against v1):**

| metric | v1 fp32 | **v2 bf16x2 (PROMOTED)** | reading |
|---|---|---|---|
| p50 latency | 0.9608 ms | **0.7722 ms** | −19.6% |
| speedup | 3.920x | **4.879x** | |
| **PE %** | 96.19 | **91.66** | v2 slightly idle — the phase-3 surface |
| MFU % | 45.57 | 42.55 | bf16 rate |
| Vec % | 14.88 | 26.62 | limb subtracts (hidden) |
| Scl % | 9.07 | 13.24 | bf16 casts (hidden) |
| DMA % | 19.38 | 23.93 | hidden; HBM flat |
| HBMrd / HBMwr | 84 / 34 MB | **84 / 34 MB** | one-pass floor, IDENTICAL |
| matmul_instruction_count | 4616 | 6664 | 512 transp + 2048·2(fp32) → 512 + 2048·3(bf16) |
| vector_engine_instruction_count | 400 | 566 | limb subtracts |
| scalar_engine_instruction_count | 225 | 246 | bf16 casts |
| **TRUE PE-active/inf** | 0.9242 ms | **0.7078 ms** | −23.4% — the real win (phase 2) |
| psum_read_sbuf_write_count | 132 | 132 | PSUM pressure unchanged |

### 1.2 The PE-idle read (the whole phase-3 argument)

TRUE PE-active is **0.7078 ms** of the **0.7722 ms** wall → **~64 µs PE-idle (8.3%)**.
PE-active is the fixed floor: `2·M·N·K` at the bf16 systolic rate, run 3× for the
3-product compensated split. **Cutting PE-active further needs either fewer products
(fails the gate) or a lower-precision matmul (fails the gate)** — closed. Every non-PE
engine is well under 50% and HBM is at the one-pass floor, so the only latency left to
chase is the **64 µs PE-idle**, and the only precision-neutral way to chase it is to
schedule the *same* matmuls so the PE array stalls less.

**Which operand is the "weight load".** `nc_matmul(stationary, moving) = stationary.T @
moving`, contraction on the partition axis of both. The tile shapes force the roles: the
stationary operand's free dim must be ≤128 and the moving operand's ≤512. Here
`xT_hi[kt]`/`xT_lo[kt]` are [k_in=128, m_in=128] (free=128 → **stationary**, loaded into
the array) and `w_hi[kt,c]`/`w_lo[kt,c]` are [k_in=128, n=512] (free=512 → **moving**,
streamed). So the *activation transpose is the stationary/weight-loaded operand* and the
weight `w` streams. A stationary load costs ~`num_partitions`≈128 array cycles; it is
**skippable when consecutive matmuls reuse the same stationary**. The cost model
(`kernel-cost-analysis`, Formula A) charges Matmul as `dst_free·100/freq` and
does **not** bill the stationary load separately — i.e. it *assumes* the load pipelines
behind the previous matmul's moving stream. Whether that assumption holds at v2's reuse
pattern is exactly what D1 measures.

---

## 2. Shape-lever closure (identical to the siblings)

| Lever | Applies? | Reason (this op's fixed shape M4096 N2048 K2048) |
|---|---|---|
| **Edge / partial tiles** | **No — vacuous** | M=32·128, K=16·128, N=4·512 all divide evenly. No ragged tile, no remainder loop, no mask anywhere. Edge specialization needs an edge; there is none. |
| **Tile-size / partition-free regime** | **No — layout forced** | `nc_matmul` needs k_in on the partition axis of both operands and produces `[m_in(par), n(free)]`; m_in is forced onto the stationary/partition side, n onto the moving/free side. Swapping m↔n would require transposing the N=2048-wide result back — far larger than any tiling gain. Also the RMSNorm reduces over N; keeping N on the free axis makes it a cheap in-partition `tensor_reduce`. Forced. |
| **N-chunk (moving-free) width** | **No — already maximal** | N_CHUNK=512 = `psum_fmax` = one fp32 PSUM bank in the free dim (knowledgebase `6288aaad`: "tile budget is psum_fmax"). Larger exceeds a bank; smaller wastes systolic streaming width. Pinned. |
| **M-blocking** (the `matmul` task's phase-2 win) | **No — vacuous** | That win removed *redundant w HBM reloads* by reusing a loaded w-tile across output-row tiles. Here `w_hi`/`w_lo` are fully resident (128 KB/part, budget ~208 KB) loaded **once** before the M-loop; each `x`/`z` tile is read exactly once (HBMrd=84 MB = the x+w+z one-pass floor). There is no redundant HBM traffic to block for. (D1 below is *not* M-blocking — it does not touch HBM; it reschedules on-chip PSUM/stationary reuse.) |
| **LNC2 / multi-core sharding** | **No — out of contract** | Scored single-core (`--logical-nc-config=1`). Using LNC2 would change the scoring contract, not optimize within it. |

**K note vs the siblings.** This op's K=2048 (16 K-tiles) is 2× `add_rmsnorm_matmul`'s
K=1024 (8 K-tiles) — the only shape difference that could make a *scheduling* lever
behave differently here. It doubles the K-accumulation depth per PSUM bank and doubles
the number of distinct stationary limbs (32 vs 16 per M-tile), which is precisely what
gives D1 (§3) a longer reuse run to group. Everything else closes identically.

---

## 3. Directions enumerated, ranked

### D1 — GEMM loop reorder for stationary (weight-load) reuse  *(PRIMARY; measure)*

**Idea.** v2's GEMM is **N-chunk-outer**: for each of the 4 N-chunks it fully
accumulates all 16 K-tiles' 3 products into one [128,512] PSUM bank before moving to the
next chunk. Consequence: each stationary limb `xT_hi[kt]` is loaded into the array for
chunk 0, and **loaded again** for chunks 1/2/3 — 4 loads per limb per M-tile. Within a
chunk the reuse run is only 2 (P1→P2 share `xT_hi[kt]`, then P3 changes to `xT_lo[kt]`,
then the next kt changes again).

Reorder to **K-tile-outer with 4 live PSUM accumulators**, grouped by stationary limb:

```python
acc = [zeros[128,512] psum  for c in 4]          # 4 live banks, accumulate across all kt
for kt in 16:
    # hi-pass: xT_hi[kt] STATIONARY for 8 consecutive matmuls (P1,P2 over 4 chunks)
    for c in 4:
        acc[c] += xT_hi[kt] @ w_hi[kt,c]
        acc[c] += xT_hi[kt] @ w_lo[kt,c]
    # lo-pass: xT_lo[kt] STATIONARY for 4 consecutive matmuls (P3 over 4 chunks)
    for c in 4:
        acc[c] += xT_lo[kt] @ w_hi[kt,c]
for c in 4: y[:,chunk_c] = acc[c] + z_tile[c]     # residual add, then the fp32 norm epilogue (v2, unchanged)
```

Now `xT_hi[kt]` is stationary across **8** consecutive matmuls and `xT_lo[kt]` across
**4** — reuse runs of 8 and 4 instead of 2. Stationary loads drop from 128 → **32 per
M-tile** (16 kt × 2 limbs). If the array's weight-load is not fully hidden at v2's
reuse pattern, this recovers part of the 64 µs idle.

**PSUM feasibility.** 4 live [128,512] accumulators = 4 banks (512 ≤ 2048 elem/bank) +
1 bank for the [128,128] transpose = **5 of 8 banks**. Fits. `psum_read_sbuf_write_count`
should stay ~132 (same number of evictions; they just happen after the kt loop).

**Correctness (NOT bit-exact — re-gate required).** The 3 products and the RNE
split are unchanged, so the bf16 error is the same *class*. But the fp32 PSUM
accumulation **order** changes (hi-pass P1,P2 for all kt, then lo-pass P3 for all kt, vs
v2's per-kt P1,P2,P3 interleave). fp32 add is non-associative, so rel-L2 will move by
~ulp-level, expected to stay ≈1.5447e-5. **This re-opens the correctness gate** — must
run full-5-seed and confirm rel-L2 ≈ 1.54e-5 and PASS (unlike the sibling's D1 / this
op's v2_psum_split, which were bit-exact). If rel-L2 jumps materially, that is a real
scheduling/aliasing bug to investigate, not a precision tradeoff.

**Expected latency — measure; prior is no-op/small-regress.** Two measured precedents
say the compiler may already extract this and/or the enlarged live set may hurt:
- `v2_psum_split` (this task, phase 2): a source reschedule that the compiler had
  already applied → byte-identical no-op.
- `bmm` phase 3: multi-bank PSUM "issue-before-drain" pipelining was a **compiler
  no-op** (`affine_range` already pipelines the rotating bank), *and* enlarging the live
  resident/PSUM working set **regressed monotonically** (the
  `cross-batch-blocking-antilever` lesson) because it constrains the software pipeline.
  Holding 4 live accumulators across the full 16-tile K-loop is the same enlarged-live-set
  risk — here it may cost more than the stationary-reuse saves.
The reason it is still worth **one** datum: the reuse-run-length change (2→8) is a
structural property of the source loop order the compiler is unlikely to synthesize from
the N-chunk-outer form (it would have to reorder across the entire chunk loop), and this
op's 16 K-tiles give the longest reuse run in the family. Realistic expectation: within
noise of v2, or a small regress; optimistic ceiling ~5.0–5.3x if idle is genuinely
weight-load-bound. **Promote iff full-5-seed PASS AND p50 beats v2 out-of-noise (>1.8%
band).** Otherwise record the floor-confirmation; v2 stays promoted.

**Risk:** low-moderate. No new primitive, no dtype/algebra change; only the loop
structure and PSUM-bank count change. The one real risk (regression from 4 live banks)
is itself an informative measured datum.

### D2 — 2-bank / 2-chunk grouping variant  *(secondary; only if D1's profile points here)*

If D1 shows the 4-live-bank version regressed *specifically* on PSUM/pipeline pressure
(PE-idle up, not down) rather than the reorder being a pure no-op, try the intermediate:
group by stationary over **2 chunks at a time** (2 live banks + transpose = 3 banks),
i.e. an outer 2-iteration chunk-pair loop with kt-outer inside. This keeps a reuse run of
4 (hi-pass over 2 chunks) while halving the live PSUM set — testing whether the
regression is the enlarged live set (D2 recovers) or the reorder itself (D2 also no-op).
**Contingent, not proactive** — build only if D1's digest shows a specific PSUM/pipeline
bubble; otherwise skip and record why (mirrors the sibling's contingent-D2 discipline).

### D3 — PSUM-source activation-limb split  *(CLOSED — already measured in phase 2)*

The sibling's phase-3 primary (split `xT_hi`/`xT_lo` directly from the transpose PSUM
bank, dropping the fp32 `xT_f` copy). **Already built and measured here as
`runs/matmul_add_rmsnorm_v2_psum_split.py`** during phase 2: byte-identical compiler
no-op (all instruction counts `==`, TRUE PE-active 0.7078→0.7079 ms, rel-L2 bit-exact,
+0.08% within noise). neuronx-cc copy-propagates the exact fp32 PSUM→SBUF copy. **Do not
rebuild** — cite the phase-2 datum.

### D4 — off-PE transpose to remove the 512 transpose matmuls  *(CLOSED — record-only)*

Both siblings closed this: SBUF→SBUF `dma_transpose` of a [128,128] tile is infeasible
(hwdge needs `src.shape[0]==16`, swdge needs an HBM source — shape/memory block, not just
dtype), and `nc_transpose`(vector) lands in fp32 PSUM needing a re-cast and measured a
+2% regress. The 512 identity-matmul transposes (~27 µs of PE-active) are already hidden
under the PE-bound matmul. **Do not explore.**

### D5 — precision / product-count changes  *(FORBIDDEN this phase)*

3-product bf16 is pinned: margin 1.30x, v2b 4-product was a decision-SKIP (sibling v3b
MEASURED-REJECT +28% for a ~1.6% accuracy move swamped by the fp32 floor), plain bf16
fails 117×. Every phase-3 candidate keeps the exact 3-product arithmetic.

### D6 — split-before-transpose (wide limb ops)  *(CLOSED — record-only)*

Splitting `x` into limbs *before* transpose (3 wide [128,2048] ops instead of granular
per-sub-tile ops) **doubles the transpose PE work** (transpose `x_hi` and `x_lo`
separately → 32 transpose matmuls/M-tile instead of 16). Adds PE-active — the one thing
we cannot afford. Phase 2 already fixed "split *after* the transpose costs one transpose,
not two." **Do not implement.**

---

## 4. Execution plan (≤2 candidates; measure-first)

1. **Re-anchor (same-session control).** Re-measure v2 full-5-seed via
   `runs/dump_metrics.py` to pin this session's PE-active / PE% / idle and latency
   anchor (profiler jitter ~±1.8%). Confirm PE≈91.7%, HBM 84/34 MB, rel-L2 1.5447e-5.
2. **v3 (D1, stationary-reuse reorder):** `runs/matmul_add_rmsnorm_v3_stationary_reorder.py`,
   forked from v2. Change *only* the GEMM loop nest (K-tile-outer, 4 live PSUM banks,
   hi-pass/lo-pass grouping); epilogue (residual add + fp32 RMSNorm + full-width output
   scale) byte-for-byte v2. `--fast` PASS gate first, then **full 5-seed** twice for
   stability + the same-session v2 anchor. Capture the profiler digest (watch PE% and
   TRUE PE-active, and `psum_read_sbuf_write_count`).
   - **Promote iff** full-5-seed PASS AND p50 beats v2 out-of-noise (>1.8%).
   - **Otherwise** record the within-noise floor-confirmation (or the regression as a
     first-class negative datum, per `bmm`'s anti-lever discipline); v2 stays promoted.
3. **v4 (D2) only if D1's profile shows a specific PSUM/pipeline bubble.** Contingent,
   same gates. If D1 is a clean no-op, skip D2 and record why (proactive D2 forbidden).
4. **Close-out:** whichever of v2/v3(/v4) is fastest is the phase-3 (and task) result.
   Report speedup vs the 3.768493 ms baseline on the **full** 5-seed correctness gate.

## 5. Evidence to record

- `benchmark.csv`: one row per perf-relevant candidate (v3, and v4 if run), plus the
  D3(closed-in-phase-2)/D4/D5/D6 decisions as record-only notes.
- `candidates.jsonl`: DAG node v3→v2 (and v4→v3 if run) with metrics, `rel_l2`,
  `per_seed_rel_l2`, the non-bit-exact-reorder note, and the `per_seed_latency_ms=null`
  / `latency_scope` caveat carried from v2.
- `profile/`: v3 digest with the PE-idle before/after read (TRUE PE-active,
  `psum_read_sbuf_write_count`), and a short shape-closure note (or a reference to §2 /
  the sibling's closure doc) for AC-6.

## 6. Correctness invariants (never regress)

- fp32 residual add `y = x@w + z` **before** the norm; fp32 RMSNorm reduction over N
  (`square` / full-2048 `tensor_reduce(axis=[1])` / `mean_eps` / `rsqrt`); **eps added
  AFTER the `/N` mean**, matching the reference `np.mean(y**2,axis=-1)+eps`.
- `g` is on the OUTPUT free axis (N) → `[1,N]→[128,N]` broadcast multiply applied
  **after** the norm; **never folded into w** (folding would scale y before the norm and
  break `rms=sqrt(mean(y^2))`). `inv_rms` is **not** commuted out (norm reduces over N →
  the full [128,N] row must be assembled before `inv_rms` is known).
- The 3-product bf16 split and its PINNED split order (`w`→`w_hi`→`w_res`→`w_lo`;
  `xT`→`xT_hi`→`xT_res`→`xT_lo`; products `hi@hi + hi@lo + lo@hi`, drop `lo@lo`) are
  unchanged. D1 changes only the *order* of fp32 PSUM accumulation, so rel-L2 must stay
  ≈1.5447e-5 (not bit-exact; re-gate full-5-seed).
- Raw-2D I/O + exact signature `kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor)`;
  full-width [128,N] output store.
- Every promotion gated on **full 5-seed** `l2_norm_passed`, not `--fast` alone.
- **CAVEAT (carried from phase 2):** the adapter fixes seed 42 for all 5 profiler seeds,
  so on-device 5-seed PASS is a determinism/stability gate, weak on *input* diversity;
  the offline 7-draw sim (worst bf16-only 4.454e-6) covers input diversity. `v1`
  (3.920x, pure fp32) is retained as the guaranteed-correct fp32 fallback.

--- Original Design Draft End ---
