# rmsnorm_matmul (M4096 N2048 K1024, fp32) — Phase 3 Plan: Shape-Specialization Closure + Gated Precision-Split Attempt

## Goal Description

Close out Phase 3 of the `rmsnorm_matmul` kernel (fixed shape M=4096, N=2048, K=1024,
fp32; `out = rmsnorm(x) @ w`, RMSNorm over the K axis) as **regime / shape
specialization**: analyze where time goes across the tensor's structure and specialize
*only* where a measured win justifies the added complexity.

The starting point is the promoted kernel `runs/rmsnorm_matmul_v1.py` (**1.066x** over the
0.502647 ms baseline, full 5-seed PASS). Phase 2 already established that v1 sits at the
**fp32 systolic PE floor**: the only non-floor PE work (the transpose of `x` sub-tiles) is
*fully hidden* under the PE-bound matmul (the experimental `load_transpose2d` route removed
the transpose from the PE entirely and still landed within the ~±1.8% noise band; PE stayed
pinned at 97%). The measured MFU of 46% equals the bf16-peak floor (218.5 µs) divided by the
measured latency (471.6 µs) — i.e. MFU is *literally* the fp32 rate ceiling, not schedulable
inefficiency.

Phase 3 therefore has two jobs, and neither may ever regress correctness (the NKIBench
relative-L2 gate `||v_k − v_r||₂ < 2e-5 · ||v_r||₂` on seeds `[0, 21, 42, 63, 84]`, fp32,
enforced by `verify.py` gating on `l2_norm_passed`) or latency below v1:

1. **Confirm there is no shape-structure lever left** that Phase 2 did not already cover
   (edge/partial tiles, tile-size/partition-free regimes, N-chunk width, M-blocking, LNC2
   sharding) — recording the *absence* of a specialization surface as a first-class finding,
   not a gap.
2. **Run the two remaining swings**, each strictly gated:
   - **P1 (primary, cheap, correctness-safe):** reorder the main matmul so each stationary
     transposed-activation tile `xT[kt]` is loaded once and streamed against all N-chunks —
     the one main-matmul micro-lever Phase 2 never touched. Most-likely a within-noise
     confirmation that stationary fills are already hidden; done because it is cheap and
     numerically equivalent, and closes the last structural question by *measurement*.
   - **P3 (stretch, gated):** a compensated bf16×2 split-matmul — the only lever that can
     move the fp32 rate ceiling — attempted *only after* an offline numpy pre-check shows a
     comfortable correctness margin, and promoted *only* on a full-5-seed PASS plus an
     out-of-noise latency win.

The **most-likely and fully-acceptable outcome is a floor-confirmation exit**: v1 stays
promoted at 1.066x, now backed by measured evidence that (a) the shape has no edge/tile
specialization surface, (b) the main-matmul stationary fills are already hidden, and (c) the
fp32 rate is the hard ceiling that the 2e-5 gate forbids breaking. This mirrors the sibling
`matmul` task's phase-3 result, reached by measurement rather than assumption.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. Correctness is always the NKIBench relative-L2 gate via
`verify.py` (which gates on `l2_norm_passed`); "noise band" refers to the ~±1.8% run-to-run
latency variance observed on this remote profiler. The scoring command (run from inside
`workspaces/rmsnorm_matmul/`) is:

```
python3 ../../verify.py \
    --op rmsnorm_matmul --candidate runs/<file>.py [--fast]
```

- AC-1: **Correctness never regresses (HARD gate).** No candidate is promoted unless it
  passes the relative-L2 gate on **all 5 seeds** `[0, 21, 42, 63, 84]`.
  - Positive Tests (expected to PASS):
    - The final promoted kernel reports `correct: 5/5` on a full (non-`--fast`) `verify.py` run.
    - If no candidate beats v1 out-of-noise, v1 remains promoted and still reports `correct: 5/5`.
  - Negative Tests (expected to FAIL / be rejected):
    - Any candidate with even one failing seed is rejected and never promoted.
    - A candidate promoted on `--fast` (1 seed) evidence alone, without a full-5-seed confirmation, is rejected.

- AC-2: **fp32 numerical fidelity preserved on every non-P3 path (HARD).** The v1/P1 fp32
  compute path keeps `nc_matmul` accumulation in **fp32 PSUM** and the same per-output-element
  K-accumulation order; the RMSNorm reduction stays a single full-1024 free-axis
  `tensor_reduce` with `inv_rms = rsqrt(sumsq · 1/K)` (1/K folded into the rsqrt scale).
  - Positive Tests (expected to PASS):
    - P1's output is numerically equivalent to the `v2_postscale` base (same fp32 PSUM
      accumulation per output element; only the N-chunk loop moves inside the K loop), and
      passes the full-5-seed gate.
    - The reduction remains one full-K `tensor_reduce`; no split/partial-K reduction is introduced.
  - Negative Tests (expected to FAIL / be rejected):
    - Any v1/P1-path change that lowers matmul accumulation below fp32 (e.g. bf16/tf32 PSUM) is rejected on this path (bf16 mantissa ~4e-3 ≫ 2e-5; closed by D4).
    - Reordering that changes the per-output-element K-accumulation set (not just interleaving) such that the full-5-seed result drifts outside the gate is rejected.

- AC-3: **P1 = stationary-activation reuse reorder, measured not assumed.** P1 reorders the
  main matmul to `kt`-outer / `c`-inner with up to 4 live `[128,512]` fp32 PSUM accumulators
  so each stationary `xT[kt]` tile is loaded **once** and streamed against all 4 N-chunks.
  Its purpose is to *measure* whether any stationary fill is exposed on the main matmul.
  - AC-3.1: **P1 correctness.** P1 (built on the `v2_postscale` eviction-fold base) passes the full-5-seed gate.
    - Positive: `verify.py` full run reports `correct: 5/5` for the P1 kernel.
    - Negative: a P1 variant that spills SBUF/PSUM or serializes on PSUM-bank dependencies such that a seed fails is rejected.
  - AC-3.2: **P1 promotion requires an out-of-noise win with profiler evidence.** P1 is
    promoted over v1 only if a full-5-seed run beats a **fresh same-session v1 control** by
    more than the noise band **and** the profiler digest shows a consistent PE/MFU/latency
    change (evidence that exposed fill actually left the PE).
    - Positive: P1 full-5-seed latency beats the same-session v1 control by > the noise-band margin (see DEC-1), with PE%/MFU moving consistently → promote P1.
    - Negative (expected, most likely): P1 lands within the noise band with PE% unchanged at ~97% → **not** promoted; recorded as a valid closing datum that stationary fills are already hidden (consistent with the Phase-2 `load_transpose2d` result). Promoting such a within-noise "win" is rejected.

- AC-4: **P3 is offline-gated before any remote spend (process gate).** Before spending a
  remote P3 iteration, an **offline numpy simulation** of the bf16×2 split — seeded with
  `np.random.seed(42)` to reproduce the exact input the harness scores (the adapter fixes the
  input seed to 42 for all profiler seeds; see Feasibility Hints) — must compute the
  NKIBench-style relative-L2 of the idealized split against the fp32 reference and report the
  **actual worst rel-L2 number**.
  - AC-4.1: **Offline simulator faithfulness.** The offline sim reproduces the fp32 reference
    (v1 path) rel-L2 within expected tolerance *before* the bf16×2 result is trusted, and
    matches the harness's input draw (seed 42, draw order, dtypes), the RMSNorm formula
    (divide-by-K-inside-rms then normalize, then matmul), and the exact P3 limb construction.
    - Positive: an fp32-vs-fp32 offline control reproduces a rel-L2 near the ~1.3e-7 role seen for the v1 reduction; the bf16×2 sim uses `x_hi = bf16(x)`, `x_lo = bf16(x − x_hi)` (round-to-nearest-even) and the 3-product sum, dropping `x_lo·w_lo`.
    - Negative: an offline simulator whose fp32 control does not reproduce the reference (indicating a formula/seed/dtype mismatch) is not trusted for the bf16×2 decision.
  - AC-4.2: **Offline result routes the remote spend (one-way, with margin).** The offline
    number is treated as a **practical no-spend gate**, not a formal impossibility proof
    (hardware rounding could occasionally cancel error). If the offline idealized bf16×2 worst
    rel-L2 is **at or above** the 2e-5 gate, **or only marginally below it** (see DEC-2 for the
    numeric "comfortable" threshold), **do not** spend a remote P3 run — record it as the
    precision-floor datum. Only a **comfortably-below** offline margin authorizes a remote P3 attempt.
    - Positive: offline worst rel-L2 comfortably below the threshold → proceed to a remote P3 attempt (AC-5).
    - Negative (expected likely): offline worst rel-L2 ≥ 2e-5 or only marginally below → **no** remote P3 run; recorded as the definitive "fp32 precision floor cannot be beaten within the 2e-5 gate" datum (upgrading D4's single-precision-swap calibration to a modeled attempt).

- AC-5: **P3 promotion requires a measured full-5-seed PASS AND an out-of-noise latency win
  (HARD).** Even with a comfortable offline margin, the offline number is *evidence, not
  certification*: P3 is promoted only if a remote **full-5-seed** run PASSES the gate *and*
  beats the fresh same-session v1 control by more than the noise band.
  - Positive: remote P3 full-5-seed `correct: 5/5` **and** latency > noise-band faster than the same-session v1 control → promote P3 (the only path that breaks the fp32 floor).
  - Negative: any seed fails, or the latency gain is within noise / consumed by split+cast+extra-residency overhead → reject P3, record as the precision-floor confirmation, keep v1 promoted.

- AC-6: **Shape-specialization space is explicitly closed (documentation gate).** The plan's
  analysis that the fixed shape offers no edge/partial-tile, no tile-size/partition-free, no
  N-chunk-width, no M-blocking, and no LNC2 lever is recorded as an evidenced finding.
  - Positive: `docs/` / evidence records state, per lever, *why* it is vacuous or already-at-constraint (even divisibility → no ragged tiles; `nc_matmul` forces k-on-partition and `[m_in(par), n(free)]` output; N_CHUNK=512=psum_fmax already maximal; w already fully resident so M-blocking is vacuous; scored single-core `--logical-nc-config=1`).
  - Negative: asserting a lever is "not applicable" without the shape/constraint reason is rejected.

- AC-7: **Every perf-relevant candidate and every gated decision is recorded as evidence
  (process gate).** Each scored candidate appends a row to `benchmark.csv` and a node to
  `candidates.jsonl` (with parent links forming a DAG); the offline P3 sim result and the
  floor-confirmation exit (if taken) are recorded with their numbers.
  - Positive: `benchmark.csv` + `candidates.jsonl` contain the fresh v1 control, P1, the offline-sim datum, and any remote P3 attempt, each with latency and (where applicable) rel-L2 and profiler digest.
  - Negative: a promotion or reject with no corresponding evidence row/node is rejected.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
The full planned exploration executed and recorded: (1) a fresh same-session v1 full-5-seed
control anchors the noise band; (2) P1 (kt-outer / 4-live-PSUM-accumulator stationary-reuse
reorder on the `v2_postscale` base) is implemented, scored full-5-seed against the control,
with before/after PE%/MFU/Vec%/Scl%/DMA% deltas captured; (3) the offline numpy bf16×2
simulator is built, validated against the fp32 reference, and run (optionally across extra
synthetic seeds for robustness) to produce the worst rel-L2; (4) if and only if the offline
margin is comfortable, one remote P3 full-5-seed attempt is scored; (5) the best correct
kernel is kept and, if nothing beats v1 out-of-noise, the floor-confirmation exit is written
with the full lever-closure analysis. No speculative rewrites beyond P1 and the gated P3.

### Lower Bound (Minimum Acceptable Scope)
The fresh v1 control is run, P1 is implemented and scored full-5-seed against it, and the
offline P3 pre-check is run. If P1 is within noise and the offline P3 margin is not
comfortable (the expected case), **v1 remains promoted at 1.066x** and Phase 3 exits as a
documented specialization-space closure (AC-6) with the fp32-floor and hidden-fill evidence
recorded (AC-7). A remote P3 run is *not* required when the offline gate forbids it.

### Allowed Choices
- Can use: the `v2_postscale` eviction-fold kernel as the P1/P3 base while keeping v1 as the
  promoted fallback and speedup reference; `nl.affine_range` loops; up to 4 live `[128,512]`
  fp32 PSUM banks for P1; proven primitives (`nc_matmul`, PSUM accumulate, `tensor_scalar`,
  `tensor_reduce`, `activation`); bf16 casts (`nl.copy`/activation with `dtype=bf16`) and
  bf16 `nc_matmul` operands for P3 (the native path; already compiled in D4); `tensor_tensor`
  for the split subtraction; system `python3` (numpy 2.0.2) with pure-numpy bf16
  round-to-nearest-even for the offline sim; `--fast` for triage and **full 5-seed before any
  promotion**.
- Cannot use: lowering the v1/P1 fp32 matmul accumulation below fp32 (closed by D4); plain
  bf16/tf32 matmul as a promotion path (P4, rejected — mantissa ~4e-3 ≫ 2e-5); the
  `dma_transpose` route (D2, closed: fp32-ineligible / crashes on this remote);
  `nc_transpose(engine=vector)` as a promotion (D3, closed: available but regresses +2.08%);
  promoting on `--fast` alone; promoting a within-noise result; multi-core / LNC2 sharding
  (out of contract, scored single-core); editing anything under `../../AccelOpt/NKIBench/`.

> **Note on Deterministic Designs**: This plan is a small, bounded set of gated directions
> (P1 primary, P3 offline-then-remote gated, P2 contingency-only), not a single deterministic
> design. The upper and lower bounds differ mainly in whether a remote P3 run happens — and
> that is decided *by measurement* (the offline gate), not by choice. Within each direction
> the mechanics are essentially fixed by the hardware constraints described in the draft.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

**Shape-lever closure (AC-6) — why every classic phase-3 lever is vacuous or already-maxed:**
- Edge/partial tiles: **none** — M=4096=32·128, K=1024=8·128, N=2048=4·512 all divide evenly;
  no masks, no remainder loop, nothing to specialize.
- Tile-size / partition-free regime: **forced** — `nc_matmul` needs the contraction dim
  (k_in) on the partition axis; the output tile is `[m_in(par), n(free)]`. m/n are not
  swappable without transposing the whole result back.
- N-chunk width: **already maximal** — N_CHUNK=512 = `psum_fmax` = one fp32 PSUM bank
  (precedent `6288aaad`: "tile budget is psum_fmax, not pmax").
- M-blocking (the sibling matmul's phase-2 win): **vacuous** — that win removed redundant `w`
  HBM reloads; here w is already fully resident (8 MB, loaded once) and DMA is only ~20% busy.
- LNC2 sharding: **out of contract** — scored single-core (`--logical-nc-config=1`).

**P1 — stationary-activation reuse reorder (AC-3).** v1/`v2_postscale` runs the matmul
`c`-outer (4 N-chunks) / `kt`-inner (8 K-tiles), so the **stationary operand `xT[kt]`** (the
transposed *activation*; `w` is the *moving* operand) is re-presented on every one of the
1024 `nc_matmul` calls. Reorder to `kt`-outer / `c`-inner with 4 live PSUM accumulators so
each `xT[kt]` is loaded once and streamed against all 4 N-chunks (256 stationary fills instead
of 1024):
```
acc[0..3] = zeros[128,512]          # 4 live fp32 PSUM banks (4 of 8 — fits)
for kt in 8:
    xt = xT[kt]                     # stationary loaded ONCE per kt
    for c in 4:
        acc[c] += nc_matmul(stationary=xt, moving=w_sb[kt, c-slice])
# evict each acc[c] with the per-row inv_rms tensor_scalar (v2_postscale fold), then store
```
This is numerically equivalent to the base (same fp32 PSUM K-accumulation per output element;
only the N loop moves inside). **Most-likely within-noise:** `affine_range` already lets the
scheduler pipeline fills behind the previous stream, and Phase-2's `load_transpose2d` showed
PE-side transpose work is fully hidden — so the open question "does `nc_matmul` keep the
stationary resident across consecutive same-stationary calls, or re-fill each call?" is
answered *empirically* by the PE%/latency delta, not assumed. Watch for SBUF/PSUM spills or
PSUM-bank serialization. Theoretical ceiling ~8.7% *only if* fills were fully exposed.

**P3 — offline-gated compensated bf16×2 split-matmul (AC-4/AC-5).**
- Split each fp32 operand into two bf16 limbs: `x_hi = bf16(x)`, `x_lo = bf16(x − x_hi)`; same
  for w. Accumulate 3 bf16 products in fp32 PSUM: `x_hi·w_hi + x_hi·w_lo + x_lo·w_hi` (drop
  `x_lo·w_lo`). Idealized effective mantissa ≈ 16 bits → ~1.5e-5, *just under* the 2e-5 gate —
  which is exactly why it must be measured, not assumed.
- **Offline pre-check first (zero remote spend).** The adapter fixes the input seed to 42 for
  all 5 profiler seeds (`adapter/nkibench_case.py`: `np.random.seed(42)` before both the
  kernel and reference draws; explicit comment that multi-seed runs reuse the same inputs).
  So a pure-numpy bf16×2 sim seeded with 42 reproduces the *exact* input the remote gate
  scores. Build the fp32 reference (RMSNorm-then-matmul per the numpy reference), the idealized
  bf16×2 result, compute NKIBench relative-L2, and report the worst number. Because idealized
  numpy RNE + exact fp32 accumulate is *at least as good as* hardware bf16, an offline result
  at/above (or marginally below) 2e-5 means hardware almost-certainly fails — but this is a
  practical no-spend gate, **not** a proof (hardware rounding can occasionally cancel error),
  so require a *comfortable* margin (DEC-2) before spending remote, and still require a remote
  full-5-seed PASS to promote. Account for: dropped `x_lo·w_lo`, limb-construction rounding,
  K=1024 summation, RMSNorm error stacking, and output-relative amplification on small-norm rows.

**P2 (contingency only).** Fold the RMSNorm scale onto w-load rather than per-tile — run *only*
if P1 surfaces an unexpected exposed Vector/Scalar bubble. Default: skip (the norm is
measured-hidden: Vec 15%/Scl 11%, and `v2_postscale` already moved the scale to eviction).

**Promotion discipline.** Always compare to a **fresh same-session** v1 full-5-seed control
(remote drift is real; historical 0.4716 vs same-session 0.4715 ≈ <0.03%). Promote only on a
full-5-seed out-of-noise win. A clean within-noise read is a *valid closing result*.

### Relevant References
- `runs/rmsnorm_matmul_v1.py` — promoted kernel (1.066x); the fp32 floor and the fallback.
- `runs/rmsnorm_matmul_v2_postscale.py` — eviction-fold enabler; the base P1/P3 build on.
- `runs/rmsnorm_matmul_probe_bf16_calib.py` — D4 bf16 main-matmul calibration (fp32/bf16 ≈ 3.23x; the fp32-floor datum) and a structural template for the P3 bf16 operand casts.
- `runs/rmsnorm_matmul_probe_loadtranspose.py` — D2b; decisive "transpose fully hidden" datum.
- `../../AccelOpt/NKIBench/reference/rmsnorm_matmul_M4096_N2048_K1024_numpy_1.py` — the numpy reference the offline P3 sim must reproduce (RMSNorm divides by K inside rms, then normalize, then matmul).
- `adapter/nkibench_case.py` — the seed-42-fixed input assembly (`DEFAULT_INPUT_SEED=42`, `NKIBENCH_SEEDS=[0,21,42,63,84]`); the basis for the faithful offline sim.
- `../../verify.py` — the scoring harness (`--fast` = 1 seed/20 iters; full = 5 seeds/100 iters; gates on `l2_norm_passed`, `rtol=2e-5`).
- `../matmul/docs/plan-phase2.md`, `../matmul/benchmark.csv` — the sibling task's floor-confirmation precedent (its micro-lever landed <2.5%, below noise; B=4 stayed promoted).
- `.claude/skills/kernel-cost-analysis`, `kernel-optimization-kb` — theoretical floor and precedents (e.g. `6288aaad` PSUM-bank / psum_fmax sizing).

## Dependencies and Sequence

### Milestones
1. **Re-anchor + close the shape-lever question.**
   - Phase A: Run a fresh same-session v1 full-5-seed control; record latency + profiler digest to `benchmark.csv`/`candidates.jsonl` (anchors the noise band for all comparisons).
   - Phase B: Record the shape-lever closure analysis (AC-6) — each lever vacuous or already-at-constraint, with the reason.
2. **P1 — measure the main-matmul stationary-fill lever.**
   - Step 1: Implement the `kt`-outer / 4-live-PSUM stationary-reuse reorder on the `v2_postscale` base (keep the eviction-fold inv_rms scale).
   - Step 2: `--fast` triage; if it beats the control by > the noise band, run full 5-seed. Capture PE%/MFU/Vec%/Scl%/DMA% before/after.
   - Step 3: Promote only on an out-of-noise full-5-seed win with consistent profiler evidence; otherwise record the within-noise closing datum. (P2 runs here only if a norm bubble appears.)
   - Depends on Milestone 1 (control anchors the comparison).
3. **P3 — offline-gated precision-split attempt.**
   - Step 1: Build the offline numpy bf16×2 simulator; validate its fp32 control reproduces the reference.
   - Step 2: Run it (seed 42; optionally extra synthetic seeds) → worst rel-L2.
   - Step 3: Apply the no-spend gate — comfortably-below → one remote full-5-seed P3 attempt; at/above/marginal → record the precision-floor datum and stop.
   - Step 4: Promote a remote P3 only on full-5-seed PASS + out-of-noise win; else keep the fallback.
   - Depends on Milestone 1 (control); independent of P1's outcome (P3 can proceed regardless of whether P1 won).
4. **Keep the best correct kernel + exit.**
   - Keep whichever correct kernel is fastest out-of-noise; if none beats v1, **v1 stays promoted (1.066x)** and the floor-confirmation exit is written (shape closure + hidden fills + fp32 ceiling), with all evidence recorded.
   - Depends on Milestones 2 and 3.

<Each direction gets ≤5 iterations; stop a direction early on a clear reject or a clean
within-noise read. Never promote a candidate that fails any seed or regresses below v1.>

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Run a fresh same-session v1 full-5-seed control; record latency + profiler digest to `benchmark.csv`/`candidates.jsonl` to anchor the noise band | AC-1, AC-7 | coding | - |
| task2 | Record the shape-specialization closure analysis (edge tiles / partition-free / N-chunk / M-blocking / LNC2), each with its shape-or-constraint reason | AC-6 | coding | - |
| task3 | Implement P1: reorder main matmul to `kt`-outer / `c`-inner with 4 live `[128,512]` fp32 PSUM accumulators on the `v2_postscale` base; keep eviction-fold `inv_rms` | AC-2, AC-3, AC-3.1 | coding | task1 |
| task4 | Score P1: `--fast` triage, then full 5-seed if > noise band; capture PE%/MFU/Vec%/Scl%/DMA% before/after; record DAG node | AC-3.2, AC-7 | coding | task3 |
| task5 | (Codex) Adversarially review P1's promotion case: is any latency delta out-of-noise, and is the profiler evidence (PE%/MFU shift) consistent with real fill removal vs scheduling noise? | AC-3.2 | analyze | task4 |
| task6 | Build the offline numpy bf16×2 simulator (seed 42, reference RMSNorm-then-matmul, `x_hi=bf16(x)`/`x_lo=bf16(x-x_hi)` RNE, 3-product sum dropping `x_lo·w_lo`); validate its fp32 control reproduces the reference rel-L2 | AC-4, AC-4.1 | coding | - |
| task7 | Run the offline sim (seed 42; optionally extra synthetic seeds) → worst NKIBench relative-L2; record the number | AC-4, AC-7 | coding | task6 |
| task8 | (Codex) Review the offline sim's faithfulness and the no-spend decision: does the fp32 control validate the model, is the margin comfortable per the DEC-2 threshold, and are all error sources accounted? | AC-4.2 | analyze | task7 |
| task9 | If (and only if) the offline margin is comfortably below the gate, implement + score one remote P3 bf16×2 full-5-seed attempt (correctness read first, then latency vs control); record DAG node | AC-1, AC-5, AC-7 | coding | task8 |
| task10 | (Contingency) If P1 surfaces an unexpected exposed Vec/Scl bubble, implement + score P2 (fold RMSNorm scale onto w-load); else record P2 as skipped-by-measurement | AC-2, AC-7 | coding | task4 |
| task11 | Decide & record the exit: keep the best correct out-of-noise kernel; if none beats v1, write the floor-confirmation closure (shape closure + hidden fills + fp32 ceiling) and keep v1 promoted; update `benchmark.csv`/`candidates.jsonl`/`profile/` | AC-1, AC-6, AC-7 | coding | task4, task5, task9 |

## Claude-Codex Deliberation

### Agreements
- P1's framing is technically coherent: the stationary operand is `xT` (the transposed
  activation), `w` is the moving operand, the expected win is empirical (measure, don't
  assume), and "numerically equivalent" is the correct claim rather than "bit-identical".
- Comparing against a **fresh same-session v1 control** and gating promotion on an out-of-noise
  full-5-seed win is the right discipline; a within-noise read is a valid closing result.
- Given PE=97%, the Phase-2 `load_transpose2d` within-noise result, and the fp32 structural
  penalty, **no obvious shape lever appears to be missed** — floor-confirmation is a valid,
  defensible endpoint.
- P3's naive error model (~16-bit effective mantissa, ~1.5e-5 vs 2e-5) is razor-thin and
  optimistic; it must be validated, and an offline numpy pre-check is the right cheap gate
  before spending remote iterations. P3 must report the worst rel-L2 *number*, not pass/fail.

### Resolved Disagreements
- **P1 dataflow naming (Codex first pass → resolved):** the draft called P1 "weight-fill"
  amortization, but `w` is the moving operand. Resolved: relabel as **stationary-activation
  (`xT`) fill reuse across N-chunks**; the mechanics and the 1024→256 stationary-fill count are
  unchanged, only the label was wrong.
- **P1 "bit-identical" claim (Codex first pass → resolved):** too strong given `affine_range`
  reschedule freedom. Resolved: downgrade to **"numerically equivalent (same per-output-element
  fp32 K-accumulation), re-verified full-5-seed."**
- **Lineage ambiguity v1 vs v2_postscale (Codex first pass → resolved):** the draft said "start
  from promoted v1" but the P1 mechanics build on the eviction-fold. Resolved: **P1/P3 build on
  `v2_postscale`** (the proven within-noise enabler); **v1 stays the promoted fallback and the
  1.066x speedup reference.**
- **Offline P3 gate as a "certain" kill criterion (Codex second pass → resolved):** "idealized
  offline ≥ 2e-5 ⇒ hardware certainly fails" is too strong because hardware rounding can
  occasionally cancel error. Resolved: the offline sim is a **practical no-spend gate with a
  margin** (don't spend if at/above **or marginally below** the gate), **not** an impossibility
  proof; a comfortable offline pass still **requires** a remote full-5-seed PASS to promote.
  The numeric "comfortable" threshold is carried as DEC-2.

### Convergence Status
- Final Status: `converged` (2 convergence rounds; Codex second pass returned `AGREE` on P1
  framing and the promotion gate, with one `REQUIRED_CHANGES` — the offline-gate reframing —
  which is incorporated above. Remaining `UNRESOLVED` items are empirical-by-design, not
  user-arbitration points: whether `nc_matmul` preserves stationary residency is exactly what
  P1 measures, and the offline-vs-hardware rounding caveat is now baked into AC-4.2.)

## Pending User Decisions

- DEC-1: **P1/P3 promotion noise-gate margin.** How much must a candidate beat the fresh
  same-session v1 control (on a full-5-seed run) to be promoted?
  - Claude Position: `> 1.8%` (one noise band), the draft's stated band, with same-session v1
    bracketing to reduce drift risk; grey-zone results (1.8–2.5%) require a confirmatory re-run.
  - Codex Position: Consider a stricter `> 2.5–3%` given the sibling task's ~1.8% observed noise
    and remote profiler variance, to avoid promoting drift as a win.
  - Tradeoff Summary: A stricter gate reduces false promotions but could reject a genuine ~2%
    P1 win (within the plausible outcome range); a looser gate risks promoting noise. This
    inherits Phase-2's DEC-2 (left `PENDING` there); given the phase-2 outcome was
    floor-confirmation, it never bound. Recommendation: `> 1.8%` with a confirmatory re-run in
    the 1.8–2.5% grey zone.
  - Decision Status: `PENDING`

- DEC-2: **Numeric "comfortably below 2e-5" threshold for the offline P3 no-spend gate.** Below
  what offline idealized-bf16×2 worst rel-L2 is a remote P3 attempt authorized?
  - Claude Position: The offline sim is faithful to the exact scored input (seed 42), so a
    modest margin suffices — e.g. authorize a remote attempt only if offline worst rel-L2
    `≤ ~1.5e-5` (the draft's own estimate), and treat `1.5e-5–2e-5` as "marginal → no spend".
  - Codex Position: Because the offline model is optimistic and hardware rounding can move the
    score either way, require a clear margin — define "comfortable" as `≤ ~1.5e-5` (or stricter)
    *before* implementation, and do not spend remote runs on a marginal or at/above result.
  - Tradeoff Summary: A tighter threshold (e.g. `≤ 1e-5`) almost never authorizes a remote P3
    run (likely correct, since bf16×2's idealized floor already sits near the gate) but risks
    skipping a genuine ~1.3e-5 pass; a looser threshold (`< 2e-5`) risks spending a remote run
    that then fails on hardware rounding. Claude and Codex agree on `≤ ~1.5e-5` as the default;
    the exact cutoff is the user's call. Recommendation: `≤ 1.5e-5` authorizes a remote attempt;
    `> 1.5e-5` records the precision-floor datum with no remote spend.
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Phase", "Step", "P1/P2/P3/P4", "D1/D2/D3/D4", "DEC-", or similar workflow/plan markers.
- These terms are for plan documentation only, not for the resulting kernel source or the
  offline simulator script.
- Use descriptive, domain-appropriate naming in code (e.g. `inv_rms`, `xT`, `stationary_reuse`,
  `bf16_split`, `x_hi`, `x_lo`, `post_scale`, `same_session_control`) instead.

--- Original Design Draft Start ---

# rmsnorm_matmul (M4096 N2048 K1024, fp32) — Phase 3 implementation draft

## Goal

Phase 3 = **regime / shape specialization**: analyze where time goes across the
tensor's structure and specialize *only* where a measured win justifies the added
complexity. Start from the promoted kernel (`runs/rmsnorm_matmul_v1.py`, **1.066x**,
full 5-seed PASS) and never regress correctness (NKIBench relative-L2 gate
`< 2e-5·||v_r||` on seeds `[0,21,42,63,84]`, fp32) or latency below v1.

The honest framing up front, so the phase is scoped correctly: **Phase 2 already
established that v1 sits at the fp32 systolic PE floor** with the only non-floor PE
work (the transpose) *fully hidden* under the matmul (`load_transpose2d` removed the
transpose entirely and landed within noise, PE pinned at 97%). Phase 3's job is to
(a) confirm there is no *shape-structure* lever left that Phase 2 didn't already
cover, and (b) run the one remaining swing that could beat the fp32 floor — a
precision-split matmul — as a gated calibration, since it is the only thing that can
move the structural ceiling.

## What "shape specialization" can and cannot buy here (the decisive analysis)

Phase 3 for other operators means: edge tiles, tile-size regimes, partition/free
splits per shape. I checked each against *this* op's fixed shape and found **the shape
offers essentially no specialization surface**:

| Specialization lever | Applies here? | Why |
|---|---|---|
| **Edge / partial tiles** | **No** | Every dim divides evenly: M=4096=32·128, K=1024=8·128, N=2048=4·512. There is **no ragged tile anywhere** — no masks, no remainder loop. Nothing to specialize. |
| **Tile-size regime (partition/free split)** | **No** | The layout is *forced*. `nc_matmul` needs the contraction dim (k_in) on the partition axis; the required output tile is `[m_in(par), n(free)]`. So m_in must be the stationary/partition dim and n the moving/free dim. You cannot swap m↔n without transposing the whole result back. |
| **N-chunk (moving free) width** | **Already optimal** | v1 uses N_CHUNK=512 = `psum_fmax` = exactly one PSUM bank — the documented per-matmul moving-free maximum (precedent `6288aaad`: "tile budget is psum_fmax, not pmax"). Larger is impossible; smaller wastes the array. |
| **M-blocking (sibling matmul's phase-2 win)** | **No** | That win removed *redundant w HBM reloads* by reusing a loaded w-tile across B output-row tiles. Here w is **already fully resident** (8 MB, loaded once) and DMA is only ~20% busy — there is no redundant traffic to block for. |
| **LNC2 sharding** | **Out of contract** | Scored single-core, `--logical-nc-config=1`. Not a lever on this harness. |

So the classic phase-3 specializations are all either **vacuous** (no edge tiles) or
**already at their constraint** (N=512=psum_fmax; w resident). This is itself a
finding worth recording, not a gap.

### Where the time actually goes (measured + modeled, reconciled)

```
v1: latency=0.4716ms  MFU=46%  PE=97%  Vec=15%  Scl=11%  DMA=20%  HBMrd=25MB HBMwr=34MB
```

- **bf16-peak floor** for this GEMM = 2·M·N·K / (128·128·2.4e9·2) = **218.5 µs**. This
  equals the cost model's "stream-only" PE floor exactly, and **218.5 / 471.6 = 46.3%
  = the measured MFU.** So MFU=46% is *literally* "measured latency vs the bf16-peak
  denominator" — it is a structural fp32 ceiling, not inefficiency.
- The 2.16× gap between measured (471.6 µs) and the bf16 floor (218.5 µs) is dominated
  by the **fp32 PE-rate penalty**: D4's same-kernel bf16 swap ran **3.23× faster**
  end-to-end. That penalty is unschedulable — it is the array being bf16-native.
- Vec 15% / Scl 11% / DMA 20%: RMSNorm and all data movement are **comfortably hidden**
  under the PE-bound matmul. Confirmed by D1 (post-scale fold moved the scale pass off
  the input with no latency change) and D2b (transpose fully removed → within noise).

**Conclusion:** the only two things that could move latency are (1) shaving any
*exposed* systolic weight-fill on the main matmul — a within-fp32 micro-lever — and
(2) breaking the fp32 rate ceiling itself, which requires lower precision and thus
must clear the 2e-5 gate. Everything else is hidden or already optimal.

## Candidate directions, ranked by benefit ÷ risk

### P1 — Main-matmul loop reorder to amortize the stationary weight-fill (PRIMARY; cheap, correctness-safe)

This is the **one main-matmul micro-lever Phase 2 never touched** — Phase 2 worked
entirely on the *transpose*, never on the GEMM's own schedule.

**Observation.** v1's matmul loop is `c`-outer (4 N-chunks), `kt`-inner (8 K-tiles):
```
for c in 4:            # N-chunk
    acc = zeros[128,512]
    for kt in 8:       # K-tile — stationary xT[kt] changes every call
        acc += nc_matmul(stationary=xT[kt], moving=w_sb[kt, c-slice])
```
The **stationary operand `xT[kt]` is reloaded into the systolic array on every one of
the 1024 `nc_matmul` calls.** A `[128,128]` stationary costs ~128 cycles to fill the
array vs 512 cycles to stream the moving tile. If those fills are *exposed*, they are
128/(128+512) ≈ 20% of the matmul, ~55 µs.

**Fix.** Reorder to `kt`-outer, `c`-inner with **4 live PSUM accumulators** (4 banks,
fits the 8-bank budget), so each stationary `xT[kt]` is loaded **once** and streamed
against all 4 N-chunks before moving on:
```
acc[0..3] = zeros[128,512]         # 4 live PSUM banks
for kt in 8:
    xt = xT[kt]                    # load stationary ONCE
    for c in 4:
        acc[c] += nc_matmul(stationary=xt, moving=w_sb[kt, c-slice])
```
This is the weight-stationary dataflow the knowledgebase documents for amortizing
weight load (`scheduling-and-pipelining`; `6288aaad` on PSUM-bank accumulator sizing).

- **Theoretical ceiling:** fill goes from 1024×128 to 256×128 cycles → saves ~41 µs =
  **~8.7% of latency** *if fills are currently fully exposed*.
- **Realistic expectation: likely within noise.** Two strong priors say the compiler
  already hides these fills: (i) v1's loops are `nl.affine_range` (unordered), so the
  scheduler already has freedom to pipeline fills behind the previous matmul's stream;
  (ii) D2b removed the *transpose's* fills+compute entirely and moved nothing — direct
  evidence that PE-side fills on this kernel are not on the critical path. So P1 is most
  likely a **within-noise confirmation**, not a win.
- **Why do it anyway:** it is *cheap* (loop reorder + 4 PSUM banks), **bit-identical**
  in accumulation order (same K-accumulation, just N streamed inside), and it closes the
  last structural question — "is any weight-fill exposed on the main matmul?" — with a
  measurement instead of an assumption. Correctness risk ≈ 0 (no arithmetic change).
- **Watch:** 4 live `[128,512]` fp32 PSUM banks = 4 of 8 banks — fine. Confirm the
  reorder doesn't spill SBUF (xT already resident) or serialize on PSUM-bank
  dependencies. Keep the D1 post-scale eviction fold (apply `inv_rms` via `tensor_scalar`
  reading PSUM directly) since it is already proven and removes the input-scale pass.
- **Iterations:** ≤2 (implement on the D1 base; `--fast` triage; full-5-seed if it beats
  v1 by > the ~1.8% noise band).

### P2 — Fold the RMSNorm scale onto w-load instead of per-tile (LOW value, only if P1 idle) 

A variant to consider only if P1 shows unexpected exposed Vector/Scalar time: since
Vec+Scl are ~26% and hidden, there is no expected win, but if the reorder surfaces a
norm bubble, fusing the square+reduce+rsqrt tighter (or computing sum-of-squares via a
`nc_matmul(x, x)`-style path that shares the PE) could rebalance. **Default: skip** —
the norm is measured-hidden; this is a contingency, not a planned iteration.

### P3 — bf16×2 compensated split-matmul (STRETCH; the ONLY lever above the fp32 floor; GATED)

The fp32 PE rate is the structural ceiling (MFU=46% = bf16-peak/measured). The *only*
way past it is to do the matmul in bf16 arithmetic while recovering enough precision to
clear the 2e-5 gate — a **split/compensated GEMM**:

- Split each fp32 operand into two bf16 limbs: `x ≈ x_hi + x_lo`, `w ≈ w_hi + w_lo`
  (`x_hi = bf16(x)`, `x_lo = bf16(x - x_hi)`; same for w).
- Accumulate 3 bf16 products into fp32 PSUM: `x_hi·w_hi + x_hi·w_lo + x_lo·w_hi`
  (drop the negligible `x_lo·w_lo`). Effective mantissa ≈ 2·8 = 16 bits → rel error
  ≈ 2^-16 ≈ **1.5e-5**, just under the **2e-5** gate.
- Speed: 3 bf16 matmul passes, but bf16 is ~3.23× faster than fp32 (D4), so the compute
  is ~3×/3.23× ≈ 0.93× of the fp32 matmul — plus the split/quantize overhead. Net
  ceiling maybe ~1.05–1.2× *if and only if* it passes correctness.

- **Risk: HIGH on correctness.** 1.5e-5 vs 2e-5 is razor-thin; the accumulation of 3
  bf16 rounding steps plus the RMSNorm's own error could push it over on some seed.
  The sibling matmul task explicitly flagged "compensated-bf16x3 ~1.21x optimistic,
  uncertain vs 2e-5" and treated the 2e-5 gate as forbidding it. So this is a **gated,
  one-shot calibration**, not a planned promotion:
  - Implement the 3-pass split; score **full 5-seed** (not just `--fast`) to read the
    *actual* rel-L2 margin on the worst seed, plus latency.
  - **Promote only if** it PASSES all 5 seeds *and* beats v1 out-of-noise. Otherwise
    record it as the definitive "fp32 precision floor cannot be beaten within the gate"
    datum (upgrading D4's single-precision-swap calibration to a real attempt) and stop.
- **Risk: API.** Needs bf16 casts (`nl.copy`/activation with `dtype=bf16`) and bf16
  `nc_matmul` operands — bf16 matmul is the *native* path so this is well-supported
  (D4 already compiled a bf16 matmul on this remote). The split subtraction is plain
  `tensor_tensor`. Lower API risk than Phase 2's transpose probes.
- **Iterations:** ≤2 (implement 3-pass; full-5-seed correctness+latency; decide).

### P4 — REJECT: plain bf16 / tf32 matmul
Already closed by D4 (bf16 mantissa ~4e-3 » 2e-5; fails correctness). Not re-run.

## Plan of attack (order)

1. **Re-anchor the noise band:** one same-session v1 full-5-seed control (as in Phase 2)
   so P1/P3 are compared against a fresh control, not the historical 0.4716.
2. **P1 (primary):** reorder the main matmul to kt-outer / 4-live-PSUM-accumulator
   weight-stationary form on the D1 post-scale base. `--fast` triage; if > noise band
   over control, full 5-seed. Record PE% / MFU before-after to see if any fill left the
   PE. Most-likely outcome: within-noise → keep whichever of {v1, P1} is simpler/faster.
3. **P3 (gated stretch):** implement the bf16×2 split matmul; **full-5-seed** correctness
   read first (the gate is the whole question), then latency. Promote only on PASS +
   out-of-noise win; else record as the precision-floor confirmation.
4. Keep the best correct kernel. Expected: if neither P1 nor P3 clears the noise band /
   gate, **v1 remains promoted** and Phase 3 exits as a *specialization-space closure*:
   documented evidence that the shape has no edge/tile-regime lever and the fp32 rate is
   the ceiling — mirroring the sibling matmul's phase-3 floor-confirmation.

Never promote a candidate that fails any seed or regresses below v1. Each direction gets
≤5 iterations; stop a direction early on a clear reject or a clean within-noise read.

## Correctness invariants (must hold; unchanged from v1 unless noted)

- Every dim divides evenly → **no masks / partial tiles** anywhere (this is also why
  edge-tile specialization is vacuous).
- Reduction stays a single full-1024 free-axis `tensor_reduce`; `inv_rms = rsqrt(sumsq·1/K)`
  (folded 1/K) — rel-L2 1.3e-7 role, identical to v1.
- **P1:** matmul accumulation stays **fp32 in PSUM**, same K-accumulation order (only the
  N-chunk loop moves inside the K loop) → **bit-identical** result to the D1 base. Output
  tile `[m_in(par), n(free)]` and per-row `inv_rms` broadcast at eviction unchanged.
- **P3 only:** accumulation becomes 3 bf16 products summed in fp32 PSUM. This *changes the
  numerics* (that's the point) → its correctness is re-established empirically on the full
  5-seed gate, and it is promoted *only* if the measured worst-seed rel-L2 < 2e-5. The
  fp32 v1/P1 path is untouched and remains the guaranteed fallback.

## Risks / things to watch

- **P1 within-noise (most likely).** Do not promote a < ~1.8% "win"; re-run the control
  and require margin. A clean within-noise read is a *valid closing result*, not a
  failure — it confirms fills are already hidden (consistent with D2b).
- **P3 correctness cliff.** The 1.5e-5 vs 2e-5 margin is thin and seed-dependent; always
  score **all 5 seeds** before believing a pass. RMSNorm error stacks on top of the
  matmul split error. If any seed fails, it's a reject, full stop.
- **PSUM pressure (P1).** 4 live `[128,512]` fp32 banks = 4/8 banks; P3's split may want
  more transient PSUM/SBUF for limbs — watch for spills in the profiler (sibling saw
  residency regressions when banks/buffers grew).
- **Noise band.** ~±1.8% run-to-run (sibling + Phase-2 observed). `--fast` for triage,
  **full 5-seed** before any promotion.
- **Older remote NKI.** P1 uses only proven primitives (nc_matmul, PSUM accumulate,
  tensor_scalar) — no API risk. P3's bf16 matmul path already compiled in D4.

## Phase-3 success criterion

A correct (full 5-seed PASS) kernel that beats v1's **1.066x** by more than the noise
band. Realistic outcomes, in order of likelihood:
1. **Floor-confirmation (most likely):** P1 within noise, P3 fails the gate → **v1 stays
   promoted (1.066x)**, now with the added evidence that (a) the shape has no edge/tile
   specialization surface, (b) the main-matmul weight-fills are already hidden, and (c)
   the fp32 rate is the hard ceiling that the 2e-5 gate forbids breaking. This is a
   complete, defensible phase-3 result — the same shape of outcome as the sibling matmul
   task, reached by measurement rather than assumption.
2. **Small P1 win:** if some weight-fill was exposed, the reorder lands ~1.07–1.09x →
   promote P1.
3. **P3 surprise:** if the bf16×2 split both passes all 5 seeds and beats v1 out-of-noise
   (~1.1–1.3x), promote it — the only path that breaks the fp32 floor. Treated as a
   high-value long shot, gated strictly on the correctness measurement.

--- Original Design Draft End ---
