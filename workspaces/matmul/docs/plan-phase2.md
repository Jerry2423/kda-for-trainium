# matmul Phase 2 — Profile-driven optimization (beat baseline)

## Goal Description

Take the first correct kernel `runs/matmul_v1.py` (fp32 dense GEMM, M=4096,
K=5120, N=12288; currently 0.855x / 15.885 ms with MFU=41%) and reduce on-device
latency **below the baseline (13.5785 ms, i.e. > 1.0x speedup)** while preserving
correctness on all five NKIBench seeds. The measured bottleneck is the PE array
**stalling on rhs HBM reload** (rhs re-loaded once per M-tile ≈ 8 GB; DMA 76%
active; PE floor is only 6.62 ms so ~58% of wall-clock is PE idle). The primary
lever is reducing rhs HBM traffic via **M-blocking** (reuse each loaded rhs K-tile
across B output-row tiles), optionally combined with **rhs DMA double-buffering**.
The stretch target is to approach the fp32 PE floor (~6.62 ms ≈ 2.05x) as far as
DMA reduction allows; if a direction cannot progress, record before/after latency
+ profiler evidence explaining why rather than silently moving on.

## Acceptance Criteria

- AC-1: Correctness is never regressed — the best Phase-2 candidate passes the
  NKIBench relative-L2 gate (`< 2e-5`, fp32) on all five seeds `[0,21,42,63,84]`.
  - Positive Tests (expected to PASS):
    - `verify.py --op matmul --candidate runs/<file>.py` (full 5-seed) reports
      `correct: 1/1` with every per-seed `l2_norm_passed` true.
    - `--fast` (seed 42) reports `PASS` during iteration.
  - Negative Tests (expected to FAIL):
    - A candidate that mis-maps M-block output rows (e.g. two blocked M-tiles
      writing the same `out[mt,:, :]` rows, or a swapped accumulator) fails L2.
    - A candidate that downcasts any operand to bf16/tf32 fails the 2e-5 gate.
  - AC-1.1 (margin sanity, **diagnostic only — not a promotion gate**): a correct
    candidate should keep mean relative L2 well under 2e-5 (fp32 accumulation only
    reorders the K sum). A result merely squeaking under 2e-5 is investigated
    before promotion, but the sole hard correctness gate is the official 2e-5 pass
    on all seeds — a clear pass is never rejected for missing an arbitrary margin.

- AC-2: The best Phase-2 candidate achieves **speedup > 1.0x** (latency
  < 13.5785 ms) on a stable p50 measurement (full, non-`--fast` run), i.e. it
  beats the baseline. This is the Phase-2 exit bar.
  - Positive Tests (expected to PASS):
    - Full `verify.py` prints `speedup > 1.000x` for the promoted candidate, and
      the result is stable (not a single lucky run — confirmed by the harness p50
      over its iters, re-run if variance looks high).
  - Negative Tests (expected to FAIL):
    - A candidate at ≤ 1.0x is NOT promoted as the Phase-2 result (it may still be
      recorded as a rejected/among-directions data point).
    - A candidate that lowers HBMrd but regresses latency (e.g. via SBUF spills)
      does not satisfy AC-2.
  - AC-2.1 (stretch, non-gating): progress toward the ~6.62 ms fp32 PE floor is
    tracked; higher is better, but the hard bar is > 1.0x. Not reaching the floor
    is acceptable if justified by profiler evidence (residual DMA / scheduling).

- AC-3: Every candidate is justified by **profiler evidence**, not latency alone.
  For each candidate (kept or rejected) record latency, speedup, MFU, PE-active,
  DMA-active, and HBMrd/HBMwr, plus the B value / buffer layout tried.
  - Positive Tests (expected to PASS):
    - `benchmark.csv` gains a row per perf-relevant candidate; `candidates.jsonl`
      gains a node with `parent` links (root parent = `matmul_v1`); `profile/`
      holds the metric digest for each major direction.
    - A kept candidate shows *both* reduced HBMrd/DMA-wait *and* improved (or at
      least non-regressed) latency vs its parent.
  - Negative Tests (expected to FAIL):
    - Promoting a candidate on reduced HBMrd alone while latency regressed is
      rejected (HBM reduction is necessary, not sufficient).
    - A candidate showing SBUF/PSUM spills or unexpected extra HBM traffic in the
      profiler digest is rejected even if it happens to score.

- AC-4: The kernel remains a single self-contained `@nki.jit def kernel(v1, v2)`
  with the exact tiled I/O contract (inputs `v1 (32,128,40,128)`,
  `v2 (40,128,12288)`; output `(32,128,12288)` `nl.shared_hbm`), fp32 throughout,
  no harness/reference/baseline edits.
  - Positive Tests (expected to PASS):
    - The adapter assembles + runs each candidate end-to-end with no signature /
      shape / import errors.
  - Negative Tests (expected to FAIL):
    - Changing the signature or output shape, or editing NKIBench
      `{kernels,reference,seeds,summary.json}`, is rejected.

- AC-5: Each optimization direction is explored for **at most 5 iterations**;
  every direction ends in an explicit keep / revise / reject decision backed by
  the recorded before/after evidence.
  - Positive Tests (expected to PASS):
    - The Phase-2 evidence trail shows, per direction (D1 M-blocking, D2
      double-buffer), the candidates tried (≤5), their metrics, and a verdict.
  - Negative Tests (expected to FAIL):
    - Silently abandoning a direction with no recorded before/after, or exceeding
      5 iterations on one direction without a decision, violates the contract.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A single correct `kernel(v1, v2)` that: (a) M-blocks the outer loop so each rhs
K-tile `[128,512]` is loaded once and fed to B stationary lhsT tiles into B
distinct `[128,512]` fp32 PSUM accumulators before being discarded (B swept over
2, 4, 6, and — experimentally — 8, picking the empirical best); and optionally
(b) double-buffers the rhs DMA (ping-pong prefetch of the next K-tile while the
current matmuls run) on top of the best B. All correctness invariants from
Phase 1 hold (fp32, transpose lhs onto the partition axis, K-accumulate all 40
tiles before store, static loop bounds, no masking). Full profiler evidence is
recorded per candidate.

### Lower Bound (Minimum Acceptable Scope)
A single correct M-blocked `kernel(v1, v2)` at whatever B first clears
**> 1.0x** on a stable full-run p50 while passing all 5 seeds, with its
before/after latency + profiler digest recorded. Double-buffering (D2) is
optional and only pursued if the profiler still shows DMA wait after the best B.
If no B beats 1.0x, the deliverable is the best correct candidate plus recorded
profiler evidence explaining the residual bottleneck (this still satisfies AC-1
/AC-3/AC-5 but not AC-2, and must be reported honestly).

### Allowed Choices
- Can use: M-blocking (B output-row tiles sharing a loaded rhs tile); B ∈ {2,4,6,8}
  chosen empirically; multiple `[128,512]` fp32 PSUM accumulators (≤ 8 banks);
  rhs ping-pong SBUF double-buffering / prefetch; the Phase-1 identity-transpose
  idiom; `nl.affine_range`/`nl.sequential_range`; full or K-blocked lhsT residency.
- Cannot use: any non-fp32 dtype on the numeric path; moving free dim > 512 (fp32
  PSUM bank limit on trn2); N-outer loop order (re-transposes lhs 24x — strictly
  worse); partial-K writes to `out` (must accumulate all 40 K-tiles before store);
  masking/remainder logic (all dims divide evenly); edits to NKIBench
  reference/baseline/summary; a top-level `runs/`/`outputs/`.

> **Note on Determinism**: The math (fp32 GEMM), the tiled I/O contract, and the
> transpose requirement are fixed. The real degrees of freedom this phase are the
> M-block factor B, whether rhs is double-buffered, and lhsT residency (full vs
> K-blocked). The optimum B is an empirical question (SBUF/PSUM pressure vs DMA
> relief), not a fixed choice — sweep and measure.

## Feasibility Hints and Suggestions

> **Note**: Reference only — conceptual, not prescriptive.

### Conceptual Approach

The bottleneck (grounded): Matmul cost on trn2 = `dst_free_elems * 100/240` ns.
Main matmuls `32*24*40 * 512 = 6.554 ms`; transpose `1280 * 128 = 0.068 ms` (1%).
PE floor ≈ 6.62 ms; measured 15.885 ms → 42% ≈ measured MFU (41%). So ~58% is PE
**idle on rhs DMA**: rhs (252 MB) is reloaded once per M-tile → 8.05 GB (= measured
HBMrd 7584 MB), loaded inline per matmul with no prefetch.

D1 — M-blocking (reuse rhs across B M-tiles). Restructure so rhs traffic drops to
≈ 8/B GB:

```
# lhsT for a BLOCK of B M-tiles: [mb, kt] -> [k_in(par), m_in(free)]
# SBUF-legal: par_dim on partition axis, (mb, kt) as leading index dims.
lhs_t = nl.ndarray((B, 40, par_dim(128), 128), fp32, sbuf)   # B*40*128*4/1024 KB/part

for mblock in range(32 // B):                 # 32/B blocks of B M-tiles
    for mb in range(B):                       # transpose each block member's lhs
        for kt in range(40):
            load v1[mblock*B+mb, :, kt, :] -> sbuf; transpose -> lhs_t[mb, kt]
    for c in range(24):                        # N-chunks of 512
        acc[mb] = psum_zeros([128,512]) for mb in range(B)   # B PSUM banks
        for kt in range(40):
            rhs_sb = load v2[kt, :, c*512:(c+1)*512]          # rhs loaded ONCE...
            for mb in range(B):                               # ...reused B times
                acc[mb] += nc_matmul(lhs_t[mb, kt], rhs_sb)   # [m_in,512]
        for mb in range(B):
            store out[mblock*B+mb, :, c*512:(c+1)*512] = copy(acc[mb])
```

Correctness invariants (Codex CORE_RISK): the B accumulators must be **distinct**
PSUM banks, each lhsT[mb] a distinct SBUF region, and each `mb` must write a
**distinct** `out[mblock*B+mb, :, ...]` row-block. A shared/aliased accumulator or
a wrong M index corrupts rows while possibly passing a subset of seeds — so the
host numpy check (task) must verify a full tile from a **non-zero mblock, mb>0**.

Budget (honest, per-partition SBUF ≤ 192 KB; PSUM = 8 banks of [128,512] fp32):
- lhsT = `B*40*128*4/1024` KB/part: B=2→40, B=4→80, B=6→120, B=8→160. Plus rhs
  ping-pong (~4), B-wide output staging (B*2 KB), transpose temp (~1). Totals:
  B=2≈49, B=4≈93, B=6≈137, B=8≈181 KB. **B=8 is tight** (~11 KB slack → spill
  risk); B=2/4/6 comfortable.
- PSUM: B accumulator banks. B=8 uses all 8 → **no bank headroom for D2**; B≤6
  leaves banks free. The transpose PSUM is transient (freed before the N-loop).

Sweep order (Codex ALTERNATIVE): **B=2 first** (most DMA relief per unit SBUF/PSUM
pressure, safest), then B=4, B=6; treat B=8 as experimental (verify no spill in
the profiler). Pick the empirical best B by measured latency, not by HBMrd alone.

D2 — rhs DMA double-buffering (only after the final best B is chosen, and only if
DMA wait remains). Two rhs SBUF buffers; prefetch K-tile 0 before the loop; each K
iteration computes the B matmuls on `cur` while DMA-ing K-tile+1 into `1-cur`.
Precedents `bc877398`, `3c7e053b`. It adds no PSUM-lived values (double-buffering is
SBUF-only), but it needs the schedule to overlap — validate overlap in the profiler
(DMA-active should drop / hide behind PE), not by latency alone (Codex: manual
ping-pong can serialize if the compiler sees aliasing). **B=8 + D2 rule:** if the
best D1 candidate is B=8 (all 8 PSUM banks in use, and SBUF ~181 KB/part already
tight), D2's extra rhs buffer may not fit — skip D2 for B=8 unless a separate
experimental branch shows it compiles without spills; prefer applying D2 to a
B ≤ 6 candidate that left headroom.

Expected HBMrd scaling per B (sanity target only; **latency + no-spill is the
promotion criterion**, not HBMrd): rhs traffic ≈ 8.05/B GB plus lhs (~0.084 GB read
once per M-block pass = 0.084 GB total, B-independent) + output (0.2 GB). So total
HBMrd ≈ `8.05/B + ~0.28` GB: B=2 → ~4.3 GB, B=4 → ~2.3 GB, B=6 → ~1.6 GB,
B=8 → ~1.3 GB. A measured HBMrd materially above this at a given B signals spilling.

D3 — transpose: build lhsT once per M-block (a side effect of D1). Do NOT invest
in eliminating transposes (1% of floor).

Rejected: bf16/tf32 downcast (breaks 2e-5); moving free > 512 (PSUM bank);
N-outer (re-transpose 24x).

### Relevant References
- `runs/matmul_v1.py` — Phase-1 baseline candidate to evolve; proven transpose +
  K-accumulate + `[128,512]` PSUM structure.
- `runs/_layout_check.py` — host numpy check to extend for the M-block indexing.
- `../AccelOpt/NKIBench/kernels/matmul_M4096_N12288_K5120_0.py` — baseline kernel;
  its `v6/v7/v8` blocked SBUF shapes show a legal multi-index-dim residency form.
- `.claude/skills/kernel-cost-analysis` — the PE-floor cost model.
- `.claude/skills/kernel-optimization-kb` — precedents `6288aaad`
  (tile budget to kill redundant weight DMA), `bc877398` / `3c7e053b`
  (ping-pong double-buffering).
- `verify.py` — scoring + the profiler metric digest (`summary_metrics`).

## Dependencies and Sequence

### Milestones
1. D1 — M-blocking (primary):
   - Phase A: extend the host numpy check for M-block indexing (verify a full tile
     from mblock>0, mb>0, exact K-accum order).
   - Phase B: implement M-blocked kernel, B parameterized; score B=2, then 4, 6
     (`--fast` reads); record each in benchmark.csv/candidates.jsonl/profile.
   - Phase C: try B=8 experimentally; reject if profiler shows spills or a latency
     regression. Pick best B; full 5-seed confirm.
2. D2 — rhs double-buffering (conditional on residual DMA wait after best B):
   - Step 1: add ping-pong rhs prefetch on the best-B kernel; score; keep only if
     MFU rises / DMA-wait falls without correctness or spill regressions.
   - Step 2: full 5-seed confirm on the promoted candidate.
3. Promote + report:
   - Step 1: full-run p50 for the best candidate; confirm > 1.0x (AC-2) and
     stability; record final evidence.
   - Step 2: if no candidate beats 1.0x, report the best correct one + profiler
     evidence for the residual bottleneck.

Dependencies: D1 before D2 (never tune B and prefetch simultaneously — Codex:
failures become unattributable). Each candidate depends on a passing correctness
check before it counts toward AC-2. B=8 depends on observing spill/no-spill in the
profiler.

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Extend `runs/_layout_check.py` for M-block indexing: verify a full `(mblock>0, mb>0, n_chunk)` output tile with the exact per-block K-accumulation, confirming distinct-row / distinct-accumulator mapping | AC-1, AC-4 | coding | - |
| task2 | Implement M-blocked kernel (`runs/matmul_v2_b{B}.py` or a B-parameterized source), B accumulators in distinct PSUM banks, rhs loaded once per K-tile and reused across B M-tiles; fp32; store distinct out row-blocks | AC-1, AC-3, AC-4 | coding | task1 |
| task3 | Score B=2, then B=4, B=6 with `verify.py --fast`; record latency+MFU+PE+DMA+HBMrd per B in benchmark.csv/candidates.jsonl/profile; pick empirical best B | AC-2, AC-3, AC-5 | coding | task2 |
| task4 | Try B=8 experimentally; inspect profiler for SBUF/PSUM spills (inferred from unexplained HBMrd/HBMwr rise or latency regression — no explicit spill counter exists); keep only if no spill and latency improves; else reject with recorded evidence. Completes the D1 (best-B) selection. | AC-3, AC-5 | coding | task3 |
| task5 | After the final D1 selection (best B chosen, incl. the B=8 keep/reject verdict): IF the best-B profiler still shows DMA wait AND that B leaves PSUM headroom (B ≤ 6), add rhs ping-pong double-buffering (D2); score; keep only if DMA-wait/MFU improves without correctness/spill regression (≤5 iters). If best B is 8 (all 8 PSUM banks used), skip D2 unless a separate experimental branch proves resource headroom. | AC-2, AC-3, AC-5 | coding | task4 |
| task6 | Full 5-seed `verify.py` on the best selected candidate (the winner of D1 ± D2, whichever is fastest — D2 may be skipped); confirm all seeds pass and stable p50 speedup > 1.0x; record final evidence (parent DAG to matmul_v1) | AC-1, AC-2, AC-3 | coding | task5 |
| task7 | (Optional) Codex review of the promoted kernel for M-block accumulator/row-mapping correctness + prefetch aliasing before promotion | AC-1 | analyze | task2 |

Candidate naming convention (Codex OPTIONAL): `runs/matmul_v2_b2.py`,
`matmul_v2_b4.py`, `matmul_v2_b6.py`, `matmul_v2_b8.py`, and (if D2 kept)
`matmul_v2_b{best}_dbuf.py` — so `candidates.jsonl` node ids map 1:1 to files.

## Claude-Codex Deliberation

### Agreements
- The bottleneck is PE idle on rhs HBM reload, not compute or transpose; M-blocking
  to reuse rhs is the primary lever.
- fp32 must hold; bf16/tf32 downcast is off-limits (2e-5 gate).
- B accumulators must be distinct PSUM banks; each blocked M-tile writes distinct
  output rows; correctness must be verified with a non-trivial tile (mblock>0, mb>0).
- Record full profiler evidence (HBMrd, DMA-active, MFU, PE-active, latency) per
  candidate; reduced HBMrd alone is necessary but not sufficient — latency must
  improve and there must be no spills.
- D1 before D2; ≤5 iterations per direction with explicit keep/revise/reject.

### Resolved Disagreements
- B sizing: the draft led with "B=4 safe, B=8 stretch." Codex argued **start at
  B=2** (most DMA relief per unit pressure, safest) and that B=8 is genuinely risky
  (≈181 KB/partition leaves ~11 KB scratch → spill risk; uses all 8 PSUM banks →
  no D2 headroom). **Resolution: sweep B=2→4→6, B=8 experimental only**; pick by
  measured latency. Budget re-computed to confirm (B≤6 comfortable).
- "Reduced HBMrd = success": Codex flagged that lower HBMrd with spills can regress
  latency. **Resolution: AC-3 requires both HBM/DMA reduction AND non-regressed
  latency AND no spills**; HBM reduction alone never promotes a candidate.
- D1 vs D2 ordering: **Resolution: find best B without prefetch first**, add D2
  only if DMA wait remains, so failures stay attributable.

### Convergence Round 1 (second Codex pass, reviewing candidate plan v1)
Codex found the plan "broadly reasonable, technically aligned … complete enough to
execute after a small dependency cleanup." No conceptual blockers, no UNRESOLVED
items. Four `REQUIRED_CHANGES`, all accepted and applied:
1. `task5` (D2) depended only on `task3` but must run after the *final* D1
   selection (incl. the B=8 verdict). → repointed `task5` to depend on `task4`.
2. `task6` hard-depended on `task5`, but D2 is conditional/skippable. → repointed
   `task6` to `task5` with "best selected candidate (D2 may be skipped)" wording.
3. B=8 + D2 interaction was under-specified. → added an explicit rule: if best B=8
   (all 8 PSUM banks, SBUF tight), skip D2 unless a separate branch proves headroom.
4. "Spill" needed a detection method (no explicit counter). → AC-3 / task4 now infer
   spills from unexplained HBMrd/HBMwr rise or latency regression.
`OPTIONAL_IMPROVEMENTS` folded in: per-B expected-HBMrd sanity targets (latency
still the promotion bar), a fixed `matmul_v2_b{B}[_dbuf]` naming convention, and
AC-1.1 explicitly marked diagnostic-only (not a hidden gate).

Convergence matrix (round 1):
| Topic | Claude (plan v1) | Codex | Resolution |
|---|---|---|---|
| task5/task6 dependencies | task5←task3, task6←task4+task5 | must follow final D1 selection; D2 skippable | resolved (repointed) |
| B=8 + D2 co-viability | "D2 on best B" + "pair D2 with B≤6" (implicit tension) | make the rule explicit | resolved (explicit B=8 rule) |
| spill detection | "reject on spills" | define how to detect (no counter) | resolved (infer from HBM/latency) |
| AC-1.1 strictness | margin sanity | make diagnostic-only, not a gate | resolved (clarified) |

Round 2 not required: no `REQUIRED_CHANGES` remain; Codex's UNRESOLVED = "None
blocking."

### Convergence Status
- Final Status: `converged` (Codex first-pass + one convergence round; first-pass
  CORE_RISKS became AC-1/AC-3 negative tests + correctness invariants,
  MISSING_REQUIREMENTS became AC-3 evidence fields + AC-2 stability,
  ALTERNATIVE_DIRECTIONS (B=2 first, B-sweep, K-blocked lhsT fallback) became the
  sweep order; round-1 REQUIRED_CHANGES all applied. No unresolved items.)

## Pending User Decisions

Codex's `QUESTIONS_FOR_USER` are answerable from the harness / this session; none
block implementation:

- DEC-1: Does precedent `6288aaad` use B=4 or B=8 / a different layout?
  - Claude Position: Immaterial — we choose B empirically by measured latency on
    THIS shape, not by copying the precedent's constant. The precedent only
    validates the *technique* (derive tile budget from PSUM/SBUF to kill redundant
    weight DMA). Decision Status: `Resolved — B chosen empirically (sweep 2/4/6/8)`.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: The precedent is a pattern source, not a constant to copy.
- DEC-2: Are profiler spill indicators available?
  - Claude Position: `verify.py` surfaces `summary_metrics` (MFU/PE/Vec/Scl/DMA/
    HBMrd/HBMwr); explicit spill counters may not be exposed, so we infer spills
    from an HBMrd/HBMwr rise or a latency regression at higher B. Decision Status:
    `Resolved — infer from HBM traffic + latency; treat unexplained HBM rise as spill`.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: No dedicated spill metric; use HBM traffic + latency as proxy.
- DEC-3 / DEC-4: fixed M/K/N + fp32 specialization allowed?
  - Claude Position: Yes — the NKIBench case is a single fixed shape (M4096 N12288
    K5120, fp32); specializing hard for it (static bounds, no masks) is expected
    and already done in Phase 1. Decision Status: `Resolved — fixed shape, fp32,
    static bounds; masks unnecessary (dims divide evenly)`.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: Fixed by the benchmark case definition.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology
  (`AC-`, `Milestone`, `Phase`, `Step`, `D1/D2`, task IDs). Use domain names
  (`m_block`, `lhs_t`, `psum_acc`, `rhs_buf`, `B` / `M_BLOCK`).
- fp32 on the entire numeric path; distinct PSUM banks per blocked M-tile; comment
  the output store tying `out[mblock*B+mb, :, n0:n0+512]` axes to `[m_in, n]`.
- Keep the kernel a single `@nki.jit def kernel(v1, v2)`.

--- Original Design Draft Start ---

# matmul Phase 2 — Profile-driven optimization draft

## Starting point (best correct kernel)

`runs/matmul_v1.py` (Phase 1): M-outer fp32 GEMM, identity-transpose lhs,
K-accumulate 40 tiles into a `[128,512]` PSUM bank, stream 24 N-chunks.
- **Correct** on all 5 seeds. **Latency 15.885 ms → 0.855x** (baseline 13.5785 ms).
- Profiler: **MFU=41%, PE=89%, Vec=2%, Scl=0%, DMA=76%, HBMrd=7584 MB, HBMwr=201 MB.**

## Bottleneck diagnosis (grounded, not guessed)

**Theoretical PE floor (kernel-cost-analysis, trn2).** The Matmul cost
model is `latency = dst_free_elems * 100 / tensor_freq(240)`; there is no separate
weight-load term (stationary load overlaps in the systolic array).
- Main matmuls: `32 M-tiles * 24 N-chunks * 40 K-tiles = 30720`, each dst `[128,512]`
  → `30720 * 512 * 100/240 = 6.554 ms`.
- Transpose matmuls: `32*40 = 1280`, each dst `[128,128]` → `0.068 ms` (**only 1.0%**).
- **PE floor ≈ 6.62 ms.**

**The gap is stall, not work.** `6.62 / 15.885 = 42%` — this *equals* the measured
MFU (41%). So ~58% of wall-clock is the PE array **idle**, and with DMA=76% active
and HBMrd=7584 MB, it is idle **waiting on rhs DMA**.

**Why rhs DMA dominates.** rhs is 252 MB; M-outer reloads it once per M-tile → `252 MB
* 32 ≈ 8.05 GB` (matches HBMrd 7584 MB). Every one of the 30720 main matmuls
`nl.load`s its rhs `[128,512]` tile inline immediately before the matmul, with no
prefetch, so the PE stalls on each load. **rhs reload + no DMA/compute overlap is the
bottleneck.** The transpose (1%) and fp32 rate are NOT the gate.

## Optimization directions (ranked by benefit / risk)

### D1 — M-blocking to reuse rhs across B M-tiles  [HIGHEST VALUE, low risk]
Process a block of **B M-tiles together**: load each rhs K-tile once, feed it into
B stationary lhsT tiles (B matmuls into B separate PSUM accumulators) before
discarding it. rhs HBM traffic drops from 8.05 GB to `8.05/B`:
- B=4 → 2.01 GB (lhsT resident 80 KB/partition), B=8 → 1.01 GB (160 KB/partition).
SBUF budget (192 KB/partition) allows up to B=8 for lhsT, but must also hold the
streamed rhs tile(s) + output tiles → **B=4 is the safe first step, B=8 the stretch.**
Precedent: `6288aaad` (derive tile budget from PSUM/SBUF constraint to kill
redundant weight DMA per tile). Expected: DMA pressure ~4x lower; if the kernel is
DMA-stall-bound this should move MFU substantially toward the 6.6 ms floor.
PSUM note: B accumulators each `[128,512]` = B banks; B=4 uses 4 of 8 banks — fits.

### D2 — Double-buffer the rhs DMA (ping-pong prefetch)  [HIGH VALUE, medium risk]
Even with reuse, the rhs load for K-tile kt+1 should overlap the matmul on kt.
Pre-allocate two rhs SBUF buffers; prefetch the first K-tile before the loop; each
iteration computes on `cur` while DMA-ing the next tile into `1-cur`. Hides the
remaining rhs latency behind PE work.
Precedents: `bc877398`, `3c7e053b` (ping-pong K/V and block-metadata buffers).
Composes with D1 (prefetch the next shared rhs tile). Risk: buffer-index /
dependency bugs; verify correctness each step.

### D3 — lhs load + transpose amortization  [LOW VALUE]
lhs is only 84 MB and loaded once per M-tile already; the transpose is 1% of the PE
floor. Keep the transpose but ensure the lhsT for a whole M-block is built once (a
side effect of D1). **Do NOT invest in eliminating transposes** — negligible payoff.

### D4 — widen moving free dim beyond 512  [REJECT on trn2]
PSUM fp32 bank = 512 free; can't grow a single accumulator past 512 on trn2. Reject.

### Rejected / deferred
- Downcast to bf16/tf32 for 4x PE throughput — **rejected**, breaks the 2e-5 L2 gate
  (fp32 mandatory). This is why the PE floor is what it is.
- N-outer loop — would re-transpose lhs 24x; strictly worse. Reject.

## Plan of attack (≤5 iterations per direction, per the phase contract)

1. **D1 first** (biggest, safest): implement M-blocking B=4, score `--fast`, compare
   latency + HBMrd + MFU vs v1. If SBUF ok and it helps, try B=8. Keep the best B.
2. **D2 on top of the best D1**: add rhs double-buffering; score; keep if MFU rises.
3. Re-profile; if still DMA-bound, revisit B; if now PE-bound (MFU high), stop —
   we're near the fp32 floor.
Each candidate: `verify.py --fast` for a read, full 5-seed before promoting; record
`benchmark.csv` + `candidates.jsonl` (parent = matmul_v1) + `profile/` digest.
**Never regress correctness** (all 5 seeds must keep passing).

## Target

Beat baseline: **> 1.0x** (i.e. < 13.5785 ms) as the Phase-2 exit bar; stretch toward
the 6.62 ms PE floor (theoretical max ≈ 2.05x) as far as DMA reduction allows. If a
direction cannot progress, record the before/after latency + profiler evidence
explaining why rather than silently moving on.

## Correctness / evidence contract (unchanged from Phase 1)
- fp32 throughout; all 5 seeds `[0,21,42,63,84]` pass relative-L2 `< 2e-5`.
- Single `@nki.jit def kernel(v1,v2)`; candidates in `runs/`; never edit baseline/reference.
- Parent DAG in `candidates.jsonl`; per-direction profiling under `profile/`.

--- Original Design Draft End ---
