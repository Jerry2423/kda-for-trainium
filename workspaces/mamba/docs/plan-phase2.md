# mamba (M7168 C256 S16, fp32) — Phase 2 Plan: Profile-Driven Sequence Tiling

## Goal Description

Turn the phase-1 mamba selective-scan kernel (`runs/mamba_v1.py`, currently
**0.832× — slower than the 1.258 ms baseline**) into a kernel that beats the
baseline by a wide margin, using the profile evidence already gathered. The
phase-1 regression is fully diagnosed: the whole-M `[128, 7168]` fp32 working set
(~196 KB/partition) sits at the trn2 208 KB SBUF limit, leaving the compiler no
double-buffering slack, so it **spills to HBM** (`HBMrd = 72 MB` vs the ~16 MB
read-once floor) and PE/Vec co-limit at ~51% each.

The single decisive lever is **sequence tiling**: chunk the M=7168 scan (free)
axis into fixed-width tiles (start 512), carry the scan's last column forward as
the next tile's `initial`. Two AccelOpt reference kernels measured through this
harness's own `verify.py` path prove the outcome — `mamba_v3` (seq-tile inner,
`static_range`) reaches **1.504×** and `mamba_optimized` (seq-tile outer,
`sequential_range`, carried `scan_state`) reaches **1.608×** — both by dropping
`HBMrd` to 16 MB and lifting PE/Vec to 93–99% / 82–89% **without changing the
math**. The op sequence (`activation(exp)`, `tensor_tensor`,
`tensor_tensor_scan`, `b`/`c` broadcast) is identical to phase 1 and must stay
identical; only the loop/tile structure changes.

A secondary, opportunistic goal: at ~1.6× the bottleneck engine is the Tensor
engine (PE=99%) even though mamba has no explicit matmul — the `b`/`c`
partition-dim `broadcast_to((128, M))` is lowered to `nc_matmul(ones, row)` on
the PE. Relocating that broadcast off the saturated PE (which sits next to an
idle DMA at 11%) is the only lever that can push past the 1.608× reference. This
is explicitly a stretch, not a phase gate.

Correctness must never regress: the relative-L2 gate is `||v_k − v_r||₂ <
3e-5 · ||v_r||₂` on seeds `[0, 21, 42, 63, 84]`, fp32 (mamba's looser 3e-5, per
`adapter/nkibench_case.py`), enforced by `verify.py`'s `l2_norm_passed`. The
numpy oracle for this exact op sequence is rel-L2 4.08e-7 — ~75× inside the gate.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. "Latency", "speedup", `HBMrd`, and the per-engine
active % (`PE`, `Vec`, `Scl`, `DMA`) all refer to the values surfaced by
`verify.py` / the remote profiler digest; speedup = `1.258274 ms / candidate_ms`.

- AC-1: **Correctness is never regressed.** Every candidate that is recorded or
  promoted passes the NKIBench relative-L2 gate (`< 3e-5 · ||v_r||`) on all five
  seeds `[0, 21, 42, 63, 84]` in fp32, verified by `verify.py` (`l2_norm_passed`).
  This is a HARD requirement, not a trend.
  - Positive Tests (expected to PASS):
    - A full-seed `verify.py` run (no `--fast`) on the promoted phase-2 kernel
      reports pass on all five seeds.
    - A `--fast` (seed-42) run on any candidate under active iteration reports a
      pass before it is considered for a full-seed run.
  - Negative Tests (expected to FAIL):
    - Any candidate whose carry loop uses `affine_range` (breaking the
      loop-carried scan dependency) fails the rel-L2 gate on at least one seed and
      is NOT recorded as passing or promoted.
    - A candidate promoted on `--fast` alone (seed 42) without a full 5-seed
      confirmation is rejected by the process gate even if seed 42 passes.
- AC-2: **At least one D1 sequence-tiling variant lands ≥ 1.5× with a full 5-seed
  pass**, becoming the new phase-2 promoted best. This is the primary success
  target (it reproduces two measured references, so it is a hard expectation, not
  a stretch).
  - AC-2.1: **D1b implemented** — `mamba_optimized` shape: seq-tile OUTER loop over
    `nl.sequential_range(n_seq_tile)`, channels then states inside, a
    `[n_channel_tile, 128, S]` `scan_state` carried across tiles, output stored
    per seq-tile slice (no whole-M accumulator).
    - Positive: D1b compiles, passes 5-seed rel-L2, and its digest shows
      `HBMrd ≈ 16 MB` (spill eliminated) with PE/Vec materially above the phase-1
      51/51% (target envelope ≈ PE 99 / Vec 89).
    - Negative: If D1b's outer `sequential_range` reproduces the AccelOpt
      "wrong-answer / worse-perf" failure (fails rel-L2, or `HBMrd` stays high, or
      latency does not improve), it is NOT promoted; the process falls through to
      AC-2.2.
  - AC-2.2: **D1a implemented and measured** — `mamba_v3` shape: seq-tile INNER
    loop over `nl.static_range(n_seq_tile)`, per-state `scan_init` `[128, 1]`
    carried as `scan_res[:, seq_tile-1]`. D1a is run at least once regardless of
    D1b's outcome (local compiler/version effects could invert the D1a/D1b
    ranking; running both is cheap), and the faster passing variant is promoted.
    - Positive: D1a compiles, passes 5-seed rel-L2, digest shows `HBMrd ≈ 16 MB`;
      the faster of {D1a, D1b} that passes is the promoted phase-2 best.
    - Negative: A variant that fails 5-seed correctness is never selected as the
      winner even if it is the faster of the two.
  - AC-2.3: **Carry extraction preserves identity.** The carried state is exactly
    the previous tile's final scan column, shape-preserved: D1a carries a `[128,1]`
    slice (not squeezed / not broadcast from the wrong axis); D1b writes state `s`'s
    final value into the `[i_channel_tile, :, i_state]` slot without cross-mixing
    the two channel tiles or the 16 states.
    - Positive: 5-seed rel-L2 passes (a corrupted carry would exceed 3e-5, given the
      4.08e-7 oracle margin), confirming carry indexing is correct.
    - Negative: A carry that indexes the wrong `(channel-tile, state)` slot or the
      wrong column produces a rel-L2 failure and is rejected.
- AC-3: **The op sequence and math are unchanged from phase 1.** Every phase-2
  candidate uses the same primitive sequence (`activation(exp, scale=A)`,
  `tensor_tensor(delta,u)`, `tensor_tensor(·,b_bcast)`, `tensor_tensor_scan`,
  `tensor_tensor(·,c_bcast)`, accumulate). Only loop order, tile width, broadcast
  placement, and hoisting change.
  - Positive: A diff against `runs/mamba_v1.py` shows only loop/tile/placement
    changes; rel-L2 stays in the ~1e-6 regime (not merely under 3e-5).
  - Negative: A candidate that alters the numerics (e.g. a different scan
    formulation, a reduced-precision cast, or reordered non-associative math) is
    out of scope for phase 2 even if it happens to pass.
- AC-4: **Each recorded candidate is backed by profiler evidence.** Every perf
  change is appended to `benchmark.csv`; every candidate is a node in
  `candidates.jsonl` with DAG `parent` links; per-engine digest
  (`PE/Vec/Scl/DMA %`, `HBMrd/HBMwr`, latency) is captured. Kernel `.py` sources
  live under `runs/` (tracked).
  - Positive: After the phase, `benchmark.csv` and `candidates.jsonl` contain one
    row/node per measured candidate with parent links forming a DAG rooted at
    `mamba_v1`, and each carries its digest.
  - Negative: A candidate that changed latency but left `benchmark.csv` /
    `candidates.jsonl` un-updated, or whose `.py` was written outside `runs/`, is a
    process failure.
- AC-5 (opportunistic, NOT a phase gate): **Ceiling-raising past 1.608× is
  attempted via D3a and kept only on measured evidence.** D3a hoists the `b`/`c`
  broadcast out of the 2-iteration channel loop. Because this harness exposes only
  per-engine active % (no IR / no `nc_matmul` counts), the keep/drop decision uses
  **latency as the primary metric** with the PE-active-*time* proxy
  (`latency_ms × PE_active_fraction`) as the directional check and DMA % as a
  guardrail.
  - Positive: D3a passes 5-seed rel-L2 AND reduces latency below the D1 winner,
    with PE-active-time not contradicting the "less PE work" hypothesis → kept.
    Reaching < 0.78 ms (beating the 1.608× reference) is the aspirational marker.
  - Negative: If D3a leaves latency unchanged/worse (the compiler likely already
    CSE'd the channel-independent broadcast) it is marked no-op and dropped, and
    the D1 winner remains the promoted best. Failing to beat 1.608× does NOT fail
    the phase.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A phase-2 kernel that lands the D1 sequence-tiling winner (≥ 1.5×, 5-seed pass,
`HBMrd ≈ 16 MB`), then, within the ≤5-iteration budget, opportunistically explores
the ceiling-raisers: D3a (hoist `b`/`c` broadcast off the channel loop), a D2
seq-tile width sweep `{256, 512, 1024}`, D4 (hoist state-independent
`deltaU = delta·u`), and — only if PE remains the clear limiter with DMA headroom
— D3b (stride-0 DMA broadcast). Every kept change is evidence-gated (latency + a
full 5-seed pass) and every candidate is recorded in `benchmark.csv` /
`candidates.jsonl` with a profiler digest. A phase-3 preview note (unroll the
16-state loop, hard-tune seq-tile width for the fixed S=16 / C=256 / M=7168
shapes) is captured but not implemented.

### Lower Bound (Minimum Acceptable Scope)
A single sequence-tiled kernel (either D1b or D1a) that eliminates the spill
(`HBMrd ≈ 16 MB`), passes the 5-seed rel-L2 gate, achieves ≥ 1.5× over the
baseline, is promoted as the new phase-2 best, and is recorded with its profiler
digest in `benchmark.csv` and `candidates.jsonl`. None of D2/D3/D4 is required if
the iteration budget or evidence does not justify them.

### Allowed Choices
- Can use: `nl.sequential_range` and `nl.static_range` for the carry loop;
  `nki.isa.tensor_tensor_scan` with a `[P,1]` or array-slice `initial`; either the
  seq-tile-outer (`scan_state` array) or seq-tile-inner (`scan_init` `[128,1]`)
  structure; seq-tile widths that divide 7168 exactly (256, 512, 1024, and
  optionally 384/768 only if the sweep shows tile-size sensitivity); `broadcast_to`
  for `b`/`c`; hoisting channel-independent broadcasts; stride-0 DMA broadcast
  (`dma_broadcast` / access-pattern) for D3b; hoisting the state-independent
  `deltaU`.
- Cannot use: `nl.affine_range` on any loop that carries the scan `initial`
  (silently corrupts the recurrence); any change to the op sequence or numerics
  (AC-3); precision reductions below fp32; promotion on `--fast` alone; editing the
  benchmark definition under `../../AccelOpt/NKIBench/{kernels,reference,seeds,
  summary.json}` or hand-tuning the baseline; writing kernel sources outside
  `runs/`.

> **Note on Deterministic Designs**: The D1 primary lever is highly determined —
> it must reproduce one of two specific, measured reference structures with a
> fixed carry-loop-kind constraint. Within D1, the only real choice is D1b vs D1a
> (resolved by measuring both). The ceiling-raisers (D2/D3/D4) are genuinely
> optional and evidence-gated, so the upper and lower bounds diverge there.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are
> conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

D1b (lead — mirrors `profile/refs/ref_optimized.py`, the 1.608× reference):

```
seq_tile = 512;  n_seq_tile = 7168 / 512 = 14   (exact)
scan_state = zeros([n_channel_tile, 128, S])                 # carried across tiles
for i_seq_tile in sequential_range(n_seq_tile):              # OUTER, loop-carried
    scanC_accum = zeros([n_channel_tile, 128, seq_tile])     # live set = seq_tile wide (2 KB/part)
    for i_channel_tile in affine_range(n_channel_tile):      # 2 tiles
        delta_i = load delta[cs:cs+128, seq_start:seq_end]   # once per (tile, seq_tile)
        u_i     = load u[cs:cs+128, seq_start:seq_end]
        for i_state in affine_range(S):                      # 16
            A_i     = load a[cs:cs+128, i_state]
            deltaA  = activation(exp, delta_i, scale=A_i)     # Scalar engine
            B_i     = load b[i_state:i_state+1, seq_start:seq_end]
            deltaU  = tensor_tensor(delta_i, u_i, multiply)
            deltaBu = tensor_tensor(deltaU, B_i.broadcast_to((128,seq_tile)), multiply)
            init    = scan_state[i_channel_tile,:,i_state] if i_seq_tile>0 else 0
            scan_res= tensor_tensor_scan(deltaA, deltaBu, initial=init, mul, add)
            if i_seq_tile < n_seq_tile-1:                     # carry the LAST column forward
                scan_state[i_channel_tile,:,i_state:i_state+1] = scan_res[:, seq_tile-1:seq_tile]
            C_i     = load c[i_state:i_state+1, seq_start:seq_end]
            scanC   = tensor_tensor(scan_res, C_i.broadcast_to((128,seq_tile)), multiply)
            scanC_accum[i_channel_tile] += scanC
    store output[:, seq_start:seq_end] = scanC_accum          # per seq-tile slice
```

D1a (fallback — mirrors `profile/refs/ref_v3.py`, 1.504×): channels outer, states
next, seq-tile INNER over `static_range(n_seq_tile)` with a per-state
`scan_init = zeros([128,1])` updated by `scan_init[...] = scan_res[:, seq_tile-1]`.
Note the AccelOpt source comment: `sequential_range` on the INNER seq loop gave
wrong answers and worse perf there, which is exactly why D1a uses `static_range`
(full unroll) and D1b uses `sequential_range` on the OUTER loop.

D3a (ceiling-raiser): `b[s,:]` / `c[s,:]` are channel-independent, so broadcast
them once per (seq-tile, state) and reuse across both channel tiles instead of
re-broadcasting inside the channel loop — halving the implicit broadcast matmuls
in principle. Keep only if latency drops (the compiler may already CSE this).

D2/D4/D3b: sweep `seq_tile ∈ {256, 1024}` (512 anchor) on the D1 winner; hoist
`deltaU = delta·u` out of the state loop (it does not depend on `s`); and, last
and only if PE stays the clear limiter, load `b`/`c` via a stride-0 partition
access pattern into a `[128, seq_tile]` tile to spend idle DMA instead of PE.

### Relevant References
- `runs/mamba_v1.py` — phase-1 kernel (the whole-M structure to replace).
- `profile/refs/ref_optimized.py` — the 1.608× D1b target structure (seq-tile
  outer, `sequential_range`, `[n_ct,128,S]` `scan_state`).
- `profile/refs/ref_v3.py` — the 1.504× D1a fallback structure (seq-tile inner,
  `static_range`, `[128,1]` carry; carries the AccelOpt `sequential_range`
  warning comment).
- `profile/refs/README.md` — the measured reference sweep table + diagnosis.
- `AccelOpt/NKIBench/reference/mamba_M7168_C256_S16_numpy_1.py` — the numpy oracle
  (identity `transform_to_nki_inputs`; math settled).
- `adapter/nkibench_case.py` — seeds `[0,21,42,63,84]`, mamba rel-tol 3e-5,
  compiler flags `--disable-dge --logical-nc-config=1`.
- `verify.py` — scoring harness (`--op mamba --candidate runs/<f>.py [--fast]`).
- `benchmark.csv`, `candidates.jsonl` — evidence ledgers (DAG parent links).

## Dependencies and Sequence

### Milestones

1. **Milestone 1 — D1 sequence tiling (primary; targets AC-1, AC-2, AC-3, AC-4).**
   Kills the spill, lands ≥ 1.5× with a 5-seed pass. This is the phase's floor.
   - Phase A: Implement D1b (seq-tile outer, `sequential_range`, `scan_state`);
     score seed-42 `--fast`; read the digest.
   - Phase B: If D1b passes and reproduces the ≈0.78 ms / 16 MB / PE≈99/Vec≈89
     envelope on 5 seeds, promote it. Regardless, also implement and measure D1a
     (seq-tile inner, `static_range`, `[128,1]` carry) at least once; promote the
     faster variant that passes 5 seeds. If D1b misbehaves (correctness or perf),
     D1a is the guaranteed fallback.
2. **Milestone 2 — Ceiling-raisers on top of the D1 winner (opportunistic;
   targets AC-5).** Only pursued within the remaining iteration budget; each gated
   on a full 5-seed pass + a latency improvement above noise.
   - Step 1: D3a — hoist `b`/`c` broadcast out of the channel loop. Keep only if
     latency drops (PE-active-time as directional proxy; DMA % guardrail).
   - Step 2: D2 — sweep `seq_tile ∈ {256, 1024}` against the current winner; keep
     the best.
   - Step 3: D4 — hoist state-independent `deltaU = delta·u` (moved ahead of D3b;
     lower risk).
   - Step 4: D3b — stride-0 DMA broadcast, LAST, and only if PE remains the clear
     limiter after D3a+D4 and DMA has headroom; else not pursued.
3. **Milestone 3 — Harden and preview phase 3 (targets AC-1, AC-4).** Full 5-seed
   `verify.py` on the final best; ensure `benchmark.csv` / `candidates.jsonl` are
   complete; record a phase-3 preview (unroll the 16-state loop, hard-tune
   seq-tile width for the fixed shapes) without implementing it.

Dependency summary: Milestone 1 gates everything (no ceiling-raiser is meaningful
until the spill is killed and a stable D1 winner exists). Within Milestone 2, D3a
→ D2 → D4 → D3b is ordered by confidence/risk, and D3b additionally depends on the
post-D3a+D4 digest still showing PE as the limiter. Milestone 3 depends on the
final selected candidate.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Implement D1b: seq-tile outer, `sequential_range`, `[n_ct,128,S]` carried `scan_state`, per-seq-tile store; seq_tile=512 | AC-2, AC-2.1, AC-3 | coding | - |
| task2 | Score task1 on seed-42 `--fast`, read digest (PE/Vec/DMA/HBMrd) | AC-1, AC-4 | coding | task1 |
| task3 | Implement D1a: seq-tile inner, `static_range`, `[128,1]` `scan_init` carry | AC-2, AC-2.2, AC-2.3, AC-3 | coding | - |
| task4 | Score task3 on seed-42 `--fast`, read digest | AC-1, AC-4 | coding | task3 |
| task5 | Run full 5-seed `verify.py` (drop `--fast`) on the faster passing D1 variant; promote it as phase-2 best; record in benchmark.csv + candidates.jsonl | AC-1, AC-2, AC-4 | coding | task2, task4 |
| task6 | Implement D3a: hoist `b`/`c` broadcast out of the channel loop on top of the D1 winner | AC-5 | coding | task5 |
| task7 | Score D3a; decide keep/drop via latency-primary + PE-active-time proxy + DMA guardrail; if no-op, drop | AC-5, AC-1, AC-4 | coding | task6 |
| task8 | D2 sweep `seq_tile ∈ {256, 1024}` (512 anchor) against current winner; keep best | AC-4 | coding | task5 |
| task9 | Implement D4: hoist state-independent `deltaU = delta·u` out of the state loop | AC-3, AC-4 | coding | task8 |
| task10 | Analyze post-D3a+D4 digest: is PE still the clear limiter with DMA headroom to justify D3b? | AC-5 | analyze | task7, task9 |
| task11 | Implement D3b (stride-0 DMA broadcast) ONLY if task10 says yes; verify PE falls, DMA stays unsaturated, 5-seed passes | AC-5, AC-1 | coding | task10 |
| task12 | Harden: full 5-seed run on final best, complete evidence ledgers, write phase-3 preview note | AC-1, AC-4 | coding | task5 |

## Claude-Codex Deliberation

### Agreements
- D1 sequence tiling is the correct primary lever; it reproduces two measured,
  correctness-passing references and kills the spill without touching the math.
- The 5-seed rel-L2 gate is mandatory before any promotion; `--fast` (seed 42)
  is only a fast pre-check, never a promotion basis.
- D3/D4 are opportunistic ceiling-raisers, not phase gates. Beating 1.608× is not
  a reliable planning assumption; landing D1-level 1.5–1.6× is the realistic goal.
- The carry loop must be `sequential_range` (D1b) or `static_range` (D1a), never
  `affine_range`; carry extraction must preserve `[128,1]` shape (D1a) and
  `(channel-tile, state)` identity (D1b).
- D4 (hoist `deltaU`) is lower-risk than D3b and should precede it.

### Resolved Disagreements
- **D3 keep/drop metric (Codex REQUIRED_CHANGE, accepted):** Claude's draft gated
  D3a/D3b on "PE active % must drop." Codex noted this is confounded — PE % can
  stay ~99% even if total PE work halves (the shorter kernel is still PE-saturated
  for its whole runtime), and PE % can drop while latency doesn't improve.
  Resolution: **latency is the primary keep/drop metric**, with PE-active-*time*
  (`latency_ms × PE_active_fraction`) as the directional proxy and DMA % as a
  guardrail. Correctness (5-seed) is always mandatory. Encoded in AC-5.
- **D1b-else-D1a binary vs run-both (Codex REQUIRED_CHANGE, accepted):** Claude's
  revised draft made acceptance binary (D1b, fall back to D1a only on failure).
  Codex argued for measuring BOTH D1a and D1b at least once, since local
  compiler/version effects could invert the reference ranking and the cost of
  running both is low. Resolution: **run both at least once, promote the faster
  passing variant** (D1b still leads implementation as the best-measured ref).
  Encoded in AC-2.1 / AC-2.2.
- **D3b ordering (Codex first-pass, accepted):** moved D3b after D4 and behind an
  explicit "PE still the clear limiter + DMA headroom" gate, reflecting its higher
  compile/correctness/descriptor risk.
- **Verifiability of PE-work reduction (Codex UNRESOLVED, accepted as a known
  limit):** this harness has no IR/BIR access, so a true `nc_matmul`-count
  reduction from D3a/D3b cannot be proven — only performance-counter correlation.
  The plan therefore claims correlation (latency + PE-active-time), not verified
  matmul-count reduction, and drops D3a if latency is unmoved.

### Convergence Status
- Final Status: `converged` (2 Codex passes; all REQUIRED_CHANGES incorporated;
  no opposing positions remain open).

## Pending User Decisions

_None._ The one quantitative distinction the workflow flags — whether the ~1.6× /
0.78 ms figure is a hard requirement or a directional target — is already resolved
by the draft itself and encoded in the ACs: the **correctness gate (rel-L2 3e-5,
5 seeds) is a HARD requirement (AC-1)**; the **≥1.5× D1 speedup is a hard success
target (AC-2)** because it reproduces two measured references; and **beating
1.608× via D3 is an opportunistic direction (AC-5)**, not a phase gate. No
Claude/Codex disagreement was left unresolved (see Deliberation), so there is
nothing requiring an explicit user decision before implementation.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such
  as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `scan_state`,
  `scan_init`, `seq_tile_size`, `i_seq_tile`, `deltaU`, `B_i_bcast`), matching the
  existing `runs/mamba_v1.py` and reference-kernel conventions.
- Kernel `.py` sources go under `runs/` (tracked); other run artifacts and all of
  `profile/` are git-ignored. Record evidence in `benchmark.csv` and
  `candidates.jsonl` (DAG parent links), not in the plan.

--- Original Design Draft Start ---

# mamba (M7168 C256 S16, fp32) — Phase 2 implementation draft (profile-driven)

## Starting point and the phase-1 puzzle

Phase 1 (`runs/mamba_v1.py`, commit 719684c) produced the first correct kernel but
it is **0.832× — slower than the baseline** (1.5116 ms vs 1.258 ms). That regression
is the whole content of phase 2: the phase-1 "load delta/u once instead of 16×"
hygiene was supposed to help and instead hurt. Phase 1's own digest already named the
two suspects — `HBMrd=72MB` (4.5× the ~16 MB read-once floor → **compiler spilling**)
and `PE=52% / Vec=51%` co-limit (**not** the pure-Vector-bound story the phase-1 draft
hypothesized). Phase 2 confirms both with direct measurement and fixes them.

Correctness gate: relative-L2 `< 3e-5 · ||v_r||` on seeds `[0,21,42,63,84]`, fp32.
The math is settled and unchanged from phase 1 (numpy oracle rel-L2 4.08e-7); every
phase-2 candidate keeps the identical op sequence, only the **loop/tile structure**
changes. Never regress correctness.

## Measured evidence (the profile-driven core of this phase)

I profiled the AccelOpt `samples/nki/mamba_*.py` reference kernels through our own
`verify.py` path (seed 42, `--fast`) to get real per-engine breakdowns for the ~1.6×
target the prompt cites. Evidence saved under `profile/refs/` (gitignored copies +
`README.md`):

| kernel | structure | latency | speedup | PE | Vec | Scl | DMA | HBMrd |
|--------|-----------|---------|---------|----|----|----|----|-------|
| baseline | state-outer, delta/u reloaded 16× | 1.2583 ms | 1.000× | — | — | — | — | — |
| **our v1** (= AccelOpt `mamba_v2`) | channels-outer, load-once, **whole-M** scan | 1.5116 ms | 0.832× | 52% | 51% | 16% | 11% | **72 MB** |
| AccelOpt `mamba_v3` | seq-tile **inner** (`static_range` 512), `[128,1]` carry | 0.8367 ms | 1.504× | 93% | 82% | 22% | 13% | 16 MB |
| AccelOpt `mamba_optimized` | seq-tile **outer** (`sequential_range` 512), carried `scan_state` | **0.7823 ms** | **1.608×** | **99%** | 89% | 23% | 11% | 16 MB |

Three facts fall straight out of this table:

1. **Our v1 reproduces `mamba_v2` to the digit** (1.5116 ms, PE52/Vec51/72MB). The
   regression is real and understood, not measurement noise.

2. **One lever explains the entire 1.51 → 0.78 ms gap: sequence tiling** (chunk the
   M=7168 free axis into 512-wide tiles, carry the scan's last column forward as the
   next tile's `initial`). It changes *nothing* about the math and yet:
   - **kills the spill**: `HBMrd` 72 MB → **16 MB** (the read-once floor).
   - **unlocks pipelining**: PE/Vec 52/51% → **99/89%**.

3. **At 1.6× the bottleneck engine is the Tensor engine (PE=99%) — and mamba has no
   matmul.** This is the ceiling-raising lever for beating the reference (see §"Beyond
   1.6×").

### Why whole-M spills — SBUF arithmetic (trn2, 208 KB/partition usable)

A `[128, 7168]` fp32 tile is **28.0 KB/partition**. v1 holds live across the 16-state
loop: `delta_i`, `u_i`, `scanC_accum` (3 × 28 = 84 KB) **plus** the per-state
temporaries `deltaA`, `deltaBu`, `scan_res`, `scanC` (~4 × 28 = 112 KB) → **~196 KB
peak, right at the 208 KB usable limit.** The allocator has no slack for
double-buffering, so it spills intermediates to HBM → the 72 MB read traffic. Sequence
tiling at 512 shrinks every tile to **2.0 KB/partition**; the whole hoisted working set
is ~20–45 KB, leaving ample room for the compiler to double-buffer (which is what turns
PE/Vec from 51% into 89–99%).

### Why PE is busy at all — partition-broadcast is a hidden matmul

Confirmed via the NKI arch guide + the NKI kernel library source: `broadcast_to((128, M))`
of a `[1, M]` SBUF row **across the partition dimension** is lowered to
`nc_matmul(ones[128,1], row[1,M])` on the **Tensor engine** (result lands in PSUM).
mamba broadcasts `b[s,:]` and `c[s,:]` this way once per state. Those implicit matmuls
are the *only* PE work in the kernel, and they sit in a serial dependency with the
Vector multiplies that consume them — exactly the "PE≈Vec≈50%, neither saturated"
signature v1 shows, and the "PE=99%" ceiling `mamba_optimized` hits once pipelining
removes the stalls. This is what §"Beyond 1.6×" attacks.

## Ranked optimization directions

Ranked by expected benefit × confidence. Each is a ≤5-iteration exploration with
before/after `verify.py` latency + the profiler digest as the keep/revise/reject gate.

### D1 — Sequence tiling with carried scan state  ⭐ (primary; ~1.5–1.6× expected)

**What.** Chunk M=7168 into `seq_tile` (start 512; 7168/512 = 14, exact) tiles. Keep
channels-outer / state-inner and load `delta`/`u` once per channel tile (the phase-1
hygiene is *correct*, it was just starved of SBUF). Per state, scan each seq-tile with
`initial = scan_init` where `scan_init` is the previous tile's last column, a
per-partition `[128,1]` carried state. `nki.isa.tensor_tensor_scan` accepts a `[P,1]`
tile as `initial` (documented; = the `result[:,i-1]` fed into column 0).

**Two sub-variants, both measured on the references above — try both:**
- **D1a (= `mamba_v3` shape): seq-tile INNER, `static_range`.** State loop outside,
  seq-tile loop inside with a per-state `scan_init` carried by `static_range`
  (the AccelOpt comment warns `sequential_range` here gave *wrong* answers and worse
  perf; `static_range` fully unrolls the 14 tiles). Measured 1.504×.
- **D1b (= `mamba_optimized` shape): seq-tile OUTER, `sequential_range`.** Seq-tile
  loop outermost over a `[n_channel_tile, 128, S]` `scan_state` array; channels then
  states inside; store each seq-tile's output slice directly (no whole-M accumulator).
  Measured **1.608×** — the best reference. The outer `sequential_range` cleanly
  expresses the cross-tile carry and keeps the live accumulator at `seq_tile` width.

**Why it wins.** Kills the spill (28 KB → 2 KB tiles) and enables double-buffering.
**Risk.** Low — it's the exact structure of two measured, correctness-passing references;
same op sequence, same scan primitive. **Correctness watch:** the carried `initial`
introduces a loop-carried dependency, so the carry loop must be `sequential_range`
(D1b) or `static_range` (D1a), never `affine_range` — getting this wrong silently
corrupts the scan. Validate on seed 42 first, then full 5-seed before promoting.

### D2 — Tune the seq-tile width (cheap sweep on top of D1's winner)

**What.** Once D1's structure is chosen, sweep `seq_tile ∈ {256, 512, 1024}` (all
divide 7168: 28/14/7 tiles). 512 is AccelOpt's "magic number"; our silu task found
**finer** free-axis tiling beat wider (optimum ~4 KB/partition burst) — so 256 (1 KB)
is worth a shot, and 1024 (4 KB) brackets the other side. Pure knob turn, no structural
change.

**Why.** The optimum balances pipeline depth (more tiles = more overlap) against
per-tile fixed overhead + carry-dependency serialization. **Benefit** likely a few %.
**Risk** minimal. Rank second because it only refines D1.

### D3 — Move the b/c partition-broadcast OFF the Tensor engine  ⭐ (ceiling-raiser; beat 1.6×)

**What.** At 1.6× the kernel is **PE=99%-bound on the implicit broadcast matmuls**,
while **DMA sits at 11%**. Relocate the `b`/`c` partition-broadcast off the saturated
PE. Three costed options (from the API research), to try in order of confidence:

- **D3a — Hoist the broadcast out of the channel loop.** `b[s,:]` and `c[s,:]` are
  **channel-independent** — but v1/references re-broadcast them inside the 2-iteration
  channel loop. Broadcasting once per (seq-tile, state) and reusing across both channel
  tiles **halves** the broadcast matmuls (896 → 448, ≈ −50% PE work). Pure hoist,
  obviously correct, no new API. **Highest-confidence ceiling raiser.**
- **D3b — Partition-stride-0 DMA broadcast.** Load `b[s,:]`/`c[s,:]` directly into a
  `[128, seq_tile]` SBUF tile via a stride-0 partition access pattern
  (`src.ap(pattern=[[0,128],[1,seq_tile]])` → `dma_copy`), spending idle DMA bandwidth
  instead of PE. Real technique (the NKI kernel library `dma_broadcast`); caveat is extra
  descriptor traffic (128 partitions), but DMA has huge headroom here.
- **D3c — `nc_stream_shuffle` broadcast on VectorE.** Keeps it off PE at the cost of
  Vector cycles; only attractive if Vec has more slack than DMA after D1 (Vec is 89%,
  so likely *not* — rank D3c last).

**Why.** This is the only lever that can push past the 1.608× reference, because that
reference is PE-bound. **Risk** moderate: D3a is safe; D3b/D3c change how a tile is
materialized and need a correctness re-check + a digest read to confirm PE actually
drops without a new bottleneck appearing. Gated on D1 landing first.

### D4 — Hoist state-independent `deltaU = delta·u` out of the state loop (minor)

**What.** `deltaU = delta_i · u_i` does **not** depend on the state `s`, yet it is
recomputed inside the 16-state loop (1 of ~5 Vector passes/state). Compute it once per
(channel tile / seq-tile) and reuse. Removes ~20% of the Vector `tensor_tensor` work.

**Why / risk.** Correct and cheap. But it only helps if the kernel is Vector-bound
*after* D1+D3 — at PE=99% the Vector engine is the *second* bottleneck (89%), so this is
a follow-on, not a lead. It also adds one live `[128, seq_tile]` buffer (trivial at
2 KB). Rank low; revisit only if a post-D3 digest shows Vec back on top. **Note:** this
is the phase-1 draft's "biggest vector lever" — the measurement demotes it, which is
the point of a profile-driven phase.

## Plan of attack (≤5 iterations)

1. **Iter 1 — D1b** (`mamba_optimized` shape: seq-tile outer, `sequential_range`,
   carried `scan_state`, seq_tile=512). Expected ≈1.6×. Score seed-42 `--fast`; read
   the digest; if it reproduces PE≈99/Vec≈89/16MB, **promote** as the new phase-2 best.
   Fallback D1a (`static_range` inner) if the outer `sequential_range` misbehaves.
2. **Iter 2 — D3a** (hoist b/c broadcast out of the channel loop) on top of D1b. Target
   PE < 99%, latency < 0.78 ms. Keep only if the digest confirms PE drops.
3. **Iter 3 — D2** seq-tile sweep {256, 1024} against the D1b(+D3a) winner. Keep best.
4. **Iter 4 — D3b** (stride-0 DMA broadcast) *if* D3a left PE still on top; compare
   against the DMA-active digit.
5. **Iter 5 — D4** *only if* a digest shows Vector back as the bottleneck; else spend
   the iteration hardening the current best (full 5-seed run + shape-specialization
   preview for phase 3).

Record every candidate in `benchmark.csv` + `candidates.jsonl` (DAG parent links),
profiling evidence under `profile/`. Full 5-seed `verify.py` (drop `--fast`) before any
promotion. Phase-3 preview: S=16, C=256 (exactly 2 channel tiles), M=7168 are all fixed
— unroll the state loop and hard-tune the seq-tile width once the structure is locked.

## Risks / watch-items

- **Carry-dependency loop kind.** The carried `initial` makes the seq-tile loop
  loop-carried → must be `sequential_range`/`static_range`, never `affine_range`.
  Mis-choosing silently corrupts the scan (AccelOpt hit exactly this). First check on
  every D1 variant: seed-42 rel-L2 passes.
- **`scan_state` extraction.** Carrying "the last column" (`scan_res[:, seq_tile-1]`)
  into a `[128,1]` slice must land at partition-aligned `[P,1]`; verify against the
  reference indexing (`scan_init[...] = scan_res[0:128, seq_tile-1]`).
- **D3b descriptor cost.** Stride-0 DMA into `[128, seq_tile]` still emits ≥128
  descriptors; confirm the DMA digit stays well below saturation and that PE actually
  falls — otherwise revert to D3a.
- **Don't over-fit `--fast`.** All reference numbers above are seed-42/fast; confirm the
  ranking holds on the full 5-seed / higher-iter run before promoting, as other tasks
  showed fast-mode can mis-rank close candidates.

--- Original Design Draft End ---
