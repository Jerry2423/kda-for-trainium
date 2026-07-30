# mamba (M7168 C256 S16, fp32) — Phase 3 Plan: Land the Sequence-Tiling Floor, then Specialize the Fixed Shapes

## Goal Description

Phase 3 is nominally "shape specialization," but the honest workspace state forces a
two-part phase. **The only kernel in `runs/` is `runs/mamba_v1.py` at 0.832× — SLOWER
than the 1.258 ms baseline** (`candidates.jsonl` has one node; `benchmark.csv` one row).
**Phase 2 designed the sequence-tiling win (measured 1.5–1.6× on reference kernels
through this harness) but never landed a kernel** — its RLCR loop was blocked by a stale
phase-1 lock. So the proven 1.6× structure has no `runs/` artifact.

Therefore phase 3 must **(A) first land the proven sequence-tiling lever (S0) as the new
promoted baseline `mamba_v2`** (this is not re-litigation — it is the evidence-verified
~1.6× structure that simply was never written to code), then **(B) specialize the fixed
M=7168 / C=256 / S=16 shapes on top of it**: relocate the `b`/`c` partition-broadcast off
the saturated Tensor engine (S1), sweep the seq-tile regime for the fixed M (S2), relieve
the Vector engine via op fusion + hoisting (S3), and static-unroll the compile-time-
constant loops (S4).

The structural analysis is unambiguous. Every axis divides evenly (C=256 = exactly 2
tiles of 128; M=7168 = 2¹⁰·7 divides cleanly by {128,256,448,512,896,1024,1792}; S=16
fixed), so there are **no edge tiles to specialize**. At the ~1.6× point the bottleneck
engine is the **Tensor engine (PE≈99%)** even though mamba has no matmul — the
partition-dim `broadcast_to((128, seq_tile))` of `b[s,:]`/`c[s,:]` is lowered to
`nc_matmul(ones, row)` on the PE, and those broadcast matmuls are the only PE work; they
saturate it while **DMA idles at 11%**. Specialization = pick the seq-tile regime that
minimizes PE pressure, move the broadcast off the PE, and exploit the constant S=16 /
2-channel loops.

Correctness must never regress: the relative-L2 gate is `||v_k − v_r||₂ < 3e-5 · ||v_r||₂`
on seeds `[0,21,42,63,84]`, fp32 (mamba's looser 3e-5, per `adapter/nkibench_case.py`),
enforced by `verify.py`'s `l2_norm_passed`. The numpy oracle for this exact op sequence is
rel-L2 4.08e-7 — ~75× inside the gate. Every phase-3 candidate keeps fp32 and the identical
**mathematical recurrence and value-level semantics** (`activation(exp,scale=A)`,
`tensor_tensor(delta,u)`, `tensor_tensor(·,b)`, `tensor_tensor_scan`, `tensor_tensor(·,c)`,
accumulate over states); only loop/tile structure, broadcast *mechanism*, and op *fusion*
(fusing adjacent ops that compute the identical result) change.

## Quantitative Thresholds (referenced by the criteria below)

- **Correctness (HARD):** rel-L2 `< 3e-5 · ||v_r||` on all 5 seeds; expected regime ~1e-6.
  This is a pass/fail gate, never a trend.
- **S0 success (HARD):** speedup ≥ 1.5× AND spill eliminated. "Spill eliminated" = `HBMrd`
  returns near the 16 MB read-once floor; concretely `HBMrd ≤ 20 MB` (vs the 72 MB spill).
- **Promotion "clears noise" (directional heuristic, not a hard gate):** a candidate is
  promoted over its parent only if it improves latency by a margin larger than run-to-run
  jitter — as a rule of thumb ≳ 3% on the full 5-seed run, or a delta clearly outside a
  same-session control band. A sub-noise delta ⇒ keep the parent / prefer the simpler,
  more-stable structure. Never promote on `--fast` (seed 42) alone.
- **S1b guardrails (directional):** `HBMrd` must stay near the floor (must NOT balloon
  toward the ~100+ MB a naive per-partition broadcast could imply — reject if it rises
  materially above ~25 MB), and DMA must NOT become the new saturating bottleneck. Concretely,
  reject S1b if `DMA_active% ≥ 85%` (DMA is now the near-saturated engine), OR if DMA becomes
  the single highest-active engine of the digest without BOTH a latency improvement beyond
  noise AND a PE-active-time drop.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. "Latency", "speedup", `HBMrd`, and the per-engine active %
(`PE`, `Vec`, `Scl`, `DMA`) all refer to values surfaced by `verify.py` / the remote
profiler digest; speedup = `1.258274 ms / candidate_ms`.

- AC-1: **Correctness is never regressed.** Every recorded or promoted candidate that is
  measured passes the NKIBench relative-L2 gate (`< 3e-5 · ||v_r||`) on the seeds it is run
  against, and no candidate is *promoted* without a full 5-seed pass in fp32, verified by
  `verify.py` (`l2_norm_passed`). HARD requirement, not a trend.
  - Positive Tests (expected to PASS):
    - A full-seed `verify.py` run (no `--fast`) on the promoted kernel passes all five seeds
      `[0,21,42,63,84]`.
    - A `--fast` (seed-42) screen on any candidate under active iteration passes before it is
      considered for a full-seed run.
  - Negative Tests (expected to FAIL / be rejected):
    - Any candidate whose carry loop uses `affine_range` (breaking the loop-carried scan
      dependency) fails rel-L2 on ≥1 seed and is NOT recorded as passing or promoted.
    - A candidate promoted on `--fast` (seed 42) alone without a full 5-seed confirmation is
      rejected by the process gate even if seed 42 passes.
- AC-2: **S0 (sequence tiling) lands as the new promoted baseline `mamba_v2` at ≥1.5× with
  a full 5-seed pass and the spill eliminated (`HBMrd ≤ 20 MB`).** This is the phase's HARD
  success gate (it reproduces two measured references, so it is a hard expectation, not a
  stretch).
  - Positive Tests (expected to PASS):
    - The promoted `mamba_v2` passes 5-seed rel-L2, reports latency in the reference band
      (≈0.78–0.84 ms), speedup ≥1.5×, and `HBMrd ≤ 20 MB` (not the 72 MB spill).
    - `candidates.jsonl` gains a `mamba_v2` node with `parent = mamba_v1` and its digest.
  - Negative Tests (expected to FAIL / be rejected):
    - A candidate that keeps the whole-M structure (`HBMrd` stays ≈72 MB) or does not reach
      ≥1.5× is NOT promoted as `mamba_v2`.
  - AC-2.1: **S0-lead (D1b) implemented** — mirror `profile/refs/ref_optimized.py` (1.608×):
    seq-tile OUTER over `nl.sequential_range(n_seq_tile)`, a `[n_channel_tile, 128, S]`
    `scan_state` carried across tiles, channels then states inside, output stored per
    seq-tile slice (no whole-M accumulator); the scan `initial` for tiles > 0 is the
    array-slice `scan_state[i_ct, 0:128, i_state]` (the exact known-good form used by
    `ref_optimized.py`), else scalar `0`; after each scan write `scan_res[:, seq_tile-1:seq_tile]`
    back into the `[i_ct, 0:128, i_state:i_state+1]` slot.
    - Positive: D1b compiles, passes 5-seed rel-L2, digest shows `HBMrd ≤ 20 MB` and PE/Vec
      materially above phase-1's 51/51% (target envelope ≈ PE 99 / Vec 89).
    - Negative: If D1b's outer `sequential_range` reproduces AccelOpt's "wrong-answer /
      worse-perf" failure (fails rel-L2, `HBMrd` stays high, or latency does not improve), it
      is NOT promoted; the process falls through to AC-2.2.
  - AC-2.2: **S0-fallback (D1a) implemented and measured** — mirror `profile/refs/ref_v3.py`
    (1.504×): seq-tile INNER over `nl.static_range(n_seq_tile)` (fully unrolled), per-state
    `scan_init = zeros([128,1])` carried via the ref idiom `scan_init[...] =
    scan_res[0:128, seq_tile-1]` (the `[...]` write preserves the `[128,1]` buffer; the RHS
    is the tile's last column). D1a is run at least once regardless of D1b's outcome (local
    compiler/version effects could invert the reference ranking; running both is cheap). The
    **faster passing variant** is promoted as `mamba_v2`; if latencies tie within noise,
    prefer the simpler/more-stable structure.
    - Positive: D1a compiles, passes 5-seed rel-L2, digest shows `HBMrd ≤ 20 MB`; the faster
      of {D1a, D1b} that passes 5 seeds is promoted.
    - Negative: A variant that fails 5-seed correctness is never selected even if faster.
  - AC-2.3: **Carry extraction preserves identity.** The carried state is exactly the
    previous tile's final scan column, shape-preserved: D1a writes the last column into a
    `[128,1]` `scan_init` via `scan_init[...] = scan_res[0:128, seq_tile-1]` (not from the
    wrong axis); D1b writes state `s`'s final value into `[i_channel_tile, 0:128,
    i_state:i_state+1]` without cross-mixing the 2 channel tiles or 16 states.
    - Positive: 5-seed rel-L2 passes (a corrupted carry would exceed 3e-5 given the 4.08e-7
      oracle margin), confirming carry indexing is correct.
    - Negative: A carry that indexes the wrong `(channel-tile, state)` slot or the wrong
      column produces a rel-L2 failure and is rejected.
- AC-3: **The mathematical recurrence and value-level semantics are unchanged from phase 1.**
  Every phase-3 candidate computes the identical result values via the same recurrence
  (`deltaA = exp(A·delta)`, `deltaBu = (delta·u)·b`, scan, `scanC = scan·c`, sum over
  states). Only loop order, tile width, broadcast *mechanism*, and op *fusion* change.
  Fusing adjacent operations that compute the identical result (e.g. folding
  `scanC = scan·c` and its accumulation into one `scalar_tensor_tensor` pass) IS allowed;
  the state-accumulation order and fp32 precision are preserved.
  - Positive: A diff against `runs/mamba_v1.py` shows only loop/tile/broadcast-mechanism/
    fusion changes; rel-L2 stays in the ~1e-6 regime (not merely under 3e-5).
  - Negative: A candidate that alters the result semantics (different scan formulation,
    reduced precision below fp32, or reordered non-associative math across the state
    reduction) is out of scope even if it happens to pass.
- AC-4: **Each recorded candidate is backed by profiler evidence.** Every perf change is
  appended to `benchmark.csv`; every candidate is a node in `candidates.jsonl` with DAG
  `parent` links; the per-engine digest (`PE/Vec/Scl/DMA %`, `HBMrd/HBMwr`, latency) is
  captured. `--fast` seed-42 screens MAY be recorded as evidence rows explicitly marked as
  screens (`seeds=[42]`, verdict `screen`), and are NOT required to carry a full 5-seed
  pass; only rows with verdict `promoted` require the full 5-seed run. Kernel `.py` sources
  live under `runs/` (tracked).
  - Positive: After the phase, `benchmark.csv` / `candidates.jsonl` contain one row/node per
    measured candidate with parent links forming a DAG rooted at `mamba_v1`, each carrying
    its digest; promoted nodes carry a 5-seed pass, screen rows are labelled as such.
  - Negative: A candidate that changed latency but left the ledgers un-updated, or whose
    `.py` was written outside `runs/`, or a `promoted` node lacking a 5-seed pass, is a
    process failure.
- AC-5 (opportunistic, NOT a phase gate): **S1 — move the `b`/`c` partition-broadcast off
  the Tensor engine — is attempted and kept only on measured evidence.** Because the harness
  exposes only per-engine active % (no IR / no `nc_matmul` counts), the keep/drop decision
  uses **latency as the primary metric**, with **PE-active-*time* (`latency_ms ×
  PE_active_fraction`)** as the directional proxy (PE% alone is confounded — it can stay
  ~99% even if total PE work halves) and **DMA % plus `HBMrd`** as guardrails.
  - AC-5.1: **S1a — hoist the broadcast out of the 2-channel loop.** `b[s,:]`/`c[s,:]` are
    channel-independent; broadcast once per (seq-tile, state) and reuse across both channel
    tiles. Because `broadcast_to` may be a lazy view re-emitted per consumer, the hoist may
    require explicitly materializing `B_i_bcast`/`C_i_bcast` into an SBUF tile once to
    actually reduce PE work. (Note: this hoist maps naturally onto D1b's loop order; if D1a
    wins S0, S1a may need loop restructuring or a larger live range to be applicable — gate
    on measurement.)
    - Positive: S1a passes 5-seed rel-L2 AND reduces latency below the S0 winner beyond
      noise, with PE-active-time dropping → kept.
    - Negative: If S1a leaves latency and PE-active-time unchanged (compiler already CSE'd /
      the hoist was a lazy-view no-op), it is recorded as a no-op and dropped; the S0 winner
      remains best.
  - AC-5.2: **S1b — stride-0 DMA-load broadcast — is a compile-feasibility PROBE first, then
    a perf lever.** Load `b[s,:]`/`c[s,:]` from HBM into a `[128, seq_tile]` SBUF tile via a
    partition-stride-0 access pattern (broadcast-view or `src.ap(pattern=[[0,128],
    [1,seq_tile]])` + `dma_copy`), spending idle DMA instead of saturated PE. S1b MUST
    compile and profile under the harness's **globally applied** compiler flags
    `--disable-dge --logical-nc-config=1` with **no per-op DGE tuning** (any `dge_mode=none`
    intent is redundant — DGE is already off globally; if the pattern relies on dynamic
    descriptor generation that `--disable-dge` forbids, S1b is infeasible and dropped in
    favor of S1a).
    - Positive: S1b compiles under the harness flags, passes 5-seed rel-L2 (broadcast
      *values* identical → rel-L2 stays ~1e-6), latency drops below the S1a/S0 winner beyond
      noise, PE-active-time falls, DMA stays below the 85% saturation guardrail, and `HBMrd`
      stays near the floor (≤ ~25 MB; not ballooning toward 100+ MB).
    - Negative: If S1b fails to compile under `--disable-dge`, or `DMA_active% ≥ 85%` /
      `HBMrd > ~25 MB` without PE-active-time falling, or latency does not improve beyond
      noise, it is dropped and S1a (or the S0 winner) remains best.
  - Reaching < 0.78 ms (beating the 1.608× reference) is the aspirational marker for S1;
    failing to beat 1.608× does NOT fail the phase.
- AC-6 (opportunistic specialization — the phase's headline shape lever): **S2 — the
  seq-tile regime is specialized for the fixed M=7168 via a selected-width sweep.** Once the
  S0/S1 winner's structure is fixed, sweep the exact divisors `seq_tile ∈ {256,448,896,1024}`
  against the 512 anchor (28/16/8/7 tiles vs 14), optionally extending to the wider `1792`
  (4 tiles) or finer `128` (56 tiles) edge if the sweep trend points there, and hard-code
  the winning width for M=7168. The hypothesis (mamba is PE-bound on broadcasts whose count
  is inversely proportional to seq_tile ⇒ **wider may win**, opposite to the silu task's
  "finer wins") is a hypothesis to test, not an assumed outcome; the sweep interacts with S1
  (if S1b moves broadcasts to DMA, descriptor count scales with `n_seq_tile`, flipping the
  pressure). Sweep S2 **after** S1 lands; if S1 fails or is a no-op, sweep S2 on the S0
  winner (do NOT block S2 on S1).
  - Positive: The sweep runs; the width that passes 5-seed rel-L2 and clears noise (without
    reintroducing the spill, `HBMrd ≤ 20 MB`) is hard-coded for M=7168 and recorded.
  - Negative: A width that reintroduces the spill (`HBMrd` back above the floor) or fails
    correctness is not selected; if no divisor beats 512 beyond noise, 512 is kept and the
    result is recorded as "512 optimal for M=7168."
- AC-7 (opportunistic, correctness-neutral): **S3/S4 secondary levers are attempted where
  the digest justifies them and kept only on a measured latency drop.**
  - AC-7.1: **S3 (Vec relief) if a post-S1 digest shows Vec as the limiter.** S3a fuses
    `scanC = scan_res·c` + accumulate into one `nisa.scalar_tensor_tensor` pass
    (data=scan_res, op0=multiply, operand0=c, op1=add, operand1=scanC_accum — both operands
    SBUF, not PSUM, per the API constraint). S3a is compile-gated: if this NKI build rejects
    the planned `c` operand shape, it is recorded as infeasible and dropped. S3b hoists the
    state-independent `deltaU = delta·u` out of the 16-state loop (recomputed 16× today;
    compute once per (channel-tile, seq-tile)); S3b is low-risk and may be folded in
    regardless of the Vec-limiter check (maps naturally onto D1b; may need restructuring
    under D1a).
    - Positive: S3a/S3b compile, pass 5-seed rel-L2 (same result values), and reduce latency
      beyond noise → kept.
    - Negative: A fusion that changes result semantics, fails to compile, or does not move
      latency beyond noise is rejected/dropped.
  - AC-7.2: **S4 (static-unroll of the compile-time-constant loops).** Replace the
    `affine_range` on the S=16 state loop and the n_channel_tile=2 channel loop with
    `nl.static_range` (full unroll) to give the compiler a longer dependency-free window.
    The seq carry loop MUST remain `sequential_range` (D1b) / `static_range` (D1a).
    - Positive: S4 passes 5-seed rel-L2 and reduces latency beyond noise → kept.
    - Negative: If unrolling increases register/code pressure and latency does not drop (or
      regresses), S4 is dropped.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
S0 landed and promoted as `mamba_v2` (≥1.5×, 5-seed pass, `HBMrd ≤ 20 MB`), then — within
the ≤5-iteration budget — S1 (S1a hoist, and S1b stride-0 DMA broadcast only where it
compiles under the harness flags and the digest confirms PE-active-time falls with DMA/HBM
headroom), S2 (the `{256,448,896,1024}` vs 512 selected-width sweep with the winner
hard-coded for M=7168), S3 (`scalar_tensor_tensor` fuse + hoist `deltaU`), and S4
(static-unroll of the S=16 / 2-channel loops) each explored, evidence-gated (latency + full
5-seed pass), and recorded in `benchmark.csv` / `candidates.jsonl` with a profiler digest.
The best passing candidate is hardened with a full 5-seed run and reported as the final
speedup. Promoted candidates are named monotonically as `mamba_vN` in the order they are
actually promoted (skipped/dropped levers do not consume a version number).

### Lower Bound (Minimum Acceptable Scope)
A single sequence-tiled kernel (D1b or D1a) that eliminates the spill (`HBMrd ≤ 20 MB`),
passes the 5-seed rel-L2 gate, achieves ≥1.5× over the baseline, is promoted as `mamba_v2`
(parent `mamba_v1`), and is recorded with its digest in `benchmark.csv` and
`candidates.jsonl`. None of S1/S2/S3/S4 is required if the iteration budget or evidence does
not justify them — S0 landing IS phase success.

### Allowed Choices
- Can use: `nl.sequential_range` / `nl.static_range` for the carry loop;
  `nki.isa.tensor_tensor_scan` with a scalar-`0`, `[P,1]` tile, or `[P]` array-slice
  `initial` (the `scan_state[i_ct,:,i_state]` array-slice form is known-good from
  `ref_optimized.py`); either the seq-tile-outer (`scan_state` array) or seq-tile-inner
  (`scan_init` `[128,1]`) structure; `broadcast_to` for `b`/`c`; hoisting/materializing
  channel-independent broadcasts (S1a); a stride-0 partition DMA broadcast via a
  broadcast-view or `src.ap(...)` + `dma_copy` (S1b) **provided it compiles under the global
  `--disable-dge --logical-nc-config=1` with no per-op DGE dependency**;
  `nisa.scalar_tensor_tensor` for the `(scan_res·c)+accum` fusion (S3a, compile-gated);
  hoisting the state-independent `deltaU = delta·u` (S3b); `nl.static_range` unroll of the
  state/channel loops (S4); seq-tile widths that divide 7168 exactly
  ({128,256,448,512,896,1024,1792}).
- Cannot use: `nl.affine_range` on any loop that carries the scan `initial` (silently
  corrupts the recurrence); any change to the result semantics or numerics (AC-3); precision
  reductions below fp32; a per-op `dge_mode` knob as a required lever (DGE is disabled
  globally); promotion on `--fast` alone or on a sub-noise latency delta; editing the
  benchmark definition under `../../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}`
  or hand-tuning the baseline; writing kernel sources outside `runs/`.

> **Note on Deterministic Designs**: S0 is highly determined — it must reproduce one of two
> specific, measured reference structures with a fixed carry-loop-kind constraint; the only
> real choice within S0 is D1b vs D1a (resolved by measuring both). The specialization
> levers S1–S4 are genuinely optional and evidence-gated, so upper and lower bounds diverge
> there.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

S0-lead D1b (mirrors `profile/refs/ref_optimized.py`, the 1.608× reference):

```
seq_tile = 512;  n_seq_tile = 7168 / 512 = 14   (exact)
scan_state = zeros([n_channel_tile, 128, S])                 # carried across tiles
for i_seq_tile in sequential_range(n_seq_tile):              # OUTER, loop-carried
    scanC_accum = zeros([n_channel_tile, 128, seq_tile])     # live set = seq_tile wide (2 KB/part)
    for i_channel_tile in affine_range(n_channel_tile):      # 2 tiles
        delta_i = load delta[cs:cs+128, seq_start:seq_end]
        u_i     = load u[cs:cs+128, seq_start:seq_end]
        for i_state in affine_range(S):                      # 16
            A_i     = load a[cs:cs+128, i_state]
            deltaA  = activation(exp, delta_i, scale=A_i)     # Scalar engine
            B_i     = load b[i_state:i_state+1, seq_start:seq_end]
            deltaU  = tensor_tensor(delta_i, u_i, multiply)   # (S3b: hoist this out of state loop)
            deltaBu = tensor_tensor(deltaU, B_i.broadcast_to((128,seq_tile)), multiply)
            init    = scan_state[i_channel_tile, 0:128, i_state] if i_seq_tile>0 else 0
            scan_res= tensor_tensor_scan(deltaA, deltaBu, initial=init, mul, add)
            if i_seq_tile < n_seq_tile-1:                     # carry the LAST column forward
                scan_state[i_channel_tile, 0:128, i_state:i_state+1] = scan_res[:, seq_tile-1:seq_tile]
            C_i     = load c[i_state:i_state+1, seq_start:seq_end]
            scanC   = tensor_tensor(scan_res, C_i.broadcast_to((128,seq_tile)), multiply)
            scanC_accum[i_channel_tile] += scanC              # (S3a: fuse this into scalar_tensor_tensor)
    store output[:, seq_start:seq_end] = scanC_accum
```

S0-fallback D1a (mirrors `profile/refs/ref_v3.py`, 1.504×): channels outer, states next,
seq-tile INNER over `static_range(n_seq_tile)` with a per-state `scan_init = zeros([128,1])`
updated by `scan_init[...] = scan_res[0:128, seq_tile-1]` (the `[...]` write keeps the
`[128,1]` buffer shape; the RHS is the tile's final column). The AccelOpt source warns that
`sequential_range` on the INNER seq loop gave wrong answers + worse perf — which is exactly
why D1a uses `static_range` (full unroll) and D1b uses `sequential_range` on the OUTER loop.

S1a: `b`/`c` are channel-independent → broadcast once per (seq-tile, state) and reuse across
both channel tiles. If a plain hoist is CSE'd or is a lazy-view no-op, materialize
`B_i_bcast`/`C_i_bcast` into an SBUF tile explicitly. Keep only if PE-active-time drops.

S1b (precedent `5f08e8cb` `mlp_tkg_down_projection.py`): replace the PE partition-broadcast
with a broadcast-view fed to `dma_copy`. First a compile-feasibility probe under
`--disable-dge`; then check latency, PE-active-time, DMA%, and `HBMrd`. If DMA/HBM climb
without PE falling, revert to S1a.

S2: selected-width sweep `seq_tile ∈ {256,448,896,1024}` vs the 512 anchor on the current
winner (edge-extend to 1792/128 only if the trend points there); hard-code the best width
for M=7168.

S3a: `nisa.scalar_tensor_tensor(data=scan_res, op0=multiply, operand0=c, op1=add,
operand1=scanC_accum)` does `(scan_res·c)+accum` in one Vector pass (both operands SBUF, not
PSUM). Compile-gated. S3b: compute `deltaU` once per (channel-tile, seq-tile), reuse across
16 states.

S4: `nl.static_range` on the S=16 and n_channel_tile=2 loops (the seq carry loop stays
`sequential_range`/`static_range`).

### Relevant References
- `runs/mamba_v1.py` — phase-1 kernel (the whole-M structure to replace).
- `profile/refs/ref_optimized.py` — the 1.608× D1b target (seq-tile outer, `sequential_range`,
  `[n_ct,128,S]` `scan_state`, array-slice `initial`).
- `profile/refs/ref_v3.py` — the 1.504× D1a fallback (seq-tile inner, `static_range`,
  `[128,1]` `scan_init[...]` carry; carries the AccelOpt `sequential_range`-warning comment).
- `profile/refs/README.md` — the measured reference sweep table + diagnosis.
- `docs/plan-phase2.md` — the phase-2 plan whose D1 sequence-tiling lever this phase lands
  (naming map: D1→S0, D3a→S1a, D3b→S1b, D2→S2, D4→S3b).
- `AccelOpt/NKIBench/reference/mamba_M7168_C256_S16_numpy_1.py` — numpy oracle.
- `adapter/nkibench_case.py` — seeds `[0,21,42,63,84]`, mamba rel-tol 3e-5, compiler flags
  `--disable-dge --logical-nc-config=1` (DGE globally OFF).
- `verify.py` — scoring harness; surfaces PE/Vec/Scl/DMA %, HBMrd/HBMwr, MFU.
- `benchmark.csv`, `candidates.jsonl` — evidence ledgers (DAG parent links).
- `5f08e8cb` (`mlp_tkg_down_projection.py`), `d1124a76` — DMA-broadcast precedents for S1b.

## Dependencies and Sequence

### Milestones

1. **Milestone 1 — S0: land the sequence-tiling floor (primary; targets AC-1..AC-4).** Kills
   the spill, lands ≥1.5× with a 5-seed pass, promotes `mamba_v2`. The phase's gate.
   - Phase A: Implement D1b (seq-tile outer, `sequential_range`, `scan_state`, seq_tile=512);
     score seed-42 `--fast`; read the digest.
   - Phase B: Implement + measure D1a (seq-tile inner, `static_range`, `[128,1]` carry) at
     least once. Run the faster passing variant on the full 5 seeds; promote it as `mamba_v2`.
     If D1b misbehaves, D1a is the guaranteed fallback.
2. **Milestone 2 — S1: move the `b`/`c` broadcast off the PE (opportunistic; targets AC-5).**
   On top of the S0 winner; each step gated on 5-seed pass + latency clearing noise.
   - Step 1: S1a — hoist/materialize the broadcast out of the channel loop; keep only if
     PE-active-time drops (else record no-op).
   - Step 2: S1b — compile-feasibility probe under `--disable-dge` first (write + compile a
     minimal stride-0 broadcast); if feasible, implement the full stride-0 DMA broadcast;
     keep only if latency drops, PE-active-time falls, and DMA/`HBMrd` stay below
     saturation/inflation; else revert to S1a.
3. **Milestone 3 — S2: specialize the seq-tile regime for M=7168 (opportunistic; AC-6).**
   Sweep `{256,448,896,1024}` vs 512 (edge-extend if warranted) on the current winner (after
   S1 if S1 landed, else on the S0 winner); hard-code the winning width. Do not block on S1.
4. **Milestone 4 — S3/S4: secondary levers (opportunistic, correctness-neutral; AC-7).** S3
   (`scalar_tensor_tensor` fuse if Vec-limited + hoist `deltaU`, S3b foldable regardless) and
   S4 (static-unroll) run on the current winner. Keep only on a measured drop.
5. **Milestone 5 — Harden + final lock (targets AC-1, AC-4).** Full 5-seed `verify.py` (no
   `--fast`) on the final best candidate; complete `benchmark.csv` / `candidates.jsonl` with
   DAG parent links and per-engine digests; report the final speedup vs baseline.

Dependency summary: Milestone 1 gates everything (no specialization is meaningful until the
spill is killed and a stable S0 winner exists). Milestone 2 (S1) depends on M1; S1b
additionally depends on a passing compile-feasibility probe under `--disable-dge`. Milestone
3 (S2) runs after the S1 disposition when S1 is attempted (so it sweeps on the right
structure) but falls back to the S0 winner if S1 does not land — it is NOT blocked by S1
succeeding. Milestone 4 (S3/S4) runs on the current winner and, for the S3a fuse, on a
digest showing Vec on top. Milestone 5 depends on whichever candidate is finally selected.
The ≤5-iteration budget is a soft exploration budget, not a hard cap on profiler calls; S0
is mandatory and S1–S4 are kept strictly where the measured win justifies the added
complexity.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Implement D1b (S0-lead): seq-tile outer, `sequential_range`, `[n_ct,128,S]` carried `scan_state`, array-slice `initial`, per-seq-tile store; seq_tile=512 | AC-2, AC-2.1, AC-3 | coding | - |
| task2 | Score task1 seed-42 `--fast`, read digest (PE/Vec/DMA/HBMrd); record as a screen row | AC-1, AC-4 | coding | task1 |
| task3 | Implement D1a (S0-fallback): seq-tile inner, `static_range`, `[128,1]` `scan_init[...]` carry | AC-2, AC-2.2, AC-2.3, AC-3 | coding | - |
| task4 | Score task3 seed-42 `--fast`, read digest; record as a screen row | AC-1, AC-4 | coding | task3 |
| task5 | Full 5-seed `verify.py` (no `--fast`) on the faster passing D1 variant; promote as `mamba_v2` (parent `mamba_v1`); record in benchmark.csv + candidates.jsonl | AC-1, AC-2, AC-4 | coding | task2, task4 |
| task6 | Implement S1a: hoist/materialize `b`/`c` broadcast out of the channel loop on top of the S0 winner | AC-5, AC-5.1 | coding | task5 |
| task7 | Score S1a; keep/drop via latency-primary + PE-active-time proxy + DMA/HBMrd guardrail; if no-op, drop | AC-5, AC-5.1, AC-1, AC-4 | coding | task6 |
| task8 | Write + compile a minimal stride-0 partition DMA broadcast probe; confirm it compiles+profiles under `--disable-dge --logical-nc-config=1` with no per-op DGE dependency; assess HBMrd-inflation risk | AC-5.2 | coding | task5 |
| task9 | Implement full S1b (stride-0 DMA broadcast) ONLY if task8 shows feasible; verify PE-active-time falls, DMA/HBMrd stay unsaturated, 5-seed passes; promote as the next `mamba_vN` if it wins | AC-5, AC-5.2, AC-1 | coding | task8, task7 |
| task10 | S2 selected-width sweep `seq_tile ∈ {256,448,896,1024}` vs 512 (edge-extend to 1792/128 if warranted) on the current winner (S1 winner if landed, else S0); hard-code best for M=7168; promote as the next `mamba_vN` if it wins | AC-6, AC-1, AC-4 | coding | task7, task9 |
| task11 | Analyze the current-winner digest: is Vec now the clear limiter (justifying the S3a fuse), and does the bottleneck favor S3/S4 next? | AC-7 | analyze | task7, task9 |
| task12 | Implement S3: `scalar_tensor_tensor` fuse (S3a, compile-gated, if Vec-limited) + hoist `deltaU` (S3b, foldable regardless); keep on measured drop; promote as the next `mamba_vN` if it wins | AC-7, AC-7.1, AC-3 | coding | task10, task11 |
| task13 | Implement S4: static-unroll the S=16 / 2-channel loops on the current winner; keep only on measured latency drop | AC-7, AC-7.2, AC-1 | coding | task10 |
| task14 | Harden: full 5-seed run (no `--fast`) on the final selected candidate; complete evidence ledgers with DAG links + digests; report final speedup vs baseline | AC-1, AC-4 | coding | task10, task12, task13 |

## Claude-Codex Deliberation

### Agreements
- S0 (sequence tiling) is the correct primary lever and the phase's hard gate; it reproduces
  two measured, correctness-passing references and kills the spill without touching the math.
  Land it as `mamba_v2` before any specialization.
- The 5-seed rel-L2 gate is mandatory before any promotion; `--fast` (seed 42) is a
  pre-check only, never a promotion basis.
- Keep/drop uses latency as primary, PE-active-*time* (`latency × PE%`) as the directional
  proxy (PE% alone is confounded), and DMA %/`HBMrd` as guardrails — no IR/`nc_matmul` access
  exists, so this is correlation, not verified matmul-count reduction.
- The carry loop must be `sequential_range` (D1b) or `static_range` (D1a), never
  `affine_range`; carry extraction must preserve the `[128,1]` shape (D1a `scan_init[...]`
  idiom) and `(channel-tile, state)` identity (D1b array-slice write).
- Promote the faster passing D1 variant; tie within noise → prefer the simpler structure.
- S1/S2/S3/S4 are opportunistic specializations, kept only where the measured win justifies
  the complexity; beating 1.608× is aspirational, not a phase gate.

### Resolved Disagreements
- **S1b under global `--disable-dge` (Codex CORE_RISK, accepted):** the draft's "prefer
  `dge_mode=none`" note is redundant/confusing because the harness already forces DGE off
  globally. Resolution: S1b carries **no per-op DGE dependency**; it is a **compile-
  feasibility probe first** (a minimal stride-0 broadcast that must compile+profile under
  `--disable-dge --logical-nc-config=1`, task8) and a perf lever second. If the pattern
  relies on dynamic descriptor generation that `--disable-dge` forbids, S1b is infeasible and
  dropped for S1a. Encoded in AC-5.2 + task8 (`coding`).
- **S1b HBMrd guardrail (Codex CORE_RISK, accepted):** a stride-0 broadcast can inflate HBM
  reads (each partition reading the same row → 16 MB floor toward 100+ MB), erasing the win
  even with DMA at 11%. Resolution: `HBMrd` (≤ ~25 MB) and DMA-not-saturated are explicit
  S1b guardrails. Encoded in AC-5.2 + Quantitative Thresholds.
- **S1a may be a lazy-view no-op (Codex CORE_RISK, accepted):** hoisting the Python
  `broadcast_to` expression may not reduce PE work if the view is re-emitted per consumer.
  Resolution: S1a may require explicit SBUF materialization; gate on measured PE-active-time
  drop; if unmoved, record no-op. Encoded in AC-5.1.
- **Fast-screen ledger contradiction (Codex REQUIRED_CHANGE round 1, accepted):** AC-4's
  "every recorded candidate carries a digest" tensioned with AC-1's "no promotion on `--fast`
  alone." Resolution: `--fast` screens MAY be recorded as evidence rows explicitly marked
  `seeds=[42]` / verdict `screen`; only `promoted` rows require the full 5-seed pass. Encoded
  in AC-1 + AC-4.
- **Numeric thresholds (Codex REQUIRED_CHANGE round 1, accepted):** vague terms (`≈16 MB`,
  `few %`, `toward saturation`) were pinned to a Quantitative Thresholds section: spill
  killed = `HBMrd ≤ 20 MB`; promotion clears noise ≳ 3% on the full run; S1b inflation limit
  ~25 MB; DMA must not become the saturating engine.
- **D1a carry shape / `tensor_tensor_scan` initial (Codex REQUIRED_CHANGE round 1, accepted):**
  aligned AC-2.2/AC-2.3 and the pseudocode with the exact reference idioms — D1a uses
  `scan_init[...] = scan_res[0:128, seq_tile-1]` (the `[...]` keeps `[128,1]`); D1b uses the
  `[P]` array-slice `scan_state[i_ct,:,i_state]` as `initial` (known-good from
  `ref_optimized.py`). Documented the accepted `initial` shapes (scalar-0, `[P,1]`, `[P]`).
- **AC-3 vs S3a fusion (Codex REQUIRED_CHANGE round 1, accepted):** "same primitive sequence"
  would forbid the `scalar_tensor_tensor` fusion the plan itself proposes. Rewrote AC-3 as
  "same mathematical recurrence / value-level semantics," explicitly allowing fusion of
  adjacent ops that compute the identical result while preserving state-accumulation order
  and fp32 precision.
- **Task DAG corrections (Codex REQUIRED_CHANGE round 1, accepted):** task8 retagged
  `coding` (a compile probe is empirical, not analysis); task9 depends on task8+task7; task10
  depends on the S1 disposition (task7, task9) so it sweeps on the current winner; task11 no
  longer hard-blocks on S1b when infeasible (task7, task9); task13 runs on the current winner
  (task10); task14 depends on the final selected candidate (task10, task12, task13).
- **S2 not blocked by S1 / "selected-width sweep" (Codex TECHNICAL_GAP + round-1
  REQUIRED_CHANGE, accepted):** if S1b fails or S1a is a no-op, sweep S2 on the S0 winner;
  and the sweep is a *selected-width* sweep over `{256,448,896,1024}` vs 512 (edge-extend to
  1792/128 if the trend warrants), not an exhaustive "fastest divisor." Encoded in AC-6.
- **Hard-stop / budget policy (Codex MISSING_REQUIREMENT, accepted):** 5 iterations ≈ 9+
  profiles. Resolution: the ≤5-iteration budget is a soft exploration budget; S0 landing is
  the hard success (lower bound); S1–S4 are kept strictly on evidence within budget. Encoded
  in Path Boundaries + the Dependency summary.
- **Monotonic candidate naming (Codex OPTIONAL, accepted):** promoted candidates are named
  `mamba_vN` in the order actually promoted; skipped/dropped levers do not consume a number.
- **DMA saturation guardrail pinned (Codex REQUIRED_CHANGE round 2, accepted):** round 1 left
  the DMA guardrail as "toward saturation." Resolution: pinned numerically — reject S1b if
  `DMA_active% ≥ 85%`, or if DMA becomes the single highest-active engine without BOTH a
  latency win beyond noise AND a PE-active-time drop. Encoded in Quantitative Thresholds +
  AC-5.2.

### Convergence Status
- Final Status: `converged` (3 Codex passes: 1 first-pass + 2 convergence rounds; all
  REQUIRED_CHANGES from both rounds incorporated; the two UNRESOLVED items —
  `scalar_tensor_tensor` operand shape for S3a and S1b compile viability under
  `--disable-dge` — are handled as compile-gated probes in the plan, not open user decisions).

## Pending User Decisions

_None._ Codex's `QUESTIONS_FOR_USER` and round-1 `UNRESOLVED` items are resolved by the
draft's own framing, the established phase-2 precedent for this same task, and the plan's
compile-gating philosophy — no explicit user decision is required before implementation:
- *"Is the stride-0 DMA broadcast known to work under `--disable-dge`?"* — resolved by making
  S1b a compile-feasibility probe (task8/AC-5.2): the harness itself answers this; no
  assumption is baked in.
- *"Is the budget 5 ideas or 5 kernels?"* — resolved: the ≤5-iteration budget is soft; the
  hard gate is S0 landing + evidence-gated specializations (Path Boundaries).
- *"Prioritize landing a strong `mamba_v2` even if S1/S2 deferred?"* — resolved: yes, S0 is
  the hard success target (lower bound); S1–S4 are opportunistic.
- *"Does `scalar_tensor_tensor` accept the planned `c` operand shape?"* — resolved: S3a is
  compile-gated (AC-7.1); infeasibility ⇒ drop, no user decision needed.
The quantitative metrics (rel-L2 3e-5 correctness gate = HARD; ≥1.5× S0 speedup = HARD
success target reproducing measured refs; < 0.78 ms / beating 1.608× = aspirational) were
already confirmed in the phase-2 plan for this identical task and are unchanged, so they are
not re-litigated here.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Step", "Phase", "S0/S1/S2", "D1a/D1b", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code (e.g. `scan_state`, `scan_init`,
  `seq_tile_size`, `i_seq_tile`, `deltaU`, `B_i_bcast`), matching `runs/mamba_v1.py` and the
  reference-kernel conventions.
- Kernel `.py` sources go under `runs/` (tracked); other run artifacts and all of `profile/`
  are git-ignored. Record evidence in `benchmark.csv` and `candidates.jsonl` (DAG parent
  links), not in the plan.

--- Original Design Draft Start ---

# mamba (M7168 C256 S16, fp32) — Phase 3 implementation draft (regime / shape specialization)

## Starting point — a blunt correction that reframes the whole phase

Phase 3 is nominally "shape specialization," but the honest state of the workspace
forces a two-part phase, and the draft must say so up front:

- **The only kernel that exists in `runs/` is `runs/mamba_v1.py` at 0.832× — SLOWER
  than the 1.258 ms baseline.** `candidates.jsonl` has exactly one node; `benchmark.csv`
  one perf row.
- **Phase 2 never landed a kernel.** Its draft (`docs/draft-phase2.md`) and plan
  (`docs/plan-phase2.md`) are excellent and committed, but the phase-2 RLCR loop was
  **blocked by a stale phase-1 lock** (see `logs/phase2.3-loop.log`: it held
  `plan-phase1.md`, `current_round=0`, never reviewed, never retired). So the
  profile-driven **sequence-tiling win that phase 2 designed (measured 1.5–1.6× on the
  reference kernels through our own harness) was never written to code.**

Therefore phase 3 cannot "specialize" a fast kernel — there isn't one yet. Phase 3
must **(A) first land the proven phase-2 lever (sequence tiling) as the new baseline**,
then **(B) specialize the now-fixed shapes on top of it.** Part A is not busywork or
re-litigation: it is the evidence-verified 1.6× structure that simply has no `runs/`
artifact. Part B is where the genuine phase-3 content lives.

Correctness gate is unchanged: relative-L2 `< 3e-5 · ||v_r||₂` on seeds
`[0,21,42,63,84]`, fp32. The numpy oracle for this op sequence is rel-L2 4.08e-7
(~75× inside the gate). Every phase-3 candidate keeps fp32 and the identical op
sequence (`activation(exp,scale=A)`, `tensor_tensor(delta,u)`, `tensor_tensor(·,b)`,
`tensor_tensor_scan`, `tensor_tensor(·,c)`, accumulate); only loop/tile structure,
broadcast *mechanism*, and op *fusion* change. Never regress correctness.

## The measured evidence phase 2 gathered (still the ground truth for phase 3)

Profiled through our `verify.py` path (seed 42, `--fast`), saved under `profile/refs/`:

| kernel | structure | latency | speedup | PE | Vec | Scl | DMA | HBMrd |
|--------|-----------|---------|---------|----|----|----|----|-------|
| baseline | state-outer, delta/u reloaded 16× | 1.2583 ms | 1.000× | — | — | — | — | — |
| **our v1** (= AccelOpt `mamba_v2`) | channels-outer, load-once, **whole-M** | 1.5116 ms | 0.832× | 52% | 51% | 16% | 11% | **72 MB** |
| `ref_v3` | seq-tile **inner** (`static_range` 512), `[128,1]` carry | 0.8367 ms | 1.504× | 93% | 82% | 22% | 13% | 16 MB |
| `ref_optimized` | seq-tile **outer** (`sequential_range` 512), `[n_ct,128,S]` `scan_state` | **0.7823 ms** | **1.608×** | **99%** | 89% | 23% | 11% | 16 MB |

Two facts drive phase 3:

1. **Sequence tiling is the floor** (0.832× → 1.6×): chunk M=7168 into 512-wide tiles,
   carry the scan's last column forward as the next tile's `initial`. It changes
   nothing about the math; it kills the SBUF spill (72 MB → 16 MB read-once floor) and
   lets the compiler pipeline PE against Vec (51/51% → 99/89%). Land this first.

2. **At 1.6× the bottleneck engine is the Tensor engine (PE=99%) — and mamba has no
   matmul.** The `b[s,:]`/`c[s,:]` partition-dim `broadcast_to((128, seq_tile))` is
   lowered to `nc_matmul(ones[1,128], row[1,seq_tile])` on the PE (confirmed:
   `trainium_inferentia2_arch.md` — "NKI invokes such matmul under the hood when
   `broadcast_input.broadcast_to((M, …))` is called"; and `fused_mamba.md` —
   "partition-dim broadcast often requires a separate instruction on TensorE"). Those
   broadcast matmuls are the *only* PE work, and they saturate it while **DMA sits idle
   at 11%.** This is the ceiling to attack in the specialization half.

## Where time goes across this tensor's structure (the phase-3 analysis)

The prompt asks to "analyze where time goes across the tensor's structure." The
structure is small and fully static, which is exactly why specialization pays:

- **Partition axis (channels):** C=256 = **exactly 2 tiles of 128** — no remainder, no
  edge tile. The 2-iteration channel loop is a compile-time constant.
- **Free axis (sequence):** M=7168 = 2¹⁰·7. Divides exactly by {128, 256, 448, 512,
  896, 1024, 1792} — every candidate seq-tile is clean; again **no edge tile.**
- **State axis:** S=16, fixed and tiny — the inner state loop is a compile-time constant.
- **Engine time at the 1.6× point:** PE 99% (b/c broadcast matmuls — pure overhead,
  not real compute), Vec 89% (the 4–5 `tensor_tensor` per state + the scan), Scalar 23%
  (the `exp`), **DMA 11% (idle).**

The structural hotspot is unambiguous: **the b/c partition-broadcast matmuls saturate
the PE.** There are *no* edge tiles to specialize (everything divides evenly), so the
phase-3 specialization is **not** "handle the ragged tile" — it is: pick the seq-tile
regime that minimizes PE pressure for the fixed M, relocate the broadcast off the PE,
and exploit the compile-time-constant S=16 / 2-channel loops.

## Ranked specialization directions

Ranked by expected benefit × confidence. Each is a ≤5-iteration probe gated on
`verify.py` latency + the profiler digest, correctness (5-seed) always mandatory.

### S0 — Land sequence tiling with carried scan state  ⭐ (FLOOR; must land; ~1.5–1.6×)

Carried over from the phase-2 plan (its D1), never coded. This is the phase-3 baseline;
everything else stacks on it.

- **S0-lead = D1b (mirror `profile/refs/ref_optimized.py`, the 1.608× ref):** seq-tile
  **outer** over `nl.sequential_range(n_seq_tile)`; a `[n_channel_tile, 128, S]`
  `scan_state` carried across tiles; channels then states inside; store each seq-tile's
  output slice (no whole-M accumulator). `initial = scan_state[i_ct,:,i_state]` for
  tiles > 0, else 0; after each scan, write `scan_res[:, seq_tile-1:seq_tile]` back into
  the `[i_ct, :, i_state]` slot.
- **S0-fallback = D1a (mirror `profile/refs/ref_v3.py`, 1.504×):** seq-tile **inner**
  over `nl.static_range(n_seq_tile)` (fully unrolled), per-state `scan_init` `[128,1]`
  carried as `scan_init[...] = scan_res[:, seq_tile-1]`.

**Correctness watch (from the AccelOpt source):** the carried `initial` is a
loop-carried dependency, so the carry loop MUST be `sequential_range` (D1b) or
`static_range` (D1a) — **never `affine_range`** (silently corrupts the scan; AccelOpt
hit this). `tensor_tensor_scan`'s `initial` accepts a `[P,1]` tile or a scalar 0
(confirmed in the API docs). Land D1b, measure seed-42 then 5-seed; keep D1a as the
guaranteed fallback if our exact build makes the outer `sequential_range` misbehave.
Promote the faster passing variant as `mamba_v2` (parent `mamba_v1`).

### S1 — Move the b/c broadcast OFF the Tensor engine  ⭐ (ceiling-raiser; beat 1.6×)

This is the one lever that can pass the 1.608× reference, because that reference is
PE=99%-bound on broadcast matmuls while DMA idles at 11%. Two sub-levers, safe → strong:

- **S1a — Hoist the broadcast out of the 2-channel loop (D3a).** `b[s,:]`/`c[s,:]` are
  channel-independent; with C fixed at exactly 2 tiles, broadcasting once per
  (seq-tile, state) and reusing across both channel tiles **halves** the broadcast
  matmuls (at seq_tile=512: 896 → 448). Pure hoist, obviously correct, no new API.
  **Caveat:** the compiler may already CSE the channel-independent broadcast, in which
  case S1a is a no-op — gate on measured latency, not intent.
- **S1b — Stride-0 DMA-load broadcast (D3b) — the real ceiling-raiser.** Instead of
  broadcasting a `[1, seq_tile]` SBUF row across partitions via `nc_matmul`, load
  `b[s,:]`/`c[s,:]` from HBM directly into a `[128, seq_tile]` SBUF tile with a
  **stride-0 partition access pattern** (every partition reads the same HBM row). This
  spends idle DMA bandwidth instead of saturated PE. **Direct precedent: `5f08e8cb`
  (`mlp_tkg_down_projection.py`)** replaced a `nc_stream_shuffle`/PE partition-broadcast
  of a `[1,H]` bias with exactly this — a `TensorView.broadcast(dim=0, size=T)` view fed
  to `dma_copy` (compiler lowers it to one broadcasted-DMA pattern). Implement via the
  broadcast-view (or `src.ap(pattern=[[0,128],[1,seq_tile]])`) + `dma_copy`, prefer
  `dge_mode=none` for the static contiguous load (precedent `d1124a76`).

**Why S1 is the phase-3 keystone.** It attacks the #1 measured bottleneck and is the
only direction with headroom below 0.78 ms. **Keep/drop metric (from the phase-2
deliberation): latency is primary**, PE-active-*time* (`latency × PE%`) is the
directional proxy (PE% alone is confounded — it can stay ~99% even if total PE work
halves), DMA% is the guardrail (S1b must not push DMA toward saturation). **Risk:** S1b
emits ≥128 descriptors per broadcast; confirm DMA stays well below saturation and PE
actually falls, else fall back to S1a. Correctness is trivially preserved — the
broadcast *values* are identical; only how the tile is materialized changes.

### S2 — Specialize the seq-tile regime for the fixed M=7168 (the "tile-size regime" lever)

This is the prompt's headline specialization. Once S0's structure is chosen, sweep the
exact divisors `seq_tile ∈ {256, 448, 512, 896, 1024}` (28/16/14/8/7 tiles). **The
regime tradeoff here is the OPPOSITE of the silu task's "finer wins":**

- silu was DMA-streaming (no scan, no broadcast) and preferred finer tiles (~4 KB burst).
- mamba is **PE-bound on broadcast matmuls whose count is `n_seq_tile × S × (channels or
  1 if hoisted)` — i.e. inversely proportional to seq_tile width.** Larger tiles ⇒ fewer
  broadcasts ⇒ less PE pressure. So the hypothesis is **wider wins** (896/1024), bounded
  by (a) SBUF/double-buffer slack — a `[128,1024]` fp32 tile is 4 KB/part, still tiny —
  and (b) the outer carry serialization (more tiles = more sequential carry points).
  512 was the reference's "magic number," but that was *before* attacking the broadcast;
  the optimal regime interacts with S1 (if S1b moves broadcasts to DMA, the DMA
  descriptor count then scales with `n_seq_tile`, flipping the pressure back toward
  wider tiles). Sweep S2 **after** S1 lands, and hard-code the winner for the fixed M.

Benefit likely a few %, but it's a pure knob and it's exactly the shape-specialization
the phase asks for. Risk minimal.

### S3 — Relieve the Vector engine (fuse + hoist) — only after PE drops

Once S1 moves the broadcast off PE, **Vec (89%) becomes the limiter.** Two moves:

- **S3a — Fuse `scanC = scan_res·c` + accumulate into one Vector op.** Today it's
  `tensor_tensor(scan_res, c_bcast, multiply)` then `scanC_accum += scanC` (two Vector
  passes). `nisa.scalar_tensor_tensor(dst, data=scan_res, op0=multiply,
  operand0=c_row/c_bcast, op1=add, operand1=scanC_accum)` does `(scan_res·c)+accum` in
  **one pass at `tensor_tensor` cost** (confirmed in API docs). Constraint: `data` and
  `operand1` can't both be in PSUM — both are SBUF here, fine. (If S1b makes c a
  free-dim-foldable operand this fuses even more cleanly.)
- **S3b — Hoist state-independent `deltaU = delta·u` out of the state loop (D4).**
  `deltaU` doesn't depend on `s`, yet it's recomputed in all 16 state iterations. Compute
  once per (channel-tile, seq-tile); removes ~20% of the Vector `tensor_tensor` work.
  Adds one live `[128, seq_tile]` buffer (2 KB — trivial).

Both are correctness-neutral (same math). Rank below S1/S2 because they only pay once PE
is no longer the ceiling — but a post-S1 digest showing Vec on top makes them the next
lever, and S3b is safe enough to fold in regardless.

### S4 — Static-unroll the compile-time-constant loops (cheap shape specialization)

S=16 and n_channel_tile=2 are compile-time constants. Replacing their `affine_range`
with `static_range` (full unroll) can give the compiler a longer, dependency-free
instruction window to double-buffer across — a classic fixed-shape specialization. Note
D1b already needs the *seq* loop to be `sequential_range` (carry); unrolling the **state
and channel** loops is independent of that and correctness-neutral. Cheap probe; keep
only on a measured latency drop.

## Plan of attack (≤5 iterations)

1. **Iter 1 — S0 (land the floor).** Implement D1b (seq_tile=512), score seed-42
   `--fast`, read digest; expect ≈0.78 ms / 16 MB / PE≈99 / Vec≈89. Also implement +
   measure D1a once. Promote the faster passing variant as `mamba_v2` after a **full
   5-seed** run. This alone takes us from 0.832× to ~1.6×.
2. **Iter 2 — S1 (off-PE broadcast).** S1a hoist first (safe); if the digest shows PE
   unchanged (compiler already CSE'd), go S1b (stride-0 DMA broadcast, `5f08e8cb`
   pattern). Keep only if latency drops below the S0 winner; PE-active-time proxy + DMA
   guardrail. Target < 0.78 ms. `mamba_v3`.
3. **Iter 3 — S2 (seq-tile regime sweep {256,448,896,1024} vs 512)** against the current
   winner. Hard-code the winning width for M=7168. `mamba_v4`.
4. **Iter 4 — S3 (Vec relief)** *if* the post-S1 digest shows Vec as the limiter:
   `scalar_tensor_tensor` fuse (S3a) + hoist `deltaU` (S3b). Else spend it on S4 unroll.
   `mamba_v5`.
5. **Iter 5 — Harden + final specialization lock.** Full 5-seed `verify.py` (no
   `--fast`) on the best candidate; ensure `benchmark.csv` / `candidates.jsonl` complete
   with DAG parent links and per-engine digests; try S4 unroll if not yet done. Report
   final speedup vs baseline.

Record every candidate in `benchmark.csv` + `candidates.jsonl` (DAG parent links),
profiling evidence under `profile/`. Kernel sources under `runs/` (tracked). Full 5-seed
before any promotion — never promote on `--fast` alone (other tasks showed fast-mode can
mis-rank close candidates).

## Correctness watch-items

- **Carry-loop kind.** The carried `initial` makes the seq-tile loop loop-carried → must
  be `sequential_range` (D1b) or `static_range` (D1a), never `affine_range`. First check
  on every S0 variant: seed-42 rel-L2 passes.
- **Carry extraction identity.** The carried state is exactly the previous tile's final
  scan column: D1a carries a `[128,1]` slice (not squeezed / not from the wrong axis);
  D1b writes state `s`'s final value into `[i_channel_tile, :, i_state]` without
  cross-mixing the 2 channel tiles or the 16 states. A corrupted carry exceeds 3e-5 given
  the 4.08e-7 oracle margin, so 5-seed pass ⇒ carry indexing is correct.
- **S1b broadcast values.** The stride-0 DMA broadcast must materialize the *same* row in
  every partition (partition stride 0, free stride 1). Verify rel-L2 stays ~1e-6, not
  merely < 3e-5.
- **Op sequence unchanged.** No precision reduction below fp32, no reordered
  non-associative math — a diff vs the S0 winner shows only structure / broadcast-mechanism
  / fusion changes.

## Risks / watch-items

- **S1a may be a no-op** (compiler CSE) — that's fine, it de-risks S1b; gate on latency.
- **S1b descriptor cost** — ≥128 descriptors/broadcast; if DMA climbs toward saturation
  without PE falling, revert to S1a. This is the one moderate-risk lever.
- **S2 regime interacts with S1** — the PE-vs-DMA pressure balance shifts once the
  broadcast moves engines, so sweep S2 *after* S1 is decided, not before.
- **Don't over-fit `--fast`** — confirm the ranking on the full 5-seed / higher-iter run
  before promoting.
- **Scope discipline** — S0 (the ~1.6× floor) is the phase's hard success target; S1–S4
  are evidence-gated specializations, kept only where the measured win justifies the
  added complexity, exactly as the prompt directs.

--- Original Design Draft End ---
