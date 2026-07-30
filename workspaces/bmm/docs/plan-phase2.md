# bmm Phase 2 — Profile-Driven Optimization (Schedule-Bound fp32 Batched Matmul)

## Goal Description

Turn the phase-1 correctness base `runs/bmm_v1.py` (first correct fp32 kernel for the
`bmm` operator: batched matmul `out[b] = lhs[b] @ rhs[b]`, `b in 0..15`,
`lhs (16,4096,64)=(B,M,K)`, `rhs (16,64,4096)=(B,K,N)` fp32 → `out (16,4096,4096)`,
B=16/M=4096/K=64/N=4096, currently **0.663x / 3.8477 ms** against the **2.550 ms**
baseline, full 5-seed L2 PASS) into a materially faster kernel, **primarily through
pure-fp32 PSUM-bank pipelining** that recovers v1's per-instruction schedule loss.

The phase-1 profile is the whole story: **PE=95% but MFU=11%** — the Tensor Engine is
*occupied* yet *stalled between instructions*, doing little useful work. HBM traffic is
already at the read-once/write-once floor (HBMrd=34 MB, HBMwr=1074 MB), so DMA levers are
dead. v1 does **25% FEWER total matmul sites than the 2.550 ms baseline** (4608 vs 6144;
the 4096 main matmuls are *identical*, v1 does 8× fewer transposes) **yet runs 51% slower
→ ~2× slower per site.** The lever is therefore **PE feeding / scheduling**, not op count
and not DMA.

The work proceeds strictly measure-first: a Round-0 measurement battery must **confirm or
refute** the schedule-bound thesis before any code change. Then three ranked directions:
**D1** PSUM-bank pipelining (pure fp32, primary, high confidence), **D2** off-PE `lhs`
transpose via `load_transpose2d` (enabler), and **D3** a compensated bf16x2 3-product main
matmul (a tightly-gated, measure-first bet that is the only lever touching the fp32
structural ceiling). `runs/bmm_v1.py` is retained as the pure-fp32 fallback throughout.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. Correctness everywhere means NKIBench's relative-L2 gate
(`||v_k − v_r||₂ < 2e-5·||v_r||₂`) across seeds `[0,21,42,63,84]` via `verify.py` (the gate
is `l2_norm_passed`; not allclose). "Same-session noise band" is the ~1.8–2.5% p50 jitter
observed on sibling ops; a real win must land OUT of that band on a full (non-`--fast`) run.

- AC-1: **Round-0 measurement battery (measure-first; mostly zero remote-risk) is produced
  and either CONFIRMS or REFUTES the schedule-bound thesis before any optimization code is
  written.** The battery is captured to `profile/` and includes a same-session v1 **and**
  baseline anchor.
  - Positive Tests (expected to PASS):
    - `runs/dump_metrics.py` (a copy of the swiglu idiom retargeted with op string `"bmm"`)
      reports, for `runs/bmm_v1.py`, the TRUE `tensor_engine_active_time_ns` normalized
      per-inference and `matmul_instruction_count` (not the coarse PE%×latency proxy).
    - The same `dump_metrics` run on the **read-only** baseline kernel
      `bmm_B16_M4096_K64_N4096_0.py` (profiling only — the baseline file is never edited)
      records its true PE-active time and matmul-instruction count in the same session.
    - The confirm/refute rule is multi-signal (not a single ratio): it compares
      matmul-instruction count, TRUE PE-active time, per-matmul stall/issue gap
      (PE-active ÷ matmul count), total PE-active vs wall-clock, transpose-instruction
      share, and (where the profiler exposes it) PSUM→SBUF copy/store overlap. The thesis
      is CONFIRMED if the baseline achieves lower per-instruction stall (more matmul-instr
      in less true PE-active time) and this bounds D1 headroom; otherwise it is REFUTED and
      the phase pivots to a documented negative result (see Lower Bound).
    - A bf16 PE-ratio calibration probe (`runs/bmm_probe_bf16_calib.py`: v1 with the main
      matmul operands cast to bf16, single product) runs and is read via `dump_metrics`;
      its correctness is expected to FAIL and it is recorded as **record-only**, existing
      solely to measure this op's fp32-vs-bf16 pass ratio.
    - An offline numpy sim (`runs/offline_bf16_split_sim.py`, zero remote spend) reproduces
      the scored input draw + numpy reference and reports the worst-over-seeds 3-product
      compensated-bf16x2 rel-L2, modeling the SAME RNE rounding, limb split, 3-product
      order (`hi·hi + hi·lo + lo·hi`), and dropped `lo·lo` term the intended kernel uses.
    - HBM read/write bytes and any available SBUF/PSUM spill indicator are captured for v1
      as the floor reference for later spill-detection.
  - Negative Tests (expected to FAIL / be rejected):
    - Editing, moving, or regenerating the baseline or reference kernel to "measure" it is
      rejected — the baseline is profiled in place, read-only.
    - Concluding "schedule-bound" from PE=95% + low MFU ALONE (without the multi-signal
      comparison against the baseline anchor) is rejected as over-classification.
    - Treating the bf16 calib probe or the offline sim as a correctness candidate (it is
      not) is rejected.
- AC-2: **D1 — PSUM-bank pipelining (PRIMARY, pure fp32).** A candidate that issues `G`
  independent n-chunk matmuls into `G` DISTINCT pre-declared PSUM banks while preserving
  v1's hoisted 512 transposes and v1's EXACT single-pass K=64 correctness.
  - Positive Tests (expected to PASS):
    - The candidate declares a multi-bank accumulator (e.g.
      `acc = nl.ndarray((G, par_dim(128), 512), buffer=nl.psum)`) and issues the `G` main
      matmuls into `acc[0..G-1]` before their PSUM→SBUF copies/stores drain, mirroring the
      baseline's `v10` discipline. Output tile math stays a single `=` write (K=64 in one
      `nc_matmul`, no K-accumulation reorder), so correctness is identical to v1.
    - Staged isolation is exercised: Stage A pre-declares the accumulator across the 8
      n-chunks WITHOUT changing issue order (isolates declaration scope from scheduling);
      Stage B groups issue-before-copy/store (isolates the scheduling effect). A cheap G=1
      pre-declared-bank control and/or a no-op structural candidate is used when useful to
      attribute a latency move to allocation scope vs multi-bank scheduling.
    - `G` is swept over `{2, 4}` (G=2 first as the safer bank-pressure probe); the n-block
      width (512-group vs baseline-style 1024-group) may also be swept, within ≤5 iterations.
    - A promoted D1 candidate passes the full 5-seed L2 gate, keeps HBM read/write at the
      v1 floor with **no spill** (validated by profiler/spill evidence, not assumed from
      shape), lands an out-of-band p50 latency improvement over the same-session v1 anchor,
      AND shows improved true PE-active time OR reduced per-matmul stall vs v1.
    - The `(G, par_dim(128), 512)` PSUM allocation compiles without implicit spill or bank
      aliasing (confirmed by evidence, not by shape alone).
  - Negative Tests (expected to FAIL / be rejected):
    - A candidate that changes the single-pass K=64 contraction into a multi-pass/K-accum
      form, or otherwise perturbs accumulation order, is rejected (it is not a pure
      correctness-preserving reschedule).
    - A "multi-bank" candidate that the compiler serializes anyway (no true PE-active or
      stall improvement) or that spills PSUM/SBUF above the v1 floor is rejected.
    - A latency change within the ~1.8–2.5% same-session noise band is not a win and does
      not replace v1/best-known.
- AC-3: **D2 — off-PE `lhs` transpose via `load_transpose2d` (ENABLER).** Replace the 512
  PE identity-matmul transposes with `nl.load_transpose2d(v1[b, m_slice, k])` → `[k, m]`
  loaded already-transposed from HBM.
  - Positive Tests (expected to PASS):
    - A minimal compile + tile-orientation debug check runs FIRST: the transposed-load
      output for at least one small slice is verified to match v1's PE-transpose output
      orientation exactly (catches swapped-axis / signature misunderstandings before a full
      run), followed by a full 5-seed L2 PASS.
    - The transposed-load path removes the 512 PE transpose passes and the transpose PSUM
      bank + its copy, and is KEPT only if it lowers latency directly OR enables a stronger
      D1 result (frees a PSUM bank so the output banks fit); HBM traffic does not increase
      beyond noise.
  - Negative Tests (expected to FAIL / be rejected):
    - If `load_transpose2d` fails to lower or does not help here, the candidate falls back
      to v1's proven PE identity-transpose idiom rather than forcing the transposed load.
    - A transposed-load candidate that passes compile/shape but fails numerical correctness
      on any seed (a silent axis swap) is rejected.
- AC-4: **D3 — compensated bf16x2 3-product main matmul (GATED BET, measure-first).** Split
  each main-matmul operand into bf16 hi/lo limbs; accumulate 3 products
  (`hi·hi + hi·lo + lo·hi`, drop `lo·lo`) in fp32 PSUM. Pursued ONLY if BOTH Round-0 gates
  pass.
  - Positive Tests (expected to PASS):
    - Gate (a): the Round-0 bf16 PE-ratio probe shows measured fp32/bf16 ratio **> 3**
      (so 3 bf16 products cost less than the fp32 emulation), assessed against the added
      limb-prep Vec/Scalar cost, not PE time alone.
    - Gate (b): the offline sim's WORST-over-seeds rel-L2 is at or below the tightened
      threshold **≤ 7e-6** (a maximum, not a target — see DEC-1), leaving quadrature margin
      under the 2e-5 gate given the sibling underestimate.
    - A promoted D3 candidate passes the full 5-seed **on-device** L2 gate (the offline sim
      is a green-light, not a guarantee) AND lands an out-of-band p50 win over the best
      pure-fp32 (D1/D2) candidate.
  - Negative Tests (expected to FAIL / be rejected):
    - If the measured ratio is ~2, D3 is SKIPPED entirely (record-only) — 3 products would
      raise PE time and regress, exactly as the swiglu sibling's all-3 split did (0.409x).
    - A D3 candidate promoted on the offline sim alone (without an on-device 5-seed L2 PASS)
      is rejected.
    - Emitting a bf16 output (as opposed to bf16 intermediate limbs with fp32 PSUM and fp32
      output) is rejected — the output is the final result and the 2e-5 gate forbids it.
- AC-5: **Correctness invariance + fallback.** Every promoted candidate clears the full
  5-seed L2 gate; `runs/bmm_v1.py` is retained unchanged as the pure-fp32 fallback; the
  exact output layout `(16,4096,4096)=(B,M,N)` is preserved and verified across all seeds.
  - Positive Tests (expected to PASS):
    - `verify.py --op bmm --candidate runs/<file>.py` (no `--fast`) reports
      `l2_norm_passed=True` for every seed for any promoted candidate.
    - v1 still exists and still passes 5 seeds at the end of the phase.
  - Negative Tests (expected to FAIL / be rejected):
    - A candidate that passes `--fast` (seed 42) only, or passes a subset of seeds, is not
      promotable.
    - Deleting or overwriting v1 (the fallback) is rejected.
- AC-6: **Rejection rules (guard against false wins).** A candidate is rejected if it shows
  HBM spill above the v1 floor, seed-to-seed correctness instability, compile fragility, or
  a latency change within the same-session noise band.
  - Positive Tests (expected to PASS):
    - Each promote/revise/reject decision is recorded with the deciding number and with an
      attribution of WHERE any win came from (scheduling = D1, transpose removal = D2, or
      precision = D3).
  - Negative Tests (expected to FAIL / be rejected):
    - A candidate promoted on a `--fast` single-seed screen without a full 5-seed
      confirmation is rejected.
    - A candidate whose only evidence is an in-noise-band p50 delta is rejected.
- AC-7: **Bookkeeping and discipline.** Evidence is recorded per the KDA loop.
  - Positive Tests (expected to PASS):
    - Each perf change is appended to `benchmark.csv`; each candidate to `candidates.jsonl`
      with parent DAG links; profiling evidence lands under `profile/`; kernels live in
      `runs/`.
    - Before each comparison, v1 is re-run same-session as the noise anchor; timing records
      capture p50 plus a spread indicator (min/max or stddev) so marginal regressions are
      not hidden by the noise band.
  - Negative Tests (expected to FAIL / be rejected):
    - Writing kernels outside `runs/`, or editing the baseline/reference, is rejected.
    - Promoting on a cross-session (not same-session) comparison is rejected.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
All three directions land with full evidence: D1 pure-fp32 PSUM-bank pipelining recovers
the per-site schedule gap (projected ~1.2–1.35x), D2 off-PE transpose is folded in as an
enabler for a small additional gain, and D3 compensated bf16x2 clears both Round-0 gates
and an on-device 5-seed L2 PASS to add headroom on top of D1 (trajectory toward ~1.5x).
The phase closes with a documented best candidate, a retained pure-fp32 fallback
(`bmm_v1.py`), complete Round-0 measurement evidence, and `benchmark.csv` / `candidates.jsonl`
/ `profile/` all updated with per-decision attribution (scheduling vs transpose vs precision).

### Lower Bound (Minimum Acceptable Scope)
The Round-0 battery is produced and the schedule-bound thesis is explicitly confirmed or
refuted from the same-session v1+baseline anchors. If confirmed, D1 alone yields at least
one pure-fp32 candidate that passes the full 5-seed L2 gate and clears 1.0x (beats the
2.550 ms baseline) out of the noise band, with `bmm_v1.py` retained as fallback. If Round-0
**refutes** the thesis, the minimum acceptable outcome is a documented, evidence-backed
negative result (why v1's gap is not schedule loss) plus retained v1 — not a forced speedup.

### Allowed Choices
- Can use: multi-bank pre-declared PSUM accumulators (baseline-style, indexed to distinct
  banks); `nl.load_transpose2d` for the off-PE transpose; compensated bf16x2 3-product main
  matmul with fp32 PSUM accumulation and fp32 output (gated); a `G` sweep over `{2,4}` and
  an n-block-width sweep (512 vs 1024); staged/isolation microvariants (Stage A declaration
  scope, Stage B issue grouping, G=1 control, no-op structural control); the swiglu
  `dump_metrics.py` and `offline_bf16_split_sim.py` idioms retargeted to bmm.
- Cannot use: a bf16 (or any sub-fp32) OUTPUT; `dma_transpose` at fp32 (proven-infeasible on
  this remote); vector-engine transpose (`nc_transpose engine=vector`, measured +2%
  regression on a sibling); packing 2 batches onto the K partition axis (block-diagonal
  `out[b]` would sum two batches — numerically wrong); editing the NKIBench baseline,
  reference, seeds, or summary; `allclose` as the correctness gate; promoting on `--fast`
  or on in-noise-band deltas.

> **Note on Deterministic Designs**: The draft prescribes a specific ranked direction set
> (D1 → D2 → D3) with fixed Round-0 gates and a fixed correctness gate (2e-5 rel-L2, 5
> seeds). Within that, the implementation retains real choices (G value, n-block width,
> transpose method, whether D3 runs at all), so the bounds above are genuinely a range, not
> a single point — but the closed/forbidden list is fixed per the draft and is not to be
> re-litigated by iterations.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

Round 0 (measure-first, before any optimization code):
```
# 1. v1 true PE-active + matmul instruction count (per-inference normalized)
dump_metrics runs/bmm_v1.py                       # copy swiglu dump_metrics.py, op="bmm"
# 2. baseline true PE-active + matmul count (READ-ONLY, profile in place)
dump_metrics <NKIBench baseline bmm_..._0.py>
# 3. bf16 PE-ratio calib probe (record-only; correctness expected FAIL)
dump_metrics runs/bmm_probe_bf16_calib.py         # v1 with main-matmul operands -> bf16
# 4. offline bf16x2 rel-L2 sim (numpy, zero remote spend)
python runs/offline_bf16_split_sim.py             # worst-over-seeds 3-product rel-L2
# Confirm/refute: compare matmul-instr count, TRUE PE-active, per-matmul stall,
# PE-active vs wall, transpose share, copy/store overlap across v1 vs baseline.
```

D1 — PSUM-bank pipelining (mirrors the matmul sibling's multi-bank `acc`):
```
for b in affine_range(16):
    rhs_sb = load v2[b]                                  # [64, 4096] resident once/batch
    for mt in affine_range(32):
        lhs_t = transpose(load lhs tile)                 # v1's hoisted transpose (kept)
        # Stage A: pre-declare across chunks; Stage B: group issue-before-drain
        acc = ndarray((G, par_dim(128), 512), buffer=psum)   # G distinct banks
        for g in range(G):
            acc[g] = nc_matmul(lhs_t, rhs_sb[:, chunk(g)]) # issue G matmuls first
        for g in range(G):
            store out[...] = copy(acc[g])                  # then drain copies/stores
```
Note the bank budget: 8 output banks + 1 transpose bank > 8 physical banks, so either group
n (small G) or offload the transpose (D2) to free a bank. Start G=2, then G=4.

D2 — off-PE transpose (proven portable at fp32 on this remote):
```
xT = nl.load_transpose2d(v1[b, m_slice, k])   # [128,64] HBM -> [64,128] transposed load
# verify one small slice matches v1's PE-transpose orientation BEFORE full run
```

D3 — compensated bf16x2 (only if both Round-0 gates pass):
```
a_hi = bf16(a); a_lo = bf16(a - a_hi)         # RNE limbs
# accumulate in fp32 PSUM: a_hi@b_hi + a_hi@b_lo + a_lo@b_hi  (drop lo@lo)
```

### Relevant References
- `workspaces/bmm/runs/bmm_v1.py` — the phase-1 base and pure-fp32 fallback; the loop nest
  D1 reschedules (single fresh `acc`/`psum_t` per iteration is the suspected starvation).
- `../AccelOpt/NKIBench/kernels/bmm_B16_M4096_K64_N4096_0.py` — the 2.550 ms baseline; its
  `v8`/`v10` giant pre-declared multi-bank PSUM tensors are the D1 template (READ-ONLY).
- `workspaces/matmul/runs/matmul_v2_b4.py` — proven multi-bank `acc = zeros((B, par_dim(128),
  N_CHUNK), buffer=psum)` idiom with distinct banks per block member (D1 pattern precedent).
- `workspaces/rmsnorm_matmul/runs/rmsnorm_matmul_probe_loadtranspose.py` — proven fp32
  `load_transpose2d` (full 5-seed PASS, transpose hidden, PE stayed 97%) (D2 precedent).
- `workspaces/rmsnorm_matmul/runs/rmsnorm_matmul_probe_bf16_calib.py` — bf16 PE-ratio
  calibration probe idiom (Round-0 #3 template).
- `workspaces/swiglu/runs/dump_metrics.py` — reads TRUE `tensor_engine_active_time_ns` +
  `matmul_instruction_count`, per-inference normalized (Round-0 #1/#2 template; retarget
  op string to `"bmm"`).
- `workspaces/swiglu/runs/offline_bf16_split_sim.py` — numpy compensated-bf16x2 rel-L2 sim
  idiom (Round-0 #4 template; adapt the input draw + reference to bmm's single matmul).
- `verify.py` — the correctness/perf harness; gates on `l2_norm_passed`, prints the
  MFU/PE/Vec/Scl/DMA/HBM digest.

## Dependencies and Sequence

### Milestones
1. **Round 0 — evidence (blocks everything else).**
   - Phase A: create `runs/dump_metrics.py`, `runs/bmm_probe_bf16_calib.py`,
     `runs/offline_bf16_split_sim.py` (retargeted sibling idioms).
   - Phase B: run the same-session v1 + baseline anchors, the bf16 calib probe, and the
     offline sim; capture to `profile/`; confirm or refute the schedule-bound thesis and
     record the D1 headroom bound and the D3 gate readings.
2. **D1 — PSUM-bank pipelining (primary; depends on Round 0 CONFIRMING the thesis, else
   pivot to negative result).**
   - Step 1: Stage A — pre-declare multi-bank `acc` without reordering issue (isolate
     declaration scope); optional G=1 / no-op controls.
   - Step 2: Stage B — group issue-before-drain; sweep G∈{2,4} (G=2 first) and n-block
     width; validate no-spill + out-of-band win + true-PE-active/stall improvement; promote
     on full 5-seed L2 PASS.
3. **D2 — off-PE transpose (enabler; best evaluated combined with D1).**
   - Step 1: minimal compile + tile-orientation debug-slice check.
   - Step 2: fold into the best D1 candidate; keep only if it lowers latency or frees the
     bank that lets more output banks fit; else fall back to the PE transpose.
4. **D3 — compensated bf16x2 (gated bet; depends on BOTH Round-0 gates passing).**
   - Step 1: confirm gate (a) ratio > 3 and gate (b) offline worst-seed rel-L2 ≤ 7e-6.
   - Step 2: build the 3-product limb kernel on top of the best D1/D2 candidate; promote
     only on a full 5-seed on-device L2 PASS and an out-of-band win over the best fp32.

Dependency summary: Round 0 gates D1 (thesis) and D3 (both precision gates). D2 depends on
D1 (evaluated combined). D3 builds on the best D1/D2 candidate. v1 is the fallback for all.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Create `runs/dump_metrics.py`, `runs/bmm_probe_bf16_calib.py`, `runs/offline_bf16_split_sim.py` as bmm-retargeted sibling idioms | AC-1 | coding | - |
| task2 | Run Round-0 battery: same-session v1 + read-only baseline anchors, bf16 calib probe, offline sim; capture to `profile/` | AC-1 | coding | task1 |
| task3 | Analyze Round-0 evidence: multi-signal confirm/refute of schedule-bound thesis; bound D1 headroom; read D3 gates | AC-1, AC-4 | analyze | task2 |
| task4 | D1 Stage A: pre-declare multi-bank `acc` without reordering issue (+ optional G=1 / no-op controls); score | AC-2 | coding | task3 |
| task5 | D1 Stage B: group issue-before-drain; sweep G∈{2,4} (G=2 first) and n-block width; validate no-spill + out-of-band + PE-active/stall; promote on 5-seed PASS | AC-2, AC-5, AC-6, AC-7 | coding | task4 |
| task6 | D2: compile + tile-orientation debug-slice check for `load_transpose2d`, then fold into best D1; keep or fall back | AC-3, AC-5, AC-7 | coding | task5 |
| task7 | D3 (conditional on both Round-0 gates): build 3-product bf16x2 limb kernel on best D1/D2; on-device 5-seed L2 + out-of-band win to promote | AC-4, AC-5, AC-6, AC-7 | coding | task3, task6 |
| task8 | Final bookkeeping: `benchmark.csv` rows, `candidates.jsonl` DAG, `profile/` evidence, per-decision attribution (schedule/transpose/precision) | AC-7 | coding | task5, task6, task7 |

## Claude-Codex Deliberation

### Agreements
- Measure-first Round 0 is the right guardrail before spending any iteration; it must
  produce a same-session v1 **and** baseline anchor.
- D1 (pure-fp32 PSUM-bank pipelining, staged to isolate declaration scope from scheduling)
  is the correct center of gravity and highest-confidence win.
- D2 is correctly framed as an enabler, not assumed-free performance; it needs its own
  compile + orientation validation.
- D3 is appropriately gated and must be pursued only on strong offline margin plus an
  on-device 5-seed L2 PASS; if the fp32/bf16 ratio is ~2 it is skipped entirely.
- Keeping `bmm_v1.py` as the pure-fp32 fallback and requiring a full 5-seed remote L2 pass
  for every promotion is correct.
- The closed directions (M-blocking, store-burst fattening / ping-pong, bf16 output,
  `dma_transpose` fp32, vector-engine transpose) remain correctly closed unless metrics
  change.

### Resolved Disagreements
- **Round-0 confirm/refute rule too coarse (Codex REQUIRED_CHANGE):** Codex objected that
  "baseline does more matmul-instr in less PE-active time" alone can over-classify mixed
  causes (fp32 emulation, issue limits, SBUF pressure, copy serialization). Resolution:
  AC-1 now requires a multi-signal rule (matmul count, true PE-active, per-matmul stall,
  PE-active vs wall, transpose share, copy/store overlap) and an explicit REFUTE branch.
- **D1 multi-bank correctness-by-shape (Codex REQUIRED_CHANGE):** Codex noted
  `(G,par_dim(128),512)` does not *guarantee* distinct physical banks or non-serialization.
  Resolution: AC-2 requires no-spill/no-bank-aliasing validation from profiler/spill
  evidence (not shape), plus staged isolation (Stage A/B) and G=1 / no-op controls to
  attribute any move to allocation vs scheduling.
- **D2 orientation risk (Codex REQUIRED_CHANGE):** a transposed load can pass compile/shape
  yet swap axes. Resolution: AC-3 adds a mandatory tile-orientation debug-slice check
  against v1's transpose output BEFORE the full 5-seed run.
- **D3 offline gate too loose (Codex first-pass + REQUIRED_CHANGE):** given the sibling's
  ~3.4× offline underestimate (add_rmsnorm_matmul on-device 1.528e-5 vs offline 4.45e-6,
  combining in quadrature with the fp32 floor), a `< 2e-5` offline gate is unsafe.
  Resolution: AC-4 tightens the offline gate to a worst-seed **≤ 7e-6 maximum** (not a
  target) and keeps the on-device 5-seed L2 PASS as the true promotion gate.
- **D1 gain overstatement risk (Codex CORE_RISK):** Codex flagged that matching baseline
  per-site efficiency is not guaranteed. Resolution: the Lower Bound only commits to >1.0x
  *if Round 0 confirms the thesis*, and explicitly allows a documented negative result if
  it refutes — the ~1.2–1.35x figure is framed as a projection bounded by Round-0 #1/#2.

### Convergence Status
- Final Status: `converged` (Codex round-2 review returned AGREE — "reasonable and
  converged enough to execute" — with no material disagreement; all REQUIRED_CHANGES folded
  into AC-1..AC-4 above; two items carried to Pending User Decisions).

## Pending User Decisions

- DEC-1: **D3 offline rel-L2 gate threshold.**
  - Claude Position: worst-over-seeds offline rel-L2 **≤ 7e-6** as the maximum to green-light
    an on-device D3 attempt (7e-6 quadrature-combined with a ~1e-5 fp32 floor ≈ 1.2e-5,
    comfortably under the 2e-5 on-device gate).
  - Codex Position: agrees ≤ 7e-6 is acceptable; ≤ 5e-6 is stricter/safer but may reject a
    viable candidate. Either is defensible.
  - Tradeoff Summary: tighter (5e-6) spends fewer risky remote runs but may skip a candidate
    that would have passed on-device; looser (7e-6) risks one wasted on-device run if
    quadrature is worse than modeled. The on-device 5-seed L2 PASS remains the true gate
    either way, so the downside of 7e-6 is bounded to one screening run.
  - Decision Status: `PENDING` (recommendation: adopt ≤ 7e-6; both sides agree it is
    acceptable).
- DEC-2: **Iteration-budget posture.**
  - Claude Position: bounded — the draft's ≤5-iterations-per-direction, measure-first, stop
    on the first out-of-band full-pass win; only consider spend-to-maximize after a
    full-pass candidate already beats the baseline.
  - Codex Position: agrees with the bounded posture first; spend-to-maximize should start
    only after a full-pass candidate beats baseline.
  - Tradeoff Summary: bounded conserves remote spend and matches sibling discipline;
    spend-to-maximize could squeeze more speedup at higher remote cost with diminishing
    returns.
  - Decision Status: `PENDING` (recommendation: bounded posture, per both sides).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Step", "Phase", "D1/D2/D3", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `acc`, `psum_banks`,
  `lhs_t`, `rhs_sb`, `bf16_hi`/`bf16_lo`), matching the existing `runs/*.py` conventions.

--- Original Design Draft Start ---

# bmm — Phase 2 draft: profile-driven optimization

**Operator:** `bmm` (NKIBench case 2). Batched matmul `out[b] = lhs[b] @ rhs[b]`,
`b in 0..15`. `lhs (16,4096,64)=(B,M,K)`, `rhs (16,64,4096)=(B,K,N)` fp32 →
`out (16,4096,4096)`. **B=16, M=4096, K=64, N=4096.** Baseline **2.550 ms**.

**Start point:** `runs/bmm_v1.py` = the phase-1 correctness base, **0.663x (3.8477 ms)**,
full 5-seed L2 PASS. Profile: MFU=11% **PE=95%** Vec=20% Scl=14% DMA=39%,
HBMrd=34 MB (read floor), HBMwr=1074 MB (write floor). Traffic is already minimal.

---

## 1. The anomaly that defines phase 2 (this is the whole story)

Phase 1 recorded "PE-bound, not write-bound" (Codex reconciled the draft's wrong
write-bound framing). Phase 2 sharpens that into an **actionable** finding by counting
`nc_matmul` sites in v1 vs the baseline:

| kernel | transpose sites | main-matmul sites | total sites | latency | µs / site |
|---|---|---|---|---|---|
| baseline `..._0.py` | `16·4·4·8` = **2048** | `16·4·4·8·2` = **4096** | **6144** | 2.550 ms | 0.415 |
| **v1** | `16·32` = **512** | `16·32·8` = **4096** | **4608** | 3.848 ms | 0.835 |

- The **4096 main matmuls are IDENTICAL** in both (`[64,128]×[64,512]→[128,512]`, fp32).
- v1 does **8× FEWER transposes** (512 vs 2048): it hoists the lhs transpose above the
  n-loop; the baseline re-transposes each m-subtile once per 1024-wide n-block (4× redundant).
- **v1 does 25% FEWER total matmul sites, yet is 51% slower → 2× slower per site.**

⇒ v1 is **not** limited by op count (it already does less work than the 2.55 ms baseline).
It is **schedule-bound**: the PE is *occupied* (PE=95%) but *stalled between
instructions*, doing little useful work (MFU=11%). The baseline reaches 2.55 ms with
MORE work purely because it feeds the PE better. **The phase-1 draft's DMA levers
(store-burst fattening, ping-pong) are dead** — DMA=39% is already hidden under compute
and traffic is at the read-once/write-once floor. **The lever is PE feeding.**

### Root cause hypothesis (to confirm in round 0)
The baseline pre-declares giant multi-bank PSUM tensors indexed by every loop var —
`v8 = zeros((16,4,4,8, 64,128))` (transposes), `v10 = zeros((16,4,4,8,2, 128,512))`
(outputs). Each `(loop-index)` combo writes a **distinct logical PSUM bank**, which the
compiler rotates through the 8 physical banks and software-pipelines: matmul(c+1) issues
while copy(c)/store(c) drain. v1 instead uses a **single** `acc` (and a single `psum_t`)
declared fresh inside the loop; the tight `matmul → nl.copy → nl.store` dependency on one
rotating bank (plus a serial `transpose → copy` at the head of each m-tile's 8-matmul
burst) starves the PE. This is exactly the multi-bank idiom the `matmul` sibling used to
hit PE=100% (`acc = zeros((B,128,512))`, distinct bank per block member).

## 2. Structural ceiling shared with the baseline (do NOT try to remove)
- **fp32 on a bf16-native PE**: each fp32 matmul emulates in multiple bf16 passes.
- **K=64 fills only 64 of 128 partition rows** → the contraction axis is half-empty every
  pass. Cannot be fixed by packing 2 batches onto K: `out[b]` are block-diagonal, so a
  128-row stacked contraction would *sum* two batches' products — numerically wrong.

Both are present in the baseline too, so they explain why even a perfectly-scheduled fp32
bmm can't reach the naive ~0.90 ms FLOP floor (MFU stays low). They do **not** explain
v1's gap *to the baseline* — that gap is schedule (§1). Precision (§4, D3) is the only
lever that touches this ceiling, and it is a gated, measure-first bet.

## 3. Round 0 — measurements before any code change (mostly zero remote-risk)

All via the sibling `dump_metrics.py` idiom (reads the profiler's TRUE
`tensor_engine_active_time_ns` + `matmul_instruction_count`, not the coarse PE%×lat
proxy). Create `runs/dump_metrics.py` = the swiglu copy with op string `"bmm"`.

1. **v1 true PE-active + instr count** — `dump_metrics runs/bmm_v1.py`. Establishes v1's
   real PE-busy time and matmul-instruction count (expect ~2× sites for fp32 emulation).
2. **Baseline true PE-active + instr count** — `dump_metrics` on the read-only baseline
   `..._0.py` (profiling only, never edit it). This QUANTIFIES the per-instruction stall:
   if the baseline does more matmul-instr in less PE-active time, the gap is confirmed
   schedule loss and bounds the D1 headroom exactly.
3. **fp32/bf16 PE-ratio calibration** — `runs/bmm_probe_bf16_calib.py`: v1 with the main
   matmul operands cast to bf16 (single product, **correctness will FAIL — record-only**),
   read via `dump_metrics`. Gives THIS op's fp32-vs-bf16 pass ratio. **This decides D3**:
   the `matmul` sibling measured fp32 ≈ **3.62×** bf16 (so bf16x2's 3 products WIN), but
   the `swiglu` sibling measured fp32 ≈ **2×** (so bf16x2's 3 products LOSE +50%). bmm
   could go either way — measure, don't assume.
4. **offline bf16x2 rel-L2 sim** — `runs/offline_bf16_split_sim.py` (numpy, ZERO remote
   spend): reproduce the scored input draw + the numpy reference, compute the worst-case
   3-product compensated-bf16x2 rel-L2 across seeds. Gate D3 on `< 2e-5`. Note bmm's
   output is a **raw matmul with NO downstream averaging** (unlike rmsnorm's `/K`), and
   K=64 is short, so error ≈ single-pass matmul rounding — likely safe but must be shown.

## 4. Optimization directions, ranked by expected benefit × confidence

### D1 — PSUM-bank pipelining (PRIMARY; high confidence, high benefit, pure fp32)
Restructure v1's inner loop to expose independent matmuls into **distinct pre-declared
PSUM banks**, mirroring the baseline's `v10` discipline, while KEEPING v1's hoisted 512
transposes. Concretely: per `(b, mt)`, declare a multi-bank output accumulator
`acc = nl.ndarray((G, par_dim(128), 512), buffer=psum)` and issue the G n-chunk matmuls
into `acc[0..G-1]` before their copies/stores drain (G ∈ {2,4}; 8 output banks + 1
transpose bank > 8 physical, so group n or offload the transpose — see D2). This lets the
compiler pipeline matmul-issue ahead of PSUM→SBUF copy and store.
- **Expected:** recover the 2×/site schedule gap. v1 has 25% FEWER matmul sites than the
  2.55 ms baseline and identical main matmuls, so matching the baseline's per-site
  efficiency projects to **well under 2.55 ms (~1.2–1.35x)**; round-0 #1/#2 give the exact
  bound. Even conservative parity-per-site clears 1.0x on the transpose savings alone.
- **Risk:** low — no precision change, correctness identical (single-pass `=`, no K-accum
  reorder). Sweep G ∈ {2,4}; watch PSUM/SBUF pressure (the matmul sibling saw B=8 regress
  full-run). ≤5 iterations: G-sweep + n-block width (512-group vs 1024-group like baseline).

### D2 — off-PE lhs transpose via `load_transpose2d` (ENABLER; low risk, small–med gain)
Replace the 512 identity-matmul transposes with `nl.load_transpose2d(v1[b, m_slice, k])`
→ `[k,m]` loaded already-transposed from HBM. **Proven portable at fp32 on this remote**
(rmsnorm `probe_loadtranspose`: full 5-seed PASS, transpose fully hidden, PE stayed 97%).
Removes 512 PE passes AND the transpose PSUM bank + its copy → **frees a PSUM bank so all
output banks fit** and simplifies the D1 schedule. Best evaluated *combined with D1*.
- **Risk:** low (measured-portable). If it fails to lower here, fall back to the PE
  identity-transpose (v1's proven idiom). ≤2 iterations.

### D3 — compensated bf16x2 3-product main matmul (GATED BET; measure-first)
Split each main-matmul operand into bf16 hi/lo limbs; accumulate 3 products
(`hi·hi + hi·lo + lo·hi`, drop `lo·lo`) in fp32 PSUM. **Only pursue if BOTH round-0 gates
pass:** (a) #3 shows fp32/bf16 ratio **> 3** (else 3 products cost more than fp32, and this
regresses exactly like swiglu's all-3 split, 0.409x); (b) #4 offline rel-L2 **< 2e-5**.
- **Upside if it ports:** ~1.2–1.3x *on top of* D1 (matmul/rmsnorm/add_rmsnorm all won big
  here). **Downside if the ratio is ~2:** SKIP entirely — it would raise PE time.
- This is the only lever that touches the fp32 ceiling (§2). Requires an on-device 5-seed
  L2 PASS to promote (offline sim is a green-light, not a guarantee). ≤3 iterations
  (single-precision limb build + resident-limb reuse; combine with D1 banking).

### Closed / not-pursued directions (record-only, do not spend iterations)
- **M-blocking** (the `matmul` sibling's winner): N/A — rhs reload is already eliminated
  (rhs[b] resident once/batch), reads are 34 MB at the floor. Nothing to amortize.
- **Store-burst fattening / output ping-pong** (phase-1 draft's levers): dead — DMA=39%
  hidden, HBMwr at the 1074 MB write floor. Codex already closed this.
- **bf16 output**: forbidden — output IS the final result; the 2e-5 gate bans it.
- **`dma_transpose` fp32**: proven-INFEASIBLE on this remote (rmsnorm probes: fp32 is not
  2-byte, `dma_transpose`/SFKVectorizer crash). Do not re-probe.
- **Vector-engine transpose** (`nc_transpose engine=vector`): rmsnorm MEASURED +2%
  regression (Vec co-bottleneck). Reject.

## 5. Method & discipline (per direction, ≤5 iterations)
- **Noise anchor:** re-run v1 same-session as the control before each comparison (siblings
  saw ~0.08–0.5% jitter; treat a **~1.8–2.5% band** as noise). Promote only OUT-of-band
  wins on a **full 5-seed** run (drop `--fast`); `--fast` (seed 42) only for screening.
- **Evidence per direction:** before/after p50 latency (`verify.py`), the MFU/PE/Vec/Scl/
  DMA/HBM digest, and TRUE PE-active + `matmul_instruction_count` (`dump_metrics`) to
  distinguish real PE-work change from serialization. Record keep/revise/reject with the
  number that decided it.
- **Never regress correctness:** every promoted candidate must clear the 5-seed L2 gate;
  keep v1 as the pure-fp32 fallback (like the rmsnorm/swiglu families).
- **Bookkeeping:** append each perf change to `benchmark.csv`; each candidate to
  `candidates.jsonl` with parent links (DAG); profiling evidence under `profile/`. Kernels
  go in `runs/`; never edit the baseline/reference.

## 6. Expected trajectory
`v1 0.663x → D1 (schedule fix, pure fp32) ~1.2–1.35x → +D2 (off-PE transpose) small gain
→ +D3 (bf16x2, ONLY if round-0 ratio>3 & rel-L2<2e-5) up to ~1.5x.` D1 is the
high-confidence, provable win (baseline reaches 2.55 ms with more PE work); D3 is the
upside gamble that round 0 will accept or kill before we spend a full remote run on it.

--- Original Design Draft End ---
