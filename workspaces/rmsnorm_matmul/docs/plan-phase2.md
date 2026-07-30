# rmsnorm_matmul Phase 2 — Move the x-Transpose Off the Tensor Engine (M4096 N2048 K1024, fp32)

## Goal Description

Start from the best correct kernel, `runs/rmsnorm_matmul_v1.py` (**1.066x** speedup,
latency 0.4716 ms vs baseline 0.502647 ms, full 5-seed PASS), and reduce on-device
latency **without ever regressing correctness**. Correctness is NKIBench's relative-L2
gate `||v_k - v_r||_2 < 2e-5 * ||v_r||_2` on seeds `[0, 21, 42, 63, 84]`, fp32, as
enforced by `verify.py` (`l2_norm_passed`).

Profiling and the theoretical cost model agree that the fp32 main matmul is already at
the systolic floor (PE=97%, MFU=46% is a **structural fp32 ceiling**, not inefficiency),
and that the RMSNorm vector/scalar work and DMA are comfortably hidden under the PE-bound
matmul. The **only non-floor PE work** is the 256 identity-transpose matmuls that move
x's K-dim onto the partition axis (~13.65 µs pure, up to ~27 µs with poorly-amortized
systolic fill). The phase-2 thesis is to **move that transpose off the 97%-busy Tensor
Engine onto an otherwise-idle engine (DMA, or the Vector engine)**, enabled by folding
the per-row `inv_rms` scale into PSUM eviction so the transpose can read **raw** x
directly from HBM.

The algebraic enabler (verified against the NKIBench numpy reference and in the draft on
all 5 seeds): RMSNorm's `inv_rms[m]` is a per-row scalar, so it commutes with the matmul —
`(x·inv_rms) @ w == (x @ w)·inv_rms`. Applying the scale on the **output** at eviction
decouples the norm from the transpose's critical path and lets the transpose source raw x
from HBM.

**Realistic outcome (corrected from the draft's "1.10–1.18x").** Wall-clock latency is
0.4716 ms; the transpose is at most 13.65–27 µs of PE work. Removing 27 µs →
0.4446 ms → 0.502647/0.4446 = **1.130x**; removing 13.65 µs → 0.4580 ms → **1.098x**. A
1.18x total speedup would require removing ~45.6 µs, far more than the transpose costs, so
it is not achievable by transpose removal alone. Moreover, measured latency (~471.6 µs) is
~2x the modeled PE compute (~232 µs), so ~240 µs of wall-clock is non-PE
(overhead/exposed DMA/scheduling gaps); if the transpose partially overlaps today, the
realized win can shrink toward the ~1.8% run-to-run noise band (the sibling `matmul` task
found its analogous lever ≤ 2.5%, below noise). The **honest target** is therefore: a
correct full-5-seed kernel that beats v1 by **more than the ~1.8% noise band** — realistically
low-single-digit % up to **~1.10–1.13x** total if the transpose is genuinely on the critical
path — **or**, if every off-PE transpose route is infeasible/within-noise on this remote
target, a defensible floor-confirmation exit (v1 stays promoted with measured evidence that
it sits at the fp32 PE floor and the only non-floor work cannot be moved off-PE), mirroring
the sibling matmul task's outcome.

**Hard API constraint discovered during planning (shifts the prior toward floor-confirmation).**
The local venv is **client-only** — there is no local `neuronxcc`/NKI package, so all API
availability is remote-only. The NKI API reference *documents* `nisa.dma_transpose` (D2) as
requiring a **2-byte dtype** on both its fast hardware-DGE and software-DGE paths, so a
**4-byte fp32** transpose is very likely rejected by design (independently confirmed by the
sibling matmul task's analysis: "fp32 is ineligible"). The D2 probe is therefore expected to
*confirm ineligibility* rather than open the main swing. Meanwhile the Vector-engine
`nc_transpose` (D3) is documented at a `[32,32]` tile limit (so each `[128,128]` transpose
would need 16 sub-transposes → 256 transposes become ~4096 Vector ops), and its `engine=`
kwarg is in the *same class* the sibling task saw rejected. The one clearly-de-risked piece
is D1: `tensor_scalar` reading a PSUM tile directly is **confirmed working on the remote**
(used in every NKIBench baseline), so the post-scale eviction fold rests on solid ground. Net:
floor-confirmation is the more probable exit, D5 (norm-emits-transposed-layout, uses only
proven primitives) may be the only surviving *off-PE* path, and every route still requires a
recorded remote probe before any redesign.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. Verification uses `verify.py` on the remote trn2 profiler,
which gates on `l2_norm_passed`. The scoring command is:

```bash
python3 \
    ../../verify.py --op rmsnorm_matmul --candidate runs/<file>.py [--fast]
```

- AC-1: **Correctness never regresses (HARD gate).** Any candidate that is *promoted*
  passes the NKIBench relative-L2 criterion on **every** seed in `[0, 21, 42, 63, 84]`,
  validated by a FULL `verify.py` run (no `--fast`). `--fast` (seed 42) is for triage only.
  - Positive Tests (expected to PASS):
    - A full 5-seed `verify.py` run of the promoted candidate reports every seed passing the L2 gate (`correct: 5/5`).
    - The `--fast` seed-42 read passes before the full run is attempted.
  - Negative Tests (expected to FAIL / be rejected):
    - Any single seed reporting `l2_norm_passed = false`.
    - Promoting a candidate on the basis of the `--fast` seed-42 run alone.
    - A candidate that passes numerically but does not complete/return on the remote profiler.

- AC-2: **fp32 numerical fidelity preserved (HARD).** Every `nc_matmul` operand and every
  PSUM accumulation tensor on the productive path stays `np.float32`. No bf16/tf32/fp16 is
  introduced on the matmul path. Any lower-precision experiment (D4) is a **record-only
  calibration probe** that is never promoted and never influences the fp32 acceptance path.
  - Positive Tests (expected to PASS):
    - All matmul operands and PSUM tiles declared `np.float32`; rel-L2 stays well below `2e-5`.
    - A one-shot bf16/tf32 calibration probe, if run, is recorded in `candidates.jsonl` as `is_correctness_candidate: false` / non-promotable.
  - Negative Tests (expected to FAIL / be rejected):
    - Any productive-path operand downcast to bf16/tf32/fp16 to chase PE throughput.
    - A tolerance-scraping result near `2e-5` indicating unintended precision loss.
    - Any promotion decision that cites the D4 probe's latency.

- AC-3: **Post-scale eviction fold (D1) is algebraically correct AND operationally real.**
  The kernel computes `(x @ w)·inv_rms`, where `inv_rms` is the per-row `[128,1]` operand
  applied when the matmul result is moved PSUM→SBUF. Full-seed rel-L2 stays `<< 2e-5`
  (numpy-modeled `4.85e-7`). The reduction remains a single full-1024 free-axis
  `tensor_reduce` with `inv_rms = rsqrt(sumsq·(1/K))`.
  - AC-3.1: The eviction op is confirmed to lower on the remote target either as a genuine
    fused eviction (a `tensor_scalar`/`scalar_tensor_tensor` reading the PSUM tile) **or**,
    if PSUM cannot be a vector/scalar-op source, as an explicit `nl.copy` PSUM→SBUF followed
    by the per-row scale — with the chosen form documented and its cost measured (not assumed).
    - Positive: the per-row `[128,1]` `inv_rms` aligns to the output tile's partition axis (`m_in`); full-seed L2 passes; the scale is applied exactly once per output element.
    - Negative: `inv_rms` applied per-K, per-N, or as a global scalar; the scale dropped; the scale applied to the input *and* the output (double scaling).
  - Positive Tests (expected to PASS):
    - A candidate implementing `(x@w)·inv_rms` passes full 5-seed L2 with margin.
  - Negative Tests (expected to FAIL / be rejected):
    - Output matching an un-normalized `x @ w` (scale silently dropped) or a mis-broadcast scale.

- AC-4: **An off-PE transpose is promoted only on measured, PE-attributed, out-of-noise
  evidence.** A candidate that moves the transpose off the PE (D2 or D3) is promoted only if
  the profiler shows the PE **absolute active time / instruction attribution** actually drops
  (not merely a PE% shift, since PE% can move with total duration) **and** end-to-end latency
  beats v1 by more than the declared noise gate on a FULL 5-seed run.
  - Positive Tests (expected to PASS):
    - Profiler digest shows fewer/absent PE transpose instructions (or lower PE active time) after the change, alongside a full-5-seed latency that beats same-session v1 by more than the noise gate (see DEC-2).
  - Negative Tests (expected to FAIL / be rejected):
    - A latency-only improvement claim with no PE-attribution evidence.
    - A "win" within the ~1.8% noise band (indistinguishable from drift).
    - A change that lowers PE% only because total duration grew (e.g., DMA became the new bottleneck).

- AC-5: **Availability probes gate every redesign (HARD process gate).** Because the local
  venv is **client-only** (no local `neuronxcc`/NKI package to introspect), API availability
  can only be established by a remote compile+run. Before wiring `nisa.dma_transpose` (D2),
  `nisa.nc_transpose(engine=...)` (D3), or a PSUM-sourced eviction op (D1) into the fused
  kernel, a minimal standalone probe confirms the API **compiles, runs, and produces a
  correct result** on the remote target. A probe that *proves an API unavailable* (e.g. the
  documented 2-byte-dtype requirement rejecting fp32 `dma_transpose`) is a **successful,
  valuable** probe outcome, not a failure — it is recorded and the direction is closed.
  - Positive Tests (expected to PASS):
    - A minimal `[128,128]` fp32 `dma_transpose` (D2) probe returns a definitive verdict — either a correct transpose scored with `--fast`, or a recorded compile/dtype rejection (fp32 requires a 2-byte dtype per the API docs) that closes D2.
    - A minimal `nc_transpose(engine=...)` (D3) probe returns a definitive verdict on both the `engine=` kwarg and the `[32,32]` Vector tile limit; recorded as a DAG node.
    - A D1-lowering probe confirms the PSUM tile can be a `tensor_scalar`/`scalar_tensor_tensor` source (documented supported; `tensor_scalar` PSUM input is proven on the remote), and whether the eviction adds PE/vector/scalar work.
  - Negative Tests (expected to FAIL / be rejected):
    - Wiring an un-probed API into the kernel and only discovering the rejection during full integration.
    - Treating "compiles" as sufficient proof for D2/D3 (correctness and latency/attribution must also be measured).
    - Assuming fp32 `dma_transpose` works because the *docs* list the API, without the remote probe.

- AC-6: **Evidence is recorded for every perf-relevant candidate AND every probe.** Each
  perf-relevant candidate gets a `benchmark.csv` row, a `candidates.jsonl` DAG node (with
  `parent` links), and a `profile/` digest capturing PE% **and absolute PE active time**
  before/after. Failed/rejected compile probes are recorded as first-class DAG nodes too
  ("API rejected on remote" is valuable evidence).
  - Positive Tests (expected to PASS):
    - After each candidate/probe, `benchmark.csv` / `candidates.jsonl` / `profile/` contain the new node with parent link and the PE-attribution digest.
  - Negative Tests (expected to FAIL / be rejected):
    - A promoted candidate with no `profile/` PE-attribution evidence.
    - A probe result (including a rejection) that is run but left unrecorded.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A correct (full 5-seed PASS), fp32 kernel that removes the identity-transpose from the
Tensor Engine via whichever off-PE route the remote probes prove viable — in probe-decided
priority: D2 `dma_transpose` (only if the fp32/2-byte gate unexpectedly permits it), D3
`nc_transpose(engine=)` (only if the kwarg and `[32,32]` limit permit it), else D5 (norm emits
the transposed `[k_in, m_in]` layout directly using proven primitives, so no separate transpose
exists) — built on the D1 post-scale eviction fold, beating v1 by more than the noise gate
(realistically up to ~1.10–1.13x total), with full profiler evidence that PE active time
dropped. A single record-only D4 bf16/tf32 calibration probe may be run to re-confirm the fp32
penalty magnitude.

### Lower Bound (Minimum Acceptable Scope)
A defensible phase-2 exit that keeps `runs/rmsnorm_matmul_v1.py` promoted, backed by
**measured evidence** that: (a) v1 sits at the fp32 PE floor, and (b) every attempted off-PE
transpose route (D2, and D3 if applicable) is either unavailable on this remote NKI or fails
to beat the noise band — recorded as DAG nodes and a profiler digest. This mirrors the
sibling matmul task's floor-confirmation outcome and is a valid, complete phase-2 result. At
minimum, the D2 availability probe and a same-session v1 control must be run and recorded.

### Allowed Choices
- Can use: the D1 post-scale eviction fold (proven-safe enabler); `nisa.dma_transpose`
  (D2, gated on its availability probe); `nisa.nc_transpose(engine=...)` (D3, only if the
  kwarg probe compiles); a second raw-x HBM load to feed the norm's free-axis reduce **or**
  computing the norm from the transposed tile (start with the two-read variant, gated by
  measured DMA exposure); the DLoC-style norm-emits-transposed-layout fusion (D5, stretch);
  a single non-promotable D4 calibration probe.
- Cannot use: any change that regresses correctness on any seed; any bf16/tf32/fp16 on the
  productive matmul path; promoting on `--fast` alone; promoting a within-noise result;
  editing the NKIBench benchmark definition (`../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}`);
  hand-tuning a baseline; assuming an API exists without a remote probe.

> **Note on Deterministic Designs**: This plan is exploratory (multiple ranked candidate
> directions with availability gates), not a single deterministic design. The upper and lower
> bounds deliberately diverge: the upper bound is a measured off-PE-transpose win; the lower
> bound is a measured floor-confirmation. Which one is reached depends on remote API
> availability and measured latency, both of which are unknowable without probing.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

The v1 kernel (M-outer loop) currently does, per M-tile: load x `[m_in, k]` → fused
RMSNorm in SBUF (`square → full-K reduce → rsqrt(scale=1/K) → per-row tensor_scalar`) →
transpose 8 normalized `[128,128]` K-sub-tiles to `[k_in, m_in]` **on the PE** via
`nc_matmul(is_transpose=True)` → matmul-accumulate 8 K-tiles into a `[128,512]` fp32 PSUM,
streaming 4 N-chunks; w is fully resident (8 MB), each x loaded once.

The phase-2 restructure, conceptually:

```
# D1 enabler (transpose still on PE): decouple the scale from the transpose
per M-tile:
    load x_sb = [m_in, k]                       # raw x
    inv_rms[m_in,1] = rsqrt( sum_k(x^2) * 1/K ) # norm from raw x, unchanged reduce
    transpose RAW x sub-tiles -> xT[k_in, m_in] # was normalized x; now raw x
    for each N-chunk:
        acc[m_in,512] = sum_kt nc_matmul(xT[kt], w[kt])          # (x @ w) in PSUM
        out_sb = acc * inv_rms[m_in,1]           # per-row scale AT EVICTION (post-scale)
        store out_sb

# D2 swing (transpose off PE): x arrives pre-transposed from HBM
per M-tile:
    load xT_sb = dma_transpose(x[mt])  -> [k_in, m_in]   # HBM->SBUF, no PE transpose
    load x_sb  = x[mt]                 -> [m_in, k]       # 2nd read, only for the norm
    inv_rms = rsqrt(sum_k(x_sb^2)*1/K)
    matmul xT_sb @ w -> acc ; out = acc * inv_rms         # D1 post-scale
# (D2 variant b: derive the norm from xT_sb via a partition-axis reduction, no 2nd read)
```

D3 replaces `dma_transpose` with `nc_transpose(dst, data, engine=<vector>)` on the Vector
engine (only if the `engine=` kwarg compiles — see risks). D5 has the norm pass emit the
`[k_in, m_in]` layout directly (e.g. sum-of-squares via a cross-partition reduction) so no
separate transpose exists at all.

**Probe-first ordering.** Because the local venv is client-only, resolve API availability on
the remote target *before* investing in the redesign: probe D2 (`dma_transpose` fp32
`[128,128]`), D3 (`nc_transpose(engine=)`), and D1's core assumption (can a vector/scalar op
read a PSUM tile directly, or is an `nl.copy` PSUM→SBUF required first?). This resolves the
draft's internal ordering tension (its "Plan of attack" starts with D1, its "Validation plan"
says probes first) in favor of probes-first for the API-dependent pieces, while D1's
*arithmetic* is already proven and can be implemented as the safe base in parallel.

### Relevant References
- `runs/rmsnorm_matmul_v1.py` — the promoted phase-1 kernel and guaranteed fallback; its
  identity-`nc_matmul` transpose idiom is the known-good baseline.
- `../../AccelOpt/NKIBench/reference/rmsnorm_matmul_M4096_N2048_K1024_numpy_1.py` — the numpy
  reference (`normalized = x / sqrt(mean_k(x^2))`, then `@ w`); confirms the per-row commute.
- `workspaces/matmul/candidates.jsonl` — sibling task evidence: `nisa.tensor_copy(engine=)`
  and `nisa.activation(op=nl.copy)` **rejected** on this remote NKI (D3 risk); fp32 is a hard
  ~3.62x PE penalty (MFU~46% structural); its transpose/eviction lever bounded ≤ 2.5% (< noise).
- `profile/rmsnorm_matmul_v1.txt` — the v1 profiler digest (PE=97%, MFU=46%, etc.).
- `kernel-cost-analysis` skill — theoretical per-engine floor to compare against.
- `kernel-optimization-kb` skill — precedents: post-scale/eviction-fold
  (`28094369`, `d22d23db`, `5f08e8cb`, `63e18e33`); off-PE transpose (`597cf19e`,
  `86fd24ec`, `710a49f3`); DLoC RMSNorm (`5795192b`).
- `nki-api-reference` skill — signature references (latest/full API, aspirational vs the
  older remote): `dma_transpose` (`api-nki-isa-memory.md`: `dst` SBUF, `src` HBM/SBUF, **2-byte
  dtype required** on both DGE paths — fp32 ineligible); `nc_transpose(dst, data, engine=...)`
  (`api-nki-isa-memory.md`: PE `[128,128]` / **Vector `[32,32]`** tile limits); `tensor_scalar`
  / `scalar_tensor_tensor` (`api-nki-isa-tensor.md` / `-misc.md`: **PSUM `data` input allowed**;
  for `scalar_tensor_tensor`, `data` and `operand1` cannot both be PSUM). Treat docs as the
  upper-bound surface; the remote target is older — probe before relying on any `engine=` form.

## Dependencies and Sequence

### Milestones

1. **Milestone 1 — De-risk (probes + control).** Establish what is actually available on the
   remote target before any redesign.
   - Phase A: Re-run a same-session v1 control (full 5-seed) to anchor the noise band and give
     a fair comparison point (not just the historical 1.066x).
   - Phase B: Minimal remote probes — D2 `dma_transpose` fp32 `[128,128]`; D3
     `nc_transpose(engine=)`; D1 PSUM-as-vector/scalar-op-source. Each recorded as a DAG node,
     including rejections. Output: which directions are alive.

2. **Milestone 2 — D1 post-scale eviction fold (safe base / enabler).** Implement `(x@w)·inv_rms`
   with the transpose **still on the PE** (transpose raw x instead of normalized x; apply the
   per-row `inv_rms` at PSUM→SBUF eviction using the form the D1 probe confirmed). Prove full
   5-seed correctness. This is *treated as an enabler, not automatically a safe base*: it is
   promoted only if full-seed latency is no worse than v1 within the noise policy, and is used
   for D2/D3 regardless (it decouples the transpose from the norm so raw x can be transposed).
   - Depends on Milestone 1 Phase B (D1 lowering probe).

3. **Milestone 3 — Off-PE transpose swing.** With D1 in place, remove the PE transpose via the
   best *available* route, as determined by the Milestone 1 probes. Note the planning-time
   expectation: D2 (`dma_transpose`) is documented fp32-ineligible (2-byte dtype) and D3's
   `engine=` kwarg is in the class the sibling task saw rejected — so this milestone may find
   **no viable direct off-PE route**, in which case D5 (Milestone 4, norm-emits-transposed
   layout, proven primitives only) becomes the candidate off-PE path. If D2/D3 *do* work: feed
   the norm either via a second raw-x HBM load (start here; x=16 MB, two reads = 32 MB, model
   says DMA ~160→~205 µs < 218 µs PE) or from the transposed tile via a partition-axis
   reduction; the two-read path is explicitly gated by **measured DMA exposure** — if it grows
   exposed DMA or wall time, switch to the norm-from-transposed variant or abandon. Measure
   absolute PE active time + full-5-seed latency.
   - Depends on Milestone 2 (D1) and Milestone 1 (D2/D3 probes).

4. **Milestone 4 — Decide and record.** Keep the best of `{v1, D1, D2, D3}`; full 5-seed before
   any promotion; require the AC-4 out-of-noise + PE-attribution evidence. If an off-PE route
   won and budget remains, explore D5 (norm emits transposed layout). Run the D4 bf16/tf32
   calibration probe once for the record (non-promotable). If every off-PE route is infeasible
   or within-noise, execute the defensible floor-confirmation exit: v1 stays promoted with
   documented evidence.
   - Depends on Milestone 3.

Per-direction iteration cap: **≤ 5 iterations**; stop a direction early on a clear reject.
`--fast` (seed 42) for triage; **full 5-seed** before promoting.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Re-run same-session full-5-seed v1 control; record latency + digest to anchor the noise band | AC-1, AC-6 | coding | - |
| task2 | Write & run minimal remote probe: `dma_transpose` fp32 `[128,128]` (D2) — expected to hit the documented 2-byte-dtype gate; capture the exact verdict (compile/dtype rejection vs correct transpose + `--fast` latency); record DAG node incl. rejection | AC-5, AC-6 | coding | - |
| task3 | Write & run minimal remote probe: `nc_transpose(engine=...)` (D3) — does the `engine=` kwarg compile on this remote NKI, and does the `[32,32]` Vector tile limit hold (→ ~16× sub-tiling)? Record node incl. rejection | AC-5, AC-6 | coding | - |
| task4 | Write & run D1-lowering probe: can `tensor_scalar`/`scalar_tensor_tensor` read a PSUM tile directly, or is `nl.copy` PSUM→SBUF required first? Measure added engine work | AC-3.1, AC-5 | coding | - |
| task5 | Implement D1 post-scale eviction fold: transpose raw x, apply per-row `inv_rms` at eviction; full-5-seed correctness + latency vs v1 | AC-1, AC-2, AC-3 | coding | task1, task4 |
| task6 | (Codex) Review D1 candidate: correctness of the commute, eviction form, any hidden extra pass; confirm it is a valid enabler | AC-3 | analyze | task5 |
| task7 | Implement off-PE transpose swing (D2 preferred, D3 fallback) on top of D1; two-read norm first, gated by measured DMA exposure | AC-1, AC-2, AC-4 | coding | task2, task3, task5 |
| task8 | Collect profiler evidence for the off-PE candidate: absolute PE active time / instruction attribution before/after; full-5-seed latency vs same-session v1 | AC-4, AC-6 | coding | task7 |
| task9 | (Codex) Adversarially review the off-PE candidate's promotion case: is the win outside noise? Is PE attribution real, not a duration artifact? | AC-4 | analyze | task8 |
| task10 | Stretch D5: norm emits transposed `[k_in,m_in]` layout so no separate transpose exists — only if an off-PE route won and budget remains | AC-1, AC-4 | coding | task8 |
| task11 | One-shot D4 bf16/tf32 calibration probe (non-promotable, record-only) to re-confirm the fp32 penalty magnitude | AC-2, AC-6 | coding | task1 |
| task12 | Decide & record: keep best of `{v1,D1,D2,D3}`; update `benchmark.csv`/`candidates.jsonl`/`profile/`; if all off-PE routes infeasible/within-noise, write the floor-confirmation exit | AC-1, AC-6 | coding | task8, task9 |

## Claude-Codex Deliberation

### Agreements
- The fp32 main matmul is at the systolic floor; MFU~46% is a structural ceiling, not
  inefficiency. The only recoverable lever is the 256 identity-transpose matmuls.
- The `(x·inv_rms)@w == (x@w)·inv_rms` commute is algebraically sound (matches the numpy
  reference) and is the correct enabler for reading raw x.
- Probes-first ordering is mandatory: the local venv is client-only, so all API availability
  (`dma_transpose`, `engine=` kwargs, PSUM-as-op-source) can only be established by a remote
  compile+run. "Compiles" is not sufficient for D2/D3 — correctness + latency/attribution too.
- `nc_transpose(engine=...)` (D3) is **likely unavailable**, not a co-equal fallback: the
  sibling task empirically proved engine-steering kwargs are rejected on this remote target,
  and the Vector path is documented at a `[32,32]` tile limit (→ ~16× sub-tiling of each
  `[128,128]`). D3 is a quick reject-probe, not a budgeted fallback path.
- `nisa.dma_transpose` (D2, the draft's "main swing") is **documented fp32-ineligible** — both
  its hardware-DGE and software-DGE paths require a 2-byte dtype — so for this fp32 workload the
  D2 probe is expected to confirm ineligibility, not open a win. This further shifts the prior
  toward the floor-confirmation exit and raises D5's relative importance.
- D1's post-scale eviction fold rests on a **remote-proven** primitive: `tensor_scalar` reading
  a PSUM tile directly is used in every NKIBench baseline, so the eviction-fold lowering is
  low-risk (the D1 probe mainly measures added work, not existence).
- The draft's "1.10–1.18x" is too optimistic. Corrected ceiling from transpose removal alone
  is ~1.10–1.13x total; the honest gate is "beat v1 by more than the ~1.8% noise band," and a
  measured floor-confirmation is an acceptable exit.
- Promotion requires FULL 5-seed PASS, absolute PE-attribution evidence (not PE% alone), and a
  same-session v1 control; failed probes are recorded as first-class DAG nodes.
- fp32 fidelity is non-negotiable on the productive path; D4 is record-only.

### Resolved Disagreements
- **Metric target (Claude proposed correction; Codex confirmed arithmetic).** Draft said
  1.10–1.18x. Resolution: ceiling ~1.10–1.13x total from transpose removal; gate on beating the
  noise band; floor-confirmation is a valid exit. Rationale: 471.6 µs wall-clock, ≤27 µs transpose.
- **Ordering (draft internal contradiction).** "Plan of attack" said D1-first; "Validation"
  said probes-first. Resolution: probes-first for API-dependent pieces (D1 PSUM-source, D2, D3);
  D1's proven *arithmetic* proceeds as the safe base in parallel. Rationale: client-only venv.
- **D1 status.** Codex objected to calling D1 an automatic "safe base." Resolution: D1 is an
  *enabler* — promoted only if full-seed latency is no worse than v1 within noise, but used for
  D2/D3 regardless. Rationale: folding scale into eviction may change instruction mix/scheduling.
- **AC-3 operationalization.** Codex: too algebraic. Resolution: AC-3.1 now requires confirming
  the eviction *lowers* (fused PSUM-source op, or `nl.copy`+scale) and measuring its added work.
- **PE evidence.** Codex: PE% alone is ambiguous. Resolution: AC-4/AC-6 require **absolute PE
  active time / instruction attribution**, since PE% shifts with total duration.
- **Two-read norm path.** Codex: not proven-hidden. Resolution: kept as the starting variant but
  explicitly gated by measured DMA exposure, with norm-from-transposed as the fallback.

### Convergence Status
- Final Status: `converged` (1 convergence round; all REQUIRED_CHANGES incorporated; only two
  genuine user-preference decisions carried to `## Pending User Decisions`).

## Pending User Decisions

- DEC-1: **Priority of D5 (norm-emits-transposed-layout) given that D2 is likely fp32-ineligible
  and D3 is likely rejected.** Planning surfaced that D2 (`dma_transpose`) is documented to
  require a 2-byte dtype (fp32 ineligible) and D3's `engine=` kwarg is in the rejected class —
  so the "D2/D3 both die" branch is the *expected* case, not a corner case. This makes the
  question live now, not hypothetical.
  - Claude Position: Keep D5 as a stretch that runs only if an off-PE route (D2/D3) already won
    and budget remains — it is the largest rewrite with the highest correctness-surface risk, and
    a floor-confirmation exit is a fully acceptable phase-2 result.
  - Codex Position: If D2 is unavailable and D3 is rejected, D5 may be the *only* surviving
    non-speculative off-PE path; consider promoting it above "stretch" in that specific branch.
  - Tradeoff Summary: Given D2/D3 are likely dead, elevating D5 is the main remaining path to any
    off-PE win, but it is a large, risky rewrite that could consume the whole iteration budget on
    an uncertain payoff (~1.10–1.13x ceiling). Keeping it stretch protects the budget and the safe
    floor-confirmation exit; elevating it maximizes the chance of a (small) win. Recommend: attempt
    D5 only if the D1 base is clean AND ≥ 3 iterations remain; otherwise take floor-confirmation.
  - Decision Status: `PENDING`

- DEC-2: **Promotion noise-gate margin.** How much must a candidate beat same-session v1 to be
  promoted?
  - Claude Position: `> 1.8%` (one noise band), the draft's stated band, with same-session v1
    bracketing to reduce drift risk.
  - Codex Position: Consider a stricter `> 2.5–3%` given the sibling's ~1.8% observed noise and
    remote profiler variance, to avoid promoting drift as a win.
  - Tradeoff Summary: A stricter gate reduces false promotions but may reject a genuine ~2%
    transpose win (which is within the plausible outcome range); a looser gate risks promoting
    noise. Depends on how expensive remote runs are and required confidence.
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Phase", "Step", "D1/D2/D3", or similar workflow/plan markers.
- These terms are for plan documentation only, not for the resulting kernel source.
- Use descriptive, domain-appropriate naming in code (e.g. `inv_rms`, `xT`, `post_scale`,
  `dma_transpose_load`) instead.
- Preserve v1's correctness invariants in any new kernel: even divisibility → no masks/partial
  tiles; a single full-1024 free-axis `tensor_reduce` with `inv_rms = rsqrt(sumsq·(1/K))`; fp32
  PSUM accumulation; output tile layout `[m_in(par), n(free)] → v3[mt, :, n0:n0+512]`.
- Keep `runs/rmsnorm_matmul_v1.py` untouched as the guaranteed fallback; new candidates get new
  filenames under `runs/`.

--- Original Design Draft Start ---

# rmsnorm_matmul (M4096 N2048 K1024, fp32) — Phase 2 implementation draft

## Goal

Start from the best correct kernel (`runs/rmsnorm_matmul_v1.py`, **1.066x**, full
5-seed PASS) and reduce on-device latency **without ever regressing correctness**
(NKIBench relative-L2 gate `< 2e-5 * ||v_r||` on seeds `[0,21,42,63,84]`, fp32).
Phase-2 discipline: identify the real bottleneck from profiling + the theoretical
floor, enumerate directions, rank by expected benefit vs risk, and explore each for
**at most five iterations**, collecting before/after latency (`verify.py`) and
profiler evidence to justify keep / revise / reject.

## Where phase 1 left us (measured + modeled)

Phase-1 profiler digest (both `--fast` and full 5-seed identical):

```
latency=0.4716ms  speedup=1.066x
MFU=46%  PE=97%  Vec=15%  Scl=11%  DMA=20%  HBMrd=25MB  HBMwr=34MB
```

Theoretical per-engine cost model (`kernel-cost-analysis`, trn2, this
tiling 32 M-tiles × 4 N-chunks × 8 K-tiles, n=512):

| Engine | work | cost | vs PE |
|---|---|---|---|
| **PE main matmul** | 1024 nc_matmul, dst free=512 | **218.45 µs** | = 100% of the systolic floor |
| **PE transpose** | 256 nc_matmul(is_transpose), dst free=128 | **13.65 µs pure** (up to ~27 µs with weight-fill) | +6.25% (∼9–12.5% w/ fill) |
| Vector (norm reduce + scale) | 32×(reduce+scale) | ~69 µs | 32% — hidden |
| Scalar (square) | 32× | ~27 µs | 12.5% — hidden |
| DMA (x load + w load + out store) | 16+8+32 MB | ~160 µs | 73% of PE — under ceiling |

Two conclusions that set the whole phase:

1. **The main matmul is already at the fp32 systolic floor — zero wasted PE cycles.**
   The tiling is exact (32·128=4096, 8·128=1024, 4·512=2048; k=128, n=512 fill the
   array). There is *no* micro-tiling win left in the matmul itself. This matches the
   sibling `matmul` task's finding that fp32 is a hard ~3.62x PE penalty and MFU~46–49%
   is a **structural ceiling**, not inefficiency (`kda-matmul-progress`).

2. **The single recoverable lever is the 256 identity-transpose matmuls** (moving x's
   K-dim onto the partition axis). They are the *only* non-floor PE work: 6.25% of the
   matmul in the pure model, ~9–12.5% counting the poorly-amortized 128-cycle systolic
   fill of a standalone transpose (consistent with PE=97% measured). Everything else —
   RMSNorm vector/scalar work, DMA — is comfortably **hidden under the PE-bound
   matmul**, so it is not worth touching for its own sake. **DMA is only ~20% busy**
   (fp32 inflates the PE-time denominator), i.e. ~54 µs of headroom under the PE
   ceiling → an idle engine we can offload transposes onto.

Note M-blocking — the sibling `matmul` task's phase-2 win — **does not apply here**:
w is already fully resident (8 MB, loaded once) and each x tile is loaded once, so we
are already compute-bound with no redundant HBM traffic to remove.

## The phase-2 thesis

**Get the x-transpose off the 97%-busy Tensor Engine and onto an idle engine (DMA,
or Vector via `nc_transpose(engine=)`), removing the only non-floor PE work.**
Upper bound from the cost model: PE 232.1 → 218.45 µs ⇒ **~5.9% (up to ~12%) faster**
if the transpose is fully offloaded and hidden. The floor keeps MFU where it is; we
just stop paying the transpose tax on the bottleneck engine.

A key algebraic enabler (verified in numpy, all 5 seeds): **RMSNorm's `inv_rms[m]` is
a per-row scalar, so it commutes with the matmul** —
`(x·inv_rms) @ w == (x @ w)·inv_rms`. Measured rel-L2 of the post-scaled form
`(x@w)·inv_rms` vs reference: **~4.85e-7 « 2e-5** on seeds `[0,21,42,63,84]`
(input-scaled v1 is ~1.3e-7; both pass comfortably). This lets us transpose **raw x**
straight from HBM (no dependency on the norm) and apply the per-row scale on the
*output* at PSUM eviction — decoupling the norm from the matmul's critical path and,
crucially, letting the transpose read x directly from HBM via DMA.

Precedents (`kernel-optimization-kb`) back this ordering:
- **Post-scale / eviction-fold** of a per-row factor: proven, dtype-agnostic
  (`28094369`, `d22d23db`, `5f08e8cb`, `63e18e33` — fold scale into the PSUM→SBUF
  eviction with a single `tensor_scalar`/`scalar_tensor_tensor` instead of a separate
  input pass).
- **Off-PE transpose:** DVE/Vector-engine transpose (`597cf19e`, `tiled_dve_transpose`
  family) is fp32-safe and targets exactly the idle engine in a PE-bound kernel.
  **DMA-transpose** (`86fd24ec`, `710a49f3` +25% on some configs) matches our spare
  DMA — **but every library precedent is gated on 2-byte dtype**, with explicit
  small-tile / overhead-loss warnings for fp32. So DMA-transpose is the highest-ceiling
  but least-certain lever for fp32; treat as an experiment, not a given.
- **DLoC RMSNorm** (`5795192b`, −19.5% at T≥512): norm pass *emits the matmul's staged
  layout directly*. Highest ceiling, largest change; our M=4096 clears its size gate.

## Candidate directions, ranked by benefit ÷ risk

### D1 — Post-scale eviction fold (LOW risk, enabler; do first)
Move `inv_rms` off the input and onto the output: transpose **raw** x, matmul into
PSUM, then apply the per-row `[128,1]` `inv_rms` when copying PSUM→SBUF via
`tensor_scalar(op0=multiply, operand0=inv_rms)` (or `scalar_tensor_tensor`) — this
*replaces* the existing `nl.copy` eviction, adding no new pass.
- **Benefit on its own:** ~neutral latency (both scale passes are hidden under PE;
  input-scale touches 1024 elems/row, output-scale touches 2048 — slightly more, but
  on the idle Vector engine). Its value is as the **enabler for D2/D3**: the transpose
  no longer depends on the normalized values, so it can source raw x from HBM.
- **Correctness:** verified rel-L2 ~4.85e-7 « 2e-5, all 5 seeds. `inv_rms` still a
  per-partition `[128,1]` operand (same shape/idiom as v1). Reduction/rsqrt unchanged.
- **Risk:** low — arithmetic proven, one instruction relocated. Watch: `tensor_scalar`
  reading a PSUM tile (confirmed supported); the per-row operand must align to the
  output tile's partition (m_in) — it does, since output partition = m_in like v1.
- **Iterations:** ~1–2 (implement, verify parity at ≥ v1's 1.066x).

### D2 — Off-PE transpose via `nisa.dma_transpose` (MAIN swing, MEDIUM–HIGH risk)
With D1 in place, load x from HBM already transposed to `[k_in(par), m_in(free)]` via
`nisa.dma_transpose` (HBM→SBUF), eliminating all 256 PE transpose matmuls. The norm
still needs x in `[m_in(par), k(free)]` for the free-axis reduce — so either (a) also
keep the normal x load for the norm (x is only 16 MB; two reads = 32 MB, still far
under the DMA ceiling — DMA ~160→~205 µs, still < 218 µs PE), or (b) compute the norm
from the transposed tile via a partition-axis reduction. Start with (a) — simplest,
and the cost model says DMA has the headroom.
- **Benefit:** removes the 13.65–27 µs of transpose PE ⇒ **~1.06–1.12x on top of v1**
  (target ≈ 0.42–0.44 ms) if hidden under DMA. This is the phase's main expected win.
- **Risk (HIGH, must de-risk first):** (i) **fp32 dma_transpose is unproven** — the
  fast HW-DGE path is 2-byte-only; fp32 falls to a slow path (~50% BW SBUF→SBUF,
  ~90% HBM→SBUF) and library commits *disable* it for small tiles. (ii) The sibling
  task found this **remote NKI is older** and rejected engine-steering kwargs, so
  `dma_transpose` may not exist / may reject args here. **Mitigation: a 1-instruction
  availability probe before committing** to the redesign (see Validation). If it
  errors or measures slower, fall back to D3.
- **Iterations:** ≤3 (probe; wire transposed load + post-scale; measure vs v1;
  optionally the norm-from-transposed variant).

### D3 — Off-PE transpose via `nc_transpose(engine=vector)` (fp32-safe fallback)
If DMA transpose is unavailable/slow, do the `[128,128]` transpose on the **Vector
Engine** instead of the PE: `nisa.nc_transpose(dst, data, engine=<vector>)`. Vector is
only ~32% utilized, so 256 small transposes may hide there and off-load the PE.
- **Benefit:** same PE saving as D2 (~5.9%+), without needing DMA-transpose support.
- **Risk:** MEDIUM — Vector-path `nc_transpose` is documented ≤ `[32,32]` per op (may
  force 4×4 sub-tiling of each 128×128 → 16× the transpose instruction count on
  Vector; must check it still fits under the Vector budget) and the **`engine=` kwarg
  may be rejected on this older remote NKI** (same class of failure the sibling task
  hit with `tensor_copy(engine=)`). Probe the kwarg before committing.
- **Iterations:** ≤2 (probe kwarg + tile limit; measure).

### D4 — Lower-precision matmul (bf16/tf32) — REJECT by gate, 1-shot probe only
The structural ceiling is the fp32 PE rate; bf16 is ~3.6x faster on trn2. But bf16
mantissa error (~4e-3) and even tf32 (~1e-3) are **orders above the 2e-5 gate**, and
the sibling task empirically confirmed bf16 fails correctness here. **Rejected**; at
most a single calibration probe (like the sibling's) to re-confirm the fp32 penalty
magnitude for the record — not a promotable candidate.

### D5 — DLoC-style norm-emits-transposed-layout fusion (STRETCH, only if D2/D3 win)
Have the norm pass produce the transposed `[k_in, m_in]` layout directly (compute the
sum-of-squares as a cross-partition reduction, e.g. `nc_matmul(ones[128,1], x)` or a
partition reduce), so no separate transpose exists at all. Precedent −19.5%. Largest
rewrite and highest correctness-surface risk; only pursue if D2/D3 prove the off-PE
transpose direction and iterations remain.

## Plan of attack (order)

1. **D1** (post-scale eviction fold) — implement on a copy of v1, confirm ≥ 1.066x and
   full-seed parity. This is the safe base and the enabler.
2. **Probe D2** — a tiny standalone `nisa.dma_transpose` fp32 call to confirm it exists
   and lower/measure BW on this remote target *before* the redesign.
   - If OK → wire the transposed HBM load + D1 post-scale; measure vs v1.
   - If unavailable/slower → **D3** (`nc_transpose(engine=vector)`), probing the kwarg
     and the 32×32 tile limit first.
3. Keep the best of {v1, D1, D2, D3}. If an off-PE transpose wins and budget remains,
   consider **D5**; run the **D4** calibration probe once for the record.

Never promote a candidate that fails any seed or regresses below v1's 1.066x. Each
direction gets ≤5 iterations; stop a direction early on a clear reject.

## Correctness invariants (unchanged from v1, must hold)

- Every dim divides evenly → **no masks / partial tiles** anywhere.
- Reduction is a single full-1024 free-axis `tensor_reduce`; `inv_rms = rsqrt(sumsq·1/K)`
  (folded 1/K) — bit-identical role to v1 (rel-L2 1.3e-7). Post-scale variant verified
  rel-L2 4.85e-7 « 2e-5.
- Matmul accumulation stays **fp32 in PSUM** (matches numpy `np.matmul`). Transpose is
  exact in fp32 regardless of engine (identity-matmul / nc_transpose / dma_transpose
  are all value-preserving permutations).
- Output tile layout `[m_in(par), n(free)] → v3[mt, :, n0:n0+512]` unchanged, so
  `inv_rms[m]` (per output partition) broadcasts correctly at eviction.

## Risks / things to watch

- **Older remote NKI API surface.** `dma_transpose` and `engine=` kwargs may not exist
  / may reject args (sibling task precedent). **Always probe with a throwaway call
  before a redesign**; the known-good transpose is the identity-matmul (v1) and the
  profiler-example `matmul_kernel.py` idiom — keep v1 as the guaranteed fallback.
- **fp32 DMA-transpose throughput.** Even if available, small 4-byte tiles may run slow
  enough to erase the PE saving — measure end-to-end, don't trust the model's no-fp32-
  throttle DMA numbers blindly.
- **Second x read** (norm + transposed load) doubles x HBMrd to 32 MB — still under the
  DMA ceiling per the model, but confirm DMA% doesn't climb past PE and flip the
  bottleneck.
- **Noise band.** Sibling task saw ~±1.8% run-to-run; a <2% "win" is noise. Re-run
  controls and require a margin before promoting. Score `--fast` for triage, **full
  5-seed** before promoting.
- **PSUM/SBUF pressure** from any extra resident transposed-x buffer — watch for spills
  in the profiler (sibling's B=8/B=16 regressed on residency).

## Validation plan

1. **Availability probes first** (cheap, before any redesign): a minimal kernel calling
   `nisa.dma_transpose` (D2) and `nisa.nc_transpose(engine=...)` (D3) on a `[128,128]`
   fp32 tile, scored with `--fast`, to confirm the API compiles on this remote target.
2. Per candidate, `--fast` (seed 42) for a quick correctness + latency read:
   ```bash
   python3 \
       ../../verify.py --op rmsnorm_matmul --candidate runs/<file>.py --fast
   ```
3. On PASS **and** a latency ≥ v1 (with margin over the noise band), full 5-seed run
   (drop `--fast`) before recording as promoted.
4. Record each perf change in `benchmark.csv`, each candidate node in `candidates.jsonl`
   (parent = its base, DAG links), and the profiler digest (MFU/PE/Vec/Scl/DMA/HBM)
   under `profile/` — especially PE% before/after to confirm the transpose actually
   left the Tensor Engine.

## Phase-2 success criterion

A correct (full 5-seed PASS) kernel that beats v1's **1.066x** by more than the ~±1.8%
noise band — realistically **~1.10–1.18x** if the off-PE transpose lands (PE drops from
232 µs toward the 218 µs floor). If every off-PE transpose route is infeasible on this
remote NKI or fails to beat noise, the defensible phase-2 exit is: **v1 remains
promoted, with measured evidence that it sits at the fp32 PE floor and the only
non-floor work (the transpose) cannot be moved off-PE on this target** — mirroring the
sibling matmul task's floor-confirmation outcome.

--- Original Design Draft End ---
