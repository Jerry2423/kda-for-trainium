# matmul (M4096 N12288 K5120, fp32) — Phase 1: First Correct NKI Kernel

## Goal Description

Implement the first **correct** NKI kernel for the dense GEMM `out = lhs @ rhs`
(M=4096, K=5120, N=12288, fp32) on AWS Trainium (trn2), with a single
`@nki.jit def kernel(v1, v2)` entry point whose signature matches the NKIBench
baseline's tiled inputs. The kernel must consume the tiled inputs
(`v1 (32,128,40,128)`, `v2 (40,128,12288)`) and produce the tiled output
(`v3 (32,128,12288)`), and must pass NKIBench's relative-L2 correctness gate
across all five seeds `[0, 21, 42, 63, 84]`.

Phase 1 prioritizes a clean, fully-understood, correct kernel over speed. The
loop structure should nonetheless be non-pathological for a compute-bound GEMM
(the M-outer structure below), so we do not start from a structure that must be
thrown away in phase 2. Speedup over the baseline (13.578506 ms) is a secondary,
best-effort target this phase.

## Acceptance Criteria

- AC-1: The kernel passes NKIBench's relative-L2 correctness gate on all five
  seeds `[0, 21, 42, 63, 84]` (`||v_k - v_r||_2 < 2e-5 * ||v_r||_2`, fp32),
  measured through `verify.py` on the remote profiler.
  - Positive Tests (expected to PASS):
    - `python3 ../../verify.py --op matmul --candidate runs/<file>.py --fast`
      reports `PASS` (seed 42) with a scored latency printed.
    - Full run without `--fast` reports `correct: 1/1` and every per-seed
      `l2_norm_passed` is true across `[0,21,42,63,84]`.
  - Negative Tests (expected to FAIL when the kernel is wrong):
    - A kernel that swaps the output orientation (writes `[n, m]` instead of
      `[m_in, n]` into `v3[mt, :, n0:n0+512]`) fails the L2 gate.
    - A kernel that omits the lhs transpose (feeds `lhs_tile [m_in, k_in]`
      directly as the `nc_matmul` stationary operand) fails the L2 gate.
    - A kernel that downcasts any matmul operand, the transpose output, the
      accumulator, or the store to bf16/tf32 fails the 2e-5 L2 gate.
  - AC-1.1 (diagnostic guidance, not a hard gate): passing the official 2e-5 L2
    gate on all seeds is the correctness requirement. The mean relative L2 is
    *also* read as a sanity signal — a structurally correct fp32 kernel should
    land well under 2e-5 (typically ≪ 1e-5). A result that merely squeaks under
    2e-5 (e.g. 1.9e-5) is treated as suspect and investigated before promotion,
    but a clear pass is NOT rejected solely for missing an arbitrary < 1e-6
    target (fp32 accumulation-order differences can legitimately sit above 1e-6).

- AC-2: The kernel is a single self-contained `@nki.jit def kernel(v1, v2)` that
  matches the baseline's tiled I/O contract exactly and requires no changes to
  `verify.py`, the adapter, or the NKIBench reference/baseline.
  - Positive Tests (expected to PASS):
    - The kernel file lives under `workspaces/matmul/runs/`, defines exactly one
      `kernel(v1, v2)` returning a `(32,128,12288)` `nl.shared_hbm` tensor, and
      the adapter assembles + runs it without signature errors.
    - `resolve_case(..., candidate_kernel_path=runs/<file>.py)` +
      `verify.py --op matmul --candidate ...` executes end-to-end (compile +
      profile) with no "no profiling_results" / signature / import errors.
  - Negative Tests (expected to FAIL):
    - A kernel exposing a different signature (e.g. `kernel(lhs, rhs, out)`) or
      returning the wrong shape is rejected by the profiler / correctness path.
    - Editing any file under `../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json}`
      is disallowed (guarded by CLAUDE.md); candidate must stand alone in `runs/`.

- AC-3: All numeric paths remain fp32 end-to-end. Every allocated buffer
  (`lhsT`, PSUM accumulator, `out_sb`), every constant (the identity), every
  `nl.load`/`nl.copy`, and the returned `nl.shared_hbm` output tensor are fp32
  or inherit fp32 from fp32 sources. `nc_matmul` operands are both fp32 (no
  tf32/fp32 mix). No implicit or explicit downcast anywhere on the numeric path.
  (`nl.store` itself carries no dtype; correctness is that the SBUF value it
  writes and the HBM tensor it targets are both fp32.)
  - Positive Tests (expected to PASS):
    - Reading the source shows every `nl.ndarray`/`nl.zeros`/`nl.load`/`nl.copy`
      and `nl.shared_constant` on the numeric path is fp32; the returned tensor
      is `np.float32` `shared_hbm`.
  - Negative Tests (expected to FAIL):
    - Any tile declared `bf16`/`float16`/`tf32`, or a `nc_matmul` with tf32/fp32
      mixed operands, degrades the L2 margin and is rejected.

- AC-4: The tiled index mapping matches the numpy reference exactly (verified in
  numpy), so no seed produces a transposed or mis-tiled result.
  - `v1[mt,mi,kt,ki] == lhs[mt*128+mi, kt*128+ki]`,
    `v2[kt,ki,n] == rhs[kt*128+ki, n]`,
    `v3[mt,mi,n] == out[mt*128+mi, n]`.
  - Positive Tests (expected to PASS):
    - A standalone numpy check reproduces `out = lhs @ rhs` from the tiled
      layout using the kernel's exact index arithmetic (partition = `m_in` for
      output, contraction on `k_in`).
  - Negative Tests (expected to FAIL):
    - An off-by-tile index (e.g. `rhs[kt, :, n0:n0+512]` read with the wrong
      `kt` stride, or `lhsT` indexed with `40` on the partition axis) fails L2.

- AC-5: Evidence is recorded for this candidate: a row in `benchmark.csv`, a node
  in `candidates.jsonl` (parent = baseline, forming a DAG), and a profiler-metric
  digest saved under `profile/`.
  - Positive Tests (expected to PASS):
    - After scoring, `benchmark.csv` gains one row
      (`timestamp,op,candidate,parent,passed,latency_ms,speedup,notes`) and
      `candidates.jsonl` gains one JSON object with a `parent` link.
    - `profile/` contains the captured metric digest (MFU / PE / Vec / Scl / DMA
      / HBM) for this candidate's run.
  - Negative Tests (expected to FAIL):
    - Scoring a candidate without appending to `benchmark.csv` / `candidates.jsonl`
      leaves the evidence trail incomplete (fails the workflow requirement).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A single correct `kernel(v1, v2)` using the **M-outer** structure: load a
`[128,128]` fp32 identity once; for each of the 32 M-tiles, transpose that
M-tile's 40 K-sub-tiles of `lhs` (`[m_in,k_in]` → `[k_in,m_in]`) into an SBUF
`lhsT` buffer via the identity `nc_matmul(is_transpose=True)` idiom; then stream
the 24 N-chunks of width 512, accumulating over the 40 K-tiles into a
`[128,512]` fp32 PSUM tile, copying each finished tile to SBUF and storing it to
`v3[mt, :, n0:n0+512]`. All loops use static bounds. Optionally, a small on-host
numpy sanity check of the index arithmetic is written before remote scoring. The
implementation may mirror the proven indexing of the NKIBench baseline and the
profiler's `examples/matmul_kernel.py`.

### Lower Bound (Minimum Acceptable Scope)
Any single correct `kernel(v1, v2)` that passes AC-1 on all five seeds and
satisfies AC-2/AC-3/AC-4, even if it is slower than the baseline, as long as it
is fp32-correct and its structure is understood tile-by-tile (not blindly copied
without understanding). A K-blocked variant (keeping only 4–8 transposed K-tiles
resident at a time instead of all 40) is acceptable if the full-residency
version hits an SBUF/compiler limit — **but the fallback must still accumulate
all 40 K-tiles into the same PSUM tile before any store to `v3`**. The correct
fallback nesting is `for mt: for n_chunk: acc=0; for k_block: {transpose block;
accumulate block into acc}; store acc`. Never write a partial-K result to `v3`
(that would require an explicit fp32 read-modify-write of HBM, which we do not
do).

### Allowed Choices
- Can use: `nisa.nc_matmul` (incl. `is_transpose=True` identity idiom),
  `nl.load`/`nl.store`/`nl.copy`, `nl.ndarray`/`nl.zeros` in `nl.sbuf`/`nl.psum`/
  `nl.shared_hbm`, `nl.shared_constant` for the identity, `nl.affine_range`/
  `nl.sequential_range`, `par_dim`. K-tile residency may be full (40) or blocked
  (4–8). Loop order may be adjusted (M-outer preferred; N-outer permitted as a
  fallback if residency/compile issues arise).
- Cannot use: any dtype other than fp32 on the numeric path; masking/remainder
  logic (all dims divide evenly, so none is needed); edits to NKIBench
  reference/baseline/summary; hand-tuning of a baseline; a top-level `runs/` or
  `outputs/` outside the task workspace.

> **Note on Determinism**: The math (fp32 dense GEMM), the tiled I/O contract,
> and the transpose requirement are fixed by the harness — those are not
> choices. The genuine degrees of freedom are loop order, K-tile residency
> (full vs blocked), and range types. Upper and lower bounds converge on
> "one correct fp32 M-outer kernel"; the bounds differ mainly in whether a
> pre-scoring numpy sanity check and the M-outer (vs fallback) structure are used.

## Feasibility Hints and Suggestions

> **Note**: Reference only — conceptual, not prescriptive.

### Conceptual Approach

Why a transpose is needed: `nisa.nc_matmul(stationary, moving)` computes
`stationary.T @ moving` with the **contraction dim (K) on the partition axis of
BOTH operands**. `rhs` tiles arrive as `[k_in(par=128), n(free)]` — K already on
partitions, usable directly as `moving`. `lhs` tiles arrive as
`[m_in(par=128), k_in(free=128)]` — K on the free axis, so each must be
transposed to `[k_in(par), m_in(free)]` before use as `stationary`. After the
transpose, `stationary.T @ moving = lhs_tile @ rhs_tile = [m_in, n]`, which lands
with `m_in` on the output partition axis and `n` on the output free axis — the
exact `v3[mt, :, n0:n0+512]` orientation (this orientation must be checked in
code, per Codex).

```
identity[par_dim(128),128] fp32  <- nl.load(nl.shared_constant(np.identity(128, f32)))
# lhsT SBUF buffer, NKI-legal layout: kt is a LEADING index dim, par_dim is the
# partition axis, m_in is the free axis (mirrors the baseline's v7/v9 shapes):
#   lhs_t = nl.ndarray((40, nl.par_dim(128), 128), dtype=f32, buffer=nl.sbuf)
#           # [kt, k_in(par)=128, m_in(free)=128]

for mt in affine_range(32):                       # 32 M-tiles (output row blocks)
    # Transpose this M-tile's lhs: 40 K-sub-tiles [m_in,k_in] -> lhs_t[kt] = [k_in,m_in]
    for kt in affine_range(40):
        # lhs_tile = v1[mt, :, kt, :]              # [m_in(par)=128, k_in(free)=128]
        # transpose via identity matmul -> PSUM [k_in, m_in], then copy to SBUF
        psumT[k_in, m_in] = nc_matmul(stationary=v1[mt,:,kt,:], moving=identity,
                                      is_transpose=True, is_moving_onezero=True)
        lhs_t[kt, :, :]   = copy(psumT)            # SBUF [kt, k_in(par)=128, m_in(free)=128], fp32

    for n0 in [0, 512, ..., 11776]:               # 24 static N-chunks of width 512
        acc[par_dim(128),512] = psum_zeros(fp32)  # output tile [m_in, n_chunk]
        for kt in affine_range(40):               # accumulate ALL 40 K tiles before store
            # rhs_tile = v2[kt, :, n0:n0+512]      # [k_in(par)=128, 512(free)]
            acc[:, :] += nc_matmul(stationary=lhs_t[kt], moving=v2[kt,:,n0:n0+512])
        out_sb[par_dim(128),512] = copy(acc)      # PSUM -> SBUF, fp32
        store v3[mt, :, n0:n0+512] = out_sb        # [m_in, n_chunk], fp32
```

Accumulation primitive (per Codex CORE_RISK): use `acc[...] += nisa.nc_matmul(...)`
into a single PSUM tile allocated with `nl.zeros(..., buffer=nl.psum)` — the K
loop forms one accumulation group before the PSUM→SBUF eviction, exactly as the
baseline (`v10[...] += nisa.nc_matmul(...)`) does. Do not introduce a separate
tensor-add.

SBUF layout legality (per Codex REQUIRED_CHANGE): `lhs_t` must be allocated so
that `nl.par_dim(128)` is the partition axis and `kt` is a **leading index
dimension**, not itself placed on the partition axis — e.g.
`nl.ndarray((40, nl.par_dim(128), 128), ...)` indexed `lhs_t[kt]` →
`[k_in(par), m_in(free)]`. This mirrors the baseline's `v7`/`v9` SBUF shapes,
which are proven to compile. Do NOT use an array-of-tiles form that puts `kt`
ahead of the partition axis in a way the baseline doesn't demonstrate.

SBUF residency (M-outer, full K residency): `lhsT` = 40 × `[128,128]` fp32 =
40·128·4 = **20 KB per partition** (192 KB budget) — comfortable, but the budget
must also count the identity tile, the streamed `rhs` tile, `out_sb`, and any
compiler temporaries (per Codex). If allocation/compile issues appear, fall back
to K-blocking (4–8 resident transposed tiles).

HBM cost note (correcting Codex's TECHNICAL_GAP): M-outer re-reads `rhs` once per
M-tile = `251.7 MB × 32 = 8.05 GB` of rhs reads. Codex's own formula
`32·40·128·12288·4` **also equals 8.05 GB** — its "≈80 GB" figure was a 10×
arithmetic slip; the draft's "~8 GB" stands. This op is compute-bound
(515 GFLOP; baseline ≈38 TFLOP/s effective), so rhs reload is not the phase-1
gate.

Validation sequencing (per Codex MISSING_REQUIREMENT — isolate layout bugs
early): before full remote scoring, run a host-side numpy check that reproduces
`out = lhs @ rhs` from the tiled layout using the kernel's exact index arithmetic
and the transpose orientation. Then `verify.py --fast` (seed 42) for a quick
correctness+latency read; then full 5-seed. Optionally test a single `(mt, n0)`
path first if a correctness failure needs isolating.

### Relevant References
- `../AccelOpt/NKIBench/reference/matmul_M4096_N12288_K5120_numpy_2.py` — the
  numpy reference: `get_inputs`, `forward`, `transform_to_nki_inputs`,
  `transform_nki_outputs`; defines the exact tiled layout.
- `../AccelOpt/NKIBench/kernels/matmul_M4096_N12288_K5120_0.py` — the baseline
  kernel; proven transpose-via-identity + K-accumulate + `[128,512]` PSUM
  indexing to mirror.
- the profiler backend's canonical single-file matmul example — the same
  transpose idiom and load/store structure to mirror.
- `verify.py` (repo root) + `adapter/nkibench_case.py` — how the candidate is
  assembled and scored; `_l2_gate` gates on `l2_norm_passed`.
- `.claude/skills/nki-api-reference` — `nc_matmul` / `nc_transpose` / PSUM
  bank / dtype constraints (fp32 dst on trn2, moving free ≤ 512).

## Dependencies and Sequence

### Milestones
1. Layout & API grounding (done in research):
   - Phase A: Confirm tiled index mapping in numpy (done — AC-4 mapping holds).
   - Phase B: Confirm `nc_matmul` semantics, transpose idiom, fp32 PSUM 512
     limit, output orientation (done via api-docs + baseline).
2. Draft → plan (this document):
   - Phase A: Codex first-pass analysis folded in (accumulation primitive named,
     fp32-everywhere, static bounds, orientation check, HBM number corrected).
   - Phase B: Convergence review + final plan.
3. Implement first correct kernel:
   - Step 1: Write `runs/matmul_v1.py` (M-outer, full K residency, fp32).
   - Step 2: Host-side numpy sanity check of index/transpose arithmetic.
   - Step 3: `verify.py --fast` (seed 42) → fix any correctness failure
     (K-block fallback if SBUF/compile issue).
   - Step 4: Full 5-seed `verify.py` → confirm AC-1 across all seeds.
4. Record evidence (AC-5):
   - Step 1: Append `benchmark.csv` row + `candidates.jsonl` node (parent =
     baseline).
   - Step 2: Save profiler metric digest under `profile/`.

Dependencies: Milestone 3 depends on 1–2; Step 3 depends on Step 1; Step 4
depends on a passing Step 3. The K-block fallback depends only on observing an
SBUF/compile failure in Step 3.

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Write host-side numpy check reproducing `out=lhs@rhs` from tiled layout with the kernel's exact index/transpose arithmetic; verify at least one full `(mt, n_chunk)` tile using the exact K-loop accumulation order (not only a global reshape) | AC-4 | coding | - |
| task2 | Implement `runs/matmul_v1.py`: M-outer, identity-transpose lhs, K-accumulate into `[128,512]` fp32 PSUM, store `v3`; add a source comment at the store tying output axes to `v3[mt, :, n0:n0+512]` (partition=`m_in`, free=`n`) | AC-1, AC-2, AC-3, AC-4 | coding | task1 |
| task3 | Score with `verify.py --fast` (seed 42); diagnose any L2 failure (orientation/index/dtype) distinctly from any compile/SBUF-resource failure so the K-block-fallback decision is traceable; apply K-block fallback only on a resource/compile limit | AC-1, AC-1.1, AC-3 | coding | task2 |
| task4 | Full 5-seed `verify.py` run; confirm all seeds pass L2 | AC-1 | coding | task3 |
| task5 | Record `benchmark.csv` row, `candidates.jsonl` node (parent=baseline), profiler digest under `profile/` | AC-5 | coding | task4 |
| task6 | (Optional) Codex review of the final kernel source for orientation/dtype/accumulation correctness before promotion | AC-1, AC-3 | analyze | task2 |

## Claude-Codex Deliberation

### Agreements
- fp32 must hold across the entire numeric path (loads, identity, transpose out,
  PSUM accumulator, PSUM→SBUF copy, store); 2e-5 rel-L2 forbids bf16/tf32.
- The lhs transpose is mandatory (K must be on the partition axis of both
  `nc_matmul` operands); rhs tiles are already correctly oriented.
- No masking is needed (32·128=4096, 40·128=5120, 24·512=12288 all exact); use
  static loop bounds for compiler friendliness.
- Output tile orientation (`[m_in, n]`) and store indexing into
  `v3[mt, :, n0:n0+512]` must be asserted in code, not assumed.
- Mirror the proven baseline / canonical-example indexing for `load`,
  `nc_matmul`, PSUM copy, and store rather than translating freely.
- Record evidence (benchmark.csv, candidates.jsonl, profile/).

### Resolved Disagreements
- HBM reload magnitude: Codex claimed rhs reloads total ≈80 GB ("off by 10x")
  vs the draft's ~8 GB. Verified in numpy: `251.7 MB × 32 = 8.05 GB`, and Codex's
  own formula `32·40·128·12288·4 = 8.05 GB`. **Resolution: the draft's ~8 GB is
  correct**; Codex's 80 GB was the arithmetic slip. Either way the op is
  compute-bound, so this does not change the M-outer decision.
- Accumulation primitive concern: Codex flagged that `acc += nc_matmul(...)` may
  need an explicit add. **Resolution**: the baseline uses exactly
  `psum[...] += nisa.nc_matmul(...)` (accumulation group in PSUM), so the `+=`
  form is legal and idiomatic; no separate tensor-add. Named explicitly in the
  plan.
- Single-tile-first vs full loops for first correctness: adopted as a *debugging*
  fallback (isolate a `(mt, n0)` path only if L2 fails), not the default path;
  the default writes the full M-outer kernel and validates with a host numpy
  check first.

### Convergence Round 1 (second Codex pass, reviewing candidate plan v1)
Codex found **no conceptual blockers** and agreed the M-outer / full-`lhsT`
structure, fp32 discipline, and validation sequence are sound. It raised four
`REQUIRED_CHANGES`, all accepted and applied:
1. Make the `lhs_t` SBUF layout NKI-legal and explicit — `par_dim(128)` is the
   partition axis, `kt` a leading index dim (mirrors baseline `v7`/`v9`).
   Applied to the Conceptual Approach pseudocode + a dedicated note.
2. Clarify the K-block fallback must accumulate all 40 K-tiles into one PSUM tile
   before any store (no partial-K writes to `v3`). Applied to the Lower Bound.
3. Reword AC-1.1 from a hard `< 1e-6` gate to diagnostic guidance (the 2e-5 gate
   is the requirement). Applied to AC-1.1.
4. Tighten AC-3 dtype wording (`nl.store` carries no dtype; correctness is that
   all buffers/constants/loads/copies/PSUM and the returned HBM tensor are fp32).
   Applied to AC-3.
`OPTIONAL_IMPROVEMENTS` folded in: source comment at the store tying axes to
`v3[mt, :, n0:n0+512]` (AC-5/Impl Notes); numpy check verifies one full
`(mt, n_chunk)` tile with the exact K accumulation order, not just a global
reshape (task1); record compile/resource failures separately from L2 failures
so the fallback decision is traceable (task3).

Convergence matrix (round 1):
| Topic | Claude | Codex | Resolution |
|---|---|---|---|
| lhs_t SBUF layout | array-of-tiles pseudocode | needs par_dim-first legal layout | resolved (applied) |
| K-block fallback order | noted as fallback | must accumulate all K before store | resolved (applied) |
| AC-1.1 strictness | < 1e-6 target | too strict as hard gate | resolved (softened) |
| AC-3 dtype wording | "store declares dtype" | store carries no dtype | resolved (reworded) |
| Empirical unknowns (SBUF compiles? metric field names? actual L2 margin) | — | flagged | deferred to implementation (task3/task5) |

Round 2 not required: no `REQUIRED_CHANGES` remain and Codex's only `UNRESOLVED`
items are empirical (resolved by running the kernel), not plan disagreements.

### Convergence Status
- Final Status: `converged` (first-pass analysis + one convergence round; all
  CORE_RISKS, MISSING_REQUIREMENTS, and REQUIRED_CHANGES are folded into
  ACs/hints or resolved. The one numeric disagreement (HBM GB) was settled by
  direct computation. Remaining Codex items are empirical and deferred to the
  implementation loop; Codex's QUESTIONS_FOR_USER are answerable from the repo
  and recorded below with answers rather than left pending.)

## Pending User Decisions

Codex raised four `QUESTIONS_FOR_USER`; all are answerable from the harness, so
none block implementation. Recorded here with answers for traceability:

- DEC-1: Required kernel filename/signature?
  - Claude Position: Signature is fixed — `@nki.jit def kernel(v1, v2)` returning
    `(32,128,12288)` `nl.shared_hbm`; filename is free under `runs/` (e.g.
    `runs/matmul_v1.py`). The adapter swaps only the kernel source.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: Resolved from `adapter/nkibench_case.py` (`kernel_fn="kernel"`)
    and the baseline signature. Decision Status: `Resolved — kernel(v1,v2), file free under runs/`.
- DEC-2: Can `verify.py --fast` run locally, or is validation always remote?
  - Claude Position: Compilation + measurement are **remote** (this Cloud Desktop
    has no Trainium); only host-side numpy index checks are local. Use
    `--fast` (seed 42) remotely for quick reads, full 5-seed before promotion.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: Per CLAUDE.md / verify.py. Decision Status: `Resolved — remote profiler only`.
- DEC-3: Must the output be fp32 in memory?
  - Claude Position: Yes — `v3` is declared `np.float32` `shared_hbm`; the L2 gate
    is fp32 and the reference is fp32 `np.matmul`. Decision Status: `Resolved — fp32 output`.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: Fixed by the baseline/reference contract.
- DEC-4: Optimizing only single-core latency, or other core configs later?
  - Claude Position: Scoring is single-core (`--num-cores 1`, `--logical-nc-config=1`)
    per the NKIBench contract; multi-core is out of scope for this task.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: Fixed by `verify.py` defaults / baselines.json. Decision Status: `Resolved — single-core`.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology
  such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code (e.g. `identity_local`,
  `lhs_t`, `psum_acc`, `out_tile`) rather than workflow markers.
- fp32 everywhere on the numeric path; assert output orientation with a comment
  tying tile axes back to `v3[mt, :, n0:n0+512]`.

--- Original Design Draft Start ---

# matmul (M4096 N12288 K5120, fp32) — Phase 1 implementation draft

## Goal

Produce the **first correct** NKI kernel for the dense GEMM `out = lhs @ rhs`
(M=4096, K=5120, N=12288, fp32), passing NKIBench's relative-L2 gate
(`||v_k - v_r|| < 2e-5 * ||v_r||`) on all five seeds `[0,21,42,63,84]`. Prefer a
clean, understood, correct kernel over speed; leave aggressive tuning to
phase 2/3. But choose a loop structure that is already reasonable for a
compute-bound op so we don't start from a pathological baseline.

## What the operator is

Plain dense matmul. No fusion, no epilogue. The only subtlety is the **tiled
layout** the harness hands us and the **transpose** the Tensor Engine forces.

## Tiled layout (from the numpy reference)

`transform_to_nki_inputs` reshapes the natural inputs (row-major):

- `lhs (4096, 5120)` -> `v1 (32, 128, 40, 128)` = `[m_tile, m_in, k_tile, k_in]`
  - `v1[mt, mi, kt, ki] == lhs[mt*128 + mi, kt*128 + ki]`  (verified in numpy)
- `rhs (5120, 12288)` -> `v2 (40, 128, 12288)` = `[k_tile, k_in, n]`
  - `v2[kt, ki, n] == rhs[kt*128 + ki, n]`  (verified in numpy)

Output `v3 (32, 128, 12288)` = `[m_tile, m_in, n]`, later reshaped to
`(4096, 12288)` by `transform_nki_outputs`, so
`v3[mt, mi, n] == out[mt*128 + mi, n]`.

Dimensions in tiles: `M_TILES=32`, `K_TILES=40`, `N=12288`. All the 128s are
exact (4096/128=32, 5120/128=40), and 12288 = 24 * 512, so N tiles evenly into
512-wide PSUM-bank chunks. **No masking/remainders needed** — every tile is full.

## Hardware constraints that shape the kernel

From `kernel-cost-analysis` grounding + `nki-api-reference`:

- `nisa.nc_matmul(stationary, moving)` computes `stationary.T @ moving`. The
  **contraction dim must be on the partition axis of BOTH operands** (<=128).
  - `stationary` free dim -> output **partition** dim (our M, <=128).
  - `moving` free dim -> output **free** dim (our N, <=512 for fp32 PSUM bank).
- PSUM: dst must be fp32 on trn2; a single bank holds **512 fp32** in the free
  dim. So accumulate into `[128, 512]` PSUM tiles.
- K>128 handled by looping K tiles with `accumulate` into the same PSUM tile
  (accumulation group), then one copy PSUM->SBUF per output tile.
- **fp32 is mandatory.** rel-L2 2e-5 is far tighter than bf16/tf32 round-off, so
  we cannot downcast the matmul inputs for throughput. The compute floor is the
  fp32 PE-array rate; MFU is what we optimize, not precision.

### The transpose problem

`nc_matmul` needs K on partitions for both operands.

- `rhs` tile `v2[kt, :, n0:n0+512]` is `[k_in=128 (par), 512 (free)]` — K already
  on the partition axis. **Use directly as the `moving` operand.** ✓
- `lhs` tile `v1[mt, :, kt, :]` is `[m_in=128 (par), k_in=128 (free)]` — K is on
  the **free** axis. We must transpose it to `[k_in (par), m_in (free)]` before
  it can be the `stationary` operand.

Transpose idiom (same one the NKIBench baseline and the profiler's canonical
`examples/matmul_kernel.py` use): load a `[128,128]` identity into SBUF once,
then `nisa.nc_matmul(stationary=lhs_tile, moving=identity, is_transpose=True,
is_moving_onezero=True)` writes the transposed tile `[k_in, m_in]` into PSUM;
copy it to SBUF. After transpose, `stationary = lhsT[kt] = [k_in(par), m_in(free)]`,
`moving = rhs[kt] = [k_in(par), n(free)]`, so
`stationary.T @ moving = lhs_tile @ rhs_tile = [m_in, n]`. Correct. ✓

## Loop structure (Phase-1 choice)

Two candidate orders, and why I pick M-outer:

- **N-outer** (rhs stays resident, re-transpose lhs per N-block): would
  re-run the lhs transpose 24x (once per 512-wide N chunk). For a compute-bound
  GEMM that is pure PE waste (transpose runs on the same Tensor Engine) — bad.
- **M-outer** (transpose each M-block's lhs once, stream all of N): the lhs
  transpose runs exactly once per (m_tile, k_tile). Transpose cost is
  32*40 = 1280 `nc_matmul`s of `[128,128]`; the productive matmul is
  32*40*24 = 30720 `nc_matmul`s of `[128 x 512]`. Transpose overhead is a few %
  of PE time. rhs gets re-read from HBM per M-tile (32x, ~8 GB), but this op is
  compute-bound so HBM reload is not the gate in phase 1. **Pick M-outer.**

Chosen structure (simple, one M-tile at a time — minimal SBUF, easy to verify):

```
identity[128,128] <- load once

for mt in range(32):                      # 32 M-tiles (output rows)
    # transpose this M-tile's lhs: all 40 K-tiles -> lhsT[k_tile][k_in, m_in]
    for kt in range(40):
        lhs_tile = v1[mt, :, kt, :]        # [m_in(par)=128, k_in(free)=128]
        psumT = nc_matmul(lhs_tile, identity, is_transpose=True,
                          is_moving_onezero=True)   # -> [k_in, m_in] in PSUM
        lhsT[kt] = copy(psumT)             # SBUF [k_in(par)=128, m_in(free)=128]

    for n0 in range(0, 12288, 512):        # 24 N-chunks of 512
        acc = psum_zeros([128, 512])       # output tile [m_in, n_chunk]
        for kt in range(40):               # accumulate over K
            rhs_tile = v2[kt, :, n0:n0+512]        # [k_in(par)=128, 512(free)]
            acc += nc_matmul(lhsT[kt], rhs_tile)   # [m_in, 512], accumulate
        out_sb = copy(acc)                 # PSUM -> SBUF
        store v3[mt, :, n0:n0+512] = out_sb
```

SBUF residency per M-tile: `lhsT` = 40 * [128 x 128] fp32 = 40 * 64 KB-tile
= 2.56 MB total, i.e. 40*128*4 = 20 KB per partition (192 KB budget) — fits with
lots of headroom. rhs tiles are streamed (loaded per (n0, kt)); a natural
phase-2 improvement is to block N so a loaded rhs tile feeds several things, but
phase 1 keeps it simple and correct.

## Correctness reasoning

- Every dim divides evenly (32*128=4096, 40*128=5120, 24*512=12288) → no
  masks, no partial tiles.
- The transpose is exact in fp32 (identity matmul, `is_moving_onezero` just a
  perf hint, not a numeric change). Internal matmul accumulation is fp32.
- K accumulation order: summing 40 K-tiles in fp32 in PSUM. Relative-L2 over the
  whole tensor is tolerant to fp32 summation order at 2e-5; the reference is
  also fp32 `np.matmul`. Expect comfortable pass.

## Risks / things to watch

- **Index/layout bug** producing a transposed or mis-tiled output — the most
  likely failure. Mitigate by mirroring the baseline's proven indexing exactly
  where possible and reasoning tile-by-tile.
- PSUM over-allocation if I keep too many `[128,512]` banks live — keep a single
  `acc` per N-chunk (8 banks total; one 512-tile = 1 bank).
- SBUF overflow if lhsT residency is mis-sized — 20 KB/partition is safe.

## Validation plan

1. `--fast` (seed 42, few iters) first for a quick correctness+latency read:
   `python3 \
       ../../verify.py --op matmul --candidate runs/<file>.py --fast`
2. On PASS, full 5-seed run (drop `--fast`) before recording as promoted.
3. Record perf row in `benchmark.csv`, candidate node in `candidates.jsonl`
   (parent = baseline), and the profiler metric digest under `profile/`.

## Phase-1 success criterion

L2 gate passes on all five seeds. Speedup is secondary this phase, but the
M-outer structure should already be in the ballpark of the baseline rather than
pathologically slow. Any speedup >= ~1.0x with a correct result is an acceptable
phase-1 exit; real MFU work is phase 2.

--- Original Design Draft End ---
