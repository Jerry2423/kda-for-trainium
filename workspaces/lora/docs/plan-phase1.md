# lora Phase 1 — First Correct NKI Kernel (fused low-rank residual)

## Goal Description

Produce the FIRST correct NKI kernel for the `lora` operator — `runs/lora_v1.py` —
computing `out = x@w + (x@a)@b` in pure fp32 on AWS Trainium (trn2), passing the
NKIBench relative-L2 gate (`||v_k - v_r||_2 < 2e-5 * ||v_r||_2`, fp32) across all five
seeds `[0, 21, 42, 63, 84]`. Prioritize a clean, fully-understood kernel over speed,
but structure it so the low-rank residual `(x@a)@b` is **fused into the base GEMM's
PSUM output accumulation before eviction** — the intermediate never touches HBM. This
fused shape is both the prompt's stated win and the simplest correct form (one output
store per tile, no separate add pass).

The kernel reuses the proven M-outer structure of the sibling `matmul_v1` kernel, which
passed the same `2e-5` gate at an fp32 floor ~1e-6. Speed levers (wider N-chunk,
resident `b`, M-blocking, bf16x2 split) are explicitly deferred to phase 2/3.

**Shapes / dtype:** x (M=4096, K=5120), w (K=5120, N=12288), a (K=5120, R=128),
b (R=128, N=12288), out (M=4096, N=12288); all fp32. M_TILES=32, K_TILES=40, R=128,
N_TILES=96 (at N_CHUNK=128).

**Kernel signature:** `def kernel(v1, v2, v3, v4)` returning `v5`, where
`v1`=x `(8,4,128,40,128)`, `v2`=w `(40,128,12288)`, `v3`=a `(40,128,128)`,
`v4`=b `(128,12288)`, and `v5`=out `(8,4,128,96,128)`.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification.

- AC-1: **Correctness gate.** `runs/lora_v1.py` passes the NKIBench relative-L2 gate
  (`< 2e-5`, fp32) on every seed in `[0, 21, 42, 63, 84]`. `2e-5` is a HARD requirement
  (the NKIBench correctness contract that `verify.py` gates on via `l2_norm_passed`),
  not an optimization trend. Phase 1 has NO speed target; any speedup is acceptable so
  long as correctness holds.
  - Positive Tests (expected to PASS):
    - `verify.py --op lora --candidate runs/lora_v1.py` reports `l2_norm_passed=True`
      for all 5 seeds in the full (non-`--fast`) run before promotion.
    - The `--fast` run passes during iteration.
  - Negative Tests (expected to FAIL):
    - A kernel with a swapped `(m_hi, m_lo)` decode fails the L2 gate on at least one seed.
    - A kernel that omits the low-rank residual (base GEMM only) fails the L2 gate
      (the residual is not within `2e-5` of zero relative to the full output).

- AC-2: **Fused single-bank output accumulation.** The `(x@a)@b` residual is added into
  the SAME output PSUM tile as the base GEMM `x@w` for each output n-tile, before that
  tile is copied to SBUF and stored. There is NO HBM store or reload of the `(x@a)`
  intermediate or of the `(x@a)@b` residual. A separate, transient PSUM computation for
  the down-projection `tT` (then copied to SBUF) is permitted and expected — "single
  bank" refers only to the final output accumulation, not to a claim that the kernel
  uses exactly one PSUM object.
  - Positive Tests (expected to PASS):
    - Source inspection shows the base-GEMM K-accumulation and the low-rank matmul
      target the same `acc` PSUM tile per n-tile, followed by one `nl.copy` + one
      `nl.store` to `v5`.
    - No `nl.store`/`nl.load` of any `(x@a)` or `(x@a)@b` intermediate to/from HBM.
  - Negative Tests (expected to FAIL):
    - Storing `(x@a)@b` to HBM and reloading it for a separate add pass violates AC-2.
    - Accumulating the residual into a different PSUM tile and adding it post-hoc in a
      second output pass violates the fused-single-bank requirement.

- AC-3: **Correct 2-level M-index.** The kernel decodes `m_hi = mt // 4`, `m_lo = mt % 4`
  for both the `v1` (x) read and the `v5` (out) store, matching the verified layout
  `m = (m_hi*4 + m_lo)*128 + m_in`.
  - Positive Tests (expected to PASS):
    - The host-side layout check (AC-6) confirms `v1[m_hi,m_lo,m_in,kt,ki] == x[(m_hi*4+m_lo)*128+m_in, kt*128+ki]` and the analogous `v5` mapping for sampled indices including nonzero `m_hi` and nonzero `m_lo`.
  - Negative Tests (expected to FAIL):
    - Using a flat `(32,128,...)` M-index (the sibling `matmul` shape) mis-reads/mis-writes rows and fails AC-1.
    - Swapping `m_hi`/`m_lo` (`m_hi = mt % 4`, `m_lo = mt // 4`) fails AC-1.

- AC-4: **`lhs_t` computed once and shared.** For each M-tile, the transposed x tiles
  `lhs_t[kt] = [k_in(par), m_in(free)]` are computed once (40 identity-transpose matmuls)
  and reused as the shared operand for BOTH the base GEMM `x@w` and the down-projection
  `x@a`.
  - Positive Tests (expected to PASS):
    - Source inspection shows a single `lhs_t` buffer produced in the transpose loop and
      consumed by both the down-projection matmul and the base-GEMM matmul.
  - Negative Tests (expected to FAIL):
    - Recomputing (re-transposing) x separately for the down-projection duplicates work
      and is rejected by this criterion.

- AC-5: **Down-projection `tT = (x@a)^T` via matmul, fp32 SBUF-resident before the N loop.**
  `tT` is a single `[R=128, m_in=128]` tile computed by accumulating over K:
  `tT += nc_matmul(stationary=a_local[kt] [k_in,R], moving=lhs_t[kt] [k_in,m_in])`. The
  PSUM tile backing this accumulation is explicitly zero-initialized
  (`nl.zeros(..., buffer=nl.psum)`) before the K loop, and `tT` is copied to fp32 SBUF
  BEFORE the N-tile loop begins (the N loop reuses PSUM banks, so `tT` must survive in
  SBUF, not PSUM).
  - Positive Tests (expected to PASS):
    - The host-side layout check confirms the accumulated `tT` equals `(x@a).T` for the
      sampled M-tile.
    - Source inspection shows `tT`'s PSUM tile zero-initialized before the K loop and
      `nl.copy(..., dtype=np.float32)` to SBUF before the N loop.
  - Negative Tests (expected to FAIL):
    - Leaving `tT` in PSUM across the N loop (where banks are reused for `acc`) corrupts
      it and fails AC-1.
    - Omitting the PSUM zero-init leaves stale accumulator state and fails AC-1.

- AC-6: **Host-side numpy layout check passes before the remote run.** A non-scored
  `runs/_layout_check.py` (sibling-`matmul` precedent) mirrors the kernel's index /
  transpose / accumulation arithmetic in numpy against a locally-computed gold for
  sampled output tiles, and asserts all mappings. It MUST cover: the 2-level M-index for
  `v1`/`v5`; the `v3` (a) mapping `v3[kt,ki,r] == a[kt*128+ki, r]`; the `v4` (b) mapping
  `v4[r,n] == b[r,n]`; the `tT == (x@a).T` down-projection identity; and the fused
  `acc == x_rows@w_cols + (x_rows@a)@b_cols` for at least one sample with nonzero
  `m_hi`, nonzero `m_lo`, nonzero `kt`, nonzero `r`, and a last/near-last `nt` (to catch
  tail n-tile and 2-level-M mistakes).
  - Positive Tests (expected to PASS):
    - `python3 runs/_layout_check.py` exits 0 with all asserts passing.
  - Negative Tests (expected to FAIL):
    - An index/transpose bug in the mirrored arithmetic trips an assert (checked by
      temporarily perturbing an index during development).

- AC-7: **Pure fp32 throughout.** Every allocation (`lhs_t`, `a_local`, `tT`, `acc`,
  loaded `w_tile`, loaded `b_tile`, identity, output staging) is fp32, and every
  PSUM->SBUF copy uses `nl.copy(..., dtype=np.float32)` (fp32-preserving, per the
  `matmul_v1` precedent). No bf16/fp16 anywhere in phase 1.
  - Positive Tests (expected to PASS):
    - Source inspection shows `dtype=np.float32` on all buffers and copies.
  - Negative Tests (expected to FAIL):
    - Any implicit or explicit cast of `tT`, `acc`, or `lhs_t` to bf16/fp16 (which would
      compound error across the 128-term contraction) is rejected.

- AC-8: **Evidence recorded.** A `benchmark.csv` perf row and a `candidates.jsonl` row
  (`id=lora_v1`, `parent=baseline`) with per-seed pass/rel-L2, latency, speedup, and
  worst rel-L2 are written, plus a per-engine / MFU / HBM profile digest under
  `profile/`. (Bookkeeping criterion: it records the correctness evidence from AC-1 but
  does not itself gate numerical correctness — a missing profile digest must not be read
  as a correctness failure if AC-1 evidence exists.)
  - Positive Tests (expected to PASS):
    - `benchmark.csv` gains a `lora_v1` row; `candidates.jsonl` gains the `lora_v1` DAG
      entry; `profile/` holds the digest for this candidate.
  - Negative Tests (expected to FAIL):
    - Promoting the candidate with no `candidates.jsonl` entry or no recorded per-seed
      L2 evidence violates the workflow contract.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A single clean `runs/lora_v1.py` implementing the fused-single-bank fp32 M-outer kernel
with `N_CHUNK=128` (one output n-tile per chunk, direct store to `v5[m_hi,m_lo,:,nt,:]`),
`a` fully resident in SBUF, `b` streamed per n-tile, the identity-transpose helper and
`nl.copy` fp32 idioms taken from `matmul_v1`, and an accompanying `runs/_layout_check.py`
host-side numpy validation. Passes all 5 seeds at the full (non-`--fast`) measurement,
with evidence recorded per AC-8.

### Lower Bound (Minimum Acceptable Scope)
The same fused-single-bank fp32 kernel passing all 5 seeds under the `2e-5` gate, with
the correct 2-level M-index and the low-rank residual fused into the base GEMM's output
accumulation (no HBM round-trip). The host-side layout check exists and passes.

The upper and lower bounds nearly converge: this is a highly deterministic phase-1
design. The only material latitude is in debugging aids and comment/verification
thoroughness, not in the kernel's algorithmic shape.

### Allowed Choices
- Can use: the `matmul_v1` identity `nc_matmul(is_transpose=True)` transpose idiom;
  `nl.copy(..., dtype=np.float32)` for PSUM->SBUF; `a_local` fully resident across all
  M-tiles; streaming `w` and `b` tiles per n-tile; a host-side numpy layout check
  patterned on the sibling `_layout_check.py`.
- Allowed DURING debugging only: a conservative two-accumulator fallback (base GEMM and
  low-rank residual in separate PSUM tiles, added before store) to isolate a fused-path
  bug, and/or a temporary auxiliary check-path for `(x@a)`/`tT`. The PROMOTED candidate
  MUST use the fused single-bank form (AC-2); these fallbacks are diagnostic scaffolding,
  not the deliverable.
- Cannot use: bf16/fp16 or any dtype trick (phase 1 is pure fp32 — AC-7); `N_CHUNK=512`
  with a strided 4-subtile store, resident `b`, or M-blocking (all phase-2 levers);
  editing the NKIBench baseline, reference, or `summary.json`; a flat `(32,128,...)`
  M-index.

> **Note on Deterministic Designs**: The draft specifies a highly deterministic design
> (fixed operator, verified tiled layout, fixed fused-single-bank fp32 approach with
> `N_CHUNK=128`). The path boundaries above reflect this narrow constraint: upper and
> lower bounds nearly coincide, and the algorithmic choice is fixed per the draft
> specification.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

Reuse `matmul_v1`'s M-outer structure. Constants: `M_TILES=32`, `K_TILES=40`, `R=128`,
`N=12288`, `N_CHUNK=128`, `N_TILES=96`.

Preload once (reused across all 32 M-tiles):
- 128×128 identity into SBUF (transpose helper) — `nl.shared_constant(np.identity(128))`
  loaded to SBUF, per `matmul_v1`.
- `a` fully resident: `a_local[K_TILES=40, par_dim(128), 128]` (~20 KB/part).

Per M-tile `mt` (decode `m_hi = mt // 4`, `m_lo = mt % 4` for `v1`/`v5`):

```
# 1. Transpose x once (shared by both GEMMs).
lhs_t = sbuf[K_TILES, par(128), 128]          # [k_in(par), m_in(free)]
for kt in range(40):
    lhs_sb = load v1[m_hi, m_lo, :, kt, :]     # [m_in(par), k_in(free)]
    psum_t = nc_matmul(lhs_sb, identity, is_transpose=True, is_moving_onezero=True)
    lhs_t[kt] = copy(psum_t, dtype=fp32)       # [k_in(par), m_in(free)]

# 2. Down-projection tT = (x@a)^T = [R, m_in], accumulate over K.
tT_psum = psum.zeros([par(128)=R, 128])        # explicit zero-init
for kt in range(40):
    tT_psum += nc_matmul(stationary=a_local[kt] [k_in,R], moving=lhs_t[kt] [k_in,m_in])
tT = copy(tT_psum, dtype=fp32)                 # to SBUF, BEFORE the N loop

# 3. Per n-tile: base GEMM + fused low-rank into ONE output bank.
for nt in range(96):
    acc = psum.zeros([par(128)=m_in, 128])     # explicit zero-init
    for kt in range(40):                       # base x@w
        w_tile = load v2[kt, :, 128*nt : 128*nt+128]      # [k_in, 128]
        acc += nc_matmul(stationary=lhs_t[kt] [k_in,m_in], moving=w_tile [k_in,128])
    b_tile = load v4[:, 128*nt : 128*nt+128]              # [r=128, 128]
    acc += nc_matmul(stationary=tT [r,m_in], moving=b_tile [r,128])  # (x@a)@b, fused
    out_sb = copy(acc, dtype=fp32)
    store v5[m_hi, m_lo, :, nt, :] = out_sb
```

Why correct: `lhs_t.T @ w_tile + tT.T @ b_tile = x@w + (x@a)@b`. The residual matmul
`nc_matmul(stationary=tT [R,m_in], moving=b_tile [R,128]) = tT.T @ b_tile = (x@a)@b`
produces `[m_in, n]`, the identical output layout to the base GEMM, so it accumulates
directly into the same bank.

Matmul-count sanity (N_CHUNK=128 config): per M-tile there are `40` transpose matmuls +
`40` down-projection matmuls + `96*40 = 3840` base matmuls + `96` fused low-rank matmuls.
The low-rank path adds ~`(40 + 96) / 3840 ≈ 3.5%` over the base GEMM — cheap, as expected.

Suggested debug ladder (isolate failures fast): (a) run the host-side layout check first;
(b) verify a base-only variant matches an M-index-adapted `matmul_v1`; (c) add `tT`;
(d) add the fused low-rank residual; (e) remote-verify with `--fast`, then full 5-seed.

SBUF budget (trn2 ~192 KB/part): `lhs_t` 20 KB + `a_local` 20 KB + identity 0.5 KB +
`tT` 0.5 KB + streamed `w`/`b` tiles (small, double-bufferable) — comfortable. If the
compiler shows pressure holding `lhs_t[40]` + `a_local` + identity + `tT` + output
staging simultaneously, the fallback is to stream `a` per M-tile (phase-2 tradeoff).
PSUM: one 128-wide fp32 bank live for `acc` + a transient bank for the transpose and for
`tT` — within the 8-bank budget. (Do not assume sub-bank packing; NKI may reserve at bank
granularity — this is fine here.)

### Relevant References
- `workspaces/matmul/runs/matmul_v1.py` — the proven M-outer fp32 GEMM this kernel
  extends; identity-transpose idiom and fp32 `nl.copy` come from here.
- `workspaces/matmul/runs/_layout_check.py` — host-side numpy index/transpose/accumulation
  validation pattern to mirror for `runs/_layout_check.py`.
- `../AccelOpt/NKIBench/reference/lora_M4096_N12288_K5120_R128_numpy_1.py` — the numpy
  reference (`forward`, `transform_to_nki_inputs`, `transform_nki_outputs`) defining the
  operator and the verified tiled layout.
- `../AccelOpt/NKIBench/kernels/lora_M4096_N12288_K5120_R128_0.py` — the NKIBench baseline
  kernel (structure reference only; do not edit; do not hand-tune).
- `../../verify.py` — the scorer; gates on `l2_norm_passed`, prints per-engine/MFU/HBM digest.

## Dependencies and Sequence

### Milestones
1. **Host-side layout validation**: build and pass `runs/_layout_check.py`.
   - Phase A: mirror `transform_to_nki_inputs` (numpy reshapes) for x/w/a/b/out.
   - Phase B: assert the 2-level M-index, the a/b mappings, `tT == (x@a).T`, and the
     fused `acc` gold for sampled tiles including nonzero `m_hi`/`m_lo`/`kt`/`r` and a
     tail `nt`.
2. **Kernel implementation**: write `runs/lora_v1.py` (depends on Milestone 1 confirming
   the arithmetic).
   - Phase A: preload identity + `a_local`; M-outer transpose loop producing `lhs_t`.
   - Phase B: down-projection `tT` (zero-init PSUM, accumulate over K, copy to fp32 SBUF).
   - Phase C: per-n-tile fused base+low-rank accumulation and store.
3. **Remote verification & evidence**: score and record (depends on Milestone 2).
   - Phase A: `--fast` iteration until it passes; localize any failure via the debug ladder.
   - Phase B: full (non-`--fast`) 5-seed run before declaring correct.
   - Phase C: write `benchmark.csv`, `candidates.jsonl` (`id=lora_v1`, `parent=baseline`),
     and the `profile/` digest.

Dependency summary: Milestone 1 gates Milestone 2 (the local check must pass before
spending remote runs); Milestone 2 gates Milestone 3. Within Milestone 2, Phase A → B → C
are ordered by data dependency (`lhs_t` before `tT`; `tT` before the fused N loop).

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Write `runs/_layout_check.py`: mirror the tiled layout and assert the 2-level M-index, a/b mappings, `tT=(x@a).T`, and the fused `acc` gold on sampled tiles (incl. nonzero m_hi/m_lo/kt/r and a tail nt) | AC-3, AC-5, AC-6 | coding | - |
| task2 | Implement `runs/lora_v1.py`: preload identity + `a_local`; M-outer `lhs_t` transpose loop | AC-4, AC-7 | coding | task1 |
| task3 | Implement the down-projection `tT` (zero-init PSUM, K-accumulate, fp32 copy to SBUF before N loop) | AC-5, AC-7 | coding | task2 |
| task4 | Implement the per-n-tile fused base-GEMM + low-rank single-bank accumulation and `v5` store | AC-2, AC-3, AC-7 | coding | task3 |
| task5 | Verify with `verify.py` (`--fast` iteration, then full 5-seed); localize failures via the debug ladder | AC-1 | coding | task4 |
| task6 | Record evidence: `benchmark.csv` row, `candidates.jsonl` (`id=lora_v1`, `parent=baseline`), `profile/` digest | AC-8 | coding | task5 |
| task7 | (Optional, if L2 fails and root cause is unclear) Consult Codex on the failing tile/arithmetic before further code changes | AC-1 | analyze | task5 |

## Claude-Codex Deliberation

### Agreements
- The core algebra is correct: `lhs_t = x_tile.T`; `tT += a_local.T @ lhs_t = (x@a)^T`;
  `acc += tT.T @ b_tile = (x@a)@b_tile`; and `lhs_t.T @ w + tT.T @ b = x@w + (x@a)@b`.
- `nc_matmul` axis placement is consistent (contraction on `par_dim(128)` for the base
  GEMM, the down-projection, and the low-rank update).
- The fused final accumulation is clean: base GEMM and low-rank update land in the same
  output PSUM tile before one eviction/store.
- `N_CHUNK=128` is internally consistent: 96 n-tiles, direct `v5[m_hi,m_lo,:,nt,:]`
  stores, `3840` base + `40` down-projection + `96` low-rank matmuls per M-tile.
- Reusing `matmul_v1`'s identity transpose and `nl.copy(..., dtype=np.float32)` is the
  right correctness-first choice; the fp32 `tT` copy is a hard requirement, not an optimization.
- The host-side layout check and staged debug ladder are appropriate, not over-engineered
  for this risk level.

### Resolved Disagreements
- **Draft cost figure (24 vs 96)**: the draft quoted "~6% / 24 extra matmuls" (the
  `N_CHUNK=512` count) while choosing `N_CHUNK=128` for phase 1. Resolution: restate for
  the chosen config — `96` fused low-rank matmuls + `40` down-projection per M-tile over
  `3840` base matmuls (~3.5%). Both Codex passes agree the two configs must not be mixed.
- **"single-bank" ambiguity**: Codex flagged that the AC should not read as "only one
  PSUM object in the kernel," since `tT` legitimately uses a separate transient PSUM
  computation. Resolution: AC-2 now scopes "single bank" to the FINAL output accumulation
  and explicitly permits the transient `tT` PSUM + copy.
- **Explicit PSUM zero-init**: Codex required stating that both `tT`'s PSUM tile and each
  n-tile `acc` are explicitly zero-initialized (`nl.zeros(buffer=nl.psum)`) to avoid
  stale/lazy state. Resolution: folded into AC-5 (tT) and the design pseudocode (acc),
  and into task3/task4.
- **AC-8 status**: Codex noted evidence-recording is bookkeeping, not a correctness gate.
  Resolution: AC-8 explicitly labeled bookkeeping — a missing profile digest is not a
  correctness failure if AC-1 evidence exists.
- **Layout-check coverage**: Codex required the local check to also assert the `a` (`v3`)
  and `b` (`v4`) mappings and include a sample with nonzero `m_hi`/`m_lo`/`kt`/`r` and a
  tail `nt`. Resolution: folded into AC-6 and task1.
- **Kernel signature/output shape AC**: Codex asked for an explicit signature/output-shape
  statement. Resolution: added to the Goal Description and reflected in AC-2's store test.
- **N_CHUNK=128 vs 512** (Codex v1 QUESTIONS_FOR_USER): resolved from the draft's own
  decision (option b) — phase 1 uses `N_CHUNK=128`; `512` is a phase-2 lever. Both passes concur.
- **Identity-transpose helper & fp32-preserving copy** (Codex v1 QUESTIONS_FOR_USER):
  resolved from repo evidence — `matmul_v1` provides both the proven identity
  `nc_matmul(is_transpose=True)` helper and `nl.copy(..., dtype=np.float32)`, which passed
  the same `2e-5` gate. No open question remains.

### Convergence Status
- Final Status: `converged`
- Rounds executed: 1 first-pass analysis + 1 second-pass reasonability review. The second
  pass returned no blocking `UNRESOLVED` items ("No blocking user decision. The plan is
  reasonable after tightening the initialization and AC wording"); all `REQUIRED_CHANGES`
  are deterministic tightenings folded into the ACs and tasks above.

## Pending User Decisions

None. Both Codex passes converged with no blocking user decision. Every `QUESTIONS_FOR_USER`
item from the first pass was resolved from the draft's own specification or from
repository evidence (the proven `matmul_v1` idioms), and the second pass reported no
`UNRESOLVED` opposite opinions. The single quantitative metric — the `2e-5` relative-L2
gate — is confirmed a HARD requirement (the NKIBench correctness contract, gated by
`verify.py`); phase 1 carries no speed threshold.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g., `lhs_t`, `a_local`,
  `tT`, `acc`, `m_hi`, `m_lo`, `nt`, `kt`), mirroring the sibling `matmul_v1` style.

--- Original Design Draft Start ---

# lora — Phase 1 draft: first correct NKI kernel

## Goal

Produce the FIRST correct NKI kernel for `lora`, passing the NKIBench relative-L2
gate (`2e-5`, fp32) across all five seeds `[0,21,42,63,84]`. Prioritize a clean,
fully-understood kernel over speed — but structure it so the low-rank update is
**fused into the base GEMM's output accumulation** (the prompt's stated win), since
that fused shape costs no more than the un-fused one and is the natural correct form.

## Operator

`out = x@w + (x@a)@b`  (fp32)

| tensor | math shape | role |
|--------|-----------|------|
| `x` | (M=4096, K=5120) | activations |
| `w` | (K=5120, N=12288) | base weight |
| `a` | (K=5120, R=128) | low-rank down-projection |
| `b` | (R=128, N=12288) | low-rank up-projection |
| `out` | (M=4096, N=12288) | result |

A large base matmul `x@w` (dominant: 4096×5120×12288) plus a cheap low-rank residual
`(x@a)@b` (R=128). `x@a` is (M,R); `@b` lifts it back to (M,N).

## Tiled layout (from the reference `transform_to_nki_inputs`) — VERIFIED numerically

The kernel entry is `kernel(v1, v2, v3, v4)` and returns `v5`:

- `v1` (x): `(8, 4, 128, 40, 128)` = `[m_hi, m_lo, m_in, k_tile, k_in]`.
  Row `m = (m_hi*4 + m_lo)*128 + m_in`, col `k = k_tile*128 + k_in`.
  There are `8*4 = 32` M-tiles of 128 rows, `40` K-tiles of 128.
- `v2` (w): `(40, 128, 12288)` = `[k_tile, k_in, n]`. Already `[k_in(contraction), n]`.
- `v3` (a): `(40, 128, 128)` = `[k_tile, k_in, r]`. Already `[k_in(contraction), r]`.
- `v4` (b): `(128, 12288)` = `[r, n]`. Already `[r(contraction), n]`.
- `v5` (out): `(8, 4, 128, 96, 128)` = `[m_hi, m_lo, m_in, n_tile, n_in]`.
  Row `m = (m_hi*4 + m_lo)*128 + m_in`, col `n = n_tile*128 + n_in`. `96` N-tiles of 128.

I confirmed each decomposition with a numpy reshape/index round-trip (all True).

Note the M-tile index in `v1`/`v5` is a **2-level** `(m_hi, m_lo)` pair (8×4), unlike
the sibling `matmul` case whose x/out were a flat `(32, 128, ...)`. Everything else
(contraction on partition for w/a/b, N free) matches the sibling `matmul` exactly.

## Tensor-engine contract (recap from sibling `matmul_v1`)

`nisa.nc_matmul(stationary, moving) = stationary.T @ moving`, with the **contraction
dim on the PARTITION axis** of both operands, both operands resident in SBUF.

- `w`, `a`, `b` all already have their contraction dim (`k_in` resp. `r`) as the
  first/partition axis in the tiled layout — load them and use directly, **no transpose**.
- `x` tiles arrive as `[m_in(par), k_in(free)]` — contraction `k_in` is on the FREE
  axis, so each must be transposed to `[k_in(par), m_in(free)]` via the identity
  `nc_matmul(is_transpose=True)` idiom before use. This transposed `x` tile
  (`lhs_t`) is the shared operand for BOTH `x@w` and `x@a`.

## Design — M-outer, single fused PSUM accumulation

Reuse the proven M-outer structure of `matmul_v1`. Constants: `M_TILES=32`,
`K_TILES=40`, `R=128`, `N=12288`, `N_CHUNK=512` (one fp32 PSUM bank), `N_CHUNKS=24`.

Preload once (reused across all 32 M-tiles; both are small):
- 128×128 identity into SBUF (transpose helper).
- `a` fully resident: `a_local[K_TILES, par_dim(128), 128]` = 40·128·4 = 20 KB/part.
- `b` streamed per N-chunk for phase 1 (`[r=128, 512]`, 2 KB/part transient); note
  preloading `b` resident (24·512·4 = 48 KB/part) is a clean phase-2 lever since `b`
  is reused across all M-tiles.

Per M-tile `mt` (decoded to `m_hi = mt//4`, `m_lo = mt%4` for `v1`/`v5` indexing):

1. **Transpose x once (shared).** For each of 40 K-tiles: load `x` tile
   `[m_in(par), k_in(free)]`, transpose → `lhs_t[kt] = [k_in(par), m_in(free)]`
   in resident SBUF (`[K_TILES, par_dim(128), 128]`, 20 KB/part).

2. **Low-rank down-projection, transpose-free.** Compute `tT = (x@a)ᵀ`, a single
   `[R=128, m_in=128]` tile, by accumulating over K:
   `tT += nc_matmul(stationary=a_local[kt] [k_in,R], moving=lhs_t[kt] [k_in,m_in])`
   → `a.Tᵀ... = [R, m_in]`. Verified numerically that this equals `(x@a).T`. Keep
   `tT` in SBUF (copy out of its PSUM bank) so it is available to every N-chunk.

3. **Per N-chunk (24 chunks of 512): base GEMM + fused low-rank into ONE bank.**
   ```
   acc = psum.zeros([m_in=128, 512])
   for kt in 40:                     # base x@w
       load w tile [k_in=128, 512]
       acc += nc_matmul(stationary=lhs_t[kt] [k_in,m_in], moving=w_tile [k_in,512])
   # FUSE low-rank in the SAME bank — no HBM round-trip for the intermediate:
   load b tile [r=128, 512]
   acc += nc_matmul(stationary=tT [r=128, m_in], moving=b_tile [r=128, 512])
   copy acc -> out_sb ; store to v5[m_hi, m_lo, :, 2*c : 2*c+... ]
   ```
   `nc_matmul(stationary=tT [R,m_in], moving=b_tile [R,n]) = tT.T @ b = (x@a)@b`
   `= [m_in, n]`, identical output layout to the base GEMM, so it accumulates
   directly. One extra matmul per N-chunk (24 total) + 40 down-proj matmuls per
   M-tile — ~6% over the 960 base matmuls/M-tile. Cheap, as expected.

   **Store index detail:** N_CHUNK=512 spans 4 output N-tiles of 128. `v5`'s N axis
   is `[n_tile(96), n_in(128)]`. Either (a) store with a reshaped/strided write across
   the 4 sub-tiles the 512-chunk covers, or (b) — simpler and less bug-prone for a
   first correct kernel — set `N_CHUNK=128` so each chunk maps to exactly one `n_tile`
   (`nc = c`), giving 96 chunks. Start with option (b) for phase-1 correctness; option
   (a) / 512-wide is a phase-2 tile-width lever. (Even at 128-wide, each PSUM tile is
   one bank's worth; correctness is unaffected, only matmul granularity.)

## Why this is correct and fusion-clean

- Pure fp32 `nc_matmul` throughout — matches the sibling `matmul_v1`, which passed
  the same `2e-5` gate with a fp32 floor ~1e-6. No dtype tricks in phase 1.
- The low-rank result is added into the base GEMM's PSUM bank **before eviction**, so
  the `(x@a)@b` intermediate never touches HBM — exactly the prompt's stated win, and
  it is also the simplest correct form (one output store per tile, no separate add pass).
- `x` is transposed exactly once per M-tile and reused by both GEMMs; `w/a/b` need no
  transpose. Minimal instruction surface.

## SBUF/PSUM budget (per partition, trn2 ~192 KB SBUF)

- `lhs_t` 20 KB + `a_local` 20 KB + identity 0.5 KB + `tT` 0.5 KB + streamed
  `w`/`b` tiles (2 KB each, double-bufferable) ≈ 45 KB — comfortable.
- PSUM: one 512-wide fp32 bank live for `acc`; a transient bank for the transpose
  and for `tT`. Well within the 8-bank budget.

## Validation & evidence

Score from `workspaces/lora/`:
```
python3 \
    ../../verify.py --op lora --candidate runs/lora_v1.py --fast
```
Gate on `l2_norm_passed` across all 5 seeds (drop `--fast` before promoting).
Record the perf row in `benchmark.csv`; add a `candidates.jsonl` row
(`id=lora_v1`, `parent=baseline`) with seeds, latency, speedup, worst rel-L2, and the
per-engine / MFU / HBM digest under `profile/`.

## Risks / watch-items

- **M-tile 2-level index** (`m_hi,m_lo`) is the one layout difference from the sibling
  `matmul` — get the `v1`/`v5` indexing right (verified above); a swapped pair silently
  scrambles rows and fails L2.
- **`tT` PSUM→SBUF copy** must happen before the N-chunk loop reuses PSUM banks;
  keep `tT` in SBUF, not PSUM, across the chunk loop.
- **Store granularity** for N_CHUNK=512 across 4 `n_tile`s — the reason phase 1 uses
  N_CHUNK=128 (one chunk = one `n_tile`) to keep the first correct kernel unambiguous.

## Phase-2/3 outlook (not this phase)

- Widen `N_CHUNK` to 512 (fewer, larger matmuls; strided 4-subtile store).
- Preload `b` resident (48 KB/part) — reused across all M-tiles, kills 32× reloads.
- M-blocking to amortize `w` reloads (the base GEMM reloads all of `w` per M-tile).
- bf16x2 3-product split on the base GEMM if it stays PE-bound and the fp32 speedup
  floor is the ceiling (the promoted lever on every sibling GEMM: matmul 1.274×,
  transpose_matmul 1.334×). Guard with the offline rel-L2 simulator as usual.

--- Original Design Draft End ---
