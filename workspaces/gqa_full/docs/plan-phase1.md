# gqa_full — Phase 1 Plan: First Correct NKI Kernel

## Goal Description

Produce the first **correct** NKI kernel for the `gqa_full` operator (NKIBench case
`0`): grouped-query full (non-causal) softmax attention on AWS Trainium (trn2), fp32.
Shapes (natural layout): `q (1,4096,16,128)`, `k,v (1,4096,8,128)`; `B=1, N=4096,
QH=16, KH=8, n_rep=QH/KH=2, D=128`.

The reference computes, per query head `qh` (its kv head is `kh = qh // 2`, because
`xk = np.repeat(k, n_rep=2, axis=head)`):

```
S = q_h @ k_h.T / sqrt(D)        # [N_q, N_k] scores
A = softmax_over_Nk(S)           # row-softmax over the KEY axis
O = A @ v_h                      # [N_q, D] context
```

So the operator is exactly a **per-head `bmm_softmax` (scores + row-softmax over N)
followed by a second matmul (context)**. `bmm_softmax` is a solved, promoted,
5-seed-correct sibling on this harness (`bmm_softmax_v4`, on-device rel-L2 2.57e-6);
its softmax epilogue is reused verbatim, with a `1/sqrt(D)` scale added.

The single, hard success condition for this phase is passing the NKIBench relative-L2
correctness gate across all five seeds. The design is a per-head fusion: build one
query tile's full score row in SBUF, softmax it in place over the 4096-wide key axis,
consume it immediately in the context matmul, and discard — so the `N×N` scores never
spill to HBM. Beating the 15.579 ms baseline is expected as a by-product (the baseline
spills the whole score matrix to HBM and round-trips it) but is **explicitly deferred
to phase 2/3 and is NOT a gate for this phase**. Prioritize a clean, obviously-correct
kernel where every tile index is understood, over speed.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. **AC-1 is the only hard gate**; AC-2 through AC-7 are
correctness-supporting properties and promotion sanity checks that make an AC-1 pass
trustworthy rather than accidental.

- AC-1: **(HARD GATE)** The kernel passes the NKIBench relative-L2 correctness gate for
  every seed in `[0, 21, 42, 63, 84]` at `rel_tol = 2e-5`, fp32
  (`||v_k - v_r||_2 < 2e-5 * ||v_r||_2`), as reported by `verify.py`'s
  `l2_norm_passed` flag. This is the sole formal phase-1 success condition.
  - Positive Tests (expected to PASS):
    - A full (non-`--fast`) `verify.py --op gqa_full --candidate runs/gqa_full_v1.py`
      run reports `l2_norm_passed = True` for all 5 seeds (5/5 correct).
    - A `--fast` run of the same candidate reports the L2 gate passing on its measured
      seed(s) before the full 5-seed confirmation.
  - Negative Tests (expected to FAIL, i.e. be caught):
    - A kernel with the head map reversed (`qh = grp*8 + kh` instead of `qh = 2*kh + grp`)
      or the key/value subtile flattening mismatched fails at least one seed.
    - A kernel that leaves any output tile unwritten (stale/zero) fails the gate.

- AC-2: **(promotion sanity check, NOT the formal gate)** The on-device rel-L2 lands in
  the expected low band — roughly `1e-7 .. few×1e-6` — consistent with the fp32 matmul
  emulation floor and the sibling (`bmm_softmax_v4` = 2.57e-6; standalone layout
  reconstruction = 2.43e-7). A result technically under `2e-5` but far above this band
  is treated as *suspicious / likely a latent bug*, not a clean pass. This criterion
  does **not** override AC-1's gate; AC-1 remains the only hard threshold.
  - Positive Tests:
    - On-device rel-L2 ≈ `≤ 5e-6` on the measured seeds.
    - Per-seed rel-L2 values are recorded (not just pass/fail) so the trend is visible.
  - Negative Tests:
    - A rel-L2 in `[1e-5, 2e-5)` is flagged for investigation before any promotion,
      even though it nominally passes AC-1.

- AC-3: The tiled layout index maps hold end-to-end, including the head map
  `qh = 2*kh + grp` and the context-loop subtile ordering `j = 4*a + b`, so that the
  score column offset `128*(4*a+b)` (key axis `n_k`) aligns with the value subtile
  `t_v = j` (value axis `n_v`) — both being absolute sequence position `n`.
  - Positive Tests:
    - The standalone numpy reconstruction through the index maps yields rel-L2 ≈ 2.43e-7
      vs the reference (already verified; re-run confirmed PASS).
    - The kernel's store indices match `v4 = [0, kh, grp, t_q, pos, d]` and its load
      indices match the `q/k/v` maps in the draft.
  - Negative Tests:
    - Iterating the context `j` in any flattening other than `j = 4*a + b` (e.g. a
      transposed `a`/`b` order) misaligns key columns with value subtiles and fails AC-1.

- AC-4: Output coverage is exact — the kernel writes precisely `8 (kh) × 2 (grp) ×
  32 (t_q) = 512` output tiles of `[128,128]`, covering all 16 query heads and all
  4096 query positions, each tile written exactly once (no gaps, no duplicates).
  - Positive Tests:
    - Every `(kh, grp, t_q)` in `8×2×32` stores once to `v4[0, kh, grp, t_q, :, :]`.
  - Negative Tests:
    - A dropped `grp` or `t_q` iteration leaves stale/zero output and fails AC-1
      (optionally surfaced faster by an output sentinel pre-fill in a debug run).

- AC-5: The numeric method matches the reference. fp32 throughout; the full score row
  is materialized (in SBUF) before the reduction, and softmax is computed over the
  **entire** 4096-wide key axis (not chunked/online); the operation order reproduces the
  reference: `score*scale → tensor_reduce(max, negate=True) → activation(exp, bias=neg_max)
  → tensor_reduce(add) → reciprocal → tensor_scalar(mul)`, with `scale = 1/sqrt(128) =
  0.08838835`.
  - Positive Tests:
    - The epilogue matches `bmm_softmax_v4` verbatim plus an explicit full-width
      `score * scale` multiply applied before the max reduction.
  - Negative Tests:
    - Any chunked/online softmax with partial (per-512-chunk) max/sum, OR bf16 anywhere,
      OR fusing the row-sum into the `activation(reduce_op=add, reduce_res=)` accumulator
      (measured +75% wall on the sibling), is rejected for this phase.

- AC-6: PSUM/SBUF usage is legal and the accumulations are correct. The full-width score
  and attention tensors are materialized in **SBUF, not PSUM**: each 512-column score
  matmul result is copied out to SBUF before the next chunk (no live `[128,4096]` PSUM
  tile). The `A_t` transpose result is copied to SBUF before it is used as the stationary
  operand of the context matmul. The `O_psum[128,128]` accumulator is zero-initialized
  per `(kh, grp, t_q)` and genuinely accumulated over all 32 `j` subtiles before it is
  copied/stored (loop-carried PSUM state, mirroring the baseline's `psum += nc_matmul`
  idiom on this same operator).
  - Positive Tests:
    - No more than the 8 available PSUM banks are live at any point; the context
      accumulation is loop-carried across all 32 `j` before the copy-out.
  - Negative Tests:
    - Relying on a live full-width (`[128,4096]`) PSUM tile, reading `O_psum` before all
      32 accumulations complete, or reusing a non-reset PSUM accumulator is rejected.

- AC-7: **(secondary, NON-gating diagnostic)** The profiler digest shows no SBUF spill
  and HBM traffic near the read-once / write-once floor (order-of-magnitude ~100 MB read
  + ~33 MB write for the drafted reuse design). Because the lower-bound path explicitly
  permits correctness-first reloads/retransposes, elevated HBM read is a **warning to
  investigate, never an automatic phase-1 failure** — the quoted byte figures are
  telemetry, not a correctness expectation. Correctness (AC-1) remains the only gate.
  - Positive Tests:
    - HBM traffic within ~1.2× of the read/write floor with no spill for `gqa_full_v1`.
  - Negative Tests:
    - HBM `≫` floor (indicating spill or unexpected re-fetch) is logged as a diagnostic
      concern to investigate; it does **not** by itself fail phase 1 if AC-1 passes.

## Path Boundaries

This is a **deterministic design**: the draft specifies a concrete kernel
(`gqa_full_v1`). The bounds below describe the acceptable correctness envelope, not a
menu of competing designs.

### Upper Bound (Maximum Acceptable Scope)
A single correct `gqa_full_v1` kernel implemented exactly as drafted: per-`kh` resident
`k_t[d=128, 4096]` built from 32 identity-transpose subtiles and `v_sb[p=128, 4096]`
loaded with no transpose, both reused across the 2 query groups and all 32 `t_q` tiles;
a per-`t_q` `q` transpose; the `8 × 512` score build into SBUF; the sibling softmax
epilogue plus the explicit `1/sqrt(D)` scale; and the 32-step `A`-transpose-and-accumulate
context matmul into a zero-initialized PSUM accumulator. It passes the 5-seed gate with
rel-L2 in the expected low band, writes all 512 output tiles once, and shows no spill.
This is "complete correctness with the drafted fusion" — nothing beyond it (no perf
levers) is in scope for this phase.

### Lower Bound (Minimum Acceptable Scope)
Any fp32 kernel that passes the 5-seed rel-L2 `2e-5` gate for `gqa_full`, even if it is
simpler or slower than the drafted design — for example, reloading/retransposing `k`/`v`
per group instead of caching per head, at the cost of extra HBM traffic. Correctness is
the floor; the per-head reuse and no-spill property are desirable but may be relaxed if
they turn out to complicate a first correct pass.

### Allowed Choices
- Can use: fp32 for all inputs, intermediates, reductions, and outputs; the
  `nisa.nc_matmul(is_transpose=True)` identity-transpose idiom (established on this
  operator's baseline and on `bmm_softmax_v4`); `affine_range` loops with a loop-carried
  `psum += nc_matmul` accumulation (established on this operator's baseline); per-head
  operand reuse of `k_t`/`v_sb`, OR per-group reload as the fallback lower-bound path; an
  optional debug-only output sentinel pre-fill and per-seed rel-L2 logging.
- Cannot use (deferred to phase 2/3 as performance levers, not correctness needs): bf16
  or bf16x2 split; flash-style / online (chunked) softmax with partial max/sum; folding
  `1/sqrt(D)` into the `activation(scale=)` parameter; the two-phase transpose-all
  schedule; and `M_SUB` / M-block sizing sweeps. These do not affect phase-1 correctness
  and their inclusion is out of scope here.

> **Note on Deterministic Design**: The draft fixes the algorithm and layout, so the
> upper bound is a single concrete kernel and the lower bound relaxes only *how much
> operand reuse* is required, never *what is computed*. "Allowed Choices" is
> correspondingly narrow by the draft's specification.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach

Per-head fusion (`gqa_full_v1`), pseudocode:

```
out = ndarray((1,8,2,32,128,128), fp32, shared_hbm)
identity_local[128,128] = load(shared_constant(I128))          # once, for all transposes

for kh in affine_range(8):
    # per-head shared operands (reused across grp and all 32 t_q)
    k_t[d=128, 4096]:  for a in 8, b in 4:                      # 32 k transposes
        k_sub[c=128, d=128] = load(v2[0,a,b,:,kh,:])
        k_t[:, 128*(4*a+b):+128] = copy( nc_matmul(k_sub, identity, is_transpose=True) )
    v_sb[p=128, 4096]: for t_v in 32:                           # NO transpose, direct load
        v_sb[:, 128*t_v:+128] = load(v3[0,t_v,:, kh*128:kh*128+128])

    for grp in affine_range(2):                                 # qh = 2*kh + grp
        for t_q in affine_range(32):
            # --- scores ---
            q_sb[p=128, d=128]  = load(v1[t_q,:,qh,:])
            q_t[d=128, m_q=128] = copy( nc_matmul(q_sb, identity, is_transpose=True) )
            score[128,4096] in SBUF:
                for c in 8: acc = nc_matmul(q_t, k_t[:,512*c:+512]); score[:,512*c:+512]=copy(acc)
            # --- softmax over the 4096 free axis, in SBUF, fp32 ---
            score   = score * scale                              # scale = 1/sqrt(128)
            neg_max = tensor_reduce(max, score, axis=free, negate=True)
            exp_t   = activation(exp, score, bias=neg_max, scale=1.0)
            row_sum = tensor_reduce(add, exp_t, axis=free)       # explicit Vector add (NOT fused)
            recip   = reciprocal(row_sum)
            A       = tensor_scalar(exp_t, mul=recip)            # [128,4096]
            # --- context ---
            O_psum[128,128] (zero-init, accumulate):
                for j in 32:
                    A_t = copy( nc_matmul(A[:,128*j:+128], identity, is_transpose=True) )  # [n_k,m_q]
                    O_psum += nc_matmul(A_t, v_sb[:,128*j:+128])                            # [m_q, d]
            store(out[0,kh,grp,t_q,:,:], copy(O_psum))
return out
```

Key correctness anchors, from Codex first-pass + convergence review:
- Set `qh = 2*kh + grp` explicitly and use it for both the `v1` load and the `out` store.
- The context loop must use `j = 4*a + b` flattening so key columns and value subtiles
  index the same absolute position `n`.
- `O_psum` must be zero-initialized per `(kh, grp, t_q)` and accumulated over all 32 `j`
  before copy/store; `A_t` must be copied to SBUF before use as the stationary operand.
- The full-width score/attention tensors live in **SBUF**; PSUM holds only the
  per-chunk matmul/transpose results and the `[128,128]` `O` accumulator (≤ 8 banks live).
- fp32 everywhere; full-width (not chunked) softmax in the reference op order.

### SBUF budget (per partition, one `kh` live)
`k_t` 16 KB + `v_sb` 16 KB + `score` 16 KB + `exp_t/A` 16 KB + `q_t` 0.5 KB + `A_t`
0.5 KB + `O` 0.5 KB + identity 0.5 KB + `[128,1]` scalars ≈ **66 KB** of the ~208 KB
usable ⇒ no spill; HBM stays near the read-once/write-once floor.

### Relevant References
- `workspaces/gqa_full/docs/draft-phase1.md` — the full design draft (preserved below).
- `../AccelOpt/NKIBench/reference/gqa_full_B1_N4096_QH16_KH8_D128_numpy_2.py` — the numpy
  reference and the reshape-only `transform_to_nki_inputs` / `transform_nki_outputs`.
- `../AccelOpt/NKIBench/kernels/gqa_full_B1_N4096_QH16_KH8_D128_0.py` — the 15.579 ms
  baseline; already uses the identity `nc_matmul(is_transpose=True)` and the
  `psum += nc_matmul` accumulation idioms on this exact operator.
- `workspaces/bmm_softmax/runs/bmm_softmax_v4.py` — the promoted sibling whose full-width
  fp32 softmax epilogue (max-negate fold, explicit row-sum) is reused verbatim.
- `/tmp/gqa_layout_check.py` — the standalone numpy reconstruction that validates the
  q/k/v/out index maps at rel-L2 2.43e-7 (re-run confirmed PASS).
- `verify.py` (repo root) — the scoring harness; gates on `l2_norm_passed`.

## Dependencies and Sequence

### Milestones
1. Kernel implementation: write `runs/gqa_full_v1.py` per the drafted design.
   - Phase A: set up shared operands per `kh` (`k_t` via 32 transposes, `v_sb` direct
     load) and the once-loaded identity tile.
   - Phase B: per `(grp, t_q)` — transpose `q`, build the `8×512` score row in SBUF,
     run the full-width fp32 softmax epilogue with the explicit scale, then the 32-step
     `A`-transpose-and-accumulate context matmul, and store `O`.
2. Correctness verification: score the candidate and confirm the gate.
   - Step 1: cross-check the kernel's load/store index expressions against the
     already-passing numpy reconstruction map (`AC-3`).
   - Step 2: `--fast` scoring to catch gross errors early, then the full 5-seed run for
     the AC-1 gate; record per-seed rel-L2 (`AC-2`).
   - Step 3: if the L2 gate fails, diagnose via the accuracy-debugging methodology.
3. Evidence recording: log the perf change in `benchmark.csv`, the candidate in
   `candidates.jsonl` (as the DAG root), and any profiling digest under `profile/`.

Dependencies: Milestone 2 depends on Milestone 1; Milestone 3 depends on a passing
Milestone 2. Within Milestone 1, Phase B depends on the shared operands from Phase A.

## Task Breakdown

Each task includes exactly one routing tag: `coding` (implemented by Claude) or
`analyze` (executed via Codex, `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Implement `runs/gqa_full_v1.py`: per-`kh` `k_t`/`v_sb` shared operands, per-`(grp,t_q)` q-transpose + `8×512` SBUF score build, full-width fp32 softmax epilogue with explicit `1/sqrt(D)` scale, 32-step A-transpose + zero-init PSUM-accumulate context, store all 512 tiles. Set `qh = 2*kh + grp`; use `j = 4*a+b` ordering. | AC-3, AC-4, AC-5, AC-6 | coding | - |
| task2 | Cross-check the kernel's load/store index expressions against the passing numpy reconstruction map before benchmarking. | AC-3 | coding | task1 |
| task3 | Score the candidate: `--fast` first, then the full 5-seed `verify.py` run; confirm `l2_norm_passed` 5/5 and record per-seed rel-L2. | AC-1, AC-2 | coding | task2 |
| task4 | If the L2 gate fails, diagnose the root cause (layout/index, accumulation, or op-order) using the accuracy-debugging methodology and propose the fix. | AC-1 | analyze | task3 |
| task5 | Read the profiler digest for SBUF spill / HBM-floor telemetry (non-gating) and record `benchmark.csv` + `candidates.jsonl` (DAG root) + `profile/` evidence. | AC-7 | coding | task3 |

## Claude-Codex Deliberation

### Agreements
- The plan is correctness-first and correctly phase-scoped: the only hard gate is the
  NKIBench `rel-L2 < 2e-5` across the five fp32 seeds; performance work is deferred.
- The operator decomposes into a per-head `bmm_softmax` (scores + row-softmax over N)
  plus a context matmul; reusing the promoted `bmm_softmax_v4` softmax epilogue is sound.
- The `nc_matmul(is_transpose=True)` identity idiom and the `affine_range` +
  `psum += nc_matmul` context accumulation are established-legal on this exact operator
  (the baseline uses both), so they are safe phase-1 building blocks.
- The standalone numpy reconstruction validates the q/k/v/out index maps (rel-L2 2.43e-7,
  re-run confirmed) but is necessary-not-sufficient: it does not prove NKI store/load
  legality, `qh` binding, or exactly-once output coverage — hence AC-3/AC-4/AC-6.
- Highest correctness risks to guard explicitly: the `qh = 2*kh + grp` binding, the
  `j = 4*a+b` key/value alignment, the loop-carried zero-initialized `O_psum`
  accumulation, full-width (not chunked) softmax in the reference op order, and fp32
  everywhere.

### Resolved Disagreements
- AC-2 as gate vs. sanity check: Codex flagged that a strict `1e-7..few-e-6` expectation
  must not contradict AC-1's `2e-5` gate. Resolution: AC-2 is explicitly reworded as a
  *promotion sanity check* (a value in `[1e-5, 2e-5)` triggers investigation) while AC-1
  remains the sole hard threshold. Adopted.
- Score-tensor location: Codex required making explicit that the full `[128,4096]`
  score/attention lives in **SBUF, not PSUM**. Resolution: AC-5 and AC-6 now state
  full-width SBUF materialization and forbid a live full-width PSUM tile. Adopted.
- AC-7 HBM figures: Codex noted the `~100MB read + ~33MB write` floor must not become an
  implied correctness expectation, since the lower-bound reload path may exceed it.
  Resolution: AC-7 is marked explicitly NON-gating; elevated HBM is a warning to
  investigate, not a phase-1 failure. Adopted.
- Codex `QUESTIONS_FOR_USER` from the first pass were all resolved during refinement:
  (a) a known-good transpose idiom IS available (baseline + `bmm_softmax_v4`); (b) the
  literal-sibling vs. reload tradeoff is captured by the upper/lower bounds; (c) debug
  checksums are optional (task2 cross-checks indices against the passing reconstruction;
  `verify.py` is the gate). No item required a user decision.

### Convergence Status
- Final Status: `converged` — the second Codex review returned no blocking `UNRESOLVED`
  items and no high-impact `DISAGREE`; all `REQUIRED_CHANGES` were wording tightenings
  that have been folded into AC-2, AC-5, AC-6, and AC-7. Rounds executed: 1.

## Pending User Decisions

None. All Codex first-pass questions and convergence items were resolved during plan
refinement (see Resolved Disagreements). No item requires an explicit user decision, and
no quantitative metric is ambiguous: the `rel_tol = 2e-5` 5-seed gate is the hard
NKIBench contract, and beating the 15.579 ms baseline is a deferred phase-2/3 direction
(not a phase-1 gate) per the draft.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Phase", "Step", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `k_t`, `v_sb`,
  `score`, `neg_max`, `exp_t`, `row_sum`, `o_psum`, `qh`, `kh`, `grp`, `t_q`).
- The kernel is a single `@nki.jit def kernel(v1, v2, v3)` entry point whose signature
  consumes the tiled inputs and returns the tiled output shape.

--- Original Design Draft Start ---

# gqa_full — Phase 1 Draft (first correct NKI kernel)

## Goal

Produce the first **correct** NKI kernel for `gqa_full` (NKIBench case 0):
grouped-query full (non-causal) softmax attention. Prioritize a clean,
obviously-correct kernel that passes the relative-L2 gate across all five seeds
`[0,21,42,63,84]` (`rel_tol=2e-5`, fp32); speed is phase-2/3 work. The fused
per-head design below should already clear the 15.579 ms baseline by a wide
margin, because the baseline spills the whole score matrix to HBM (see "Why the
baseline is slow").

## Operator

- Shapes/dtype (natural layout): `q (1,4096,16,128)`, `k,v (1,4096,8,128)` fp32.
  `B=1, N=4096, QH=16, KH=8, n_rep=QH/KH=2, D=128`.
- Reference (`gqa_full_..._numpy_2.py`), per query head `qh` (its kv head is
  `kh = qh // 2`, since `xk=np.repeat(k, n_rep=2, axis=head)`):
  ```python
  S    = q_h @ k_h.T / sqrt(D)                    # [N_q, N_k] scores
  A    = softmax_over_Nk(S)                        # row-softmax over the key axis
  O    = A @ v_h                                   # [N_q, D] context
  ```
  So it is exactly a **per-head `bmm_softmax` (scores + row-softmax) followed by
  a second matmul (context)**. `bmm_softmax` is a solved, promoted, 5-seed-correct
  sibling on this harness — reuse its core verbatim and bolt on the context matmul.

## Tiled layout — DERIVED AND EMPIRICALLY VERIFIED (rel-L2 2.4e-7)

`transform_to_nki_inputs` is reshape-only; `transform_nki_outputs` reshapes our
result to the ref shape. I reconstructed the reference output from the tiled
tensors through the index maps below and got **rel-L2 = 2.4e-7** vs the reference
(`/tmp/gqa_layout_check.py`, seed 0) — the maps are correct, not guessed:

- **q** `v1 = (32,128,16,128) = [t_q, p, qh, d]`:  `q[n_q, qh, d]` with
  `n_q = 128*t_q + p`. A fixed `(t_q, qh)` slice `v1[t_q,:,qh,:]` is
  `[p=n_q_sub(par)=128, d(free)=128]`.
- **k** `v2 = (1,8,4,128,8,128) = [0, a, b, c, kh, d]`:  `k[n_k, kh, d]` with
  **`n_k = 512*a + 128*b + c = 128*(4*a+b) + c`**. A fixed `(a,b,kh)` slice
  `v2[0,a,b,:,kh,:]` is `[c=n_k_sub(par)=128, d(free)=128]`; the 32 subtiles
  `(a in 8, b in 4)` tile the full `N_k=4096` at column offset `128*(4*a+b)`.
- **v** `v3 = (1,32,128,1024) = [0, t_v, p, kh*128+d]`:  `v[n_v, kh, d]` with
  `n_v = 128*t_v + p`. A fixed `t_v` slice `v3[0,t_v,:, kh*128 : kh*128+128]` is
  `[p=n_v_sub(par)=128, d(free)=128]` for head `kh`.
- **out** `v4 = (1,8,2,32,128,128) = [0, kh, grp, t_q, pos, d]`, maps to
  `ref[0, qh=2*kh+grp, n=128*t_q+pos, d]`. So `O_tile[pos(par)=128, d(free)=128]`
  stores directly to `v4[0, kh, grp, t_q, :, :]`.

Key alignment fact (verified): the **`n_k` (key) axis of the scores/attn matches
the `n_v` (value) axis of `v`** — both are absolute sequence position `n`. So the
context matmul iterates `n` subtiles `0..31` uniformly; k-columns built at offset
`128*(4*a+b)` and v-subtiles at `t_v` both index position `n`, and they line up.

## The two matmuls (nc_matmul = stationary.T @ moving, contraction on partition)

Both operands live in SBUF; the contraction dim must be on the **partition** axis
of both. `D=128` and `N_k` subtiles are 128, so each is a clean 128-contraction.

**Matmul 1 — scores `S[m_q, n_k] = sum_d q_h[m_q,d]·k_h[n_k,d]`** (contract `d`):
- result `[m_q(par)=128, n_k(free)]` ⇒ softmax reduces over the **free** axis (good,
  same as `bmm_softmax_v4`). ⇒ `stationary=[d(par),m_q(free)]`,
  `moving=[d(par),n_k(free)]` — **both q and k need `d` on the partition axis**.
- q native is `[p=m_q(par), d(free)]` → transpose once per `(kh,grp,t_q)` via the
  identity `nc_matmul(is_transpose=True, is_moving_onezero=True)` idiom → `q_t[d,m_q]`.
- k native is `[c=n_k_sub(par), d(free)]` → transpose the 32 subtiles once per `kh`
  into a resident `k_t[d(par)=128, n_k(free)=4096]` (reused across `grp` and all
  32 `t_q` ⇒ 64 reuses; column offset `128*(4*a+b)`).
- 8 chunks of `N_CHUNK=512` (one fp32 PSUM bank) build `score[128,4096]`.

**Matmul 2 — context `O[m_q,d] = sum_{n_k} A[m_q,n_k]·v_h[n_k,d]`** (contract `n_k`):
- result `[m_q(par)=128, d(free)=128]` ⇒ `stationary=[n_k(par),m_q(free)]`,
  `moving=[n_k(par),d(free)]`.
- `A` from softmax is `[m_q(par)=128, n_k(free)=4096]` → for each of 32 `n_k`
  subtiles, transpose `A[:,128*j:+128]` → `A_t[n_k_sub=128, m_q=128]` (identity
  idiom), and **accumulate** into one `[128,128]` PSUM bank over `j=0..31`
  (`v24 += nc_matmul` idiom in the baseline).
- **v needs NO transpose**: v native `[p=n_v_sub(par)=128, d(free)=128]` is already
  the required moving layout. Load `v_h` once per `kh` into `v_sb[p(par)=128,
  t_v*128+d]` (so subtile `j` = `v_sb[:, 128*j:+128]`), reused across `grp`+`t_q`.

## Softmax epilogue (verbatim `bmm_softmax_v4`, + the `1/sqrt(D)` scale)

Over the `N_k=4096` free axis, fp32, max-shifted for overflow safety:
```
score  = score * scale                 # scale = 1/sqrt(128) = 0.08838835; reproduces the
                                        #   reference's exact op order (attn = q@kT * scale, THEN max)
neg_max = tensor_reduce(max, score, axis=free, negate=True)   # -row_max, negate folds the *-1 step
exp_t   = activation(exp, score, bias=neg_max, scale=1.0)     # exp(score - row_max)
row_sum = tensor_reduce(add, exp_t, axis=free)                # explicit Vector add (do NOT fuse
                                        #   into activation reduce_res — measured +75% on the sibling)
recip   = reciprocal(row_sum)
A       = tensor_scalar(exp_t, mul=recip)                     # per-row [128,1] scale over free axis
```
The explicit `score*scale` full-width multiply reproduces the reference's operation
order bit-for-bit (I verified this path at rel-L2 2.4e-7). Folding the scale into
the `activation(scale=)` param (and scaling `neg_max`) removes that full-width op
and is numerically equivalent — noted as a **phase-2 lever**, not used in phase-1.

## Kernel plan (`gqa_full_v1`)

```
out = ndarray((1,8,2,32,128,128), fp32, shared_hbm)
identity_local[128,128] = load(shared_constant(I128))         # once, for all transposes

for kh in affine_range(8):
    # --- per-head shared operands (reused across grp and all 32 t_q) ---
    k_t[d=128, 4096]  : for a in 8, b in 4:                    # 32 k transposes
        k_sub[c=128,d=128] = load(v2[0,a,b,:,kh,:])
        k_t[:, 128*(4*a+b):+128] = copy( nc_matmul(k_sub, identity, is_transpose=True) )  # [d, n_k_sub]
    v_sb[p=128, 4096] : for t_v in 32:                         # no transpose, direct load
        v_sb[:, 128*t_v:+128] = load(v3[0,t_v,:, kh*128:kh*128+128])   # [p=n_v_sub, d]

    for grp in affine_range(2):                                # qh = 2*kh + grp
        for t_q in affine_range(32):
            # scores
            q_sb[p=128,d=128] = load(v1[t_q,:,qh,:])
            q_t[d=128,m_q=128] = copy( nc_matmul(q_sb, identity, is_transpose=True) )
            score[128,4096]:
                for c in 8: acc[128,512] = nc_matmul(q_t, k_t[:,512*c:+512]); score[:,512*c:+512]=copy(acc)
            # softmax over the 4096 free axis (block above)  -> A[128,4096]
            # context
            O_psum[128,128] (accumulate):
                for j in 32:
                    A_t[n_k=128,m_q=128] = copy( nc_matmul(A[:,128*j:+128], identity, is_transpose=True) )
                    O_psum += nc_matmul(A_t, v_sb[:,128*j:+128])    # [m_q, d]
            store(out[0,kh,grp,t_q,:,:], copy(O_psum))              # [pos, d]
return out
```

## Why the baseline is slow (and what we fix)

Baseline latency **15.579 ms**. It materializes the whole per-head/per-tile score
matrix across the grid (`v13/v14 ≈ [32,2,4,2,2,4,128,512]` ≈ the full `QH*N*N`
scores) and does a chunked online max/sum over that resident set — far larger than
SBUF, so scores **spill to HBM and round-trip** (write scores, read back for
exp/normalize) on top of the output write.

**Fix = per-head, per-m_q fusion** (the same win `bmm_softmax` had over its
baseline). One m_q tile's full score row is only `[128,4096] fp32 = 16 KB/partition`,
trivially resident. Build the row, softmax it in place, immediately consume it in
the context matmul, discard. Scores never touch HBM; traffic drops toward the
read-once/write-once floor (read q+k+v ≈ 100 MB, write out ≈ 33 MB). Note: this
already achieves "scores never fully materialize" at the whole-matrix level —
true flash-style online softmax over `n_k` chunks is a further phase-2 lever, and
is **not needed to fit** here (the 16 KB row fits with room to spare).

## SBUF budget (per partition, one `kh` live)

`k_t` 16 KB + `v_sb` 16 KB + `score` 16 KB + `exp_t/A` 16 KB + `q_t` 0.5 KB +
`A_t` 0.5 KB + `O` 0.5 KB + identity 0.5 KB + `[128,1]` scalars ≈ **66 KB** of the
~208 KB usable ⇒ no spill; HBM stays at the read-/write-once floor.

## Correctness argument

- Layout maps verified end-to-end at **rel-L2 2.4e-7** against the reference.
- Both matmuls are single-128-contraction `nc_matmul` (fp32 emulation floor
  ~1.8e-7 on the sibling cores); softmax is max-shifted fp32 in the reference's
  exact op order. Well under the `2e-5` gate.
- `n_k`↔`n_v` position alignment verified; k/v loaded once per `kh` and correctly
  shared across the 2 query groups (`n_rep=2` = the `qh=2*kh+grp` map).

## Phase-2 / phase-3 outlook (not this phase)

- Two-phase transpose-all schedule (pack all q/A transposes up front, then stream
  matmuls) — the promoted `bmm_softmax_v4`/`bmm_v2` lever; hides softmax behind a
  longer matmul stream. M-block sizing (`M_SUB`) sweep.
- Fold `1/sqrt(D)` into `activation(scale=)` to drop the full-width scale op.
- Flash-style online softmax over `n_k` chunks (deferred; not needed to fit).
- bf16x2 split on the two big matmuls if PE-bound (the matmul-family lever), gated
  on the fp32 rel-L2 floor headroom under `2e-5`.

## Validate / score

```bash
python3 \
    ../../verify.py --op gqa_full --candidate runs/gqa_full_v1.py --fast
```

--- Original Design Draft End ---
