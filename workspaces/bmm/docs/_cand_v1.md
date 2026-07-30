# bmm Phase 1 — First Correct fp32 NKI Kernel

## Goal Description
Produce the first CORRECT NKI/Trainium kernel for the `bmm` operator (NKIBench case 2):
batched matmul `out[b] = lhs[b] @ rhs[b]` for `b in 0..15`, fp32, with
`lhs (16,4096,64)=(B,M,K)`, `rhs (16,64,4096)=(B,K,N)`, `out (16,4096,4096)=(B,M,N)`
(B=16, M=4096, K=64, N=4096). The kernel must pass NKIBench's relative-L2 correctness
gate (rel_tol=2e-5) across all five seeds [0,21,42,63,84], validated by verify.py.
Phase 1 prioritizes a clean, fully-understood, correct kernel over speed; performance
is captured as evidence only, with NO phase-1 performance floor.

## Acceptance Criteria

- AC-1: The kernel passes the NKIBench relative-L2 correctness gate for bmm across all
  five seeds via a full (non-fast) verify.py run.
  - Positive Tests: `verify.py --op bmm --candidate runs/bmm_v1.py` (no --fast) reports
    l2_norm_passed=True for every seed; a --fast single-seed gate passes first.
  - Negative Tests: a kernel that downcasts to bf16, drops/mis-tiles K terms, or mis-maps
    the output produces l2_norm_passed=False on at least one seed and is rejected.
- AC-2: The contraction is a faithful single-pass fp32 matmul: K=64 contracted in ONE
  nc_matmul per output tile, no K-accumulation loop, no bf16/tf32 anywhere.
  - Positive Tests: each (b,mt,c) output tile is produced by exactly one main nc_matmul
    with K=64 on the partition axis of both operands; source contains no `+=` over K-tiles
    and no dtype downcast.
  - Negative Tests: a K-tile loop with `+=` accumulation, or any bf16/tf32 cast, is present
    and rejected on review.
- AC-3: The layout/transpose/matmul mapping is correct for this shape.
  - Positive Tests: rhs tile loads directly as [k=64(par), n(free)]; lhs tile
    [m=128(par), k=64(free)] is transposed via the identity idiom to [k=64(par), m=128(free)]
    (is_transpose=True, is_moving_onezero=True) and copied to SBUF; main
    nc_matmul(stationary=lhs_t[64,128], moving=rhs[64,512]) yields [m=128(par), n=512(free)].
  - Negative Tests: feeding lhs with K on the free axis as the stationary operand, or a
    matmul whose partition axis is not the K contraction, yields wrong results / fails to trace.
- AC-4: The kernel output reshapes to the reference shape (16,4096,4096) row-major with
  full, non-overlapping coverage (32 M-tiles x 8 N-chunks x 16 batches).
  - Positive Tests: the returned tensor (direct 3D (16,4096,4096) OR the 4D fallback
    (16,32,128,4096)) reshapes to (16,4096,4096) and matches np.matmul(lhs,rhs) within the gate.
  - Negative Tests: an output whose tiling leaves a gap/overlap, or whose row-major order
    differs from the reference, fails the L2 gate.
- AC-5: Phase-1 bookkeeping is recorded: benchmark.csv perf row, candidates.jsonl DAG root
  (parent = bmm_B16_M4096_K64_N4096_0.py), and a profile/ metrics digest.
  - Positive Tests: after a full run, benchmark.csv has the bmm_v1 row, candidates.jsonl has
    the root entry, and profile/ holds the MFU/PE/Vec/DMA/HBMrd/HBMwr digest.
  - Negative Tests: a promoted candidate with no benchmark.csv/candidates.jsonl/profile
    evidence is incomplete and rejected.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A clean single-file kernel `runs/bmm_v1.py` with a `@nki.jit def kernel(v1, v2)` entry point:
loads a 128x128 identity once; per batch loads rhs[b] resident as [64(par),4096(free)] (16 KB/part);
per M-tile loads and identity-transposes the lhs tile once; streams 8 N-chunks of 512 producing
[128,512] PSUM tiles copied to SBUF and stored into a direct 3D output (16,4096,4096); captures the
full profiler metrics digest into profile/ as write-bound evidence; records benchmark.csv and
candidates.jsonl. No perf optimization (no store-burst fattening, no ping-pong) in phase 1.

### Lower Bound (Minimum Acceptable Scope)
Any single fp32 kernel that passes the 5-seed L2 gate for bmm and is recorded as the DAG root —
including the documented fallbacks (4D output (16,32,128,4096); rhs loaded per-chunk or in 1024-wide
blocks) if the direct-3D / full-resident form fails to trace.

### Allowed Choices
- Can use: fp32 throughout; the proven identity-transpose idiom; N_CHUNK=512 for the main matmul;
  either direct 3D output (16,4096,4096) or 4D fallback (16,32,128,4096); rhs resident [64,4096] OR
  per-chunk / 1024-wide rhs loads; module-level + in-function NKI imports.
- Cannot use: bf16/tf32 anywhere; a K-split accumulation loop; a single matmul with PSUM free width
  >512 fp32; hand-tuning of the baseline; edits to any ../../AccelOpt/NKIBench/{kernels,reference,seeds,summary.json} file.

> Deterministic-design note: the math mapping (transpose orientation, K=64 single pass, N=512 chunk,
> output tiling) is fixed per the draft; the only genuinely open choices are output shape (3D vs 4D
> fallback) and rhs residency granularity, both gated by whether the tracer accepts the cleaner form.

## Feasibility Hints and Suggestions

### Conceptual Approach
```
identity_local = load 128x128 identity into SBUF        # once, reused for all transposes
for b in affine_range(16):
    rhs_sb = load v2[b, 0:64, 0:4096] -> [64(par), 4096(free)]     # once per batch (16 KB/part)
    for mt in affine_range(32):                                    # 4096/128 M-tiles
        lhs_sb = load v1[b, mt*128:+128, 0:64] -> [128(par), 64(free)]
        lhs_t_psum = nc_matmul(lhs_sb, identity_local, is_transpose=True, is_moving_onezero=True)
                     -> PSUM [64(par),128(free)]
        lhs_t = copy(lhs_t_psum) -> SBUF [64(par),128(free)]
        for c in affine_range(8):                                  # 4096/512 N-chunks
            acc = nc_matmul(lhs_t, rhs_sb[:, c*512:+512]) -> PSUM [128(par),512(free)]
            out_sb = copy(acc) -> SBUF [128,512]
            store out[b, mt*128:+128, c*512:+512] = out_sb
return out   # (16,4096,4096); reshapes row-major to reference
```
All slices become affine index expressions (nl.arange(64) K-partition, nl.arange(128) M, nl.arange(512)
N-chunk); allocations carry par_dim (transpose tiles par_dim(64); main/output tiles par_dim(128)).

### Relevant References
- `../../AccelOpt/NKIBench/kernels/bmm_B16_M4096_K64_N4096_0.py` — proven baseline; exact
  identity-transpose call (line 34), 4D output store (line 42), rhs-once-per-batch load (line 30).
- `workspaces/matmul/runs/matmul_v1.py` — sibling dense GEMM: identity-transpose idiom, N_CHUNK=512,
  and a strided 3D output store `out[mt, arange(128)[:,None], n0+arange(512)[None,:]]` (structurally
  the bmm 3D store with a scalar leading index) — proven to compile and pass the L2 gate.
- `../../AccelOpt/NKIBench/reference/bmm_B16_M4096_K64_N4096_numpy_1.py` — reference: reshape-only
  transform_to_nki_inputs, row-major transform_nki_outputs.
- `verify.py` — correctness gate (l2_norm_passed across seeds); `--fast` = single seed.

## Dependencies and Sequence

### Milestones
1. Kernel implementation: write `runs/bmm_v1.py` (identity load -> batch loop -> M-tile transpose ->
   N-chunk matmul -> store). Depends on the layout/transpose mapping being fixed.
2. Fast correctness gate: verify.py --fast (1 seed). If it fails to trace, apply the documented
   fallback (4D output and/or 1024-wide rhs). Depends on Milestone 1.
3. Full correctness measurement: verify.py without --fast (5 seeds); capture the metrics digest.
   Depends on Milestone 2 passing.
4. Bookkeeping: append benchmark.csv row, candidates.jsonl DAG root, write profile/ digest.
   Depends on Milestone 3.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Implement runs/bmm_v1.py per the mapping (identity transpose, K=64 single pass, N=512 chunks, direct 3D output) | AC-2, AC-3 | coding | - |
| task2 | Run fast (1-seed) verify.py gate; if trace fails, apply 4D-output / 1024-rhs fallback | AC-1, AC-4 | coding | task1 |
| task3 | Run full 5-seed verify.py; confirm l2_norm_passed all seeds | AC-1 | coding | task2 |
| task4 | Capture profiler metrics digest into profile/; append benchmark.csv + candidates.jsonl root | AC-5 | coding | task3 |
| task5 | Confirm theoretical write-bound framing against measured HBMwr/DMA%/PE% (evidence for phase 2, not a phase-1 gate) | AC-5 | analyze | task3 |

## Claude-Codex Deliberation

### Agreements
- Math mapping (transpose orientation, K=64 single matmul pass, N=512 chunk) is correct.
- Phase 1 is correctness-first: no performance floor; latency/metrics are evidence only.
- fp32 only; no bf16/tf32; no K-accumulation loop; PSUM main free width capped at 512.
- Pseudocode slices must be affine index forms with par_dim allocations.

### Resolved Disagreements
- Output shape (direct 3D vs baseline 4D): Claude keeps direct 3D (16,4096,4096) as primary because
  matmul_v1 proves a structurally identical strided 3D store; Codex prefers baseline 4D for lowest
  risk. Resolution: 3D primary, 4D (16,32,128,4096) as an explicit gated fallback if the tracer
  rejects the partition-offset 3D store. Both reshape row-major identically.
- rhs residency ([64,4096] resident vs 1024-wide blocks): resident kept as primary (16 KB/part, safe);
  1024-wide / per-chunk load documented as fallback. Draft's "avoids 32x reload" framing softened —
  the baseline also loads rhs once per batch, so the win is simplicity, not reload elimination.
- "bit-faithful to NumPy" softened to "relative-L2 safe": correctness is argued against the 2e-5 gate,
  not bit-equivalence.

### Convergence Status
- Final Status: converged

## Pending User Decisions

- DEC-1: Verifier seed behavior. Codex flags that the local adapter may seed inputs with a fixed seed
  even for multi-seed requests.
  - Claude Position: Trust verify.py's l2_norm_passed gate as the source of truth per CLAUDE.md; the
    5-seed pass history of sibling ops indicates seeds are varied. Not in phase-1 scope.
  - Codex Position: Consider whether distinct 5-seed evidence needs an independent seed-varied path.
  - Tradeoff Summary: Investigating the adapter adds scope for little correctness gain given single-pass
    fp32 has no seed-dependent accumulation order; the L2 gate already guards correctness.
  - Decision Status: PENDING

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Step", "Phase", or similar workflow markers. Use descriptive, domain-appropriate
  naming (batch/m-tile/n-chunk, lhs_t, rhs_sb, etc.).
