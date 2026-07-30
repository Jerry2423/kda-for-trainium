# add_rmsnorm_matmul (M4096 N2048 K1024, fp32 I/O) — Shape-Specialization Closure (Phase 3, AC-6)

`out = rmsnorm(x + z) @ (g ⊙ w)`: a residual add, RMSNorm over the K axis, a per-K
contraction-axis scale `g`, and a GEMM. Fixed shape M=4096, N=2048, K=1024, fp32 I/O.
Scored single-core on the remote profiler (`--disable-dge --logical-nc-config=1`), correctness
by NKIBench relative-L2 < 2e-5 on seeds `[0,21,42,63,84]` (`verify.py` gates on
`l2_norm_passed` for all seeds).

Phase 3 = regime/shape specialization: specialize only where the tensor's structure admits a
lever *and* a measured win justifies the complexity. This is the near-exact sibling of
`rmsnorm_matmul` (same M/N/K, same fp32 contract) plus a residual add `a = x + z`, a per-K `g`
scale, and an `eps` — so the shape levers close **identically** to that sibling
(`workspaces/rmsnorm_matmul/docs/shape-specialization-closure-phase3.md`), and this doc is that
closure specialized to this op. The finding: this fixed shape offers **no shape-specialization
surface** — every classic phase-3 lever is either vacuous (no ragged structure to exploit) or
already pinned at its hardware constraint. The absence of a lever is recorded as a first-class
result, not a gap.

## Per-lever closure (all five levers)

| Lever | Applies? | Shape / constraint reason |
|---|---|---|
| **Edge / partial tiles** | **No — vacuous** | Every dim divides its tile size evenly: M=4096=32·128, K=1024=8·128, N=2048=4·512. No ragged tile, no remainder loop, no masked/partial tile anywhere. Edge-tile specialization requires an edge; there is none. |
| **Tile-size / partition-free regime** | **No — layout forced** | `nc_matmul(stationary, moving) = stationary.T @ moving` requires the contraction dim (k_in) on the PARTITION axis of both SBUF operands and produces an output tile `[m_in(par), n(free)]`. So m_in is forced onto the stationary/partition axis and n onto the moving/free axis. Swapping m↔n would require transposing the entire N=2048-wide result back — a cost far larger than any tiling gain. No free partition/free split to specialize. |
| **N-chunk (moving-free) width** | **No — already maximal** | N_CHUNK=512 = `psum_fmax` = exactly one fp32 PSUM bank in the free dim — the documented per-matmul moving-free maximum (knowledgebase precedent `6288aaad`: "tile budget is psum_fmax, not pmax"). Larger exceeds a PSUM bank; smaller wastes the systolic streaming width. Already at the constraint. |
| **M-blocking** (the `matmul` task's phase-2 win) | **No — vacuous** | That win removed *redundant w HBM reloads* by reusing a loaded w-tile across output-row tiles. Here `w'` (g folded in, two bf16 limbs) is only 64 KB/partition (budget 192 KB), loaded **fully resident once** before the M-loop, then reused across all 32 M-tiles; each `x` and each `z` tile is loaded exactly once. HBMrd=42MB is already minimal (one read of x + one read of z + one read of w). There is no redundant HBM traffic to block for — the reuse M-blocking would create already exists. (The only difference from the sibling here is the extra one-time `z` read: 42MB vs the sibling's 25MB. That is intrinsic to the residual add, not a blocking lever.) |
| **LNC2 / multi-core sharding** | **No — out of contract** | Scored single-core (`--logical-nc-config=1`). LNC2 sharding is not a lever on this harness; using it would change the scoring contract, not optimize within it. |

## Why nothing else is left inside the fixed 3-product bf16 schedule

The promoted `add_rmsnorm_matmul_v3_bf16_split` (4.632x, 0.4013 ms) is PE-bound at PE=89%,
MFU=41%. PE-active (~355 µs) is the fixed floor: `2·M·N·K` at the bf16 systolic rate, run 3×
for the 3-product compensated split. Cutting PE-active further needs either fewer products
(fails the gate — plain bf16 ~117× over) or a lower-precision matmul (fails the gate). Closed
and pinned (D5 forbidden this phase; the 4-product D3 was already a MEASURED-REJECT: +28%
latency for a 1.6% accuracy move swamped by the fp32 hardware floor).

The one op-specific phase-3 surface was a **PE-idle gap** — v3 sits at PE=89% (~44 µs idle of
the 401 µs wall) vs the sibling's identical bf16-split kernel at PE=96% (~15 µs idle). The
~29 µs of extra idle is this op's extra per-M-tile non-PE work (the residual add `a=x+z`, the
extra `z` read, and the granular per-sub-tile activation limb-split) becoming *exposed* once the
matmul got fast. Phase 3 attacked that gap with a **bit-exact schedule simplification** (D1),
not the arithmetic:

- **D1 (`v4`, aT-split-from-PSUM) — MEASURED, bit-exact, within-noise floor-confirmation.**
  Split the transposed activation limbs directly from the transpose PSUM bank (`psum_t`),
  dropping v3's intermediate fp32 SBUF copy `aT_f` — 4→3 ops per K-sub-tile (32→24 per M-tile in
  the source). Result: compiles cleanly (the PSUM-source `nl.copy`/`tensor_tensor` primitives are
  codegen-feasible), full-5-seed PASS with rel-L2 **bit-for-bit identical to v3** (1.528161e-5,
  |Δ|=0), but latency within noise (v4 0.4014/0.4015 ms inside the v3 anchor band 0.4005–0.4017,
  +0.09%). The profiler digest is decisive: v3 and v4 have **identical** Vec/Scalar/matmul
  instruction counts (314/376/3336), identical HBM traffic (42/34 MB), no spill, PE/Vec active
  within 0.3%. The trn2 compiler already copy-propagated v3's exact fp32 `aT_f` copy into the
  downstream cast and subtract — i.e. it had already applied D1 at lowering — so the removed
  source ops carried no distinct hardware work and were never on the critical path (hidden under
  the PE-bound matmul, exactly like the sibling's transpose/stationary-fills at PE=96%). v4 is
  kept as tracked, more-readable equivalent evidence; v3 stays promoted (AC-5).
- **D2 (`v5`, residual-add / norm-exposure reduction) — NOT TRIGGERED.** AC-9 authorizes D2 only
  on a measured Vec/Scalar bubble. D1's digest shows none: Vec 21.7% / Scl 23.9% unchanged from
  v3, both hidden under PE 89.5%, identical instruction counts. The residual add and the full-K
  fp32 RMSNorm reduction are already hidden, so no numerically-neutral schedule tweak moves the
  wall clock. Building v5 would be a proactive D2 (forbidden). Skipped by measurement.

## Conclusion

Every shape-specialization lever is closed by the shape or by a hardware constraint, and the one
op-specific micro-lever (the PSUM-source limb split, D1) is *measured* to be already applied by
the compiler and hidden under the PE-bound 3-product matmul. The only lever that moved latency
in this op's history was breaking the fp32 rate ceiling with the compensated bf16×2 split (phase
2, v3, 3.898x→4.632x), which is already done and pinned. Within the fixed 3-product bf16
schedule, **v3 is at the floor at 4.632x**; v1 (3.754x) and v2 (3.898x) remain pure-fp32
fallbacks against the fixed-seed-42 adapter input-diversity caveat (AC-8.1). No further
shape-specialization or numerically-neutral schedule lever is open.
