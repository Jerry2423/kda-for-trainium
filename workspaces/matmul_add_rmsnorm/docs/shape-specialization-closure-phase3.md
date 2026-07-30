# matmul_add_rmsnorm (M4096 N2048 K2048, fp32 I/O) — Shape-Specialization Closure (Phase 3, AC-6)

`out = rmsnorm(x @ w + z) * g`: a GEMM, a residual add, RMSNorm over the **N (output/free)** axis,
and a per-N free-axis scale `g`. Fixed shape M=4096, N=2048, K=2048, fp32 I/O. Scored single-core
on the remote profiler (`--disable-dge --logical-nc-config=1`), correctness by NKIBench
relative-L2 < 2e-5 on seeds `[0,21,42,63,84]` (`verify.py` gates on `l2_norm_passed` for all seeds).

Phase 3 = regime/shape specialization: specialize only where the tensor's structure admits a
lever *and* a measured win justifies the complexity. This is the MIRROR of the rmsnorm-first
siblings (`add_rmsnorm_matmul`, `rmsnorm_matmul`) — same M/N, same fp32 contract, same bf16x2
3-product split — but with the GEMM FIRST (so the norm reduces over the matmul-output N axis, `g`
sits on the output free axis and is never folded into `w`, and `inv_rms` cannot commute out). The
shape levers therefore close **identically** to those siblings
(`workspaces/add_rmsnorm_matmul/docs/shape-specialization-closure-phase3.md`), and this doc is that
closure specialized to this op. The finding: this fixed shape offers **no shape-specialization
surface** — every classic phase-3 lever is either vacuous (no ragged structure to exploit) or
already pinned at its hardware constraint. The absence of a lever is recorded as a first-class
result, not a gap.

## Per-lever closure (all five levers)

| Lever | Applies? | Shape / constraint reason (M4096 N2048 K2048) |
|---|---|---|
| **Edge / partial tiles** | **No — vacuous** | Every dim divides its tile size evenly: M=4096=32·128, K=2048=16·128, N=2048=4·512. No ragged tile, no remainder loop, no masked/partial tile anywhere. Edge-tile specialization requires an edge; there is none. |
| **Tile-size / partition-free regime** | **No — layout forced** | `nc_matmul(stationary, moving) = stationary.T @ moving` requires the contraction dim (k_in) on the PARTITION axis of both SBUF operands and produces an output tile `[m_in(par), n(free)]`. So m_in is forced onto the stationary/partition axis and n onto the moving/free axis. Swapping m↔n would require transposing the entire N=2048-wide result back — a cost far larger than any tiling gain. Also the RMSNorm reduces over N; keeping N on the free axis makes it a cheap in-partition `tensor_reduce`. Forced. |
| **N-chunk (moving-free) width** | **No — already maximal** | N_CHUNK=512 = `psum_fmax` = exactly one fp32 PSUM bank in the free dim — the documented per-matmul moving-free maximum (knowledgebase precedent `6288aaad`: "tile budget is psum_fmax, not pmax"). Larger exceeds a PSUM bank; smaller wastes the systolic streaming width. Already at the constraint. |
| **M-blocking** (the `matmul` task's phase-2 win) | **No — vacuous** | That win removed *redundant w HBM reloads* by reusing a loaded w-tile across output-row tiles. Here the two bf16 `w_hi`/`w_lo` limbs are 128 KB/partition total (= the SAME bytes as v1's one fp32 w; two bf16 limbs at 2 B == one fp32 at 4 B; budget ~208 KB), loaded **fully resident once** before the M-loop, then reused across all 32 M-tiles; each `x` and each `z` tile is loaded exactly once. HBMrd=84 MB is already minimal (one read of x + one read of z + one read of w). There is no redundant HBM traffic to block for. (The stationary-reuse reorder, D1/v3 below, is NOT M-blocking — it does not touch HBM; it reschedules on-chip PSUM/stationary reuse.) |
| **LNC2 / multi-core sharding** | **No — out of contract** | Scored single-core (`--logical-nc-config=1`). LNC2 sharding is not a lever on this harness; using it would change the scoring contract, not optimize within it. |

## K-note (the one shape difference vs the K=1024 siblings)

This op's K=2048 (16 K-tiles) is **2×** `add_rmsnorm_matmul`'s and `rmsnorm_matmul`'s K=1024
(8 K-tiles) — the only shape difference that could make a *scheduling* lever behave differently
here. It doubles the K-accumulation depth per PSUM bank and doubles the number of distinct
stationary limbs (**32 vs 16 per M-tile**: 16 K-tiles × {`xT_hi`, `xT_lo`}), which is precisely
what gives the D1 stationary-reuse reorder (§ below) a longer reuse run to group — a 2→8
(`xT_hi[kt]` over P1+P2 across 4 chunks) and 2→4 (`xT_lo[kt]` over P3) run length instead of the
sibling's shorter K. Everything else closes identically to the siblings.

## Why nothing else is left inside the fixed 3-product bf16 schedule

The promoted `matmul_add_rmsnorm_v2_bf16_split` (4.879x, 0.7722 ms) is PE-bound at PE=91.66%,
MFU=42.55%. TRUE PE-active (0.7078 ms) is the fixed floor: `2·M·N·K` at the bf16 systolic rate,
run 3× for the 3-product compensated split. Cutting PE-active further needs either fewer products
(fails the gate — plain bf16 ~117× over) or a lower-precision matmul (fails the gate). Closed and
pinned (D5 forbidden this phase; the 4-product v2b was already a decision-SKIP — v2's per-seed
rel-L2 1.544749e-5 sits comfortably below the 1.8e-5 danger band, and offline the 4th product
moves rel-L2 only 4.454e-6→3.491e-6 for a ~+25% PE cost, the sibling v3b analog MEASURED-REJECT
+28%).

The one op-specific phase-3 surface was the **~64 µs PE-idle** (PE=91.66%, 8.3% of the 772 µs
wall). The only precision-neutral way to chase it is to schedule the *same* matmuls so the PE
array stalls less. Phase 3 measured two such levers, both non-arithmetic:

- **D1 (`v3`, stationary-reuse GEMM loop reorder) — MEASURED, bit-exact, small out-of-noise
  regression (+1.95%), NOT promoted.** Reorder the GEMM from N-chunk-outer to K-tile-outer with 4
  live [128,512] fp32 PSUM accumulators grouped by shared stationary limb, lengthening the
  stationary-reuse run 2→8/4 and cutting stationary loads 128→32 per M-tile. Result: compiles
  cleanly (no PSUM-bank/spill error), full-5-seed PASS with rel-L2 **bit-for-bit identical to v2**
  (1.544749420463104e-5 on every seed, |Δ|=0 — the kt loop is outer so each bank's per-bank
  reduction order P1,P2,P3 over kt=0..15 is preserved; only the cross-bank interleaving changed;
  Codex confirmed at 0.9 conf). The stationary-reuse lever *did* fire —
  `tensor_engine_instruction_count` dropped 11074→9853 (fewer stationary loads) — but the PRIMARY
  mechanism metric moved the WRONG way: TRUE PE-active 0.7089→0.7110 ms (+0.30%), PE% 91.72→90.28
  (PE-idle 8.28%→9.72%). The cut loads were already hidden behind the moving stream, and the 4 live
  accumulators enlarged the `affine_range` live PSUM set, slightly raising per-instruction PE stall
  — the enlarged-live-set regression mode of `BL-20260710-cross-batch-blocking-is-an-antilever`.
  Same-session bracketed p50 (re-verified 2026-07-12 after a prior-session crash: v2 anchor A 0.7729
  / v3 0.7879,0.7868 / v2 anchor B 0.7728; the crashed session measured the twin 0.7876,0.7882 /
  0.7729,0.7727): NON-overlapping bands, **+1.88% regression** (crashed session +1.95% — SAME
  finding) → **NOT promoted** (AC-5 promotes only on a >1.8% *win*). All regression sentinels flat
  (matmul 6664, psum_read 132, HBM 84/34 MB) — a pure reschedule.
- **D2 (`v4`, 2-bank/2-chunk grouping) — CONTINGENT diagnostic, compiler no-op relative to v2.**
  Triggered by AC-9 because v3's PE-idle went UP (a specific PSUM/pipeline bubble), not a clean
  no-op. Halving the live set (4→2 banks, reuse run 4) recovers fully to v2: p50 0.7726 ms,
  PE 91.66%, TRUE PE-active 0.7082 ms, and `tensor_engine_instruction_count` back to **11074 (==
  v2, NOT v3's 9853)** — neuronx-cc scheduled the 2-chunk grouping into v2's exact instruction
  stream. Full-5-seed PASS, rel-L2 bit-exact 1.544749e-5. This confirms the v3 regression was the
  enlarged live set (halving it recovers), but v4 does NOT beat v2 — it just returns to v2's
  schedule. Neither reorder improves on v2.
- **D3 (`v2_psum_split`, PSUM-source activation-limb split) — MEASURED in phase 2, byte-identical
  compiler no-op.** The sibling's headline phase-3 primary (split `xT_hi`/`xT_lo` directly from the
  transpose PSUM bank, dropping the intermediate fp32 `xT_f` copy) was already built and measured
  here during phase 2: all instruction counts `==` (matmul 6664, Vec 566, Scl 246, psum_read 132),
  TRUE PE-active 0.7078→0.7079 ms, rel-L2 bit-exact 1.544749e-5, +0.08% latency within noise.
  neuronx-cc already copy-propagates the exact fp32 PSUM→SBUF copy
  (`BL-20260710-compiler-copy-propagates-exact-psum-sbuf-copy`). Closed by this op's own
  measurement; not re-litigated in phase 3.
- **D4 (off-PE transpose), D6 (split-before-transpose) — closed record-only.** D4: SBUF→SBUF
  `dma_transpose` of a [128,128] tile is infeasible (HW-DGE needs `src.shape[0]==16`, SW-DGE needs
  an HBM source), and `nc_transpose` lands in fp32 PSUM needing a re-cast and measured +2% on the
  siblings; the 512 identity-matmul transposes are already hidden under the PE-bound matmul. D6:
  splitting `x` into limbs *before* the transpose doubles the transpose PE work (32 transpose
  matmuls/M-tile instead of 16) — adding the one thing we cannot afford. Both closed by sibling
  precedent + this op's cost structure.

## Conclusion

Every shape-specialization lever is closed by the shape or by a hardware constraint. The op-specific
micro-lever measured in phase 2 (the PSUM-source limb split, D3) is *already applied by the compiler*
and hidden under the PE-bound matmul. The one genuinely-untested precision-neutral lever (the D1
stationary-reuse reorder, `v3`) is measured in phase 3 to be a **small out-of-noise regression
(+1.88% re-verified / +1.95% crashed session, bit-exact)** — the stationary-reuse lever fires
(fewer PE-engine instructions) but the cut
loads were already hidden and the 4 live PSUM banks enlarge the pipeline's live set; the 2-bank
diagnostic (`v4`) recovers to v2 exactly, isolating the enlarged live set as the cause without
beating v2. The only lever that ever moved latency in this op's history was breaking the fp32 rate
ceiling with the compensated bf16×2 split (phase 2, v2, 3.920x→4.879x), which is already done and
pinned. Within the fixed 3-product bf16 schedule, **v2 is at the floor at 4.879x**; `v1` (3.920x,
pure fp32) remains the guaranteed-correct fallback against the fixed-seed-42 adapter input-diversity
caveat (AC-8.1). No further shape-specialization or numerically-neutral schedule lever is open.
