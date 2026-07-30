# bmm_softmax — Phase-2 Bottleneck Evidence (task7, analyze/Codex)

EVIDENCE ONLY — not a phase-1 correctness gate. Characterizes the measured bottleneck of
the phase-1 kernel and confirms/refutes the baseline-spill diagnosis, as phase-2 input.

## Measured digests (remote Trainium, full 5-seed, window-normalized per inference)

| | v1 (fused, mine) | NKIBench baseline | pure bmm_v1 (ref) |
|---|---|---|---|
| p50 latency | 4.5995 ms | 7.3020 ms | 3.8477 ms |
| TRUE PE-active/inf | 3.6424 ms (79.19%) | 3.7345 ms (51.14%) | 3.6552 ms (95%) |
| TRUE DMA-active/inf | 1.4083 ms (30.62%) | 3.4246 ms (46.90%) | — |
| Vec % | 59.37% | 46.70% | 20% |
| Scl % | 59.00% | 51.02% | 14% |
| matmul_instruction_count | 8704 | 10240 | 8704 |
| HBMrd | 33.6 MB | 700.5 MB | 34 MB |
| HBMwr | 1073.7 MB | 1740.6 MB | 1074 MB |
| psum_read_sbuf_write_count | 4608 | 5632 | — |
| on-device rel-L2 | 2.5683e-6 | 1.8253e-6 | — |

## Q1 — Baseline-spill diagnosis: CONFIRMED (with a quantitative refinement)

Read/write floors: input read floor 33.55 MB, output write floor 1073.74 MB.

Baseline excess:
- HBMrd excess = 700.5 − 33.55 = **666.95 MB**
- HBMwr excess = 1740.6 − 1073.74 = **666.86 MB**
- Total excess round-trip = **1333.8 MB**

The near-perfect read/write symmetry (~667 MB each way) is strong evidence of an intermediate
tensor written to HBM and reread — i.e. the baseline round-trips a score-like intermediate
through HBM. My fused kernel eliminates it entirely: HBMrd sits at the 33.6 MB input floor and
HBMwr at the 1073.7 MB output floor, with spill counters = 0.

**Codex adversarial refinement (adopted):** the round-trip is **~667 MB, ~62% of a full
`(B,M,N)` fp32 score tensor (1073.74 MB)**, NOT the entire score matrix. So the correct claim
is "the baseline spills/rereads a substantial score-like intermediate (~667 MB each way,
~1.33 GB total)," not "the baseline spills the entire score matrix." (The draft's "materializes
essentially the entire score matrix and spills ~1GB" overstated it; the measured spill is ~62%
of that — likely the baseline keeps part of the working set on-chip and spills the remainder.)

## Q2 — v1's true bottleneck: PE floor + exposed softmax Vec/Scl tail (NOT DMA)

- Wall 4.5995 ms − TRUE PE-active 3.6424 ms = **0.957 ms** of wall not covered by PE.
- DMA-active 1.4083 ms is largely overlapped (PE + DMA > wall) and traffic is at the floor, so
  the kernel is **not DMA/spill-bound**.
- Vec and Scl are both ~59% (the new exp/reduce/reciprocal/scale softmax work). Most hides under
  the PE-bound (79%) matmul, but ~0.96 ms is an **exposed softmax Vec/Scl tail + scheduling**.
- **Single most promising phase-2 lever (Codex + mine):** reduce/hide the full-row softmax
  Vec/Scl tail. Candidate moves: the two-phase transpose-all schedule (`bmm_v2`, 1.253x on pure
  bmm) to deepen the matmul stream so more of the softmax hides, and the `activation` fused
  row-sum (`reduce_op=nl.add`) to drop one Vector pass. The GEMM path itself is already near the
  pure-bmm PE floor — little surface there.

## Q3 — Fusion did not change the matmul: AGREED

TRUE PE-active 3.6424 ms (mine) vs 3.6552 ms (pure bmm) = −0.35%, within measurement noise;
matmul_instruction_count 8704 identical. The matmul workload is preserved by the fusion.

## Q4 — Red flags: none material

- 1.585x is plausible and mechanistically explained: baseline sheds ~1.33 GB excess HBM traffic
  (DMA-active 3.42 → 1.41 ms).
- HBMwr == output floor and HBMrd == input floor are strong positive signals: no score spill,
  no score reread.
- rel-L2 2.57e-6 vs the 2e-5 gate is numerically trustworthy for fp32 softmax (~7.8x margin).
- Only caveat (adopted above): quantify the baseline spill as ~667 MB, not a full score matrix.

## Verdict

Baseline-spill diagnosis CONFIRMED at the traffic level (with the ~667 MB / ~62% refinement).
v1 is PE-bound at the pure-bmm floor with an exposed ~0.96 ms softmax Vec/Scl tail — the phase-2
surface. Codex (high effort) concurred on all four points and supplied the ~62% refinement.
