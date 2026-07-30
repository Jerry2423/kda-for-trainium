# bmm_softmax — Phase 2 Exit Decision

**Promoted:** `runs/bmm_softmax_v4.py` — **1.946x** (3.7460 ms p50), full 5-seed L2 PASS,
on-device rel-L2 **2.5683307869e-6** (bit-for-bit identical to v1, ~7.8x under the 2e-5 gate).
**Fallback kept:** `runs/bmm_softmax_v1.py` (fp32, phase-1 base / DAG root, 1.585x).

Phase 2 improved the fused `out[b] = softmax_N(lhs[b] @ rhs[b])` kernel from **1.585x → 1.946x**
(a **22.8% wall reduction**, 4.5995 → 3.7460 ms) as a pure schedule + epilogue-fold change —
**no numeric change** (matmul math and softmax values bit-identical to v1).

## Promoted kernel = D1 + D2-fold + D3-M16

`bmm_softmax_v4` = the sibling `bmm_v2` two-phase transpose-all schedule (D1) + the max-negate
fold (the benign half of D2) + the interior `M_SUB=16` schedule depth (D3):
- **Two-phase schedule:** per batch, transpose all lhs subtiles of an m-block up front into a
  resident `lhs_t_pack[64, M_SUB*128]`, then run the block's main matmuls with no transpose
  interleaved, each subtile immediately followed by its softmax epilogue + 4096-wide store.
- **Max-negate fold:** `tensor_reduce(nl.max, negate=True)` writes `−row_max` directly, dropping
  the separate `neg_max = tensor_scalar(*−1)` op (one fewer Vector instruction per tile).
- **M_SUB=16** (2 m-blocks per batch, within one batch): the measured interior optimum.

## Per-direction verdict

| Dir | Description | Verdict | Evidence |
|-----|-------------|---------|----------|
| **D1** | Port `bmm_v2` two-phase transpose-all schedule | **KEEP (promoted-through)** | `bmm_softmax_v2` 1.857x; profile/bmm_softmax_v2_digest.txt, bmm_softmax_v2_d1_compare.md |
| **D2-fold** | `tensor_reduce(negate=True)` max-negate fold | **KEEP** | `bmm_softmax_v3b` 1.933x, Vec instr 3826→3418; profile/bmm_softmax_d2_compare.md |
| **D2-reduce_res** | fused `activation(reduce_op=add, reduce_res)` exp+row-sum | **REJECT (measured)** | `bmm_softmax_v3_fused_reduce_reject` +75% wall; profile/bmm_softmax_d2_compare.md |
| **D3** | `M_SUB ∈ {8,16,32}` within-batch sweep | **KEEP M_SUB=16** | M8 1.906x < M32 1.933x < **M16 1.946x**; profile/bmm_softmax_d3_msub_sweep.md |
| **GpSimd normalize** | move normalize off Vector (AC-6) | **NOT BUILT (precondition false)** | PE still the bind (PE 90% > Vec 72%); profile/bmm_softmax_ac7_audit.md |

### D1 — two-phase transpose-all (KEEP, promoted-through)
Ported the sibling `bmm_v2` schedule into the fused kernel (M_SUB=32). Pure schedule change:
`matmul_instruction_count` 8704 == v1 (purity guard), spill=0, HBM 33.6/1073.7 MB at the
read-once/write-once floor, rel-L2 bit-identical (transpose-before-use is exact). Full-5-seed
PASS **1.857x** (3.9250 ms).

**Bucket = HYBRID.** By AC-3 profiler semantics the literal bucket is **(c)**: TRUE PE-active
dropped only 3.6424 → 3.4595 ms (−5.0%), NOT "materially toward ~2.0 ms" — unlike pure bmm
(3.66 → 2.01 ms), the heavy softmax epilogue between subtile matmul bursts kept the stream from
deepening to bmm_v2's 0.231 µs stall. But **(c)'s label "D1-failed" is wrong as an OUTCOME**:
D1 delivered a large, out-of-noise **−14.7% wall win** by *hiding* the softmax Vec/Scalar tail
(exposed tail wall−PE 0.957 → 0.460 ms, −52%). This is a second route to a win the plan's
buckets did not anticipate (a PE collapse was assumed to be the only route). Classified
honestly (NOT (c)-dressed-as-(a)); promotable on DEC-1 (14.7% ≫ 3%). Codex (high effort)
concurred: "(c) by AC-3 profiler semantics; promotable tail-hiding win by DEC-1; proceed to D2."

### D2 — softmax-epilogue fusion (KEEP the fold, REJECT the reduce_res)
D2 as worded bundled two fusions; they were isolated to attribute the outcome (AC-4).
- **Max-negate fold: KEEP.** `bmm_softmax_v3b` 1.933x (3.7715 ms), −3.7% over D1 (robust:
  same-session bracket v2 {3.9257, 3.9242} vs v3b {3.7821, 3.7802} non-overlapping).
  `vector_engine_instruction_count` 3826 → 3418 (the `neg_max` op removed), Vec-active
  5.5416 → 5.3616 ms → satisfies AC-4's positive test (Vector pass dropped).
- **Fused exp+row-sum via `reduce_res`: REJECT (measured).** `bmm_softmax_v3_fused_reduce_reject`
  is correct (rel-L2 bit-identical) but +75% wall (6.8883 ms). The `reduce_res` accumulator
  side-effect in the same Scalar `activation` makes the compiler **re-run the whole
  transpose+matmul+score-build stream twice per inference**: apples-to-apples (`--fast`)
  `matmul_instruction_count` 8704 → 17408 (2.0×), `psum_read_sbuf_write_count` 4224 → 8448 (2.0×),
  `hbm_read` 33.6 → 67.2 MB (2.0×) while `hbm_write` stayed 1× (output once). Far from removing
  a Vector pass, it doubled the kernel. Fell back to the explicit `tensor_reduce(add)` row-sum
  (keep D1) per AC-4's negative test. **New lesson** (see memory / bitlesson): the fused
  `activation(reduce_res=)` free-axis reduce, whose "no additional cost" docstring claim was
  verified for the *reduce itself*, can trigger a whole-stream recompute in this
  matmul→SBUF-score→activation fusion — a measured anti-lever, not the free row-sum the plan
  hypothesized.

### D3 — M_SUB within-batch sweep (KEEP M_SUB=16)
Justified because D1 left softmax exposed (bucket-(b)-like: Vec 59→70%, Scl 59→67%). Swept
`M_SUB ∈ {8,16,32}` on the D1+fold kernel, within one batch (batch loop still `affine_range(B)`;
never cross-batch — the inherited anti-lever). **Non-monotonic interior optimum at M_SUB=16**:
M8 1.906x (PE-active 3.5799) < M32 1.933x (3.4352) < **M16 1.946x (3.3778 ms)**; same-session
bracket M16 {3.7461, 3.7462} vs M32 {3.7719, 3.7735} non-overlapping (~0.7%). All keep
matmul 8704, spill=0, HBM floor, rel-L2 bit-identical. **Refines the two-phase lesson:**
"deepen the stream to the whole reuse group" is optimal for a *light* epilogue (pure bmm store)
but a *heavy* per-tile epilogue (full-row softmax) has an interior optimum below the whole batch.
M_SUB=16 adopted as a within-winner tie-break (the phase promotion rests on the large D1+fold
win; the 0.7% M16 edge is robust and PE-active-corroborated).

### GpSimd normalize (AC-6) — NOT built
Gated on "only if Vector is still the bind after D1+D2." After the winner, **PE is still the
bind** (TRUE PE-active 3.378 ms > Vec 2.68 ms; PE% 90 > Vec% 72). Precondition false → out of
scope, not built (per AC-6 negative test). The wheel's `tensor_scalar(engine=gpsimd)` rsqrt-only
constraint would also have forced a reformulation; moot here.

## AC-7 — inherited measured rejects NOT built (Codex-confirmed)
- **bf16x2 3-product matmul split** — not built; every candidate fp32, matmul_instruction_count
  stayed 8704 (fp32 2.0-pass emulation). Evidence: `[[BL-20260710-bf16x2-loses-when-fp32-emulates-in-2-passes]]`.
- **bf16 exp/softmax** — not built; softmax all fp32, rel-L2 2.57e-6 (a bf16 exp over N=4096 ≈ 1e-2).
- **cross-batch blocking / double-buffer** — not built; batch loop `affine_range(B)`, M_SUB
  blocking within one batch only. Evidence: `[[BL-20260710-cross-batch-blocking-is-an-antilever-on-affine-range]]`.
- **removing the max-reduce or normalize pass** — not done; v4 keeps both (the fold removed only
  the separate `neg_max` op, not the max-reduce).

## Metric diff — promoted v4 vs v1

| metric | v1 (fallback) | v4 (promoted) | delta |
|---|---|---|---|
| p50 wall | 4.5995 ms | **3.7460 ms** | **−18.6% wall (1.585x → 1.946x)** |
| TRUE PE-active/inf | 3.6424 ms | 3.3778 ms | −7.3% |
| exposed tail (wall − PE) | 0.957 ms | ~0.368 ms | −62% |
| Vec-active/inf | 2.730 ms (59.37%) | ~2.68 ms (71.57%) | ↓ abs, %↑ (wall shrank) |
| Scalar % | 59.00% | 69.95% | re-exposed |
| DMA-active/inf | 1.408 ms (30.62%) | 1.397 ms (37.30%) | flat |
| matmul_instruction_count | 8704 | **8704** | **0 (purity guard ✓)** |
| vector_engine_instruction_count | 3639 | 3472 | −167 (fold + M16) |
| psum_read_sbuf_write_count | 4608 | 4224 | −384 |
| HBM read / write | 33.6 / 1073.7 MB | **33.6 / 1073.7 MB** | **at floor, spill=0 ✓** |
| on-device rel-L2 (all seeds) | 2.5683307869e-6 | **2.5683307869e-6** | **bit-identical ✓** |

## Exit rationale (DEC-2/DEC-3)
The phase beat its hard gates: correctness bit-identical (rel-L2 « 2e-5) and a robust wall win
(1.946x ≫ v1 1.585x, DEC-1 margin cleared ~7.5×). The D1→D2→D3 chain produced two composable
wins (two-phase schedule + max-negate fold) and one measured reject (fused reduce_res); D3
found an interior M_SUB optimum. The remaining bind is PE (matmul), and the pure-fp32 schedule
levers are exhausted (bf16x2 excluded on inherited evidence; GpSimd normalize precondition
false). Wall 3.746 ms sits above the directional ~1.75 ms Scalar-exp heuristic, but per DEC-3
we exit on exhausted profiler-justified wins within the iteration budget, not on hitting that
figure. `bmm_softmax_v1` retained as the fp32 fallback / DAG root.
