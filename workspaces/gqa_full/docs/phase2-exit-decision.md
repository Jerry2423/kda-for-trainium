# gqa_full — Phase 2 exit decision

## Outcome

**PROMOTED: `runs/gqa_full_v5_scoresplit.py` at 2.167x (7.1907 ms)** over the
15.579 ms baseline — up from the phase-1 fp32 base `gqa_full_v1` (1.462x, 10.6579 ms).
`gqa_full_v1` (fp32, 1.462x) is retained as the **guaranteed fp32 fallback** (DAG
root); `gqa_full_v4_msub2` (2.070x) is the fastest **fp32-only** candidate (D3 uses
bf16 limbs on the score matmul only).

All four ranked directions were explored to their evidence-based conclusion (the plan
Upper Bound). Every KEPT/PROMOTED candidate on the winning path was scored FULL-5-SEED
(v1, D1a `gqa_full_v2_twophase`, D2c combined `gqa_full_v3c`, D1b M_SUB={1,2}, and D3
`gqa_full_v5_scoresplit` — all PASS, per-seed `l2_norm_passed=True`); the D2 isolation
nodes (`gqa_full_v3a`, `gqa_full_v3b`) and the off-optimum M_SUB={4,8} sweep points were
`--fast` (seed-42) screens only, used for attribution/direction and not promoted. HBM
stayed at the read-once/write-once floor (67.2 / 33.6 MB, no spill) at every candidate.

## The lever ladder (each measured, DAG under gqa_full_v1)

| kernel | lever | p50 (ms) | speedup | rel-L2 | scored | note |
|--------|-------|----------|---------|--------|--------|------|
| gqa_full_v1 | fp32 base (phase 1) | 10.6579 | 1.462x | 2.874266e-6 | full-5 | fp32 fallback / DAG root |
| gqa_full_v2_twophase | D1a two-phase context loop | 10.5204 | 1.481x | 2.874266e-6 (bit-identical) | full-5 | KEPT bit-exact base |
| gqa_full_v3a_scalefold | D2 scale-fold alone | 9.6523 | 1.614x | 2.873728e-6 | --fast | isolation node (not promoted) |
| gqa_full_v3b_defernorm | D2 defer-normalize alone | 8.6820 | 1.795x | 2.874244e-6 | --fast | isolation node (not promoted) |
| gqa_full_v3c_scalefold_defernorm | D2 combined | 7.7904 | 2.000x | 2.873922e-6 | full-5 | KEPT base for D1b |
| gqa_full_v4_msub1 | D1b M_SUB=1 control | 7.7902 | 2.000x | 2.873922e-6 | full-5 | == v3c (pure-reschedule control) |
| **gqa_full_v4_msub2** | **D1b M_SUB=2 (interior optimum)** | **7.5276** | **2.070x** | 2.873922e-6 | full-5 | fastest fp32-only; D3 parent |
| gqa_full_v4_msub4 | D1b M_SUB=4 | 7.5470 | 2.064x | 2.873922e-6 | --fast | past the optimum (not promoted) |
| gqa_full_v4_msub8 | D1b M_SUB=8 spill probe | 7.5448 | 2.065x | 2.873922e-6 | --fast | NO spill; saturates (not promoted) |
| **gqa_full_v5_scoresplit** | **D3 bf16x2 score split** | **7.1907** | **2.167x** | 5.421586e-6 | full-5 | **PROMOTED FINAL** |

## Why each direction landed where it did

**D1a (two-phase context loop) — KEPT as bit-exact base, small standalone win.**
Restructuring the per-tile `transpose->copy->matmul` chain into transpose-all then
matmul-stream let the compiler coalesce the 32 narrow per-tile A_t PSUM drains into
~8 wide copies (psum_read_sbuf_write_count 21568->9280, Vec/Scl instr ~halved), but
wall only moved -1.22% — the ~5.5 ms exposed tail is dominated at M_SUB=1 by the
per-tile score->softmax->context dependency, not the A_t interleave. `matmul_instruction_count`
identical (58112), rel-L2 bit-identical (AC-1.1 purity held).

**D2 (scale-fold + defer-normalize) — the biggest single jump, 1.481x -> 2.000x.**
Drafted as a "small standalone win"; it was the largest. Each fold removes ONE
4096-wide Vector pass that sat in the exposed serialization tail: scale-fold drops
the `score*scale` (-8.25% alone), defer-normalize drops the `attn=exp*recip`
pre-normalize (-17.47% alone, and frees the 16 KB attn buffer). Combined: Vec instr
14585->4741 (-67%), wall -25.95% vs D1a. Additive and monotonic across the isolated
folds -> no numeric/scheduling surprise. rel-L2 ~= v1 floor (algebraically exact,
fp32 reassociation only). Codex AGREE: recip is a per-row scalar so (exp/sum)@v ==
(exp@v)/sum is exact; deferred exp@v is fp32-safe (exp<=1, sum<=4096, v O(1)).

**D1b (M_SUB query-tile batching) — interior optimum at M_SUB=2, 2.070x.**
The heavy-epilogue interior-optimum pattern (softmax + the 32x A_t transpose is a
heavy epilogue): M_SUB=1 too shallow (7.7907), M_SUB=2 best (7.5273), M_SUB={4,8}
saturate/slightly worse (~7.545). M_SUB=1 reproduces D2c bit-for-bit (clean control).
M_SUB=8 spill probe did NOT spill (usable SBUF ~208 KB). Promotion bracket
(same-session interleaved full-5-seed): A(M1){7.7902,7.7902} vs B(M2){7.5276,7.5274},
gap 3.37% NON-OVERLAPPING. Pure reschedule (matmul 58112 identical, rel-L2 bit-identical).

**D3 (bf16x2 score split) — gate OPENED, WON, 2.070x -> 2.167x.**
Ranked last and expected "likely model/measured-reject" because at v1 PE was only 48%
busy. But D2 collapsed the Vec tail and made PE the wall-limiter (68% at M_SUB=2:
PE-active 5.129 >> Vec 3.076 >> Scl 2.894 ms/inf) -> the plan's own gate condition
fired. Offline whole-attention pre-check (softmax exponentiates the score, so the
sibling GEMM-quadrature floor is NOT used) authorized: worst 4.684e-6 across 7 seeds.
On-device: TRUE PE-active 5.129->4.709ms (-8.2%), wall -4.47%, rel-L2 5.421586e-6 ==
quadrature sqrt(2.874e-6^2 + 4.643e-6^2)=5.46e-6 (0.7% off), 3.7x under the 2e-5 gate.
Bracket A(M2){7.5272,7.5273} vs B(D3){7.1890,7.1899} gap 4.49% NON-OVERLAPPING.
The moving-512 per-instruction-rate mechanism (3 bf16 < fp32 emulation) + resident
limbs (2 bf16 == 1 fp32 bytes, HBM at floor, no reload). D3 budget <=2: iter 1 landed
the win -> STOP (4-product is a known measured-reject).

## Independent review (Codex, task8, analyze)
high effort AGREE on all three numerical claims (D1a bit-exact reschedule; D2
defer-normalize/scale-fold exact + fp32-safe; D3 whole-path validation methodology +
5.42e-6 sound). profile/gqa_full_phase2_codex_review.md.

## Correctness caveat (bf16 route)
The adapter reuses seed-42 input for all 5 profiler seeds, so the on-device 5-seed
PASS is weak on input DIVERSITY. The offline sim's 7 distinct-seed draws (worst
4.684e-6) carry the real margin. `gqa_full_v1` (fp32, 1.462x) is the guaranteed
fallback and `gqa_full_v4_msub2` (2.070x) the fp32-only next-best if a future
evaluator draws genuinely distinct per-seed inputs.

## Not pursued (as planned)
- Cross-kv-head / cross-query-group blocking — measured anti-lever (kda-bmm-progress).
- Fused `activation reduce_res` row-sum — measured +75% anti-lever (kda-bmm-softmax).
- Online/flash chunked softmax — adds Vec rescaling, wrong direction while Vec co-limits.
- Transpose elimination — fundamental; D1 hides the transposes, does not remove them.
- Shape specialization — phase 3.
