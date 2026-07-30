# gqa_full — Phase 2 Draft (profile-driven optimization)

## Starting point

Best correct kernel = `runs/gqa_full_v1.py` (DAG root, phase-1 promotion):
**1.462x / 10.6579 ms** over the 15.579 ms baseline, full 5-seed L2 PASS,
on-device rel-L2 **2.874e-6** (identical on every seed, ~7x under the 2e-5 gate).
fp32 throughout; per-head fusion (per-head bmm_softmax scores + full-row softmax
over the 4096-wide key axis, then a 32-step context matmul), scores never touch
HBM. Traffic is already at the read-once/write-once floor (HBMrd 67.2 MB == q+k+v,
HBMwr 33.6 MB == out, DMA 3.9%, no spill).

Phase-2 goal: minimize on-device latency without regressing correctness. Explore
each ranked direction for AT MOST five iterations, with before/after `verify.py`
latency + profiling evidence justifying keep / revise / reject.

## Round-0 bottleneck (evidence in `profile/gqa_full_phase2_bottleneck_evidence.txt`)

Per-inference, from `profile/gqa_full_v1_digest.txt` (metric window 2.0 inf):

| engine | active/inf | % |
|---|---|---|
| wall (p50) | 10.658 ms | — |
| **PE (TRUE)** | 5.125 ms | 48.1% |
| Vec | 4.321 ms | 40.5% |
| Scl | 4.505 ms | 42.3% |
| DMA | 0.415 ms | 3.9% (HBM floor, no spill) |

**The bind is SERIALIZATION, not any single engine.** The largest engine is PE at
5.125 ms, but wall is 10.658 ms — the **exposed tail (wall − PE) = 5.532 ms** (52%
of wall runs outside the PE stream). `sum(PE+Vec+Scl) = 13.95 ms` vs wall 10.66 ms
⇒ only 3.29 ms overlaps today. A schedule that overlaps down to the max engine
(5.125 ms) would be ~2.08x faster → up to ~3.0x total. **The primary lever is
overlap/schedule, not shrinking one engine.** This is the exact shape the sibling
`bmm_softmax` hit in phase 2: two-phase transpose-all cut its exposed tail
0.957→0.460 ms for 1.585→1.946x **without collapsing PE** — a tail-hiding win.

**Transposes are ~half the Tensor-Engine work** (the burden `bmm_softmax` lacked):
`tensor_engine_instruction_count 116385` vs `matmul_instruction_count 58112`.
Per-inference nc_matmul **sites** (512 tiles = kh8·grp2·tq32):

- real matmuls: score 8/tile = 4096; context 32/tile = 16384 → **20480**
- transposes: q 1/tile = 512; **A_t 32/tile = 16384**; k 256 → **17152** (46% of sites)

PE-cycle proxy (free-dim weighted): score 33% / context 33% / A_t-transpose 33%.
The **A_t transpose (16384 sites, the single largest PE class)** is the per-tile
32× attention transpose *inside* the context loop, run today as a serial
`transpose(PE) → copy(Vec/Scl) → matmul(PE)` chain 32× per tile — this interleave
is what keeps the PE stream shallow and holds the Vec/Scl copies in the exposed
tail.

## Ranked directions (benefit vs risk)

### D1 — Two-phase transpose-all in the context loop (+ M-block / M_SUB sweep) — PRIMARY

The promoted `bmm_softmax_v4` / `bmm_v2` lever, adapted to gqa's context matmul.
Today, per `(kh,grp,t_q)` the context loop does `for j in 32: transpose A[:,j] →
copy → o_psum += A_t·v[:,j]`, a PE→Vec→PE serial chain that also stalls the score
matmuls of the next tile. Restructure into two phases so the PE stream runs deep:

1. **Transpose-all first:** for the M-block of query tiles, transpose all 32 A
   subtiles (and q) up front into resident SBUF `A_t` tiles.
2. **Matmul-stream second:** then issue the 32 context matmuls back-to-back into
   the accumulating PSUM bank, so the transpose copies drain in parallel with a
   long uninterrupted matmul stream instead of gating it.

Sweep the M-block width **M_SUB ∈ {8, 16, 32}** query tiles processed together
before draining, exactly as `bmm_softmax_v4`. The sibling lesson
([[BL-20260711-heavy-epilogue-shifts-twophase-msub-optimum-interior]]): a **heavy
per-tile epilogue (full-row softmax) shifts the optimum to an INTERIOR M_SUB**
(bmm_softmax won at M16, not the whole-group M32). gqa's epilogue is heavier still
(softmax **plus** the 32× A_t transpose), so expect the interior optimum ≤ 16 —
test 16 first, then 8, then 32. Stay **within** the natural per-`(kh,grp)` reuse
group; never block across kv-heads (cross-batch/cross-group blocking is a proven
anti-lever, see [[kda-bmm-progress]] cross-batch-blocking-antilever).

- **Expected outcome:** tail-hiding win (wall drops toward PE-active ≈ 5 ms,
  PE-active roughly flat), NOT a PE collapse. Predict ~1.7–2.0x.
- **Risk:** medium. Pure reschedule → must stay **bit-identical** (rel-L2
  2.874e-6, matmul_instruction_count 58112, psum count, HBM floor all unchanged).
  SBUF: M_SUB=32 A_t tiles = 32·16 KB = 512 KB/partition would blow the budget —
  so M_SUB is *also* SBUF-bounded; the score/attn resident tile is 16 KB and a
  handful of A_t live tiles fit, but a full 32-wide A_t hold does not. This is a
  second reason the optimum is interior; verify no spill (DMA stays ~4%) at each
  M_SUB.
- **Iterations (≤5):** (1) two-phase at M_SUB=16 vs v1 baseline; (2) M_SUB=8;
  (3) M_SUB=32 (expect regress / SBUF pressure); pick the interior optimum;
  (4–5) reserve for a revise if the first cut regresses or spills.

### D2 — Scale-fold + defer-normalize (cheap, stackable, low-risk Vec reduction)

Two numerically-verified folds that shrink the co-limiting Vector engine
(offline check, seed 0, rel-L2 vs the v1 path = **5.36e-7** ≪ 2e-5 gate, below the
v1 device floor 2.874e-6; positive scale preserves row-argmax so the max-shift
stays valid):

1. **Fold `1/sqrt(D)` into `activation(scale=)`** and scale `neg_max` by the same
   factor (bias = `scale·(−max_unscaled)`), removing the full-width
   `score*scale` `tensor_scalar` (a 4096-wide Vec op every tile). This is the
   phase-1 draft's noted lever #2.
2. **Defer normalization past the context matmul:** run the context matmul on the
   **unnormalized** `exp` tile, then scale the small `[128,128]` output `O` by the
   `[128,1]` reciprocal. This turns the 4096-wide `attn = exp·recip` normalize
   (per tile) into a 128-wide `O·recip` — a **32× smaller** Vec op per tile —
   because `(exp/sum) @ v == (exp @ v) / sum` (per-row scalar pulls out of the
   contraction). `recip` is still computed from the full-width row-sum.

- **Expected outcome:** directly cuts Vec-active (the 40.5%/4.32 ms co-limiter);
  compounds with D1's overlap. Small standalone win; meaningful stacked.
- **Risk:** low. Numerically verified; still fp32; reference op order preserved up
  to the associativity of a positive per-row scalar. Build UNDER the D1 winner.
- **Iterations:** 1 (fold both, measure). Keep only if rel-L2 stays < 2e-5 (expect
  ~2.9e-6, unchanged) and wall improves or is neutral.

### D3 — bf16x2 3-product split on the score matmul (GATED, ranked last)

`matmul_instruction_count / real-GEMM-site = 58112/20480 = 2.84` (fp32 emulation
pass-multiple). The **score** matmul has moving `[128,512]` — the moving-512
regime where sibling GEMMs ran fp32 at ~1.8–1.95×/instr and the compensated
bf16x2 3-product split WON (matmul, rmsnorm_matmul, tmm). The **context** matmul
moving is `[128,128]` (small, weak split candidate — skip it).

BUT: PE is only 48% busy, and the split leaves the **46% transpose sites** in
fp32. Cutting PE moves the wall only AFTER D1+D2 have closed the Vec/Scl exposed
tail and PE has become the true bind. So D3 is **contingent**, gated on:
(a) post-D1/D2 profile showing PE ≥ ~Vec/Scl and PE the wall-limiter; and
(b) rel-L2 headroom — v1 is at 2.87e-6, only ~7× under 2e-5; the score-only split
adds ~4.45e-6 in quadrature (sibling floor), still passing, but confirm on-device.

- **Expected outcome:** uncertain; likely model/measured-reject given PE is not yet
  the bind. Explore ≤2 iterations only if the gate opens.
- **Risk:** high (correctness headroom + may not touch the wall). Keep v1 (and the
  D1/D2 winner) as fp32 fallback if it fails or washes.

### Not pursued (note-only)

- **Flash-style online softmax over n_k chunks:** its memory benefit is already
  captured (scores never hit HBM; DMA at 3.9%, no spill). Online softmax would
  ADD per-chunk running-max rescaling passes on the Vector engine — the WRONG
  direction while Vec co-limits. Only revisit if a future shape forces the score
  row out of SBUF (not the case here: 16 KB/partition fits).
- **Further transpose elimination:** the two transposes (q for scores, A for
  context) are fundamental to softmax-over-free-axis + the two matmul contraction
  layouts; D1 hides them rather than removing them.

## Success criteria / exit

- Correctness never regresses: every seed `[0,21,42,63,84]` passes rel-L2 < 2e-5
  (full run, not just `--fast`, before any promotion). Record on-device rel-L2.
- Promote the fastest candidate that holds correctness; keep `gqa_full_v1` as the
  fp32 fallback in the DAG. Log every perf change to `benchmark.csv`, every
  candidate to `candidates.jsonl` (parent links), profiling evidence to `profile/`.
- Target: close the 5.5 ms exposed tail via D1 (+D2), landing meaningfully above
  1.462x (aspirationally toward the ~2x that full PE/Vec overlap allows); treat
  D3 as upside only if PE becomes the bind.

## Validate / score

```bash
# fast (1 seed) during iteration:
python3 \
    ../../verify.py --op gqa_full --candidate runs/<kernel>.py --fast
# full 5-seed before promotion (drop --fast); full metrics:
python3 \
    runs/dump_metrics.py --op gqa_full --candidate runs/<kernel>.py
```
