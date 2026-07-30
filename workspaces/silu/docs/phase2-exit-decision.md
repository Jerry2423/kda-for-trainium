# silu phase-2 exit decision

## Final result: v1 UNCHANGED + roofline confirmed (the plan's explicit lower bound)

`runs/silu_v1.py` (0.3009 ms, 3.398x over the 1.022441 ms baseline) **remains the
phase-2 result, unchanged.** No candidate cleared the AC-8 same-session noise band;
every explored lever either regressed or is redundant with the compiler's existing
software-pipeline. This is the plan's explicitly-legitimate lower-bound outcome, not a
failure: v1 sits at the achieved single-core streaming HBM roofline, so there is no
multiplicative headroom and the only slack (a ~3% / ~9 µs DMA-issue bubble) could not
be harvested without coarsening the pipeline that already hides it.

## Primary deliverable (AC-2): roofline confirmed with evidence
- Traffic floor = `2 · 4096 · 7168 · 4 B = 234.88 MB` = measured `HBMrd 117 + HBMwr 117 MB`
  exactly. Zero redundant traffic.
- Effective aggregate BW = `234.88 MB / 0.3009 ms ≈ 781 GB/s`, framed as the **achieved
  single-core streaming roofline** for this kernel/schedule — NOT the HBM-fabric maximum.
- `DMA = 97%` active; the cost model's `368 GB/s` is per-unidirectional-stream (serialized
  0.638 ms is ~2x conservative; measured 0.3009 ms is below the overlapped one-way
  0.319 ms and within ~6% of the 736 GB/s two-stream model).
- Scalar compute floor (~0.191 ms cost-model / ~0.102 ms measured-active) is hidden under
  DMA either way; k-batching does not change the total Scalar element count.
- Independently reviewed by Codex (high effort): AGREE on both claims, caveats folded in.
- Written up in `docs/phase2-roofline-confirmation.md`; digest `profile/silu_phase2_roofline.txt`.

## Secondary deliverable (bounded DMA-bubble harvest): explored, nothing promotable

| Candidate | Lever | Result | HBM | Verdict |
|-----------|-------|--------|-----|---------|
| silu_v1 (parent) | k=1, affine_range(32) | 0.3009 ms, DMA 97% | 117+117 | kept |
| silu_v2_k2 | D1 k=2 batching | 0.3350 ms (--fast), DMA 85% | 117+117 | rejected (screen regress) |
| silu_v2_k3 | D1 k=3 + exact tail | 0.7729 ms (--fast), DMA 71% | 117+117 | rejected (screen regress) |
| silu_v2_k4 | D1 k=4 + D3 in-place | 0.3971 ms (full 5-seed), DMA 72% | 117+117 | rejected (screen regress); AC-5 in-place validated |
| silu_v2_pingpong | D2 explicit ping-pong | 0.5889 ms (--fast), DMA 97% | 117+117 | rejected (redundant/worse than affine_range) |

- **D1 (AC-3):** monotone worsening; k=1 (v1) is best. Wider bursts → fewer
  `affine_range` iterations → coarser pipeline → more DMA idle. Digest
  `profile/silu_v2_d1_ksweep.txt`.
- **AC-3.1 k∈{5,6,7} probe:** trigger is "k=4 still monotone-best (latency still
  decreasing)". It is not (min at k=1), so the probe was **not run**. The soft ~32767
  free-dim bound (k=5 → 35840) is documented but is not the stopping reason; the sweep
  stops because it is monotonically worse than k=1.
- **D3 in-place (AC-5):** `silu_v2_k4` uses `nisa.activation` with `dst == data` (one
  live 112 KB tile; separate buffers = 224 KB > 208 KB). Compiled with no
  fallback/legality warning, passed the full 5-seed L2 gate (exit 0, correct 1/1), HBM
  at floor, no extra spills — the undocumented `src == dst` aliasing is empirically safe.
- **D2 (AC-4):** exactly one confirmation run; regressed (`sequential_range` forbids the
  compiler pipelining that `affine_range` licenses). Recorded as redundant. Digest
  `profile/silu_v2_pingpong.txt`.
- **D4 (AC-6):** rejected on the confirmed `--disable-dge` premise; conditional attempt
  not triggered (no unexplained bubble — D1/D2 only raised it).
- **D5 (AC-6):** rejected outright (fp32 gate fixes the 234.88 MB floor; no dtype trick).
  Digest `profile/silu_phase2_d4_d5_rejection.txt`.

## AC-8 gate outcome
No candidate beat v1 in the `--fast` screen (all regressed by ≥11%), so none qualified
for the full interleaved `A0,B0,A1,B1,A2` promotion sequence. Per AC-8, `--fast` is
screen-only; a screen regression cannot be promoted, and a within-noise tie keeps v1.
Result: **keep v1.**

## AC-1 (correctness)
Every candidate passed the rel-L2 gate (correct 1/1); silu_v2_k4 additionally passed
the FULL 5-seed run. No bf16/tf32/fp16 introduced anywhere. v1's promoted status is
unchanged and its correctness is untouched.

## Evidence (AC-7)
5 candidate rows in `benchmark.csv`; 7 nodes in `candidates.jsonl` (all parented off
`silu_v1`, including first-class D4/D5 rejection nodes); 5 `profile/` digests with HBM
counters; 4 new candidate `.py` sources under `runs/` (committed/tracked).
