# lora — Phase 3 Exit Decision

## Outcome: FINALIZE `lora_v3_bf16_split` (D2) at 1.297x; `lora_v2_mblk4` (0.988x) kept as the fp32 fallback

Phase 3 asked one narrow question: **is there any restructuring that beats D2's 1.297x, or is
D2 the finalize?** The answer is **D2 is the finalize.** The single genuinely-untested lever —
the canonical LoRA weight-fold `lora_v4_fold` (E1) — was built, offline-gated, and measured on
a full 5-seed same-session A/B; it is a **wash** (1.09% over D2, inside the 3% promotion bar).
E2 and E3 model-reject against the profile, an offline SBUF budget, and measured sibling
precedent.

| kernel | precision | latency (full 5-seed) | speedup | worst rel-L2 | HBMrd | HBMwr | role |
|--------|-----------|-----------------------|---------|--------------|-------|-------|------|
| lora_v1 (phase 1) | fp32 | 38.3562 ms | 0.382x | 4.874e-7 | 7813 MB | 201 MB | superseded |
| lora_v2_mblk4 (D1) | fp32 | 14.8385 ms | 0.988x | 4.874e-7 | 2150 MB | 201 MB | **fp32 fallback** |
| **lora_v3_bf16_split (D2)** | bf16x2 base | **11.2989 ms** (p3 A/B) | **1.298x** | 6.240e-7 | 2112 MB | 201 MB | **FINALIZE** |
| lora_v4_fold (E1) | bf16x2 fold | 11.1756 ms | 1.312x | 4.4484e-6 | 2297 MB | 453 MB | **measured reject (wash)** |

`out = x@w + (x@a)@b` (M=4096, K=5120, N=12288, R=128, fp32). Baseline 14.6645 ms. D2's
phase-3 A/B re-confirm (11.2992 / 11.2986 ms, band 0.0006 ms) reproduces its phase-2
11.3034 ms; 1.297x is the kernel of record.

## E1 (weight-fold `lora_v4_fold`) — BUILT, gated, measured, REJECTED as a wash

E1 tests the LoRA algebraic identity `x@w + (x@a)@b = x@(w + a@b)`: materialize `w' = w + a@b`
once to fp32 HBM (an internal `shared_hbm` scratch), then run a pure `x@w'` bf16x2 3-product
GEMM — literally the sibling `matmul_v3_bf16_split` base GEMM on folded weights, with D2's
down-proj / up-proj / resident-`a` / `tT` machinery deleted.

### AC-2 — offline no-spend gate AUTHORIZED (before any remote spend)
The extended `runs/offline_lora_bf16_split_sim.py` (route `[F]` = `mm_bf16x2_3prod(x, fp32(w+a@b))`
vs the fp32 reference) authorized the fold:
- **AC-2.1 / AC-2.2:** folded worst-over-seeds **4.456e-6** (seed-42 gate 4.450e-6; diversity
  seeds 0/21/63/84 worst 4.456e-6) — `< 8e-6` (model-consistency band) `< 1.3e-5` (absolute
  cap). It correctly **loses the ~11.4x dilution** that the base-only route `[A]` enjoys:
  `[F]` 4.450e-6 ≈ base-only-in-isolation 4.453e-6, **not** the diluted `[A]` 3.930e-7 — the
  whole output (incl. the 99.6%-magnitude-dominant low-rank term) flows through one bf16x2 GEMM.
- **AC-2.3:** the fp32 reassociation control `[F-fp32]` = `rel_l2(fp32(x@(w+a@b)), x@w+(x@a)@b)`
  = **6.079e-7** (~1.25x the fp32 floor 4.874e-7, ~7.3x below the bf16 fold) → the folded-route
  error is **bf16-dominated, not a reassociation artifact**.
- **AC-2.4:** the fail-closed independent-reference control still `raise`s (not `assert`,
  survives `python -O`; verified: `NKIBENCH_ROOT=<bogus> python -O` exits 1) — the gate can
  never authorize on an unvalidated draw model. fp32 control vs the independent reference
  module = 0.000e+00 (draw model validated).
- **Codex (high effort)** independently reviewed and **AGREED on all 4 points**, concurring
  with authorizing exactly one gated remote run (only a minor wording note on a print label,
  addressed).

### AC-1 — E1 PASSES the correctness gate
On-device full 5-seed PASS at **rel-L2 4.4484e-6** (all seeds; `l2_norm_passed=True`), ~4.5x
under the 2e-5 gate. This matches the offline fold 4.450e-6 and the **predicted device
quadrature** `sqrt(4.874e-7² + 4.45e-6²) = 4.476e-6` — the bf16 term **DOMINATES** here,
**inverting** D2's fp32-floor-dominated 6.240e-7 (D2 keeps the low-rank fp32, diluting its base
split error 11.4x below the floor; the fold does not).

### AC-3 — same-session interleaved A/B → WASH, NOT promoted
Interleaved bracket (full 5-seed p50): **D2-before 11.2992 → E1 11.1756 → D2-after 11.2986 ms**,
so `band = |11.2992 − 11.2986| = 0.0006 ms` (negligible drift). The comparator:
`PROMOTE iff E1_p50 < min(D2_before, D2_after) − max(band, 0.34 ms)` = `< 11.2986 − 0.34 =
10.9586 ms`. **E1 = 11.1756 ms > 10.9586 ms → NOT promoted.** E1 beats the faster D2 bracket by
only **0.123 ms = 1.09%**, far inside the `max(band, 3% = 0.34 ms)` bar. Wash.

### AC-4 — no unintended spill (both bands hold)
- **HBM write 453 MB** = 201 output + 252 intentional `w'` materialization — the AC-4.1 band
  exactly, `≤ 500 MB` allowance → **no write spill**.
- **HBM read 2297 MB** (D2 2112 → 2297, +8.8%) — within the AC-4.2 modeled ~2361 MB band,
  `≤ 2600 MB` → **no re-fetch / read spill**. The fold's extra `w` prologue read + the `w'`
  main stream land below the naive estimate (the fold removes D2's repeated per-`m_hi` `b`
  reloads).

### Mechanism — why the fold is a wash (the plan's thesis, confirmed)
E1's **TRUE PE-active is 10.6548 ms vs D2's 10.7351 ms (−0.75%)** — the fold *does* remove the
low-rank tail's PE work. But the **`a@b` materialization PE is additive on a PE-bound op** (it
cannot hide under the main GEMM's PE), and the **+252 MB `w'` write + +186 MB read** only
partially hide under the DMA idle — so the net wall is only **−1.12%**, a wash. The optimistic
`14.6645 / 10.656 = 1.376x` figure was correctly demoted in planning to an *unreachable*
theoretical bound (it assumes zero materialization cost). **The extra HBM round-trip does not
pay for the tail removal**; D2's PSUM-fused fp32 low-rank (no HBM round-trip, HBMwr 201 MB
byte-identical) is strictly better. This is the phase-1 prompt's low-rank-fusion hint answered:
fusing into the base PSUM (D2) beats materializing folded weights to HBM (E1).

## E2 / E3 — model-based rejects (not built)

- **E2 (bf16x2 split of the up-projection `(x@a)@b`):** numerically safe (offline route `[B']`
  base+up split 4.438e-6 « gate) but **no PE upside** — the up-proj is 768 / 97536 = 0.79% of
  matmul instructions; splitting cuts ~0.13% of total PE, an order below the ~1.3% measurement
  noise, while adding `tT`/`b` limb builds. **Moot under the fold** (E1 removes the up-proj
  entirely). Recorded; no spend.
- **E3 (double-buffer the lhs limbs to close D2's 5.0% / 0.569 ms PE-idle bubble):** the
  double-buffered resident set — 2× lhs limbs (B=4) 160 KB + `a_local` 20 KB + `tT` 2 KB +
  per-chunk `w` transients ~12 KB = **~194 KB/partition > the 192 KB trn2 SBUF limit** before
  compiler temporaries → predicted **hard spill** (the `tmm_v7_dbuf_rhs` read-floor-break
  signature). The identical double-buffer lever was **BUILT and measured-rejected on two
  siblings** (`tmm_v7_dbuf_rhs` engaged its prefetch overlap but broke the AC-4 read floor
  229→283 MB + a write-spill; `bmm` phase-3 found cross-block blocking a monotone anti-lever).
  Ceiling here is only the 5% idle gap. Pre-gated on SBUF (AC-4.3) → not built.

## Settled regimes (no phase-3 action)
- **N_CHUNK = 512** — one fp32 PSUM bank (max moving-free width); 1024 illegal; smaller raises
  the matmul-site count.
- **M-block B = 4** — the natural arithmetic-free `m_lo` block; sibling matmul D4 sweep B=2
  0.983x, B=8 0.968x, B=16 0.519x.
- **Edge tiles: none** — all dimensions are exact tile multiples (edge-free, no ragged regime).

## AC-5 — honest seed caveat
All 5 on-device profiler "seeds" `[0,21,42,63,84]` draw the **same seed-42 input** (the
adapter reseeds `np.random.seed(42)` before every draw), so the on-device 5-seed PASS is one
distinct input, not five. The true distinct-input numeric margin is the **offline diversity
worst** on seeds `[0,21,63,84]`: **4.456e-6** for the fold (E1), **3.934e-7** for D2's base-only
composite. Both the on-device worst and the offline diversity worst are reported above and in
`candidates.jsonl`; they are never conflated.

## AC-6 — Evidence
- `benchmark.csv`: `lora_v4_fold` measured row + a `lora_v3_bf16_split` same-session A/B re-confirm control row.
- `candidates.jsonl`: `lora_v4_fold` measured-reject node (parent `lora_v3_bf16_split`);
  `E2-upproj-split-model-reject`, `E3-dbuf-lhs-limbs-model-reject` model-reject nodes with
  numeric basis (parent `lora_v3_bf16_split`). DAG:
  `lora_v1 → lora_v2_mblk4 → lora_v3_bf16_split → {lora_v4_fold, E2, E3, D3, D4}`.
- `profile/`: `lora_phase3_offline_fold_gate.txt` (AC-2 gate), `lora_v4_fold_fast_screen.txt`
  (screen), `lora_phase3_ab_bracket.txt` (AC-3 A/B), `lora_v4_fold_digest.txt` (E1 + D2
  same-session `dump_metrics`: TRUE PE-active, per-seed rel-L2, HBM counters).
- `runs/offline_lora_bf16_split_sim.py` (extended with `composite_fold_bf16x2` + `fold_fp32_control`),
  `runs/_layout_check.py` (extended with the fp32 fold identity + sign-flipped negative control),
  `runs/lora_v4_fold.py` (the built kernel), all run clean.

## Bottom line
lora is PE-bound at the base-GEMM systolic floor with an edge-free shape and the dominant lever
(bf16x2 3-product count) at its proven numeric floor. The prompt's intended low-rank fusion is
already realized in D2 without an HBM round-trip. The one measured build (the canonical LoRA
weight-fold) definitively confirms that materializing folded weights to HBM does not beat the
PSUM-fused fp32 tail: a wash (1.09%, inside the 3% bar). **FINALIZE `lora_v3_bf16_split` at
1.297x; keep `lora_v2_mblk4` at 0.988x as the fp32 fallback.**
