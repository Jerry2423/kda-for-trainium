# bmm_softmax Phase 1 — Exit Decision (promotion evidence)

## Result

`runs/bmm_softmax_v1.py` — first correct fp32 fused NKI kernel for `bmm_softmax`
(B16 M4096 K64 N4096: batched matmul + row-softmax over N). **PROMOTED** as the
phase-1 DAG root (no phase-1 performance floor; correctness-first).

- **Correctness (AC-2, sole gate):** full 5-seed `verify.py` PASS — `l2_norm_passed=True`
  on all seeds `[0,21,42,63,84]`. On-device rel-L2 = **2.5683e-6** on every seed
  (~7.8x under the `2e-5` gate). `--fast` pre-check passed first.
- **Latency (evidence only):** p50 = **4.5995ms**, **1.585x** over the 7.290ms baseline.
- **Softmax path:** `full_row_4096` (the PRIMARY 4096-wide fused softmax). It traced,
  compiled, and ran directly — the AC-2.1 correctness-gated fallbacks (chunked
  512+`loop_reduce` softmax; 4D `(16,32,128,4096)` output; per-chunk/1024-wide rhs loads)
  were **not** needed.

## Structure (what the kernel does)

Batch-outer, mirrors the solved sibling `bmm_v1` through the matmul, then a fused softmax
epilogue replaces the plain store:

1. Load a 128x128 identity into SBUF once (transpose moving operand).
2. Per batch `b`: load `rhs[b]` resident as `[k=64(par), n=4096(free)]` (16 KB/part).
3. Per m-tile (32 = 4096/128): load lhs `[m=128(par), k=64(free)]`, identity-transpose
   once via `nc_matmul(is_transpose=True, is_moving_onezero=True)` → PSUM `[64,128]`, copy
   to SBUF; then 8 n-chunks of 512: single-pass K=64 `nc_matmul` → `[128,512]` PSUM, copy
   into a resident `score[128,4096]` SBUF tile.
4. Full-row fused softmax over the N free axis, entirely in SBUF:
   `row_max = tensor_reduce(nl.max, score, axis=[1])` →
   `neg_max = tensor_scalar(row_max, *-1)` →
   `exp_t = activation(nl.exp, score, bias=neg_max[128,1])` →
   `row_sum = tensor_reduce(nl.add, exp_t, axis=[1])` →
   `recip = reciprocal(row_sum)` →
   `out_t = tensor_scalar(exp_t, *recip[128,1])`.
5. One 4096-wide store into a direct 3D output `(16,4096,4096)`.

fp32 throughout; no bf16/tf32; no K-accumulation loop; **no softmax reduce/activation/
elementwise op on a PSUM tile** (PSUM banks only hold matmul/transpose, copied out
immediately).

## Profiler digest (full 5-seed, `runs/dump_metrics.py`)

| metric | value | note |
|---|---|---|
| p50 latency | 4.5997 ms | 1.585x over 7.290ms baseline |
| MFU | 9.5% | bf16-peak denominator; fp32 caps MFU low by construction |
| PE (tensor) active | 79.19% | TRUE PE-active/inf = **3.6424 ms** |
| Vec active | 59.37% | vs pure bmm 20% — the new softmax vector work |
| Scl active | 59.00% | vs pure bmm 14% — exp/activation on the Scalar engine |
| DMA active | 30.62% | TRUE DMA-active/inf = 1.4083 ms |
| HBM read | 33.6 MB | == lhs+rhs 33.55MB (read-once, rhs once/batch) |
| HBM write | 1073.7 MB | == output B*M*N*4 = 1073.74MB (write-once) |
| matmul_instruction_count | 8704 | **identical to pure bmm_v1** (512 transpose + 8192 main) |
| psum_read_sbuf_write_count | 4608 | |
| spill_save/reload_bytes | absent (=0) | **NO spill** |

## Analysis (phase-2 seeds)

- **The matmul cost is unchanged by fusion.** TRUE PE-active/inf 3.6424ms ≈ pure `bmm_v1`
  3.6552ms; matmul_instruction_count 8704 identical. The softmax was fused into the same
  per-m-tile loop, so the Tensor Engine does exactly the bmm work.
- **PE-bound (79%) with a co-limiting materialized softmax stack.** Vec jumped 20→59% and
  Scl 14→59% vs pure bmm — the exp/reduce/reciprocal/scale work. Most of it hides under the
  PE-bound matmul, but the wall (4.5995ms) exceeds TRUE PE-active (3.6424ms) by ~0.96ms — an
  exposed softmax tail. This is the phase-2 surface.
- **Traffic at the read-once/write-once floor; no spill.** HBMwr 1073.7MB == the output
  floor, HBMrd 33.6MB == lhs+rhs once. Fusion keeps the full `[128,4096]` score row on-chip
  (16 KB/part), so scores never touch HBM. This is the mechanism behind the 1.585x over the
  baseline. **Baseline-spill diagnosis CONFIRMED at the traffic level (task7, Codex-concurred):**
  the read-only baseline profiles at HBMrd 700.5MB / HBMwr 1740.6MB — an excess of ~667MB read
  + ~667MB write = ~1.33GB score-like round-trip vs the floor. Refinement: that is ~62% of a
  full fp32 score tensor (1073.74MB), so the correct claim is "baseline spills/rereads a
  substantial score-like intermediate", NOT "spills the entire score matrix". Details in
  `docs/phase2-bottleneck-evidence.md`.

## Phase-2 outlook (NOT implemented in phase 1)

- Two-phase transpose-all schedule (`bmm_v2`, 1.253x on pure bmm) to remove the per-tile
  transpose→matmul serialization.
- `activation` fused row-sum (`reduce_op=nl.add`) to drop one Vector pass from the softmax.
- Pull the exposed ~0.96ms softmax Vec/Scl tail under the PE-bound matmul.

Phase-1 kept the softmax primitives separate for clarity/verifiability; the fusions above
are explicitly phase-2 levers (in the plan's "Cannot use" for phase 1).
