# matmul (M4096 N12288 K5120, fp32) — Phase 1 implementation draft

## Goal

Produce the **first correct** NKI kernel for the dense GEMM `out = lhs @ rhs`
(M=4096, K=5120, N=12288, fp32), passing NKIBench's relative-L2 gate
(`||v_k - v_r|| < 2e-5 * ||v_r||`) on all five seeds `[0,21,42,63,84]`. Prefer a
clean, understood, correct kernel over speed; leave aggressive tuning to
phase 2/3. But choose a loop structure that is already reasonable for a
compute-bound op so we don't start from a pathological baseline.

## What the operator is

Plain dense matmul. No fusion, no epilogue. The only subtlety is the **tiled
layout** the harness hands us and the **transpose** the Tensor Engine forces.

## Tiled layout (from the numpy reference)

`transform_to_nki_inputs` reshapes the natural inputs (row-major):

- `lhs (4096, 5120)` -> `v1 (32, 128, 40, 128)` = `[m_tile, m_in, k_tile, k_in]`
  - `v1[mt, mi, kt, ki] == lhs[mt*128 + mi, kt*128 + ki]`  (verified in numpy)
- `rhs (5120, 12288)` -> `v2 (40, 128, 12288)` = `[k_tile, k_in, n]`
  - `v2[kt, ki, n] == rhs[kt*128 + ki, n]`  (verified in numpy)

Output `v3 (32, 128, 12288)` = `[m_tile, m_in, n]`, later reshaped to
`(4096, 12288)` by `transform_nki_outputs`, so
`v3[mt, mi, n] == out[mt*128 + mi, n]`.

Dimensions in tiles: `M_TILES=32`, `K_TILES=40`, `N=12288`. All the 128s are
exact (4096/128=32, 5120/128=40), and 12288 = 24 * 512, so N tiles evenly into
512-wide PSUM-bank chunks. **No masking/remainders needed** — every tile is full.

## Hardware constraints that shape the kernel

From `kernel-cost-analysis` grounding + `nki-api-reference`:

- `nisa.nc_matmul(stationary, moving)` computes `stationary.T @ moving`. The
  **contraction dim must be on the partition axis of BOTH operands** (<=128).
  - `stationary` free dim -> output **partition** dim (our M, <=128).
  - `moving` free dim -> output **free** dim (our N, <=512 for fp32 PSUM bank).
- PSUM: dst must be fp32 on trn2; a single bank holds **512 fp32** in the free
  dim. So accumulate into `[128, 512]` PSUM tiles.
- K>128 handled by looping K tiles with `accumulate` into the same PSUM tile
  (accumulation group), then one copy PSUM->SBUF per output tile.
- **fp32 is mandatory.** rel-L2 2e-5 is far tighter than bf16/tf32 round-off, so
  we cannot downcast the matmul inputs for throughput. The compute floor is the
  fp32 PE-array rate; MFU is what we optimize, not precision.

### The transpose problem

`nc_matmul` needs K on partitions for both operands.

- `rhs` tile `v2[kt, :, n0:n0+512]` is `[k_in=128 (par), 512 (free)]` — K already
  on the partition axis. **Use directly as the `moving` operand.** ✓
- `lhs` tile `v1[mt, :, kt, :]` is `[m_in=128 (par), k_in=128 (free)]` — K is on
  the **free** axis. We must transpose it to `[k_in (par), m_in (free)]` before
  it can be the `stationary` operand.

Transpose idiom (same one the NKIBench baseline and the profiler's canonical
`examples/matmul_kernel.py` use): load a `[128,128]` identity into SBUF once,
then `nisa.nc_matmul(stationary=lhs_tile, moving=identity, is_transpose=True,
is_moving_onezero=True)` writes the transposed tile `[k_in, m_in]` into PSUM;
copy it to SBUF. After transpose, `stationary = lhsT[kt] = [k_in(par), m_in(free)]`,
`moving = rhs[kt] = [k_in(par), n(free)]`, so
`stationary.T @ moving = lhs_tile @ rhs_tile = [m_in, n]`. Correct. ✓

## Loop structure (Phase-1 choice)

Two candidate orders, and why I pick M-outer:

- **N-outer** (rhs stays resident, re-transpose lhs per N-block): would
  re-run the lhs transpose 24x (once per 512-wide N chunk). For a compute-bound
  GEMM that is pure PE waste (transpose runs on the same Tensor Engine) — bad.
- **M-outer** (transpose each M-block's lhs once, stream all of N): the lhs
  transpose runs exactly once per (m_tile, k_tile). Transpose cost is
  32*40 = 1280 `nc_matmul`s of `[128,128]`; the productive matmul is
  32*40*24 = 30720 `nc_matmul`s of `[128 x 512]`. Transpose overhead is a few %
  of PE time. rhs gets re-read from HBM per M-tile (32x, ~8 GB), but this op is
  compute-bound so HBM reload is not the gate in phase 1. **Pick M-outer.**

Chosen structure (simple, one M-tile at a time — minimal SBUF, easy to verify):

```
identity[128,128] <- load once

for mt in range(32):                      # 32 M-tiles (output rows)
    # transpose this M-tile's lhs: all 40 K-tiles -> lhsT[k_tile][k_in, m_in]
    for kt in range(40):
        lhs_tile = v1[mt, :, kt, :]        # [m_in(par)=128, k_in(free)=128]
        psumT = nc_matmul(lhs_tile, identity, is_transpose=True,
                          is_moving_onezero=True)   # -> [k_in, m_in] in PSUM
        lhsT[kt] = copy(psumT)             # SBUF [k_in(par)=128, m_in(free)=128]

    for n0 in range(0, 12288, 512):        # 24 N-chunks of 512
        acc = psum_zeros([128, 512])       # output tile [m_in, n_chunk]
        for kt in range(40):               # accumulate over K
            rhs_tile = v2[kt, :, n0:n0+512]        # [k_in(par)=128, 512(free)]
            acc += nc_matmul(lhsT[kt], rhs_tile)   # [m_in, 512], accumulate
        out_sb = copy(acc)                 # PSUM -> SBUF
        store v3[mt, :, n0:n0+512] = out_sb
```

SBUF residency per M-tile: `lhsT` = 40 * [128 x 128] fp32 = 40 * 64 KB-tile
= 2.56 MB total, i.e. 40*128*4 = 20 KB per partition (192 KB budget) — fits with
lots of headroom. rhs tiles are streamed (loaded per (n0, kt)); a natural
phase-2 improvement is to block N so a loaded rhs tile feeds several things, but
phase 1 keeps it simple and correct.

## Correctness reasoning

- Every dim divides evenly (32*128=4096, 40*128=5120, 24*512=12288) → no
  masks, no partial tiles.
- The transpose is exact in fp32 (identity matmul, `is_moving_onezero` just a
  perf hint, not a numeric change). Internal matmul accumulation is fp32.
- K accumulation order: summing 40 K-tiles in fp32 in PSUM. Relative-L2 over the
  whole tensor is tolerant to fp32 summation order at 2e-5; the reference is
  also fp32 `np.matmul`. Expect comfortable pass.

## Risks / things to watch

- **Index/layout bug** producing a transposed or mis-tiled output — the most
  likely failure. Mitigate by mirroring the baseline's proven indexing exactly
  where possible and reasoning tile-by-tile.
- PSUM over-allocation if I keep too many `[128,512]` banks live — keep a single
  `acc` per N-chunk (8 banks total; one 512-tile = 1 bank).
- SBUF overflow if lhsT residency is mis-sized — 20 KB/partition is safe.

## Validation plan

1. `--fast` (seed 42, few iters) first for a quick correctness+latency read:
   `python3 \
       ../../verify.py --op matmul --candidate runs/<file>.py --fast`
2. On PASS, full 5-seed run (drop `--fast`) before recording as promoted.
3. Record perf row in `benchmark.csv`, candidate node in `candidates.jsonl`
   (parent = baseline), and the profiler metric digest under `profile/`.

## Phase-1 success criterion

L2 gate passes on all five seeds. Speedup is secondary this phase, but the
M-outer structure should already be in the ballpark of the baseline rather than
pathologically slow. Any speedup >= ~1.0x with a correct result is an acceptable
phase-1 exit; real MFU work is phase 2.
