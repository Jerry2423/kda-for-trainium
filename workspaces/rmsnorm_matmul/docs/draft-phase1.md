# rmsnorm_matmul (M4096 N2048 K1024, fp32) — Phase 1 implementation draft

## Goal

Produce the **first correct** NKI kernel for the fused op
`out = rmsnorm(x) @ w`, passing NKIBench's relative-L2 gate
(`||v_k - v_r|| < 2e-5 * ||v_r||`) on all five seeds `[0,21,42,63,84]`, fp32.
Prefer a clean, understood, correct kernel over speed; leave aggressive tuning
to phase 2/3. But choose a loop structure that is already reasonable (loads each
input once, keeps the small `w` resident) so we don't start from a pathological
baseline like the stock NKIBench kernel.

## What the operator is

RMSNorm over the K axis, feeding a dense matmul:

```
inv_rms[m]      = 1 / sqrt( mean_k( x[m,k]^2 ) )      # per-row scalar, reduce over K
normalized[m,k] = x[m,k] * inv_rms[m]
out[m,n]        = sum_k normalized[m,k] * w[k,n]
```

M=4096, K=1024, N=2048, all fp32. The norm is a memory-bound per-row reduction
over K; the matmul is compute-bound. The phase-1 win the prompt points at is to
**fuse the norm into the matmul's input staging** — compute `inv_rms` and scale
each x-row in SBUF, then feed the normalized tile straight into the transpose +
matmul without a round-trip to HBM.

## Tiled layout (from the numpy reference, verified in numpy)

`transform_to_nki_inputs` reshapes the natural (row-major) inputs:

- `x (4096,1024)` -> `v1 (32, 128, 1024)` = `[m_tile, m_in, k]`
  - `v1[mt, mi, k] == x[mt*128 + mi, k]`  (verified)
- `w (1024,2048)` -> `v2 (8, 128, 2048)` = `[k_tile, k_in, n]`
  - `v2[kt, ki, n] == w[kt*128 + ki, n]`  (verified)

Output `v3 (32, 128, 2048)` = `[m_tile, m_in, n]`, reshaped back to `(4096,2048)`
by `transform_nki_outputs`, so `v3[mt, mi, n] == out[mt*128 + mi, n]`.

Dimensions in tiles: `M_TILES=32`, `K_TILES=8`, `N=2048 = 4*512`. All 128s are
exact (4096/128=32, 1024/128=8) and 2048 = 4*512, so N tiles evenly into
512-wide PSUM-bank chunks. **No masking / partial tiles anywhere.**

## Hardware constraints that shape the kernel

From `kernel-cost-analysis` grounding + the baseline's proven idioms:

- `nisa.nc_matmul(stationary, moving)` computes `stationary.T @ moving`. The
  **contraction dim must be on the partition axis of BOTH operands** (<=128).
  - `stationary` free dim -> output **partition** dim (our M, <=128).
  - `moving` free dim -> output **free** dim (our N, <=512 for an fp32 PSUM bank).
- PSUM: dst is fp32 on trn2; a single bank holds **512 fp32** in the free dim.
  So the output tile is `[128, 512]` (one N-chunk) per PSUM bank.
- K>128 handled by looping the 8 K-tiles with `+=` into the same PSUM tile
  (one accumulation group), then a single copy PSUM->SBUF per output tile.
- **fp32 is mandatory.** rel-L2 2e-5 is far tighter than bf16/tf32 round-off, so
  we cannot downcast the matmul operands. The compute floor is the fp32 PE rate;
  the matmul memory (`kda-matmul-progress`) records fp32 as a hard ~3.6x PE
  penalty on trn2 — a structural ceiling we accept in phase 1.

### The transpose problem

`nc_matmul` needs K on the partition axis of both operands.

- `w` tile `v2[kt, :, n0:n0+512]` is `[k_in=128 (par), 512 (free)]` — K already on
  the partition axis. **Use directly as the `moving` operand.** ✓
- `x` tile `v1[mt, :, kt*128:(kt+1)*128]` is `[m_in=128 (par), k_in=128 (free)]` —
  K is on the **free** axis. We must transpose each `[128,128]` sub-tile to
  `[k_in (par), m_in (free)]` before it can be the `stationary` operand.

Transpose idiom (identical to the baseline and the profiler's canonical
`matmul_kernel.py`): load a `[128,128]` identity into SBUF once, then
`nisa.nc_matmul(stationary=x_sub, moving=identity, is_transpose=True,
is_moving_onezero=True)` writes the transposed sub-tile `[k_in, m_in]` into PSUM;
copy it to SBUF. After transpose, `stationary = xT[kt] = [k_in(par), m_in(free)]`,
`moving = w[kt] = [k_in(par), n(free)]`, so
`stationary.T @ moving = x_tile @ w_tile = [m_in, n]` — output partition = m_in,
which matches `v3`'s layout. ✓ (The alternative, w-stationary, would put N on the
output partition axis and force a second transpose — rejected.)

## Fusion choice: input-scaling (normalize before matmul)

Two mathematically-equivalent placements of the `inv_rms` scale, both verified in
numpy to pass the gate on seed 42 (threshold 2e-5):

- **Input-scaling** `(x * inv_rms) @ w`: rel-L2 **1.3e-7**. Scales the x tile
  (K=1024 wide) once per row, before transpose. This is what the prompt asks for
  and what the baseline does.
- **Output-scaling** `(x @ w) * inv_rms`: rel-L2 4.8e-7. Scales the wider output
  (N=2048) after matmul. Larger element count, no staging benefit.

**Pick input-scaling.** `inv_rms[m]` is a per-partition scalar, so the scale is a
single `nisa.tensor_scalar(..., operand0=inv_rms_per_partition)` over the resident
x tile — it fuses into input staging exactly as intended and touches fewer
elements (1024 < 2048).

## Loop structure (Phase-1 choice)

Key structural decision vs the stock baseline: `w` is only **8.4 MB** and fits
fully in SBUF at **64 KB/partition** (budget 192 KB). So load `w` **once** up
front and reuse it across all 32 M-tiles — this removes the baseline's 4x w
reload *and* makes the outer loop a trivial one-M-tile-at-a-time sweep (no
M-blocking needed; w-reuse is automatic). We also load each x tile **once** and
compute the norm from the resident tile, removing the baseline's double x-load.

```
identity[128,128] <- load once
for kt in range(8):                       # load w fully resident, once
    w_sb[kt] = load(v2[kt])               # SBUF [k_in=128(par), 2048(free)]

for mt in range(32):                      # 32 M-tiles (output rows), reuse w_sb
    x_sb   = load(v1[mt])                 # [m_in=128(par), 1024(free)]  (one load)

    # ---- fused RMSNorm over K, entirely in SBUF ----
    sq     = activation(square, x_sb)                 # [128,1024]
    sumsq  = tensor_reduce(add, sq, axis=free)        # [128,1]
    inv_rms= activation(rsqrt, sumsq, scale=1/K)      # [128,1]  = rsqrt(mean sq)
    xn     = tensor_scalar(x_sb, multiply, inv_rms)   # [128,1024]  per-row scale

    # ---- transpose normalized x: 8 K-sub-tiles [128,128] -> xT[kt][k_in,m_in] ----
    for kt in range(8):
        psumT   = nc_matmul(xn[:, kt*128:(kt+1)*128], identity,
                            is_transpose=True, is_moving_onezero=True)  # [k_in,m_in]
        xT[kt]  = copy(psumT)             # SBUF [k_in=128(par), m_in=128(free)]

    # ---- matmul: stream all of N, accumulate over K ----
    for n0 in range(0, 2048, 512):        # 4 N-chunks of 512
        acc = psum_zeros([128, 512])      # output tile [m_in, n_chunk]
        for kt in range(8):               # accumulate over K
            acc += nc_matmul(xT[kt], w_sb[kt][:, n0:n0+512])   # [m_in,512]
        out_sb = copy(acc)                # PSUM -> SBUF
        store v3[mt, :, n0:n0+512] = out_sb
```

SBUF residency: `w_sb` = 8 * [128 x 2048] fp32 = 64 KB/partition; `x_sb` 4 KB,
`xn` 4 KB, `xT` = 8 * [128 x 128] = 4 KB, identity 0.5 KB — well under 192 KB.
PSUM: one `[128,128]` transpose bank + one `[128,512]` acc bank live at a time.

Instruction budget: productive matmuls = 32*8*4 = **1024** of `[128k x 512n]`;
transpose matmuls = 32*8 = **256** of `[128x128]` (~20% extra PE tiles, a known
fp32-transpose tax to revisit in phase 2). Total matmul FLOP = 2*M*N*K = 17.2
GFLOP; the baseline's 0.5026 ms implies ~34 TFLOP/s to beat/match.

## Correctness reasoning

- Every dim divides evenly (32*128=4096, 8*128=1024, 4*512=2048) → no masks, no
  partial tiles.
- The reduction order is numerically irrelevant here: a single free-axis
  `tensor_reduce` over the full 1024-wide tile vs the baseline's 2x512 split
  gives a max rel diff of **0.0** in fp32 (verified). `inv_rms` via
  `rsqrt(mean sq)` matches `x/sqrt(mean sq)` to 9e-8.
- The transpose is exact in fp32 (identity matmul; `is_*_onezero` are perf hints,
  not numeric changes). Matmul accumulation is fp32 in PSUM, same as the numpy
  reference's `np.matmul`.
- End-to-end input-scaled result vs reference: rel-L2 **1.3e-7 << 2e-5**.
  Comfortable pass expected on all seeds.

## Risks / things to watch

- **Index/layout bug** (transposed or mis-tiled output) is the most likely
  failure — mirror the baseline's proven `nl.arange` indexing where possible and
  reason tile-by-tile.
- `inv_rms` must be a **per-partition** operand to `tensor_scalar` (shape
  `[128,1]`), broadcasting across the free axis — matches baseline v9/v11.
- PSUM over-allocation if too many banks stay live — keep one `acc` per N-chunk
  and free the transpose bank before the matmul loop.
- SBUF: 64 KB/partition for resident `w` is safe, but confirm the profiler
  doesn't spill; if it does, fall back to streaming w per M-block (still cheap).

## Validation plan

1. `--fast` (seed 42, few iters) first for a quick correctness+latency read:
   `python3 \
       ../../verify.py --op rmsnorm_matmul --candidate runs/<file>.py --fast`
2. On PASS, full 5-seed run (drop `--fast`) before recording as promoted.
3. Record the perf row in `benchmark.csv`, the candidate node in
   `candidates.jsonl` (parent = baseline), and the profiler metric digest
   (MFU / PE / Vec / DMA / HBM) under `profile/` for phase-2 bottleneck triage.

## Phase-1 success criterion

L2 gate passes on all five seeds. Speedup is secondary this phase, but the
w-resident, load-once structure should already be at least on par with the
baseline (0.5026 ms) rather than pathologically slow — the fused-norm + single
x/w load removes the baseline's redundant HBM traffic. Any correct result with
speedup >= ~1.0x is an acceptable phase-1 exit; MFU/bottleneck work is phase 2.
