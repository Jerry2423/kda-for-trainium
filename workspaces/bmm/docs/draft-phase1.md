# bmm — Phase 1 draft: first correct fp32 NKI kernel

**Operator:** `bmm` (NKIBench case 2). Batched matmul `out[b] = lhs[b] @ rhs[b]`
for `b in 0..15`. Shapes/dtype: `lhs (16,4096,64)`, `rhs (16,64,4096)` fp32 →
`out (16,4096,4096)` fp32. **B=16, M=4096, K=64, N=4096.**

Baseline kernel: `../../AccelOpt/NKIBench/kernels/bmm_B16_M4096_K64_N4096_0.py`
(measured baseline latency **2.550 ms**, from `baselines.json`).
Numpy reference: `../../AccelOpt/NKIBench/reference/bmm_B16_M4096_K64_N4096_numpy_1.py`.

Phase-1 goal: produce the **first CORRECT** NKI kernel that passes the relative-L2
gate (`||v_k − v_r||₂ < 2e-5·||v_r||₂`, fp32) across all five seeds `[0,21,42,63,84]`.
Prioritize a clean, fully-understood kernel over speed.

---

## 1. Input/output layout contract (the one thing that differs from `matmul`)

The reference's `transform_to_nki_inputs` only **reshapes** the inputs — it does
**not** pre-tile them (unlike the dense-`matmul` case, whose inputs arrive already
split into `[m_tile,128,k_tile,128]`). So the kernel consumes **natural batched
layout**:

- `v1 = lhs = (16, 4096, 64) = (B, M, K)` — for batch `b`, `v1[b]` is `[M, K]`.
- `v2 = rhs = (16, 64, 4096) = (B, K, N)` — for batch `b`, `v2[b]` is `[K, N]`.

`transform_nki_outputs` reshapes the kernel result to `ref.shape = (16,4096,4096)`
in row-major order, so the kernel may return **`out = (16, 4096, 4096) = (B, M, N)`**
directly (row-major-contiguous; reshapes to the reference shape trivially). This is
cleaner than the baseline's `(16,32,128,4096)` and reshapes identically.

**Consequence of K=64:** the contraction depth is 64 ≤ 128, so the *entire* K
dimension fits in a single Tensor-Engine pass. **There is no K-accumulation loop**
— each output tile is produced by exactly one `nc_matmul` (no `+=` over K-tiles).
This makes the kernel markedly simpler than `matmul_v1` (which loops 40 K-tiles).

## 2. Tensor-Engine mechanics (how the matmul maps to hardware)

`nisa.nc_matmul(stationary, moving) = stationary.T @ moving`, and the **contraction
dim must be on the PARTITION axis of BOTH operands**. We want
`out[m,n] = Σ_k lhs[m,k]·rhs[k,n]`, contracting over K.

- **moving = rhs tile** `[k(par)=64, n(free)]`. `v2[b]` is `[K,N]` → K is already the
  leading (partition-mappable) axis, so an rhs tile loads **directly** as
  `[k=64(par), n(free)]`. No transpose needed.
- **stationary must be** `[k(par)=64, m_in(free)=128]` so that
  `stationary.T @ moving = [m_in,k] @ [k,n] = [m_in,n]` (output partition = m_in,
  free = n).
  But `v1[b]` is `[M,K]` → a loaded lhs tile is `[m_in=128(par), k=64(free)]`, with
  K on the **free** axis. So the lhs tile must be **transposed** to `[k, m_in]`.

**Transpose idiom (identical to the baseline and `matmul_v1`, proven to compile):**
`nc_matmul(stationary=lhs_sb[m_in=128(par), k=64(free)], moving=identity[128,128],
is_transpose=True)` → `[k=64(par), m_in=128(free)]` in PSUM. This is exactly the
baseline's line-34 pattern (`v7[…128,64]` → `v8[…64,128]`). Copy that PSUM tile to
SBUF to use as the stationary operand.

- **main matmul:** `nc_matmul(stationary=lhs_t[k=64(par), m_in=128(free)],
  moving=rhs_chunk[k=64(par), n=512(free)])` → `[m_in=128(par), n=512(free)]` PSUM
  tile = the output tile. Contraction K=64 on the partition of both. ✓

**Tile sizes.** M is tiled by 128 (partition limit) → 32 M-tiles. N is chunked by
**512** (the proven fp32 moving-free width — one matmul pass; used by both the
baseline `v10` and `matmul_v1`) → 8 N-chunks. K=64 is a single untiled contraction.

## 3. Loop nest (clean, correct, write-efficient-enough for phase 1)

```
identity_local = load 128×128 identity into SBUF        # once, reused for all transposes
for b in affine_range(16):                              # batch
    rhs_sb = load v2[b, 0:64, 0:4096]  -> [64(par), 4096(free)]   # once per batch (1 MB, 16 KB/part)
    for mt in affine_range(32):                         # M-tiles (4096/128)
        lhs_sb = load v1[b, mt*128:+128, 0:64] -> [128(par), 64(free)]
        # transpose lhs tile on the PE:
        lhs_t_psum = nc_matmul(lhs_sb, identity_local, is_transpose=True) -> [64(par),128(free)]
        lhs_t = copy(lhs_t_psum) -> SBUF [64(par),128(free)]
        for c in affine_range(8):                       # N-chunks (4096/512)
            acc = nc_matmul(lhs_t, rhs_sb[:, c*512:+512]) -> PSUM [128(par),512(free)]
            out_sb = copy(acc) -> SBUF [128,512]
            store out[b, mt*128:+128, c*512:+512] = out_sb
return out   # (16,4096,4096)
```

Why load `rhs[b]` **once per batch**: `rhs[b]` is `[64,4096]` = 1 MB (16 KB per
partition on 64 partitions — trivially within trn2's ~208 KB usable/partition).
Loading it once and slicing per N-chunk avoids the 32× reload that streaming it
inside the M-loop would incur. `lhs[b]` (`[4096,64]`) is streamed one 128-row tile
at a time (can't fit 4096 rows in 128 partitions). This keeps HBM **reads** minimal
(lhs 16.8 MB + rhs 16.8 MB, each read once).

## 4. Correctness reasoning

- Every `(b, mt, c)` computes `out[b, mt·128 : mt·128+128, c·512 : c·512+512]`
  exactly `= lhs[b][those 128 rows, :] @ rhs[b][:, those 512 cols]` — a full,
  exact fp32 contraction over all K=64 (single matmul, no partial sums to
  reconcile). Union over `(mt, c)` tiles the full `[4096,4096]` output with no
  overlap or gap (32·128 = 4096, 8·512 = 4096). Union over `b` covers all 16
  batches. ⇒ bit-faithful to `np.matmul(lhs, rhs)` up to fp32 matmul rounding.
- No accumulation-order divergence (K un-split), no bf16 anywhere → error is pure
  single-pass fp32 matmul rounding, far under the 2e-5 relative-L2 gate. (The
  `matmul` sibling passes 5-seed L2 with the *same* fp32 identity-transpose idiom
  and a *deeper* 40-way K accumulation, so K=64 single-pass is strictly safer.)

## 5. Bottleneck framing (for phases 2–3; not acted on in phase 1)

Theoretical floors on trn2 (from `kernel-cost-analysis` cost model):

| Component | Floor |
|---|---|
| PE main matmuls (dst free-elements 4096·32·16 ÷ 2.40 GHz) | 0.874 ms |
| PE lhs transposes (512 × 128 free ÷ 2.40 GHz) | 0.027 ms |
| **PE total** | **~0.90 ms** |
| **HBM output write** (1.074 GB fp32 @ ~781 GB/s) | **~1.375 ms** |

⇒ **This kernel is WRITE-BOUND, not PE-bound** — the opposite regime from the
dense `matmul` sibling (which was PE-bound and won via M-blocking to cut rhs
*reloads*). Here HBM reads are tiny (33 MB total) and rhs reloads are already
eliminated, so **M-blocking is not the lever**. The output is genuinely 1.074 GB of
fp32 and the L2 gate (2e-5) forbids a bf16 output, so **~1.375 ms is a near-hard
ceiling (~1.85× over baseline)**. The phase-2/3 levers will be: (a) **write-DMA
efficiency** — accumulate a full `[128, 4096]` M-tile row in SBUF and issue one
16 KB-contiguous store per M-tile instead of eight 2 KB stores, to fatten burst
size; (b) **overlap** compute/transpose behind the output DMA (ping-pong output
SBUF buffers). Phase 1 does **not** do these — it stores per 512-chunk for maximum
clarity; it just records the measured `HBMwr` / `DMA%` / `PE%` digest so phase 2
starts from evidence.

## 6. Phase-1 validation & bookkeeping

1. Write kernel to `runs/bmm_v1.py` with a single `@nki.jit def kernel(v1, v2)`
   entry point and the module-level + in-function NKI imports (matching the
   baseline / `matmul_v1` convention that is known to trace).
2. Fast gate first (1 seed):
   ```
   python3 \
       ../../verify.py --op bmm --candidate runs/bmm_v1.py --fast
   ```
3. On PASS, run the full 5-seed measurement (drop `--fast`) before recording, and
   capture the printed `metrics:` digest (`MFU/PE/Vec/DMA/HBMrd/HBMwr`) into
   `profile/` as the phase-1 bottleneck evidence (expect write-bound: high `DMA%`,
   `HBMwr ≈ 1074 MB`, modest `PE%`).
4. Append the perf row to `benchmark.csv` and the candidate (parent =
   `bmm_B16_M4096_K64_N4096_0.py`) to `candidates.jsonl` as the DAG root.

## 7. Risks / watch-items

- **3D HBM output store**: storing an SBUF `[128,512]` tile into
  `out[b, m_slice[:,None], n_slice[None,:]]` on a `(16,4096,4096)` tensor is a
  standard strided 2D access (partition stride = 4096 elems, free stride = 1);
  `matmul_v1` does the equivalent on a 3D output. If the tracer rejects the 3D
  form, fall back to the baseline's 4D output shape `(16,32,128,4096)`.
- **`is_transpose` operand shapes**: stationary `[128(par),64(free)]`, identity
  `[128,128]` → output `[64(par),128(free)]`. Mirror the baseline's exact index
  expressions to avoid a shape/`is_moving_onezero` mismatch.
- **PSUM width**: main-matmul output `[128,512]` fp32 = 512 elems/partition, one
  PSUM bank (2048) — safe. Do not widen a single matmul past 512 fp32 free.
