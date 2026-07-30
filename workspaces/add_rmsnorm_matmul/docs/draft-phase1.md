# add_rmsnorm_matmul — Phase 1 draft (first correct NKI kernel)

## 1. Operator and contract

**Op:** `add_rmsnorm_matmul`, NKIBench case `2`. Fused residual-add + RMSNorm + dense GEMM.

**Reference computation** (`AccelOpt/NKIBench/reference/add_rmsnorm_matmul_M4096_N2048_K1024_numpy_1.py`):

```python
def forward(x, w, eps, z, g):
    y = x + z                                  # residual add
    t = np.sum(np.square(y)/K, axis=-1, keepdims=True)   # mean of squares over K
    t = (t + eps)
    y = y / np.sqrt(t)                          # RMSNorm (per-row scale)
    y = y * g                                   # per-K learned scale (g is length K)
    return np.matmul(y, w)                      # dense GEMM
```

**Shapes / dtype (all fp32):**
- `x`, `z`: `(M=4096, K=1024)`
- `w`: `(K=1024, N=2048)`
- `g`: `(K=1024,)`  — learned scale along the contraction dim
- `eps`: python float scalar (1e-5)
- output: `(M=4096, N=2048)`

**Signature (matches baseline):** `def kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor)`.

**I/O layout — RAW 2D, NOT pre-tiled.** This case's `transform_to_nki_inputs` is the
IDENTITY (returns inputs unchanged), so the kernel receives the raw 2D tensors above and
must return a raw 2D `(4096, 2048)`. This differs from the sibling `rmsnorm_matmul`, whose
reference pre-reshaped inputs to 3D `(32,128,1024)` / `(8,128,2048)`. So this kernel slice-
tiles itself (`x_tensor[i*128 + ix, iy]`), exactly like the NKIBench baseline does.

**Correctness gate:** relative-L2 `||v_k - v_r||_2 < 2e-5 * ||v_r||_2`, fp32, across seeds
`[0,21,42,63,84]`. (`verify.py` gates on `l2_norm_passed`.)

**Score:** `baseline_latency / candidate_latency`, p50 on-device, single core,
`--disable-dge --logical-nc-config=1`. Baseline latency = **1.859287 ms** (baselines.json).

## 2. Why the baseline is slow — the dominant Phase-1 win

The NKIBench baseline (`kernels/add_rmsnorm_matmul_M4096_N2048_K1024_0.py`) loads **all of w
inside the M-loop**:

```python
for i in range(32):            # M-tiles
    ...RMSNorm on this M-tile...
    for n in range(4):         # N-chunks
        for k in range(8):     # K-tiles
            w_tile = nl.load(w_tensor[k*128:(k+1)*128, n*512:(n+1)*512])   # <-- reloaded 32*4*8 = 1024x
            res_psum += nl.matmul(rmsnorm_out_tile[:, k*128:...], w_tile)
```

That is **1024 weight loads** streaming the full 8 MB weight matrix **32 times** ≈ **256 MB**
of redundant HBM reads. This is why the baseline latency (1.859 ms) is ~3.7× the sibling
`rmsnorm_matmul` baseline (0.503 ms) despite near-identical arithmetic.

**w is only 8 MB = 64 KB/partition** (SBUF budget ~192 KB/partition). Loading it **fully
resident once** and reusing it across all 32 M-tiles is the single biggest, lowest-risk win,
and it is exactly the structure my promoted sibling `rmsnorm_matmul_v1` used (1.066x there,
where the baseline was already w-efficient). Here the baseline is w-*inefficient*, so the
same structure should recover a large multiple. This is the Phase-1 kernel's core.

## 3. Relationship to the sibling `rmsnorm_matmul` (high-confidence reuse)

This op is `rmsnorm_matmul` plus a residual add and a per-K learned scale. My promoted
sibling kernel `workspaces/rmsnorm_matmul/runs/rmsnorm_matmul_v1.py` is directly adaptable.
Three deltas to fold in:

1. **Residual add** `a = x + z` before the norm. One extra `nl.load(z)` per M-tile and one
   `nl.add` (or fold into the square activation's input). Then everything (`square`, reduce,
   norm, matmul) runs on `a` instead of raw `x`.
2. **`+ eps` inside the rsqrt**: `inv_rms = rsqrt(mean_k(a^2) + eps)`. The baseline does
   `mean = square_sum / K; mean = nl.add(mean, eps); rms_reciprocal = nl.rsqrt(mean)`.
3. **Per-K learned scale `g`**: `y = normalized * g`, where `g` has length K (varies along
   the contraction axis). Handling in §5.

Everything else — w-resident load, per-row fused RMSNorm, identity-matmul transpose of the
normalized activation to put the contraction dim on the partition axis, K-accumulate into a
`[128,512]` fp32 PSUM bank, N in 4 chunks of 512 — carries over verbatim. Off-PE transpose
routes and the fp32-rate ceiling were already fully explored in the sibling's phases 2-3;
Phase 1 here only needs a clean, correct, w-resident kernel.

## 4. Tiling plan (M-outer, w-resident) — the Phase-1 kernel

Constants: `M=4096, K=1024, N=2048`; `M_TILES=32` (4096/128), `K_TILES=8` (1024/128),
`N_CHUNK=512` (one fp32 PSUM bank in the free dim), `N_CHUNKS=4`. All dims divide evenly →
no edge tiles.

Setup (once):
- `out = nl.ndarray((4096, 2048), dtype=fp32, buffer=nl.shared_hbm)` — 2D, matching the
  identity output transform.
- Load `g` once into SBUF as `[1, K]` (`g_tensor.reshape((1,K))`), like the baseline.
- Load identity `[128,128]` const into SBUF once (moving operand for the transpose idiom).
- Load `w` fully resident: `w_sb[kt] = [k_in(par)=128, n=2048]` for `kt in 0..7`
  (8·128·2048·4B = 8 MB = 64 KB/partition), from `w_tensor[kt*128:(kt+1)*128, :]`.

Per M-tile `mt in affine_range(32)`:
1. Load `x_tile`, `z_tile` = `[128, 1024]` from `x_tensor[mt*128 + ix, iy]`,
   `z_tensor[mt*128 + ix, iy]`.
2. `a = nl.add(x_tile, z_tile)`  → `[128,1024]` in SBUF.
3. **Fused RMSNorm over K** on `a`:
   - `sq = activation(op=nl.square, data=a, bias=bias_zero, scale=1.0)` (Scalar Engine, fp32).
   - `sumsq = tensor_reduce(nl.add, sq, axis=[1])` → `[128,1]` (single full-1024 free reduce).
   - `mean = sumsq / K`; `mean = nl.add(mean, eps)`; `inv_rms = nl.rsqrt(mean)` → `[128,1]`.
     (Mirror the baseline's runtime-`eps` path to avoid the v2/v3 scalar-bias portability
     question. Optionally fold `1/K` into the rsqrt `scale` as the sibling did, keeping the
     `+eps` as a separate add on `sumsq` — verify equivalence; keep the simple form if unsure.)
   - `norm = tensor_scalar(a, op0=nl.multiply, operand0=inv_rms)` → `[128,1024]` per-row scale.
4. **Apply g** (per-K scale) — see §5. Phase-1 choice: `g_bcast = g_tile.broadcast_to((128,K))`
   then `y = nl.multiply(norm, g_bcast)` → `[128,1024]` (baseline's approach, free-axis, safe).
5. **Transpose** the 8 normalized `[128,128]` K-sub-tiles of `y` to `yT[kt] = [k_in(par),
   m_in(free)]` via the identity nc_matmul idiom (`nisa.nc_matmul(y[:,kt*128:...],
   identity, is_transpose=True, is_moving_onezero=True)` → PSUM → copy to SBUF). Needed because
   `nc_matmul(stationary, moving) = stationary.T @ moving` requires the contraction dim (k_in)
   on the partition axis of both operands.
6. **Matmul**: for `c in affine_range(4)`: `acc = zeros([128,512], psum)`; for `kt in
   affine_range(8)`: `acc += nc_matmul(stationary=yT[kt] [k_in,m_in], moving=w_sb[kt]
   [k_in, 512c:512c+512])` = `[m_in,512]`. Then `out_sb = copy(acc)` and
   `nl.store(out[mt*128 + ix, c*512 + iz], out_sb)`.

SBUF budget check: w_sb 64 KB/part + a/sq/y/norm (~4 KB each) + yT (8·128·4B=4 KB) +
identity + small vectors ≪ 192 KB/part. Fine.

## 5. Handling `g` — decision for Phase 1 vs. a Phase-2 lever

`out[m,n] = inv_rms[m] · sum_k ( a[m,k] · g[k] · w[k,n] )`.

- `inv_rms[m]` is **per-row** (indexed by m, the output partition) → it commutes with the
  matmul (a scalar per output row). The sibling proved a **post-scale eviction fold** (apply
  it via `tensor_scalar` reading PSUM at eviction) is exact to ~4.8e-7. That's a Phase-2
  micro-lever; **Phase 1 applies it inline** on the normalized activation (simplest, correct).
- `g[k]` is **per-contraction-column** (indexed by k) → it does **NOT** commute past the
  matmul. Two correct placements:
  - **(Phase-1) on the activation**: `y = norm * g_bcast`, `g_bcast` = broadcast of the
    `[1,K]` g-row to `[128,K]` along the partition axis (baseline's approach). Free-axis
    multiply, obviously correct.
  - **(Phase-2 lever) fold g into resident w once**: `w'[k,n] = g[k]·w[k,n]`. Since w_sb tiles
    are `[k_in(par), n]`, g is a per-partition `[128,1]` scale → one `tensor_scalar` per w-tile
    at load time, done 8× total instead of 32× on the activation. Combined with inv_rms
    post-scale eviction, the per-M-tile inner work drops to add + norm + transpose + matmul.
    Defer to Phase 2 (measure first).

**Phase 1 = inline `g_bcast` multiply** for a clean, obviously-correct baseline. Record the
g-into-w + post-scale-eviction fold as the Phase-2 opener.

## 6. Primitives (all proven on this remote in the sibling task)

- `nisa.activation(op=nl.square, ...)` and `nisa.activation(op=nl.rsqrt, ...)` — Scalar Engine,
  fp32; `output = op(data*scale + bias)`. Use a `[128,1]` zero-bias tile for portability.
- `nisa.tensor_reduce(nl.add, ..., axis=[1])` — full-K free-axis reduce → `[128,1]`.
- `nisa.tensor_scalar(data, op0=nl.multiply, operand0=inv_rms[128,1])` — per-row free-axis
  broadcast scale.
- `nl.broadcast_to(g_tile, (128,K))` — `[1,K]`→`[128,K]` partition broadcast for the g multiply
  (or use baseline's `g_tile.broadcast_to((128,K))`).
- `nisa.nc_matmul(..., is_transpose=True, is_moving_onezero=True)` with a `[128,128]` identity
  — transpose idiom (PROVEN; the sibling confirmed dma_transpose is fp32-ineligible and
  nc_transpose(vector) regresses, so the identity-matmul transpose is the right Phase-1 choice).
- `nisa.nc_matmul(stationary, moving)` = `stationary.T @ moving`, fp32 accumulate in PSUM.

All are the exact ops used in the promoted `rmsnorm_matmul_v1`.

## 7. Correctness risks & mitigations

- **eps placement**: must be added AFTER the `/K` mean and BEFORE the sqrt/rsqrt
  (`rsqrt(mean + eps)`). Mirror the baseline's `nl.add(mean, eps)` exactly.
- **g axis**: g is length-K = the contraction axis. On the activation it broadcasts along the
  free (K) axis of `[128,K]` — matches `g_bcast[128,K]`. If ever folded into w it is a
  per-partition `[128,1]` scale (k_in on w's partition axis). Do not confuse with a per-row scale.
- **Draw order / seeds**: the profiler fixes seed 42 for the input draw (adapter note); the
  reference draws `x, w, eps, z, g` in that order in `get_inputs`. The kernel is agnostic to
  draw order (it just consumes the tiled args), so this is only a correctness-of-comparison
  concern the harness already handles.
- **Output layout**: return raw 2D `(4096,2048)`; `transform_nki_outputs` reshapes to ref
  shape (identity here). Store with `out[mt*128 + ix, c*512 + iz]` indexing.
- Validate on `--fast` (seed 42) first, then full 5-seed before recording. Expect PE-bound,
  correct on all seeds (fp32 throughout; no precision shortcuts in Phase 1).

## 8. Deliverable & how it's scored

- Kernel file: `runs/add_rmsnorm_matmul_v1.py`.
- Validate/score (from `workspaces/add_rmsnorm_matmul/`):
  ```bash
  python3 \
      ../../verify.py --op add_rmsnorm_matmul --candidate runs/add_rmsnorm_matmul_v1.py --fast
  # then drop --fast for the full 5-seed / higher-iter measurement before recording.
  ```
- Record the perf change in `benchmark.csv`; add the candidate to `candidates.jsonl` with
  parent `add_rmsnorm_matmul_M4096_N2048_K1024_0.py`; read the profiler digest (PE/MFU/Vec/
  Scl/DMA/HBM) to confirm the bottleneck for Phase 2.

## 9. Expected outcome

First correct fp32 kernel passing all 5 seeds, with a large speedup over the 1.859 ms
baseline driven almost entirely by making w resident (eliminating ~256 MB of redundant weight
reloads). The kernel should land PE-bound (fp32 systolic floor, MFU ~46% like the sibling),
setting up Phase 2 (g-into-w fold + inv_rms post-scale eviction) and the Phase-3 compensated
bf16x2 split (the sibling's +28% surprise win, gated on the 2e-5 L2 tolerance holding under
K-averaging — which it did there at ~4.5e-6).
