# matmul_add_rmsnorm — Phase 1 draft (first correct NKI kernel)

## 1. Operator and contract

**Op:** `matmul_add_rmsnorm`, NKIBench case `1`. Fused dense GEMM → residual-add → RMSNorm.

**Reference** (`AccelOpt/NKIBench/reference/matmul_add_rmsnorm_M4096_N2048_K2048_numpy_1.py`):

```python
def forward(x, w, eps, z, g):
    y   = np.matmul(x, w) + z                               # GEMM then residual add
    rms = np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True) + eps)  # row RMS over N (axis=-1)
    return y * g / rms                                      # per-col g, per-row 1/rms
```

**Shapes / dtype (all fp32):**
- `x`: `(M=4096, K=2048)`
- `w`: `(K=2048, N=2048)`
- `z`: `(M=4096, N=2048)`   — residual, added to the GEMM output
- `g`: `(N=2048,)`          — learned scale along the **output** dim N
- `eps`: python float scalar (1e-5)
- output: `(M=4096, N=2048)`

**Signature (matches baseline):** `def kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor)`
(eps is the 3rd positional arg, a runtime python scalar).

**I/O layout — RAW 2D, NOT pre-tiled.** This case's `transform_to_nki_inputs` is the
IDENTITY (`return inputs`), so the kernel receives the raw 2D tensors above and returns a
raw 2D `(4096, 2048)`; the harness's `transform_nki_outputs` reshapes to the ref shape
(already 2D → identity). The kernel slice-tiles itself (`x_tensor[i*128 + ix, ik]`), exactly
like the NKIBench baseline and the sibling `add_rmsnorm_matmul` (which is also raw-2D I/O).

**Correctness gate:** relative-L2 `||v_k - v_r||_2 < 2e-5 * ||v_r||_2`, fp32, across seeds
`[0,21,42,63,84]` (`verify.py` gates on `l2_norm_passed`). Phase 1 stays **pure fp32** —
no precision tricks; correctness first.

**Score:** `baseline_latency / candidate_latency`, p50 on-device, single core,
`--disable-dge --logical-nc-config=1`. Baseline latency = **3.768493 ms** (baselines.json).

## 2. This op is the MIRROR of the rmsnorm siblings (matmul-first)

The two prior fused ops — `rmsnorm_matmul` and `add_rmsnorm_matmul` — do **norm → GEMM**.
This op does **GEMM → add → norm**, the reverse. That single reordering changes three things,
all of which make this op *structurally simpler* than the siblings:

| aspect | siblings (norm→GEMM) | this op (GEMM→add→norm) |
|---|---|---|
| reduction axis of the norm | over **K** (contraction) | over **N** (the GEMM **output** free axis) |
| does the norm need a transpose? | norm result is on the m-partition, matmul needs k-partition → **transpose the activation** | matmul output is already `[m_in(par), n(free)]`; norm reduces the **free** axis → **no norm transpose** |
| `g` placement | length **K** = contraction axis → does NOT commute past the matmul, must fold into w or apply on activation | length **N** = free axis of the output → trivial `[1,N]→[128,N]` broadcast multiply, exactly like the baseline |
| residual `z` | `x+z` **before** norm, shape `(M,K)` | `+z` **after** GEMM, shape `(M,N)`, added to the matmul result |

So the *only* transpose required is `x → xT` for the Tensor Engine (contraction-on-partition),
identical to the promoted `matmul_v1` / `add_rmsnorm_matmul_v1` idiom. The norm is a clean
free-axis reduce over the natural matmul output layout — the *easy* direction. This also means
the fully-fused SBUF structure the baseline already uses is the right shape; Phase 1's job is
to keep it and **kill the weight reload**.

## 3. Why the baseline is slow — the dominant Phase-1 win

The NKIBench baseline (`kernels/matmul_add_rmsnorm_M4096_N2048_K2048_0.py`) loads **all of w
inside the M-loop**:

```python
for i in range(M//128):            # 32 M-tiles
    for n in range(N//512):        # 4 N-chunks
        for k in range(K//128):    # 16 K-tiles
            w_tile = nl.load(w_tensor[k*128:(k+1)*128, n*512:(n+1)*512])  # reloaded 32*4*16 = 2048x
            res_psum += nl.matmul(x_tiles[:, k*128:...], w_tile)
```

That is **2048 weight loads** streaming the full 16 MB weight matrix **32 times ≈ 537 MB** of
redundant HBM reads. That reload traffic (not the compute) is why the baseline is 3.768 ms —
and it is the same pathology the siblings had (`add_rmsnorm_matmul`: baseline 1.859 ms →
w-resident v1 0.495 ms = **3.754x** from the reload fix alone).

**w is 16 MB = 128 KB/partition** (16 K-tiles × 2048 × 4 B). SBUF budget is ~192 KB/partition,
and the per-M-tile working set is ~40 KB/partition (x, xT, the assembled matmul output, z, sq —
each ≤ `[128,2048]` fp32 = 8 KB/part). **128 + 40 = ~168 KB < 192 KB → w fully resident is
feasible** (tighter than the siblings, whose K=1024 gave a 64 KB weight; here K=2048 doubles it,
so budget headroom is only ~24 KB — flag as a risk, see §7). Loading w **once** and reusing it
across all 32 M-tiles is the single biggest, lowest-risk Phase-1 win, and matches the proven
sibling structure.

## 4. Phase-1 kernel design (v1, fp32, w-resident, explicit nc_matmul)

Use the **explicit `nisa.nc_matmul` + identity-matmul transpose** idiom (proven on this remote
by `matmul_v1`, `rmsnorm_matmul_v1`, `add_rmsnorm_matmul_v1`), NOT the baseline's high-level
`nl.matmul`. Reason: the known Phase-2 win across every sibling is the **compensated bf16x2
split-matmul**, which requires operating on the transposed limbs directly through `nc_matmul`;
starting from the explicit idiom makes Phase 2 a clean extension rather than a rewrite.

**Constants:** `M_TILES=32`, `K_TILES=16` (2048/128), `N=2048`, `N_CHUNK=512` (one fp32 PSUM
bank), `N_CHUNKS=4`, `INV_N = 1/N`.

**Tensor-engine mapping.** `nc_matmul(stationary, moving) = stationary.T @ moving`, contraction
(k_in) on the **partition** axis of both, both in SBUF. We want `out[m,n] = sum_k x[m,k]·w[k,n]`:
- `w` is `[k(par), n(free)]` in HBM already → load directly as the **moving** operand.
- `x` is `[m(par), k(free)]` → k is on the free axis → transpose each `[128,128]` K-sub-tile to
  `xT[kt] = [k_in(par), m_in(free)=128]` via the identity `nc_matmul(is_transpose=True,
  is_moving_onezero=True)` idiom, then use as the **stationary** operand.
- product → `[m_in(par), n(free)]` — the natural layout for the row-wise (over-N) norm.

**Preamble (once):**
1. `bias_zero = zeros([par_dim(128),1])` for Scalar-Engine activations (square, rsqrt) — a
   `[128,1]` bias is portable across NeuronCore generations (scalar bias needs v3+).
2. `g_tile = nl.load(g_tensor.reshape((1,N)))` → `[1,N]`, broadcast `[1,N]→[128,N]` at use
   (free-axis, exactly the baseline's `g_bcast`).
3. `identity_local[128,128]` from `nl.shared_constant(np.identity(128))` — the moving operand
   for the transpose, loaded once.
4. **w fully resident:** `w_sb[kt] = [par_dim(128), N]` for kt in 0..15, `nl.load` once
   (16 MB total, 128 KB/part).

**Per-M-tile loop (`for mt in nl.affine_range(32)`):**
1. Load `x_sb = x_tensor[mt*128+ix, ik]` → `[128, K=2048]`.
2. Transpose the 16 K-sub-tiles: `xT[kt] = [k_in, m_in=128]` via identity nc_matmul → PSUM →
   copy to SBUF (mirrors `matmul_v1`).
3. **GEMM, assemble the full row into SBUF** (RMSNorm needs the whole N-row before reducing):
   `rmsnorm_in = [128, N]`; for each of 4 N-chunks `c`: `acc = zeros([128,512], psum)`; for
   each of 16 K-tiles: `acc += nc_matmul(xT[kt], w_sb[kt, :, c*512:...])`; then
   `rmsnorm_in[:, c*512:...] = copy(acc)`.
4. **Residual add:** `y = rmsnorm_in + z_tile`, where `z_tile = z_tensor[mt*128+ix, iy]`
   (`[128,N]`, free-axis `tensor_tensor` add — matches reference `y = matmul + z`).
5. **Fused RMSNorm over N (free axis), entirely in SBUF:**
   - `sq = activation(op=square, data=y, bias=bias_zero)` → `[128,N]`
   - `sumsq = tensor_reduce(add, sq, axis=[1])` → `[128,1]` (single full-N free-axis reduce)
   - `mean_eps = tensor_scalar(sumsq, mul, INV_N, add, eps)` → `sum/N + eps` (eps added AFTER
     the mean, NOT scaled by 1/N — matches `np.mean(...) + eps`)
   - `inv_rms = activation(op=rsqrt, data=mean_eps, bias=bias_zero)` → `[128,1] = 1/rms`
6. **Output scale:** `out = y * g * inv_rms`. Apply as `tmp = y * inv_rms` (per-row `[128,1]`
   `tensor_scalar`) then `out_sb = tmp * g_bcast` (per-col `[1,N]→[128,N]` `nl.multiply`), or
   the equivalent baseline order (`y * inv_rms` then `* g_bcast`). Both reproduce
   `y * g / rms` (associativity of scalar multiplies; validated by the fp32 control below).
7. `nl.store(out[mt*128+ix, iy], out_sb)`.

Return the `(M,N)` `nl.shared_hbm` output.

## 5. Correctness notes (must-match details)

- **eps placement:** `mean(y²) + eps`, eps added *after* the `/N` mean (baseline does
  `mean = square_sum / N; mean = mean + eps`). Do NOT fold eps into the `1/N` scale.
- **Reduce axis:** the norm is over **N** (`axis=-1`), which is the free axis of the assembled
  `[128,N]` tile → `tensor_reduce(axis=[1])`. (Contrast the siblings, which reduced over K.)
- **g is length N** (free/output axis) → `[1,N]` broadcast, NEVER folded into w (it does not sit
  on the contraction axis here). This is *simpler* than the sibling `add_rmsnorm_matmul`, whose
  g was length-K and needed a fold.
- **fp32 throughout** — matmul in fp32 PSUM, norm in fp32. The 2e-5 rel-L2 gate is tight even
  for pure fp32 (siblings measured on-device fp32 rel-L2 ~1.46e-5, only ~1.37x under the gate,
  because trn2 emulates fp32 matmul in multiple bf16 passes). Phase 1 must not add any extra
  precision loss; keep every intermediate fp32.
- **Input draw order** (for later offline sims, not needed to code v1): `get_inputs` draws
  `x → w → eps → z → g`; eps is a non-random 1e-5.

## 6. SBUF budget (per partition, 128 partitions)

| buffer | shape/part | KB/part |
|---|---|---|
| w resident (16 K-tiles) | 16 × 2048 fp32 | 128.0 |
| x_sb | 2048 fp32 | 8.0 |
| xT (16 × [.,128]) | 16 × 128 fp32 | 8.0 |
| rmsnorm_in / y / sq (reused) | 2048 fp32 each | ~8–24 |
| z_tile | 2048 fp32 | 8.0 |
| g_tile, identity, [128,1] scalars | small | <2 |
| **total** | | **~168–184 KB** |

Under the ~192 KB/part budget, but with less headroom than the siblings (their w was 64 KB).
If it does not fit / spills, the fallback is to **reuse buffers aggressively** (compute `sq`
in place over `y`, don't keep both `rmsnorm_in` and `y`) before considering partial-w or
M-blocking. Correctness does not depend on w being fully resident — it's a perf choice — so a
spill would only cost speed, not correctness.

## 7. What Phase 1 does NOT do (defer)

- **No bf16x2 split** (the sibling Phase-2/3 win, +28% there). Phase 1 is pure fp32.
- **No g-into-w fold / post-scale eviction refactor** — g is free-axis here so the fold is a
  no-op; the inv_rms/g eviction-fold micro-opts are a Phase-2 concern.
- **No off-PE transpose exploration** (dma_transpose fp32-ineligible, nc_transpose(vector)
  regressed — both CLOSED in sibling phase-2; do not re-explore).
- **No M-blocking / loop reorder** — Phase-2/3 levers, only if the profiler shows an exposed
  fill or DMA bubble.

## 8. Risks / open questions

- **SBUF headroom (~24 KB):** w-resident is feasible but tight (§6). If the compiler spills,
  reuse the `y`/`sq`/`rmsnorm_in` buffers in place; correctness is unaffected.
- **fp32 rel-L2 margin is thin** (~1.37x under gate on the siblings). If v1 fails the gate,
  invoke `kernel-accuracy-debugging`; the likely culprit would be an eps/mean-order or
  reduce-axis mistake, not precision (pure fp32 should pass as it did on all siblings).
- **512 identity transposes** (16/M-tile × 32) live on the PE alongside the matmul. On the
  siblings these were fully hidden under the PE-bound matmul; expected here too, but confirm
  in the Phase-2 profiler digest rather than assuming.

## 9. Deliverable & verification

- Kernel: `runs/matmul_add_rmsnorm_v1.py`, single `@nki.jit def kernel(x_tensor, w_tensor, eps,
  z_tensor, g_tensor)`, structure per §4.
- Score (from `workspaces/matmul_add_rmsnorm/`):
  ```bash
  python3 \
      ../../verify.py --op matmul_add_rmsnorm --candidate runs/matmul_add_rmsnorm_v1.py --fast
  ```
  then full 5-seed (drop `--fast`) before recording.
- Record the perf change in `benchmark.csv`, the candidate (with parent link) in
  `candidates.jsonl`, and the profiler digest under `profile/`.
- **Phase-1 success = full-5-seed PASS.** Expected speedup: large (killing 537 MB of reload
  traffic), plausibly in the ~3–4x range by analogy to `add_rmsnorm_matmul_v1` (3.754x) scaled
  for this op's 2× matmul work (K=2048 vs 1024), but Phase 1 is graded on correctness, not the
  exact number.

See sibling evidence: `workspaces/add_rmsnorm_matmul/` (raw-2D I/O + fused-norm template),
`workspaces/matmul/runs/matmul_v1.py` (nc_matmul transpose idiom), memory
`kda-add-rmsnorm-matmul-progress`, `kda-rmsnorm-matmul-progress`, `kda-matmul-progress`.
