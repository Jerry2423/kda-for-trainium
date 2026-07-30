# adamw (M10944 N2048, fp32) — Phase 1 implementation draft

## Goal

Produce the **first correct** NKI kernel for the fused **AdamW optimizer step**
over four `(10944, 2048)` fp32 tensors (`theta, g, m, v`) → one `(10944, 2048)`
output `new_theta`, passing NKIBench's relative-L2 gate
(`||v_k - v_r||_2 < 2e-5 * ||v_r||_2`) on all five seeds `[0,21,42,63,84]`.
Prefer a clean, understood, correct kernel over speed; leave aggressive tuning to
phase 2/3. But choose a loop/compute structure that is already reasonable (minimal
DMA, single-op-per-tile free axis, minimal Vector-engine pressure) so we don't
inherit the baseline's pathological 20-buffer / heavily-fragmented shape.

## What the operator is

Pure **elementwise** update. No reduction, no matmul, no cross-partition traffic.
Every output element depends only on the four co-located input elements. The numpy
reference (`../../AccelOpt/NKIBench/reference/adamw_M10944_N2048_numpy_1.py`):

```
theta_t     = theta - 1e-5 * theta
m_t         = 0.9 * m + 0.1 * g
v_t         = 0.999 * v + 0.001 * g * g
v_hat       = v_t * 1000
new_theta_t = theta_t - 0.01 * m_t / (sqrt(v_hat) + 1e-8)
```

The only real content of the kernel is: **(a) the tiled layout**, **(b) which
fused instruction sequence computes the update in the fewest Vector ops**, and
**(c) keeping the pass close to HBM-bound.**

## Tiled layout — an IDENTITY reshape (verified)

`transform_to_nki_inputs` reshapes each `(10944, 2048)` input to `(10944, 2048)`
— i.e. a **no-op identity reshape**. So the tiled inputs the kernel receives are
exactly the natural 2D arrays, indexed `[row, col]`. This matches the baseline,
whose signature is `def kernel(v1, v2, v3, v4)` with each `vN` a 2D
`(10944, 2048)` HBM tensor. **Input arg order (read off the baseline's `nl.load`
calls):**

| arg | tensor  | reference role |
|-----|---------|----------------|
| v1  | `theta` | parameters     |
| v2  | `g`     | gradient       |
| v3  | `m`     | 1st moment     |
| v4  | `v`     | 2nd moment     |

Output is a fresh `shared_hbm` `(10944, 2048)` fp32 tensor, reshaped back to
`(10944, 2048)` by `transform_nki_outputs` — again identity. Signature therefore
matches the baseline exactly.

Because the op is purely elementwise, layout is irrelevant to *correctness* — we
just apply the same update to every element. No transpose, no cross-partition
work.

### Row-tiling into 128-partition blocks (with a masked tail)

Partition dim ≤ 128, so we tile the row axis (`M = 10944`) into blocks of 128
partitions and keep the full `N = 2048` free axis in one instruction (2048 <
32767 activation free-dim limit, and well within Vector limits — no inner
free-dim loop). `10944 = 128 * 85 + 64`, so we need **86 tiles**, the **last one
partial (64 valid rows)**. This mirrors the baseline's tiling choice.

- **Tail handling:** mask every `nl.load` and the final `nl.store` with the
  row-bound predicate `-128*i0 - arange(128)[:,None] + 10943 >= 0` (exactly the
  baseline's predicate). Compute ops on the padding rows produce garbage but are
  **never stored**, so they need no mask — keeping the compute clean.
- **No-mask alternative (noted for later):** `10944 = 96 * 114` and `96 ≤ 128`,
  so partition-dim `P = 96` gives 114 exact tiles with zero masking. DMA byte
  volume is identical (Formula-E per-stream cost `≈ 245 µs` either way — 128-part
  ×86 vs 96-part ×114 differ by <1%), so this only removes the (nearly free) mask
  predicate at the cost of more loop iterations. Not worth it for phase 1; keep
  the well-understood 128×86 masked shape.

Per-partition SBUF for a `[128, 2048]` fp32 tile = `2048*4 = 8 KB`. Even holding
~10 such tiles live (≈80 KB/partition) sits comfortably under the ~208 KB usable
SBUF, leaving room for the phase-2 double buffer. **No middle-axis tiling needed**
(contrast silu, whose 32×7168 row forced a middle tile).

## Algebraic simplification (fold the constants) — verified in numpy

The reference multiplies then divides by 1000; fold it so the denominator's `sqrt`
sees the pre-scaled value and the numerator's `0.01` collapses into the `m_t`
coefficients. Two identities:

- `v_hat = 1000 * (0.999*v + 0.001*g²) = 999*v + g²`
  → the `*1000` disappears; `0.999*1000 = 999`, `0.001*1000 = 1`.
- `0.01 * m_t = 0.01 * (0.9*m + 0.1*g) = 0.009*m + 0.001*g = 0.001 * (9*m + g)`
- `theta_t = theta - 1e-5*theta = 0.99999 * theta`

Two eps-handling options, **both numerically identical** and both pass:

- **eps outside sqrt** (matches reference exactly): `sqrt(v_hat) + 1e-8`, then
  reciprocal.
- **eps dropped / rsqrt** (chosen): `1/sqrt(v_hat)`. Since `v_hat = 999*v + g² > 0`
  (`v = |normal| ≥ 0`) and the `1e-8` eps is `~1e-8` vs a denominator of O(30),
  dropping it changes the result by ~3e-10 relative — far below the gate.

**Numpy verification** (worst rel-L2 across all 5 seeds, fp32):

| formulation                     | worst rel-L2 | gate 2e-5 |
|---------------------------------|--------------|-----------|
| full simplified (eps outside)   | 3.42e-08     | PASS      |
| rsqrt, no eps                   | 3.42e-08     | PASS      |
| **6-op `scalar_tensor_tensor` chain (chosen)** | **3.43e-08** | **PASS** |

Margin is ~580× under the gate, so the simplification is safe with huge headroom.

## Chosen instruction sequence — 6 ops/tile (2 Scalar + 4 Vector)

Each op processes one `[128, 2048]` tile. Using the fused NKI primitives
(signatures confirmed via `nki-api-reference`):

- `nisa.activation(op, data, bias, scale)` computes `op(scale*data + bias)` on the
  **Scalar** engine.
- `nisa.scalar_tensor_tensor(data, op0, operand0, op1, operand1)` computes
  `(data op0 operand0) op1 operand1` on the **Vector** engine, where `operand0` is
  a **scalar** (its pre-scale is free, ≈ one `tensor_tensor` latency) and
  `operand1` is a **full tile**. This is the workhorse: it folds each constant
  pre-multiply into the tile-tile combine for free.

```
g2   = activation(op=square,  data=g)                              # Scalar : g²
vhat = scalar_tensor_tensor(v,  mult, 999.0,   add,   g2)          # Vector : 999v + g²
rden = activation(op=rsqrt,   data=vhat)                           # Scalar : 1/sqrt(vhat)
mm   = scalar_tensor_tensor(m,  mult, 9.0,     add,   g)           # Vector : 9m + g
term = scalar_tensor_tensor(mm, mult, 0.001,   mult,  rden)        # Vector : (0.001·mm)·rden
out  = scalar_tensor_tensor(theta, mult, 0.99999, subtract, term)  # Vector : 0.99999·theta − term
store(out)
```

`0.001 * (9m + g) = 0.009m + 0.001g = 0.01 * m_t` ✓, and multiplying by `rden =
1/sqrt(v_hat)` gives `0.01*m_t / sqrt(v_hat)` ✓. Final `stt` yields
`0.99999*theta − term` ✓.

Why this split:
- The two `sqrt`/`square` nonlinearities go on the **Scalar** engine
  (`activation`), keeping them **off** the Vector engine.
- The **four inherent tile-tile combines** (`999v+g²`, `9m+g`, `mm·rden`,
  `theta−term`) must live on the Vector engine (`tensor_tensor` family);
  `scalar_tensor_tensor` fuses each one's scalar pre-multiply for free, so **4 is
  the algorithmic minimum** of Vector ops for this dependency graph. Contrast the
  baseline, which spends ~11 Vector/Scalar ops through 20 SBUF buffers.

## Hardware grounding: cost model + bottleneck (trn2)

Per `[128, 2048]` fp32 tile, from `kernel-cost-analysis`
(trn2: Vector 0.96 GHz, Scalar 1.20 GHz, DMA 16×23 GB/s):

- **DMA (Formula E, HBM↔SBUF):** load `[128,2048]` = `8192 B/part · ceil(128/16)/23
  ≈ 2849 ns`. 4 loads + 1 store = `5 · 2849 ≈ 14.2 µs/tile` × 86 = **1.225 ms**
  model DMA-issue floor (= 448 MB / 368 GB/s aggregate). But measured HBM on this
  harness runs ~**781 GB/s** (2× the model's conservative 368) → **~0.574 ms**
  real DMA ceiling.
- **Vector floor:** 4 `scalar_tensor_tensor` (Formula A, cpe=1) `= 4 · 2048·100/96
  ≈ 8533 ns/tile` × 86 = **0.734 ms**.
- **Scalar floor:** 2 `activation` (cpe=1) `= 2 · 2048·100/120 ≈ 3413 ns/tile` × 86
  = **0.294 ms**.
- **Baseline measured = 1.305 ms.**

**Bottleneck read:** by the conservative model, DMA-issue (1.225 ms) dominates and
Vector (0.734 ms) hides under it — so a clean fused single-pass kernel should land
near the baseline immediately and the win is becoming truly DMA-bound. **But** if
real HBM ≈ 781 GB/s, the DMA floor drops to ~0.574 ms and the **Vector floor
(0.734 ms) becomes the true bottleneck**. This is the phase-1 finding that sets up
later phases: **adamw is NOT trivially DMA-bound like silu** — its 4 tile-tile
Vector ops are comparable to the real DMA floor, so reducing/rebalancing Vector
pressure (and guaranteeing DMA/compute overlap) is the real lever. (Caveat: the
model prices `reciprocal` at cpe=26; we avoid it entirely by using `rsqrt` on the
Scalar engine, and the baseline's measured 1.3 ms already shows the model
overstates the true Vector chain cost — trust measured-vs-floor over raw theory.)

Phase-1 target is simply **correctness at ≈ baseline latency (~1.0–1.3x)** with a
clean fused single pass. The overlap / Vector-rebalance / shape-specialization
levers below are explicitly deferred.

## Kernel structure (phase 1)

```python
@nki.jit
def kernel(v1, v2, v3, v4):          # v1=theta, v2=g, v3=m, v4=v
    P, N, T = 128, 2048, 86
    out_hbm = nl.ndarray((10944, 2048), dtype=fp32, buffer=nl.shared_hbm)
    for i0 in nl.affine_range(T):    # affine_range → compiler pipelines DMA w/ compute
        rows = 128*i0 + arange(128)[:, None]
        m_pred = (-128*i0 - arange(128)[:, None] + 10943 >= 0)   # tail mask
        # 4 masked loads [128,2048] HBM→SBUF
        theta = load(v1[rows, arange(2048)], mask=m_pred)
        g     = load(v2[rows, arange(2048)], mask=m_pred)
        m     = load(v3[rows, arange(2048)], mask=m_pred)
        v     = load(v4[rows, arange(2048)], mask=m_pred)
        # 6-op fused chain (unmasked — padding rows never stored)
        g2   = activation(op=square, data=g)
        vhat = scalar_tensor_tensor(g2? ...)      # see sequence above
        rden = activation(op=rsqrt, data=vhat)
        mm   = scalar_tensor_tensor(m,  mult, 9.0,     add,  g)
        term = scalar_tensor_tensor(mm, mult, 0.001,   mult, rden)
        out  = scalar_tensor_tensor(theta, mult, 0.99999, subtract, term)
        store(out_hbm[rows, arange(2048)], out, mask=m_pred)
    return out_hbm
```

- `nl.affine_range(86)` (not `sequential_range`): iterations are independent, so
  the compiler is free to pipeline the next tile's loads under this tile's
  compute/store.
- SBUF tiles: 4 loaded (`theta,g,m,v`) + up to 5 intermediates (`g2, vhat, rden,
  mm, term`) + `out`. ≈ 80 KB/partition live, room for double-buffering. Phase 1
  keeps them distinct/named for clarity; buffer reuse is a phase-2 knob.
- fp32 scalar constants typed as `np.float32` (mirroring the baseline's
  `np.dtype(np.float32).type(...)`).

## Correctness plan

1. Implement as `runs/adamw_v1.py`.
2. Score with `--fast` first, then full 5-seed before recording:
   ```
   python3 \
       ../../verify.py --op adamw --candidate runs/adamw_v1.py --fast
   ```
3. Gate is `l2_norm_passed` across seeds `[0,21,42,63,84]` — trust `verify.py`.
4. If the L2 gate fails, invoke `kernel-accuracy-debugging` (likely suspects:
   wrong input arg mapping `v1..v4`, a mask predicate off-by-one on the tail, or
   an `scalar_tensor_tensor` operand/reverse-flag ordering error) before guessing.
5. Record the result in `benchmark.csv` and the candidate in `candidates.jsonl`
   (parent = baseline). Save the profiler digest under `profile/`.

## Risks / open questions

- **`scalar_tensor_tensor` operand roles:** `operand0` must be the scalar and
  `operand1` the full tile; confirm the `reverse0/reverse1` defaults give
  `(data op0 scalar) op1 tile` (subtract must be `theta_scaled − term`, not the
  reverse). Verified against the numpy model above.
- **`activation(op=square)` availability** on the Scalar engine (vs a Vector
  `tensor_tensor(g,g)`). If `square` is unavailable/slower, fall back to a Vector
  `g*g`, which shifts one op onto the Vector engine (5 Vector ops) — still
  correct, slightly worse Vector floor; a phase-2 concern only.
- **Tail mask correctness** on the 64-row last tile — the single highest-risk
  correctness item; the predicate is copied verbatim from the (correct) baseline.

## Deferred to phase 2 / 3 (explicitly out of scope for phase 1)

- **DMA/compute overlap & Vector rebalance** (the real lever): confirm the fused
  pass is DMA-bound in *measurement*; if Vector-bound, move an op to Scalar/GpSimd
  or restructure to cut a tile-tile combine.
- **Double-buffering / burst-size tuning** on the load stream (cf. silu phase 3:
  finer free-axis tiling won; here the free axis is a single 2048 already).
- **Shape specialization** (phase 3): e.g. the `P=96 × 114` no-mask tiling, or
  wider/narrower row tiles, static unrolling (cf. mamba v5).
