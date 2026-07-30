# rope_single_freq_apply (D128, B*H*N=262144, fp32) — Phase 1 implementation draft

## Goal

Produce the **first correct** NKI kernel for the single-frequency RoPE apply,
passing NKIBench's relative-L2 gate (`||v_k - v_r||_2 < 2e-5 * ||v_r||_2`, fp32)
on all five seeds `[0, 21, 42, 63, 84]`. Correctness-first: a clean, fully
understood kernel over speed. But pick a loop/compute structure that is already
reasonable (minimal DMA, no redundant copies) and, crucially, **use phase 1 to
measure the real bottleneck** — because unlike the silu case this op is *not*
obviously HBM-bound (see the hardware grounding below).

Baseline latency (`rope_single_freq_apply_B1_H64_N4096_D128_0.py`) = **1.1418 ms**
(cached in `baselines.json`), measured through the same profiler path.

## What the operator is

Rotary position embedding, applied elementwise with a **cross-half interaction**.
Split `x` into two halves over the head dim `D=128` (`half=64`):

```
x0 = x[:64, :]        # lower half of D
x1 = x[64:, :]        # upper half of D
out0 = x0 * cos - x1 * sin      # -> output rows   0:64
out1 = x0 * sin + x1 * cos      # -> output rows  64:128
out  = concat([out0, out1], axis=0)   # (128, S)
```

Each output element depends on the **co-located column** of both halves plus the
co-located `cos`/`sin`. There is no reduction and no matmul — but it is *not* a
pure single-input elementwise map like silu: every output column mixes `x0` and
`x1`, so the kernel does **four tensor×tensor products + two tensor±tensor
combines = six vector passes** per element. That distinction drives the whole
bottleneck story below.

## Tiled layout (verified in numpy)

`transform_to_nki_inputs(inputs)` returns `inputs` unchanged — **it is the
identity**. So, unlike silu (which reshapes to a 3D `(128,32,7168)` tiled view),
the kernel here consumes the *natural* 2D tensors directly:

- `x_in`: `(128, 262144)` = `(D, B*H*N)` fp32 — partition axis = `D` (=128), free axis = `S` (=262144)
- `cos`, `sin`: `(64, 262144)` = `(D/2, S)` fp32 — partition axis = 64, free axis = `S`
- output: `(128, 262144)` fp32, same layout as `x_in`

`transform_nki_outputs(k_res, ref)` just wraps `(k_res,)`. So the kernel signature
matches the baseline exactly:

```python
@nki.jit
def kernel(x_in, cos, sin):   # returns one shared_hbm out of shape (128, S)
```

**Verified in numpy** (seed 42, the adapter's fixed input seed): `transform_to_nki_inputs`
is identity, and the op sequence above reproduces `forward(...)` with **rel-L2 =
0.0 (exact match)**. So the math is settled before we write a line of NKI; the
only remaining content is (a) the partition layout, (b) the free-axis tiling, and
(c) keeping loads/stores/compute lean.

### SBUF budget forces a free-axis (S) tile

Per-partition data for the whole `S=262144` free axis is `262144 * 4 B = 1 MB`,
far over the ~208 KB usable SBUF/partition (trn2). So we tile the **free axis**
`S` into chunks of width `W` and loop. `S = 2^18`, so any power-of-two `W`
divides it exactly → **mask-free, rectangular tiles, no tail handling**. Live
tiles per iteration are ~8 `[64, W]` fp32 buffers (`x0, x1, c, s` + products/
combines); budget:

| `W`  | iters `S/W` | 8 tiles `[64,W]` | % usable | fits |
|------|-------------|------------------|----------|------|
| 2048 | 128         | 64 KB/part       | 30%      | yes  |
| 4096 | 64          | 128 KB/part      | 54%      | yes (room for a phase-2 double buffer) |
| 8192 | 32          | 256 KB/part      | 108%     | no   |

Phase-1 default: **`W = 2048`** (128 pipeline iterations, comfortable headroom).
This is a *starting point*, not a tuned value — free-axis tile width is an
explicit phase-2/3 lever (the silu campaign found **finer wins**, optimum ~4 KB/
partition ≈ `W=1024` for fp32; see `[[kda-silu-progress]]`).

## Hardware grounding: cost model + bottleneck (trn2, single core)

**HBM traffic floor (hard lower bound — bytes that must move):**

```
read  : x 128 MB + cos 64 MB + sin 64 MB = 256 MB
write : out 128 MB
total : 402.65 MB
```

At ~800 GB/s effective HBM BW (measured on this profiler for the silu streaming
kernel), the HBM floor = **0.503 ms** → a **2.27× ceiling** vs the 1.1418 ms
baseline *if* we ever became fully DMA-bound. This is the one number I trust as a
hard floor.

**Vector-engine cost (the reason this op differs from silu):** six
`tensor_tensor` passes, each over `S = 262144` free elements/partition. Per the
cost model (Formula A/B, trn2 Vector @ 0.96 GHz):

- both operands in SBUF → 2 cyc/elem → `6 * 2 * 262144 * 100/96` = **3.28 ms**
- optimistic 1 cyc/elem → **1.64 ms**

Both estimates are **at or above the measured 1.1418 ms baseline** — which does
exactly these six `tensor_tensor` ops. So the theoretical vector model
*over-predicts* the real device here (real trn2 fp32 vector throughput evidently
beats the naive 2-cyc formula). **Takeaway: the cost model is not a reliable
floor for this op; the profiler's measured `Vec%` vs `DMA%` digest is the source
of truth.** Two consequences:

1. Phase 1 must **read the profiler `summary_metrics`** (Vec / Scl / DMA % +
   HBMrd/HBMwr) on the first correct kernel to learn whether we are vector-bound
   or DMA/scheduling-bound. That verdict, not an assumption, sets phase-2's
   direction. (Silu was ~97% DMA on a single cheap Scalar op; rope with 6 vector
   passes may sit much higher on `Vec%`.)
2. If the profiler shows we are **vector-bound**, the dominant lever is the
   **packed 128-partition layout** described next; if **DMA/scheduling-bound**,
   the lever is finer free-axis tiling (silu precedent).

**Instruction selection note:** the products are genuine tensor×tensor
(`x0 * cos`), so `tensor_scalar` / `scalar_tensor_tensor` (the cheaper 1-cyc
paths) do **not** apply — one operand would have to be a scalar. No fused
"a*b − c*d" primitive exists. So six `tensor_tensor` is the minimum for the
64-partition layout; the only way to cut vector work is to pack onto 128
partitions (halving the op count), below.

## Phase-1 kernel structure (layout A — 64-partition, no copy)

Loop over free-axis chunks; each iteration is fully independent → `nl.affine_range`
so the compiler software-pipelines DMA against compute (the silu lesson: let
`affine_range` build one deep pipeline).

```
for j in nl.affine_range(S // W):          # 128 independent iterations
    cols = j*W : (j+1)*W
    x0 = load x_in[0:64,   cols]           # [64, W] -> SBUF partition 0
    x1 = load x_in[64:128, cols]           # [64, W] -> SBUF partition 0 (fresh tile!)
    c  = load cos[:, cols]                 # [64, W]
    s  = load sin[:, cols]                 # [64, W]

    e_cos = tensor_tensor(x0, c, multiply)     # x0*cos
    o_sin = tensor_tensor(x1, s, multiply)     # x1*sin
    out0  = tensor_tensor(e_cos, o_sin, subtract)   # x0*cos - x1*sin
    e_sin = tensor_tensor(x0, s, multiply)     # x0*sin
    o_cos = tensor_tensor(x1, c, multiply)     # x1*cos
    out1  = tensor_tensor(o_cos, e_sin, add)        # x1*cos + x0*sin

    store out[0:64,   cols] = out0
    store out[64:128, cols] = out1
```

**Why this is cleaner than the baseline (which needs an `nl.copy`):** the baseline
loads all of `x_in` as a 128-partition tile and slices `x1` from partitions
`64:128`, so its lower operand sits at partition base 64 and must be `nl.copy`-ed
to base 0 before `tensor_tensor` (which requires both operands at the same base
partition). By instead **loading `x1 = x_in[64:128, cols]` into its own fresh
`[64,W]` tile**, the destination lands at partition 0 (confirmed via the NKI docs:
`nl.load` returns a new SBUF tile based at partition 0 regardless of the HBM row
offset). So `x0, x1, c, s` are all base-0 and aligned — **six `tensor_tensor`
with zero copies**. Storing `out1` back to HBM rows `64:128` is the mirror image
and is a supported partition-offset store.

- 4 loads + 2 stores per tile → total HBM traffic = the 402.65 MB floor exactly
  (read-once/write-once); no redundant reads.
- Uses only 64 of 128 vector lanes. This does **not** waste vector wall-clock in
  layout A (op latency depends on free-dim size, not partition count), but it *is*
  the slack that layout B exploits.

## Phase-2/3 levers to preview (set up, not implement now)

- **Layout B — packed 128-partition compute (the big vector lever).** Because
  vector op latency is per-free-element and *independent of partition count*,
  packing both output halves onto all 128 partitions lets 3 ops do the work of 6:
  `t1 = x * cos_stacked` (`[128,W]`), `t2 = x_swap * sin_stacked` (`[128,W]`),
  `out = t1 ± t2` — where `x_swap` is `x` with its two halves swapped and the
  lower half negated, and `cos_stacked`/`sin_stacked` are `cos`/`sin` broadcast to
  128 partitions. This **halves vector time** but adds a cross-partition
  swap/negate and either a partition-broadcast copy or a second `cos`/`sin` DMA
  (which would raise HBM traffic). Whether it wins depends entirely on the phase-1
  `Vec%` vs `DMA%` verdict — hence measuring first.
- **Finer free-axis tiling** (`W` → 1024 or below). The silu campaign showed finer
  chunks amortize the pipeline fill/drain bubble when DMA-bound (optimum ~4 KB/
  partition burst). Cheap to sweep once we know the bottleneck.
- **In-place / buffer reuse** to shrink live SBUF and enable deeper double
  buffering (only if scheduling-bound).

## Correctness plan

1. Numpy oracle (already done): identity transform + op sequence → rel-L2 0.0.
2. First candidate = layout A above, `W=2048`. Score with
   `verify.py --op rope_single_freq_apply --candidate runs/<file>.py --fast`
   (seed 42), then the full 5-seed run before recording as the phase-1 baseline.
3. Record the profiler digest (Vec/Scl/DMA %, HBMrd/HBMwr) in `benchmark.csv` and
   `candidates.jsonl` — this is the phase-1 deliverable that steers phase 2.

## Risks / watch-items

- **Partition-offset load/store semantics** — the whole no-copy design rests on
  `nl.load(x_in[64:128, :])` landing at partition 0 and `nl.store(out[64:128,:])`
  writing rows 64:128. Confirmed in the NKI docs; if the compiler rejects it,
  fall back to the baseline's 128-partition-load + `nl.copy` realign (still 6
  `tensor_tensor`, one extra copy).
- **`tensor_tensor` operand memory spaces** — both operands in SBUF is the 2-cyc
  path; that's inherent to a two-tensor multiply and matches the baseline.
- **Bottleneck surprise** — the explicit hypothesis is that rope may be
  *vector-bound* (6 passes) where silu was DMA-bound (1 pass). Phase 1 exists to
  confirm/refute this before committing phase-2 effort to the wrong lever.
