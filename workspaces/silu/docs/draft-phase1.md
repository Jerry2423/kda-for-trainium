# silu (M4096 N7168, fp32) — Phase 1 implementation draft

## Goal

Produce the **first correct** NKI kernel for elementwise SiLU / swish
`y = x / (1 + exp(-x)) = x * sigmoid(x)` over a `(4096, 7168)` fp32 tensor,
passing NKIBench's relative-L2 gate (`||v_k - v_r|| < 2e-5 * ||v_r||`) on all
five seeds `[0,21,42,63,84]`. Prefer a clean, understood, correct kernel over
speed; leave aggressive tuning to phase 2/3. But choose a loop/compute structure
that is already reasonable (minimal DMA, compute hidden under DMA) so we don't
start from the baseline's pathological 5-buffer / 4-vector-pass shape.

## What the operator is

Pure elementwise activation. No reduction, no matmul, no fusion. Every output
element depends only on the co-located input element:

```
y[i,j] = x[i,j] * sigmoid(x[i,j]),   sigmoid(t) = 1 / (1 + exp(-t))
```

The only real content of the kernel is: **(a) the tiled layout**, **(b) which
instruction sequence computes silu**, and **(c) keeping the pass HBM-bound.**

## Tiled layout (from the numpy reference, verified in numpy)

`transform_to_nki_inputs` reshapes the natural (row-major) input:

- `x (4096, 7168)` -> `v1 (128, 32, 7168)` = `[p, m, f]`
  - `v1[p, m, f] == x[p*32 + m, f]`  (**verified**: 20000 random probes match)

Output `v2 (128, 32, 7168)` = `[p, m, f]`, reshaped back to `(4096, 7168)` by
`transform_nki_outputs`, so `v2[p, m, f] == y[p*32 + m, f]`. The signature
therefore matches the baseline: `def kernel(v1)` returning a `shared_hbm` v2 of
the same shape.

Note the layout: the **partition axis is dim 0 (size 128)**, the middle axis
(size 32) and the free axis (size 7168) are both per-partition. Because silu is
purely elementwise, the layout is irrelevant to *correctness* — we just apply the
same op to every element — so no transpose, no cross-partition traffic. All the
128s / 32 / 7168 are exact (128*32 = 4096 = M, 7168 = N), so **no masking or
partial tiles anywhere.**

### SBUF budget forces a middle-axis tile

Per-partition data = `32 * 7168 = 229376` fp32 elements = **896 KB/partition**,
which far exceeds the ~208 KB usable SBUF per partition (trn2). So we cannot hold
a partition's whole row in SBUF; we tile the middle (32) axis.

Natural phase-1 tiling: **loop `i0 in range(32)`**, each iteration operating on
`v1[:, i0, :] = [128, 7168]` = **28 KB/partition**. That fits SBUF with huge
headroom (leaves room for the phase-2 double buffer), and `7168` is within the
Scalar-engine activation free-dim limit (well under 32767), so a single
activation instruction covers the whole 7168-wide slice — no inner free-dim loop
needed. This gives the minimal instruction count: **32 loads + 32 stores** of
`[128, 7168]`, each partition reading a contiguous 7168-run.

## Hardware grounding: cost model + bottleneck (trn2)

Numbers from `kernel-cost-analysis` (trn2: Scalar 1.20 GHz, Vector
0.96 GHz, HBM aggregate 368 GB/s = 16 x 23 GB/s):

- **HBM floor** — read 117 MB + write 117 MB fp32: `235 MB / 368 GB/s = 0.638 ms`.
  (Per-slice Formula-E: load `[128,7168]` = `7168*4*ceil(128/16)/23 ≈ 9.97 us`,
  x32 = 319 us load + 319 us store = 638 us.)
- **Compute floor, fused silu** — Scalar activation, cpe=1, free=7168:
  `1*7168*100/120 = 5.97 us/slice` x32 = **0.191 ms**. Comfortably **under** the
  0.638 ms DMA floor -> the fused-silu kernel is **HBM-bandwidth bound**, using
  zero Vector-engine time.
- **Baseline measured = 1.022 ms**; AccelOpt's reported ~1.67x = **0.612 ms**,
  which is essentially the 0.638 ms HBM floor. **All the headroom is in becoming
  DMA-bound** — i.e. removing extra SBUF passes / instruction overhead and
  overlapping compute with DMA. The baseline leaves ~1.6x on the table by doing
  four Vector passes (exp + add + reciprocal + multiply) through five separate
  large SBUF buffers.

Caveat on the model: the cost model prices `reciprocal` at cpe=26, which would
put the baseline's Vector chain at ~6.9 ms — far above its measured 1.02 ms. So
the model over-states reciprocal, and we should **not** claim "reciprocal is the
bottleneck" from theory alone. The trustworthy signal is measured-vs-HBM-floor
(1.02 vs 0.64 ms), and the real per-engine numbers the profiler returns
(`verify.py` prints MFU / Vec / Scl / DMA / HBM). Phase 1 doesn't need to win
this; it needs a correct kernel whose structure is already HBM-bound-shaped so
phase 2 tunes DMA overlap, not a rewrite.

## Compute-sequence choice: a correctness-first decision ladder

The crux is which instruction sequence computes silu, trading op-count (speed)
against LUT-approximation risk under the 2e-5 rel-L2 gate. The reference computes
`x/(1+exp(-x))` in fp32 (numpy exp ~0.5 ULP), i.e. effectively fp32-exact. The
**baseline uses the hardware `exp` LUT and passes the gate**, which proves this
target's activation LUTs are L2-accurate for this function class. Three rungs,
best-and-simplest first:

**v1 (primary) — fused `nl.silu`, single Scalar instruction.**
```python
y_tile = nisa.activation(op=nl.silu, data=x_tile)   # = x * sigmoid(x)
```
Simplest possible kernel: `load -> silu -> store`. One Scalar op, zero Vector
ops, HBM-bound at the floor, one intermediate buffer. `nl.silu` is a first-class
op specifier added in NKI 2.21 (release notes; confirmed in
`nki-api-reference`), computing exactly `x*sigmoid(x)` internally in fp32. The
**only** risk is that the silu LUT is a distinct table from `exp` and its
approximation error could exceed 2e-5 rel-L2. Given the exp LUT passes, this is
*likely* fine — but it must be scored, not assumed.

**v2 (fallback if v1 fails L2) — `nl.sigmoid` activation + `nl.multiply`.**
```python
sig_tile = nisa.activation(op=nl.sigmoid, data=x_tile)   # Scalar
y_tile   = nl.multiply(x_tile, sig_tile)                 # Vector, tensor_tensor
```
Two ops (1 Scalar + 1 Vector), still drops the baseline's `add` and expensive
`reciprocal`. Exactly equals `x*sigmoid(x)`. Isolates the sigmoid LUT (separate
from silu), so if v1 failed on a silu-specific table, this may pass. Vector
`multiply` (cpe=2, both-SBUF) is ~478 us total, still under the 638 us DMA floor,
so it stays HBM-bound.

**v3 (guaranteed-correct safety net) — exp-exact = baseline math.**
```python
e     = nisa.activation(op=nl.exp, data=x_tile, scale=-1.0, bias=0.0)  # exp(-x)
denom = nisa.tensor_scalar(data=e, op0=nl.add, operand0=1.0)           # 1 + exp(-x)
recip = nisa.reciprocal(data=denom)
y     = nl.multiply(x_tile, recip)
```
This is the accepted baseline's exact sequence (hardware `exp` + exact arith), so
it **passes the gate by construction** — the exit guarantee that phase 1 ends
with a correct kernel regardless of LUT behavior. Slower (4 ops), but only used
if both LUT rungs fail.

**Plan:** implement and score v1 first (expected pass). Only descend the ladder
on an actual L2 failure; each rung is a small, independent edit the RLCR loop can
score. Record which rung passes as the phase-1 kernel and note the LUT-accuracy
finding for phase 2/3.

## Kernel skeleton (v1)

```python
@nki.jit
def kernel(v1):                      # v1: (128, 32, 7168) fp32
    P, MID, F = 128, 32, 7168
    v2 = nl.ndarray((P, MID, F), dtype=np.float32, buffer=nl.shared_hbm)
    for i0 in nl.affine_range(MID):                        # 32 middle-axis slices
        x_tile = nl.load(v1[:, i0, :])                     # [128, 7168] HBM->SBUF
        y_tile = nisa.activation(op=nl.silu, data=x_tile)  # Scalar: x*sigmoid(x)
        nl.store(v2[:, i0, :], value=y_tile)               # [128, 7168] SBUF->HBM
    return v2
```
(Exact index-expression form — `nl.arange(128)[:,None]` etc. — to match the
baseline's addressing idiom; the plan step will pin the precise API spellings.)

## What phase 1 deliberately does NOT do (hooks for phase 2/3)

- **No double buffering yet.** v1 is serialized load->compute->store per slice.
  Phase 2's main lever is overlapping DMA with compute (ping-pong SBUF buffers,
  `affine_range` software pipelining) to actually hit the 0.638 ms HBM floor.
- **No multi-slice batching.** ~208 KB / 28 KB ≈ 7 slices could share one SBUF
  residency to amortize loop overhead; deferred to phase 2 once we see the real
  DMA-vs-overhead split from the profiler.
- **No dtype tricks.** fp32 in/out is mandated by the gate; the compute is
  Scalar-only and already under the DMA floor, so there's nothing to gain from
  bf16 here (unlike a compute-bound op) — and it would risk the L2 gate.
- **No layout change.** Elementwise means layout is correctness-neutral; keep the
  baseline's `(128, 32, 7168)` shape so the harness reconciliation is untouched.

## Acceptance for phase 1

1. Passes relative-L2 on all five seeds (run without `--fast` before promoting).
2. Clean, single-loop structure; every tile understood; no masking.
3. Record the run in `benchmark.csv` and the candidate DAG in `candidates.jsonl`;
   keep the profiler's per-engine digest under `profile/` (it drives phase 2).
   Expectation: v1 already lands well below 1.022 ms (toward the ~0.61 ms floor)
   since it removes three of the baseline's four passes — but phase 1 is judged
   on correctness, not the speedup.
