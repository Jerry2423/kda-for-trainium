# mamba (M7168 C256 S16, fp32) — Phase 1 implementation draft

## Goal

Produce the **first correct** NKI kernel for the Mamba selective-scan (SSM
recurrence), passing NKIBench's relative-L2 gate
(`||v_k - v_r||_2 < 3e-5 * ||v_r||_2`, fp32 — mamba uses the looser 3e-5, see
`adapter/nkibench_case.py:55`) on all five seeds `[0, 21, 42, 63, 84]`.
Correctness-first: a clean, fully-understood kernel over speed. But pick a
loop/compute structure that is already reasonable (**no 16× redundant reload of
delta/u**, no needless copies) and, crucially, **use phase 1 to measure the real
bottleneck** — because the naive "sequential scan over M=7168" framing is
misleading here (see below), and I need the profiler's `Vec%`/`Scl%`/`DMA%`
digest to steer phase 2.

Baseline latency (`mamba_M7168_C256_S16_0.py`) = **1.258274 ms** (cached in
`baselines.json`), measured through the same profiler path. AccelOpt's optimized
sample reaches ~1.6× here → target ≈ **0.79 ms**.

## What the operator is (selective-scan SSM)

Per the numpy reference (`mamba_M7168_C256_S16_numpy_1.py`):

```
deltaA   = exp(delta[:,None,:] * a[:,:,None])         # (C,S,M)
deltaB_u = delta[:,None,:] * b[None,:,:] * u[:,None,:] # (C,S,M)
for i in range(M):                                     # first-order recurrence over sequence M
    state_i = deltaA[...,i] * state_{i-1} + deltaB_u[...,i]
out = sum_S( c[None,:,:] * scan_res )                  # (C,M), reduce over state S
```

Shapes / dtype (NKIBench case 2, fp32):

| tensor | shape | meaning |
|--------|-------|---------|
| `delta` | (256, 7168) | (C, M) |
| `u`     | (256, 7168) | (C, M) |
| `a`     | (256, 16)   | (C, S) |
| `b`     | (16, 7168)  | (S, M) |
| `c`     | (16, 7168)  | (S, M) |
| out     | (256, 7168) | (C, M) |

`C=256` channels, `M=7168` sequence (the scan axis), `S=16` state.

### The key realization: the scan is a hardware primitive, not the challenge

The recurrence `state_i = deltaA_i·state_{i-1} + deltaB_u_i` is a **first-order
linear recurrence** — exactly what `nki.isa.tensor_tensor_scan` computes in a
**single Vector-Engine instruction**, keeping the running carry in-engine without
round-tripping SBUF per timestep (confirmed via the NKI API docs + the
`fused_mamba` tutorial). Its semantics:

```
result[:,0] = op1( op0(data0[:,0], initial), data1[:,0] )
result[:,i] = op1( op0(data0[:,i], result[:,i-1]), data1[:,i] )
```

With `op0=multiply, op1=add, data0=deltaA, data1=deltaBu, initial=0` this **is**
the mamba scan. The M=7168 axis is the free axis of the scan; channels sit on
partitions. So the "sequential dependency over M" that the prompt flags as the
challenge is already handled by the primitive — **the baseline already uses it**
(`kernel.py:49`). Phase 1 therefore is *not* about inventing a parallel/chunked
scan; it is about understanding every tile and measuring where time actually
goes.

## Tiled layout (verified in numpy)

`transform_to_nki_inputs(inputs)` returns `inputs` unchanged — **it is the
identity** (reference file lines 27-28). So the kernel consumes the *natural* 2D
tensors and the signature matches the baseline exactly:

```python
@nki.jit
def kernel(delta, u, a, b, c):   # returns one shared_hbm out of shape (C, M) = (256, 7168)
```

`transform_nki_outputs` just reshapes `k_res` to the reference shape `(C,M)`.

**Partition mapping:** channels `C=256` → partition axis, tiled into
`n_channel_tile = 256/128 = 2` tiles of 128 (the assert `channels % 128 == 0`
holds). Sequence `M=7168` → free axis. State `S=16` → a loop.

### Numpy oracle (settled before writing NKI)

I reproduced the exact per-(channel-tile, state) kernel formulation in numpy
(seed 42, the adapter's fixed input seed):

- `deltaA[c,m] = exp(a[c,s] · delta[c,m])`   ← per-partition scalar scale `a[:,s]`
- `deltaBu[c,m] = (delta[c,m]·u[c,m]) · b[s,m]`  ← `b[s,:]` broadcast over channels
- `st = tensor_tensor_scan(deltaA, deltaBu, initial=0)` along M
- `out += c[s,:] · st`  (accumulate over the 16 states)

Result: **rel-L2 = 4.08e-7 vs the reference** → PASS the 3e-5 gate with ~75×
margin. `||out|| ≈ 2.055`, `max|out| ≈ 0.0127` (tiny magnitudes — the inputs are
`N(0, 0.05)`, so `exp(delta·a) ≈ 1`, decays are near-unity; no overflow risk, and
fp32 throughout the scan is numerically comfortable). **The math is settled; the
only remaining content is the NKI loop structure and keeping DMA/compute lean.**

## Hardware grounding: why the baseline is (almost certainly) Vector-bound

**HBM traffic floor (read-once / write-once):**

```
read  : delta 7.34 + u 7.34 + a 0.016 + b 0.46 + c 0.46 = 15.61 MB
write : out 7.34 MB
total : 22.95 MB
```

At the ~800 GB/s effective HBM BW this profiler showed on the silu streaming
kernel, the DMA floor is **~0.029 ms** — two orders of magnitude below the
1.258 ms baseline. Even the *baseline's* inflated traffic (it reloads `delta` and
`u` inside the 16-iteration state loop → ~242 MB) implies only **~192 GB/s
effective**, far under 800 GB/s. **Conclusion: the baseline is nowhere near
DMA-bound — it is Vector-Engine bound.** The Vector work per channel tile is:

- 16 × `tensor_tensor_scan` over 7168 free elements (serial recurrence, in-engine)
- 16 × `deltaU = delta·u`  (**state-independent** — recomputed every state!)
- 16 × `deltaBu = deltaU·b`
- 16 × `scanC = scan·c`
- 16 × `+=` accumulate

i.e. **~5 vector passes over [128,7168] per state × 16 states × 2 channel tiles**.
The `exp` is an `activation` → it runs on the **Scalar Engine**, off the vector
critical path (good). This breakdown is the phase-1 hypothesis the profiler must
confirm, and it already names the phase-2 levers (below).

## Phase-1 kernel structure (clean, whole-M, delta/u loaded once)

Adopt the baseline's proven building blocks (`activation(exp)`,
`tensor_tensor`, `tensor_tensor_scan`, broadcast of `b`/`c` rows) but with the
**channels-outer / state-inner** loop order so `delta`/`u` are loaded **once per
channel tile** instead of 16× — the mamba analog of rope phase-1's "no redundant
copy" hygiene. This is exactly the AccelOpt `mamba_v2` structure; it is
obviously-correct (same ops, same scan, just a loop reorder + hoisted loads) and
strictly cleaner than the baseline. Whole-M scan (no sequence tiling) keeps it
simple and is proven legal — the baseline scans the full 7168 free axis.

```
for i_channel_tile in affine_range(2):                 # 256 channels / 128
    cs = i_channel_tile*128
    delta_i = load delta[cs:cs+128, 0:7168]            # [128,7168]  loaded ONCE
    u_i     = load u[cs:cs+128, 0:7168]                # [128,7168]  loaded ONCE
    scanC_accum = zeros([128,7168])                    # fp32 accumulator over states
    for i_state in affine_range(16):
        A_i     = load a[cs:cs+128, i_state]           # [128] per-partition scalar
        deltaA  = activation(exp, delta_i, scale=A_i)  # Scalar engine
        B_i     = load b[i_state:i_state+1, 0:7168]    # [1,7168] -> broadcast to [128,7168]
        deltaU  = tensor_tensor(delta_i, u_i, multiply)
        deltaBu = tensor_tensor(deltaU, B_i_bcast, multiply)
        scan    = tensor_tensor_scan(deltaA, deltaBu, initial=0, op0=mul, op1=add)
        C_i     = load c[i_state:i_state+1, 0:7168]    # [1,7168] -> broadcast
        scanC   = tensor_tensor(scan, C_i_bcast, multiply)
        scanC_accum += scanC
    store output[cs:cs+128, 0:7168] = scanC_accum
```

- HBM reads drop from ~242 MB → ~16.5 MB (`delta`+`u` once, `b`/`c` rows over the
  state loop, `a` tiny); writes 7.34 MB. Near the read-once floor. **If** the op
  turns out DMA-bound this is a big win; if Vector-bound (my hypothesis) it's
  free hygiene that doesn't hurt.
- `affine_range` on both loops so the compiler software-pipelines DMA against
  compute (the silu/rope lesson). The `+=` into `scanC_accum` is the proven
  baseline reduction pattern over the 16 affine-range states.

### SBUF budget (whole-M fits)

Per partition, an `[128,7168]` fp32 tile = 28.7 KB. Live across the state loop:
`delta_i`, `u_i`, `scanC_accum` = 86 KB; plus per-state temporaries
(`deltaA`, `deltaBu`, `scan`, `scanC`, reusing `deltaU→deltaBu`) ≈ 3-4 × 28.7 KB.
Peak ≈ 170-200 KB/partition — within the ~208 KB usable, and **less** than the
baseline (which holds `scanC_accum` for *both* channel tiles = 57 KB). If it ever
spills, sequence-tiling (phase-2 lever) is the natural fix, not needed for
correctness.

## Correctness plan

1. Numpy oracle (done): identity transform + op sequence → rel-L2 4.08e-7.
2. First candidate = the structure above. Score `--fast` (seed 42), then the full
   5-seed run before recording as the phase-1 baseline.
3. Record the profiler digest (`MFU`/`PE`/`Vec`/`Scl`/`DMA` %, `HBMrd`/`HBMwr`)
   in `benchmark.csv` + `candidates.jsonl` — this is the phase-1 deliverable that
   steers phase 2.
4. Fallback: if the loop reorder somehow regresses or fails to compile, the plain
   baseline structure (delta/u reloaded per state) is a known-correct floor to
   fall back to.

## Phase-2/3 levers to preview (set up, not implement now — gated on the measurement)

- **Hoist `deltaU = delta·u` out of the state loop (biggest vector lever).** It is
  **state-independent** — the baseline recomputes it 16×. Computing it once per
  channel tile removes ~1 of ~5 vector passes per state (~20% of vector work if
  Vector-bound). Cheap, obviously correct; the top phase-2 candidate.
- **Sequence tiling + carry (AccelOpt `mamba_optimized`).** Tile M into chunks
  (e.g. 512), `sequential_range` over chunks, carry the scan's last column forward
  as the next tile's `initial`. Shrinks live SBUF and enables deeper double
  buffering; note it does *not* reduce total scan vector-cycles, so its value
  depends on the bottleneck verdict (helps if scheduling/SBUF-bound).
- **Engine rebalancing.** `exp` already on ScalarE; check whether more of the
  per-state elementwise work (the `·b`, `·c` broadcasts) can move to Scalar/GpSimd
  to unblock the Vector critical path if Vector-bound.
- **Shape specialization (phase 3).** `S=16`, `C=256` (exactly 2 tiles),
  `M=7168` are all fixed — unroll the 16-state loop, specialize the 2-tile channel
  loop, tune the sequence-tile width.

## Risks / watch-items

- **Bottleneck surprise.** The explicit hypothesis is *Vector-bound* (32 scans +
  per-state multiplies), not DMA-bound. Phase 1 exists to confirm/refute this
  before phase 2 commits effort to the wrong lever (e.g. chasing DMA when the scan
  dominates).
- **`tensor_tensor_scan` operand rules.** `data0`/`data1` can't *both* be in PSUM;
  both in SBUF is fine (our case). Inputs auto-cast to fp32 and math is fp32 —
  matches the reference and the numerically-comfortable oracle.
- **`activation` scale broadcast.** `scale=A_i` (per-partition scalar `a[:,s]`)
  must broadcast along the free axis to give `exp(a[c,s]·delta[c,m])` — verified
  against the oracle and identical to the baseline's usage.
- **Whole-M SBUF pressure.** Fits per the budget above, but is the first thing to
  watch if the compiler reports a spill; sequence-tiling is the ready fallback.
