# rope_single_freq_apply — Phase 1 bottleneck digest + phase-2 verdict

Candidate: `runs/rope_v1.py` (layout A, 64-partition, no-copy, `W = 2048`).
Path label: **`layout_a_no_copy`** (no-copy partition-offset load/store compiled and
passed under `--disable-dge --logical-nc-config=1`; the copy-realign fallback was
not needed). Raw evidence: `profile/rope_v1_perseed_metrics.json`.

## Correctness (full 5-seed gate)

| seed | l2_norm_passed | relative_l2_error | tolerance |
|------|----------------|-------------------|-----------|
| 0    | true           | 0.0               | rtol=2e-5 |
| 21   | true           | 0.0               | rtol=2e-5 |
| 42   | true           | 0.0               | rtol=2e-5 |
| 63   | true           | 0.0               | rtol=2e-5 |
| 84   | true           | 0.0               | rtol=2e-5 |

`all_seeds_passed = true`. rel-L2 is **exactly 0.0** on every seed — this is exact
fp32 arithmetic (4 multiplies + 1 subtract + 1 add, no approximation, no dtype
downcast), so there is no seed-to-seed jitter to worry about, with unbounded margin
against the 2e-5 gate.

## Performance + engine digest (p50, non-`--fast`, warmup=10 iters=100)

| metric | value |
|--------|-------|
| latency p50 | **0.9445 ms** (probe run 0.9436 ms) |
| speedup vs baseline (1.1418 ms) | **1.209x** |
| Vec active % | **91.6%** |
| DMA active % | **93.5%** (`software_dynamic_dma` 99.6%) |
| Scl active % | 0.18% |
| PE active % | 0.21% |
| MFU % | 0% (no matmul) |
| MBU % (HBM-fabric bandwidth util) | **29.9%** |
| HBMrd | **268.44 MB** (256 MiB) |
| HBMwr | **134.22 MB** (128 MiB) |
| HBM total | **402.65 MB** (384 MiB) |
| effective BW = total_bytes / latency | **~427 GB/s** |

## AC-2 — HBM at the read-once/write-once floor

Measured HBM traffic is **exactly** the theoretical payload floor, to the byte:
`hbm_read_bytes = 268435456 = x 128 MiB + cos 64 MiB + sin 64 MiB`,
`hbm_write_bytes = 134217728 = out 128 MiB`, total `402653184 B = 402.65 MB`
(`inputs_outputs_weights_size_bytes` reports the same 402653184). The read/write
split matches the ~268/~134 MB expectation. Four loads + two stores per tile,
no redundant reads: **read-once / write-once confirmed** (0% over floor, well
inside the ±10% AC-2 tolerance).

## Bottleneck verdict: VECTOR-BOUND (co-limited with DMA-active), HBM at floor

Reading the numbers **together**, not any single percentage:

1. **Vec is near-saturated (91.6%) doing the six `tensor_tensor` passes.** This is
   the structural cost of the 64-partition layout: each of the six passes streams
   all `S` free elements on only 64 of 128 vector lanes.
2. **HBM traffic is exactly at the floor** (402.65 MB) — there is no wasted traffic
   to remove; DMA cannot be made cheaper by cutting bytes.
3. **We are NOT at the HBM bandwidth wall.** Effective BW is ~427 GB/s — well below
   the ~781 GB/s single-core streaming roofline the silu campaign measured on this
   profiler, and MBU is only **29.9%** of HBM-fabric peak. So although `DMA active`
   reads 93.5%, the DMA engine is busy issuing/waiting, **not** bandwidth-bound:
   the pure-DMA floor at 781 GB/s would be ~0.516 ms, but we measure 0.944 ms.
4. The ~0.43 ms gap between the 0.516 ms pure-DMA ceiling and the 0.944 ms achieved
   latency is **vector-engine time that is not hidden under DMA** — the six vector
   passes co-limit the wall clock. This is the opposite of silu (97% DMA / 1% Vec,
   at the 781 GB/s roofline): rope's six vector passes make it **vector-bound**.

This matches the AC-5 decision rule for the vector-bound branch: *high `Vec_pct`
with near-floor HBM traffic → phase-2 lever is layout B*.

### Phase-2 lever: layout B (packed 128-partition compute) — halve the vector passes

Because vector op latency is per-free-element and independent of partition count,
packing both output halves onto all 128 partitions lets **3 ops do the work of 6**
(`t1 = x * cos_stacked`, `t2 = x_swap * sin_stacked`, `out = t1 ± t2` over `[128,W]`,
where `x_swap` swaps the two halves and negates the lower one). That roughly halves
the vector time, which is the co-limiting term.

**Constraint carried into phase 2 from this digest:** DMA is already co-saturated in
*active time* at the traffic floor, so layout B must **not raise HBM traffic** — the
`cos`/`sin` broadcast to 128 partitions must be an SBUF partition-broadcast, not a
second HBM read of `cos`/`sin` (which would push traffic above floor and could
negate the vector win). The 29.9% MBU leaves fabric-bandwidth headroom, but adding
reads still costs DMA-active time on an already-93.5%-busy DMA engine. The finer-`W`
lever (silu precedent) is **secondary** here: it only harvests a DMA fill/drain
bubble, and DMA is not the dominant limiter for rope — so it is a phase-3 polish, not
the phase-2 primary.

Theoretical note (context, not a gate): the per-engine cost model predicts the six
`tensor_tensor` passes at 1.64–3.28 ms — already above the measured 1.1418 ms
baseline that does exactly these ops — so the cost model over-predicts real trn2
fp32 vector throughput for this op. The measured digest above is the source of
truth for the verdict, as the plan requires.

## Theoretical per-engine cost cross-check (task7, `kernel-cost-analysis`)

Cost-model figures for phase-2 planning only (the measured digest above remains the
source of truth). `TensorTensorArith` uses Formula B (DualMemSpace); both operands
in SBUF → 2 cyc/elem; trn2 Vector @ 0.96 GHz. Op latency is **per free-element and
independent of partition count**, so it depends only on the `S = 262144` free
elements streamed per partition.

| quantity | value |
|----------|-------|
| per `tensor_tensor` pass (2 cyc/elem) | `2·262144·100/96` = 546.1 µs |
| **Layout A** vector floor (6 passes) | **3.277 ms** (optimistic 1 cyc/elem: 1.638 ms) |
| **Layout B** vector floor (3 passes) | **1.638 ms** (optimistic 1 cyc/elem: 0.819 ms) |
| **Layout A / Layout B vector ratio** | **2.00x** (exactly 6/3) |
| DMA cost-model floor @ 368 GB/s aggregate | 1.094 ms |

Two takeaways for phase 2:

1. **Layout B halves the theoretical vector floor** (3.277 → 1.638 ms, exactly 2x),
   because 3 packed `[128, W]` ops replace 6 `[64, W]` ops while each op streams the
   same free-element count per partition. This is the theoretical basis for the
   layout-B lever the measured verdict points to.
2. The model **over-predicts by ~3.47x** here — the layout-A floor (3.277 ms, or
   1.638 ms optimistic) sits well above the measured 0.9445 ms — confirming the plan's
   caution that this op's cost model is not a reliable floor. So layout B's *measured*
   win will likely be **less than the theoretical 2x**: real trn2 fp32 vector runs
   faster than the model, DMA-active is already co-saturated at the traffic floor
   (0.516 ms pure-DMA ceiling), and layout B adds a cross-partition swap/negate plus
   an SBUF `cos`/`sin` broadcast. Expect a solid-but-sub-2x improvement, floored by
   the ~0.516 ms DMA ceiling.
