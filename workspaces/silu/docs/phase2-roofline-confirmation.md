# silu (M4096 N7168, fp32) — Phase-2 roofline confirmation (achieved streaming roofline)

This is the **primary** phase-2 deliverable: a rigorous, evidence-backed confirmation
that the promoted phase-1 kernel `runs/silu_v1.py` already sits at the **achieved
single-core streaming HBM roofline** for this elementwise fp32 access pattern and
compiler schedule — so there is **no multiplicative headroom**, only a single-digit-%
DMA-issue/scheduling bubble to (optionally) harvest.

Grounding evidence: `profile/silu_v1.txt` (full 5-seed profiler digest).

| latency | speedup | MFU | PE | Vec | Scl | DMA | HBMrd | HBMwr |
|---------|---------|-----|----|-----|-----|-----|-------|-------|
| 0.3009 ms | 3.398x | 0% | 1% | 1% | 34% | **97%** | 117 MB | 117 MB |

## 1. Traffic is exactly the read-once / write-once minimum

SiLU (`x / (1 + e^-x)`) is a pure elementwise map: every input element is read once
and every output element is written once. The fp32 in/out contract fixes the byte
count:

```
per direction = M · N · 4 B = 4096 · 7168 · 4 = 117,440,512 B = 117.44 MB
total         = 2 · 117.44 MB                                  = 234.88 MB
```

`verify.py` prints the HBM counters as **decimal MB** (`bytes / 1e6`, whole-number
`%.0f`): `117,440,512 / 1e6 = 117.44 → prints "117 MB"` per direction. The measured
digest is exactly `HBMrd 117 MB + HBMwr 117 MB`, i.e. **234.88 MB total = the
theoretical read-once/write-once floor**. There are zero redundant SBUF passes and
zero recompute (contrast the baseline, which does four Vector passes — exp, add,
reciprocal, multiply — through five SBUF buffers; v1 collapses to one fused Scalar
op with `Vec = 1%`). **There is no HBM traffic left to remove.**

## 2. v1 is at the achieved bandwidth ceiling for this stream

Effective aggregate HBM bandwidth implied by v1:

```
234.88 MB / 0.3009 ms ≈ 780.6 GB/s ≈ 781 GB/s
```

Two independent cross-checks against the cost model:

- **The model's `368 GB/s` is a per-unidirectional-stream figure.** Its *serialized*
  prediction — reading and writing on one stream back-to-back — is
  `234.88 MB / 368 GB/s ≈ 0.638 ms`. v1 (0.3009 ms) is **2.1x below** that, because
  real HBM overlaps the read and write streams; the serialized model does not. So the
  serialized 0.638 ms is ~2x conservative for this streaming pattern, and the right
  comparison is the **overlapped one-way** estimate:
  `117.44 MB / 368 GB/s ≈ 0.319 ms`. The measured `0.3009 ms` is essentially equal to
  — and in fact slightly **below** — that conservative one-way estimate. (This is why
  the plan reworded the "0.319 ms floor" to "cost-model conservative one-way
  estimate": v1 already sits under it.) Equivalently, if both directions overlap at the
  368 GB/s per-stream rate the aggregate is `2 × 368 = 736 GB/s`; the measured
  781 GB/s is `781/736 = 1.061` — within ~6% of that two-stream model. That closeness
  is what makes "368 GB/s is an approximate per-stream sustained figure, not a hard
  physical cap" the correct reading (a strict cap would be *exceeded* by the
  measurement, which would instead flag a units/accounting mismatch).

- **The Scalar compute floor is hidden under DMA.** The fused SiLU on the Scalar
  Engine (cpe=1, free=7168, ×32 iters → 229,376 free-axis cycles / 1.2 GHz effective
  ≈ **0.191 ms** in the cost model) is comfortably under the DMA time
  (`Scl = 34%` « `DMA = 97%`). Note the cost-model 0.191 ms and the profiler's
  *measured* Scalar-active time (`34% × 0.3009 ms ≈ 0.102 ms`) are on **different
  accounting bases** (~1.87x apart) — they are not the same quantity and should not be
  conflated. What matters is that **both** are well under the 0.3009 ms latency
  (0.191 ms and 0.102 ms « 0.291 ms of DMA-active), so compute is not on the critical
  path under either figure; DMA is. Batching `k` middle slices per activation does not
  change the total Scalar element count (`32·7168 = 8·(4·7168) = 229,376` free-axis
  cycles either way), so the "compute stays hidden" conclusion is invariant across the
  D1 sweep.

- **`DMA = 97%` active + near-model bandwidth is the saturation confirmation.** Only
  ~3% of the runtime (`3% × 0.3009 ms ≈ 9 µs`) is DMA-idle. Note "active" means the DMA
  engine is *busy*, not that it transfers at the theoretical peak on every active cycle
  — the 97% figure alone does not *prove* absolute saturation. But taken together with
  the achieved bandwidth sitting within ~6% of the overlapped two-stream model
  (781 vs 736 GB/s, below), it is a fair practical confirmation that DMA/HBM streaming
  is the limiting roofline and the ~9 µs idle bubble is the only slack.

## 3. Framing: achieved streaming roofline, NOT the HBM-fabric maximum

The `781 GB/s` figure is the **achieved single-core streaming roofline for THIS
kernel** (this access pattern + this compiler schedule, single logical NeuronCore,
`--logical-nc-config=1`). It is **not** a claim that 781 GB/s is the global Trainium
HBM-fabric mathematical maximum — a different kernel, more cores, or a different
access pattern could move more or fewer bytes/s. Overstating it as the fabric maximum
would be rejected in review (AC-2 negative test). What is defensible and what this
document claims: for this fp32 elementwise SiLU on one core, v1 has removed all
redundant traffic (234.88 MB = floor) and is DMA-saturated (97%) at ~781 GB/s
effective, so **no multiplicative latency win is physically available.**

## 4. Consequence for phase 2

AccelOpt's "~1.67x here" was relative to the **baseline** (4 Vector passes, 5 SBUF
buffers). v1 already captured that and more (3.398x) by collapsing to read-once →
fused-silu → write-once. The fp32 in/out contract fixes the 234.88 MB traffic floor,
so the **only** physically available slack is the ~3% DMA-issue/scheduling bubble
(~9 µs). Realistic phase-2 ceiling: **single-digit-% latency (0.3009 → ~0.29 ms), or
zero.** Any candidate that claims that slack must clear the numeric AC-8 same-session
noise band on a full 5-seed run before it can replace v1; a within-noise tie keeps v1.
"v1 unchanged plus this roofline confirmation" is an explicitly legitimate terminal
outcome.
