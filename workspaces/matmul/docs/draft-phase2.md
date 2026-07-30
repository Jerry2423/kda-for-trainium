# matmul Phase 2 — Profile-driven optimization draft

## Starting point (best correct kernel)

`runs/matmul_v1.py` (Phase 1): M-outer fp32 GEMM, identity-transpose lhs,
K-accumulate 40 tiles into a `[128,512]` PSUM bank, stream 24 N-chunks.
- **Correct** on all 5 seeds. **Latency 15.885 ms → 0.855x** (baseline 13.5785 ms).
- Profiler: **MFU=41%, PE=89%, Vec=2%, Scl=0%, DMA=76%, HBMrd=7584 MB, HBMwr=201 MB.**

## Bottleneck diagnosis (grounded, not guessed)

**Theoretical PE floor (kernel-cost-analysis, trn2).** The Matmul cost
model is `latency = dst_free_elems * 100 / tensor_freq(240)`; there is no separate
weight-load term (stationary load overlaps in the systolic array).
- Main matmuls: `32 M-tiles * 24 N-chunks * 40 K-tiles = 30720`, each dst `[128,512]`
  → `30720 * 512 * 100/240 = 6.554 ms`.
- Transpose matmuls: `32*40 = 1280`, each dst `[128,128]` → `0.068 ms` (**only 1.0%**).
- **PE floor ≈ 6.62 ms.**

**The gap is stall, not work.** `6.62 / 15.885 = 42%` — this *equals* the measured
MFU (41%). So ~58% of wall-clock is the PE array **idle**, and with DMA=76% active
and HBMrd=7584 MB, it is idle **waiting on rhs DMA**.

**Why rhs DMA dominates.** rhs is 252 MB; M-outer reloads it once per M-tile → `252 MB
* 32 ≈ 8.05 GB` (matches HBMrd 7584 MB). Every one of the 30720 main matmuls
`nl.load`s its rhs `[128,512]` tile inline immediately before the matmul, with no
prefetch, so the PE stalls on each load. **rhs reload + no DMA/compute overlap is the
bottleneck.** The transpose (1%) and fp32 rate are NOT the gate.

## Optimization directions (ranked by benefit / risk)

### D1 — M-blocking to reuse rhs across B M-tiles  [HIGHEST VALUE, low risk]
Process a block of **B M-tiles together**: load each rhs K-tile once, feed it into
B stationary lhsT tiles (B matmuls into B separate PSUM accumulators) before
discarding it. rhs HBM traffic drops from 8.05 GB to `8.05/B`:
- B=4 → 2.01 GB (lhsT resident 80 KB/partition), B=8 → 1.01 GB (160 KB/partition).
SBUF budget (192 KB/partition) allows up to B=8 for lhsT, but must also hold the
streamed rhs tile(s) + output tiles → **B=4 is the safe first step, B=8 the stretch.**
Precedent: `6288aaad` (derive tile budget from PSUM/SBUF constraint to kill
redundant weight DMA per tile). Expected: DMA pressure ~4x lower; if the kernel is
DMA-stall-bound this should move MFU substantially toward the 6.6 ms floor.
PSUM note: B accumulators each `[128,512]` = B banks; B=4 uses 4 of 8 banks — fits.

### D2 — Double-buffer the rhs DMA (ping-pong prefetch)  [HIGH VALUE, medium risk]
Even with reuse, the rhs load for K-tile kt+1 should overlap the matmul on kt.
Pre-allocate two rhs SBUF buffers; prefetch the first K-tile before the loop; each
iteration computes on `cur` while DMA-ing the next tile into `1-cur`. Hides the
remaining rhs latency behind PE work.
Precedents: `bc877398`, `3c7e053b` (ping-pong K/V and block-metadata buffers).
Composes with D1 (prefetch the next shared rhs tile). Risk: buffer-index /
dependency bugs; verify correctness each step.

### D3 — lhs load + transpose amortization  [LOW VALUE]
lhs is only 84 MB and loaded once per M-tile already; the transpose is 1% of the PE
floor. Keep the transpose but ensure the lhsT for a whole M-block is built once (a
side effect of D1). **Do NOT invest in eliminating transposes** — negligible payoff.

### D4 — widen moving free dim beyond 512  [REJECT on trn2]
PSUM fp32 bank = 512 free; can't grow a single accumulator past 512 on trn2. Reject.

### Rejected / deferred
- Downcast to bf16/tf32 for 4x PE throughput — **rejected**, breaks the 2e-5 L2 gate
  (fp32 mandatory). This is why the PE floor is what it is.
- N-outer loop — would re-transpose lhs 24x; strictly worse. Reject.

## Plan of attack (≤5 iterations per direction, per the phase contract)

1. **D1 first** (biggest, safest): implement M-blocking B=4, score `--fast`, compare
   latency + HBMrd + MFU vs v1. If SBUF ok and it helps, try B=8. Keep the best B.
2. **D2 on top of the best D1**: add rhs double-buffering; score; keep if MFU rises.
3. Re-profile; if still DMA-bound, revisit B; if now PE-bound (MFU high), stop —
   we're near the fp32 floor.
Each candidate: `verify.py --fast` for a read, full 5-seed before promoting; record
`benchmark.csv` + `candidates.jsonl` (parent = matmul_v1) + `profile/` digest.
**Never regress correctness** (all 5 seeds must keep passing).

## Target

Beat baseline: **> 1.0x** (i.e. < 13.5785 ms) as the Phase-2 exit bar; stretch toward
the 6.62 ms PE floor (theoretical max ≈ 2.05x) as far as DMA reduction allows. If a
direction cannot progress, record the before/after latency + profiler evidence
explaining why rather than silently moving on.

## Correctness / evidence contract (unchanged from Phase 1)
- fp32 throughout; all 5 seeds `[0,21,42,63,84]` pass relative-L2 `< 2e-5`.
- Single `@nki.jit def kernel(v1,v2)`; candidates in `runs/`; never edit baseline/reference.
- Parent DAG in `candidates.jsonl`; per-direction profiling under `profile/`.
