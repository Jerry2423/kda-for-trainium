# rmsnorm_matmul (M4096 N2048 K1024, fp32) — Phase 2 implementation draft

## Goal

Start from the best correct kernel (`runs/rmsnorm_matmul_v1.py`, **1.066x**, full
5-seed PASS) and reduce on-device latency **without ever regressing correctness**
(NKIBench relative-L2 gate `< 2e-5 * ||v_r||` on seeds `[0,21,42,63,84]`, fp32).
Phase-2 discipline: identify the real bottleneck from profiling + the theoretical
floor, enumerate directions, rank by expected benefit vs risk, and explore each for
**at most five iterations**, collecting before/after latency (`verify.py`) and
profiler evidence to justify keep / revise / reject.

## Where phase 1 left us (measured + modeled)

Phase-1 profiler digest (both `--fast` and full 5-seed identical):

```
latency=0.4716ms  speedup=1.066x
MFU=46%  PE=97%  Vec=15%  Scl=11%  DMA=20%  HBMrd=25MB  HBMwr=34MB
```

Theoretical per-engine cost model (`kernel-cost-analysis`, trn2, this
tiling 32 M-tiles × 4 N-chunks × 8 K-tiles, n=512):

| Engine | work | cost | vs PE |
|---|---|---|---|
| **PE main matmul** | 1024 nc_matmul, dst free=512 | **218.45 µs** | = 100% of the systolic floor |
| **PE transpose** | 256 nc_matmul(is_transpose), dst free=128 | **13.65 µs pure** (up to ~27 µs with weight-fill) | +6.25% (∼9–12.5% w/ fill) |
| Vector (norm reduce + scale) | 32×(reduce+scale) | ~69 µs | 32% — hidden |
| Scalar (square) | 32× | ~27 µs | 12.5% — hidden |
| DMA (x load + w load + out store) | 16+8+32 MB | ~160 µs | 73% of PE — under ceiling |

Two conclusions that set the whole phase:

1. **The main matmul is already at the fp32 systolic floor — zero wasted PE cycles.**
   The tiling is exact (32·128=4096, 8·128=1024, 4·512=2048; k=128, n=512 fill the
   array). There is *no* micro-tiling win left in the matmul itself. This matches the
   sibling `matmul` task's finding that fp32 is a hard ~3.62x PE penalty and MFU~46–49%
   is a **structural ceiling**, not inefficiency (`kda-matmul-progress`).

2. **The single recoverable lever is the 256 identity-transpose matmuls** (moving x's
   K-dim onto the partition axis). They are the *only* non-floor PE work: 6.25% of the
   matmul in the pure model, ~9–12.5% counting the poorly-amortized 128-cycle systolic
   fill of a standalone transpose (consistent with PE=97% measured). Everything else —
   RMSNorm vector/scalar work, DMA — is comfortably **hidden under the PE-bound
   matmul**, so it is not worth touching for its own sake. **DMA is only ~20% busy**
   (fp32 inflates the PE-time denominator), i.e. ~54 µs of headroom under the PE
   ceiling → an idle engine we can offload transposes onto.

Note M-blocking — the sibling `matmul` task's phase-2 win — **does not apply here**:
w is already fully resident (8 MB, loaded once) and each x tile is loaded once, so we
are already compute-bound with no redundant HBM traffic to remove.

## The phase-2 thesis

**Get the x-transpose off the 97%-busy Tensor Engine and onto an idle engine (DMA,
or Vector via `nc_transpose(engine=)`), removing the only non-floor PE work.**
Upper bound from the cost model: PE 232.1 → 218.45 µs ⇒ **~5.9% (up to ~12%) faster**
if the transpose is fully offloaded and hidden. The floor keeps MFU where it is; we
just stop paying the transpose tax on the bottleneck engine.

A key algebraic enabler (verified in numpy, all 5 seeds): **RMSNorm's `inv_rms[m]` is
a per-row scalar, so it commutes with the matmul** —
`(x·inv_rms) @ w == (x @ w)·inv_rms`. Measured rel-L2 of the post-scaled form
`(x@w)·inv_rms` vs reference: **~4.85e-7 « 2e-5** on seeds `[0,21,42,63,84]`
(input-scaled v1 is ~1.3e-7; both pass comfortably). This lets us transpose **raw x**
straight from HBM (no dependency on the norm) and apply the per-row scale on the
*output* at PSUM eviction — decoupling the norm from the matmul's critical path and,
crucially, letting the transpose read x directly from HBM via DMA.

Precedents (`kernel-optimization-kb`) back this ordering:
- **Post-scale / eviction-fold** of a per-row factor: proven, dtype-agnostic
  (`28094369`, `d22d23db`, `5f08e8cb`, `63e18e33` — fold scale into the PSUM→SBUF
  eviction with a single `tensor_scalar`/`scalar_tensor_tensor` instead of a separate
  input pass).
- **Off-PE transpose:** DVE/Vector-engine transpose (`597cf19e`, `tiled_dve_transpose`
  family) is fp32-safe and targets exactly the idle engine in a PE-bound kernel.
  **DMA-transpose** (`86fd24ec`, `710a49f3` +25% on some configs) matches our spare
  DMA — **but every library precedent is gated on 2-byte dtype**, with explicit
  small-tile / overhead-loss warnings for fp32. So DMA-transpose is the highest-ceiling
  but least-certain lever for fp32; treat as an experiment, not a given.
- **DLoC RMSNorm** (`5795192b`, −19.5% at T≥512): norm pass *emits the matmul's staged
  layout directly*. Highest ceiling, largest change; our M=4096 clears its size gate.

## Candidate directions, ranked by benefit ÷ risk

### D1 — Post-scale eviction fold (LOW risk, enabler; do first)
Move `inv_rms` off the input and onto the output: transpose **raw** x, matmul into
PSUM, then apply the per-row `[128,1]` `inv_rms` when copying PSUM→SBUF via
`tensor_scalar(op0=multiply, operand0=inv_rms)` (or `scalar_tensor_tensor`) — this
*replaces* the existing `nl.copy` eviction, adding no new pass.
- **Benefit on its own:** ~neutral latency (both scale passes are hidden under PE;
  input-scale touches 1024 elems/row, output-scale touches 2048 — slightly more, but
  on the idle Vector engine). Its value is as the **enabler for D2/D3**: the transpose
  no longer depends on the normalized values, so it can source raw x from HBM.
- **Correctness:** verified rel-L2 ~4.85e-7 « 2e-5, all 5 seeds. `inv_rms` still a
  per-partition `[128,1]` operand (same shape/idiom as v1). Reduction/rsqrt unchanged.
- **Risk:** low — arithmetic proven, one instruction relocated. Watch: `tensor_scalar`
  reading a PSUM tile (confirmed supported); the per-row operand must align to the
  output tile's partition (m_in) — it does, since output partition = m_in like v1.
- **Iterations:** ~1–2 (implement, verify parity at ≥ v1's 1.066x).

### D2 — Off-PE transpose via `nisa.dma_transpose` (MAIN swing, MEDIUM–HIGH risk)
With D1 in place, load x from HBM already transposed to `[k_in(par), m_in(free)]` via
`nisa.dma_transpose` (HBM→SBUF), eliminating all 256 PE transpose matmuls. The norm
still needs x in `[m_in(par), k(free)]` for the free-axis reduce — so either (a) also
keep the normal x load for the norm (x is only 16 MB; two reads = 32 MB, still far
under the DMA ceiling — DMA ~160→~205 µs, still < 218 µs PE), or (b) compute the norm
from the transposed tile via a partition-axis reduction. Start with (a) — simplest,
and the cost model says DMA has the headroom.
- **Benefit:** removes the 13.65–27 µs of transpose PE ⇒ **~1.06–1.12x on top of v1**
  (target ≈ 0.42–0.44 ms) if hidden under DMA. This is the phase's main expected win.
- **Risk (HIGH, must de-risk first):** (i) **fp32 dma_transpose is unproven** — the
  fast HW-DGE path is 2-byte-only; fp32 falls to a slow path (~50% BW SBUF→SBUF,
  ~90% HBM→SBUF) and library commits *disable* it for small tiles. (ii) The sibling
  task found this **remote NKI is older** and rejected engine-steering kwargs, so
  `dma_transpose` may not exist / may reject args here. **Mitigation: a 1-instruction
  availability probe before committing** to the redesign (see Validation). If it
  errors or measures slower, fall back to D3.
- **Iterations:** ≤3 (probe; wire transposed load + post-scale; measure vs v1;
  optionally the norm-from-transposed variant).

### D3 — Off-PE transpose via `nc_transpose(engine=vector)` (fp32-safe fallback)
If DMA transpose is unavailable/slow, do the `[128,128]` transpose on the **Vector
Engine** instead of the PE: `nisa.nc_transpose(dst, data, engine=<vector>)`. Vector is
only ~32% utilized, so 256 small transposes may hide there and off-load the PE.
- **Benefit:** same PE saving as D2 (~5.9%+), without needing DMA-transpose support.
- **Risk:** MEDIUM — Vector-path `nc_transpose` is documented ≤ `[32,32]` per op (may
  force 4×4 sub-tiling of each 128×128 → 16× the transpose instruction count on
  Vector; must check it still fits under the Vector budget) and the **`engine=` kwarg
  may be rejected on this older remote NKI** (same class of failure the sibling task
  hit with `tensor_copy(engine=)`). Probe the kwarg before committing.
- **Iterations:** ≤2 (probe kwarg + tile limit; measure).

### D4 — Lower-precision matmul (bf16/tf32) — REJECT by gate, 1-shot probe only
The structural ceiling is the fp32 PE rate; bf16 is ~3.6x faster on trn2. But bf16
mantissa error (~4e-3) and even tf32 (~1e-3) are **orders above the 2e-5 gate**, and
the sibling task empirically confirmed bf16 fails correctness here. **Rejected**; at
most a single calibration probe (like the sibling's) to re-confirm the fp32 penalty
magnitude for the record — not a promotable candidate.

### D5 — DLoC-style norm-emits-transposed-layout fusion (STRETCH, only if D2/D3 win)
Have the norm pass produce the transposed `[k_in, m_in]` layout directly (compute the
sum-of-squares as a cross-partition reduction, e.g. `nc_matmul(ones[128,1], x)` or a
partition reduce), so no separate transpose exists at all. Precedent −19.5%. Largest
rewrite and highest correctness-surface risk; only pursue if D2/D3 prove the off-PE
transpose direction and iterations remain.

## Plan of attack (order)

1. **D1** (post-scale eviction fold) — implement on a copy of v1, confirm ≥ 1.066x and
   full-seed parity. This is the safe base and the enabler.
2. **Probe D2** — a tiny standalone `nisa.dma_transpose` fp32 call to confirm it exists
   and lower/measure BW on this remote target *before* the redesign.
   - If OK → wire the transposed HBM load + D1 post-scale; measure vs v1.
   - If unavailable/slower → **D3** (`nc_transpose(engine=vector)`), probing the kwarg
     and the 32×32 tile limit first.
3. Keep the best of {v1, D1, D2, D3}. If an off-PE transpose wins and budget remains,
   consider **D5**; run the **D4** calibration probe once for the record.

Never promote a candidate that fails any seed or regresses below v1's 1.066x. Each
direction gets ≤5 iterations; stop a direction early on a clear reject.

## Correctness invariants (unchanged from v1, must hold)

- Every dim divides evenly → **no masks / partial tiles** anywhere.
- Reduction is a single full-1024 free-axis `tensor_reduce`; `inv_rms = rsqrt(sumsq·1/K)`
  (folded 1/K) — bit-identical role to v1 (rel-L2 1.3e-7). Post-scale variant verified
  rel-L2 4.85e-7 « 2e-5.
- Matmul accumulation stays **fp32 in PSUM** (matches numpy `np.matmul`). Transpose is
  exact in fp32 regardless of engine (identity-matmul / nc_transpose / dma_transpose
  are all value-preserving permutations).
- Output tile layout `[m_in(par), n(free)] → v3[mt, :, n0:n0+512]` unchanged, so
  `inv_rms[m]` (per output partition) broadcasts correctly at eviction.

## Risks / things to watch

- **Older remote NKI API surface.** `dma_transpose` and `engine=` kwargs may not exist
  / may reject args (sibling task precedent). **Always probe with a throwaway call
  before a redesign**; the known-good transpose is the identity-matmul (v1) and the
  profiler-example `matmul_kernel.py` idiom — keep v1 as the guaranteed fallback.
- **fp32 DMA-transpose throughput.** Even if available, small 4-byte tiles may run slow
  enough to erase the PE saving — measure end-to-end, don't trust the model's no-fp32-
  throttle DMA numbers blindly.
- **Second x read** (norm + transposed load) doubles x HBMrd to 32 MB — still under the
  DMA ceiling per the model, but confirm DMA% doesn't climb past PE and flip the
  bottleneck.
- **Noise band.** Sibling task saw ~±1.8% run-to-run; a <2% "win" is noise. Re-run
  controls and require a margin before promoting. Score `--fast` for triage, **full
  5-seed** before promoting.
- **PSUM/SBUF pressure** from any extra resident transposed-x buffer — watch for spills
  in the profiler (sibling's B=8/B=16 regressed on residency).

## Validation plan

1. **Availability probes first** (cheap, before any redesign): a minimal kernel calling
   `nisa.dma_transpose` (D2) and `nisa.nc_transpose(engine=...)` (D3) on a `[128,128]`
   fp32 tile, scored with `--fast`, to confirm the API compiles on this remote target.
2. Per candidate, `--fast` (seed 42) for a quick correctness + latency read:
   ```bash
   python3 \
       ../../verify.py --op rmsnorm_matmul --candidate runs/<file>.py --fast
   ```
3. On PASS **and** a latency ≥ v1 (with margin over the noise band), full 5-seed run
   (drop `--fast`) before recording as promoted.
4. Record each perf change in `benchmark.csv`, each candidate node in `candidates.jsonl`
   (parent = its base, DAG links), and the profiler digest (MFU/PE/Vec/Scl/DMA/HBM)
   under `profile/` — especially PE% before/after to confirm the transpose actually
   left the Tensor Engine.

## Phase-2 success criterion

A correct (full 5-seed PASS) kernel that beats v1's **1.066x** by more than the ~±1.8%
noise band — realistically **~1.10–1.18x** if the off-PE transpose lands (PE drops from
232 µs toward the 218 µs floor). If every off-PE transpose route is infeasible on this
remote NKI or fails to beat noise, the defensible phase-2 exit is: **v1 remains
promoted, with measured evidence that it sits at the fp32 PE floor and the only
non-floor work (the transpose) cannot be moved off-PE on this target** — mirroring the
sibling matmul task's floor-confirmation outcome.
