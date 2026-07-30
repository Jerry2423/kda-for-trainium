# rope_single_freq_apply — Phase 2 exit decision + phase-3 steering verdict

## Outcome: layout B WINS decisively. Promoted `rope_v3_layoutB_pe.py` — 0.696 ms, **1.641x** over baseline (1.358x over layout A), exact (rel-L2 = 0.0 all 5 seeds).

| kernel | tile W | latency p50 | vs baseline | vs layout A | verdict |
|--------|--------|-------------|-------------|-------------|---------|
| baseline (NKIBench) | 512 | 1.1418 ms | 1.000x | — | — |
| `rope_v1` layout A | 2048 | 0.9445 ms | 1.209x | 1.000x | phase-1 promoted |
| `rope_v2_layoutB_scalar` | 2048 | 0.7801 ms | 1.464x | 1.211x | D1b kept, superseded |
| `rope_v2_layoutB_scalar_w512` | 512 | 0.7337 ms | 1.557x | 1.286x | exact fallback |
| **`rope_v3_layoutB_pe`** | **512** | **0.696 ms** | **1.641x** | **1.358x** | **PROMOTED** |

All layout-B candidates are **exact** (per-seed rel-L2 = 0.0 on all of `[0,21,42,63,84]`).
The PE/hybrid realization (D1c) unexpectedly beat the all-Scalar sibling (0.734 ms) by 5.2%
repeatably — promoted per DEC-1 (win > the ~3-5% noise band). The Scalar sibling serves as
a guaranteed-exact fallback.

## What won (D1c hybrid): PE permutation + Scalar broadcasts

The whole bet was whether the cross-partition builds (64->128 broadcast of cos/sin,
64<->64 half-swap + negate of x) could **hide on idle engines** rather than bouncing back
onto the bound Vector engine. The winning D1c hybrid splits the work across two engines:

- **`x_swap_neg = [-x1; +x0]` built via a 0/+-1 permutation `nc_matmul(swap_t, a)` =
  `swap_t.T @ a` on the PE (Tensor) engine -> PSUM.** The `swap_t` constant is
  `[[0,I],[-I,0]]` so `swap_t.T = [[0,-I],[I,0]]`, compiled as a `nl.shared_constant`
  loaded to SBUF once. This is genuine `[128,512]` streaming matmul work on the
  otherwise-idle PE (PE was 0.21% in layout A).

- **`cos_pack`/`sin_pack` kept as Scalar-engine SBUF broadcasts** via
  `nisa.activation(op=nl.copy, scale=+-1)` cross-half copies — the same mechanism as the
  D1b all-Scalar sibling.

- **Placement:** `M1 = A(SBUF) * cos_pack(SBUF)`, `M2 = x_swap_neg(PSUM) * sin_pack(SBUF)`
  — at most one PSUM operand per `tensor_tensor`. Result: the most engine-balanced profile
  (PE 50%, Vec 69%, Scl 47%, DMA 99% — all compute engines hide under DMA).

- **fp32 `nc_matmul` was not documented bit-exact**, so AC-1.1 required the full 5-seed
  gate. Empirically: rel-L2 = 0.0 (exact) — no tf32 decomposition on the 0/+-1 selection
  pattern.

- The Scalar-only sibling (`rope_v2_layoutB_scalar_w512`, 0.734 ms) serves as a
  **guaranteed-exact fallback** if a future compiler change breaks the `nc_matmul` exactness.

## Active-time table (PE/hybrid final metrics)

Absolute active-time (active_ms = pct x latency) proves the win **materialized**, not merely
relocated (AC-4):

| engine | layout A | layout B PE/hybrid | reading |
|--------|----------|-------------------|---------|
| Vec | 0.865 ms | 0.475 ms | ~halved by 6->3 packing |
| Scl | 0.002 ms | 0.329 ms | cos/sin broadcasts hide under DMA |
| PE  | 0.002 ms | 0.354 ms | x_swap_neg permutation hides under DMA |
| DMA | 0.883 ms | 0.689 ms | sole wall-clock limiter |

Vec-active fell ~2x (0.865 -> 0.475 ms) — exactly the 6->3 `tensor_tensor` pass reduction.
PE and Scl absorbed the build work and both hide under DMA. DMA-active dropped even though
HBM bytes are identical (402.65 MB, at floor) because the natural single `[128,W]` x-load
issues cheaper than layout A's two separate `[64,W]` x-loads.

## Directions not pursued (recorded, per AC-5/AC-7)

- **bf16 downcast (D3) — REJECTED up front (AC-7).** RoPE is pure elementwise with no
  reduction to average away rounding; bf16 error ~ 4e-3 >> the 2e-5 gate. fp32 mandatory.
- **Explicit ping-pong / burst-batching (D3) — REJECTED up front (AC-7).** `affine_range`
  already builds the software pipeline; silu precedent shows manual ping-pong/wider bursts
  regress.
- **D1b (all-Scalar) was tried first** (exact, 0.778 ms KEEP < 0.80 ms threshold); D1c
  (PE/hybrid) was then implemented per the Round-0 review's no-deferral contract and
  unexpectedly beat D1b by 5.2%.
- D1 design-iteration count: **2** (D1b + D1c), well within the <=5 cap.

## Phase-3 steering verdict (final PE/hybrid numbers)

- **Layout B beat layout A** (1.358x kernel-over-kernel, 1.209x -> 1.641x over baseline).
- **Winning build-engine realization: HYBRID (D1c)** — PE for `x_swap_neg` permutation,
  Scalar for `cos`/`sin` broadcasts.
- The op is now **DMA-co-limited**: DMA-active 0.689 ms vs the 0.516 ms pure-DMA ceiling
  (effBW 578 GB/s = 74% of 781 GB/s roofline, MBU ~40%). All compute engines hide under
  DMA: PE 50%, Vec 69%, Scl 47%.
- **Phase 3: chase DMA scheduling/burst-shape/MBU-raising levers on layout B** — not layout
  changes and not the PE path (already at 50%). The W-sweep took the fill/drain bubble to
  W=512 (plan's finest contract point).
