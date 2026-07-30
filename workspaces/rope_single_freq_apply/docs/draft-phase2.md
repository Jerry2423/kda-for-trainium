# rope_single_freq_apply — Phase 2 implementation draft (profile-driven optimization)

## Goal

Start from the promoted phase-1 kernel (`runs/rope_v1.py`, layout A, **0.9445 ms,
1.209x** over the 1.1418 ms baseline) and cut on-device latency **without ever
regressing correctness** (rel-L2 must stay `< 2e-5`, and today is exactly `0.0`
on all five seeds). Use the phase-1 profiler verdict — not a fresh guess — to pick
the lever, then explore each ranked direction for **at most five iterations**,
collecting before/after `verify.py` latency + the profiler engine digest to justify
keep / revise / reject.

## Phase-1 verdict recap (the source of truth — see `docs/phase1-bottleneck-digest.md`)

| metric | value | reading |
|--------|-------|---------|
| latency p50 | 0.9445 ms | current best (layout A, `W=2048`) |
| Vec active % | **91.6%** | six `tensor_tensor` passes on **64 of 128 lanes** |
| DMA active % | **93.5%** (sw-dma 99.6%) | co-limiter, but *not* bandwidth-bound |
| MBU % | 29.9% | HBM fabric bandwidth only 30% used |
| HBMrd / HBMwr / total | 268.44 / 134.22 / **402.65 MB** | **exactly the read-once/write-once floor** |
| eff BW | ~427 GB/s | « ~781 GB/s silu streaming roofline |

**Verdict: VECTOR-BOUND, co-limited with DMA-active, HBM at floor.** The pure-DMA
ceiling at 781 GB/s is `402.65 MB / 781 GB/s = 0.516 ms`; we measure 0.944 ms. The
**~0.43 ms gap is vector time that is not hidden under DMA** — the six vector passes
co-limit the wall clock. Two hard constraints fall out of this and gate every
direction below:

1. **HBM is already at the floor** (0% over). No traffic to remove; and any *added*
   HBM read (e.g. a second read of `cos`/`sin`, or reloading `x` twice) pushes above
   floor and costs DMA-active time on an already-93.5%-busy engine → forbidden.
2. **The prize is the ~0.43 ms of unhidden vector time.** The only way to shrink it
   is to do **fewer / wider vector passes**. The floor we are chasing is the
   ~0.516 ms DMA ceiling → best-case latency ~0.55–0.65 ms → **~1.75–2.0x** total.

## Why "fewer vector passes" means 128-partition packing (instruction-selection check)

`out0 = x0·cos − x1·sin`, `out1 = x0·sin + x1·cos` is **4 products + 2 combines = 6
`tensor_tensor` passes** in the 64-partition layout, and that is **minimal for 64
partitions**:

- No fused multiply-add collapses it. `cos`/`sin` are **full `[64, W]` tiles** (they
  vary along the free axis), *not* per-partition `[P,1]` scalars — so
  `nisa.tensor_scalar` (2 ops, but operands must be scalar/`[P,1]`) and
  `nisa.scalar_tensor_tensor` (`(data op0 scalar) op1 tile`) do **not** apply, and
  there is no `accumulate` flag on `tensor_tensor`. (Confirmed against
  `api-nki-isa-tensor.md` / `api-nki-isa-misc.md`.)
- `tensor_tensor` cost is **per free-element and independent of partition count** (128
  SIMD lanes run in parallel), so a `[128, W]` pass costs the same wall-clock as a
  `[64, W]` pass. Packing both output halves onto all 128 lanes therefore lets
  **3 passes do the work of 6**, roughly halving the co-limiting vector term.

This is the phase-1 digest's designated lever and the primary phase-2 direction.

### The packed-compute algebra (layout B)

`x_in` is **already** `[x0; x1]` on 128 partitions in HBM (`A`). Build:

- `A       = [x0; x1]`  (natural load, 1 DMA — no copy, cheaper than layout A's two loads)
- `Aswap±  = [−x1; +x0]` (swap the two 64-partition halves **and negate the top half**)
- `Ccos    = [cos; cos]` (broadcast `cos` 64→128 partitions)
- `Csin    = [sin; sin]` (broadcast `sin` 64→128 partitions)

Then **3 `tensor_tensor` over `[128, W]`**:

```
M1  = A     ⊙ Ccos      # [ x0·cos ;  x1·cos ]
M2  = Aswap± ⊙ Csin     # [-x1·sin ;  x0·sin ]
out = M1 + M2           # [ x0·cos − x1·sin ; x1·cos + x0·sin ]  ✓  == [out0; out1]
```

Baking the sign into `Aswap±` (rather than a `[−sin;+sin]` `Csin`) makes the final
combine a **single `add` across all 128 partitions** — a partition-dependent
add-top/sub-bottom is *not* one `tensor_tensor`, so the sign must live in exactly one
operand. Store `out` as one `[128, W]` DMA. Loads (x + cos + sin) and the store stay
**exactly at the HBM floor** — no extra HBM traffic.

**Exactness:** IEEE fp32 makes `a + (−b) ≡ a − b` and `(−x1)·sin ≡ −(x1·sin)`
bit-identically (negation is a sign-bit flip), so the packed order reproduces layout
A's arithmetic → rel-L2 should stay `0.0`. **Caveat to verify (see risks):** if the
broadcast/swap is built via a PE matmul, `nc_matmul` on fp32 may decompose internally
(bf16/tf32) and perturb the result — must re-check all 5 seeds.

### The crux: which engine builds `Aswap±`, `Ccos`, `Csin` — and does it hide?

These three are **cross-partition** data moves (replicate 64→128; swap halves). The
whole bet is that they land on an **idle** engine and hide under the DMA/vector floor.
Established fact (checked against the API + optimization knowledgebase): the only
engines that move data across partitions are **DMA**, **PE (matmul)**, and
**`nc_stream_shuffle`** — and:

- **`nc_stream_shuffle` is ruled out**: it runs on the **Vector Engine** (the engine
  we are trying to unload) and only shuffles **within 32-partition quadrants**, so it
  cannot express a 64↔64 half-swap (`api-nki-isa-misc.md`).
- **Vector/Scalar copies cannot cross partitions** — they are partition-locked. So a
  plain `nl.copy`/ScalarE activation can only *negate/scale in place*, not replicate
  or swap partitions.

That leaves two realizations to try (this is what the ≤5 iterations explore):

- **D1a — PE permutation matmul (idle engine, PE=0.2%).** `Ccos = [I;I]·cos`,
  `Csin = [I;I]·sin`, `Aswap± = S±·A` with `S± = [[0,−I],[I,0]]` are all 0/±1
  permutation matmuls on the **totally idle** Tensor engine → PSUM. Pro: uses dead
  silicon, so it can hide fully. Cons: (i) matmul output is PSUM and
  `tensor_tensor` forbids **both** operands in PSUM, so at most one packed operand per
  TT may be PSUM (needs careful SBUF/PSUM placement, maybe one extra copy);
  (ii) **fp32-matmul exactness risk** — must verify the gate still passes; (iii) three
  W-streaming matmuls are not literally free even when PE is idle.
- **D1b — SBUF→SBUF DMA broadcast/swap (no HBM traffic).** Replicate/swap partitions
  via SBUF-resident DMA (precedent: `TensorView.broadcast()` on the DMA path,
  `5f08e8cb`; GpSimd SBUF→SBUF `dma_engine.gpsimd_dma`, `dma-and-engines.md`). The
  negate (top-half sign) is exact on **ScalarE** (`activation(copy, scale=−1)`, idle).
  Pro: pure data movement → **arithmetically exact**, all-SBUF (no PSUM constraint).
  Con: adds **DMA-active** time on the already-93.5%-busy DMA engine — but MBU is only
  30%, so there is fabric-bandwidth headroom; the question the profiler answers is
  whether the added SBUF↔SBUF *active* time hides under the compute.
- **D1c — hybrid**: broadcast `Ccos`/`Csin` via DMA→SBUF (exact), build `Aswap±` via
  PE→PSUM, negate on ScalarE. Placement `M1 = A(SBUF)⊙Ccos(SBUF)`,
  `M2 = Aswap±(PSUM)⊙Csin(SBUF)` satisfies the "not both PSUM" rule with zero extra
  copies. Balances load across DMA + PE + Scalar.

## Ranked directions (benefit vs risk)

| # | direction | expected benefit | risk | iters |
|---|-----------|------------------|------|-------|
| **D1** | **Layout B: 128-partition packing (6→3 vector passes)** | **high** — toward the 0.516 ms DMA floor, up to ~1.75–2.0x | **high** — cross-partition build must hide on an idle engine; PE path has fp32-exactness + PSUM constraints | ≤5 (across D1a/b/c variants) |
| D2 | Finer free-axis `W` sweep on the best kernel | low-med — harvests DMA fill/drain bubble; effBW 427«781 hints at a pipelining gap | low — mask-free power-of-two `W`, no arithmetic change | ≤2 |
| D3 | Rejected-by-evidence (documented, ~0 iters) | — | — | 0–1 |

### D1 — Layout B (primary). Plan per iteration

1. **D1b first (lowest correctness risk).** Implement DMA/ScalarE broadcast+swap, all
   SBUF, keep `W=2048`. Score `--fast` (seed 42) then full 5-seed. Read the digest:
   did **Vec%** drop toward ~half and did **latency** fall below 0.80 ms?
2. **D1a / D1c** if D1b's DMA-active additions eat the win: move the builds onto the
   idle **PE** (and Scalar for negate). Verify the fp32 gate on the PE path *before*
   trusting latency.
3. Keep the variant with the lowest latency **that still passes all 5 seeds**.

**Keep / revise / reject rule for D1:**
- **Keep & promote** if latency `< 0.80 ms` (> 1.42x; clear win over 1.209x) with all
  seeds passing. Stretch success `< 0.65 ms` (~1.75x).
- **Revise** (try the next D1 variant) if `0.80–0.90 ms`: the packing helped but the
  build didn't fully hide — try moving the build to a more idle engine.
- **Reject, keep layout A** if `≥ 0.90 ms` after exhausting D1a/b/c, or if the gate
  fails on every exact-preserving variant. Record *why* (which engine the build landed
  on, Vec%/DMA%/PE% after) so phase 3 doesn't re-tread it.

### D2 — Finer free-axis `W` (secondary, cheap)

Sweep `W ∈ {1024, 512, 1536}` on whichever kernel D1 promotes (mask-free needs `W`
dividing `S = 2^18`; 1536 does not — restrict to powers of two `{1024, 512}` unless a
padded-tail variant is warranted). The silu campaign `[[kda-silu-progress]]` found
**finer wins** (optimum ~4 KB/partition burst) by amortizing the pipeline fill/drain
bubble. Here the digest calls this **secondary** — DMA is co-saturated in *active
time* at the floor, so the ceiling is small — but effBW 427 GB/s « 781 and MBU 30%
leave room for a bubble, and it is a near-zero-risk probe. **Keep** any `W` that
lowers latency with 5 seeds passing; else record the sweep and drop.

### D3 — Rejected by evidence (state, don't burn iterations)

- **bf16 / lower-precision downcast — REJECT.** RoPE is pure elementwise with **no
  reduction to average away rounding** (unlike `[[kda-rmsnorm-matmul-progress]]` where
  K-averaging rescued compensated-bf16). bf16 elementwise error ≈ 2⁻⁸ ≈ 4e-3 »» the
  2e-5 gate → fails. fp32 is mandatory.
- **Explicit ping-pong / wider burst-batching — REJECT (precedent).** The silu
  campaign found wider burst-batching + manual ping-pong **regressed**; `affine_range`
  already builds the software pipeline. At most one confirmatory probe if D1/D2 stall.

## Correctness guardrails (every candidate)

- rel-L2 stays `< 2e-5`; today it is `0.0` exact — treat any nonzero rel-L2 as a red
  flag and diff the arithmetic ordering. The **PE-matmul path (D1a)** is the one place
  packing can perturb fp32 → **always run the full 5-seed gate, not just `--fast`,
  before promoting a PE-path candidate.**
- Never raise HBM traffic above the 402.65 MB floor. Confirm `HBM_total_MB` in the
  digest is unchanged (broadcast/swap must be SBUF-resident or PE-permutation, never a
  re-read of `x`/`cos`/`sin` from HBM).
- Record every perf change in `benchmark.csv`, every candidate in `candidates.jsonl`
  (parent = `rope_v1`), and keep each direction's profiler digest under `profile/`.

## Measurement protocol

For each candidate, from `workspaces/rope_single_freq_apply/`:

```bash
# fast probe (seed 42) during iteration
python3 \
    ../../verify.py --op rope_single_freq_apply --candidate runs/<file>.py --fast
# full 5-seed + higher-iter latency before any promote
python3 \
    ../../verify.py --op rope_single_freq_apply --candidate runs/<file>.py
```

Read from the printed digest / `summary_metrics`: **latency p50, Vec%, DMA%, Scl%,
PE%, MBU%, HBMrd/HBMwr/total**. The decisive signal for D1 is **Vec% dropping toward
half while HBM_total stays at floor and PE%/DMA% absorb the build** — that is the
packing win materializing rather than just relocating.

## Risks / watch-items

- **The build may not hide (D1's core risk).** If the swap/broadcast lands back on the
  Vector engine (e.g. compiler routes a copy there) or serializes the DMA, we trade 3
  vector passes for 3 movement ops and net nothing. Mitigation: pin the build to
  PE/DMA/Scalar explicitly and read the per-engine digest, not just latency.
- **PSUM "not both operands" rule** forces careful SBUF/PSUM placement on the PE path
  (D1a/c); a mis-placement forces an extra copy that can erase the win.
- **fp32 PE-matmul may not be bit-exact** → gate risk on D1a; prefer the exact
  DMA/Scalar path (D1b) unless PE demonstrably wins on latency *and* passes 5 seeds.
- **Small absolute headroom.** The hard floor is 0.516 ms (2.21x max). Set
  expectations: a solid but sub-2x win is the realistic target; don't over-invest
  iterations past a clear plateau — bank the best correct kernel and move on.

## Deliverable

The best correct kernel promoted (with `benchmark.csv` / `candidates.jsonl` /
`profile/` evidence), plus a short verdict noting whether layout B beat layout A and
which build-engine realization won — steering phase-3 shape specialization.
