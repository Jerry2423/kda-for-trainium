# Phase 2 · task1 — cross-partition-build primitive semantics (confirmed)

Source: `nki-api-reference` skill (arch + API refs) cross-checked by Codex (high effort,
198 s). Both agree on all six claims. This is the semantic foundation for the layout-B
build (AC-2 / AC-3 / AC-1.1).

## The governing rule: Cross-Partition Connectivity (trn2 / NeuronCore-v3)

`start_partition` allocator + cross-partition rule (`trainium_inferentia2_arch.md`
lines 1096–1154):

- **Vector / Scalar / GpSimd compute engines are partition-locked for `num_partition > 64`**:
  a 128-partition op must start at partition 0 and maps each src partition 1:1 to the same
  dst partition — **no cross-partition movement**.
- For `32 < num_partition ≤ 64` these engines support **cross-half movement**
  (src `0:64` ↔ dst `64:128`); for `≤ 32`, cross-quadrant.
- **DMA engines are the exception** — descriptor-driven, not partition-locked; `nisa.dma_copy`
  moves HBM↔SBUF, within HBM, or **within SBUF** (`nki-dma-overview.md` line 12).

## The six confirmed claims

1. **Connectivity rule** as stated above. ✔
2. **Broadcast `[cos]→[cos;cos]` and half-swap `[x0;x1]→[x1;x0]` cannot be a single
   128-partition compute op**, but each = **two ≤64-partition cross-half moves**
   (`tensor_copy`/`activation` on Vec/Scl/GpSimd), OR `nisa.dma_copy` SBUF→SBUF (DMA). ✔
   (GpSimd cannot touch PSUM — keep GpSimd builds SBUF-only, which is our case.)
   `nl.load`/`nl.store` are HBM↔SBUF only — **no SBUF→SBUF load/store**.
3. **`nc_stream_shuffle`** runs on the **Vector** engine and is **mod-32 quadrant-confined**
   (`shuffle_mask[i]` = src partition mod 32) → **cannot express a 64↔64 half-swap**. ✔
   (Ruled out — it also loads the very engine we are unloading.)
4. **`nisa.activation(op=nl.copy, scale=−1.0)`** on **Scalar** = exact fp32 negate for finite
   inputs (Scalar math is fp32; `×(−1.0)` is an IEEE sign flip, no rounding), when data and
   dst are fp32. ✔ (NaN-payload/sNaN wording aside — out of scope; NKIBench inputs are finite
   `np.random.normal`.) The negate FUSES into the top-half cross-half copy in one Scalar op.
5. **`nc_matmul` fp32 is NOT documented bit-exact** (mixed-precision; only *accumulation* is
   fp32; tf32 path exists). Operands must be **SBUF**, dst **PSUM**, **moving free ≤ 512**
   (so `W > 512` forces the matmul into 512-wide sub-tiles). A 0/±1 permutation build
   (`Ccos=[I;I]·cos`, `Aswap±=S±·A`) is **genuine `[128,≤512]` streaming matmul work**, not a
   free copy → **must pass the full 5-seed gate before trusting** (AC-1.1). ✔
6. **`tensor_tensor`** permits SBUF/SBUF, SBUF/PSUM, PSUM/SBUF — **not PSUM/PSUM**. ✔

## Design steer (folded into the plan): D1b = Scalar-first, not DMA-first

The plan draft leaned toward SBUF→SBUF **DMA** for the exact D1b build. Both the docs and
Codex say **start on the Scalar engine instead**:

- DMA is already **93.5 % active** and co-limiting the wall clock; low MBU (30 %) proves
  *fabric-bandwidth* headroom but NOT DMA-*sequencer/descriptor* slack. SBUF→SBUF DMA adds
  active work to the co-limiting engine (and ~256·S·4 B of extra SBUF↔SBUF movement).
- **Scalar is idle (0.18 %)** and its copy/`activation(copy, scale=−1)` is exact fp32.

**The bet, quantified:** the 4 cross-half builds (broadcast cos, broadcast sin, swap-lo,
swap-hi) are each a `[64, W]` op (64 of 128 lanes). If they land on **Vector**, that is
3 packed `[128,W]` passes + 4 `[64,W]` copies ≈ **7×S vector-equiv > layout A's 6×S → LOSS**
(the AC-4 "bounced back onto Vector" failure). Routed to **Scalar**, Vector drops 6→3 passes;
the risk is that 4 `[64,W]` Scalar copies ≈ 4×S Scalar-equiv could make **Scalar** the new
co-limiter (still < layout A's 6×S vector, but > the 3×S vector floor). Mitigations if so
(revise order): split the 4 builds across **Scalar + GpSimd** (both idle) so no single build
engine exceeds the 3×S floor; then escalate to the **PE** permutation build (D1a/D1c).

**Decisive keep/revise/reject metric (AC-4/AC-5):** the **build engine's active %/time after
the change**, read together with the **Vec% drop** and **latency**. Keep if Vec% drops
materially AND latency falls (< 0.80 ms) AND the build engine does not become the new
critical engine AND HBM stays at floor. Revise (redistribute the build / move to PE) if Vec%
drops but latency does not (build serialized or the build engine went dominant). Reject (keep
layout A) if ≥ 0.90 ms after exhausting D1a/b/c.

## Implementation-shape consequences

- **A** = natural `nl.load(x_in[0:128, …])` into a `[128, W]` tile (x0→0:64, x1→64:128):
  1 HBM load, cheaper than layout A's two.
- **Ccos/Csin**: `nl.load` cos/sin into the `[0:64]` half of a `[128,W]` tile, then one
  cross-half Scalar copy `→ [64:128]`. HBM reads cos/sin **once** (floor preserved).
- **Aswap±**: `Aswap[0:64] = −A[64:128]` (cross-half Scalar `activation(copy, scale=−1)`);
  `Aswap[64:128] = A[0:64]` (cross-half Scalar copy). Sign baked into `Aswap±` → final
  combine is a single 128-partition `add` (AC-3).
- **3 `tensor_tensor` over `[128,W]`**: `M1=A⊙Ccos`, `M2=Aswap±⊙Csin`, `out=M1+M2`; single
  `[128,W]` store. HBM = x(128 MiB rd) + cos(64) + sin(64) + out(128 wr) = 402.65 MB floor.
- **fp32 exact**: DMA/Scalar builds are pure data movement + a sign flip → rel-L2 expected
  `0.0` (AC-1). Only the PE path (D1a/D1c) risks perturbation → AC-1.1 full-seed gate.
