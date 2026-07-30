# swiglu — Phase 1 draft (first correct NKI kernel)

## 1. Operator and contract

**Op:** `swiglu`, NKIBench case `2`. Fused SwiGLU feed-forward: two input projections
(up, gate), a SiLU gate on the gate projection multiplied by the up projection, then a
down projection.

**Reference computation** (`AccelOpt/NKIBench/reference/swiglu_M4096_N3072_K1024_numpy_2.py`):

```python
def forward(x, w_up, w_down, w_gate):
    up_feature   = np.matmul(x, w_up)                            # (M,N)
    gate_feature = np.matmul(x, w_gate)                          # (M,N)
    activated    = gate_feature / (1 + np.exp(-gate_feature))    # = gate * sigmoid(gate) = SiLU(gate)
    return np.matmul(activated * up_feature, w_down)             # (M,K)
```

So with `h = SiLU(gate) * up`:  `up = x@w_up`, `gate = x@w_gate`, `out = h @ w_down`.

**Shapes / dtype (all fp32):**
- `x`      : `(M=4096, K=1024)`
- `w_up`   : `(K=1024, N=3072)`
- `w_gate` : `(K=1024, N=3072)`
- `w_down` : `(N=3072, K=1024)`   — down-projection contracts over N, emits K
- output   : `(M=4096, K=1024)`

Three GEMMs, one elementwise SiLU-gate on the `(4096, 3072)` intermediate `h`.

**Signature (matches baseline):** `def kernel(v1, v2, v3, v4)`. From the reference's
`transform_to_nki_inputs`, the argument order is the **append order**, which is NOT the
`forward()` order:

| arg | tensor  | tiled shape             | meaning |
|-----|---------|-------------------------|---------|
| v1  | x       | `(8, 4, 128, 8, 128)`   | input activations |
| v2  | w_up    | `(8, 128, 3072)`        | up weight |
| v3  | **w_down** | `(24, 128, 1024)`    | down weight |
| v4  | **w_gate** | `(8, 128, 3072)`     | gate weight |

(The baseline kernel confirms this: it loads `v4` first as `w_gate_local` and `v2` as
`w_up_local`, and stores into `v5 (32,128,1024)`.)

**Correctness gate:** relative-L2 `||v_k - v_r||_2 < 2e-5 * ||v_r||_2`, fp32, across seeds
`[0, 21, 42, 63, 84]`. (`verify.py` gates on `l2_norm_passed` — trust it.)

**Score:** `baseline_latency / candidate_latency`, p50 on-device, single core,
`--disable-dge --logical-nc-config=1`. Baseline latency = **2.0742 ms** (baselines.json).

## 2. The tiled layout — exact decode

All reshapes are row-major (numpy default). Decoding each input to its `[partition, free]`
role for the Tensor Engine:

- **v1 = x**, `(8, 4, 128, 8, 128)`. Flattened offset of `v1[a,b,mi,d,e]` corresponds to
  `x[(a*4+b)*128 + mi, d*128 + e]`. So the M-tile index is `mt = a*4 + b` (i.e.
  `a = mt//4`, `b = mt%4`), partition axis is `mi` (m_in ∈ [0,128)), and the free axis
  `(d,e)` walks `k ∈ [0,1024)`. **Crucially, `(8,4,128,8,128)` is row-major identical to
  `(32, 128, 1024)`** — the exact silu-style no-copy `reshape` view. So I will do
  `x3 = v1.reshape((32, 128, 1024))` and read `x3[mt, m_in(par), k(free)]`. Each x tile is
  `[m_in(par)=128, k(free)=1024]`.

- **v2 = w_up**, `(8, 128, 3072)`: `v2[kt, ki, n] = w_up[kt*128 + ki, n]` →
  `[k_in(par)=128, n(free)=3072]`. This is the matmul **moving** operand directly (no
  transpose; contraction `k_in` already on partition).

- **v4 = w_gate**, `(8, 128, 3072)`: same layout as w_up → `[k_in(par), n(free)]`, moving
  operand directly.

- **v3 = w_down**, `(24, 128, 1024)`: `v3[nt, ni, kp] = w_down[nt*128 + ni, kp]` →
  `[n_in(par)=128, kp(free)=1024]`. The down GEMM contracts over N, so `n_in` on partition
  is exactly right: w_down is the **moving** operand for the down GEMM (no transpose).

- **out = v5**, `(32, 128, 1024)`: `v5[mt, mi, kp] = out[mt*128 + mi, kp]` →
  `[m_in(par), kp(free)]`. Same M-tile numbering as x. ✓

**Tensor-Engine rule** (proven across the matmul / add_rmsnorm_matmul siblings):
`nisa.nc_matmul(stationary, moving) = stationary.T @ moving`, with the contraction dim on
the **partition** axis of *both* operands, both resident in SBUF, moving free-dim ≤ 512
(one fp32 PSUM bank).

**Two transposes are unavoidable in fp32**, because in each GEMM one operand has its
contraction dim on the *free* axis:
1. **up/gate**: contraction is `k`. x tile is `[m_in(par), k(free)]` — k is on the free
   axis. Transpose x → `xT[k_in(par), m_in(free)]` (8 sub-tiles of `[128,128]`), then
   `nc_matmul(stationary=xT[kt], moving=w[kt]) = [m_in, k_in] @ [k_in, n] = [m_in, n]`.
   **This single transpose of x is SHARED by both the up and gate GEMMs** — x is read once
   and transposed once per M-tile, then reused as the stationary operand against both
   `w_up` and `w_gate`.
2. **down**: contraction is `n`. h is produced as `[m_in(par), n(free)]` — n is on the free
   axis. Transpose h → `hT[n_in(par), m_in(free)]` (24 sub-tiles of `[128,128]`), then
   `nc_matmul(stationary=hT[nt], moving=w_down[nt]) = [m_in, n_in] @ [n_in, kp] = [m_in, kp]`.

Both transposes use the standard identity-matmul idiom
(`nisa.nc_matmul(tile, identity, is_transpose=True, is_moving_onezero=True)` → PSUM →
`nl.copy` to SBUF), identical to `matmul_v1` and `add_rmsnorm_matmul_v1`.

## 3. Why the baseline is slow — and where Phase-1 already improves

The NKIBench baseline (`kernels/swiglu_M4096_N3072_K1024_0.py`) does two wasteful things:

1. **It spills the entire intermediate `h` to HBM and reloads it.** It computes the
   up/gate/SiLU-gate result into an HBM scratch `v20 = _spill_163 (3,8,8,128,512)` with
   `nl.store`, then in the down phase reloads it (`_reload_166`, `nl.load(v20...)`). That is
   an extra `4096*3072*4 B = 50 MB` write **plus** 50 MB read of a tensor that fits in SBUF.
2. **Its transpose structure is baroque** — it transposes x with a per-`(i3,i4)` identity
   matmul into `v11/v12`, driven by a leading `i0 in range(3)` loop that appears to recompute
   x's transpose **three times** (once per N-third), rather than transposing x once and
   reusing it. The gate and up phases (`i0` loop and `i7` loop) each independently reload and
   re-transpose x's sub-tiles.

**My Phase-1 kernel keeps `h` fully resident in SBUF and transposes x exactly once per
M-tile, shared across up+gate.** Sizing (per M-tile, one row-block of 128):
- `h` tile is `[128, 3072]` fp32 = `3072*4 = 12 KB/partition`. Comfortably resident (trn2
  usable SBUF ≈ 208 KB/partition). No HBM spill of `h` at all.
- This alone removes ~100 MB of `h` spill+reload traffic vs the baseline.

This will not be a huge speedup on its own (see §5 — the op is PE-bound), but it is the
clean, obviously-correct structure that later phases build on.

## 4. Weight residency — the key structural constraint (differs from siblings)

`add_rmsnorm_matmul_v1` won 3.75x by holding its **single** 8 MB weight fully resident. That
is **not possible here**: the three weights are
- `w_up`   : `1024*3072*4 = 12 MB` = `3072*8*4 / … ` → **96 KB/partition** (8 k-tiles × 3072 × 4 B)
- `w_gate` : 12 MB = **96 KB/partition**
- `w_down` : `3072*1024*4 = 12 MB` = **96 KB/partition** (24 n-tiles × 1024 × 4 B)

Total **288 KB/partition > 208 KB usable** — they do not all fit resident simultaneously,
even before counting x/h/PSUM staging. So Phase 1 will **stream weights** and accept the
resulting weight-DMA cost (the same trade `matmul_v1` accepted at 0.855x before M-blocking
lifted it). This is the honest Phase-1 baseline; amortizing weight DMA is the explicit
Phase-2 lever (§6).

**Loop structure (Phase 1, M-outer, one M-tile at a time — block factor B=1):**

```
x3 = v1.reshape((32,128,1024))                      # no-copy view
load 128x128 identity into SBUF (once)
for mt in affine_range(32):                          # 32 M-tiles
    # ---- load + transpose x for this M-tile (SHARED by up and gate) ----
    x_sb  = load x3[mt]                              # [m_in(par)=128, k=1024]
    xT[kt] = transpose(x_sb[:,128*kt:...]) for kt in 0..7   # 8x [k_in,m_in], via identity matmul

    # ---- up and gate projections, N-chunk by N-chunk (6 chunks of 512) ----
    for c in affine_range(6):                        # N=3072 / 512
        up_acc   = zeros[128,512] psum
        gate_acc = zeros[128,512] psum
        for kt in affine_range(8):
            w_up_sb   = load v2[kt, :, 512*c:...]     # [k_in,512]  moving
            w_gate_sb = load v4[kt, :, 512*c:...]     # [k_in,512]  moving
            up_acc   += nc_matmul(xT[kt], w_up_sb)    # [m_in,512]
            gate_acc += nc_matmul(xT[kt], w_gate_sb)  # [m_in,512]
        # ---- SiLU-gate fused at eviction: h = SiLU(gate) * up ----
        sg = activation(op=nl.silu, data=gate_acc)    # [m_in,512], Scalar Engine, fp32
        h_sb[:, 512*c:...] = multiply(sg, up_acc)     # [m_in,512] -> resident h [128,3072]

    # ---- transpose h (24 sub-tiles) then down projection ----
    hT[nt] = transpose(h_sb[:,128*nt:...]) for nt in 0..23   # 24x [n_in,m_in]
    for c2 in affine_range(2):                        # K=1024 / 512
        out_acc = zeros[128,512] psum
        for nt in affine_range(24):
            w_down_sb = load v3[nt, :, 512*c2:...]     # [n_in,512] moving
            out_acc  += nc_matmul(hT[nt], w_down_sb)   # [m_in,512]
        store out_sb -> v5[mt, :, 512*c2:...]
```

**SiLU is fused at PSUM eviction**, exactly matching the reference's
`gate/(1+exp(-gate)) * up`. `nl.silu` is the single-instruction Scalar-Engine SiLU
(`x*sigmoid(x)`) used by the promoted `silu` sibling — one activation call replaces the
baseline's 4-op `exp → +1 → reciprocal → multiply` sequence and is exactly equal to it. The
subsequent `* up` is a Vector-Engine `nl.multiply`. Both are on the `[128,512]` chunk and are
cheap relative to the matmuls.

**SBUF budget check (per M-tile, B=1):** identity `[128,128]`=0.5 KB; `x_sb` 4 KB; `xT`
8×`[128,128]`=4 KB; `h_sb` 12 KB; `hT` 24×`[128,128]`=12 KB; streamed w tiles a few
`[128,512]`=2 KB each. Total well under 208 KB/partition. **PSUM:** up_acc + gate_acc are 2
banks; the transpose PSUM tile is 1 bank; out_acc 1 bank — ≤ 8 banks. ✓

## 5. Theoretical floor — why ~1.0x is the honest Phase-1 expectation

Total matmul MACs: up + gate + down = `2 * (M*K*N) + M*N*K` on the fused sizes =
`4096*1024*3072 (up) + 4096*1024*3072 (gate) + 4096*3072*1024 (down)` = **3 × 1.29e10 ≈
3.86e10 MACs**, plus the two transposes (`4096*1024 + 4096*3072 ≈ 2.1e7` element-moves, tiny).

The `matmul` sibling measured fp32 throughput ≈ `2.58e11 MAC / 13.35 ms ≈ 1.93e13 MAC/s` on
this exact trn2 profiler path. So the **fp32 PE floor for swiglu ≈ 3.86e10 / 1.93e13 ≈
2.0 ms** — essentially the baseline latency (2.074 ms). **The baseline is already close to the
fp32 systolic floor.** A correct, clean fp32 kernel therefore lands near **~1.0x**; the
weight-DMA of B=1 streaming may pull it slightly under 1.0x (as with `matmul_v1`'s 0.855x
before M-blocking). Phase 1's job is *correctness first*, not a win.

This matches the sibling pattern (rmsnorm_matmul, matmul, add_rmsnorm_matmul all hit
PE≈94–100% at the fp32 floor). The **real** swiglu win comes in Phase 3 (§6).

## 6. What Phases 2–3 target (recorded so Phase 1 stays disciplined)

- **Phase 2 — amortize weight DMA via M-blocking (the `matmul_v2_b4` lever).** Process a
  block of `B` M-tiles together so each streamed weight tile (`w_up`, `w_gate`, `w_down`) is
  loaded once and reused across `B` stationary `xT`/`hT` tiles, cutting weight HBM traffic
  ~`B`-fold. `matmul` found **B=4 optimal** (1.017x; B=8/16 regressed on SBUF/PSUM pressure);
  swiglu has more live SBUF per M-tile (resident `h`, two accumulators), so the sweet spot may
  be smaller — sweep B∈{2,4} and measure. Possibly also hold `w_down` (or one projection)
  resident while streaming the others.
- **Phase 3 — compensated bf16×2 split-matmul on all three GEMMs (the proven ceiling-breaker).**
  The sibling `rmsnorm_matmul` (1.066x→1.363x) and `add_rmsnorm_matmul` (→4.632x) both broke
  the fp32 PE floor with a 3-product compensated bf16 split (`hi = bf16(v)`,
  `lo = bf16(v - hi)`, keep 3 of 4 cross-products, drop `lo*lo`), accumulating in fp32 PSUM.
  It must be **offline-gated first** (numpy multi-seed sim reproducing the exact scored draw)
  to confirm worst-case rel-L2 ≪ 2e-5 before spending remote runs. **Caution specific to
  swiglu**: error compounds across *three* chained GEMMs and a nonlinearity in the middle, so
  the offline sim must model the full pipeline (up/gate split → SiLU → down split), and the
  margin may be tighter than the single-GEMM siblings. This is the phase where swiglu's score
  can move well above 1.0x.

## 7. Correctness & numerical notes

- **fp32 end-to-end.** No dtype games in Phase 1 — every load, matmul, activation, and store
  is fp32, so the only error is the hardware fp32 systolic-accumulation floor (the siblings
  measured rel-L2 ≈ 1.4–1.5e-5 at K=1024, comfortably under 2e-5). swiglu's down GEMM
  contracts over N=3072 (3× longer accumulation) so its floor may be marginally higher; the
  5-seed run will confirm it clears 2e-5. If any seed is marginal, the SiLU-gate and the two
  projections are all exact fp32, so the only lever is nothing to change — it either passes or
  reveals a layout bug.
- **SiLU exactness.** `nl.silu` computes `x*sigmoid(x) = x/(1+e^-x)`, algebraically identical
  to the reference's `gate/(1+exp(-gate))`. Confirm on the first run that rel-L2 is at the
  fp32 floor (not elevated), which would validate the fused activation against the reference's
  4-op form.
- **No masking needed.** M=4096=32×128, K=1024=8×128, N=3072=24×128=6×512, K_out=1024=2×512 —
  every tile is exact and rectangular. No edge/tail handling.
- **Layout verification first.** Before trusting perf, I will confirm the v1↔x reshape and the
  v2/v3/v4↔weight roles with a tiny probe (or by reasoning against the baseline's own indexing,
  which I've already cross-checked in §1–2). The reshape-view claim (`(8,4,128,8,128)` ≡
  `(32,128,1024)`) is the one assumption most worth a sanity check, mirroring how the `silu`
  sibling validated its `(128,32,7168)`≡`(128,224,1024)` view.

## 8. Deliverable for Phase 1

`runs/swiglu_v1.py` — a single `@nki.jit def kernel(v1, v2, v3, v4)` implementing §4:
M-outer, x transposed once and shared by up+gate, up/gate streamed over 6 N-chunks with SiLU
fused at eviction into a **resident** `h`, h transposed and down-projected over 24 n-tiles
into the `[m_in, 1024]` output, weights streamed (B=1). Score with the 5-seed full run;
record in `benchmark.csv` + `candidates.jsonl` (parent = `baseline:swiglu_M4096_N3072_K1024_0.py`).
Expected: **full 5-seed PASS, ~0.9–1.0x** — a correct, clean base for the Phase-2 M-blocking
and Phase-3 bf16-split wins.
