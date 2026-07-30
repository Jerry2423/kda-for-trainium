# transpose_matmul — Phase 1 Draft (first correct NKI kernel)

## 1. Operator & shapes

`out = lhs^T @ rhs`, fp32.
- `lhs`: stored **(K, M) = (2048, 4096)** — i.e. K-major.
- `rhs`: **(K, N) = (2048, 10944)**.
- `out`: **(M, N) = (4096, 10944)**.
- MACs = M·N·K = 4096·10944·2048 = **9.17e10**.

NKIBench tiles the operands (partition dim ≤ 128) via `transform_to_nki_inputs`:
```
lhs (2048,4096) -> reshape (128, 16, 4096)   # v1
rhs (2048,10944) -> reshape (128, 16, 10944) # v2
out                                          # v3 = (32, 128, 10944)
```

## 2. The layout insight (the main lever, already used by the baseline)

For the reshape `(K, ·) -> (128, 16, ·)`, index maps as `k = k_in*16 + kt`
(verified numerically: `reshape(128,16,·)[k_in,kt,·] == orig[k_in*16+kt, ·]`).
So:

- `v1[k_in(128,par), kt(16), m(4096)]` — **K is on the partition axis**.
- `v2[k_in(128,par), kt(16), n(10944)]` — same.

The Tensor Engine's `nisa.nc_matmul(stationary, moving)` computes
`stationary.T @ moving` and requires the **contraction dim on the partition axis
of BOTH operands**. Here the contraction is K, and K is *already* on the
partition axis of both v1 and v2. Therefore:

> **No transpose is needed.** `lhs^T @ rhs` where lhs is (K,M) is exactly what
> `nc_matmul` computes when we feed `stationary = v1[:, kt, m0:m0+128]`
> (= `[k_in=128, m_sub=128]`) and `moving = v2[:, kt, n0:n0+W]`
> (= `[k_in=128, n=W]`): result `= [m_sub, n]`, accumulated over the 16 `kt`
> tiles. Full contraction = `k_in(128) × kt(16) = 2048 = K`. ✔

This contrasts with my prior `matmul_v1`/`bmm_v2`, where lhs arrived M-major and
needed an identity `nc_matmul(is_transpose=True)` per tile. Here that whole
transpose stage disappears — the transposed-lhs operand is a free byproduct of
the input layout.

## 3. Bottleneck: compute-bound fp32 GEMM

- Baseline latency = **4.849615 ms** (from `baselines.json`), 9.17e10 MACs
  ⇒ ~1.9e13 MAC/s effective — right at the fp32 PE floor observed in my
  matmul-family work (`[[kda-bmm-progress]]`, `[[kda-matmul-progress]]`).
- HBM traffic even with generous re-loads is far under the roofline: lhs 33.5 MB
  (1×) + rhs 89.6 MB (re-loaded a few ×) + out 179 MB ≈ 0.5–0.7 ms at ~781 GB/s
  (`[[kda-silu-progress]]` roofline). **DMA is not the wall.**
- ⇒ Phase 1 correctness cost ≈ baseline; the real speedup levers are compute
  (phase 2 bf16×2, phase 3 shape specialization). Phase 1 target: a clean,
  fully-correct kernel at roughly baseline latency (≥ ~1.0×), not a speed win.

## 4. Phase-1 kernel design (`runs/tmm_v1.py`)

Clean **M-block-outer streaming** GEMM, no transpose, single-bank PSUM tiles.

Constants:
```
M_TILES = 32   (4096/128)     K_TILES = 16   (contraction tiles)
M_BLK   = 8 tiles (=1024 m)   M_BLOCKS = 4
N       = 10944               N_CHUNK = 456   N_CHUNKS = 24   (456*24 = 10944 exactly)
```

**N_CHUNK = 456** is chosen deliberately: it is an exact divisor of N and ≤ 512
(one fp32 PSUM bank is 512 wide). This gives **zero tail masking anywhere** —
the single largest correctness-bug surface (the baseline's `mask=... >= 0`
arithmetic) is eliminated for the phase-1 correct baseline. The ~2% free-axis
under-fill vs 512 is a phase-2 concern, not a phase-1 one.

```
out = nl.ndarray((32, 128, 10944), fp32, shared_hbm)

for mb in affine_range(M_BLOCKS):                 # 4 m-blocks of 1024
    # Resident lhs block for these 1024 m-rows: [k_in=128, kt=16, 1024]
    #   (16*1024*4 = 64 KB/partition). Loaded ONCE per m-block; read-only after.
    lhs_blk = sbuf[par_dim(128), 16, 1024]
    for kt in affine_range(16):
        lhs_blk[:, kt, :] = load(v1[:128, kt, 1024*mb : 1024*mb+1024])

    for c in affine_range(N_CHUNKS):              # 24 n-chunks of 456
        # rhs chunk, all 16 kt for this n-slice: [k_in=128, kt=16, 456]
        #   (16*456*4 ≈ 28.5 KB/partition). Loaded ONCE per (mb,c), reused
        #   across the 8 m-subtiles below (so rhs is read 4× total, once per mb).
        rhs_chunk = sbuf[par_dim(128), 16, 456]
        for kt in affine_range(16):
            rhs_chunk[:, kt, :] = load(v2[:128, kt, 456*c : 456*c+456])

        for s in affine_range(M_BLK):             # 8 m-subtiles in the block
            acc = nl.zeros(par_dim(128), 456, psum)     # 1 PSUM bank
            for kt in affine_range(16):
                acc += nc_matmul(
                    lhs_blk[:, kt, 128*s : 128*s+128],  # stationary [k_in,128]
                    rhs_chunk[:, kt, :])                # moving     [k_in,456]
            out_sb = copy(acc -> sbuf)                  # [128, 456]
            store(out[8*mb + s, :128, 456*c : 456*c+456], out_sb)
```

Result of the inner matmul: `stationary.T @ moving = [m_sub=128, k_in] @ [k_in,
456] = [m_sub=128, 456]`, accumulated over the 16 `kt` tiles → the full K=2048
contraction. Output tile `[m_sub(par), n(free)]` stores directly into
`out[8*mb+s, :, n0:n0+456]`, matching `v3[m_tile, m_in, n]`.

**Resident SBUF budget/partition**: lhs_blk 64 KB + rhs_chunk ~28.5 KB (+ small
out_sb/load temps) ≈ 93 KB, well under 192 KB even with double-buffering of the
per-chunk rhs. **PSUM**: one 456-wide bank live per `s` (+ compiler rotation) —
far under the 8 banks. No spill; HBM stays near the once-lhs / 4×-rhs / once-out
floor.

**Why M-block-outer (not the dead-simple m-tile-outer of `matmul_v1`)**: reloading
rhs per m-tile would be 32× rhs traffic (~2.9 GB → ~3.7 ms DMA, dangerously close
to the PE floor). Blocking M into 4 resident blocks caps rhs re-reads at 4×
(~0.36 GB), keeping the kernel comfortably PE-bound. M_BLK is a correctness-neutral
knob (8 matches baseline granularity; 16 fits SBUF and would halve rhs traffic) —
left as an explicit tunable for phase 2/3.

## 5. Correctness argument

- Math is **bit-exact fp32**: same operands, same K-accumulation order semantics
  as a standard tiled GEMM; `nc_matmul` fp32 accumulate in PSUM, one copy to
  SBUF, one store. No dtype casts, no approximation.
- No masking ⇒ no partial-tile arithmetic to get wrong; every tile is full-size.
- Gate = NKIBench relative-L2 `||v_k − v_r||₂ < 2e-5·||v_r||₂` across seeds
  `[0,21,42,63,84]`. A faithful fp32 GEMM sits at ~0 rel-L2 (well inside 2e-5);
  `verify.py` gates on `l2_norm_passed` — trust it.

## 6. Evidence plan

Run from `workspaces/transpose_matmul/`:

1. **Correctness/score (fast)** while iterating:
   ```
   python3 \
       ../../verify.py --op transpose_matmul --candidate runs/tmm_v1.py --fast
   ```
2. **Promote**: re-run without `--fast` (full 5-seed / higher-iter) before recording.
3. Record the perf row in `benchmark.csv`
   (`timestamp,op,candidate,parent,passed,latency_ms,speedup,notes`).
4. Record the candidate in `candidates.jsonl` (DAG: `parent =
   baseline:transpose_matmul_M4096_K2048_N10944_0.py`), with `seeds`,
   `latency_ms`, `baseline_latency_ms=4.849615`, `speedup`, `rel_l2_gate=2e-5`,
   and a `structure` note.
5. Save the profiler digest (MFU / PE / Vec / Scl / DMA / HBMrd / HBMwr, printed
   by `verify.py`) under `profile/` — this is the round-0 bottleneck baseline
   that phase 2 works against. **Confirm PE ≈ dominant engine** (validates the
   compute-bound hypothesis before choosing the phase-2 lever).

## 7. Forward-looking (NOT phase 1 — for context only)

- **Phase 2 lever candidate: bf16×2 3-product split** on the matmul. My
  matmul-family results are split: it *won* big on `matmul_add_rmsnorm`
  (`[[kda-matmul-add-rmsnorm-progress]]`, +4.88×, because per-instruction fp32
  *rate* on a moving-512 GEMM dominates, not the emulation instruction *count*),
  but *lost* on swiglu (`[[kda-swiglu-progress]]`, fp32 emulates in ~2 passes).
  This kernel is a large moving-N (456–512) fp32 GEMM — the *favorable* case —
  so bf16×2 is the leading phase-2 hypothesis, to be MEASURED not assumed. The
  rel-L2 headroom (2e-5 gate; compensated-bf16 lands ~4.5e-6 offline, ~1.5e-5
  on-device in quadrature with the fp32 floor) must be re-checked on-device.
- **Phase 3 lever candidate: shape specialization** — M_BLK width (8→16),
  N_CHUNK (456 vs 512+mask), and stationary/moving reuse tuned to the measured
  PE-idle gap.

**Phase-1 deliverable: `runs/tmm_v1.py`, correct across all 5 seeds, ~baseline
latency, PE-bound confirmed in `profile/`.**
