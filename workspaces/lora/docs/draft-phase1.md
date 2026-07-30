# lora — Phase 1 draft: first correct NKI kernel

## Goal

Produce the FIRST correct NKI kernel for `lora`, passing the NKIBench relative-L2
gate (`2e-5`, fp32) across all five seeds `[0,21,42,63,84]`. Prioritize a clean,
fully-understood kernel over speed — but structure it so the low-rank update is
**fused into the base GEMM's output accumulation** (the prompt's stated win), since
that fused shape costs no more than the un-fused one and is the natural correct form.

## Operator

`out = x@w + (x@a)@b`  (fp32)

| tensor | math shape | role |
|--------|-----------|------|
| `x` | (M=4096, K=5120) | activations |
| `w` | (K=5120, N=12288) | base weight |
| `a` | (K=5120, R=128) | low-rank down-projection |
| `b` | (R=128, N=12288) | low-rank up-projection |
| `out` | (M=4096, N=12288) | result |

A large base matmul `x@w` (dominant: 4096×5120×12288) plus a cheap low-rank residual
`(x@a)@b` (R=128). `x@a` is (M,R); `@b` lifts it back to (M,N).

## Tiled layout (from the reference `transform_to_nki_inputs`) — VERIFIED numerically

The kernel entry is `kernel(v1, v2, v3, v4)` and returns `v5`:

- `v1` (x): `(8, 4, 128, 40, 128)` = `[m_hi, m_lo, m_in, k_tile, k_in]`.
  Row `m = (m_hi*4 + m_lo)*128 + m_in`, col `k = k_tile*128 + k_in`.
  There are `8*4 = 32` M-tiles of 128 rows, `40` K-tiles of 128.
- `v2` (w): `(40, 128, 12288)` = `[k_tile, k_in, n]`. Already `[k_in(contraction), n]`.
- `v3` (a): `(40, 128, 128)` = `[k_tile, k_in, r]`. Already `[k_in(contraction), r]`.
- `v4` (b): `(128, 12288)` = `[r, n]`. Already `[r(contraction), n]`.
- `v5` (out): `(8, 4, 128, 96, 128)` = `[m_hi, m_lo, m_in, n_tile, n_in]`.
  Row `m = (m_hi*4 + m_lo)*128 + m_in`, col `n = n_tile*128 + n_in`. `96` N-tiles of 128.

I confirmed each decomposition with a numpy reshape/index round-trip (all True).

Note the M-tile index in `v1`/`v5` is a **2-level** `(m_hi, m_lo)` pair (8×4), unlike
the sibling `matmul` case whose x/out were a flat `(32, 128, ...)`. Everything else
(contraction on partition for w/a/b, N free) matches the sibling `matmul` exactly.

## Tensor-engine contract (recap from sibling `matmul_v1`)

`nisa.nc_matmul(stationary, moving) = stationary.T @ moving`, with the **contraction
dim on the PARTITION axis** of both operands, both operands resident in SBUF.

- `w`, `a`, `b` all already have their contraction dim (`k_in` resp. `r`) as the
  first/partition axis in the tiled layout — load them and use directly, **no transpose**.
- `x` tiles arrive as `[m_in(par), k_in(free)]` — contraction `k_in` is on the FREE
  axis, so each must be transposed to `[k_in(par), m_in(free)]` via the identity
  `nc_matmul(is_transpose=True)` idiom before use. This transposed `x` tile
  (`lhs_t`) is the shared operand for BOTH `x@w` and `x@a`.

## Design — M-outer, single fused PSUM accumulation

Reuse the proven M-outer structure of `matmul_v1`. Constants: `M_TILES=32`,
`K_TILES=40`, `R=128`, `N=12288`, `N_CHUNK=512` (one fp32 PSUM bank), `N_CHUNKS=24`.

Preload once (reused across all 32 M-tiles; both are small):
- 128×128 identity into SBUF (transpose helper).
- `a` fully resident: `a_local[K_TILES, par_dim(128), 128]` = 40·128·4 = 20 KB/part.
- `b` streamed per N-chunk for phase 1 (`[r=128, 512]`, 2 KB/part transient); note
  preloading `b` resident (24·512·4 = 48 KB/part) is a clean phase-2 lever since `b`
  is reused across all M-tiles.

Per M-tile `mt` (decoded to `m_hi = mt//4`, `m_lo = mt%4` for `v1`/`v5` indexing):

1. **Transpose x once (shared).** For each of 40 K-tiles: load `x` tile
   `[m_in(par), k_in(free)]`, transpose → `lhs_t[kt] = [k_in(par), m_in(free)]`
   in resident SBUF (`[K_TILES, par_dim(128), 128]`, 20 KB/part).

2. **Low-rank down-projection, transpose-free.** Compute `tT = (x@a)ᵀ`, a single
   `[R=128, m_in=128]` tile, by accumulating over K:
   `tT += nc_matmul(stationary=a_local[kt] [k_in,R], moving=lhs_t[kt] [k_in,m_in])`
   → `a.Tᵀ... = [R, m_in]`. Verified numerically that this equals `(x@a).T`. Keep
   `tT` in SBUF (copy out of its PSUM bank) so it is available to every N-chunk.

3. **Per N-chunk (24 chunks of 512): base GEMM + fused low-rank into ONE bank.**
   ```
   acc = psum.zeros([m_in=128, 512])
   for kt in 40:                     # base x@w
       load w tile [k_in=128, 512]
       acc += nc_matmul(stationary=lhs_t[kt] [k_in,m_in], moving=w_tile [k_in,512])
   # FUSE low-rank in the SAME bank — no HBM round-trip for the intermediate:
   load b tile [r=128, 512]
   acc += nc_matmul(stationary=tT [r=128, m_in], moving=b_tile [r=128, 512])
   copy acc -> out_sb ; store to v5[m_hi, m_lo, :, 2*c : 2*c+... ]
   ```
   `nc_matmul(stationary=tT [R,m_in], moving=b_tile [R,n]) = tT.T @ b = (x@a)@b`
   `= [m_in, n]`, identical output layout to the base GEMM, so it accumulates
   directly. One extra matmul per N-chunk (24 total) + 40 down-proj matmuls per
   M-tile — ~6% over the 960 base matmuls/M-tile. Cheap, as expected.

   **Store index detail:** N_CHUNK=512 spans 4 output N-tiles of 128. `v5`'s N axis
   is `[n_tile(96), n_in(128)]`. Either (a) store with a reshaped/strided write across
   the 4 sub-tiles the 512-chunk covers, or (b) — simpler and less bug-prone for a
   first correct kernel — set `N_CHUNK=128` so each chunk maps to exactly one `n_tile`
   (`nc = c`), giving 96 chunks. Start with option (b) for phase-1 correctness; option
   (a) / 512-wide is a phase-2 tile-width lever. (Even at 128-wide, each PSUM tile is
   one bank's worth; correctness is unaffected, only matmul granularity.)

## Why this is correct and fusion-clean

- Pure fp32 `nc_matmul` throughout — matches the sibling `matmul_v1`, which passed
  the same `2e-5` gate with a fp32 floor ~1e-6. No dtype tricks in phase 1.
- The low-rank result is added into the base GEMM's PSUM bank **before eviction**, so
  the `(x@a)@b` intermediate never touches HBM — exactly the prompt's stated win, and
  it is also the simplest correct form (one output store per tile, no separate add pass).
- `x` is transposed exactly once per M-tile and reused by both GEMMs; `w/a/b` need no
  transpose. Minimal instruction surface.

## SBUF/PSUM budget (per partition, trn2 ~192 KB SBUF)

- `lhs_t` 20 KB + `a_local` 20 KB + identity 0.5 KB + `tT` 0.5 KB + streamed
  `w`/`b` tiles (2 KB each, double-bufferable) ≈ 45 KB — comfortable.
- PSUM: one 512-wide fp32 bank live for `acc`; a transient bank for the transpose
  and for `tT`. Well within the 8-bank budget.

## Validation & evidence

Score from `workspaces/lora/`:
```
python3 \
    ../../verify.py --op lora --candidate runs/lora_v1.py --fast
```
Gate on `l2_norm_passed` across all 5 seeds (drop `--fast` before promoting).
Record the perf row in `benchmark.csv`; add a `candidates.jsonl` row
(`id=lora_v1`, `parent=baseline`) with seeds, latency, speedup, worst rel-L2, and the
per-engine / MFU / HBM digest under `profile/`.

## Risks / watch-items

- **M-tile 2-level index** (`m_hi,m_lo`) is the one layout difference from the sibling
  `matmul` — get the `v1`/`v5` indexing right (verified above); a swapped pair silently
  scrambles rows and fails L2.
- **`tT` PSUM→SBUF copy** must happen before the N-chunk loop reuses PSUM banks;
  keep `tT` in SBUF, not PSUM, across the chunk loop.
- **Store granularity** for N_CHUNK=512 across 4 `n_tile`s — the reason phase 1 uses
  N_CHUNK=128 (one chunk = one `n_tile`) to keep the first correct kernel unambiguous.

## Phase-2/3 outlook (not this phase)

- Widen `N_CHUNK` to 512 (fewer, larger matmuls; strided 4-subtile store).
- Preload `b` resident (48 KB/part) — reused across all M-tiles, kills 32× reloads.
- M-blocking to amortize `w` reloads (the base GEMM reloads all of `w` per M-tile).
- bf16x2 3-product split on the base GEMM if it stays PE-bound and the fp32 speedup
  floor is the ceiling (the promoted lever on every sibling GEMM: matmul 1.274×,
  transpose_matmul 1.334×). Guard with the offline rel-L2 simulator as usual.
