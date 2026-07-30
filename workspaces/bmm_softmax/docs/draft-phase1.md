# bmm_softmax — Phase 1 Draft (first correct fused NKI kernel)

## Goal

Produce the first **correct** NKI kernel for `bmm_softmax` (NKIBench case 2): a
batched matmul followed by a row-softmax over the N axis. Prioritize a clean,
obviously-correct kernel that passes the relative-L2 gate across all five seeds
`[0,21,42,63,84]`; speed is phase-2/3 work. It must beat nothing to be a valid
phase-1 base, but the fused design below should already clear the baseline by a
wide margin (see "Why this beats the baseline").

## Operator

- Shapes/dtype: `lhs v1 (16,4096,64)=(B,M,K)`, `rhs v2 (16,64,4096)=(B,K,N)`,
  `out (16,4096,4096)=(B,M,N)`, fp32. `B=16, M=4096, K=64, N=4096`.
- Reference (`bmm_softmax_..._numpy_1.py`):
  ```python
  x       = lhs @ rhs                       # (B,M,N)
  max_x   = np.max(x, axis=2, keepdims=True)   # row max over N
  exp_x   = np.exp(x - max_x)
  sum_exp = np.sum(exp_x, axis=2, keepdims=True)
  out     = exp_x / sum_exp                  # softmax over N, per (b,m) row
  ```
  So softmax is over the **N=4096 axis**, independently for each of the `B*M`
  rows. `transform_to_nki_inputs` is reshape-only; `transform_nki_outputs`
  reshapes our result to the ref shape, so returning `(16,4096,4096)` is an
  identity reshape (this is what the sibling `bmm_v1` did and it passed L2).

## The matmul core is a solved sibling

`bmm` (NKIBench case 2, workspace `../bmm/`) has the **identical** GEMM
`(B16,M4096,K64,N4096)` and a promoted, 5-seed-correct kernel. Reuse its core
verbatim; only the epilogue changes (softmax instead of a plain store).

Facts inherited from the `bmm` core (see `../bmm/benchmark.csv`):
- `nc_matmul(stationary, moving) = stationary.T @ moving`; contraction dim `k`
  must be on the **partition** axis of both operands, both operands in SBUF.
- `K=64 <= 128` ⇒ the whole contraction is **one** Tensor-Engine pass per output
  tile (single `nc_matmul`, no K-accumulation loop).
- `moving = rhs[b]` tile `[k=64(par), n(free)]` loads directly (v2[b] is `[k,n]`).
- `stationary` must be `[k=64(par), m_in=128(free)]`; a loaded lhs tile is
  `[m_in=128(par), k=64(free)]`, so transpose it once via the identity
  `nc_matmul(is_transpose=True, is_moving_onezero=True)` idiom → `[k=64, m_in=128]`,
  copy to SBUF.
- fp32 `nc_matmul` on this core measured rel-L2 `1.83e-7` (fp32 emulation floor),
  far under the `2e-5` gate. Adding fp32 softmax vector ops keeps us at that floor.

## Why the baseline is slow (and what we fix)

Baseline latency is **7.29ms**, ~2.9x the pure-`bmm` baseline (2.55ms), despite
computing the same 1GB output. The baseline materializes essentially the entire
`(B,M,N)` score matrix in SBUF (`v11 ≈ [4,8,16,4,2,128,512]`) and does a chunked
online max/sum across it — that resident set is ~1GB and **spills to HBM**, so
the scores round-trip through HBM twice (write scores, read back for exp/divide)
on top of the 1GB output write.

**Fix = fusion.** Process one m-tile at a time. Its full score row is only
`[128, 4096] fp32 = 16 KB/partition`, trivially resident. Compute the row,
softmax it in place, store the normalized row, discard. Scores never touch HBM;
HBM traffic drops to the once-each floor (read lhs+rhs ≈ 34 MB, write out ≈ 1074 MB).

## Kernel plan (phase-1: `bmm_softmax_v1`)

Batch-outer, mirror `bmm_v1`'s structure exactly through the matmul, then replace
the plain store with a **full-row fused softmax epilogue** over the 4096 free axis.

```
out = ndarray((16,4096,4096), fp32, shared_hbm)
identity_local[128,128] = load(shared_constant(I128))     # once, for the transpose

for b in affine_range(16):
    rhs_sb[64, 4096] = load(v2[b])                         # resident, 16 KB/part
    for mt in affine_range(32):                            # 32 = 4096/128 m-tiles
        lhs_sb[128, 64]  = load(v1[b, 128*mt:.., :])
        psum_t[64,128]   = nc_matmul(lhs_sb, identity_local,
                                     is_transpose=True, is_moving_onezero=True)
        lhs_t[64,128]    = copy(psum_t)                    # stationary [k, m_in]

        # --- build the full score row [128, 4096] in SBUF (8 chunks of 512) ---
        score[128, 4096]                                   # 16 KB/part
        for c in affine_range(8):                          # 8 = 4096/512
            acc[128,512] = nc_matmul(lhs_t, rhs_sb[:, 512*c:512*c+512])  # 1 PSUM bank
            score[:, 512*c:512*c+512] = copy(acc)          # PSUM -> SBUF

        # --- fused softmax over the free axis (N=4096), all in SBUF, fp32 ---
        row_max[128,1] = tensor_reduce(max, score, axis=free)
        neg_max[128,1] = tensor_scalar(row_max, mul=-1.0)  # bias = -row_max
        exp_t[128,4096] = activation(exp, score, bias=neg_max, scale=1.0)  # exp(score - row_max)
        row_sum[128,1]  = tensor_reduce(add, exp_t, axis=free)
        recip[128,1]    = reciprocal(row_sum)
        out_t[128,4096] = tensor_scalar(exp_t, mul=recip)  # per-row multiply
        store(out[b, 128*mt:.., :], out_t)                 # one 4096-wide store
```

### Why full-row (not the baseline's chunked online softmax)

Because the full row (16 KB/part) is trivially resident, we do **not** need the
flash-style online max/sum with per-chunk rescaling that the baseline uses. One
`tensor_reduce(max)`, one `activation(exp, bias=-max)`, one `tensor_reduce(add)`,
one `reciprocal`, one `tensor_scalar(*recip)` — each a single instruction over the
whole 4096-wide axis (SBUF allows up to 32767 free elements; the 512 PSUM cap does
not apply since we reduce the SBUF score tile, not PSUM). This is both simpler and
strictly fewer vector passes than the online scheme.

### Correctness reasoning

- **Math matches the reference step-for-step**: row max over N, subtract, exp,
  row sum over N, divide. Softmax is invariant to the max shift; subtracting the
  row max both matches the reference numerics and prevents any exp overflow
  (scores ~ N(0,64), std≈8; `exp(score-max) <= 1`, sum <= 4096, all fp32-safe).
- **Axis is correct**: each m-tile row tile is `[m_in=128(par), n=4096(free)]`;
  reducing the free axis is reducing over N = reference `axis=2`. Store maps
  `out[b, 128*mt+p, n]` (partition→m, free→n), row-major = ref `(B,M,N)`.
- **fp32 throughout**: matmul at the emulation floor (`1.83e-7` on the sibling)
  plus fp32 exp/sum/divide ⇒ expected rel-L2 « `2e-5`. No bf16 anywhere in phase 1.
- **`activation` semantics** (confirmed via NKI API docs): computes
  `op(data*scale + bias)`; `bias` may be a `[128,1]` per-partition vector on
  NeuronCore-v3 (trn2) — exactly the per-row `-row_max`. The baseline uses this
  same `bias=[128,1]` pattern, so it is a supported idiom on this target.

### Resident SBUF budget (per partition, fp32) — no spill

`rhs_sb` 16 KB + `score` 16 KB + `exp_t` 16 KB + `lhs_sb`/`lhs_t` < 1 KB ≈ **~48 KB**
of the 192 KB budget. (`out_t` can reuse `exp_t` in place via the `tensor_scalar`
dst to save 16 KB if the compiler wants it; not required.) Well clear of spill, so
HBM stays at the read-once/write-once floor.

## Risks / things to verify during the RLCR loop

1. **`activation` fused-reduce vs separate reduce.** The API also supports fusing
   the row-sum into the `exp` `activation` (`reduce_op=nl.add, reduce_res=...`) in
   one Scalar-Engine pass. Phase-1 keeps the sum as a **separate** `tensor_reduce`
   for maximum clarity/verifiability; the fusion is a phase-2 lever (saves one
   Vector pass), noted but not taken now.
2. **`tensor_scalar` with a `[128,1]` vector operand** for the per-row multiply by
   `recip` (and the `*-1` for `neg_max`). The baseline uses `tensor_scalar` with a
   `[128,1]` `operand0` (`v16`) for exactly the final per-row multiply, so the
   pattern is supported. Confirm the API arg form on first compile.
3. **Output declaration `(16,4096,4096)`** returned directly (vs the baseline's
   `(16,32,128,4096)`). `bmm_v1` used the flat `(B,M,N)` form and passed the gate;
   `transform_nki_outputs` reshapes to ref shape (identity here).
4. **PSUM free-axis cap.** Keep every reduce/activation on the **SBUF** `score`/
   `exp_t` tiles (4096 allowed), never on a PSUM tile (512 cap). The 8 chunk
   matmuls each land in a `[128,512]` PSUM bank and are copied out immediately.

## Acceptance for phase 1

- 5-seed relative-L2 PASS (`< 2e-5`) via
  `python3 ../../verify.py --op bmm_softmax --candidate runs/bmm_softmax_v1.py` (drop `--fast` for the promote measurement).
- Record the perf row in `benchmark.csv` and the candidate in `candidates.jsonl`
  (parent = `baseline:bmm_softmax_B16_K64_M4096_N4096_0.py`), with the profiler
  digest (PE/Vec/Scl/DMA %, MFU, HBM rd/wr) so phase 2 knows the bottleneck engine.

## Phase-2 / phase-3 outlook (not implemented now)

- **Phase 2 (profile-driven):** adopt the sibling's proven **two-phase transpose
  M-block=32** schedule (`bmm_v2`, 1.253x on pure bmm) to remove per-tile
  transpose→matmul serialization; fuse the row-sum into the `exp` activation to
  drop a Vector pass. Bottleneck will likely be DMA (1 GB output write) or the
  matmul PE time — read the digest to decide. Softmax vector work (exp over 1 GB
  of scores) is new pressure on Scalar/Vector vs pure bmm; watch Scl/Vec %.
- **Phase 3 (shape specialization):** N=4096 is exactly 8×512 (PSUM-bank) and
  M=4096 exactly 32×128 — edge-free, like the `bmm` sibling, so classic shape
  specialization has little surface. Precision (bf16x2 for the matmul) is a
  possible lever but softmax is exp of scores, so any matmul error is amplified
  through exp — the `2e-5` gate is tighter here than for a plain GEMM; treat bf16
  with caution and gate on an offline rel-L2 sim first (as the sibling did).
```
