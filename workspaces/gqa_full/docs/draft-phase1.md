# gqa_full — Phase 1 Draft (first correct NKI kernel)

## Goal

Produce the first **correct** NKI kernel for `gqa_full` (NKIBench case 0):
grouped-query full (non-causal) softmax attention. Prioritize a clean,
obviously-correct kernel that passes the relative-L2 gate across all five seeds
`[0,21,42,63,84]` (`rel_tol=2e-5`, fp32); speed is phase-2/3 work. The fused
per-head design below should already clear the 15.579 ms baseline by a wide
margin, because the baseline spills the whole score matrix to HBM (see "Why the
baseline is slow").

## Operator

- Shapes/dtype (natural layout): `q (1,4096,16,128)`, `k,v (1,4096,8,128)` fp32.
  `B=1, N=4096, QH=16, KH=8, n_rep=QH/KH=2, D=128`.
- Reference (`gqa_full_..._numpy_2.py`), per query head `qh` (its kv head is
  `kh = qh // 2`, since `xk=np.repeat(k, n_rep=2, axis=head)`):
  ```python
  S    = q_h @ k_h.T / sqrt(D)                    # [N_q, N_k] scores
  A    = softmax_over_Nk(S)                        # row-softmax over the key axis
  O    = A @ v_h                                   # [N_q, D] context
  ```
  So it is exactly a **per-head `bmm_softmax` (scores + row-softmax) followed by
  a second matmul (context)**. `bmm_softmax` is a solved, promoted, 5-seed-correct
  sibling on this harness — reuse its core verbatim and bolt on the context matmul.

## Tiled layout — DERIVED AND EMPIRICALLY VERIFIED (rel-L2 2.4e-7)

`transform_to_nki_inputs` is reshape-only; `transform_nki_outputs` reshapes our
result to the ref shape. I reconstructed the reference output from the tiled
tensors through the index maps below and got **rel-L2 = 2.4e-7** vs the reference
(`/tmp/gqa_layout_check.py`, seed 0) — the maps are correct, not guessed:

- **q** `v1 = (32,128,16,128) = [t_q, p, qh, d]`:  `q[n_q, qh, d]` with
  `n_q = 128*t_q + p`. A fixed `(t_q, qh)` slice `v1[t_q,:,qh,:]` is
  `[p=n_q_sub(par)=128, d(free)=128]`.
- **k** `v2 = (1,8,4,128,8,128) = [0, a, b, c, kh, d]`:  `k[n_k, kh, d]` with
  **`n_k = 512*a + 128*b + c = 128*(4*a+b) + c`**. A fixed `(a,b,kh)` slice
  `v2[0,a,b,:,kh,:]` is `[c=n_k_sub(par)=128, d(free)=128]`; the 32 subtiles
  `(a in 8, b in 4)` tile the full `N_k=4096` at column offset `128*(4*a+b)`.
- **v** `v3 = (1,32,128,1024) = [0, t_v, p, kh*128+d]`:  `v[n_v, kh, d]` with
  `n_v = 128*t_v + p`. A fixed `t_v` slice `v3[0,t_v,:, kh*128 : kh*128+128]` is
  `[p=n_v_sub(par)=128, d(free)=128]` for head `kh`.
- **out** `v4 = (1,8,2,32,128,128) = [0, kh, grp, t_q, pos, d]`, maps to
  `ref[0, qh=2*kh+grp, n=128*t_q+pos, d]`. So `O_tile[pos(par)=128, d(free)=128]`
  stores directly to `v4[0, kh, grp, t_q, :, :]`.

Key alignment fact (verified): the **`n_k` (key) axis of the scores/attn matches
the `n_v` (value) axis of `v`** — both are absolute sequence position `n`. So the
context matmul iterates `n` subtiles `0..31` uniformly; k-columns built at offset
`128*(4*a+b)` and v-subtiles at `t_v` both index position `n`, and they line up.

## The two matmuls (nc_matmul = stationary.T @ moving, contraction on partition)

Both operands live in SBUF; the contraction dim must be on the **partition** axis
of both. `D=128` and `N_k` subtiles are 128, so each is a clean 128-contraction.

**Matmul 1 — scores `S[m_q, n_k] = sum_d q_h[m_q,d]·k_h[n_k,d]`** (contract `d`):
- result `[m_q(par)=128, n_k(free)]` ⇒ softmax reduces over the **free** axis (good,
  same as `bmm_softmax_v4`). ⇒ `stationary=[d(par),m_q(free)]`,
  `moving=[d(par),n_k(free)]` — **both q and k need `d` on the partition axis**.
- q native is `[p=m_q(par), d(free)]` → transpose once per `(kh,grp,t_q)` via the
  identity `nc_matmul(is_transpose=True, is_moving_onezero=True)` idiom → `q_t[d,m_q]`.
- k native is `[c=n_k_sub(par), d(free)]` → transpose the 32 subtiles once per `kh`
  into a resident `k_t[d(par)=128, n_k(free)=4096]` (reused across `grp` and all
  32 `t_q` ⇒ 64 reuses; column offset `128*(4*a+b)`).
- 8 chunks of `N_CHUNK=512` (one fp32 PSUM bank) build `score[128,4096]`.

**Matmul 2 — context `O[m_q,d] = sum_{n_k} A[m_q,n_k]·v_h[n_k,d]`** (contract `n_k`):
- result `[m_q(par)=128, d(free)=128]` ⇒ `stationary=[n_k(par),m_q(free)]`,
  `moving=[n_k(par),d(free)]`.
- `A` from softmax is `[m_q(par)=128, n_k(free)=4096]` → for each of 32 `n_k`
  subtiles, transpose `A[:,128*j:+128]` → `A_t[n_k_sub=128, m_q=128]` (identity
  idiom), and **accumulate** into one `[128,128]` PSUM bank over `j=0..31`
  (`v24 += nc_matmul` idiom in the baseline).
- **v needs NO transpose**: v native `[p=n_v_sub(par)=128, d(free)=128]` is already
  the required moving layout. Load `v_h` once per `kh` into `v_sb[p(par)=128,
  t_v*128+d]` (so subtile `j` = `v_sb[:, 128*j:+128]`), reused across `grp`+`t_q`.

## Softmax epilogue (verbatim `bmm_softmax_v4`, + the `1/sqrt(D)` scale)

Over the `N_k=4096` free axis, fp32, max-shifted for overflow safety:
```
score  = score * scale                 # scale = 1/sqrt(128) = 0.08838835; reproduces the
                                        #   reference's exact op order (attn = q@kT * scale, THEN max)
neg_max = tensor_reduce(max, score, axis=free, negate=True)   # -row_max, negate folds the *-1 step
exp_t   = activation(exp, score, bias=neg_max, scale=1.0)     # exp(score - row_max)
row_sum = tensor_reduce(add, exp_t, axis=free)                # explicit Vector add (do NOT fuse
                                        #   into activation reduce_res — measured +75% on the sibling)
recip   = reciprocal(row_sum)
A       = tensor_scalar(exp_t, mul=recip)                     # per-row [128,1] scale over free axis
```
The explicit `score*scale` full-width multiply reproduces the reference's operation
order bit-for-bit (I verified this path at rel-L2 2.4e-7). Folding the scale into
the `activation(scale=)` param (and scaling `neg_max`) removes that full-width op
and is numerically equivalent — noted as a **phase-2 lever**, not used in phase-1.

## Kernel plan (`gqa_full_v1`)

```
out = ndarray((1,8,2,32,128,128), fp32, shared_hbm)
identity_local[128,128] = load(shared_constant(I128))         # once, for all transposes

for kh in affine_range(8):
    # --- per-head shared operands (reused across grp and all 32 t_q) ---
    k_t[d=128, 4096]  : for a in 8, b in 4:                    # 32 k transposes
        k_sub[c=128,d=128] = load(v2[0,a,b,:,kh,:])
        k_t[:, 128*(4*a+b):+128] = copy( nc_matmul(k_sub, identity, is_transpose=True) )  # [d, n_k_sub]
    v_sb[p=128, 4096] : for t_v in 32:                         # no transpose, direct load
        v_sb[:, 128*t_v:+128] = load(v3[0,t_v,:, kh*128:kh*128+128])   # [p=n_v_sub, d]

    for grp in affine_range(2):                                # qh = 2*kh + grp
        for t_q in affine_range(32):
            # scores
            q_sb[p=128,d=128] = load(v1[t_q,:,qh,:])
            q_t[d=128,m_q=128] = copy( nc_matmul(q_sb, identity, is_transpose=True) )
            score[128,4096]:
                for c in 8: acc[128,512] = nc_matmul(q_t, k_t[:,512*c:+512]); score[:,512*c:+512]=copy(acc)
            # softmax over the 4096 free axis (block above)  -> A[128,4096]
            # context
            O_psum[128,128] (accumulate):
                for j in 32:
                    A_t[n_k=128,m_q=128] = copy( nc_matmul(A[:,128*j:+128], identity, is_transpose=True) )
                    O_psum += nc_matmul(A_t, v_sb[:,128*j:+128])    # [m_q, d]
            store(out[0,kh,grp,t_q,:,:], copy(O_psum))              # [pos, d]
return out
```

## Why the baseline is slow (and what we fix)

Baseline latency **15.579 ms**. It materializes the whole per-head/per-tile score
matrix across the grid (`v13/v14 ≈ [32,2,4,2,2,4,128,512]` ≈ the full `QH*N*N`
scores) and does a chunked online max/sum over that resident set — far larger than
SBUF, so scores **spill to HBM and round-trip** (write scores, read back for
exp/normalize) on top of the output write.

**Fix = per-head, per-m_q fusion** (the same win `bmm_softmax` had over its
baseline). One m_q tile's full score row is only `[128,4096] fp32 = 16 KB/partition`,
trivially resident. Build the row, softmax it in place, immediately consume it in
the context matmul, discard. Scores never touch HBM; traffic drops toward the
read-once/write-once floor (read q+k+v ≈ 100 MB, write out ≈ 33 MB). Note: this
already achieves "scores never fully materialize" at the whole-matrix level —
true flash-style online softmax over `n_k` chunks is a further phase-2 lever, and
is **not needed to fit** here (the 16 KB row fits with room to spare).

## SBUF budget (per partition, one `kh` live)

`k_t` 16 KB + `v_sb` 16 KB + `score` 16 KB + `exp_t/A` 16 KB + `q_t` 0.5 KB +
`A_t` 0.5 KB + `O` 0.5 KB + identity 0.5 KB + `[128,1]` scalars ≈ **66 KB** of the
~208 KB usable ⇒ no spill; HBM stays at the read-/write-once floor.

## Correctness argument

- Layout maps verified end-to-end at **rel-L2 2.4e-7** against the reference.
- Both matmuls are single-128-contraction `nc_matmul` (fp32 emulation floor
  ~1.8e-7 on the sibling cores); softmax is max-shifted fp32 in the reference's
  exact op order. Well under the `2e-5` gate.
- `n_k`↔`n_v` position alignment verified; k/v loaded once per `kh` and correctly
  shared across the 2 query groups (`n_rep=2` = the `qh=2*kh+grp` map).

## Phase-2 / phase-3 outlook (not this phase)

- Two-phase transpose-all schedule (pack all q/A transposes up front, then stream
  matmuls) — the promoted `bmm_softmax_v4`/`bmm_v2` lever; hides softmax behind a
  longer matmul stream. M-block sizing (`M_SUB`) sweep.
- Fold `1/sqrt(D)` into `activation(scale=)` to drop the full-width scale op.
- Flash-style online softmax over `n_k` chunks (deferred; not needed to fit).
- bf16x2 split on the two big matmuls if PE-bound (the matmul-family lever), gated
  on the fp32 rel-L2 floor headroom under `2e-5`.

## Validate / score

```bash
python3 \
    ../../verify.py --op gqa_full --candidate runs/gqa_full_v1.py --fast
```
