# matmul_add_rmsnorm — Phase 2 draft (profile-driven optimization)

## 0. TL;DR

Phase-1 `matmul_add_rmsnorm_v1` is correct and **PE-bound at the fp32 systolic floor**
(PE=96%, MFU=46%, 0.9612 ms, 3.920x over the 3.768493 ms baseline; full-5-seed PASS).
Every other engine is hidden under that floor (Vec 15%, Scl 9%, DMA 20%; HBMrd 84 MB is
already the ~80 MB single-pass floor). On the bf16-native trn2 PE array a *correct* fp32
GEMM runs multiple internal bf16 passes, which is exactly why MFU is capped near ~46%.
**The only lever that cuts wall-clock is cutting PE time**, and the only correctness-viable
way to do that on this shape is the **compensated bf16x2 split-matmul** — the same win that
was PROMOTED on both siblings (`rmsnorm_matmul`: 1.066x→1.363x, +28%;
`add_rmsnorm_matmul`: 3.754x→4.632x, +23%).

The single new risk vs the siblings — the bf16 matmul error entering the norm path — is
already de-risked by a zero-remote-spend offline numpy sim (§3): worst bf16-only rel-L2 =
**4.454e-6** across 7 seeds, predicted device quadrature **1.526e-5** (~1.31x under the
2e-5 gate). SBUF fits (§5). Plan is **one** promotion candidate (v2, the bf16x2 split built
directly on v1) plus a costed accuracy-repair fallback (v2b, the 4-product split) if the
measured on-device rel-L2 crosses a marginal threshold.

## 1. Phase-1 baseline and the measured bottleneck

Promoted `runs/matmul_add_rmsnorm_v1.py` (0.9612 ms = 3.920x), full-5-seed PASS. Profiler
digest (from `benchmark.csv` / `candidates.jsonl`):

| engine | v1 | reading |
|---|---|---|
| **PE** | **96%** | saturated — the binding constraint |
| MFU | 46% | fp32 emulation rate penalty (bf16-native array runs fp32 as multi-pass) |
| Vec | 15% | RMSNorm reduce + output scale — hidden |
| Scl | 9% | square + rsqrt activations — hidden |
| DMA | 20% | HBMrd 84 MB (x 32 + w 16 + z 32 ≈ 80 MB one-pass floor), HBMwr 34 MB — hidden |

**Diagnosis: the kernel is PE-bound and the PE is running at the fp32 emulation rate.** Two
independent lines of evidence make this a *floor*, not a schedule artifact:
1. All non-PE engines are well under 50% and HBMrd is already at the single-pass floor
   (84 MB) — there is no memory-traffic or Vec/Scl headroom to reclaim; the earlier
   full-width-vs-chunked output-store A/B already confirmed the epilogue is not the binding
   engine (chunking it *raised* PE to 98% and regressed 6.4%).
2. Both siblings sat at the identical fp32 floor (PE 94–97%, MFU 44–46%) and both were only
   broken by dropping the matmul to bf16-class precision.

The theoretical picture (see `kernel-cost-analysis` for the cost model): a
single-pass bf16 matmul of this shape has a PE floor of roughly `M·N·(K/128)` dst-free
cycles ≈ the sibling's ~218 µs class scaled for K=2048; the fp32 emulation costs ~2.1x
that, which is what 0.9612 ms reflects. Cutting to a **3-product** bf16 split is ~3 passes
vs fp32's ~4 (trn2 emulates fp32 matmul in ~4 bf16 passes), i.e. a ~0.75x PE-time ratio →
the directional expectation is **~0.75 ms / ~5.0x**. Promotion depends only on a *measured*
out-of-noise win + full-5-seed PASS, not on hitting that number.

## 2. Why bf16x2 is the ONLY real lever (what is closed)

Enumerate and rank the directions; everything except the split is closed by the profile or
by sibling precedent:

| direction | expected benefit | verdict |
|---|---|---|
| **compensated bf16x2 split-matmul** | **cut PE time ~0.75x → ~5x** | **PRIMARY — the only lever that touches the binding engine** |
| off-PE transpose (dma_transpose / nc_transpose / load_transpose2d) | move the 512 identity transposes off PE | **CLOSED** on both siblings: dma_transpose is fp32/bf16 SBUF→SBUF infeasible (hwdge needs src.shape[0]==16, swdge needs HBM src), nc_transpose(vector) regressed +2%, load_transpose2d hidden. And the transposes are ALREADY hidden under the PE-bound matmul here (PE=96%), so there is no idle to reclaim. |
| output-store structure (full-width vs per-chunk) | Vec/Scl overlap | **CLOSED in phase-1 R1**: full-width store measured 6.4% faster; chunking the pure-SBUF epilogue only adds Vec/Scl+store ops. |
| M-blocking / loop reorder | fill/DMA overlap | **N/A**: w is fully resident (single HBM pass), so there is no weight-reload bubble to block against; DMA is 20% and hidden. |
| N_CHUNK sizing | PSUM bank utilization | **CLOSED**: 512 = one fp32 PSUM bank = maximal moving-free width; siblings confirmed optimal. |
| fp32 loop reorder / g-into-w fold | — | **N/A / no-op here** (see §4: g is free-axis, does NOT fold; inv_rms does NOT commute out). |

So Phase 2 is essentially one direction explored carefully, with a costed fallback — not a
scatter of micro-opts.

## 3. De-risking the bf16 split for THIS op (the mirror twist)

**This is the one place matmul_add_rmsnorm genuinely differs from the siblings**, and it must
be measured, not assumed. In the norm→GEMM siblings the bf16 error entered *only the matmul*;
`inv_rms` was computed from the *exact fp32 activation* and commuted out as a post-scale, so
the norm path was pristine. Here the op is **GEMM → add → norm**: the bf16 matmul error lands
in `y = x@w + z`, and `y` feeds **both** `inv_rms[m] = 1/sqrt(mean_n(y²)+eps)` **and** the
output numerator `y·g`. The error propagates through the norm — a path no sibling sim ever
exercised.

I built and ran an offline numpy sim (`runs/offline_bf16_split_sim.py`, zero remote spend;
evidence `profile/matmul_add_rmsnorm_offline_bf16_split_sim.txt`, candidates node
`matmul_add_rmsnorm_offline_bf16_split_sim`) that reproduces the EXACT scored input (seed 42,
draw order x→w→z→g, z=(M,N), g=(N,), eps=1e-5) and this composite epilogue:

| model | rel-L2 vs fp32 reference | meaning |
|---|---|---|
| fp32 control (exact matmul + norm+scale) | **0.000e+00** | seed/draw-order/dtype/eps/formula all match — model is exact |
| plain single-limb bf16 | 2.350e-3 | fails the gate 117x — confirms single-limb is out |
| **bf16x2 3-product** (drop lo@lo) | **4.452e-6** (seed 42); **4.454e-6 worst over 7 seeds** | ~4.5x under the 2e-5 gate; ~3.3x below the fp32 sibling's own on-device 1.46e-5 |
| bf16x2 4-product (keeps lo@lo) | 3.491e-6 | sizes the dropped cross term — negligible improvement |

**The composite norm path does NOT blow up the error.** In fact it partially self-cancels: a
coherent relative perturbation `δ` in `y` scales the numerator by ~`δ` and `inv_rms` by ~`−δ`,
so `out = y·inv_rms` is first-order insensitive to a common-mode scaling of `y`. The measured
4.454e-6 is the residual after that cancellation. The dropped `lo@lo` term is negligible
(3-product 4.454e-6 vs 4-product 3.491e-6), so **3-product is the right choice**.

**KEY calibration — QUADRATURE (learned from the sibling):** the offline number is the
*bf16-only* term. The on-device rel-L2 combines the hardware fp32 floor (present in v1 too —
trn2 emulates "fp32" with rounding the numpy sim can't see, ~1.46e-5 on the sibling) with the
bf16 error **in quadrature**:
`sqrt(1.46e-5² + 4.454e-6²) = 1.526e-5`. This is not a hypothesis — the sibling
`add_rmsnorm_matmul` predicted 1.526e-5 by the identical method and *measured 1.528e-5 on
device*. So the expected on-device rel-L2 here is **~1.53e-5, ~1.31x under the 2e-5 gate**:
comfortable, but the margin is real and thin enough to keep the 4-product repair costed (§6).

## 4. Two op-specific simplifications vs the sibling bf16-split kernel

The sibling `add_rmsnorm_matmul_v3` needed a **v2 enabler refactor** first (g-into-w' fold +
inv_rms post-scale eviction) because its g was per-K (contraction axis) and its inv_rms
commuted out. **Neither applies here**, so we skip the enabler and diff directly on v1:

1. **g is NOT folded into w.** g is length-N on the *output* free axis, applied *after* the
   norm (`out = y·g/rms`). Folding `g[n]` into `w[k,n]` would scale `y` *before* the norm and
   change `rms = sqrt(mean(y²))` — algebraically wrong. g stays exactly where v1 has it: a
   `[1,N]→[128,N]` broadcast multiply on the output. (Contrast the sibling, whose per-K g
   folded cleanly into the resident weight.)
2. **inv_rms does NOT commute out to a post-scale eviction.** The norm reduces over N, so the
   *entire* `[128,N]` row `y` must be assembled in SBUF before `inv_rms` is known — we cannot
   apply it at PSUM→SBUF eviction chunk-by-chunk the way the sibling did (its norm reduced
   over K, independent of the matmul output). v1's structure — assemble full y, then a single
   full-N reduce, then a full-width output scale — is already the correct shape and is *kept*.

So the v1→v2 diff is **localized to the matmul only**: split w into two bf16 limbs once at
load; split each transposed x sub-tile into two bf16 limbs; replace the single fp32
`nc_matmul` accumulation with the 3-product bf16 accumulation. The residual add, RMSNorm, and
output scale are byte-for-byte v1 (all fp32).

## 5. Candidate v2 design (compensated bf16x2 3-product split, built on v1)

`runs/matmul_add_rmsnorm_v2_bf16_split.py`. Same signature, raw-2D I/O, M-outer loop, and
fp32 epilogue as v1. Diffs:

**Pinned, auditable split order** (bf16(.) = `nl.copy(dtype=nl.bfloat16)`, round-to-nearest-even):
- Weight, once at load (replaces the fp32 `w_sb`): for each of 16 K-tiles,
  `w_f = load(w[kt])` (fp32) → `w_hi[kt] = bf16(w_f)` → `w_res = w_f − w_hi` (fp32, exact for
  O(1)) → `w_lo[kt] = bf16(w_res)`. Store `w_hi`, `w_lo` as resident `[16,128,N]` bf16 (32 KB
  each = 64 KB/part total, **identical bytes to v1's one fp32 w**).
- Activation, per M-tile per K-sub-tile (replaces the fp32 `xT`): transpose the RAW x
  sub-tile to `xT_f = [k_in, m_in]` via the exact fp32 identity `nc_matmul` (unchanged from
  v1) → `xT_hi[kt] = bf16(xT_f)` → `xT_res = xT_f − xT_hi` → `xT_lo[kt] = bf16(xT_res)`.
  Splitting after the transpose is identical to before it (transpose is exact, bf16 rounding
  is element-wise).

**Matmul (replaces v1's single fp32 accumulation):** per N-chunk `c` (4 chunks of 512), per
K-tile `kt` (16), accumulate three bf16 products into the fp32 PSUM bank:
```
acc += nc_matmul(xT_hi[kt], w_hi[kt, :, chunk])   # hi @ hi
acc += nc_matmul(xT_hi[kt], w_lo[kt, :, chunk])   # hi @ lo
acc += nc_matmul(xT_lo[kt], w_hi[kt, :, chunk])   # lo @ hi   (drop lo@lo)
```
Then, **exactly as v1**: `y[:, chunk] = acc + z_tile` (fp32 residual add before the norm),
and after all 4 chunks the fp32 RMSNorm (square → full-N reduce → `sumsq/N + eps` → rsqrt)
and the full-width output scale `out = y·inv_rms·g`.

**Loop-order note (de-risk aT-split cost):** v3 of the sibling split each `[128,128]` aT
sub-tile via an intermediate fp32 SBUF copy (`aT_f`) then two casts; the sibling's phase-3 D1
showed the compiler already elides the redundant copy and the sub-tile split work is hidden
under the PE-bound matmul, so I will write the straightforward per-sub-tile split and *check
the digest* rather than pre-optimizing it. If the digest shows PE idle (<~95%) I have the
sibling's bit-exact "split aT limbs directly from PSUM" reschedule (`aT_hi=bf16(psum_t)`,
`aT_lo=bf16(psum_t−aT_hi)`, dropping the fp32 `aT_f`) as a zero-precision-change tweak lever.

## 6. Accuracy-repair fallback v2b (4-product), costed, built only if triggered

Trigger (from the sibling's playbook): if v2's *measured on-device* rel-L2 exceeds **1.5e-5**
on any seed (the marginal threshold; expected ~1.53e-5 by §3 quadrature, so this may fire),
build `runs/matmul_add_rmsnorm_v2b_bf16_split4.py` = v2 + the fourth product
`acc += nc_matmul(xT_lo[kt], w_lo[kt, :, chunk])`. Offline this only moves rel-L2 4.454e-6 →
3.491e-6 (~22% of the bf16 term, ~1% of the quadrature) while adding a 4th matmul pass
(+~25% PE time). On the sibling the analogous v3b measured **+28% latency for a ~1.6% rel-L2
improvement → MEASURED-REJECT** (a false repair when the fp32 floor dominates). I expect the
same here and will keep v2b as a recorded negative datum unless v2 actually fails the gate
(in which case the 4-product is a genuine correctness necessity, not a false repair). This is
"build + measure, don't skip-by-model" per the sibling discipline.

## 7. SBUF budget (per partition, 128 partitions) — fits

| buffer | bytes/part | note |
|---|---|---|
| `w_hi` + `w_lo` (16 K-tiles, N=2048, bf16) | 64 + 64 = **128** KB | **same as v1's one fp32 w** (2 bf16 limbs = 1 fp32) |
| identity | <1 KB | |
| M-loop transients: x_sb (8) + xT_hi/lo (4+4) + aT split temps (~1) + y (8) + sq (8) + z_tile (2) + out_sb (8) | ~**43** KB | peak in the M-loop |
| **M-loop total** | **~171 KB** | < ~192 KB budget |
| preamble peak (w limbs + transient w_f + w_res) | ~145 KB | < budget |

Comfortable. If the compiler ever spills, reuse `sq` over `y`'s buffer (compute norm in place)
before touching residency — correctness is unaffected (residency is a perf choice).

## 8. Acceptance / how each candidate is judged

Score from `workspaces/matmul_add_rmsnorm/`:
```bash
python3 \
    ../../verify.py --op matmul_add_rmsnorm --candidate runs/<file>.py --fast   # seed-42 quick
# drop --fast for the full 5-seed gate before recording/promoting
```
Numeric per-seed rel-L2 (verify.py prints only the bool gate + latency): copy the sibling's
`runs/rel_l2_probe.py` into this workspace and run it to record the numeric rel-L2 (needed to
evaluate the §6 trigger).

- **v2 (bf16x2 3-product) — the intended promotion.** Promote iff: full-5-seed
  `l2_norm_passed = True` AND measured p50 is out-of-noise faster than a same-session v1
  control (record a fresh v1 run as the noise anchor; sibling jitter was <0.1%). Expect
  ~0.75 ms / ~5x and rel-L2 ~1.53e-5.
- **v2b (4-product) — fallback only.** Build+measure only if v2's measured rel-L2 crosses
  1.5e-5; keep as negative datum unless v2 fails the gate outright.
- **v1 retained as the pure-fp32 fallback** (DEC: keep a correct fp32 path if any bf16 seed
  is ever marginal).
- Never regress correctness; ≤5 iterations on the one direction.

## 9. Deliverables

- `runs/matmul_add_rmsnorm_v2_bf16_split.py` (promotion candidate), and `v2b_bf16_split4.py`
  only if triggered.
- `runs/offline_bf16_split_sim.py` (already written) + `runs/rel_l2_probe.py` (copy from
  sibling) as evidence helpers.
- Record each perf change in `benchmark.csv`; each candidate (with parent links) in
  `candidates.jsonl`; profiler digests + the offline-sim output under `profile/`.

See sibling evidence: `workspaces/add_rmsnorm_matmul/runs/add_rmsnorm_matmul_v3_bf16_split.py`
(the split-matmul template), `workspaces/add_rmsnorm_matmul/docs/plan-phase2.md`,
`workspaces/rmsnorm_matmul/runs/rmsnorm_matmul_v4_bf16_split.py`; memory
`kda-add-rmsnorm-matmul-progress`, `kda-rmsnorm-matmul-progress`, `kda-matmul-progress`.
