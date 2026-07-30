# bmm_softmax — Phase 2 Draft (profile-driven optimization)

Start-of-phase base: **`runs/bmm_softmax_v1.py`** — the phase-1 fused kernel, PROMOTED
at **1.585x** (4.5995 ms) over the 7.290 ms baseline, full 5-seed L2 PASS, on-device
rel-L2 **2.5683e-6** (~7.8x under the 2e-5 gate). This draft ranks the phase-2
levers, each explored for ≤5 iterations, and states the correctness guardrails and
measurement protocol. It does NOT change the numerical result.

---

## 1. Measured bottleneck (from phase-1 evidence, already collected)

Remote-profiler digest (`profile/bmm_softmax_v1_digest.txt`, full 5-seed, per-inference
after the 2.0x window normalization):

| metric | v1 (fused) | pure bmm_v1 (ref) | NKIBench baseline |
|---|---|---|---|
| p50 wall | **4.5995 ms** | 3.8477 ms | 7.3020 ms |
| TRUE PE-active/inf | **3.6424 ms** (79%) | 3.6552 ms (95%) | 3.7345 ms |
| Vec-active/inf | 2.73 ms (59.37%) | ~0.9 ms (20%) | 46.70% |
| Scalar-active/inf | 2.71 ms (59.00%) | ~0.6 ms (14%) | 51.02% |
| DMA-active/inf | 1.4083 ms (30.62%) | — | 3.4246 ms |
| matmul_instruction_count | **8704** | 8704 | 10240 |
| HBM read / write | 33.6 MB / 1073.7 MB | 34 / 1074 MB | 700.5 / 1740.6 MB |
| spill | **0** | 0 | ~1.33 GB round-trip |

**Two facts drive the whole phase (both already established, `docs/phase2-bottleneck-evidence.md`):**

1. **The kernel is PE-bound and the matmul is byte-identical to pure `bmm_v1`.**
   TRUE PE-active 3.6424 ms ≈ pure bmm_v1 3.6552 ms; matmul_instruction_count 8704
   identical. The fusion did NOT change the matmul workload — it runs `bmm_v1`'s
   *per-m-tile transpose→matmul* schedule. The exposed tail is
   `wall − PE-active = 4.5995 − 3.6424 = **0.957 ms**` of softmax Vec/Scalar work
   not covered by the matmul.

2. **Traffic is at the read-once/write-once floor, no spill.** HBMwr == output floor
   (1073.7 MB), HBMrd == input floor (33.6 MB), DMA-active 1.41 ms fully overlapped.
   Phase 2 has **no DMA/spill surface** — the win must come from the compute engines
   (PE, Vec, Scalar), not from moving bytes.

**The prize (why phase 2 has real headroom, not just 0.96 ms):** the softmax Vec/Scalar
stack (Vec 2.73, Scalar 2.71 ms) is *currently hidden* under the 3.64 ms PE bind — only
0.96 ms leaks out. The sibling `bmm` proved the *same* matmul core can run its PE-active
from 3.66 ms → **2.01 ms** by a pure schedule change (`bmm_v2` two-phase transpose-all,
per-matmul stall 0.420→0.231 µs, `[[kda-bmm-progress]]`). If we port that schedule, PE
drops toward ~2.0 ms, which **exposes the softmax as the new bind**. So the two levers
are complementary: one cuts PE (and re-exposes softmax), the other shrinks softmax.

---

## 2. Ranked directions

### D1 — Port the `bmm_v2` two-phase transpose-all schedule (PRIMARY; highest value, lowest risk)

**Hypothesis.** v1 runs `bmm_v1`'s schedule: per m-tile it does `transpose → copy → 8
matmuls`, so a serial `transpose→copy` dependency sits at the head of every matmul burst.
`bmm_v2` separated these: transpose **all 32 m-subtiles of the batch up front** into a
resident `[k=64, 32*128=4096]` SBUF pack, then run all main matmuls with **no transpose
interleaved**. On the identical matmul core this cut PE-active 3.66→2.01 ms (**1.89x
kernel-over-kernel**, `bmm_v1`→`bmm_v2`) because the long uninterrupted matmul stream lets
the compiler hide every PSUM→SBUF copy behind the next matmul.

**Port into the fused kernel.** Keep softmax per-m-tile (a full `[128,4096]` score row is
16 KB/part, resident; all 32 rows at once would be 512 KB/part — impossible, so softmax
must stay per-tile). Structure per batch `b`:
- **Phase A:** load + identity-transpose all 32 lhs subtiles into `lhs_t_pack[64, 32*128]`.
- **Phase B:** for each subtile `s` (in `affine_range`): 8 single-pass K=64 matmuls build
  `score[128,4096]`, then the fused softmax epilogue, then the 4096-wide store. `score`,
  `exp_t`, `out_t` stay allocated *inside* the `affine_range(32)` loop so the compiler can
  overlap subtile `s`'s softmax (Vec/Scalar) with subtile `s+1`'s matmul burst (PE) — the
  same free-pipelining `affine_range` gave `bmm_v2`.

**SBUF budget (verified):** pack 16 KB + rhs 16 KB + score 16 KB + exp_t 16 KB + out_t 16 KB
+ identity 0.5 KB ≈ **80.5 KB/partition** of the 208 KB trn2 usable — 127 KB headroom, no
spill. (Can drop toward 48 KB by doing softmax in-place on one buffer; not required.)

**Correctness risk: ~zero.** Transpose-before-use is exact, so transposing all subtiles up
front then matmul is *bit-identical* to interleaving — the matmul math is unchanged. The
softmax epilogue is byte-for-byte the phase-1 code. This is a pure schedule change.

**Expected benefit.** PE-active 3.64→~2.0 ms. Wall becomes bounded by `max(PE~2.0,
Vec, Scalar)` + residual gaps → expect a decisive move past 1.585x toward ~2.0–2.5x even
before D2. **Open question (the thing to measure):** in `bmm_v2` the epilogue between
matmul bursts was a light store; here it is the heavier softmax (Vec+Scalar). Whether
`affine_range` still hides subtile `s`'s softmax under subtile `s+1`'s matmul is exactly
what the profile must confirm.

**Iterations (≤5):** (1) port transpose-all + per-subtile softmax, `--fast` correctness
+ measure; (2) if softmax does NOT overlap (Vec/Scalar exposed between bursts), try
in-place softmax to shrink the live set; (3–5) reserved for the D3 M-block sweep below,
which shares this schedule.

### D2 — Fuse `exp`+row-sum and fold the max-negate (COMPLEMENTARY; needed once D1 exposes Vec)

**Hypothesis.** v1's softmax runs **3 full-width Vec passes** (max-reduce, add-reduce for
the row sum, normalize) + 1 Scalar pass (`exp`) + a tiny `neg_max`. Two of these collapse
for free (confirmed against `nki-api-reference`):
- `nisa.tensor_reduce(op=nl.max, negate=True)` writes `−row_max` directly, at no extra
  cost — kills the separate `neg_max = tensor_scalar(*−1)` op.
- `nisa.activation(op=nl.exp, bias=neg_max, reduce_op=nl.add, reduce_res=row_sum)`
  computes `exp_t` **and** the row sum in the *same* Scalar pass — the docs state the
  fused free-axis reduce is "no additional performance cost" beyond reading the
  accumulator out. This **removes the entire 4096-wide `tensor_reduce(add)` Vector pass.**

Net: softmax Vec load drops from 3 full-width passes → **2** (max-reduce, normalize),
Scalar unchanged. This is the lever that pays off *after* D1 re-exposes Vec as the bind.

**Correctness risk: low.** Same math; the fused sum accumulates the same `exp` values in
fp32. Reduction ordering is the caller's responsibility per docs but the accumulation is
the identical set of terms → rel-L2 expected to stay ~2.6e-6, far under 2e-5. Gate on the
full 5-seed run before promoting.

**Iterations (≤5):** (1) apply both fusions on top of D1, `--fast` + measure Vec drop;
(2) if the fused `reduce_res` shows any rel-L2 drift, fall back to the explicit add-reduce
(keep D1). Precedent: attention epilogues fuse exp/sum this way
(`[[kda-*]]`; knowledgebase `scheduling-and-pipelining.md` §5, `compute-fusion.md` §1).

### D3 — M-block / schedule-depth sweep (CHEAP; subsumed by D1 or a small tweak)

`bmm_v2` swept `M_SUB` and found whole-batch (`M_SUB=32`) optimal (stall
0.420→0.396(8)→0.340(16)→0.231(32) µs). Port `M_SUB=32` first. BUT here the epilogue
between bursts is the heavier softmax — a *smaller* M-block might pipeline better if the
softmax can't hide under a 32-deep stream. If D1 leaves Vec/Scalar exposed, sweep
`M_SUB ∈ {8, 16, 32}` (2 extra iterations, reuses D1's kernel). Low-medium value.

---

## 3. Explicit rejects (inherited measured evidence — do NOT build)

- **bf16x2 matmul split (3-product).** The matmul core is byte-identical to `bmm`, whose
  bf16 calib probe measured an fp32/bf16 pass-ratio of **2.0** (need >3). At ratio 2.0 the
  3-product bf16x2 split *raises* PE (it emulates fp32 in ~2 passes already), exactly the
  swiglu/`bmm` reject. `[[kda-bmm-progress]]`, `[[BL-20260710-bf16x2-loses-when-fp32-emulates-in-2-passes]]`.
  No build.
- **bf16 `exp`/softmax to halve the Scalar pass.** bf16 carries ~3 decimal digits (~1e-2
  rel error); softmax over N=4096 would blow past the 2e-5 gate (current margin only 7.8x).
  Reject on numerics.
- **Cross-batch blocking / cross-batch double-buffer.** Measured **ANTI-LEVER** in `bmm`
  phase 3: blocking adjacent batches regresses monotonically (stall 0.231→0.296→0.332 µs)
  — the enlarged cross-batch live set constrains the `affine_range` pipeline; the batch
  boundary is a *helpful* reset, not a bubble. `[[kda-bmm-progress]]`,
  `[[BL-20260710-cross-batch-blocking-is-an-antilever-on-affine-range]]`. The D1/D3
  schedule depth stays **within one batch** (`M_SUB ≤ 32`), never across the batch axis.
- **Removing the max-reduce or the normalize pass.** Both are required (overflow-safe
  max-shift; softmax normalization) and already minimal (1 pass each after D2). Keep.

## 4. Optional lower-priority lever (only if D1+D2 leave Vec exposed)

- **Engine rebalancing — move the normalize `tensor_scalar(*recip)` off Vector.**
  `tensor_scalar` runs on Vector/Scalar/**GpSimd**; GpSimd sits at ~4% idle. If after
  D1+D2 the profile still shows Vec as the bind (2 passes), pinning the normalize to
  GpSimd (`engine=`) takes a full 4096-wide pass off the Vec critical path. Cheap, bit-
  exact; test only if warranted by the profile. (Knowledgebase `dma-and-engines.md`:
  VectorE↔ScalarE↔GpSimd offload.)

---

## 5. Correctness guardrails (never regress)

- fp32 throughout the matmul and softmax; no bf16, no tf32.
- Max-shifted softmax preserved: `exp(score − row_max)` (overflow-safe), then divide by
  the row sum. Reduction over the **N free axis** (reference axis 2).
- No softmax reduce/activation/elementwise op on a PSUM tile — PSUM banks hold only
  matmul/transpose, copied to SBUF immediately (as in v1).
- Every candidate: `--fast` pre-check, then **full 5-seed** `verify.py` before any
  promotion; require `l2_norm_passed=True` on all seeds `[0,21,42,63,84]` and record the
  worst on-device rel-L2 (must stay « 2e-5).

## 6. Measurement protocol (per candidate)

From `workspaces/bmm_softmax/`:
```bash
python3 \
    ../../verify.py --op bmm_softmax --candidate runs/<file>.py --fast   # gate first
# then drop --fast for the promotion measurement
```
For each direction capture the digest (`runs/dump_metrics.py`) and diff vs v1 on:
TRUE PE-active/inf, Vec/Scalar-active, matmul_instruction_count (should stay 8704 for
D1/D3; unchanged by D2), HBM read/write (must stay at the floor), psum copies, and wall.
Keep evidence under `profile/`; log every perf change in `benchmark.csv`; record each
candidate in `candidates.jsonl` with parent links (DAG root = `bmm_softmax_v1`).

## 7. Expected outcome & exit

- **Primary target:** decisively beat 1.585x. D1 alone (PE 3.64→~2.0 ms) should reach
  ~2.0–2.5x; D1+D2 (Vec 3→2 passes, softmax exposed but smaller) should approach the
  compute floor set by `max(PE~2.0 ms, Scalar exp ~1.75 ms, Vec~1.8 ms)` ≈ ~2.0–2.5 ms
  → roughly **2.7–3.6x** over baseline. Numbers are hypotheses; the profile gates them.
- **Hard floor on trn2:** the Scalar `exp` pass (~1.75 ms theoretical for 512 × 4096-wide
  passes) is irreducible here — the Vector-fused `nisa.exponential` that would move it is
  NeuronCore-v4-only. So do not expect to beat ~1.75 ms wall regardless of schedule.
- **Promote** the best correct candidate; **keep `bmm_softmax_v1`** as the simple fp32
  fallback. Write `docs/phase2-exit-decision.md` with keep/revise/reject per direction and
  the before/after evidence, then update `[[kda-bmm-softmax-progress]]`.
