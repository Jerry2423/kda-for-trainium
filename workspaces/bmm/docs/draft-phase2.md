# bmm — Phase 2 draft: profile-driven optimization

**Operator:** `bmm` (NKIBench case 2). Batched matmul `out[b] = lhs[b] @ rhs[b]`,
`b in 0..15`. `lhs (16,4096,64)=(B,M,K)`, `rhs (16,64,4096)=(B,K,N)` fp32 →
`out (16,4096,4096)`. **B=16, M=4096, K=64, N=4096.** Baseline **2.550 ms**.

**Start point:** `runs/bmm_v1.py` = the phase-1 correctness base, **0.663x (3.8477 ms)**,
full 5-seed L2 PASS. Profile: MFU=11% **PE=95%** Vec=20% Scl=14% DMA=39%,
HBMrd=34 MB (read floor), HBMwr=1074 MB (write floor). Traffic is already minimal.

---

## 1. The anomaly that defines phase 2 (this is the whole story)

Phase 1 recorded "PE-bound, not write-bound" (Codex reconciled the draft's wrong
write-bound framing). Phase 2 sharpens that into an **actionable** finding by counting
`nc_matmul` sites in v1 vs the baseline:

| kernel | transpose sites | main-matmul sites | total sites | latency | µs / site |
|---|---|---|---|---|---|
| baseline `..._0.py` | `16·4·4·8` = **2048** | `16·4·4·8·2` = **4096** | **6144** | 2.550 ms | 0.415 |
| **v1** | `16·32` = **512** | `16·32·8` = **4096** | **4608** | 3.848 ms | 0.835 |

- The **4096 main matmuls are IDENTICAL** in both (`[64,128]×[64,512]→[128,512]`, fp32).
- v1 does **8× FEWER transposes** (512 vs 2048): it hoists the lhs transpose above the
  n-loop; the baseline re-transposes each m-subtile once per 1024-wide n-block (4× redundant).
- **v1 does 25% FEWER total matmul sites, yet is 51% slower → 2× slower per site.**

⇒ v1 is **not** limited by op count (it already does less work than the 2.55 ms baseline).
It is **schedule-bound**: the PE is *occupied* (PE=95%) but *stalled between
instructions*, doing little useful work (MFU=11%). The baseline reaches 2.55 ms with
MORE work purely because it feeds the PE better. **The phase-1 draft's DMA levers
(store-burst fattening, ping-pong) are dead** — DMA=39% is already hidden under compute
and traffic is at the read-once/write-once floor. **The lever is PE feeding.**

### Root cause hypothesis (to confirm in round 0)
The baseline pre-declares giant multi-bank PSUM tensors indexed by every loop var —
`v8 = zeros((16,4,4,8, 64,128))` (transposes), `v10 = zeros((16,4,4,8,2, 128,512))`
(outputs). Each `(loop-index)` combo writes a **distinct logical PSUM bank**, which the
compiler rotates through the 8 physical banks and software-pipelines: matmul(c+1) issues
while copy(c)/store(c) drain. v1 instead uses a **single** `acc` (and a single `psum_t`)
declared fresh inside the loop; the tight `matmul → nl.copy → nl.store` dependency on one
rotating bank (plus a serial `transpose → copy` at the head of each m-tile's 8-matmul
burst) starves the PE. This is exactly the multi-bank idiom the `matmul` sibling used to
hit PE=100% (`acc = zeros((B,128,512))`, distinct bank per block member).

## 2. Structural ceiling shared with the baseline (do NOT try to remove)
- **fp32 on a bf16-native PE**: each fp32 matmul emulates in multiple bf16 passes.
- **K=64 fills only 64 of 128 partition rows** → the contraction axis is half-empty every
  pass. Cannot be fixed by packing 2 batches onto K: `out[b]` are block-diagonal, so a
  128-row stacked contraction would *sum* two batches' products — numerically wrong.

Both are present in the baseline too, so they explain why even a perfectly-scheduled fp32
bmm can't reach the naive ~0.90 ms FLOP floor (MFU stays low). They do **not** explain
v1's gap *to the baseline* — that gap is schedule (§1). Precision (§4, D3) is the only
lever that touches this ceiling, and it is a gated, measure-first bet.

## 3. Round 0 — measurements before any code change (mostly zero remote-risk)

All via the sibling `dump_metrics.py` idiom (reads the profiler's TRUE
`tensor_engine_active_time_ns` + `matmul_instruction_count`, not the coarse PE%×lat
proxy). Create `runs/dump_metrics.py` = the swiglu copy with op string `"bmm"`.

1. **v1 true PE-active + instr count** — `dump_metrics runs/bmm_v1.py`. Establishes v1's
   real PE-busy time and matmul-instruction count (expect ~2× sites for fp32 emulation).
2. **Baseline true PE-active + instr count** — `dump_metrics` on the read-only baseline
   `..._0.py` (profiling only, never edit it). This QUANTIFIES the per-instruction stall:
   if the baseline does more matmul-instr in less PE-active time, the gap is confirmed
   schedule loss and bounds the D1 headroom exactly.
3. **fp32/bf16 PE-ratio calibration** — `runs/bmm_probe_bf16_calib.py`: v1 with the main
   matmul operands cast to bf16 (single product, **correctness will FAIL — record-only**),
   read via `dump_metrics`. Gives THIS op's fp32-vs-bf16 pass ratio. **This decides D3**:
   the `matmul` sibling measured fp32 ≈ **3.62×** bf16 (so bf16x2's 3 products WIN), but
   the `swiglu` sibling measured fp32 ≈ **2×** (so bf16x2's 3 products LOSE +50%). bmm
   could go either way — measure, don't assume.
4. **offline bf16x2 rel-L2 sim** — `runs/offline_bf16_split_sim.py` (numpy, ZERO remote
   spend): reproduce the scored input draw + the numpy reference, compute the worst-case
   3-product compensated-bf16x2 rel-L2 across seeds. Gate D3 on `< 2e-5`. Note bmm's
   output is a **raw matmul with NO downstream averaging** (unlike rmsnorm's `/K`), and
   K=64 is short, so error ≈ single-pass matmul rounding — likely safe but must be shown.

## 4. Optimization directions, ranked by expected benefit × confidence

### D1 — PSUM-bank pipelining (PRIMARY; high confidence, high benefit, pure fp32)
Restructure v1's inner loop to expose independent matmuls into **distinct pre-declared
PSUM banks**, mirroring the baseline's `v10` discipline, while KEEPING v1's hoisted 512
transposes. Concretely: per `(b, mt)`, declare a multi-bank output accumulator
`acc = nl.ndarray((G, par_dim(128), 512), buffer=psum)` and issue the G n-chunk matmuls
into `acc[0..G-1]` before their copies/stores drain (G ∈ {2,4}; 8 output banks + 1
transpose bank > 8 physical, so group n or offload the transpose — see D2). This lets the
compiler pipeline matmul-issue ahead of PSUM→SBUF copy and store.
- **Expected:** recover the 2×/site schedule gap. v1 has 25% FEWER matmul sites than the
  2.55 ms baseline and identical main matmuls, so matching the baseline's per-site
  efficiency projects to **well under 2.55 ms (~1.2–1.35x)**; round-0 #1/#2 give the exact
  bound. Even conservative parity-per-site clears 1.0x on the transpose savings alone.
- **Risk:** low — no precision change, correctness identical (single-pass `=`, no K-accum
  reorder). Sweep G ∈ {2,4}; watch PSUM/SBUF pressure (the matmul sibling saw B=8 regress
  full-run). ≤5 iterations: G-sweep + n-block width (512-group vs 1024-group like baseline).

### D2 — off-PE lhs transpose via `load_transpose2d` (ENABLER; low risk, small–med gain)
Replace the 512 identity-matmul transposes with `nl.load_transpose2d(v1[b, m_slice, k])`
→ `[k,m]` loaded already-transposed from HBM. **Proven portable at fp32 on this remote**
(rmsnorm `probe_loadtranspose`: full 5-seed PASS, transpose fully hidden, PE stayed 97%).
Removes 512 PE passes AND the transpose PSUM bank + its copy → **frees a PSUM bank so all
output banks fit** and simplifies the D1 schedule. Best evaluated *combined with D1*.
- **Risk:** low (measured-portable). If it fails to lower here, fall back to the PE
  identity-transpose (v1's proven idiom). ≤2 iterations.

### D3 — compensated bf16x2 3-product main matmul (GATED BET; measure-first)
Split each main-matmul operand into bf16 hi/lo limbs; accumulate 3 products
(`hi·hi + hi·lo + lo·hi`, drop `lo·lo`) in fp32 PSUM. **Only pursue if BOTH round-0 gates
pass:** (a) #3 shows fp32/bf16 ratio **> 3** (else 3 products cost more than fp32, and this
regresses exactly like swiglu's all-3 split, 0.409x); (b) #4 offline rel-L2 **< 2e-5**.
- **Upside if it ports:** ~1.2–1.3x *on top of* D1 (matmul/rmsnorm/add_rmsnorm all won big
  here). **Downside if the ratio is ~2:** SKIP entirely — it would raise PE time.
- This is the only lever that touches the fp32 ceiling (§2). Requires an on-device 5-seed
  L2 PASS to promote (offline sim is a green-light, not a guarantee). ≤3 iterations
  (single-precision limb build + resident-limb reuse; combine with D1 banking).

### Closed / not-pursued directions (record-only, do not spend iterations)
- **M-blocking** (the `matmul` sibling's winner): N/A — rhs reload is already eliminated
  (rhs[b] resident once/batch), reads are 34 MB at the floor. Nothing to amortize.
- **Store-burst fattening / output ping-pong** (phase-1 draft's levers): dead — DMA=39%
  hidden, HBMwr at the 1074 MB write floor. Codex already closed this.
- **bf16 output**: forbidden — output IS the final result; the 2e-5 gate bans it.
- **`dma_transpose` fp32**: proven-INFEASIBLE on this remote (rmsnorm probes: fp32 is not
  2-byte, `dma_transpose`/SFKVectorizer crash). Do not re-probe.
- **Vector-engine transpose** (`nc_transpose engine=vector`): rmsnorm MEASURED +2%
  regression (Vec co-bottleneck). Reject.

## 5. Method & discipline (per direction, ≤5 iterations)
- **Noise anchor:** re-run v1 same-session as the control before each comparison (siblings
  saw ~0.08–0.5% jitter; treat a **~1.8–2.5% band** as noise). Promote only OUT-of-band
  wins on a **full 5-seed** run (drop `--fast`); `--fast` (seed 42) only for screening.
- **Evidence per direction:** before/after p50 latency (`verify.py`), the MFU/PE/Vec/Scl/
  DMA/HBM digest, and TRUE PE-active + `matmul_instruction_count` (`dump_metrics`) to
  distinguish real PE-work change from serialization. Record keep/revise/reject with the
  number that decided it.
- **Never regress correctness:** every promoted candidate must clear the 5-seed L2 gate;
  keep v1 as the pure-fp32 fallback (like the rmsnorm/swiglu families).
- **Bookkeeping:** append each perf change to `benchmark.csv`; each candidate to
  `candidates.jsonl` with parent links (DAG); profiling evidence under `profile/`. Kernels
  go in `runs/`; never edit the baseline/reference.

## 6. Expected trajectory
`v1 0.663x → D1 (schedule fix, pure fp32) ~1.2–1.35x → +D2 (off-PE transpose) small gain
→ +D3 (bf16x2, ONLY if round-0 ratio>3 & rel-L2<2e-5) up to ~1.5x.` D1 is the
high-confidence, provable win (baseline reaches 2.55 ms with more PE work); D3 is the
upside gamble that round 0 will accept or kill before we spend a full remote run on it.
