# bmm_softmax — Phase 3 Exit Decision

**Outcome: TERMINAL (AC-7). Keep `runs/bmm_softmax_v4.py` (1.946x, 3.7456 ms p50) as the
phase-3 kernel of record. No candidate promoted.** The reachable epilogue engine-placement
levers (D1) produced no statistically promotable improvement over v4; the compiler's own
engine placement already balances the fused softmax epilogue against the matmul stream.
`bmm_softmax_v1` (1.585x, fp32 fallback / DAG root) and `bmm_softmax_v4` both retained.

Phase reframe (fixed in the plan): classic shape specialization has no surface here (every
axis divides cleanly, tiles at the 512-wide PSUM-bank wall, full-row resident softmax), so
phase 3 attacked the **engine schedule of the fused softmax epilogue** — the one structural
dimension phase 2 stopped short of. 5 optimization iterations used (the AC-6b diagnostic +
3 copy-placement variants + 1 normalize-placement variant), within the ≤5 budget.

Codex (high effort) independently reviewed the round-0 + D1 evidence and concurred:
**TERMINAL AC-7, keep v4, no promotion** — adding the conservative framing adopted below.

**Round-1 evidence repair (2026-07-11):** the round-0 code review flagged that the depth-matched
gap number in the draft ("+17.9%", pure-bmm-M16 PE-active 2.8582 ms) did not match the saved
profile artifact (which showed 3.0781 ms), and that AC-3 task2 recorded only aggregate engine
counts rather than a per-op placement record. Both are now fixed: (1) a same-session interleaved
bracket shows the pure-bmm-M16 control is measurement-bimodal (its *raw* PE-active-ns is constant;
the jitter is a p50/window artifact), so the honest gap is a **range +9%–18%**, not a single
point; (2) the per-op placement is recorded (attributed from forced-engine variant deltas — a true
per-instruction dump is remote-infeasible here). A re-run of the analyze checkpoint on the
corrected evidence again returned TERMINAL (Codex round-1). The no-promotion result never depended
on the exact gap magnitude.

---

## Round 0 — controlled anchors & causal check (AC-3 / AC-3.1 / AC-6b)

All same-session, `--fast` (seed 42), `runs/dump_metrics.py` TRUE
`tensor_engine_active_time_ns` (window-normalized), not the coarse PE%/DMA% proxy.
**(Round-1 evidence repair: the round-0 draft cited a single pure-bmm-M16 point that a
re-run did not reproduce; the numbers below are reconciled against a same-session
interleaved bracket — see `profile/bmm_softmax_phase3_round0_depth_bracket.txt`.)**

| signal | v4 (fused, M16) | pure-bmm M16 (depth-matched) | AC-6b copy-elim diag |
|---|---|---|---|
| wall p50 | 3.737–3.746 ms | 2.886 / 3.104 ms (bimodal) | 6.6288 ms (+77%) |
| TRUE PE-active/inf | **3.363 ms (mean)** | **2.862 / 3.078 ms (bimodal)** | **3.5710 ms (+5.9%)** |
| raw tensor_engine_active_ns | ~6.73 ms | **~6.590 ms (constant)** | 7.15 ms |
| Vec active / instr | 71.8% / 3472 | 22.6% / 3996 | 34% / 1651 |
| Scl active / instr | 70.1% / 4628 | 19.0% / 3989 | 29% / 1356 |
| matmul_instruction_count | 8704 | 8704 | 8704 |
| psum_read_sbuf_write_count | 4224 | 4224 | 1152 |
| HBM read / write | 33.6 / 1073.7 MB | 33.6 / 1073.7 MB | 33.6 / 1073.7 MB (floor) |
| rel-L2 (seed 42) | 2.5683307869e-6 | 1.83e-7 (pure bmm) | 2.5683307869e-6 (bit-id) |

**Depth-matched PE-active gap = +9% to +18% (an honest RANGE, control-noise-dependent).**
v4's PE-active is stable (mean 3.363 ms, ~0.3% spread). The pure-bmm-M16 control is BIMODAL in
the reported metric — but its *raw* `tensor_engine_active_time_ns` is essentially **constant**
(6.5904/6.5905/6.5905 ms across 3 runs); the jitter is entirely in the reported p50, hence the
metric window (`total_time_ns/p50` = 2.303/2.303/2.141), which the window-normalization
propagates into TRUE PE-active/inf {2.862, 2.862, 3.078}. So the gap is +17.5% vs the control's
low mode and +9.3% vs its high mode. **Both ends positive → real softmax-specific PE inflation
at matched M16 depth is confirmed; only the magnitude is control-noise-dependent.** (The round-0
draft's single "+17.9%" cherry-picked the low mode; this is corrected to the range.)

This is still **far smaller than the draft's depth-confounded "+68%" headline** (which compared
v4-M16 vs pure-bmm at its *own* optimum M_SUB=32 = 2.011 ms — a different depth). The plan
explicitly demanded the depth-matched comparator (AC-3 negative test: "treating the +68% figure
as the controlled gap"); the phase rests on the +9%–18% range, not +68%. Crucially, **the
terminal conclusion does not depend on the exact gap magnitude** — the copy-elim diagnostic
(+77% wall) and the D1 no-promotion sweep are independent of it.

**Static epilogue engine placement (AC-3, recorded not assumed —
`profile/bmm_softmax_phase3_epilogue_placement.txt`):** a true per-instruction engine-annotated
dump (per chunk c=0..7) needs the local BIR/NTFF artifacts that this remote-profiler harness does
not produce (it returns only per-ENGINE aggregate counts — see CLAUDE.md). So the drain-copy
engine split is **attributed from the measured instruction-count deltas** of the forced-engine
variants (decisive for the one question that matters — all-on-one vs already-spread — even though
it is not exact per-op): from v4 Vec 3472/Scl 4628, all-Scalar Vec 1720/Scl 5860, all-Vector Vec
6014/Scl 1651, the compiler puts **~41% of the drain copies on Vector and ~60–70% on Scalar** —
it **already spreads the drain across both engines, weighted to Scalar**, not all-on-one. The
normalize is on Vector in v4 (moving it to Scalar gave byte-identical counts → compiler-preferred).
Any hand-forced rebalance is therefore measured against a placement the compiler already split.
(Codex round-1 review accepted this attribution as sound for the phase decision, with the caveat
that it is not exact per-op attribution.)

**AC-3.1 causal-premise check — PREMISE HOLDS (not downgraded):** the depth-matched gap is
moderate but real (+9% to +18%, control-noise-dependent) AND the AC-6b copy-elimination
diagnostic **materially inflates PE-active (+5.9%) and wrecks wall (+77%)**. Holding all 8 PSUM banks alive through the
4096-wide max+exp (removing the drain copies) starves the next subtile's 8 matmuls — confirming
(a) the drain copies are the fast bank-free path, so copy-elimination stays a paper reject
(AC-6, now measured), and (b) the bank-drain speed genuinely gates the matmul stream. So the
back-pressure story is CONFIRMED, and pursuing D1 was justified (not a lean-terminal shortcut).

---

## D1 — epilogue engine rebalancing (AC-4)

### Phase A — copy placement (AC-4.1), normalize held at v4's (all `--fast`, same-session)

All variants bit-exact (rel-L2 2.5683307869e-6, matmul 8704, psum 4224, HBM at floor, spill 0):

| variant | wall p50 | Δwall | TRUE PE-active | ΔPE | Vec instr | Scl instr | verdict |
|---|---|---|---|---|---|---|---|
| v4 (compiler placement) | 3.7456 | — | 3.3707 | — | 3472 | 4628 | anchor |
| (ii) all copies → Scalar | 3.7281 | −0.47% | 3.4241 | **+1.6%** | 1720 | 5860 | **no-op** (wall in-noise, PE rose) |
| (iii) all copies → Vector | 3.9681 | +5.9% | 3.4629 | +2.7% | 6014 | 1651 | **ANTI-LEVER** |
| (iv) 4 Vec / 4 Scalar split | 4.2743 | +14.1% | 3.5053 | +4.0% | 3653 | 3546 | **ANTI-LEVER (worst)** |

**Key empirical result:** the 4/4 split is the *worst* of the sweep. Per the plan's own test
("the 4/4-split-vs-all-Vec comparison empirically answers whether Vector and Scalar drain PSUM
in parallel"), this is a clean **NO — Vector and Scalar do not drain PSUM in parallel here;
forcing a split strictly hurts.** The 2/6 and 6/2 extensions do NOT fire (their trigger was an
out-of-noise 4/4 signal; the 4/4 signal is strongly negative). The compiler's own Vec 3472 /
Scl 4628 placement beats every hand-forced placement.

**API note (Plan Evolution):** the plan's primary lever `nisa.tensor_copy(engine=...)` was
flagged by the sibling `matmul` task as "API-infeasible on this remote." That was an **older
call form**; the current form `nisa.tensor_copy(acc, engine=nki.isa.engine.scalar|vector,
dtype=...)` **COMPILED and RAN cleanly** on this remote (status=success, purity intact,
bit-exact). So the D1 lever was genuinely reachable — and it still did not win. (The sibling's
"infeasible" record is superseded for this remote/NKI version.)

### Phase B — normalize placement (AC-4.2), on the winning (= compiler/v4) copy assignment

| variant | wall p50 | Δwall | TRUE PE-active | ΔPE | Vec instr | Scl instr | verdict |
|---|---|---|---|---|---|---|---|
| v4 (normalize on Vector, `tensor_scalar`) | 3.7456 | — | 3.3707 | — | 3472 | 4628 | anchor |
| normalize → Scalar (`activation(op=nl.copy, scale=recip)`) | 3.7373 | −0.22% | 3.3647 | −0.18% | 3473 | 4628 | **compiler no-op** |

Per-engine instruction counts are **byte-identical to v4** (Vec 3473 vs 3472, Scl 4628 == 4628):
the Scalar `activation(op=nl.copy, scale=recip)` lowered to the same instruction stream as v4's
Vector `tensor_scalar`. Both deltas deep inside the ~1.5–2% noise band. No-op.

### AC-8 same-session interleaved bracket (closest two candidates vs v4)

| candidate | bracket (p50) | gap vs v4 | PE-active | promotable? |
|---|---|---|---|---|
| v4 | {3.7456, 3.7462} | — | 3.3707 | (anchor) |
| all-Scalar copies | {3.7281, 3.7315} | ~0.4% | +1.6% ↑ | **NO** — <2% bar, PE rose (no corroborating mechanism) |
| normalize-Scalar | {3.7355, 3.7373} | ~0.2% | −0.18% (flat) | **NO** — compiler no-op |

The all-Scalar bracket is barely non-overlapping, but the gap is ~0.4% (far below the ~2%
promotion bar) and TRUE PE-active *rose* +1.6% — no corroborating throughput mechanism (DEC-1:
wall is the gate, PE-active must corroborate). This is the classic `--fast` scheduling variance
on a higher-Scalar-pressure variant that reverses on the full run
(`[[BL-20260709-fast-vs-full-run-latency]]`). Not promotable.

---

## D2 — M_SUB re-sweep (AC-5): SKIPPED (correctly, per AC-5 negative test)

D2 is contingent on D1 landing a promotable win. **D1 did not land** (all copy variants
no-op/anti-lever; normalize a compiler no-op), so D2 is skipped by AC-5's own negative test
("running D2 when D1 did not land"). Additionally: M_SUB was already swept in phase 2 (interior
optimum M16: M8 1.906x < M32 1.933x < M16 1.946x, `[[BL-20260711-heavy-epilogue-shifts-twophase-msub-optimum-interior]]`),
and D1 did **not** change the engine balance, so the phase-2 optimum stands. Building D2 would
be chasing a contradicted precondition (mirrors sibling `bmm`'s D2-skip after a regressing D1).

---

## Closed levers (AC-6) — record-only, NOT built (no iterations spent)

| lever | one-line reason closed |
|---|---|
| bf16x2 3-product matmul split | fp32/bf16 pass ratio 2.0 < 3.0 ⇒ split *raises* PE on a PE-bound kernel; `[[BL-20260710-bf16x2-loses-when-fp32-emulates-in-2-passes]]` |
| bf16 `exp`/softmax | ~1e-2 rel error over N=4096 ≫ the 2e-5 gate (v4 margin only 7.8x) |
| cross-batch blocking / double-buffer | measured anti-lever in `bmm` (stall 0.231→0.296→0.332 µs monotone); `[[BL-20260710-cross-batch-blocking-is-an-antilever-on-affine-range]]` |
| GpSimd normalize / copies | API-infeasible: `tensor_scalar(engine=gpsimd)` is rsqrt-only; GpSimd cannot read PSUM |
| fused `activation(reduce_res=)` exp+row-sum | measured +75% wall via whole-stream 2× recompute; `[[BL-20260711-activation-reduce_res-fused-rowsum-triggers-score-stream-recompute]]` |
| copy-elimination as a **production** kernel | measured +77% wall / +5.9% PE (AC-6b diagnostic below); the drain copies are the fast bank-free path |
| wider matmul tile / narrower N_CHUNK / K-packing / DMA store-burst / bf16 output | all closed in `bmm` (512-wide PSUM-bank wall; block-diagonal; DMA hidden at floor; 2e-5 bans bf16 out) |

**AC-6b diagnostic** (`runs/bmm_softmax_copyelim_diag.py`, one `--fast` probe, counts toward
budget): reduce/exp read the resident PSUM score tile directly (no drain copies). Result:
+77% wall, +5.9% PE-active, psum copies 4224→1152, rel-L2 bit-identical. Confirmed the paper
rejection and the back-pressure premise. Diagnostic-only — never a production candidate.

---

## Per-direction decision (keep / revise / reject)

| Direction | Decision | Evidence |
|---|---|---|
| D1 copy placement all-Scalar | **REJECT** (no-op) | wall −0.47% (in-noise), PE-active +1.6% (rose); no corroborating mechanism |
| D1 copy placement all-Vector | **REJECT** (anti-lever) | wall +5.9%, PE +2.7%, Vec saturated 88% |
| D1 copy placement 4/4 split | **REJECT** (anti-lever, worst) | wall +14.1%; proves Vec+Scl do NOT drain PSUM in parallel |
| D1 normalize on Scalar | **REJECT** (compiler no-op) | byte-identical instruction counts, wall/PE in-noise |
| D2 M_SUB re-sweep | **SKIP** | precondition (D1 landed) false; phase-2 optimum M16 stands |
| **v4 (phase-2 kernel)** | **KEEP** (terminal) | engine-balanced optimum among reachable D1 levers |
| **v1 (fp32 fallback)** | **KEEP** | DAG root / guaranteed-correct fallback |

**Conservative framing (per Codex):** this is not a proof that v4 is *globally* engine-balanced;
it is that **the reachable D1 engine-placement levers produced no statistically promotable
improvement, and v4 remains the best supported kernel.** The back-pressure premise is confirmed;
the lever to relieve it via manual engine placement is exhausted (the compiler already does it).

**Final: `bmm_softmax_v4` at 1.946x (3.7456 ms) is the phase-3 kernel of record.**
