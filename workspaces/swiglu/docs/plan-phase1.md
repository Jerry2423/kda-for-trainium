# swiglu Phase 1 — First Correct fp32 NKI Kernel (M-outer, resident `h`, shared x-transpose)

## Goal Description

Produce the first **correct** NKI kernel for the fused SwiGLU feed-forward operator
(NKIBench case `2`) on AWS Trainium (trn2), delivered as `runs/swiglu_v1.py` — a single
`@nki.jit def kernel(v1, v2, v3, v4)` entry point that consumes the pre-tiled inputs and
returns the tiled output `v5 (32,128,1024)`.

The operator fuses three fp32 GEMMs and a SiLU gate:

```
up   = x @ w_up            # (M,N)
gate = x @ w_gate          # (M,N)
h    = (gate * sigmoid(gate)) * up   # SiLU(gate) * up, elementwise on (M,N)
out  = h @ w_down          # (M,K)
```

with `M=4096`, `K=1024`, `N=3072`, all fp32. The kernel is structured M-outer (one
128-row M-tile at a time, block factor `B=1`): load and transpose the `x` tile **once**
and reuse it as the stationary operand for both the up and gate GEMMs; stream `w_up`,
`w_gate`, `w_down` from HBM; fuse the SiLU gate at PSUM eviction into a **fully SBUF-resident**
intermediate `h` (no HBM spill of `h`, unlike the baseline); transpose `h` and run the down
GEMM into the output.

Phase 1's objective is **correctness first, not speed**: pass the NKIBench relative-L2 gate
on all five seeds. The operator is PE-bound near the fp32 systolic floor (~2.0 ms vs the
2.0742 ms baseline), so a clean fp32 kernel is expected to land near `~1.0x` (possibly
slightly below on `B=1` weight-DMA). Amortizing weight DMA (M-blocking) and breaking the
fp32 floor (compensated bf16×2 split) are explicitly deferred to Phases 2 and 3.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. The verification command for every AC is (run from `workspaces/swiglu/`):

```bash
python3 \
    ../../verify.py --op swiglu --candidate runs/swiglu_v1.py
```

(`--fast` for a quick single-draw check during iteration; drop it for the full 5-seed measurement before promoting.)

- AC-1: The kernel **compiles and runs** end-to-end on the remote trn2 profiler path with entry `def kernel(v1, v2, v3, v4)` and returns a tensor of tiled shape `(32, 128, 1024)`.
  - Positive Tests (expected to PASS):
    - `verify.py --op swiglu --candidate runs/swiglu_v1.py --fast` compiles and produces a latency measurement (no compile/runtime error).
    - Every `nisa.nc_matmul` call has both operands resident in SBUF and a moving free dim `<= 512`; every activation reads an SBUF tile (not a raw PSUM bank).
    - The returned tensor reshapes cleanly to the reference output shape via `transform_nki_outputs`.
  - Negative Tests (expected to FAIL when the kernel is wrong):
    - Passing an HBM tensor directly as an `nc_matmul` operand, or a moving free dim `> 512`, fails to compile.
    - Returning a tensor whose tiled shape is not `(32, 128, 1024)` is rejected by the harness.

- AC-2: The kernel is **numerically correct** — it passes the NKIBench relative-L2 gate `||v_k - v_r||_2 < 2e-5 * ||v_r||_2` (fp32) on **every** seed in `[0, 21, 42, 63, 84]`. This is a HARD requirement (the benchmark correctness contract).
  - Positive Tests (expected to PASS):
    - The full 5-seed `verify.py` run reports `l2_norm_passed == True` for all of `[0, 21, 42, 63, 84]`.
    - The measured rel-L2 sits at the fp32 systolic-accumulation floor (siblings measured ~1.4–1.5e-5 at K=1024), confirming the fused SiLU and the layout are exact, not merely under the gate by luck.
  - Negative Tests (expected to FAIL when working correctly):
    - A transposed/swapped `[partition, free]` interpretation of any tile (x, xT, h, hT, or a weight) produces a rel-L2 far above 2e-5 (garbage), failing the gate — surfacing a layout bug rather than passing silently.
    - Substituting a mismatched activation form (e.g. a different sigmoid approximation, or dropping the `* up` gate) fails the gate.
  - AC-2.1: The `x` layout assumption `v1 (8,4,128,8,128)` ≡ `(32,128,1024)` (row-major, no-copy reshape) is validated before perf is trusted.
    - Positive: a tiny probe (or a passing full-kernel run) confirms `x3 = v1.reshape((32,128,1024))` maps `x3[mt, m_in, k]` to `x[mt*128+m_in, k]`, mirroring how the `silu` sibling validated its `(128,32,7168)`≡`(128,224,1024)` view.
    - Negative: an incorrect reshape/stride assumption yields a failing rel-L2 across all seeds.

- AC-3: The intermediate `h = SiLU(gate) * up` is held **fully resident in SBUF** for each M-tile — the kernel performs **no HBM spill/reload of `h`** (in contrast to the baseline's `_spill_163`/`_reload_166`).
  - Positive Tests (expected to PASS):
    - `runs/swiglu_v1.py` allocates no HBM scratch tensor for `h`; `h` lives in an SBUF buffer of `[128, 3072]` fp32 (12 KB/partition) per M-tile.
    - The profiler's HBM read/write bytes reflect weight + x + out traffic only, without the extra ~50 MB write + ~50 MB read of an `h` spill.
  - Negative Tests (expected to FAIL / be rejected in review):
    - Introducing an `nl.hbm`/`nl.store` staging tensor for `h` and reloading it violates AC-3 (this is exactly the baseline behavior the plan removes).

- AC-4: The `x` tile is **loaded and transposed exactly once per M-tile** and the resulting `xT` is reused as the stationary operand for **both** the up and gate GEMMs (no per-projection or per-N-third re-transpose of `x`).
  - Positive Tests (expected to PASS):
    - For each M-tile, there are 8 transpose `nc_matmul(is_transpose=True)` calls on `x` sub-tiles total (not 16, not 24), and both `up_acc` and `gate_acc` read the same `xT[kt]` tiles.
  - Negative Tests (expected to FAIL / be rejected in review):
    - Re-transposing `x` inside the N-chunk loop, or once per projection, duplicates transpose work and is rejected in review (this is one of the two baseline inefficiencies the plan removes).

- AC-5: Phase-1 evidence is recorded: the candidate is scored and its result is logged in `benchmark.csv` and `candidates.jsonl` with the correct parent link.
  - Positive Tests (expected to PASS):
    - `benchmark.csv` gains a row for `swiglu_v1` with `passed`, `latency_ms`, `speedup`, and a note.
    - `candidates.jsonl` gains a record for `swiglu_v1` with `parent = baseline:swiglu_M4096_N3072_K1024_0.py`.
  - Negative Tests (expected to FAIL / be rejected in review):
    - Promoting or reporting a candidate whose evidence rows are missing, or whose parent link is absent/incorrect, is rejected.

> **Note on the score metric**: The `~0.9–1.0x` speedup is an **optimization trend/direction for Phase 1, not a pass/fail acceptance bar** (the draft states "Phase 1's job is correctness first, not a win"). A correct kernel that lands slightly below 1.0x still satisfies Phase 1; the speedup wins are Phases 2–3. The only HARD numeric requirement in Phase 1 is the rel-L2 `< 2e-5` gate on all five seeds (AC-2).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. This draft
specifies a **highly deterministic** design (fixed layout decode, fixed loop structure, fixed
Tensor-Engine idiom proven across sibling kernels), so the bounds are narrow.

### Upper Bound (Maximum Acceptable Scope)
A single `runs/swiglu_v1.py` implementing the M-outer loop exactly as in draft §4: reshape
`v1` to `(32,128,1024)` (no-copy view), load a 128×128 identity once, and for each of 32
M-tiles — transpose `x` once into 8 `xT` sub-tiles (shared by up+gate), stream `w_up`/`w_gate`
over 6 N-chunks of 512 accumulating over 8 k-tiles into two PSUM banks, fuse SiLU at eviction
(PSUM→SBUF copy, then `nl.silu`, then `* up`) into a resident `h_sb [128,3072]`, transpose `h`
into 24 `hT` sub-tiles, and stream `w_down` over 2 K-out chunks of 512 accumulating over 24
n-tiles into the output. Weights are streamed (`B=1`). Includes the layout probe (AC-2.1) and
the exact-4-op SiLU fallback wired in as a ready swap if `nl.silu` misses the gate. Full 5-seed
scoring with evidence recorded. No Phase-2 M-blocking and no Phase-3 bf16 split.

### Lower Bound (Minimum Acceptable Scope)
The same single `runs/swiglu_v1.py` that passes the rel-L2 gate on all five seeds (AC-2) with
the correct entry/return shape (AC-1), keeps `h` resident (AC-3), transposes `x` once shared by
up+gate (AC-4), and has its result recorded in `benchmark.csv` + `candidates.jsonl` (AC-5).
The fallback SiLU need not be exercised if the fused `nl.silu` passes; the standalone layout
probe may be folded into a passing full-kernel run if that already validates the reshape.

### Allowed Choices
- **Can use**: `nisa.nc_matmul` (with `is_transpose=True` identity idiom for transposes),
  `nl.load`/`nl.copy`/`nl.store`, `nisa.activation(op=nl.silu, ...)` on an SBUF tile,
  `nl.multiply`, fp32 PSUM accumulators, `nl.affine_range` loops, the no-copy `reshape` view of
  `v1`. Two acceptable SiLU implementations: (a) fused `nl.silu` [preferred], and (b) the exact
  reference-shaped 4-op fallback `exp(scale=-1) → +1 → reciprocal → * gate` [used only if (a)
  misses the gate]. Free choice of N-chunk loop nesting and buffer naming.
- **Cannot use**: any HBM spill/reload of `h` (violates AC-3); re-transposing `x` per projection
  or per N-third (violates AC-4); bf16 / dtype-splitting or any reduced-precision matmul in
  Phase 1 (deferred to Phase 3); M-blocking `B>1` in Phase 1 (deferred to Phase 2); a moving
  free dim `>512`; feeding HBM tensors directly to `nc_matmul`; running activation directly on a
  raw PSUM bank (must copy PSUM→SBUF first); editing the NKIBench baseline/reference/seeds.

> **Note on Deterministic Design**: Because the draft fixes the algorithm and layout, the upper
> and lower bounds nearly converge — the only latitude is whether the fallback SiLU path is
> exercised and whether the layout probe is a standalone script or folded into the full run. The
> Tensor-Engine idiom, tiled-layout decode, loop structure, and precision (fp32) are fixed per
> the draft specification.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Per-M-tile pseudocode (draft §4), `B=1`:

```
x3 = v1.reshape((32,128,1024))                 # no-copy row-major view
identity[128,128] loaded once into SBUF
for mt in affine_range(32):
    x_sb = load x3[mt]                          # [m_in(par)=128, k(free)=1024]
    for kt in 0..7:                             # transpose ONCE, shared by up+gate
        xT[kt] = copy(nc_matmul(x_sb[:,128*kt:+128], identity, is_transpose=True))  # [k_in,m_in]

    for c in affine_range(6):                   # N=3072 / 512
        up_acc, gate_acc = psum[128,512], psum[128,512]
        for kt in affine_range(8):
            w_up_sb   = load v2[kt,:,512*c:+512]     # [k_in,512] moving
            w_gate_sb = load v4[kt,:,512*c:+512]     # [k_in,512] moving
            up_acc   += nc_matmul(xT[kt], w_up_sb)   # [m_in,512]
            gate_acc += nc_matmul(xT[kt], w_gate_sb) # [m_in,512]
        up_sb   = copy(up_acc)                   # PSUM -> SBUF BEFORE activation
        gate_sb = copy(gate_acc)                 # PSUM -> SBUF BEFORE activation
        sg = activation(op=nl.silu, data=gate_sb)    # [m_in,512], Scalar Engine
        h_sb[:,512*c:+512] = multiply(sg, up_sb)     # -> resident h_sb[128,3072]

    for nt in 0..23:                            # transpose h (24 sub-tiles)
        hT[nt] = copy(nc_matmul(h_sb[:,128*nt:+128], identity, is_transpose=True))  # [n_in,m_in]
    for c2 in affine_range(2):                  # K=1024 / 512
        out_acc = psum[128,512]
        for nt in affine_range(24):
            w_down_sb = load v3[nt,:,512*c2:+512]    # [n_in,512] moving
            out_acc  += nc_matmul(hT[nt], w_down_sb) # [m_in,512]
        store copy(out_acc) -> v5[mt,:,512*c2:+512]
return v5
```

Key correctness invariants (from the Codex-first + convergence passes):
- **PSUM→SBUF copy before activation.** Every sibling and the baseline copy the accumulator to
  SBUF before running an activation on it; do the same for `up_acc`/`gate_acc` (do not run
  `nl.silu` directly on a raw PSUM bank).
- **Down-GEMM index mapping**: `nc_matmul(stationary=hT[nt] [n_in,m_in], moving=w_down[nt, kchunk] [n_in,512]) = [m_in, n_in] @ [n_in, 512] = [m_in, 512]` — contraction on `n_in` (partition of both), correct.
- **SBUF/PSUM budget** (per-partition, `B=1`): identity 0.5 KB + `x_sb` 4 KB + `xT` 4 KB +
  `h_sb` 12 KB + `hT` 12 KB + `up_sb`/`gate_sb` 2 KB each + a few streamed `[128,512]` weight
  tiles (2 KB each) + `out_sb` 2 KB ≈ well under the ~208 KB/partition usable SBUF. PSUM: up+gate
  = 2 banks, transpose = 1 bank, out_acc = 1 bank — never more than 8, and up/gate accumulators
  must not be held live into the down phase.
- **SiLU exactness**: `nl.silu(x) = x*sigmoid(x) = x/(1+e^-x)` is algebraically identical to the
  reference `gate/(1+exp(-gate))`. The standalone `silu` sibling passed the gate with
  `nisa.activation(op=nl.silu)`, but `gate = x@w_gate` has a wider distribution than raw-normal
  SiLU inputs and error propagates through the down GEMM, so this remains an **empirical** check.
  If any seed misses, swap in the exact reference-shaped 4-op form (see fallback below) — do NOT
  substitute a different sigmoid form.
- **Exact-4-op fallback** (only if fused `nl.silu` misses the gate), mirroring the baseline:
  `e = activation(op=nl.exp, data=gate_sb, scale=-1.0, bias=0)` → `d = tensor_scalar(e, add, 1.0)`
  → `r = reciprocal(d)` → `sg = multiply(gate_sb, r)` → `h = multiply(sg, up_sb)`.
- **No masking**: M=4096=32×128, K=1024=8×128, N=3072=24×128=6×512, K_out=1024=2×512 — all tiles
  are exact and rectangular; no edge/tail handling.

### Relevant References
- `../AccelOpt/NKIBench/reference/swiglu_M4096_N3072_K1024_numpy_2.py` — numpy reference + `transform_to_nki_inputs`/`transform_nki_outputs` (the layout decode source of truth).
- `../AccelOpt/NKIBench/kernels/swiglu_M4096_N3072_K1024_0.py` — the baseline; confirms arg roles (`v4=w_gate` loaded first, `v2=w_up`, `v3=w_down`, output `v5`), and shows the `_spill_163`/`_reload_166` HBM spill of `h` (removed) and the triple `x` re-transpose under `i0 in range(3)` (removed).
- `workspaces/matmul/runs/matmul_v1.py` — the M-outer + identity-transpose + N-chunk-512 GEMM idiom this kernel mirrors (`nc_matmul(stationary, moving) = stationary.T @ moving`, PSUM→SBUF copy).
- `workspaces/add_rmsnorm_matmul/runs/add_rmsnorm_matmul_v1.py` and `workspaces/rmsnorm_matmul/runs/rmsnorm_matmul_v1.py` — the transpose + `activation` + evict-to-SBUF patterns and the per-partition zero-bias activation setup.
- `workspaces/silu/runs/silu_v1.py` (and `silu_v3_s7.py`) — the fused `nisa.activation(op=nl.silu, ...)` usage and the no-copy `reshape`-view validation precedent.
- `../../verify.py` — the correctness/latency harness that gates on `l2_norm_passed`.

## Dependencies and Sequence

### Milestones
1. **Layout & idiom lock-in** (foundation):
   - Phase A: Decode the tiled layout of `v1..v4` and `v5` against the reference's
     `transform_to_nki_inputs` and cross-check against the baseline's own indexing (already done
     in draft §1–2). Confirm the `v1 (8,4,128,8,128)` ≡ `(32,128,1024)` no-copy reshape (AC-2.1),
     via a tiny probe or by folding the check into the first full run.
   - Phase B: Confirm the Tensor-Engine transpose idiom and down-GEMM index mapping against the
     `matmul_v1` / `add_rmsnorm_matmul_v1` siblings.
2. **Kernel implementation** (depends on Milestone 1):
   - Step 1: Implement the M-outer loop with shared `xT` transpose (AC-4).
   - Step 2: Implement up/gate GEMMs over 6 N-chunks with SiLU fused at eviction into resident `h` (AC-3), copying PSUM→SBUF before activation.
   - Step 3: Implement `h` transpose + down GEMM over 2 K-out chunks into the output (AC-1).
3. **Verification & evidence** (depends on Milestone 2):
   - Step 1: Run `verify.py --fast` for a quick correctness/compile check; iterate on any compile or layout error.
   - Step 2: If any seed misses the rel-L2 gate, swap in the exact-4-op SiLU fallback and re-verify.
   - Step 3: Run the full 5-seed `verify.py` (AC-2); record `benchmark.csv` + `candidates.jsonl` with the baseline parent link (AC-5).

Dependencies: Milestone 2 cannot start until the layout decode (Milestone 1) is locked; the
5-seed evidence run (Milestone 3.3) depends on a passing correctness check (Milestone 3.1/3.2).

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Confirm/validate the `v1 (8,4,128,8,128)`≡`(32,128,1024)` no-copy reshape and the `v2/v3/v4/v5` layout roles (probe or reasoned cross-check vs baseline) | AC-2.1 | coding | - |
| task2 | Implement M-outer loop: reshape `v1`, load identity once, per M-tile load `x` and transpose once into 8 shared `xT` sub-tiles | AC-1, AC-4 | coding | task1 |
| task3 | Implement up/gate GEMMs over 6 N-chunks of 512, two PSUM accumulators over 8 k-tiles, PSUM→SBUF copy, fused `nl.silu` + `* up` into resident `h_sb` | AC-1, AC-3 | coding | task2 |
| task4 | Implement `h` transpose (24 `hT` sub-tiles) + down GEMM over 2 K-out chunks of 512 accumulating over 24 n-tiles into `v5` | AC-1 | coding | task3 |
| task5 | Run `verify.py --fast` to confirm compile + single-draw correctness; fix compile/layout errors | AC-1, AC-2.1 | coding | task4 |
| task6 | If any seed misses the 2e-5 gate, wire in the exact reference-shaped 4-op SiLU fallback and re-verify | AC-2 | coding | task5 |
| task7 | Run full 5-seed `verify.py`; record `benchmark.csv` + `candidates.jsonl` (parent = baseline) | AC-2, AC-5 | coding | task5 |
| task8 | (Optional) If measured rel-L2 is elevated (not near the fp32 floor) or an unexpected latency cliff appears, ask Codex to sanity-check the layout/precision path before spending more remote runs | AC-2 | analyze | task7 |

## Claude-Codex Deliberation

### Agreements
- The M-outer structure with a single shared `x` transpose (up+gate) and a resident `h` (no HBM
  spill) is the right clean, correct Phase-1 design; it removes both baseline inefficiencies.
- SBUF/PSUM budget is sound: `h_sb`+`hT` ≈ 24 KB/partition is comfortably resident; PSUM stays
  within 8 banks provided up/gate accumulators are not held live into the down phase.
- The down-GEMM index mapping `nc_matmul(hT[nt], w_down[nt,kchunk]) = [m_in, kchunk]` is correct
  (contraction on `n_in`, on the partition axis of both operands).
- PSUM→SBUF copy **before** activation is the correct constraint (matches the baseline and all
  siblings); activation must not read a raw PSUM bank.
- No bf16/dtype splitting and no M-blocking in Phase 1 — those are Phases 3 and 2 respectively.
- Correctness on all five seeds is the Phase-1 exit criterion; the `~1.0x` speedup is a trend, not a bar.

### Resolved Disagreements
- **`nl.silu` directly on PSUM vs SBUF copy first**: Codex flagged that activation may not be
  valid on a raw PSUM bank. Resolved by inspecting the baseline and every sibling — all copy the
  accumulator to SBUF before activation. The plan adopts the PSUM→SBUF-copy-then-activate pattern.
- **`nl.silu` exactness vs the reference's 4-op form**: Codex noted algebraic equality does not
  guarantee rel-L2 equality (different sigmoid/exp approximations), especially after the down GEMM
  amplifies error. Resolved by (a) preferring fused `nl.silu` (proven to pass in the standalone
  `silu` sibling) and (b) wiring in the exact reference-shaped 4-op fallback as a ready swap gated
  on the empirical 5-seed result. Codex agreed this is a reasonable first attempt with a safe net.
- **`hT` resident vs chunk-wise production**: Codex offered chunk-wise `hT` to save 12 KB/partition.
  Resolved as unnecessary for Phase 1 (budget is ample); keep `hT` resident for simplicity. Recorded
  as a Phase-2 lever if SBUF pressure ever appears.
- **Expected `~0.9–1.0x`**: Codex judged this possibly optimistic because `B=1` reloads all three
  12 MB weights per M-tile (weight-DMA can dominate). Resolved by treating the speedup as a Phase-1
  trend/direction, not an acceptance bar, and recording M-blocking as the explicit Phase-2 lever.

### Convergence Status
- Final Status: `converged`
- Rounds: Codex first-pass analysis (1) + one second-pass reasonability review; the second pass
  returned **no structural REQUIRED_CHANGES** (only the already-incorporated SBUF-copy-before-
  activation and exact-4-op-fallback items), so convergence conditions were met.

## Pending User Decisions

No blocking user decisions remain. All Codex first-pass questions were substantively resolved
during the convergence passes (see Resolved Disagreements). The items below are **empirical
verification gates**, not opposing-opinion decisions — they are answered by the `verify.py` run
itself, not by a human choice, and are listed here for transparency.

- DEC-1: Whether fused `nl.silu` clears the 2e-5 rel-L2 gate on all five seeds, or the exact 4-op fallback is needed.
  - Claude Position: Prefer fused `nl.silu` (single Scalar-Engine op, proven in the `silu` sibling); fall back to the exact 4-op form only if a seed misses.
  - Codex Position: `nl.silu` is a reasonable first attempt, but standalone-`silu` passing is not full proof for GEMM-produced gate values; the fallback must copy the baseline 4-op sequence exactly.
  - Tradeoff Summary: Fused path is simpler/cheaper; the fallback is a guaranteed-exact match to the reference at slightly higher op count. Both are in scope; the choice is made **empirically by the 5-seed result**, with the fallback already wired as a swap.
  - Decision Status: `RESOLVED — empirical: try nl.silu first, swap to exact 4-op fallback iff any seed misses the gate.`
- DEC-2: Whether the fp32 rel-L2 floor stays under 2e-5 given the down GEMM contracts over N=3072 (a 3× longer accumulation than the K=1024 siblings).
  - Claude Position: Siblings measured ~1.4–1.5e-5 at K=1024; expected to clear 2e-5, confirmed by the 5-seed run.
  - Codex Position: The exact margin for three chained fp32 GEMMs is empirical; the plan should pass, but the full gate is the authority.
  - Tradeoff Summary: There is no precision lever left in a fully-fp32 kernel — it either clears the gate or reveals a layout bug (which task1/task5 would catch). No design change is implied.
  - Decision Status: `RESOLVED — empirical: the 5-seed verify.py run is the authority; a miss indicates a layout bug, not a precision knob to turn.`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `x_transposed`, `up_acc`, `gate_acc`, `h_sbuf`, `w_down_tile`, `out_tile`), matching the naming style of the `matmul_v1` / sibling kernels.
- English for all repository-facing files, comments, and commit messages (per repo CLAUDE.md).
- Do not edit the NKIBench baseline, reference, seeds, or `summary.json`; the candidate lives only in `runs/swiglu_v1.py`.

--- Original Design Draft Start ---

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

--- Original Design Draft End ---
