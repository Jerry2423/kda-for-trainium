# <Plan Title>

## Goal Description
<Clear, direct description of what needs to be accomplished>

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: <First criterion>
  - Positive Tests (expected to PASS):
    - <Test case that should succeed when criterion is met>
    - <Another success case>
  - Negative Tests (expected to FAIL):
    - <Test case that should fail/be rejected when working correctly>
    - <Another failure/rejection case>
  - AC-1.1: <Sub-criterion if needed>
    - Positive: <...>
    - Negative: <...>
- AC-2: <Second criterion>
  - Positive Tests: <...>
  - Negative Tests: <...>
...

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
<Affirmative description of the most comprehensive acceptable implementation>
<This represents completing the goal without over-engineering>
Example: "The implementation includes X, Y, and Z features with full test coverage"

### Lower Bound (Minimum Acceptable Scope)
<Affirmative description of the minimum viable implementation>
<This represents the least effort that still satisfies all acceptance criteria>
Example: "The implementation includes core feature X with basic validation"

### Allowed Choices
<Options that are acceptable for implementation decisions>
- Can use: <technologies, approaches, patterns that are allowed>
- Cannot use: <technologies, approaches, patterns that are prohibited>

> **Note on Deterministic Designs**: If the draft specifies a highly deterministic design with no choices (e.g., "must use JSON format", "must use algorithm X"), then the path boundaries should reflect this narrow constraint. In such cases, upper and lower bounds may converge to the same point, and "Allowed Choices" should explicitly state that the choice is fixed per the draft specification.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
<Text description, pseudocode, or diagrams showing ONE possible implementation path>

### Relevant References
<Code paths and concepts that might be useful>
- <path/to/relevant/component> - <brief description>

## Dependencies and Sequence

### Milestones
1. <Milestone 1>: <Description>
   - Phase A: <...>
   - Phase B: <...>
2. <Milestone 2>: <Description>
   - Step 1: <...>
   - Step 2: <...>

<Describe relative dependencies between components, not time estimates>

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | <...> | AC-1 | coding | - |
| task2 | <...> | AC-2 | analyze | task1 |

## Claude-Codex Deliberation

### Agreements
- <Point both sides agree on>

### Resolved Disagreements
- <Topic>: Claude vs Codex summary, chosen resolution, and rationale

### Convergence Status
- Final Status: `converged` or `partially_converged`

## Pending User Decisions

- DEC-1: <Decision topic>
  - Claude Position: <...>
  - Codex Position: <...>
  - Tradeoff Summary: <...>
  - Decision Status: `PENDING` or `<User's final decision>`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# transpose_matmul — Phase 3 draft (regime / shape specialization)

Operator: `transpose_matmul` (NKIBench case 2). `out = lhs^T @ rhs`,
lhs (K,M)=(2048,4096) K-major, rhs (K,N)=(2048,10944), out (M,N)=(4096,10944),
fp32. MACs = M·N·K ≈ 9.17e10. Baseline **4.849615 ms**.

Start point: **tmm_v3_mblk16_bf16_split** (`runs/tmm_v3_mblk16_bf16_split.py`),
the promoted phase-2 kernel — **1.334x (3.6338 ms)**, full-5-seed L2 PASS
(rel-L2 4.4515e-6). Compensated bf16x2 3-product split of both operands
(hi@hi + hi@lo + lo@hi, drop lo@lo), no explicit transpose (K arrives on the
partition axis), M_BLK=16 (2 blocks of 2048 rows), N_CHUNK=456. `tmm_v1.py`
(fp32, 1.026x) is retained as the guaranteed pure-fp32 fallback.
Phase-2 disposition: `profile/tmm_v2_bf16_split_digest.md`.

The phase-3 brief is *shape specialization*: "analyze where time goes across
the tensor's structure and specialize only where the measured win justifies the
added complexity." The honest answer this phase must defend is **the dominant
lever is exhausted and the shape is edge-free**, so phase 3 is a bounded
gap-closing screen with the strong prior that the correct outcome is to
**FINALIZE tmm_v3** — mirroring the `bmm` phase-3 result (already at the floor →
finalize) rather than the `silu`/`rmsnorm_matmul` result (a new lever appeared).

---

## 1. Where the time goes (established, quantified against the hard floor)

tmm_v3 is **PE-BOUND and within 3.76% of the theoretical arithmetic ceiling.**
Full-5-seed dump (`profile/tmm_phase2r1_full_v3.txt`):

| metric | value | reading |
|---|---|---|
| wall p50 | **3.6338 ms** | 1.334x over baseline |
| TRUE PE-active/inf | **3.564 ms** | the matmul IS the wall clock |
| PE active % | **98.08%** | PE is essentially the entire wall |
| Vec / Scl / GpSimd / DMA % | 13.7 / 6.7 / 14.2 / 22.6 | all hidden well under PE |
| HBMrd / HBMwr | 229.0 / 179.3 MB | below the v1 floor; no spill (psum drain 768) |
| matmul_instruction_count | 36864 | 3.0 instr/site over 12288 sites |
| MFU | 48.2% | bf16-mix on a fp32-defined problem |

### The cost-model floor (kernel-cost-analysis, trn2)

The Tensor Engine streams the *moving* operand one (bf16 double-pumped: two)
column(s) per cycle; the stationary weight load pipelines behind the previous
matmul's moving stream, so a 456-wide moving fully hides the 128-cycle load. The
floor is therefore total-moving-columns / 2 / freq:

```
moving-col-cycles = (M/128 m-subtiles) · (K/128 kt) · (3 products) · N
                  = 32 · 16 · 3 · 10944 = 16,809,984
PE floor (bf16, 2 col/cyc, 2.40 GHz) = 16,809,984 / 2 / 2.40e9 = 3.502 ms
```

- **Ceiling speedup = 4.849615 / 3.502 = 1.385x.**
- v3 PE-active 3.564 ms is **1.77% above this floor** (per-instruction 96.7 ns
  measured vs 95.0 ns floor → ~1.7 ns residual bubble/instr; the weight load is
  confirmed fully hidden by the wide moving stream).
- v3 wall 3.6338 ms is **3.76% above the ceiling**; **1.96% of that (69.8 µs) is
  the PE-idle gap** (wall − PE-active), and the remaining ~1.8% is the PE-active
  bubble already near the floor.

**Conclusion:** the only two slivers left are (a) the ~1.8% PE-active bubble —
essentially unrecoverable, it's the systolic warm-up per matmul — and (b) the
~2.0% PE-idle gap where the Tensor Engine waits on the limb-build prologue /
DMA. The dominant term (PE-active) is set by the product count, which §3 shows
is a hard numeric floor.

---

## 2. Shape structure — already fully specialized, edge-free (the crux for a "shape" phase)

The phase-3 brief asks to specialize tile-size regimes / partition-free splits /
edge tiles. **This shape has no ragged structure to specialize:**

| axis | size | tiling | remainder |
|---|---|---|---|
| M (output rows) | 4096 | 32 × 128 subtiles, M_BLK=16 | **0** (exact) |
| K (contraction) | 2048 | 16 × 128 kt tiles on partition | **0** (exact) |
| N (output cols) | 10944 | 456 × 24 chunks | **0** (exact divisor) |

- N=10944 = 2⁶·3²·19. The phase-1 choice N_CHUNK=456 (= 10944/24, ≤512 PSUM
  width) makes **every tile full-size — zero tail masking anywhere.** There is
  no edge tile to give a different regime.
- Phase 2 already swept the one free tiling knob and **measured-rejected wider
  chunks in both forms**: N_CHUNK=512 mask-free 192-tail (`tmm_v4`, wall +1.12%)
  and N_CHUNK=512 masked-tail (`tmm_v5`, wall +4.2%). The kernel is
  **PE-column-bound, not issue-overhead-bound**, so fewer/wider matmuls cannot
  help — total PE columns pushed are invariant.
- M_BLK is pinned at 16 by the AC-4 read floor (M_BLK=8 → 448 MB read regression;
  M_BLK=32 → single 4096-row block would need 256 KB/partition of resident bf16
  limbs, **spilling** the 208 KB trn2 SBUF budget).

So the classic "shape specialization" moves (edge-tile fast path, partition/free
re-split, per-regime tile size) are **all no-ops or already-rejected here.** A
"shape" phase on a perfectly-divisible, single-regime GEMM is largely a
confirmation that no specialization is warranted.

---

## 3. The dominant lever (product count) is a proven numeric floor — cannot be lowered

PE-active scales **linearly** with the number of bf16 products per site (3.502 ms
at 3 products; would be 2.335 ms at 2). The only way to beat the ceiling is fewer
products while still clearing the 2e-5 relative-L2 gate. I checked this offline
(zero remote spend, full K=2048 contraction, N(0,1) inputs, seeds {42,0,84}):

| scheme | products | rel-L2 | vs 2e-5 gate |
|---|---|---|---|
| plain bf16 | 1 | 2.35e-3 | **FAIL ×117** |
| bf16 split lhs only (drop rhs_lo) | 2 | 1.66e-3 | **FAIL ×83** |
| bf16 split rhs only (drop lhs_lo) | 2 | 1.66e-3 | **FAIL ×83** |
| **bf16 3-product (current)** | **3** | **4.45e-6** | **PASS (4.5× under)** |
| plain fp16 | 1 | 2.94e-4 | **FAIL ×15** |
| fp16 split (drop one lo) | 2 | 2.08e-4 | **FAIL ×10** |
| fp16 3-product | 3 | 3.68e-7 | PASS (over-accurate) |

**Reading:** any 2-product scheme leaves one operand at 7-bit (bf16) or 10-bit
(fp16) mantissa, and the un-refined operand's rounding error dominates at
~1.7e-3 / ~2e-4 — 10–83× over the gate. Refining **both** operands (≥3 products)
is mandatory. **fp16 does not rescue a 2-product scheme** (still 10× over), and
the fp16 exponent (5 bits) is a correctness risk on other seeds even where the
mantissa would suffice — a bad trade for zero speed gain (fp16 limbs cost exactly
the same PE cycles as bf16). So **3 products is the numeric floor; PE-active
cannot drop, and 1.385x is a hard ceiling.** D4 (4-product) is the wrong
direction (more work for accuracy already 4.5× spare).

This closes the only lever that could move the dominant term. Phase 3 therefore
cannot chase PE-active; it can only try to shave the ~2% PE-idle gap.

---

## 4. Lever enumeration for the residual ~2% PE-idle gap

Ranked. All are gap-only (best case ≈ the 69.8 µs / 1.96% idle); none can touch
PE-active. Every candidate must hold **correctness (rel-L2 unchanged 4.4515e-6),
AC-4 (HBM read ≤ 229 MB, no spill), and beat a same-session tmm_v3 control band**
before adoption — the phase-2 discipline.

| # | lever | mechanism | expected | risk | priority |
|---|---|---|---|---|---|
| **E1** | **fewer/wider N chunks (912-wide, 24→12 iters)** | halves the number of limb-build prologues (rhs limb rebuild per chunk) → fewer PE-idle prologue bubbles | LOW (≤~1.5%) | MED (912>512 needs 2 PSUM banks/site → more matmuls; likely affine_range anti-lever) | screen `--fast` first |
| **E2** | **double-buffer / prefetch the next rhs chunk's limbs** while the current chunk's 16 subtiles matmul | overlap the rhs limb build (Vec/Scl, currently partly exposed at chunk boundaries) with PE | LOW (≤~2%, caps at the idle gap) | MED (2× rhs-limb SBUF; can enlarge the live set and *constrain* the affine_range pipeline — the bmm anti-lever) | screen `--fast` only if E1 shows the prologue is the idle source |
| E3 | build lhs limbs once, keep bf16 rhs-limb build off the PE critical path via engine placement (GpSimd vs Vec for the subtract) | move limb-build cost onto an idle engine | very LOW | LOW-MED | screen only if E1/E2 localize the gap to limb-build |
| E4 | 4-product / higher precision | — | **none** (§3: accuracy already 4.5× spare) | HIGH latency | **SKIP** |
| E5 | edge-tile / per-regime tile size | — | **none** (§2: shape is edge-free) | — | **SKIP (no-op)** |

### Why the honest prior is FINALIZE, not "expect a win"

- The gap is **1.96%** and PE is at **98%**. The largest conceivable E1+E2 win is
  the idle gap itself (~2%), and the realistic capture is a fraction of that.
- **E1 (wider chunks) already lost in phase 2** in its N_CHUNK=512 form (+1.12%
  and +4.2%). 912-wide is the same class of lever (fewer, wider matmuls on a
  PE-column-bound kernel) and additionally needs 2 PSUM banks per site — the
  strong prior is it regresses again. It is on the list only because 912 is an
  *exact* divisor (no tail) and it halves the *prologue* count (a different
  sub-mechanism than v4/v5's chunk-width change), so one `--fast` screen is
  cheap insurance, not an expected win.
- **E2 (double-buffer)** directly attacks the idle gap, but the `bmm` phase-3
  memory is a live warning: enlarging a cross-iteration resident live set
  *constrained* the affine_range software pipeline and **regressed monotonically**
  with no spill. `silu` phase-2 also saw ping-pong regress. This is the most
  principled attempt but is flagged as a likely anti-lever on this compiler.
- Sibling precedent: `bmm` phase 3 concluded "within 13.4% of the fp32 ceiling →
  finalize; every reschedule regressed." tmm is **far tighter (3.76% of ceiling)**,
  so the room is smaller still.

**Expected phase-3 outcome:** run the E1 `--fast` screen (cheap) and, if it does
not clear the band, run one E2 `--fast` screen; the most probable result is both
reject (measured, recorded as negative evidence like v4/v5), and **tmm_v3 is
FINALIZED at 1.334x with the ceiling analysis documented.** A ≤1–2% gap-closing
win is possible but not the base case. Either way this is a *measured* verdict,
not an assumption — the phase-2 protocol (same-session anchor band, AC-4 read
floor, correctness parity) decides.

---

## 5. Measurement protocol (unchanged discipline; gap-only levers)

Per `[[BL-20260709-fast-vs-full-run-latency]]` and the phase-2 AC-set:

1. **No offline gate needed** — E1/E2/E3 are bit-identical rescheduling/tiling of
   the *same* 3-product arithmetic (rel-L2 stays 4.4515e-6). Confirm parity on
   the run, don't re-derive it.
2. `verify.py --fast` on the candidate → correctness + latency direction.
3. `runs/dump_metrics.py --fast` on the candidate AND a same-session tmm_v3
   anchor → read **wall p50**, **TRUE PE-active/inf**, **hbm_read_bytes**,
   **psum_read_sbuf_write_count**. Establish the tmm_v3 control band (|v3a−v3b|).
4. **ADOPT only if:** wall beats v3 by > max(band, 3%) **AND** rel-L2 == 4.4515e-6
   **AND** hbm_read ≤ 229 MB **AND** no spill (psum 768, write 179.3 MB). This is
   the exact bar phase-2 used to promote v3 and reject v4/v5.
5. **REJECT (measured) otherwise** — record the wall delta, PE-active, and HBM as
   first-class negative evidence in `benchmark.csv` / `candidates.jsonl` (the v4/v5
   pattern). PE-active RISING or read growing = immediate reject.
6. On the (unlikely) adopt path, confirm on the **full 5-seed** run before
   promoting; keep tmm_v1 as the fp32 fallback regardless.

---

## 6. Exit criteria / plan for the ≤5-iteration budget

- **Iter 0 (analysis, done in this draft):** the dominant lever is a proven
  numeric floor (§3), the shape is edge-free (§2), tmm_v3 is within 3.76% of the
  1.385x hard ceiling (§1). Records the ceiling as the phase-3 headline datum.
- **Iter 1:** E1 (N_CHUNK=912, 12 chunk-iters) `--fast` screen vs same-session
  v3 anchor. Strong prior: reject (wider-chunk class already lost). Deliverable
  `runs/tmm_v6_nchunk912.py` (parent tmm_v3) — record measured.
- **Iter 2 (conditional on E1's profile localizing the idle to limb-build):**
  E2 double-buffer rhs limbs `--fast` screen. Prior: likely affine_range
  anti-lever (bmm). Deliverable `runs/tmm_v7_dbuf_rhs.py` — record measured.
- **Iter 3 (only if a screen clears the band):** full-5-seed confirm + control
  band → promote; else FINALIZE tmm_v3.

**Success (base case):** the phase-3 ceiling analysis is documented, the two
gap-only screens are measured-rejected (or one narrowly adopted), and the
op is **FINALIZED at tmm_v3 1.334x** (or slightly better) with full-5-seed L2
PASS, HBM at the floor, and tmm_v1 retained as the fp32 fallback. The valuable
phase-3 deliverable here is the *proof the kernel is near-optimal* (ceiling +
product-count floor + edge-free shape), not a new speedup. Record every candidate
in `benchmark.csv` + `candidates.jsonl` (DAG parent links); profiling evidence
under `profile/`. Never regress correctness or the AC-4 read floor.

--- Original Design Draft End ---
