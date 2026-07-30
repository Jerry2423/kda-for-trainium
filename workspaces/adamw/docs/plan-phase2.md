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

# adamw (M10944 N2048, fp32) — Phase 2 implementation draft (profile-driven opt)

## Starting point

Phase-1 winner **`runs/adamw_v1.py`** — the first correct fused kernel:
- **0.6180 ms, 2.112x** over baseline (full 5-seed PASS; `--fast` 0.6166 ms / 2.116x).
- Structure: `nl.affine_range(86)` over `M=10944` in `[128, 2048]` row tiles (last
  tile partial, 64 valid rows); 4 **masked** loads (`row < 10944`) → 6-op fused chain
  (2 Scalar `activation` square/rsqrt + 4 Vector `scalar_tensor_tensor`) → 1 masked
  store. Folded algebra `new_theta = 0.99999·theta − 0.001·(9m+g)·rsqrt(999v+g²)`.
- Profiler digest: **DMA 95%, Vec 64%, Scl 28%, PE 0%**; HBMrd 359 MB, HBMwr 90 MB.

Phase 1 already collapsed the baseline's fragmented 20-buffer / `[128,512]` traffic to
the read-once/write-once floor, which is why it jumped straight to 2.112x (far past the
plan's cautious 1.0–1.3x). Phase 2's job is to squeeze the remaining scheduling slack —
not to cut traffic (there is none left to cut).

## Bottleneck: DMA-bound, at the traffic floor

Read the phase-1 numbers as a roofline:

| metric | v1 | meaning |
|---|---|---|
| DMA active | **95%** | the constraint |
| Vec active | 64% | #2 engine, **hidden** under DMA |
| Scl active | 28% | the two nonlinearities (square, rsqrt), off Vector |
| PE / MFU | 0% | no matmul (irrelevant) |
| HBMrd | 359 MB | 4 × 89.66 MB = **read-once** for theta,g,m,v |
| HBMwr | 90 MB | 1 × 89.66 MB = **write-once** for new_theta |

Total traffic = **448.27 MB** (4R + 1W). Effective bandwidth =
448.27 MB / 0.6180 ms ≈ **727 GB/s**. The silu kernel on this same trn2 profiler
sustained ~**799.5 GB/s** at its best tiling — so there is a **~5% DMA-idle bubble**
(727 vs ~800 GB/s) and *nothing else*.

**There is no traffic lever.** All four inputs are genuinely read (every element of
each feeds the update), the output is written exactly once, and the HBM tensors are
fp32 supplied by the harness — reading/storing in bf16 would neither reduce HBM bytes
(the tensors *are* fp32 in HBM) nor survive the 2e-5 L2 gate on a pure elementwise op
with no reduction to average the error down (cf. rmsnorm_matmul, where K-averaging is
what made compensated-bf16 acceptable; there is no such averaging here). So:

> **The entire phase-2 prize is the ~5% DMA bubble. Ceiling ≈ 448.27 MB / 799.5 GB/s
> ≈ 0.561 ms ≈ 2.33x. Realistic target ≈ 2.15–2.30x.** The lever is *scheduling*
> (pipeline depth / burst width), never traffic.

### Vector floor sanity check (so finer tiling doesn't backfire)

Measured Vec-active ≈ 0.64 × 0.618 = **0.395 ms**, comfortably under DMA's 0.59 ms
proxy — the 4-Vector chain is hidden. Going *finer* (more, smaller chunks) raises the
**per-instruction fixed overhead** (Vector `semaphore_start` = 268 ns, `write_drain` =
161 ns each). If we go too fine, the Vector instruction count × fixed overhead can lift
Vec-active across the DMA line and Vector becomes the new bottleneck. This is exactly
why silu's latency-vs-fineness curve **turned up** past its optimum. So finer tiling is
a *bounded* sweep with a turn to detect, not "smaller is always better."

## Direct precedent: silu phase-3 (finer free-axis tiling harvests the DMA bubble)

Same profiler, same DMA-bound-elementwise shape. silu went
**v1 (depth-32, 28 KB burst) → v3_s7 (depth-224, 4 KB burst): 0.3009 → 0.2940 ms
(+2.2%), effBW 780.6 → 799.5 GB/s**, purely by reshaping the row-major tensor into a
**memory-identical finer view** and walking a single deeper `affine_range`. Key
recorded findings that shape this plan:
- **Finer WON; wider REGRESSED.** silu phase-2's wider burst-batching + ping-pong both
  regressed; only *finer* free-axis tiling harvested the bubble.
- **The curve turns up.** s=8 (896-wide, 3.5 KB burst) regressed vs s=7 (1024-wide,
  4 KB); 1024 = 2¹⁰ was the observed minimum — descriptor/issue overhead below ~4 KB
  overtook the pipeline-depth benefit.
- **It's a pure scheduling lever** — HBM traffic identical, SBUF tiny.

### The one adaptation for adamw: it is a 5-DMA-per-iteration kernel, not 2

silu moves **2 tensors/iter** (1 load + 1 store); adamw moves **5** (4 loads + 1 store).
At a given chunk width the DMA engine issues 2.5× more descriptors per iteration, so the
fixed per-descriptor overhead is amortized over more work and the burst-efficiency
crossover should sit at a **wider** chunk than silu's 1024. Also, v1 *already* runs an
8 KB burst at depth 86 (much finer than silu-v1's 28 KB / depth-32 start), so the
headroom is smaller and the optimum is likely near or slightly above 1024, not far
below it. → **Anchor the sweep at 1024 but probe wider (1216, 1536), not just finer.**

## The phase-2 kernel shape: mask-free `(128, ITERS, CH)` reshape-view stream

`M·N = 22 413 312 = 128 × 175 104` **exactly**, so the whole tensor is a clean
`(128, 175104)` flat view with all 128 partition lanes live. `175104 = 2¹⁰ · 3² · 19`,
so it has exact divisors across the whole interesting burst band. Reshape each input
(and the output) to `(128, ITERS, CH)` with `ITERS · CH = 175104` and walk one flat
`nl.affine_range(ITERS)` — identical to silu v3_s7. This delivers **two wins at once**:

1. **Mask-free.** Unlike v1's `10944 = 128·85 + 64` partial tail (a predicate on every
   load/store), every chunk here is exactly rectangular → **no mask on any DMA**,
   removing the tail footgun and the predicate cost.
2. **Tunable pipeline depth / burst.** `CH` sets the per-partition burst (`CH·4` bytes)
   and `ITERS = 175104/CH` sets the pipeline depth — the exact silu lever.

**Correctness is layout-invariant and verified.** The op is pure elementwise, so any
element-bijective reshape-and-reshape-back is exact — confirmed in numpy:
`(M,N) → (128, ITERS, CH) → (M,N)` round-trips **bit-exact** (`np.array_equal` True),
and the folded algebra over the reshaped view holds **rel-L2 = 3.4e-8** vs the numpy
reference (≈ 580× under the 2e-5 gate). The compute chain is unchanged from v1 (2 Scalar
+ 4 Vector), just applied per `[128, CH]` chunk. No mask, no tail, no algebra change.

Rejected layout alternative: the `P = 96×114` "no-mask row shape" (10944 = 96·114) is
strictly *worse* — it leaves 32/128 partition lanes idle, i.e. 25% lower per-instruction
bandwidth on a DMA-bound kernel. The flat 128-partition reshape is mask-free *and* uses
all lanes, so it dominates. Do not pursue 96×114.

## Directions, ranked by expected benefit ÷ risk

### D1 — mask-free reshape-view + bounded finer/wider CH sweep  *(PRIMARY)*
The whole phase. Port v1 → the `(128, ITERS, CH)` stream above and sweep `CH` over exact
divisors of 175104 to find the DMA-bubble minimum. Expected **+2–8%** (→ ~2.15–2.30x),
directly analogous to silu's +2.2%. Risk: **low** (proven precedent, identical shape,
correctness layout-invariant). Guard rail: watch Vec% — if it climbs toward DMA% at the
finer end, that side of the sweep is bounded by the Vector floor, not the bubble.

Candidate `CH` (all exact divisors; burst = CH·4 B/partition):

| CH | ITERS | burst/part | role |
|----|-------|-----------|------|
| 1824 | 96 | 7.12 KB | ≈ v1's 8 KB burst, but **mask-free** — isolates the mask-removal gain from the finer-tiling gain |
| 1536 | 114 | 6.00 KB | wider probe (5-DMA/iter hypothesis) |
| 1216 | 144 | 4.75 KB | mid probe |
| **1024** | **171** | **4.00 KB** | **anchor** = silu's proven 2¹⁰ optimum |
| 768 | 228 | 3.00 KB | finer bracket probe (only if 1024 screens ≥ 1216) |

### D2 — explicit double-buffer / ping-pong across chunks  *(SECONDARY, likely-reject)*
If `affine_range` auto-pipelining leaves residual bubble that CH tuning can't close, try
manual 2-buffer prefetch (scheduling precedents #1/#2). **Priced low** because silu
phase-2 already tested ping-pong on this exact profiler and it **regressed** — the
compiler's `affine_range` pipeliner over a deep flat loop already overlaps DMA with
compute. Budget: **at most 1 probe**, and only if D1 plateaus below ~2.25x with a
visible DMA bubble the sweep didn't close. Expected: reject.

### D3 — compute-chain rebalance (keep Vector hidden)  *(DEFENSIVE, folded into D1)*
Not a latency lever on its own (DMA-bound, compute hidden), but a *guard*: if the finer
end of the D1 sweep lifts Vec% toward DMA%, rebalance one Vector op off the Vector
engine — e.g. fold the final `0.99999·theta − term` into a Scalar `activation`
(bias/scale) or move a `scalar_tensor_tensor` to GpSimd — to keep Vector under DMA.
Only act if the sweep exposes Vector; otherwise document and skip.

## Measurement discipline (mirror silu's promote/reject rule)

Profiler p50 has run-to-run noise, and the whole prize here is a few percent, so screen
and promote against an explicit noise band — same protocol that promoted silu v3_s7:

1. **Screen** each candidate with `--fast` (seed 42) vs a same-session `--fast` re-run of
   v1; treat `|Δ| < Jf ≈ 0.0005 ms` as noise (no decision). Record every result in
   `benchmark.csv` regardless.
2. **Bounded sweep, detect the turn.** Run the anchor + neighbors; when latency starts
   rising on one side, take **one** bracket probe past the observed minimum on the other
   side (the silu s=8 rule) and **stop**. ≤ 5 scored candidates total for D1.
3. **Promote-test** the sweep-best with the **full 5-seed** measurement, interleaved
   `v1 / best / v1 / best / v1` (A0,B0,A1,B1,A2). Promote **only if** `Bbar < Abar − J`
   **and** every B < max(A) **and** HBMrd+HBMwr still = 448 MB (traffic floor intact,
   i.e. no accidental extra pass) **and** the 5-seed L2 gate PASSES. Otherwise keep v1.
4. **Never regress correctness.** Any candidate that fails the L2 gate on any seed is
   rejected outright; the folded algebra and mask-free rectangular shape must be
   preserved exactly (no eps re-introduction needed — v_hat = 999v+g² > 0 holds).

## Iteration budget (≤ 5 for D1; D2/D3 conditional)

1. **CH=1024** reshape-view port (silu anchor) — screen vs v1.
2. **CH=1536** (wider) — screen.
3. **CH=1216** (mid) — screen. → pick best of {1024, 1536, 1216}.
4. **Bracket probe** on the winning side (768 if finer wins; 1824 if wider wins) —
   screen, confirm the turn.
5. **Promote-test** the sweep-best (interleaved full 5-seed) → promote or keep v1.

D2 (1 ping-pong probe) and D3 (1 rebalance) only fire if D1 plateaus below target with
a diagnosable cause; each is a single probe, documented keep/revise/reject.

## Expected outcome & evidence

- **Most likely:** a `CH ≈ 1024–1536` reshape-view kernel lands ~**2.15–2.25x**
  (effBW ~760–790 GB/s), mask-free, traffic unchanged at 448 MB. Promote as
  `adamw_v2_ch<CH>.py`; keep v1 as the documented fallback.
- **Possible null result:** if v1's existing 8 KB / depth-86 tiling is already inside
  the bubble minimum, the sweep may not clear the noise band (Δ < J). Then the honest
  outcome is "v1 already near the DMA floor; finer tiling within noise" — record it and
  keep v1 (still a valid phase-2 finding: the floor was already reached in phase 1).
- Per candidate: before/after `--fast` latency + full profiler digest (DMA/Vec/Scl/HBM)
  in `benchmark.csv`; candidate row + parent DAG link in `candidates.jsonl`; the winning
  direction's profile under `profile/`.

--- Original Design Draft End ---
