# NKI Kernel Optimization Prompts

Task prompts for the KDA-for-Trainium workflow, organized by operator and phase.

> **The per-operator `<op>/phaseN.md` files are generated, not checked in.** Run
> `python3 prompts/_gen_prompts.py` to (re)generate them for all NKIBench operators
> from the templates + per-op metadata in that script. This keeps the repo free of
> benchmark-derived content and lets you regenerate prompts whenever the templates
> or operator set change.

```
prompts/
  _gen_prompts.py     # generator: writes <op>/phase{1,2,3}.md for every op
  <op>/               # generated (git-ignored per-op prompt dirs)
    phase1.md   # research + first CORRECT NKI kernel
    phase2.md   # profile-driven optimization (<=5 iters per direction)
    phase3.md   # shape / regime specialization
```

## Common workflow (per phase)

**Automated (default):** `bash scripts/run-op.sh <op>` from the repo root runs all
three phases headlessly (draft → gen-plan → commit → RLCR loop → commit). See the
top-level README §2 for flags (`--from-phase`/`--to-phase`, `--dry-run`).

**Manual (per phase):**
1. Start a fresh Claude Code session inside the task workspace (`workspaces/<op>/`).
2. Paste the phase prompt (`prompts/<op>/phaseN.md`).
3. The agent investigates: the numpy reference, the baseline NKI kernel, the
   profiler's `summary_metrics`, the NKI skills (`.claude/skills/`), and NKI docs.
4. The agent writes its plan draft to `docs/draft-phaseN.md`.
5. Run `/humanize:gen-plan --input docs/draft-phaseN.md --output docs/plan-phaseN.md`
   to turn the draft into a detailed plan (per-phase names: gen-plan errors if the
   output already exists).
6. Run `/humanize:start-rlcr-loop docs/plan-phaseN.md --skip-quiz` to implement and iterate.
7. Score each candidate (from the task workspace `workspaces/<op>/`):
   `python3 ../../verify.py --op <op> --candidate runs/<file>.py --fast`
8. Record perf changes in `benchmark.csv`, candidates in `candidates.jsonl` (DAG),
   profiling evidence under `profile/`.

## Shared requirements (unless a phase overrides)

- The kernel MUST pass the NKIBench correctness gate: relative-L2
  `‖v_k − v_r‖₂ < rel_tol·‖v_r‖₂` (rel_tol = 2e-5; 3e-5 for mamba) across seeds
  `[0,21,42,63,84]`, fp32. `verify.py` enforces this.
- Optimize p50 on-device latency; the score is speedup over the baseline kernel.
- Implement in NKI Python with a single `@nki.jit def kernel(...)` entry point
  matching the baseline's tiled input signature.
- Actively use the NKI skills for evidence-based decisions (see each phase's Tools
  section) rather than guessing.
- The kernel runs in TILED space (partition dim ≤ 128). The reference `forward`
  runs in natural space; the harness reconciles them via `transform_nki_outputs`.

## Iterative refinement

These prompts are starting points. Re-invoke a phase with a higher target speedup
or stricter promotion rules as the search matures. Add human hints directly into
the prompt when you have them: likely bottleneck engine, tiling strategy to try,
directions known to be risky.
