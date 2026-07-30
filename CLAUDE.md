# Agent Instructions — KDA for Trainium

This is a task implementation workspace: drive NKI kernel optimization on NKIBench,
scored via the remote Trainium profiler. Mirror the GPU KDA discipline; swap the
target layer for NKI/Trainium.

## Rules

- Use English for all repository-facing files, prompts, and commit messages.
- **Never edit** `$NKIBENCH_ROOT/{kernels,reference,seeds,summary.json}` — it is the
  benchmark definition, from [AccelOpt](https://github.com/zhang677/AccelOpt)
  (defaults to `../AccelOpt/NKIBench`). Candidates go in the task's
  `workspaces/<op>/runs/`; never hand-tune a baseline.
- Generated artifacts belong in the task workspace: `workspaces/<op>/runs/` and
  `workspaces/<op>/profile/`. Candidate kernel **`.py` sources under `runs/` are
  tracked** (so `codex review` and git history see the actual kernel, not just the
  `benchmark.csv` / `candidates.jsonl` evidence rows); other `runs/` artifacts and
  all of `profile/` are git-ignored. Do not write to a top-level `runs/`,
  `outputs/`, or `profile/` — those are not created and only `outputs/` is ignored.
- Correctness is NKIBench's relative-L2 gate across seeds `[0,21,42,63,84]`, not
  allclose. `verify.py` already gates on `l2_norm_passed` — trust it.
- `verify.py` is a placeholder here (no profiler backend shipped); it documents the
  input/output contract. Wire in your own evaluation backend before scoring.

## Workspace layout

Each operator gets its own evidence directory under `workspaces/<op>/`:
```
kda-trainium/               (shared tools — verify.py, adapter, prompts, scripts, baselines.json)
  workspaces/
    silu/                   (per-task: docs/, runs/, profile/, benchmark.csv, candidates.jsonl)
    matmul/
    ...
```
Create a new task: `bash scripts/new-task.sh <op>`. Run the KDA loop from
**inside** `workspaces/<op>/` so evidence paths are workspace-relative.

## Workflow (per operator)

**Automated (default):** from the repo root, `bash scripts/run-op.sh <op>` runs all
three phases headlessly. Per phase it makes the agent write `docs/draft-phaseN.md`,
runs `/humanize:gen-plan` → `docs/plan-phaseN.md`, **commits** draft+plan (the RLCR
loop requires a clean tree for `codex review`; gen-plan's auto-start does NOT commit,
so the driver owns this), runs `/humanize:start-rlcr-loop --skip-quiz`, and commits
the result. Logs in `workspaces/<op>/logs/`; re-running resumes (skips completed
steps). Scope with `--from-phase N --to-phase N`; preview with `--dry-run`.

**Manual (interactive), if driving by hand:**
1. `cd workspaces/<op>/` — the KDA loop runs here; evidence is relative to this dir.
2. Read the op's numpy reference (`../../AccelOpt/NKIBench/reference/<op>_...py`) and
   the baseline kernel.
3. Paste `prompts/<op>/phaseN.md`. Agent investigates first, then writes `docs/draft-phaseN.md`.
4. `/humanize:gen-plan --input docs/draft-phaseN.md --output docs/plan-phaseN.md` → detailed plan.
5. `/humanize:start-rlcr-loop docs/plan-phaseN.md --skip-quiz` → implement/iterate.
   (Per-phase filenames: gen-plan errors if the output already exists.)
6. Score every candidate (from this dir):
   `python3 ../../verify.py --op <op> --candidate runs/<file>.py --fast`
7. Record each perf change in `benchmark.csv`; each candidate in `candidates.jsonl`
   (with parent links as a DAG); profiling evidence under `profile/`.
8. Progress phase1 → phase2 (profile-driven opt) → phase3 (shape specialization).

## Skills (linked into .claude/skills/, symlinked from the NKI kernel library)

Claude Code discovers skills from `~/.claude/skills/` and `<project>/.claude/skills/`.
We use the project-level `.claude/skills/` here. Only skills that work WITHOUT
the NKI kernel library's local build-based test workflow are linked:

- `kernel-cost-analysis` → theoretical per-engine cost / bottleneck engine (no hardware).
- `kernel-optimization-kb` → real optimization precedents from NKI-kernel-library git history.
- `kernel-accuracy-debugging` → bug-pattern reference when a correctness (L2) gate fails.
- `nki-concept-docs` / `nki-api-reference` → NKI concepts and API lookups.

### Where performance data comes from (IMPORTANT)

Do NOT try to run `nki-separation-analysis`, `nki-roofline-analysis`,
`nki-dma-compute-order-check`, or `neuron-nki-optimization-analysis` here — they
collect data via `a local build-based separation test`
inside the NKI kernel library package (local BIR/NTFF artifacts) and via Neuron
Explorer. This harness measures on the REMOTE profiler instead, which already
returns per-engine utilization, MFU, HBM read/write bytes, and DMA active time
in each ProfileResult's `summary_metrics`. Read those numbers (surfaced by
`verify.py` / the profiler response) for bottleneck analysis; use
`kernel-cost-analysis` for the theoretical floor to compare against.

## External loop engine

`humanize` (Claude Code plugin): `/humanize:gen-plan`, `/humanize:start-rlcr-loop`.
