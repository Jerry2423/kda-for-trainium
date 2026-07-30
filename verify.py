#!/usr/bin/env python3
"""NKIBench evaluation harness — PLACEHOLDER.

This repository ships the KDA-for-Trainium *driver* (adapter, prompts, workspace
evidence), but NOT a profiler backend. The original `verify.py` scored kernels on
a remote NKI profiler through a proprietary client that is not part of this
open-source release. This placeholder documents the input/output contract so you
can wire in your own profiling/evaluation backend.

Implement `verify.py` against your backend so it honors the contract below; the
rest of the harness (adapter, prompts, run-op.sh, summary.py) is unchanged.

────────────────────────────────────────────────────────────────────────────────
INPUT  (command-line)
────────────────────────────────────────────────────────────────────────────────
  --op <name>            One NKIBench operator (e.g. silu). Mutually exclusive
                         with --pilot.
  --pilot                The pilot operator subset (see adapter PILOT_OPS).
  (no selector)          All operators defined in $NKIBENCH_ROOT/summary.json.
  --candidate <path>     Path to a candidate kernel .py for a SINGLE op (requires
                         --op; not combinable with --build-baselines). Defaults to
                         the operator's baseline kernel.
  --build-baselines      Measure baseline kernels and cache to baselines.json.
  --fast                 Quick check: 1 seed, fewer warmup/iters.
  --num-cores {1,2,4,8}  Cores to compile/run single-logical-nc on (default 1).
  --timeout <seconds>    Per-op profiling timeout (default 900).

  Environment:
    NKIBENCH_ROOT        Path to AccelOpt's NKIBench (default ../AccelOpt/NKIBench;
                         github.com/zhang677/AccelOpt). Read-only benchmark def.

  What a case provides (via adapter/nkibench_case.py -> resolve_case(...).
  profile_kwargs()): a self-contained kernel module `src_code` whose `get_inputs()`
  yields TILED inputs, plus the numpy reference `initial_code` (`get_numpy_inputs`,
  `forward`, `transform_nki_outputs`), the multi-seed set [0,21,42,63,84], the
  relative-L2 tolerance (rtol=2e-5; 3e-5 for mamba, atol=0), and the compiler
  flags (--disable-dge --logical-nc-config=1). Your backend consumes these to
  compile+run the kernel and compare against the golden.

────────────────────────────────────────────────────────────────────────────────
OUTPUT
────────────────────────────────────────────────────────────────────────────────
  Per operator, your backend must yield:
    passed      : bool   — correctness gate = every seed passes relative-L2
                           (‖v_k − v_r‖₂ < rel_tol·‖v_r‖₂, fp32).
    latency_ms  : float  — p50 on-device latency (None if the kernel failed).
    error       : str|None — failure detail when not passed.
    summary_metrics : dict — per-engine util / MFU / HBM read+write bytes / DMA
                           active time, for bottleneck triage (optional but used
                           by the one-line digest and by the agent's analysis).

  Derived + printed:
    speedup     = baseline_latency_ms / candidate_latency_ms   (per op)
    aggregate   = geomean of per-op speedups
    A PASS/FAIL + latency + speedup line per op, then the geomean summary.

  Side effects:
    --build-baselines  writes {op: {baseline_latency_ms, case_id, kernel}} to
                       baselines.json (used as the speedup denominator).

  Exit code: 0 if all scored ops pass (or all baselines measured), else nonzero.

────────────────────────────────────────────────────────────────────────────────
The correctness contract is fixed by NKIBench/AccelOpt and MUST NOT be relaxed:
relative-L2 across all five seeds, fp32, rtol as above. See README.md and CLAUDE.md.
"""

from __future__ import annotations

import sys


def main() -> int:
    sys.stderr.write(
        "verify.py is a placeholder in this open-source release — it does not "
        "include a profiler backend.\n"
        "Implement it against your own profiling/evaluation backend, honoring the "
        "input/output contract documented at the top of this file.\n"
        "The adapter (adapter/nkibench_case.py) already assembles each NKIBench "
        "case into the kwargs your backend needs (see NKIBenchCase.profile_kwargs).\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
