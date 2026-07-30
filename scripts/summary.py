#!/usr/bin/env python3
"""Summarize KDA-for-Trainium batch progress across all workspaces.

Reads each workspaces/<op>/candidates.jsonl for the best PASSING speedup and
each logs/phaseN.done marker for how far the op got, then prints a per-op table
+ running geomean over the ops that have a passing kernel. Read-only.

    python scripts/summary.py            # table + geomean + rerun list
    python scripts/summary.py --json     # machine-readable
"""
from __future__ import annotations
import argparse, glob, json, math, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def phases_done(op: str) -> list[int]:
    return [p for p in (1, 2, 3)
            if (ROOT / "workspaces" / op / "logs" / f"phase{p}.done").exists()]


# The NKIBench correctness contract is the full multi-seed gate; a candidate only
# counts as a *validated* result (eligible for best / geomean / promoted) if it
# cleared all of these seeds. This drops fast-screen rows (seeds=[42] or seeds=None)
# and explicitly rejected/dropped/screen verdicts, which carry passed=true but are
# NOT full-gate results — including them would overstate validated performance
# (e.g. a rejected 1-seed silu fast screen at 3.480x edging out the promoted 3.478x).
FULL_GATE_SEEDS = {0, 21, 42, 63, 84}
_NON_VALIDATED_VERDICTS = ("REJECT", "DROPPED", "SCREEN")


def is_validated(d: dict) -> bool:
    """True iff the candidate passed the FULL multi-seed gate with a numeric speedup
    and is not an explicitly rejected/dropped/screen record."""
    if not d.get("passed") or not isinstance(d.get("speedup"), (int, float)):
        return False
    seeds = d.get("seeds")
    if not (isinstance(seeds, (list, tuple)) and FULL_GATE_SEEDS.issubset(set(seeds))):
        return False
    v = str(d.get("verdict", "")).upper()
    return not any(bad in v for bad in _NON_VALIDATED_VERDICTS)


def scan_op(op: str) -> dict:
    ws = ROOT / "workspaces" / op
    cj = ws / "candidates.jsonl"
    best = None          # (speedup, candidate)
    promoted = None      # (speedup, candidate)
    n_pass = 0
    if cj.exists():
        for line in cj.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only full-gate validated results feed best / geomean / promoted;
            # fast screens and rejected/dropped records are ignored here.
            if is_validated(d):
                sp = d.get("speedup")
                n_pass += 1
                if best is None or sp > best[0]:
                    best = (sp, d.get("candidate"))
                # A verdict counts as promoted only if it affirmatively says so
                # ("promoted", "PROMOTED-breaks-fp32-floor", ...) and is NOT a negated
                # form. Guard against the substring trap where "PROMOT" also appears
                # inside "NOT-PROMOTED" / "NON-PROMOTED" verdicts (e.g.
                # "floor-confirmation-not-promoted") — those must be excluded.
                v = str(d.get("verdict", "")).upper()
                negated = any(
                    neg in v for neg in ("NOT-PROMOT", "NON-PROMOT", "NOT PROMOT")
                )
                if "PROMOT" in v and not negated:
                    if promoted is None or sp > promoted[0]:
                        promoted = (sp, d.get("candidate"))
    ph = phases_done(op)
    # An op counts as started if it has ANY phase-done marker OR a non-empty
    # candidates.jsonl. Check phase markers FIRST so a phase that produced markers
    # but no (or empty) candidates.jsonl — e.g. a docs-only or failed phase — still
    # reports its progress instead of being misreported as "not started".
    started = bool(ph) or (cj.exists() and cj.stat().st_size > 0)
    if ph == [1, 2, 3]:
        status = "COMPLETE"
    elif ph:
        missing = ",".join(str(p) for p in (1, 2, 3) if p not in ph)
        status = f"partial (rerun phase {missing})"
    elif started:
        status = "in progress (phase1)"
    else:
        status = "not started"
    return dict(op=op, best=best, promoted=promoted, n_pass=n_pass,
                phases=ph, status=status, started=started)


def all_ops() -> list[str]:
    # every workspace that exists, sorted by best speedup later
    return sorted(p.name for p in (ROOT / "workspaces").iterdir() if p.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    rows = [scan_op(op) for op in all_ops()]

    if args.json:
        out = [{**r, "best_speedup": r["best"][0] if r["best"] else None,
                "best_candidate": r["best"][1] if r["best"] else None,
                "promoted_speedup": r["promoted"][0] if r["promoted"] else None}
               for r in rows]
        print(json.dumps(out, indent=2))
        return

    # sort by best speedup desc, ops without a result at the bottom
    rows.sort(key=lambda r: (r["best"][0] if r["best"] else -1), reverse=True)

    print(f"{'op':26s} {'best':>8s}  {'phases':>6s}  status")
    print("-" * 70)
    speeds = []
    reruns = []
    for r in rows:
        bs = f"{r['best'][0]:.3f}x" if r["best"] else "—"
        if r["best"]:
            speeds.append(r["best"][0])
        ph = "".join(str(p) for p in r["phases"]) or "-"
        if "rerun" in r["status"] or (r["started"] if "started" in r else False):
            pass
        if "rerun" in r["status"]:
            reruns.append(r)
        print(f"{r['op']:26s} {bs:>8s}  {ph:>6s}  {r['status']}")

    print("-" * 70)
    if speeds:
        gm = math.exp(sum(math.log(s) for s in speeds) / len(speeds))
        print(f"ops with a passing kernel : {len(speeds)}/{len(rows)}")
        print(f"geomean(best speedup)     : {gm:.3f}x   (range {min(speeds):.3f}x .. {max(speeds):.3f}x)")
    else:
        print("no passing kernels yet")

    if reruns:
        print("\nneeds rerun (interrupted — phase1/earlier results preserved):")
        for r in reruns:
            missing = [p for p in (1, 2, 3) if p not in r["phases"]]
            lo, hi = missing[0], missing[-1]
            print(f"  bash scripts/run-op.sh {r['op']} --from-phase {lo} --to-phase {hi}")


if __name__ == "__main__":
    main()
