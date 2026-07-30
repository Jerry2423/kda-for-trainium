#!/usr/bin/env bash
# Create a per-task evidence workspace under workspaces/<op>/.
#
# The repo root holds SHARED tooling (verify.py, adapter/, prompts/, scripts/,
# baselines.json, .claude/skills). Each optimization task gets its own evidence
# directory so multiple tasks don't clobber each other's draft/plan/candidates.
#
# Usage:  bash scripts/new-task.sh <op>       e.g.  bash scripts/new-task.sh gqa_full
set -euo pipefail

OP="${1:-}"
if [[ -z "$OP" ]]; then
  echo "usage: bash scripts/new-task.sh <operator>" >&2
  exit 1
fi
# Reject anything but a bare operator name (no slashes / path chars) so a typo
# can't silently create a nested or misplaced workspace.
if [[ ! "$OP" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "error: operator name must match [A-Za-z0-9_]+, got: '$OP'" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
WS="$ROOT/workspaces/$OP"

if [[ -d "$WS" ]]; then
  echo "workspace already exists: $WS"
  exit 0
fi

mkdir -p "$WS"/{docs,runs,profile}
touch "$WS/docs/.gitkeep"
printf "timestamp,op,candidate,parent,passed,latency_ms,speedup,notes\n" > "$WS/benchmark.csv"
: > "$WS/candidates.jsonl"

echo "created $WS with {docs,runs,profile,benchmark.csv,candidates.jsonl}"
echo "next: paste prompts/$OP/phase1.md (or author it) and run the KDA loop from $WS"
