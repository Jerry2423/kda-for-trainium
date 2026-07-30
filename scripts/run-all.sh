#!/usr/bin/env bash
# Serial batch driver: run scripts/run-op.sh for many operators, one at a time.
#
#     bash scripts/run-all.sh                 # all 14 ops, fast-baseline first
#     bash scripts/run-all.sh silu bmm matmul # just these, in the given order
#     bash scripts/run-all.sh --from-phase 1 --to-phase 1   # phase subset, all ops
#     bash scripts/run-all.sh --dry-run
#
# Why serial (not `run-op.sh a & run-op.sh b &`): run-op.sh commits with a
# repo-wide `git add -A`, and the RLCR loop requires a clean tree — two ops in
# parallel would cross-contaminate commits, collide on .git/index.lock, and one
# would block the other's loop from starting. So we run strictly one op at a time.
#
# Resilient: one op failing does NOT abort the batch — it's recorded and we move
# on. Resumable: an op whose logs/phaseN.done markers already cover the requested
# phases is skipped (delegated to run-op.sh's own per-step resume). A batch
# summary is printed at the end and written to logs/run-all-summary.txt.
set -uo pipefail   # NOT -e: we handle per-op failure ourselves so the batch continues

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# Default op order: fast baseline first (early results, cheap ops surface bugs
# before the expensive ones). Overridable by passing op names as positional args.
DEFAULT_ORDER=(rmsnorm_matmul silu rope_single_freq_apply mamba adamw
               add_rmsnorm_matmul swiglu bmm matmul_add_rmsnorm transpose_matmul
               bmm_softmax matmul lora gqa_full)

# ---- parse args: collect op names (positional) + pass-through flags -----------
OPS=(); PASSTHRU=(); DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-phase|--to-phase|--model) PASSTHRU+=("$1" "${2:?}"); shift 2 ;;
    --dry-run) DRY=1; PASSTHRU+=("$1"); shift ;;
    -h|--help)
      echo "usage: bash scripts/run-all.sh [op ...] [--from-phase N] [--to-phase N] [--model M] [--dry-run]" >&2
      echo "  no op args => all 14, fast-baseline first" >&2
      exit 0 ;;
    -*) echo "unknown flag: $1" >&2; exit 1 ;;
    *)  OPS+=("$1"); shift ;;
  esac
done
[[ ${#OPS[@]} -eq 0 ]] && OPS=("${DEFAULT_ORDER[@]}")

# ---- environment: source once so every run-op.sh child inherits it ------------
# (run-op.sh also sources kda-env.sh itself, but doing it here refreshes creds a
# single time for the whole batch instead of per-op.)
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/kda-env.sh"
if [[ "$DRY" != 1 ]]; then
  bash "$REPO_ROOT/scripts/nki-auth.sh" || echo "WARN: nki-auth failed; profiler scoring may fail"
fi

LOGS="$REPO_ROOT/logs"; mkdir -p "$LOGS"
SUMMARY="$LOGS/run-all-summary.txt"
: > "$SUMMARY"

echo "batch: ${#OPS[@]} op(s) -> ${OPS[*]}"
echo "flags to run-op.sh: ${PASSTHRU[*]:-(none)}"
echo ""

declare -a OK_OPS=() FAIL_OPS=()
i=0
for op in "${OPS[@]}"; do
  i=$((i+1))
  echo "############################################################"
  echo "# [$i/${#OPS[@]}] $op"
  echo "############################################################"
  # Each op is fully self-contained in run-op.sh (env, creds check, workspace
  # scaffold, clean-tree gate, phases, commits). We only branch on its exit code.
  if bash "$REPO_ROOT/scripts/run-op.sh" "$op" "${PASSTHRU[@]}"; then
    OK_OPS+=("$op");  echo "[$op] OK"    | tee -a "$SUMMARY"
  else
    rc=$?
    FAIL_OPS+=("$op"); echo "[$op] FAILED (exit $rc) — see workspaces/$op/logs/" | tee -a "$SUMMARY"
    # A failed op can leave the tree dirty (partial draft/plan not yet committed),
    # which would make the NEXT op's clean-tree gate fail. Surface that clearly.
    if [[ "$DRY" != 1 && -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all | grep -vE '^\?\? .*\.humanize[-/]')" ]]; then
      echo "[$op] WARNING: repo left dirty by the failure; committing WIP so the batch can continue" | tee -a "$SUMMARY"
      git -C "$REPO_ROOT" add -A
      git -C "$REPO_ROOT" commit -q -m "$op: WIP (batch salvage after failure)" || true
    fi
  fi
  echo ""
done

# ---- summary ------------------------------------------------------------------
{
  echo ""
  echo "==================== batch summary ===================="
  echo "OK   (${#OK_OPS[@]}): ${OK_OPS[*]:-none}"
  echo "FAIL (${#FAIL_OPS[@]}): ${FAIL_OPS[*]:-none}"
  echo "evidence per op: workspaces/<op>/{benchmark.csv,candidates.jsonl,runs/,logs/}"
} | tee -a "$SUMMARY"

# Nonzero exit iff any op failed, so callers/CI can detect it.
[[ ${#FAIL_OPS[@]} -eq 0 ]]
