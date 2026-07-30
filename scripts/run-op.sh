#!/usr/bin/env bash
# Headless KDA driver: run the full 3-phase NKI optimization loop for ONE
# operator with no manual copy-paste and no manual gen-plan -> loop handoff.
#
#     bash scripts/run-op.sh <op> [--from-phase N] [--to-phase N] [--model M] [--dry-run]
#
# Each phase is three `claude -p` (headless) calls plus two git commits:
#
#   1. investigate + write docs/draft-phaseN.md          (agent)
#   2. /humanize:gen-plan  draft -> docs/plan-phaseN.md   (slash command)
#      -> git commit "draft + plan"                       (driver)
#   3. /humanize:start-rlcr-loop  implement + iterate      (slash command + Stop hook)
#      -> git commit "result"                             (driver)
#
# Why the commit between (2) and (3): the RLCR loop's setup script refuses to
# start on a dirty git tree (it needs a clean baseline for `codex review --base`),
# and gen-plan's --auto-start does NOT commit for you. So the driver owns that
# commit deterministically instead of relying on the agent to remember it.
#
# Why three separate `claude -p` calls: a slash command (/humanize:*) is expanded
# by the CLI from the top-level prompt; the agent cannot invoke one mid-turn. So
# each humanize command must be its own headless invocation.
#
# Headless facts this relies on (all verified): CLAUDE_PROJECT_DIR (set by
# kda-env.sh) drives humanize's state resolution independent of CWD; there is no
# AskUserQuestion tool in `-p` mode, so gen-plan's human-review gate cannot block;
# the Stop hook re-loops under `-p`, so start-rlcr-loop iterates headlessly.
#
# Resume: re-running skips any step whose output already exists (draft-phaseN.md,
# plan-phaseN.md) or whose done-marker exists (logs/phaseN.done). Safe to Ctrl-C
# and re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# ---- args -----------------------------------------------------------------
OP=""; FROM=1; TO=3; MODEL=""; DRY=0
usage() {
  echo "usage: bash scripts/run-op.sh <op> [--from-phase N] [--to-phase N] [--model M] [--dry-run]" >&2
  exit 1
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-phase) FROM="${2:?}"; shift 2 ;;
    --to-phase)   TO="${2:?}";   shift 2 ;;
    --model)      MODEL="${2:?}"; shift 2 ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    usage ;;
    -*)           echo "unknown flag: $1" >&2; usage ;;
    *)            [[ -z "$OP" ]] && OP="$1" || { echo "unexpected arg: $1" >&2; usage; }; shift ;;
  esac
done
[[ -n "$OP" ]] || usage
[[ "$OP" =~ ^[A-Za-z0-9_]+$ ]] || { echo "error: operator name must match [A-Za-z0-9_]+, got: '$OP'" >&2; exit 1; }
[[ "$FROM" =~ ^[1-3]$ && "$TO" =~ ^[1-3]$ && "$FROM" -le "$TO" ]] || { echo "error: phases must be 1..3 with from<=to" >&2; exit 1; }

# ---- environment (codex provider/model overrides + PROJECT_DIR; backend env) ----
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/kda-env.sh"
# Run the backend auth/setup step up front (placeholder by default; implement it
# for your own profiling backend if it needs credentials).
if [[ "$DRY" != 1 ]]; then bash "$REPO_ROOT/scripts/nki-auth.sh" || echo "WARN: nki-auth.sh failed; scoring may fail until you implement it"; fi

# ---- workspace + baseline preflight ---------------------------------------
WS="$REPO_ROOT/workspaces/$OP"
if [[ ! -d "$WS" ]]; then
  echo "workspace $WS missing; scaffolding it"
  bash "$REPO_ROOT/scripts/new-task.sh" "$OP"
fi
if ! grep -q "\"$OP\"" "$REPO_ROOT/baselines.json" 2>/dev/null; then
  echo "WARN: no cached baseline for '$OP' in baselines.json — speedup will print as '—'."
  echo "      cache it first:  python3 verify.py --build-baselines --op $OP"
fi
for P in $(seq "$FROM" "$TO"); do
  [[ -f "$REPO_ROOT/prompts/$OP/phase$P.md" ]] || { echo "error: missing prompts/$OP/phase$P.md (author it or run _gen_prompts.py)" >&2; exit 1; }
done

# ---- clean-tree preflight (mirrors the loop's own requirement) ------------
# Skipped under --dry-run so you can preview the planned steps on a dirty tree.
DIRTY="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all 2>/dev/null | grep -vE '^\?\? .*\.humanize[-/]' || true)"
if [[ "$DRY" != 1 && -n "$DIRTY" ]]; then
  echo "error: repo is not clean; commit or stash first so the loop has a clean baseline:" >&2
  echo "$DIRTY" >&2
  exit 1
fi

LOGS="$WS/logs"; mkdir -p "$LOGS"

CLAUDE_ARGS=(--permission-mode bypassPermissions --add-dir "$(dirname "$REPO_ROOT")")
[[ -n "$MODEL" ]] && CLAUDE_ARGS+=(--model "$MODEL")

# run_claude <prompt> <logfile>   (runs from the task workspace so relative
# evidence paths — docs/, runs/, ../../verify.py — resolve as the prompts expect)
run_claude() {
  local prompt="$1" log="$2"
  if [[ "$DRY" == 1 ]]; then
    echo "[dry-run] (cd $WS && claude -p <prompt> ${CLAUDE_ARGS[*]})  -> $log"
    return 0
  fi
  ( cd "$WS" && claude -p "$prompt" "${CLAUDE_ARGS[@]}" ) 2>&1 | tee "$log"
}

commit_all() { # <message>
  if [[ "$DRY" == 1 ]]; then echo "[dry-run] git add -A && git commit -m \"$1\""; return 0; fi
  git -C "$REPO_ROOT" add -A
  git -C "$REPO_ROOT" commit -q -m "$1" || echo "(nothing to commit)"
}

# ---- phase loop -----------------------------------------------------------
for ((P=FROM; P<=TO; P++)); do
  echo "==================== $OP — phase $P ===================="
  DRAFT="docs/draft-phase$P.md"
  PLAN="docs/plan-phase$P.md"
  PROMPT_FILE="$REPO_ROOT/prompts/$OP/phase$P.md"

  # Step 1 — draft. Append an automation directive so the agent stops after the
  # draft instead of trying to run the humanize commands itself (it can't in -p).
  if [[ -f "$WS/$DRAFT" ]]; then
    echo "[phase $P] skip draft (exists: $DRAFT)"
  else
    echo "[phase $P] step 1/3: investigate + write $DRAFT"
    run_claude "$(cat "$PROMPT_FILE")

--- AUTOMATION DIRECTIVE (driver-injected) ---
You are running headless under scripts/run-op.sh. Do the investigation this
phase asks for, then write your implementation-plan draft to $DRAFT and STOP.
Do NOT run /humanize:gen-plan or /humanize:start-rlcr-loop yourself — the driver
runs those in the next steps. The final action of this turn is writing $DRAFT." \
      "$LOGS/phase$P.1-draft.log"
    [[ "$DRY" == 1 || -f "$WS/$DRAFT" ]] || { echo "error: agent did not write $DRAFT (see $LOGS/phase$P.1-draft.log)" >&2; exit 1; }
  fi

  # Step 2 — gen-plan (no --auto-start; the driver owns the commit + loop start).
  if [[ -f "$WS/$PLAN" ]]; then
    echo "[phase $P] skip gen-plan (exists: $PLAN)"
  else
    echo "[phase $P] step 2/3: /humanize:gen-plan -> $PLAN"
    run_claude "/humanize:gen-plan --input $DRAFT --output $PLAN" "$LOGS/phase$P.2-genplan.log"
    [[ "$DRY" == 1 || -f "$WS/$PLAN" ]] || { echo "error: gen-plan did not write $PLAN (see $LOGS/phase$P.2-genplan.log)" >&2; exit 1; }
  fi

  # Commit draft + plan so the loop starts on a clean tree.
  commit_all "$OP phase $P: draft + plan"

  # Step 3 — RLCR loop (Stop hook drives iteration; default max iterations).
  MARK="$LOGS/phase$P.done"
  if [[ -f "$MARK" ]]; then
    echo "[phase $P] skip loop (done marker present: $MARK)"
  else
    echo "[phase $P] step 3/3: /humanize:start-rlcr-loop $PLAN --skip-quiz"
    run_claude "/humanize:start-rlcr-loop $PLAN --skip-quiz" "$LOGS/phase$P.3-loop.log"
    [[ "$DRY" == 1 ]] || touch "$MARK"
    commit_all "$OP phase $P: result (candidates + evidence)"
  fi
  echo "[phase $P] done"
done

echo "==================== $OP — all requested phases complete ===================="
echo "evidence: $WS/{benchmark.csv,candidates.jsonl,runs/,docs/}   logs: $LOGS/"
