# One-shot terminal setup for the KDA-for-Trainium workflow. SOURCE it (do not
# run) once per new terminal, from anywhere:
#
#     source /path/to/kda-trainium/scripts/setup.sh
#     # or, from the repo root:  source scripts/setup.sh
#
# It (1) sets the backend + codex environment (via kda-env.sh, which must be
# sourced), and (2) runs the backend auth/setup step (via nki-auth.sh — a
# placeholder by default — as a subprocess so its `set -e` can't exit your shell).
#
# Guard: warn if executed instead of sourced (env would vanish on exit).
# Detect "sourced" in both bash and zsh.
_kda_sourced=0
if [ -n "${ZSH_VERSION:-}" ]; then
  case "${ZSH_EVAL_CONTEXT:-}" in *:file*) _kda_sourced=1 ;; esac
elif [ -n "${BASH_SOURCE:-}" ]; then
  [ "${BASH_SOURCE[0]}" != "${0}" ] && _kda_sourced=1
fi
if [ "$_kda_sourced" != "1" ]; then
  echo "error: source this script, don't run it:  source scripts/setup.sh" >&2
  exit 1
fi
unset _kda_sourced

# Resolve this script's directory whether sourced from bash or zsh, any CWD.
_KDA_SETUP_SRC="${BASH_SOURCE[0]:-${(%):-%x}}"
_KDA_SCRIPTS_DIR="$(cd "$(dirname "$_KDA_SETUP_SRC")" && pwd)"

# 1. Environment (must be sourced so exports land in the current shell).
source "$_KDA_SCRIPTS_DIR/kda-env.sh"

# 2. Backend auth/setup step (subprocess: isolates its `set -euo pipefail`).
bash "$_KDA_SCRIPTS_DIR/nki-auth.sh" || true

unset _KDA_SETUP_SRC _KDA_SCRIPTS_DIR
