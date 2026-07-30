# Source this in every terminal before running the KDA-for-Trainium workflow:
#     source scripts/kda-env.sh
#
# It sets the environment the workflow needs:
#   * verify.py backend  -> whatever env your profiling/evaluation backend requires
#   * humanize / codex    -> provider/model overrides for your Codex backend

# --- profiling/evaluation backend (verify.py) ---
# verify.py is a placeholder in this release. If the backend you wire in needs any
# environment (endpoint URL, credentials, a named profile, etc.), export it here
# and/or in scripts/nki-auth.sh.

# --- humanize codex integration ---
# The RLCR loop (humanize) drives Codex for reviews. If your Codex backend needs
# a provider-prefixed model id and/or cannot accept a reasoning-effort override,
# set them here. Adjust DEFAULT_CODEX_MODEL to a model your backend exposes;
# HUMANIZE_NO_REASONING_EFFORT=1 skips the `-c model_reasoning_effort=` flag for
# backends whose auth rejects it (leave unset if yours accepts it).
export DEFAULT_CODEX_MODEL="${DEFAULT_CODEX_MODEL:-<your-codex-model>}"
export HUMANIZE_NO_REASONING_EFFORT="${HUMANIZE_NO_REASONING_EFFORT:-1}"

# codex/humanize scripts need a project root and a git repo (RLCR uses codex review).
export CLAUDE_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

echo "KDA env ready: codex=$DEFAULT_CODEX_MODEL  root=$CLAUDE_PROJECT_DIR"
