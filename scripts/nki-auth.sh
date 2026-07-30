#!/usr/bin/env bash
# Backend auth/setup step for the profiling/evaluation backend verify.py uses.
#
# PLACEHOLDER — fill in for your own backend.
# scripts/setup.sh and scripts/run-op.sh call this before scoring. Implement
# whatever your backend needs to become usable (log in, fetch/export a token,
# set an endpoint, etc.). The rest of the harness only assumes that after this
# runs successfully, verify.py can reach and use your backend.
set -euo pipefail

echo "nki-auth.sh is a placeholder — implement backend auth/setup for your profiler." >&2
echo "See README.md (Setup) for the environment this harness expects." >&2
exit 1
