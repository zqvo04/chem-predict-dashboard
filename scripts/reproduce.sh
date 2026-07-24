#!/usr/bin/env bash
# Reproduce every measured number the repo currently claims.
#
#   ./scripts/reproduce.sh
#
# As the funnel is built (STEP 3+), each step appends its metric-regenerating
# command here, so one script always rebuilds the whole VALIDATION.md story from
# a clean checkout. Today it covers: the test suite + the Gate 0 data audit.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Test suite =="
python -m pytest tests/ -q

echo
echo "== Gate 0 data audit (VALIDATION.md) =="
echo "Needs network to ChEMBL on first run; results cache under data/cache/."
python scripts/gate0_audit.py

echo
echo "Done. Compare the Gate 0 tables above against VALIDATION.md."
