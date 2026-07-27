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
echo "== Panel data layer (STEP 2 / 15) =="
echo "Builds/caches the per-isoform regression datasets + cross-measured join."
python -m src.data.panel_data jak

echo
echo "== Per-isoform regressors (STEP 3) =="
echo "5-seed scaffold-split MAE/RMSE/R2/Spearman per isoform (a few minutes)."
python -m src.models.isoform_regressor

echo
echo "== Selectivity gap validation (STEP 4) =="
echo "5-seed predicted-vs-measured gap Spearman + enrichment on the cross-measured set."
python -m src.selectivity
python scripts/make_hero_figure.py

echo
echo "== Conformal coverage (STEP 5) =="
echo "5-seed empirical coverage at 90% nominal, the 3-arm gap-interval comparison"
echo "(summed / flat / difficulty-scaled, STEP 14) + coverage-vs-nominal figure."
python -m src.conformal
python scripts/make_coverage_figure.py

echo
echo "== Applicability domain (STEP 6) =="
echo "In- vs out-of-domain error margin + money plot."
python -m src.applicability
python scripts/make_ad_figure.py

echo
echo "== Binder gate (STEP 10, Tier 0.5) =="
echo "Build the physchem-matched presumed-inactives (needs network to ten ChEMBL"
echo "targets; caches under data/) and the binder gate: ROC-AUC + Youden's-J point."
python -m src.data.negatives
python -m src.models.binder_gate

echo
echo "== Tiered wide screen (STEP 7) =="
echo "Screen the diverse library down the funnel to a selective + in-domain shortlist."
python -m src.data.library
python -m src.funnel

echo
echo "== Funnel loop, one worked case (STEP 8) =="
echo "B -> SELECT -> A (generate + re-score) -> before/after report + figure."
python scripts/run_loop.py

echo
echo "== Panel registry + leakage check (STEP 15) =="
echo "Lists the registered selectivity panels and checks each is disjoint from the"
echo "binder gate's negative basket and the wide library's targets."
python -m src.panels

echo
echo "== Gate 0 data audit (VALIDATION.md) =="
echo "Needs network to ChEMBL on first run; results cache under data/cache/."
python scripts/gate0_audit.py

echo
echo "== Assay-type + time-split audit (VALIDATION.md) =="
echo "Re-tests the headline gap claim on the ATP-independent Ki/Kd subset and on a"
echo "publication-year cut. Either can invalidate a headline number — that is the point."
python scripts/assay_time_audit.py

echo
echo "Done. Compare the Gate 0 tables above against VALIDATION.md."
