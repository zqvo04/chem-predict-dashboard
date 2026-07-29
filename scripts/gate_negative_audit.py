#!/usr/bin/env python
"""Audit the binder gate against molecules *measured* as non-binders.

STEP 10 validated the gate on **presumed** inactives — actives on other targets with
no JAK record. This asks the complementary question the evidence store made
answerable: how does it do on molecules a chemist actually put on a JAK assay and
found dead?

Those are the right-censored records (`>` with no pchembl). The strict set used here
keeps only molecules whose every panel record is `>` at **>= 10 uM**, because a `>`
bound below that is compatible with a potent compound and is not evidence of
non-binding — a distinction that changes the answer materially.

    python scripts/gate_negative_audit.py [panel]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import db as _db                              # noqa: E402
from src.funnel import _context, _gate                      # noqa: E402
from src.models.features import iter_morgan_batches         # noqa: E402
from src.panels import DEFAULT_PANEL, get_panel             # noqa: E402
from src.selectivity import POTENCY_FLOOR, SELECTIVE_GAP    # noqa: E402

MIN_BOUND_NM = 10_000       # ">= 10 uM" — the threshold at which '>' means "inactive"


def held_out_non_binders(con, panel) -> list[str]:
    """Molecules with no quantified value on this panel and every bound >= 10 uM."""
    return [r[0] for r in con.execute("""
        WITH t AS (
          SELECT inchikey,
                 count(*) FILTER (WHERE pchembl_value IS NOT NULL)  AS n_quant,
                 min(CASE WHEN standard_relation <> '=' AND standard_units = 'nM'
                          THEN standard_value END)                  AS min_bound
          FROM activity WHERE target_chembl_id IN ?
          GROUP BY 1)
        SELECT m.parent_smiles
        FROM t JOIN molecule m USING (inchikey)
        WHERE t.n_quant = 0 AND t.min_bound >= ?""",
        [list(panel.chembl_ids.values()), MIN_BOUND_NM]).fetchall()]


def main() -> None:
    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    if not _db.exists():
        raise SystemExit("No evidence store. Run: python -m src.data.db --restore")

    with _db.session(read_only=True) as con:
        smiles = held_out_non_binders(con, panel)
    if not smiles:
        raise SystemExit(f"No measured non-binders for panel {panel.name}.")

    gate = _gate(panel)
    proba = gate.predict_proba(smiles)
    scoreable = ~np.isnan(proba)
    passed = proba[scoreable] >= gate.threshold

    isoforms, models, _, _ = _context(panel, use_cache=True)
    preds: dict[str, list] = {i: [] for i in isoforms}
    for X, _mask in iter_morgan_batches(smiles):
        if X.shape[0]:
            for iso in isoforms:
                preds[iso].append(models[iso].predict(X))
    p = {iso: np.concatenate(preds[iso]) for iso in isoforms}
    floor = p[panel.target] >= POTENCY_FLOOR
    gap = p[panel.target] - np.maximum.reduce([p[o] for o in panel.offs])
    both = floor & (gap >= SELECTIVE_GAP)

    print(f"panel {panel.name}: {len(smiles)} molecules measured as non-binders "
          f"(every record '>' at >= {MIN_BOUND_NM/1000:.0f} uM, none in training)\n")
    print(f"  binder gate passes them   {passed.sum():5d} / {scoreable.sum():<5d} "
          f"= {passed.mean():6.1%}   (its own test set rejected "
          f"{gate.metrics.neg_reject_mean:.1%} of negatives)")
    print(f"  clear the potency floor   {floor.sum():5d} / {len(floor):<5d} "
          f"= {floor.mean():6.1%}   (mean predicted {panel.target} pchembl "
          f"{p[panel.target].mean():.2f})")
    print(f"  clear floor + selectivity {both.sum():5d} / {len(both):<5d} "
          f"= {both.mean():6.1%}")
    print("\nThese are adversarially hard by construction — molecules someone thought\n"
          "worth testing — not a random library draw. Read alongside the 18.9 % gate\n"
          "pass rate on the wide library (STEP 13).")


if __name__ == "__main__":
    main()
