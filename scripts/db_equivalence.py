#!/usr/bin/env python
"""D0 gate: how far does the evidence store's view differ from the committed parquet?

The committed `assets/<panel>/*.parquet` are what the validated models were trained
on. Rebuilding the same datasets from the store will *not* reproduce them, and the
point of this script is to say by how much and **why**, rather than to discover it
later as a mystery.

Two sources of difference, deliberately reported apart:

  release drift    the committed files were built from an unrecorded ChEMBL release;
                   the store records its own. New molecules and revised measurements
                   land as additions and value changes.
  standardisation  `panel_data._canonical` notes the committed files predate
                   `standardize()` and are ~0.5 % un-standardised.

Neither is a bug. Both would silently move every training set — and therefore every
`model_id`, invalidating every exported contract — if a migration swapped the data
source without measuring first.

    python scripts/db_equivalence.py [panel]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import db as _db                              # noqa: E402
from src.panels import DEFAULT_PANEL, get_panel             # noqa: E402
from src.standardize import standardize                     # noqa: E402


def compare(con, panel, isoform: str) -> dict:
    """Committed parquet vs the store's view, for one panel member."""
    committed_path = panel.data_bundled / f"{isoform}.parquet"
    if not committed_path.exists():
        return {"isoform": isoform, "status": "no committed parquet"}
    committed = pd.read_parquet(committed_path)

    view = con.execute(
        "SELECT smi, pchembl, n_meas FROM v_pchembl_median WHERE target_chembl_id = ?",
        [panel.chembl_ids[isoform]]).df()
    if view.empty:
        return {"isoform": isoform, "status": "not ingested"}

    # The store standardises on ingest; the committed file mostly does not. Compare on
    # the standardised form of both so the release difference is not swamped by the
    # standardisation difference, then report the latter separately.
    committed = committed.copy()
    committed["std"] = committed["smi"].map(standardize)
    unstandardised = int((committed["std"] != committed["smi"]).sum())
    committed = committed.dropna(subset=["std"])

    left = set(committed["std"])
    right = set(view["smi"])
    both = left & right

    merged = (committed[committed["std"].isin(both)][["std", "pchembl"]]
              .drop_duplicates("std")
              .merge(view[view["smi"].isin(both)][["smi", "pchembl"]]
                     .drop_duplicates("smi"),
                     left_on="std", right_on="smi", suffixes=("_committed", "_store")))
    delta = (merged["pchembl_store"] - merged["pchembl_committed"]).abs()

    return {
        "isoform": isoform,
        "committed_rows": len(committed),
        "store_rows": len(view),
        "shared": len(both),
        "only_committed": len(left - right),
        "only_store": len(right - left),
        "shared_pct": round(100 * len(both) / max(len(left), 1), 1),
        "unstandardised_in_committed": unstandardised,
        "unstandardised_pct": round(100 * unstandardised / max(len(committed), 1), 2),
        "pchembl_identical": int((delta < 1e-9).sum()),
        "pchembl_changed": int((delta >= 1e-9).sum()),
        "pchembl_max_abs_delta": round(float(delta.max()), 3) if len(delta) else 0.0,
        "pchembl_mean_abs_delta": round(float(delta.mean()), 4) if len(delta) else 0.0,
    }


def main() -> None:
    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    with _db.session(read_only=True) as con:
        release = con.execute("SELECT release FROM source LIMIT 1").fetchone()
        print(f"panel {panel.name}   store release: {release[0] if release else 'none'}")
        print("committed assets: release UNRECORDED (this is the hole STEP 16 closes)\n")
        rows = [compare(con, panel, iso) for iso in panel.isoforms]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if "shared_pct" in df:
        print(f"\nOverlap {df['shared_pct'].min():.1f}-{df['shared_pct'].max():.1f} % of "
              f"committed molecules; {df['pchembl_changed'].sum()} shared molecules "
              f"changed pchembl.")
        print("Gate: the committed parquet stays the training data until a retrain is "
              "run as its own announced, contract-invalidating step.")


if __name__ == "__main__":
    main()
