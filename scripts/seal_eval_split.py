#!/usr/bin/env python3
"""Seal the round-0 evaluation split (N11 / G9).

Writes once and refuses to overwrite. The training set grows every round; the
split it is measured against must not, or the rounds are not comparable to each
other and a regression can hide as a changed denominator.

The cut is temporal, not scaffold-based. STATE.md section 4c measured a Tanimoto
1-NN lookup at 0.782 against the model's 0.779 on a scaffold split and at 0.572
against 0.734 under this cut — a scaffold split scores every round on the protocol
where a lookup ties the model.

    python scripts/seal_eval_split.py [panel]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import panel_data                                     # noqa: E402
from src.panels import DEFAULT_PANEL, get_panel                     # noqa: E402


def main() -> None:
    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    path = panel.data_bundled / "eval_split.parquet"
    if path.exists():
        raise SystemExit(f"{path} already exists — the split is sealed. "
                         "Delete it deliberately if you mean to reseal.")

    year = panel_data.year_first(panel)
    molecules = pd.concat(
        [panel_data.build_isoform_dataset(panel, iso)["smi"] for iso in panel.isoforms]
    ).drop_duplicates()
    frame = pd.DataFrame({"smi": molecules})
    frame["year_first"] = frame["smi"].map(year)
    frame = frame.dropna(subset=["year_first"]).reset_index(drop=True)
    frame["fold"] = ["eval" if y > panel_data.EVAL_TIME_CUT else "train"
                     for y in frame["year_first"]]

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    counts = frame["fold"].value_counts()
    print(f"Sealed {len(frame)} molecules at cut {panel_data.EVAL_TIME_CUT}: "
          f"train {counts.get('train', 0)}, eval {counts.get('eval', 0)} -> {path}")


if __name__ == "__main__":
    main()
