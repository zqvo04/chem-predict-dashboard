#!/usr/bin/env python3
"""Seal the measured-negative train/eval split.

Writes once and refuses to overwrite. A held-out set that moves with the evidence
store is not held out, and the whole point of this population is to answer
"does the retrained gate reject non-binders whose chemotype it has not seen".

The split is scaffold-based, not temporal: these molecules carry no pchembl, so
most of them are absent from the sealed time split, and the question is about
chemotype rather than publication date.

    python scripts/seal_measured_negatives.py [panel]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import panel_data                                     # noqa: E402
from src.models.scaffold_split import scaffold_split                # noqa: E402
from src.panels import DEFAULT_PANEL, get_panel                     # noqa: E402

TEST_FRAC = 0.25
SEED = 0


def main() -> None:
    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    path = panel.data_bundled / "measured_negatives.parquet"
    if path.exists():
        raise SystemExit(f"{path} already exists — the split is sealed. "
                         "Delete it deliberately if you mean to reseal.")

    population = panel_data.measured_negatives(panel)
    _, eval_idx = scaffold_split(population["smi"].tolist(),
                                 test_frac=TEST_FRAC, seed=SEED)
    population["fold"] = "train"
    population.loc[list(eval_idx), "fold"] = "eval"

    path.parent.mkdir(parents=True, exist_ok=True)
    population.to_parquet(path, index=False)
    counts = population["fold"].value_counts()
    print(f"Sealed {len(population)} measured negatives: "
          f"train {counts.get('train', 0)}, eval {counts.get('eval', 0)} -> {path}")


if __name__ == "__main__":
    main()
