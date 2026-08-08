#!/usr/bin/env python3
"""N14: what `early_stopping='auto'` actually does, per isoform (STATE.md section 4).

`_fit` never passes `early_stopping`, so scikit-learn's 'auto' applies: on for
n_samples > 10_000, off below. STATE.md section 4 recorded that the JAK isoform
sizes straddle that threshold and called the consequence "regularisation may
differ per isoform". This audit measures it instead of assuming, and reports two
things a single size comparison misses:

  1. Whether early stopping ever *fired*. When on, 10 % of the training rows are
     held out as a validation set and never trained on. If `n_iter_` reaches
     `max_iter` the mechanism never stopped anything, so that 10 % bought nothing.

  2. Whether the deployed model and the models behind the VALIDATION.md metrics
     land on the same side of the threshold. `evaluate` fits on a scaffold-split
     train fold (~80 %), the deployed model on 100 %, so an isoform sitting just
     above 10_000 is configured one way when measured and the other way when
     shipped.

Reads the committed models and datasets; no training, no network.

    python scripts/early_stopping_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import panel_data                                    # noqa: E402
from src.models.features import morgan_matrix                      # noqa: E402
from src.models.isoform_regressor import _load                     # noqa: E402
from src.models.scaffold_split import scaffold_split               # noqa: E402
from src.panels import DEFAULT_PANEL, get_panel                    # noqa: E402

AUTO_THRESHOLD = 10_000        # scikit-learn's n_samples cut for early_stopping='auto'


def main() -> None:
    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    print(f"Panel {panel.name} — early_stopping='auto' fires above "
          f"n_samples > {AUTO_THRESHOLD:,}\n")
    print(f"{'isoform':8} {'deployed n':>11} {'eval train n':>13} "
          f"{'on?':>5} {'n_iter_':>8} {'max_iter':>9} {'fired?':>7}")

    for iso in panel.isoforms:
        data = panel_data.build_isoform_dataset(panel, iso)
        smiles = data["smi"].tolist()
        _, mask = morgan_matrix(smiles)
        kept = [s for s, keep in zip(smiles, mask) if keep]
        train_idx, _ = scaffold_split(kept, test_frac=0.2, seed=0)

        model = _load(panel.model_bundled / f"{iso}_reg.pkl").model
        on = model.do_early_stopping_
        fired = model.n_iter_ < model.max_iter
        print(f"{iso:8} {len(kept):11,} {len(train_idx):13,} "
              f"{str(on):>5} {model.n_iter_:8} {model.max_iter:9} {str(fired):>7}")

    print(f"\nWhen on, {panel.isoforms[0]}-style models hold out "
          "validation_fraction=0.1 of the training rows and never fit on them.\n"
          "An isoform whose deployed n and eval-train n straddle the threshold is\n"
          "configured differently when measured than when shipped.")


if __name__ == "__main__":
    main()
