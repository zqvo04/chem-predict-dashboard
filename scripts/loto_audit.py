#!/usr/bin/env python3
"""D7: how strong is paralog transfer, before anything fancier? (ROADMAP D7)

ROADMAP D7 keeps proteochemometrics out until there is a baseline to beat, on the
grounds that seven targets cannot support PCM anyway (STATE.md section 9). The
baseline it names is paralog transfer: train on a panel's *other* members and
predict the held-out one. This measures it.

Every held-out target is scored on the same molecules two ways:

  WITHIN   trained on that target's own scaffold-split train half — the normal model
  LOTO     trained on the panel's other members only, never on this target

and the LOTO column is split by whether the test molecule appears in the paralogs'
data at all. That split is the whole interpretation. A molecule measured on JAK1
and JAK2 sits in the training set with a correlated label, so transfer onto it is
partly recall of that molecule; transfer onto molecules the paralogs never saw is
the number that says whether the panel generalises across proteins.

Committed assets only, no network.

    python scripts/loto_audit.py            # ~10 minutes
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import panel_data                                     # noqa: E402
from src.models import isoform_regressor as ir                      # noqa: E402
from src.models.features import morgan_matrix                       # noqa: E402
from src.models.scaffold_split import scaffold_split                # noqa: E402
from src.panels import JAK, PI3K                                    # noqa: E402

MIN_TEST = 30       # below this a Spearman is not worth printing


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) < MIN_TEST:
        return float("nan"), float("nan")
    return (float(spearmanr(y_true, y_pred).statistic),
            float(mean_absolute_error(y_true, y_pred)))


def _fit_predict(train: pd.DataFrame, query_smiles: list[str]) -> np.ndarray:
    """Train on (smi, pchembl) and predict the query; NaN where a SMILES fails."""
    Xtr, mtr = morgan_matrix(train["smi"].tolist())
    model = ir._fit(Xtr, train["pchembl"].to_numpy()[mtr])
    Xq, mq = morgan_matrix(query_smiles)
    out = np.full(len(query_smiles), np.nan)
    out[mq] = model.predict(Xq)
    return out


def audit_panel(panel) -> None:
    frames = {iso: panel_data.build_isoform_dataset(panel, iso) for iso in panel.isoforms}
    print(f"\n{'=' * 76}\nPanel {panel.name} — "
          + ", ".join(f"{iso} n={len(frames[iso])}" for iso in panel.isoforms))
    print("=" * 76)
    print(f"  {'held out':9} {'test n':>7} {'WITHIN rho':>11} {'LOTO rho':>9} "
          f"{'LOTO seen':>10} {'LOTO unseen':>12} {'unseen n':>9}")

    for held in panel.isoforms:
        own = frames[held]
        smiles = own["smi"].tolist()
        tr, te = scaffold_split(smiles, test_frac=0.2, seed=0)
        test = own.iloc[te].reset_index(drop=True)

        within = _fit_predict(own.iloc[tr], test["smi"].tolist())
        paralogs = pd.concat([frames[o] for o in panel.isoforms if o != held],
                             ignore_index=True)
        loto = _fit_predict(paralogs, test["smi"].tolist())

        y = test["pchembl"].to_numpy()
        seen = test["smi"].isin(set(paralogs["smi"])).to_numpy()
        ok = ~np.isnan(loto)

        w_rho, _ = _score(y[ok], within[ok])
        l_rho, _ = _score(y[ok], loto[ok])
        s_rho, _ = _score(y[ok & seen], loto[ok & seen])
        u_rho, _ = _score(y[ok & ~seen], loto[ok & ~seen])
        print(f"  {held:9} {int(ok.sum()):>7} {w_rho:>11.3f} {l_rho:>9.3f} "
              f"{s_rho:>10.3f} {u_rho:>12.3f} {int((ok & ~seen).sum()):>9}")


def main() -> None:
    print("D7 — paralog transfer baseline (Spearman on the held-out target)")
    for panel in (JAK, PI3K):
        audit_panel(panel)
    print("\nHow to read this. WITHIN is the normal model on that target's own data.")
    print("LOTO never saw the target. 'seen' molecules are in the paralogs' data with")
    print("a correlated label, so transfer onto them is partly recall; 'unseen' is the")
    print("honest cross-protein number, and the gap between the two columns is how much")
    print("of the transfer is shared chemistry rather than shared pharmacology.")


if __name__ == "__main__":
    main()
