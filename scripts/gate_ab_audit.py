#!/usr/bin/env python3
"""B: does a measured negative class change what the binder gate decides?

Two gates, one difference. Both get the same positives, the same presumed
negatives, the same fingerprints, the same estimator and the same threshold rule.
The candidate additionally gets the train fold of the measured negatives.

  BASELINE   presumed negatives only            — what ships today
  CANDIDATE  presumed + measured[train]         — the negative class redefined

Both are scored on populations neither trained on:

  measured[eval]   held-out measured non-binders  — the thing being fixed
  positives[eval]  held-out actives               — the cost of fixing it
  sealed 414       the falsification audit's arm  — untouched by either gate (G6)

The middle one is not optional. A gate that rejects every measured negative by
rejecting everything is not an improvement, and only the positive arm shows it.
STATE.md sections 2a and 11 are the same lesson twice.

Nothing here writes a model. Whether to deploy is a separate decision, because it
changes a committed asset and every model_id under it (G0).

    python scripts/gate_ab_audit.py            # ~5 minutes
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import panel_data                                     # noqa: E402
from src.data.negatives import build_negatives, positive_smiles     # noqa: E402
from src.models.binder_gate import _fit                             # noqa: E402
from src.models.features import morgan_matrix                       # noqa: E402
from src.panels import DEFAULT_PANEL                                # noqa: E402

SWEEP = (0.5, 0.7, 0.9, 0.95, 0.99)
PRESUMED_HOLDOUT = 0.2


def build_gate(positives: list[str], negatives: list[str]):
    """Fit a gate from explicit class members; nothing is cached."""
    smiles = list(positives) + list(negatives)
    y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    X, mask = morgan_matrix(smiles)
    return _fit(X, y[mask])


def probabilities(model, smiles: list[str]) -> np.ndarray:
    """P(binder) for each parseable SMILES."""
    X, mask = morgan_matrix(list(smiles))
    out = np.full(len(mask), np.nan)
    if X.shape[0]:
        out[mask] = model.predict_proba(X)[:, 1]
    return out[~np.isnan(out)]


def youden_threshold(model, positives: list[str], negatives: list[str]) -> float:
    """Youden's J on held-out positives vs held-out presumed negatives.

    Deliberately not computed on the measured negatives: a threshold tuned on the
    population it is then judged against is not a measurement.
    """
    p_pos, p_neg = probabilities(model, positives), probabilities(model, negatives)
    y = np.concatenate([np.ones(len(p_pos)), np.zeros(len(p_neg))])
    fpr, tpr, thresholds = roc_curve(y, np.concatenate([p_pos, p_neg]))
    return float(thresholds[np.argmax(tpr - fpr)])


def at_matched_recall(p_meas: np.ndarray, p_pos: np.ndarray,
                      recall: float) -> tuple[float, float]:
    """(threshold, measured-negative pass rate) where the gate keeps `recall` of actives.

    Two gates compared at their own Youden points can differ only in where they sit
    on the same curve. Matching the positive recall first removes that.
    """
    threshold = float(np.quantile(p_pos, 1.0 - recall))
    return threshold, float((p_meas >= threshold).mean())


def main() -> None:
    panel = DEFAULT_PANEL

    split = panel_data.load_eval_split(panel)
    train_smiles = set(split.loc[split["fold"] == "train", "smi"])
    eval_smiles = set(split.loc[split["fold"] == "eval", "smi"])

    all_positives = positive_smiles(panel)
    pos_train = sorted(all_positives & train_smiles)
    pos_eval = sorted(all_positives & eval_smiles)

    presumed = build_negatives(panel)["smi"].tolist()
    cut = int(len(presumed) * PRESUMED_HOLDOUT)
    holdout_presumed, train_presumed = presumed[:cut], presumed[cut:]

    measured = panel_data.load_measured_negatives(panel)
    meas_train = measured.loc[measured["fold"] == "train", "smi"].tolist()
    meas_eval = measured.loc[measured["fold"] == "eval", "smi"].tolist()
    sealed = sorted(set(panel_data.censored_library_molecules(panel)["smi"]))

    print(f"positives  train {len(pos_train)}  held-out {len(pos_eval)}")
    print(f"presumed   train {len(train_presumed)}  held-out {len(holdout_presumed)}")
    print(f"measured   train {len(meas_train)}  held-out {len(meas_eval)}")
    print(f"sealed 414 {len(sealed)} (in neither gate)\n")

    arms = {
        "BASELINE  (presumed only)": train_presumed,
        "CANDIDATE (presumed + measured[train])":
            sorted(set(train_presumed) | set(meas_train)),
    }

    curves = {}
    for name, negatives in arms.items():
        model = build_gate(pos_train, negatives)
        threshold = youden_threshold(model, pos_eval, holdout_presumed)
        groups = {
            "measured[eval]  (want LOW)": meas_eval,
            "sealed 414      (want LOW)": sealed,
            "presumed[eval]  (want LOW)": holdout_presumed,
            "positives[eval] (want HIGH)": pos_eval,
        }
        print("=" * 72)
        print(f"{name}   negatives n={len(negatives)}   threshold {threshold:.3f}")
        print("=" * 72)
        print(f"  {'group':30} {'n':>6} {'pass%':>7} {'median P':>10}")
        for label, smis in groups.items():
            p = probabilities(model, smis)
            print(f"  {label:30} {len(p):6d} {100 * (p >= threshold).mean():6.1f}% "
                  f"{np.median(p):10.3f}")

        p_meas, p_pos = probabilities(model, meas_eval), probabilities(model, pos_eval)
        p_sealed = probabilities(model, sealed)
        curves[name] = (p_meas, p_pos, p_sealed)
        print(f"\n  {'threshold':>10} {'measured[eval] pass':>21} {'positives kept':>16}")
        for t in SWEEP:
            print(f"  {t:10.2f} {100 * (p_meas >= t).mean():20.1f}% "
                  f"{100 * (p_pos >= t).mean():15.1f}%")
        print()

    print("=" * 72)
    print("HEAD TO HEAD at matched positive recall")
    print("=" * 72)
    print(f"  {'recall kept':>12} " + "".join(f"{n.split()[0]:>14}" for n in curves)
          + f"{'delta':>10}")
    for recall in (0.99, 0.975, 0.95, 0.90):
        rates = {}
        for name, (p_meas, p_pos, _) in curves.items():
            _, rate = at_matched_recall(p_meas, p_pos, recall)
            rates[name] = rate
        values = list(rates.values())
        print(f"  {recall:11.1%} " + "".join(f"{100 * v:13.1f}%" for v in values)
              + f"{100 * (values[1] - values[0]):+9.1f}")

    print("\n  Same, on the sealed 414 (neither gate saw them):")
    print(f"  {'recall kept':>12} " + "".join(f"{n.split()[0]:>14}" for n in curves)
          + f"{'delta':>10}")
    for recall in (0.99, 0.975, 0.95, 0.90):
        values = []
        for _, (_, p_pos, p_sealed) in curves.items():
            _, rate = at_matched_recall(p_sealed, p_pos, recall)
            values.append(rate)
        print(f"  {recall:11.1%} " + "".join(f"{100 * v:13.1f}%" for v in values)
              + f"{100 * (values[1] - values[0]):+9.1f}")

    print("\nHow to read this. Negative delta means the candidate rejects more real")
    print("non-binders at the same cost in actives. Read the matched-recall table,")
    print("not each gate's own Youden point — two gates at different operating points")
    print("on the same curve is not a difference in what they learned.")


if __name__ == "__main__":
    main()
