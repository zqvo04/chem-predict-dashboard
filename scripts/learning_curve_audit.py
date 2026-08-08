#!/usr/bin/env python3
"""N7: does more training data still move the models? (ROADMAP section 4)

The loop's retraining channel needs five links: a label arrives, it is stored, the
retrain reads it, **the model changes materially**, the selection changes. Links 1-3
are designed and link 4 has never been measured. There is reason to doubt it — a
round adds roughly 70 molecules to a training set in the thousands, a few percent.

Two axes, measured separately, because they are expected to differ. The regressors
already have ~10k molecules of published SAR. The binder gate's negatives are all
"presumed inactive by absence", and measured negatives would be new information.

Method: fix a scaffold split, then refit on nested prefixes of the training half and
score every model on the same held-out test half. Nested (not independent draws) so
the curve moves with data volume rather than with resampling noise, and the test
half is identical at every point so the y-axis is comparable across the curve.

The subsets are random within the training half. A real round is not random — it is
whatever the acquisition function picked — so this measures the value of *volume*,
which is the link-4 question, not the value of a well-chosen batch.

    python scripts/learning_curve_audit.py            # ~10 minutes
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import panel_data                                    # noqa: E402
from src.models import binder_gate as bg                           # noqa: E402
from src.models import isoform_regressor as ir                     # noqa: E402
from src.models.features import morgan_matrix                      # noqa: E402
from src.models.scaffold_split import scaffold_split               # noqa: E402
from src.panels import DEFAULT_PANEL                               # noqa: E402

FRACTIONS = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
SEEDS = (0, 1, 2)
BATCH = 70          # molecules a DMTA round is expected to add (ROADMAP section 4)


def _curve(X, y, kept, fit, score):
    """Mean metric at each training fraction, on a per-seed fixed test half."""
    rows = []
    for seed in SEEDS:
        train_idx, test_idx = scaffold_split(kept, test_frac=0.2, seed=seed)
        order = np.random.default_rng(seed).permutation(len(train_idx))
        per_frac = []
        for frac in FRACTIONS:
            sub = train_idx[order[: max(2, int(len(train_idx) * frac))]]
            per_frac.append(score(fit(X[sub], y[sub]), X[test_idx], y[test_idx]))
        rows.append((len(train_idx), per_frac))
    n_train = rows[0][0]
    return n_train, np.array([r[1] for r in rows])


def _report(label: str, n_train: int, values: np.ndarray, metric: str) -> None:
    mean, std = values.mean(axis=0), values.std(axis=0)
    print(f"\n{label}   (train half n={n_train}, {len(SEEDS)} seeds, fixed test half)")
    print(f"  {'train n':>9} {'frac':>6} {metric:>18}")
    for frac, m, s in zip(FRACTIONS, mean, std):
        print(f"  {int(n_train * frac):>9} {frac:>6.0%} {m:>11.4f} ± {s:.4f}")

    # Tail slope, expressed in the unit the loop actually adds: one round's batch.
    d_n = n_train * (FRACTIONS[-1] - FRACTIONS[-2])
    per_batch = (mean[-1] - mean[-2]) / d_n * BATCH
    print(f"  tail slope: {mean[-1] - mean[-2]:+.4f} over the last {int(d_n)} molecules"
          f"  ->  {per_batch:+.5f} per {BATCH}-molecule round")
    print(f"  seed-to-seed std at full data: ±{std[-1]:.4f}"
          f"   ({'BELOW' if abs(per_batch) < std[-1] else 'ABOVE'} the noise floor)")


def regression_axis() -> None:
    iso = DEFAULT_PANEL.target
    data = panel_data.build_isoform_dataset(DEFAULT_PANEL, iso)
    smiles = data["smi"].tolist()
    X, mask = morgan_matrix(smiles)
    y = data["pchembl"].to_numpy()[mask]
    kept = [s for s, k in zip(smiles, mask) if k]

    for metric, score in (
        ("Spearman", lambda m, Xt, yt: float(spearmanr(yt, m.predict(Xt)).statistic)),
        ("MAE", lambda m, Xt, yt: float(mean_absolute_error(yt, m.predict(Xt)))),
    ):
        n_train, values = _curve(X, y, kept, ir._fit, score)
        _report(f"REGRESSION — {iso} pchembl", n_train, values, metric)


def gate_axis() -> None:
    smiles, y = bg._dataset(DEFAULT_PANEL)
    X, mask = morgan_matrix(smiles)
    y = y[mask]
    kept = [s for s, k in zip(smiles, mask) if k]
    n_train, values = _curve(
        X, y, kept, bg._fit,
        lambda m, Xt, yt: float(roc_auc_score(yt, m.predict_proba(Xt)[:, 1])))
    _report("BINDER GATE — presumed-inactive classification", n_train, values, "ROC-AUC")


def main() -> None:
    print("=" * 70)
    print("N7 — learning curves, per axis")
    print("=" * 70)
    regression_axis()
    gate_axis()
    print("\nHow to read this. A tail slope below the seed-to-seed noise floor means")
    print("another round's worth of molecules cannot be distinguished from a reseed,")
    print("so retraining is not that axis' learning channel — acquisition is.")


if __name__ == "__main__":
    main()
