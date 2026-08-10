#!/usr/bin/env python3
"""N1: does the model beat a Tanimoto nearest-neighbour lookup? (ROADMAP section 5)

STATE.md section 4 describes the models' judgement principle as "do this molecule's
substructure fragments resemble the fragments of molecules that were active in the
training set". A 1-nearest-neighbour lookup over the same fingerprints answers
exactly that question with no model at all — so it is the baseline the headline
claims have to clear, and until now nobody had run it.

The target is the **gap axis**. The potency axis is already 93 % falsified against
measured non-binders (STATE.md section 2), so its baseline comparison changes
nothing. The gap Spearman of ~0.80 is the one live claim in the repo, and if a
similarity lookup reproduces it, the claim shrinks to "similarity search".

Baseline and model see the identical split, the identical fingerprints and the
identical metric — the neighbour search reuses `applicability`'s own generator, so
this is not a re-implementation of the representation.

    python scripts/nn_baseline_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdkit import DataStructs
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.applicability import _bitvects                             # noqa: E402
from src.data import panel_data                                     # noqa: E402
from src.data.panel_data import EVAL_TIME_CUT as TIME_CUT           # noqa: E402
from src.data.panel_data import year_first as _year_first           # noqa: E402
from src.models import isoform_regressor as ir                      # noqa: E402
from src.models.features import morgan_matrix                       # noqa: E402
from src.models.scaffold_split import scaffold_split                # noqa: E402
from src.panels import DEFAULT_PANEL                                # noqa: E402
from src.selectivity import _enrichment, evaluate_split             # noqa: E402

SEEDS = (0, 1, 2)


def nn_predict(query: list[str], train: list[str], train_y: np.ndarray) -> np.ndarray:
    """Label of each query's most Tanimoto-similar training molecule; NaN if unparseable."""
    train_fps = _bitvects(train)
    usable = [i for i, fp in enumerate(train_fps) if fp is not None]
    fps = [train_fps[i] for i in usable]
    labels = train_y[usable]

    out = np.full(len(query), np.nan)
    for i, fp in enumerate(_bitvects(query)):
        if fp is not None:
            out[i] = labels[int(np.argmax(DataStructs.BulkTanimotoSimilarity(fp, fps)))]
    return out


def _gap_data():
    cross = panel_data.build_cross_measured(DEFAULT_PANEL)
    smiles = cross["smi"].tolist()
    X, mask = morgan_matrix(smiles)
    cross = cross[mask].reset_index(drop=True)
    kept = [s for s, k in zip(smiles, mask) if k]
    gap = (cross[DEFAULT_PANEL.target].to_numpy()
           - cross[list(DEFAULT_PANEL.offs)].max(axis=1).to_numpy())
    return X, cross, gap, kept


def _gap_row(X, cross, gap, kept, tr, te) -> tuple[tuple, tuple]:
    """(model, baseline) each as (Spearman, enrichment) on one split."""
    model_sp, _, model_en = evaluate_split(X, cross, gap, tr, te)
    nn = nn_predict([kept[i] for i in te], [kept[i] for i in tr], gap[tr])
    ok = ~np.isnan(nn)
    return ((model_sp, model_en),
            (float(spearmanr(gap[te][ok], nn[ok]).statistic), _enrichment(gap[te][ok], nn[ok])))


def gap_scaffold() -> None:
    X, cross, gap, kept = _gap_data()
    rows = [_gap_row(X, cross, gap, kept, *scaffold_split(kept, 0.2, seed=s)) for s in SEEDS]
    model = np.array([r[0] for r in rows])
    base = np.array([r[1] for r in rows])
    print(f"\nGAP axis, scaffold split (n={len(kept)}, {len(SEEDS)} seeds)")
    print(f"  {'':22} {'Spearman':>18} {'top-decile enrich':>20}")
    print(f"  {'model (diff-of-reg)':22} {model[:, 0].mean():>11.3f} ± {model[:, 0].std():.3f}"
          f" {model[:, 1].mean():>13.2f}x")
    print(f"  {'1-NN Tanimoto':22} {base[:, 0].mean():>11.3f} ± {base[:, 0].std():.3f}"
          f" {base[:, 1].mean():>13.2f}x")
    print(f"  {'model advantage':22} {model[:, 0].mean() - base[:, 0].mean():>+11.3f}"
          f" {model[:, 1].mean() - base[:, 1].mean():>+13.2f}x")


def gap_time() -> None:
    X, cross, gap, kept = _gap_data()
    year = _year_first(DEFAULT_PANEL)
    years = np.array([year.get(s, np.nan) for s in kept])
    tr = np.flatnonzero(years <= TIME_CUT)
    te = np.flatnonzero(years > TIME_CUT)
    (m_sp, m_en), (b_sp, b_en) = _gap_row(X, cross, gap, kept, tr, te)
    print(f"\nGAP axis, {TIME_CUT} time split (train {len(tr)}, test {len(te)})")
    print(f"  {'model (diff-of-reg)':22} {m_sp:>11.3f} {m_en:>13.2f}x")
    print(f"  {'1-NN Tanimoto':22} {b_sp:>11.3f} {b_en:>13.2f}x")
    print(f"  {'model advantage':22} {m_sp - b_sp:>+11.3f} {m_en - b_en:>+13.2f}x")


def potency_scaffold() -> None:
    """One reference point on the other axis, so the gap result is not read alone."""
    iso = DEFAULT_PANEL.target
    data = panel_data.build_isoform_dataset(DEFAULT_PANEL, iso)
    smiles = data["smi"].tolist()
    X, mask = morgan_matrix(smiles)
    y = data["pchembl"].to_numpy()[mask]
    kept = [s for s, k in zip(smiles, mask) if k]

    tr, te = scaffold_split(kept, 0.2, seed=0)
    pred = ir._fit(X[tr], y[tr]).predict(X[te])
    nn = nn_predict([kept[i] for i in te], [kept[i] for i in tr], y[tr])
    ok = ~np.isnan(nn)
    print(f"\nPOTENCY axis ({iso}), scaffold split seed 0 (train {len(tr)}, test {len(te)})")
    print(f"  {'':22} {'Spearman':>11} {'MAE':>10}")
    print(f"  {'model':22} {spearmanr(y[te], pred).statistic:>11.3f}"
          f" {mean_absolute_error(y[te], pred):>10.3f}")
    print(f"  {'1-NN Tanimoto':22} {spearmanr(y[te][ok], nn[ok]).statistic:>11.3f}"
          f" {mean_absolute_error(y[te][ok], nn[ok]):>10.3f}")


def main() -> None:
    print("=" * 72)
    print("N1 — model vs Tanimoto 1-nearest-neighbour, same split, same fingerprints")
    print("=" * 72)
    gap_scaffold()
    gap_time()
    potency_scaffold()
    print("\nHow to read this. The baseline has no model: it copies the measured value")
    print("of the single most similar training molecule. Whatever the model scores")
    print("above that line is what the learning is worth.")


if __name__ == "__main__":
    main()
