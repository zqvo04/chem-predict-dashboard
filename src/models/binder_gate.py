"""STEP 10: the JAK binder gate (Stage-B Tier 0.5).

A binary "is this a plausible JAK binder?" classifier that runs *before* the
per-isoform regressors. The regressors are trained only on quantified JAK pchembl,
so on an off-domain molecule they revert to their training mean (~6.3) and the
molecule sails over the potency floor with a meaningless selectivity gap. The gate
drops those molecules first, so Tier 1 only ranks chemistry that plausibly engages
JAK at all.

  positives  JAK actives, pchembl >= 6, any isoform          (src/data/jak.py)
  negatives  physchem-matched presumed-inactives              (src/data/negatives.py)

ECFP4 + HistGradientBoosting, scaffold-split evaluation over several seeds — the
same honest protocol as the regressors. The operating point is **Youden's J** (the
threshold maximising TPR − FPR) measured on held-out folds and averaged over seeds
— the standard balanced cut, data-derived rather than a round number. It is
deliberately *not* a high positive-recall threshold: known JAK actives score ~1.0,
so a recall-set cut only admits molecules that already look like a *known* JAK
chemotype and rejects the novel scaffolds a discovery screen exists to find. The
balanced cut strips the confident non-binders (ethanol, off-target junk) while
leaving plausible-but-uncertain chemistry for Tier 1 to rank.

The gate does not touch the regression, the selectivity gap, conformal intervals
or AD — it filters what reaches them — so every number those modules report is
unchanged. It is the genuine binder/non-binder gate DESIGN_DECISIONS section 1 kept
in reserve, now that the funnel's live output shows AD alone did not carry the
censored-label burden.

CLI (evaluate + cache, print metrics):
    python -m src.models.binder_gate
"""
from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from ..data import jak
from ..data.negatives import build_negatives, jak_positive_smiles
from .features import morgan_matrix
from .scaffold_split import scaffold_split

SEEDS = (0, 1, 2, 3, 4)
_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = _ROOT / "data" / "models" / "jak"                   # runtime cache, gitignored
BUNDLED_MODEL_DIR = _ROOT / "assets" / "models" / "jak"         # committed for deploys


@dataclass
class GateMetrics:
    n_pos: int
    n_neg: int
    n_seeds: int
    auc_mean: float
    auc_std: float
    ap_mean: float
    ap_std: float
    threshold: float            # Youden's J operating point; P(binder) below it is gated out
    pos_recall_mean: float      # fraction of held-out actives kept at that threshold
    neg_reject_mean: float      # fraction of held-out negatives dropped at that threshold


@dataclass
class BinderGate:
    model: HistGradientBoostingClassifier
    threshold: float
    metrics: GateMetrics

    def predict_proba(self, smiles: list[str]) -> np.ndarray:
        """P(JAK binder) aligned with input; NaN where a SMILES fails to parse."""
        X, mask = morgan_matrix(list(smiles))
        out = np.full(len(mask), np.nan)
        if X.shape[0]:
            out[mask] = self.model.predict_proba(X)[:, 1]
        return out

    def is_binder(self, smiles: list[str]) -> np.ndarray:
        """Gate verdict: P(binder) >= threshold. Unparseable SMILES fail closed."""
        p = self.predict_proba(smiles)
        return np.where(np.isnan(p), False, p >= self.threshold)


def _fit(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    # Balanced sample weights so the score reflects separability, not the class
    # ratio left by physchem matching.
    w = np.where(y == 1, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=0)
    model.fit(X, y, sample_weight=w)
    return model


def _dataset(use_cache: bool = True) -> tuple[list[str], np.ndarray]:
    """Combined (smiles, label) with 1 = JAK binder, 0 = matched presumed-inactive."""
    pos = sorted(jak_positive_smiles(use_cache=use_cache))
    neg = build_negatives(use_cache=use_cache)["smi"].tolist()
    smiles = pos + neg
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return smiles, y


def evaluate(seeds: tuple[int, ...] = SEEDS, use_cache: bool = True) -> GateMetrics:
    """Scaffold-split AUC / average precision over seeds, and the Youden's-J threshold.

    The threshold is Youden's J (argmax of TPR − FPR on the held-out ROC) averaged
    over seeds — measured on molecules whose scaffolds the model never trained on —
    and reported with the recall it keeps and the negative-rejection it buys.
    """
    smiles, y = _dataset(use_cache=use_cache)
    X, mask = morgan_matrix(smiles)
    y = y[mask]
    kept = [s for s, k in zip(smiles, mask) if k]

    aucs, aps, thresholds, pos_probs, neg_probs = [], [], [], [], []
    for seed in seeds:
        tr, te = scaffold_split(kept, test_frac=0.2, seed=seed)
        proba = _fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        yte = y[te]
        aucs.append(float(roc_auc_score(yte, proba)))
        aps.append(float(average_precision_score(yte, proba)))
        fpr, tpr, thr = roc_curve(yte, proba)
        thresholds.append(float(thr[np.argmax(tpr - fpr)]))
        pos_probs.append(proba[yte == 1])
        neg_probs.append(proba[yte == 0])

    threshold = float(np.mean(thresholds))
    pos_all, neg_all = np.concatenate(pos_probs), np.concatenate(neg_probs)
    return GateMetrics(
        n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()), n_seeds=len(seeds),
        auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
        ap_mean=float(np.mean(aps)), ap_std=float(np.std(aps)),
        threshold=threshold,
        pos_recall_mean=float((pos_all >= threshold).mean()),
        neg_reject_mean=float((neg_all < threshold).mean()),
    )


def _load(path: Path) -> BinderGate:
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return BinderGate(model=d["model"], threshold=d["threshold"],
                      metrics=GateMetrics(**d["metrics"]))


def _save(gate: BinderGate, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"model": gate.model, "threshold": gate.threshold,
                     "metrics": asdict(gate.metrics)}, fh)


def train_and_cache(use_cache: bool = True) -> BinderGate:
    """Evaluate (seeded splits), refit on all data, cache the deployed gate."""
    if use_cache:
        for directory in (MODEL_DIR, BUNDLED_MODEL_DIR):
            path = directory / "binder_gate.pkl"
            if path.exists():
                return _load(path)

    metrics = evaluate(use_cache=use_cache)
    smiles, y = _dataset(use_cache=use_cache)
    X, mask = morgan_matrix(smiles)
    gate = BinderGate(model=_fit(X, y[mask]), threshold=metrics.threshold, metrics=metrics)
    _save(gate, MODEL_DIR / "binder_gate.pkl")
    return gate


def _main() -> None:
    m = train_and_cache(use_cache=False).metrics
    print(f"JAK binder gate  (pos {m.n_pos} / neg {m.n_neg}, {m.n_seeds} seeds, scaffold split)")
    print(f"  ROC-AUC            : {m.auc_mean:.3f} ± {m.auc_std:.3f}")
    print(f"  Average precision  : {m.ap_mean:.3f} ± {m.ap_std:.3f}")
    print(f"  Youden's-J threshold : {m.threshold:.3f} "
          f"(keeps {m.pos_recall_mean:.1%} of actives, rejects {m.neg_reject_mean:.1%} of negatives)")


if __name__ == "__main__":
    _main()
