"""Phase 3: per-target activity regression model (QSAR).

Predicts pchembl_value (potency; higher = more potent) from Morgan fingerprints
with a RandomForest. The model is trained on-the-fly for a target from ChEMBL
data, evaluated with a scaffold split (no scaffold leakage), and cached to disk
so the dashboard pays the training cost only once per target.

CLI:
    python -m src.models.target_model EGFR

Honest scope notes:
  - The accessible training range is limited to molecules with a *quantified*
    pchembl in ChEMBL, which skews toward measured (often active-ish) compounds.
    True hard-negatives are under-represented.
  - Predictions are only trustworthy inside the model's applicability domain
    (chemotypes near the training set). Novel scaffolds extrapolate poorly.
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from ..data import chembl_client as cc
from .features import morgan_matrix
from .scaffold_split import scaffold_split

_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_MODEL_DIR = _ROOT / "assets" / "models"  # committed demo models (instant load)
MODEL_DIR = _ROOT / "data" / "models"            # runtime cache (gitignored)
MIN_TRAIN_MOLECULES = 50  # data-sufficiency gate: below this we refuse to train


@dataclass
class ModelMetrics:
    n_molecules: int
    n_test: int
    r2: float
    rmse: float
    pchembl_min: float
    pchembl_max: float


@dataclass
class TargetModel:
    target: cc.Target
    model: HistGradientBoostingRegressor
    metrics: ModelMetrics

    def predict(self, smiles: list[str]) -> np.ndarray:
        """Predicted pchembl aligned with the input; NaN where SMILES fail."""
        X, mask = morgan_matrix(list(smiles))
        preds = np.full(len(mask), np.nan)
        if X.shape[0]:
            preds[mask] = self.model.predict(X)
        return preds


def build_training_set(target_id: str, max_records: int = 4000) -> pd.DataFrame:
    """Full measured activity range for a target, one median pchembl per molecule.

    Median over replicate measurements reduces cross-assay noise.
    """
    acts = cc.fetch_activities(target_id, pchembl_gte=None, max_records=max_records)
    if acts.empty:
        return pd.DataFrame(columns=["molecule_chembl_id", "canonical_smiles", "pchembl_value"])

    acts = acts.copy()
    acts["pchembl_value"] = pd.to_numeric(acts["pchembl_value"], errors="coerce")
    acts = acts.dropna(subset=["canonical_smiles", "pchembl_value"])
    return (
        acts.groupby("molecule_chembl_id", sort=False)
            .agg(canonical_smiles=("canonical_smiles", "first"),
                 pchembl_value=("pchembl_value", "median"))
            .reset_index()
    )


def _fit_model(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    # Gradient boosting beats RandomForest here on both accuracy and pickle
    # size (~1.5 MB vs ~54 MB), which matters for the free-tier deployment.
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=0)
    model.fit(X, y)
    return model


def _load_cached(chembl_id: str) -> TargetModel | None:
    """Prefer a committed demo model, then the runtime cache."""
    for directory in (BUNDLED_MODEL_DIR, MODEL_DIR):
        path = directory / f"{chembl_id}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                return pickle.load(fh)
    return None


def train_target_model(query: str, organism: str | None = "Homo sapiens",
                       max_records: int = 4000, test_frac: float = 0.2,
                       use_cache: bool = True) -> TargetModel:
    """Resolve a target, train a scaffold-validated pchembl regressor, cache it."""
    target = cc.resolve_target(query, organism=organism)
    if use_cache:
        cached = _load_cached(target.chembl_id)
        if cached is not None:
            return cached

    data = build_training_set(target.chembl_id, max_records=max_records)
    if len(data) < MIN_TRAIN_MOLECULES:
        raise ValueError(
            f"Target {target.chembl_id} has only {len(data)} usable molecules "
            f"(< {MIN_TRAIN_MOLECULES}); not enough to train a reliable model.")

    smiles = data["canonical_smiles"].tolist()
    y_all = data["pchembl_value"].to_numpy()
    X, mask = morgan_matrix(smiles)
    y = y_all[mask]
    kept_smiles = [s for s, keep in zip(smiles, mask) if keep]

    train_idx, test_idx = scaffold_split(kept_smiles, test_frac=test_frac)
    if len(test_idx) == 0 or len(train_idx) == 0:
        raise ValueError("Scaffold split produced an empty train or test set.")

    eval_model = _fit_model(X[train_idx], y[train_idx])
    pred = eval_model.predict(X[test_idx])
    metrics = ModelMetrics(
        n_molecules=len(y),
        n_test=len(test_idx),
        r2=float(r2_score(y[test_idx], pred)),
        rmse=float(np.sqrt(mean_squared_error(y[test_idx], pred))),
        pchembl_min=float(y.min()),
        pchembl_max=float(y.max()),
    )

    # Deployed model is refit on all data (eval model was only for honest metrics).
    final_model = _fit_model(X, y)
    bundle = TargetModel(target=target, model=final_model, metrics=metrics)

    if use_cache:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / f"{target.chembl_id}.pkl", "wb") as fh:
            pickle.dump(bundle, fh)
    return bundle


def _main() -> None:
    ap = argparse.ArgumentParser(description="Train a per-target pchembl regressor")
    ap.add_argument("target", help="Target name or ChEMBL id, e.g. EGFR")
    ap.add_argument("--organism", default="Homo sapiens")
    ap.add_argument("--max-records", type=int, default=4000)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    tm = train_target_model(args.target, organism=args.organism,
                            max_records=args.max_records, use_cache=not args.no_cache)
    m = tm.metrics
    print(f"Target : {tm.target.chembl_id}  {tm.target.pref_name}")
    print(f"Train  : {m.n_molecules} molecules, pchembl range {m.pchembl_min:.2f}-{m.pchembl_max:.2f}")
    print(f"Eval   : scaffold-split test n={m.n_test}  R2={m.r2:.3f}  RMSE={m.rmse:.3f}")


if __name__ == "__main__":
    _main()
