"""STEP 3: per-isoform pchembl regressor (the Stage-B Tier-1 engine).

One HistGradientBoosting regressor per panel member, predicting pchembl from ECFP4
— the v1 QSAR approach, reused per isoform. Evaluated with a **scaffold split over
several seeds** so the reported MAE / RMSE / R2 / Spearman come with mean +- std,
not a single fragile draw. The deployed model is refit on all data; the seeded
splits are only for honest metrics.

Models are cached under the panel's own namespace (`assets/models/<panel>/`), so a
second panel cannot load — or overwrite — the first one's regressors.

CLI (evaluate + cache one panel, print the metrics table):
    python -m src.models.isoform_regressor [panel]
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..data import panel_data
from ..panels import DEFAULT_PANEL, PanelSpec, get_panel
from .features import morgan_matrix
from .scaffold_split import scaffold_split

SEEDS = (0, 1, 2, 3, 4)


@dataclass
class IsoformMetrics:
    isoform: str
    n_molecules: int
    n_seeds: int
    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float
    r2_mean: float
    r2_std: float
    spearman_mean: float
    spearman_std: float


@dataclass
class IsoformModel:
    isoform: str
    model: HistGradientBoostingRegressor
    metrics: IsoformMetrics

    def predict(self, smiles: list[str]) -> np.ndarray:
        """Predicted pchembl aligned with input; NaN where SMILES fail to parse."""
        X, mask = morgan_matrix(list(smiles))
        out = np.full(len(mask), np.nan)
        if X.shape[0]:
            out[mask] = self.model.predict(X)
        return out


def _fit(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=0)
    model.fit(X, y)
    return model


def _eval_one_seed(X: np.ndarray, y: np.ndarray, smiles: list[str], seed: int) -> dict:
    train_idx, test_idx = scaffold_split(smiles, test_frac=0.2, seed=seed)
    pred = _fit(X[train_idx], y[train_idx]).predict(X[test_idx])
    yt = y[test_idx]
    return {
        "mae": mean_absolute_error(yt, pred),
        "rmse": float(np.sqrt(mean_squared_error(yt, pred))),
        "r2": r2_score(yt, pred),
        "spearman": float(spearmanr(yt, pred).statistic),
    }


def evaluate(panel: PanelSpec, name: str, seeds: tuple[int, ...] = SEEDS,
             use_cache: bool = True,
             data: pd.DataFrame | None = None) -> IsoformMetrics:
    """Scaffold-split metrics over several seeds, as mean +- std.

    `data` (columns `smi`, `pchembl`) trains on a caller-supplied set instead of
    the panel's own; see `train_and_cache` for why the seam exists.
    """
    if data is None:
        data = panel_data.build_isoform_dataset(panel, name, use_cache=use_cache)
    smiles = data["smi"].tolist()
    X, mask = morgan_matrix(smiles)
    y = data["pchembl"].to_numpy()[mask]
    kept = [s for s, keep in zip(smiles, mask) if keep]

    per = [_eval_one_seed(X, y, kept, s) for s in seeds]
    agg = {k: (float(np.mean([p[k] for p in per])), float(np.std([p[k] for p in per])))
           for k in ("mae", "rmse", "r2", "spearman")}
    return IsoformMetrics(
        isoform=name, n_molecules=len(y), n_seeds=len(seeds),
        mae_mean=agg["mae"][0], mae_std=agg["mae"][1],
        rmse_mean=agg["rmse"][0], rmse_std=agg["rmse"][1],
        r2_mean=agg["r2"][0], r2_std=agg["r2"][1],
        spearman_mean=agg["spearman"][0], spearman_std=agg["spearman"][1],
    )


def _load(path: Path) -> IsoformModel:
    # Pickled as a plain dict (not the dataclass) so the cache loads regardless of
    # whether it was written from `python -m ...` (__main__) or an import.
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return IsoformModel(isoform=d["isoform"], model=d["model"],
                        metrics=IsoformMetrics(**d["metrics"]))


def _save(bundle: IsoformModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"isoform": bundle.isoform, "model": bundle.model,
                     "metrics": asdict(bundle.metrics)}, fh)


def train_and_cache(panel: PanelSpec, name: str, use_cache: bool = True,
                    data: pd.DataFrame | None = None) -> IsoformModel:
    """Evaluate (seeded splits), refit on all data, cache the deployed model.

    A committed model in assets/ is used when no runtime cache exists, so a fresh
    deploy loads in milliseconds instead of retraining on ~10k molecules.

    `data` (columns `smi`, `pchembl`) is the training-set injection seam: the
    learning curve, the leave-one-target-out baseline and the nearest-neighbour
    comparison all have to train on a *subset*, which the panel-derived path
    cannot express. An injected run neither reads nor writes the model cache — a
    subset-trained model must never land where the deployed one lives.
    """
    injected = data is not None
    if use_cache and not injected:
        for directory in (panel.model_cache, panel.model_bundled):
            path = directory / f"{name}_reg.pkl"
            if path.exists():
                return _load(path)

    metrics = evaluate(panel, name, use_cache=use_cache, data=data)
    if data is None:
        data = panel_data.build_isoform_dataset(panel, name, use_cache=use_cache)
    X, mask = morgan_matrix(data["smi"].tolist())
    y = data["pchembl"].to_numpy()[mask]
    bundle = IsoformModel(isoform=name, model=_fit(X, y), metrics=metrics)
    if not injected:
        # retrains land in the runtime cache, never on top of a committed asset
        _save(bundle, panel.model_cache / f"{name}_reg.pkl")
    return bundle


def _main() -> None:
    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    print(f"Panel {panel.name}")
    print(f"{'isoform':7} {'n':>6} {'MAE':>13} {'RMSE':>13} {'R2':>13} {'Spearman':>13}")
    for name in panel.isoforms:
        m = train_and_cache(panel, name, use_cache=False).metrics
        print(f"{m.isoform:7} {m.n_molecules:6d} "
              f"{m.mae_mean:.3f}±{m.mae_std:.3f}  "
              f"{m.rmse_mean:.3f}±{m.rmse_std:.3f}  "
              f"{m.r2_mean:.3f}±{m.r2_std:.3f}  "
              f"{m.spearman_mean:.3f}±{m.spearman_std:.3f}")


if __name__ == "__main__":
    _main()
