"""STEP 5: split-conformal regression intervals (Stage-B Tier 2).

Attaches a calibrated prediction interval to every per-isoform pchembl estimate.
Inductive/split conformal: fit on a proper-train split, compute absolute residuals
on a disjoint calibration split, and take the finite-sample-corrected quantile as
the interval half-width `q`. Under exchangeability the interval
`[pred - q, pred + q]` covers the truth with the nominal probability (default 90%).

All three splits (proper-train / calibration / test) are scaffold-disjoint, so
coverage is measured on novel chemotypes — the honest stress case. The gap `S`
interval is propagated from the two contributing isoform half-widths.

CLI (empirical coverage at 90% nominal, per isoform):
    python -m src.conformal
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import jak
from .models.features import morgan_matrix
from .models.scaffold_split import scaffold_split

TARGET_ISOFORMS = ("JAK1", "JAK2", "JAK3")
SEEDS = (0, 1, 2, 3, 4)
DEFAULT_ALPHA = 0.10          # 90% nominal coverage


def conformal_quantile(cal_residuals: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """Finite-sample conformal half-width: the ceil((n+1)(1-alpha))/n residual quantile."""
    n = len(cal_residuals)
    level = min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(cal_residuals, level, method="higher"))


def predict_interval(pred: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric conformal interval around a point prediction."""
    return pred - q, pred + q


def gap_interval(q_target: float, q_off: float) -> float:
    """Conservative half-width for the gap S = pred(target) - max(off): the two add."""
    return q_target + q_off


def _fit(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                         random_state=0).fit(X, y)


def _seed_errors(isoform: str, seed: int, use_cache: bool = True
                 ) -> tuple[np.ndarray, np.ndarray]:
    """One seed's (test |residuals|, calibration |residuals|) with disjoint scaffolds."""
    data = jak.build_isoform_dataset(isoform, use_cache=use_cache)
    smiles = data["smi"].tolist()
    X, mask = morgan_matrix(smiles)
    y = data["pchembl"].to_numpy()[mask]
    kept = [s for s, keep in zip(smiles, mask) if keep]

    trpool, test = scaffold_split(kept, test_frac=0.2, seed=seed)
    ptr_rel, cal_rel = scaffold_split([kept[i] for i in trpool], test_frac=0.25, seed=seed)
    ptr, cal = trpool[ptr_rel], trpool[cal_rel]

    model = _fit(X[ptr], y[ptr])
    cal_res = np.abs(y[cal] - model.predict(X[cal]))
    test_err = np.abs(y[test] - model.predict(X[test]))
    return test_err, cal_res


@dataclass
class CoverageMetrics:
    isoform: str
    nominal: float
    coverage_mean: float
    coverage_std: float
    width_mean: float
    width_std: float
    n_seeds: int


def evaluate_coverage(isoform: str, alpha: float = DEFAULT_ALPHA,
                      seeds: tuple[int, ...] = SEEDS, use_cache: bool = True) -> CoverageMetrics:
    """Empirical coverage + mean interval width at a nominal level, over several seeds."""
    covs, widths = [], []
    for seed in seeds:
        test_err, cal_res = _seed_errors(isoform, seed, use_cache=use_cache)
        q = conformal_quantile(cal_res, alpha)
        covs.append(float((test_err <= q).mean()))
        widths.append(2 * q)
    return CoverageMetrics(
        isoform=isoform, nominal=1 - alpha,
        coverage_mean=float(np.mean(covs)), coverage_std=float(np.std(covs)),
        width_mean=float(np.mean(widths)), width_std=float(np.std(widths)),
        n_seeds=len(seeds),
    )


def _main() -> None:
    print(f"Split-conformal coverage at {int((1-DEFAULT_ALPHA)*100)}% nominal "
          f"({len(SEEDS)} seeds, scaffold-disjoint):")
    print(f"  {'isoform':7} {'coverage':>16} {'interval width (pchembl)':>26}")
    for iso in TARGET_ISOFORMS:
        m = evaluate_coverage(iso)
        print(f"  {iso:7} {m.coverage_mean:6.3f} ± {m.coverage_std:.3f}     "
              f"{m.width_mean:6.2f} ± {m.width_std:.2f}")


if __name__ == "__main__":
    _main()
