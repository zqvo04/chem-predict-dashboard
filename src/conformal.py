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

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import panel_data
from .panels import DEFAULT_PANEL, PanelSpec, get_panel
from .models.features import morgan_matrix
from .models.scaffold_split import scaffold_split

SEEDS = (0, 1, 2, 3, 4)
DEFAULT_ALPHA = 0.10          # 90% nominal coverage

QUANTILES_FILE = "conformal_quantiles.json"


def bundled_quantiles(panel: PanelSpec) -> Path:
    """Deploy-time half-widths for one panel's bundled models.

    Calibration is deterministic given the pinned dataset + seed, so the committed
    table is exactly what `halfwidth`/`calibrate_gap` would recompute — it is
    committed only so a fresh deploy skips the model fits. Namespaced per panel
    (STEP 15); the JAK table moved from `assets/` to `assets/jak/` unchanged, keys
    and values included.
    """
    return panel.data_bundled / QUANTILES_FILE


def conformal_quantile(cal_residuals: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """Finite-sample conformal half-width: the ceil((n+1)(1-alpha))/n residual quantile."""
    n = len(cal_residuals)
    level = min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(cal_residuals, level, method="higher"))


def predict_interval(pred: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric conformal interval around a point prediction."""
    return pred - q, pred + q


def gap_interval(q_target: float, q_off: float) -> float:
    """Legacy worst-case half-width for the gap: the two isoform half-widths add.

    Superseded by the directly-calibrated `gap_halfwidth` (STEP 14) and kept only
    because it documents what the funnel used to do. Summing assumes the two
    isoform errors are independent and adverse; measured, they correlate at
    **+0.65**, so they largely cancel in the difference and the sum over-pays —
    it delivered 99.7 % coverage at a 90 % nominal level, twice the necessary width.
    """
    return q_target + q_off


# --- directly-calibrated, difficulty-scaled gap interval (STEP 14) ------------ #

SIGMA_KNOTS = 21          # piecewise-linear sigma curve, on nn-similarity in [0,1]
_SIGMA_FLOOR = 0.15       # keeps sigma away from zero where calibration data is thin


def _gap_predict(models: dict, X: np.ndarray, panel: PanelSpec) -> np.ndarray:
    p = {i: models[i].predict(X) for i in panel.isoforms}
    return p[panel.target] - np.maximum.reduce([p[o] for o in panel.offs])


def _fit_sigma(nn_sim: np.ndarray, residual: np.ndarray) -> dict:
    """A difficulty curve sigma(nn_similarity), returned as interpolation knots.

    The gap error is not constant: it grows as a molecule sits further from the
    training set. Fitting that dependence and *scaling* the interval by it is what
    makes coverage hold for hard molecules instead of only on average. Stored as
    knots rather than a pickled model — it is a 1-D monotone-ish curve, and knots
    are inspectable, tiny and version-proof.
    """
    model = HistGradientBoostingRegressor(max_iter=120, random_state=0).fit(
        nn_sim.reshape(-1, 1), np.log(residual + 1e-3))
    xs = np.linspace(0.0, 1.0, SIGMA_KNOTS)
    ys = np.clip(np.exp(model.predict(xs.reshape(-1, 1))), _SIGMA_FLOOR, None)
    return {"x": xs.tolist(), "y": ys.tolist()}


def sigma_at(nn_sim, knots: dict) -> np.ndarray:
    """Evaluate the difficulty curve at given nearest-neighbour similarities."""
    return np.interp(np.asarray(nn_sim, dtype=float), knots["x"], knots["y"])


def calibrate_gap(panel: PanelSpec = DEFAULT_PANEL, alpha: float = DEFAULT_ALPHA,
                  seed: int = 0, use_cache: bool = True) -> dict:
    """Calibrate the gap interval directly, scaled by distance from training.

    Split-conformal on the **gap residual itself** rather than on each isoform
    separately, over the cross-measured set (the only place a *measured* gap
    exists). The calibration half is split again: one part fits the difficulty
    curve, the other supplies the conformal quantile, so the curve is never fitted
    and scored on the same molecules.

    Returns `{"q": float, "sigma": {"x": [...], "y": [...]}}`; the half-width for a
    molecule is `q * sigma(nn_similarity)`.
    """
    from .applicability import tanimoto_nn_similarity

    cross = panel_data.build_cross_measured(panel, use_cache=use_cache)
    smiles = cross["smi"].tolist()
    X, mask = morgan_matrix(smiles)
    cross = cross[mask].reset_index(drop=True)
    kept = [s for s, k in zip(smiles, mask) if k]
    meas = {i: cross[i].to_numpy() for i in panel.isoforms}
    meas_gap = meas[panel.target] - np.maximum.reduce([meas[o] for o in panel.offs])

    trpool, _ = scaffold_split(kept, test_frac=0.2, seed=seed)
    ptr_rel, cal_rel = scaffold_split([kept[i] for i in trpool], test_frac=0.25, seed=seed)
    ptr, cal = trpool[ptr_rel], trpool[cal_rel]

    models = {i: _fit(X[ptr], meas[i][ptr]) for i in panel.isoforms}
    train_smi = [kept[i] for i in ptr]
    fit_idx, quant_idx = cal[::2], cal[1::2]

    def resid_and_sim(idx):
        r = np.abs(meas_gap[idx] - _gap_predict(models, X[idx], panel))
        s = tanimoto_nn_similarity([kept[i] for i in idx], train_smi)
        return r, s

    r_fit, s_fit = resid_and_sim(fit_idx)
    r_q, s_q = resid_and_sim(quant_idx)
    knots = _fit_sigma(s_fit, r_fit)
    q = conformal_quantile(r_q / sigma_at(s_q, knots), alpha)
    return {"q": float(q), "sigma": knots}


def _bundled_gap(panel: PanelSpec, alpha: float, seed: int) -> dict | None:
    path = bundled_quantiles(panel)
    if not path.exists():
        return None
    table = json.loads(path.read_text(encoding="utf-8"))
    return table.get(f"gap|alpha={alpha}|seed={seed}")


@lru_cache(maxsize=8)
def gap_calibration(panel: PanelSpec = DEFAULT_PANEL, alpha: float = DEFAULT_ALPHA,
                    seed: int = 0, use_cache: bool = True) -> dict:
    """The deployed gap calibration — committed table first, else computed."""
    if use_cache:
        cached = _bundled_gap(panel, alpha, seed)
        if cached is not None:
            return cached
    return calibrate_gap(panel, alpha, seed, use_cache=use_cache)


def gap_halfwidth(nn_sim, panel: PanelSpec = DEFAULT_PANEL, alpha: float = DEFAULT_ALPHA,
                  seed: int = 0, use_cache: bool = True) -> np.ndarray:
    """Per-molecule gap half-width, widening as a molecule leaves the training set.

    `nn_sim` is the nearest-neighbour Tanimoto similarity to training, taken as the
    **minimum across the contributing isoforms** — the gap is only as trustworthy as
    the most-extrapolating model behind it, the same worst-case rule the
    applicability-domain verdict already uses.
    """
    cal = gap_calibration(panel, alpha, seed, use_cache)
    return cal["q"] * sigma_at(nn_sim, cal["sigma"])


def _fit(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                         random_state=0).fit(X, y)


def _seed_errors(panel: PanelSpec, isoform: str, seed: int, use_cache: bool = True
                 ) -> tuple[np.ndarray, np.ndarray]:
    """One seed's (test |residuals|, calibration |residuals|) with disjoint scaffolds."""
    data = panel_data.build_isoform_dataset(panel, isoform, use_cache=use_cache)
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


def _bundled_quantile(panel: PanelSpec, isoform: str, alpha: float,
                      seed: int) -> float | None:
    """The committed half-width for this exact (isoform, alpha, seed), if present."""
    path = bundled_quantiles(panel)
    if not path.exists():
        return None
    table = json.loads(path.read_text(encoding="utf-8"))
    value = table.get(f"{isoform}|alpha={alpha}|seed={seed}")
    return float(value) if value is not None else None


def halfwidth(panel: PanelSpec, isoform: str, alpha: float = DEFAULT_ALPHA,
              seed: int = 0, use_cache: bool = True) -> float:
    """A calibrated interval half-width q for the deployed screen (one split).

    Calibration means fitting a model on the proper-train split, so the committed
    table is consulted first; any (alpha, seed) it does not cover is computed.
    """
    if use_cache:
        cached = _bundled_quantile(panel, isoform, alpha, seed)
        if cached is not None:
            return cached
    _, cal_res = _seed_errors(panel, isoform, seed, use_cache=use_cache)
    return conformal_quantile(cal_res, alpha)


@dataclass
class CoverageMetrics:
    isoform: str
    nominal: float
    coverage_mean: float
    coverage_std: float
    width_mean: float
    width_std: float
    n_seeds: int


def evaluate_coverage(panel: PanelSpec, isoform: str, alpha: float = DEFAULT_ALPHA,
                      seeds: tuple[int, ...] = SEEDS, use_cache: bool = True) -> CoverageMetrics:
    """Empirical coverage + mean interval width at a nominal level, over several seeds."""
    covs, widths = [], []
    for seed in seeds:
        test_err, cal_res = _seed_errors(panel, isoform, seed, use_cache=use_cache)
        q = conformal_quantile(cal_res, alpha)
        covs.append(float((test_err <= q).mean()))
        widths.append(2 * q)
    return CoverageMetrics(
        isoform=isoform, nominal=1 - alpha,
        coverage_mean=float(np.mean(covs)), coverage_std=float(np.std(covs)),
        width_mean=float(np.mean(widths)), width_std=float(np.std(widths)),
        n_seeds=len(seeds),
    )


SIM_BUCKETS = ((0.0, 0.35), (0.35, 0.45), (0.45, 0.60), (0.60, 1.01))


def evaluate_gap_coverage(panel: PanelSpec = DEFAULT_PANEL, alpha: float = DEFAULT_ALPHA,
                          seeds: tuple[int, ...] = SEEDS, use_cache: bool = True) -> dict:
    """Marginal *and* per-difficulty coverage of the gap interval, old vs new.

    Marginal coverage alone hides the failure this replaced: a single constant
    width over-covers molecules close to training and badly under-covers the ones
    far from it, which is exactly where a wide screen operates. Coverage is
    therefore reported per nearest-neighbour-similarity bucket.
    """
    from .applicability import tanimoto_nn_similarity

    cross = panel_data.build_cross_measured(panel, use_cache=use_cache)
    smiles = cross["smi"].tolist()
    X, mask = morgan_matrix(smiles)
    cross = cross[mask].reset_index(drop=True)
    kept = [s for s, k in zip(smiles, mask) if k]
    meas = {i: cross[i].to_numpy() for i in panel.isoforms}
    meas_gap = meas[panel.target] - np.maximum.reduce([meas[o] for o in panel.offs])

    # Three arms, because the middle one is the trap: "old" (summed half-widths) is
    # what shipped, "flat" is the obvious fix (calibrate on the gap, one width for
    # everyone), and "new" additionally scales that width by distance from training.
    # Without the flat arm the comparison would hide why the scaling is needed.
    arms = ("old", "flat", "new")
    out = {f"{k}_{a}": [] for a in arms
           for k in ("marginal", "width", "crosses_zero")}
    out.update({f"bucket_{a}": {b: [] for b in SIM_BUCKETS} for a in arms})

    for seed in seeds:
        trpool, test = scaffold_split(kept, test_frac=0.2, seed=seed)
        ptr_rel, cal_rel = scaffold_split([kept[i] for i in trpool], test_frac=0.25, seed=seed)
        ptr, cal = trpool[ptr_rel], trpool[cal_rel]
        models = {i: _fit(X[ptr], meas[i][ptr]) for i in panel.isoforms}
        train_smi = [kept[i] for i in ptr]

        g_te = _gap_predict(models, X[test], panel)
        r_te = np.abs(meas_gap[test] - g_te)
        s_te = tanimoto_nn_similarity([kept[i] for i in test], train_smi)

        cal_gap = calibrate_gap(panel, alpha, seed, use_cache=use_cache)
        w_new = cal_gap["q"] * sigma_at(s_te, cal_gap["sigma"])
        q_iso = {i: conformal_quantile(
            np.abs(meas[i][cal] - models[i].predict(X[cal])), alpha) for i in panel.isoforms}
        w_old = np.full(len(test), q_iso[panel.target] + max(q_iso[o] for o in panel.offs))
        r_cal = np.abs(meas_gap[cal] - _gap_predict(models, X[cal], panel))
        w_flat = np.full(len(test), conformal_quantile(r_cal, alpha))

        for tag, w in (("new", w_new), ("old", w_old), ("flat", w_flat)):
            out[f"marginal_{tag}"].append(float((r_te <= w).mean()))
            out[f"width_{tag}"].append(float(2 * np.mean(w)))
            out[f"crosses_zero_{tag}"].append(
                float(((g_te - w <= 0) & (0 <= g_te + w)).mean()))
            for b in SIM_BUCKETS:
                sel = (s_te >= b[0]) & (s_te < b[1])
                if sel.sum() >= 15:
                    out[f"bucket_{tag}"][b].append(float((r_te[sel] <= w[sel]).mean()))
    return out


def _main() -> None:
    import sys

    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    print(f"Split-conformal coverage at {int((1-DEFAULT_ALPHA)*100)}% nominal, "
          f"panel {panel.name} ({len(SEEDS)} seeds, scaffold-disjoint):")
    print(f"  {'isoform':7} {'coverage':>16} {'interval width (pchembl)':>26}")
    for iso in panel.isoforms:
        m = evaluate_coverage(panel, iso)
        print(f"  {iso:7} {m.coverage_mean:6.3f} ± {m.coverage_std:.3f}     "
              f"{m.width_mean:6.2f} ± {m.width_std:.2f}")

    g = evaluate_gap_coverage(panel)
    mean = lambda v: float(np.mean(v)) if len(v) else float("nan")
    print(f"\nGap interval — summed half-widths (old) vs directly calibrated (new):")
    print(f"  {'':22} {'marginal':>10} {'width':>8} {'crosses 0':>11}")
    for tag, label in (("old", "summed (old)"), ("flat", "flat calibrated"),
                       ("new", "calibrated (new)")):
        print(f"  {label:22} {mean(g[f'marginal_{tag}']):10.3f} "
              f"{mean(g[f'width_{tag}']):8.2f} {mean(g[f'crosses_zero_{tag}']):10.0%}")
    print(f"\n  coverage by nearest-neighbour similarity to training:")
    print(f"  {'bucket':22} {'old':>10} {'flat':>10} {'new':>10}")
    for b in SIM_BUCKETS:
        print(f"  [{b[0]:.2f}, {b[1]:.2f})".ljust(24)
              + f"{mean(g['bucket_old'][b]):10.3f} {mean(g['bucket_flat'][b]):10.3f} "
              f"{mean(g['bucket_new'][b]):10.3f}")


if __name__ == "__main__":
    _main()
