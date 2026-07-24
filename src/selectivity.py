"""STEP 4: selectivity gap — the shared scoring core (Stage-B Tier 1/2).

Selectivity is the pchembl **gap** between the target isoform and its worst
off-isoform:

    S(target) = pchembl_pred(target) - max_off pchembl_pred(off-isoform)

`+1` gap ~= 10x selective, `+2` ~= 100x. Two estimators (hybrid, DESIGN_DECISIONS
section 2):

  * **difference-of-regressors** — subtract the per-isoform regressors' predictions.
    Uses all per-isoform data, scores any molecule, so it drives the wide screen.
  * **direct gap regressor** — one regressor trained on the cross-measured set to
    predict the measured gap. Avoids stacked error but is data-limited, so it
    re-ranks / validates survivors.

This module is imported by both the app (Stage B) and the Colab notebook (Stage A
re-score), so the loop scores through the *same* code. Gate 4 validates predicted
gap against the *measured* gap on a scaffold-split of the cross-measured set.

CLI (evaluate both estimators against the measured gap):
    python -m src.selectivity
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .data import jak
from .models.features import morgan_matrix
from .models.scaffold_split import scaffold_split

TARGET = "JAK1"
OFFS = ("JAK2", "JAK3")
POTENCY_FLOOR = 6.0          # pchembl_pred(target) floor for the selective ranking
SELECTIVE_GAP = 1.0          # >= 10x = "selective" for enrichment
SEEDS = (0, 1, 2, 3, 4)


def selectivity_gap(preds: pd.DataFrame, target: str = TARGET,
                    offs: tuple[str, ...] = OFFS) -> np.ndarray:
    """S = pred(target) - max_off pred(off), from a frame of per-isoform predictions."""
    return preds[target].to_numpy() - preds[list(offs)].max(axis=1).to_numpy()


def _fit(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                         random_state=0).fit(X, y)


@dataclass
class SelectivityMetrics:
    target: str
    n_cross: int
    n_seeds: int
    spearman_diff_mean: float
    spearman_diff_std: float
    spearman_direct_mean: float
    spearman_direct_std: float
    enrichment_mean: float          # top-decile enrichment of >=10x-selective
    enrichment_std: float
    base_selective_rate: float      # fraction of cross-measured that are >=10x selective


def _enrichment(y_true_gap: np.ndarray, y_pred_gap: np.ndarray, frac: float = 0.1) -> float:
    """Enrichment factor of >=10x-selective molecules in the top-`frac` by predicted gap."""
    k = max(1, int(round(len(y_pred_gap) * frac)))
    top = np.argsort(y_pred_gap)[::-1][:k]
    hit = (y_true_gap >= SELECTIVE_GAP)
    base = hit.mean()
    if base == 0:
        return float("nan")
    return float(hit[top].mean() / base)


def evaluate_gap(target: str = TARGET, offs: tuple[str, ...] = OFFS,
                 seeds: tuple[int, ...] = SEEDS, use_cache: bool = True) -> SelectivityMetrics:
    """Gate 4: predicted vs measured gap on a scaffold split of the cross-measured set.

    Self-contained and leak-free: per-isoform regressors are trained only on the
    cross-measured TRAIN molecules (scaffold-disjoint from test), so the gap is
    never evaluated on training scaffolds. This under-uses data vs the deployed
    models (which train on each isoform's full set), so it is a conservative
    estimate of deployed selectivity performance.
    """
    cross = jak.build_cross_measured(use_cache=use_cache)
    smiles = cross["smi"].tolist()
    X, mask = morgan_matrix(smiles)
    cross = cross[mask].reset_index(drop=True)
    kept = [s for s, keep in zip(smiles, mask) if keep]
    isoforms = [target, *offs]
    measured_gap = (cross[target].to_numpy()
                    - cross[list(offs)].max(axis=1).to_numpy())

    sp_diff, sp_direct, enrich = [], [], []
    for seed in seeds:
        tr, te = scaffold_split(kept, test_frac=0.2, seed=seed)
        # difference-of-regressors: one regressor per isoform on the train split
        pred_te = {iso: _fit(X[tr], cross[iso].to_numpy()[tr]).predict(X[te])
                   for iso in isoforms}
        pred_gap = pred_te[target] - np.maximum.reduce([pred_te[o] for o in offs])
        # direct gap regressor
        direct_gap = _fit(X[tr], measured_gap[tr]).predict(X[te])

        gap_te = measured_gap[te]
        sp_diff.append(spearmanr(gap_te, pred_gap).statistic)
        sp_direct.append(spearmanr(gap_te, direct_gap).statistic)
        enrich.append(_enrichment(gap_te, pred_gap))

    return SelectivityMetrics(
        target=target, n_cross=len(kept), n_seeds=len(seeds),
        spearman_diff_mean=float(np.mean(sp_diff)), spearman_diff_std=float(np.std(sp_diff)),
        spearman_direct_mean=float(np.mean(sp_direct)), spearman_direct_std=float(np.std(sp_direct)),
        enrichment_mean=float(np.nanmean(enrich)), enrichment_std=float(np.nanstd(enrich)),
        base_selective_rate=float((measured_gap >= SELECTIVE_GAP).mean()),
    )


def _main() -> None:
    m = evaluate_gap()
    print(f"Selectivity for {m.target} over {'/'.join(OFFS)}  "
          f"(cross-measured n={m.n_cross}, {m.n_seeds} seeds)")
    print(f"  Spearman (difference-of-regressors): {m.spearman_diff_mean:.3f} ± {m.spearman_diff_std:.3f}")
    print(f"  Spearman (direct gap regressor)    : {m.spearman_direct_mean:.3f} ± {m.spearman_direct_std:.3f}")
    print(f"  Top-decile enrichment (>=10x sel.) : {m.enrichment_mean:.2f} ± {m.enrichment_std:.2f}x  "
          f"(base rate {m.base_selective_rate:.1%})")


if __name__ == "__main__":
    _main()
