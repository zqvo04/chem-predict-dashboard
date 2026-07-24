"""STEP 4 tests: selectivity gap math + enrichment (offline)."""
import numpy as np
import pandas as pd

from src import selectivity as sel


def test_gap_is_target_minus_worst_off():
    preds = pd.DataFrame({"JAK1": [8.0, 6.0], "JAK2": [6.0, 6.5], "JAK3": [5.0, 7.0]})
    gap = sel.selectivity_gap(preds, target="JAK1", offs=("JAK2", "JAK3"))
    # row0: 8 - max(6,5) = 2 ; row1: 6 - max(6.5,7) = -1
    assert np.allclose(gap, [2.0, -1.0])


def test_gap_uses_worst_not_mean_off():
    preds = pd.DataFrame({"JAK1": [8.0], "JAK2": [4.0], "JAK3": [7.9]})
    # worst off is JAK3 (7.9), so gap is 0.1 — mean would wrongly give ~2.05
    assert np.isclose(sel.selectivity_gap(preds)[0], 0.1)


def test_enrichment_perfect_ranking():
    # predicted gap perfectly orders molecules; top decile is all selective.
    true_gap = np.array([2.0, 1.5, 1.2] + [0.0] * 27)   # 3/30 selective, base 10%
    pred_gap = true_gap.copy()
    ef = sel._enrichment(true_gap, pred_gap, frac=0.1)   # top 3 -> all 3 hits
    assert ef == 10.0                                     # precision 1.0 / base 0.1


def test_enrichment_random_ranking_is_about_one():
    rng = np.random.default_rng(0)
    true_gap = (rng.random(2000) < 0.1).astype(float) * 2.0   # ~10% selective
    pred_gap = rng.random(2000)                                # unrelated
    ef = sel._enrichment(true_gap, pred_gap, frac=0.1)
    assert 0.5 < ef < 1.7                                       # near 1, no signal
