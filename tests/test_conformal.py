"""STEP 5 tests: conformal quantile + interval math (offline, deterministic)."""
import numpy as np
import pytest

from src import conformal as cf


def test_conformal_quantile_finite_sample_correction():
    # residuals 1..10; at alpha=0.1, level = ceil(11*0.9)/10 = 10/10 = 1.0 -> max = 10.
    res = np.arange(1, 11, dtype=float)
    assert cf.conformal_quantile(res, alpha=0.1) == 10.0
    # at alpha=0.3, level = ceil(11*0.7)/10 = 8/10 = 0.8 -> the 0.8 'higher' quantile.
    q = cf.conformal_quantile(res, alpha=0.3)
    assert q == np.quantile(res, 0.8, method="higher")


def test_conformal_quantile_caps_level_at_one():
    # tiny calibration set can't reach high nominal exactly; level caps at 1.0 (max).
    res = np.array([2.0, 4.0, 6.0])
    assert cf.conformal_quantile(res, alpha=0.05) == 6.0


def test_predict_interval_is_symmetric():
    pred = np.array([7.0, 5.0])
    lo, hi = cf.predict_interval(pred, q=0.8)
    assert np.allclose(lo, [6.2, 4.2]) and np.allclose(hi, [7.8, 5.8])


def test_gap_interval_adds_halfwidths():
    assert cf.gap_interval(0.6, 0.7) == pytest.approx(1.3)


def test_empirical_coverage_matches_nominal_on_synthetic():
    # Gaussian residuals: the conformal quantile of |errors| should cover ~90%.
    rng = np.random.default_rng(0)
    cal = np.abs(rng.normal(0, 1, 5000))
    test = np.abs(rng.normal(0, 1, 5000))
    q = cf.conformal_quantile(cal, alpha=0.1)
    cov = (test <= q).mean()
    assert 0.88 <= cov <= 0.92
