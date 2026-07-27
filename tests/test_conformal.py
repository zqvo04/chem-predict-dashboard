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


# --- difficulty-scaled gap interval (STEP 14) ------------------------------- #

_KNOTS = {"x": [0.0, 0.5, 1.0], "y": [1.0, 0.5, 0.2]}


def test_sigma_at_interpolates_and_clamps():
    assert cf.sigma_at(0.0, _KNOTS) == pytest.approx(1.0)
    assert cf.sigma_at(0.25, _KNOTS) == pytest.approx(0.75)     # linear between knots
    assert cf.sigma_at(1.0, _KNOTS) == pytest.approx(0.2)
    # outside the knot range numpy.interp clamps to the end values
    assert cf.sigma_at(-1.0, _KNOTS) == pytest.approx(1.0)
    assert cf.sigma_at(2.0, _KNOTS) == pytest.approx(0.2)


def test_sigma_is_vectorised():
    out = cf.sigma_at([0.0, 0.5, 1.0], _KNOTS)
    assert np.allclose(out, [1.0, 0.5, 0.2])


def test_gap_halfwidth_widens_as_similarity_drops(monkeypatch):
    """The whole point of the change: a molecule far from training must get a wider
    interval than one close to it. A constant width covered 92% of near molecules
    but only 46% of far ones at a 90% nominal level."""
    monkeypatch.setattr(cf, "gap_calibration",
                        lambda *a, **k: {"q": 2.0, "sigma": _KNOTS})
    far, near = cf.gap_halfwidth(0.0), cf.gap_halfwidth(1.0)
    assert far > near
    assert far == pytest.approx(2.0) and near == pytest.approx(0.4)


def test_fit_sigma_returns_usable_knots():
    """sigma must decrease with similarity when the residuals do."""
    rng = np.random.default_rng(0)
    sim = rng.uniform(0, 1, 600)
    resid = np.abs(rng.normal(0, 1.0 - 0.7 * sim))      # noisier when sim is low
    knots = cf._fit_sigma(sim, resid)
    assert len(knots["x"]) == cf.SIGMA_KNOTS == len(knots["y"])
    assert min(knots["y"]) >= cf._SIGMA_FLOOR
    assert cf.sigma_at(0.05, knots) > cf.sigma_at(0.95, knots)


def test_empirical_coverage_matches_nominal_on_synthetic():
    # Gaussian residuals: the conformal quantile of |errors| should cover ~90%.
    rng = np.random.default_rng(0)
    cal = np.abs(rng.normal(0, 1, 5000))
    test = np.abs(rng.normal(0, 1, 5000))
    q = cf.conformal_quantile(cal, alpha=0.1)
    cov = (test <= q).mean()
    assert 0.88 <= cov <= 0.92
