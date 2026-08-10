"""A2: EV = P(binder) * regressor + (1 - P) * floor.

모델을 학습시키지 않는다. 검증 대상은 산술과 정렬이고, 그 둘이 틀리면 감사 결과 전체가
틀린다. 학습된 모델의 성능은 scripts/two_part_audit.py가 잰다.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.two_part import CENSORED_FLOOR, TwoPart
from src.panels import DEFAULT_PANEL


class _ConstGate:
    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, X):
        p = np.full(X.shape[0], self.p)
        return np.column_stack([1.0 - p, p])


class _ConstReg:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X):
        return np.full(X.shape[0], self.value)


def _model(p: float, values: dict[str, float]) -> TwoPart:
    return TwoPart(gate=_ConstGate(p),
                   regressors={iso: _ConstReg(v) for iso, v in values.items()})


def test_expected_potency_interpolates_between_regressor_and_floor():
    t = DEFAULT_PANEL.target
    model = _model(0.25, {iso: 6.4 for iso in DEFAULT_PANEL.isoforms})
    ev = model.expected_potency(["CCO"])

    expected = 0.25 * 6.4 + 0.75 * CENSORED_FLOOR
    assert ev[t][0] == pytest.approx(expected)


def test_expected_gap_cancels_the_floor():
    t, o1, o2 = DEFAULT_PANEL.isoforms
    model = _model(0.4, {t: 8.0, o1: 6.0, o2: 5.0})
    gap = model.expected_gap(["CCO"], DEFAULT_PANEL)

    # 0.4 * (8.0 - max(6.0, 5.0)); the floor appears in every isoform and cancels
    assert gap[0] == pytest.approx(0.4 * 2.0)


def test_a_certain_binder_reduces_to_the_regressor():
    t = DEFAULT_PANEL.target
    model = _model(1.0, {iso: 7.3 for iso in DEFAULT_PANEL.isoforms})
    assert model.expected_potency(["CCO"])[t][0] == pytest.approx(7.3)


def test_unparseable_smiles_stay_nan_and_keep_alignment():
    t = DEFAULT_PANEL.target
    model = _model(0.5, {iso: 6.0 for iso in DEFAULT_PANEL.isoforms})
    ev = model.expected_potency(["CCO", "not a molecule", "c1ccccc1"])

    assert len(ev[t]) == 3
    assert np.isnan(ev[t][1])
    assert not np.isnan(ev[t][0]) and not np.isnan(ev[t][2])
