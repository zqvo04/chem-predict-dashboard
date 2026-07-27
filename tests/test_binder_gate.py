"""STEP 10 tests: the JAK binder gate (Tier 0.5).

Synthetic, separable positive/negative sets (no network) exercise the
evaluate/train/predict paths and the recall-set threshold; a live smoke test
self-skips unless the bundled gate is present, where it checks a trivial molecule
is rejected.
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.models import binder_gate as bg
from src.panels import JAK

# Positives and negatives that SHARE Murcko scaffolds (polar vs nonpolar terminal
# groups on the same ring system), so every scaffold group holds both classes and a
# scaffold split can never produce a single-class fold — yet the polar/nonpolar
# Morgan difference gives the classifier real, learnable signal.
_CORES = [
    "c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "c1ccc2ncccc2c1",
    "c1cccs1", "c1ccco1", "c1ccncn1", "c1cc[nH]n1", "c1ccc(-c2ccccc2)cc1",
]
_POS_SUBS = ["O=C(N)", "O=C(O)", "N", "O"]   # polar: amide, acid, amine, hydroxyl
_NEG_SUBS = ["CCCC", "CC", "C", "CCCCC"]     # nonpolar alkyl chains
_POS = [s + c for c in _CORES for s in _POS_SUBS]
_NEG = [s + c for c in _CORES for s in _NEG_SUBS]


def _patch(monkeypatch):
    monkeypatch.setattr(bg, "positive_smiles",
                        lambda panel, use_cache=True: set(_POS))
    monkeypatch.setattr(bg, "build_negatives",
                        lambda panel, use_cache=True: pd.DataFrame({"smi": _NEG}))


def test_evaluate_is_finite_and_separable(monkeypatch):
    _patch(monkeypatch)
    m = bg.evaluate(JAK, seeds=(0, 1, 2), use_cache=False)
    assert m.n_pos == len(_POS) and m.n_neg == len(_NEG)
    assert np.isfinite(m.auc_mean) and np.isfinite(m.ap_mean)
    assert m.auc_mean > 0.6                          # separable families
    assert 0.0 <= m.threshold <= 1.0                 # Youden's-J operating point
    assert 0.0 <= m.pos_recall_mean <= 1.0 and 0.0 <= m.neg_reject_mean <= 1.0


def test_train_and_cache_gates_and_aligns(tmp_path, monkeypatch):
    panel = dataclasses.replace(JAK, root=tmp_path)
    _patch(monkeypatch)
    gate = bg.train_and_cache(panel, use_cache=False)
    assert (panel.model_cache / "binder_gate.pkl").exists()

    proba = gate.predict_proba(["O=C(N)c1ccccc1", "not_a_smiles"])
    assert np.isfinite(proba[0]) and np.isnan(proba[1])          # aligned, NaN on bad SMILES

    verdict = gate.is_binder(["O=C(N)c1ccccc1", "not_a_smiles"])
    assert verdict.dtype == bool and verdict[1] == False          # bad SMILES fails closed
    # A polar (positive family) molecule scores above a nonpolar (negative family) one.
    assert gate.predict_proba(["O=C(N)c1ccccc1"])[0] > gate.predict_proba(["CCCCc1ccccc1"])[0]


def test_cache_roundtrips(tmp_path, monkeypatch):
    panel = dataclasses.replace(JAK, root=tmp_path)
    _patch(monkeypatch)
    written = bg.train_and_cache(panel, use_cache=False)
    loaded = bg._load(panel.model_cache / "binder_gate.pkl")
    assert loaded.threshold == written.threshold
    assert loaded.metrics.auc_mean == written.metrics.auc_mean


def test_live_bundled_gate_rejects_a_trivial_molecule():
    try:
        gate = bg.train_and_cache(JAK, use_cache=True)
    except Exception as err:                                       # noqa: BLE001
        pytest.skip(f"binder gate not built / data unavailable: {err}")
    if not (JAK.model_bundled / "binder_gate.pkl").exists() and \
       not (JAK.model_cache / "binder_gate.pkl").exists():
        pytest.skip("no bundled or cached binder gate to check")
    # Ethanol is not a JAK binder; the gate must not pass it.
    assert not gate.is_binder(["CCO"])[0]
