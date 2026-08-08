"""STEP 3 tests: seeded scaffold split + per-isoform regressor.

Uses a small synthetic dataset (no network, fast) to exercise the eval/predict
paths; the real 5-seed metrics live in VALIDATION.md.
"""
import dataclasses

import numpy as np
import pytest

from src.models import isoform_regressor as ir
from src.panels import JAK
from src.models.scaffold_split import _scaffold, scaffold_split

# A handful of scaffolds with substituent variants -> several groups to split.
_SMILES = [
    "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1", "Clc1ccccc1", "Oc1ccccc1",
    "c1ccncc1", "Cc1ccncc1", "CCc1ccncc1", "Clc1ccncc1", "Oc1ccncc1",
    "c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1", "CCc1ccc2ccccc2c1", "Oc1ccc2ccccc2c1",
    "c1ccc2[nH]ccc2c1", "Cc1ccc2[nH]ccc2c1", "CCc1ccc2[nH]ccc2c1",
]


def test_scaffold_split_seed_is_deterministic():
    a = scaffold_split(_SMILES, seed=3)
    b = scaffold_split(_SMILES, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_scaffold_split_seeds_differ_but_stay_leak_free():
    seen = set()
    for seed in range(5):
        tr, te = scaffold_split(_SMILES, seed=seed)
        assert len(tr) + len(te) == len(_SMILES)          # full cover
        assert set(tr).isdisjoint(set(te))                # no index overlap
        tr_sc = {_scaffold(_SMILES[i]) for i in tr}
        te_sc = {_scaffold(_SMILES[i]) for i in te}
        assert tr_sc.isdisjoint(te_sc)                    # no scaffold crosses
        seen.add(tuple(te))
    assert len(seen) > 1                                  # seeds explore >1 split


def _synthetic(panel, name, use_cache=True):
    import pandas as pd
    rng = np.random.default_rng(0)
    smis, pch = [], []
    for i, s in enumerate(_SMILES):
        for _ in range(4):                                # 4 replicates per scaffold-variant
            smis.append(s)
            pch.append(5.0 + (i % 5) + rng.normal(0, 0.2))
    return pd.DataFrame({"smi": smis, "pchembl": pch})


def test_evaluate_returns_finite_metrics(monkeypatch):
    monkeypatch.setattr(ir.panel_data, "build_isoform_dataset", _synthetic)
    m = ir.evaluate(JAK, "JAK1", seeds=(0, 1, 2))
    assert m.n_seeds == 3
    for v in (m.mae_mean, m.rmse_mean, m.r2_mean, m.spearman_mean,
              m.mae_std, m.rmse_std):
        assert np.isfinite(v)
    assert m.mae_mean >= 0 and m.rmse_mean >= 0


def test_train_and_cache_predicts_and_aligns(tmp_path, monkeypatch):
    panel = dataclasses.replace(JAK, root=tmp_path)
    monkeypatch.setattr(ir.panel_data, "build_isoform_dataset", _synthetic)
    bundle = ir.train_and_cache(panel, "JAK1", use_cache=False)
    assert (panel.model_cache / "JAK1_reg.pkl").exists()
    preds = bundle.predict(["c1ccccc1", "not_a_smiles"])
    assert np.isfinite(preds[0]) and np.isnan(preds[1])   # aligned, NaN on bad SMILES


def test_injected_data_is_what_gets_trained_on(tmp_path):
    """G1: the seam trains on the frame it is handed, not on the panel's own data.

    No monkeypatching — that is the point of the seam: STATE.md section 1a traced
    the undetected breakages to tests replacing module globals because the data
    could not be passed in.
    """
    panel = dataclasses.replace(JAK, root=tmp_path)
    full = _synthetic(panel, "JAK1")
    frame = full.iloc[:-8]                     # a real subset, every scaffold kept
    bundle = ir.train_and_cache(panel, "JAK1", use_cache=False, data=frame)
    assert bundle.metrics.n_molecules == len(frame) < len(full)
    assert np.isfinite(bundle.predict(["c1ccccc1"])[0])


def test_injected_run_never_writes_the_model_cache(tmp_path):
    """G1/G0: a subset-trained model must not land where the deployed one lives."""
    panel = dataclasses.replace(JAK, root=tmp_path)
    ir.train_and_cache(panel, "JAK1", use_cache=False,
                       data=_synthetic(panel, "JAK1"))
    assert not (panel.model_cache / "JAK1_reg.pkl").exists()


def test_default_path_still_loads_the_committed_jak_model():
    """G0: adding the seam must not move the deployed JAK assets.

    The default call has to keep short-circuiting to `assets/models/jak/` — if it
    ever starts retraining, the deployed model_id changes and the whole validated
    JAK story detaches from the numbers in VALIDATION.md.
    """
    bundle = ir.train_and_cache(JAK, "JAK1")
    committed = ir._load(JAK.model_bundled / "JAK1_reg.pkl")
    assert bundle.metrics == committed.metrics
    assert np.array_equal(bundle.predict(_SMILES), committed.predict(_SMILES))


def test_cache_roundtrips_via_plain_dict(tmp_path, monkeypatch):
    # The cache must load regardless of __main__/import context (plain-dict pickle).
    panel = dataclasses.replace(JAK, root=tmp_path)
    monkeypatch.setattr(ir.panel_data, "build_isoform_dataset", _synthetic)
    written = ir.train_and_cache(panel, "JAK1", use_cache=False)
    loaded = ir._load(panel.model_cache / "JAK1_reg.pkl")
    assert loaded.isoform == "JAK1"
    assert loaded.metrics.r2_mean == written.metrics.r2_mean
    assert np.isfinite(loaded.predict(["c1ccccc1"])[0])
