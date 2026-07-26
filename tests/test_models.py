"""Phase 3 tests: featurization, scaffold split, and target model training."""
import numpy as np
import pytest

from src.models.features import FP_DTYPE, FP_SIZE, iter_morgan_batches, morgan_matrix
from src.models.scaffold_split import scaffold_split, _scaffold
from src.models import target_model as tm


def test_morgan_matrix_shape_and_mask():
    smiles = ["CCO", "not_a_smiles", "c1ccccc1"]
    X, mask = morgan_matrix(smiles)
    assert X.shape == (2, FP_SIZE)          # only the 2 valid ones featurized
    assert list(mask) == [True, False, True]
    assert set(np.unique(X)).issubset({0.0, 1.0})


def test_morgan_matrix_is_uint8():
    """Fingerprints are single bits; float64 spent 8 bytes each and was what pushed
    the deployed funnel over its memory limit. uint8 is numerically inert here."""
    X, _ = morgan_matrix(["CCO", "c1ccccc1"])
    assert X.dtype == FP_DTYPE
    assert morgan_matrix([])[0].dtype == FP_DTYPE          # empty case keeps the dtype


def test_iter_morgan_batches_matches_the_whole_matrix():
    """Batched featurisation must be indistinguishable from doing it in one go —
    that equivalence is the whole basis for streaming the wide screen."""
    smiles = ["CCO", "c1ccccc1", "not_a_smiles", "CCN", "c1ccncc1", "bad", "CCCC"]
    X_all, mask_all = morgan_matrix(smiles)

    parts = list(iter_morgan_batches(smiles, batch_size=2))
    assert len(parts) == 4                                  # 7 inputs / batch 2
    X_batched = np.concatenate([X for X, _ in parts if X.shape[0]])
    mask_batched = np.concatenate([m for _, m in parts])

    assert np.array_equal(mask_batched, mask_all)
    assert np.array_equal(X_batched, X_all)


def test_iter_morgan_batches_handles_an_all_invalid_batch():
    parts = list(iter_morgan_batches(["bad", "worse"], batch_size=2))
    X, mask = parts[0]
    assert X.shape == (0, FP_SIZE) and not mask.any()


def test_scaffold_split_no_scaffold_crosses():
    smiles = [
        "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1",        # benzene scaffold
        "c1ccncc1", "Cc1ccncc1",                       # pyridine scaffold
        "c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1",           # naphthalene scaffold
    ]
    train_idx, test_idx = scaffold_split(smiles, test_frac=0.3)

    assert len(train_idx) + len(test_idx) == len(smiles)
    assert set(train_idx).isdisjoint(set(test_idx))
    train_scaffolds = {_scaffold(smiles[i]) for i in train_idx}
    test_scaffolds = {_scaffold(smiles[i]) for i in test_idx}
    assert train_scaffolds.isdisjoint(test_scaffolds)


def test_training_gate_rejects_tiny_targets(monkeypatch):
    monkeypatch.setattr(tm.cc, "resolve_target",
                        lambda q, organism=None: tm.cc.Target("CHEMBLX", "x", "", "SINGLE PROTEIN", 0.0))
    monkeypatch.setattr(tm, "build_training_set",
                        lambda tid, max_records=4000: __import__("pandas").DataFrame(
                            {"molecule_chembl_id": ["a"], "canonical_smiles": ["CCO"],
                             "pchembl_value": [7.0]}))
    with pytest.raises(ValueError, match="not enough"):
        tm.train_target_model("whatever", use_cache=False)


def test_live_train_egfr():
    try:
        model = tm.train_target_model("EGFR", max_records=800, use_cache=False)
    except RuntimeError as err:
        pytest.skip(f"ChEMBL unreachable: {err}")

    assert model.target.chembl_id == "CHEMBL203"
    assert model.metrics.n_molecules >= tm.MIN_TRAIN_MOLECULES
    assert np.isfinite(model.metrics.r2) and np.isfinite(model.metrics.rmse)
    # a per-target Morgan-RF regressor should carry real signal on EGFR
    assert model.metrics.r2 > 0.2

    preds = model.predict(["COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC", "bad_smiles"])
    assert np.isfinite(preds[0]) and np.isnan(preds[1])
