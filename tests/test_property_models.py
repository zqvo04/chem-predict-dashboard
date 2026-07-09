"""Phase 3b tests: generic solubility + toxicity property models."""
import numpy as np

from src.models import property_models as pm
from src import pipeline


def test_augmented_matrix_width_and_mask():
    X, mask = pm._augmented_matrix(["CCO", "not_a_smiles", "c1ccccc1"])
    assert X.shape == (2, pm.FP_SIZE + pm._N_DESCRIPTORS)
    assert list(mask) == [True, False, True]


def test_bundled_models_load_and_predict():
    models = pm.load_property_models()
    assert models is not None, "bundled property_models.pkl should be committed"

    out = models.predict(["CCO", "bad", "c1ccc(cc1)C(=O)O"])
    assert list(out.columns) == ["logS_pred", "tox_prob"]
    # ethanol is very soluble -> high logS; middle row is unparseable -> NaN
    assert out.loc[0, "logS_pred"] > -1.0
    assert np.isnan(out.loc[1, "logS_pred"]) and np.isnan(out.loc[1, "tox_prob"])
    assert ((out["tox_prob"].dropna() >= 0) & (out["tox_prob"].dropna() <= 1)).all()


def test_sol_norm_maps_to_unit_interval():
    out = pipeline._sol_norm([-7.0, -6.0, -3.5, -1.0, 0.5])
    assert out.tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]
