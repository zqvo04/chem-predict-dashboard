"""STEP 7 tests: loop-contract build / round-trip / model-pin guard (offline)."""
import pandas as pd
import pytest

from src import loop_contract as lc


def _shortlist():
    return pd.DataFrame([{
        "smi": "c1ccccc1", "origin": "screen", "parent_smiles": None,
        "pred_JAK1": 8.4, "lo_JAK1": 7.3, "hi_JAK1": 9.5, "in_domain_JAK1": True,
        "pred_JAK2": 6.7, "lo_JAK2": 5.5, "hi_JAK2": 7.9, "in_domain_JAK2": True,
        "pred_JAK3": 6.1, "lo_JAK3": 4.9, "hi_JAK3": 7.3, "in_domain_JAK3": True,
        "gap": 1.7, "gap_lo": 0.4, "gap_hi": 3.0, "meets_floor": True, "verdict": "in_domain",
    }])


def test_build_contract_shape():
    ids = {"JAK1": "CHEMBL2835@abc", "JAK2": "CHEMBL2971@def", "JAK3": "CHEMBL2148@ghi"}
    c = lc.build_contract(_shortlist(), "JAK1", ["JAK2", "JAK3"], ids, alpha=0.10)
    assert c["schema_version"] == lc.SCHEMA_VERSION
    assert c["target_isoform"] == "JAK1" and c["off_isoforms"] == ["JAK2", "JAK3"]
    assert c["provenance"]["model_ids"] == ids
    m = c["molecules"][0]
    assert m["smiles"] == "c1ccccc1"
    assert m["selectivity"]["gap"] == 1.7
    assert m["per_isoform"]["JAK1"]["interval"] == [7.3, 9.5]
    assert m["deep_dive"] is None


def test_contract_round_trip(tmp_path):
    ids = {"JAK1": "CHEMBL2835@abc", "JAK2": "CHEMBL2971@def", "JAK3": "CHEMBL2148@ghi"}
    c = lc.build_contract(_shortlist(), "JAK1", ["JAK2", "JAK3"], ids, alpha=0.10)
    p = tmp_path / "case.json"
    lc.write_contract(c, p)
    assert lc.read_contract(p) == c


def test_assert_models_match_guards_stage_a():
    ids = {"JAK1": "CHEMBL2835@abc", "JAK2": "CHEMBL2971@def", "JAK3": "CHEMBL2148@ghi"}
    c = lc.build_contract(_shortlist(), "JAK1", ["JAK2", "JAK3"], ids, alpha=0.10)
    lc.assert_models_match(c, ids)                       # identical -> ok
    with pytest.raises(ValueError, match="Model mismatch"):
        lc.assert_models_match(c, {**ids, "JAK1": "CHEMBL2835@CHANGED"})


def test_model_id_is_stable_and_content_addressed():
    from sklearn.dummy import DummyRegressor
    import numpy as np
    m = DummyRegressor().fit(np.zeros((3, 2)), [1, 2, 3])
    a, b = lc.model_id("CHEMBL2835", m), lc.model_id("CHEMBL2835", m)
    assert a == b and a.startswith("CHEMBL2835@")
