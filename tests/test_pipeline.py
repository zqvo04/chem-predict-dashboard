"""Phase 4 tests: composite scoring helpers and the full screen."""
import numpy as np
import pytest

from src import pipeline


def test_activity_norm_clips_to_unit_interval():
    out = pipeline._activity_norm([4.0, 5.0, 7.5, 10.0, 12.0])
    assert out.tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]


def test_canonical_and_qed():
    assert pipeline._canonical("C(C)O") == "CCO"      # canonicalized
    assert pipeline._canonical("nonsense") is None
    assert 0.0 <= pipeline._qed("CCO") <= 1.0


def test_live_screen_egfr():
    try:
        target, model, scored = pipeline.screen("EGFR", max_records=4000, use_cache=True)
    except RuntimeError as err:
        pytest.skip(f"ChEMBL/PubChem unreachable: {err}")

    assert target.chembl_id == "CHEMBL203"
    # only drug-like molecules survive, scores are valid and sorted
    assert scored["druglike"].all()
    assert ((scored["composite"] >= 0) & (scored["composite"] <= 1)).all()
    assert scored["composite"].is_monotonic_decreasing

    known = scored[scored["source"] == "chembl_known"]
    novel = scored[scored["source"] == "pubchem_novel"]
    assert len(known) > 0
    # known scored on measured potency, novel purely on the model prediction
    assert known["measured_pchembl"].notna().all()
    if len(novel):
        assert novel["measured_pchembl"].isna().all()
        assert novel["pred_pchembl"].notna().all()
