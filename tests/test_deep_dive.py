"""STEP 8 tests: loop closure logic (offline, scoring mocked)."""
import pandas as pd
import pytest

from src import deep_dive
from src.loop_contract import build_contract


IDS = {"JAK1": "CHEMBL2835@a", "JAK2": "CHEMBL2971@b", "JAK3": "CHEMBL2148@c"}


def _scored(smis):
    # minimal frame matching what funnel.score_molecules returns
    n = len(smis)
    d = {"smi": list(smis), "gap": [1.5] * n, "gap_lo": [0.0] * n, "gap_hi": [3.0] * n,
         "meets_floor": [True] * n, "in_domain": [True] * n, "verdict": ["in_domain"] * n}
    for iso in ("JAK1", "JAK2", "JAK3"):
        d[f"pred_{iso}"] = [7.0] * n
        d[f"lo_{iso}"] = [6.0] * n
        d[f"hi_{iso}"] = [8.0] * n
        d[f"in_domain_{iso}"] = [True] * n
    return pd.DataFrame(d)


def _contract():
    picks = _scored(["c1ccccc1C(=O)O"])
    return build_contract(picks, "JAK1", ["JAK2", "JAK3"], IDS, alpha=0.10)


def _patch(monkeypatch):
    monkeypatch.setattr(deep_dive.funnel, "current_model_ids", lambda *a, **k: IDS)
    monkeypatch.setattr(deep_dive.funnel, "score_molecules",
                        lambda smis, *a, **k: _scored(smis))


def test_run_deep_dive_generates_and_rescoring_tags_origin(monkeypatch):
    _patch(monkeypatch)
    res = deep_dive.run_deep_dive(_contract(), max_analogues_per_case=8)
    assert (res.before["origin"] == "screen").all()
    assert (res.after["origin"] == "generated").all()
    assert res.after["parent_smiles"].notna().all()     # analogues linked to a parent
    assert len(res.after) > 0


def test_model_mismatch_blocks_rescoring(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(deep_dive.funnel, "current_model_ids",
                        lambda *a, **k: {**IDS, "JAK1": "CHEMBL2835@CHANGED"})
    with pytest.raises(ValueError, match="Model mismatch"):
        deep_dive.run_deep_dive(_contract())


def test_rescore_contract_is_stage_a(monkeypatch):
    _patch(monkeypatch)
    res = deep_dive.run_deep_dive(_contract(), max_analogues_per_case=5)
    a = deep_dive.rescore_contract(res)
    assert a["provenance"]["stage"] == "A_rescore"
    assert all(m["origin"] == "generated" for m in a["molecules"])


def test_report_labels_hypothesis(monkeypatch):
    _patch(monkeypatch)
    res = deep_dive.run_deep_dive(_contract(), max_analogues_per_case=5)
    report = deep_dive.report_markdown(res)
    assert "in-silico hypothesis" in report.lower()
    assert "before" in report and "after" in report
