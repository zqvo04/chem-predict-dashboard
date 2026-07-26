"""STEP 8 tests: loop closure logic (offline, scoring mocked)."""
import json
from pathlib import Path

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


def test_a_malformed_contract_fails_before_any_scoring(monkeypatch):
    """The contract is hand-carried into Colab, so the wrong file is a normal
    accident. It has to fail by name, not as a KeyError inside the scoring."""
    _patch(monkeypatch)
    c = _contract()
    c["molecules"] = []
    with pytest.raises(ValueError, match="no molecules"):
        deep_dive.run_deep_dive(c)


# --- The Colab handoff (notebooks/deep_dive.ipynb) ------------------------- #

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "deep_dive.ipynb"


def _code_cells():
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_reads_the_contract_before_it_clones():
    """The contract names the commit the clone is pinned to, so it has to be read
    first — a clone-then-upload order cannot pin anything."""
    cells = _code_cells()
    upload = next(i for i, s in enumerate(cells) if "files.upload()" in s)
    clone = next(i for i, s in enumerate(cells) if "git clone" in s)
    assert upload < clone


def test_notebook_pins_the_checkout_to_the_contracts_code_version():
    """Cloning the default branch would re-score through whatever HEAD is today:
    the models are pickles in the repo, so that is a different experiment from the
    one the contract describes. The pin — and its hard failure — is the handoff."""
    cells = _code_cells()
    assert 'ref = prov["code_version"]' in "".join(cells)
    clone = next(s for s in cells if "git clone" in s)
    assert '"git", "checkout", "-q", ref' in clone
    assert "raise SystemExit" in clone      # never fall through to the default branch


def test_notebook_returns_the_rescore_contract():
    """The loop is a cycle: Stage A has to hand its A_rescore contract back."""
    cells = _code_cells()
    assert any("write_contract(a_contract" in s and "files.download" in s for s in cells)
