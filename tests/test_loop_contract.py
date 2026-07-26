"""STEP 7 tests: loop-contract build / round-trip / model-pin guard (offline)."""
from pathlib import Path

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


# --- Colab handoff -------------------------------------------------------- #

def test_colab_url_pins_the_contracts_own_code_version(monkeypatch):
    """The point of the link is not the saved click: pinning it to the contract's
    commit means the notebook Colab opens is the one the contract was exported
    from, so assert_models_match passes by construction."""
    monkeypatch.setattr(lc, "repo_slug", lambda: "owner/repo")
    url = lc.colab_url({"provenance": {"code_version": "abc1234"}})
    assert url == ("https://colab.research.google.com/github/owner/repo/"
                   "blob/abc1234/notebooks/deep_dive.ipynb")


def test_colab_url_is_none_without_a_remote_or_a_known_commit(monkeypatch):
    monkeypatch.setattr(lc, "repo_slug", lambda: None)
    assert lc.colab_url({"provenance": {"code_version": "abc1234"}}) is None
    monkeypatch.setattr(lc, "repo_slug", lambda: "owner/repo")
    assert lc.colab_url({"provenance": {"code_version": "unknown"}}) is None


@pytest.mark.parametrize("remote", [
    "https://github.com/owner/repo.git",
    "https://github.com/owner/repo",
    "git@github.com:owner/repo.git",
    "http://proxy@127.0.0.1:8080/git/owner/repo",
])
def test_repo_slug_parses_the_usual_remote_spellings(monkeypatch, remote):
    monkeypatch.setattr(lc, "_git", lambda *a: remote)
    assert lc.repo_slug() == "owner/repo"


def test_commit_on_remote_sees_a_pushed_commit(monkeypatch):
    """The Colab link and the notebook's pinned checkout both resolve through
    GitHub, so an unpushed commit is a dead handoff — the app has to be able to
    say so before the user lands in Colab."""
    monkeypatch.setattr(lc, "_git", lambda *a: "refs/remotes/origin/main")
    assert lc.commit_on_remote("abc1234") is True


def test_commit_on_remote_reports_an_unpushed_or_unknowable_commit(monkeypatch):
    monkeypatch.setattr(lc, "_git", lambda *a: "")          # contained by no remote branch
    assert lc.commit_on_remote("abc1234") is False
    monkeypatch.setattr(lc, "_git", lambda *a: None)        # unknown sha / no git
    assert lc.commit_on_remote("abc1234") is None
    assert lc.commit_on_remote("unknown") is None


# --- Contract validation (the Colab upload lands here) --------------------- #

def _valid():
    ids = {"JAK1": "CHEMBL2835@abc", "JAK2": "CHEMBL2971@def", "JAK3": "CHEMBL2148@ghi"}
    return lc.build_contract(_shortlist(), "JAK1", ["JAK2", "JAK3"], ids, alpha=0.10)


def test_validate_accepts_a_contract_this_code_built():
    lc.validate_contract(_valid())


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c.pop("molecules"), "missing molecules"),
    (lambda c: c["provenance"].pop("model_ids"), "missing model_ids"),
    (lambda c: c.update(schema_version="2.0"), "incompatible"),
    (lambda c: c.update(molecules=[]), "no molecules"),
    (lambda c: c["molecules"][0].update(smiles=""), "no SMILES"),
])
def test_validate_names_what_is_wrong(mutate, message):
    """A hand-carried file arrives wrong in plausible ways — the wrong JSON, a
    future schema, an empty selection. Each has to fail by name, not as a KeyError
    inside the scoring code half an hour into a Colab session."""
    c = _valid()
    mutate(c)
    with pytest.raises(ValueError, match=message):
        lc.validate_contract(c)


def test_validate_rejects_json_that_is_not_a_contract():
    with pytest.raises(ValueError, match="expected a JSON object"):
        lc.validate_contract([1, 2, 3])


def test_colab_url_points_at_a_notebook_that_actually_exists():
    """The URL is built from NOTEBOOK_PATH; if the notebook moves, the link 404s."""
    assert (Path(__file__).resolve().parents[1] / lc.NOTEBOOK_PATH).exists()


# --- Deployment fallbacks (the web app has no git) ------------------------- #

def test_code_version_and_slug_fall_back_to_the_deployment_env(monkeypatch):
    """The container serving the dashboard has neither git nor .git, so without
    this the commit reads 'unknown' on the web and the Colab handoff disappears
    exactly where most people meet the funnel."""
    monkeypatch.setattr(lc, "_git", lambda *a: None)
    monkeypatch.setenv("RENDER_GIT_COMMIT", "ab1a969" + "f" * 33)   # Render gives the full sha
    monkeypatch.setenv("RENDER_GIT_REPO_SLUG", "owner/repo")
    assert lc.code_version() == "ab1a969"
    assert lc.repo_slug() == "owner/repo"
    assert lc.colab_url({"provenance": {"code_version": lc.code_version()}}) == (
        "https://colab.research.google.com/github/owner/repo/"
        "blob/ab1a969/notebooks/deep_dive.ipynb")


def test_git_wins_over_the_env_and_a_repo_url_still_parses(monkeypatch):
    monkeypatch.setenv("CHEM_PREDICT_COMMIT", "eeeeeee")
    monkeypatch.setattr(lc, "_git", lambda *a: "abc1234")
    assert lc.code_version() == "abc1234"
    monkeypatch.setattr(lc, "_git", lambda *a: None)
    monkeypatch.setenv("CHEM_PREDICT_REPO", "https://github.com/owner/fork.git")
    assert lc.code_version() == "eeeeeee" and lc.repo_slug() == "owner/fork"


def test_no_git_and_no_env_stays_honest(monkeypatch):
    """No silent fallback to a branch name: a contract that cannot name its commit
    cannot be pinned, and the notebook refuses to run against an unpinned checkout."""
    monkeypatch.setattr(lc, "_git", lambda *a: None)
    for name in ("CHEM_PREDICT_COMMIT", "RENDER_GIT_COMMIT",
                 "CHEM_PREDICT_REPO", "RENDER_GIT_REPO_SLUG"):
        monkeypatch.delenv(name, raising=False)
    assert lc.code_version() == "unknown"
    assert lc.repo_slug() is None
    assert lc.colab_url({"provenance": {"code_version": "unknown"}}) is None
