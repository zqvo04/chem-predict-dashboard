"""G10: what a committed Program card is allowed to claim.

An anti-target needs evidence of class 1-3 — an approved-drug safety label, a
mechanism, or human loss-of-function genetics. Class 4 is structural similarity,
which says "it could bind", not "binding it is bad". STATE.md section 7's gaps 7b
and 7c are both that mistake, so the rule is checked rather than remembered.
"""
import json
from pathlib import Path

import pytest

from scripts.make_cases_index import violations

CASES = Path(__file__).resolve().parents[1] / "assets" / "cases"


def _cards():
    return [json.loads(p.read_text()) for p in sorted(CASES.glob("*/panel.json"))]


def test_every_committed_card_satisfies_g10():
    cards = _cards()
    assert cards, "no committed Program cards to check"
    for card in cards:
        assert violations(card) == []


def test_a_similarity_only_registration_is_rejected():
    card = {"case_id": "x", "offs": ["JAK2"],
            "anti_target_evidence": {"JAK2": {"class": [4], "registered": True}}}
    assert any("G10 requires" in v for v in violations(card))


def test_a_similarity_only_watch_item_is_allowed():
    card = {"case_id": "x", "offs": ["JAK3"],
            "anti_target_evidence": {"JAK3": {"class": [4], "registered": False,
                                              "watch": True}}}
    assert violations(card) == []


def test_an_off_target_with_no_evidence_entry_is_rejected():
    card = {"case_id": "x", "offs": ["JAK2", "TYK2"],
            "anti_target_evidence": {"JAK2": {"class": [1], "registered": True}}}
    assert any("no evidence entry" in v for v in violations(card))


def test_the_index_refuses_to_write_on_a_violation(tmp_path, monkeypatch):
    import scripts.make_cases_index as index

    bad = tmp_path / "bad-case"
    bad.mkdir()
    (bad / "panel.json").write_text(json.dumps(
        {"case_id": "bad", "panel": "jak", "indication": "x", "offs": ["JAK2"],
         "anti_target_evidence": {"JAK2": {"class": [4], "registered": True}}}))
    monkeypatch.setattr(index, "CASES", tmp_path)
    monkeypatch.setattr(index, "ROOT", tmp_path)
    with pytest.raises(SystemExit):
        index.main()
    assert not (tmp_path / "CASES.md").exists()
