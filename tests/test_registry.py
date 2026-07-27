"""STEP 15 tests: the run registry (append-only round history) and Campaign.

All offline. The registry root is redirected to tmp_path, so nothing here touches
the developer's real `data/registry/`.
"""
import json

import pandas as pd
import pytest

from src import campaign as camp
from src import registry as reg
from src.panels import JAK, PI3K


def test_rounds_are_empty_before_anything_happens(tmp_path):
    assert reg.rounds("nothing-yet", root=tmp_path) == []
    assert reg.list_campaigns(root=tmp_path) == []


def test_append_round_assigns_increasing_indices(tmp_path):
    for i in range(3):
        r = reg.append_round("c1", kind="screen", model_ids={"JAK1": "x@1"},
                             n_molecules=10 + i, root=tmp_path)
        assert r.index == i
    got = reg.rounds("c1", root=tmp_path)
    assert [r.index for r in got] == [0, 1, 2]
    assert [r.n_molecules for r in got] == [10, 11, 12]


def test_index_is_derived_from_disk_not_memory(tmp_path):
    """Two independent callers must produce two rounds, not one clobbering the other."""
    reg.append_round("c1", kind="screen", model_ids={}, n_molecules=1, root=tmp_path)
    reg.append_round("c1", kind="select", model_ids={}, n_molecules=2, root=tmp_path)
    assert len(reg.rounds("c1", root=tmp_path)) == 2


def test_scores_round_trip_through_parquet(tmp_path):
    df = pd.DataFrame({"smi": ["CCO", "c1ccccc1"], "gap": [1.0, 2.0]})
    r = reg.append_round("c1", kind="screen", model_ids={}, n_molecules=2,
                         scores=df, root=tmp_path)
    assert r.scores_path is not None
    back = reg.round_scores("c1", 0, root=tmp_path)
    pd.testing.assert_frame_equal(df, back)


def test_round_without_scores_reports_none(tmp_path):
    reg.append_round("c1", kind="select", model_ids={}, n_molecules=1, root=tmp_path)
    assert reg.round_scores("c1", 0, root=tmp_path) is None
    with pytest.raises(KeyError):
        reg.round_scores("c1", 7, root=tmp_path)


def test_a_torn_final_line_costs_one_round_not_the_file(tmp_path):
    """A crash mid-append must not make the earlier history unreadable."""
    reg.append_round("c1", kind="screen", model_ids={}, n_molecules=1, root=tmp_path)
    reg.append_round("c1", kind="screen", model_ids={}, n_molecules=2, root=tmp_path)
    path = reg.campaign_dir("c1", tmp_path) / reg.ROUNDS_FILE
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"index": 2, "kind": "scr')          # torn write
    got = reg.rounds("c1", root=tmp_path)
    assert [r.index for r in got] == [0, 1]


@pytest.mark.parametrize("bad", ["../escape", "/abs", "", "a/b", "."])
def test_campaign_ids_cannot_escape_the_registry(bad, tmp_path):
    with pytest.raises(ValueError, match="Invalid campaign id"):
        reg.campaign_dir(bad, tmp_path)


# --- Campaign + validation tiers ------------------------------------------- #

def test_validated_tier_is_reserved_for_panels_with_committed_evidence():
    v = camp.assess(JAK, n_cross_measured=3624)
    assert v.tier == camp.VALIDATED
    assert v.supports_selectivity_claim


def test_a_new_panel_with_plenty_of_data_is_bootstrap_not_validated():
    """Data volume must never promote a panel to 'validated' — only measurement does."""
    v = camp.assess(PI3K, n_cross_measured=5000)
    assert v.tier == camp.BOOTSTRAP
    assert v.supports_selectivity_claim
    assert "hypothesis" in v.reason


def test_too_little_cross_measured_data_blocks_the_selectivity_claim():
    v = camp.assess(PI3K, n_cross_measured=camp.MIN_CROSS_MEASURED - 1)
    assert v.tier == camp.INSUFFICIENT
    assert not v.supports_selectivity_claim


def test_campaign_round_trips_through_json(tmp_path):
    c = camp.Campaign(campaign_id="jak-jak1", panel_name="jak", library="wide",
                      model_ids={"JAK1": "CHEMBL2835@abc"}, conformal_alpha=0.10,
                      validation=camp.assess(JAK, 3624), code_version="deadbee",
                      created="2026-07-27T00:00:00+00:00")
    camp.save(c, root=tmp_path)
    back = camp.load("jak-jak1", root=tmp_path)
    assert back.to_dict() == c.to_dict()
    assert back.panel is JAK
    assert reg.list_campaigns(root=tmp_path) == ["jak-jak1"]


def test_saved_campaign_is_readable_json(tmp_path):
    c = camp.Campaign(campaign_id="c1", panel_name="jak", library="wide",
                      model_ids={}, conformal_alpha=0.10,
                      validation=camp.assess(JAK, 3624))
    path = camp.save(c, root=tmp_path)
    d = json.loads(path.read_text())
    assert d["validation"]["tier"] == camp.VALIDATED


# --- the generalisation must not have moved the validated panel ------------- #

def test_jak_model_ids_match_the_committed_contracts():
    """STEP 15's whole constraint: parameterising the code by panel must leave the
    validated JAK screen bit-identical. Model ids are behavioural digests, so if any
    JAK model changed, this fails — and every contract ever exported would stop
    validating. Skips when the bundled models are absent (no assets in the checkout).
    """
    from src.funnel import current_model_ids

    if not all((JAK.model_bundled / f"{iso}_reg.pkl").exists() for iso in JAK.isoforms):
        pytest.skip("bundled JAK models not present")
    assert current_model_ids(JAK) == {
        "JAK1": "CHEMBL2835@61edd879da",
        "JAK2": "CHEMBL2971@5649ee4718",
        "JAK3": "CHEMBL2148@0772985d8c",
    }


def test_registered_panels_are_disjoint_from_the_gate_and_library():
    """Repeat of the STEP 10 / STEP 13 leakage lesson, as a guard rather than a note:
    a panel sharing a target with the binder gate's negatives would train the gate to
    reject its own actives, and one sharing with the library would screen a haystack
    built from its own answers."""
    from src.panels import PANELS, disjointness_report

    for name, panel in PANELS.items():
        report = disjointness_report(panel)
        assert report["gate_negative_conflicts"] == [], name
        assert report["library_conflicts"] == [], name


def test_a_panel_needs_off_targets():
    """A single target has no gap to be selective by — that is Mode 1's job, not the
    funnel's, and the spec refuses it rather than producing a meaningless zero."""
    import dataclasses

    with pytest.raises(ValueError, match="no off-targets"):
        dataclasses.replace(JAK, offs=())


def test_a_panel_needs_a_chembl_id_for_every_member():
    import dataclasses

    with pytest.raises(ValueError, match="no ChEMBL id"):
        dataclasses.replace(JAK, offs=("JAK2", "JAK9"))
