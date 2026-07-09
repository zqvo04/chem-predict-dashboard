"""Phase 1 tests for the ChEMBL client.

Unit tests run fully offline (network is monkeypatched / not touched).
The live smoke test skips itself if the API is unreachable.
"""
import pandas as pd
import pytest
from rdkit import Chem

from src.data import chembl_client as cc


def test_to_candidates_dedup_validation_and_sort():
    raw = pd.DataFrame([
        # M1: two assays -> keep max pchembl (8.5, Ki) and count = 2
        {"molecule_chembl_id": "M1", "canonical_smiles": "CCO", "standard_type": "IC50", "pchembl_value": "7.0"},
        {"molecule_chembl_id": "M1", "canonical_smiles": "CCO", "standard_type": "Ki", "pchembl_value": "8.5"},
        {"molecule_chembl_id": "M2", "canonical_smiles": "c1ccccc1", "standard_type": "IC50", "pchembl_value": "6.0"},
        # M3: invalid SMILES -> dropped
        {"molecule_chembl_id": "M3", "canonical_smiles": "not_a_smiles", "standard_type": "IC50", "pchembl_value": "9.0"},
        # M4: no pchembl -> dropped
        {"molecule_chembl_id": "M4", "canonical_smiles": "CCO", "standard_type": "IC50", "pchembl_value": None},
    ])
    out = cc.to_candidates(raw)

    assert list(out["molecule_chembl_id"]) == ["M1", "M2"]  # sorted by potency desc
    m1 = out.iloc[0]
    assert m1["pchembl_value"] == 8.5
    assert m1["n_activities"] == 2
    assert m1["standard_type"] == "Ki"  # attributes come from the best assay


def test_to_candidates_empty_returns_typed_frame():
    out = cc.to_candidates(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == cc._CANDIDATE_COLUMNS


def test_target_ranking_prefers_single_protein_human(monkeypatch):
    fake = {"targets": [
        {"target_chembl_id": "C1", "target_type": "PROTEIN-PROTEIN INTERACTION",
         "organism": "Homo sapiens", "score": 17, "pref_name": "complex"},
        {"target_chembl_id": "C2", "target_type": "SINGLE PROTEIN",
         "organism": "Mus musculus", "score": 15, "pref_name": "mouse"},
        {"target_chembl_id": "C3", "target_type": "SINGLE PROTEIN",
         "organism": "Homo sapiens", "score": 10, "pref_name": "human"},
    ]}
    monkeypatch.setattr(cc, "_get", lambda path, params, **kw: fake)

    ranked = cc.search_targets("EGFR")
    # single-protein + human wins despite the lowest API score
    assert ranked[0].chembl_id == "C3"


@pytest.mark.parametrize("query,expected_id", [("EGFR", "CHEMBL203")])
def test_live_smoke(query, expected_id):
    try:
        target, candidates = cc.get_candidates(query, max_records=200, use_cache=False)
    except RuntimeError as err:
        pytest.skip(f"ChEMBL unreachable: {err}")

    assert target.chembl_id == expected_id
    assert len(candidates) > 0
    assert candidates["canonical_smiles"].map(
        lambda s: Chem.MolFromSmiles(s) is not None).all()
