"""Phase 2 tests: Lipinski Ro5 + PAINS filtering."""
import pandas as pd

from src.filters.druglikeness import apply_druglikeness, _FILTER_COLUMNS


def test_clean_drug_passes():
    df = pd.DataFrame({"canonical_smiles": ["COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC"]})  # top EGFR hit
    out = apply_druglikeness(df).iloc[0]
    assert out["ro5_violations"] == 0
    assert out["ro5_pass"] and out["pains_pass"] and out["druglike"]


def test_pains_substructure_fails():
    df = pd.DataFrame({"canonical_smiles": ["O=C1CSC(=S)N1"]})  # rhodanine core (PAINS)
    out = apply_druglikeness(df).iloc[0]
    assert not out["pains_pass"]
    assert not out["druglike"]


def test_ro5_double_violation_fails():
    df = pd.DataFrame({"canonical_smiles": ["C" * 40]})  # MW 563, LogP 15.8 -> 2 violations
    out = apply_druglikeness(df).iloc[0]
    assert out["ro5_violations"] == 2
    assert not out["ro5_pass"]
    assert not out["druglike"]


def test_input_not_mutated_and_columns_added():
    df = pd.DataFrame({"canonical_smiles": ["CCO"]})
    out = apply_druglikeness(df)
    assert list(df.columns) == ["canonical_smiles"]          # original untouched
    assert all(c in out.columns for c in _FILTER_COLUMNS)    # all flags present
    assert out.iloc[0]["druglike"]


def test_empty_frame_returns_typed_columns():
    out = apply_druglikeness(pd.DataFrame({"canonical_smiles": []}))
    assert out.empty
    assert all(c in out.columns for c in _FILTER_COLUMNS)
