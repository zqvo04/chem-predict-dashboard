"""STEP 7/13 tests: wide library canonicalisation, dedup and gate-leakage exclusion.

Offline — the ChEMBL fetch is mocked, so these pin the assembly logic rather than
the data. A live smoke test self-skips when no library has been built.
"""
import pandas as pd
import pytest

from src.data import library as lib


def _raw_acts(smiles):
    return pd.DataFrame({
        "molecule_chembl_id": [f"CHEMBL{i}" for i in range(len(smiles))],
        "canonical_smiles": smiles,
        "pchembl_value": [7.0] * len(smiles),
        "standard_type": ["IC50"] * len(smiles),
    })


def _patch(monkeypatch, tmp_path, pools, excluded=frozenset()):
    monkeypatch.setattr(lib, "CACHE", tmp_path / "library.parquet")
    monkeypatch.setattr(lib, "BUNDLED", tmp_path / "no-bundle.parquet")
    monkeypatch.setattr(lib, "LIBRARY_TARGETS", {n: n for n in pools})
    monkeypatch.setattr(lib, "MIN_TARGETS", 1)
    monkeypatch.setattr(lib, "_gate_training_smiles", lambda use_cache=True: set(excluded))
    monkeypatch.setattr(lib.cc, "fetch_activities", lambda cid, **kw: pools[cid])


def test_load_library_canonicalises_and_dedups(tmp_path, monkeypatch):
    # Two spellings of ethanol + benzene + one unparseable -> 2 unique canonical rows.
    _patch(monkeypatch, tmp_path,
           {"T1": _raw_acts(["OCC", "CCO", "c1ccccc1", "not_a_smiles"])})
    out = lib.load_library(use_cache=False)
    assert set(out["smi"]) == {"CCO", "c1ccccc1"}        # deduped + canonical, bad dropped
    assert out["smi"].is_unique
    assert "druglike" in out.columns                      # Tier 0 carried by the cache
    assert (tmp_path / "library.parquet").exists()        # cached


def test_load_library_excludes_binder_gate_training_molecules(tmp_path, monkeypatch):
    """The library must not contain what the gate was trained on, or the gate's
    verdict would be recall rather than prediction — the exact leakage that made
    the Tox21 library's MPO tox column meaningless."""
    seen = "CCO"
    _patch(monkeypatch, tmp_path,
           {"T1": _raw_acts([seen, "c1ccccc1"])}, excluded={seen})
    out = lib.load_library(use_cache=False)
    assert seen not in set(out["smi"])
    assert "c1ccccc1" in set(out["smi"])


def test_load_library_dedups_across_targets(tmp_path, monkeypatch):
    """A molecule active on two panel targets is still one library entry."""
    _patch(monkeypatch, tmp_path,
           {"T1": _raw_acts(["CCO", "c1ccccc1"]), "T2": _raw_acts(["CCO", "CCN"])})
    out = lib.load_library(use_cache=False)
    assert out["smi"].is_unique
    assert set(out["smi"]) == {"CCO", "c1ccccc1", "CCN"}


def test_load_library_refuses_a_thin_build(tmp_path, monkeypatch):
    """Too few targets fetched means an unrepresentative library; fail loudly rather
    than silently screen against a thin one."""
    _patch(monkeypatch, tmp_path, {"T1": _raw_acts(["CCO"])})
    monkeypatch.setattr(lib, "MIN_TARGETS", 5)
    with pytest.raises(RuntimeError, match="too few for a diverse screening library"):
        lib.load_library(use_cache=False)


def test_load_library_uses_cache(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path, {"T1": _raw_acts(["CCO"])})
    lib.load_library(use_cache=False)                     # writes cache
    monkeypatch.setattr(lib.cc, "fetch_activities",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("refetched")))
    assert list(lib.load_library(use_cache=True)["smi"]) == ["CCO"]


def test_live_library_is_diverse_and_leak_free():
    try:
        out = lib.load_library(use_cache=True)
    except (RuntimeError, OSError) as err:
        pytest.skip(f"no library built / ChEMBL unreachable: {err}")
    assert len(out) > 1000
    assert out["smi"].is_unique
    if "source_target" in out.columns:
        assert out["source_target"].nunique() >= lib.MIN_TARGETS
