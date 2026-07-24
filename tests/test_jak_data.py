"""STEP 2 tests: JAK data layer — median dedup, SMILES hygiene, cross-measured join.

Uses a synthetic activities frame (no network) to pin the collapse/join logic, plus
a live smoke test that self-skips when ChEMBL is unreachable.
"""
import pandas as pd
import pytest

from src.data import jak


def _acts(rows):
    return pd.DataFrame(rows, columns=["canonical_smiles", "pchembl_value"])


def test_collapse_medians_and_dedups():
    # ethanol measured 3x (median of 6,7,8 = 7), benzene once, one bad SMILES dropped.
    acts = _acts([("CCO", 6.0), ("CCO", 7.0), ("CCO", 8.0),
                  ("c1ccccc1", 9.0), ("not_a_smiles", 5.0)])
    out = jak._collapse(acts)
    assert set(out["smi"]) == {"CCO", "c1ccccc1"}          # bad SMILES dropped
    row = out[out["smi"] == "CCO"].iloc[0]
    assert row["pchembl"] == 7.0                            # median, not mean/max
    assert row["n_meas"] == 3                               # provenance count


def test_collapse_canonicalises_before_grouping():
    # same molecule written two ways collapses to one row.
    acts = _acts([("OCC", 6.0), ("CCO", 8.0)])
    out = jak._collapse(acts)
    assert len(out) == 1
    assert out.iloc[0]["n_meas"] == 2


def test_collapse_empty():
    assert jak._collapse(_acts([])).empty


def test_build_isoform_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown isoform"):
        jak.build_isoform_dataset("JAK9", use_cache=False)


def test_build_and_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(jak, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(jak.cc, "fetch_activities",
                        lambda tid, **kw: _acts([("CCO", 7.0), ("c1ccccc1", 8.0)]))
    first = jak.build_isoform_dataset("JAK1", use_cache=True)
    assert (tmp_path / "JAK1.parquet").exists()
    assert first["smi"].is_unique                          # no duplicate molecules
    # second call loads from cache (fetch would raise if called again)
    monkeypatch.setattr(jak.cc, "fetch_activities",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("refetched")))
    pd.testing.assert_frame_equal(first, jak.build_isoform_dataset("JAK1", use_cache=True))


def test_cross_measured_is_intersection(tmp_path, monkeypatch):
    monkeypatch.setattr(jak, "CACHE_DIR", tmp_path)
    per = {
        "JAK1": _acts([("CCO", 8.0), ("c1ccccc1", 7.0)]),          # CCO, benzene
        "JAK2": _acts([("CCO", 6.0), ("CCN", 7.0)]),               # CCO, ethylamine
        "JAK3": _acts([("CCO", 5.0), ("c1ccccc1", 6.0)]),          # CCO, benzene
    }
    monkeypatch.setattr(jak.cc, "fetch_activities",
                        lambda tid, **kw: per[{v: k for k, v in jak.TARGETS.items()}[tid]])
    cross = jak.build_cross_measured(use_cache=False)
    assert list(cross["smi"]) == ["CCO"]                   # only molecule in all three
    assert set(cross.columns) == {"smi", "JAK1", "JAK2", "JAK3"}


def test_live_summary_smoke():
    try:
        tbl = jak.summary(use_cache=True)
    except (RuntimeError, OSError) as err:
        pytest.skip(f"ChEMBL unreachable: {err}")
    assert set(tbl["isoform"]) == {"JAK1", "JAK2", "JAK3"}
    assert (tbl["n_molecules"] > 1000).all()               # thousands per isoform
    assert tbl.attrs["n_cross_measured"] > 500
