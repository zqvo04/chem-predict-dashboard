"""STEP 2 tests: panel data layer — median dedup, SMILES hygiene, cross-measured join.

Uses a synthetic activities frame (no network) to pin the collapse/join logic, plus
a live smoke test that self-skips when ChEMBL is unreachable.

Cache/bundle paths come from the panel spec since STEP 15, so the round-trip tests
build a throwaway panel rooted at tmp_path instead of monkeypatching module-level
directories — which exercises the real lookup order rather than bypassing it.
"""
import dataclasses

import pandas as pd
import pytest

from src.data import panel_data as jak
from src.panels import JAK


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


def _tmp_panel(tmp_path):
    """The JAK panel with its assets redirected into tmp_path (no bundle present)."""
    return dataclasses.replace(JAK, root=tmp_path)


def test_build_isoform_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown isoform"):
        jak.build_isoform_dataset(JAK, "JAK9", use_cache=False)


def test_build_and_cache_roundtrip(tmp_path, monkeypatch):
    panel = _tmp_panel(tmp_path)                           # no bundle -> forces the fetch path
    monkeypatch.setattr(jak.cc, "fetch_activities",
                        lambda tid, **kw: _acts([("CCO", 7.0), ("c1ccccc1", 8.0)]))
    first = jak.build_isoform_dataset(panel, "JAK1", use_cache=True)
    assert (panel.data_cache / "JAK1.parquet").exists()
    assert first["smi"].is_unique                          # no duplicate molecules
    # second call loads from cache (fetch would raise if called again)
    monkeypatch.setattr(jak.cc, "fetch_activities",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("refetched")))
    pd.testing.assert_frame_equal(
        first, jak.build_isoform_dataset(panel, "JAK1", use_cache=True))


def test_panel_namespaces_its_cache(tmp_path, monkeypatch):
    """Two panels must not read each other's datasets — the whole point of STEP 15."""
    from src.panels import PI3K

    monkeypatch.setattr(jak.cc, "fetch_activities",
                        lambda tid, **kw: _acts([("CCO", 7.0)]))
    jak_panel = dataclasses.replace(JAK, root=tmp_path)
    pi3k_panel = dataclasses.replace(PI3K, root=tmp_path)
    jak.build_isoform_dataset(jak_panel, "JAK1", use_cache=True)
    jak.build_isoform_dataset(pi3k_panel, "PIK3CD", use_cache=True)
    assert jak_panel.data_cache != pi3k_panel.data_cache
    assert (jak_panel.data_cache / "JAK1.parquet").exists()
    assert not (pi3k_panel.data_cache / "JAK1.parquet").exists()


def test_cross_measured_is_intersection(tmp_path, monkeypatch):
    panel = _tmp_panel(tmp_path)
    per = {
        "JAK1": _acts([("CCO", 8.0), ("c1ccccc1", 7.0)]),          # CCO, benzene
        "JAK2": _acts([("CCO", 6.0), ("CCN", 7.0)]),               # CCO, ethylamine
        "JAK3": _acts([("CCO", 5.0), ("c1ccccc1", 6.0)]),          # CCO, benzene
    }
    monkeypatch.setattr(jak.cc, "fetch_activities",
                        lambda tid, **kw: per[{v: k for k, v in JAK.chembl_ids.items()}[tid]])
    cross = jak.build_cross_measured(panel, use_cache=False)
    assert list(cross["smi"]) == ["CCO"]                   # only molecule in all three
    assert set(cross.columns) == {"smi", "JAK1", "JAK2", "JAK3"}


def test_live_summary_smoke():
    try:
        tbl = jak.summary(JAK, use_cache=True)
    except (RuntimeError, OSError) as err:
        pytest.skip(f"ChEMBL unreachable: {err}")
    assert set(tbl["isoform"]) == {"JAK1", "JAK2", "JAK3"}
    assert (tbl["n_molecules"] > 1000).all()               # thousands per isoform
    assert tbl.attrs["n_cross_measured"] > 500


# --- assay provenance (assay/time audit) ---------------------------------- #

def _acts_full(rows):
    return pd.DataFrame(rows, columns=["canonical_smiles", "pchembl_value",
                                       "standard_type", "assay_type", "document_year"])


def test_collapse_separates_the_atp_independent_subset():
    """An IC50 for an ATP-competitive kinase inhibitor moves with the assay's ATP
    level; a Ki/Kd does not. Keeping them separable is what lets the audit ask
    whether the selectivity gap is real or an artefact of pooled assay types."""
    acts = _acts_full([("CCO", 6.0, "IC50", "B", 2010),
                       ("CCO", 8.0, "Ki", "B", 2012),
                       ("CCO", 9.0, "Kd", "B", 2014),
                       ("c1ccccc1", 7.0, "EC50", "F", 2020)])
    out = jak._collapse(acts).set_index("smi")
    assert out.loc["CCO", "pchembl"] == 8.0           # median over all four types
    assert out.loc["CCO", "pchembl_kikd"] == 8.5      # median over Ki/Kd only
    assert out.loc["CCO", "n_kikd"] == 2
    assert out.loc["c1ccccc1", "n_kikd"] == 0
    assert pd.isna(out.loc["c1ccccc1", "pchembl_kikd"])


def test_collapse_records_arrival_year_and_assay_mix():
    """year_first is the molecule's arrival date — the only cut a time split can
    make honestly, since a later re-measurement does not make it new chemistry."""
    acts = _acts_full([("CCO", 6.0, "IC50", "B", 2018),
                       ("CCO", 7.0, "IC50", "F", 2012),
                       ("CCO", 8.0, "IC50", "B", 2020)])
    row = jak._collapse(acts).iloc[0]
    assert row["year_first"] == 2012                   # earliest, not latest or median
    assert row["frac_binding"] == pytest.approx(2 / 3)


def test_collapse_tolerates_activities_cached_without_provenance():
    """An activities frame cached before these fields were requested must still
    build — the provenance columns come back NaN rather than failing the build."""
    out = jak._collapse(_acts([("CCO", 7.0)]))
    assert out.iloc[0]["pchembl"] == 7.0
    assert out.iloc[0]["n_kikd"] == 0
    assert pd.isna(out.iloc[0]["year_first"])
