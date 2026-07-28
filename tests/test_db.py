"""STEP 16 tests: the evidence store — schema, idempotent ingest, and the views.

All offline. `fetch_activities_evidence` and `release` are stubbed, and the store is
built in tmp_path, so nothing here touches the network or the developer's real
`data/chem.duckdb`.
"""
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb", reason="duckdb is a dev-only dependency")

from src.data import db as _db          # noqa: E402
from src.data import ingest             # noqa: E402


def _acts(rows):
    """Raw evidence-shaped activity frame."""
    return pd.DataFrame(rows, columns=[
        "activity_id", "molecule_chembl_id", "canonical_smiles", "assay_chembl_id",
        "assay_type", "assay_description", "standard_type", "standard_relation",
        "standard_value", "standard_units", "pchembl_value", "document_chembl_id",
        "document_year"])


_ROWS = [
    # ethanol, measured three times -> median 7.0, and once as a Ki
    (1, "CHEMBL1", "CCO", "CHEMBLA1", "B", "binding", "IC50", "=", 100.0, "nM", 6.0, "D1", 2001),
    (2, "CHEMBL1", "CCO", "CHEMBLA1", "B", "binding", "IC50", "=", 100.0, "nM", 7.0, "D1", 2003),
    (3, "CHEMBL1", "CCO", "CHEMBLA2", "F", "cell",    "Ki",   "=", 100.0, "nM", 8.0, "D2", 2005),
    # benzene, one measurement
    (4, "CHEMBL2", "c1ccccc1", "CHEMBLA1", "B", "binding", "IC50", "=", 10.0, "nM", 9.0, "D1", 2002),
    # a right-censored non-binder: stored, but must not reach the view
    (5, "CHEMBL3", "CCCC", "CHEMBLA1", "B", "binding", "IC50", ">", 10000.0, "nM", None, "D3", 2004),
]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.cc, "release", lambda: "ChEMBL_99")
    monkeypatch.setattr(ingest.cc, "fetch_activities_evidence",
                        lambda tid, **kw: _acts(_ROWS))
    con = _db.connect(tmp_path / "t.duckdb")
    yield con
    con.close()


def test_schema_starts_empty(tmp_path):
    con = _db.connect(tmp_path / "empty.duckdb")
    s = _db.summary(con)
    assert s["activity"] == 0 and s["molecule"] == 0 and s["sources"] == []
    con.close()


def test_schema_version_mismatch_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "v.duckdb"
    _db.connect(path).close()
    monkeypatch.setattr(_db, "SCHEMA_VERSION", _db.SCHEMA_VERSION + 1)
    with pytest.raises(RuntimeError, match="schema"):
        _db.connect(path)


def test_ingest_records_source_and_provenance(store):
    ingest.ingest_target(store, "CHEMBLT1")
    src = store.execute("SELECT source_id, release FROM source").fetchall()
    assert src == [("chembl_99", "ChEMBL_99")]
    ing = store.execute("SELECT target_chembl_id, n_rows FROM ingestion").fetchall()
    assert ing == [("CHEMBLT1", 5)]


def test_reingest_inserts_nothing(store):
    """Idempotence on ChEMBL's own activity_id — an interrupted backfill is re-runnable."""
    first = ingest.ingest_target(store, "CHEMBLT1")
    second = ingest.ingest_target(store, "CHEMBLT1")
    assert first["activities"] == 5
    assert second["activities"] == 0
    assert store.execute("SELECT count(*) FROM activity").fetchone()[0] == 5


def test_censored_rows_are_stored_but_excluded_from_the_view(store):
    ingest.ingest_target(store, "CHEMBLT1")
    assert _db.summary(store)["censored_rows"] == 1        # the '>' row is kept
    view = store.execute("SELECT smi FROM v_pchembl_median").df()
    assert "CCCC" not in set(view["smi"])                  # but never reaches training


def test_view_reproduces_collapse_semantics(store):
    """Median per molecule, Ki/Kd sub-median, measurement count, earliest year."""
    ingest.ingest_target(store, "CHEMBLT1")
    v = store.execute("SELECT * FROM v_pchembl_median ORDER BY smi").df()
    eth = v[v["smi"] == "CCO"].iloc[0]
    assert eth["pchembl"] == 7.0            # median of 6, 7, 8 — not mean, not max
    assert eth["n_meas"] == 3
    assert eth["pchembl_kikd"] == 8.0       # the single Ki
    assert eth["n_kikd"] == 1
    assert eth["year_first"] == 2001
    assert eth["frac_binding"] == pytest.approx(2 / 3)   # two of three are assay_type B


def test_molecules_are_keyed_by_inchikey_so_spellings_collapse(store, monkeypatch):
    """Two spellings of ethanol must be one molecule, not two."""
    rows = [(*_ROWS[0][:2], "OCC", *_ROWS[0][3:]), _ROWS[1]]
    monkeypatch.setattr(ingest.cc, "fetch_activities_evidence",
                        lambda tid, **kw: _acts(rows))
    ingest.ingest_target(store, "CHEMBLT1")
    assert store.execute("SELECT count(*) FROM molecule").fetchone()[0] == 1
    assert store.execute("SELECT count(*) FROM activity").fetchone()[0] == 2


def test_assays_are_captured(store):
    """Assay identity is what makes the ATP-confound question askable at all."""
    ingest.ingest_target(store, "CHEMBLT1")
    assays = store.execute(
        "SELECT assay_chembl_id, assay_type FROM assay ORDER BY 1").fetchall()
    assert assays == [("CHEMBLA1", "B"), ("CHEMBLA2", "F")]


def test_assay_stratified_query_is_one_predicate(store):
    """The point of the evidence table: STEP 4's audit becomes a WHERE clause."""
    ingest.ingest_target(store, "CHEMBLT1")
    kikd = store.execute("""
        SELECT m.parent_smiles, median(a.pchembl_value) AS pchembl
        FROM activity a JOIN molecule m USING (inchikey)
        WHERE a.standard_type IN ('Ki','Kd') AND a.standard_relation = '='
        GROUP BY 1""").df()
    assert list(kikd["parent_smiles"]) == ["CCO"]
    assert kikd.iloc[0]["pchembl"] == 8.0


def test_molecule_rows_carry_the_standardisation_policy(store):
    from src.standardize import STANDARDIZE_VERSION

    ingest.ingest_target(store, "CHEMBLT1")
    versions = {r[0] for r in
                store.execute("SELECT DISTINCT standardize_version FROM molecule").fetchall()}
    assert versions == {STANDARDIZE_VERSION}


def test_molecule_physchem_lands_in_the_right_columns(store):
    """Guards the column-order class of bug: an insert that misaligns fields would
    still populate every column, so only the values reveal it."""
    ingest.ingest_target(store, "CHEMBLT1")
    row = store.execute(
        "SELECT parent_smiles, mw, logp, hbd, arom_rings, qed FROM molecule "
        "WHERE parent_smiles = 'c1ccccc1'").fetchone()
    smi, mw, logp, hbd, arom, qed = row
    assert smi == "c1ccccc1"
    assert 77 < mw < 79                      # benzene is 78.11 g/mol
    assert 1.0 < logp < 2.5                  # ~1.7
    assert hbd == 0 and arom == 1
    assert 0.0 <= qed <= 1.0
