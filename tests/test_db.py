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
    con = _db.connect(tmp_path / "t.duckdb", announce=False)
    yield con
    con.close()


def test_schema_starts_empty(tmp_path):
    con = _db.connect(tmp_path / "empty.duckdb", announce=False)
    s = _db.summary(con)
    assert s["activity"] == 0 and s["molecule"] == 0 and s["sources"] == []
    con.close()


def test_schema_version_mismatch_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "v.duckdb"
    _db.connect(path, announce=False).close()
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


def test_library_overlap_reports_none_without_a_store(monkeypatch, tmp_path):
    """'Not checked' must never be reported as 'disjoint' — the whole point of the
    STEP 16 finding is that target-level disjointness did not imply molecule-level."""
    from src import panels
    from src.data import db as dbmod

    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "absent.duckdb")
    assert panels.library_molecule_overlap(panels.JAK) is None


def test_library_overlap_counts_gate_positives(tmp_path, monkeypatch):
    from src import panels
    from src.data import db as dbmod

    path = tmp_path / "ov.duckdb"
    con = dbmod.connect(path, announce=False)
    con.execute("INSERT INTO source VALUES ('s','ChEMBL','99',now(),'u')")
    con.execute("INSERT INTO ingestion VALUES ('i','s','T','{}',0,'v',now())")
    con.execute("INSERT INTO molecule (inchikey, parent_smiles, standardize_version) "
                "VALUES ('K1','CCO','v'), ('K2','c1ccccc1','v')")
    con.execute("INSERT INTO library VALUES ('L','d','s',now())")
    con.execute("INSERT INTO library_member VALUES ('L','K1',NULL,true), ('L','K2',NULL,true)")
    # K1 is an active on a JAK isoform => a gate positive sitting inside the library
    con.execute("INSERT INTO activity (activity_id, inchikey, target_chembl_id, "
                "standard_relation, pchembl_value, source_id, ingestion_id) "
                "VALUES (1,'K1','CHEMBL2835','=',8.0,'s','i')")
    con.close()
    monkeypatch.setattr(dbmod, "DB_PATH", path)

    out = panels.library_molecule_overlap(panels.JAK)
    assert out["library_molecules"] == 2
    assert out["gate_positives_in_library"] == 1


def test_connection_url_is_an_absolute_local_path(tmp_path):
    """The 'connection' is a file, not an endpoint — four slashes mark it absolute."""
    url = _db.connection_url(tmp_path / "s.duckdb")
    assert url.startswith("duckdb:////")
    assert str((tmp_path / "s.duckdb").resolve()) in url
    ro = _db.connection_url(tmp_path / "s.duckdb", read_only=True)
    assert ro.endswith("?access_mode=read_only")


def test_connection_help_names_the_real_path(tmp_path):
    text = _db.connection_help(tmp_path / "s.duckdb")
    assert str((tmp_path / "s.duckdb").resolve()) in text
    assert "read-only" in text and "one writer" in text.lower()


def test_creation_announces_once_then_stays_quiet(tmp_path, capsys):
    path = tmp_path / "announce.duckdb"
    _db.connect(path).close()
    assert "connect to the evidence store" in capsys.readouterr().out
    _db.connect(path).close()                       # already exists -> no repeat
    assert "connect to the evidence store" not in capsys.readouterr().out


def test_announce_can_be_silenced_for_library_use(tmp_path, capsys):
    _db.connect(tmp_path / "quiet.duckdb", announce=False).close()
    assert capsys.readouterr().out == ""


# --- the committed evidence export ------------------------------------------ #

def test_export_then_restore_round_trips(tmp_path, store):
    """The export is what puts the data in the repo, so it has to come back intact."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import export_store

    ingest.ingest_target(store, "CHEMBLT1")
    out = tmp_path / "evidence"
    manifest = export_store.export(store, out)
    assert manifest["tables"]["activity"]["rows"] == 5
    assert (out / "manifest.json").exists()

    counts = _db.restore_from_assets(path=tmp_path / "restored.duckdb", assets=out)
    assert counts["activity"] == 5
    assert counts["molecule"] == 3
    assert counts["sources"] == ["chembl_99"]


def test_restore_is_idempotent(tmp_path, store):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import export_store

    ingest.ingest_target(store, "CHEMBLT1")
    out = tmp_path / "evidence"
    export_store.export(store, out)
    target = tmp_path / "twice.duckdb"
    first = _db.restore_from_assets(path=target, assets=out)
    second = _db.restore_from_assets(path=target, assets=out)
    assert first["activity"] == second["activity"] == 5      # replaced, not appended


def test_restore_without_an_export_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="No committed evidence export"):
        _db.restore_from_assets(path=tmp_path / "x.duckdb", assets=tmp_path / "nothing")


def test_committed_export_matches_the_documented_row_counts():
    """The README quotes these numbers; drift between doc and data is a silent lie."""
    import json

    manifest_path = _db.EVIDENCE_ASSETS / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("no committed evidence export in this checkout")
    m = json.loads(manifest_path.read_text())
    assert m["tables"]["activity"]["rows"] == 103420
    assert m["tables"]["molecule"]["rows"] == 75870
    assert m["tables"]["assay"]["rows"] == 5207
    assert m["tables"]["library_member"]["rows"] == 38592
    assert m["schema_version"] == _db.SCHEMA_VERSION
