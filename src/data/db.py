"""The evidence store: one row per measurement, with provenance (STEP 16).

Before this, the repo stored *answers* and discarded the evidence behind them.
`panel_data._collapse` reduced every molecule to a median pchembl at ingest time,
and the raw records survived only in `data/cache/activities_<sha1>.parquet` — keyed
by a hash of the request params, so "every activity for CHEMBL203" was a question
the cache could not answer. Two consequences, both measured and both expensive:

  * The ATP-competition confound (README's headline caveat, Spearman 0.682 -> 0.462
    on the Ki/Kd subset) could only be *bounded*, because `standard_type` was the
    finest instrument available — assay identity was never even retrieved.
  * A campaign was not reproducible. `Campaign` pinned `code_version` and
    `model_ids` but not the data, and the ChEMBL release was recorded nowhere.

This module holds the schema that fixes both. The rule it exists to enforce:
**store evidence, derive answers.** `activity` is never collapsed; the per-molecule
median that `build_isoform_dataset` returns is the `v_pchembl_median` *view*, so an
assay-stratified training set is a WHERE clause rather than a bespoke script.

Deliberately not in here: fingerprints and models (derived, large, already cached
well), and the run registry (append-only JSONL, a different problem with a different
right answer — see DATABASE_DESIGN.md).

**The deployed app never imports this.** `assets/*.parquet` stays the deploy
surface; the database is a build-time artifact that *generates* those files, and
`duckdb` is a dev dependency only.

CLI (create/inspect the store):
    python -m src.data.db
"""
from __future__ import annotations

import contextlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = _ROOT / "data" / "chem.duckdb"        # runtime artifact, gitignored

SCHEMA_VERSION = 1

# Right-censored records ('>' 10 uM, no pchembl) are *stored* but excluded from the
# views. DESIGN_DECISIONS section 1 rejected fetching them when every pull had to be
# justified by a model that would consume it immediately; with an evidence table the
# cost is one column and no decision. Storing is not modelling — nothing downstream
# changes until someone writes a query against them.
_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

-- ── provenance ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS source (
    source_id     TEXT PRIMARY KEY,     -- 'chembl_37' — what a campaign pins
    name          TEXT NOT NULL,
    release       TEXT NOT NULL,        -- read from the live /status endpoint
    retrieved_utc TIMESTAMP NOT NULL,
    base_url      TEXT
);

CREATE TABLE IF NOT EXISTS ingestion (   -- one fetch run; replaces the sha1 cache key
    ingestion_id     TEXT PRIMARY KEY,
    source_id        TEXT NOT NULL,
    target_chembl_id TEXT,
    params_json      TEXT NOT NULL,      -- the exact query, readable not hashed
    n_rows           INTEGER,
    code_version     TEXT,
    started_utc      TIMESTAMP
);

-- ── entities ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS target (
    target_chembl_id TEXT PRIMARY KEY,
    pref_name   TEXT,
    organism    TEXT,
    target_type TEXT,
    source_id   TEXT
);

CREATE TABLE IF NOT EXISTS molecule (
    inchikey      TEXT PRIMARY KEY,      -- of the standardised neutral parent
    parent_smiles TEXT NOT NULL,         -- src/standardize.py output
    raw_smiles    TEXT,                  -- as the source delivered it
    mol_chembl_id TEXT,
    mw REAL, logp REAL, tpsa REAL, hbd INTEGER, hba INTEGER,
    rotb INTEGER, arom_rings INTEGER, qed REAL,
    -- Which standardisation policy produced parent_smiles. The committed
    -- assets/jak parquet predates the current policy and is ~0.5 % un-standardised;
    -- tagging the policy keeps legacy rows identifiable instead of silently mixed in.
    standardize_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assay (
    assay_chembl_id  TEXT PRIMARY KEY,
    assay_type       TEXT,               -- B(inding) / F(unctional) / A / T / P
    description      TEXT,
    target_chembl_id TEXT
);

-- ── the evidence table ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS activity (
    -- ChEMBL's own activity_id, so re-ingesting a target inserts nothing new and an
    -- interrupted backfill can simply be re-run. Same idempotence property that made
    -- model_id worth fixing in STEP 12.
    activity_id        BIGINT PRIMARY KEY,
    inchikey           TEXT NOT NULL,
    target_chembl_id   TEXT NOT NULL,
    assay_chembl_id    TEXT,
    standard_type      TEXT,             -- IC50 / Ki / Kd / EC50
    standard_relation  TEXT,             -- '=' '>' '<'  — the censored class
    standard_value     DOUBLE,
    standard_units     TEXT,
    pchembl_value      DOUBLE,
    document_chembl_id TEXT,
    document_year      INTEGER,
    source_id          TEXT NOT NULL,
    ingestion_id       TEXT NOT NULL
);

-- ── the wide screening library ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS library (
    library_id  TEXT PRIMARY KEY,        -- 'chembl20panel'
    description TEXT,
    source_id   TEXT,
    built_utc   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS library_member (
    library_id    TEXT NOT NULL,
    inchikey      TEXT NOT NULL,
    source_target TEXT,                  -- which panel target it was drawn from
    druglike      BOOLEAN,               -- Tier 0 verdict, carried by the library
    PRIMARY KEY (library_id, inchikey)
);

CREATE INDEX IF NOT EXISTS activity_target ON activity(target_chembl_id);
CREATE INDEX IF NOT EXISTS activity_mol    ON activity(inchikey);
CREATE INDEX IF NOT EXISTS activity_assay  ON activity(assay_chembl_id);
"""

# `_collapse`, expressed as SQL. Keeping it a view rather than a table is the whole
# point: the assay-stratified variants STEP 4's audit had to hand-build are the same
# query with one more predicate.
_VIEWS = """
CREATE OR REPLACE VIEW v_pchembl_median AS
SELECT a.target_chembl_id,
       m.inchikey,
       m.parent_smiles                                                    AS smi,
       median(a.pchembl_value)                                            AS pchembl,
       count(*)                                                           AS n_meas,
       median(a.pchembl_value) FILTER (WHERE a.standard_type IN ('Ki','Kd'))
                                                                          AS pchembl_kikd,
       count(*)                FILTER (WHERE a.standard_type IN ('Ki','Kd'))
                                                                          AS n_kikd,
       min(a.document_year)                                               AS year_first,
       avg(CASE WHEN s.assay_type = 'B' THEN 1.0
                WHEN s.assay_type IS NULL THEN NULL ELSE 0.0 END)         AS frac_binding
FROM activity a
JOIN molecule m USING (inchikey)
LEFT JOIN assay s USING (assay_chembl_id)
WHERE a.pchembl_value IS NOT NULL
  AND (a.standard_relation = '=' OR a.standard_relation IS NULL)
GROUP BY 1, 2, 3;

-- Molecule x target matrix, the shape a multi-task model needs. 37 targets with a
-- shared molecule index, which 37 separate parquet files could not provide.
CREATE OR REPLACE VIEW v_molecule_target_matrix AS
SELECT inchikey, smi, target_chembl_id, pchembl, n_meas
FROM v_pchembl_median;
"""


def connect(path: Path | None = None, read_only: bool = False):
    """Open the store, creating the schema on first use.

    `duckdb` is imported here rather than at module scope so that importing
    `src.data` on a deployment that has no dev dependencies does not fail.
    """
    import duckdb

    path = Path(path or DB_PATH)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        _init(con)
    return con


def _init(con) -> None:
    con.execute(_DDL)
    con.execute(_VIEWS)
    current = con.execute("SELECT max(version) FROM schema_version").fetchone()[0]
    if current is None:
        con.execute("INSERT INTO schema_version VALUES (?)", [SCHEMA_VERSION])
    elif current != SCHEMA_VERSION:
        raise RuntimeError(
            f"Store at schema {current}, code expects {SCHEMA_VERSION}. "
            "The store is a rebuildable artifact — delete it and re-run the backfill.")


@contextlib.contextmanager
def session(path: Path | None = None, read_only: bool = False):
    con = connect(path, read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def exists(path: Path | None = None) -> bool:
    return Path(path or DB_PATH).exists()


def summary(con) -> dict:
    """Row counts per table plus the sources present — what a backfill produced."""
    tables = ("source", "ingestion", "target", "molecule", "assay", "activity",
              "library", "library_member")
    out = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
    out["sources"] = [r[0] for r in
                      con.execute("SELECT source_id FROM source ORDER BY 1").fetchall()]
    out["censored_rows"] = con.execute(
        "SELECT count(*) FROM activity WHERE standard_relation <> '='").fetchone()[0]
    return out


def _main() -> None:
    with session() as con:
        s = summary(con)
        print(f"store: {DB_PATH}")
        for k, v in s.items():
            print(f"  {k:16} {v}")


if __name__ == "__main__":
    _main()
