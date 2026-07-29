# Data layer design — from cached files to a queryable corpus

**Status: design, not built.** The measurements in §1 are real (taken from the
working tree); everything else is a proposal. When built, results go to
VALIDATION.md and surviving rationale to DESIGN_DECISIONS.md.

> **On the apparent contradiction with STEP 15.** Phase 2 chose append-only JSONL
> over a database for the run registry, and that decision stands. The registry is
> small, append-only, written once per round and read in order — a database would
> have added a dependency and schema migrations to buy nothing. The **data layer** is
> the opposite problem: hundreds of thousands of measurements across 37 targets,
> queried by joins nobody can enumerate in advance. Different problem, different
> answer. The registry stays where it is.

---

## 1. What is actually wrong today (measured)

Four defects, one cause: **the repo stores answers, not evidence.**

| defect | evidence in the working tree |
|---|---|
| **The cache cannot be queried** | `data/cache/activities_<sha1>.parquet`, keyed by a hash of the request params (`src/data/cache.py`). There is no way to ask "every activity for CHEMBL203" without reconstructing the exact params that produced it. 15 files, 66 305 rows, all opaque. |
| **Individual measurements are destroyed on ingest** | `panel_data._collapse` groups to one median row per molecule. The raw rows survive only in the opaque cache, and only until it is cleared. |
| **Assay identity is never retrieved** | `_ONLY_FIELDS` in `chembl_client.py` requests `standard_type`, `assay_type`, `document_year` — but **not** `assay_chembl_id`, not the assay description, not the confidence score. |
| **The source release is unrecorded** | `BASE_URL = ".../chembl/api/data"` carries no version; no `provenance.json` is committed. A `Campaign` pins `code_version` and `model_ids` but **not** the data it was built from. |

The third and fourth are the expensive ones.

**Why the missing assay identity is expensive.** The headline caveat in README —
selectivity Spearman falling from 0.682 to 0.462 on the ATP-independent Ki/Kd subset
— is the sharpest known limitation in the project, and the reason it could only be
*bounded* rather than *fixed* is that `standard_type` is the finest instrument
available. Assay-level records would let the same question be asked properly: same
assay, same conditions, same paper.

**Why the missing version is expensive.** This is the same class of bug as STEP 12's
`model_id`: something the system depends on that it does not track. Model identity
was fixed; data identity was not. Today a campaign is **not reproducible** — ChEMBL
updates, the same code sees different data, and nothing anywhere notices.

**Scale.** 37 distinct targets are already known to the repo (20 library + 10 gate
negatives + 3 JAK + 4 PI3K). Extrapolating from the cached pulls, the full corpus is
on the order of **0.5–1 M activity rows** — small. Nothing here is a big-data problem;
it is a *bookkeeping* problem, and the fix is worth doing precisely because it is cheap.

---

## 2. Principles

1. **Store evidence, derive answers.** One row per measurement, forever. The
   collapsed median that `build_isoform_dataset` returns becomes a *view*, not a
   file. Assay-stratified training sets then cost a `WHERE` clause instead of a
   bespoke script.
2. **Ingestion is append-only and immutable.** An activity row is never updated. A
   new ChEMBL release is new rows carrying a new `source_id`. "Which data" then has a
   precise answer, and old campaigns stay reproducible after a refresh.
3. **The database is a derived artifact, not the source of truth.** ChEMBL is. The
   file is gitignored and rebuildable, exactly like `data/models/`.
4. **The deployed app must not need it.** `assets/*.parquet` stays the deploy
   surface: 512 MB ceiling, ~4.5 s cold start, no network. The database is a
   *build-time* concern that **generates** those assets.
5. **The public API does not change.** `panel_data.build_isoform_dataset(panel, iso)`
   keeps its signature and its return frame. Everything above it — regressors, gate,
   AD, conformal, funnel — stays untouched, which is what keeps JAK bit-identical.

---

## 3. Engine

**Recommendation: DuckDB, as a build-time-only dependency.**

| | DuckDB | SQLite | Postgres |
|---|---|---|---|
| new dependency | one pin | **none** (stdlib) | server to run |
| reads/writes parquet natively | **yes** | no | no |
| analytical joins / pivots | **built for it** | adequate | yes |
| fits `assets/` round-trip | **directly** | via pandas | via pandas |
| ops cost | zero | zero | real |

DuckDB wins on the two things this workload actually is: it reads the existing
parquet files **in place** (so migration starts as a query, not a copy), and the
molecule × 37-target pivot that multi-task modelling needs is its native workload.
It is embedded, single-file and server-free, so it costs nothing operationally.

The honest counter-argument is that the repo pins dependencies deliberately and
SQLite is free. At 1 M rows SQLite would work. **If avoiding a dependency matters
more than parquet interop, take SQLite — the schema below is unchanged either way.**

**This is the least important decision in this document.** The schema and the
ingestion discipline are the deliverable; the engine is swappable behind
`src/data/db.py`.

**Deployment:** `duckdb` goes in a `requirements-dev.txt`, not `requirements.txt`.
The Streamlit app never imports it.

---

## 4. Schema

```sql
-- ── provenance ────────────────────────────────────────────────────────────────
CREATE TABLE source (
  source_id      TEXT PRIMARY KEY,   -- 'chembl_34'  — what a campaign pins
  name           TEXT NOT NULL,      -- 'ChEMBL'
  release        TEXT NOT NULL,      -- '34'         — read from /status at fetch time
  retrieved_utc  TIMESTAMP NOT NULL,
  base_url       TEXT
);

CREATE TABLE ingestion (             -- one fetch run; replaces the opaque cache key
  ingestion_id   TEXT PRIMARY KEY,
  source_id      TEXT NOT NULL REFERENCES source,
  target_chembl_id TEXT,
  params_json    TEXT NOT NULL,      -- the exact query, readable instead of hashed
  n_rows         INTEGER,
  code_version   TEXT,
  started_utc    TIMESTAMP
);

-- ── entities ──────────────────────────────────────────────────────────────────
CREATE TABLE target (
  target_chembl_id TEXT PRIMARY KEY,
  pref_name TEXT, organism TEXT, target_type TEXT, gene_symbol TEXT,
  source_id TEXT REFERENCES source
);

CREATE TABLE molecule (
  inchikey        TEXT PRIMARY KEY,  -- of the standardised neutral parent
  parent_smiles   TEXT NOT NULL,     -- src/standardize.py output
  raw_smiles      TEXT,              -- as ChEMBL delivered it
  mol_chembl_id   TEXT,
  mw REAL, logp REAL, tpsa REAL, hbd INT, hba INT, rotb INT, arom_rings INT, qed REAL,
  standardize_version TEXT NOT NULL  -- see §6; the 0.5 % problem lives here
);

CREATE TABLE assay (                 -- NEW: not fetched today
  assay_chembl_id  TEXT PRIMARY KEY,
  assay_type       TEXT,             -- B(inding) / F(unctional) / A / T / P
  description      TEXT,
  confidence_score INT,
  target_chembl_id TEXT REFERENCES target
);

-- ── the evidence table ────────────────────────────────────────────────────────
CREATE TABLE activity (              -- ONE ROW PER MEASUREMENT. never collapsed.
  activity_id     BIGINT PRIMARY KEY,     -- ChEMBL's own id: idempotent re-ingest
  inchikey        TEXT NOT NULL REFERENCES molecule,
  target_chembl_id TEXT NOT NULL REFERENCES target,
  assay_chembl_id TEXT REFERENCES assay,
  standard_type   TEXT,              -- IC50 / Ki / Kd / EC50
  standard_relation TEXT,            -- '=' '>' '<'   ← see below
  standard_value  DOUBLE,
  standard_units  TEXT,
  pchembl_value   DOUBLE,
  document_chembl_id TEXT,
  document_year   INT,
  source_id       TEXT NOT NULL REFERENCES source,
  ingestion_id    TEXT NOT NULL REFERENCES ingestion
);
CREATE INDEX activity_target ON activity(target_chembl_id);
CREATE INDEX activity_mol    ON activity(inchikey);
CREATE INDEX activity_assay  ON activity(assay_chembl_id);

-- ── panels, persisted rather than hardcoded ───────────────────────────────────
CREATE TABLE panel        (panel_name TEXT PRIMARY KEY, target_isoform TEXT, label TEXT);
CREATE TABLE panel_member (panel_name TEXT, isoform TEXT, target_chembl_id TEXT,
                           is_target BOOLEAN, PRIMARY KEY (panel_name, isoform));
```

Three columns are doing real work and deserve naming:

**`activity_id` as the primary key** makes re-ingestion idempotent. Re-pulling a
target inserts nothing new, so an interrupted backfill can simply be re-run — the
same property that made `model_id` worth fixing in STEP 12.

**`standard_relation` reopens a question that was closed.** DESIGN_DECISIONS §1
rejected fetching right-censored `>` values as "sparse and biased… at best a partial
patch, not worth the complexity now." That judgement was made when every fetch had to
be justified by a model that would consume it immediately. With an evidence table,
storing them costs one column and no decision — and the censored-label gap (§1's
defect #4, the reason the binder gate exists at all) becomes something that can be
*re-examined with data* rather than permanently argued from first principles. Storing
is not modelling; nothing downstream changes until someone writes a query.

**`assay_chembl_id` is the one that unblocks the ATP confound.** With it, "same
compound, same assay protocol, different isoform" becomes expressible, which is the
comparison the selectivity gap actually wants to be built from.

### Views — where the current behaviour lives

```sql
-- Reproduces panel_data._collapse exactly.
CREATE VIEW v_pchembl_median AS
SELECT a.target_chembl_id, m.inchikey, m.parent_smiles AS smi,
       median(a.pchembl_value)                                        AS pchembl,
       count(*)                                                       AS n_meas,
       median(a.pchembl_value) FILTER (WHERE a.standard_type IN ('Ki','Kd')) AS pchembl_kikd,
       count(*)                FILTER (WHERE a.standard_type IN ('Ki','Kd')) AS n_kikd,
       min(a.document_year)                                           AS year_first,
       avg(CASE WHEN s.assay_type = 'B' THEN 1.0 ELSE 0.0 END)        AS frac_binding
FROM activity a
JOIN molecule m USING (inchikey)
LEFT JOIN assay s USING (assay_chembl_id)
WHERE a.pchembl_value IS NOT NULL AND a.standard_relation = '='
GROUP BY 1, 2, 3;
```

That view **is** `_collapse`, in SQL. Which is the point: the ATP-independent training
set that STEP 4's audit had to hand-build becomes

```sql
... WHERE a.standard_type IN ('Ki','Kd')          -- assay-independent subset
... WHERE s.confidence_score >= 8                 -- direct single-protein only
... GROUP BY a.assay_chembl_id                    -- within-assay selectivity pairs
```

and the multi-task training matrix multi-task modelling needs is one pivot over
`v_pchembl_median`, across all 37 targets, instead of 37 parquet files with no
common index.

---

## 5. Code layout

| file | responsibility |
|---|---|
| `src/data/db.py` | connection, DDL, schema version + migrations. The only file that knows the engine. |
| `src/data/ingest.py` | fetch → standardise → upsert; writes `source` + `ingestion` rows |
| `src/data/panel_data.py` | **unchanged public API**; reads the view, falls back to bundled parquet |
| `scripts/build_db.py` | backfill all registered targets; export `assets/` parquet |

The fallback in `panel_data` is what protects the deployment: on a machine with no
database (Streamlit Cloud, a fresh clone) it reads the committed parquet exactly as
today. The database is how those parquet files get *made*.

**`Campaign` gains `data_version`** — the set of `source_id`s its training data came
from — and the loop contract carries it at schema 1.2. That closes the reproducibility
hole: a contract will then pin code, models, *and* data.

---

## 6. The migration risk that will bite

`panel_data._canonical` carries this note:

> the committed `assets/jak/*.parquet` were built before this and are ~0.5 %
> un-standardised (measured); rebuilding them here therefore also requires retraining
> the isoform regressors and re-running `scripts/reproduce.sh`

So a DB rebuilt with current standardisation **will not** reproduce the committed JAK
parquet byte-for-byte. About 0.5 % of rows will differ, the training sets will shift,
and **every `model_id` will change — invalidating every exported contract**, exactly
as STEP 12 did.

This must not happen as a side effect of a data-layer refactor. The plan:

1. **Build the DB alongside.** Prove `v_pchembl_median` matches the committed parquet
   on the 99.5 % that is unaffected, and produce an explicit diff of the rest.
2. **Quarantine, don't fix.** Record the standardisation policy per molecule
   (`standardize_version`), so the legacy rows are identifiable rather than silently
   mixed in.
3. **Retrain as its own gated step**, with its own VALIDATION entry, announced as a
   contract-invalidating change — never bundled into the migration commit.

Getting this wrong looks like success: the tests pass, the numbers move slightly, and
nobody notices the JAK screen is no longer the one that was validated.

---

## 7. Build order and gates

| step | delivers | gate |
|---|---|---|
| **D0** schema + `db.py` + ingest one target | the tables exist | `v_pchembl_median` for one JAK isoform equals `assets/jak/JAK1.parquet` on the 99.5 %; the diff is enumerated |
| **D1** backfill all 37 targets; add `assay_chembl_id`, `standard_relation` to the fetch | a queryable corpus | row counts per target match the existing datasets; re-running ingest inserts 0 rows (idempotence) |
| **D2** `data_version` into `Campaign` + contract 1.2 | **reproducibility** | a campaign records its `source_id`s; a contract from a different release fails validation loudly |
| **D3** assay-stratified views | the ATP confound is *askable* | STEP 4's audit reproduces from a query rather than a script |
| **D4** molecule × target matrix view | multi-task training set | the matrix covers all 37 targets with a shared molecule index |

**D0–D2 is the valuable part and is small.** D3 and D4 are the improvements the data
layer exists to enable, and each is a separate piece of work with its own validation.

Throughout: **Gate Z — JAK stays bit-identical.** The existing `model_id` test is the
tripwire, and it must stay green through D0–D4. The moment it goes red, the change
has stopped being a data-layer refactor.

---

## 8. What deliberately stays out

- **The run registry.** Stays JSONL. Different lifetime, human-readable matters, and
  it is already right (§ preamble).
- **Fingerprints and models.** Derived, large, already cached well. A database would
  only make them harder to load.
- **Round score tables.** Parquet beside the registry, as now. They may earn a place
  in the DB once cross-campaign queries are a real need — not before.
- **A general chemistry warehouse.** The scope is the 37 targets this project uses
  and whatever a panel adds. Building for hypothetical breadth is how this becomes a
  six-month project that never returns to the science.

---

## 9. Open questions

1. **Molecule identity: InChIKey or standardised SMILES?** InChIKey is canonical,
   fixed-width and a better key, but it is a second identity system alongside the
   SMILES everything currently passes around. Cheap to decide, annoying to change.
2. **Do `>` censored values get ingested from the start?** Storing them is nearly
   free and reopens the censored-label question; it also roughly doubles the row
   count for some targets. Recommendation: ingest, exclude in the views.
3. **BindingDB and PubChem BioAssay as additional sources.** The schema already
   supports them (`source_id`), but each brings its own duplicate-resolution problem
   against ChEMBL. Worth doing *after* D2, never as part of it.
4. **Should `panel` live in the DB or stay in `src/panels.py`?** The table above
   assumes both — code as the source of truth, DB as a queryable mirror. If they can
   drift, they will; a test should assert they agree.
