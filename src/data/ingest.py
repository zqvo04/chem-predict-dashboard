"""Ingestion into the evidence store: fetch -> standardise -> idempotent upsert (STEP 16).

Every write records **where it came from**: the ChEMBL release in `source`, the exact
query in `ingestion`. That is the hole this closes — until now a `Campaign` pinned
`code_version` and `model_ids` but not its data, so re-running it after a ChEMBL
refresh silently trained on something else.

Idempotence is keyed on ChEMBL's own `activity_id`, so re-ingesting a target inserts
nothing new and an interrupted backfill is simply re-run. Nothing here mutates: a new
release is new rows under a new `source_id`, and old campaigns keep resolving to the
data they were built on.

CLI (backfill the registered panels and the wide library):
    python -m src.data.ingest              # every registered panel
    python -m src.data.ingest jak          # one panel
    python -m src.data.ingest --library    # the wide screening library
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, QED

from ..standardize import STANDARDIZE_VERSION, standardize_mol
from . import chembl_client as cc
from . import db as _db

RDLogger.DisableLog("rdApp.*")

LIBRARY_ID = "chembl20panel"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_source(con, release: str | None = None) -> str:
    """Register the live ChEMBL release, returning its `source_id`."""
    release = release or cc.release()
    source_id = release.lower()                      # 'chembl_37'
    con.execute(
        "INSERT OR IGNORE INTO source VALUES (?, ?, ?, ?, ?)",
        [source_id, "ChEMBL", release, _now(), cc.BASE_URL])
    return source_id


def _molecule_rows(smiles: pd.Series) -> pd.DataFrame:
    """Standardise unique SMILES to parent + InChIKey + cached physchem.

    InChIKey is the primary key rather than the SMILES string: it is canonical and
    fixed-width, and it is the identifier a second source (BindingDB, PubChem) could
    ever be joined on. The parent SMILES is kept alongside because that is what the
    featuriser and every existing dataset actually consume.
    """
    rows = []
    for raw in pd.unique(smiles.dropna()):
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            continue
        parent = standardize_mol(mol)
        if parent is None or parent.GetNumAtoms() == 0:
            continue
        try:
            key = Chem.MolToInchiKey(parent)
        except Exception:                            # RDKit InChI can fail on exotica
            continue
        if not key:
            continue
        rows.append({
            "inchikey": key,
            "parent_smiles": Chem.MolToSmiles(parent),
            "raw_smiles": raw,
            "mol_chembl_id": None,
            "mw": float(Descriptors.MolWt(parent)),
            "logp": float(Descriptors.MolLogP(parent)),
            "tpsa": float(Descriptors.TPSA(parent)),
            "hbd": int(Lipinski.NumHDonors(parent)),
            "hba": int(Lipinski.NumHAcceptors(parent)),
            "rotb": int(Descriptors.NumRotatableBonds(parent)),
            "arom_rings": int(Descriptors.NumAromaticRings(parent)),
            "qed": float(QED.qed(parent)),
            "standardize_version": STANDARDIZE_VERSION,
        })
    return pd.DataFrame(rows)


def _upsert(con, table: str, df: pd.DataFrame, columns: list[str]) -> int:
    """Insert rows, skipping any whose primary key is already present.

    Columns are named explicitly on both sides rather than relying on `SELECT *`
    matching the DDL's order: a positional insert keeps working after a column is
    added or reordered, and writes the values into the wrong fields when it does.
    That is a corruption that no test would catch until the numbers looked odd.
    """
    if df.empty:
        return 0
    before = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    cols = ", ".join(columns)
    con.register("_payload", df[columns])
    con.execute(f"INSERT OR IGNORE INTO {table} ({cols}) SELECT {cols} FROM _payload")
    con.unregister("_payload")
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] - before


_MOL_COLUMNS = ["inchikey", "parent_smiles", "raw_smiles", "mol_chembl_id", "mw",
                "logp", "tpsa", "hbd", "hba", "rotb", "arom_rings", "qed",
                "standardize_version"]
_ACT_COLUMNS = ["activity_id", "inchikey", "target_chembl_id", "assay_chembl_id",
                "standard_type", "standard_relation", "standard_value",
                "standard_units", "pchembl_value", "document_chembl_id",
                "document_year", "source_id", "ingestion_id"]


def ingest_target(con, target_chembl_id: str, source_id: str | None = None,
                  max_records: int = 60000, use_cache: bool = True) -> dict:
    """Fetch one target's activities and write them as evidence rows."""
    from ..loop_contract import code_version

    source_id = source_id or ensure_source(con)
    acts = cc.fetch_activities_evidence(target_chembl_id, max_records=max_records,
                                        use_cache=use_cache)
    if acts.empty:
        return {"target": target_chembl_id, "fetched": 0, "activities": 0,
                "molecules": 0, "assays": 0}

    params = {"target_chembl_id": target_chembl_id, "max_records": max_records,
              "fields": "evidence"}
    ingestion_id = hashlib.sha1(
        json.dumps({**params, "source": source_id}, sort_keys=True).encode()
    ).hexdigest()[:16]
    con.execute("INSERT OR IGNORE INTO ingestion VALUES (?, ?, ?, ?, ?, ?, ?)",
                [ingestion_id, source_id, target_chembl_id, json.dumps(params, sort_keys=True),
                 int(len(acts)), code_version(), _now()])

    # target
    con.execute("INSERT OR IGNORE INTO target VALUES (?, ?, ?, ?, ?)",
                [target_chembl_id, None, None, None, source_id])

    # molecules — standardise the unique structures only
    mols = _molecule_rows(acts["canonical_smiles"])
    n_mol = _upsert(con, "molecule", mols, _MOL_COLUMNS)
    key_of = dict(zip(mols["raw_smiles"], mols["inchikey"])) if not mols.empty else {}

    # assays
    if "assay_chembl_id" in acts.columns:
        assays = (acts[["assay_chembl_id", "assay_type", "assay_description"]]
                  .dropna(subset=["assay_chembl_id"]).drop_duplicates("assay_chembl_id")
                  .rename(columns={"assay_description": "description"}))
        assays["target_chembl_id"] = target_chembl_id
        n_assay = _upsert(con, "assay", assays,
                          ["assay_chembl_id", "assay_type", "description",
                           "target_chembl_id"])
    else:
        n_assay = 0

    # activities
    a = acts.copy()
    a["inchikey"] = a["canonical_smiles"].map(key_of)
    a = a.dropna(subset=["inchikey", "activity_id"])
    a["target_chembl_id"] = target_chembl_id
    a["source_id"] = source_id
    a["ingestion_id"] = ingestion_id
    for col in ("standard_value", "pchembl_value"):
        a[col] = pd.to_numeric(a.get(col), errors="coerce")
    a["document_year"] = pd.to_numeric(a.get("document_year"), errors="coerce").astype("Int64")
    a["activity_id"] = pd.to_numeric(a["activity_id"], errors="coerce").astype("int64")
    for col in ("assay_chembl_id", "standard_type", "standard_relation",
                "standard_units", "document_chembl_id"):
        if col not in a.columns:
            a[col] = None
    # One activity_id can appear twice in a paginated pull; the PK would reject the
    # batch rather than the duplicate, so collapse here.
    a = a.drop_duplicates("activity_id")
    n_act = _upsert(con, "activity", a, _ACT_COLUMNS)

    # Counted with SQL's semantics (NULL is not '<> ='), so this agrees with what
    # `db.summary` reports rather than quietly differing by the null-relation rows.
    rel = acts.get("standard_relation")
    censored = int((rel.notna() & (rel != "=")).sum()) if rel is not None else 0
    return {"target": target_chembl_id, "fetched": int(len(acts)), "activities": n_act,
            "molecules": n_mol, "assays": n_assay, "censored": censored}


def ingest_panel(con, panel, source_id: str | None = None, use_cache: bool = True) -> list[dict]:
    source_id = source_id or ensure_source(con)
    return [ingest_target(con, cid, source_id, use_cache=use_cache)
            for cid in panel.chembl_ids.values()]


def ingest_library(con, source_id: str | None = None, use_cache: bool = True) -> dict:
    """Register the wide screening library's molecules in the store.

    The library is target-agnostic and shared by every campaign, so it is its own
    table rather than a panel's. Membership is recorded by InChIKey, which is what
    lets "is this library molecule also a training molecule?" — the leakage question
    STEP 13 had to answer with a bespoke SMILES join — become a plain SQL join.
    """
    from .library import load_library

    source_id = source_id or ensure_source(con)
    lib = load_library(use_cache=use_cache)
    con.execute("INSERT OR IGNORE INTO library VALUES (?, ?, ?, ?)",
                [LIBRARY_ID, "Bioactive drug-like molecules from 20 diverse ChEMBL targets",
                 source_id, _now()])

    mols = _molecule_rows(lib["smi"])
    n_mol = _upsert(con, "molecule", mols, _MOL_COLUMNS)
    key_of = dict(zip(mols["raw_smiles"], mols["inchikey"])) if not mols.empty else {}

    members = pd.DataFrame({
        "library_id": LIBRARY_ID,
        "inchikey": lib["smi"].map(key_of),
        "source_target": lib["source_target"] if "source_target" in lib.columns else None,
        "druglike": lib["druglike"] if "druglike" in lib.columns else None,
    }).dropna(subset=["inchikey"]).drop_duplicates("inchikey")
    n_member = _upsert(con, "library_member", members,
                       ["library_id", "inchikey", "source_target", "druglike"])
    return {"library": LIBRARY_ID, "molecules": n_mol, "members": n_member,
            "library_rows": int(len(lib))}


def _main() -> None:
    from ..panels import PANELS, get_panel

    args = [a for a in sys.argv[1:]]
    do_library = "--library" in args
    names = [a for a in args if not a.startswith("--")]

    with _db.session() as con:
        source_id = ensure_source(con)
        print(f"source: {source_id}")
        for name in (names or list(PANELS)):
            panel = get_panel(name)
            print(f"\npanel {panel.name}")
            for row in ingest_panel(con, panel, source_id):
                print(f"  {row['target']:14} fetched={row['fetched']:6d} "
                      f"+activities={row['activities']:6d} +molecules={row['molecules']:6d} "
                      f"+assays={row['assays']:5d} censored={row.get('censored', 0)}")
        if do_library:
            print("\nlibrary")
            print(" ", ingest_library(con, source_id))
        print("\nstore:", _db.summary(con))


if __name__ == "__main__":
    _main()
