"""JAK per-isoform datasets for selectivity modelling (STEP 2).

Builds one clean, cached **pchembl-regression** dataset per JAK isoform, plus the
cross-measured join that grounds selectivity validation. Reuses the Phase-1 ChEMBL
client; one **median pchembl per (molecule, isoform)**; unparseable SMILES dropped;
canonicalised so molecules join across isoforms.

There is no active/inactive labelling: the Gate 0 audit showed the inactive class
is nearly empty, so the task is regression and selectivity is a pchembl *gap*
(see VALIDATION.md and DESIGN_DECISIONS.md sections 1-2).

CLI (build + cache all three + print the summary table):
    python -m src.data.jak
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from . import chembl_client as cc

RDLogger.DisableLog("rdApp.*")

# Canonical human single-protein ChEMBL targets (confirmed in Gate 0).
TARGETS = {"JAK1": "CHEMBL2835", "JAK2": "CHEMBL2971", "JAK3": "CHEMBL2148"}
MAX_RECORDS = 40000  # full coverage; pagination does not truncate at this size

_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _ROOT / "data" / "jak"


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Chem.MolToSmiles(mol) if mol else None


def _collapse(activities: pd.DataFrame) -> pd.DataFrame:
    """Raw activities -> one median-pchembl row per canonical molecule.

    Columns: smi, pchembl, n_meas. Unparseable SMILES and non-numeric pchembl are
    dropped. n_meas records how many measurements the median was taken over
    (provenance for downstream noise-awareness).
    """
    if activities.empty:
        return pd.DataFrame(columns=["smi", "pchembl", "n_meas"])
    df = activities.copy()
    df["pchembl"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["canonical_smiles", "pchembl"])
    df["smi"] = df["canonical_smiles"].map(_canonical)
    df = df.dropna(subset=["smi"])
    out = (df.groupby("smi", sort=False)
             .agg(pchembl=("pchembl", "median"), n_meas=("pchembl", "size"))
             .reset_index())
    return out


def build_isoform_dataset(name: str, use_cache: bool = True) -> pd.DataFrame:
    """Clean, cached median-pchembl dataset for one isoform (smi, pchembl, n_meas)."""
    if name not in TARGETS:
        raise ValueError(f"Unknown isoform {name!r}; expected one of {list(TARGETS)}")
    path = CACHE_DIR / f"{name}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)

    acts = cc.fetch_activities(TARGETS[name], pchembl_gte=None, max_records=MAX_RECORDS)
    data = _collapse(acts)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)
    return data


def build_cross_measured(use_cache: bool = True) -> pd.DataFrame:
    """Molecules measured on all three isoforms (smi, JAK1, JAK2, JAK3 pchembl)."""
    path = CACHE_DIR / "cross_measured.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)

    frames = [build_isoform_dataset(n, use_cache=use_cache)
                  .set_index("smi")["pchembl"].rename(n) for n in TARGETS]
    cross = pd.concat(frames, axis=1, join="inner").reset_index()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cross.to_parquet(path, index=False)
    return cross


def summary(use_cache: bool = True) -> pd.DataFrame:
    """Per-isoform count + pchembl distribution table."""
    rows = []
    for name in TARGETS:
        d = build_isoform_dataset(name, use_cache=use_cache)["pchembl"]
        rows.append({"isoform": name, "n_molecules": len(d),
                     "pchembl_min": round(float(d.min()), 2),
                     "pchembl_median": round(float(d.median()), 2),
                     "pchembl_max": round(float(d.max()), 2)})
    tbl = pd.DataFrame(rows)
    tbl.attrs["n_cross_measured"] = len(build_cross_measured(use_cache=use_cache))
    return tbl


def _write_provenance() -> None:
    prov = {"built": date.today().isoformat(),
            "targets": TARGETS, "max_records": MAX_RECORDS,
            "n_molecules": {n: int(len(build_isoform_dataset(n))) for n in TARGETS},
            "n_cross_measured": int(len(build_cross_measured()))}
    (CACHE_DIR / "provenance.json").write_text(json.dumps(prov, indent=2))


def _main() -> None:
    tbl = summary()
    print(tbl.to_string(index=False))
    print(f"\n3-way cross-measured: {tbl.attrs['n_cross_measured']}")
    _write_provenance()
    print(f"Cached -> {CACHE_DIR}")


if __name__ == "__main__":
    _main()
