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
from rdkit import RDLogger

from ..standardize import standardize
from . import chembl_client as cc

RDLogger.DisableLog("rdApp.*")

# Canonical human single-protein ChEMBL targets (confirmed in Gate 0).
TARGETS = {"JAK1": "CHEMBL2835", "JAK2": "CHEMBL2971", "JAK3": "CHEMBL2148"}
MAX_RECORDS = 40000  # full coverage; pagination does not truncate at this size

_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _ROOT / "data" / "jak"          # runtime cache, gitignored
BUNDLED_DIR = _ROOT / "assets" / "jak"      # committed, so a fresh deploy skips retrieval


def _cached(filename: str) -> Path | None:
    """Runtime cache first, then the committed copy; None if neither exists."""
    for directory in (CACHE_DIR, BUNDLED_DIR):
        path = directory / filename
        if path.exists():
            return path
    return None


def _canonical(smiles: str) -> str | None:
    """Neutral parent form — the same standardisation every query path applies.

    NOTE: the committed `assets/jak/*.parquet` were built before this and are
    ~0.5 % un-standardised (measured); rebuilding them here therefore also
    requires retraining the isoform regressors and re-running
    `scripts/reproduce.sh`, since the training sets shift slightly.
    """
    return standardize(smiles)


# Assay types whose pchembl is an ATP-concentration-independent constant. An IC50
# for an ATP-competitive kinase inhibitor shifts with the assay's ATP level (and
# the JAK isoforms do not share an ATP Km), so a gap built from IC50s measured
# under different conditions carries an assay artefact. Ki/Kd do not have that
# problem, which is what makes them the control subset.
EQUILIBRIUM_TYPES = ("Ki", "Kd")

_COLUMNS = ["smi", "pchembl", "n_meas", "pchembl_kikd", "n_kikd",
            "year_first", "frac_binding"]


def _collapse(activities: pd.DataFrame) -> pd.DataFrame:
    """Raw activities -> one median-pchembl row per canonical molecule.

    Columns: smi, pchembl, n_meas, plus the assay provenance needed to ask whether
    a result is an artefact of how it was measured:

      pchembl_kikd  median over Ki/Kd records only (NaN when the molecule has none)
      n_kikd        how many of its measurements were Ki/Kd
      year_first    earliest publication year — the molecule's arrival date, which
                    is what a time split has to cut on
      frac_binding  fraction measured in a binding (biochemical) assay rather than
                    a functional (typically cellular) one

    Unparseable SMILES and non-numeric pchembl are dropped. n_meas records how many
    measurements the median was taken over (provenance for noise-awareness).
    """
    if activities.empty:
        return pd.DataFrame(columns=_COLUMNS)
    df = activities.copy()
    df["pchembl"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["canonical_smiles", "pchembl"])
    df["smi"] = df["canonical_smiles"].map(_canonical)
    df = df.dropna(subset=["smi"])

    # Provenance fields are optional: an activities frame cached before they were
    # requested simply yields NaN rather than failing the build.
    def _col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(pd.NA, index=df.index)

    df["_kikd"] = df["pchembl"].where(_col("standard_type").isin(EQUILIBRIUM_TYPES))
    df["_year"] = pd.to_numeric(_col("document_year"), errors="coerce")
    df["_binding"] = (_col("assay_type") == "B").astype(float).where(
        _col("assay_type").notna())

    out = (df.groupby("smi", sort=False)
             .agg(pchembl=("pchembl", "median"), n_meas=("pchembl", "size"),
                  pchembl_kikd=("_kikd", "median"), n_kikd=("_kikd", "count"),
                  year_first=("_year", "min"), frac_binding=("_binding", "mean"))
             .reset_index())
    return out


def build_isoform_dataset(name: str, use_cache: bool = True) -> pd.DataFrame:
    """Clean, cached median-pchembl dataset for one isoform (smi, pchembl, n_meas)."""
    if name not in TARGETS:
        raise ValueError(f"Unknown isoform {name!r}; expected one of {list(TARGETS)}")
    if use_cache:
        cached = _cached(f"{name}.parquet")
        if cached is not None:
            return pd.read_parquet(cached)

    acts = cc.fetch_activities(TARGETS[name], pchembl_gte=None, max_records=MAX_RECORDS)
    data = _collapse(acts)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(CACHE_DIR / f"{name}.parquet", index=False)
    return data


def build_cross_measured(use_cache: bool = True) -> pd.DataFrame:
    """Molecules measured on all three isoforms (smi, JAK1, JAK2, JAK3 pchembl)."""
    if use_cache:
        cached = _cached("cross_measured.parquet")
        if cached is not None:
            return pd.read_parquet(cached)

    frames = [build_isoform_dataset(n, use_cache=use_cache)
                  .set_index("smi")["pchembl"].rename(n) for n in TARGETS]
    cross = pd.concat(frames, axis=1, join="inner").reset_index()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cross.to_parquet(CACHE_DIR / "cross_measured.parquet", index=False)
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
