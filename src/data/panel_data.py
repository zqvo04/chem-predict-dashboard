"""Per-isoform datasets for selectivity modelling (STEP 2, generalised in STEP 15).

Builds one clean, cached **pchembl-regression** dataset per panel member, plus the
cross-measured join that grounds selectivity validation. Reuses the Phase-1 ChEMBL
client; one **median pchembl per (molecule, isoform)**; unparseable SMILES dropped;
canonicalised so molecules join across isoforms.

There is no active/inactive labelling: the Gate 0 audit showed the inactive class
is nearly empty, so the task is regression and selectivity is a pchembl *gap*
(see VALIDATION.md and DESIGN_DECISIONS.md sections 1-2).

Every function takes a `PanelSpec` (`src/panels.py`), which supplies both the
ChEMBL ids and the cache/bundle directories. This module was `src/data/jak.py`
until STEP 15; the JAK panel resolves to the same `assets/jak/` files it always
did, so the generalisation changed no data.

CLI (build + cache one panel + print the summary table):
    python -m src.data.panel_data [panel]
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from ..panels import DEFAULT_PANEL, PanelSpec, get_panel
from ..standardize import standardize
from . import chembl_client as cc

RDLogger.DisableLog("rdApp.*")

MAX_RECORDS = 40000  # full coverage; pagination does not truncate at this size


def _cached(panel: PanelSpec, filename: str) -> Path | None:
    """Runtime cache first, then the committed copy; None if neither exists."""
    for directory in (panel.data_cache, panel.data_bundled):
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


def build_isoform_dataset(panel: PanelSpec, name: str,
                          use_cache: bool = True) -> pd.DataFrame:
    """Clean, cached median-pchembl dataset for one panel member (smi, pchembl, n_meas)."""
    if name not in panel.chembl_ids:
        raise ValueError(f"Unknown isoform {name!r} for panel {panel.name!r}; "
                         f"expected one of {list(panel.isoforms)}")
    if use_cache:
        cached = _cached(panel, f"{name}.parquet")
        if cached is not None:
            return pd.read_parquet(cached)

    acts = cc.fetch_activities(panel.chembl_ids[name], pchembl_gte=None,
                               max_records=MAX_RECORDS)
    data = _collapse(acts)
    panel.data_cache.mkdir(parents=True, exist_ok=True)
    data.to_parquet(panel.data_cache / f"{name}.parquet", index=False)
    return data


def build_cross_measured(panel: PanelSpec, use_cache: bool = True) -> pd.DataFrame:
    """Molecules measured on *every* panel member (smi + one pchembl column each).

    The inner join is what makes a *measured* gap exist, so it is also what bounds
    how much of the panel can be validated: a panel whose members are rarely
    co-assayed yields a small cross-measured set and, with it, a weak calibration.
    `Campaign` reads the size of this frame to decide a panel's validation tier.
    """
    if use_cache:
        cached = _cached(panel, "cross_measured.parquet")
        if cached is not None:
            return pd.read_parquet(cached)

    frames = [build_isoform_dataset(panel, n, use_cache=use_cache)
                  .set_index("smi")["pchembl"].rename(n) for n in panel.isoforms]
    cross = pd.concat(frames, axis=1, join="inner").reset_index()
    panel.data_cache.mkdir(parents=True, exist_ok=True)
    cross.to_parquet(panel.data_cache / "cross_measured.parquet", index=False)
    return cross


def measured_negatives(panel: PanelSpec) -> pd.DataFrame:
    """Molecules assayed against the panel and found weak; columns inchikey, smi.

    A censored record ("IC50 > x") and no pchembl anywhere in the panel. These are
    the negative class the binder gate has never had: its presumed negatives are
    actives of *other* targets, so it learned to recognise panel chemistry rather
    than binding, and on 2026-08-09 it passed 65.3 % of these at a median
    P(binder) of 0.921.

    The sealed 414 (`censored_library_molecules`) are removed — they are the
    falsification audit's negative arm and must stay out of every training set (G6).
    What is left is the censored population outside the wide library, which is
    mostly panel-programme chemistry rather than off-target chemistry.
    """
    root = panel.root / "assets" / "evidence"
    act = pd.read_parquet(root / "activity.parquet")
    molecule = pd.read_parquet(root / "molecule.parquet").dropna(subset=["parent_smiles"])

    rows = act[act["target_chembl_id"].isin(panel.chembl_ids.values())]
    quantified = set(rows.loc[rows["pchembl_value"].notna(), "inchikey"])
    censored = set(rows.loc[rows["standard_relation"].isin(CENSORED_RELATIONS), "inchikey"])
    sealed = set(censored_library_molecules(panel)["inchikey"])

    usable = (censored - quantified) - sealed
    return (molecule.loc[molecule["inchikey"].isin(usable), ["inchikey", "parent_smiles"]]
            .rename(columns={"parent_smiles": "smi"})
            .drop_duplicates("inchikey")
            .sort_values("inchikey")
            .reset_index(drop=True))


def load_measured_negatives(panel: PanelSpec) -> pd.DataFrame:
    """The sealed measured-negative split; columns inchikey, smi, fold.

    Read-only. `scripts/seal_measured_negatives.py` writes it once, because a
    held-out set that is recomputed per run is not held out — the evidence store
    grows and the evaluation moves with it.
    """
    path = _cached(panel, "measured_negatives.parquet")
    if path is None:
        raise FileNotFoundError(
            f"No sealed measured-negative split for panel {panel.name!r}. "
            "Run: python scripts/seal_measured_negatives.py " + panel.name)
    return pd.read_parquet(path)


EVAL_TIME_CUT = 2020        # STATE.md section 4c: the cut the model beats a lookup on


def year_first(panel: PanelSpec) -> pd.Series:
    """smi -> earliest year any panel member published a quantified measurement.

    The committed per-isoform parquets carry only (smi, pchembl, n_meas): the
    `year_first` column `_collapse` produces never made it into the bundle. The
    evidence store holds the same provenance offline, so provenance-dated work
    reads it from there rather than forcing a network rebuild.
    """
    root = panel.root / "assets" / "evidence"
    act = pd.read_parquet(root / "activity.parquet")
    molecule = pd.read_parquet(root / "molecule.parquet")
    rows = act[act["target_chembl_id"].isin(panel.chembl_ids.values())
               & act["pchembl_value"].notna()]
    year = rows.groupby("inchikey")["document_year"].min()
    joined = molecule[["inchikey", "parent_smiles"]].join(year, on="inchikey")
    return joined.dropna(subset=["document_year"]).set_index("parent_smiles")["document_year"]


CENSORED_RELATIONS = (">", ">=")


def same_document_cross_measured(panel: PanelSpec) -> pd.DataFrame:
    """교차측정 세트를 한 논문 안에서 만든 것 (ROADMAP P4.1).

    `build_cross_measured`는 아이소폼마다 모든 문헌에 걸친 중앙값을 취하므로, 한 분자의
    선택성 gap이 서로 다른 논문·서로 다른 ATP 농도에서 나온 두 값의 차이일 수 있다.
    이 함수는 분자마다 **패널 레코드를 가장 많이 담은 단일 문헌** — 동수면 문헌 id가 작은
    쪽, 그래서 결정적 — 을 고르고 그 문헌 안의 아이소폼별 중앙값만 쓴다.

    컬럼은 `build_cross_measured`와 같은 `smi + 아이소폼별 한 컬럼`이라, 두 프레임 모두
    `selectivity.evaluate_split`에 그대로 들어간다. 그것이 비교를 가능하게 하는 조건이다.
    """
    root = panel.root / "assets" / "evidence"
    act = pd.read_parquet(root / "activity.parquet")
    molecule = pd.read_parquet(root / "molecule.parquet")[["inchikey", "parent_smiles"]]

    by_target = {cid: iso for iso, cid in panel.chembl_ids.items()}
    rows = act[act["target_chembl_id"].isin(by_target)
               & act["pchembl_value"].notna()].copy()
    rows["isoform"] = rows["target_chembl_id"].map(by_target)

    grouped = rows.groupby(["inchikey", "document_chembl_id"])
    per_doc = pd.DataFrame({"n_iso": grouped["isoform"].nunique(),
                            "n_rec": grouped.size()}).reset_index()
    covered = per_doc[per_doc["n_iso"] == len(panel.isoforms)]
    if covered.empty:
        return pd.DataFrame(columns=["smi", *panel.isoforms])

    chosen = (covered.sort_values(["inchikey", "n_rec", "document_chembl_id"],
                                 ascending=[True, False, True])
                     .drop_duplicates("inchikey")[["inchikey", "document_chembl_id"]])

    picked = rows.merge(chosen, on=["inchikey", "document_chembl_id"])
    wide = (picked.groupby(["inchikey", "isoform"])["pchembl_value"]
                  .median().unstack("isoform"))
    out = (wide.join(molecule.drop_duplicates("inchikey").set_index("inchikey"))
               .rename(columns={"parent_smiles": "smi"})
               .dropna()
               .reset_index(drop=True))
    return out[["smi", *panel.isoforms]].drop_duplicates("smi").reset_index(drop=True)


def evidence_isoform_frame(panel: PanelSpec, isoform: str) -> pd.DataFrame:
    """증거 저장소에서 유도한 한 아이소폼의 (smi, pchembl_kikd, year_first).

    `build_isoform_dataset`은 네트워크 재빌드 시 이 두 컬럼을 만들지만, 커밋된 JAK 번들은
    그 이전 스키마라 `(smi, pchembl, n_meas)`뿐이다. 배포 자산 재빌드는 G0 사건이므로,
    assay 종류나 출판 연도가 필요한 감사는 같은 provenance를 이쪽에서 읽는다 —
    `year_first`가 이미 만든 선례와 같다.

    `year_first`는 **모든** assay 종류에 걸친 최초 연도다. 분자가 문헌에 등장한 시점이
    질문이지, Ki로 측정된 시점이 아니다.
    """
    root = panel.root / "assets" / "evidence"
    act = pd.read_parquet(root / "activity.parquet")
    molecule = pd.read_parquet(root / "molecule.parquet")[["inchikey", "parent_smiles"]]

    rows = act[(act["target_chembl_id"] == panel.chembl_ids[isoform])
               & act["pchembl_value"].notna()]
    kikd = (rows[rows["standard_type"].isin(EQUILIBRIUM_TYPES)]
            .groupby("inchikey")["pchembl_value"].median().rename("pchembl_kikd"))
    year = rows.groupby("inchikey")["document_year"].min().rename("year_first")

    out = (pd.concat([year, kikd], axis=1)
             .join(molecule.drop_duplicates("inchikey").set_index("inchikey")))
    return (out.dropna(subset=["parent_smiles", "year_first"])
               .rename(columns={"parent_smiles": "smi"})
               .reset_index(drop=True)[["smi", "pchembl_kikd", "year_first"]])


def censored_library_molecules(panel: PanelSpec) -> pd.DataFrame:
    """Library molecules whose only panel measurement is a censored non-binding.

    One row per (molecule, isoform) that has a censored record and no pchembl for
    that molecule anywhere in the panel. `pchembl_upper` is the **weakest** claim
    among that molecule's records for the isoform — the largest IC50 quoted, hence
    the loosest bound — so a falsification counted against it is a lower bound.

    This is the negative arm of the falsification audit and one of the counts the
    suitability screen reports, so it lives here with the other per-panel datasets
    rather than in whichever script needed it first.
    """
    root = panel.root / "assets" / "evidence"
    act = pd.read_parquet(root / "activity.parquet")
    member = pd.read_parquet(root / "library_member.parquet")
    molecule = pd.read_parquet(root / "molecule.parquet")

    isoform_of = {cid: iso for iso, cid in panel.chembl_ids.items()}
    rows = act[act["target_chembl_id"].isin(isoform_of)].copy()
    rows["isoform"] = rows["target_chembl_id"].map(isoform_of)

    # A molecule with a pchembl on *any* panel member is in the training data for
    # that member, so it is not a clean external test even where another isoform
    # only censored it.
    quantified = set(rows.loc[rows["pchembl_value"].notna(), "inchikey"])

    censored = rows[
        rows["standard_relation"].isin(CENSORED_RELATIONS)
        & rows["pchembl_value"].isna()
        & rows["standard_value"].notna()
        & (rows["standard_units"] == "nM")     # the other 1 % of units are not worth converting
        & ~rows["inchikey"].isin(quantified)
    ].copy()
    censored["pchembl_upper"] = 9.0 - np.log10(censored["standard_value"])

    druglike = member.loc[member["druglike"], ["inchikey"]]
    sealed = (censored.merge(druglike, on="inchikey")
              .groupby(["inchikey", "isoform"], as_index=False)["pchembl_upper"].max()
              .merge(molecule[["inchikey", "parent_smiles"]], on="inchikey")
              .rename(columns={"parent_smiles": "smi"}))
    return sealed.sort_values(["inchikey", "isoform"]).reset_index(drop=True)


def load_eval_split(panel: PanelSpec) -> pd.DataFrame:
    """The evaluation split sealed at round 0 (G9); columns smi, year_first, fold.

    Read-only on purpose. `scripts/seal_eval_split.py` writes it once and refuses to
    overwrite, because a split that can be recomputed is not sealed — the training
    set grows from round to round and the split must not move with it.
    """
    path = _cached(panel, "eval_split.parquet")
    if path is None:
        raise FileNotFoundError(
            f"No sealed evaluation split for panel {panel.name!r}. "
            "Run: python scripts/seal_eval_split.py " + panel.name)
    return pd.read_parquet(path)


def summary(panel: PanelSpec, use_cache: bool = True) -> pd.DataFrame:
    """Per-isoform count + pchembl distribution table."""
    rows = []
    for name in panel.isoforms:
        d = build_isoform_dataset(panel, name, use_cache=use_cache)["pchembl"]
        rows.append({"isoform": name, "n_molecules": len(d),
                     "pchembl_min": round(float(d.min()), 2),
                     "pchembl_median": round(float(d.median()), 2),
                     "pchembl_max": round(float(d.max()), 2)})
    tbl = pd.DataFrame(rows)
    tbl.attrs["n_cross_measured"] = len(build_cross_measured(panel, use_cache=use_cache))
    return tbl


def _write_provenance(panel: PanelSpec) -> None:
    prov = {"built": date.today().isoformat(), "panel": panel.name,
            "targets": panel.chembl_ids, "max_records": MAX_RECORDS,
            "n_molecules": {n: int(len(build_isoform_dataset(panel, n)))
                            for n in panel.isoforms},
            "n_cross_measured": int(len(build_cross_measured(panel)))}
    panel.data_cache.mkdir(parents=True, exist_ok=True)
    (panel.data_cache / "provenance.json").write_text(json.dumps(prov, indent=2))


def _main() -> None:
    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    tbl = summary(panel)
    print(f"Panel {panel.name}: {panel.target} vs {'/'.join(panel.offs)}")
    print(tbl.to_string(index=False))
    print(f"\n{len(panel.isoforms)}-way cross-measured: {tbl.attrs['n_cross_measured']}")
    _write_provenance(panel)
    print(f"Cached -> {panel.data_cache}")


if __name__ == "__main__":
    _main()
