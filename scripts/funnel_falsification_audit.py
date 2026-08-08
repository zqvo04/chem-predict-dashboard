#!/usr/bin/env python3
"""0': the deployed funnel scored against measured answers, both directions.

STATE.md section 2 recorded that 91.3 % of the funnel's potency predictions on 414
measured non-binders exceed the molecule's own measured upper bound — and flagged
itself as having no reproduction script, which is why the number never reached
VALIDATION.md. This is that script.

It runs TWO arms, because one is not interpretable on its own. STATE.md section 2
says so in its own words: "전부 음성 샘플이므로 이것은 위양성률이지 재현율이 아니다."
A cascade scored only against non-binders always improves by tightening, so
conservatism is indistinguishable from accuracy without the other side.

  NEGATIVE ARM — 414 measured non-binders inside the wide library.
      Censored ChEMBL records ("IC50 > x") with no pchembl anywhere in the panel.
      Constructed from the committed evidence store, never trained on (they carry
      no pchembl, so `build_isoform_dataset` excludes them by construction), and
      sealed to a committed file so the docking gate (G8) and every later round
      score the identical set.

  POSITIVE ARM — measured actives published after the time cut.
      Answers the question the negative arm cannot: how many real actives does
      the cascade throw away? The deployed models trained on these molecules, so
      running them would measure memorisation. The regressors are therefore
      refit on the pre-cut data only, through the injection seam, and the actives
      are scored by models that never saw them.

The arms differ in scope on purpose. The negative arm covers every tier because
the deployed models are legitimately blind to those molecules. The positive arm
covers Tier 1 only: an honest Tier 0.5 / Tier 2 recall needs the binder gate and
the AD reference refit on the same cut, and neither takes injected data yet.

    python scripts/funnel_falsification_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import funnel                                             # noqa: E402
from src.data import panel_data                                    # noqa: E402
from src.models.isoform_regressor import train_and_cache           # noqa: E402
from src.panels import DEFAULT_PANEL                               # noqa: E402
from src.selectivity import POTENCY_FLOOR                          # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = _ROOT / "assets" / "evidence"
SEALED = DEFAULT_PANEL.data_bundled / "sealed_negatives.parquet"

CENSORED_RELATIONS = (">", ">=")
TIME_CUT = 2020            # STATE.md section 5: the 2020 cut leaves 1,525 molecules
ACTIVE_PCHEMBL = 6.0       # matches POTENCY_FLOOR: "active" on the funnel's own scale


# --------------------------------------------------------------------------- #
# The sealed negative set
# --------------------------------------------------------------------------- #

def build_sealed() -> pd.DataFrame:
    """The library molecules whose only panel measurement is a censored non-binding.

    One row per (molecule, isoform) with a censored record and no pchembl for that
    molecule anywhere in the panel. `pchembl_upper` is the **weakest** claim among
    that molecule's records for the isoform — the largest IC50 quoted, hence the
    loosest bound — so a falsification counted against it is a lower bound on the
    real rate, exactly as STATE.md section 2 describes.
    """
    act = pd.read_parquet(EVIDENCE / "activity.parquet")
    member = pd.read_parquet(EVIDENCE / "library_member.parquet")
    molecule = pd.read_parquet(EVIDENCE / "molecule.parquet")

    isoform_of = {cid: iso for iso, cid in DEFAULT_PANEL.chembl_ids.items()}
    panel_rows = act[act["target_chembl_id"].isin(isoform_of)].copy()
    panel_rows["isoform"] = panel_rows["target_chembl_id"].map(isoform_of)

    # A molecule with a pchembl on *any* panel member is in the training data for
    # that member, so it is not a clean external test even where another isoform
    # only censored it.
    quantified = set(panel_rows.loc[panel_rows["pchembl_value"].notna(), "inchikey"])

    censored = panel_rows[
        panel_rows["standard_relation"].isin(CENSORED_RELATIONS)
        & panel_rows["pchembl_value"].isna()
        & panel_rows["standard_value"].notna()
        & (panel_rows["standard_units"] == "nM")          # the other 1 % of units are not worth converting
        & ~panel_rows["inchikey"].isin(quantified)
    ].copy()
    censored["pchembl_upper"] = 9.0 - np.log10(censored["standard_value"])

    druglike = member.loc[member["druglike"], ["inchikey"]]
    sealed = (censored.merge(druglike, on="inchikey")
              .groupby(["inchikey", "isoform"], as_index=False)["pchembl_upper"].max()
              .merge(molecule[["inchikey", "parent_smiles"]], on="inchikey")
              .rename(columns={"parent_smiles": "smi"}))
    return sealed.sort_values(["inchikey", "isoform"]).reset_index(drop=True)


def load_sealed() -> pd.DataFrame:
    """The sealed set, written on first run and read verbatim afterwards (G6/G8).

    Rebuilding it on every run would let the population drift with the evidence
    store, and a docking configuration committed against a moving target is not
    the pre-registration G8 asks for.
    """
    if SEALED.exists():
        return pd.read_parquet(SEALED)
    sealed = build_sealed()
    SEALED.parent.mkdir(parents=True, exist_ok=True)
    sealed.to_parquet(SEALED, index=False)
    print(f"  sealed {sealed['inchikey'].nunique()} molecules -> {SEALED}")
    return sealed


# --------------------------------------------------------------------------- #
# Negative arm
# --------------------------------------------------------------------------- #

def negative_arm() -> None:
    print("=" * 78)
    print("NEGATIVE ARM — measured non-binders, scored by the deployed funnel")
    print("=" * 78)

    sealed = load_sealed()
    smiles = sealed[["inchikey", "smi"]].drop_duplicates("inchikey")
    print(f"\n  molecules {len(smiles)}   (molecule, isoform) bounds {len(sealed)}")

    scored = funnel.score_molecules(smiles["smi"].tolist())
    scored["inchikey"] = smiles["inchikey"].to_numpy()[: len(scored)]

    target = DEFAULT_PANEL.target
    pred_long = scored.melt(
        id_vars=["inchikey"],
        value_vars=[f"pred_{iso}" for iso in DEFAULT_PANEL.isoforms],
        var_name="isoform", value_name="pred")
    pred_long["isoform"] = pred_long["isoform"].str.removeprefix("pred_")
    paired = sealed.merge(pred_long, on=["inchikey", "isoform"])
    paired["exceeds"] = paired["pred"] > paired["pchembl_upper"]

    per_mol = paired.groupby("inchikey")["exceeds"].any()
    over = paired.loc[paired["exceeds"], "pred"] - paired.loc[paired["exceeds"], "pchembl_upper"]
    print(f"\n  FALSIFICATION RATE  {per_mol.sum()} / {len(per_mol)} = {per_mol.mean():.1%}"
          "   (any isoform's prediction exceeds that isoform's measured upper bound)")
    print(f"  median exceedance   {over.median():.2f} log")

    # Restricted to the target isoform, which is what STATE.md section 2 reported
    # before this script existed. Kept alongside rather than instead of: a molecule
    # falsified on JAK2 is still a falsified prediction.
    on_target = paired[paired["isoform"] == target]
    ot_over = on_target.loc[on_target["exceeds"], "pred"] - \
        on_target.loc[on_target["exceeds"], "pchembl_upper"]
    print(f"  {target} only        {int(on_target['exceeds'].sum())} / {len(on_target)}"
          f" = {on_target['exceeds'].mean():.1%},  median exceedance {ot_over.median():.2f} log")
    print(f"  median predicted {target}  {scored[f'pred_{target}'].median():.2f}"
          f"   vs median measured upper bound  {sealed['pchembl_upper'].median():.2f}")

    gate = funnel._gate(DEFAULT_PANEL, True)
    n = len(scored)
    print(f"\n  Per-tier pass rate, each measured on all {n} independently:")
    print(f"    Tier 0.5  binder gate (P >= {gate.threshold:.2f}) "
          f"{int(scored['is_binder'].sum()):>5} = {scored['is_binder'].mean():.1%}")
    print(f"    Tier 1    potency floor (pred {target} >= {POTENCY_FLOOR}) "
          f"{int(scored['meets_floor'].sum()):>5} = {scored['meets_floor'].mean():.1%}")
    print(f"    Tier 2    in-domain (of those reaching it) "
          f"{int(scored['in_domain'].sum()):>5} = {scored['in_domain'].mean():.1%}")

    cascade = scored[scored["is_binder"] & scored["meets_floor"]]
    print(f"\n  Cascaded, as the screen actually runs:"
          f"\n    gate -> floor  {len(cascade)}"
          f"\n    + in-domain    {int(cascade['in_domain'].sum())}   <- what reaches a shortlist")


# --------------------------------------------------------------------------- #
# Positive arm
# --------------------------------------------------------------------------- #

def _year_first() -> pd.Series:
    """smi -> earliest year any panel member published a quantified measurement.

    The committed `assets/jak/*.parquet` carry only (smi, pchembl, n_meas) — the
    `year_first` column `_collapse` produces never made it into the bundle, so
    `scripts/assay_time_audit.py` needs a network rebuild to run. The evidence
    store holds the same provenance offline, so the cut is taken from there.
    """
    act = pd.read_parquet(EVIDENCE / "activity.parquet")
    molecule = pd.read_parquet(EVIDENCE / "molecule.parquet")
    panel_rows = act[act["target_chembl_id"].isin(DEFAULT_PANEL.chembl_ids.values())
                     & act["pchembl_value"].notna()]
    year = panel_rows.groupby("inchikey")["document_year"].min()
    joined = molecule[["inchikey", "parent_smiles"]].join(year, on="inchikey")
    return joined.dropna(subset=["document_year"]).set_index("parent_smiles")["document_year"]


def _cut(data: pd.DataFrame, year: pd.Series) -> pd.DataFrame:
    """`data` with a `year_first` column, dropping the rows provenance cannot date."""
    out = data.copy()
    out["year_first"] = out["smi"].map(year)
    return out.dropna(subset=["year_first"])


def positive_arm() -> None:
    print()
    print("=" * 78)
    print(f"POSITIVE ARM — actives first published after {TIME_CUT}, Tier 1 only")
    print("=" * 78)

    target = DEFAULT_PANEL.target
    year = _year_first()
    frames = {iso: _cut(panel_data.build_isoform_dataset(DEFAULT_PANEL, iso), year)
              for iso in DEFAULT_PANEL.isoforms}

    models = {}
    for iso, data in frames.items():
        pre = data[data["year_first"] <= TIME_CUT]
        models[iso] = train_and_cache(DEFAULT_PANEL, iso, use_cache=False, data=pre).model
        print(f"  {iso}: refit on {len(pre)} pre-{TIME_CUT} molecules "
              f"(deployed model uses all {len(data)})")

    post = frames[target][frames[target]["year_first"] > TIME_CUT]
    actives = post[post["pchembl"] >= ACTIVE_PCHEMBL].reset_index(drop=True)
    if actives.empty:
        print("\n  No post-cut actives; nothing to report.")
        return

    scored = funnel._predict(actives[["smi", "pchembl"]].copy(), models,
                             list(DEFAULT_PANEL.isoforms), target, DEFAULT_PANEL.offs)
    kept = scored["meets_floor"]
    print(f"\n  post-{TIME_CUT} measured actives (pchembl >= {ACTIVE_PCHEMBL})  {len(scored)}")
    print(f"  Tier 1 potency floor keeps  {int(kept.sum())} = {kept.mean():.1%}")
    print(f"  THE FLOOR DISCARDS          {int((~kept).sum())} = {(~kept).mean():.1%} of real actives")
    lost = scored.loc[~kept, "pchembl"]
    if len(lost):
        print(f"    their measured pchembl: median {lost.median():.2f}, max {lost.max():.2f}")
    print("\n  Read against the negative arm's Tier 1 pass rate. A floor that lets"
          "\n  most non-binders through AND discards real actives is not conservative,"
          "\n  it is uninformative on this axis.")


def main() -> None:
    negative_arm()
    positive_arm()


if __name__ == "__main__":
    main()
