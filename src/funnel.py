"""STEP 7: the tiered wide screen (Stage B) + the SELECT export.

Runs the diverse wide library down the cost funnel and returns a shortlist that is
**selective and in-domain**, each row carrying everything the loop contract needs:

  Tier 0  Ro5 + PAINS                         (near-free, drop gross liabilities)
  Tier 1  per-isoform regressors -> gap S     (cheap; rank, keep the top band)
  Tier 2  conformal interval + applicability  (on survivors only; keep in-domain)

The expensive per-molecule work (AD nearest-neighbour, intervals) runs only on
Tier-1 survivors, so the funnel economics are real. `screen_to_contract` turns a
user's picks into the versioned loop-contract dict for the deep dive.

CLI (screen the library, print the shortlist head):
    python -m src.funnel
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .applicability import in_domain
from .conformal import DEFAULT_ALPHA, halfwidth
from .data import jak
from .data.library import load_library
from .filters.druglikeness import apply_druglikeness
from .loop_contract import build_contract, model_id
from .models.features import morgan_matrix
from .models.isoform_regressor import train_and_cache
from .selectivity import OFFS, POTENCY_FLOOR, TARGET


def screen_library(target: str = TARGET, offs: tuple[str, ...] = OFFS,
                   library: pd.DataFrame | None = None, tier1_keep: int = 300,
                   shortlist: int = 60, use_cache: bool = True) -> pd.DataFrame:
    """Run the wide library down the funnel; return the ranked selective+in-domain shortlist."""
    isoforms = [target, *offs]
    models = {iso: train_and_cache(iso, use_cache=use_cache).model for iso in isoforms}
    q = {iso: halfwidth(iso, use_cache=use_cache) for iso in isoforms}
    train = {iso: jak.build_isoform_dataset(iso, use_cache=use_cache)["smi"].tolist()
             for iso in isoforms}
    lib = load_library(use_cache=use_cache) if library is None else library

    # Tier 0: drug-likeness
    df = apply_druglikeness(lib, smiles_col="smi")
    df = df[df["druglike"]].reset_index(drop=True)

    # Tier 1: per-isoform predictions -> gap S, potency floor, rank, keep top band
    X, mask = morgan_matrix(df["smi"].tolist())
    df = df[mask].reset_index(drop=True)
    for iso in isoforms:
        df[f"pred_{iso}"] = models[iso].predict(X)
    off_max = df[[f"pred_{o}" for o in offs]].max(axis=1)
    df["gap"] = df[f"pred_{target}"] - off_max
    df["meets_floor"] = df[f"pred_{target}"] >= POTENCY_FLOOR
    df = df[df["meets_floor"]].sort_values("gap", ascending=False).head(tier1_keep).reset_index(drop=True)
    if df.empty:
        return df

    # Tier 2: conformal intervals + applicability domain (survivors only)
    worst_off = df[[f"pred_{o}" for o in offs]].idxmax(axis=1).str.replace("pred_", "")
    for iso in isoforms:
        df[f"lo_{iso}"] = df[f"pred_{iso}"] - q[iso]
        df[f"hi_{iso}"] = df[f"pred_{iso}"] + q[iso]
        df[f"in_domain_{iso}"] = in_domain(df["smi"].tolist(), train[iso])["in_domain"]
    q_off = np.array([q[o] for o in worst_off])
    half = q[target] + q_off
    df["gap_lo"] = df["gap"] - half
    df["gap_hi"] = df["gap"] + half
    df["in_domain"] = np.logical_and.reduce([df[f"in_domain_{iso}"] for iso in isoforms])
    df["verdict"] = np.where(df["in_domain"], "in_domain", "uncertain")

    return df.sort_values("gap", ascending=False).head(shortlist).reset_index(drop=True)


def current_model_ids(target: str = TARGET, offs: tuple[str, ...] = OFFS,
                      use_cache: bool = True) -> dict[str, str]:
    """The pinned id of each deployed isoform model (for the contract / Stage-A guard)."""
    ids = {}
    for iso in [target, *offs]:
        bundle = train_and_cache(iso, use_cache=use_cache)
        ids[iso] = model_id(jak.TARGETS[iso], bundle.model)
    return ids


def screen_to_contract(picks: pd.DataFrame, target: str = TARGET,
                       offs: tuple[str, ...] = OFFS, alpha: float = DEFAULT_ALPHA) -> dict:
    """Turn the user's selected shortlist rows into a loop-contract dict."""
    return build_contract(picks, target, list(offs),
                          current_model_ids(target, offs), alpha)


def _main() -> None:
    sl = screen_library()
    cols = ["smi", f"pred_{TARGET}", "gap", "gap_lo", "gap_hi", "in_domain", "verdict"]
    print(f"Shortlist: {len(sl)} selective + drug-like candidates "
          f"({int(sl['in_domain'].sum())} in-domain)")
    print(sl[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    _main()
