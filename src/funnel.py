"""STEP 7: the tiered wide screen (Stage B) + the SELECT export.

Runs the diverse wide library down the cost funnel and returns a shortlist that is
**selective and in-domain**, each row carrying everything the loop contract needs:

  Tier 0  Ro5 + PAINS                         (near-free, drop gross liabilities)
  Tier 1  per-isoform regressors -> gap S     (cheap; rank, keep the top band)
  Tier 2  conformal interval + applicability  (on survivors only; keep in-domain)

The expensive per-molecule work (AD nearest-neighbour, intervals) runs only on
Tier-1 survivors, so the funnel economics are real. `score_molecules` exposes the
same scoring for re-scoring a small set (the Stage-A loop closure), and
`screen_to_contract` turns a user's picks into the versioned loop-contract dict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .applicability import in_domain, load_reference
from .conformal import DEFAULT_ALPHA, halfwidth
from .data import jak
from .data.library import load_library
from .filters.druglikeness import apply_druglikeness
from .loop_contract import build_contract, model_id
from .models.features import morgan_matrix
from .models.isoform_regressor import train_and_cache
from .selectivity import OFFS, POTENCY_FLOOR, TARGET


def _quiet(_message: str) -> None:
    """Default progress sink — the CLI and tests don't report steps."""


def _context(target: str, offs: tuple[str, ...], use_cache: bool, step=_quiet):
    isoforms = [target, *offs]
    models = {}
    for iso in isoforms:
        step(f"Loading the {iso} regressor")
        models[iso] = train_and_cache(iso, use_cache=use_cache).model
    q = {}
    for iso in isoforms:
        step(f"Calibrating the {iso} conformal interval")
        q[iso] = halfwidth(iso, use_cache=use_cache)
    refs = {iso: load_reference(iso, use_cache=use_cache) for iso in isoforms}
    return isoforms, models, q, refs


def _predict(df: pd.DataFrame, models, isoforms, target, offs) -> pd.DataFrame:
    """Tier 1 (cheap): per-isoform predictions, gap S, potency floor."""
    X, mask = morgan_matrix(df["smi"].tolist())
    df = df[mask].reset_index(drop=True)
    for iso in isoforms:
        df[f"pred_{iso}"] = models[iso].predict(X)
    df["gap"] = df[f"pred_{target}"] - df[[f"pred_{o}" for o in offs]].max(axis=1)
    df["meets_floor"] = df[f"pred_{target}"] >= POTENCY_FLOOR
    return df


def _trust(df: pd.DataFrame, q, refs, isoforms, target, offs) -> pd.DataFrame:
    """Tier 2 (pricier, survivors only): conformal intervals + applicability domain."""
    if df.empty:
        return df
    worst_off = df[[f"pred_{o}" for o in offs]].idxmax(axis=1).str.replace("pred_", "")
    for iso in isoforms:
        df[f"lo_{iso}"] = df[f"pred_{iso}"] - q[iso]
        df[f"hi_{iso}"] = df[f"pred_{iso}"] + q[iso]
        df[f"in_domain_{iso}"] = in_domain(df["smi"].tolist(), reference=refs[iso])["in_domain"]
    half = q[target] + np.array([q[o] for o in worst_off])
    df["gap_lo"] = df["gap"] - half
    df["gap_hi"] = df["gap"] + half
    df["in_domain"] = np.logical_and.reduce([df[f"in_domain_{iso}"] for iso in isoforms])
    df["verdict"] = np.where(df["in_domain"], "in_domain", "uncertain")
    return df


def score_molecules(smiles: list[str], target: str = TARGET, offs: tuple[str, ...] = OFFS,
                    use_cache: bool = True) -> pd.DataFrame:
    """Full Tier-1+Tier-2 scoring of a given (small) set — used for Stage-A re-scoring."""
    isoforms, models, q, refs = _context(target, offs, use_cache)
    df = _predict(pd.DataFrame({"smi": list(smiles)}), models, isoforms, target, offs)
    return _trust(df, q, refs, isoforms, target, offs)


def screen_library(target: str = TARGET, offs: tuple[str, ...] = OFFS,
                   library: pd.DataFrame | None = None, tier1_keep: int = 300,
                   shortlist: int = 60, use_cache: bool = True, on_step=None) -> pd.DataFrame:
    """Run the wide library down the funnel; return the ranked selective+in-domain shortlist.

    `on_step(message)`, when given, is called before each stage so a caller (the
    dashboard) can report honest progress on a run that takes minutes.
    """
    step = on_step or _quiet
    isoforms, models, q, refs = _context(target, offs, use_cache, step)
    step("Loading the wide library")
    lib = load_library(use_cache=use_cache) if library is None else library

    # Tier 0 depends on the library alone, never on the models, so the cached
    # library already carries its verdict; only an ad-hoc library needs the run.
    precomputed = "druglike" in lib.columns
    step(f"Tier 0 — Ro5 + PAINS over {len(lib)} molecules"
         + (" (carried by the library cache)" if precomputed else ""))
    df = lib if precomputed else apply_druglikeness(lib, smiles_col="smi")
    df = df[df["druglike"]].reset_index(drop=True)

    step(f"Tier 1 — per-isoform prediction + gap S over {len(df)} drug-like molecules")
    df = _predict(df, models, isoforms, target, offs)          # Tier 1
    df = df[df["meets_floor"]].sort_values("gap", ascending=False).head(tier1_keep).reset_index(drop=True)

    step(f"Tier 2 — conformal interval + applicability domain over {len(df)} survivors")
    df = _trust(df, q, refs, isoforms, target, offs)           # Tier 2 (survivors only)
    if df.empty:
        return df
    return df.sort_values("gap", ascending=False).head(shortlist).reset_index(drop=True)


def current_model_ids(target: str = TARGET, offs: tuple[str, ...] = OFFS,
                      use_cache: bool = True) -> dict[str, str]:
    """The pinned id of each deployed isoform model (for the contract / Stage-A guard)."""
    return {iso: model_id(jak.TARGETS[iso], train_and_cache(iso, use_cache=use_cache).model)
            for iso in [target, *offs]}


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
