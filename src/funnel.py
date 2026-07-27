"""STEP 7: the tiered wide screen (Stage B) + the SELECT export.

Runs the diverse wide library down the cost funnel and returns a shortlist that is
**selective and in-domain**, each row carrying everything the loop contract needs:

  Tier 0  Ro5 + PAINS                         (near-free, drop gross liabilities)
  Tier 1  per-isoform regressors -> gap S     (cheap; rank, keep the top band)
  Tier 2  conformal interval + applicability  (on survivors only; keep in-domain)
          + MPO properties (QED / solubility / tox alert)

The expensive per-molecule work (AD nearest-neighbour, intervals) runs only on
Tier-1 survivors, so the funnel economics are real. `score_molecules` exposes the
same scoring for re-scoring a small set (the Stage-A loop closure), and
`screen_to_contract` turns a user's picks into the versioned loop-contract dict.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .applicability import in_domain, load_reference
from .conformal import DEFAULT_ALPHA, gap_halfwidth, halfwidth
from .data.library import load_library
from .filters.druglikeness import apply_druglikeness
from .loop_contract import build_contract, model_id
from .models.binder_gate import train_and_cache as train_gate
from .models.features import iter_morgan_batches
from .models.isoform_regressor import train_and_cache
from .mpo import annotate as mpo_annotate
from .panels import DEFAULT_PANEL, PanelSpec
from .selectivity import POTENCY_FLOOR


GAP_DIST_NAME = "gap_distribution.npz"


def _gap_dist_paths(panel: PanelSpec) -> tuple[Path, Path]:
    """(runtime cache, committed bundle) for this panel's gap-percentile reference.

    The library is target-agnostic and stays shared, but this distribution is that
    library scored *through one panel's models*, so it belongs to the panel — it
    moved from `assets/library/` to `assets/<panel>/` in STEP 15. The bytes did not
    change: the provenance guard inside still pins the same model ids, so the
    committed JAK file remains valid where it stands.
    """
    return panel.data_cache / GAP_DIST_NAME, panel.data_bundled / GAP_DIST_NAME


def _quiet(_message: str) -> None:
    """Default progress sink — the CLI and tests don't report steps."""


@lru_cache(maxsize=4)
def _gate(panel: PanelSpec = DEFAULT_PANEL, use_cache: bool = True):
    """The deployed binder gate (Tier 0.5) for a panel, loaded once per process."""
    return train_gate(panel, use_cache=use_cache)


def _context(panel: PanelSpec, use_cache: bool, step=_quiet):
    isoforms = list(panel.isoforms)
    models = {}
    for iso in isoforms:
        step(f"Loading the {iso} regressor")
        models[iso] = train_and_cache(panel, iso, use_cache=use_cache).model
    q = {}
    for iso in isoforms:
        step(f"Calibrating the {iso} conformal interval")
        q[iso] = halfwidth(panel, iso, use_cache=use_cache)
    refs = {iso: load_reference(panel, iso, use_cache=use_cache) for iso in isoforms}
    return isoforms, models, q, refs


def _predict(df: pd.DataFrame, models, isoforms, target, offs) -> pd.DataFrame:
    """Tier 1 (cheap): per-isoform predictions, gap S, potency floor.

    Featurises and scores in batches: one slice of fingerprints is built, all three
    isoform models score it, and it is released before the next slice. Holding the
    whole library's fingerprints at once cost ~113 MB and was the largest single
    contributor to the deployed app's memory ceiling.
    """
    masks, preds = [], {iso: [] for iso in isoforms}
    for X, mask in iter_morgan_batches(df["smi"].tolist()):
        masks.append(mask)
        if X.shape[0]:
            for iso in isoforms:
                preds[iso].append(models[iso].predict(X))

    mask = np.concatenate(masks) if masks else np.zeros(0, dtype=bool)
    df = df[mask].reset_index(drop=True)
    if df.empty:
        # Nothing survived an upstream filter (e.g. the binder gate rejected the
        # whole set): return the empty frame carrying the Tier-1 columns rather than
        # crash. Callers already handle an empty shortlist.
        for iso in isoforms:
            df[f"pred_{iso}"] = pd.Series(dtype=float)
        df["gap"] = pd.Series(dtype=float)
        df["meets_floor"] = pd.Series(dtype=bool)
        return df
    for iso in isoforms:
        df[f"pred_{iso}"] = np.concatenate(preds[iso])
    df["gap"] = df[f"pred_{target}"] - df[[f"pred_{o}" for o in offs]].max(axis=1)
    df["meets_floor"] = df[f"pred_{target}"] >= POTENCY_FLOOR
    return df


def _trust(df: pd.DataFrame, q, refs, isoforms, panel: PanelSpec) -> pd.DataFrame:
    """Tier 2 (pricier, survivors only): conformal intervals + applicability domain + MPO."""
    if df.empty:
        return df
    for iso in isoforms:
        df[f"lo_{iso}"] = df[f"pred_{iso}"] - q[iso]
        df[f"hi_{iso}"] = df[f"pred_{iso}"] + q[iso]
        ad = in_domain(df["smi"].tolist(), reference=refs[iso])
        df[f"in_domain_{iso}"] = ad["in_domain"]
        df[f"nn_sim_{iso}"] = ad["nn_sim"]     # feeds the gap interval below

    # The gap interval is calibrated directly against the *measured* gap and scaled
    # by how far the molecule sits from training, rather than summing the two
    # isoform half-widths. Summing assumed independent adverse errors; measured,
    # they correlate at +0.65 and largely cancel, so it delivered 99.7% coverage at
    # a 90% nominal level and crossed zero for the entire shortlist. Calibrating on
    # the gap fixes the width but not its *shape*: one constant width covers only
    # 46% of the low-similarity molecules that make up most of a wide screen, which
    # is why it is scaled by nn-similarity rather than left flat (STEP 14).
    nn_min = np.minimum.reduce([df[f"nn_sim_{iso}"].to_numpy() for iso in isoforms])
    half = gap_halfwidth(nn_min, panel)
    df["gap_halfwidth"] = half
    df["gap_lo"] = df["gap"] - half
    df["gap_hi"] = df["gap"] + half
    df["in_domain"] = np.logical_and.reduce([df[f"in_domain_{iso}"] for iso in isoforms])
    df["verdict"] = np.where(df["in_domain"], "in_domain", "uncertain")
    # Selectivity is not the only thing that disqualifies a molecule: annotate the
    # survivors with the cheap property axes so an insoluble or toxicophore-bearing
    # candidate is visible as such. The ranking below stays on the gap.
    props = mpo_annotate(df["smi"].tolist())
    for column in props.columns:
        df[column] = props[column].to_numpy()
    return df


def _annotate_binder(df: pd.DataFrame, panel: PanelSpec, use_cache: bool) -> pd.DataFrame:
    """Attach the Tier-0.5 binder-gate probability and verdict (no filtering here)."""
    if df.empty:
        return df
    gate = _gate(panel, use_cache)
    df["binder_prob"] = gate.predict_proba(df["smi"].tolist())
    df["is_binder"] = df["binder_prob"] >= gate.threshold
    return df


def score_molecules(smiles: list[str], panel: PanelSpec = DEFAULT_PANEL,
                    use_cache: bool = True) -> pd.DataFrame:
    """Full Tier-0.5+Tier-1+Tier-2 scoring of a given (small) set — used for Stage-A re-scoring.

    The binder gate is reported as a column, not applied as a filter: a caller scoring
    one molecule (or a handful of analogues) wants the non-binder verdict shown, not the
    row silently dropped. `screen_library` is where the gate prunes.
    """
    isoforms, models, q, refs = _context(panel, use_cache)
    df = _predict(pd.DataFrame({"smi": list(smiles)}), models, isoforms,
                  panel.target, panel.offs)
    df = _trust(df, q, refs, isoforms, panel)
    return _annotate_binder(df, panel, use_cache)


def _gap_distribution_provenance(panel: PanelSpec, use_cache: bool) -> str:
    """What a cached gap distribution must match to still be valid.

    The distribution is a function of the library, the three regressors and the
    gate threshold. Pinning the model ids (stable behavioural digests since
    STEP 12) means a retrain invalidates the cache automatically instead of
    silently ranking against a stale reference.
    """
    return json.dumps({"models": current_model_ids(panel, use_cache),
                       "gate": round(float(_gate(panel, use_cache).threshold), 6),
                       "offs": list(panel.offs)}, sort_keys=True)


def _compute_gap_distribution(panel: PanelSpec, use_cache: bool) -> np.ndarray:
    isoforms, models, _, _ = _context(panel, use_cache)
    lib = load_library(use_cache=use_cache)
    lib = lib[lib["druglike"]].reset_index(drop=True) if "druglike" in lib.columns else lib
    # Percentile is against the population the funnel actually ranks — plausible
    # binders — so a molecule the gate would reject is not part of the reference.
    gate = _gate(panel, use_cache)
    lib = lib[gate.predict_proba(lib["smi"].tolist()) >= gate.threshold].reset_index(drop=True)
    scored = _predict(lib, models, isoforms, panel.target, panel.offs)
    return np.sort(scored["gap"].to_numpy())


@lru_cache(maxsize=4)
def library_gap_distribution(panel: PanelSpec = DEFAULT_PANEL,
                             use_cache: bool = True) -> np.ndarray:
    """Sorted Tier-1 gap S over the gate-clearing library — the percentile reference.

    A lone gap value means little on its own, but the molecule's *position among
    thousands of others* is what the gap model was validated on (Spearman 0.80,
    4.5x enrichment), so the single-molecule view leads with the percentile.

    Computing it means screening the whole library — ~8.7 s — which the
    single-molecule view was paying on every cold process just to place one
    molecule that itself scores in 0.3 s. It depends on nothing the user types, so
    it is precomputed and shipped, guarded by a provenance string so a retrained
    model or a moved gate threshold rebuilds it rather than ranking against a stale
    reference.
    """
    want = _gap_distribution_provenance(panel, use_cache)
    cache_path, bundled_path = _gap_dist_paths(panel)
    if use_cache:
        for path in (cache_path, bundled_path):
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    if str(z["provenance"]) == want:
                        return z["gaps"].astype(float)

    gaps = _compute_gap_distribution(panel, use_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, gaps=gaps.astype(np.float32), provenance=want)
    return gaps


def gap_percentile(gap: float, panel: PanelSpec = DEFAULT_PANEL,
                   use_cache: bool = True) -> float:
    """Where a gap sits in the library distribution, as a percentile (0-100)."""
    dist = library_gap_distribution(panel, use_cache)
    return float(np.searchsorted(dist, gap) / len(dist) * 100.0)


def screen_library(panel: PanelSpec = DEFAULT_PANEL,
                   library: pd.DataFrame | None = None, tier1_keep: int = 300,
                   shortlist: int = 60, use_cache: bool = True, on_step=None) -> pd.DataFrame:
    """Run the wide library down the funnel; return the ranked selective+in-domain shortlist.

    `on_step(message)`, when given, is called before each stage so a caller (the
    dashboard) can report honest progress on a run that takes minutes.
    """
    step = on_step or _quiet
    isoforms, models, q, refs = _context(panel, use_cache, step)
    step("Loading the wide library")
    lib = load_library(use_cache=use_cache) if library is None else library

    # Tier 0 depends on the library alone, never on the models, so the cached
    # library already carries its verdict; only an ad-hoc library needs the run.
    precomputed = "druglike" in lib.columns
    step(f"Tier 0 — Ro5 + PAINS over {len(lib)} molecules"
         + (" (carried by the library cache)" if precomputed else ""))
    df = lib if precomputed else apply_druglikeness(lib, smiles_col="smi")
    df = df[df["druglike"]].reset_index(drop=True)

    # Tier 0.5 — the binder gate prunes molecules the regressors would only score at
    # their training mean (ethanol, off-target junk), before the gap is ever computed.
    gate = _gate(panel, use_cache)
    step(f"Tier 0.5 — binder gate over {len(df)} drug-like molecules "
         f"(pass P(binder) ≥ {gate.threshold:.2f})")
    df["binder_prob"] = gate.predict_proba(df["smi"].tolist())
    df = df[df["binder_prob"] >= gate.threshold].reset_index(drop=True)

    step(f"Tier 1 — per-isoform prediction + gap S over {len(df)} plausible binders")
    df = _predict(df, models, isoforms, panel.target, panel.offs)   # Tier 1
    df = df[df["meets_floor"]].sort_values("gap", ascending=False).head(tier1_keep).reset_index(drop=True)

    step(f"Tier 2 — conformal interval + applicability domain over {len(df)} survivors")
    df = _trust(df, q, refs, isoforms, panel)                  # Tier 2 (survivors only)
    if df.empty:
        return df
    return df.sort_values("gap", ascending=False).head(shortlist).reset_index(drop=True)


def current_model_ids(panel: PanelSpec = DEFAULT_PANEL,
                      use_cache: bool = True) -> dict[str, str]:
    """The pinned id of each deployed isoform model (for the contract / Stage-A guard)."""
    return {iso: model_id(panel.chembl_ids[iso],
                          train_and_cache(panel, iso, use_cache=use_cache).model)
            for iso in panel.isoforms}


def screen_to_contract(picks: pd.DataFrame, panel: PanelSpec = DEFAULT_PANEL,
                       alpha: float = DEFAULT_ALPHA, campaign_id: str | None = None) -> dict:
    """Turn the user's selected shortlist rows into a loop-contract dict."""
    return build_contract(picks, panel.target, list(panel.offs),
                          current_model_ids(panel), alpha, campaign_id=campaign_id)


def _main() -> None:
    import sys

    from .panels import get_panel

    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    sl = screen_library(panel)
    cols = ["smi", f"pred_{panel.target}", "gap", "gap_lo", "gap_hi", "binder_prob",
            "in_domain", "verdict", "mpo"]
    print(f"Shortlist: {len(sl)} selective + drug-like candidates "
          f"({int(sl['in_domain'].sum())} in-domain)")
    print(sl[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    _main()
