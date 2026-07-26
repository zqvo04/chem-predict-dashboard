"""Presumed-inactive negative set for the JAK binder gate (STEP 10 / P1).

The per-isoform regressors are trained only on *quantified* JAK pchembl, so they
never see a non-binder. On anything off-domain a tree ensemble reverts to its
training mean (~6.3 pchembl), so ethanol scores as a 760 nM JAK1 "inhibitor" and
83 % of a diverse target-agnostic library clears the potency floor — the floor
gates nothing. Worse, the selectivity gap `S` between three shrunk-to-mean
predictions is a model artefact, not chemistry.

DESIGN_DECISIONS section 1 anticipated exactly this and kept the fix in reserve:
"A DUD-E-style decoy class for a genuine binder/non-binder gate is kept in reserve
as an optional future addition." The funnel's live output is what made it
necessary. This module builds the missing negative class.

**Presumed inactive by absence.** Real, drug-like molecules measured active on
*other* ChEMBL targets, carrying no JAK record, are taken as presumed JAK
non-binders. They are real chemistry (not invented decoys) and match the
deployment scenario — a wide library *is* a set of molecules with no known JAK
activity.

**Physchem-matched.** The negatives are subsampled to match the JAK actives' joint
(MW, logP) distribution, so a classifier trained on them cannot separate the
classes on molecular size or lipophilicity alone — it is forced toward JAK SAR
rather than a gross-property shortcut (the standard decoy-bias guard). The residual
label noise — a promiscuous other-kinase inhibitor that would in fact hit JAK but
has no JAK record — is real and bounded: known cross-actives are removed by the JAK
exclusion, and the two kinase targets are a minority of the basket.

CLI (build + cache + bundle-ready parquet):
    python -m src.data.negatives
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from ..standardize import standardize
from . import chembl_client as cc
from . import jak

RDLogger.DisableLog("rdApp.*")

# A diverse, drug-like basket of *other* targets: two kinases (hard negatives that
# look like JAK inhibitors but are not) and eight non-kinases across proteases,
# metalloenzymes, GPCRs and an ion channel (easy negatives that catch trivial and
# grossly off-target chemistry). Verified SINGLE PROTEIN / Homo sapiens.
NEGATIVE_TARGETS = {
    "EGFR": "CHEMBL203",          # kinase (hard)
    "CDK2": "CHEMBL301",          # kinase (hard)
    "Thrombin": "CHEMBL204",
    "CarbonicAnhydrase2": "CHEMBL205",
    "DopamineD2": "CHEMBL217",
    "COX2": "CHEMBL230",
    "Acetylcholinesterase": "CHEMBL220",
    "hERG": "CHEMBL240",
    "HistamineH3": "CHEMBL264",
    "CannabinoidCB1": "CHEMBL218",
}
POSITIVE_PCHEMBL = 6.0    # a JAK molecule at >= this is a "binder" positive
MAX_RECORDS = 4000        # per other-target active pull
MATCH_BINS = 12           # quantile bins per axis for the (MW, logP) match
MATCH_SEED = 0

_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _ROOT / "data" / "jak"          # runtime cache, gitignored
BUNDLED_DIR = _ROOT / "assets" / "jak"      # committed, so a fresh deploy skips retrieval


def jak_positive_smiles(use_cache: bool = True) -> set[str]:
    """Standardised SMILES of every JAK molecule measured at pchembl >= 6 (any isoform)."""
    mols: set[str] = set()
    for iso in jak.TARGETS:
        d = jak.build_isoform_dataset(iso, use_cache=use_cache)
        mols |= set(d.loc[d["pchembl"] >= POSITIVE_PCHEMBL, "smi"])
    return mols


def _mw_logp(smiles: list[str]) -> np.ndarray:
    """(MW, logP) for each SMILES; NaN row where the SMILES does not parse."""
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        rows.append([Descriptors.MolWt(mol), Descriptors.MolLogP(mol)]
                    if mol is not None else [np.nan, np.nan])
    return np.array(rows, dtype=float)


def _matched_indices(pos_desc: np.ndarray, neg_desc: np.ndarray,
                     bins: int = MATCH_BINS, seed: int = MATCH_SEED) -> np.ndarray:
    """Indices into the negative pool whose (MW, logP) matches the positive shape.

    Bin both classes on the positives' quantile edges; per bin, draw as many
    negatives as there are positives (capped by what the pool holds). The kept
    negatives therefore mirror the positive joint (MW, logP) histogram — the match
    that denies the classifier a size/lipophilicity shortcut.
    """
    def edges(col: np.ndarray) -> np.ndarray:
        e = np.quantile(col, np.linspace(0, 1, bins + 1))
        e[0], e[-1] = -np.inf, np.inf
        return e

    mw_e, lp_e = edges(pos_desc[:, 0]), edges(pos_desc[:, 1])

    def cell(desc: np.ndarray) -> np.ndarray:
        return (np.digitize(desc[:, 0], mw_e[1:-1]) * bins
                + np.digitize(desc[:, 1], lp_e[1:-1]))

    pos_cell, neg_cell = cell(pos_desc), cell(neg_desc)
    pos_per_cell = np.bincount(pos_cell, minlength=bins * bins)
    rng = np.random.default_rng(seed)

    kept: list[int] = []
    for c in range(bins * bins):
        pool = np.where(neg_cell == c)[0]
        take = min(len(pool), int(pos_per_cell[c]))
        if take:
            kept.extend(rng.choice(pool, size=take, replace=False).tolist())
    return np.array(sorted(kept), dtype=int)


def build_negatives(use_cache: bool = True) -> pd.DataFrame:
    """Physchem-matched presumed-inactive negatives (columns: smi, source_target).

    Fetches the other-target actives, standardises to neutral parents, drops any
    molecule present in a JAK dataset, deduplicates, then matches the survivors to
    the JAK actives' (MW, logP) distribution.
    """
    if use_cache:
        for directory in (CACHE_DIR, BUNDLED_DIR):
            path = directory / "negatives.parquet"
            if path.exists():
                return pd.read_parquet(path)

    jak_pos = jak_positive_smiles(use_cache=use_cache)
    frames, skipped = [], []
    for name, cid in NEGATIVE_TARGETS.items():
        # One target's transient API failure must not discard the other nine fetches;
        # the basket is deliberately larger than it needs to be for exactly this.
        try:
            acts = cc.fetch_activities(cid, pchembl_gte=POSITIVE_PCHEMBL,
                                       max_records=MAX_RECORDS, use_cache=use_cache)
        except (RuntimeError, OSError):
            skipped.append(name)
            continue
        cand = cc.to_candidates(acts)
        if cand.empty:
            continue
        smi = cand["canonical_smiles"].map(standardize)
        frames.append(pd.DataFrame({"smi": smi, "source_target": name}).dropna(subset=["smi"]))

    if len(frames) < 3:
        raise RuntimeError(
            f"Only {len(frames)} of {len(NEGATIVE_TARGETS)} negative targets fetched "
            f"(skipped {skipped}); too few for a physchem-matched negative set.")
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[~pool["smi"].isin(jak_pos)]
    # First occurrence keeps one source label per molecule; a molecule active on
    # several other targets is still one presumed-inactive example.
    pool = pool.drop_duplicates(subset="smi").reset_index(drop=True)

    pos_desc = _mw_logp(sorted(jak_pos))
    pos_desc = pos_desc[~np.isnan(pos_desc).any(axis=1)]
    neg_desc = _mw_logp(pool["smi"].tolist())
    valid = ~np.isnan(neg_desc).any(axis=1)
    pool, neg_desc = pool[valid].reset_index(drop=True), neg_desc[valid]

    kept = _matched_indices(pos_desc, neg_desc)
    negatives = pool.iloc[kept].reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    negatives.to_parquet(CACHE_DIR / "negatives.parquet", index=False)
    return negatives


def _main() -> None:
    neg = build_negatives(use_cache=False)
    pos = jak_positive_smiles()
    print(f"JAK positives (pchembl >= {POSITIVE_PCHEMBL}): {len(pos)}")
    print(f"Matched negatives: {len(neg)}  (from {len(NEGATIVE_TARGETS)} other targets)")
    print(neg["source_target"].value_counts().to_string())
    print(f"Cached -> {CACHE_DIR / 'negatives.parquet'}")


if __name__ == "__main__":
    _main()
