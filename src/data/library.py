"""STEP 7 / 13: the wide screening library (the funnel top).

A large, diverse, **target-agnostic** set of drug-like molecules — the haystack the
cheap tiers search, deliberately *not* drawn from JAK actives (that was v1's fake
novelty).

**Source: bioactive drug-like molecules from 20 diverse ChEMBL targets** (STEP 13).
The library was previously the Tox21 collection, which was the wrong haystack twice
over:

  * **It is a toxicology panel**, not a discovery library — environmental chemicals,
    pesticides and industrial reagents. With the binder gate in place only 23 of its
    6 882 drug-like molecules looked like plausible JAK binders, which made the
    shortlist and the gap percentile (4.3 points per molecule) nearly meaningless.
  * **It was the toxicity model's own training set** — 7 613 / 7 613 overlap — so the
    MPO `tox_prob` column was recalling memorised labels, not predicting.

The replacement is drawn with the same client the rest of the repo uses, from
targets chosen to be **disjoint from everything the binder gate was trained on**:
none of the JAK isoforms (its positives) and none of `negatives.NEGATIVE_TARGETS`
(its negatives). Molecules appearing in either training class are dropped by SMILES
as well, so the gate's verdict on a library molecule is a genuine prediction rather
than recall — the same leakage this replacement exists to remove.

Half the panel is kinases (chemistry a JAK screen could plausibly enrich from,
without being JAK) and half is non-kinase — proteases, GPCRs, nuclear receptors,
epigenetic and metabolic enzymes — so the library stays genuinely diverse rather
than becoming a kinase-inhibitor set.

CLI (build + cache, print size):
    python -m src.data.library
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from ..filters.druglikeness import apply_druglikeness
from ..standardize import standardize
from . import chembl_client as cc

RDLogger.DisableLog("rdApp.*")

# 20 targets, verified SINGLE PROTEIN / Homo sapiens, none of them a JAK isoform or
# one of the binder gate's negative targets. Ten kinases and ten non-kinases.
LIBRARY_TARGETS = {
    # kinases — plausible JAK-adjacent chemistry, but different targets
    "SRC": "CHEMBL267", "p38a": "CHEMBL260", "ABL1": "CHEMBL1862",
    "VEGFR2": "CHEMBL279", "BTK": "CHEMBL5251", "AURKA": "CHEMBL4722",
    "MET": "CHEMBL3717", "ALK": "CHEMBL4247", "PIM1": "CHEMBL2147",
    "GSK3B": "CHEMBL262",
    # non-kinases — the diversity half
    "HDAC1": "CHEMBL325", "PARP1": "CHEMBL3105", "A2A": "CHEMBL251",
    "5HT2A": "CHEMBL224", "BACE1": "CHEMBL4822", "CathepsinK": "CHEMBL268",
    "PDE5": "CHEMBL1827", "ER": "CHEMBL206", "AR": "CHEMBL1871",
    "GR": "CHEMBL2034",
}
PCHEMBL_GTE = 6.0      # bioactive on its own target; nothing to do with JAK
MAX_RECORDS = 3000     # per target
MIN_TARGETS = 10       # refuse to build a thin library if fetches fail

_ROOT = Path(__file__).resolve().parents[2]
CACHE = _ROOT / "data" / "library" / "library.parquet"          # runtime cache, gitignored
BUNDLED = _ROOT / "assets" / "library" / "library.parquet"      # committed for deploys


def _gate_training_smiles(use_cache: bool = True) -> set[str]:
    """Every molecule the binder gate was trained on, positives and negatives.

    Excluded from the library so the gate's verdict on it is a prediction. Imported
    lazily: `negatives` imports this module's sibling `jak`, and the exclusion is
    only needed on a rebuild.
    """
    from .negatives import build_negatives, positive_smiles

    return set(positive_smiles(use_cache=use_cache)) | set(
        build_negatives(use_cache=use_cache)["smi"])


def load_library(use_cache: bool = True) -> pd.DataFrame:
    """Diverse, target-agnostic drug-like library as a frame of standardised SMILES.

    Molecules are reduced to their neutral parent (`src.standardize`) before
    dedup: ChEMBL ships salts and charged forms, and screening a counterion is
    wrong twice over — it inflates the molecular weight Tier 0 checks against Ro5,
    and it changes the fingerprint Tier 1 and the applicability domain read.

    Carries the funnel's Tier-0 verdict (Ro5 + PAINS) as columns: it is a property
    of the library and the filter rules alone — no model enters it — so computing
    it once here saves the screen ~16 s of PAINS matching on every cold start.
    """
    if use_cache:
        for path in (CACHE, BUNDLED):     # runtime rebuild wins over the shipped copy
            if path.exists():
                return pd.read_parquet(path)

    excluded = _gate_training_smiles(use_cache=use_cache)
    frames, skipped = [], []
    for name, cid in LIBRARY_TARGETS.items():
        # A single target's transient API failure must not discard the other
        # nineteen fetches; the panel is deliberately larger than it needs to be.
        try:
            acts = cc.fetch_activities(cid, pchembl_gte=PCHEMBL_GTE,
                                       max_records=MAX_RECORDS, use_cache=use_cache)
        except (RuntimeError, OSError):
            skipped.append(name)
            continue
        cand = cc.to_candidates(acts)
        if cand.empty:
            continue
        smi = cand["canonical_smiles"].map(standardize)
        frames.append(pd.DataFrame({"smi": smi, "source_target": name}).dropna(subset=["smi"]))

    if len(frames) < MIN_TARGETS:
        raise RuntimeError(
            f"Only {len(frames)} of {len(LIBRARY_TARGETS)} library targets fetched "
            f"(skipped {skipped}); too few for a diverse screening library.")

    pool = pd.concat(frames, ignore_index=True)
    pool = pool[~pool["smi"].isin(excluded)]
    pool = pool.drop_duplicates(subset="smi").reset_index(drop=True)

    lib = apply_druglikeness(pool, smiles_col="smi")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    lib.to_parquet(CACHE, index=False)
    return lib


def _main() -> None:
    lib = load_library(use_cache=False)
    print(f"Wide library: {len(lib)} unique canonical molecules from "
          f"{lib['source_target'].nunique()} targets (target-agnostic), "
          f"{int(lib['druglike'].sum())} clearing Tier 0")
    print(lib["source_target"].value_counts().to_string())
    print(f"cached -> {CACHE}")


if __name__ == "__main__":
    _main()
