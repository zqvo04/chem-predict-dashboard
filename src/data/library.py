"""STEP 7: the wide screening library (the funnel top).

A large, diverse, **target-agnostic** set of drug-like molecules — the haystack the
cheap tiers search, deliberately *not* drawn from JAK actives (that was v1's fake
novelty). Default source: the Tox21 compound collection (~8k diverse drug-like
molecules, already wired for download in property_models), canonicalised and
deduplicated. Demo-scale and offline-cached; the pipeline scales to larger
libraries (ZINC / Enamine REAL) bounded only by Tier-1 throughput.

CLI (build + cache, print size):
    python -m src.data.library
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from ..filters.druglikeness import apply_druglikeness
from ..models.property_models import _download

RDLogger.DisableLog("rdApp.*")

_ROOT = Path(__file__).resolve().parents[2]
CACHE = _ROOT / "data" / "library" / "library.parquet"          # runtime cache, gitignored
BUNDLED = _ROOT / "assets" / "library" / "library.parquet"      # committed for deploys


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Chem.MolToSmiles(mol) if mol else None


def load_library(use_cache: bool = True) -> pd.DataFrame:
    """Diverse, target-agnostic drug-like library as a frame of canonical SMILES.

    Carries the funnel's Tier-0 verdict (Ro5 + PAINS) as columns: it is a property
    of the library and the filter rules alone — no model enters it — so computing
    it once here saves the screen ~16 s of PAINS matching on every cold start.
    """
    if use_cache:
        for path in (CACHE, BUNDLED):     # runtime rebuild wins over the shipped copy
            if path.exists():
                return pd.read_parquet(path)

    import gzip
    raw = gzip.decompress(_download("tox21.csv.gz"))
    smiles = pd.read_csv(io.BytesIO(raw))["smiles"].dropna().map(_canonical).dropna()
    lib = apply_druglikeness(pd.DataFrame({"smi": pd.Series(smiles.unique())}),
                             smiles_col="smi")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    lib.to_parquet(CACHE, index=False)
    return lib


def _main() -> None:
    lib = load_library()
    print(f"Wide library: {len(lib)} unique canonical drug-like molecules "
          f"(target-agnostic), {int(lib['druglike'].sum())} clearing Tier 0"
          f"\ncached -> {CACHE}")


if __name__ == "__main__":
    _main()
