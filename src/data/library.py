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

from ..models.property_models import _download

RDLogger.DisableLog("rdApp.*")

_ROOT = Path(__file__).resolve().parents[2]
CACHE = _ROOT / "data" / "library" / "library.parquet"


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Chem.MolToSmiles(mol) if mol else None


def load_library(use_cache: bool = True) -> pd.DataFrame:
    """Diverse, target-agnostic drug-like library as a frame of canonical SMILES."""
    if use_cache and CACHE.exists():
        return pd.read_parquet(CACHE)

    import gzip
    raw = gzip.decompress(_download("tox21.csv.gz"))
    smiles = pd.read_csv(io.BytesIO(raw))["smiles"].dropna().map(_canonical).dropna()
    lib = pd.DataFrame({"smi": pd.Series(smiles.unique())})

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    lib.to_parquet(CACHE, index=False)
    return lib


def _main() -> None:
    lib = load_library()
    print(f"Wide library: {len(lib)} unique canonical drug-like molecules "
          f"(target-agnostic)\ncached -> {CACHE}")


if __name__ == "__main__":
    _main()
