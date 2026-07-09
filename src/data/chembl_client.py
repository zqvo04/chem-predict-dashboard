"""ChEMBL data client: target resolution + bioactivity fetch (Phase 1).

Option A (retrieval-first) backbone of the E2E screening pipeline:

    target name  ->  ChEMBL target id  ->  validated candidate SMILES

Everything runs on CPU with only HTTP + RDKit — no GPU, no paid services.
Raw activity pages are cached to parquet so repeat runs don't hit the API.

CLI:
    python -m src.data.chembl_client EGFR --top 10
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import pandas as pd
import requests
from rdkit import Chem, RDLogger

from . import cache

RDLogger.DisableLog("rdApp.*")  # silence RDKit warnings on invalid SMILES

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
DEFAULT_ACTIVITY_TYPES = ("IC50", "Ki", "Kd", "EC50")
_ONLY_FIELDS = (
    "molecule_chembl_id,canonical_smiles,standard_type,"
    "standard_value,standard_units,pchembl_value"
)
_HEADERS = {"User-Agent": "chem-predict-dashboard/0.1 (portfolio project)"}


@dataclass
class Target:
    chembl_id: str
    pref_name: str
    organism: str
    target_type: str
    score: float


def _get(path: str, params: dict, retries: int = 3, timeout: int = 30) -> dict:
    """GET a ChEMBL endpoint with exponential-backoff retry.

    Honors the environment HTTPS_PROXY automatically via requests.
    """
    url = f"{BASE_URL}/{path}"
    params = {**params, "format": "json"}
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as err:
            last_err = err
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"ChEMBL request to {path!r} failed after {retries} tries: {last_err}")


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #
def search_targets(query: str, organism: str | None = "Homo sapiens",
                   limit: int = 25) -> list[Target]:
    """Search targets by free text, ranked for our use case.

    ChEMBL's own relevance score often ranks protein complexes / PPIs above the
    plain single protein (e.g. "EGFR/PPP1CA" outranks EGFR). We re-rank to
    prefer SINGLE PROTEIN entries in the requested organism, then fall back to
    the API score.
    """
    data = _get("target/search", {"q": query, "limit": limit})
    targets = [
        Target(
            chembl_id=t.get("target_chembl_id"),
            pref_name=t.get("pref_name") or "",
            organism=t.get("organism") or "",
            target_type=t.get("target_type") or "",
            score=float(t.get("score") or 0.0),
        )
        for t in data.get("targets", [])
        if t.get("target_chembl_id")
    ]

    def rank(t: Target):
        return (
            t.target_type == "SINGLE PROTEIN",
            organism is None or t.organism == organism,
            t.score,
        )

    targets.sort(key=rank, reverse=True)
    return targets


def resolve_target(query: str, organism: str | None = "Homo sapiens") -> Target:
    """Resolve a target name or ChEMBL id to a single best Target."""
    if query.upper().startswith("CHEMBL"):
        data = _get(f"target/{query.upper()}", {})
        return Target(
            chembl_id=data["target_chembl_id"],
            pref_name=data.get("pref_name") or "",
            organism=data.get("organism") or "",
            target_type=data.get("target_type") or "",
            score=0.0,
        )
    targets = search_targets(query, organism=organism)
    if not targets:
        raise ValueError(f"No ChEMBL target found for query: {query!r}")
    return targets[0]


# --------------------------------------------------------------------------- #
# Bioactivity fetch
# --------------------------------------------------------------------------- #
def fetch_activities(target_id: str,
                     activity_types: tuple[str, ...] = DEFAULT_ACTIVITY_TYPES,
                     pchembl_gte: float | None = 6.0,
                     max_records: int = 2000,
                     page_size: int = 1000,
                     use_cache: bool = True) -> pd.DataFrame:
    """Fetch raw bioactivities for a target, paginating up to max_records.

    Filtering is done server-side (pchembl threshold + activity type) to keep
    payloads small. pchembl_value >= 6 (~1 uM) is the usual "active" cutoff.
    Pass pchembl_gte=None to fetch the full measured range (any quantified
    pchembl) — used to build regression training sets in Phase 3.
    """
    params = {
        "target_chembl_id": target_id,
        "standard_type__in": ",".join(activity_types),
        "only": _ONLY_FIELDS,
    }
    if pchembl_gte is not None:
        params["pchembl_value__gte"] = pchembl_gte
    else:
        params["pchembl_value__isnull"] = "false"
    cache_params = {**params, "max_records": max_records}
    if use_cache:
        hit = cache.load("activities", cache_params)
        if hit is not None:
            return hit

    rows: list[dict] = []
    offset = 0
    while len(rows) < max_records:
        page = _get("activity", {
            **params,
            "limit": min(page_size, max_records - len(rows)),
            "offset": offset,
        })
        acts = page.get("activities", [])
        if not acts:
            break
        rows.extend(acts)
        if page.get("page_meta", {}).get("next") is None:
            break
        offset += len(acts)

    df = pd.DataFrame(rows)
    if use_cache:
        cache.save("activities", cache_params, df)
    return df


_CANDIDATE_COLUMNS = [
    "molecule_chembl_id", "canonical_smiles",
    "pchembl_value", "standard_type", "n_activities",
]


def to_candidates(activities: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw activities into one validated row per molecule.

    - drops rows with no SMILES or no numeric pchembl_value
    - keeps each molecule's best (max) pchembl_value
    - drops molecules whose SMILES RDKit cannot parse
    - returns sorted by potency, most potent first
    """
    if activities.empty:
        return pd.DataFrame(columns=_CANDIDATE_COLUMNS)

    df = activities.copy()
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["canonical_smiles", "pchembl_value"])
    df = df.sort_values("pchembl_value", ascending=False)

    candidates = (
        df.groupby("molecule_chembl_id", sort=False)
          .agg(
              canonical_smiles=("canonical_smiles", "first"),
              pchembl_value=("pchembl_value", "max"),
              standard_type=("standard_type", "first"),
              n_activities=("pchembl_value", "size"),
          )
          .reset_index()
    )

    valid = candidates["canonical_smiles"].apply(
        lambda s: Chem.MolFromSmiles(s) is not None
    )
    candidates = candidates[valid]
    return candidates.sort_values("pchembl_value", ascending=False).reset_index(drop=True)


def get_candidates(query: str, organism: str | None = "Homo sapiens",
                   **fetch_kwargs) -> tuple[Target, pd.DataFrame]:
    """End-to-end Phase 1 entry point: target name -> (Target, candidate table)."""
    target = resolve_target(query, organism=organism)
    activities = fetch_activities(target.chembl_id, **fetch_kwargs)
    return target, to_candidates(activities)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch candidate molecules for a target from ChEMBL")
    ap.add_argument("target", help="Target name or ChEMBL id, e.g. EGFR / CHEMBL203")
    ap.add_argument("--organism", default="Homo sapiens")
    ap.add_argument("--pchembl-gte", type=float, default=6.0)
    ap.add_argument("--max-records", type=int, default=2000)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    target, candidates = get_candidates(
        args.target,
        organism=args.organism,
        pchembl_gte=args.pchembl_gte,
        max_records=args.max_records,
        use_cache=not args.no_cache,
    )
    print(f"Target : {target.chembl_id}  {target.pref_name}  "
          f"({target.organism}, {target.target_type})")
    print(f"Result : {len(candidates)} unique valid candidate molecules\n")
    print(candidates.head(args.top).to_string(index=False))


if __name__ == "__main__":
    _main()
