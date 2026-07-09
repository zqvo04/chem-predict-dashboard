"""PubChem similarity expansion: seed SMILES -> novel candidate SMILES.

Used in Phase 4 to bring in molecules the per-target model has NOT been trained
on, so its activity predictions are genuine rather than memorized. PUG-REST
only, throttled to respect PubChem's rate limits (<= 5 requests/second).
"""
from __future__ import annotations

import time
from urllib.parse import quote

import requests
from rdkit import Chem

_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_HEADERS = {"User-Agent": "chem-predict-dashboard/0.1 (portfolio project)"}
_THROTTLE = 0.25  # seconds between requests


def _get(url: str, timeout: int = 40) -> dict:
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    time.sleep(_THROTTLE)
    return resp.json()


def _similar_cids(smiles: str, threshold: int, max_records: int) -> list[int]:
    url = (f"{_BASE}/compound/fastsimilarity_2d/smiles/{quote(smiles)}/cids/JSON"
           f"?Threshold={threshold}&MaxRecords={max_records}")
    try:
        data = _get(url)
        return data.get("IdentifierList", {}).get("CID", [])
    except requests.RequestException:
        return []


def _cids_to_smiles(cids: list[int]) -> dict[int, str]:
    if not cids:
        return {}
    ids = ",".join(map(str, cids))
    url = f"{_BASE}/compound/cid/{ids}/property/SMILES/JSON"
    try:
        data = _get(url)
        return {p["CID"]: p.get("SMILES")
                for p in data["PropertyTable"]["Properties"]}
    except requests.RequestException:
        return {}


def expand(seeds: list[str], threshold: int = 90,
           max_per_seed: int = 30, cap: int = 150) -> dict[int, str]:
    """Return {CID: canonical_smiles} of molecules similar to the seeds.

    Degrades gracefully: network errors yield fewer (or zero) results rather
    than raising, so the pipeline can proceed on known actives alone.
    """
    out: dict[int, str] = {}
    for smiles in seeds:
        cids = _similar_cids(smiles, threshold, max_per_seed)
        for cid, raw in _cids_to_smiles(cids).items():
            if raw is None or cid in out:
                continue
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                continue
            out[cid] = Chem.MolToSmiles(mol)  # canonical form for dedup
            if len(out) >= cap:
                return out
    return out
