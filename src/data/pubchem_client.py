"""PubChem access: name -> structure, and similarity expansion of a seed SMILES.

Two independent uses:

  * `resolve_name` turns what a user types ("ruxolitinib") into a structure, so
    the dashboard can score a molecule by name rather than by SMILES.
  * `expand` brings in molecules the per-target model has NOT been trained on
    (Phase 4), so its activity predictions are genuine rather than memorized.

PUG-REST only, throttled to respect PubChem's rate limits (<= 5 requests/second).
Both paths return **standardised** SMILES: PubChem serves marketed drugs as their
salt form, and a counterion would silently corrupt the fingerprint, the Ro5
molecular weight and the applicability-domain distance.
"""
from __future__ import annotations

import time
from urllib.parse import quote

import requests

from ..standardize import standardize

_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_HEADERS = {"User-Agent": "chem-predict-dashboard/0.1 (portfolio project)"}
_THROTTLE = 0.25  # seconds between requests


def _get(url: str, timeout: int = 40) -> dict:
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    time.sleep(_THROTTLE)
    return resp.json()


class NameNotFound(LookupError):
    """PubChem knows no compound by that name."""


def resolve_name(name: str) -> tuple[str, int, str]:
    """Resolve a compound name or synonym to (standardised SMILES, CID, matched title).

    Raises NameNotFound if PubChem has no match, and requests.RequestException if
    the lookup itself failed — the two are different problems for the caller and a
    silent None would conflate "no such drug" with "you are offline".
    """
    query = (name or "").strip()
    if not query:
        raise NameNotFound("Enter a compound name.")
    url = f"{_BASE}/compound/name/{quote(query)}/property/SMILES,Title/JSON"
    try:
        data = _get(url)
    except requests.HTTPError as err:
        if err.response is not None and err.response.status_code == 404:
            raise NameNotFound(f"PubChem has no compound named {query!r}.") from err
        raise
    records = data.get("PropertyTable", {}).get("Properties", [])
    for record in records:                       # first record whose structure parses
        smiles = standardize(record.get("SMILES"))
        if smiles:
            return smiles, int(record["CID"]), record.get("Title") or query
    raise NameNotFound(f"PubChem returned no usable structure for {query!r}.")


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
            parent = standardize(raw)         # neutral parent form, for dedup and scoring
            if parent is None:
                continue
            out[cid] = parent
            if len(out) >= cap:
                return out
    return out
