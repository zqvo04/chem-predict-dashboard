"""Tiny on-disk cache for ChEMBL responses.

Avoids re-hitting the API for the same query. Keyed by a hash of the request
params so different targets / thresholds never collide. Parquet keeps the
cached candidate tables small and fast to reload.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _path(name: str, params: dict) -> Path:
    blob = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha1(blob.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{name}_{digest}.parquet"


def load(name: str, params: dict) -> pd.DataFrame | None:
    """Return the cached frame for these params, or None on a miss."""
    path = _path(name, params)
    return pd.read_parquet(path) if path.exists() else None


def save(name: str, params: dict, df: pd.DataFrame) -> None:
    """Persist a frame for these params, creating the cache dir if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_path(name, params), index=False)
