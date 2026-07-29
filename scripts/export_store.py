#!/usr/bin/env python
"""Export the evidence store to committed parquet, so the data is in the repo.

The store itself (`data/chem.duckdb`, 52 MB) is a rebuildable local artifact and
stays gitignored. Compressed to ZSTD parquet the same content is **5.7 MB**, which is
small enough to commit — and that changes what the data *is*: not a file on whoever
ran the backfill's laptop, but something a reader can open from the repository, query
in a browser, or restore into a local store in seconds without touching ChEMBL.

That matters for this project specifically. Every number in VALIDATION.md is supposed
to be checkable; "trust me, the store says so" is exactly the kind of claim the rest
of the repo refuses to make.

    python scripts/export_store.py          # store  -> assets/evidence/*.parquet
    python -m src.data.db --restore         # assets -> a fresh local store
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import db as _db                     # noqa: E402

EXPORT_DIR = Path(__file__).resolve().parents[1] / "assets" / "evidence"
TABLES = ("source", "ingestion", "target", "molecule", "assay", "activity",
          "library", "library_member")


def export(con, out_dir: Path = EXPORT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": _db.SCHEMA_VERSION,
        "sources": [r[0] for r in
                    con.execute("SELECT source_id FROM source ORDER BY 1").fetchall()],
        "tables": {},
    }
    for table in TABLES:
        path = out_dir / f"{table}.parquet"
        con.execute(
            f"COPY {table} TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        manifest["tables"][table] = {
            "rows": con.execute(f"SELECT count(*) FROM {table}").fetchone()[0],
            "bytes": path.stat().st_size,
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    if not _db.exists():
        raise SystemExit(
            "No store to export. Build it first:\n"
            "  pip install -r requirements-dev.txt\n"
            "  python -m src.data.ingest jak pi3k --library")
    with _db.session(read_only=True) as con:
        manifest = export(con)
    total = sum(t["bytes"] for t in manifest["tables"].values())
    print(f"exported -> {EXPORT_DIR}  ({total / 1e6:.2f} MB, "
          f"sources: {', '.join(manifest['sources'])})")
    for name, t in manifest["tables"].items():
        print(f"  {name:16} {t['rows']:>8,} rows  {t['bytes'] / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
