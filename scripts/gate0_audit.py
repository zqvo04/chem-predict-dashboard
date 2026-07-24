"""Gate 0 data audit: reproduce the JAK per-isoform counts + cross-measured
selectivity signal recorded in VALIDATION.md.

    python scripts/gate0_audit.py

Pulls JAK1/2/3 from ChEMBL (cached under data/cache after the first run), takes one
median pchembl per (molecule, isoform), and reports:
  - per-isoform unique molecules + active/inactive/gray counts,
  - the 3-way and pairwise cross-measured counts,
  - gap-based selective counts (S = pchembl(target) - max(off-isoforms)).

This is the audit that overturned the active/inactive classification plan (the
inactive class is nearly empty) in favour of pchembl regression + a selectivity
gap. See VALIDATION.md and DESIGN_DECISIONS.md sections 1-2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.chembl_client import fetch_activities  # noqa: E402
from src.data.jak import MAX_RECORDS, TARGETS, _canonical  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def _median_pchembl(target_id: str) -> pd.DataFrame:
    """One median pchembl per canonical molecule for a target."""
    acts = fetch_activities(target_id, pchembl_gte=None, max_records=MAX_RECORDS).copy()
    acts["pchembl_value"] = pd.to_numeric(acts["pchembl_value"], errors="coerce")
    acts = acts.dropna(subset=["canonical_smiles", "pchembl_value"])
    acts["smi"] = acts["canonical_smiles"].map(_canonical)
    acts = acts.dropna(subset=["smi"])
    return acts.groupby("smi").pchembl_value.median()


def main() -> None:
    per = {name: _median_pchembl(cid) for name, cid in TARGETS.items()}

    print("Per-isoform (median pchembl per molecule):")
    print(f"  {'isoform':7} {'unique':>7} {'active>=6':>9} {'inactive<=5':>11} "
          f"{'gray':>6}  pchembl_range")
    for name, s in per.items():
        active = int((s >= 6).sum())
        inactive = int((s <= 5).sum())
        gray = int(((s > 5) & (s < 6)).sum())
        print(f"  {name:7} {len(s):7d} {active:9d} {inactive:11d} {gray:6d}  "
              f"[{s.min():.1f}-{s.max():.1f}]")

    m = pd.concat([per["JAK1"].rename("JAK1"), per["JAK2"].rename("JAK2"),
                   per["JAK3"].rename("JAK3")], axis=1, join="inner")
    print(f"\n3-way cross-measured (all of JAK1/2/3): {len(m)}")
    for a, b in [("JAK1", "JAK2"), ("JAK1", "JAK3"), ("JAK2", "JAK3")]:
        n = len(pd.concat([per[a], per[b]], axis=1, join="inner"))
        print(f"  pairwise {a}-{b}: {n}")

    print("\nGap-based selective on the 3-way set (S = pchembl(tgt) - max(off)):")
    print(f"  {'target':7} {'S>=1(10x)':>10} {'S>=2(100x)':>11} {'median_S':>9} {'max_S':>6}")
    for tgt in TARGETS:
        offs = [o for o in TARGETS if o != tgt]
        gap = m[tgt] - m[offs].max(axis=1)
        print(f"  {tgt:7} {int((gap >= 1).sum()):10d} {int((gap >= 2).sum()):11d} "
              f"{gap.median():+9.2f} {gap.max():6.2f}")

    p = pd.concat([per["JAK1"].rename("JAK1"), per["JAK2"].rename("JAK2")],
                  axis=1, join="inner")
    g = p["JAK1"] - p["JAK2"]
    print(f"\nPairwise JAK1-JAK2 (n={len(p)}): "
          f"|S|>=1={int((g.abs() >= 1).sum())}  "
          f"sel-JAK1(S>=1)={int((g >= 1).sum())}  "
          f"sel-JAK2(S<=-1)={int((g <= -1).sum())}")


if __name__ == "__main__":
    main()
