"""Phase 4: end-to-end screening pipeline + composite scoring.

    target
      -> retrieve known actives                       (Phase 1, ChEMBL)
      -> expand with novel PubChem analogues           (Phase 4, breaks circularity)
      -> drug-likeness filter                          (Phase 2, Ro5 + PAINS)
      -> per-target activity prediction                (Phase 3, QSAR regressor)
      -> composite score -> ranked Top N

The composite blends predicted potency with QED drug-likeness. Known actives act
as a positive control (they should rank high); pubchem_novel rows are the actual
screening output the model is genuinely predicting.

CLI:
    python -m src.pipeline EGFR --top 15
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED

from .data import chembl_client as cc
from .data import pubchem_client as pc
from .filters.druglikeness import apply_druglikeness
from .models.target_model import TargetModel, train_target_model

ACTIVITY_WEIGHT = 0.7
QED_WEIGHT = 0.3
_POOL_COLUMNS = ["id", "canonical_smiles", "measured_pchembl", "source"]


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def _qed(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return float(QED.qed(mol)) if mol else np.nan


def _activity_norm(pchembl: pd.Series | np.ndarray) -> np.ndarray:
    """Map predicted pchembl to [0, 1] on a fixed potency scale (5 = ~10 uM,
    10 = ~0.1 nM). Fixed (not min-max) so scores are comparable across targets."""
    return np.clip((np.asarray(pchembl, dtype=float) - 5.0) / 5.0, 0.0, 1.0)


def _known_pool(target_query: str, organism: str | None,
                max_records: int, use_cache: bool):
    target, known = cc.get_candidates(target_query, organism=organism,
                                      max_records=max_records, use_cache=use_cache)
    known = known.copy()
    known["canonical_smiles"] = known["canonical_smiles"].map(_canonical)
    known = known.dropna(subset=["canonical_smiles"])
    known = known.rename(columns={"molecule_chembl_id": "id",
                                  "pchembl_value": "measured_pchembl"})
    known["source"] = "chembl_known"
    return target, known[_POOL_COLUMNS]


def _novel_pool(known: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    seeds = (known.sort_values("measured_pchembl", ascending=False)
                  ["canonical_smiles"].head(n_seeds).tolist())
    novel = pc.expand(seeds, threshold=85)
    known_smiles = set(known["canonical_smiles"])
    rows = [{"id": f"CID{cid}", "canonical_smiles": smi,
             "measured_pchembl": np.nan, "source": "pubchem_novel"}
            for cid, smi in novel.items() if smi not in known_smiles]
    return pd.DataFrame(rows, columns=_POOL_COLUMNS)


def screen(target_query: str, organism: str | None = "Homo sapiens",
           expand: bool = True, n_seeds: int = 5,
           max_records: int = 4000, use_cache: bool = True
           ) -> tuple[cc.Target, TargetModel, pd.DataFrame]:
    """Run the full screen and return (target, model, scored pool).

    The pool is drug-likeness-filtered and sorted by composite score, with a
    `source` column ("chembl_known" | "pubchem_novel"). Callers slice per
    source: known actives are the positive control, pubchem_novel rows are the
    screening output. Known rows are scored on measured potency, novel on the
    model's prediction — so the two tracks are ranked, not mixed into one list.
    """
    target, pool = _known_pool(target_query, organism, max_records, use_cache)
    model = train_target_model(target_query, organism=organism,
                               max_records=max_records, use_cache=use_cache)

    if expand and not pool.empty:
        pool = pd.concat([pool, _novel_pool(pool, n_seeds)], ignore_index=True)

    # known rows come first, so dedup keeps the measured/known copy
    pool = pool.drop_duplicates(subset="canonical_smiles", keep="first").reset_index(drop=True)

    pool = apply_druglikeness(pool)
    pool = pool[pool["druglike"]].reset_index(drop=True)

    pool["pred_pchembl"] = model.predict(pool["canonical_smiles"].tolist())
    pool = pool.dropna(subset=["pred_pchembl"])

    # Known actives already have a measured potency, so score them on truth and
    # only trust the model for novel molecules. This removes the memorization
    # inflation that would otherwise let training molecules crowd out novel ones.
    pool["activity_pchembl"] = pool["measured_pchembl"].fillna(pool["pred_pchembl"])
    pool["qed"] = pool["canonical_smiles"].map(_qed)
    pool["activity_norm"] = _activity_norm(pool["activity_pchembl"])
    pool["composite"] = ACTIVITY_WEIGHT * pool["activity_norm"] + QED_WEIGHT * pool["qed"]

    scored = pool.sort_values("composite", ascending=False).reset_index(drop=True)
    return target, model, scored


def _main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end target screening")
    ap.add_argument("target", help="Target name or ChEMBL id, e.g. EGFR")
    ap.add_argument("--organism", default="Homo sapiens")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--max-records", type=int, default=4000)
    ap.add_argument("--no-expand", action="store_true")
    args = ap.parse_args()

    target, model, scored = screen(args.target, organism=args.organism,
                                   expand=not args.no_expand,
                                   max_records=args.max_records)
    m = model.metrics
    known = scored[scored["source"] == "chembl_known"]
    novel = scored[scored["source"] == "pubchem_novel"]
    cols = ["id", "pred_pchembl", "measured_pchembl", "qed", "composite"]

    print(f"Target : {target.chembl_id}  {target.pref_name}")
    print(f"Model  : scaffold R2={m.r2:.3f} RMSE={m.rmse:.3f} (n={m.n_molecules})")
    print(f"Pool   : {len(known)} known actives + {len(novel)} novel candidates (drug-like)\n")

    print(f"== Top {args.top} known actives (positive control, scored on measured potency) ==")
    print(known.head(args.top)[cols].to_string(index=False))
    print(f"\n== Top {args.top} NOVEL candidates (model-predicted, not in training) ==")
    print(novel.head(args.top)[cols].to_string(index=False) if not novel.empty
          else "  (expansion returned nothing — check network)")


if __name__ == "__main__":
    _main()
