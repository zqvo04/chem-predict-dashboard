"""Chirality audit: what the achiral ECFP4 representation cannot see.

    python scripts/chirality_audit.py

README "Known gaps" #7 has always listed "achiral ECFP4" as a limitation, but the
size of it was never measured, so it stayed a caveat rather than a claim. This
script measures it from the committed assets only — no network, no models.

Three questions, because they have different answers and drive different decisions:

  1. How much of the training data does the representation collapse?
     Two distinct standardised SMILES that produce the same 2048-bit fingerprint are
     one input to the model with two labels. That is label noise by construction: it
     does not shrink as more data arrives.

  2. Is that collapse actually chirality, or fingerprint hashing?
     A collision is only a chirality finding if the members are the same molecule
     once stereochemistry is stripped. Hash collisions would be a different defect
     with a different fix, so the two are separated rather than assumed.

  3. What would a two-stage design lose?
     If achiral screening (stage B) feeds a chirality-aware refinement (stage A),
     the only actives lost for good are those whose pooled value falls below the
     potency floor -- the funnel never passes them on. Group mean is the proxy for
     what a model trained on the collapsed pair learns.

The library is measured too, for a different reason: a chirality-aware model needs
molecules whose stereocentres are *specified*, and screening-library molecules are
not guaranteed to be. Unspecified centres are what stage A can enumerate.

Numbers land in STATE.md. Re-run after any change to `src.standardize` or to the
fingerprint generator in `src.models.features` -- both feed this directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.features import _GENERATOR  # noqa: E402
from src.panels import get_panel  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_ROOT = Path(__file__).resolve().parents[1]
POTENCY_FLOOR = 6.0        # the funnel's Tier-1 floor (src/funnel.py)


def _flat_smiles(smiles: str) -> str | None:
    """Canonical SMILES with stereochemistry removed -- the achiral identity."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def _collision_groups(frame: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """Molecules grouped by fingerprint, keeping only the groups with >1 member."""
    groups: dict[str, list[tuple[str, float]]] = {}
    for smi, pchembl in zip(frame["smi"], frame["pchembl"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        key = _GENERATOR.GetFingerprint(mol).ToBitString()
        groups.setdefault(key, []).append((smi, float(pchembl)))
    return {k: v for k, v in groups.items() if len(v) > 1}


def _spread(groups: list[list[tuple[str, float]]]) -> dict:
    """Within-group pchembl spread -- how wrong the model must be on some member."""
    if not groups:
        return {"n_groups": 0, "n_molecules": 0}
    diffs = np.array([max(p for _, p in g) - min(p for _, p in g) for g in groups])
    return {
        "n_groups": len(groups),
        "n_molecules": sum(len(g) for g in groups),
        "median_diff": float(np.median(diffs)),
        "p90_diff": float(np.percentile(diffs, 90)),
        "max_diff": float(diffs.max()),
        "n_ge_1log": int((diffs >= 1).sum()),
    }


def audit_training(panel_name: str = "jak") -> pd.DataFrame:
    """Per-isoform collapse, split into chirality and everything else."""
    panel = get_panel(panel_name)
    rows = []
    for isoform in panel.isoforms:
        frame = pd.read_parquet(panel.data_bundled / f"{isoform}.parquet")
        n_stereo = sum(
            1 for s in frame["smi"]
            if (m := Chem.MolFromSmiles(s)) is not None
            and rdMolDescriptors.CalcNumAtomStereoCenters(m) > 0)

        groups = _collision_groups(frame)
        stereo, other = [], []
        for members in groups.values():
            flat = {_flat_smiles(s) for s, _ in members}
            (stereo if len(flat) == 1 else other).append(members)

        # What a two-stage design loses: the group's pooled value sinks a real
        # active below the floor, so stage B never hands it to stage A.
        sunk = sum(len([p for _, p in g if p >= POTENCY_FLOOR])
                   for g in stereo + other
                   if max(p for _, p in g) >= POTENCY_FLOOR
                   and np.mean([p for _, p in g]) < POTENCY_FLOOR)
        n_active = int((frame["pchembl"] >= POTENCY_FLOOR).sum())

        rows.append({"isoform": isoform, "n": len(frame),
                     "pct_stereocentre": round(100 * n_stereo / len(frame), 1),
                     **{f"chiral_{k}": v for k, v in _spread(stereo).items()},
                     **{f"other_{k}": v for k, v in _spread(other).items()},
                     "n_active": n_active, "actives_sunk": sunk,
                     "pct_actives_sunk": round(100 * sunk / n_active, 2)})
    return pd.DataFrame(rows)


def audit_library() -> dict:
    """Is the screening library's stereochemistry even specified?

    A chirality-aware model has nothing to use on an unspecified centre, and the
    training set is roughly half specified -- so screening the library with such a
    model scores part of it off-distribution. Stage A can enumerate instead.
    """
    lib = pd.read_parquet(_ROOT / "assets" / "library" / "library.parquet")
    parsed = defined = unspecified = 0
    counts: dict[int, int] = {}
    for smi in lib["smi"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        parsed += 1
        if rdMolDescriptors.CalcNumAtomStereoCenters(mol) > 0:
            defined += 1
        n_unspec = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
        if n_unspec > 0:
            unspecified += 1
            counts[n_unspec] = counts.get(n_unspec, 0) + 1
    return {"n_parsed": parsed,
            "n_defined": defined, "pct_defined": round(100 * defined / parsed, 1),
            "n_unspecified": unspecified,
            "pct_unspecified": round(100 * unspecified / parsed, 1),
            "unspecified_centre_counts": dict(sorted(counts.items()))}


def main() -> None:
    train = audit_training()
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("Training data -- fingerprint collapse (assets/jak/*.parquet)\n")
    print(train.to_string(index=False))

    lib = audit_library()
    print(f"\nScreening library (assets/library/, n={lib['n_parsed']})")
    print(f"  defined stereocentres    : {lib['n_defined']} ({lib['pct_defined']} %)")
    print(f"  unspecified stereocentres: {lib['n_unspecified']} "
          f"({lib['pct_unspecified']} %)")
    print(f"  unspecified centres per molecule: {lib['unspecified_centre_counts']}")
    print("\nUnspecified centres are what stage A enumerates; 2^n isomers per "
          "molecule, and n is 1 for most of them.")


if __name__ == "__main__":
    main()
