"""STEP 8: conditional analogue generation over a chosen scaffold (CPU).

A compact, deterministic generator that decorates a seed molecule's aromatic
positions with common medicinal-chemistry substituents, yielding valid novel
analogues. This is the CPU fallback that lets the loop close and be tested without
a GPU; the Colab notebook can swap in a heavier GPU generative model at this seam.

Every generated molecule is an **in-silico hypothesis** — it must be re-scored
through the Stage-B models and AD-filtered, and it is never presented as a hit.

A hypothesis nobody can synthesise is not a hypothesis, so analogues are also
filtered on **synthetic accessibility** (RDKit's SA score, 1 = trivial to make,
10 = intractable). The generator decorates ring positions without any notion of a
synthetic route, so this is the cheap guard that keeps un-makeable molecules out
of the deep-dive report.
"""
from __future__ import annotations

import os
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import RDConfig

RDLogger.DisableLog("rdApp.*")

# sascorer ships in RDKit's Contrib tree, which is not on the import path by default.
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

# Common medchem substituents, as fragments attached at a dummy atom.
SUBSTITUENTS = ["F", "Cl", "Br", "C", "OC", "O", "N", "C#N", "C(F)(F)F", "S(C)(=O)=O"]

# SA score above this is rejected. 6 is the usual "hard to make" line: it keeps
# decorated drug-like scaffolds while dropping the strained or over-substituted
# products this generator can produce by attaching to an awkward position.
MAX_SA_SCORE = 6.0


def _attach(mol: Chem.Mol, aromatic_idx: int, sub_smiles: str) -> str | None:
    """Attach a substituent fragment at an aromatic carbon; return canonical SMILES or None."""
    frag = Chem.MolFromSmiles("*" + sub_smiles)
    if frag is None:
        return None
    combo = Chem.RWMol(Chem.CombineMols(mol, frag))
    dummies = [a for a in combo.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummies:
        return None
    dummy = dummies[0]
    neighbour = dummy.GetNeighbors()[0].GetIdx()
    combo.AddBond(aromatic_idx, neighbour, Chem.BondType.SINGLE)
    combo.RemoveAtom(dummy.GetIdx())
    m = combo.GetMol()
    try:
        Chem.SanitizeMol(m)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
        return None
    return Chem.MolToSmiles(m)


def sa_score(smiles: str) -> float | None:
    """Synthetic accessibility, 1 (trivial) to 10 (intractable); None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    return sascorer.calculateScore(mol) if mol is not None else None


def generate_analogues(seed_smiles: str, max_analogues: int = 40,
                       max_sa: float = MAX_SA_SCORE) -> list[str]:
    """Novel valid analogues of a seed, by single-substituent aromatic decoration.

    Analogues scoring worse than `max_sa` on synthetic accessibility are dropped:
    the decoration is structure-blind, and an analogue nobody can make is not a
    testable hypothesis.
    """
    mol = Chem.MolFromSmiles(seed_smiles)
    if mol is None:
        return []
    seed_canon = Chem.MolToSmiles(mol)
    out: list[str] = []
    seen = {seed_canon}
    positions = [a.GetIdx() for a in mol.GetAtoms()
                 if a.GetIsAromatic() and a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0]
    for idx in positions:
        for sub in SUBSTITUENTS:
            smi = _attach(mol, idx, sub)
            if not smi or smi in seen:
                continue
            seen.add(smi)
            score = sa_score(smi)
            if score is None or score > max_sa:
                continue
            out.append(smi)
            if len(out) >= max_analogues:
                return out
    return out
