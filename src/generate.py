"""STEP 8: conditional analogue generation over a chosen scaffold (CPU).

A compact, deterministic generator that decorates a seed molecule's aromatic
positions with common medicinal-chemistry substituents, yielding valid novel
analogues. This is the CPU fallback that lets the loop close and be tested without
a GPU; the Colab notebook can swap in a heavier GPU generative model at this seam.

Every generated molecule is an **in-silico hypothesis** — it must be re-scored
through the Stage-B models and AD-filtered, and it is never presented as a hit.
"""
from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Common medchem substituents, as fragments attached at a dummy atom.
SUBSTITUENTS = ["F", "Cl", "Br", "C", "OC", "O", "N", "C#N", "C(F)(F)F", "S(C)(=O)=O"]


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


def generate_analogues(seed_smiles: str, max_analogues: int = 40) -> list[str]:
    """Novel valid analogues of a seed, by single-substituent aromatic decoration."""
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
            if smi and smi not in seen:
                seen.add(smi)
                out.append(smi)
                if len(out) >= max_analogues:
                    return out
    return out
