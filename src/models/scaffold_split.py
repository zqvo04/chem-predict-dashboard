"""Bemis-Murcko scaffold split for honest QSAR evaluation.

A random train/test split lets near-duplicate analogues leak across the split
and inflates reported performance. Splitting by Murcko scaffold guarantees no
scaffold appears in both sets, so the test score reflects generalization to new
chemotypes — the number that actually matters for screening.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_split(smiles: list[str], test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Split indices so scaffolds never cross train/test.

    Largest scaffold groups are assigned to train first (the standard
    deterministic variant); smaller groups fill the test set up to test_frac.
    Returns (train_idx, test_idx) as sorted int arrays.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[_scaffold(smi)].append(i)

    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n_test_target = int(round(len(smiles) * test_frac))

    train_idx: list[int] = []
    test_idx: list[int] = []
    for group in ordered:
        if len(test_idx) + len(group) <= n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))
