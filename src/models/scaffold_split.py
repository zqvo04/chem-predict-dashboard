"""Bemis-Murcko scaffold split for honest QSAR evaluation.

A random train/test split lets near-duplicate analogues leak across the split
and inflates reported performance. Splitting by Murcko scaffold guarantees no
scaffold appears in both sets, so the test score reflects generalization to new
chemotypes — the number that actually matters for screening.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_split(smiles: list[str], test_frac: float = 0.2,
                   seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Split indices so scaffolds never cross train/test.

    With ``seed=None`` (default) this is the deterministic variant: largest
    scaffold groups are placed first, so the split is stable across runs.

    Passing an integer ``seed`` instead shuffles the scaffold groups with that
    seed before filling the test set, giving a different-but-still-leak-free split
    per seed — used to report mean ± std over several scaffold splits. Scaffolds
    still never cross train/test; only *which* scaffolds land in test changes.
    Returns (train_idx, test_idx) as sorted int arrays.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[_scaffold(smi)].append(i)

    if seed is None:
        ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    else:
        ordered = sorted(groups.values(), key=lambda g: g[0])  # stable base order
        random.Random(seed).shuffle(ordered)
    n_test_target = int(round(len(smiles) * test_frac))

    train_idx: list[int] = []
    test_idx: list[int] = []
    for group in ordered:
        if len(test_idx) + len(group) <= n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))
