"""Molecular featurization: Morgan (ECFP4) fingerprints.

2048-bit ECFP4 (radius 2) is the standard, cheap descriptor for CPU QSAR.
The generator is built once and reused.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

FP_SIZE = 2048
_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)


def morgan_matrix(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Featurize a list of SMILES.

    Returns (X, mask):
      - X:    (n_valid, FP_SIZE) float array of fingerprints
      - mask: (len(smiles),) bool array marking which inputs parsed

    The mask lets callers realign predictions with the original rows when some
    SMILES fail to parse.
    """
    rows: list[np.ndarray] = []
    mask: list[bool] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            mask.append(False)
            continue
        rows.append(_GENERATOR.GetFingerprintAsNumPy(mol))
        mask.append(True)
    X = np.array(rows, dtype=float) if rows else np.empty((0, FP_SIZE))
    return X, np.array(mask, dtype=bool)
