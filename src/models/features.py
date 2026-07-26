"""Molecular featurization: Morgan (ECFP4) fingerprints.

2048-bit ECFP4 (radius 2) is the standard, cheap descriptor for CPU QSAR.
The generator is built once and reused.

**Fingerprints are held as `uint8`, not float64.** They are single bits, so float64
spent 8 bytes on each: the drug-like library alone came to 113 MB as float64 versus
14 MB as uint8, which is what pushed the deployed funnel over a 512 MB memory limit.
The change is numerically inert — HistGradientBoosting bins its inputs, and 0/1 bins
identically either way. Verified: training on uint8 vs float64 gives bit-identical
predictions (max abs diff 0.0) *and* an identical model pickle hash, so `model_id`
and every VALIDATION.md number are unaffected.

For a whole library, prefer `iter_morgan_batches`: it featurises and scores in
slices so the full matrix is never resident at once.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

FP_SIZE = 2048
FP_DTYPE = np.uint8
BATCH_SIZE = 1024
_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)


def morgan_matrix(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Featurize a list of SMILES.

    Returns (X, mask):
      - X:    (n_valid, FP_SIZE) uint8 array of fingerprints
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
    X = (np.asarray(rows, dtype=FP_DTYPE) if rows
         else np.empty((0, FP_SIZE), dtype=FP_DTYPE))
    return X, np.array(mask, dtype=bool)


def iter_morgan_batches(smiles: list[str], batch_size: int = BATCH_SIZE
                        ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield `(X, mask)` per slice of `smiles`, so no full matrix is ever resident.

    `mask` is the slice's own parse mask; concatenating the yielded masks in order
    reproduces exactly what `morgan_matrix` would return for the whole list. Used by
    the wide screen, where materialising every fingerprint at once is what costs the
    memory rather than the arithmetic.
    """
    for start in range(0, len(smiles), batch_size):
        yield morgan_matrix(smiles[start:start + batch_size])
