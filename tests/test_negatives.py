"""STEP 10 tests: the presumed-inactive negative set for the binder gate.

Synthetic frames (no network) pin the JAK-exclusion, dedup and physchem-match
logic; a live smoke test self-skips when ChEMBL is unreachable.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import negatives as neg


def test_matched_indices_track_the_positive_physchem_shape():
    # Positives cluster at (MW≈300, logP≈2). Negatives are split between that
    # cluster and a far one at (MW≈600, logP≈6). The match must prefer the near
    # cluster, so a classifier can't separate the classes on MW/logP alone.
    rng = np.random.default_rng(0)
    pos = np.column_stack([rng.normal(300, 10, 200), rng.normal(2, 0.2, 200)])
    near = np.column_stack([rng.normal(300, 10, 200), rng.normal(2, 0.2, 200)])
    far = np.column_stack([rng.normal(600, 10, 200), rng.normal(6, 0.2, 200)])
    negs = np.vstack([near, far])

    kept = neg._matched_indices(pos, negs, bins=6, seed=0)
    assert len(kept) > 0
    frac_near = np.mean(kept < 200)               # first 200 rows are the near cluster
    assert frac_near > 0.9                         # matched set is almost all near-cluster


def test_matched_indices_is_deterministic():
    rng = np.random.default_rng(1)
    pos = rng.normal(300, 40, (150, 2))
    negs = rng.normal(300, 40, (300, 2))
    a = neg._matched_indices(pos, negs, seed=7)
    b = neg._matched_indices(pos, negs, seed=7)
    assert np.array_equal(a, b)


def _raw_acts(smiles):
    return pd.DataFrame({
        "molecule_chembl_id": [f"CHEMBL{i}" for i in range(len(smiles))],
        "canonical_smiles": smiles,
        "pchembl_value": [7.0] * len(smiles),
        "standard_type": ["IC50"] * len(smiles),
    })


def test_build_excludes_jak_molecules_and_dedups(tmp_path, monkeypatch):
    import dataclasses

    from src.panels import JAK

    panel = dataclasses.replace(JAK, root=tmp_path)
    monkeypatch.setattr(neg, "usable_negative_targets",
                        lambda p: {"T1": "CHEMBLX", "T2": "CHEMBLY", "T3": "CHEMBLZ"})
    # Isolate the exclusion/dedup logic from the physchem match (matching has its
    # own tests above): keep every candidate that survives exclusion.
    monkeypatch.setattr(neg, "_matched_indices",
                        lambda pos, negd, **kw: np.arange(len(negd)))
    # A drug-like JAK positive that also appears in a negative pull must be dropped.
    jak_mol = "O=C(Nc1ccccc1)c1ccc(Cl)cc1"
    monkeypatch.setattr(neg, "positive_smiles",
                        lambda p, use_cache=True: {jak_mol})

    pool = {
        "CHEMBLX": _raw_acts([jak_mol, "CCOc1ccc(C(=O)O)cc1", "CCOc1ccc(C(=O)O)cc1"]),
        "CHEMBLY": _raw_acts(["COc1ccc(CCN)cc1", "CN1CCN(c2ccccc2)CC1"]),
        "CHEMBLZ": _raw_acts(["O=C(O)c1ccc(N)cc1", "Cc1ccc(S(N)(=O)=O)cc1"]),
    }
    monkeypatch.setattr(neg.cc, "fetch_activities",
                        lambda cid, **kw: pool[cid])

    out = neg.build_negatives(panel, use_cache=False)
    assert (panel.data_cache / "negatives.parquet").exists()
    assert len(out) > 0
    assert jak_mol not in set(out["smi"])          # JAK molecule excluded
    assert out["smi"].is_unique                    # the duplicate collapsed
    assert set(out.columns) == {"smi", "source_target"}


def test_live_build_smoke():
    try:
        out = neg.build_negatives(use_cache=True)   # default panel = JAK
    except (RuntimeError, OSError) as err:
        pytest.skip(f"ChEMBL unreachable / negatives not built: {err}")
    assert len(out) > 500
    assert out["smi"].is_unique
    assert set(out["source_target"]).issubset(set(neg.NEGATIVE_TARGETS))
