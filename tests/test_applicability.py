"""STEP 6 tests: applicability-domain signals (offline, deterministic)."""
import numpy as np

from src import applicability as ad


def test_tanimoto_nn_identical_molecule_is_one():
    train = ["c1ccccc1", "CCO", "CCN"]
    sim = ad.tanimoto_nn_similarity(["CCO"], train)   # exact match in train
    assert np.isclose(sim[0], 1.0)


def test_tanimoto_nn_dissimilar_is_low():
    train = ["c1ccccc1"]                               # benzene
    sim = ad.tanimoto_nn_similarity(["CCCCCCCCCO"], train)   # long aliphatic alcohol
    assert sim[0] < ad.TANIMOTO_IN_DOMAIN             # far -> out of domain


def test_leverage_flags_an_extrapolating_descriptor():
    # tight training cluster; a query far out in descriptor space has high leverage.
    train = ["CCO", "CCN", "CCC", "CCCO", "CCCN"]      # small molecules
    query = ["CCO", "C(=O)(O)c1ccc2ccc3ccc4ccccc4c3c2c1"]   # near vs huge PAH acid
    desc_tr, desc_q = ad._descriptors(train), ad._descriptors(query)
    h, thr = ad.leverage(desc_q, desc_tr)
    assert h[0] < thr < h[1]                           # near in-domain, far out


def test_in_domain_requires_both_signals():
    train = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"]
    flags = ad.in_domain(["c1ccccc1", "CCCCCCCCCCCCCCO"], train)
    assert flags["in_domain"][0]                       # benzene: close + typical
    assert not flags["in_domain"][1]                   # long chain: far / atypical
