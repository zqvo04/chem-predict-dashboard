"""STEP 8 tests: CPU analogue generation (offline)."""
from rdkit import Chem

from src.generate import generate_analogues


def test_generates_valid_novel_analogues():
    seed = "c1ccccc1C(=O)O"                             # benzoic acid (aromatic Hs)
    ana = generate_analogues(seed, max_analogues=15)
    assert len(ana) > 0
    seed_canon = Chem.MolToSmiles(Chem.MolFromSmiles(seed))
    for smi in ana:
        assert Chem.MolFromSmiles(smi) is not None      # valid
        assert smi != seed_canon                        # novel vs seed
    assert len(set(ana)) == len(ana)                    # deduped


def test_respects_the_cap():
    ana = generate_analogues("c1ccccc1", max_analogues=5)
    assert len(ana) <= 5


def test_bad_smiles_yields_nothing():
    assert generate_analogues("not_a_smiles") == []


def test_nonaromatic_seed_has_no_decoration_sites():
    # pure aliphatic: no aromatic CH positions to decorate.
    assert generate_analogues("CCCCO") == []
