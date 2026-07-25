"""STEP 8 tests: CPU analogue generation (offline)."""
from rdkit import Chem

from src.generate import generate_analogues, sa_score


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


def test_synthetic_accessibility_gates_the_output():
    """A hypothesis nobody can make is not a hypothesis. The generator decorates
    ring positions with no notion of a synthetic route, so SA is the cheap guard."""
    seed = "N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1"     # ruxolitinib
    unfiltered = generate_analogues(seed, max_analogues=500, max_sa=99)
    tightened = generate_analogues(seed, max_analogues=500, max_sa=3.5)
    assert 0 < len(tightened) < len(unfiltered)
    assert all(sa_score(s) <= 3.5 for s in tightened)


def test_default_threshold_keeps_decorated_drug_like_scaffolds():
    """Honest scope: at the default of 6.0 this drops nothing for a drug-like seed.
    It exists to catch pathological products, not to prune ordinary ones."""
    seed = "N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1"
    assert (generate_analogues(seed, max_analogues=500)
            == generate_analogues(seed, max_analogues=500, max_sa=99))


def test_sa_score_rejects_unparseable_input():
    assert sa_score("not_a_smiles") is None
