"""Structure standardisation: one canonical parent form per compound."""
from rdkit import Chem

from src.standardize import standardize

# Ruxolitinib as the free base and as the phosphate salt PubChem returns for the
# marketed drug. These must standardise to the same structure — the whole reason
# the module exists is that a counterion otherwise moves a known JAK inhibitor out
# of the applicability domain it belongs in.
_RUX_FREE_BASE = "N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1"
_RUX_PHOSPHATE = _RUX_FREE_BASE + ".OP(=O)(O)O"


def test_salt_and_free_base_collapse_to_one_structure():
    assert standardize(_RUX_PHOSPHATE) == standardize(_RUX_FREE_BASE)


def test_counterion_is_dropped_not_kept_as_the_parent():
    parent = standardize("CC(=O)Oc1ccccc1C(=O)O.[Na+]")
    assert parent is not None
    assert "Na" not in parent


def test_charged_form_is_neutralised():
    assert standardize("CC(=O)[O-]") == standardize("CC(=O)O")


def test_output_is_canonical_regardless_of_input_spelling():
    assert standardize("OCC") == standardize("CCO") == "CCO"


def test_unparseable_and_non_string_input_return_none():
    assert standardize("not_a_smiles") is None
    assert standardize(None) is None
    assert standardize(123) is None


def test_tautomer_canonicalisation_is_opt_in():
    """Off by default: turning it on rewrites ~10 % of the training sets, which
    would invalidate the deployed models — a retraining decision, not a default."""
    enol = "C=C(O)CC"
    assert standardize(enol) == Chem.CanonSmiles(enol)      # untouched by default
    assert standardize(enol, tautomer=True) != standardize(enol)


def test_stereochemistry_survives():
    """ECFP4 ignores stereo, but the SMILES we store and display must not."""
    assert "@" in standardize(_RUX_FREE_BASE)
