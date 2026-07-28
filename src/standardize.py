"""Structure standardisation — one canonical form for every SMILES that enters the pipeline.

Every model here featurises a SMILES with ECFP4, so two spellings of the same
compound produce two different fingerprints. The failure that matters is not
cosmetic: PubChem returns marketed drugs as their **salt** form (e.g. ruxolitinib
phosphate), and a counterion changes the fingerprint, the molecular weight the Ro5
filter reads, and the Tanimoto distance the applicability domain measures. A user
typing a famous JAK inhibitor would get "out of domain" for a molecule the models
were trained on.

`standardize()` is the single funnel every ingest and query path goes through:

    largest fragment  -> drop counterions, solvates, co-crystal formers
    uncharge          -> neutralise the parent
    canonical SMILES  -> one spelling

**Tautomers are deliberately not canonicalised by default.** Measured on this
repo's own data, tautomer canonicalisation rewrites 9.7 % of the JAK1 training set
and 18.4 % of the wide library, and costs ~8.6 ms/molecule against ~0.4 ms for the
steps above. Turning it on therefore changes what the deployed models were trained
on and invalidates every number in VALIDATION.md — a deliberate retraining
decision, not a side effect of cleaning input. `tautomer=True` is available for
whoever makes that decision.
"""
from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# Names the policy below, so a stored structure can be traced to the rules that
# produced it. The committed assets/jak parquet predates `standardize()` entirely and
# is ~0.5 % un-standardised (measured); rows carrying that provenance are tagged
# "legacy" in the evidence store rather than silently mixed with these.
STANDARDIZE_VERSION = "largest-fragment+uncharge/v1"

_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()
_TAUTOMER = rdMolStandardize.TautomerEnumerator()


def standardize_mol(mol: Chem.Mol, tautomer: bool = False) -> Chem.Mol | None:
    """Parent form of a molecule: largest fragment, neutralised."""
    try:
        mol = _LARGEST_FRAGMENT.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        if tautomer:
            mol = _TAUTOMER.Canonicalize(mol)
    except Exception:                      # RDKit raises a variety of C++ errors here
        return None
    return mol


def standardize(smiles: str, tautomer: bool = False) -> str | None:
    """Canonical SMILES of the neutral parent, or None if unparseable.

    Replaces the bare `MolToSmiles(MolFromSmiles(s))` the loaders used to do: that
    gave one *spelling* per input string but not one *structure* per compound.
    """
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = standardize_mol(mol, tautomer=tautomer)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)
