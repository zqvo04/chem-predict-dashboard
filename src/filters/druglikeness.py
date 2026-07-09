"""Phase 2: drug-likeness filtering.

Two cheap, standard medicinal-chemistry gates applied before we spend model
inference on a molecule:

- **Lipinski's Rule of 5** — flags poor oral-absorption risk. A molecule is
  allowed at most one violation (the conventional interpretation).
- **PAINS** — pan-assay interference substructures (rhodanines, catechols, ...)
  that produce false hits in bioassays. Any match is a fail.

Everything is RDKit-only and runs in microseconds per molecule.
"""
from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Rule-of-5 thresholds (a property is a violation if it exceeds the value).
RO5_THRESHOLDS = {"mw": 500.0, "logp": 5.0, "hbd": 5, "hba": 10}

_DESCRIPTOR_COLUMNS = ["mw", "logp", "hbd", "hba", "tpsa"]
_FILTER_COLUMNS = _DESCRIPTOR_COLUMNS + ["ro5_violations", "ro5_pass", "pains_pass", "druglike"]


def _build_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


_PAINS_CATALOG = _build_pains_catalog()  # built once at import; matching is read-only


def _descriptors(mol: Chem.Mol) -> dict:
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
    }


def _ro5_violations(desc: dict) -> int:
    return (
        int(desc["mw"] > RO5_THRESHOLDS["mw"])
        + int(desc["logp"] > RO5_THRESHOLDS["logp"])
        + int(desc["hbd"] > RO5_THRESHOLDS["hbd"])
        + int(desc["hba"] > RO5_THRESHOLDS["hba"])
    )


def _evaluate(smiles: str, max_ro5_violations: int) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # defensive; Phase 1 already validated, but never crash
        return {c: None for c in _DESCRIPTOR_COLUMNS} | {
            "ro5_violations": None, "ro5_pass": False,
            "pains_pass": False, "druglike": False,
        }
    desc = _descriptors(mol)
    violations = _ro5_violations(desc)
    ro5_pass = violations <= max_ro5_violations
    pains_pass = not _PAINS_CATALOG.HasMatch(mol)
    return desc | {
        "ro5_violations": violations,
        "ro5_pass": ro5_pass,
        "pains_pass": pains_pass,
        "druglike": ro5_pass and pains_pass,
    }


def apply_druglikeness(df: pd.DataFrame, smiles_col: str = "canonical_smiles",
                       max_ro5_violations: int = 1) -> pd.DataFrame:
    """Return a copy of df with drug-likeness descriptors and pass flags added.

    Added columns: mw, logp, hbd, hba, tpsa, ro5_violations, ro5_pass,
    pains_pass, druglike. The input frame is not mutated.
    """
    if df.empty:
        return df.assign(**{c: pd.Series(dtype="object") for c in _FILTER_COLUMNS})

    evaluated = pd.DataFrame(
        [_evaluate(s, max_ro5_violations) for s in df[smiles_col]],
        index=df.index,
    )
    return pd.concat([df, evaluated], axis=1)
