"""Multi-parameter optimisation — the properties a shortlist has to clear besides selectivity.

A funnel that ranks on the selectivity gap alone will happily hand a medicinal
chemist an insoluble molecule carrying a toxicophore alert. Real screening
cascades score against a target product profile, not a single endpoint, so Tier 2
annotates every survivor with three cheap properties alongside its gap:

  QED         RDKit's drug-likeness estimate (0-1, higher better)
  logS        predicted solubility from the bundled ESOL model
  tox_prob    P(hit in any Tox21 assay) from the bundled Tox21 model — an alert,
              not a safety verdict

They are combined as a **geometric mean of desirabilities**, deliberately not a
weighted sum: a weighted sum lets an excellent QED mask an unacceptable
solubility, whereas a geometric mean sends the whole score to zero when any one
property is disqualifying. That is the behaviour a screening cascade needs.

`mpo_score` never reorders the funnel by itself — the shortlist is still ranked by
selectivity. It is a second axis the dashboard shows, so a selective-but-unusable
molecule is visible as such rather than silently promoted.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

from .models.property_models import load_property_models

RDLogger.DisableLog("rdApp.*")

# Solubility desirability ramp, in log10(mol/L). -4 is roughly 100 uM at a typical
# lead molecular weight — adequate for a discovery compound; below -6 (~1 uM) a
# molecule is hard to formulate and hard to assay honestly.
LOGS_POOR, LOGS_GOOD = -6.0, -4.0

# Toxicity desirability ramp on P(any Tox21 hit). The Tox21 aggregate is a broad
# alert with ROC-AUC ~0.75, so it is scored as a soft penalty across a wide band
# rather than a hard gate at any single probability.
TOX_GOOD, TOX_POOR = 0.2, 0.6


def _ramp(x: np.ndarray, poor: float, good: float) -> np.ndarray:
    """Linear desirability in [0,1]; handles either direction via poor/good order."""
    return np.clip((x - poor) / (good - poor), 0.0, 1.0)


def qed(smiles: list[str]) -> np.ndarray:
    """RDKit QED per molecule, NaN where the SMILES does not parse."""
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        out.append(QED.qed(mol) if mol is not None else np.nan)
    return np.array(out, dtype=float)


def annotate(smiles: list[str]) -> pd.DataFrame:
    """QED, predicted logS, tox alert probability and the combined MPO desirability.

    Returns a frame aligned with `smiles`. If the property-model bundle is absent
    the solubility and toxicity columns are NaN and `mpo` falls back to QED alone,
    so the funnel degrades rather than failing.
    """
    smiles = list(smiles)
    frame = pd.DataFrame({"qed": qed(smiles)})

    models = load_property_models()
    if models is None:
        frame["logS_pred"] = np.nan
        frame["tox_prob"] = np.nan
        frame["mpo"] = frame["qed"]
        return frame

    props = models.predict(smiles)
    frame["logS_pred"] = props["logS_pred"].to_numpy()
    frame["tox_prob"] = props["tox_prob"].to_numpy()

    desirabilities = np.vstack([
        frame["qed"].to_numpy(),
        _ramp(frame["logS_pred"].to_numpy(), LOGS_POOR, LOGS_GOOD),
        _ramp(frame["tox_prob"].to_numpy(), TOX_POOR, TOX_GOOD),
    ])
    # Geometric mean: one disqualifying property drags the whole score down.
    # An unparseable molecule gives an all-NaN column; nanmean warns on it and we
    # overwrite the result with NaN below, so the warning is noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        frame["mpo"] = np.exp(np.nanmean(np.log(np.clip(desirabilities, 1e-6, 1.0)), axis=0))
    frame.loc[frame["qed"].isna(), "mpo"] = np.nan
    return frame
