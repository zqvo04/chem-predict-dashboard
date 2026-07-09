"""Phase 3b: generic drug-property models from MoleculeNet.

Two static, target-independent models trained once on public MoleculeNet data:

  - **solubility** (ESOL, regression)    -> predicted logS (higher = more soluble)
  - **toxicity**   (Tox21, classification) -> P(hit in any of 12 assays), used as
    a broad "toxicophore alert" (a proxy, not a verdict — see notes below)

Both reuse the project's Morgan fingerprints and scaffold-split evaluation. The
trained bundle is small and shipped in assets/models/property_models.pkl.

CLI (re-train and refresh the bundle):
    python -m src.models.property_models

Honest scope notes:
  - Tox21 assays are specific mechanisms (nuclear-receptor / stress-response),
    so the aggregate "any hit" is a screening alert, not a general safety claim.
  - ESOL is ~1100 molecules; the model is a rough solubility prior.
"""
from __future__ import annotations

import gzip
import io
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

from .features import FP_SIZE, _GENERATOR
from .scaffold_split import scaffold_split

# Solubility is driven by physicochemistry (logP, TPSA, MW ...) far more than by
# substructure alone, so we augment the Morgan fingerprint with a few RDKit
# descriptors. This lifts scaffold-split ESOL R^2 from ~0.41 to ~0.86.
_N_DESCRIPTORS = 7

_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "data" / "moleculenet"
BUNDLE_PATH = _ROOT / "assets" / "models" / "property_models.pkl"

_BASE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
_TOX21_ASSAYS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                 "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]


@dataclass
class PropertyModels:
    solubility: HistGradientBoostingRegressor
    toxicity: HistGradientBoostingClassifier
    metrics: dict = field(default_factory=dict)

    def predict(self, smiles: list[str]) -> pd.DataFrame:
        """Predicted logS and toxicity probability, aligned with input rows."""
        X, mask = _augmented_matrix(list(smiles))
        logS = np.full(len(mask), np.nan)
        tox = np.full(len(mask), np.nan)
        if X.shape[0]:
            logS[mask] = self.solubility.predict(X)
            tox[mask] = self.toxicity.predict_proba(X)[:, 1]
        return pd.DataFrame({"logS_pred": logS, "tox_prob": tox})


def _descriptors(mol: Chem.Mol) -> list[float]:
    return [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol)]


def _augmented_matrix(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Morgan fingerprint concatenated with RDKit descriptors, plus a parse mask."""
    rows: list[np.ndarray] = []
    mask: list[bool] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            mask.append(False)
            continue
        fp = _GENERATOR.GetFingerprintAsNumPy(mol)
        rows.append(np.concatenate([fp, _descriptors(mol)]))
        mask.append(True)
    width = FP_SIZE + _N_DESCRIPTORS
    X = np.array(rows, dtype=float) if rows else np.empty((0, width))
    return X, np.array(mask, dtype=bool)


def _download(filename: str) -> bytes:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cached = DATA_DIR / filename
    if cached.exists():
        return cached.read_bytes()
    resp = requests.get(_BASE + filename, timeout=90)
    resp.raise_for_status()
    cached.write_bytes(resp.content)
    return resp.content


def load_esol() -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(_download("delaney-processed.csv")))
    return pd.DataFrame({
        "smiles": df["smiles"],
        "logS": df["measured log solubility in mols per litre"],
    }).dropna()


def load_tox21() -> pd.DataFrame:
    raw = gzip.decompress(_download("tox21.csv.gz"))
    df = pd.read_csv(io.BytesIO(raw))
    assays = df[_TOX21_ASSAYS]
    measured = assays.notna().any(axis=1)
    label = (assays == 1).any(axis=1).astype(int)
    out = pd.DataFrame({"smiles": df["smiles"], "tox": label})[measured]
    return out.dropna(subset=["smiles"])


def _xy(df: pd.DataFrame, target_col: str):
    X, mask = _augmented_matrix(df["smiles"].tolist())
    y = df[target_col].to_numpy()[mask]
    kept = [s for s, keep in zip(df["smiles"], mask) if keep]
    return X, y, kept


def _train_solubility() -> tuple[HistGradientBoostingRegressor, dict]:
    X, y, smiles = _xy(load_esol(), "logS")
    tr, te = scaffold_split(smiles, test_frac=0.2)
    eval_m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=0).fit(X[tr], y[tr])
    pred = eval_m.predict(X[te])
    metrics = {"task": "solubility", "n": len(y),
               "r2": float(r2_score(y[te], pred)),
               "rmse": float(np.sqrt(mean_squared_error(y[te], pred)))}
    final = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=0).fit(X, y)
    return final, metrics


def _train_toxicity() -> tuple[HistGradientBoostingClassifier, dict]:
    X, y, smiles = _xy(load_tox21(), "tox")
    tr, te = scaffold_split(smiles, test_frac=0.2)
    eval_m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=0).fit(X[tr], y[tr])
    proba = eval_m.predict_proba(X[te])[:, 1]
    metrics = {"task": "toxicity", "n": len(y),
               "roc_auc": float(roc_auc_score(y[te], proba)),
               "positive_rate": float(y.mean())}
    final = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=0).fit(X, y)
    return final, metrics


def train_property_models() -> PropertyModels:
    sol, sol_m = _train_solubility()
    tox, tox_m = _train_toxicity()
    return PropertyModels(solubility=sol, toxicity=tox,
                          metrics={"solubility": sol_m, "toxicity": tox_m})


def load_property_models(allow_train: bool = False) -> PropertyModels | None:
    """Load the bundled models. If missing, optionally train (needs network)."""
    if BUNDLE_PATH.exists():
        with open(BUNDLE_PATH, "rb") as fh:
            return pickle.load(fh)
    if not allow_train:
        return None
    models = train_property_models()
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BUNDLE_PATH, "wb") as fh:
        pickle.dump(models, fh)
    return models


def _main() -> None:
    models = train_property_models()
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BUNDLE_PATH, "wb") as fh:
        pickle.dump(models, fh)
    sm, tm = models.metrics["solubility"], models.metrics["toxicity"]
    print(f"Solubility (ESOL) : n={sm['n']}  R2={sm['r2']:.3f}  RMSE={sm['rmse']:.3f}")
    print(f"Toxicity  (Tox21) : n={tm['n']}  ROC-AUC={tm['roc_auc']:.3f}  "
          f"positives={tm['positive_rate']:.1%}")
    print(f"Saved bundle -> {BUNDLE_PATH}")


if __name__ == "__main__":
    _main()
