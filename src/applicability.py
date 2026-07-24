"""STEP 6: applicability domain (Stage-B Tier 2).

Two orthogonal "is this prediction extrapolation?" signals:

  * **Tanimoto distance** — 1 − (max ECFP4 Tanimoto similarity to any training
    molecule). Fingerprint-space: far from every training molecule = out-of-domain.
  * **Descriptor leverage** — the hat value h = xᵀ(XᵀX)⁻¹x in RDKit-descriptor
    space; > 3·p/n (the standard threshold) = extrapolation in physicochemical space.

A prediction is **in-domain** only if both agree; either alarm marks it
out-of-domain. AD propagates to the selectivity gap: `S` is *uncertain* if any
contributing isoform model is out-of-domain (worst-case). AD also carries the
non-binder burden regression alone cannot (DESIGN_DECISIONS section 1).

The money plot (`scripts/make_ad_figure.py`) shows prediction error rises as
molecules leave the domain — the check that AD is real, not decorative.

CLI (in- vs out-of-domain error margin per isoform):
    python -m src.applicability
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import rdFingerprintGenerator

from .data import jak
from .models.features import morgan_matrix
from .models.scaffold_split import scaffold_split
from .conformal import _fit

TARGET_ISOFORMS = ("JAK1", "JAK2", "JAK3")
SEEDS = (0, 1, 2, 3, 4)
TANIMOTO_IN_DOMAIN = 0.30       # nearest-neighbour similarity below this = out-of-domain
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _bitvects(smiles: list[str]) -> list:
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        out.append(_GEN.GetFingerprint(mol) if mol is not None else None)
    return out


def tanimoto_nn_similarity(query: list[str], train: list[str]) -> np.ndarray:
    """Max ECFP4 Tanimoto similarity of each query to the training set."""
    train_fps = [fp for fp in _bitvects(train) if fp is not None]
    sims = np.zeros(len(query))
    for i, fp in enumerate(_bitvects(query)):
        sims[i] = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) if fp is not None else 0.0
    return sims


def _descriptors(smiles: list[str]) -> np.ndarray:
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        rows.append([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                     Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
                     Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol)]
                    if mol is not None else [np.nan] * 7)
    return np.array(rows, dtype=float)


def leverage(query_desc: np.ndarray, train_desc: np.ndarray) -> tuple[np.ndarray, float]:
    """Hat values h = xᵀ(XᵀX)⁻¹x in standardised descriptor space, and the 3·p/n threshold."""
    mu, sd = train_desc.mean(0), train_desc.std(0)
    sd[sd == 0] = 1.0
    Xtr = (train_desc - mu) / sd
    Xq = (query_desc - mu) / sd
    cov_inv = np.linalg.pinv(Xtr.T @ Xtr)
    h = np.einsum("ij,jk,ik->i", Xq, cov_inv, Xq)
    threshold = 3.0 * Xtr.shape[1] / Xtr.shape[0]
    return h, float(threshold)


def in_domain(query: list[str], train: list[str]) -> dict:
    """Per-molecule AD flags for a query set against a training set."""
    nn_sim = tanimoto_nn_similarity(query, train)
    h, h_thr = leverage(_descriptors(query), _descriptors(train))
    tan_ok = nn_sim >= TANIMOTO_IN_DOMAIN
    lev_ok = h <= h_thr
    return {"nn_sim": nn_sim, "tanimoto_ok": tan_ok,
            "leverage": h, "leverage_ok": lev_ok,
            "in_domain": tan_ok & lev_ok}


@dataclass
class ADMetrics:
    isoform: str
    n_seeds: int
    err_in_mean: float
    err_out_mean: float
    frac_out_mean: float


def evaluate_ad(isoform: str, seeds: tuple[int, ...] = SEEDS, use_cache: bool = True) -> ADMetrics:
    """Mean |error| for in- vs out-of-domain test molecules (the money-plot claim)."""
    data = jak.build_isoform_dataset(isoform, use_cache=use_cache)
    smiles = data["smi"].tolist()
    X, mask = morgan_matrix(smiles)
    y = data["pchembl"].to_numpy()[mask]
    kept = [s for s, k in zip(smiles, mask) if k]

    err_in, err_out, frac_out = [], [], []
    for seed in seeds:
        tr, te = scaffold_split(kept, test_frac=0.2, seed=seed)
        model = _fit(X[tr], y[tr])
        err = np.abs(y[te] - model.predict(X[te]))
        flags = in_domain([kept[i] for i in te], [kept[i] for i in tr])
        dom = flags["in_domain"]
        if dom.any():
            err_in.append(err[dom].mean())
        if (~dom).any():
            err_out.append(err[~dom].mean())
        frac_out.append(float((~dom).mean()))
    return ADMetrics(
        isoform=isoform, n_seeds=len(seeds),
        err_in_mean=float(np.mean(err_in)), err_out_mean=float(np.mean(err_out)),
        frac_out_mean=float(np.mean(frac_out)),
    )


def _main() -> None:
    print(f"In- vs out-of-domain test error ({len(SEEDS)} seeds, scaffold split):")
    print(f"  {'isoform':7} {'|err| in':>10} {'|err| out':>11} {'ratio':>7} {'% out':>7}")
    for iso in TARGET_ISOFORMS:
        m = evaluate_ad(iso)
        ratio = m.err_out_mean / m.err_in_mean
        print(f"  {iso:7} {m.err_in_mean:10.3f} {m.err_out_mean:11.3f} "
              f"{ratio:6.2f}x {m.frac_out_mean:6.1%}")


if __name__ == "__main__":
    _main()
