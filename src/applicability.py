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
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import rdFingerprintGenerator

from .data import panel_data
from .panels import DEFAULT_PANEL, PanelSpec, get_panel
from .models.features import morgan_matrix
from .models.scaffold_split import scaffold_split
from .conformal import _fit

SEEDS = (0, 1, 2, 3, 4)
TANIMOTO_IN_DOMAIN = 0.30       # nearest-neighbour similarity below this = out-of-domain
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _bitvects(smiles: list[str]) -> list:
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        out.append(_GEN.GetFingerprint(mol) if mol is not None else None)
    return out


def _nn_similarity(query: list[str], train_fps: list) -> np.ndarray:
    sims = np.zeros(len(query))
    for i, fp in enumerate(_bitvects(query)):
        sims[i] = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) if fp is not None else 0.0
    return sims


def tanimoto_nn_similarity(query: list[str], train: list[str]) -> np.ndarray:
    """Max ECFP4 Tanimoto similarity of each query to the training set."""
    return _nn_similarity(query, [fp for fp in _bitvects(train) if fp is not None])


def _descriptors(smiles: list[str]) -> np.ndarray:
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        rows.append([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                     Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
                     Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol)]
                    if mol is not None else [np.nan] * 7)
    return np.array(rows, dtype=float)


def _standardisation(train_desc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The training-side leverage constants: mean, sd, (XᵀX)⁻¹ and the 3·p/n threshold."""
    mu, sd = train_desc.mean(0), train_desc.std(0)
    sd = sd.copy()
    sd[sd == 0] = 1.0
    Xtr = (train_desc - mu) / sd
    return mu, sd, np.linalg.pinv(Xtr.T @ Xtr), 3.0 * Xtr.shape[1] / Xtr.shape[0]


def _hat(query_desc: np.ndarray, mu: np.ndarray, sd: np.ndarray,
         cov_inv: np.ndarray) -> np.ndarray:
    Xq = (query_desc - mu) / sd
    return np.einsum("ij,jk,ik->i", Xq, cov_inv, Xq)


def leverage(query_desc: np.ndarray, train_desc: np.ndarray) -> tuple[np.ndarray, float]:
    """Hat values h = xᵀ(XᵀX)⁻¹x in standardised descriptor space, and the 3·p/n threshold."""
    mu, sd, cov_inv, threshold = _standardisation(train_desc)
    return _hat(query_desc, mu, sd, cov_inv), float(threshold)


@dataclass(frozen=True)
class ADReference:
    """The training-side half of the AD check, precomputed.

    Both signals depend on the training set only through fixed quantities — the
    training fingerprints for the Tanimoto nearest neighbour, and the
    standardisation plus (XᵀX)⁻¹ for the leverage. Rebuilding those from SMILES
    costs ~10 s per isoform and dominated Tier 2, so they are built once, cached,
    and shipped.
    """
    fps: tuple
    mu: np.ndarray
    sd: np.ndarray
    cov_inv: np.ndarray
    threshold: float


def build_reference(train: list[str]) -> ADReference:
    """Precompute the training side of both AD signals."""
    mu, sd, cov_inv, threshold = _standardisation(_descriptors(train))
    return ADReference(fps=tuple(fp for fp in _bitvects(train) if fp is not None),
                       mu=mu, sd=sd, cov_inv=cov_inv, threshold=threshold)


def _save_reference(ref: ADReference, path: Path) -> None:
    # Fingerprints go out as their raw 256-byte RDKit bit blocks: exact, compact,
    # and ~35x faster to reload than re-deriving them from SMILES.
    packed = np.frombuffer(
        b"".join(DataStructs.BitVectToBinaryText(fp) for fp in ref.fps), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fps=packed.reshape(len(ref.fps), -1), mu=ref.mu,
                        sd=ref.sd, cov_inv=ref.cov_inv, threshold=ref.threshold)


def _load_reference(path: Path) -> ADReference:
    with np.load(path) as z:
        return ADReference(
            fps=tuple(DataStructs.CreateFromBinaryText(row.tobytes()) for row in z["fps"]),
            mu=z["mu"], sd=z["sd"], cov_inv=z["cov_inv"], threshold=float(z["threshold"]))


@lru_cache(maxsize=None)
def load_reference(panel: PanelSpec, isoform: str, use_cache: bool = True) -> ADReference:
    """The cached AD reference for one panel member's full training set.

    Runtime cache first, then the committed copy, then build-and-cache — the same
    order every other artifact in the repo uses. Keyed by panel as well as isoform:
    the reference *is* a training set, so two panels sharing an isoform name but
    not a dataset must not share this cache entry.
    """
    if use_cache:
        for directory in (panel.ad_cache, panel.ad_bundled):
            path = directory / f"{isoform}.npz"
            if path.exists():
                return _load_reference(path)

    train = panel_data.build_isoform_dataset(panel, isoform,
                                             use_cache=use_cache)["smi"].tolist()
    ref = build_reference(train)
    _save_reference(ref, panel.ad_cache / f"{isoform}.npz")
    return ref


def in_domain(query: list[str], train: list[str] | None = None, *,
              reference: ADReference | None = None) -> dict:
    """Per-molecule AD flags for a query set against a training set.

    Pass `reference` (from `load_reference`) to reuse a precomputed training side;
    `train` builds it on the spot, which is what the seeded evaluation below needs.
    """
    ref = reference if reference is not None else build_reference(train)
    nn_sim = _nn_similarity(query, list(ref.fps))
    h = _hat(_descriptors(query), ref.mu, ref.sd, ref.cov_inv)
    tan_ok = nn_sim >= TANIMOTO_IN_DOMAIN
    lev_ok = h <= ref.threshold
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


def evaluate_ad(panel: PanelSpec, isoform: str, seeds: tuple[int, ...] = SEEDS,
                use_cache: bool = True) -> ADMetrics:
    """Mean |error| for in- vs out-of-domain test molecules (the money-plot claim)."""
    data = panel_data.build_isoform_dataset(panel, isoform, use_cache=use_cache)
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
    import sys

    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    print(f"In- vs out-of-domain test error, panel {panel.name} "
          f"({len(SEEDS)} seeds, scaffold split):")
    print(f"  {'isoform':7} {'|err| in':>10} {'|err| out':>11} {'ratio':>7} {'% out':>7}")
    for iso in panel.isoforms:
        m = evaluate_ad(panel, iso)
        ratio = m.err_out_mean / m.err_in_mean
        print(f"  {iso:7} {m.err_in_mean:10.3f} {m.err_out_mean:11.3f} "
              f"{ratio:6.2f}x {m.frac_out_mean:6.1%}")


if __name__ == "__main__":
    _main()
