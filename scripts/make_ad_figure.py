"""Applicability-domain money plot: prediction error rises out of domain.

Left: mean |error| vs nearest-neighbour Tanimoto similarity to the training set —
error climbs as molecules get less like anything seen in training. Right: mean
|error| for in- vs out-of-domain test molecules (the combined Tanimoto+leverage
flag). Also prints the in/out error ratio that Gate 6 checks.

    python scripts/make_ad_figure.py  ->  figures/applicability_error.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.applicability import (SEEDS, TANIMOTO_IN_DOMAIN, TARGET_ISOFORMS,  # noqa: E402
                               _fit, in_domain)
from src.data import jak                             # noqa: E402
from src.models.features import morgan_matrix        # noqa: E402
from src.models.scaffold_split import scaffold_split  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures" / "applicability_error.png"
BINS = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])


def main() -> None:
    errs, sims, dom = [], [], []
    for iso in TARGET_ISOFORMS:
        data = jak.build_isoform_dataset(iso)
        smiles = data["smi"].tolist()
        X, mask = morgan_matrix(smiles)
        y = data["pchembl"].to_numpy()[mask]
        kept = [s for s, k in zip(smiles, mask) if k]
        for seed in SEEDS:
            tr, te = scaffold_split(kept, test_frac=0.2, seed=seed)
            model = _fit(X[tr], y[tr])
            errs.append(np.abs(y[te] - model.predict(X[te])))
            flags = in_domain([kept[i] for i in te], [kept[i] for i in tr])
            sims.append(flags["nn_sim"])
            dom.append(flags["in_domain"])
    err = np.concatenate(errs); sim = np.concatenate(sims); dom = np.concatenate(dom)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))

    centres, means = [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (sim >= lo) & (sim < hi)
        if m.sum() > 20:
            centres.append((lo + hi) / 2)
            means.append(err[m].mean())
    axL.plot(centres, means, marker="o", color="#d62728")
    axL.axvline(TANIMOTO_IN_DOMAIN, ls="--", color="grey", lw=1)
    axL.text(TANIMOTO_IN_DOMAIN + 0.01, axL.get_ylim()[1], " in-domain →",
             color="grey", va="top", fontsize=9)
    axL.set_xlabel("nearest-neighbour Tanimoto similarity to training set")
    axL.set_ylabel("mean |prediction error|  (pchembl)")
    axL.set_title("Error rises as molecules leave the domain")

    in_err, out_err = err[dom].mean(), err[~dom].mean()
    axR.bar(["in-domain", "out-of-domain"], [in_err, out_err],
            color=["#2ca02c", "#d62728"])
    axR.set_ylabel("mean |prediction error|  (pchembl)")
    axR.set_title(f"Out-of-domain error is {out_err / in_err:.2f}× higher")
    for i, v in enumerate([in_err, out_err]):
        axR.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"in-domain |err|={in_err:.3f}  out-of-domain |err|={out_err:.3f}  "
          f"ratio={out_err / in_err:.2f}x  (%out={(~dom).mean():.1%})")
    print(f"Saved -> {FIG}")


if __name__ == "__main__":
    main()
