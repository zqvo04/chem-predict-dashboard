"""Hero figure: the selectivity ranking flip.

Scores the cross-measured JAK molecules with the deployed per-isoform regressors
and plots predicted **potency** (pchembl for the target) against predicted
**selectivity gap** S. Points are shaded by the *measured* gap, so the figure
doubles as a validation: selective-by-model molecules really are selective-by-data.

The flip: the molecule ranked #1 by potency alone (rightmost) is not the molecule
ranked #1 by selectivity (topmost). A potency-only screen would surface the former;
the selectivity-aware screen surfaces the latter. Both are highlighted.

    python scripts/make_hero_figure.py   ->  figures/selectivity_ranking_flip.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.features import morgan_matrix          # noqa: E402
from src.models.isoform_regressor import train_and_cache  # noqa: E402
from src.selectivity import OFFS, TARGET               # noqa: E402
from src.data import jak                                # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures" / "selectivity_ranking_flip.png"


def main() -> None:
    cross = jak.build_cross_measured()
    smiles = cross["smi"].tolist()
    X, mask = morgan_matrix(smiles)
    cross = cross[mask].reset_index(drop=True)

    models = {iso: train_and_cache(iso, use_cache=True).model for iso in (TARGET, *OFFS)}
    pred = {iso: models[iso].predict(X) for iso in (TARGET, *OFFS)}
    potency = pred[TARGET]
    gap = potency - np.maximum.reduce([pred[o] for o in OFFS])
    measured_gap = cross[TARGET].to_numpy() - cross[list(OFFS)].max(axis=1).to_numpy()

    top_potency = int(np.argmax(potency))     # #1 if you rank by potency only
    top_gap = int(np.argmax(gap))             # #1 if you rank by selectivity

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(potency, gap, c=measured_gap, cmap="coolwarm", vmin=-2, vmax=2,
                    s=14, alpha=0.6, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(f"measured gap  ({TARGET} − max off)")

    ax.axhline(1.0, ls="--", lw=1, color="grey")
    ax.text(ax.get_xlim()[0], 1.03, "  ≥10× selective", color="grey", fontsize=9, va="bottom")

    for idx, label, color in [(top_potency, "#1 by potency only", "#1f77b4"),
                              (top_gap, "#1 by selectivity", "#d62728")]:
        ax.scatter(potency[idx], gap[idx], s=140, facecolors="none",
                   edgecolors=color, linewidths=2)
        ax.annotate(label, (potency[idx], gap[idx]), textcoords="offset points",
                    xytext=(8, 8), color=color, fontsize=10, fontweight="bold")

    ax.set_xlabel(f"predicted potency  pchembl_pred({TARGET})")
    ax.set_ylabel(f"predicted selectivity gap  S = {TARGET} − max({', '.join(OFFS)})")
    ax.set_title("Selectivity ranking flip: the most potent is not the most selective")
    fig.tight_layout()

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"potency #1: pchembl={potency[top_potency]:.2f} gap={gap[top_potency]:+.2f}")
    print(f"selectivity #1: pchembl={potency[top_gap]:.2f} gap={gap[top_gap]:+.2f}")
    print(f"Saved -> {FIG}")


if __name__ == "__main__":
    main()
