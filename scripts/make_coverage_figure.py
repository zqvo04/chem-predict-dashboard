"""Conformal coverage figure + the 90% gate numbers.

Computes each isoform's scaffold-disjoint calibration/test residuals once, then
sweeps the nominal level to show empirical coverage tracks the diagonal, and
prints the 90%-nominal coverage that Gate 5 checks.

    python scripts/make_coverage_figure.py  ->  figures/conformal_coverage.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.conformal import (SEEDS, TARGET_ISOFORMS, _seed_errors,  # noqa: E402
                           conformal_quantile)

FIG = Path(__file__).resolve().parents[1] / "figures" / "conformal_coverage.png"
NOMINALS = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])


def main() -> None:
    # Cache (test_err, cal_res) per isoform+seed so each nominal level reuses them.
    errs = {iso: [_seed_errors(iso, s) for s in SEEDS] for iso in TARGET_ISOFORMS}

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="ideal (y = x)")

    print("Split-conformal coverage (scaffold-disjoint, mean over seeds):")
    for iso in TARGET_ISOFORMS:
        emp = []
        for nom in NOMINALS:
            alpha = 1 - nom
            covs = [float((te <= conformal_quantile(cal, alpha)).mean())
                    for te, cal in errs[iso]]
            emp.append(np.mean(covs))
        emp = np.array(emp)
        ax.plot(NOMINALS, emp, marker="o", label=iso)
        j = int(np.where(NOMINALS == 0.90)[0][0])
        print(f"  {iso}: 90% nominal -> {emp[j]:.3f} empirical")

    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage (scaffold-split test)")
    ax.set_title("Conformal intervals: empirical coverage tracks nominal")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"Saved -> {FIG}")


if __name__ == "__main__":
    main()
