#!/usr/bin/env python3
"""Two audits of the headline selectivity claim, using the assay provenance in the data.

The repo claims the predicted selectivity gap ranks molecules like the measured gap
does (Spearman ~0.80, ~4.5x enrichment). Both audits ask whether that claim
survives a harder question — neither adds a feature, and either could invalidate a
headline number, which is the point.

  1. ASSAY HETEROGENEITY.  The training label is a median pchembl over IC50, Ki, Kd
     and EC50 pooled together. For an ATP-competitive kinase inhibitor an IC50
     depends on the assay's ATP concentration while a Ki/Kd does not, and JAK1/2/3
     do not share an ATP Km — so a gap assembled from IC50s measured under
     different conditions can carry an assay artefact rather than a real
     selectivity difference. Re-running the validation on the Ki/Kd-only subset
     removes that confound. If the correlation holds, the artefact explanation is
     ruled out; if it collapses, the headline number was measuring the assay.

  2. TIME SPLIT.  A scaffold split asks "does this generalise to new chemotypes?".
     What a deployed screen actually faces is "does this predict what nobody had
     published yet?". Cutting on each molecule's first publication year is the
     closest free proxy for prospective validation, and it is strictly harder than
     a scaffold split: the test molecules are not merely new scaffolds, they are
     chemistry that did not exist in the literature at training time.

Both report the identical metric as Gate 4 (`src.selectivity.evaluate_split`), so
the numbers are directly comparable to VALIDATION.md.

    python scripts/assay_time_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import jak                                          # noqa: E402
from src.models.features import morgan_matrix                     # noqa: E402
from src.models.scaffold_split import scaffold_split              # noqa: E402
from src.selectivity import OFFS, SEEDS, TARGET, evaluate_split   # noqa: E402

MIN_TEST = 60          # below this a Spearman on the test split is not worth reporting
SUBSAMPLE_DRAWS = 5    # random draws for the sample-size control
TOP_FRAC = 0.1         # top-decile, matching src.selectivity._enrichment


def enrichment_ceiling(base_rate: float, frac: float = TOP_FRAC) -> float:
    """The largest enrichment the top-`frac` could possibly show at this base rate.

    Raw enrichment is not comparable across splits with different base rates: if
    34 % of the test set is already selective, a perfect ranker tops out at 1/0.34
    = 2.9x, so a 2.6x is near-perfect rather than a decline. Every enrichment below
    is therefore also reported as a percentage of this ceiling.
    """
    return min(1.0 / base_rate, 1.0 / frac) if base_rate > 0 else float("nan")


def _cross(column: str) -> pd.DataFrame:
    """Cross-measured frame built from one pchembl column, carrying year_first."""
    frames, years = [], []
    for iso in jak.TARGETS:
        d = jak.build_isoform_dataset(iso).set_index("smi")
        frames.append(d[column].rename(iso).dropna())
        years.append(d["year_first"].rename(iso))
    cross = pd.concat(frames, axis=1, join="inner")
    # The molecule's arrival date is the earliest year any isoform measured it.
    cross["year_first"] = pd.concat(years, axis=1).min(axis=1).reindex(cross.index)
    return cross.reset_index()


def _prepare(cross: pd.DataFrame):
    smiles = cross["smi"].tolist()
    X, mask = morgan_matrix(smiles)
    cross = cross[mask].reset_index(drop=True)
    kept = [s for s, keep in zip(smiles, mask) if keep]
    gap = cross[TARGET].to_numpy() - cross[list(OFFS)].max(axis=1).to_numpy()
    return X, cross, gap, kept


def _scaffold_seeds(cross: pd.DataFrame) -> dict:
    X, cross, gap, kept = _prepare(cross)
    if len(kept) < MIN_TEST * 5:
        return {"n": len(kept), "too_small": True}
    rows = [evaluate_split(X, cross, gap, *scaffold_split(kept, 0.2, seed=s))
            for s in SEEDS]
    diff, direct, enrich = (np.array([r[i] for r in rows]) for i in range(3))
    return {"n": len(kept), "too_small": False,
            "spearman": (diff.mean(), diff.std()),
            "direct": (direct.mean(), direct.std()),
            "enrichment": (np.nanmean(enrich), np.nanstd(enrich)),
            "base_rate": float((gap >= 1.0).mean())}


def _report(label: str, res: dict) -> None:
    if res.get("too_small"):
        print(f"\n{label}\n  n={res['n']} — too few to split; no comparable number.")
        return
    ceiling = enrichment_ceiling(res["base_rate"])
    print(f"\n{label}")
    print(f"  cross-measured n           : {res['n']}")
    print(f"  Spearman (difference)      : {res['spearman'][0]:.3f} ± {res['spearman'][1]:.3f}")
    print(f"  Spearman (direct)          : {res['direct'][0]:.3f} ± {res['direct'][1]:.3f}")
    print(f"  Top-decile enrichment      : {res['enrichment'][0]:.2f} ± {res['enrichment'][1]:.2f}x"
          f"   ({res['enrichment'][0] / ceiling:.0%} of the {ceiling:.1f}x ceiling)")
    print(f"  base >=10x-selective rate  : {res['base_rate']:.1%}")


def _subsampled(cross: pd.DataFrame, n: int, draws: int = SUBSAMPLE_DRAWS) -> dict:
    """The all-types set cut down to `n` molecules — the control for sample size.

    Without this the assay comparison is uninterpretable: the Ki/Kd subset is not
    only a different assay type, it is ~10x smaller, and a regressor trained on a
    tenth of the data would degrade for that reason alone.
    """
    out = []
    for draw in range(draws):
        res = _scaffold_seeds(cross.sample(n=n, random_state=draw).reset_index(drop=True))
        if not res["too_small"]:
            out.append(res)
    if not out:
        return {"n": n, "too_small": True}
    pick = lambda key, i: float(np.mean([r[key][i] for r in out]))
    return {"n": n, "too_small": False,
            "spearman": (pick("spearman", 0), float(np.std([r["spearman"][0] for r in out]))),
            "direct": (pick("direct", 0), float(np.std([r["direct"][0] for r in out]))),
            "enrichment": (pick("enrichment", 0), float(np.std([r["enrichment"][0] for r in out]))),
            "base_rate": float(np.mean([r["base_rate"] for r in out]))}


def audit_assay_type() -> None:
    print("=" * 78)
    print("AUDIT 1 — is the selectivity gap an artefact of pooled assay types?")
    print("=" * 78)

    full = _scaffold_seeds(_cross("pchembl"))
    _report("all types (IC50+Ki+Kd+EC50, as deployed)", full)

    kikd_cross = _cross("pchembl_kikd")
    kikd = _scaffold_seeds(kikd_cross)
    _report("Ki/Kd only (ATP-independent)", kikd)

    if not kikd.get("too_small"):
        control = _subsampled(_cross("pchembl"), n=kikd["n"])
        _report(f"CONTROL — all types, randomly cut to n={kikd['n']} "
                f"({SUBSAMPLE_DRAWS} draws x {len(SEEDS)} seeds)", control)
        print("\n  How to read this: the Ki/Kd subset differs from the deployed set in")
        print("  TWO ways at once — assay type and sample size. Compare Ki/Kd against")
        print("  the CONTROL, not against the full set. If the control falls just as")
        print("  far, the drop is sample size and the assay confound is not shown.")


def audit_time_split() -> None:
    print()
    print("=" * 78)
    print("AUDIT 2 — does the gap predict chemistry that had not been published yet?")
    print("=" * 78)
    X, cross, gap, kept = _prepare(_cross("pchembl"))
    years = cross["year_first"].to_numpy()

    print(f"\n  cross-measured n={len(kept)}, "
          f"years {np.nanmin(years):.0f}-{np.nanmax(years):.0f}")
    print(f"\n  {'cutoff':>8} {'train':>7} {'test':>7} "
          f"{'Spearman':>10} {'direct':>8} {'enrich':>8} {'ceiling':>8} {'%max':>6} {'base':>7}")
    for cutoff in (2015, 2016, 2017, 2018, 2019, 2020):
        tr = np.flatnonzero(years <= cutoff)
        te = np.flatnonzero(years > cutoff)
        if len(te) < MIN_TEST or len(tr) < MIN_TEST:
            print(f"  {cutoff:>8} {len(tr):>7} {len(te):>7}    (too few to report)")
            continue
        diff, direct, enrich = evaluate_split(X, cross, gap, tr, te)
        base = float((gap[te] >= 1.0).mean())
        ceiling = enrichment_ceiling(base)
        print(f"  {cutoff:>8} {len(tr):>7} {len(te):>7} "
              f"{diff:>10.3f} {direct:>8.3f} {enrich:>8.2f} {ceiling:>8.2f} "
              f"{enrich / ceiling:>6.0%} {base:>7.1%}")

    ref = _scaffold_seeds(_cross("pchembl"))
    ref_ceiling = enrichment_ceiling(ref["base_rate"])
    print(f"\n  reference, scaffold split (same data, {len(SEEDS)} seeds): "
          f"Spearman {ref['spearman'][0]:.3f} ± {ref['spearman'][1]:.3f}, "
          f"enrichment {ref['enrichment'][0]:.2f}x "
          f"({ref['enrichment'][0] / ref_ceiling:.0%} of its {ref_ceiling:.1f}x ceiling), "
          f"train n≈{int(len(gap) * 0.8)}")
    print("\n  How to read this. A time split is strictly harder than a scaffold split:")
    print("  the test molecules are not just new scaffolds, they are chemistry absent")
    print("  from the literature at training time. Two things also move with the")
    print("  cutoff and must not be mistaken for the effect being measured — the")
    print("  TRAIN SIZE (compare the latest cutoff against the reference, whose train")
    print("  set is closest in size) and the BASE RATE, which caps how large an")
    print("  enrichment can possibly get. Read %max, not raw enrichment.")


def main() -> None:
    audit_assay_type()
    audit_time_split()


if __name__ == "__main__":
    main()
