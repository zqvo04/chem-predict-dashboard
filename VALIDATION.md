# Validation

Measured results only. Every number here is reproducible from a script + the data
source named beside it. Planned-but-unmeasured items live in the
[roadmap](README.md#roadmap--jak-selectivity-screening-funnel), not here.

---

## Gate 0 — JAK data audit (2026-07-24)

**Purpose.** Before building the selectivity funnel, measure the real JAK data to
(a) confirm enough cross-measured molecules exist to *validate* a selectivity gap,
and (b) test the assumption behind the active/inactive classification plan.

**Source.** ChEMBL REST (`ebi.ac.uk`), targets JAK1 `CHEMBL2835`, JAK2 `CHEMBL2971`,
JAK3 `CHEMBL2148`. All quantified-pchembl records (IC50/Ki/Kd/EC50), one **median
pchembl per (molecule, isoform)**, RDKit-canonical SMILES. `max_records=40000` per
isoform (full coverage; pagination did not truncate).

### Per-isoform counts

| Isoform | activity rows | unique molecules | pchembl ≥ 6 (active) | ≤ 5 (inactive) | 5–6 (gray) | pchembl range |
|---------|--------------:|-----------------:|---------------------:|---------------:|-----------:|:-------------:|
| JAK1 | 14 964 | 10 468 | 9 618 | **75** | 775 | 4.0 – 11.0 |
| JAK2 | 18 036 | 12 680 | 11 000 | **333** | 1 347 | 3.8 – 11.0 |
| JAK3 | 10 814 | 7 457 | 6 251 | **245** | 961 | 4.0 – 11.0 |

### Cross-measured (selectivity ground truth)

- **3-way (measured on all of JAK1/2/3):** **3624** molecules (2808 with no gray-zone value).
- Pairwise: JAK1–JAK2 = 8483, JAK1–JAK3 = 4110, JAK2–JAK3 = 4797.

### Selectivity signal — gap-based vs class-based

Strict class-based selective (active target **and** inactive at *both* off-isoforms):
JAK1 = **2**, JAK2 = 3, JAK3 = 1 — effectively empty, because the inactive class is empty.

Gap-based selective on the 3-way set (`S = pchembl(target) − max(off)`):

| Target | S ≥ 1 (≥10×) | S ≥ 2 (≥100×) | median S | max S |
|--------|------------:|-------------:|:--------:|:-----:|
| JAK1 | 593 | 30 | +0.03 | 2.57 |
| JAK2 | 320 | 53 | −0.48 | 2.72 |
| JAK3 | 129 | 39 | −1.16 | 4.75 |

Pairwise JAK1–JAK2 (n = 8483): |S| ≥ 1 for 2632 molecules — 2073 JAK1-selective, 559 JAK2-selective.

### Findings

1. **Active/inactive classification is not viable on this data.** The inactive
   class is 75 / 333 / 245 against ~10k actives — ChEMBL records here are almost
   all measured (expected) binders; true non-binders are right-censored (`>`, no
   pchembl) or absent. 75 negatives cannot train or calibrate a classifier.
2. **Selectivity lives in the pchembl gap, not a class split.** Binarizing at
   pchembl 6 collapses a 9.0 and a 7.0 to the same "active", discarding the
   selectivity signal. The gap-based positives are ample (hundreds per isoform at
   ≥10×; 2073 in the JAK1–JAK2 pairwise view).
3. **Cross-measured N is healthy (3624).** The selectivity gap *can* be validated
   against measured data; the pairwise fallback is not forced.

### Decision

- **Per-isoform model:** pchembl **regression** (not classification).
- **Selectivity:** predicted **gap** `S`, validated against the measured gap on the
  3624-molecule cross-measured set (Spearman + ≥10× enrichment).
- **Non-binder recognition:** carried by the applicability domain; DUD-E decoys
  kept in reserve as an optional future addition.
- **Scope:** 3-isoform selectivity proceeds; pairwise (JAK1–JAK2, richest at ≥100×)
  kept in reserve for a stronger-selectivity story.

Rationale in [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) §1–2.

### Reproduce

```bash
pip install -r requirements.txt
python scripts/gate0_audit.py     # to be added in STEP 1; prints the tables above
```

*(Until STEP 1 lands the committed script, the audit was run from the pull in
`src/data/chembl_client.py` with the medians and thresholds described above; raw
activity pages are cached under `data/cache/`.)*

---

## STEP 2 — JAK data layer (2026-07-24)

Persisted the three per-isoform **regression** datasets (median pchembl per
molecule) and the cross-measured join, via `src/data/jak.py`
(`python -m src.data.jak`). Cached to `data/jak/` (gitignored, regenerable).

| Isoform | molecules | pchembl min | median | max |
|---------|----------:|:-----------:|:------:|:---:|
| JAK1 | 10 468 | 4.01 | 8.00 | 11.00 |
| JAK2 | 12 680 | 3.84 | 7.36 | 10.97 |
| JAK3 | 7 457 | 4.00 | 7.24 | 10.98 |

3-way cross-measured: **3624** (matches Gate 0). Each row carries `n_meas`
(measurements the median was taken over) as provenance. Tests
(`tests/test_jak_data.py`) pin median dedup, SMILES canonicalisation/hygiene, and
the cross-measured intersection offline; a live summary test self-skips without
network.

---

## STEP 3 — per-isoform pchembl regressors (2026-07-24)

HistGradientBoosting regressor per isoform (ECFP4 → pchembl), evaluated over **5
scaffold-split seeds** (`src/models/isoform_regressor.py`,
`python -m src.models.isoform_regressor`). Mean ± std:

| Isoform | n | MAE | RMSE | R² | Spearman |
|---------|--:|:---:|:----:|:--:|:--------:|
| JAK1 | 10 468 | 0.448 ± 0.013 | 0.622 ± 0.015 | 0.768 ± 0.013 | 0.879 ± 0.010 |
| JAK2 | 12 680 | 0.512 ± 0.012 | 0.688 ± 0.014 | 0.713 ± 0.009 | 0.835 ± 0.007 |
| JAK3 | 7 457 | 0.529 ± 0.038 | 0.715 ± 0.048 | 0.712 ± 0.040 | 0.823 ± 0.031 |

Scaffold-split R² 0.71–0.77 with low seed variance — the regression the data
supports (and stronger than v1's single-target EGFR R² ≈ 0.55, consistent with the
larger per-isoform sets). RMSE ≈ 0.62–0.72 pchembl means a typical potency error
under ~1 log unit. Spearman 0.82–0.88 is the number that matters for a *ranking*
screen. The seeded splits are for honest metrics only; the deployed model is refit
on all data and cached to `data/models/jak/{isoform}_reg.pkl`.

**Gate 3 passed:** metrics stable across seeds; scaffold split applied before any
fit (no leakage). This is the Tier-1 engine the selectivity gap is built on next.

---

## STEP 4 — selectivity gap, validated against the measured gap (2026-07-24)

The gap `S = pchembl_pred(JAK1) − max(pchembl_pred(JAK2), pchembl_pred(JAK3))`,
evaluated over 5 scaffold-split seeds of the **cross-measured** set (n = 3624),
predicted vs the *measured* gap (`src/selectivity.py`, `python -m src.selectivity`).
Isoform regressors here are trained only on the scaffold-train molecules
(leak-free, and conservative vs the deployed all-data models).

| Estimator | Spearman (predicted vs measured gap) |
|-----------|:------------------------------------:|
| difference-of-regressors (wide) | 0.797 ± 0.041 |
| direct gap regressor (narrow re-rank) | 0.816 ± 0.044 |

Top-decile enrichment of ≥10×-selective molecules: **4.54 ± 0.56×** over a base
rate of 16.4% — the top 10% ranked by predicted gap concentrate 4.5× more truly
selective molecules than random.

**Gate 4 passed.** Predicted selectivity tracks measured selectivity (Spearman
≈ 0.80), and the ranking enriches for real selective molecules — the first
evidence the funnel's central claim is not hollow. The direct regressor edges the
difference form (no stacked error), exactly the hybrid rationale; both are kept
(difference screens wide, direct re-ranks/validates).

**Hero figure** (`figures/selectivity_ranking_flip.png`, `scripts/make_hero_figure.py`):
predicted potency vs predicted gap for the cross-measured molecules, shaded by
*measured* gap — the molecule ranked #1 by potency alone is not the one ranked #1
by selectivity, the rank flip a potency-only screen would miss.

### Where this could still fail

- **≥100× selectivity is thin** (30 / 53 / 39 at S ≥ 2). A strong-selectivity story
  should use the pairwise view or the ≥10× threshold, and say so.
- Median-over-assays hides cross-assay disagreement; a molecule with wide
  inter-assay spread carries a noisier gap than the point value suggests.
- The cross-measured set is biased toward well-studied chemotypes — validation
  there may over-state performance on novel scaffolds (which is why scaffold-split
  evaluation and AD both matter downstream).
