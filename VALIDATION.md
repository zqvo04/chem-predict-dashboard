# Validation

Measured results only. Every number here is reproducible from a script + the data
source named beside it. Planned-but-unmeasured items live in the
[roadmap](README.md#how-the-funnel-was-built-step-by-step), not here.

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

---

## STEP 5 — conformal prediction intervals (2026-07-24)

Split-conformal regression (`src/conformal.py`, `python -m src.conformal`): fit on
a proper-train split, calibrate the interval half-width on a disjoint calibration
split, measure coverage on a scaffold-disjoint test split, over 5 seeds at 90%
nominal.

| Isoform | empirical coverage @ 90% nominal | interval width (pchembl) |
|---------|:--------------------------------:|:------------------------:|
| JAK1 | 0.904 ± 0.010 | 2.12 ± 0.03 |
| JAK2 | 0.905 ± 0.007 | 2.36 ± 0.12 |
| JAK3 | 0.888 ± 0.023 | 2.40 ± 0.07 |

**Gate 5 passed:** empirical coverage 0.888–0.905 sits inside the 88–92% tolerance
band for a 90% target — and holds on *scaffold-disjoint* test molecules, the honest
stress case for the exchangeability assumption. The coverage-vs-nominal figure
(`figures/conformal_coverage.png`, `scripts/make_coverage_figure.py`) shows all
three isoforms tracking the ideal diagonal across nominal levels 0.5–0.95.

Interval widths of ~±1.1 pchembl (2.1–2.4 full width) are honest about the model's
limits: a 90% interval spans about a 10-fold potency range. The gap `S` interval is
the conservative sum of the two contributing isoform half-widths.

---

## STEP 6 — applicability domain, the money plot (2026-07-24)

Two orthogonal AD signals (`src/applicability.py`): nearest-neighbour ECFP4
**Tanimoto similarity** to the training set, and RDKit-descriptor **leverage**
(hat value vs the 3·p/n threshold). A molecule is in-domain only if both agree.
Money plot over the scaffold-split test sets of all three isoforms, 5 seeds
(`scripts/make_ad_figure.py`, `figures/applicability_error.png`):

- **Continuous (the decisive evidence):** mean |error| rises monotonically as
  nearest-neighbour Tanimoto similarity to training drops — from **0.44 pchembl**
  at similarity ≈ 0.85 to **0.85–0.91 pchembl** at similarity ≈ 0.35, roughly a
  2× increase. Error tracks distance-from-domain exactly as an AD should.
- **Binary flag:** in-domain |error| **0.491** vs out-of-domain **0.638** — a
  **1.30×** margin.

**Gate 6 passed** on the continuous trend: prediction error is systematically
higher out of domain. The binary margin is deliberately conservative — on
chemically homogeneous JAK ChEMBL only **1.8%** of scaffold-split test molecules
fall outside the (conventional, untuned) Tanimoto-0.3 boundary, so the two-bar
view understates what the similarity-binned curve shows plainly. AD earns its keep
on the **diverse wide library** (STEP 7), where most molecules *are* out of domain
— there it is the mechanism that restricts trust to the in-domain subset.

AD propagates to the selectivity gap: `S` is flagged **uncertain** if any
contributing isoform model is out-of-domain (worst-case), carrying the non-binder
burden regression alone cannot.

---

## STEP 7 — wide library + tiered screen + SELECT export (2026-07-24)

**Wide library** (`src/data/library.py`): 7 823 unique, canonical, drug-like,
**target-agnostic** molecules (Tox21 collection — diverse, not JAK actives).
Demo-scale and offline-cached; the pipeline scales to larger libraries.

**Tiered screen** (`src/funnel.py`, `python -m src.funnel`), JAK1 over JAK2/JAK3:

- Tier 0 Ro5 + PAINS → Tier 1 per-isoform gap `S` + potency floor (top 300 by gap)
  → Tier 2 conformal interval + applicability domain on survivors → **shortlist of 60**.
- Of the 60 selective, drug-like candidates, only **3 are in-domain**; the rest are
  flagged **uncertain**.

That 3/60 is the funnel working as designed, not a failure: a diverse
target-agnostic library is mostly *outside* the JAK training domain, so AD (STEP 6)
restricts trust to the small in-domain subset. The wide screen applies the model
broadly and cheaply; AD is what keeps the output honest. Expensive per-molecule
work (AD nearest-neighbour, intervals) runs only on the ≤300 Tier-1 survivors, not
the whole library — the funnel economics.

**Honest limitation:** the gap's conservative 90% interval (sum of two isoform
half-widths, ≈ ±2.2 pchembl) is wide and often crosses zero even when the point
gap is clearly selective — the ranking is trustworthy, the per-molecule interval
tempers confidence rather than confirming selectivity.

**SELECT export** (`src/loop_contract.py`): a picked shortlist becomes a versioned
JSON contract pinning `model_ids` (content-addressed, e.g.
`CHEMBL2835@fd9840028c`), `conformal_alpha`, and `code_version`; it round-trips and
`assert_models_match` rejects a mismatch — so the Stage-A deep dive can only
re-score through the identical models. The dashboard (`app.py`) gains a
"JAK selectivity funnel" mode: shortlist table with gap + interval + in/out-of-domain
badge, a SELECT multiselect, and a contract download button.

---

## STEP 8 — deep dive + loop closure (2026-07-24)

One real case flowed **B → SELECT → A → re-score** (`scripts/run_loop.py`,
artifacts in `examples/`):

1. **B:** screened the wide library → shortlist; the top **3 in-domain** cases
   exported as `examples/loop_case_B_export.json`.
2. **A:** `src/deep_dive.run_deep_dive` asserted the contract's `model_ids` match
   the current models, generated **90 analogues** (`src/generate.py`, CPU aromatic
   decoration), and **re-scored them through the same `src` funnel scoring**.

| set | n | median gap | max gap | % ≥10× selective | % in-domain |
|-----|--:|:----------:|:-------:|:----------------:|:-----------:|
| before (selected) | 3 | +1.29 | +1.39 | 100% | 100% |
| after (generated) | 90 | +1.21 | +1.79 | 71% | 19% |

The honest before/after story: generation reached a **higher max gap (+1.79 vs
+1.39)** and produced an in-domain analogue that **improves on its parent** —
`COc1cc(NC(=O)c2cc(Cl)ccc2O)c(Cl)cc1[N+](=O)[O-]`, gap **+1.74** (parent +1.39).
But only **19%** of generated analogues stay in-domain: decorating a scaffold
often pushes molecules out of the training domain, and AD flags them — the loop is
honest, not triumphant. `figures/loop_before_after.png` shows the shift; the
re-scored analogues are written back as `examples/loop_case_A_rescore.json`
(`stage: A_rescore`), so they re-enter Stage B — the funnel is a cycle.

**Gate 8 passed:** the loop runs end-to-end on one worked case, re-scoring uses the
identical `src` (enforced by `assert_models_match`), and the report ends with
*"in-silico hypothesis — requires wet-lab validation."*

**Scope note:** the confirmatory **docking** arm of the deep dive is documented as
an optional GPU seam in `notebooks/deep_dive.ipynb` (AutoDock Vina, orthogonal
corroboration only), not executed here — the loop-closing requirement is the
same-model re-scoring of generated analogues, which is done on CPU.

### Where this could still fail

- **≥100× selectivity is thin** (30 / 53 / 39 at S ≥ 2). A strong-selectivity story
  should use the pairwise view or the ≥10× threshold, and say so.
- Median-over-assays hides cross-assay disagreement; a molecule with wide
  inter-assay spread carries a noisier gap than the point value suggests.
- The cross-measured set is biased toward well-studied chemotypes — validation
  there may over-state performance on novel scaffolds (which is why scaffold-split
  evaluation and AD both matter downstream).
