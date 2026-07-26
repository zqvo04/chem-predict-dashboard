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

---

## AUDIT — assay-type confound and time-split validation (2026-07-26)

**Purpose.** Re-test STEP 4's headline claim (Spearman ≈ 0.80, ~4.5× enrichment)
against two harder questions it had never been asked. Neither audit adds a
feature; both were run because either could invalidate a headline number.

**Source.** ChEMBL re-fetched with assay provenance (`standard_type`, `assay_type`,
`document_year`), standardised to neutral parents. JAK1 10 463 / JAK2 12 671 /
JAK3 7 452 molecules; 3-way cross-measured **3621**. Metric is identical to
Gate 4 (`src.selectivity.evaluate_split`), so all rows below are comparable.
Reproduce: `python scripts/assay_time_audit.py`.

### Assay mix (measured, and it settles one open question)

| Isoform | molecules | with ≥1 Ki/Kd | binding (biochemical) assays |
|---------|----------:|--------------:|-----------------------------:|
| JAK1 | 10 463 | 2 372 (22.7 %) | 99.4 % |
| JAK2 | 12 671 | 2 511 (19.8 %) | 94.7 % |
| JAK3 | 7 452 | 1 162 (15.6 %) | 96.6 % |

**Biochemical-vs-cellular mixing is not a confound here.** 95–99 % of records are
binding assays, so pooled `EC50`/functional readouts are a rounding error. That
concern is measured and dismissed.

### Audit 1 — is the gap an artefact of pooled assay types?

An IC50 for an ATP-competitive kinase inhibitor moves with the assay's ATP
concentration, and JAK1/2/3 do not share an ATP Km; Ki/Kd are ATP-independent.
The Ki/Kd-only subset is therefore the control for that artefact — but it is also
~9× smaller, so a **like-sized random cut of the full set** is required before any
drop can be attributed to assay type.

| Set | n | Spearman (diff) | Spearman (direct) | Top-decile enrichment | base |
|-----|--:|:---------------:|:-----------------:|:---------------------:|-----:|
| all types (as deployed) | 3621 | 0.798 ± 0.044 | 0.816 ± 0.045 | 4.48 ± 0.56× (73 % of ceiling) | 16.4 % |
| **control** — all types, cut to n = 386 | 386 | 0.682 ± 0.045 | 0.722 ± 0.035 | 3.71 ± 0.48× (65 % of ceiling) | 17.5 % |
| **Ki/Kd only** (ATP-independent) | 386 | **0.462 ± 0.087** | 0.579 ± 0.035 | **0.48 ± 0.96×** (5 % of ceiling) | 9.6 % |

**Finding — the confound is real and material.** Sample size accounts for part of
the drop (0.798 → 0.682), but not the rest: at *matched* n the ATP-independent
subset still falls from **0.682 to 0.462**, roughly 2.5 pooled standard deviations.
Top-decile enrichment does not merely weaken, it disappears (3.71× → 0.48×, i.e.
below random), though that particular figure is too noisy to lean on — its
standard deviation exceeds its mean on a top decile of 39 molecules.

**What this does and does not mean.** It does **not** refute STEP 4: the deployed
number is correctly measured for what it measures. It does mean a meaningful part
of the headline ranking performance is carried by *how the data was measured*
rather than by selectivity biology alone, and the honest reading of "Spearman 0.80"
is now "0.80 on pooled assay types, substantially lower on the ATP-independent
subset."

**Competing explanations not yet excluded** (n = 386 is small):
- the Ki/Kd base rate is lower (9.6 % vs 17.5 %) and its gap distribution
  narrower, which compresses a rank correlation independently of assay physics;
- Ki/Kd-measured molecules may be a chemically distinct, older population rather
  than a random sample of the same chemistry.

### Audit 2 — does the gap predict chemistry that was not yet published?

Cut on each molecule's **first** publication year: train on what existed at the
cutoff, test on chemistry absent from the literature at that time. Strictly harder
than a scaffold split. One split per cutoff (no seeds), so individual rows carry
no error bar — read the trend, and compare the latest cutoff against the reference,
whose training set is closest in size.

| cutoff | train | test | Spearman | direct | enrichment | ceiling | % of max | base |
|-------:|------:|-----:|---------:|-------:|-----------:|--------:|---------:|-----:|
| 2015 | 865 | 2756 | 0.354 | 0.515 | 3.19× | 5.39× | 59 % | 18.5 % |
| 2016 | 1232 | 2389 | 0.606 | 0.546 | 3.90× | 5.01× | 78 % | 20.0 % |
| 2017 | 1358 | 2263 | 0.470 | 0.476 | 3.80× | 4.93× | 77 % | 20.3 % |
| 2018 | 1727 | 1894 | 0.510 | 0.651 | 3.08× | 4.65× | 66 % | 21.5 % |
| 2019 | 2096 | 1525 | 0.698 | 0.739 | 3.41× | 3.98× | 86 % | 25.1 % |
| **2020** | **2541** | **1080** | **0.715** | 0.741 | 2.64× | 2.94× | **90 %** | 34.0 % |
| *reference — scaffold split, 5 seeds* | *≈2896* | *724* | *0.798* | *0.816* | *4.48×* | *6.13×* | *73 %* | *16.4 %* |

Raw enrichment is **not** comparable across these rows: the top decile is capped at
`1/base_rate`, and the base rate climbs from 16 % to 34 % across the table. At a
34 % base rate no ranker can exceed 2.94×, so the 2020 row's 2.64× is 90 % of the
achievable maximum — the best row in the table, not the worst.

**Finding — the ranking claim survives prospective-style evaluation.** At the
closest-comparable training size, Spearman is **0.715 vs 0.798** and enrichment
reaches **90 % of its ceiling vs 73 %**. Predicting genuinely unpublished chemistry
costs roughly a tenth of the rank correlation, which is a smaller penalty than the
scaffold-split framing would lead one to expect.

**Caveat.** Spearman is non-monotonic across cutoffs (0.354 → 0.606 → 0.470 → 0.510
→ 0.698 → 0.715) and each row is a single split. The trend tracks training size;
the individual values are noisy and should not be quoted alone.

### What these two audits jointly imply

They are not in tension. The gap ranking **does** transfer forward in time, and a
meaningful share of what it has learned **is** tied to assay conditions — a model
can reliably reproduce an assay-correlated pattern that itself persists across
years. The follow-up that would separate them is a larger ATP-independent set
(ChEMBL alone will not supply it) or explicit ATP-concentration normalisation of
the IC50 records.

---

## STEP 10 — the binder gate (Tier 0.5) (2026-07-26)

**Why this was needed — a measured failure of the deployed screen.** The
per-isoform regressors are trained only on quantified JAK pchembl, so they never
see a non-binder. On an off-domain molecule a tree ensemble reverts to its training
mean (~6.3 pchembl), which is above the potency floor. Measured on the deployed
models:

| Probe | predicted JAK1 pchembl (before the gate) |
|-------|:----------------------------------------:|
| ethanol (`CCO`) | **6.12** (a 760 nM "inhibitor") |
| benzene | 5.77 |
| aspirin | 6.14 |

Of the 6 882 drug-like library molecules, **83.4 %** cleared the potency floor
(`pred JAK1 ≥ 6`), so Tier 1's most important gate removed almost nothing, and the
shortlist was topped by a neonicotinoid insecticide and 4-chlorothioanisole — the
selectivity gap between three shrunk-to-mean predictions is a model artefact.
Applicability domain flagged these *uncertain*, but only *after* the gap had ranked
them. DESIGN_DECISIONS §1 had reserved the fix ("a DUD-E-style decoy class … kept
in reserve"); this activates it.

**The gate.** A binary "is this a plausible JAK binder?" classifier
(`src/models/binder_gate.py`) that runs before Tier 1. Positives are JAK actives
(pchembl ≥ 6, any isoform); negatives are real drug-like molecules measured active
on ten *other* targets with no JAK record (`src/data/negatives.py`), **physchem-
matched** to the JAK actives' (MW, logP) distribution so the classifier cannot
separate the classes on molecular size or lipophilicity alone. ECFP4 +
HistGradientBoosting, scaffold-split over 5 seeds.

| Quantity | Value |
|----------|:-----:|
| positives / negatives | 15 446 / 12 144 |
| ROC-AUC (scaffold split, 5 seeds) | **0.998 ± 0.001** |
| Average precision | 0.999 ± 0.001 |
| operating point (Youden's J) | **0.544** |
| held-out actives kept | 98.2 % |
| held-out negatives rejected | 98.4 % |

**On the probe molecules** (threshold 0.544): ethanol **0.007**, benzene 0.008,
aspirin 0.001, caffeine 0.061, the neonicotinoid insecticide **0.001** — all gated
out; ruxolitinib **0.999** and tofacitinib **1.000** — pass. The gate rejects the
junk and keeps the real inhibitors.

**On the wide library.** Of 6 882 drug-like molecules, **23 pass the gate** (0.33 %),
and the resulting shortlist (20 molecules, **4 in-domain**, up from 3) is dominated
by purine / pyrrolopyrimidine / kinase-hinge chemotypes rather than pesticides and
thioanisoles. The potency floor is now downstream of a filter that actually
removes non-binders.

**Gate 10 passed:** the trivial-molecule failure is closed (ethanol/benzene/aspirin
gated out, known JAK inhibitors retained), and the wide screen ranks plausible
binders instead of shrunk-to-mean artefacts.

**Honest reading of the 0.998 AUC.** JAK inhibitors are a distinctive chemotype, so
JAK actives and other-target actives are nearly separable in ECFP4 — the gate
learns a coarse "JAK-like vs not" boundary, which is exactly what rejects ethanol
but is *not* subtle SAR. Two consequences follow. **(1)** The Youden's-J operating
point (0.544) is used deliberately rather than a high positive-recall cut: a
recall-set threshold sits at ~0.93 (known actives score ~1.0) and would admit only
molecules that already look like a *known* JAK chemotype — rejecting the novel
scaffolds a discovery screen exists to find (it collapses the library to 2). **(2)**
The gate partly overlaps the Tanimoto AD signal (both read fingerprint
resemblance to JAK actives); it is a stronger, learned version of "unlike the
training set", not a fully orthogonal second opinion. **(3)** The 23/6 882 pass rate
is low mainly because the wide library is the Tox21 collection — a tox-screening
set with little JAK-like chemistry. A JAK-oriented discovery library would pass
many more; replacing the library is the natural follow-up.

**Residual limit — presumed-negative label noise.** A promiscuous other-kinase
inhibitor that would in fact hit JAK but carries no JAK record is a mislabelled
negative. Known cross-actives are removed by the JAK exclusion, and the two kinase
targets (EGFR, CDK2) are a minority of the ten-target basket, but the noise is real
and bounds how sharp the boundary should be trusted to be.

**The regression numbers are untouched.** The gate filters what reaches Tier 1; it
does not change the regressors, the gap, the conformal intervals or AD. The one
shared-code change — a `try/except` in `_scaffold` for a negative molecule that
crashes `MurckoScaffoldSmiles` ("bad bond stereo") — was verified inert on the
existing data: **0** of the JAK1 / JAK2 / JAK3 / ESOL / Tox21 training molecules hit
the fallback, so every metric in STEP 3–8 stands as measured.

### Reproduce

```bash
python -m src.data.negatives          # build + cache the physchem-matched negatives
python -m src.models.binder_gate      # evaluate (5 seeds) + cache the deployed gate
cp data/jak/negatives.parquet assets/jak/
cp data/models/jak/binder_gate.pkl assets/models/jak/
python -m src.funnel                  # screen the library through Tier 0.5 + 1 + 2
```
