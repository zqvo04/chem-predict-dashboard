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

**Wide library** (`src/data/library.py`): 38 594 unique, canonical, **target-agnostic**
molecules from 20 diverse ChEMBL targets, 32 322 clearing Tier 0 (superseded the
original 7 823-molecule Tox21 collection — see STEP 13).
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

> **Re-run 2026-07-26** after the binder gate (STEP 10), the `model_id` fix
> (STEP 12) and the library replacement (STEP 13). Each reshaped the shortlist, so
> the selected parents changed; the numbers below are the current run and supersede
> the original ones.

| set | n | median gap | max gap | % ≥10× selective | % in-domain |
|-----|--:|:----------:|:-------:|:----------------:|:-----------:|
| before (selected) | 3 | +1.62 | +1.68 | 100% | 100% |
| after (generated) | 90 | +1.45 | +2.38 | 93% | 100% |

The honest before/after story: generation reached a **higher max gap (+2.38 vs
+1.68)** and produced an in-domain analogue that **improves on its parent** —
`Cc1cc(Nc2nc(N)c(Cl)c(Nc3cn(C)nc3S(=O)(=O)C(C)C)n2)c(OC2CC2)cc1C1COC1`, gap **+2.38**
(parent +1.68). Both sets are now **100% in-domain** and the selected parents are
already ≥10x selective, because the library is real kinase-adjacent chemistry rather
than a toxicology panel: the gate admits molecules close to the training
distribution, and decorating those scaffolds stays in-domain too.
`figures/loop_before_after.png` shows the shift; the re-scored analogues are written
back as `examples/loop_case_A_rescore.json` (`stage: A_rescore`), so they re-enter
Stage B — the funnel is a cycle.

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

---

## STEP 11 — memory: the deployed funnel fit in 512 MB (2026-07-26)

**The failure.** The deployed app raised *"Ran out of memory (used over 512MB)"* on
the funnel mode. Profiled stage by stage on the real screen:

| Stage | RSS |
|-------|----:|
| imports (numpy + pandas + rdkit + sklearn) | 222 MB |
| library **data** (6 882 rows) | 0.6 MB |
| `morgan_matrix` over the library, **float64** | **+113 MB** |

The data was never the problem — it is 0.6 MB. The cost was holding every
fingerprint at once **as float64**: 2048 single bits stored in 8 bytes each. The
screen paid it three times over (binder gate, Tier 1, and the percentile
distribution), on top of a 222 MB import floor.

**The fix.** Two changes, neither of which touches any model:

1. **`uint8` fingerprints** (`src/models/features.py`). The same matrix is 113 MB as
   float64, 14 MB as uint8.
2. **Batched featurise-and-score** (`iter_morgan_batches`). One slice is built, all
   three isoform models score it, and it is released before the next.

| | before | after |
|---|---:|---:|
| library fingerprint matrix, resident | 113 MB | 13 MB (per batch) |
| **peak RSS, full `screen_library()`** | — | **327 MB** |

**Verified numerically inert.** The dtype change cannot move a result: training on
uint8 vs float64 gives bit-identical predictions (max abs diff **0.0**) and an
identical model **pickle hash**, so `model_id` is stable and contracts still match.
The bundled models predict identically on either dtype. `iter_morgan_batches` is
pinned by a test asserting its concatenation equals `morgan_matrix` exactly.

**Gate 11 passed:** peak RSS 327 MB for the full wide screen, all 119 tests green,
shortlist unchanged (20 molecules, 4 in-domain, best gap +0.782).

**What was *not* done, and why.** Precomputing the library fingerprints into a
committed `packbits` asset (1.76 MB) was planned and then dropped on measurement:
featurising the whole library takes **0.9 s** and loading all three AD references
**0.3 s**, so the asset would buy about a second while adding a staleness-invalidation
burden. It becomes worthwhile only when the library reaches ~10⁵–10⁶ molecules,
where that 0.9 s scales to minutes.

### Known consequence — the gap percentile reference is now small

STEP 10's binder gate filters the percentile reference population to the molecules
the funnel actually ranks. On the current library that is **23 molecules**, i.e.
4.3 percentile points per molecule. The single-molecule view leads with this number,
so it now reports the reference size alongside it and says plainly when the set is
small. The underlying cause is the **Tox21 screening library**, which contains
almost no JAK-like chemistry; replacing it is the fix, not loosening the gate.

---

## STEP 12 — `model_id` was not a stable identity (2026-07-26)

**The failure.** The Colab deep dive refused to run:

```
ValueError: Model mismatch: contract pinned {'JAK1': 'CHEMBL2835@b1bfc32323', ...},
current models {'JAK1': 'CHEMBL2835@fd9840028c', ...}
```

Only **JAK1** differed; JAK2 and JAK3 matched exactly. A code or dtype change would
have moved all three, so this was not model drift.

**The cause.** `model_id` hashed `pickle.dumps(model)`, which *looks*
content-addressed but is **not idempotent**. Measured on the bundled JAK1 model:

```
dump -> load -> dump gives identical bytes: False
  hash after 0 round-trips: fd9840028c     <- Colab, loading straight from disk
  hash after 1 round-trip : b1bfc32323     <- the app, via a cache that re-serialises
```

Both digests are the *same model*. `assets/models/jak/JAK1_reg.pkl` is byte-identical
in every commit that has ever contained it (blob `b800cb7c…`) — nothing about the
model changed. The guard was rejecting the deep dive over **serialisation noise**.
The committed example contracts carried the stale digest too, so the repo's own
worked example could not be re-scored.

The old test only asserted `model_id(m) == model_id(m)` on one object in one
process, which never exercises a round-trip — which is why this passed CI while the
real workflow failed.

**The fix.** Model identity is now **behavioural**: the digest is over the model's
predictions on a fixed, committed 16-molecule probe set (`PROBE_SMILES`), formatted
at 6 decimal places so last-bit float noise cannot move it either. Two models that
predict identically on the probe are interchangeable for re-scoring, which is
exactly what `assert_models_match` needs to guarantee — and predictions are stable
across round-trips, sklearn versions and platforms.

| | pickle-hash (old) | prediction-hash (new) |
|---|:---:|:---:|
| stable across 0/1/2 pickle round-trips | ❌ | ✅ |
| distinguishes genuinely different fits | ✅ | ✅ |

**Verified end-to-end**, simulating the exact Colab failure — a contract whose ids
come from cache-round-tripped models, validated against models loaded fresh from
disk as a fresh clone does:

```
app-side ids == colab-side ids: True
assert_models_match(committed contract, fresh clone): PASS
```

**Gate 12 passed:** `python scripts/run_loop.py` runs B→SELECT→A→re-score
end-to-end, the regenerated example contracts validate from a fresh clone, and
three tests pin the property — idempotence, round-trip stability, and that a
genuinely different fit still produces a different id (so stability did not come
from the id going blind).

**Cost of the fix.** Every model id changes, so contracts exported before this
commit will not validate. `examples/*.json` are regenerated; the STEP 8 loop table
above is re-run and superseded. `PROBE_SMILES` must never be edited for the same
reason — it would silently invalidate every contract ever exported.

---

## STEP 13 — the wide library was the wrong haystack (2026-07-26)

**Two defects, one cause.** The wide library was the **Tox21 collection**, chosen in
STEP 7 because it was diverse, drug-like and already wired for download. Both of its
problems come from that convenience:

1. **It is a toxicology panel, not a discovery library.** Tox21 is environmental
   chemicals, pesticides and industrial reagents. Once the binder gate (STEP 10)
   could tell a plausible JAK binder from junk, only **23 of 6 882** drug-like
   molecules survived — 0.33 %. The shortlist and the gap percentile (4.3 points per
   molecule) were both near-meaningless, and the funnel's top-ranked molecules had
   been a neonicotinoid insecticide and 4-chlorothioanisole.
2. **It was the toxicity model's own training set.** `library.py` downloaded
   `tox21.csv.gz` and `property_models.py` trained the Tox21 classifier on the same
   file: **7 613 / 7 613 = 100 % overlap**. Every MPO `tox_prob` on the shortlist was
   a recalled training label, not a prediction — including the "MPO 0.01, Tox21 alert
   ≈ 0.9" figures the README had cited as evidence the MPO axis worked.

**The replacement.** Bioactive drug-like molecules from **20 diverse ChEMBL targets**
(ten kinases, ten non-kinases: proteases, GPCRs, nuclear receptors, epigenetic and
metabolic enzymes), pulled with the same client the rest of the repo uses. The panel
is chosen to be **disjoint from everything the binder gate was trained on** — no JAK
isoform (its positives), none of `negatives.NEGATIVE_TARGETS` (its negatives) — and
molecules appearing in either training class are dropped by SMILES as well, so the
gate's verdict on a library molecule is a genuine prediction rather than recall.
Half the panel is kinases so a JAK screen has plausible chemistry to enrich from
without the library becoming a kinase-inhibitor set.

| | Tox21 (before) | ChEMBL panel (after) |
|---|---:|---:|
| library molecules | 7 613 | **38 594** |
| clearing Tier 0 | 6 882 | **32 322** |
| clearing the binder gate | 23 (0.33 %) | **6 120 (18.9 %)** |
| shortlist | 20 | **60** |
| **in-domain shortlist** | 4 | **33** |
| gap-percentile reference | 23 | **6 120** |
| overlap with the tox model's training set | **100 %** | **0.50 %** |
| peak RSS, full screen | 327 MB | 341 MB |
| wall clock, full screen | — | 12 s |

The MPO leak is closed (0.50 % residual, 193 molecules active on both a panel target
and a Tox21 assay — real chemistry, not a wiring mistake). The shortlist is now
aminopyrimidines, ureas, benzisoxazoles and pyrazolopyrimidines — kinase chemotypes —
rather than pesticides.

**The 4.7x larger library costs 14 MB of peak memory**, because STEP 11 made the
screen stream its fingerprints; before that change this library would not have fit.

**Gate 13 passed:** the funnel screens a real discovery library, the gate's verdict on
it is a prediction rather than recall, and the shortlist carries 33 in-domain
candidates instead of 4. All 128 tests green.

**What this does not fix.** The gap interval still crosses zero for 100 % of the
shortlist — that is the conformal-calibration problem (intervals are calibrated on
ChEMBL JAK molecules and propagated by summing two half-widths), untouched here and
still open.

### Reproduce

```bash
python -m src.data.library      # rebuild from the 20-target panel (needs network)
cp data/library/library.parquet assets/library/
python -m src.funnel            # screen it
python scripts/run_loop.py      # regenerate the worked case
```

---

## STEP 14 — the gap interval was calibrated on the wrong thing (2026-07-27)

**The defect.** The funnel's selectivity gap `S` came with a 90 % interval whose
half-width was the **sum of the two contributing isoform half-widths**
(`conformal.gap_interval`). Summing is the correct worst case only if the two
isoform errors are independent and adverse. Measured on the cross-measured set they
are neither: the JAK1 and JAK2 residuals correlate at **+0.65**, so they largely
cancel in a difference and the sum over-pays for an error that does not occur.

The cost is not academic. At a 90 % nominal level the summed interval delivered
**99.7 % empirical coverage** at a mean width of **4.86** pchembl units, and on the
deployed shortlist it **crossed zero for 60 of 60 molecules** — meaning the funnel
could not call a single molecule selective, which is the one thing it exists to do.

**Three arms, because the obvious fix is a trap.** The fix is to calibrate on the
gap residual directly, over the cross-measured set (the only place a *measured* gap
exists). But a single directly-calibrated width passes the headline check while
failing the molecules the screen is actually made of, so it is measured as its own
arm rather than skipped:

| arm | what it is | marginal coverage | mean width | crosses zero |
|---|---|---:|---:|---:|
| **summed** (shipped before) | q(target) + max q(off) | 0.997 ± 0.003 | 4.86 | 99.6 % |
| **flat** (the trap) | split-conformal on the gap residual, one width for all | 0.897 ± 0.015 | 1.97 | 74.6 % |
| **scaled** (shipped now) | that width × a difficulty curve σ(nn-similarity) | 0.896 ± 0.020 | 2.17 | 69.9 % |

Marginal coverage alone would call the flat arm finished — 0.897 against a 0.90
nominal is as close as this gets. Split by how far a molecule sits from training, it
is not:

| nn-similarity to training | summed | flat | scaled |
|---|---:|---:|---:|
| **[0.00, 0.35)** | 0.921 | **0.460** | **0.921** |
| **[0.35, 0.45)** | 0.976 | 0.546 | 1.000 |
| [0.45, 0.60) | 0.998 | 0.837 | 0.938 |
| [0.60, 1.00] | 0.998 | 0.921 | 0.889 |

**A constant width is mis-sized per molecule, and it is mis-sized in the direction
that matters.** The flat arm's advertised "90 % CI" is a 46 % CI on the
lowest-similarity band — and a wide, target-agnostic screen operates almost entirely
in that band (**88.5 %** of gate survivors and **52 of 60** shortlisted molecules sit
in the bottom two rows). Scaling the width by a fitted difficulty curve lifts the
worst bucket from **0.460 to 0.889** while keeping marginal coverage on nominal.

> **Correcting the record.** The commit that introduced this change
> (`5c37da0`) presented the `0.460` bucket table as a property of the **summed**
> interval. It is not: the summed interval over-covers *everywhere* (worst bucket
> 0.921) — its failure is being uniformly too wide, not too narrow on hard
> molecules. `0.460` belongs to the **flat** arm. The three-arm comparison above is
> the measurement that separates them, and this table supersedes the one in that
> commit message.

**What shipped.** `calibrate_gap` splits the cross-measured set scaffold-wise, and
splits the calibration half again — one part fits the difficulty curve
σ(nn-similarity), the other supplies the conformal quantile — so the curve is never
fitted and scored on the same molecules. The half-width for a molecule is
`q · σ(nn_sim)` with **q = 3.003** and σ running **0.15 → 0.467**. σ is stored as 21
interpolation knots in `assets/jak/conformal_quantiles.json` rather than a pickled
model: it is a 1-D curve, and knots are inspectable, tiny and version-proof.
`nn_sim` is the **minimum across the contributing isoforms** — the gap is only as
trustworthy as the most-extrapolating model behind it, the same worst-case rule the
AD verdict already uses.

**Effect on the deployed screen** (60-molecule shortlist, 33 in-domain — unchanged):

| | summed (before) | scaled (after) |
|---|---:|---:|
| gap interval width | flat **4.53** | **1.78 – 2.80** (adaptive) |
| shortlist crossing zero | **60 / 60 (100 %)** | **32 / 60 (53 %)** |
| molecules with a directional 90 % claim | **0** | **28** |

In the single-molecule view the width now tracks how much the model actually knows
about the molecule: tofacitinib (nn 1.00) gets **1.23**, ruxolitinib (0.66) **1.78**,
ethanol (0.11) **2.80**.

**Gate 14 passed:** the gap interval is calibrated against the measured gap rather
than assembled from two unrelated ones, marginal coverage sits on nominal
(0.896 vs 0.90), the worst-similarity bucket holds at 0.889 instead of 0.460, and
the funnel can state a directional selectivity result for 28 molecules where before
it could state none. Per-isoform coverage is untouched — JAK1 **0.904 ± 0.010**,
JAK2 **0.905 ± 0.007**, JAK3 **0.888 ± 0.023**, all inside the 88–92 % gate.

**Honest limits.** The two low-similarity buckets clear the 15-molecule reporting
floor in only **2 of 5 seeds**, so those two rows rest on fewer samples than the two
above them. `gap_interval()` is kept in the codebase but marked superseded, because
it documents what the funnel used to do. And the calibration inherits STEP 4's
caveat unchanged: it is fitted on pooled assay types, so it carries the same
ATP-competition confound the headline Spearman does.

### Reproduce

```bash
python -m src.conformal          # per-isoform coverage + the 3-arm gap table
```

---

## STEP 15 — the funnel was a case study, not a workflow (2026-07-27)

**The defect is structural, not numerical.** Every number STEP 2–14 reports is
correct and every one of them is about JAK, because the target was not a parameter.
`selectivity.TARGET`/`OFFS` were module constants, the datasets lived at
`assets/jak/`, the models at `assets/models/jak/`, and the AD references were keyed
on bare isoform names. Nothing could be pointed at other chemistry, the three modes
shared no state, and — the reason this blocks everything downstream — there was no
record of what had been screened, when, through which models. Active learning is by
definition a loop where round *N* changes round *N+1*; that cannot exist over pure
function calls.

**The unit of generalisation is a panel, not a target.** `S = pred(target) −
max_off pred(off)` is undefined without off-targets, so `PanelSpec`
(`src/panels.py`) carries a target, its off-targets, their ChEMBL ids and the asset
namespace derived from its name. A single-target screen is Mode 1's job and the
spec now refuses to be constructed without off-targets rather than silently
returning a meaningless zero.

### The JAK screen did not move

The constraint on this refactor was that the validated panel keep producing
*exactly* what it produced before. Model ids are behavioural digests (STEP 12), so
they are the sharpest available check — if any JAK model had shifted, every contract
ever exported would stop validating:

| check | before | after |
|---|---|---|
| JAK model ids | `61edd879da` / `5649ee4718` / `0772985d8c` | **identical** |
| shortlist | 60 (33 in-domain) | **60 (33 in-domain)** |
| per-isoform coverage | 0.904 / 0.905 / 0.888 | **unchanged** |

Committed assets moved but did not change: `assets/conformal_quantiles.json` →
`assets/jak/`, `assets/library/gap_distribution.npz` → `assets/jak/`,
`assets/ad_reference/JAK*.npz` → `assets/ad_reference/jak/`. Byte-identical, keys
included — the gap-distribution provenance guard still matches, so nothing rebuilt.
A test pins the model ids against the values in `examples/*.json`.

### A second panel, built end to end through the same code

Nominal generalisation is easy to claim and cheap to fake, so a second panel was
actually built: **PI3K δ vs α/β/γ** — a real selectivity problem (idelalisib is the
δ-selective drug). It was chosen because it is **disjoint from the binder gate's
negative basket and from the wide library's 20 targets**, so it inherits the STEP 10
and STEP 13 leakage discipline instead of renegotiating it. That is checked by
`disjointness_report`, not assumed.

| | JAK (validated) | PI3K (bootstrap) |
|---|---|---|
| members | JAK1 / JAK2 / JAK3 | PIK3CD / PIK3CA / PIK3CB / PIK3CG |
| molecules per member | 6.3k – 11k | 2 777 – 7 723 |
| **cross-measured (all members)** | **3 624** | **1 500** |
| regressor R² (5-seed scaffold) | 0.71 – 0.77 | **0.582 – 0.710** |
| regressor Spearman | 0.82 – 0.88 | **0.736 – 0.834** |
| binder gate ROC-AUC | 0.998 | **0.999 ± 0.001** |
| gate threshold (Youden's J) | 0.544 | **0.481** (keeps 98.7 % of actives, rejects 99.0 % of negatives) |
| gate conflicts with library/negatives | none | **none** |

No code path is special-cased for either panel. PI3K's regressors are weaker than
JAK's, which is the expected consequence of less data on PIK3CB/PIK3CG rather than a
bug — and it is exactly why it is not labelled validated.

**Cost:** 230 s to fetch the four datasets, 469 s to train the regressors, 565 s to
build the negatives and the gate. Its datasets are bundled (492 kB) so the campaign
card is honest offline; its **models are deliberately not bundled** (5.8 MB), since
shipping pre-trained artifacts for an unvalidated panel would put it on the same
footing as the validated one.

### Validation tiers — the guard against re-opening v1's hole

v1 was replaced because it ranked molecules by a composite score validated against
nothing. Letting anyone create a campaign re-opens precisely that hole: a bootstrap
campaign renders the same tables as the JAK one. So every campaign carries a tier,
**derived from what has been measured rather than asserted**:

| tier | when | what it permits |
|---|---|---|
| `validated` | the panel's gates are re-run and written up here | the full claim |
| `bootstrap` | models + AD + conformal built, gates not re-run | ranking shown as a hypothesis, badged |
| `insufficient_data` | < 200 cross-measured molecules | **no selectivity claim at all** |

The 200 floor is arithmetic, not taste: `calibrate_gap` puts roughly 10 % of the
cross-measured set into the conformal quantile, and the finite-sample level
`ceil((n+1)(1−α))/n` only drops below 1.0 — becoming a real quantile rather than the
largest residual seen — at n ≥ 19. A test pins that **data volume alone never
promotes a panel**: PI3K with a hypothetical 5 000 cross-measured molecules is still
`bootstrap`.

### Registry

`data/registry/<campaign>/` holds the campaign definition, an append-only
`rounds.jsonl`, and one parquet per round's scored table. Append-only JSONL rather
than a database because the operations needed are "add a round" and "read them in
order": no dependency, greppable, and a torn final line costs one round instead of
the file (pinned by a test). The loop contract gains `campaign_id` at **schema 1.1**
— a minor bump, since the validator keys compatibility on the major version, so
contracts exported at 1.0 still validate.

**Gate 15 passed:** the JAK panel is bit-identical, a second panel builds end to end
through the same code with no special-casing and a clean leakage report, tiers
prevent an unvalidated campaign from presenting as a validated one, and rounds
persist across processes. **151 tests green** (132 before).

**What this does not do.** PI3K is *not* validated — its gap-vs-measured-gap Spearman,
conformal coverage and AD separation have not been measured, so its ranking is a
hypothesis. The registry records rounds but nothing yet *reads* round N to choose
round N+1; that is the acquisition function, and it is Phase 3. The v1 pipeline
(`src/pipeline.py`) is unchanged and still runs from its CLI — it is simply no
longer wired into the dashboard.

### Reproduce

```bash
python -m src.panels                       # registered panels + leakage report
python -m src.data.panel_data pi3k         # build the second panel (needs network)
python -m src.models.isoform_regressor pi3k
python -m src.data.negatives pi3k && python -m src.models.binder_gate pi3k
```

---

## STEP 16 — the repo stored answers and threw away the evidence (2026-07-28)

**The defect.** `panel_data._collapse` reduced every molecule to a median pchembl at
ingest, and the raw records survived only in `data/cache/activities_<sha1>.parquet` —
keyed by a hash of the request params, so "every activity for CHEMBL203" was a
question the cache could not answer. Two measured consequences:

* The **ATP-competition confound** (Spearman 0.682 → 0.462 on the Ki/Kd subset) could
  only be *bounded*, never fixed: `standard_type` was the finest instrument available
  because `assay_chembl_id` was never requested.
* A campaign was **not reproducible**. It pinned `code_version` and `model_ids` but
  not its data, and the ChEMBL release appeared nowhere in the repo — the same class
  of defect as STEP 12's `model_id`, unfixed one layer down.

**What the store holds now** (`src/data/db.py`, DuckDB, gitignored, rebuildable):

| | rows |
|---|---:|
| activities (one per measurement, never collapsed) | **103 420** |
| molecules (keyed by standardised-parent InChIKey) | 75 870 |
| distinct assays | **5 207** |
| library members | 38 592 |
| **right-censored records** (`>`, no pchembl) | **29 321 (28.4 %)** |
| sources | `chembl_37` |

Per JAK isoform, the numbers the collapsed datasets hid:

| | JAK1 | JAK2 | JAK3 |
|---|---:|---:|---:|
| measurements | 19 949 | 24 749 | 14 927 |
| **distinct assays** | **711** | **1 049** | **777** |
| censored | 4 737 | 5 511 | 3 881 |
| molecules in the committed dataset | 10 468 | 12 680 | 7 457 |

JAK1's 10 468-row training set is the collapse of 19 949 measurements taken across
**711 different assays**. None of that structure was visible before.

### The migration diff is far smaller than the design feared

DATABASE_DESIGN §6 predicted the store would not reproduce the committed parquet, and
that a careless swap would move every training set. Measured
(`python scripts/db_equivalence.py`):

| panel | molecule identity | pchembl changed | un-standardised in committed | max Δ |
|---|---|---:|---:|---:|
| **JAK** | **100 % shared** (0 only-committed, 0 only-store) | **12 / 30 586** | 74 rows (0.15–0.29 %) | 0.845 |
| **PI3K** | 100 % shared | 0 | 0 | 0 |

So the honest cost of migrating JAK is **5 molecules merging under standardisation
and 3–6 pchembl values shifting per isoform**, not the wholesale change §6 braced for.
PI3K matching exactly is the measurement validating itself: those datasets were built
this session, from ChEMBL_37, with the current standardisation, so zero drift is the
right answer and the tool reports it.

**It is still not zero, and non-zero is the whole point.** Any difference changes the
training set, therefore every `model_id`, therefore every exported contract. So
`panel_data` still reads the committed parquet; the retrain remains its own announced,
contract-invalidating step. The committed assets' own release stays **unrecorded** —
`Campaign.data_version` reports exactly that rather than inventing a default.

### What the store found in its first hour: STEP 15's leakage check was too weak

`disjointness_report` compares **target identifiers**. That is necessary and not
sufficient: a library drawn from other targets can still contain molecules separately
assayed on this panel, and those are molecules the gate was trained on. Answering it
needs a molecule-level join across every target at once — which is precisely what did
not exist before.

| panel | library molecules that are its own gate positives |
|---|---:|
| JAK | **8 / 38 592 (0.02 %)** |
| **PI3K** | **134 / 38 592 (0.35 %)** |

Two distinct findings:

1. **STEP 13's exclusion leaked slightly.** It dropped known JAK actives from the
   library **by SMILES**; matching on InChIKey finds 8 that a string comparison
   missed. Small, and it confirms the exclusion was substantially right.
2. **PI3K was never excluded at all.** The panel was added in STEP 15, long after the
   library was built, and the target-level check passed because no PI3K subunit is
   among the library's 20 targets. 134 library molecules are nonetheless PI3K actives,
   so the PI3K gate's verdict on those rows is **recall, not prediction** — the exact
   failure STEP 10 and STEP 13 were fixed for, recurring in a panel added later and
   invisible until now.

The app previously asserted that target-level disjointness made "the gate's verdict a
prediction, not recall." That did not follow, and the claim is corrected: the campaign
card now reports the molecule-level count when the store is present, and says
**"unchecked on this machine"** when it is not — never "disjoint".

**Gate 16 passed:** measurements are stored rather than discarded, every row carries
its ChEMBL release, re-ingesting a target inserts nothing (idempotent on ChEMBL's own
`activity_id`), the assay-stratified query STEP 4's audit needed is a `WHERE` clause,
and the validated JAK screen is untouched — `model_id`s unchanged, 166 tests green.

**Honest limits.** Only the 7 panel targets and the library are ingested; the 20
library targets and 10 gate-negative targets are not, so the multi-task matrix
(DATABASE_DESIGN D4) is not yet buildable. The 134-molecule PI3K leak is **measured
but not fixed** — fixing it means rebuilding the library with a molecule-level
exclusion and retraining the PI3K gate, which is a separate change. And the store
inherits every limitation of ChEMBL curation; storing more evidence does not make the
evidence better.

### Reproduce

```bash
pip install -r requirements-dev.txt
python -m src.data.ingest jak pi3k --library   # needs network, ~25 min
python -m src.data.db                          # row counts
python scripts/db_equivalence.py jak           # the migration diff
```
