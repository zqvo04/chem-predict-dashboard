# Design decisions — JAK-selectivity screening funnel

Why the funnel is built the way it is. This document is the rationale companion to
the [Roadmap](README.md#roadmap--jak-selectivity-screening-funnel). It records
decisions and the reasoning (including rejected alternatives), so future changes
argue with the reasoning rather than silently reverse it.

**Status:** planning artifact. No results are recorded here yet; measured numbers
live in `VALIDATION.md` once produced. Where a threshold is stated, it is a
*gate*, not a claim.

---

## 0. What problem the funnel actually solves

The v1 pipeline is a sound engineering skeleton with a hollow scientific value
proposition. Four gaps make its output untrustworthy *as discovery*:

1. **Fake novelty.** PubChem 2D-similarity expansion returns near-analogues of
   known actives, inside the training scaffold neighborhood. The headline
   scaffold-split R² (generalization to new chemotypes) does not describe these
   near-duplicates. The "discovery" output is interpolation dressed as prediction.
2. **No trust signal.** Predictions carry no uncertainty and no domain-of-validity;
   an in-domain estimate and a wild extrapolation are visually identical.
3. **Unvalidated ranking.** The composite score's weights are validated against no
   endpoint. There is no evidence the ranking enriches for anything.
4. **Censored, biased label.** Regression trains only on *quantified* pchembl, so
   the model never sees a true inactive and cannot recognize a non-binder — fatal
   for a tool whose job is to reject bad molecules.

The engineering (featurizer, scaffold split, cache, Trainer, deployment) and the
project's unusual honesty are worth keeping. The **value proposition** is what
gets rebuilt. The funnel is chosen because it *forces* the fixes: selectivity is a
genuine unsolved problem (not table-stakes potency), its predictions are checkable
against measured selectivity, and they are useless without uncertainty + AD.

---

## 1. Task: binary classification, not potency regression

**Decision.** Each isoform model is a binary classifier — active (pchembl ≥ 6) vs
inactive (pchembl ≤ 5) — not a pchembl regressor.

**Why.** Regression on quantified pchembl is trained only on molecules that *have*
a measured potency, which skews toward actives and, critically, **omits true
inactives**: non-binders are recorded as right-censored `>` values or not at all,
and the regressor drops them. The model therefore learns "potency among actives"
and will assign a plausible-looking pchembl to a non-binder. For a screening tool
whose core job is to *reject* bad molecules, this is a fundamental defect (gap #4).

Classification repairs it: right-censored `>` values and low-pchembl records
become a legitimate **inactive** class, so the model finally learns *active vs
inactive*. It also makes the output a **checkable** claim (did this molecule hit
this isoform?) rather than an unvalidatable continuous score.

**Label thresholds (default, tunable).** Active = pchembl ≥ 6 (~1 µM), inactive =
pchembl ≤ 5, and the **5–6 gray zone is dropped from training labels** to reduce
boundary noise (a 5.9 vs 6.1 split is measurement jitter, not biology). Both the
gray-zone-excluded and gray-zone-included results will be reported so the choice is
auditable.

**Cost.** We give up a graded potency ranking — but that ranking was never
validated in v1 anyway, and the decision that matters downstream (selective or
not) is categorical.

**Rejected alternatives.**
- *Keep regression, bound damage with AD + conformal.* Honest but never learns to
  recognize inactives — leaves gap #4 open.
- *Add DUD-E-style decoys.* Most principled true-negative source, but adds a data
  and validation burden disproportionate to a zero-cost build. Kept in reserve if
  the ChEMBL inactive class proves too small or biased.

---

## 2. Selectivity as a probability

**Decision.**
```
P(selective for target) = P(active | target) · Π_off P(inactive | off-isoform)
```
computed from the per-isoform classifier probabilities.

**Why.** It is a single, interpretable quantity built only from calibrated
per-isoform probabilities, and — unlike the v1 composite — it is **validatable**:
on molecules measured across isoforms we can compare predicted `P(selective)` to
the *measured* selectivity label (gap #3). The product form encodes the clinical
reality that selectivity requires being active at the target **and** clean at
**every** off-target — one bad off-target kills selectivity, matching the
worst-case `max` intuition from the earlier regression framing.

**Potency floor is intrinsic.** Because the target term is `P(active | target)`, a
molecule inactive everywhere gets a low `P(selective)` automatically — no separate
floor needed (this was an explicit patch in the regression framing; classification
absorbs it).

---

## 3. Metrics: PR-AUC + calibration, not R²

**Decision.** Evaluate classifiers with **PR-AUC** and **calibration** (Brier
score + reliability curve), reported as mean ± std over **≥5 scaffold-split
seeds**. Selectivity is validated with **PR-AUC / enrichment** of predicted
`P(selective)` against measured selectivity.

**Why.** Actives (and especially selective molecules) are the minority class, so
ROC-AUC is optimistic and PR-AUC is the honest summary. The whole funnel rests on
`P(selective)` being a *real* probability that we multiply together and threshold —
so **calibration is not optional**; a reliability curve is the proof. If
calibration is poor, apply Platt/isotonic scaling on a held-out fold and re-check.
R²/RMSE are dropped because they were never a validatable claim for this use case.

**Seeds.** The existing `scaffold_split` is deterministic and takes no seed; a
`seed` argument (shuffling within the greedy scaffold fill) is the one surgical
change needed to report mean ± std instead of a single fragile draw.

---

## 4. Uncertainty: conformal classification

**Decision.** Split/inductive **conformal classification** (Mondrian per class, or
APS) producing **prediction sets** with a coverage guarantee at **90 %** nominal
(default), coverage empirically verified on the scaffold-split test set.

**Why.** Conformal gives a distribution-free, finite-sample coverage guarantee
under exchangeability — the honest way to say "how sure." For a classifier the
natural object is a **prediction set** (e.g. `{active}`, `{inactive}`, or the
ambiguous `{active, inactive}`), which is cleaner and more truthful than forcing a
regression-style interval onto a probability. **Gate:** empirical coverage in
**88–92 %** at 90 % nominal, or the step is not done.

**Assumption stated loudly.** Conformal coverage holds under exchangeability
between calibration and test data. A scaffold split deliberately breaks that
(test scaffolds are novel), so coverage on genuinely new chemotypes is expected to
be the *stress case* — which is exactly why AD (below) runs alongside it.

---

## 5. Applicability domain: two definitions, propagated

**Decision.** Flag every prediction with **≥2** AD definitions —
**Tanimoto distance to nearest training molecule** and **descriptor-space
leverage** — and propagate to selectivity: a `P(selective)` call is **uncertain**
if *any* contributing isoform model is out-of-domain.

**Why.** One AD metric is gameable; two orthogonal views (fingerprint-space
neighborhood vs descriptor-space extrapolation) are harder to fool. Propagation is
worst-case because `P(selective)` is a product across isoforms and is only as
trustworthy as its shakiest factor. **Gate — the money plot:** out-of-domain
error must be *systematically higher* than in-domain error with a clear margin; if
AD does not separate error, it is decorative and the step fails.

---

## 6. Generation (stage A): hypothesis-only

**Decision.** Keep conditional generation in the offline Colab notebook, but every
generated molecule is labelled an **in-silico hypothesis requiring wet-lab
validation** and filtered through the same AD; nothing is presented as a hit.

**Why.** De-novo generation of *validated* selective molecules on a zero-cost /
Colab budget is not achievable, and pretending otherwise would recreate gap #1 in
fancier form. The defensible deliverable is a **distribution shift**: after
generation, does the population's `P(selective)` move up while staying in-domain?
That is an honest, checkable claim; "we found a drug" is not.

---

## 7. Loop data contract: versioned JSON, pinned models

**Decision.** A single versioned **JSON** object flows B → SELECT → A → re-score,
pinning `model_ids`, `conformal_alpha`, and `code_version`. Stage A asserts it is
scoring with the identical models before re-scoring. (Schema in the roadmap.)

**Why.** JSON is human-diffable, git-friendly, and loads in Colab without pyarrow.
Pinning the exact models is what makes "re-score through the *same* models" a
verifiable fact rather than a hope — stage A cannot silently diverge from stage B.
The before/after report is then a diff of two objects of identical shape.

---

## 8. Gate 0 and the pairwise fallback

**Decision.** Before building anything selectivity-specific, pull JAK1/2/3 and
measure the **3-way cross-measured count** (molecules measured on all three).
Below a threshold, the flagship narrows to **pairwise selectivity** (target over
the single off-isoform with the most co-measured molecules, e.g. JAK1 over JAK2).

**Why.** `P(selective)` can be *computed* from predictions for any molecule, but it
can only be *validated* on molecules with measured labels across the relevant
isoforms. If that co-measured set is too thin, the hero validation (§3) is not
credible, and no amount of modeling fixes it. Making this a data-driven go/no-go
gate — decided *after* seeing real counts, not before — prevents building an
elaborate story on absent ground truth.

---

## 9. CPU-only / zero-cost split

**Decision.** Stage B (screening: classifiers, conformal, AD, dashboard) runs on
CPU within the free-host budget (~1 vCPU / ~1 GB). GPU is confined to the offline
Colab notebook (stage A generation) and never touches the deployed app.

**Why.** The constraint is the project's premise. Classification, conformal, and AD
are all cheap CPU operations on ECFP4 features; only generation benefits from a
GPU, and it is inherently offline (a human selects a case first). Keeping the
shared scoring `src` modules CPU-only is also what lets the *same* code run in both
the app and the notebook.
