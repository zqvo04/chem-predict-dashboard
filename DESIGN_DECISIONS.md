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

## 2. Selectivity as a probability — hybrid estimator

**Decision.** Two estimators of selectivity, used at different funnel tiers:

- **Product form (wide, Tier 1)** — from the per-isoform classifiers:
  ```
  P(selective | target) = P(active | target) · Π_off (1 − P(active | off))
  ```
- **Direct classifier (narrow, Tier 2)** — a single classifier trained on the
  cross-measured set with a "selective vs not" label.

**Why a hybrid, not one or the other.** The two trade off along the exact axis the
funnel cares about:

- The **product form uses all per-isoform data** (every molecule measured on *any*
  isoform trains its factor) and can score *any* molecule, so it is the only
  estimator that works on the unlabelled wide library. Its weakness: it multiplies
  three probabilities, so **miscalibration compounds** — a small per-isoform bias
  becomes a larger error in the product.
- The **direct classifier avoids compounding** (it learns the selective/not
  boundary in one shot) but can only train on the **cross-measured** set, which is
  thin (Gate 0 risk). Using it to screen the whole library would extrapolate far
  past its training support.

So: product form does the cheap wide screen (Tier 1); the direct classifier
**re-ranks the survivors** and serves as an **independent validator** of the
product estimator at Tier 2. Neither is trusted alone. If the two disagree
sharply on a molecule, that is surfaced, not averaged away.

Both are **validatable** — unlike the v1 composite — against the *measured*
selectivity label on held-out cross-measured molecules (gap #3). The product form
encodes the clinical reality that selectivity needs activity at the target **and**
cleanliness at **every** off-target (one bad off-target kills it).

**Potency floor is intrinsic.** The `P(active | target)` factor drives
`P(selective)` low for molecules inactive everywhere — no separate floor needed.

**Rejected.** *Product only* — cheap but compounding error goes unchecked.
*Direct only* — cleaner but cannot screen the wide library and dies if Gate 0 is
thin. The hybrid is what lets a data-rich cheap screen coexist with a data-lean
honest validator.

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

## 6. The deep dive (stage A, Tier 3): confirmatory scoring + generation

**Decision.** The expensive tier does two distinct things on the handful of
selected molecules, in this order:

1. **Confirmatory structure-based docking** (AutoDock Vina into published
   JAK1/2/3 structures) — *orthogonal* evidence against the ligand-based ML score.
2. **Conditional generation** of analogues toward higher `P(selective)` — labelled
   in-silico hypotheses, AD-filtered.

Both are re-scored through the same `src` B modules.

**Why confirmatory scoring is the primary deep-dive, not generation.** The user's
funnel is "narrow, then dig **deep**" — dig deep means *learn more about the few
survivors*, which is a **scoring/analysis** axis. Generation instead *creates new
molecules* — an **expansion** axis, orthogonal to digging deep. They are both
useful but they are not the same thing, and conflating them (the earlier framing)
made "deep dive" mean "make more molecules", which does not deepen knowledge of the
selected case at all. So the deep dive leads with confirmatory analysis; generation
is the second, optional axis.

**Why docking, and its honest limits.** Docking is *structure-based* and therefore
**orthogonal** to the *ligand-based* ML — it can corroborate or contradict the ML's
selectivity call using different information (the actual binding pocket). But
docking scores correlate weakly with measured affinity, so docking is used for
**consensus and pose inspection, never as a second oracle**. A molecule the ML
calls selective *and* that docks preferentially into the target pocket over the
off-isoforms is a stronger hypothesis; disagreement is flagged, not buried. Vina is
CPU-capable but slow, which is why it lives in the expensive Tier-3 stage on a
handful of molecules, not in the wide screen.

**Why generation stays hypothesis-only.** De-novo generation of *validated*
selective molecules on a zero-cost / Colab budget is not achievable; pretending
otherwise recreates gap #1 in fancier form. The defensible deliverable is a
**distribution shift** — does the generated population's `P(selective)` move up
while staying in-domain? That is checkable; "we found a drug" is not.

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

**Decision.** Stage B (Tiers 0–2: classifiers, conformal, AD, dashboard) runs on
CPU within the free-host budget (~1 vCPU / ~1 GB). GPU is confined to the offline
Colab notebook (Tier 3: docking + generation) and never touches the deployed app.

**Why.** The constraint is the project's premise. Classification, conformal, and AD
are all cheap CPU operations on ECFP4 features; only docking and generation want a
GPU, and they are inherently offline (a human selects a case first). Keeping the
shared scoring `src` modules CPU-only is also what lets the *same* code run in both
the app and the notebook.

---

## 10. The funnel as explicit cost tiers

**Decision.** Order the pipeline as tiers of increasing per-molecule cost, each
rejecting most of its input, so expensive operations only ever touch a handful of
survivors:

| Tier | Cost / molecule | Operation | Rough cut |
|------|-----------------|-----------|-----------|
| 0 | near-free | Ro5 + PAINS rules | 10^5 → 10^5- |
| 1 | ms | classifiers → product `P(selective)` | 10^5 → 10^3 |
| 2 | pricier | conformal + AD (+ direct re-rank) | 10^3 → 10^2 |
| SELECT | human | judgment | 10^2 → few |
| 3 | expensive | docking + generation | few |

**Why.** This *is* the "cheap wide screen → expensive deep dive" goal, made
mechanical. The decisive rule: **the expensive per-molecule operations run only
after the cheap classifier has pruned the library.** Applicability domain in
particular is O(N_train) per query (Tanimoto to the training set) — cheap on 10^2
survivors, prohibitive on a 10^5 library. Deferring AD and conformal to Tier 2 (and
docking to Tier 3) is what keeps the funnel economically real rather than a flat
"score everything expensively" pipeline wearing a funnel diagram.

**Scalability note.** If Tier-1 survivor sets get large, AD nearest-neighbour uses
approximate search (LSH / FAISS on packed fingerprints) or a fixed diverse training
reference, keeping Tier 2 bounded.

---

## 11. The wide screening library (and the AD tension)

**Decision.** The screen's input is a large, diverse, **target-agnostic** library
(default ~10^5 drug-like molecules from ZINC20 lead-like or a PubChem random
sample, source pluggable), **separate** from the target's own ChEMBL actives.

**Why.** v1's "novel candidates" came from PubChem *similarity* expansion of known
actives — near-duplicates inside the training neighborhood (gap #1). That is not a
wide screen; it is retrieval. A genuine funnel needs a broad haystack the cheap
tiers search, unrelated to what the model already knows.

**The tension, stated openly.** A diverse library means **most of it is
out-of-domain** for models trained on ChEMBL JAK chemistry. Naively, that looks
fatal — the screen would return mostly "uncertain". It is not fatal; it is the
*reason the AD tier exists*. The honest reading:

- The wide screen **applies** the model broadly (cheap, Tier 1) to find candidate
  selective molecules.
- The AD tier (Tier 2) then **restricts trust** to the in-domain subset.
- The funnel's output is therefore "**selective *and* in-domain**" molecules — a
  defensible middle ground between "only re-rank known actives" (too narrow, v1)
  and "trust predictions on wild extrapolations" (dishonest).

So AD-vs-wide is not a bug to hide but the exact mechanism that makes a broad cheap
screen honest. The size is demo-scale on the zero-cost budget; the design scales to
larger libraries bounded only by Tier-1 throughput.
