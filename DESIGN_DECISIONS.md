# Design decisions — JAK-selectivity screening funnel

Why the funnel is built the way it is. This document is the rationale companion to
the [Roadmap](README.md#how-the-funnel-was-built-step-by-step). It records
decisions and the reasoning (including rejected alternatives), so future changes
argue with the reasoning rather than silently reverse it.

**Status:** built (STEP 0–9). This records the *why*; the measured numbers live in
[VALIDATION.md](VALIDATION.md). Where a threshold is stated, it is a *gate* the
build passed.

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

## 1. Task: pchembl regression + a validated selectivity gap

> **This decision reversed an earlier one, on data.** The plan first reframed the
> task as binary active/inactive classification to fix the censored-label gap
> (#4). **Gate 0 measured the JAK data and killed that plan** (see
> [VALIDATION.md](VALIDATION.md)): the inactive class barely exists — 75 / 333 /
> 245 inactives (pchembl ≤ 5) against 9.6k / 11k / 6.3k actives for JAK1 / 2 / 3.
> A classifier cannot be trained or calibrated on 75 negatives, and binarizing at
> pchembl 6 *destroys the selectivity signal* (a 9.0 and a 7.0 both become
> "active"). The gate did its job: it invalidated an assumption before we built on
> it.

**Decision.** Each isoform model is a **pchembl regressor** (the v1 approach,
which the data strongly supports — ~10k molecules each, pchembl range 4–11).
Selectivity is a **predicted gap** between isoform regressors, validated against
the *measured* gap on cross-measured molecules (§2).

**Why regression is the data-appropriate choice.** JAK ChEMBL is ~99 % actives
because `fetch_activities` returns only records with a quantified pchembl — and
molecules get measured because they are expected binders. True non-binders are
right-censored (`>` values, no pchembl) or simply absent. So the negative class
needed for classification is not in the data, but the *continuous potency*
regression needs is abundant. Selectivity, in this target family, lives in the
**pchembl gap among co-measured actives**, not in an active/inactive split — which
is exactly what Gate 0's gap counts confirm (593 JAK1-selective at ≥10×, 2073 in
the JAK1–JAK2 pairwise view).

**The censored-label gap (#4) is mitigated, not solved.** Regression still cannot
*recognize a non-binder* from a diverse library. Since the data denies us a real
inactive class, that burden moves to the **applicability domain** (§5): on the
wide library, AD flags "unlike anything I trained on", which is the honest
substitute for "I have never seen a true inactive." A DUD-E-style decoy class for
a genuine binder/non-binder gate is kept **in reserve** as an optional future
addition, not a dependency.

**Rejected alternatives.**
- *Binary classification (the reverted plan).* Killed by Gate 0 — no inactive
  class, and it discards the gap that selectivity is made of.
- *Fetch right-censored `>` inactives separately.* Sparse and biased (which
  compounds got tested as `>` is not random); at best a partial patch, not worth
  the complexity now.

---

## 2. Selectivity as a predicted gap — hybrid estimator

**Decision.** Selectivity is the pchembl **gap** between the target and its worst
off-isoform. Two estimators, used at different funnel tiers:

- **Difference-of-regressors (wide, Tier 1)** — from the per-isoform regressors:
  ```
  S(target) = pchembl_pred(target) − max_off  pchembl_pred(off-isoform)
  ```
- **Direct gap regressor (narrow, Tier 2)** — a single regressor trained on the
  **cross-measured** set to predict the measured gap directly.

`max` over off-isoforms (worst-case) because selectivity is limited by the
*closest* off-target; `S = +1` ≈ 10× selective, `+2` ≈ 100×.

**Why a hybrid, not one or the other.** The two trade off along the exact axis the
funnel cares about:

- The **difference-of-regressors uses all per-isoform data** (every molecule
  measured on *any* isoform trains its regressor) and can score *any* molecule, so
  it is the only estimator that works on the unlabelled wide library. Its weakness:
  it subtracts two independent predictions, so **their errors add**.
- The **direct gap regressor avoids error stacking** (it learns the gap in one
  shot) but can only train on the **cross-measured** set (3624 molecules, 3-way —
  healthy per Gate 0, but far smaller than the per-isoform sets). Using it to
  screen the whole library would extrapolate past its support.

So: difference-of-regressors does the cheap wide screen (Tier 1); the direct gap
regressor **re-ranks the survivors** and serves as an **independent validator** at
Tier 2. Neither is trusted alone; sharp disagreement is surfaced, not averaged.

Both are **validatable** — unlike the v1 composite — against the *measured* gap on
held-out cross-measured molecules (gap #3). Gate 0 confirms the ground truth
exists: hundreds of ≥10×-selective molecules per isoform (2073 in the JAK1–JAK2
pairwise view), the positive set the hero figure and Gate 4 need.

**Potency floor.** Unlike the classification product, a pure gap does *not*
self-suppress a molecule inactive everywhere (a weak-vs-weaker pair can still show
a gap). So the ranking applies a **target-potency floor** (default
`pchembl_pred(target) ≥ 6`) and always displays the pair *(target potency, gap)* —
selectivity is never shown in isolation.

**Rejected.** *Difference-only* — cheap but stacked error unchecked. *Direct-only*
— cleaner but cannot screen the wide library. The hybrid lets a data-rich cheap
screen coexist with a data-lean honest validator.

---

## 3. Metrics: regression + selectivity validation

**Decision.** Evaluate per-isoform regressors with **MAE / RMSE / R² / Spearman**,
reported as mean ± std over **≥5 scaffold-split seeds**. Validate selectivity with
the **gap Spearman** (predicted vs measured gap) and **enrichment of ≥10×-selective
molecules** on the cross-measured held-out split.

**Why.** These are the honest, validatable metrics for a continuous target, and —
critically — the selectivity metric is computed against a *measured* gap, which
the v1 composite never had. Spearman on the gap answers the only question that
matters ("does predicted selectivity rank molecules like measured selectivity
does?"), and enrichment answers the funnel's question ("do the top-ranked survivors
concentrate the truly selective ones?").

**Seeds.** The existing `scaffold_split` is deterministic and takes no seed; a
`seed` argument (shuffling within the greedy scaffold fill) is the one surgical
change needed to report mean ± std instead of a single fragile draw.

---

## 4. Uncertainty: conformal regression

**Decision.** Split/inductive **conformal regression** producing **prediction
intervals** at **90 %** nominal coverage (default) per isoform, coverage
empirically verified on the scaffold-split test set. The selectivity gap's interval
is propagated from the two isoform intervals.

**Why.** Conformal gives a distribution-free, finite-sample coverage guarantee
under exchangeability — the honest way to say "how sure" for a continuous
prediction. **Gate:** empirical coverage in **88–92 %** at 90 % nominal, or the
step is not done.

**Assumption stated loudly.** Conformal coverage holds under exchangeability
between calibration and test data. A scaffold split deliberately breaks that
(test scaffolds are novel), so coverage on genuinely new chemotypes is expected to
be the *stress case* — which is exactly why AD (below) runs alongside it.

---

## 5. Applicability domain: two definitions, propagated

**Decision.** Flag every prediction with **≥2** AD definitions —
**Tanimoto distance to nearest training molecule** and **descriptor-space
leverage** — and propagate to selectivity: a selectivity gap `S` is **uncertain**
if *any* contributing isoform model is out-of-domain.

**Why.** One AD metric is gameable; two orthogonal views (fingerprint-space
neighborhood vs descriptor-space extrapolation) are harder to fool. Propagation is
worst-case because `S` is a difference of two isoform predictions and is only as
trustworthy as its shakier half. AD also carries the non-binder burden that
regression alone cannot (§1). **Gate — the money plot:** out-of-domain
error must be *systematically higher* than in-domain error with a clear margin; if
AD does not separate error, it is decorative and the step fails.

---

## 6. The deep dive (stage A, Tier 3): confirmatory scoring + generation

**Decision.** The expensive tier does two distinct things on the handful of
selected molecules, in this order:

1. **Confirmatory structure-based docking** (AutoDock Vina into published
   JAK1/2/3 structures) — *orthogonal* evidence against the ligand-based ML score.
2. **Conditional generation** of analogues toward higher gap `S` — labelled
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
**distribution shift** — does the generated population's gap `S` move up while
staying in-domain? That is checkable; "we found a drug" is not.

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

**Why.** The gap `S` can be *computed* from predictions for any molecule, but it
can only be *validated* on molecules with a measured gap across the relevant
isoforms. If that co-measured set is too thin, the hero validation (§3) is not
credible, and no amount of modeling fixes it.

**Result (measured, see [VALIDATION.md](VALIDATION.md)).** Gate 0 ran: 3-way
cross-measured = **3624**, with ample gap-based positives (593 JAK1-selective at
≥10×; 2073 in the JAK1–JAK2 pairwise view). Above threshold → **3-isoform
selectivity proceeds**, pairwise kept in reserve for a ≥100× story. The gate also
overturned the classification plan (§1) — its real payoff.

---

## 9. CPU-only / zero-cost split

**Decision.** Stage B (Tiers 0–2: regressors, conformal, AD, dashboard) runs on
CPU within the free-host budget (~1 vCPU / ~1 GB). GPU is confined to the offline
Colab notebook (Tier 3: docking + generation) and never touches the deployed app.

**Why.** The constraint is the project's premise. Regression, conformal, and AD
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
| 1 | ms | regressors → gap `S` | 10^5 → 10^3 |
| 2 | pricier | conformal + AD (+ direct re-rank) | 10^3 → 10^2 |
| SELECT | human | judgment | 10^2 → few |
| 3 | expensive | docking + generation | few |

**Why.** This *is* the "cheap wide screen → expensive deep dive" goal, made
mechanical. The decisive rule: **the expensive per-molecule operations run only
after the cheap regressor has pruned the library.** Applicability domain in
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
