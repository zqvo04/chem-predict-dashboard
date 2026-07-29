# Where this project actually stands, and what it would take to finish it

Part 1 is an audit: what is measured, what is broken, what is claimed but unearned.
Part 2 is a roadmap that follows from it. Everything in Part 1 has a number and a
command behind it.

The short version: **the platform engineering is largely done and the science has a
disqualifying hole.** The trust layer — the thing this project exists to be — does not
survive contact with molecules that were actually tested and found inactive. That is
one problem, it is fixable, and until it is fixed nothing downstream is worth
optimising.

---

# Part 1 — Audit

## 1.1 The disqualifying finding

The binder gate rejects **98.4 %** of the negatives it was validated on and **52.3 %**
of molecules *measured* as JAK non-binders (`scripts/gate_negative_audit.py`, 646
molecules, every record `>` at ≥ 10 µM, none in training):

| | presumed inactives (STEP 10) | **measured inactives** |
|---|---:|---:|
| rejected | 98.4 % | **52.3 %** |
| regressor still clears the potency floor | — | **85.3 %** |
| mean predicted JAK1 pchembl on known-dead molecules | — | **6.60** |
| survive floor **and** ≥10× selectivity | — | 5.7 % (37 / 646) |

STEP 10 exists because "ethanol scored 6.12". The gate fixed that for *obvious* junk
and did not fix it for **drug-like molecules a chemist thought worth testing** — which
is the population a real screen faces. The 0.998 ROC-AUC is correctly measured against
the wrong class.

Fair caveats, stated because they bound the claim rather than soften it: this set is
adversarially hard by construction, and on the actual wide library the gate passes
18.9 % (STEP 13). Nothing here touches the regression, gap, conformal or AD numbers,
which are measured on quantified data and stand.

## 1.2 The headline selectivity number is still confounded

Spearman **0.798** pooled across assay types; **0.462** on the ATP-independent Ki/Kd
subset at matched sample size, with top-decile enrichment collapsing below random.
Measured in STEP 4's audit, never fixed. Everything downstream — the gap, its
interval, the ranking, any future active-learning benchmark — inherits it.

## 1.3 The output is thinner than it looks

Of the 60-molecule shortlist: **33 in-domain**, and only **28 carry a directional
selectivity claim** at 90 % (the interval excludes zero). So roughly half the headline
output is "we cannot say".

Of those 60, **57 have no JAK measurement at all** — genuinely untested hypotheses,
which is the right shape for a screen. The **3 that can be checked all measure
pchembl ≈ 5.1 or are censored**, i.e. below the potency floor the model put them
above. n = 3 proves nothing; it is not encouraging.

## 1.4 Nothing has ever been tested

Every number in this repo is retrospective. There has been no prospective validation,
no compound ordered, no assay run. The Make and Test halves of DMTA do not exist —
Phase 3's oracle is explicitly a label-revealing simulation.

## 1.5 Coverage against "end-to-end drug discovery"

| stage | status |
|---|---|
| target selection / validation | **absent** |
| hit finding (virtual) | built, validated retrospectively |
| hit-to-lead (SAR, series) | partial — analogue generation, no series concept |
| ADMET / DMPK | **2 thin models** (ESOL, Tox21-any-hit). No hERG liability model — hERG is used as a gate *negative target* and never as a safety endpoint. No CYP, no stability, no permeability |
| synthesis / sourcing | **absent** — nothing checks whether a proposed molecule can be made or bought |
| IP / freedom to operate | **absent** |
| in vivo, formulation, regulatory | **absent** |

Honest fraction of "end-to-end": one stage of about seven, done well.

## 1.6 What is genuinely solid

Not everything is a caveat, and the audit should say so:

- **Uncertainty is real and calibrated.** Conformal coverage 0.888–0.905 per isoform;
  the gap interval is directly calibrated and difficulty-scaled with worst-bucket
  coverage 0.889 (STEP 14). Very few projects at this scale do this correctly.
- **Leakage is hunted rather than assumed.** STEP 13's library replacement, STEP 16's
  molecule-level check that found a leak I had introduced in STEP 15.
- **Reproducibility is close to complete.** Models pinned by behavioural digest,
  data by ChEMBL release, code by commit; 103 420 measurements committed as 5.7 MB of
  parquet so any reader can check the numbers.
- **Negative results get published.** The 0.462 audit, the 0.460 mis-attribution
  correction, and §1.1 above are all in VALIDATION.md at full strength.

That discipline is the asset. It is also why §1.1 is survivable: the project found its
own worst result and wrote it down.

## 1.7 Engineering debt

- Training data still comes from committed parquet, not the store; the shipped models'
  `data_version` is `unrecorded`.
- Multi-task is impossible until the other 30 targets are ingested (~1 h network).
- PI3K is `bootstrap` with a known 134-molecule library leak.
- The DB is single-writer and local — no path for anyone else to submit data.

---

# Part 2 — Roadmap

## 2.0 First, name the achievable goal

"End-to-end drug development" by one person on public data and CPUs is not achievable,
and a roadmap that implies otherwise is the v1 failure at project scale. What *is*
achievable, and rare:

> **A computational hit-finding platform for target selectivity whose every claim is
> calibrated, leak-checked and reproducible — with a closed decision loop and a real
> interface to experiment.**

That is end-to-end *within a defensible boundary*. The difference between that and a
demo is §2.5.

## 2.1 R0 — fix the gate *(blocking, ~1 week)*

Nothing else matters until the trust layer survives §1.1.

- Retrain the gate with **measured** negatives from the store (2 751 for JAK1 alone)
  alongside the physchem-matched presumed ones. Keep both: censored compounds are
  biased toward things a chemist tried, which is a different bias, not none.
- Treat `> 10 µM` as a bound, not a point label. Naive relabelling swaps one error for
  another.
- **The 646-molecule set becomes a permanent held-out benchmark.** Never train on it.

**Gate:** rejection on measured non-binders rises from 52.3 % without the false-negative
rate on known actives rising materially. Report both — a gate that rejects everything
is not an improvement.

## 2.2 R1 — fix the labels *(~1 week)*

Build the selectivity gap from same-document measurements (**5 049** JAK molecules with
all three isoforms measured in one paper — larger than today's 3 624 pooled set), so
assay conditions are held approximately fixed by construction.

**Gate:** matched-set Spearman reported beside 0.798 (pooled) and 0.462 (Ki/Kd). If the
matched gap is no better, the confound is not what limits the model — also a result.

**Order matters:** R1 changes the labels an AL benchmark would be scored against, so it
must precede Phase 3 or that work measures a superseded target.

## 2.3 R2 — close the loop's remaining holes *(~1 week)*

- Rebuild the library with InChIKey-level exclusion against every registered panel;
  retrain the PI3K gate. **Gate:** `library_molecule_overlap` returns zero everywhere.
- Add `data_version` to the loop contract (schema 1.2) and make Stage A refuse a
  release mismatch the way it already refuses a model mismatch.
- Add an explicit, versioned `AdvanceCriteria` on the campaign — "advance if the gap's
  90 % lower bound > 0, in-domain, MPO ≥ 0.5" — so the output is a *decision*, not a
  ranking someone has to interpret.

## 2.4 R3 — make the property axes load-bearing *(~2 weeks)*

MPO currently rests on two thin models, one of which was found 100 % leaked (STEP 13).
Build **hERG** first — the data is already in the repo as a gate negative target — then
CYP inhibition and microsomal stability, each with the same discipline the funnel used:
scaffold split, AD, conformal, leakage check, gate.

**Gate:** each model passes an AD-separation check, or it is reported as decorative and
excluded from MPO.

## 2.5 R4 — the interface to experiment *(~1 week, and the point of the project)*

This is what converts a simulator into a platform:

- an **assay request** schema — what you would hand a collaborator or CRO
- a **result ingest** path — measurements returned, written to the store as a new
  `source`, indistinguishable in structure from ChEMBL rows
- Phase 3's oracle becomes an interface with two implementations: retrospective
  (benchmarking) and **human/CRO** (real)

The schema work is cheap. Building it *before* there is an experiment is still correct,
because it forces the question "what would we actually ask for?" — and because
retrofitting it later means reworking the loop.

## 2.6 R5 — active learning *(Phase 3, ~2 weeks, after R0–R2)*

As designed in [PHASE3_DESIGN.md](PHASE3_DESIGN.md), unchanged except that it now runs
against fixed labels and a fixed gate. The design's own honest expectation stands: at a
16.4 % base rate for ≥10× selectivity, AL may not beat random, and that result gets
published either way.

## 2.7 R6 — one prospective test *(the only step that changes the project's category)*

Everything above is retrospective. **A single round of real measurement is worth more
than all of it**, because it is the only evidence that the pipeline predicts rather
than describes.

Realistic routes for a solo, unfunded project, cheapest first:

1. **Kinase profiling panels** are commercially routine. Screening ~10 compounds
   against JAK1/2/3 is a small four-figure sum — the cheapest possible conversion of
   this project from "validated retrospectively" to "tested".
2. **Academic collaboration.** A lab already running JAK assays loses little by adding
   ten compounds; the shortlist is the offer.
3. **Open science.** Open Source Malaria and similar consortia take computational
   predictions and run them, and publish.
4. **Free-tier profiling.** Some initiatives screen submitted compounds at no cost for
   published targets.

**Gate:** publish the predictions *before* the results arrive, with intervals attached.
That is the difference between a prospective test and a retrospective story, and this
repo has the contract format to make it verifiable.

## 2.8 Sequencing and honest cost

```
R0 gate ──► R1 labels ──► R2 loop holes ──► R5 active learning
                    └────► R3 ADMET  (independent)
                    └────► R4 experiment interface (independent)
                                        └────► R6 prospective test
```

R0–R2 is roughly a month of focused work and removes every disqualifying defect in
Part 1. R3 and R4 are independent and can interleave. R6 is not gated on any of them —
it is gated on money and access — but it means far more after R0.

## 2.9 What still will not be true at the end

State it now so nobody discovers it later. Even with all of R0–R6 done, this project
will not do target validation, synthesis planning, IP, PK/PD or anything in vivo. It
will be a **calibrated, leak-audited, reproducible hit-finding platform for selectivity
with one prospective validation round** — which is an honest and unusually well-evidenced
thing to have built, and is not a drug discovery pipeline end to end. Claiming the
latter would cost exactly the credibility that the rest of this repo spent sixteen
steps earning.
