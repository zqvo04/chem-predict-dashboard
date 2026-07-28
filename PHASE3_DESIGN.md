# Phase 3 design — the acquisition loop

**Status: design, not built.** Nothing in this document has been measured except
the pool statistics in §1, which were taken from the committed datasets. Numbers
describing what active learning *achieves* do not appear here, because that is the
thing Phase 3 exists to find out. When it is built, results go to VALIDATION.md and
the rationale that survives contact with them goes to DESIGN_DECISIONS.md.

Phase 2 built the state (`Campaign`, `registry`) that a loop needs. Phase 3 is the
loop: something that reads round *N* and chooses round *N+1*.

> **Assumptions taken.** Four design forks were put to the project owner and not
> answered, so the recommended option was taken in each case and is flagged
> **[assumption]** at the point it bites. Each is cheap to reverse before
> implementation and expensive after, so they are worth a look: the hit definition
> (§2), the acquisition set (§4), the conformal-under-AL fix (§5), and running the
> benchmark on both panels (§6).

---

## 1. What the pool actually looks like (measured)

The benchmark needs molecules whose **selectivity gap is already measured**, so the
pool is the cross-measured set — the same set STEP 4 validated the gap on. Measured
from the committed datasets:

| | JAK (validated) | PI3K (bootstrap) |
|---|---:|---:|
| pool size (measured on every member) | 3 624 | 1 500 |
| median gap | +0.03 | −0.08 |
| p90 gap | +1.28 | +1.36 |
| max gap | +2.57 | +4.10 |
| **≥10× selective (gap ≥ 1)** | **593 (16.4 %)** | **236 (15.7 %)** |
| **≥100× selective (gap ≥ 2)** | **30 (0.8 %)** | **59 (3.9 %)** |
| distinct Murcko scaffolds | 1 334 (36.8 %) | 712 (47.5 %) |

Three things follow directly, and they shape the whole design:

1. **The potency floor is not binding here.** On JAK, ≥10×-selective and
   "≥10×-selective *and* predicted-potent ≥ 6" are the same 593 molecules. The
   cross-measured set is made of compounds someone thought worth assaying on the
   whole panel, so they are already potent. The AL target is therefore **purely the
   gap** — the floor cannot be what distinguishes a good acquisition from a bad one.
2. **≥10× is not a hard search problem.** At a 16 % base rate, random sampling finds
   plenty. Any honest report has to allow for "AL barely beats random here."
3. **≥100× is a genuinely rare-event problem, and the two panels differ sharply**
   (0.8 % vs 3.9 %). That difference is the most useful thing in this table: it is a
   natural experiment on whether AL's advantage depends on rarity.

Random-baseline expectation vs the perfect-oracle ceiling, by budget:

| panel | budget | random ≥10× | perfect ≥10× | random ≥100× | perfect ≥100× |
|---|---:|---:|---:|---:|---:|
| JAK | 600 (16.6 % of pool) | ~98 | 593 | **~5.0** | **30** |
| PI3K | 300 (20 % of pool) | ~47 | 236 | **~11.8** | **59** |

The ≥100× row is where a benchmark can actually resolve a difference: a 6× dynamic
range between random and perfect, with room for AL to land somewhere meaningful in
between. This is the argument for reporting both thresholds rather than picking one.

**Ceiling on greedy.** Greedy acquisition can only be as good as the model's ranking,
and STEP 4 measured that ranking at Spearman ≈ 0.80 (pooled assay types) with
top-decile enrichment 4.5×. Phase 3 should not expect greedy to approach the perfect
curve, and if it does, that is evidence of leakage rather than success.

---

## 2. The oracle

**Decision.** A **retrospective label-revealing oracle**. The pool's measured values
are hidden at the start; selecting a molecule reveals them and costs one unit of
budget.

**One oracle call reveals the molecule's whole panel row**, not just its gap. That is
what a selectivity assay physically is — you run the compound against the panel — and
it keeps the simulation faithful to the deployed pipeline, which predicts each isoform
separately and subtracts. It also means the loop retrains *the same
difference-of-regressors estimator the funnel deploys*, so a Phase 3 result is about
the shipped model rather than a stand-in.

**Rejected: a surrogate-model oracle.** Training a high-capacity model on everything
and calling its predictions "truth" measures whether the loop can recover that model,
not whether it can find selective chemistry. It is circular and it would produce
flattering numbers.

**Rejected: docking.** CPU-only, needs structures per isoform, and its correlation
with the measured gap would itself need validating — a project, not a component.

**Honest limit to record with any result.** A retrospective pool is not a design
space: everything in it was synthesised and assayed by someone who had a reason. The
loop is choosing *what to measure next from a fixed catalogue*, which is the
"screening" half of DMTA, not the "design" half. Generative expansion exists
(`src/generate.py`) but its molecules have no measured labels, so they cannot be part
of a benchmarked loop.

---

## 3. The loop protocol

```
round 0   seed: a random, scaffold-stratified sample     (4 % of pool)
          ├─ reveal labels
          └─ fit per-isoform regressors + AD reference + conformal calibration
round n   score every unlabelled molecule                (Tier 1 + Tier 2)
          ├─ acquisition ranks them
          ├─ take a batch                                (2 % of pool)
          │    ├─ 80 % chosen by the acquisition function
          │    └─ 20 % drawn at random  → the calibration stream (§5)
          ├─ reveal labels, append a Round to the registry
          └─ refit everything on the enlarged labelled set
stop      after 8 rounds ≈ 20 % of the pool labelled
```

Sizes are fractions of the pool, not constants, so the two panels stay comparable:

| | JAK | PI3K |
|---|---:|---:|
| seed (4 %) | 150 | 60 |
| batch (2 %) | 70 | 30 |
| rounds | 8 | 8 |
| **total budget** | **710 (19.6 %)** | **300 (20.0 %)** |

**Why cap at ~20 %.** Label enough of a finite pool and every strategy converges on
the same answer, because there is nothing left to be clever about. Keeping the budget
well under half the pool is what leaves room for a difference to exist.

**What is refit each round, and what is not.** The per-isoform regressors, the AD
reference and the conformal calibration are all functions of the labelled set, so all
three are refit. The **binder gate is not in the loop at all** — every molecule in a
cross-measured pool is already a known binder, so the gate would pass all of them.
This must be stated wherever Phase 3 results appear: **the AL benchmark operates
entirely inside the region the gate already cleared, and therefore says nothing about
whether the gate works.** That is STEP 10's claim and it stays STEP 10's.

---

## 4. Acquisition functions

**[assumption]** Implement all four below plus the mandatory random baseline. The set
is chosen so that each pair differs in exactly one thing, which is what makes the
comparison interpretable rather than a leaderboard.

| name | rule | what it isolates |
|---|---|---|
| **random** | uniform | the baseline. Not optional — without it "AL works" is unfalsifiable. |
| **greedy** | top predicted gap | pure exploitation; the common industrial default |
| **ucb** | `gap + κ · halfwidth` | exploitation + calibrated exploration |
| **uncertainty** | widest `halfwidth` | pure exploration — finds a better *model*, not better molecules |
| **diverse-greedy** | greedy, one molecule per Murcko scaffold per batch | whether batch redundancy is what limits greedy |

Two of these are load-bearing for reasons specific to this repo:

**`ucb` is where Phase 2 pays off.** Its exploration bonus is the STEP 14 half-width
— `q · σ(nn_similarity)` — which is *already* a per-molecule, distance-from-training
quantity on a calibrated scale. Most UCB implementations need an uncertainty estimate
invented for the purpose; this one falls out of work already validated. κ is a free
parameter and will be swept over a small grid (0.5, 1, 2) rather than tuned to the
answer.

**`uncertainty` vs `greedy` separates two goals that get conflated.** "Active
learning works" can mean *the model got better* or *we found more good molecules*.
Pure uncertainty sampling optimises the first and should lose badly on the second.
Reporting both makes the distinction unavoidable — and model quality is worth its own
curve, because a campaign that ends with a better model has produced something even
if it found nothing.

**`diverse-greedy` is expected to matter.** Only 37 % of JAK's pool is distinct
scaffolds, so a greedy batch of 70 can easily be a handful of chemotypes in
triplicate. The comparison against plain greedy measures exactly that cost.

---

## 5. The problem this design exists to get right: conformal under AL

**Split conformal's coverage guarantee holds under exchangeability between the
calibration set and the test set.** Active learning deliberately destroys that: it
selects the extremes of the model's own predictions, so a calibration set drawn from
acquired molecules is systematically unlike the pool it will be used to make claims
about. Naively refitting the STEP 14 calibration on AL-acquired data each round would
quietly void the one uncertainty guarantee this repo has actually earned — and it
would do so invisibly, since the reported coverage would still be computed on the same
skewed sample.

This is the most consequential decision in Phase 3, because the alternative failure is
silent.

**[assumption] Fix: a random calibration stream.** 20 % of every batch is drawn
uniformly at random from the unlabelled pool, and **only that stream is used for
conformal calibration**. It stays exchangeable with the pool by construction, so
coverage means what STEP 14 says it means.

- **Cost, stated up front:** one fifth of the budget is spent on exploration the
  acquisition function did not ask for. For JAK that is 14 molecules per batch, 112
  over the run. This is a real cost and the benchmark will report the acquisition
  curve *with* it, since that is what an honest deployment would pay.
- **Bonus:** the stream is an unbiased sample of the pool, so per-round coverage can
  be checked rather than assumed.
- **What it is not:** it is not the random baseline. That needs its own full run.

**Rejected: freeze the calibration from the round-0 random seed.** Cheaper and it
keeps exchangeability, but the calibration would go stale as the model improves —
intervals would stay wide while the model got sharper, and the UCB bonus would be
reading a curve fitted to a model that no longer exists.

**Rejected: weighted conformal.** Theoretically the right answer, wastes no budget,
and reweights by the inverse selection probability to restore exchangeability. It is
also considerably harder to implement and to validate, and if the weight estimates are
wrong it fails *silently* — the same class of failure as doing nothing. Worth
revisiting once the random-stream version has produced a coverage curve to check it
against; not worth leading with.

**Gate:** per-round empirical coverage on the random stream must stay in 88–92 %. If
it drifts, the acquisition loop is breaking the uncertainty estimate and that is a
finding to report, not a bug to hide.

---

## 6. Benchmark design

**[assumption] Both panels.** JAK and PI3K, because a conclusion drawn from one panel
cannot distinguish "AL helps" from "AL helps on JAK". Their ≥100× rarity differs
5-fold (0.8 % vs 3.9 %), which makes them a genuine contrast rather than a repetition.
Cost is 2× compute. This is also the first real return on Phase 2's generalisation.

**Design:** 5 acquisition strategies × 5 seeds × 2 panels × 8 rounds. Seeds control
the round-0 sample and every stochastic tie-break, and all strategies within a seed
share the same round-0 set, so early-luck differences do not masquerade as strategy
differences.

**[assumption] Primary metric: hits found vs budget spent**, reported at both
thresholds — ≥10× and ≥100× — with the random baseline and the perfect-oracle ceiling
on the same axes. Both, because §1 shows they ask different questions: ≥10× is where
AL probably has little to offer, ≥100× is where it might have a lot, and publishing
only the flattering one would be the same failure as the composite score in v1.

**Secondary metrics** (each answers a question the primary cannot):
- **model quality vs budget** — held-out Spearman of predicted vs measured gap on the
  never-labelled remainder; separates "better model" from "better molecules"
- **coverage vs round** on the random stream — the §5 gate
- **scaffold diversity of what was acquired** — whether a strategy won by finding one
  rich chemotype, which is a materially different result from finding several
- **rank of the best molecule found** — a campaign that finds the single best compound
  early is valuable even if its total count is unremarkable

**What would falsify "AL helps here."** Stated in advance, so the answer is not chosen
after seeing the data: if the acquisition curves for greedy/ucb/diverse-greedy fall
within the seed-to-seed spread of random at both thresholds on both panels, the honest
conclusion is that **active learning does not pay on this pool**, and that goes in
VALIDATION.md as a negative result with the same prominence a positive one would get.
Given the 16 % base rate at ≥10×, a partial version of this outcome is likely, and the
design expects it.

**Statistics.** 5 seeds is enough to see a large effect and not enough for a small
one. Report mean ± spread across seeds and refuse to claim any difference smaller than
that spread — no significance theatre on n=5.

---

## 7. Integration with Phase 2

The registry already stores what the loop needs; Phase 3 adds the part that *reads* it.

- New `Round.kind` values: `"acquire"` (a batch was chosen) and `"reveal"` (labels
  came back). `metrics` carries the acquisition name, κ, hits found this round,
  cumulative hits, and the round's coverage on the random stream.
- Each round's scored table already persists as parquet, so the acquisition for round
  *N+1* reads round *N* from disk rather than from memory. **This is the property that
  makes the loop resumable** — and resumability is what separates a benchmark script
  from a campaign a human can actually run across sessions.
- `Campaign.validation` is untouched. An AL campaign on a `bootstrap` panel is still
  bootstrap; running more rounds is not evidence that the panel's gates pass.

**New modules** (mirroring how the repo already splits things):

| file | responsibility |
|---|---|
| `src/acquire.py` | the acquisition functions; pure, `(scored_frame, k) -> indices` |
| `src/al.py` | the loop: seed, round, refit, record — one campaign, one strategy |
| `scripts/al_benchmark.py` | the grid over strategies × seeds × panels; writes the curves |
| `scripts/make_al_figure.py` | acquisition curves + the coverage-vs-round panel |

Keeping acquisition pure and separate from the loop is deliberate: it makes each
function testable on a synthetic frame with no models involved, which is the only way
the "does UCB actually rank by gap + κ·halfwidth" question gets a fast test.

---

## 8. Gates

Phase 3 is done when, and not before:

1. **Gate A — the loop runs and resumes.** A campaign interrupted after round 3 and
   restarted from the registry produces the identical round 4. Without this it is a
   script, not a loop.
2. **Gate B — random is really random.** The baseline's hit curve matches the
   hypergeometric expectation within seed spread (JAK ≥10×: ~98 at budget 600). This
   catches pool leakage — the single most likely way to get a spuriously good result.
3. **Gate C — coverage survives.** Per-round coverage on the random stream stays in
   88–92 %, or the deviation is reported as a finding.
4. **Gate D — the comparison resolves.** Every strategy's curve is reported with seed
   spread, and the writeup states plainly whether the differences exceed it — including
   when they do not.
5. **Gate E — JAK is still bit-identical.** Same constraint as STEP 15: Phase 3 adds a
   loop on top and must not perturb the deployed screen. Pinned by the existing
   model-id test.

---

## 9. Risks

| risk | why it is plausible | mitigation |
|---|---|---|
| **AL shows no benefit at ≥10×** | 16 % base rate leaves little headroom | expected; report as a conditional result, which is why ≥100× is measured too |
| **Pool leakage inflates everything** | the same molecules train and are scored | Gate B; scaffold-stratified seed; unlabelled remainder held out for the model-quality metric |
| **Greedy wins by chemotype redundancy** | 37 % distinct scaffolds | `diverse-greedy` and the acquired-diversity metric exist to expose this |
| **Coverage drifts and is not noticed** | AL breaks exchangeability by construction | §5's random stream; Gate C |
| **The result is JAK-specific** | one panel, one target family | both panels; §6 |
| **Compute grows quietly** | 5×5×2×8 = 400 round-fits | pool ≤ 3.6k molecules and models are small; budget a full run at well under an hour and measure it before scaling the grid |

---

## 10. Open questions

Genuinely undecided, and better answered with data than in advance:

1. **Is the direct gap regressor a better AL model than difference-of-regressors?**
   STEP 4 kept both. The oracle reveals the whole panel row, so both are trainable
   every round at no extra budget — worth measuring rather than assuming, but it
   doubles the grid, so it is a follow-up rather than part of the first run.
2. **Should the acquisition see the AD verdict?** Excluding out-of-domain molecules
   would make the loop safer and less exploratory at the same time. Currently unhandled
   by design: AD is reported per round but does not gate acquisition.
3. **κ for UCB.** Swept, not tuned. If the best κ differs between panels that is itself
   worth reporting.
4. **Does any of this transfer to the wide library?** The library has no measured gaps,
   so the loop cannot be benchmarked there — only deployed. Phase 3 should state what,
   if anything, licenses carrying a conclusion across.
