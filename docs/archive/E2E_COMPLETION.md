> **아카이브 (2026-08-03).** [ROADMAP.md](../../ROADMAP.md)로 대체됐다. §2의 순서 제약(라벨 → 게이트 → 건초더미 → AL)은 반증 체인 밖의 별도 축이었고 **끝까지 살아남아** ROADMAP §4에 그대로 있다. §5의 범위 선언은 ROADMAP §9가 이어받았다.
>
> §4의 DB 판단('두 번째 writer가 생길 때까지 DuckDB + 커밋된 parquet')도 유효하며, ROADMAP §10이 그 트리거가 아직 당겨지지 않았음을 확인한다.

---

# Completing the pipeline — what the evidence store unblocked, and in what order

**Status: design.** The numbers in §1 are measured from the committed store
(`assets/evidence/`, ChEMBL_37); everything after is a plan. Results go to
VALIDATION.md as they land.

Two things this document is not. It is not a proposal to broaden the project into
target selection, IP, PK/PD or clinical work — §5 states that boundary explicitly and
argues for declaring it rather than pretending to cover it. And it is not a list of
models to add: the constraint on this pipeline has never been model variety, it has
been **whether a claim can be checked**.

---

## 1. What the store made possible (measured)

Three things were argued about from first principles because the data to settle them
did not exist in queryable form. It does now.

### 1a. The ATP confound is addressable — and the matched set is *bigger*, not smaller

README's sharpest caveat: selectivity Spearman falls **0.682 → 0.462** on the
ATP-independent Ki/Kd subset, because a gap assembled across assay types can encode an
artefact rather than biology. The fix was never available: pairing measurements by
*condition* needs assay and document identity, which was never retrieved.

Measured now, for the three JAK isoforms:

| matching criterion | molecules with all 3 isoforms matched |
|---|---:|
| same assay description | **2 820** |
| **same document (one paper, one lab, one protocol)** | **5 049** |
| (current pooled cross-measured set, for scale) | 3 624 |

The same-document set is **larger than the pooled set the models train on today**.
That inverts the expected trade: a better-controlled gap does not cost data here, it
gains it. A same-paper gap holds assay conditions, ATP concentration, readout and
laboratory approximately fixed by construction — which is what the Ki/Kd restriction
was a crude proxy for.

**This is the highest-value item in the repo.** It attacks the one measured defect
that everything downstream inherits.

### 1b. Real non-binders exist, so the gate can stop presuming

The binder gate's negatives are "presumed inactive by absence" — molecules active on
other targets with no JAK record. DESIGN_DECISIONS §1 named the residual risk itself:
a promiscuous kinase inhibitor that *would* hit JAK but was never tested is a
mislabelled negative.

The store holds **29 321 right-censored records** — compounds someone measured and
found inactive (`>` 10 µM, no pchembl):

| target | molecules measured as non-binders |
|---|---:|
| PIK3CD | 3 755 |
| JAK2 | 3 628 |
| PIK3CG | 3 417 |
| JAK3 | 2 907 |
| JAK1 | 2 751 |
| PIK3CA | 1 911 |
| PIK3CB | 1 197 |

These are *measured* negatives for the exact target, which is strictly better evidence
than absence. They were never fetched before because the client filtered them out
server-side.

### 1c. The PI3K leak is quantified and fixable

STEP 16 measured 134 of 38 592 library molecules that are PI3K gate positives — the
gate scores rows it trained on. Fixing it needs a molecule-level (InChIKey) exclusion
when the library is built, which is a join the store now supports.

---

## 2. Sequencing — and why it matters more than usual

**1a changes the labels the gap model trains on.** Phase 3's active-learning benchmark
measures how well acquisition finds selective molecules *against those labels*. Run the
benchmark first and then fix the labels, and the benchmark measures a target that no
longer exists — weeks of compute answering a superseded question.

So the order is not negotiable:

```
  P4.1  assay-matched gap        (changes labels)            ← do first
  P4.2  measured negatives       (changes the gate)
  P4.3  library leak fix         (changes the haystack)
  P4.4  contract data_version    (small, independent)
  P4.5  multi-task               (needs 30 more targets)
  ────────────────────────────────────────────────────────
  Phase 3  active learning       (measures against 4.1-4.3)  ← then this
```

P4.4 is independent and can land at any time. P4.5 is the only item needing more
network.

---

## 3. The work

### P4.1 — assay-matched selectivity gap
Build the gap from measurements sharing a document (primary) or an assay description
(fallback), instead of pooling across assay types. Re-run STEP 4's validation on the
matched set and report Spearman and enrichment beside the pooled numbers.

**Gate:** the matched-set Spearman is reported next to 0.798 (pooled) and 0.462
(Ki/Kd-only) — the point is a *fair* comparison, not a better number. If the matched
gap is no better, that is the finding: the confound is not what limits the model.
**Risk:** matched-set molecules may be systematically different (a same-paper series
is often one chemotype), so scaffold diversity of the matched set must be reported.
**Cost:** ~1 day, no network.

### P4.2 — measured negatives for the binder gate
Add censored records as gate negatives alongside the physchem-matched ones, and
measure whether the gate improves. Keep the physchem match: censored compounds are
biased toward things a chemist thought worth testing, which is a different bias, not
no bias.

**Gate:** ROC-AUC and — the number that matters — the false-negative rate on held-out
*known actives*, since a gate that rejects real binders is worse than one that admits
junk. Any retrain changes the gate threshold, so the shortlist moves; that is a
VALIDATION entry, not a silent update.
**Cost:** ~1 day, no network.

### P4.3 — close the library leak
Rebuild the library with an InChIKey-level exclusion against every registered panel,
not just a SMILES-level one against JAK. Retrain the PI3K gate.

**Gate:** `library_molecule_overlap` returns zero for every registered panel, and the
app's molecule-level check says "passed" rather than reporting a count.
**Note:** this invalidates the PI3K gate and its screen. JAK's gate is unaffected
(its 8-molecule residual is below the level at which retraining is worth invalidating
contracts — state that, do not hide it).
**Cost:** ~half a day plus a library rebuild.

### P4.4 — pin data in the contract
`Campaign.data_version` exists; the loop contract does not carry it. Add it at schema
1.2 and make Stage A refuse a contract whose data release differs from the one the
models were built on, the way `assert_models_match` already refuses mismatched models.

**Gate:** a contract exported under `chembl_37` fails loudly against a `chembl_38`
rebuild instead of silently re-scoring.
**Cost:** an afternoon.

### P4.5 — multi-task across the corpus
Ingest the remaining 30 targets (20 library + 10 gate-negative), then train a shared
model over `v_molecule_target_matrix`. The payoff is the low-data isoforms: PIK3CB sits
at R² 0.582 on 2 777 molecules, and a shared representation is the standard remedy.

**Gate:** per-isoform R² measured against the current single-task numbers, on the same
scaffold splits. A multi-task model that helps PIK3CB and hurts JAK1 is a real result
and should be reported as a trade, not averaged away.
**Cost:** ~1 hour network + ~1 day.

---

## 4. Database: what to use, and when to change

**Recommendation: stay on DuckDB + committed parquet. Introduce Postgres only when
there is a second writer.**

That trigger is not arbitrary. Every current use is single-writer, read-mostly,
analytical, and local — exactly DuckDB's shape. What would break it is
**someone other than the person running the repo submitting data**: a collaborator, a
CRO returning assay results, the Phase 3 loop accepting real measurements. At that
point you need concurrent writes, authentication and an endpoint, and no amount of
parquet solves it.

| option | when it becomes the right answer | cost |
|---|---|---|
| **DuckDB + committed parquet** (current) | reading, analysis, letting others verify numbers | zero |
| **Supabase / managed Postgres** | a second writer: CRO results, collaborators, a web form | free tier fits 100k rows; an account, a schema migration, and secrets to manage |
| **MotherDuck** | the local store outgrows one machine but stays single-team | account; near-zero migration, same SQL |
| **Datasette** | a browsable read-only UI matters more than SQL access | a host to run it on |

Two honest notes. The repo already has a Supabase organisation, but its one project
belongs to something else and creating another is an account-level action, so nothing
was touched. And Postgres would **not** replace the committed parquet — a hosted
database that goes away takes the verifiability of every published number with it. The
parquet in `assets/evidence/` is the archival copy and should stay whatever else is
added.

---

## 5. The scope boundary, stated rather than implied

"End-to-end drug discovery" reaches from target selection to a clinical candidate.
This project covers **hit finding through hit-to-lead**, for a *selectivity* objective,
on public data, CPU-only. Outside that boundary and deliberately not attempted:

- **target selection and validation** — genetic evidence, tractability, safety of
  modulating the target at all
- **ADMET and DMPK depth** — hERG, CYP inhibition, microsomal stability, permeability.
  (hERG is currently used as a gate *negative target* and not modelled as a liability,
  which is a genuine and cheap gap; it is the most defensible thing to add next after
  §3, and it is still not "E2E".)
- **synthesis** — synthetic accessibility, retrosynthesis, whether a proposed molecule
  can be made or bought
- **IP and freedom to operate**
- **in vivo PK/PD, formulation, tox packages, regulatory**

Declaring this is itself an improvement. A platform that quietly implies coverage it
does not have is the failure mode this repo replaced v1 to escape, one level up.

---

## 6. Open questions

1. **Same-document or same-assay matching for P4.1?** Document is the stronger control
   and the larger set (5 049 vs 2 820); assay description is finer but noisier text.
   Measure both before choosing.
2. **Do censored records get equal weight as negatives?** They are measured, but
   `> 10 µM` is a bound, not a value. Treating a bound as a point label is its own
   modelling error — censored regression is the principled answer and heavier.
3. **Does the JAK 8-molecule residual leak justify a retrain?** Retraining invalidates
   every contract. My reading is no, and that the number belongs in VALIDATION instead;
   worth an explicit decision rather than drift.
4. **Should `panel_data` ever read the store?** Only alongside a deliberate retrain.
   Until then the committed parquet is the training data and the store is evidence.
