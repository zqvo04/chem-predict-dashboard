# chem-predict-dashboard

An end-to-end, **CPU-only / zero-cost** drug-discovery screening pipeline:

> target protein → candidate molecules → drug-likeness filter → property prediction → ranked dashboard

This is a portfolio project. It runs on a laptop and free web hosting — no GPU,
no paid APIs. Because of that constraint, step 1 is **retrieval-based virtual
screening** (find known/nearby actives), not de-novo generation. See the design
notes below for the honest trade-offs.

The shipped v1 is a single-target screen; the active work turns it into a closed
**JAK-selectivity funnel** (see [Roadmap](#roadmap--jak-selectivity-screening-funnel)).

### Documentation

| Doc | What it covers |
|-----|----------------|
| **README** (this file) | overview, v1 usage, roadmap summary |
| [WORKFLOW.md](WORKFLOW.md) | the full funnel pipeline, stage by stage — data flow, schemas, module map |
| [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) | why each choice (regression + gap, conformal, AD, gates) + rejected alternatives |
| [VALIDATION.md](VALIDATION.md) | measured results, starting with the Gate 0 data audit |

## Status

| Phase | Scope | State |
|------|-------|-------|
| **1** | Target → candidate SMILES (ChEMBL retrieval) | ✅ done |
| **2** | Drug-likeness filtering (Lipinski Ro5, PAINS) | ✅ done |
| **3** | Per-target activity model (QSAR pchembl regression) | ✅ done |
| **3b** | Generic property models (solubility + toxicity, MoleculeNet) | ✅ done |
| **4** | Pipeline integration + composite scoring + PubChem expansion | ✅ done |
| **5** | Streamlit dashboard + deploy | ✅ done |

Phases 1–5 are the shipped **v1**: a single-target retrieval screen. The next
arc turns it into a closed **selectivity-aware funnel** — see
[Roadmap](#roadmap--jak-selectivity-screening-funnel) below. Roadmap items are
**planned, not built**; nothing there ships a metric yet.

## Phase 1 — target → candidate SMILES

Given a target name (e.g. `EGFR`), the client:

1. resolves it to a ChEMBL target id, preferring the **SINGLE PROTEIN** entry in
   the requested organism (ChEMBL's own relevance score otherwise ranks protein
   complexes / PPIs first);
2. fetches bioactivities with `pchembl_value >= 6` (~1 µM, the usual "active"
   cutoff) for IC50 / Ki / Kd / EC50, paginating the REST API;
3. collapses them to one row per molecule (best potency), and drops any SMILES
   RDKit cannot parse;
4. caches raw activity pages to `data/cache/*.parquet` so repeat runs are offline.

### Usage

```bash
pip install -r requirements.txt

# CLI
python -m src.data.chembl_client EGFR --top 10
python -m src.data.chembl_client CHEMBL203 --pchembl-gte 7 --max-records 1000
```

```python
# Library
from src.data.chembl_client import get_candidates
target, candidates = get_candidates("EGFR", max_records=500)
# candidates: DataFrame[molecule_chembl_id, canonical_smiles, pchembl_value,
#                       standard_type, n_activities]
```

### Tests

```bash
python -m pytest tests/         # unit tests offline; live smoke test self-skips
```

## Phase 2 — drug-likeness filtering

Adds RDKit descriptors and two standard gates to any candidate table:

- **Lipinski Rule of 5** — `mw ≤ 500, logp ≤ 5, hbd ≤ 5, hba ≤ 10`; at most one
  violation allowed (configurable).
- **PAINS** — pan-assay interference substructures; any match fails.

```python
from src.data.chembl_client import get_candidates
from src.filters.druglikeness import apply_druglikeness

target, candidates = get_candidates("EGFR", max_records=500)
filtered = apply_druglikeness(candidates)   # adds mw, logp, hbd, hba, tpsa,
                                            # ro5_violations, ro5_pass, pains_pass, druglike
keep = filtered[filtered["druglike"]]
```

On EGFR this keeps ~88% of retrieved actives; the rejects are mostly large,
lipophilic molecules failing two Ro5 criteria at once.

## Phase 3 — per-target activity model (QSAR)

Trains a RandomForest regressor that predicts **pchembl_value** (potency) from
2048-bit Morgan (ECFP4) fingerprints, on-the-fly for a target from ChEMBL data.

- one **median pchembl per molecule** over the full measured range (not just actives)
- **scaffold split** for evaluation, so the reported score reflects generalization
  to new chemotypes (a random split would inflate it)
- **data-sufficiency gate**: refuses to train below 50 usable molecules
- trained model cached to `data/models/<target>.pkl`

```bash
python -m src.models.target_model EGFR
# Train : 2592 molecules, pchembl range 4.00-11.00
# Eval  : scaffold-split test n=518  R2=0.557  RMSE=0.932
```

```python
from src.models.target_model import train_target_model
model = train_target_model("EGFR")
scores = model.predict(["COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC"])  # predicted pchembl
```

**Trade-off:** first-time training on a data-rich target (~2600 molecules) takes
~50 s on one CPU core. The result is cached, but for the free-hosted dashboard
we will pre-bake models for a few showcase targets and/or cap training size to
avoid request timeouts (Phase 5).

## Phase 3b — generic drug-property models (MoleculeNet)

Two static, target-independent models trained once on public MoleculeNet data
and shipped in `assets/models/property_models.pkl` (~3 MB):

- **Solubility** (ESOL, regression) → predicted logS. Uses Morgan fingerprints
  **plus RDKit descriptors** (LogP, TPSA, MW, …), which lifts scaffold-split
  R² from ~0.41 to **0.86** — solubility is driven by physicochemistry, not just
  substructure.
- **Toxicity** (Tox21, classification) → probability of a hit in any of the 12
  assays, a broad "toxicophore alert". Scaffold-split **ROC-AUC ≈ 0.75**.

```bash
python -m src.models.property_models   # re-train and refresh the bundle
```

**Honest note:** Tox21 assays are specific mechanisms (nuclear-receptor / stress
response), so the aggregate is a screening *alert*, not a safety verdict. ESOL is
only ~1100 molecules — a useful prior, not a lab measurement.

## Phase 4 — end-to-end pipeline + composite scoring

`src/pipeline.py` chains everything together and adds **PubChem similarity
expansion** so the activity model scores molecules it has never seen:

```
target -> known actives (P1) -> + novel PubChem analogues
       -> drug-likeness filter (P2) -> activity prediction (P3)
       -> composite score -> two ranked tracks
```

```bash
python -m src.pipeline EGFR --top 10
```

Key design decisions (and why):

- **Two tracks, not one list.** `chembl_known` rows are a *positive control*
  scored on their **measured** potency; `pubchem_novel` rows are the actual
  screening output scored on the **model prediction**. Mixing them would let
  measured 0.1 nM binders bury every prediction — correct, but useless as
  "discovery". Scoring knowns on truth also removes the memorization inflation
  that otherwise lets training molecules dominate.
- **Composite = 0.5·activity + 0.2·QED + 0.15·solubility + 0.15·(1 − tox risk)**,
  where `activity` is pchembl mapped to [0,1] on a fixed potency scale
  (comparable across targets), QED is RDKit's drug-likeness estimate, and
  solubility / tox come from the Phase 3b property models. Weights live in
  `src/pipeline.py` and are easy to retune.

On EGFR this yields ~1900 drug-like known actives (control) plus ~37 novel
drug-like candidates with predicted pchembl ≈ 7.5–9.0.

## Phase 5 — Streamlit dashboard

```bash
streamlit run app.py
```

Enter a target, and the app runs the full pipeline and shows two tabs — novel
candidates (the discovery) and known actives (the control) — each with molecule
structures, predicted/measured potency, QED, and the composite score, plus the
model's scaffold-split metrics.

The activity model is **HistGradientBoosting**, not RandomForest: it scored
slightly better (R² 0.573 vs 0.557) at ~1/35th the pickle size (1.5 MB vs 54 MB),
which is what makes shipping a model and running on a 1 GB free host practical.
EGFR ships pre-baked in `assets/models/` for an instant demo; other targets train
on first run (~30–40 s) and are cached.

### Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at
   `app.py` on this branch.
3. `requirements.txt` and `packages.txt` (system libs for RDKit drawing) are
   picked up automatically.

Honest deployment caveats: the free tier is 1 vCPU / ~1 GB RAM. Pre-baked targets
are instant; a cold target does a one-off ~30–40 s fetch-and-train (Streamlit's
spinner covers it, but a very data-rich target can approach request limits).
PubChem/ChEMBL calls need outbound network, which the hosted runtime allows.

## Roadmap — JAK-selectivity screening funnel

> **Status: planned.** This section describes the next arc, not shipped code.
> Every figure and metric it mentions is a target to be produced with a seed +
> script, never a placeholder to be filled in. Where a value is not yet
> measured, the build will say *pending*. The reasoning behind each choice below
> lives in [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md).

### Why v1 is not enough (the honest starting point)

v1 is a sound *engineering* skeleton with a hollow *scientific* value proposition.
Four gaps make its output untrustworthy as discovery, and the funnel exists to
close them — not to pile features on top:

- **"Novel" candidates aren't novel.** PubChem 2D-similarity expansion returns
  near-analogues of known actives — inside the training scaffold neighborhood — so
  the headline scaffold-split R² does not apply to them. Interpolation on
  near-duplicates dressed up as prediction.
- **No trust signal.** Every prediction is emitted with equal confidence; an
  in-domain estimate and a wild extrapolation look identical. No uncertainty, no
  applicability domain.
- **Unvalidated ranking.** The composite score's weights
  (`0.5·activity + …`) are validated against no endpoint. A confident-looking
  ranking with no evidence it enriches for anything.
- **Censored, biased label.** Regression trains only on *quantified* pchembl, so
  the model never sees a true inactive and cannot recognize a non-binder — fatal
  for a tool whose job is to reject bad molecules.

The funnel fixes the trust gaps by construction: selectivity is a genuine
unsolved problem (not table-stakes potency), its predictions are **checkable
against measured selectivity**, and they are useless without uncertainty + AD —
so the build is *forced* to add both.

**One-sentence claim (the finish line).** Turn the v1 single-target screen into a
closed **cost funnel**: run a large, diverse molecule library through cheap tiers
(rule filters → ligand-based selectivity scoring → conformal + applicability
domain) to a shortlist that is **selective *and* in-domain** (stage B, CPU,
deployed); let a human **pick** a few; run an offline GPU **deep dive**
(confirmatory structure-based docking + analogue generation) on just those (stage
A, Colab); then **re-score** everything through the *same* stage-B models — closing
the loop. *Cheap wide screen → expensive deep dive*, made mechanical.

Why JAK: JAK1/JAK2/JAK3 are highly similar kinases where **isoform selectivity**
is a genuine, clinically important, unsolved problem (off-target JAK inhibition
drives immunosuppression/toxicity), and ChEMBL has thousands of per-isoform
records — enough for per-target QSAR + selectivity + uncertainty + AD **without a GPU**.

### The core reframe: single potency → a validated selectivity gap

> **Note — this reframe was revised on data.** The plan first moved to
> active/inactive *classification*; the **Gate 0 data audit killed it** (the
> inactive class is ~75–333 molecules against ~10k actives — see
> [VALIDATION.md](VALIDATION.md)). The reframe is now regression-based.

v1's model keeps its shape — a per-isoform **pchembl regressor** (the data supports
it: ~10k molecules each) — but the *decision* changes from single-target potency to
a **selectivity gap** validated against measured data:

```
gap (wide):    S(JAK1) = pchembl_pred(JAK1) − max( pchembl_pred(JAK2), pchembl_pred(JAK3) )
direct (narrow): one regressor for the measured gap, on the cross-measured set
```

A **hybrid** matches the funnel: the cheap **difference-of-regressors** screens the
whole library; a **direct gap regressor** (trained on cross-measured molecules)
re-ranks and validates the survivors. Unlike v1's unvalidated composite, `S` is
checked against the *measured* gap (Spearman + ≥10×-selective enrichment). The
non-binder problem that motivated classification is instead carried by the
**applicability domain** (§AD). Full rationale in
[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md); end-to-end flow in [WORKFLOW.md](WORKFLOW.md).

### The funnel (cost tiers)

```
  WIDE LIBRARY (~10^5 diverse drug-like, target-agnostic)      [cheap, many]
        |
  Tier 0  Ro5 + PAINS rules                near-free      10^5 -> 10^5-
  Tier 1  regressors -> gap S              ms/molecule    10^5 -> 10^3
  Tier 2  conformal interval + AD (survivors) pricier/mol 10^3 -> 10^2   [B, CPU]
        |
  [SELECT] human picks a few                              10^2 -> few
        |   export loop_contract.json
        v
  Tier 3  DEEP DIVE  (docking + generation) expensive/mol few          [A, Colab GPU]
        |
        +-- re-score all through the SAME src B modules --> before/after report
            (loop closes: re-scored analogues re-enter Tier 0)
```

Ordering rule: the **expensive per-molecule operations (AD, conformal, docking)
run only after the cheap regressor prunes the library** — that is what makes the
funnel economics real rather than "score everything expensively".

The headline deliverable is that this loop runs **end-to-end on one real case**:
a molecule flows B → SELECT → A → re-score, with one report showing the shift in
the **selectivity gap `S`** and **applicability-domain** status **before vs after** —
reported as an *in-silico hypothesis requiring wet-lab validation*, never a hit.

### Design constraints carried over from v1

- **CPU-only / zero-cost** for the deployed stage B (~1 vCPU / ~1 GB). GPU lives
  **only** in the offline Colab notebook (stage A), never in the app.
- The scoring logic — selectivity gap, conformal intervals, applicability
  domain — lives in **shared `src` modules imported by both the app
  and the notebook**, never duplicated. That shared code is what makes "re-score
  through the same models" real rather than a reimplementation that silently diverges.
- Reuse the existing featurizer (ECFP4), scaffold split, and Trainer — the
  funnel is an **additive** extension, not a rewrite.

### Confirmed design decisions (summary)

Locked choices; rationale + rejected alternatives in
[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md), mechanics in [WORKFLOW.md](WORKFLOW.md).

- **Funnel = explicit cost tiers:** expensive per-molecule ops (AD, conformal, docking) run *only after* the cheap classifier prunes — the funnel economics, made mechanical.
- **Wide library:** a ~10^5 diverse, target-agnostic set (ZINC/PubChem), separate from the target's own actives — the honest "wide" top v1 lacked.
- **Task:** per-isoform **pchembl regression** (reverted from classification on the Gate 0 data audit — no inactive class exists).
- **Selectivity:** the validated **gap** `S`, **hybrid** — difference-of-regressors screens wide, direct gap regressor re-ranks/validates the survivors.
- **Uncertainty:** conformal regression → prediction intervals, 90 % coverage, empirically verified.
- **Applicability domain:** Tanimoto + leverage; propagates to the gap (worst-case) and carries the non-binder burden regression can't. AD-vs-wide is the mechanism, not a bug: apply broadly, trust only in-domain.
- **Deep dive:** confirmatory **docking** (orthogonal evidence, not an oracle) *then* **generation** (hypothesis-only, AD-filtered) — never a claimed hit.
- **Loop contract:** versioned JSON pinning model ids + α + code version.
- **Fallback:** Gate 0 measures cross-measured count; below threshold → pairwise selectivity.

### Staged build plan (credibility-first)

Reordered so the trust machinery (regressors → conformal → AD) is built and
validated *before* selectivity is stacked on it, and the wide-library + deep-dive
tiers come after the scoring core is proven. Each step ends with a numeric gate; no
step advances on a placeholder metric.

| Step | Goal | Adds / touches | Gate (done-when) |
|------|------|----------------|------------------|
| **0** | Data go/no-go | `chembl_client` (reuse) → JAK1/2/3 pull | ✅ **done** — 3624 cross-measured; regression + gap (classification killed) |
| **1** | Credibility pass | `scripts/reproduce.sh`, CI, pinned deps | 5-seed numbers reproduce; CI green |
| **2** | JAK data layer | 3 cached isoform datasets | per-isoform count + pchembl-distribution table |
| **3** | Per-isoform regressors | Trainer (reuse), scaffold split + ≥5 seeds | **MAE / RMSE / R² / Spearman** mean ± std |
| **4** | Selectivity gap (hybrid) | **new** `src/selectivity.py`, `src/loop_contract.py` | hero figure; predicted vs measured gap validated (Spearman + enrichment) |
| **5** | Conformal uncertainty | **new** `src/conformal.py` | empirical coverage 88–92 % @ 90 % nominal |
| **6** | Applicability domain | **new** `src/applicability.py` | OOD error > in-domain, margin significant (money plot) |
| **7** | Wide library + tiered dashboard + SELECT | **new** `src/data/library.py`, extend `app.py` | screen 10^5 → shortlist; export a valid contract file |
| **8** | Colab deep dive + loop closure | **new** `notebooks/deep_dive.ipynb`, `src/docking.py` | one worked case; docking + gen; before/after gap `S` + AD |
| **9** | Loop hardening + docs | integration test, VALIDATION.md | full checklist green; loop test passes |

### Definition of "done"

- [ ] No placeholder metrics anywhere; every number reproducible from a script + seed.
- [ ] Per-isoform JAK **regressors** trained & evaluated (scaffold split, ≥5 seeds, MAE/RMSE/R²/Spearman, mean ± std).
- [ ] **Hybrid** selectivity gap `S` implemented and **validated against the measured gap**; ranking-flip hero figure.
- [ ] Conformal prediction intervals with verified coverage; AD flags with the out-of-domain money plot.
- [ ] Wide library screened through the cost tiers to a **selective + in-domain** shortlist; dashboard shows rank + interval + AD badge and exports a chosen case.
- [ ] Colab deep dive runs **confirmatory docking + hypothesis-only generation** and re-scores through the same `src` models.
- [ ] **The loop:** one documented end-to-end case flows B → SELECT → A → re-score, before vs after in one report.
- [ ] README leads with the funnel + hero figures; VALIDATION.md and DESIGN_DECISIONS.md exist; tests + CI + reproduce.sh pass.

## Known limitations

- **No novelty.** Retrieval returns molecules already known to ChEMBL for the
  target. Generating truly novel structures needs generative models (GPU) and is
  out of scope for a zero-cost build.
- **Coverage varies.** Well-studied targets (EGFR, kinases) return hundreds of
  actives; niche targets may return few or none.
- **Runtime API dependency.** First fetch needs network to `ebi.ac.uk`; results
  are cached afterward.
- **Model applicability domain.** The QSAR regressor is only reliable for
  chemotypes near its training set. Since Phase 1 currently returns known
  molecules, the model's real value appears once Phase 4 brings in novel
  candidates via similarity expansion. Quantified-pchembl labels also skew away
  from true hard-negatives.
- **"Novel" is modest.** PubChem 2D-similarity expansion returns mostly close
  analogues of known actives that happen not to have a measured value in ChEMBL
  for this target — reasonable follow-up candidates, not de-novo scaffolds.
