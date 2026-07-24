# Workflow & pipeline — JAK-selectivity screening funnel

The full end-to-end system, stage by stage: what data enters, what transforms
it, what comes out, and which module owns each step. This is the "what/how"
companion to [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) (the "why") and the
[README roadmap](README.md#roadmap--jak-selectivity-screening-funnel) (the
summary).

**Legend.** `[reuse]` = exists in v1, used as-is or with a small change.
`[new]` = to be built. `[gate]` = a numeric done-when check (see
DESIGN_DECISIONS.md). Nothing here is trained or measured yet — this is the
execution blueprint.

---

## 0. The loop at a glance

```
                          ┌───────────────────────────────────────────────┐
                          │                                               │
                          ▼                                               │
  ChEMBL ──▶ [DATA LAYER] ──▶ [FEATURIZE] ──▶ [STAGE B: WIDE SCREEN, CPU] │
             per-isoform         ECFP4          B1 per-isoform classifiers │
             active/inactive                    B2 P(selective)           │
                                                B3 conformal set          │
                                                B4 applicability domain   │
                                                    │                      │
                                     ranked selective candidates          │
                                                    │                      │
                              [SELECT] human picks a few cases (judgment)  │
                                                    │                      │
                                       export loop_contract.json           │
                                                    ▼                      │
                          [STAGE A: DEEP DIVE, offline Colab GPU]          │
                              generate analogues → higher P(selective)     │
                              (in-silico hypotheses, not hits)             │
                                                    │                      │
                              re-score via the SAME src B modules ─────────┘
                                                    │
                              before/after: P(selective) + AD report
```

Two execution surfaces, one shared scoring core:

- **Stage B** (screening) runs on CPU in the deployed Streamlit app.
- **Stage A** (generation) runs on GPU only in an offline Colab notebook.
- **`src/selectivity.py`, `src/conformal.py`, `src/applicability.py`** are
  imported by *both*, so "re-score through the same models" is literally the same
  code — never a reimplementation that can silently diverge.

---

## 1. Data layer  `[reuse + new]`

**Goal.** Three clean, cached, labelled per-isoform datasets, plus the
cross-measured join that ground-truths selectivity.

### 1.1 Retrieval  `[reuse: src/data/chembl_client.py]`
For each of JAK1 / JAK2 / JAK3 (TYK2 optional):
1. `resolve_target(name)` → ChEMBL target id (SINGLE PROTEIN, human).
2. `fetch_activities(id, pchembl_gte=None)` → all quantified pchembl records
   (IC50 / Ki / Kd / EC50), paginated, `[reuse: src/data/cache.py]` cached to
   `data/cache/*.parquet` for offline re-runs.

Output per isoform: raw activity rows
`(molecule_chembl_id, canonical_smiles, standard_type, standard_value, pchembl_value)`.

### 1.2 Deduplicate  `[reuse pattern]`
Collapse to **one row per (molecule, isoform)** using the **median** pchembl over
replicate / multi-assay measurements (matches v1 `build_training_set`; median
over max because it is the honest central estimate for a label). Drop
RDKit-unparseable SMILES; canonicalize for cross-isoform joins.

### 1.3 Label assignment  `[new: src/labels.py]`
Turn median pchembl into a class per (molecule, isoform):

| pchembl | label | used in training? |
|---------|-------|-------------------|
| ≥ 6 | **active** (1) | yes |
| ≤ 5 (incl. right-censored `>` values) | **inactive** (0) | yes |
| 5 – 6 (gray zone) | ambiguous | **dropped from training labels** |

Thresholds are the defaults from DESIGN_DECISIONS §1; both gray-zone-in and
gray-zone-out results get reported.

### 1.4 Cross-measured join  `[new]` `[gate: Gate 0]`
Inner-join the three labelled tables on canonical SMILES → the subset measured on
**all three** isoforms. This is the only set on which predicted selectivity can be
**validated** against measured selectivity.

**Gate 0 (go/no-go).** Count the 3-way cross-measured molecules.
- Above threshold → proceed with 3-isoform selectivity.
- Below → **pivot to pairwise** (target over the single off-isoform with the most
  co-measured molecules, e.g. JAK1 over JAK2). Decided *after* seeing real counts.

**Artifacts.** `data/jak/{JAK1,JAK2,JAK3}.parquet` (labelled),
`data/jak/cross_measured.parquet`, and a printed count + pchembl-distribution
summary table.

---

## 2. Featurization  `[reuse: src/models/features.py]`

SMILES → canonical → **2048-bit ECFP4** (Morgan radius 2). Identical featurizer
for every isoform and every stage — this shared representation is what makes
cross-isoform prediction and stage-A re-scoring coherent. Property models
additionally append RDKit descriptors `[reuse: property_models._augmented_matrix]`.

---

## 3. Stage B — the wide screen (CPU, deployed)

### B1. Per-isoform classifiers  `[new: src/models/isoform_classifier.py, reuses Trainer pattern]`
For each isoform, a **binary classifier** (HistGradientBoostingClassifier, the
existing property-model pattern) predicting **P(active)** from ECFP4.

- **Split:** scaffold split `[reuse: src/models/scaffold_split.py + new seed arg]`.
- **Seeds:** ≥5, report **PR-AUC + calibration (Brier / reliability)** as mean ± std.
- **Calibration:** if reliability is poor, Platt/isotonic on a held-out fold.
- **[gate: Gate 3]** PR-AUC + calibration stable across seeds; calibrated
  probabilities (this matters — B2 multiplies them).

Artifacts: `data/models/jak/{isoform}_clf.pkl` + a metrics table.

### B2. Selectivity probability  `[new: src/selectivity.py]`
Combine per-isoform probabilities for a chosen **target** isoform:

```
P(selective | target) = P(active | target) · Π_{off}  P(inactive | off)
                       = P(active | target) · Π_{off} (1 − P(active | off))
```

Pairwise fallback: `P(active | target) · (1 − P(active | off))` for the single
off-isoform. No separate potency floor is needed — the `P(active | target)` factor
already suppresses molecules that are inactive everywhere (DESIGN_DECISIONS §2).

**Validation** `[gate: Gate 4]`: on the cross-measured held-out scaffold split,
compare predicted `P(selective)` to the *measured* selective label
(PR-AUC / enrichment). This is what the hero figure rests on.

### B3. Conformal prediction sets  `[new: src/conformal.py]`
Split/inductive conformal classification (Mondrian per class or APS) on a
held-out calibration fold → a **prediction set** per molecule per isoform
(`{active}`, `{inactive}`, or ambiguous `{active, inactive}`) at **90 %** nominal
coverage.
**[gate: Gate 5]** empirical coverage 88–92 % on the scaffold-split test set;
coverage-vs-nominal plot.

### B4. Applicability domain  `[new: src/applicability.py]`
Two orthogonal flags per prediction:
1. **Tanimoto distance** to nearest training molecule (fingerprint neighborhood).
2. **Descriptor-space leverage** (extrapolation in physicochemical space).

Propagate to selectivity: `P(selective)` is flagged **uncertain** if *any*
contributing isoform model is out-of-domain (worst-case, because it is a product).
**[gate: Gate 6 — the money plot]** out-of-domain error must be systematically
higher than in-domain error with a clear margin, else AD is decorative.

### B5. Candidate pool & ranking  `[reuse: pipeline.py, druglikeness.py, pubchem_client.py]`
The molecules being screened come from:
- ChEMBL known actives for the target isoform (positive control), and/or
- PubChem 2D-similarity expansion `[reuse]` (labelled clearly as near-analogues,
  not de-novo — this is the honest scope from v1's limitations).

Each candidate is drug-likeness filtered `[reuse: apply_druglikeness]`, then
carries: `P(active)` per isoform, `P(selective)`, conformal set, AD flags,
QED / solubility / tox `[reuse: property_models]`. Ranked by `P(selective)` among
in-domain candidates.

**Hero figure** `[new]`: 2-D scatter of `P(active | target)` vs `P(selective)` —
potent-but-non-selective molecules cluster bottom-right, genuinely selective ones
top-right; highlight a molecule whose rank flips between potency-only and
selectivity-aware ordering.

---

## 4. SELECT — the B → A handoff  `[new: src/loop_contract.py]`

In the dashboard the user marks one or a few candidates and **exports** them as a
single versioned JSON — the loop data contract. This file is the only thing that
crosses from the CPU app to the GPU notebook.

```jsonc
{
  "schema_version": "1.0",
  "case_id": "JAK1-selective-<hash>",
  "target_isoform": "JAK1",
  "off_isoforms": ["JAK2", "JAK3"],          // or ["JAK2"] in pairwise mode
  "provenance": {
    "created": "<iso8601>",
    "stage": "B_export",                      // B_export | A_rescore
    "model_ids": { "JAK1": "CHEMBL2835@<sha>", "JAK2": "...", "JAK3": "..." },
    "conformal_alpha": 0.10,                   // 90% coverage
    "code_version": "<git sha>"
  },
  "molecules": [
    {
      "smiles": "<canonical>",
      "origin": "screen",                      // screen | generated
      "parent_smiles": null,                   // set for generated analogues
      "per_isoform": {
        "JAK1": { "p_active": 0.91, "pred_set": ["active"], "in_domain": true,
                  "tanimoto_nn": 0.62, "leverage_ok": true },
        "JAK2": { "p_active": 0.12, "pred_set": ["inactive"], "in_domain": true, "...": "" },
        "JAK3": { "p_active": 0.08, "pred_set": ["inactive"], "in_domain": true, "...": "" }
      },
      "selectivity": {
        "p_selective": 0.73,
        "verdict": "in_domain",                // in_domain | uncertain
        "pred_set_selective": ["selective"]
      }
    }
  ]
}
```

`model_ids` + `conformal_alpha` + `code_version` are the pins that let stage A
assert it is scoring with the identical models before it re-scores anything.

---

## 5. Stage A — the deep dive (offline Colab, GPU)  `[new: notebooks/deep_dive.ipynb]`

1. **Load** the exported `loop_contract.json`.
2. **Import the SAME `src` modules** (`selectivity`, `conformal`, `applicability`,
   `isoform_classifier`) and assert `model_ids` / `code_version` match.
3. **Generate** analogues over the chosen scaffold toward higher `P(selective)`
   (reuse an existing Molecule-Generator if present, else a compact conditional
   generator; GPU allowed *here only*). Validity-filter with RDKit.
4. **Re-score** every generated analogue through the identical B1–B4 pipeline →
   `P(active)` per isoform, `P(selective)`, conformal set, AD flags.
5. **Emit** a `loop_contract.json` with `stage: "A_rescore"`, `origin: "generated"`,
   and `parent_smiles` linking each analogue to its source case.
6. **Report** `[new]`: before-vs-after of the `P(selective)` distribution and AD
   status, `loop_before_after.png`, and a per-case markdown ending with
   *"in-silico hypothesis — requires wet-lab validation."*

**[gate: Gate 8]** one real exported case flows B → SELECT → A → re-score with a
single before/after report. Generated molecules valid; re-scoring uses identical
`src` (no reimplementation). **Loop closed.**

---

## 6. Loop closure

Re-scored analogues are ordinary contract molecules, so they can re-enter Stage B
(re-screened, re-ranked) exactly like screened candidates — the funnel is a cycle,
not a one-way street. The honest deliverable is a **distribution shift**
(population `P(selective)` moves up while staying in-domain), never a claimed hit.

---

## 7. Module map

| Module | Status | Role |
|--------|--------|------|
| `src/data/chembl_client.py` | reuse | resolve target, fetch activities |
| `src/data/cache.py` | reuse | parquet cache (offline re-runs) |
| `src/data/pubchem_client.py` | reuse | similarity expansion (near-analogues) |
| `src/models/features.py` | reuse | ECFP4 featurization (shared) |
| `src/models/scaffold_split.py` | reuse + seed arg | scaffold split for honest eval |
| `src/models/property_models.py` | reuse | generic solubility / tox priors |
| `src/filters/druglikeness.py` | reuse | Ro5 + PAINS filter |
| `src/pipeline.py` | extend | add `screen_selectivity()` alongside `screen()` |
| `src/labels.py` | **new** | pchembl → active / inactive / gray-zone |
| `src/models/isoform_classifier.py` | **new** | per-isoform P(active) classifier |
| `src/selectivity.py` | **new** | P(selective); shared by app + notebook |
| `src/conformal.py` | **new** | conformal prediction sets; shared |
| `src/applicability.py` | **new** | Tanimoto + leverage AD; shared |
| `src/loop_contract.py` | **new** | JSON contract read/write + model-pin assert |
| `app.py` | extend | selectivity tab, AD badge, SELECT/export |
| `notebooks/deep_dive.ipynb` | **new** | stage A generation + re-score + report |
| `scripts/reproduce.sh` | **new** | regenerate all headline numbers/figures |
| `.github/workflows/ci.yml` | **new** | run tests on push |

---

## 8. Artifacts produced

- **Datasets:** `data/jak/{isoform}.parquet`, `data/jak/cross_measured.parquet`
- **Models:** `data/models/jak/{isoform}_clf.pkl`, conformal calibration objects
- **Figures:** `selectivity_ranking_flip.png` (hero), `conformal_coverage.png`,
  `applicability_error.png` (money plot), `loop_before_after.png`
- **Contract:** `loop_contract.json` (B_export and A_rescore variants)
- **Docs:** `VALIDATION.md` (measured numbers, seeds, splits, "where this fails")

---

## 9. Build order (credibility-first)

The trust machinery (classifiers → conformal → AD) is built and gated *before*
selectivity is stacked on it, so nothing rests on an unvalidated layer. Full step
table with gates: [README roadmap](README.md#staged-build-plan-credibility-first).

```
Gate 0  data go/no-go ─▶ Phase I  STEP1 credibility ─▶ STEP3 classifiers
                                     ─▶ STEP5 conformal ─▶ STEP6 AD
        ─▶ Phase II  STEP4 selectivity + hero + validation
        ─▶ Phase III STEP7 dashboard+SELECT ─▶ STEP8 Colab + loop closure
        ─▶ STEP9 loop test + VALIDATION.md
```
