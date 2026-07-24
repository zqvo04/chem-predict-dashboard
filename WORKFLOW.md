# Workflow & pipeline — JAK-selectivity screening funnel

The full end-to-end system, stage by stage: what data enters, what transforms
it, what comes out, and which module owns each step. This is the "what/how"
companion to [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) (the "why") and the
[README roadmap](README.md#how-the-funnel-was-built-step-by-step) (the
summary).

**Legend.** `[reuse]` = exists in v1, used as-is or with a small change.
`[new]` = added for the funnel. `[gate]` = a numeric done-when check. **Built and
validated** (STEP 0–9); measured results are in [VALIDATION.md](VALIDATION.md).

The system is an **explicit cost funnel**: each tier is cheaper per molecule than
the next and rejects most of what it sees, so the expensive tiers only ever run on
a handful of survivors. That economics — *cheap wide screen → expensive deep
dive* — is the whole point, and it dictates the tier order below.

---

## 0. The funnel at a glance

```
  WIDE LIBRARY (~10^5 diverse drug-like molecules, target-agnostic)   [cheap, many]
        │
  Tier 0  rule filters (Ro5 + PAINS)              near-free       10^5 → ~10^5-
        │
  Tier 1  per-isoform regressors → gap S           ms/molecule    10^5 → 10^3
        │   (ligand-based; S = pchembl(tgt) − max_off pchembl(off))
        │
  Tier 2  conformal interval + applicability domain pricier/mol    10^3 → 10^2
        │   run ONLY on Tier-1 survivors; keep in-domain + selective
        │
  [SELECT]  human picks a few cases (judgment)                     10^2 → few
        │   export loop_contract.json  (B → A handoff)
        ▼
  Tier 3  DEEP DIVE (offline Colab, GPU)            expensive/mol  few
        │   (a) confirmatory structure-based docking (orthogonal evidence)
        │   (b) conditional generation of analogues (in-silico hypotheses)
        │
        └── re-score everything through the SAME src B modules ──┐
                                                                 │
                          before/after: gap S + AD report        │
                                                                 │
        (loop closes: re-scored analogues can re-enter Tier 0) ──┘
```

Two execution surfaces, one shared scoring core:

- **Stage B** (Tiers 0–2, the wide screen) runs on CPU in the deployed Streamlit app.
- **Stage A** (Tier 3, the deep dive) runs on GPU only in an offline Colab notebook.
- **`src/selectivity.py`, `src/conformal.py`, `src/applicability.py`** are
  imported by *both*, so "re-score through the same models" is literally the same
  code — never a reimplementation that can silently diverge.

The critical ordering rule: **the expensive per-molecule operations (AD nearest-
neighbour, conformal, docking) run only after the cheap classifier has pruned the
library.** Running AD or docking on the full 10^5 library would defeat the funnel;
running them on Tier-1 survivors keeps their cost bounded.

---

## 1. Inputs

### 1.1 Wide screening library  `[new: src/data/library.py]`
The thing that makes the screen *wide*: a large, diverse, **target-agnostic** set
of purchasable/enumerated drug-like molecules — distinct from the target's own
ChEMBL actives. Default: a cached ~10^5 diverse drug-like subset (ZINC20 lead-like
or a PubChem random sample), source pluggable and demo-scale; the pipeline scales
to larger libraries in principle, bounded only by Tier-1 throughput. This is the
funnel's top — the "haystack" the cheap tiers search.

**Honest scope.** The library is unlabelled: the selectivity gap `S` on it is pure
prediction, and most of a diverse library sits **outside** the model's
applicability domain. That is not a bug — it is why Tier 2 exists. The wide screen
*applies* the model broadly; it is trusted only where AD says in-domain (see §5).

### 1.2 Training / validation data (ChEMBL)  `[reuse: src/data/chembl_client.py]`
Separate from the library: the per-isoform activity data that *trains and
validates* the models. See §2.

---

## 2. Data layer  `[reuse + new]`

**Goal.** Three clean, cached, labelled per-isoform datasets, plus the
cross-measured join that ground-truths selectivity.

### 2.1 Retrieval  `[reuse: src/data/chembl_client.py, cache.py]`
For each of JAK1 / JAK2 / JAK3 (TYK2 optional): `resolve_target` → `fetch_activities(pchembl_gte=None)`
→ all quantified pchembl records (IC50 / Ki / Kd / EC50), paginated, cached to
`data/cache/*.parquet`.

### 2.2 Deduplicate  `[reuse pattern]`
One row per (molecule, isoform) via **median** pchembl over replicates. Drop
unparseable SMILES; canonicalize.

### 2.3 Regression target
Each per-isoform table carries the **median pchembl** as the regression label
(continuous, range ~4–11). No active/inactive labelling — Gate 0 showed the
inactive class is nearly empty (75 / 333 / 245), so classification is not viable;
selectivity lives in the pchembl *gap*, not a class split (DESIGN_DECISIONS §1).

### 2.4 Cross-measured join  `[new]` `[gate: Gate 0 — DONE]`
Inner-join the three tables on canonical SMILES → the subset measured on **all
three** isoforms. Only set on which the selectivity gap can be **validated**.

**Gate 0 result (measured, see [VALIDATION.md](VALIDATION.md)):** 3-way
cross-measured = **3624** (healthy); the gap-based selective positives are ample
(593 JAK1-selective at ≥10×; 2073 in the JAK1–JAK2 pairwise view). 3-isoform
selectivity proceeds; the pairwise fallback stays available for a stronger
(≥100×) story.

**Artifacts.** `data/jak/{JAK1,JAK2,JAK3}.parquet`, `data/jak/cross_measured.parquet`,
a count + pchembl-distribution table.

---

## 3. Featurization  `[reuse: src/models/features.py]`

SMILES → canonical → **2048-bit ECFP4**. Identical featurizer for every isoform
and every stage — shared representation makes cross-isoform prediction and
stage-A re-scoring coherent.

---

## 4. Stage B, Tier 1 — cheap wide scoring (CPU)

### B1. Per-isoform regressors  `[new: src/models/isoform_regressor.py, reuses Trainer pattern]`
Per isoform, a **pchembl regressor** (HistGradientBoostingRegressor, the v1
approach) from ECFP4.
- Scaffold split `[reuse + new seed arg]`, ≥5 seeds, report **MAE / RMSE / R² /
  Spearman** mean ± std.
- **[gate: Gate 3]** metrics stable across seeds; no test leakage.

### B2. Selectivity gap — hybrid  `[new: src/selectivity.py]`
Two estimators, used at different tiers (DESIGN_DECISIONS §2):

- **Wide (Tier 1), difference-of-regressors** — uses all per-isoform data, scores
  any molecule:
  ```
  S(target) = pchembl_pred(target) − max_off pchembl_pred(off-isoform)
  ```
  Cheap, applied to the whole library. Ranked among candidates clearing a
  **target-potency floor** (default `pchembl_pred(target) ≥ 6`), and shown as the
  pair *(target potency, gap)* — never gap alone.
- **Narrow (Tier 2 re-rank / validation), direct gap regressor** — a single
  regressor trained on the cross-measured set to predict the measured gap directly.
  Avoids the stacked error of the difference form but is data-limited, so used only
  to **re-rank survivors** and **validate** the difference estimator.

**Validation** `[gate: Gate 4]`: on the cross-measured held-out scaffold split,
compare predicted `S` (both estimators) to the *measured* gap — **Spearman** +
**enrichment of ≥10×-selective molecules**. Basis for the hero figure.

Tier-1 output: the library ranked by `S` (above the potency floor); keep the top
band (e.g. 10^5 → 10^3). Cheap ops only so far.

---

## 5. Stage B, Tier 2 — trust filter on survivors (CPU)

Runs **only on Tier-1 survivors**, so its higher per-molecule cost is bounded.

### B3. Conformal prediction intervals  `[new: src/conformal.py]`
Split/inductive conformal regression → a prediction interval per molecule per
isoform at **90 %** nominal coverage; the gap `S` interval is propagated from the
two isoform intervals.
**[gate: Gate 5]** empirical coverage 88–92 % on the scaffold-split test set.

### B4. Applicability domain  `[new: src/applicability.py]`
Two orthogonal flags: **Tanimoto distance** to nearest training molecule +
**descriptor-space leverage**. Propagate to selectivity: `S` is **uncertain** if
any contributing isoform model is out-of-domain (worst-case). AD also carries the
non-binder burden that regression alone cannot (DESIGN_DECISIONS §1).
**[gate: Gate 6 — money plot]** out-of-domain error systematically higher than
in-domain, clear margin.

**Scalability.** Tanimoto-to-training is O(N_train) per query; on the full library
that is prohibitive, which is exactly why AD is deferred to Tier 2. For very large
Tier-1 survivor sets, use approximate nearest-neighbour (LSH / FAISS on packed
fingerprints) or a fixed diverse training reference subset.

Tier-2 output: survivors that are **selective *and* in-domain**, with prediction
intervals and AD verdicts attached (e.g. 10^3 → 10^2). This is the ranked shortlist
the dashboard shows.

### B5. Drug-likeness + property context  `[reuse: druglikeness.py, property_models.py]`
Ro5/PAINS run as **Tier 0** (near-free, before Tier 1) to drop gross liabilities
early; QED / solubility / tox priors are attached to the shortlist for display.

**Hero figure** `[new]`: 2-D scatter of `pchembl_pred(target)` vs gap `S` —
potent-but-non-selective molecules bottom-right, genuinely selective ones
top-right; highlight a rank flip between potency-only and selectivity-aware order.

---

## 6. SELECT — the B → A handoff  `[new: src/loop_contract.py]`

In the dashboard the user marks a few shortlist candidates and **exports** them as
one versioned JSON — the loop data contract, the only thing crossing from the CPU
app to the GPU notebook.

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
    "conformal_alpha": 0.10,
    "code_version": "<git sha>"
  },
  "molecules": [
    {
      "smiles": "<canonical>",
      "origin": "screen",                      // screen | generated
      "parent_smiles": null,                   // set for generated analogues
      "per_isoform": {
        "JAK1": { "pred_pchembl": 8.4, "interval": [7.6, 9.2], "in_domain": true,
                  "tanimoto_nn": 0.62, "leverage_ok": true },
        "JAK2": { "pred_pchembl": 6.7, "interval": [5.9, 7.5], "in_domain": true, "...": "" },
        "JAK3": { "pred_pchembl": 6.1, "interval": [5.3, 6.9], "in_domain": true, "...": "" }
      },
      "selectivity": {
        "gap": 1.7,                              // S = pred(JAK1) − max(pred(JAK2), pred(JAK3))
        "gap_interval": [0.5, 2.9],              // propagated from isoform intervals
        "meets_potency_floor": true,
        "verdict": "in_domain"                   // in_domain | uncertain
      },
      "deep_dive": null                          // filled by Tier 3 (docking, etc.)
    }
  ]
}
```

`model_ids` + `conformal_alpha` + `code_version` pin the exact models so Stage A
asserts identity before re-scoring.

---

## 7. Stage A, Tier 3 — the deep dive (offline Colab, GPU)  `[new: notebooks/deep_dive.ipynb]`

Runs on the handful of selected molecules only. Two expensive analyses; both feed
the same-model re-score that closes the loop.

1. **Load** `loop_contract.json`; **import the SAME `src` modules** and assert
   `model_ids` / `code_version` match.

2. **(a) Confirmatory docking** `[new: src/docking.py wrapper]` — dock each
   selected molecule into published JAK1/2/3 structures (AutoDock Vina; CPU-capable,
   GPU-accelerated in Colab). This is *orthogonal, structure-based* evidence
   against the ligand-based ML score — **not** an affinity ground truth: docking
   scores correlate weakly with potency, so they are used for **consensus /
   corroboration and pose inspection**, never as a second oracle. A molecule the
   ML calls selective *and* that docks better into the target than the off-isoforms
   is a stronger hypothesis; disagreement is flagged, not hidden. Honest scope
   stated in the report.

3. **(b) Conditional generation** `[new]` — generate analogues over the chosen
   scaffold toward higher gap `S` (reuse a Molecule-Generator if present, else a
   compact conditional generator; GPU here only). Validity-filter with RDKit. Every
   generated molecule is an **in-silico hypothesis**, AD-filtered, never a hit.

4. **Re-score** every selected + generated molecule through the identical B1–B4
   pipeline → per-isoform `pred_pchembl` + interval, gap `S`, AD flags. Attach
   docking consensus to `deep_dive`.

5. **Emit** a `loop_contract.json` (`stage: "A_rescore"`, `origin: "generated"`,
   `parent_smiles` linking analogues to their case).

6. **Report** `[new]`: before-vs-after gap `S` distribution + AD status + docking
   consensus, `loop_before_after.png`, and a per-case markdown ending *"in-silico
   hypothesis — requires wet-lab validation."*

**[gate: Gate 8]** one real exported case flows B → SELECT → A → re-score with a
single before/after report; re-scoring uses identical `src`. **Loop closed.**

---

## 8. Loop closure

Re-scored analogues are ordinary contract molecules, so they re-enter Tier 0 and
run back down the funnel — a cycle, not a one-way street. Honest deliverable: a
**distribution shift** (population gap `S` up while in-domain), never a claimed hit.

---

## 9. Module map

| Module | Status | Role |
|--------|--------|------|
| `src/data/chembl_client.py` | reuse | resolve target, fetch activities |
| `src/data/cache.py` | reuse | parquet cache |
| `src/data/pubchem_client.py` | reuse | similarity expansion (optional, near-analogues) |
| `src/data/jak.py` | **new** | per-isoform regression datasets + cross-measured join (STEP 2) |
| `src/data/library.py` | **new** | load/cache the wide screening library |
| `src/models/features.py` | reuse | ECFP4 featurization (shared) |
| `src/models/scaffold_split.py` | reuse + seed arg | scaffold split |
| `src/models/property_models.py` | reuse | solubility / tox priors |
| `src/filters/druglikeness.py` | reuse | Tier 0 Ro5 + PAINS |
| `src/funnel.py` | **new** | tiered wide screen (Tier 0-2) + SELECT-to-contract |
| `src/pipeline.py` | reuse | v1 single-target screen (unchanged) |
| `src/models/isoform_regressor.py` | **new** | per-isoform pchembl regressor |
| `src/selectivity.py` | **new** | hybrid gap `S`; shared |
| `src/conformal.py` | **new** | conformal prediction intervals; shared |
| `src/applicability.py` | **new** | Tanimoto + leverage AD; shared |
| `src/generate.py` | **new** | CPU analogue generation (scaffold decoration) |
| `src/deep_dive.py` | **new** | Stage-A loop closure: generate + re-score + report |
| `src/docking.py` | future | optional Tier-3 docking seam (documented in the notebook) |
| `src/loop_contract.py` | **new** | JSON contract IO + model-pin assert |
| `app.py` | extend | tiered screen, AD badge, SELECT/export |
| `notebooks/deep_dive.ipynb` | **new** | Tier 3: docking + generation + re-score |
| `scripts/reproduce.sh` | **new** | regenerate headline numbers/figures |
| `.github/workflows/ci.yml` | **new** | run tests on push |

---

## 10. Artifacts

- **Datasets:** `data/jak/{isoform}.parquet`, `cross_measured.parquet`, `data/library/*.parquet`
- **Models:** `data/models/jak/{isoform}_reg.pkl`, direct gap regressor, conformal calibrators
- **Figures:** `selectivity_ranking_flip.png` (hero), `conformal_coverage.png`,
  `applicability_error.png` (money plot), `loop_before_after.png`
- **Contract:** `loop_contract.json` (B_export / A_rescore)
- **Docs:** `VALIDATION.md` (measured numbers, seeds, splits, "where this fails")

---

## 11. Build order (credibility-first)

Trust machinery (regressors → conformal → AD) is built and gated *before*
selectivity is stacked on it; the wide-library and deep-dive tiers come after the
scoring core is validated. Full step table with gates:
[README roadmap](README.md#staged-build-plan-credibility-first).

```
Gate 0  data go/no-go  [DONE — 3624 cross-measured, gap signal ample]
  ─▶ Phase I  STEP1 credibility ─▶ STEP3 regressors ─▶ STEP5 conformal ─▶ STEP6 AD
  ─▶ Phase II STEP4 selectivity (hybrid) + hero + validation
  ─▶ Phase III STEP7 wide library + tiered dashboard + SELECT
              ─▶ STEP8 Colab deep dive (docking + generation) + loop closure
  ─▶ STEP9 loop test + VALIDATION.md
```
