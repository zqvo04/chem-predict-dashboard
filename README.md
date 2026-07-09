# chem-predict-dashboard

An end-to-end, **CPU-only / zero-cost** drug-discovery screening pipeline:

> target protein → candidate molecules → drug-likeness filter → property prediction → ranked dashboard

This is a portfolio project. It runs on a laptop and free web hosting — no GPU,
no paid APIs. Because of that constraint, step 1 is **retrieval-based virtual
screening** (find known/nearby actives), not de-novo generation. See the design
notes below for the honest trade-offs.

## Status

| Phase | Scope | State |
|------|-------|-------|
| **1** | Target → candidate SMILES (ChEMBL retrieval) | ✅ done |
| **2** | Drug-likeness filtering (Lipinski Ro5, PAINS) | ✅ done |
| **3** | Per-target activity model (QSAR pchembl regression) | ✅ done |
| **4** | Pipeline integration + composite scoring + PubChem expansion | ✅ done |
| **5** | Streamlit dashboard + deploy | ✅ done |

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
- **Composite = 0.7 · activity + 0.3 · QED**, where `activity` is pchembl mapped
  to [0,1] on a fixed potency scale (comparable across targets) and QED is
  RDKit's standard drug-likeness estimate.

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
