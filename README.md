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
| 4 | Pipeline integration + composite scoring | planned |
| 5 | Streamlit dashboard + deploy | planned |

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
