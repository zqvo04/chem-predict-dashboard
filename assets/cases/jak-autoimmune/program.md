# Program: JAK1-selective inhibition for autoimmune indications

**Case id:** `jak-autoimmune` · **Panel:** `jak` · **Signed:** 2026-08-09 · **Status:** active

## Objective

`S = pred(JAK1) − max(pred(JAK2), pred(JAK3))`

## Why this sign

The sign is set by the indication, not by the family. In autoimmune disease JAK2
inhibition is to be avoided — JAK2 carries EPO and TPO signalling, so hitting it
produces anaemia and thrombocytopenia. In myeloproliferative neoplasms the same
isoform is the target (JAK2 V617F). The same panel therefore supports two programs
with opposite objectives, which is what `tests/test_wiring.py` checks.

## Anti-target evidence

Registration requires evidence of class 1–3 (approved-drug label, mechanism, or
human loss-of-function genetics). Structural similarity alone is class 4 and does
not register a target — it registers a watch item (G10).

| Off-target | Class | Evidence | Status |
|---|---|---|---|
| JAK2 | 1 + 2 | Approved-drug labels carry anaemia and thrombocytopenia; mechanism via EPO/TPO receptor signalling | **registered** |
| JAK3 | 4 only | Structural similarity within the family. No safety-label or genetic evidence has been recorded in this repo | **watch item — see gap 7b** |

## Recorded gaps

- **7a** — this card is the first place in the repo that names an indication. The
  deployed objective assumed one without saying so.
- **7b** — JAK3 is penalised by the deployed objective on class-4 evidence only.
  Tofacitinib hits JAK3 and is approved. STATE.md §4d measures JAK3 as the member
  paralog transfer reaches worst (unseen Spearman 0.302), which is consistent with
  its SAR being the most distinct in the family but is not safety evidence.
- **7c** — TYK2 is absent from the panel. The JAK family has four members; the
  selectivity calculation does not see a quarter of it.

**None of the three is fixed here.** Changing the panel changes every deployed
model id and breaks G0, so the card records them and the change is scheduled with a
retrain.

## Wiring check, as measured

Deployed models, 2026-08-09. `tests/test_wiring.py` pins the two rank orders.

| drug | pred JAK1 | pred JAK2 | pred JAK3 | autoimmune gap | MPN gap |
|---|---:|---:|---:|---:|---:|
| upadacitinib | 7.394 | 7.057 | 7.055 | +0.337 | −0.337 |
| fedratinib | 7.804 | 8.283 | 6.346 | −0.479 | +0.479 |
| ruxolitinib | 8.420 | 8.377 | 6.763 | +0.043 | −0.043 |

Autoimmune order: upadacitinib > ruxolitinib > fedratinib. MPN order: the exact
reverse. Ruxolitinib is a JAK1/2 dual inhibitor and lands in the middle under both,
which is why it is in the check — it makes dual inhibitors a test case rather than
an exception.

**All three drugs are in the training data,** so these predictions are recall. The
check tests that the indication drives the sign, not that the ranking is skilful.
For all three, JAK3 is the weakest isoform, so `max(off)` picks the same isoform
under both objectives and the two gaps are exactly antisymmetric — the claim is the
pair of rank orders, nothing more.

## Provenance

SMILES resolved from PubChem by name and standardised with `src.standardize`
(2026-08-09); `assets/evidence/molecule.parquet` carries `mol_chembl_id` null for
all 75,870 rows, so ChEMBL-id resolution is not available offline.
