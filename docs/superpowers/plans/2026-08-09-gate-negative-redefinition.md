# B — Binder Gate Negative Redefinition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether putting *measured* non-binders into the gate's negative class fixes its 65 % false-pass rate on real JAK-programme non-binders, in a controlled A/B where the negative class is the only thing that differs.

**Architecture:** No new model plumbing. The measured-negative population moves into `panel_data` beside the other evidence-derived datasets and is sealed to a committed file with a held-out fold. One audit script assembles two gates from the same primitives — presumed-only versus presumed-plus-measured — trains both with the existing `_fit`, and scores both on identical held-out sets. Nothing writes a deployed asset.

**Tech Stack:** Python 3.11, pandas + pyarrow, RDKit, scikit-learn, pytest. Offline, CPU, committed assets only.

## Global Constraints

- **G0** — `assets/models/jak/binder_gate.pkl` and every other deployed asset are immutable. Both gates in this plan are built in-process and thrown away. Deployment is a **separate decision after the measurement**, not part of this plan.
- **G6** — the sealed 414 (`assets/jak/sealed_negatives.parquet`) enter neither gate's training. They are a third, wholly untouched check.
- **Held-out discipline** — a slice of the measured negatives and the eval fold of the sealed time split are excluded from both gates' training. Scoring a gate on molecules it trained on measures recall.
- **One variable** — both gates get the identical positive set and the identical presumed-negative set. Only the measured negatives differ. Any second change makes the comparison uninterpretable.
- **Two arms** — every reported number comes with its cost. A gate that rejects more non-binders by rejecting everything is not an improvement, and the positive arm is what catches that.
- Run `python -m pytest tests/ -q`. Baseline entering this plan: **214 passed**.

---

## What was measured before this plan

`STATE.md` §2 reported the deployed gate passing 36.0 % of the sealed 414. Probing the wider censored population on 2026-08-09 gave a sharper and worse picture:

| group | n | pass at threshold 0.544 | median P(binder) |
|---|---:|---:|---:|
| **measured negatives, never trained on** | 1,720 | **65.3 %** | **0.921** |
| sealed 414 | 414 | 36.0 % | 0.245 |
| gate positives (training) | 2,000 | 99.8 % | 0.999 |
| presumed negatives (training) | 2,000 | 0.4 % | 0.007 |

Threshold sweep, measured-negative pass / positives kept: 0.54 → 65.4 % / 99.8 %; 0.90 → 51.9 % / 98.5 %; 0.99 → 34.6 % / 85.6 %.

**Three conclusions this plan is built on.**

1. **Threshold recalibration is falsified.** Pushing to 0.99 still lets a third of measured non-binders through and costs 14 % of actives. The two populations are not separated by the gate's score.
2. **The gate learned the wrong task.** Presumed negatives sit at median 0.007 and measured negatives at 0.921. The presumed negatives are actives of *other* targets — EGFR, CDK2, thrombin. The measured negatives are JAK-programme chemistry that somebody actually assayed and found weak. The gate learned "does this look like a JAK inhibitor", which is exactly what a measured non-binder from a JAK paper also looks like.
3. **The 36 % was optimistic.** The 414 come from the wide library, which is drawn from other targets' actives, so they are off-target chemistry. On real JAK-programme non-binders the false-pass rate is 65 %.

`STATE.md` §4b already measured the gate as saturated on data volume — ROC-AUC 0.993 at a tenth of its data. So this is not "add more negatives"; 1,720 is 14 % of the existing 12,144 and volume is not the lever. It is "the negative class is drawn from the wrong distribution", and the experiment is whether adding a small, hard, correctly-distributed slice changes the decision the gate makes.

**Measured populations** (2026-08-09):

| | n |
|---|---:|
| molecules with a JAK censored record | 4,644 |
| ↳ with no pchembl anywhere in the panel (true non-binders) | 2,134 |
| ↳ minus the sealed 414 → **usable** | **1,720** |
| of those, already in the presumed-negative basket | 75 |
| presumed negatives | 12,144 |
| gate positives | 15,453 |
| ↳ in the sealed split's train fold | 11,789 |

---

## File Structure

| File | Responsibility |
|---|---|
| `src/data/panel_data.py` (modify) | gains `measured_negatives(panel)` — the evidence-derived population, beside `censored_library_molecules` |
| `scripts/seal_measured_negatives.py` (create) | scaffold-splits the population into train/eval once and refuses to overwrite |
| `assets/jak/measured_negatives.parquet` (create, committed) | the sealed population with its fold |
| `scripts/gate_ab_audit.py` (create) | the controlled A/B, both arms, three scored populations |
| `tests/test_measured_negatives.py` (create) | the population's invariants: sealed-414 disjoint, folds disjoint, scaffold-disjoint |
| `scripts/reproduce.sh`, `STATE.md`, `VALIDATION.md` (modify) | the record |

---

### Task 1: The measured-negative population, sealed

**Files:**
- Modify: `src/data/panel_data.py` (add `measured_negatives` after `censored_library_molecules`)
- Create: `scripts/seal_measured_negatives.py`
- Create: `tests/test_measured_negatives.py`
- Create (committed artifact): `assets/jak/measured_negatives.parquet`

**Interfaces:**
- Consumes: `panel_data.censored_library_molecules(panel)` (for the sealed 414 to exclude), `models.scaffold_split.scaffold_split(smiles, test_frac, seed)`.
- Produces: `panel_data.measured_negatives(panel) -> pd.DataFrame` with columns `inchikey`, `smi`; `panel_data.load_measured_negatives(panel) -> pd.DataFrame` with the added column `fold` ∈ `{"train", "eval"}`.

**Why a scaffold split and not a time split.** The eval fold of the sealed time split is defined by publication year, and these molecules carry no pchembl, so most of them are not in it. What the held-out slice has to answer is "does the retrained gate reject non-binders it has not seen the chemotype of", and that is a scaffold question.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_measured_negatives.py
"""The measured-negative population and the discipline around it.

These are molecules somebody assayed against JAK and found weak — a censored
record, no pchembl anywhere in the panel. They are the negative class the gate
has never had. The invariants below are what keeps the experiment honest: the
sealed 414 stay out, the two folds do not overlap, and no scaffold crosses
between them, so a gate trained on the train fold has not seen the eval fold's
chemotypes.
"""
import pandas as pd

from src.data import panel_data
from src.models.scaffold_split import _scaffold
from src.panels import JAK


def test_population_excludes_the_sealed_414():
    """G6: the falsification audit's negative arm is not training data for anything."""
    population = panel_data.measured_negatives(JAK)
    sealed = set(panel_data.censored_library_molecules(JAK)["inchikey"])
    assert set(population["inchikey"]).isdisjoint(sealed)
    assert len(population) > 1000            # measured 1,720 on 2026-08-09


def test_population_has_no_quantified_molecule():
    """A molecule with a pchembl anywhere in the panel is training data, not a negative."""
    population = set(panel_data.measured_negatives(JAK)["smi"])
    for iso in JAK.isoforms:
        assert population.isdisjoint(set(panel_data.build_isoform_dataset(JAK, iso)["smi"]))


def test_folds_are_disjoint_and_scaffold_disjoint():
    split = panel_data.load_measured_negatives(JAK)
    train = split[split["fold"] == "train"]
    evalf = split[split["fold"] == "eval"]
    assert set(train["smi"]).isdisjoint(set(evalf["smi"]))
    assert len(evalf) > 200
    train_scaffolds = {_scaffold(s) for s in train["smi"]}
    eval_scaffolds = {_scaffold(s) for s in evalf["smi"]}
    assert train_scaffolds.isdisjoint(eval_scaffolds)


def test_the_split_is_read_from_disk_not_recomputed():
    a = panel_data.load_measured_negatives(JAK)
    b = panel_data.load_measured_negatives(JAK)
    pd.testing.assert_frame_equal(a, b)
    assert (JAK.data_bundled / "measured_negatives.parquet").exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_measured_negatives.py -q`
Expected: FAIL, `AttributeError: module 'src.data.panel_data' has no attribute 'measured_negatives'`

- [ ] **Step 3: Add the two functions to `src/data/panel_data.py`**

Insert immediately after `censored_library_molecules`:

```python
def measured_negatives(panel: PanelSpec) -> pd.DataFrame:
    """Molecules assayed against the panel and found weak; columns inchikey, smi.

    A censored record ("IC50 > x") and no pchembl anywhere in the panel. These are
    the negative class the binder gate has never had: its presumed negatives are
    actives of *other* targets, so it learned to recognise panel chemistry rather
    than binding, and on 2026-08-09 it passed 65.3 % of these at a median
    P(binder) of 0.921.

    The sealed 414 (`censored_library_molecules`) are removed — they are the
    falsification audit's negative arm and must stay out of every training set (G6).
    What is left is the censored population outside the wide library, which is
    mostly panel-programme chemistry rather than off-target chemistry.
    """
    root = panel.root / "assets" / "evidence"
    act = pd.read_parquet(root / "activity.parquet")
    molecule = pd.read_parquet(root / "molecule.parquet").dropna(subset=["parent_smiles"])

    rows = act[act["target_chembl_id"].isin(panel.chembl_ids.values())]
    quantified = set(rows.loc[rows["pchembl_value"].notna(), "inchikey"])
    censored = set(rows.loc[rows["standard_relation"].isin(CENSORED_RELATIONS), "inchikey"])
    sealed = set(censored_library_molecules(panel)["inchikey"])

    usable = (censored - quantified) - sealed
    out = (molecule.loc[molecule["inchikey"].isin(usable), ["inchikey", "parent_smiles"]]
           .rename(columns={"parent_smiles": "smi"})
           .drop_duplicates("inchikey")
           .sort_values("inchikey")
           .reset_index(drop=True))
    return out


def load_measured_negatives(panel: PanelSpec) -> pd.DataFrame:
    """The sealed measured-negative split; columns inchikey, smi, fold.

    Read-only. `scripts/seal_measured_negatives.py` writes it once, because a
    held-out set that is recomputed per run is not held out — the evidence store
    grows and the evaluation moves with it.
    """
    path = _cached(panel, "measured_negatives.parquet")
    if path is None:
        raise FileNotFoundError(
            f"No sealed measured-negative split for panel {panel.name!r}. "
            "Run: python scripts/seal_measured_negatives.py " + panel.name)
    return pd.read_parquet(path)
```

- [ ] **Step 4: Write `scripts/seal_measured_negatives.py`**

```python
#!/usr/bin/env python3
"""Seal the measured-negative train/eval split.

Writes once and refuses to overwrite. A held-out set that moves with the evidence
store is not held out, and the whole point of this population is to answer
"does the retrained gate reject non-binders whose chemotype it has not seen".

The split is scaffold-based, not temporal: these molecules carry no pchembl, so
most of them are absent from the sealed time split, and the question is about
chemotype rather than publication date.

    python scripts/seal_measured_negatives.py [panel]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import panel_data                                     # noqa: E402
from src.models.scaffold_split import scaffold_split                # noqa: E402
from src.panels import DEFAULT_PANEL, get_panel                     # noqa: E402

TEST_FRAC = 0.25
SEED = 0


def main() -> None:
    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    path = panel.data_bundled / "measured_negatives.parquet"
    if path.exists():
        raise SystemExit(f"{path} already exists — the split is sealed. "
                         "Delete it deliberately if you mean to reseal.")

    population = panel_data.measured_negatives(panel)
    smiles = population["smi"].tolist()
    _, eval_idx = scaffold_split(smiles, test_frac=TEST_FRAC, seed=SEED)
    population["fold"] = "train"
    population.loc[list(eval_idx), "fold"] = "eval"

    path.parent.mkdir(parents=True, exist_ok=True)
    population.to_parquet(path, index=False)
    counts = population["fold"].value_counts()
    print(f"Sealed {len(population)} measured negatives: "
          f"train {counts.get('train', 0)}, eval {counts.get('eval', 0)} -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Seal and run the tests**

```bash
python scripts/seal_measured_negatives.py jak
python -m pytest tests/test_measured_negatives.py -q
```
Expected: the script prints train/eval counts summing to ~1,720; 4 tests PASS.

If `test_population_has_no_quantified_molecule` fails, the SMILES in the evidence store and the SMILES in the per-isoform parquets are not canonicalised the same way. Do not loosen the assertion — compare on `inchikey` instead by joining the isoform datasets through `molecule.parquet`, because a molecule leaking from the positive class into the negative class is the one error this whole plan cannot survive.

- [ ] **Step 6: Commit**

```bash
git add src/data/panel_data.py scripts/seal_measured_negatives.py \
        tests/test_measured_negatives.py assets/jak/measured_negatives.parquet
git commit -m "Seal the measured-negative population the gate has never trained on"
```

---

### Task 2: The controlled A/B

**Files:**
- Create: `scripts/gate_ab_audit.py`

**Interfaces:**
- Consumes: `panel_data.load_measured_negatives(panel)`, `panel_data.load_eval_split(panel)`, `panel_data.censored_library_molecules(panel)`, `data.negatives.build_negatives(panel)`, `data.negatives.positive_smiles(panel)`, `models.binder_gate._fit(X, y)`, `models.features.morgan_matrix`.
- Produces: `gate_ab_audit.build_gate(positives, negatives) -> model`, `gate_ab_audit.report(model, threshold, groups) -> dict`.

**What must be identical between the two arms.** Positive set, presumed-negative set, fingerprints, estimator, seed, threshold rule. Only the measured negatives differ. If a second thing moves the comparison says nothing.

**Why `_fit` and not `train_and_cache`.** `train_and_cache` runs a five-seed scaffold-split evaluation and caches to `panel.model_cache`. Neither is wanted: the ROC-AUC it reports is on the presumed-negative task, which is exactly the task that measured 0.998 while the gate was passing 65 % of real non-binders, and writing a cache would put an experimental gate where the deployed one is loaded from.

**Threshold.** Both gates use Youden's J computed on the *same* held-out positive and presumed-negative material, so neither is tuned on the measured negatives it is about to be judged on. Report the sweep as well, because a single operating point hides whether the two curves differ or merely sit at different places on the same curve.

- [ ] **Step 1: Write `scripts/gate_ab_audit.py`**

```python
#!/usr/bin/env python3
"""B: does a measured negative class change what the binder gate decides?

Two gates, one difference. Both get the same positives, the same presumed
negatives, the same fingerprints, the same estimator and the same threshold rule.
The candidate additionally gets the train fold of the measured negatives.

  BASELINE   presumed negatives only            — what ships today
  CANDIDATE  presumed + measured[train]         — the negative class redefined

Both are scored on three populations none of them trained on:

  measured[eval]   held-out measured non-binders  — the thing being fixed
  positives[eval]  held-out actives               — the cost of fixing it
  sealed 414       the falsification audit's arm  — untouched by either gate (G6)

The middle one is not optional. A gate that rejects every measured negative by
rejecting everything is not an improvement, and only the positive arm shows it.
STATE.md sections 2a and 11 are the same lesson twice.

Nothing here writes a model. Whether to deploy is a separate decision, because it
changes a committed asset and every model_id under it (G0).

    python scripts/gate_ab_audit.py            # ~5 minutes
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import panel_data                                     # noqa: E402
from src.data.negatives import build_negatives, positive_smiles     # noqa: E402
from src.models.binder_gate import _fit                             # noqa: E402
from src.models.features import morgan_matrix                       # noqa: E402
from src.panels import DEFAULT_PANEL                                # noqa: E402

SWEEP = (0.5, 0.7, 0.9, 0.95, 0.99)


def build_gate(positives: list[str], negatives: list[str]):
    """Fit a gate from explicit class members; nothing is cached."""
    smiles = list(positives) + list(negatives)
    y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    X, mask = morgan_matrix(smiles)
    return _fit(X, y[mask])


def probabilities(model, smiles: list[str]) -> np.ndarray:
    """P(binder) for each parseable SMILES."""
    X, mask = morgan_matrix(list(smiles))
    out = np.full(len(mask), np.nan)
    if X.shape[0]:
        out[mask] = model.predict_proba(X)[:, 1]
    return out[~np.isnan(out)]


def youden_threshold(model, positives: list[str], negatives: list[str]) -> float:
    """Youden's J on held-out positives vs held-out presumed negatives.

    Deliberately not computed on the measured negatives: a threshold tuned on the
    population it is then judged against is not a measurement.
    """
    p_pos, p_neg = probabilities(model, positives), probabilities(model, negatives)
    y = np.concatenate([np.ones(len(p_pos)), np.zeros(len(p_neg))])
    fpr, tpr, thresholds = roc_curve(y, np.concatenate([p_pos, p_neg]))
    return float(thresholds[np.argmax(tpr - fpr)])


def main() -> None:
    panel = DEFAULT_PANEL

    split = panel_data.load_eval_split(panel)
    train_smiles = set(split.loc[split["fold"] == "train", "smi"])
    eval_smiles = set(split.loc[split["fold"] == "eval", "smi"])

    all_positives = positive_smiles(panel)
    pos_train = sorted(all_positives & train_smiles)
    pos_eval = sorted(all_positives & eval_smiles)

    presumed = build_negatives(panel)["smi"].tolist()
    # 75 of the measured negatives are already in the presumed basket; keeping the
    # union deduplicated stops one molecule counting twice in the candidate.
    holdout_presumed = presumed[: len(presumed) // 5]
    train_presumed = presumed[len(presumed) // 5:]

    measured = panel_data.load_measured_negatives(panel)
    meas_train = measured.loc[measured["fold"] == "train", "smi"].tolist()
    meas_eval = measured.loc[measured["fold"] == "eval", "smi"].tolist()
    sealed = sorted(set(panel_data.censored_library_molecules(panel)["smi"]))

    print(f"positives  train {len(pos_train)}  held-out {len(pos_eval)}")
    print(f"presumed   train {len(train_presumed)}  held-out {len(holdout_presumed)}")
    print(f"measured   train {len(meas_train)}  held-out {len(meas_eval)}")
    print(f"sealed 414 {len(sealed)} (in neither gate)\n")

    arms = {
        "BASELINE  (presumed only)": train_presumed,
        "CANDIDATE (presumed + measured[train])":
            sorted(set(train_presumed) | set(meas_train)),
    }

    for name, negatives in arms.items():
        model = build_gate(pos_train, negatives)
        threshold = youden_threshold(model, pos_eval, holdout_presumed)
        groups = {
            "measured[eval]  (want LOW)": meas_eval,
            "sealed 414      (want LOW)": sealed,
            "presumed[eval]  (want LOW)": holdout_presumed,
            "positives[eval] (want HIGH)": pos_eval,
        }
        print("=" * 72)
        print(f"{name}   negatives n={len(negatives)}   threshold {threshold:.3f}")
        print("=" * 72)
        print(f"  {'group':30} {'n':>6} {'pass%':>7} {'median P':>10}")
        for label, smis in groups.items():
            p = probabilities(model, smis)
            print(f"  {label:30} {len(p):6d} {100 * (p >= threshold).mean():6.1f}% "
                  f"{np.median(p):10.3f}")
        print(f"\n  {'threshold':>10} {'measured[eval] pass':>21} {'positives kept':>16}")
        p_meas, p_pos = probabilities(model, meas_eval), probabilities(model, pos_eval)
        for t in SWEEP:
            print(f"  {t:10.2f} {100 * (p_meas >= t).mean():20.1f}% "
                  f"{100 * (p_pos >= t).mean():15.1f}%")
        print()

    print("How to read this. The candidate wins only if measured[eval] pass falls")
    print("AND positives[eval] holds. Compare at matched positive recall, not at each")
    print("gate's own Youden point — two gates at different operating points on the")
    print("same curve is not a difference in what they learned.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/gate_ab_audit.py`
Expected: two blocks with four scored groups each and a threshold sweep. Runtime is dominated by two gate fits on ~25k molecules, roughly two minutes each.

**Record whatever comes out.** If the candidate does not beat the baseline, that is the result — `STATE.md` §4b predicted the gate's headroom is distributional, and this is the test of that prediction, not a confirmation exercise.

- [ ] **Step 3: Sanity-check the comparison before believing it**

Run: `python scripts/gate_ab_audit.py | grep -E "positives\[eval\]|negatives n="`
The two `positives[eval]` rows must have identical `n`. If they differ, the positive set moved between arms and the comparison is void.

- [ ] **Step 4: Commit**

```bash
git add scripts/gate_ab_audit.py
git commit -m "Add the controlled A/B for the gate's negative class"
```

---

### Task 3: Record the result and decide

**Files:**
- Modify: `STATE.md` (new §2b), `VALIDATION.md` (new section), `scripts/reproduce.sh`

- [ ] **Step 1: Add `STATE.md` §2b**

Insert after §2a. It must contain, in this order:

1. The pre-experiment probe table (measured negatives 65.3 % pass at median 0.921; sealed 414 at 36.0 %; presumed negatives at 0.007) and the sentence that the 36 % was optimistic because the 414 are off-target chemistry.
2. The threshold sweep showing recalibration falsified.
3. The A/B table: for each gate, pass rate on measured[eval], sealed 414, presumed[eval], positives[eval].
4. The comparison **at matched positive recall**, not at each gate's own Youden point.
5. A plain sentence saying whether the candidate won, lost, or was indistinguishable — with no softening if it lost.

- [ ] **Step 2: Add the same to `VALIDATION.md`** with a "### Reproduce" block containing `python scripts/gate_ab_audit.py`, and the honest-limits paragraph: the measured negatives are 1,720 against 12,144 presumed, the split is scaffold-based on one seed, and neither gate here is the deployed one.

- [ ] **Step 3: Add to `scripts/reproduce.sh`**, before the DMTA block:

```bash
echo
echo "== Binder gate A/B, negative class (STATE.md section 2b) =="
echo "Presumed-only versus presumed-plus-measured negatives, identical positives,"
echo "scored on held-out measured non-binders and held-out actives."
python scripts/gate_ab_audit.py
```

- [ ] **Step 4: Update the reproduction table and the DMTA readiness table**

In `STATE.md`'s 재현 상태 표 add `| §2b | 게이트 음성 A/B | ✅ scripts/gate_ab_audit.py |`. In §0a, the **T** row's "남은 것" column should say what the A/B settled about the gate.

- [ ] **Step 5: Run the full suite and commit**

```bash
python -m pytest tests/ -q
git add -A
git commit -m "Record what redefining the gate's negative class did"
```

- [ ] **Step 6: State the deployment decision explicitly — do not act on it**

Whatever the outcome, write one paragraph in `STATE.md` §2b naming what shipping the candidate would cost: retraining the deployed gate changes `assets/models/jak/binder_gate.pkl`, which moves the gate threshold, which changes `screen_library`'s Tier 0.5 output, which changes the deployed shortlist and every number in `VALIDATION.md` downstream of it. That is a G0 event and belongs to its own plan.

---

## Out of scope, and why

| | |
|---|---|
| Deploying the candidate gate | Changes a committed asset and cascades through every downstream number (G0). Separate plan, after the measurement. |
| Weighted training (`sample_weight` on measured negatives) | Introduces a free hyperparameter, and ROADMAP §8 fixes the tuning protocol as closed until A2. If the unweighted candidate fails, the weight is the next experiment with that protocol applied. |
| Replacing the presumed negatives entirely | 15,453 positives against 1,720 negatives is a 9:1 imbalance, and the presumed negatives still do the job they were built for — catching ethanol and gross off-target chemistry. Test the addition before testing the removal. |
| PI3K | `STATE.md` §6d measured 39 censored library molecules against JAK's 491; the population census has to be run per panel before the design transfers. |

---

## Self-Review

**Spec coverage.** ROADMAP B is "게이트 재설계 (측정 음성 투입, 414 봉인)" — measured negatives injected, 414 sealed. Task 1 builds and seals the population with the 414 excluded and a test pinning it. Task 2 injects them in a controlled comparison. Task 3 records it. N8 ("검열은 게이트만") is satisfied structurally: these molecules reach the gate's negative class and nothing else, and they carry no pchembl so no regressor can consume them.

**Placeholder scan.** No TBDs. Task 2 Step 2 and Task 3 Step 1 deliberately do not state expected numbers — the experiment is designed to be able to fail, and pre-writing the result invites fitting the report to it. Task 1 Step 5 carries a specific contingency with a specific fix rather than "handle errors".

**Type consistency.** `panel_data.measured_negatives(panel) -> DataFrame[inchikey, smi]` is defined in Task 1 and consumed by `seal_measured_negatives.py` in the same task. `panel_data.load_measured_negatives(panel) -> DataFrame[inchikey, smi, fold]` is defined in Task 1 and consumed by `gate_ab_audit.main` in Task 2. `CENSORED_RELATIONS` already exists in `panel_data` (added with `censored_library_molecules`) and Task 1's function reuses it rather than redeclaring. `build_gate(positives, negatives)`, `probabilities(model, smiles)` and `youden_threshold(model, positives, negatives)` are defined and used within Task 2 only.

**One risk.** `positive_smiles(panel)` returns molecules at pchembl ≥ 6 across all isoforms, and `build_isoform_dataset` SMILES are canonicalised by `standardize`, while the evidence store's `parent_smiles` went through the same function at ingest — so the set operations in Task 2 should align. Task 1 Step 5 checks exactly this and says what to do if they do not.
