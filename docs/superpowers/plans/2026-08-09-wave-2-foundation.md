# Wave 2 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the floor the DMTA loop stands on — a sealed time-split evaluation set, a label ledger that cannot leak sealed molecules into training, a round log that survives concurrency, a per-panel suitability screen, and a signed Program card whose objective sign is checked against approved drugs.

**Architecture:** Every piece here is a small addition to a module that already owns the concern — the evaluation split joins `src/data/panel_data.py` beside the other per-panel datasets, the ledger is one new `src/labels.py`, and the round fixes are surgical edits inside `src/registry.py`. Nothing retrains a deployed model, so `G0` holds throughout and the committed JAK assets do not move.

**Tech Stack:** Python 3.11, pandas + pyarrow (parquet), RDKit, scikit-learn, pytest. All offline against committed `assets/`.

## Global Constraints

- **G0** — the deployed JAK assets are immutable. No task retrains a model or rewrites `assets/models/jak/*` or `assets/jak/{JAK1,JAK2,JAK3}.parquet`.
- **G6 / G9** — the 414 sealed non-binders (`assets/jak/sealed_negatives.parquet`) and the sealed evaluation split never enter training or tuning. Enforced by test, not by comment.
- **D9** — case state is committed files. No hosted database, no new service.
- **Offline** — every test and every script added here runs with no network, from committed assets.
- **Style** — match the surrounding module: module docstring stating what the thing is *for*, comments explaining *why* not *what*, no new dependencies.
- Run the full suite with `python -m pytest tests/ -q`. Baseline before this plan: **181 passed**.

---

## Why this plan differs from the Wave 2 in ROADMAP.md

Wave 1 measured four things that changed the design. Each is recorded in `STATE.md` with a reproduction script.

| Measurement | Consequence for Wave 2 |
|---|---|
| §4c — on a scaffold split a Tanimoto 1-NN lookup scores 0.782 against the model's 0.779; under a 2020 time cut the lookup falls to 0.572 and the model holds 0.734 | **Task 1 seals a time split, not a scaffold split.** Sealing a scaffold split would score every future round on the protocol where a lookup ties the model. |
| §4b — a 70-molecule round moves JAK1 Spearman by +0.00008 against a seed spread of ±0.0072, and the gate's ROC-AUC by +0.00000 | **Task 2's ledger is a scoring ledger first.** `eligible_for_training` stays in the schema but is no longer the point; `sealed` and `scored_in_round` are. |
| §2a — the potency floor keeps 99.4 % of post-cut actives and 80.7 % of measured non-binders | **Task 6 is copy, not surgery.** The floor is correctly signed and nearly inert, so it stays. |
| §4d — paralog transfer medians 0.472 on unseen molecules with a 0.220–0.810 spread | **Task 4 reports transfer strength per member**, because a panel whose members do not inform each other is a different kind of panel. |

---

## File Structure

| File | Responsibility |
|---|---|
| `src/data/panel_data.py` (modify) | gains `load_eval_split(panel)` — the sealed time split, beside the other per-panel datasets it already owns |
| `scripts/seal_eval_split.py` (create) | writes `assets/<panel>/eval_split.parquet` once; refuses to overwrite |
| `assets/jak/eval_split.parquet` (create, committed) | the sealed split itself — the artifact G9 is about |
| `src/labels.py` (create) | append-only label ledger: arrival, `answer_status`, `sealed`, `scored_in_round` |
| `src/registry.py` (modify) | collision-free round indices, `kind` validation, and the docstring that currently promises a guarantee the code does not provide |
| `scripts/suitability_screen.py` (create) | Stage 0.5 — per-panel census: paralogs, cross-measured, censored population, library leakage, transfer strength |
| `assets/cases/jak-autoimmune/` (create, committed) | `program.md` + `panel.json` — the signed Program card |
| `tests/test_eval_split.py`, `tests/test_labels.py`, `tests/test_wiring.py` (create) | the gates for tasks 1, 2 and 5 |
| `tests/test_registry.py`, `tests/test_production_path.py` (modify) | concurrency gate; suitability screen smoke |

---

### Task 1: Seal the time-split evaluation set (N11 / G9)

**Files:**
- Create: `scripts/seal_eval_split.py`
- Modify: `src/data/panel_data.py` (add `load_eval_split` after `build_cross_measured`)
- Create: `tests/test_eval_split.py`
- Create (committed artifact): `assets/jak/eval_split.parquet`

**Interfaces:**
- Consumes: `funnel_falsification_audit._year_first()` — returns a `pd.Series` indexed by standardised SMILES, values are `document_year` floats. Move it into `src/data/panel_data.py` as `year_first(panel)` in this task and have the audit script import it from there, so two modules do not derive the same provenance differently.
- Produces: `panel_data.year_first(panel) -> pd.Series`, `panel_data.load_eval_split(panel) -> pd.DataFrame` with columns `smi`, `year_first`, `fold` where `fold` is `"train"` or `"eval"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_split.py
"""The sealed evaluation split (N11 / G9).

STATE.md section 4c measured that a Tanimoto lookup ties the model on a scaffold
split and loses badly under a publication-year cut. The split sealed at round 0 is
therefore the time split, and these tests pin the properties every later round
depends on: the folds do not overlap, no eval molecule predates the cut, and the
file on disk is the authority rather than something recomputed per run.
"""
import pandas as pd
import pytest

from src.data import panel_data
from src.panels import JAK

CUT = 2020


def test_eval_split_folds_are_disjoint_and_dated_correctly():
    split = panel_data.load_eval_split(JAK)
    train = split[split["fold"] == "train"]
    evalf = split[split["fold"] == "eval"]
    assert set(train["smi"]).isdisjoint(set(evalf["smi"]))
    assert (train["year_first"] <= CUT).all()
    assert (evalf["year_first"] > CUT).all()
    assert len(evalf) > 500          # a split too small to score is not a split


def test_eval_split_is_read_from_disk_not_recomputed():
    a = panel_data.load_eval_split(JAK)
    b = panel_data.load_eval_split(JAK)
    pd.testing.assert_frame_equal(a, b)
    assert (JAK.data_bundled / "eval_split.parquet").exists()


def test_year_first_covers_almost_every_training_molecule():
    year = panel_data.year_first(JAK)
    data = panel_data.build_isoform_dataset(JAK, "JAK1")
    assert data["smi"].map(year).notna().mean() > 0.99
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_eval_split.py -q`
Expected: FAIL with `AttributeError: module 'src.data.panel_data' has no attribute 'load_eval_split'`

- [ ] **Step 3: Add `year_first` and `load_eval_split` to `src/data/panel_data.py`**

Append after `build_cross_measured`:

```python
EVAL_TIME_CUT = 2020        # STATE.md section 4c: the cut the model beats a lookup on


def year_first(panel: PanelSpec) -> pd.Series:
    """smi -> earliest year any panel member published a quantified measurement.

    The committed per-isoform parquets carry only (smi, pchembl, n_meas): the
    `year_first` column `_collapse` produces never made it into the bundle. The
    evidence store holds the same provenance offline, so provenance-dated work
    reads it from there rather than forcing a network rebuild.
    """
    root = panel.root / "assets" / "evidence"
    act = pd.read_parquet(root / "activity.parquet")
    molecule = pd.read_parquet(root / "molecule.parquet")
    rows = act[act["target_chembl_id"].isin(panel.chembl_ids.values())
               & act["pchembl_value"].notna()]
    year = rows.groupby("inchikey")["document_year"].min()
    joined = molecule[["inchikey", "parent_smiles"]].join(year, on="inchikey")
    return joined.dropna(subset=["document_year"]).set_index("parent_smiles")["document_year"]


def load_eval_split(panel: PanelSpec) -> pd.DataFrame:
    """The evaluation split sealed at round 0 (G9); columns smi, year_first, fold.

    Read-only on purpose. `scripts/seal_eval_split.py` writes it once and refuses to
    overwrite, because a split that can be recomputed is not sealed — the training
    set grows from round to round and the split must not move with it.
    """
    path = _cached(panel, "eval_split.parquet")
    if path is None:
        raise FileNotFoundError(
            f"No sealed evaluation split for panel {panel.name!r}. "
            "Run: python scripts/seal_eval_split.py " + panel.name)
    return pd.read_parquet(path)
```

- [ ] **Step 4: Write `scripts/seal_eval_split.py`**

```python
#!/usr/bin/env python3
"""Seal the round-0 evaluation split (N11 / G9).

Writes once and refuses to overwrite. The training set grows every round; the
split it is measured against must not, or the rounds are not comparable to each
other and a regression can hide as a changed denominator.

The cut is temporal, not scaffold-based. STATE.md section 4c measured a Tanimoto
1-NN lookup at 0.782 against the model's 0.779 on a scaffold split and at 0.572
against 0.734 under this cut — a scaffold split scores every round on the protocol
where a lookup ties the model.

    python scripts/seal_eval_split.py [panel]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import panel_data                                     # noqa: E402
from src.panels import DEFAULT_PANEL, get_panel                     # noqa: E402


def main() -> None:
    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    path = panel.data_bundled / "eval_split.parquet"
    if path.exists():
        raise SystemExit(f"{path} already exists — the split is sealed. "
                         "Delete it deliberately if you mean to reseal.")

    year = panel_data.year_first(panel)
    molecules = pd.concat(
        [panel_data.build_isoform_dataset(panel, iso)["smi"] for iso in panel.isoforms]
    ).drop_duplicates()
    frame = pd.DataFrame({"smi": molecules})
    frame["year_first"] = frame["smi"].map(year)
    frame = frame.dropna(subset=["year_first"]).reset_index(drop=True)
    frame["fold"] = ["eval" if y > panel_data.EVAL_TIME_CUT else "train"
                     for y in frame["year_first"]]

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    counts = frame["fold"].value_counts()
    print(f"Sealed {len(frame)} molecules at cut {panel_data.EVAL_TIME_CUT}: "
          f"train {counts.get('train', 0)}, eval {counts.get('eval', 0)} -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Seal the split and run the tests**

```bash
python scripts/seal_eval_split.py jak
python -m pytest tests/test_eval_split.py -q
```
Expected: the script prints train/eval counts; 3 tests PASS.

- [ ] **Step 6: Point the falsification audit at the shared helper**

In `scripts/funnel_falsification_audit.py`, delete the local `_year_first` and replace its use in `positive_arm` with `panel_data.year_first(DEFAULT_PANEL)`. In `scripts/nn_baseline_audit.py`, change the import `from funnel_falsification_audit import TIME_CUT, _year_first` to `from src.data.panel_data import EVAL_TIME_CUT as TIME_CUT, year_first as _year_first`.

- [ ] **Step 7: Verify the audits still reproduce their recorded numbers**

```bash
python scripts/funnel_falsification_audit.py | head -12
```
Expected: falsification rate still `385 / 414 = 93.0%`, gate `36.0%`, floor `80.7%`.

- [ ] **Step 8: Commit**

```bash
git add src/data/panel_data.py scripts/seal_eval_split.py tests/test_eval_split.py \
        assets/jak/eval_split.parquet scripts/funnel_falsification_audit.py \
        scripts/nn_baseline_audit.py
git commit -m "Seal the round-0 evaluation split on the time cut, not the scaffold split"
```

---

### Task 2: Label ledger with the sealed invariant (D0 / E / N4)

**Files:**
- Create: `src/labels.py`
- Create: `tests/test_labels.py`

**Interfaces:**
- Consumes: `panel_data.load_eval_split(panel)` from Task 1; `assets/jak/sealed_negatives.parquet` written by `scripts/funnel_falsification_audit.py`.
- Produces: `labels.append(panel, smi, isoform, value, relation, source, answer_status, root=None) -> dict`, `labels.read(panel, root=None) -> pd.DataFrame`, `labels.training_rows(panel, root=None) -> pd.DataFrame`, `labels.sealed_smiles(panel) -> set[str]`.

**Design note the implementer needs:** ROADMAP framed this ledger as the feed for retraining. STATE.md §4b measured that a round's worth of molecules cannot be distinguished from a reseed on either axis, so the ledger's live job is **scoring** — recording that an answer arrived, what it said, and which round it was scored against. `training_rows` still exists and still filters, because the invariant it enforces (G6) is what stops the evaluation base being consumed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labels.py
"""The label ledger and the one invariant it exists to hold (G6).

A measurement is scored against a registered prediction or it is training data,
never both. The sealed molecules — the 414 measured non-binders and everything in
the sealed evaluation split — are the evaluation base, so a training reader that
returns even one of them has destroyed the thing every later round is measured on.
"""
import pandas as pd
import pytest

from src import labels
from src.panels import JAK


def test_append_and_read_roundtrip(tmp_path):
    labels.append(JAK, smi="CCO", isoform="JAK1", value=5.2, relation="=",
                  source="test", answer_status="answered", root=tmp_path)
    frame = labels.read(JAK, root=tmp_path)
    assert len(frame) == 1
    assert frame.loc[0, "smi"] == "CCO"
    assert frame.loc[0, "answer_status"] == "answered"
    assert frame.loc[0, "sealed"] is False or frame.loc[0, "sealed"] == False


def test_sealed_molecules_never_reach_training_rows(tmp_path):
    sealed = next(iter(labels.sealed_smiles(JAK)))
    labels.append(JAK, smi=sealed, isoform="JAK1", value=5.0, relation=">",
                  source="test", answer_status="answered", root=tmp_path)
    labels.append(JAK, smi="CCO", isoform="JAK1", value=5.2, relation="=",
                  source="test", answer_status="answered", root=tmp_path)
    train = labels.training_rows(JAK, root=tmp_path)
    assert set(train["smi"]) == {"CCO"}


def test_unanswered_rows_are_not_training_data(tmp_path):
    labels.append(JAK, smi="CCO", isoform="JAK1", value=None, relation=None,
                  source="test", answer_status="no_answer", root=tmp_path)
    labels.append(JAK, smi="CCC", isoform="JAK1", value=None, relation=None,
                  source="test", answer_status="failed", root=tmp_path)
    assert labels.training_rows(JAK, root=tmp_path).empty


def test_append_rejects_an_unknown_answer_status(tmp_path):
    with pytest.raises(ValueError):
        labels.append(JAK, smi="CCO", isoform="JAK1", value=1.0, relation="=",
                      source="test", answer_status="maybe", root=tmp_path)


def test_sealed_set_covers_both_populations():
    sealed = labels.sealed_smiles(JAK)
    assert len(sealed) > 1000      # 414 non-binders + the eval fold
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_labels.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.labels'`

- [ ] **Step 3: Write `src/labels.py`**

```python
"""The label ledger: what measurement arrived, and what it may be used for.

One append-only JSONL per panel. Append-only because the ledger's value is that it
records *when* something was known — a row that can be edited cannot support the
claim that a prediction preceded its answer.

**The invariant.** A measurement is scored against a registered prediction or it is
training data, never both. The sealed populations — the 414 measured non-binders
(`assets/<panel>/sealed_negatives.parquet`) and the eval fold of the sealed split —
are the evaluation base, so `training_rows` filters them out and a test pins that.
This is the general form of G6.

STATE.md section 4b measured that a round's worth of molecules moves neither the
regressors nor the gate beyond seed noise, so the ledger's live job is scoring, not
feeding retrains. `training_rows` stays because the invariant is what protects the
evaluation base, not because the retrain is expected to matter.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .data import panel_data
from .panels import PanelSpec

LEDGER = "labels.jsonl"
ANSWER_STATUS = ("answered", "no_answer", "failed")


def _path(panel: PanelSpec, root: Path | None) -> Path:
    return (root or panel.root / "data") / panel.name / LEDGER


@lru_cache(maxsize=4)
def sealed_smiles(panel: PanelSpec) -> frozenset[str]:
    """Every molecule reserved for evaluation: the measured non-binders + the eval fold."""
    out: set[str] = set()
    negatives = panel.data_bundled / "sealed_negatives.parquet"
    if negatives.exists():
        out |= set(pd.read_parquet(negatives)["smi"])
    split = panel_data.load_eval_split(panel)
    out |= set(split.loc[split["fold"] == "eval", "smi"])
    return frozenset(out)


def append(panel: PanelSpec, smi: str, isoform: str, value: float | None,
           relation: str | None, source: str, answer_status: str,
           scored_in_round: int | None = None, root: Path | None = None) -> dict:
    """Record one arrival. Returns the row as written."""
    if answer_status not in ANSWER_STATUS:
        raise ValueError(f"answer_status must be one of {ANSWER_STATUS}, got {answer_status!r}")
    row = {"smi": smi, "isoform": isoform, "value": value, "relation": relation,
           "source": source, "answer_status": answer_status,
           "scored_in_round": scored_in_round,
           "sealed": smi in sealed_smiles(panel),
           "arrived_utc": datetime.now(timezone.utc).isoformat()}
    path = _path(panel, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def read(panel: PanelSpec, root: Path | None = None) -> pd.DataFrame:
    """The whole ledger, oldest first. Missing file -> empty frame with the columns."""
    path = _path(panel, root)
    columns = ["smi", "isoform", "value", "relation", "source", "answer_status",
               "scored_in_round", "sealed", "arrived_utc"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows, columns=columns)


def training_rows(panel: PanelSpec, root: Path | None = None) -> pd.DataFrame:
    """Answered, unsealed rows — the only ones a retrain may read (G6)."""
    frame = read(panel, root)
    if frame.empty:
        return frame
    keep = (frame["answer_status"] == "answered") & (~frame["sealed"].astype(bool))
    return frame[keep].reset_index(drop=True)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_labels.py -q`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/labels.py tests/test_labels.py
git commit -m "Add the label ledger, with the sealed populations held out of training by test"
```

---

### Task 3: Collision-free round indices and a docstring that is true (E)

**Files:**
- Modify: `src/registry.py:116-146` (`append_round`) and `src/registry.py:94-113` (`rounds`)
- Modify: `tests/test_registry.py` (append the new tests)

**Interfaces:**
- Produces: `append_round(...)` unchanged in signature; `Round.index` is now derived from position in the file rather than stored, and `scores_path` carries a random suffix instead of the index.

**The bug.** `append_round` computes `index = len(rounds(...))` and then writes `round_{index}_scores.parquet`. Two processes both read the same length, both write the same filename, and the second silently overwrites the first's scores. The docstring claims the opposite outcome — "two processes appending to the same campaign produce two rounds rather than one overwriting the other's number" — so a reader trusts a guarantee that is not there. STATE.md §8 recorded the behaviour; this fixes it at the one place all callers route through.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_registry.py
def test_concurrent_appends_do_not_share_a_scores_file(tmp_path):
    """STATE.md section 8: two appends that read the same length must not collide.

    Simulated rather than threaded: both calls are given the same view of the
    campaign, which is exactly the race — index derived from disk, then a filename
    derived from the index.
    """
    import pandas as pd

    scores = pd.DataFrame({"smi": ["CCO"], "gap": [1.0]})
    a = registry.append_round("c1", kind="screen", model_ids={}, n_molecules=1,
                              scores=scores, root=tmp_path)
    b = registry.append_round("c1", kind="screen", model_ids={}, n_molecules=1,
                              scores=scores, root=tmp_path)
    assert a.scores_path != b.scores_path
    assert {r.index for r in registry.rounds("c1", root=tmp_path)} == {0, 1}
    assert registry.round_scores("c1", 0, root=tmp_path) is not None
    assert registry.round_scores("c1", 1, root=tmp_path) is not None


def test_append_round_rejects_an_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        registry.append_round("c1", kind="frobnicate", model_ids={}, n_molecules=1,
                              root=tmp_path)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_registry.py -q`
Expected: FAIL — `test_append_round_rejects_an_unknown_kind` fails with `Failed: DID NOT RAISE`.

- [ ] **Step 3: Make the index positional and the filename unique**

In `src/registry.py`, add near `ROUNDS_FILE`:

```python
KINDS = ("screen", "select", "rescore")
```

In `rounds()`, replace the `out.append(...)` loop body so the index comes from position:

```python
    for position, line in enumerate(lines):
        try:
            record = Round.from_dict(json.loads(line))
        except (json.JSONDecodeError, KeyError):
            continue
        # The index is the row's position in the append-only file, not a number
        # chosen before writing. Two processes appending concurrently therefore get
        # two distinct indices by construction rather than by hoping they did not
        # read the same length.
        record.index = position
        out.append(record)
```

Replace `append_round`'s body between the docstring and the `Round(...)` construction:

```python
    from uuid import uuid4

    from .loop_contract import code_version

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    directory = campaign_dir(campaign_id, root)
    directory.mkdir(parents=True, exist_ok=True)

    scores_path = None
    if scores is not None and not scores.empty:
        # Random rather than index-derived: the index is not known until the row is
        # appended, and two writers must not be able to pick the same name.
        name = f"round_{uuid4().hex[:8]}_scores.parquet"
        scores.to_parquet(directory / name, index=False)
        scores_path = name

    record = Round(index=-1, kind=kind,
                   created=datetime.now(timezone.utc).isoformat(),
                   code_version=code_version(), model_ids=dict(model_ids),
                   n_molecules=int(n_molecules), metrics=dict(metrics or {}),
                   scores_path=scores_path, notes=notes)
    with open(directory / ROUNDS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    record.index = len(rounds(campaign_id, root)) - 1
    return record
```

Replace the docstring's second paragraph with what is now true:

```python
    """Record a round; returns it with its assigned index.

    The index is the row's position in the append-only file and the scores filename
    carries a random suffix, so two processes appending concurrently produce two
    rounds with two score files rather than one overwriting the other. `kind` is
    validated against KINDS — the log is meant to distinguish a screen from a
    selection from a Stage-A rescore, and a free-form string cannot.
    """
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_registry.py -q`
Expected: all PASS, including the two new ones.

- [ ] **Step 5: Check no caller passed a kind outside KINDS**

Run: `grep -rn "append_round" app.py src/ scripts/ tests/`
Expected: `app.py:263` passes `kind="screen"`; every test passes `"screen"`, `"select"` or `"rescore"`. If any other literal appears, add it to `KINDS` rather than changing the call.

- [ ] **Step 6: Commit**

```bash
git add src/registry.py tests/test_registry.py
git commit -m "Derive round indices from file position so concurrent appends stop colliding"
```

---

### Task 4: Suitability screen (Stage 0.5)

**Files:**
- Create: `scripts/suitability_screen.py`
- Modify: `scripts/funnel_falsification_audit.py` (make `build_sealed` take a panel)
- Modify: `tests/test_production_path.py` (add the offline smoke)
- Modify: `scripts/reproduce.sh`

**Interfaces:**
- Consumes: `panels.disjointness_report(panel)`, `panels.library_molecule_overlap(panel)`, `panel_data.build_cross_measured(panel)`, `panel_data.build_isoform_dataset(panel, iso)`.
- Produces: `funnel_falsification_audit.build_sealed(panel=DEFAULT_PANEL)` — panel-parameterised; `suitability_screen.screen(panel) -> dict` with keys `n_members`, `n_cross_measured`, `censored_in_library`, `library_overlap`, `disjoint`, `n_scaffolds`.

**Why this task repairs things.** ROADMAP §4 ② puts the screen early because what it counts — cross-measured molecules, censored records, `library_molecule_overlap`, the library rebuild — runs through exactly the production path STATE.md §1a found broken. Building it forces those paths to work and leaves a smoke test behind.

- [ ] **Step 1: Make `build_sealed` panel-parameterised**

In `scripts/funnel_falsification_audit.py`, change the signature and the three constants it closes over:

```python
def build_sealed(panel: PanelSpec = DEFAULT_PANEL) -> pd.DataFrame:
    """..."""                                   # docstring unchanged
    root = panel.root / "assets" / "evidence"
    act = pd.read_parquet(root / "activity.parquet")
    member = pd.read_parquet(root / "library_member.parquet")
    molecule = pd.read_parquet(root / "molecule.parquet")

    isoform_of = {cid: iso for iso, cid in panel.chembl_ids.items()}
```

The rest of the function body is unchanged. Add `from src.panels import PanelSpec` to the imports.

- [ ] **Step 2: Write the failing smoke test**

```python
# append to tests/test_production_path.py
def test_suitability_screen_runs_offline_for_both_panels():
    """Stage 0.5 crosses the production path STATE.md section 1a found broken."""
    from scripts.suitability_screen import screen
    from src.panels import JAK, PI3K

    jak = screen(JAK)
    assert jak["n_members"] == 3
    assert jak["n_cross_measured"] > 3000
    assert jak["censored_in_library"] == 414        # matches STATE.md section 5

    pi3k = screen(PI3K)
    assert pi3k["n_members"] == 4
    assert pi3k["censored_in_library"] < jak["censored_in_library"]
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_production_path.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.suitability_screen'`

- [ ] **Step 4: Write `scripts/suitability_screen.py`**

```python
#!/usr/bin/env python3
"""Stage 0.5: is this panel one the funnel can actually be validated on?

Five counts, all computable before any biology is argued, all offline. A panel that
fails them is not a panel the selectivity claim can be tested on, whatever the
target rationale says.

  members            three or more paralogs — a gap needs something to be a gap against
  cross-measured     molecules measured on every member; below ~200 the conformal
                     gap interval cannot be calibrated (src/campaign.py MIN_CROSS_MEASURED)
  censored in lib    measured non-binders inside the wide library — the negative arm
                     of the falsification audit. STATE.md section 6d measured 491 for
                     JAK and 39 for PI3K, so this does not transfer between panels
  library overlap    library molecules that are this panel's own gate positives; a
                     leak turns prediction into recall
  scaffolds          Murcko diversity of the cross-measured set

Deliberately blind to safety evidence, so there is no circularity with Stage 0: the
screen says whether a panel is *computable*, the Program card says whether it is
*worth computing*.

    python scripts/suitability_screen.py [panel]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from funnel_falsification_audit import build_sealed                 # noqa: E402
from rdkit import Chem, RDLogger                                    # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold                     # noqa: E402

from src.campaign import MIN_CROSS_MEASURED                         # noqa: E402
from src.data import panel_data                                     # noqa: E402
from src.panels import (DEFAULT_PANEL, PANELS, PanelSpec,           # noqa: E402
                        disjointness_report, get_panel,
                        library_molecule_overlap)

RDLogger.DisableLog("rdApp.*")


def screen(panel: PanelSpec) -> dict:
    cross = panel_data.build_cross_measured(panel)
    scaffolds = set()
    for smi in cross["smi"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
    overlap = library_molecule_overlap(panel)
    return {
        "n_members": len(panel.isoforms),
        "n_cross_measured": len(cross),
        "censored_in_library": int(build_sealed(panel)["inchikey"].nunique()),
        "library_overlap": None if overlap is None else overlap.get("n_overlap", 0),
        "disjoint": all(disjointness_report(panel).get("ok", {}).values())
                    if isinstance(disjointness_report(panel).get("ok"), dict) else None,
        "n_scaffolds": len(scaffolds),
    }


def _verdict(result: dict) -> list[str]:
    problems = []
    if result["n_members"] < 3:
        problems.append("fewer than 3 members — a selectivity gap needs paralogs")
    if result["n_cross_measured"] < MIN_CROSS_MEASURED:
        problems.append(f"cross-measured {result['n_cross_measured']} < {MIN_CROSS_MEASURED} "
                        "— the gap interval cannot be calibrated")
    if result["censored_in_library"] < 100:
        problems.append(f"only {result['censored_in_library']} censored library molecules "
                        "— the falsification audit's negative arm is too thin")
    if result["library_overlap"]:
        problems.append(f"{result['library_overlap']} library molecules are this panel's "
                        "own gate positives — that is recall, not prediction")
    return problems


def main() -> None:
    names = [sys.argv[1]] if len(sys.argv) > 1 else list(PANELS)
    for name in names:
        panel = get_panel(name)
        result = screen(panel)
        print(f"\n{'=' * 60}\nPanel {panel.name} — {panel.label}\n{'=' * 60}")
        for key, value in result.items():
            print(f"  {key:22} {value}")
        problems = _verdict(result)
        print("  VERDICT                " +
              ("suitable" if not problems else "NOT suitable"))
        for problem in problems:
            print(f"    - {problem}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the screen, then the tests**

```bash
python scripts/suitability_screen.py
python -m pytest tests/test_production_path.py -q
```
Expected: JAK reports `censored_in_library 414` and `suitable`; PI3K reports a much smaller censored count and is flagged NOT suitable on that criterion. 3 tests PASS.

If `disjointness_report(panel)` does not return an `"ok"` mapping, read its actual return shape and set the `disjoint` key from whatever boolean it does provide — do not invent a key.

- [ ] **Step 6: Add it to `scripts/reproduce.sh`**

Insert before the falsification-audit block:

```bash
echo
echo "== Suitability screen, Stage 0.5 (per panel) =="
echo "The counts that decide whether a panel can be validated at all, before any"
echo "target rationale is argued."
python scripts/suitability_screen.py
```

- [ ] **Step 7: Commit**

```bash
git add scripts/suitability_screen.py scripts/funnel_falsification_audit.py \
        tests/test_production_path.py scripts/reproduce.sh
git commit -m "Add the Stage 0.5 suitability screen and put it in CI"
```

---

### Task 5: Program card and the D2 wiring check (Stage 0)

**Files:**
- Create: `assets/cases/jak-autoimmune/program.md`
- Create: `assets/cases/jak-autoimmune/panel.json`
- Create: `tests/test_wiring.py`
- Modify: `.gitignore` (only if `assets/cases/` is caught by an existing rule — check first)

**Interfaces:**
- Consumes: `funnel.score_molecules(smiles)`, `selectivity.selectivity_gap(preds, target, offs)`.
- Produces: the committed card; no new Python API.

**Why no refactor.** ROADMAP D2 asks for the indication to parameterise the objective sign. `selectivity_gap(preds, target, offs)` already takes both as arguments, so a second program is a second call, not a second code path. Building a `Program` abstraction before a second program exists would be an interface with one implementation.

**Verified numbers.** These were measured on 2026-08-09 with the deployed models:

| drug | pred JAK1 | pred JAK2 | pred JAK3 | autoimmune gap | MPN gap |
|---|---:|---:|---:|---:|---:|
| upadacitinib | 7.394 | 7.057 | 7.055 | +0.337 | −0.337 |
| fedratinib | 7.804 | 8.283 | 6.346 | −0.479 | +0.479 |
| ruxolitinib | 8.420 | 8.377 | 6.763 | +0.043 | −0.043 |

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wiring.py
"""D2 — does the indication actually drive the objective's sign?

Same panel, same models, opposite objectives. Autoimmune avoids JAK2 (EPO/TPO
signalling, hence anaemia and thrombocytopenia); myeloproliferative neoplasms
target JAK2 directly (JAK2 V617F). If the wiring is right, a JAK1-preferring drug
outranks a JAK2-preferring one under the first objective and the order reverses
under the second.

This is a WIRING check, not a ranking-performance result: all three drugs are in
the training data, so the predictions are recall. STATE.md and ROADMAP say the same
thing, and claiming more would repeat v1's mistake.

Note the degeneracy: for all three drugs JAK3 is the weakest isoform, so
max(off) picks the same isoform in both objectives and the two gaps come out exactly
antisymmetric. The test therefore asserts the two RANK ORDERS, which is the claim,
and not that the two scores carry independent information.
"""
import pandas as pd

from src import funnel
from src.selectivity import selectivity_gap

DRUGS = {
    "upadacitinib": "CC[C@@H]1CN(C(=O)NCC(F)(F)F)C[C@@H]1c1cnc2cnc3[nH]ccc3n12",
    "fedratinib": "Cc1cnc(Nc2ccc(OCCN3CCCC3)cc2)nc1Nc1cccc(S(=O)(=O)NC(C)(C)C)c1",
    "ruxolitinib": "N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1",
}


def _ranks():
    names = list(DRUGS)
    scored = funnel.score_molecules([DRUGS[n] for n in names])
    preds = pd.DataFrame({iso: scored[f"pred_{iso}"] for iso in ("JAK1", "JAK2", "JAK3")})
    frame = pd.DataFrame({
        "drug": names,
        "autoimmune": selectivity_gap(preds, "JAK1", ("JAK2", "JAK3")),
        "mpn": selectivity_gap(preds, "JAK2", ("JAK1", "JAK3")),
    })
    return (frame.sort_values("autoimmune", ascending=False)["drug"].tolist(),
            frame.sort_values("mpn", ascending=False)["drug"].tolist())


def test_indication_flips_the_ranking():
    autoimmune, mpn = _ranks()
    assert autoimmune == ["upadacitinib", "ruxolitinib", "fedratinib"]
    assert mpn == ["fedratinib", "ruxolitinib", "upadacitinib"]


def test_the_dual_inhibitor_is_middle_under_both_objectives():
    autoimmune, mpn = _ranks()
    assert autoimmune[1] == "ruxolitinib"
    assert mpn[1] == "ruxolitinib"
```

- [ ] **Step 2: Run it to verify it passes for the right reason**

Run: `python -m pytest tests/test_wiring.py -q`
Expected: 2 PASS. This test passes on first write because the wiring is already correct — it is a regression gate, not a red-green cycle. Confirm it is really testing something by temporarily swapping `"JAK1"` and `"JAK2"` in the `autoimmune` line, re-running to see it FAIL, then reverting.

- [ ] **Step 3: Write the Program card**

Create `assets/cases/jak-autoimmune/program.md`:

```markdown
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

## Provenance

Predictions in the wiring check come from the deployed models on 2026-08-09. All
three drugs are in the training data, so those numbers are recall — the check tests
that the indication drives the sign, not that the ranking is skilful.
```

- [ ] **Step 4: Write the machine-readable half**

Create `assets/cases/jak-autoimmune/panel.json`:

```json
{
  "case_id": "jak-autoimmune",
  "panel": "jak",
  "indication": "autoimmune",
  "target": "JAK1",
  "offs": ["JAK2", "JAK3"],
  "anti_target_evidence": {
    "JAK2": {"class": [1, 2], "registered": true},
    "JAK3": {"class": [4], "registered": false, "watch": true}
  },
  "recorded_gaps": ["7a", "7b", "7c"],
  "signed_utc": "2026-08-09",
  "wiring_check": "tests/test_wiring.py"
}
```

- [ ] **Step 5: Confirm the card is actually committed**

```bash
git check-ignore -v assets/cases/jak-autoimmune/program.md || echo "not ignored - good"
```
Expected: `not ignored - good`. If a rule matches, add `!assets/cases/` to `.gitignore` rather than deleting the existing rule.

- [ ] **Step 6: Commit**

```bash
git add assets/cases/jak-autoimmune/ tests/test_wiring.py
git commit -m "Sign the first Program card and pin the indication-to-sign wiring"
```

---

### Task 6: State the potency claim conditionally (A1, N15–N19)

**Files:**
- Modify: `app.py:381,542,640,796,809` (the JAK-string defaults and the Tier-1 copy — read each line before editing; the line numbers are from STATE.md §6e and may have drifted)
- Modify: `src/funnel.py` module docstring
- Modify: `README.md` (new "판단 원리" section, N17)

**Interfaces:** none — this task changes prose and one UI string. No Python API moves.

**The numbers this copy must not misstate.** From `STATE.md` §2/§2a, all reproducible via `scripts/funnel_falsification_audit.py`:
- the potency floor passes **80.7 %** of 414 measured non-binders and **99.4 %** of 1,928 post-2020 measured actives;
- the deployed cascade's predictions exceed the molecule's own measured upper bound for **93.0 %** of those 414;
- the gap claim is **0.779 vs a 1-NN lookup's 0.782** on a scaffold split, and **0.734 vs 0.572** under the 2020 time cut.

- [ ] **Step 1: Add the "판단 원리" section to README.md**

Insert after the scope declaration near the top:

```markdown
## 판단 원리 — 이 퍼널이 무엇을 보고 무엇을 안 보는가

```
분자 → [반지름 2까지 부분구조 열거 → 해시] → 2048비트
     → HistGradientBoosting → pchembl
```

판단 원리는 하나다: **"이 분자의 부분구조 조각들이, 학습셋에서 활성이었던 분자들의 조각들과
닮았는가."** 단백질도, 결합의 물리도, 3D 형태도, 카이랄성도 보지 않는다.

그래서 **에탄올이 760 nM JAK1 저해제로 점수가 난다** — 도메인 밖에서 트리 앙상블은 학습
평균(~6.3)으로 회귀하기 때문이다. Tier 0.5 바인더 게이트가 존재하는 이유가 이것이다.

**측정된 한계 세 가지** (전부 `scripts/` 아래 재현 스크립트가 있다):

| | 측정 |
|---|---|
| potency 축 | 실측 비결합자 414개에 대해 예측이 자기 측정 상한을 **93.0 %** 초과한다 |
| potency floor | 그 414개의 **80.7 %** 를 통과시킨다. 진짜 활성은 99.4 % 통과시키므로 방향은 맞지만 판별폭은 18.7 포인트뿐이다 |
| gap 축 | scaffold 분할에서 **Tanimoto 최근접이웃 조회(0.782)를 이기지 못한다(0.779).** 2020년 시간분할에서는 조회 0.572 대 모델 0.734로 갈라진다 |

**따라서 이 저장소가 방어하는 주장은 "선택성을 잘 맞힌다"가 아니라 "출판연도가 이동해도
유사도 조회보다 덜 무너진다"이다.**
```

- [ ] **Step 2: Make the Tier-1 UI copy conditional**

Read `app.py` around the Tier-1 potency labels, then change the wording so the floor is described as what it was measured to be. Replace any phrasing that presents predicted potency as a potency claim with:

```python
"Tier 1 — predicted potency floor. Measured on 414 known non-binders this floor "
"passes 80.7 % of them (and 99.4 % of real actives), so treat it as a coarse "
"screen, not a potency claim. pchembl 6 is IC50 1 uM — a screening cut, not a "
"drug-level bar."
```

- [ ] **Step 3: Record the hyperparameter policy as a choice (N15, N16, N19)**

Append to the `src/funnel.py` module docstring:

```
Hyperparameters are deliberately untuned, and that is a decision rather than an
omission: tuning before the potency axis is fixed would optimise R2, and the 93 %
falsification happened at good R2. The tuning protocol is fixed in advance in
ROADMAP section 8 and opens after A2. `early_stopping='auto'` behaves differently
per isoform (STATE.md section 4a) and the 10 % of rows it costs JAK1 and JAK2 is
below the seed noise floor (section 4b), so it is a consistency fix scheduled with
the next retrain, not a performance one.

The applicability domain is blind to chirality: stereoisomers are Tanimoto distance
0 under this fingerprint, so a molecule can read in-domain because its mirror image
was in training (STATE.md section 3d).
```

- [ ] **Step 4: Verify nothing broke**

Run: `python -m pytest tests/ -q`
Expected: all tests pass. Copy changes should not move any test; if one fails it was asserting on a UI string, and the assertion should be updated to the new copy.

- [ ] **Step 5: Commit**

```bash
git add README.md app.py src/funnel.py
git commit -m "State the potency claim as what was measured, and record the tuning policy as a choice"
```

---

## Wave 3 — what Wave 1 changed, to be planned separately

Not tasks. These are the decisions Wave 1's measurements force on the loop, recorded here so the next plan starts from them rather than from ROADMAP's pre-measurement version.

| ROADMAP item | Change | Because |
|---|---|---|
| `H` — Analyze: retrain trigger | **Replace with an acquisition-reordering trigger.** A retrain trigger that fires and changes nothing is worse than none, because it reads as a working feedback loop. | §4b — a round moves the metrics 90× below seed noise |
| `D` — oracle role | **Scoring, not label supply.** `population()` and exhaustion (G3) stay; feeding retrains does not. | §4b |
| `B` — gate redesign | **Promote, and restate.** Not "add more negatives" but "change what a negative is": the gate scores ROC-AUC 0.998 on separating JAK actives from other-target actives, while passing 36 % of measured non-binders. Two different tasks. | §4b + §2 |
| `G-2` — chemprop multitask | **Demote below B.** Multi-task assumes tasks inform each other; cross-target signal on unseen chemistry medians 0.472 with a 0.220–0.810 spread, and volume is saturated. | §4b + §4d |
| `PCM` | **Stays deferred, now with a number.** The baseline is low but irregular — not easy to beat, hard to tell whether it was beaten. | §4d |
| Round scoring protocol | Every round is scored on the **sealed time split** from Task 1. | §4c |

Also outstanding, found during Wave 1 and not yet scheduled:
- `assets/evidence/molecule.parquet` has `mol_chembl_id` null for all 75,870 rows, so molecules cannot be resolved by ChEMBL id offline (this is why Task 5's SMILES came from PubChem).
- `scripts/assay_time_audit.py` still needs a network rebuild, because the committed per-isoform parquets carry no `year_first`. Task 1's `panel_data.year_first` is the offline replacement; pointing that script at it is a small follow-up.

---

## Self-Review

**Spec coverage.** ROADMAP Wave 2 lists 0.5 suitability screen (Task 4), N11 sealed split (Task 1), E run-history state machine + label ledger + index race (Tasks 2 and 3), Stage 0 Program + wiring check (Task 5), A1 + N15/N16/N17/N19 (Task 6). All five are covered. The full round-transition state machine (which kinds may follow which) is **not** in Task 3 — Task 3 validates `kind` and fixes the collision, and legal transitions are deferred to Wave 3 where the loop that needs them is built. That is a deliberate narrowing, recorded here rather than silently dropped.

**Placeholders.** No TBDs. Every code step carries the code. Task 4 Step 5 contains one conditional instruction — read `disjointness_report`'s real return shape rather than assume a key — which is a verification step, not a placeholder, because inventing the key is the failure mode it prevents.

**Type consistency.** `panel_data.year_first(panel) -> pd.Series` is defined in Task 1 and consumed by Task 1's own script and by the two audit scripts it edits. `panel_data.load_eval_split(panel) -> pd.DataFrame` is defined in Task 1 and consumed by `labels.sealed_smiles` in Task 2. `build_sealed(panel)` is re-signatured in Task 4 Step 1 before Task 4 Step 4 calls it with a panel. `labels.append` takes `answer_status` in every call site shown. `registry.KINDS` is defined in Task 3 Step 3 and used by the test in Step 1.
