# Wave 3 — One DMTA Turn Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the loop once — three rounds that select molecules, get measured answers, score themselves honestly, and let round N change round N+1, with the whole case committed as files.

**Architecture:** Two new modules and one driver. `src/oracle.py` reveals measured answers for the sealed eval fold and refuses anything the models trained on. `src/dmta.py` holds the round — select, ask, analyze — because those three change together. `scripts/dmta_run.py` drives it and resumes from `rounds.jsonl`. Nothing retrains between rounds; the models are refit once on the train fold and held fixed, and acquisition is the only thing that moves.

**Tech Stack:** Python 3.11, pandas + pyarrow, RDKit, scikit-learn, pytest. Offline, CPU, committed assets only.

## Global Constraints

- **G0** — the deployed JAK assets are immutable. The campaign refits models on the train fold in-process and never writes `assets/models/jak/`.
- **G2** — the oracle returns `None` for any molecule in the train fold. Enforced by test.
- **G3** — the oracle reports `population()` and the loop stops when it reaches zero.
- **G4** — stopping after round 2 and re-running produces the same round 3.
- **G5** — improvement and non-improvement are reported with the same weight. Every round records the acquired batch **and** a random control batch.
- **G6** — sealed molecules never enter training. Round answers land in the ledger with `sealed=True`, and `labels.training_rows` already filters them.
- **G7** — falsification rate is computed every round and stored in the round's metrics.
- **D9** — the case is committed files. No database.
- Run `python -m pytest tests/ -q`. Baseline entering this plan: **194 passed**.

---

## Why this differs from ROADMAP's Wave 3

| ROADMAP | This plan | Measurement that forced it |
|---|---|---|
| `H` retrain trigger | **No retrain trigger.** Models are refit once and frozen; the trigger reorders acquisition instead | §4b — a 70-molecule round moves JAK1 Spearman +0.00008 against ±0.0072 seed noise |
| `D` oracle supplies labels for retraining | **Oracle scores only.** Its answers are sealed and cannot train | §4b, and G6 |
| `G` contract schema 1.2 | **Dropped from this plan.** The contract is the Colab handoff; this loop is CPU-only and never crosses to Stage A. Adding a schema version nothing reads is scaffolding | ponytail rung 1 |
| Round scored on a scaffold split | **Scored on the sealed time split** | §4c — a 1-NN lookup ties the model on a scaffold split |
| `G-1` docking in parallel | **Separate plan.** GPU, no dependency either way, not on the path to a turning loop | — |

**Measured oracle population** (2026-08-09): the eval fold holds 3,797 molecules, of which **1,078 are cross-measured** and therefore have a measured gap. **367 of those (34.0 %)** are ≥10×-selective, so the top-decile enrichment ceiling is 1/0.34 = **2.94×** — report enrichment against that ceiling, never raw.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/oracle.py` (create) | reveals a measured answer for an eval-fold molecule; `None` for anything else; reports and exhausts its population |
| `src/dmta.py` (create) | one round: build the frozen context, select a batch plus a random control, ask, analyze |
| `scripts/dmta_run.py` (create) | driver — N rounds, resumable from `rounds.jsonl` |
| `tests/test_oracle.py`, `tests/test_dmta.py` (create) | G2/G3 and G5/G7 gates |
| `assets/cases/jak-autoimmune/rounds.jsonl` (create, committed) | the case's round history |
| `CASES.md` (create, generated) | index built from the case files; never hand-written |
| `.gitignore` (modify) | let `assets/cases/` through if a rule catches it |

---

### Task 1: The oracle (G2, G3)

**Files:**
- Create: `src/oracle.py`
- Create: `tests/test_oracle.py`

**Interfaces:**
- Consumes: `panel_data.load_eval_split(panel)`, `panel_data.build_cross_measured(panel)`.
- Produces: `oracle.TimeSplitOracle(panel)` with `.answer(smi) -> dict[str, float] | None`, `.population() -> int`, `.remaining() -> list[str]`, `.exhausted -> bool`, `.asked -> set[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle.py
"""The time-split oracle: what it will answer, and what it must refuse (G2, G3).

The molecules it answers for are real and their measurements are real; what makes
them a test is that the campaign's models were refit without them. An oracle that
answers for a training molecule is measuring recall, so the refusal is the property
that matters and it is asserted first.
"""
import pytest

from src import oracle as orc
from src.data import panel_data
from src.panels import JAK


def _split():
    return panel_data.load_eval_split(JAK)


def test_answers_an_eval_fold_molecule_with_every_isoform():
    o = orc.TimeSplitOracle(JAK)
    smi = o.remaining()[0]
    answer = o.answer(smi)
    assert set(answer) == set(JAK.isoforms)
    assert all(isinstance(v, float) for v in answer.values())


def test_refuses_a_training_molecule(the_train_molecule=None):
    """G2: a molecule the campaign models trained on has no answer here."""
    split = _split()
    train_smi = split.loc[split["fold"] == "train", "smi"].iloc[0]
    o = orc.TimeSplitOracle(JAK)
    assert o.answer(train_smi) is None


def test_refuses_a_molecule_nobody_measured():
    o = orc.TimeSplitOracle(JAK)
    assert o.answer("CCO") is None


def test_population_shrinks_as_it_is_asked_and_exhausts():
    o = orc.TimeSplitOracle(JAK)
    start = o.population()
    assert start > 1000                     # measured 1,078 on 2026-08-09
    o.answer(o.remaining()[0])
    assert o.population() == start - 1
    assert not o.exhausted


def test_asking_twice_does_not_double_count():
    o = orc.TimeSplitOracle(JAK)
    smi = o.remaining()[0]
    o.answer(smi)
    after = o.population()
    o.answer(smi)
    assert o.population() == after
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_oracle.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'src.oracle'`

- [ ] **Step 3: Write `src/oracle.py`**

```python
"""The oracle: measured answers the campaign's models were not trained on.

STATE.md section 5 counted the "free proxy oracle" the README proposed and found
9 to 28 molecules — the haystack and the answer set were disjoint by construction.
Two real populations survived that count, and this is the second one: the eval fold
of the sealed time split (STATE.md section 4c), 1,078 cross-measured molecules
first published after 2020.

What makes them a test is not that the measurements are hidden — they are in the
repo — but that the campaign refits its models on the train fold only, so the
models have not seen them. `answer` therefore refuses anything outside the eval
fold (G2): answering for a training molecule would return recall wearing the
costume of a prediction.

**This oracle scores; it does not teach.** STATE.md section 4b measured a round's
worth of molecules as indistinguishable from a reseed on both axes, and its answers
are sealed under G6 regardless. The loop's feedback runs through acquisition.

The population is exhaustible. `population()` and `exhausted` exist so a run stops
when there is nothing left to ask rather than silently re-asking (G3).
"""
from __future__ import annotations

import pandas as pd

from .data import panel_data
from .panels import PanelSpec


class TimeSplitOracle:
    """Measured pchembl per isoform for eval-fold molecules; None for everything else."""

    def __init__(self, panel: PanelSpec) -> None:
        self.panel = panel
        split = panel_data.load_eval_split(panel)
        eval_smiles = set(split.loc[split["fold"] == "eval", "smi"])
        cross = panel_data.build_cross_measured(panel)
        answerable = cross[cross["smi"].isin(eval_smiles)].set_index("smi")
        self._answers: dict[str, dict[str, float]] = {
            smi: {iso: float(row[iso]) for iso in panel.isoforms}
            for smi, row in answerable.iterrows()
        }
        self.asked: set[str] = set()

    def answer(self, smi: str) -> dict[str, float] | None:
        """The measured pchembl per isoform, or None if this molecule is not askable."""
        found = self._answers.get(smi)
        if found is None:
            return None
        self.asked.add(smi)
        return dict(found)

    def remaining(self) -> list[str]:
        """Molecules with an answer that has not been asked for yet."""
        return [smi for smi in self._answers if smi not in self.asked]

    def population(self) -> int:
        return len(self._answers) - len(self.asked)

    @property
    def exhausted(self) -> bool:
        return self.population() == 0

    def measured_gap(self, smi: str) -> float | None:
        """S = measured(target) - max measured(off), or None if not askable."""
        found = self._answers.get(smi)
        if found is None:
            return None
        return found[self.panel.target] - max(found[o] for o in self.panel.offs)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_oracle.py -q`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/oracle.py tests/test_oracle.py
git commit -m "Add the time-split oracle, which refuses every molecule the models trained on"
```

---

### Task 2: One round — frozen context, selection, control arm, analysis (G5, G7, N9)

**Files:**
- Create: `src/dmta.py`
- Create: `tests/test_dmta.py`

**Interfaces:**
- Consumes: `oracle.TimeSplitOracle`, `panel_data.load_eval_split`, `isoform_regressor.train_and_cache(panel, iso, use_cache=False, data=...)`, `applicability.build_reference`, `applicability.in_domain`, `models.features.morgan_matrix`, `labels.append`.
- Produces: `dmta.Context` (dataclass: `models`, `ad_reference`, `model_ids`), `dmta.build_context(panel) -> Context`, `dmta.score(context, panel, smiles) -> pd.DataFrame`, `dmta.run_round(panel, context, oracle, batch=70, penalty=None, seed=0, root=None) -> dict`.

**Design the implementer must not simplify away.** The models are built once and frozen for the whole campaign. That is not laziness — STATE.md §4b measured that refitting on a round's worth of new molecules moves the metrics 90× below seed noise, so a loop that retrained would be reporting noise as learning. The consequence is that `model_ids` are identical in every round, which the round record states rather than hides.

**Why the control arm is not optional.** Wave 1 learned this twice: a one-armed measurement cannot separate a real effect from a shift in the denominator. Every round draws a random batch of the same size from the same remaining pool and scores it identically, so "acquisition beat random by X" is the claim rather than "the hit rate was Y".

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dmta.py
"""One DMTA round: the properties that make its numbers mean anything.

Every round carries a random control batch of the same size drawn from the same
pool (G5/N9). Wave 1 established twice — once on the falsification audit, once on
the learning curve — that a one-armed number cannot distinguish an effect from a
moved denominator, and a loop reporting its own improvement is exactly where that
mistake is most expensive.
"""
import pandas as pd
import pytest

from src import dmta, labels
from src import oracle as orc
from src.panels import JAK


@pytest.fixture(scope="module")
def context():
    return dmta.build_context(JAK)


def test_context_models_are_refit_not_the_deployed_ones(context):
    """G0: the campaign must not be scoring through the shipped models."""
    from src.models.isoform_regressor import train_and_cache
    deployed = train_and_cache(JAK, "JAK1").model
    assert context.models["JAK1"] is not deployed
    assert set(context.models) == set(JAK.isoforms)


def test_round_reports_acquired_and_random_arms(tmp_path, context):
    o = orc.TimeSplitOracle(JAK)
    result = dmta.run_round(JAK, context, o, batch=20, seed=0, root=tmp_path)
    for arm in ("acquired", "random"):
        assert result[arm]["n"] == 20
        assert 0.0 <= result[arm]["hit_rate"] <= 1.0
        assert 0.0 <= result[arm]["falsification_rate"] <= 1.0
    assert result["enrichment_ceiling"] > 1.0


def test_round_records_every_answer_in_the_ledger_as_sealed(tmp_path, context):
    o = orc.TimeSplitOracle(JAK)
    dmta.run_round(JAK, context, o, batch=10, seed=0, root=tmp_path)
    ledger = labels.read(JAK, root=tmp_path)
    assert len(ledger) == 20 * len(JAK.isoforms)      # both arms, every isoform
    assert ledger["sealed"].all()                     # G6
    assert labels.training_rows(JAK, root=tmp_path).empty


def test_round_consumes_the_oracle_population(tmp_path, context):
    o = orc.TimeSplitOracle(JAK)
    before = o.population()
    dmta.run_round(JAK, context, o, batch=15, seed=0, root=tmp_path)
    assert o.population() == before - 30              # 15 acquired + 15 control


def test_falsification_by_domain_bucket_is_reported(tmp_path, context):
    o = orc.TimeSplitOracle(JAK)
    result = dmta.run_round(JAK, context, o, batch=40, seed=0, root=tmp_path)
    buckets = result["falsification_by_bucket"]
    assert buckets                                     # non-empty
    assert all(0.0 <= v <= 1.0 for v in buckets.values())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dmta.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'src.dmta'`

- [ ] **Step 3: Write `src/dmta.py`**

```python
"""One turn of the loop: select, ask, analyze.

The three steps live in one module because they change together — a new acquisition
signal needs a new analysis to justify it, and an analysis nobody selects on is a
report rather than a loop.

**The models are frozen for the whole campaign.** They are refit once on the train
fold of the sealed split and never touched again. STATE.md section 4b measured a
70-molecule round as moving JAK1 Spearman by +0.00008 against a seed-to-seed spread
of +-0.0072, so a loop that retrained each round would be reporting noise as
learning. What changes between rounds is acquisition, and the round record states
that `model_ids` are identical throughout rather than leaving it to be inferred.

**Every round has two arms.** The acquired batch and a random batch of the same size
from the same remaining pool, scored identically (G5, N9). A one-armed hit rate
cannot separate a real effect from a moved denominator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import labels
from .applicability import ADReference, build_reference, in_domain
from .data import panel_data
from .models.features import morgan_matrix
from .models.isoform_regressor import train_and_cache
from .oracle import TimeSplitOracle
from .panels import PanelSpec
from .selectivity import SELECTIVE_GAP

# Nearest-neighbour similarity bands the falsification rate is reported over. The
# acquisition penalty in Task 3 is calibrated on exactly these bands, so they are
# defined once here.
NN_BUCKETS = ((0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 1.01))
TOP_FRAC = 0.1


@dataclass(frozen=True)
class Context:
    """The frozen scoring apparatus for one campaign."""
    models: dict
    ad_reference: ADReference
    model_ids: dict[str, str]
    train_smiles: tuple[str, ...]


def build_context(panel: PanelSpec) -> Context:
    """Refit the regressors and the AD reference on the train fold only.

    The deployed models trained on the whole dataset, eval fold included, so
    scoring the loop through them would measure memorisation. This is the same
    refit the falsification audit's positive arm uses.
    """
    from .loop_contract import model_id

    split = panel_data.load_eval_split(panel)
    train_smiles = set(split.loc[split["fold"] == "train", "smi"])

    models, ids = {}, {}
    for iso in panel.isoforms:
        data = panel_data.build_isoform_dataset(panel, iso)
        pre = data[data["smi"].isin(train_smiles)]
        bundle = train_and_cache(panel, iso, use_cache=False, data=pre)
        models[iso] = bundle.model
        ids[iso] = model_id(panel.chembl_ids[iso], bundle.model)

    return Context(models=models,
                   ad_reference=build_reference(sorted(train_smiles)),
                   model_ids=ids,
                   train_smiles=tuple(sorted(train_smiles)))


def score(context: Context, panel: PanelSpec, smiles: list[str]) -> pd.DataFrame:
    """Predicted pchembl per isoform, the gap, and the AD nearest-neighbour similarity."""
    X, mask = morgan_matrix(list(smiles))
    kept = [s for s, keep in zip(smiles, mask) if keep]
    frame = pd.DataFrame({"smi": kept})
    for iso in panel.isoforms:
        frame[f"pred_{iso}"] = context.models[iso].predict(X)
    frame["gap"] = (frame[f"pred_{panel.target}"]
                    - frame[[f"pred_{o}" for o in panel.offs]].max(axis=1))
    frame["nn_sim"] = in_domain(kept, reference=context.ad_reference)["nn_sim"]
    return frame


def _bucket(nn_sim: float) -> str:
    for low, high in NN_BUCKETS:
        if low <= nn_sim < high:
            return f"{low:.2f}-{high:.2f}"
    return "out-of-range"


def _arm_metrics(scored: pd.DataFrame, oracle: TimeSplitOracle,
                 panel: PanelSpec) -> tuple[dict, pd.DataFrame]:
    """Ask the oracle for one arm and score it. Returns (metrics, answered rows)."""
    rows = []
    for row in scored.itertuples():
        answer = oracle.answer(row.smi)
        if answer is None:
            continue
        measured_gap = answer[panel.target] - max(answer[o] for o in panel.offs)
        rows.append({"smi": row.smi, "pred": getattr(row, f"pred_{panel.target}"),
                     "measured": answer[panel.target], "pred_gap": row.gap,
                     "measured_gap": measured_gap, "nn_sim": row.nn_sim,
                     "answer": answer})
    answered = pd.DataFrame(rows)
    if answered.empty:
        return {"n": 0, "hit_rate": 0.0, "falsification_rate": 0.0,
                "median_error": float("nan")}, answered

    # Falsified = the prediction overstates the measurement by more than the
    # deployed conformal half-width would excuse. One log unit is the funnel's own
    # "typical potency error" line (VALIDATION.md STEP 3), used here so the rate is
    # comparable to the falsification audit's exceedance.
    overstated = (answered["pred"] - answered["measured"]) > 1.0
    return ({"n": int(len(answered)),
             "hit_rate": float((answered["measured_gap"] >= SELECTIVE_GAP).mean()),
             "falsification_rate": float(overstated.mean()),
             "median_error": float((answered["pred"] - answered["measured"]).median())},
            answered)


def run_round(panel: PanelSpec, context: Context, oracle: TimeSplitOracle,
              batch: int = 70, penalty: dict[str, float] | None = None,
              seed: int = 0, root=None) -> dict:
    """Select, ask, analyze — one round, both arms.

    `penalty` maps an NN-similarity bucket to a score deduction; it is what the
    previous round's analysis feeds back (Task 3). None means the first round,
    which ranks on the gap alone.
    """
    pool = oracle.remaining()
    if len(pool) < 2 * batch:
        batch = len(pool) // 2
    scored = score(context, panel, pool)
    scored["penalty"] = [0.0 if penalty is None else penalty.get(_bucket(s), 0.0)
                         for s in scored["nn_sim"]]
    scored["acquisition"] = scored["gap"] - scored["penalty"]

    acquired = scored.sort_values("acquisition", ascending=False).head(batch)
    rest = scored[~scored["smi"].isin(set(acquired["smi"]))]
    control = rest.sample(n=min(batch, len(rest)), random_state=seed)

    acq_metrics, acq_rows = _arm_metrics(acquired, oracle, panel)
    ctl_metrics, ctl_rows = _arm_metrics(control, oracle, panel)

    both = pd.concat([acq_rows, ctl_rows], ignore_index=True)
    for row in both.itertuples():
        for iso, value in row.answer.items():
            labels.append(panel, smi=row.smi, isoform=iso, value=float(value),
                          relation="=", source="TimeSplitOracle",
                          answer_status="answered", root=root)

    by_bucket = {}
    if not both.empty:
        both["bucket"] = [_bucket(s) for s in both["nn_sim"]]
        overstated = (both["pred"] - both["measured"]) > 1.0
        by_bucket = {b: float(overstated[both["bucket"] == b].mean())
                     for b in sorted(set(both["bucket"]))
                     if (both["bucket"] == b).sum() >= 5}

    base_rate = float((both["measured_gap"] >= SELECTIVE_GAP).mean()) if len(both) else 0.0
    return {
        "acquired": acq_metrics,
        "random": ctl_metrics,
        "advantage": acq_metrics["hit_rate"] - ctl_metrics["hit_rate"],
        "falsification_by_bucket": by_bucket,
        "base_rate": base_rate,
        # Raw enrichment is not comparable across rounds with different base rates:
        # at a 34 % base rate a perfect ranker tops out at 2.94x.
        "enrichment_ceiling": (1.0 / base_rate) if base_rate else float("nan"),
        "oracle_remaining": oracle.population(),
        "model_ids": dict(context.model_ids),
    }
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_dmta.py -q`
Expected: 5 PASS. The module-scoped `context` fixture refits three regressors, which takes about two minutes on the first test and is reused afterwards.

- [ ] **Step 5: Commit**

```bash
git add src/dmta.py tests/test_dmta.py
git commit -m "Add one DMTA round: frozen models, an acquired arm and a random control arm"
```

---

### Task 3: The feedback channel — acquisition penalty from measured falsification

**Files:**
- Modify: `src/dmta.py` (add `penalty_from` after `run_round`)
- Modify: `tests/test_dmta.py` (append)

**Interfaces:**
- Produces: `dmta.penalty_from(round_results: list[dict], scale: float = 1.0) -> dict[str, float]`.

**Why this is the loop's only honest feedback.** Three of Wave 1's four measurements point the same way: data volume is saturated (§4b), so retraining cannot be the channel; the model beats a lookup only under distribution shift (§4c); and the AD was the one layer STATE.md §2 found earning its keep. So the thing to learn round-over-round is *where the predictions are untrustworthy*, and the AD's nearest-neighbour similarity is the axis that already carries that. ROADMAP N13 asks for exactly this conversion — AD from hard filter to acquisition penalty — and calls it the only structural defence against novelty collapse.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dmta.py
def test_penalty_is_larger_where_falsification_was_worse():
    rounds = [{"falsification_by_bucket": {"0.00-0.30": 0.8, "0.60-1.01": 0.1}}]
    penalty = dmta.penalty_from(rounds)
    assert penalty["0.00-0.30"] > penalty["0.60-1.01"]
    assert penalty["0.60-1.01"] >= 0.0


def test_penalty_averages_across_rounds_it_is_given():
    rounds = [{"falsification_by_bucket": {"0.00-0.30": 1.0}},
              {"falsification_by_bucket": {"0.00-0.30": 0.0}}]
    assert dmta.penalty_from(rounds)["0.00-0.30"] == pytest.approx(0.5)


def test_no_history_means_no_penalty():
    assert dmta.penalty_from([]) == {}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dmta.py -q`
Expected: FAIL, `AttributeError: module 'src.dmta' has no attribute 'penalty_from'`

- [ ] **Step 3: Add `penalty_from` to `src/dmta.py`**

```python
def penalty_from(round_results: list[dict], scale: float = 1.0) -> dict[str, float]:
    """Acquisition deduction per NN-similarity bucket, from measured falsification.

    This is the loop's feedback channel, and after Wave 1 it is the only defensible
    one. Data volume is saturated on both axes (STATE.md section 4b), so retraining
    cannot be it; the AD is the layer section 2 found earning its keep; and ROADMAP
    N13 asks for exactly this conversion — the applicability domain from a hard
    filter into an acquisition penalty, which is what keeps exploration from being
    structurally zero while still charging for unreliability.

    The deduction is in log units, directly comparable to the gap it is subtracted
    from: a bucket where every prediction overstated by more than a log costs a full
    log of gap, one where none did costs nothing.
    """
    if not round_results:
        return {}
    totals: dict[str, list[float]] = {}
    for result in round_results:
        for bucket, rate in result.get("falsification_by_bucket", {}).items():
            totals.setdefault(bucket, []).append(float(rate))
    return {bucket: scale * float(np.mean(rates)) for bucket, rates in totals.items()}
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_dmta.py -q`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dmta.py tests/test_dmta.py
git commit -m "Feed measured falsification back as an acquisition penalty, not a retrain"
```

---

### Task 4: The driver, resumable (G4)

**Files:**
- Create: `scripts/dmta_run.py`
- Modify: `tests/test_dmta.py` (append the resume gate)

**Interfaces:**
- Consumes: `registry.append_round`, `registry.rounds`, `dmta.*`, `oracle.TimeSplitOracle`.
- Produces: `dmta_run.run(panel, campaign_id, n_rounds, batch, root=None) -> list[dict]`.

**Resume semantics.** `rounds.jsonl` is the state. On start the driver replays the recorded rounds to rebuild both the penalty and the set of already-asked molecules, then continues. G4 is that stopping after round 2 and re-running gives the same round 3.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dmta.py
def test_resuming_after_two_rounds_reproduces_the_third(tmp_path, context):
    """G4: the round history is the state, so a restart is not a different run."""
    import sys
    sys.path.insert(0, "scripts")
    import dmta_run

    straight = dmta_run.run(JAK, "g4-a", n_rounds=3, batch=10, root=tmp_path)
    dmta_run.run(JAK, "g4-b", n_rounds=2, batch=10, root=tmp_path)
    resumed = dmta_run.run(JAK, "g4-b", n_rounds=3, batch=10, root=tmp_path)

    assert len(resumed) == 3
    assert resumed[2]["acquired"]["n"] == straight[2]["acquired"]["n"]
    assert resumed[2]["acquired"]["hit_rate"] == straight[2]["acquired"]["hit_rate"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dmta.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'dmta_run'`

- [ ] **Step 3: Write `scripts/dmta_run.py`**

```python
#!/usr/bin/env python3
"""Drive the DMTA loop: N rounds, resumable, both arms recorded every round.

The round history in `rounds.jsonl` is the state. Restarting replays it to rebuild
the acquisition penalty and the set of molecules already asked, then continues — so
stopping after round 2 and re-running produces the same round 3 (G4).

Stopping rules, both recorded rather than silent:
  * the oracle is exhausted (G3), or
  * the acquired arm has failed to beat the random arm for STALL_ROUNDS rounds.

The second one is why the random arm exists. Without it "the hit rate went down"
cannot be told apart from "the remaining pool got harder", which it does by
construction as the loop consumes its own best candidates.

    python scripts/dmta_run.py [panel] [campaign_id] [n_rounds]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dmta, registry                                      # noqa: E402
from src.oracle import TimeSplitOracle                              # noqa: E402
from src.panels import DEFAULT_PANEL, PanelSpec, get_panel          # noqa: E402

STALL_ROUNDS = 2
BATCH = 70


def run(panel: PanelSpec, campaign_id: str, n_rounds: int = 3,
        batch: int = BATCH, root: Path | None = None) -> list[dict]:
    context = dmta.build_context(panel)
    oracle = TimeSplitOracle(panel)

    history = [r.metrics for r in registry.rounds(campaign_id, root)]
    # Replay: re-asking the recorded molecules is what advances the oracle to the
    # state the recorded rounds left it in. The answers are already in the ledger.
    for record in registry.rounds(campaign_id, root):
        for smi in record.metrics.get("asked", []):
            oracle.answer(smi)

    for index in range(len(history), n_rounds):
        if oracle.exhausted:
            print(f"  round {index}: oracle exhausted, stopping (G3)")
            break
        penalty = dmta.penalty_from(history)
        result = dmta.run_round(panel, context, oracle, batch=batch,
                               penalty=penalty, seed=index, root=root)
        history.append(result)
        registry.append_round(campaign_id, kind="select",
                              model_ids=result["model_ids"],
                              n_molecules=result["acquired"]["n"] + result["random"]["n"],
                              metrics=result, root=root)
        print(f"  round {index}: acquired hit {result['acquired']['hit_rate']:.1%} "
              f"vs random {result['random']['hit_rate']:.1%} "
              f"(advantage {result['advantage']:+.1%}), "
              f"falsified {result['acquired']['falsification_rate']:.1%}, "
              f"{result['oracle_remaining']} left")

        recent = history[-STALL_ROUNDS:]
        if len(recent) == STALL_ROUNDS and all(r["advantage"] <= 0 for r in recent):
            print(f"  stopping: acquisition has not beaten random for "
                  f"{STALL_ROUNDS} rounds")
            break

    return history


def main() -> None:
    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    campaign_id = sys.argv[2] if len(sys.argv) > 2 else "jak-autoimmune"
    n_rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    print(f"DMTA loop — panel {panel.name}, campaign {campaign_id}, {n_rounds} rounds")
    run(panel, campaign_id, n_rounds=n_rounds)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Record the asked molecules so replay works**

The driver's replay needs `metrics["asked"]`. In `src/dmta.py`, add the list to `run_round`'s return dict, right beside `model_ids`:

```python
        "asked": sorted(both["smi"]) if len(both) else [],
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_dmta.py -q`
Expected: 9 PASS. If the resume test fails on `hit_rate`, check that `run_round` is seeded on the round index (`seed=index`) — a control arm drawn with a different seed makes the two runs genuinely different rather than revealing a resume bug.

- [ ] **Step 6: Commit**

```bash
git add scripts/dmta_run.py src/dmta.py tests/test_dmta.py
git commit -m "Drive the loop from the round history so a restart is not a different run"
```

---

### Task 5: Run three rounds and commit the case (D9)

**Files:**
- Modify: `.gitignore` (only if it catches `assets/cases/`)
- Create: `assets/cases/jak-autoimmune/rounds.jsonl` (committed)
- Create: `scripts/make_cases_index.py`
- Create: `CASES.md` (generated)
- Modify: `STATE.md` (new section), `scripts/reproduce.sh`

- [ ] **Step 1: Check whether the case directory is ignored**

```bash
git check-ignore -v assets/cases/jak-autoimmune/rounds.jsonl || echo "not ignored - good"
```
If a rule matches, append `!assets/cases/` to `.gitignore`. Do not remove the existing `data/registry/` rule — in-progress rounds stay runtime state; only finished cases are promoted.

- [ ] **Step 2: Run the loop for real**

```bash
python scripts/dmta_run.py jak jak-autoimmune 3
```
Expected: three lines, each reporting the acquired hit rate, the random hit rate, the advantage, the falsification rate and the remaining population. **Record whatever comes out.** If acquisition loses to random in every round, that is the result and it is reported as such (G5) — the loop turning is the deliverable, not the loop winning.

- [ ] **Step 3: Promote the finished case**

```bash
mkdir -p assets/cases/jak-autoimmune
cp data/registry/jak-autoimmune/rounds.jsonl assets/cases/jak-autoimmune/rounds.jsonl
```

- [ ] **Step 4: Write `scripts/make_cases_index.py`**

```python
#!/usr/bin/env python3
"""Generate CASES.md from the committed case files.

Generated, never hand-written: an index maintained by hand drifts from the cases it
indexes, and the whole point of committing the case is that the file is the record.

    python scripts/make_cases_index.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "assets" / "cases"


def main() -> None:
    lines = ["# Cases", "",
             "Generated by `python scripts/make_cases_index.py`. Do not edit by hand.",
             "",
             "| Case | Panel | Indication | Rounds | Last acquired hit rate | vs random |",
             "|---|---|---|---:|---:|---:|"]
    for directory in sorted(p for p in CASES.iterdir() if p.is_dir()):
        panel_file = directory / "panel.json"
        if not panel_file.exists():
            continue
        card = json.loads(panel_file.read_text())
        rounds_file = directory / "rounds.jsonl"
        rounds = ([json.loads(line) for line in
                   rounds_file.read_text().splitlines() if line.strip()]
                  if rounds_file.exists() else [])
        if rounds:
            last = rounds[-1].get("metrics", {})
            hit = f"{last.get('acquired', {}).get('hit_rate', float('nan')):.1%}"
            adv = f"{last.get('advantage', float('nan')):+.1%}"
        else:
            hit = adv = "—"
        lines.append(f"| [{card['case_id']}](assets/cases/{directory.name}/program.md) "
                     f"| {card['panel']} | {card['indication']} | {len(rounds)} "
                     f"| {hit} | {adv} |")
    (ROOT / "CASES.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {ROOT / 'CASES.md'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Generate the index and add the loop to `reproduce.sh`**

```bash
python scripts/make_cases_index.py
```

Append to `scripts/reproduce.sh`:

```bash
echo
echo "== DMTA loop, three rounds (STATE.md section 11) =="
echo "Selection, measured answers from the sealed eval fold, and a random control"
echo "arm every round. Models are frozen; only acquisition changes."
python scripts/dmta_run.py jak jak-reproduce 3
python scripts/make_cases_index.py
```

- [ ] **Step 6: Record the result in STATE.md**

Add a section `## 11. DMTA 한 바퀴 (2026-08-09 측정)` containing: the oracle population (1,078 cross-measured in the eval fold, 34.0 % base rate, 2.94× enrichment ceiling), a row per round with acquired hit rate / random hit rate / advantage / falsification rate, the fact that `model_ids` were identical in every round and why, and — if acquisition did not beat random — that statement in plain words with no softening.

- [ ] **Step 7: Run the full suite and commit**

```bash
python -m pytest tests/ -q
git add -A
git commit -m "Turn the loop three times and commit the case"
```

---

## Self-Review

**Spec coverage.** ROADMAP Wave 3 lists G (contract 1.2), D (oracle + N8 + N4), H (analyze: stopping rule, retrain trigger, N10), three rounds with G4/G5/G7, and the committed case with `CASES.md`. Covered: D in Task 1, H in Tasks 2–3, G4 in Task 4, G5/G7 in Task 2, the case and index in Task 5. **N4** (`answer_status`) landed in Wave 2's ledger and Task 2 writes through it. **N10** (recording gate-threshold movement) is **not applicable** — this loop does not move the gate threshold, because it does not retrain. **G (contract 1.2) is deliberately dropped**, with the reason stated in the differences table: the contract is the Colab handoff and this loop never crosses to Stage A.

**Placeholders.** None. Task 5 Steps 2 and 6 deliberately do not predict the numbers — they are a real run, and writing expected values would invite fitting the report to them. The instruction is to record what comes out, including a loss.

**Type consistency.** `TimeSplitOracle.answer -> dict[str, float] | None` is defined in Task 1 and consumed by `_arm_metrics` in Task 2. `dmta.Context` is produced by `build_context` and consumed by `score` and `run_round`. `penalty_from(list[dict]) -> dict[str, float]` (Task 3) returns bucket keys formatted `f"{low:.2f}-{high:.2f}"` by `_bucket` (Task 2), and `run_round` looks them up with the same helper. `run_round`'s return dict gains `"asked"` in Task 4 Step 4, which `dmta_run.run` reads in its replay loop.

**One risk the implementer should expect.** `build_context` refits three regressors on ~12.6k molecules, roughly 90 seconds. The test fixture is module-scoped for that reason; if `tests/test_dmta.py` starts taking more than about three minutes, check that the fixture is not being rebuilt per test.
