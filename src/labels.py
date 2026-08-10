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
COLUMNS = ["smi", "isoform", "value", "relation", "source", "answer_status",
           "scored_in_round", "sealed", "arrived_utc"]


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
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows, columns=COLUMNS)


def training_rows(panel: PanelSpec, root: Path | None = None) -> pd.DataFrame:
    """Answered, unsealed rows — the only ones a retrain may read (G6)."""
    frame = read(panel, root)
    if frame.empty:
        return frame
    keep = (frame["answer_status"] == "answered") & (~frame["sealed"].astype(bool))
    return frame[keep].reset_index(drop=True)
