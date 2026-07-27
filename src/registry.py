"""The run registry: what was screened, when, through which models (STEP 15).

Until now every mode was a pure function call, so nothing survived a rerun. That
is fine for a one-shot screen and fatal for the thing this repo is being built
toward: active learning is by definition a loop where round N's results change
round N+1's selection, which cannot exist without state that outlives a process.

Layout — one directory per campaign under a gitignored runtime root, matching the
repo's existing `data/` (rebuildable) vs `assets/` (committed) split:

    data/registry/<campaign_id>/campaign.json          the campaign definition
    data/registry/<campaign_id>/rounds.jsonl           append-only round history
    data/registry/<campaign_id>/round_<n>_scores.parquet   that round's scored table

**Append-only JSONL rather than a database.** The operations this actually needs
are "add a round" and "read the rounds in order"; JSONL does both in a few lines,
adds no dependency, survives a partial write (a torn last line is one lost round,
not a corrupt file) and stays greppable and diffable. Wide score tables go to
parquet beside it, because that is the one part that is genuinely tabular and
large.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ROOT = _ROOT / "data" / "registry"     # runtime state, gitignored

CAMPAIGN_FILE = "campaign.json"
ROUNDS_FILE = "rounds.jsonl"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _check_id(campaign_id: str) -> str:
    """Reject ids that would escape the registry root or collide with its files.

    Campaign ids reach here from user input (Mode 1 names a new campaign), and they
    are used as directory names, so this is a path-traversal guard, not decoration.
    """
    if not _SAFE_ID.match(campaign_id or ""):
        raise ValueError(
            f"Invalid campaign id {campaign_id!r}: use letters, digits, '.', '_' or '-', "
            "starting with a letter or digit.")
    return campaign_id


def campaign_dir(campaign_id: str, root: Path | None = None) -> Path:
    return (root or REGISTRY_ROOT) / _check_id(campaign_id)


@dataclass
class Round:
    """One recorded step of a campaign.

    `kind` is what happened — "screen" (a library run), "select" (molecules picked
    and exported to a contract), "rescore" (Stage-A loop closure). `metrics` is
    free-form on purpose: what is worth recording differs per kind, and pinning a
    schema now would be guessing at Phase 3's acquisition functions.
    """

    index: int
    kind: str
    created: str
    code_version: str
    model_ids: dict[str, str]
    n_molecules: int
    metrics: dict = field(default_factory=dict)
    scores_path: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {"index": self.index, "kind": self.kind, "created": self.created,
                "code_version": self.code_version, "model_ids": self.model_ids,
                "n_molecules": self.n_molecules, "metrics": self.metrics,
                "scores_path": self.scores_path, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict) -> "Round":
        return cls(index=d["index"], kind=d["kind"], created=d["created"],
                   code_version=d.get("code_version", "unknown"),
                   model_ids=d.get("model_ids", {}),
                   n_molecules=d.get("n_molecules", 0),
                   metrics=d.get("metrics", {}),
                   scores_path=d.get("scores_path"), notes=d.get("notes", ""))


def rounds(campaign_id: str, root: Path | None = None) -> list[Round]:
    """Every recorded round, oldest first. Missing campaign or file -> empty list.

    A truncated final line is skipped rather than raised on: a half-written record
    is a lost round, and refusing to read the other twenty because of it would make
    the crash worse than it needs to be.
    """
    path = campaign_dir(campaign_id, root) / ROUNDS_FILE
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(Round.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def append_round(campaign_id: str, kind: str, model_ids: dict[str, str],
                 n_molecules: int, metrics: dict | None = None,
                 scores: pd.DataFrame | None = None, notes: str = "",
                 root: Path | None = None) -> Round:
    """Record a round; returns it with its assigned index.

    The index is derived from what is already on disk rather than tracked in memory,
    so two processes appending to the same campaign produce two rounds rather than
    one overwriting the other's number.
    """
    from .loop_contract import code_version

    directory = campaign_dir(campaign_id, root)
    directory.mkdir(parents=True, exist_ok=True)
    index = len(rounds(campaign_id, root))

    scores_path = None
    if scores is not None and not scores.empty:
        name = f"round_{index}_scores.parquet"
        scores.to_parquet(directory / name, index=False)
        scores_path = name

    record = Round(index=index, kind=kind,
                   created=datetime.now(timezone.utc).isoformat(),
                   code_version=code_version(), model_ids=dict(model_ids),
                   n_molecules=int(n_molecules), metrics=dict(metrics or {}),
                   scores_path=scores_path, notes=notes)
    with open(directory / ROUNDS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return record


def round_scores(campaign_id: str, index: int, root: Path | None = None
                 ) -> pd.DataFrame | None:
    """The scored table a round wrote, or None if it recorded no table."""
    for r in rounds(campaign_id, root):
        if r.index == index:
            if not r.scores_path:
                return None
            return pd.read_parquet(campaign_dir(campaign_id, root) / r.scores_path)
    raise KeyError(f"Campaign {campaign_id!r} has no round {index}.")


def list_campaigns(root: Path | None = None) -> list[str]:
    """Campaign ids with a definition on disk, alphabetically."""
    base = root or REGISTRY_ROOT
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir()
                  if d.is_dir() and (d / CAMPAIGN_FILE).exists())
