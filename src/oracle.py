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
