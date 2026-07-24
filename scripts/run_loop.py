"""Run the funnel loop end-to-end on one real case and emit the worked artifacts.

    B (wide screen) -> SELECT (top in-domain) -> A (generate + re-score) -> report

Writes examples/loop_case_B_export.json, examples/loop_case_A_rescore.json,
examples/loop_report.md, and figures/loop_before_after.png.

    python scripts/run_loop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import deep_dive, funnel                     # noqa: E402
from src.loop_contract import write_contract          # noqa: E402

EX = ROOT / "examples"
FIG = ROOT / "figures" / "loop_before_after.png"


def _strip(ax, x, gaps, dom):
    jitter = np.random.default_rng(0).normal(0, 0.05, len(gaps))
    ax.scatter(x + jitter[dom], gaps[dom], c="#2ca02c", s=24, label="in-domain")
    ax.scatter(x + jitter[~dom], gaps[~dom], c="#bbbbbb", s=18, label="uncertain")


def main() -> None:
    EX.mkdir(exist_ok=True)
    print("B: screening the wide library…")
    sl = funnel.screen_library()
    picks = sl[sl["in_domain"]].head(3)
    if picks.empty:
        picks = sl.head(3)                              # fall back to top gap if none in-domain
    b_contract = funnel.screen_to_contract(picks.reset_index(drop=True))
    write_contract(b_contract, EX / "loop_case_B_export.json")
    print(f"SELECT: exported {len(picks)} case(s) -> examples/loop_case_B_export.json")

    print("A: generating analogues + re-scoring through the SAME models…")
    result = deep_dive.run_deep_dive(b_contract)
    a_contract = deep_dive.rescore_contract(result)
    write_contract(a_contract, EX / "loop_case_A_rescore.json")
    report = deep_dive.report_markdown(result)
    (EX / "loop_report.md").write_text(report)
    print(report)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _strip(ax, 0.0, result.before["gap"].to_numpy(), result.before["in_domain"].to_numpy())
    _strip(ax, 1.0, result.after["gap"].to_numpy(), result.after["in_domain"].to_numpy())
    ax.axhline(1.0, ls="--", color="grey", lw=1)
    ax.text(-0.35, 1.03, "≥10× selective", color="grey", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"before\n(selected, n={len(result.before)})",
                        f"after\n(generated, n={len(result.after)})"])
    ax.set_ylabel(f"selectivity gap  S = {result.target} − max off")
    ax.set_title("Loop closed: generated analogues re-scored through the same models")
    handles = [plt.Line2D([], [], marker="o", ls="", color="#2ca02c", label="in-domain"),
               plt.Line2D([], [], marker="o", ls="", color="#bbbbbb", label="uncertain")]
    ax.legend(handles=handles, loc="upper left")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"Saved -> {FIG}")


if __name__ == "__main__":
    main()
