"""STEP 8: the deep dive + loop closure (Stage A logic, CPU-runnable).

Loads a B_export loop contract, asserts it pins the *current* Stage-B models,
generates analogues over each selected case, **re-scores them through the same
`src` scoring** (funnel.score_molecules), and emits a before/after comparison plus
an A_rescore contract. The Colab notebook wraps this; the docking step and a GPU
generator are optional swap-ins at the marked seams.

The honest deliverable is a distribution shift — does generation move the gap `S`
up while staying in-domain? — reported as an in-silico hypothesis, never a hit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import funnel
from .generate import generate_analogues
from .loop_contract import assert_models_match, build_contract, read_contract


@dataclass
class LoopResult:
    target: str
    offs: list[str]
    before: pd.DataFrame       # the selected cases, re-scored
    after: pd.DataFrame        # generated analogues, re-scored
    model_ids: dict


def run_deep_dive(contract: dict, max_analogues_per_case: int = 30,
                  use_cache: bool = True) -> LoopResult:
    target = contract["target_isoform"]
    offs = tuple(contract["off_isoforms"])

    # Loop integrity: re-score only if the models are identical to the export.
    current = funnel.current_model_ids(target, offs, use_cache=use_cache)
    assert_models_match(contract, current)

    seeds = [m["smiles"] for m in contract["molecules"]]
    parent_of, generated = {}, []
    for smi in seeds:
        for a in generate_analogues(smi, max_analogues_per_case):
            if a not in parent_of:
                parent_of[a] = smi
                generated.append(a)

    before = funnel.score_molecules(seeds, target, offs, use_cache=use_cache)
    before["origin"] = "screen"
    before["parent_smiles"] = None

    after = funnel.score_molecules(generated, target, offs, use_cache=use_cache)
    after["origin"] = "generated"
    after["parent_smiles"] = after["smi"].map(parent_of)

    return LoopResult(target=target, offs=list(offs), before=before, after=after,
                      model_ids=current)


def rescore_contract(result: LoopResult, alpha: float = 0.10) -> dict:
    """A_rescore contract from the generated, re-scored analogues."""
    return build_contract(result.after, result.target, result.offs,
                          result.model_ids, alpha, stage="A_rescore")


def _stats(df: pd.DataFrame) -> dict:
    return {"n": len(df), "gap_median": float(df["gap"].median()),
            "gap_max": float(df["gap"].max()),
            "frac_selective": float((df["gap"] >= 1.0).mean()),
            "frac_in_domain": float(df["in_domain"].mean())}


def report_markdown(result: LoopResult) -> str:
    b, a = _stats(result.before), _stats(result.after)
    ad_ok = result.after[result.after["in_domain"]]
    best = ad_ok.sort_values("gap", ascending=False).head(1)
    lines = [
        f"# Deep-dive report — {result.target} selectivity over {', '.join(result.offs)}",
        "",
        "| set | n | median gap | max gap | % ≥10× selective | % in-domain |",
        "|-----|--:|:----------:|:-------:|:----------------:|:-----------:|",
        f"| before (selected) | {b['n']} | {b['gap_median']:+.2f} | {b['gap_max']:+.2f} | "
        f"{b['frac_selective']:.0%} | {b['frac_in_domain']:.0%} |",
        f"| after (generated) | {a['n']} | {a['gap_median']:+.2f} | {a['gap_max']:+.2f} | "
        f"{a['frac_selective']:.0%} | {a['frac_in_domain']:.0%} |",
        "",
    ]
    if not best.empty:
        r = best.iloc[0]
        lines += [f"Best in-domain analogue: `{r['smi']}` "
                  f"(gap {r['gap']:+.2f}, parent `{r['parent_smiles']}`).", ""]
    else:
        lines += ["No generated analogue landed in-domain — the honest outcome for "
                  "this case; the hypotheses need a closer scaffold or wet validation.", ""]
    lines.append("_In-silico hypothesis — requires wet-lab validation._")
    return "\n".join(lines)


def run_from_file(contract_path: str, **kw) -> tuple[LoopResult, dict, str]:
    """Convenience: load a contract file, run the loop, return (result, A_contract, report)."""
    result = run_deep_dive(read_contract(contract_path), **kw)
    return result, rescore_contract(result), report_markdown(result)
