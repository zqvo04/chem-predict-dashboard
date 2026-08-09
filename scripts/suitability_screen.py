#!/usr/bin/env python3
"""Stage 0.5: is this panel one the funnel can actually be validated on?

Five counts, all computable before any biology is argued, all offline. A panel that
fails them is not a panel the selectivity claim can be tested on, whatever the
target rationale says.

  members            three or more paralogs — a gap needs something to be a gap against
  cross-measured     molecules measured on every member; below MIN_CROSS_MEASURED the
                     conformal gap interval cannot be calibrated (src/campaign.py)
  censored in lib    measured non-binders inside the wide library — the negative arm
                     of the falsification audit. STATE.md section 6d measured 491 for
                     JAK and 39 for PI3K, so this does not transfer between panels
  library leakage    library molecules that are this panel's own actives; a leak turns
                     prediction into recall
  scaffolds          Murcko diversity of the cross-measured set

The leakage join is computed from the committed parquets rather than through
`panels.library_molecule_overlap`, which answers the same question but needs the
duckdb store restored — that is a dev-only dependency, and a screen that silently
returns "not checked" in CI is worse than no screen.

Deliberately blind to safety evidence, so there is no circularity with Stage 0: the
screen says whether a panel is *computable*, the Program card says whether it is
*worth computing*.

    python scripts/suitability_screen.py [panel]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.campaign import MIN_CROSS_MEASURED                         # noqa: E402
from src.data import panel_data                                     # noqa: E402
from src.panels import (PANELS, PanelSpec, disjointness_report,     # noqa: E402
                        get_panel)

RDLogger.DisableLog("rdApp.*")

ACTIVE_PCHEMBL = 6.0


def _library_leakage(panel: PanelSpec) -> int:
    """Library molecules carrying an uncensored panel measurement at pchembl >= 6.

    The molecule-level version of the check. `disjointness_report` compares target
    identifiers, which is necessary and not sufficient: a library drawn from other
    targets can still contain molecules separately assayed on this panel.
    """
    root = panel.root / "assets" / "evidence"
    act = pd.read_parquet(root / "activity.parquet")
    member = pd.read_parquet(root / "library_member.parquet")
    hits = act[act["target_chembl_id"].isin(panel.chembl_ids.values())
               & (act["pchembl_value"] >= ACTIVE_PCHEMBL)
               & (act["standard_relation"].isna() | (act["standard_relation"] == "="))]
    return int(member["inchikey"].isin(set(hits["inchikey"])).sum())


def screen(panel: PanelSpec) -> dict:
    cross = panel_data.build_cross_measured(panel)
    scaffolds = set()
    for smi in cross["smi"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
    conflicts = disjointness_report(panel)
    return {
        "n_members": len(panel.isoforms),
        "n_cross_measured": len(cross),
        "censored_in_library": int(panel_data.censored_library_molecules(panel)["inchikey"].nunique()),
        "library_leakage": _library_leakage(panel),
        "target_conflicts": (conflicts["gate_negative_conflicts"]
                             + conflicts["library_conflicts"]),
        "n_scaffolds": len(scaffolds),
    }


def problems(result: dict) -> list[str]:
    out = []
    if result["n_members"] < 3:
        out.append("fewer than 3 members — a selectivity gap needs paralogs")
    if result["n_cross_measured"] < MIN_CROSS_MEASURED:
        out.append(f"cross-measured {result['n_cross_measured']} < {MIN_CROSS_MEASURED} "
                   "— the gap interval cannot be calibrated")
    if result["censored_in_library"] < 100:
        out.append(f"only {result['censored_in_library']} censored library molecules "
                   "— the falsification audit's negative arm is too thin")
    if result["library_leakage"]:
        out.append(f"{result['library_leakage']} library molecules carry this panel's own "
                   "measurements — that is recall, not prediction")
    if result["target_conflicts"]:
        out.append(f"panel members appear in the gate basket or library targets: "
                   f"{result['target_conflicts']}")
    return out


def main() -> None:
    names = [sys.argv[1]] if len(sys.argv) > 1 else list(PANELS)
    for name in names:
        panel = get_panel(name)
        result = screen(panel)
        print(f"\n{'=' * 62}\nPanel {panel.name} — {panel.label}\n{'=' * 62}")
        for key, value in result.items():
            print(f"  {key:22} {value}")
        found = problems(result)
        print("  VERDICT                " + ("suitable" if not found else "NOT suitable"))
        for problem in found:
            print(f"    - {problem}")


if __name__ == "__main__":
    main()
