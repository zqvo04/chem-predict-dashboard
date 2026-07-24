"""STEP 7: the loop data contract (the B -> SELECT -> A -> re-score handoff).

One versioned JSON object carries a selected case from the CPU dashboard (Stage B)
to the offline Colab notebook (Stage A) and back. It pins the exact models
(`model_ids`), the conformal level (`conformal_alpha`), and the code version, so
Stage A can *assert* it is re-scoring through the identical Stage-B models — the
thing that makes "re-score through the same models" verifiable rather than hoped.

Schema is documented in WORKFLOW.md section 6.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = "1.0"


def code_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).resolve().parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def model_id(chembl_id: str, model) -> str:
    """Stable id pinning a specific fitted model: '<chembl_id>@<content-hash>'."""
    digest = hashlib.sha1(pickle.dumps(model)).hexdigest()[:10]
    return f"{chembl_id}@{digest}"


def build_contract(shortlist: pd.DataFrame, target: str, offs: list[str],
                   model_ids: dict[str, str], alpha: float,
                   case_id: str | None = None, stage: str = "B_export") -> dict:
    """Assemble a contract from a scored shortlist frame (one row per molecule).

    Expected columns: smi, pred_<iso>, lo_<iso>, hi_<iso>, in_domain_<iso>,
    gap, gap_lo, gap_hi, meets_floor, verdict. Optional: origin, parent_smiles.
    """
    isoforms = [target, *offs]
    molecules = []
    for r in shortlist.itertuples(index=False):
        d = r._asdict()
        molecules.append({
            "smiles": d["smi"],
            "origin": d.get("origin", "screen"),
            "parent_smiles": d.get("parent_smiles"),
            "per_isoform": {
                iso: {"pred_pchembl": round(float(d[f"pred_{iso}"]), 3),
                      "interval": [round(float(d[f"lo_{iso}"]), 3), round(float(d[f"hi_{iso}"]), 3)],
                      "in_domain": bool(d[f"in_domain_{iso}"])}
                for iso in isoforms
            },
            "selectivity": {
                "gap": round(float(d["gap"]), 3),
                "gap_interval": [round(float(d["gap_lo"]), 3), round(float(d["gap_hi"]), 3)],
                "meets_potency_floor": bool(d["meets_floor"]),
                "verdict": d["verdict"],
            },
            "deep_dive": None,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id or f"{target}-selective-{code_version()}",
        "target_isoform": target,
        "off_isoforms": list(offs),
        "provenance": {
            "created": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "model_ids": dict(model_ids),
            "conformal_alpha": alpha,
            "code_version": code_version(),
        },
        "molecules": molecules,
    }


def write_contract(contract: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(contract, indent=2))


def read_contract(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def assert_models_match(contract: dict, model_ids: dict[str, str]) -> None:
    """Stage A guard: refuse to re-score unless the models match the export exactly."""
    pinned = contract["provenance"]["model_ids"]
    if pinned != model_ids:
        raise ValueError(
            f"Model mismatch: contract pinned {pinned}, current models {model_ids}. "
            "Re-scoring must use the identical Stage-B models.")
