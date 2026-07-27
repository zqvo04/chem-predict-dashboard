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
import os
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# 1.1 adds `campaign_id` to provenance (STEP 15). Minor bump, not major: the
# validator keys compatibility on the major version, so a 1.0 contract still reads
# and a 1.1 contract still validates against code expecting 1.0. The contract stays
# what it always was — one case's B->A snapshot — and merely records which campaign
# it was cut from; the campaign itself lives in the registry, not in here.
SCHEMA_VERSION = "1.1"

NOTEBOOK_PATH = "notebooks/deep_dive.ipynb"
_COLAB_BASE = "https://colab.research.google.com/github"

# Deployment fallbacks. The container serving the dashboard has neither a git
# binary nor a .git directory, so on the web — where most people meet the funnel —
# the commit would read "unknown" and the Colab handoff would silently disappear.
# Render injects the RENDER_* pair; CHEM_PREDICT_* is the generic escape hatch the
# Dockerfile bakes in. See README "Deploying".
_COMMIT_ENV = ("CHEM_PREDICT_COMMIT", "RENDER_GIT_COMMIT")
_REPO_ENV = ("CHEM_PREDICT_REPO", "RENDER_GIT_REPO_SLUG")


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args],
                                       cwd=Path(__file__).resolve().parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value and value != "unknown":
            return value
    return None


def code_version() -> str:
    """Short commit of the running code — from git, else from the deployment env."""
    sha = _git("rev-parse", "--short", "HEAD")
    if sha:
        return sha
    env = _env(_COMMIT_ENV)
    return env[:7] if env else "unknown"       # the env pair carries the full sha


def repo_slug() -> str | None:
    """'owner/repo' from the git origin remote, else the deployment env, else None."""
    raw = _git("config", "--get", "remote.origin.url") or _env(_REPO_ENV)
    if not raw:
        return None
    slug = raw.removesuffix(".git").rstrip("/")
    parts = [p for p in slug.replace(":", "/").split("/") if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else None


def colab_url(contract: dict) -> str | None:
    """Open the deep-dive notebook in Colab at the contract's exact code version.

    This is what turns the B->A handoff from "download a file, find the notebook
    yourself" into one link, and it does more than save a click: pinning the URL to
    the contract's `code_version` means the notebook Colab opens is the same commit
    the contract was exported from. The notebook then pins its own checkout to the
    same `code_version` (it reads it out of the uploaded contract), so
    `assert_models_match` passes by construction instead of by luck.

    Returns None when the commit or the remote is unknown. The commit must be
    pushed for the link to resolve — Colab reads it from GitHub, not from disk.
    """
    slug, ref = repo_slug(), contract["provenance"]["code_version"]
    if not slug or ref == "unknown":
        return None
    return f"{_COLAB_BASE}/{slug}/blob/{ref}/{NOTEBOOK_PATH}"


def commit_on_remote(ref: str) -> bool | None:
    """Is the pinned commit reachable from the `origin` remote? None if unknowable.

    Colab fetches both the notebook and the clone from GitHub, so a contract
    exported from an unpushed commit yields a link that 404s and a checkout that
    cannot be pinned. Checking here is what lets the handoff say so instead of
    handing over a dead link. Reads the local view of the remote refs — it does not
    fetch, so a commit pushed from elsewhere since the last fetch reads as absent.
    """
    if ref == "unknown":
        return None
    out = _git("branch", "-r", "--contains", ref, "--format=%(refname)")
    if out is None:
        return None
    return any(line.startswith("refs/remotes/origin/") for line in out.splitlines())


# A fixed, committed probe set. Model identity here is **behavioural**: two models
# that predict the same values on these molecules are interchangeable for
# re-scoring, which is exactly what `assert_models_match` needs to guarantee.
# Spans kinase-inhibitor chemotypes, simple aromatics/aliphatics and marketed drugs
# so two genuinely different regressors cannot agree across all of them by accident.
# NEVER change this tuple: it would change every model id and invalidate every
# contract ever exported.
PROBE_SMILES = (
    "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC[C@@H](CC#N)n1cc(-c2ncnc3[nH]ccc23)cn1",
    "C[C@@H]1CCN(C(=O)CC#N)C[C@@H]1N(C)c1ncnc2[nH]ccc12",
    "COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC", "O=C(O)c1ccccc1O",
    "c1ccc2[nH]ccc2c1", "c1ccc2ncncc2c1", "CCCCCCCC", "OCC(O)CO",
    "Clc1ccccc1", "c1ccncc1", "O=S(=O)(N)c1ccccc1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
)


@lru_cache(maxsize=1)
def _probe_matrix():
    """Fingerprints of the probe set, built once per process."""
    from .models.features import morgan_matrix

    X, mask = morgan_matrix(list(PROBE_SMILES))
    if not mask.all():
        raise RuntimeError("A probe SMILES failed to parse; model ids would be unstable.")
    return X


def model_id(chembl_id: str, model) -> str:
    """Stable id pinning a specific fitted model: '<chembl_id>@<behaviour-hash>'.

    The digest is over the model's **predictions on a fixed probe set**, not over
    its pickle. Hashing `pickle.dumps(model)` looked content-addressed but was not
    idempotent: the same model serialised at different round-trip depths produced
    different bytes (verified — dump/load/dump changes the digest), so a model that
    had passed through an extra cache layer on the app side hashed differently from
    the identical model loaded straight from disk in Colab. `assert_models_match`
    then rejected the deep dive over serialisation noise rather than model drift.

    Predictions are exactly what the guard cares about and are stable across
    round-trips, sklearn versions and platforms; they are formatted at fixed
    precision so last-bit float noise cannot move the id either.
    """
    preds = np.asarray(model.predict(_probe_matrix()), dtype=float).ravel()
    payload = ";".join(f"{v:.6f}" for v in preds)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"{chembl_id}@{digest}"


def build_contract(shortlist: pd.DataFrame, target: str, offs: list[str],
                   model_ids: dict[str, str], alpha: float,
                   case_id: str | None = None, stage: str = "B_export",
                   campaign_id: str | None = None) -> dict:
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
            "campaign_id": campaign_id,
        },
        "molecules": molecules,
    }


def write_contract(contract: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(contract, indent=2))


def read_contract(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def validate_contract(contract: dict) -> None:
    """Stage A guard: refuse anything that is not a contract this code can read.

    The file crosses machines by hand — downloaded from the dashboard, uploaded into
    Colab — so the wrong file, a truncated one, or one written by a future schema all
    arrive as plausible JSON. Failing here names what is wrong; failing later shows a
    `KeyError` inside the scoring code.
    """
    if not isinstance(contract, dict):
        raise ValueError(f"Not a loop contract: expected a JSON object, got {type(contract).__name__}.")

    missing = [k for k in ("schema_version", "case_id", "target_isoform",
                           "off_isoforms", "provenance", "molecules") if k not in contract]
    if missing:
        raise ValueError(f"Not a loop contract: missing {', '.join(missing)}. "
                         "Upload the JSON exported by the dashboard's SELECT step.")

    major = str(contract["schema_version"]).split(".")[0]
    if major != SCHEMA_VERSION.split(".")[0]:
        raise ValueError(f"Contract schema {contract['schema_version']} is incompatible "
                         f"with this code (schema {SCHEMA_VERSION}).")

    prov = contract["provenance"]
    missing = [k for k in ("model_ids", "conformal_alpha", "code_version", "stage")
               if k not in prov]
    if missing:
        raise ValueError(f"Contract provenance is missing {', '.join(missing)} — "
                         "without it the re-scoring cannot be pinned to the export.")

    if not contract["molecules"]:
        raise ValueError("Contract carries no molecules: select at least one row "
                         "in the funnel table before exporting.")
    for i, m in enumerate(contract["molecules"]):
        if not isinstance(m, dict) or not m.get("smiles"):
            raise ValueError(f"Molecule {i} has no SMILES — the contract is malformed.")


def assert_models_match(contract: dict, model_ids: dict[str, str]) -> None:
    """Stage A guard: refuse to re-score unless the models match the export exactly."""
    pinned = contract["provenance"]["model_ids"]
    if pinned != model_ids:
        raise ValueError(
            f"Model mismatch: contract pinned {pinned}, current models {model_ids}. "
            "Re-scoring must use the identical Stage-B models.")
