"""Campaign: a panel, a library and the models screening it, with a trust grade.

A campaign is the first-class thing the repo was missing. Before STEP 15 every mode
was a pure function over module constants, so "which target, through which models,
against which library, with what evidence" was implicit in the import graph and
could not be varied, recorded or compared. `Campaign` makes it explicit and
`src/registry.py` makes it durable.

**Relationship to `loop_contract.json`.** The contract stays what it was — one
case's B->A snapshot, carrying molecules — and gains a `campaign_id` pointing back
here (schema 1.1). The two have different lifetimes: a campaign spans many rounds
and outlives any single export, a contract is one export. Merging them would have
put a molecule list inside a campaign definition and invalidated every contract
already exported.

**The validation tier is the point.** v1 was replaced because it ranked molecules
with a composite score validated against nothing; the funnel earned its ranking by
measuring it. Letting anyone spin up a new panel re-opens exactly that hole — a
bootstrap campaign looks identical to the JAK one in the UI unless something says
otherwise. So every campaign carries a `Validation` with a tier, and the tier is
derived from what has actually been measured, not asserted by whoever made it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .panels import PanelSpec, get_panel
from .registry import CAMPAIGN_FILE, campaign_dir

VALIDATED = "validated"
BOOTSTRAP = "bootstrap"
INSUFFICIENT = "insufficient_data"

# What a campaign reports when its data release is unknown — the honest answer for
# the committed assets/ parquet, whose ChEMBL release was never written down.
UNRECORDED = "unrecorded"

# Panels whose numbers are gate-checked and written up in VALIDATION.md. Membership
# is a claim about evidence that exists in the repo, so it is a hand-maintained
# list — a panel cannot promote itself by having a lot of data.
VALIDATED_PANELS = {"jak"}

# The smallest cross-measured set from which a 90 %-nominal gap interval can be
# calibrated to anything narrower than "the largest residual we saw".
#
# `calibrate_gap` splits cross-measured 80/20 into train-pool/test, then that pool
# 75/25 into proper-train/calibration, then alternates the calibration half into a
# sigma-fit part and a quantile part — so the quantile is taken over roughly 10 % of
# the cross-measured molecules. The finite-sample conformal level is
# ceil((n+1)(1-alpha))/n, which only drops below 1.0 (i.e. becomes a real quantile
# rather than the maximum) at n >= 19. Ten percent of 200 is 20, so 200 is the floor.
MIN_CROSS_MEASURED = 200


@dataclass
class Validation:
    """Why a campaign's output should or should not be believed.

    `tier` is the badge the UI shows; `reason` is the sentence it shows beside it,
    because a badge with no explanation is just a different kind of unearned
    confidence.
    """

    tier: str
    reason: str
    n_cross_measured: int = 0
    evidence: list[str] = field(default_factory=list)

    @property
    def supports_selectivity_claim(self) -> bool:
        """May this campaign state a directional selectivity result at all?

        Below the calibration floor the gap has a point estimate but no honest
        interval, and a gap without an interval is the pre-STEP-14 mistake — a
        number that reads as a finding and is not one.
        """
        return self.tier in (VALIDATED, BOOTSTRAP)


def assess(panel: PanelSpec, n_cross_measured: int) -> Validation:
    """Grade a panel from what has actually been measured for it."""
    if panel.name in VALIDATED_PANELS:
        return Validation(
            tier=VALIDATED,
            reason=("Gate-checked end to end and written up in VALIDATION.md: "
                    "regression, selectivity gap vs measured gap, conformal coverage, "
                    "applicability domain and the binder gate."),
            n_cross_measured=n_cross_measured,
            evidence=["VALIDATION.md STEP 3-14"])
    if n_cross_measured < MIN_CROSS_MEASURED:
        return Validation(
            tier=INSUFFICIENT,
            reason=(f"Only {n_cross_measured} molecules are measured on every panel "
                    f"member (need {MIN_CROSS_MEASURED} to calibrate a gap interval), "
                    "so predictions carry no honest selectivity interval."),
            n_cross_measured=n_cross_measured)
    return Validation(
        tier=BOOTSTRAP,
        reason=(f"Models, applicability domain and conformal calibration were built "
                f"for this panel from {n_cross_measured} cross-measured molecules, but "
                "none of the funnel's gates has been re-run against it. Treat the "
                "ranking as a hypothesis, not a validated result."),
        n_cross_measured=n_cross_measured)


@dataclass
class Campaign:
    """One panel + library + model set, with its trust grade and round history."""

    campaign_id: str
    panel_name: str
    library: str
    model_ids: dict[str, str]
    conformal_alpha: float
    validation: Validation
    code_version: str = "unknown"
    # Which data release the models were trained on. Pinning code and models but not
    # data left a campaign unreproducible: the same code against two ChEMBL releases
    # trains on different molecules and nothing noticed (STEP 16).
    data_version: str = UNRECORDED
    created: str = ""

    @property
    def panel(self) -> PanelSpec:
        return get_panel(self.panel_name)

    def to_dict(self) -> dict:
        return {"campaign_id": self.campaign_id, "panel_name": self.panel_name,
                "library": self.library, "model_ids": self.model_ids,
                "conformal_alpha": self.conformal_alpha,
                "validation": asdict(self.validation),
                "code_version": self.code_version,
                "data_version": self.data_version, "created": self.created}

    @classmethod
    def from_dict(cls, d: dict) -> "Campaign":
        return cls(campaign_id=d["campaign_id"], panel_name=d["panel_name"],
                   library=d.get("library", "wide"), model_ids=d.get("model_ids", {}),
                   conformal_alpha=d.get("conformal_alpha", 0.10),
                   validation=Validation(**d["validation"]),
                   code_version=d.get("code_version", "unknown"),
                   data_version=d.get("data_version", UNRECORDED),
                   created=d.get("created", ""))


def data_version(panel: PanelSpec) -> str:
    """Which data release this panel's datasets came from.

    Returns the `source_id`s present in the evidence store for the panel's targets
    (e.g. `chembl_37`), or `"unrecorded"` when the store is absent — which is the
    honest answer for the committed `assets/` parquet, since the release they were
    built from was never written down. That gap is the whole reason STEP 16 exists,
    so it is reported rather than papered over with a plausible-looking default.

    Never raises: the deployed app has no store and must still describe a campaign.
    """
    try:
        from .data import db as _db

        if not _db.exists():
            return UNRECORDED
        with _db.session(read_only=True) as con:
            ids = con.execute(
                "SELECT DISTINCT source_id FROM activity "
                "WHERE target_chembl_id IN ?  ORDER BY 1",
                [list(panel.chembl_ids.values())]).fetchall()
        return "+".join(r[0] for r in ids) if ids else UNRECORDED
    except Exception:
        # duckdb missing, store locked by a concurrent backfill, schema mismatch —
        # none of which should stop a campaign being described.
        return UNRECORDED


def models_built(panel: PanelSpec) -> bool:
    """Does every panel member already have a regressor on this machine?"""
    return all(any((d / f"{iso}_reg.pkl").exists()
                   for d in (panel.model_cache, panel.model_bundled))
               for iso in panel.isoforms)


def build(panel: PanelSpec, campaign_id: str | None = None, library: str = "wide",
          alpha: float = 0.10, use_cache: bool = True) -> Campaign:
    """Assemble a campaign for a panel, grading it from its measured data.

    Reads the cross-measured count and the deployed model ids, so the resulting
    campaign describes the models that exist right now rather than an intention.

    Model ids are read only when the models already exist. `current_model_ids`
    trains what it cannot find, and a panel whose datasets ship but whose models do
    not (the bootstrap case) would otherwise spend minutes fitting regressors merely
    because someone opened its campaign card — the same "work nobody asked for" the
    mode-switch fix removed from the app.
    """
    from .data import panel_data
    from .funnel import current_model_ids
    from .loop_contract import code_version

    n_cross = len(panel_data.build_cross_measured(panel, use_cache=use_cache))
    return Campaign(
        campaign_id=campaign_id or f"{panel.name}-{panel.target.lower()}",
        panel_name=panel.name, library=library,
        model_ids=current_model_ids(panel, use_cache=use_cache) if models_built(panel) else {},
        conformal_alpha=alpha,
        validation=assess(panel, n_cross),
        code_version=code_version(),
        data_version=data_version(panel),
        created=datetime.now(timezone.utc).isoformat(),
    )


def save(campaign: Campaign, root: Path | None = None) -> Path:
    """Write the campaign definition into its registry directory."""
    directory = campaign_dir(campaign.campaign_id, root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CAMPAIGN_FILE
    path.write_text(json.dumps(campaign.to_dict(), indent=2), encoding="utf-8")
    return path


def load(campaign_id: str, root: Path | None = None) -> Campaign:
    path = campaign_dir(campaign_id, root) / CAMPAIGN_FILE
    if not path.exists():
        raise FileNotFoundError(f"No campaign {campaign_id!r} in the registry.")
    return Campaign.from_dict(json.loads(path.read_text(encoding="utf-8")))
