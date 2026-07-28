"""Panel specs: what a selectivity campaign screens *for* and *against* (STEP 15).

A selectivity funnel is not defined by a target. The quantity the whole cascade
ranks on,

    S = pred(target) - max_off pred(off-target)

is undefined without off-targets, so the unit the funnel generalises over is a
**panel** — a target plus the related off-targets it must be selective against —
not a single target. An arbitrary single-target screen is Mode 1's job, not the
funnel's.

Everything downstream is derived per panel and namespaced under the panel's name:
the per-isoform datasets, the regressors, the AD references, the binder gate and
the conformal calibration. Two panels therefore never share a cache entry or a
model file, which is what stops a second panel from silently ranking against the
first one's artifacts.

The JAK panel deliberately keeps the paths the committed assets already use
(`assets/jak/`, `assets/models/jak/`), so parameterising the code moved no bytes
and the validated JAK numbers are reproduced from the same files.

CLI (list panels + their leakage report against the gate/library targets):
    python -m src.panels
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PanelSpec:
    """One selectivity panel: a target, its off-targets, and where its assets live.

    Frozen and hashable (hence `members` as a tuple of pairs rather than a dict) so
    it can be an `lru_cache` key — every cached artifact in the funnel is keyed on
    the panel it belongs to.
    """

    name: str                                   # asset namespace, e.g. "jak"
    target: str                                 # the isoform to be selective *for*
    offs: tuple[str, ...]                       # the isoforms to be selective *against*
    members: tuple[tuple[str, str], ...]        # (isoform, ChEMBL target id)
    label: str = ""                             # human-readable, for the UI
    root: Path = _ROOT                          # repo root; overridden in tests

    def __post_init__(self) -> None:
        named = {n for n, _ in self.members}
        missing = [i for i in self.isoforms if i not in named]
        if missing:
            raise ValueError(
                f"Panel {self.name!r} has no ChEMBL id for {', '.join(missing)}; "
                "every member of a panel needs one or its dataset cannot be built.")
        if not self.offs:
            raise ValueError(
                f"Panel {self.name!r} has no off-targets. A selectivity gap is a "
                "difference against something — a single target is a Mode 1 screen, "
                "not a funnel panel.")

    @property
    def isoforms(self) -> tuple[str, ...]:
        """Panel members, target first — the order every per-isoform table uses."""
        return (self.target, *self.offs)

    @property
    def chembl_ids(self) -> dict[str, str]:
        return dict(self.members)

    # --- where this panel's artifacts live -------------------------------- #
    # Same runtime-cache / committed-bundle split the rest of the repo uses:
    # `data/` is gitignored and rebuildable, `assets/` is committed so a fresh
    # deploy skips retrieval and retraining.

    # `root` is a field rather than a module constant so a test can point a panel at
    # a tmp_path and exercise the real cache/bundle fallback instead of
    # monkeypatching module-level directories, which stopped working once the paths
    # became panel-derived.

    @property
    def data_cache(self) -> Path:
        return self.root / "data" / self.name

    @property
    def data_bundled(self) -> Path:
        return self.root / "assets" / self.name

    @property
    def model_cache(self) -> Path:
        return self.root / "data" / "models" / self.name

    @property
    def model_bundled(self) -> Path:
        return self.root / "assets" / "models" / self.name

    @property
    def ad_cache(self) -> Path:
        return self.root / "data" / "ad_reference" / self.name

    @property
    def ad_bundled(self) -> Path:
        return self.root / "assets" / "ad_reference" / self.name


JAK = PanelSpec(
    name="jak", target="JAK1", offs=("JAK2", "JAK3"),
    members=(("JAK1", "CHEMBL2835"), ("JAK2", "CHEMBL2971"), ("JAK3", "CHEMBL2148")),
    label="JAK1 vs JAK2/JAK3",
)

# A second panel, added to test that the generalisation is real rather than
# nominal. PI3K isoform selectivity is a genuine problem (idelalisib is the
# delta-selective drug), and — the reason this panel and not EGFR/HER2 — none of
# the four catalytic subunits appears in the binder gate's negative basket or in
# the wide library's target list, so it inherits the funnel's leakage discipline
# without having to renegotiate either. `disjointness_report` checks that rather
# than assuming it.
PI3K = PanelSpec(
    name="pi3k", target="PIK3CD", offs=("PIK3CA", "PIK3CB", "PIK3CG"),
    members=(("PIK3CD", "CHEMBL3130"), ("PIK3CA", "CHEMBL4005"),
             ("PIK3CB", "CHEMBL3145"), ("PIK3CG", "CHEMBL3267")),
    label="PI3K-delta vs alpha/beta/gamma",
)

PANELS: dict[str, PanelSpec] = {p.name: p for p in (JAK, PI3K)}
DEFAULT_PANEL = JAK


def get_panel(name: str) -> PanelSpec:
    """Look up a registered panel by its namespace name."""
    try:
        return PANELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown panel {name!r}; registered panels are {sorted(PANELS)}.") from None


def panel_for(target: str, offs) -> PanelSpec:
    """The registered panel matching a (target, off-targets) pair.

    A loop contract records the isoform names, not the panel, because that is what
    schema 1.0 pinned and contracts already in the wild must keep validating. This
    resolves them back to a panel so Stage A can re-score through the same models.
    Off-target order is not significant, so it is compared as a set.
    """
    wanted = (target, frozenset(offs))
    for panel in PANELS.values():
        if (panel.target, frozenset(panel.offs)) == wanted:
            return panel
    raise ValueError(
        f"No registered panel screens {target} against {sorted(offs)}; "
        f"registered panels are {sorted(PANELS)}.")


def disjointness_report(panel: PanelSpec) -> dict:
    """Is this panel disjoint from the binder gate's negatives and the wide library?

    STEP 10 and STEP 13 were both fixed for the same reason: when the molecules a
    model is judged on overlap the molecules it was trained on, the verdict is
    recall dressed as prediction. Adding a panel is exactly the moment that mistake
    is easiest to repeat, so the check is a function rather than a comment — a new
    panel whose target sits in the gate's negative basket would train the gate to
    call its own actives non-binders, and one that sits in the library's target
    list would screen a haystack built from its own answers.

    Returns the offending target names per source; empty lists mean disjoint. The
    ChEMBL id, not the label, is what is compared — the same protein under two
    names would otherwise read as disjoint.
    """
    from .data.library import LIBRARY_TARGETS
    from .data.negatives import NEGATIVE_TARGETS

    ids = set(panel.chembl_ids.values())
    return {
        "panel": panel.name,
        "gate_negative_conflicts": sorted(n for n, c in NEGATIVE_TARGETS.items() if c in ids),
        "library_conflicts": sorted(n for n, c in LIBRARY_TARGETS.items() if c in ids),
    }


def library_molecule_overlap(panel: PanelSpec) -> dict | None:
    """How many wide-library molecules are this panel's own gate positives?

    `disjointness_report` compares *target identifiers*, which is necessary and not
    sufficient: a library drawn from other targets can still contain molecules that
    were separately assayed on this panel. STEP 13 handled that for JAK by dropping
    known actives **by SMILES**, but a SMILES match misses two spellings of the same
    structure, and no such exclusion was ever run for a panel added later.

    Answering it needs a molecule-level join across every target at once, which is
    exactly what the evidence store exists for (STEP 16). Returns None when the store
    is absent — "not checked" is the honest answer there, not "disjoint".
    """
    try:
        from .data import db as _db

        if not _db.exists():
            return None
        with _db.session(read_only=True) as con:
            if not con.execute("SELECT count(*) FROM library_member").fetchone()[0]:
                return None
            n_lib = con.execute("SELECT count(*) FROM library_member").fetchone()[0]
            positives = con.execute(
                """SELECT count(DISTINCT lm.inchikey)
                   FROM library_member lm JOIN activity a USING (inchikey)
                   WHERE a.target_chembl_id IN ?
                     AND a.pchembl_value >= 6
                     AND (a.standard_relation = '=' OR a.standard_relation IS NULL)""",
                [list(panel.chembl_ids.values())]).fetchone()[0]
        return {"library_molecules": n_lib, "gate_positives_in_library": positives,
                "fraction": positives / n_lib if n_lib else 0.0}
    except Exception:
        return None


def usable_negative_targets(panel: PanelSpec) -> dict[str, str]:
    """The gate's negative basket with any panel member removed.

    A panel member in the negative basket would be labelled a presumed non-binder
    while also being a positive — so it is dropped rather than the panel refused.
    The basket is deliberately larger than it needs to be for this reason.
    """
    from .data.negatives import NEGATIVE_TARGETS

    ids = set(panel.chembl_ids.values())
    return {n: c for n, c in NEGATIVE_TARGETS.items() if c not in ids}


def _main() -> None:
    for panel in PANELS.values():
        report = disjointness_report(panel)
        clashes = report["gate_negative_conflicts"] + report["library_conflicts"]
        print(f"{panel.name:6} {panel.target} vs {'/'.join(panel.offs)}")
        print(f"       assets   {panel.data_bundled.relative_to(panel.root)}, "
              f"{panel.model_bundled.relative_to(panel.root)}")
        print(f"       disjoint {'yes' if not clashes else 'NO — ' + ', '.join(clashes)}"
              f"  (negatives left: {len(usable_negative_targets(panel))})")


if __name__ == "__main__":
    _main()
