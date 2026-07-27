"""Streamlit screening dashboard.

    streamlit run app.py

Three modes over the same model layer:

  * **Selectivity funnel** — run the wide, target-agnostic library down the JAK1
    cost funnel (Ro5/PAINS -> per-isoform QSAR -> gap S -> conformal + applicability
    domain) and export the picked cases as a versioned loop contract.
  * **Single molecule** — the same Tier-1+2 scoring for one compound entered as a
    SMILES or by name, reported against the library distribution.
  * **Target screen** — the v1 single-target pipeline: ChEMBL retrieval, PubChem
    expansion, drug-likeness filter, per-target QSAR, ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

# `src.pipeline` and the property-model bundle are imported lazily, inside the two
# Target-screen functions that use them: importing pipeline at module scope loaded
# the ~2.9 MB ESOL/Tox21 pickle on every app start, including the funnel and
# single-molecule modes that never touch it.

_ASSETS = Path(__file__).parent / "assets"

# Cell tints for the shortlist grid. Streamlit's dataframe parses these colour
# strings itself and cannot resolve CSS custom properties, so the two hues are
# repeated here from assets/tokens.css — keep them in step with the tokens.
_TINT_IN_DOMAIN = "rgba(1, 150, 104, 0.12)"    # --color-safe    #019668
_TINT_UNCERTAIN = "rgba(239, 68, 68, 0.12)"    # --color-danger  #EF4444

st.set_page_config(page_title="chem-predict — selectivity screening",
                   page_icon="🧪", layout="wide", initial_sidebar_state="auto")


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _stylesheet() -> str:
    return "\n".join((_ASSETS / name).read_text(encoding="utf-8")
                     for name in ("tokens.css", "app.css"))


def apply_theme() -> None:
    """Inject the token block + component CSS, flagged with Streamlit's active theme.

    `assets/tokens.css` switches its palette on `:root:has(.cp-theme-dark)`, so the
    injected components follow the in-app theme toggle rather than only the OS
    setting. Falls back to the light palette if the theme can't be read.
    """
    try:
        mode = st.context.theme.type
    except Exception:
        mode = "light"
    # st.html() sanitises <style> out of its payload, so the stylesheet goes in
    # through st.markdown(unsafe_allow_html=True); only the theme marker uses st.html.
    st.markdown(f"<style>{_stylesheet()}</style>", unsafe_allow_html=True)
    st.html(f"<span class='cp-theme-{mode}' hidden></span>")


# --------------------------------------------------------------------------- #
# Markup helpers
# --------------------------------------------------------------------------- #
def masthead(tags: list[str]) -> None:
    chips = "".join(f"<span class='cp-tag'>{t}</span>" for t in tags)
    st.html(
        "<header class='cp-masthead'>"
        "<span class='cp-wordmark'>chem<span>·</span>predict</span>"
        "<span class='cp-tagline'>Selectivity-aware screening funnel — CPU-only, "
        "reproducible, uncertainty-aware.</span>"
        f"<span class='cp-masthead-meta'>{chips}</span>"
        "</header>"
    )


def section_head(title: str, deck: str) -> None:
    st.html(f"<div class='cp-sectionhead'><h2>{title}</h2><p>{deck}</p></div>")


def stat_row(stats: list[tuple[str, str, str]], accent_first: bool = False) -> None:
    """Stat cards: (label, value, sub). Values must come from real computation."""
    cards = []
    for i, (label, value, sub) in enumerate(stats):
        cls = " is-accent" if accent_first and i == 0 else ""
        cards.append(
            f"<div class='cp-stat'><span class='cp-stat-label'>{label}</span>"
            f"<span class='cp-stat-value{cls}'>{value}</span>"
            f"<span class='cp-stat-sub'>{sub}</span></div>"
        )
    st.html(f"<div class='cp-stats'>{''.join(cards)}</div>")


def note(html: str) -> None:
    st.html(f"<div class='cp-note'>{html}</div>")


def idle_prompt(title: str, deck: str) -> None:
    """What a mode shows before the user has asked it to run anything."""
    section_head(title, deck)
    note(f"<strong>Nothing has run yet.</strong> {deck}")


def lead_card(row: pd.Series, target: str, offs: tuple[str, ...]) -> None:
    """Hero card for the best in-domain shortlist survivor.

    "Lead" here is a real rank (highest gap S among in-domain rows in an
    already gap-sorted shortlist), not invented copy — callers only pass a
    row once one exists.
    """
    with st.container(border=True):
        left, right = st.columns([1, 3])
        with left:
            mol = Chem.MolFromSmiles(row["smi"])
            if mol is not None:
                st.image(Draw.MolToImage(mol, size=(220, 180)), width="content")
        with right:
            st.html(
                "<div class='cp-lead-head'>"
                "<span class='cp-chip is-lead'>lead candidate</span>"
                "<span class='cp-chip is-in'>in-domain</span>"
                "</div>"
                f"<p class='cp-lead-smiles'>{row['smi']}</p>"
            )
            stat_row([
                (f"pred {target}", f"{row[f'pred_{target}']:.2f}",
                 f"potency vs {'/'.join(offs)}"),
                ("gap S", f"{row['gap']:+.2f}", "log-units over the worst off-isoform"),
                ("90% CI", f"[{row['gap_lo']:+.1f}, {row['gap_hi']:+.1f}]",
                 "split-conformal interval"),
                ("MPO", f"{row['mpo']:.2f}", "geometric mean of desirabilities"),
            ])


def colab_handoff(contract: dict, download_label: str) -> None:
    """Download the contract + open the deep-dive notebook at the pinned commit.

    The two belong together: the Colab link is built from the contract's own
    `code_version`, so the notebook that opens is the commit the contract was
    exported from. The notebook then reads the same `code_version` out of the
    uploaded contract and checks the clone out at it, so `assert_models_match`
    passes by construction rather than by luck.

    Both halves of that pin resolve through GitHub, so an unpushed commit is a
    dead handoff — checked here and said plainly rather than discovered in Colab.
    """
    from src.loop_contract import colab_url, commit_on_remote

    prov = contract["provenance"]
    ref = prov["code_version"]
    url = colab_url(contract)
    col_dl, col_colab = st.columns(2)
    with col_dl:
        st.download_button(download_label, data=json.dumps(contract, indent=2),
                           file_name=f"{contract['case_id']}.json",
                           mime="application/json", type="primary", width="stretch")
    with col_colab:
        if url:
            st.link_button(f"Open the deep dive in Colab @ {ref}", url, width="stretch")
        else:
            st.button("Colab link unavailable", disabled=True, width="stretch",
                      help="Needs a git origin remote and a known commit.")
    if url:
        st.caption(
            f"**1.** Download the contract. **2.** Open the notebook in Colab and "
            f"upload it in the first cell. **3.** The notebook reads `code_version` "
            f"from the contract and checks the clone out at commit `{ref}`, so "
            f"Stage A re-scores through these exact models — and refuses to run if "
            f"it cannot. It hands back an `A_rescore` contract at the end."
        )
        # The download -> upload round trip needs a filesystem, which a phone (or a
        # browser that blocks Colab's upload widget) does not usefully have. The
        # notebook's first cell takes a paste when no file is picked, so the whole
        # handoff can happen between two browser tabs.
        with st.expander("Paste it into Colab instead of downloading"):
            st.caption("Copy this, dismiss the notebook's file picker, and paste it "
                       "at the prompt. Same contract, same pin — no file involved.")
            st.code(json.dumps(contract, separators=(",", ":")), language="json")
        if commit_on_remote(ref) is False:
            st.warning(
                f"Commit `{ref}` is not on the GitHub remote. Colab reads both the "
                f"notebook and the clone from GitHub, not from this machine, so push "
                f"it before opening the deep dive — otherwise the link 404s and the "
                f"pinned checkout fails."
            )
    else:
        st.caption("No Colab link: this checkout has no GitHub origin remote or no "
                   "resolvable commit. The contract still downloads, and the notebook "
                   "runs anywhere the repo is checked out at this contract's commit.")
    provenance(
        [("case id", contract["case_id"]),
         ("molecules", str(len(contract["molecules"]))),
         ("conformal α", f"{prov['conformal_alpha']}"),
         ("code version", prov["code_version"])]
        + [(f"{iso} model", mid) for iso, mid in prov["model_ids"].items()]
    )


def provenance(items: list[tuple[str, str]]) -> None:
    body = "".join(
        f"<div class='cp-prov-item'><span class='cp-prov-key'>{k}</span>"
        f"<span class='cp-prov-val'>{v}</span></div>" for k, v in items
    )
    st.html(f"<footer class='cp-provenance'>{body}</footer>")


def mol_grid(df: pd.DataFrame, value_col: str, value_label: str,
             smiles_col: str = "canonical_smiles", id_col: str = "id",
             signed: bool = False, per_row: int = 4):
    """Structure grid with a legend per molecule. `signed` for quantities where the
    direction carries meaning (the selectivity gap), plain otherwise (pChEMBL)."""
    mols, legends = [], []
    for row in df.itertuples():
        mol = Chem.MolFromSmiles(getattr(row, smiles_col))
        if mol is None:
            continue
        mols.append(mol)
        label = getattr(row, id_col) if id_col else f"#{len(mols)}"
        value = getattr(row, value_col)
        legends.append(f"{label}   {value_label} {value:+.2f}" if signed
                       else f"{label}   {value_label} {value:.2f}")
    if not mols:
        return None
    return Draw.MolsToGridImage(mols, molsPerRow=per_row,
                                subImgSize=(240, 200), legends=legends)


# --------------------------------------------------------------------------- #
# Cached compute
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def run_funnel(panel_name: str):
    """Screen the wide library down one panel's selectivity funnel (cached).

    The status container is created *inside* the cached function on purpose: a
    cached function may only drive Streamlit elements it owns, so that its
    replayed output has somewhere to land on a cache hit.

    Each completed screen is recorded as a round in the campaign registry, which is
    what makes a rerun a *second round* rather than an indistinguishable repeat —
    the state Phase 3's active learning will read.
    """
    from src import registry
    from src.funnel import current_model_ids, screen_library
    from src.panels import get_panel

    panel = get_panel(panel_name)
    status = st.status("Screening the wide library…", expanded=True)
    try:
        shortlist = screen_library(panel, on_step=status.write)
    except Exception:
        status.update(label="Screening failed", state="error")
        raise
    status.update(label="Screen complete", state="complete", expanded=False)

    try:
        registry.append_round(
            campaign_id(panel_name), kind="screen",
            model_ids=current_model_ids(panel), n_molecules=len(shortlist),
            metrics={"in_domain": int(shortlist["in_domain"].sum()) if not shortlist.empty else 0,
                     "best_gap": float(shortlist["gap"].max()) if not shortlist.empty else None},
            scores=shortlist)
    except OSError:
        # A read-only deployment must still be able to screen; losing the audit
        # trail is worse than nothing but far better than losing the result.
        pass
    return shortlist


def campaign_id(panel_name: str) -> str:
    return f"{panel_name}-default"


@st.cache_resource(show_spinner=False)
def get_campaign(panel_name: str):
    """The campaign for a panel — built and persisted on first use.

    `cache_resource` rather than `cache_data`: a Campaign is a live object with a
    PanelSpec behind it, not a picklable frame.
    """
    from src import campaign as camp
    from src.panels import get_panel

    built = camp.build(get_panel(panel_name), campaign_id=campaign_id(panel_name))
    try:
        camp.save(built)
    except OSError:
        pass
    return built


_TIER_CHIP = {"validated": ("is-in", "validated"),
              "bootstrap": ("is-lead", "bootstrap"),
              "insufficient_data": ("is-out", "insufficient data")}


def tier_badge(campaign) -> None:
    """Show a campaign's validation tier and, always, why it has that tier.

    A badge with no reason beside it is just a different flavour of the unearned
    confidence the funnel replaced, so the sentence is not optional.
    """
    css, label = _TIER_CHIP.get(campaign.validation.tier, ("is-out", campaign.validation.tier))
    st.html(f"<div class='cp-tier'><span class='cp-chip {css}'>{label}</span>"
            f"<span class='cp-tier-why'>{campaign.validation.reason}</span></div>")


@st.cache_data(show_spinner=False)
def score_one(smiles: str):
    """Tier-1+2 score for one molecule, its gap percentile, and the reference size.

    The percentile is taken against the molecules that clear the binder gate — the
    population the funnel actually ranks — so the reference count is returned too:
    on the current Tox21 library that set is small, which makes the percentile
    coarse, and a headline number has to carry its own resolution.
    """
    from src.funnel import gap_percentile, library_gap_distribution, score_molecules

    scored = score_molecules([smiles])
    if scored.empty:
        return None, None, 0
    n_ref = len(library_gap_distribution())
    return scored.iloc[0], gap_percentile(float(scored.iloc[0]["gap"])), n_ref


@st.cache_data(show_spinner=False)
def resolve_input(text: str) -> tuple[str, str]:
    """Turn what the user typed into (standardised SMILES, provenance label).

    SMILES first, then a PubChem name lookup — so a valid structure never costs a
    network round trip, and a name still works. Both paths return the neutral
    parent: PubChem serves marketed drugs as salts, and the counterion would move
    the molecule out of the applicability domain it actually belongs in.
    """
    from src.data.pubchem_client import resolve_name
    from src.standardize import standardize

    text = text.strip()
    parent = standardize(text)
    if parent is not None:
        return parent, "parsed as SMILES"
    smiles, cid, title = resolve_name(text)
    return smiles, f"resolved by name → PubChem CID {cid} ({title})"


def funnel_cache_state() -> tuple[bool, str]:
    """(models + library available without training?, honest one-line cost estimate)."""
    from src.conformal import bundled_quantiles
    from src.data import library
    from src.panels import DEFAULT_PANEL as PANEL

    isoforms = PANEL.isoforms
    models = all(any((d / f"{iso}_reg.pkl").exists()
                     for d in (PANEL.model_cache, PANEL.model_bundled))
                 for iso in isoforms)
    domain = all(any((d / f"{iso}.npz").exists()
                     for d in (PANEL.ad_cache, PANEL.ad_bundled))
                 for iso in isoforms)
    ready = models and any(p.exists() for p in (library.CACHE, library.BUNDLED))
    if ready and domain and bundled_quantiles(PANEL).exists():
        return True, ("Models, library, conformal calibration and the applicability "
                      "reference all ship with the repo. This run screens the library "
                      "end to end — a few seconds.")
    if ready:
        return True, ("Models and library are available; the conformal calibration and "
                      "the applicability reference are built on this run — a minute or two.")
    return False, ("Nothing is cached on this machine: the run downloads the ChEMBL "
                   "activity sets and trains the three isoform regressors from "
                   "scratch. Budget 15–30 minutes on a cold CPU.")


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def render_funnel(panel_name: str = "jak") -> None:
    from src.funnel import screen_to_contract
    from src.panels import get_panel
    from src.selectivity import POTENCY_FLOOR

    panel = get_panel(panel_name)
    TARGET, OFFS = panel.target, panel.offs
    campaign = get_campaign(panel_name)

    section_head(
        f"{TARGET} selectivity funnel",
        f"A diverse, target-agnostic library run down the cost funnel — Ro5/PAINS, "
        f"then per-isoform QSAR to the selectivity gap <code>S = {TARGET} − max({', '.join(OFFS)})</code>, "
        f"then conformal intervals and applicability domain on the survivors only — "
        f"to a shortlist that is both selective and in-domain."
    )
    tier_badge(campaign)

    started_key = f"funnel_started_{panel_name}"
    _, cost_line = funnel_cache_state()
    if not st.session_state.get(started_key):
        note(f"<strong>Before you start.</strong> {cost_line}")
        if st.button("Run the funnel", type="primary", key=f"start_funnel_{panel_name}"):
            st.session_state[started_key] = True
            st.rerun()
        return

    try:
        sl = run_funnel(panel_name)
    except Exception as err:                      # surface, don't swallow
        st.error(f"{type(err).__name__}: {err}")
        st.session_state[started_key] = False
        return

    if sl.empty:
        st.warning(f"No molecule in the library cleared the potency floor "
                   f"(predicted {TARGET} pChEMBL ≥ {POTENCY_FLOOR}). Nothing to rank.")
        return

    n_dom = int(sl["in_domain"].sum())
    stat_row([
        ("Shortlist", f"{len(sl)}", "selective + drug-like survivors"),
        ("In-domain", f"{n_dom}", f"{n_dom / len(sl):.0%} of the shortlist is trustworthy"),
        ("Best gap S", f"{sl['gap'].max():+.2f}", "log-units over the worst off-isoform"),
        ("Potency floor", f"{POTENCY_FLOOR:.1f}", f"minimum predicted {TARGET} pChEMBL"),
    ], accent_first=True)

    in_domain_rows = sl[sl["in_domain"]]
    if not in_domain_rows.empty:
        lead_card(in_domain_rows.iloc[0], TARGET, OFFS)

    st.html(
        "<div class='cp-legend'><span class='cp-chip is-in'>in-domain</span>"
        "both applicability signals agree — the prediction is interpolation"
        "<span class='cp-chip is-out'>uncertain</span>"
        "at least one isoform model is extrapolating</div>"
    )

    view = pd.DataFrame({
        "SMILES": sl["smi"],
        f"pred {TARGET}": sl[f"pred_{TARGET}"].round(2),
        "gap S": sl["gap"].round(2),
        "gap 90% CI": [f"[{lo:+.1f}, {hi:+.1f}]" for lo, hi in zip(sl["gap_lo"], sl["gap_hi"])],
        "binder": sl["binder_prob"].round(2),
        "domain": sl["verdict"].map({"in_domain": "in-domain", "uncertain": "uncertain"}),
        "MPO": sl["mpo"].round(2),
    })
    # Tint the domain cell: it is the call the table exists to support, and a
    # legend alone makes the reader translate every row by hand.
    styled = view.style.map(
        lambda v: f"background-color: {_TINT_IN_DOMAIN if v == 'in-domain' else _TINT_UNCERTAIN}",
        subset=["domain"],
    )
    event = st.dataframe(
        styled, width="stretch", hide_index=True, on_select="rerun",
        selection_mode="multi-row", key=f"funnel_table_{panel_name}",
        column_config={
            "SMILES": st.column_config.TextColumn("SMILES", width="large"),
            f"pred {TARGET}": st.column_config.NumberColumn(
                f"pred {TARGET}", format="%.2f", help=f"Predicted {TARGET} pChEMBL"),
            "gap S": st.column_config.NumberColumn(
                "gap S", format="%+.2f",
                help="Selectivity gap in log-units; +1 ≈ 10× selective"),
            "gap 90% CI": st.column_config.TextColumn(
                "gap 90% CI", help="Split-conformal interval, calibrated on the "
                                   "measured gap and widened by distance from training"),
            "binder": st.column_config.NumberColumn(
                "binder", format="%.2f",
                help=f"Tier-0.5 binder-gate probability P({TARGET} binder). Every shortlisted "
                     f"molecule cleared the gate; the value is its margin above the cutoff."),
            "domain": st.column_config.TextColumn(
                "domain", help="Tanimoto-distance and descriptor-leverage signals combined"),
            "MPO": st.column_config.NumberColumn(
                "MPO", format="%.2f",
                help="Geometric mean of QED, predicted solubility and (1 − tox alert). "
                     "Near zero means one property is disqualifying — the ranking is "
                     "still on gap S, so read this as a veto, not a tie-breaker."),
        },
    )

    picked = list(event.selection.rows)
    section_head("Export a case",
                 "Select rows above to hand them to the offline deep dive. The contract "
                 "pins the exact models, the conformal level and the code version, so "
                 "Stage&nbsp;A can assert it re-scores through the identical models.")

    if not picked:
        st.info("No rows selected — pick one or more molecules in the table to build a contract.")
    else:
        chosen = sl.iloc[picked].reset_index(drop=True)
        img = mol_grid(chosen, "gap", "gap S", smiles_col="smi", id_col="", signed=True)
        if img is not None:
            st.image(img, width="stretch")
        colab_handoff(screen_to_contract(chosen, panel,
                                        campaign_id=campaign.campaign_id),
                      f"Export contract — {len(picked)} molecule"
                      f"{'s' if len(picked) != 1 else ''}")


def render_single(query: str) -> None:
    from src.data.pubchem_client import NameNotFound
    from src.funnel import screen_to_contract
    from src.selectivity import OFFS, POTENCY_FLOOR, TARGET

    section_head(
        "Score one molecule",
        f"The same Tier-1 and Tier-2 scoring the wide screen runs, on a single "
        f"compound — predicted {TARGET}/{'/'.join(OFFS)} potency, the selectivity gap "
        f"<code>S</code> with its conformal interval, the applicability-domain verdict "
        f"and the property axes."
    )
    if not query:
        st.info("Enter a SMILES string or a compound name in the sidebar "
                "(e.g. `ruxolitinib`, or `CCO`).")
        return

    try:
        smiles, provenance_label = resolve_input(query)
    except NameNotFound as err:
        st.error(str(err))
        return
    except Exception as err:
        st.error(f"Structure lookup failed ({type(err).__name__}: {err}). "
                 "A name lookup needs network access to PubChem; a SMILES string does not.")
        return

    try:
        row, percentile, n_ref = score_one(smiles)
    except Exception as err:
        st.error(f"{type(err).__name__}: {err}")
        return
    if row is None:
        st.error("RDKit could not build a molecule from that structure.")
        return

    # The binder gate reframes everything below it. The regressors emit a pChEMBL
    # for any structure, so a non-binder still gets a gap and a percentile — but the
    # wide screen would drop it before ranking, and saying so up front stops the
    # percentile from reading as endorsement of a molecule the models never saw.
    if "is_binder" in row and not bool(row["is_binder"]):
        st.warning(
            f"**Binder gate: unlikely JAK binder** — P(binder) "
            f"{float(row['binder_prob']):.2f}. This molecule looks unlike the JAK actives "
            f"the models were trained on, so the wide screen would gate it out before Tier 1. "
            f"The gap, percentile and interval below are shown for completeness — read them "
            f"as *what the regressors extrapolate*, not as a selectivity claim."
        )

    # Percentile leads. A lone gap value is weakly supported — the 90 % interval
    # spans ~±2 pchembl and usually crosses zero — while the *ranking* is what was
    # validated (Spearman 0.80 against measured gaps). Reporting the rank first
    # keeps the headline on the claim the evidence actually supports.
    gap, lo, hi = float(row["gap"]), float(row["gap_lo"]), float(row["gap_hi"])
    stat_row([
        ("Gap percentile", f"{percentile:.1f}",
         f"rank by gap S among the {n_ref} molecules clearing the binder gate"),
        ("Gap S", f"{gap:+.2f}", "log-units over the worst off-isoform"),
        (f"pred {TARGET}", f"{row[f'pred_{TARGET}']:.2f}",
         f"potency floor {POTENCY_FLOOR:.1f} — {'clears' if row['meets_floor'] else 'below'}"),
        ("Domain", "in-domain" if row["in_domain"] else "uncertain",
         "both applicability signals agree" if row["in_domain"] else "a model is extrapolating"),
    ], accent_first=True)

    # A single-molecule box invites famous drugs, and every marketed JAK inhibitor
    # is in ChEMBL and therefore in the training set. A Tanimoto nearest neighbour
    # of 1.0 means the model is reciting a molecule it was fitted on, which is a
    # different claim from a prediction and has to be said out loud.
    memorised = [i for i in (TARGET, *OFFS) if float(row[f"nn_sim_{i}"]) >= 0.999]
    if memorised:
        st.info(
            f"**This molecule is in the training set** for {', '.join(memorised)} "
            f"(Tanimoto nearest neighbour 1.000). The numbers below are a *fit*, not a "
            f"forecast — the model has seen this structure and its measured potency. "
            f"Scaffold-split metrics describe performance on chemotypes the model has "
            f"never seen, so they do not apply here."
        )

    mol = Chem.MolFromSmiles(smiles)
    left, right = st.columns([1, 2])
    with left:
        if mol is not None:
            st.image(Draw.MolToImage(mol, size=(320, 260)), width="content")
        st.caption(provenance_label)
        st.code(smiles, language=None)
    with right:
        st.markdown(f"**Selectivity gap** `{gap:+.2f}`  90% CI `[{lo:+.2f}, {hi:+.2f}]`")
        st.caption(
            f"The interval is calibrated against the *measured* gap and scaled by how "
            f"far this molecule sits from the training set (nearest neighbour "
            f"{float(row[f'nn_sim_{TARGET}']):.2f}), so it widens where the model is "
            f"extrapolating instead of quoting one width everywhere.")
        if lo <= 0.0 <= hi:
            st.warning(
                f"The interval crosses zero, so this molecule cannot be called "
                f"{TARGET}-selective on its own evidence. What the model supports is its "
                f"**rank**: it sits at the {percentile:.1f}th percentile of the {n_ref} "
                f"gate-clearing molecules"
                + (" — a small reference set, so read the percentile as coarse."
                   if n_ref < 100 else ".") +
                f" Selectivity ranking was validated separately (Spearman 0.80 vs "
                f"measured gaps, 4.5× enrichment).")
        else:
            st.success(f"The interval excludes zero — the predicted direction of "
                       f"selectivity is supported at 90 % confidence.")

        st.markdown("**Per-isoform prediction — confidence matrix**")
        st.caption("Tanimoto NN shaded by nearest-neighbour similarity to the training "
                    "set — the applicability-domain signal in fingerprint space.")
        iso_df = pd.DataFrame({
            "isoform": [TARGET, *OFFS],
            "pred pChEMBL": [round(float(row[f"pred_{i}"]), 2) for i in (TARGET, *OFFS)],
            "90% CI": [f"[{row[f'lo_{i}']:.2f}, {row[f'hi_{i}']:.2f}]" for i in (TARGET, *OFFS)],
            "Tanimoto NN": [round(float(row[f"nn_sim_{i}"]), 3) for i in (TARGET, *OFFS)],
            "in domain": ["yes" if row[f"in_domain_{i}"] else "no" for i in (TARGET, *OFFS)],
        })
        # Intensity-mapped green — --color-safe (#019668) at an alpha scaled to the
        # similarity value itself, so the grid reads its own confidence at a glance.
        styled = iso_df.style.map(
            lambda v: f"background-color: rgba(1, 150, 104, {0.08 + 0.35 * min(1.0, max(0.0, v)):.2f})",
            subset=["Tanimoto NN"],
        ).format({"pred pChEMBL": "{:.2f}", "Tanimoto NN": "{:.3f}"})
        st.dataframe(styled, width="stretch", hide_index=True)

    section_head("Property axes (MPO)",
                 "Selectivity is not the only thing that disqualifies a molecule. These "
                 "are combined as a geometric mean, so one unacceptable property pulls "
                 "the score down rather than being averaged away.")
    stat_row([
        ("MPO", f"{row['mpo']:.2f}" if pd.notna(row["mpo"]) else "—", "geometric mean of desirabilities"),
        ("QED", f"{row['qed']:.2f}" if pd.notna(row["qed"]) else "—", "RDKit drug-likeness"),
        ("logS", f"{row['logS_pred']:.2f}" if pd.notna(row["logS_pred"]) else "—",
         "predicted, ESOL model"),
        ("Tox alert", f"{row['tox_prob']:.2f}" if pd.notna(row["tox_prob"]) else "—",
         "P(any Tox21 hit) — a screening alert, not a safety verdict"),
    ])

    section_head("Hand this molecule to the deep dive",
                 "The contract pins the models, the conformal level and the code version; "
                 "the Colab link opens the notebook at that same commit.")
    # The single-molecule view scores through the default (validated JAK) campaign,
    # so its export is tagged with that campaign rather than left unattributed.
    colab_handoff(screen_to_contract(pd.DataFrame([row]),
                                     campaign_id=get_campaign("jak").campaign_id),
                  "Export contract — 1 molecule")


def render_campaign(panel_name: str) -> None:
    """Mode 1: campaigns — pick a panel, see what backs it, then screen it.

    This replaces the v1 single-target screen, which ranked molecules by a composite
    score validated against nothing and carried none of the trust layer the funnel
    was built to add (standardisation, the binder gate, applicability domain,
    conformal intervals). Re-framing it as "start a campaign on a panel" keeps the
    thing it was actually for — pointing the tool at chemistry other than JAK — while
    routing it through the validated cascade instead of around it.

    The v1 pipeline itself (`src/pipeline.py`) is untouched and still runs from its
    CLI; it is no longer wired into the dashboard.
    """
    from src import registry
    from src.panels import disjointness_report

    campaign = get_campaign(panel_name)
    panel = campaign.panel

    section_head(
        "Campaigns",
        "A campaign is a panel, a library and the models screening it, carrying the "
        "evidence that says how far to trust its output. The JAK campaign is the "
        "validated one; anything else is a bootstrap until its gates have been re-run."
    )
    tier_badge(campaign)

    stat_row([
        ("Panel", panel.label or panel.name, f"{panel.target} vs {', '.join(panel.offs)}"),
        ("Cross-measured", f"{campaign.validation.n_cross_measured}",
         "molecules with a measured gap — what calibration is built from"),
        ("Library", campaign.library, "target-agnostic ChEMBL panel, shared by campaigns"),
        ("Conformal", f"{1 - campaign.conformal_alpha:.0%}", "nominal interval coverage"),
    ], accent_first=True)

    report = disjointness_report(panel)
    clashes = report["gate_negative_conflicts"] + report["library_conflicts"]
    if clashes:
        st.warning(
            f"**Leakage risk.** {', '.join(clashes)} appears both in this panel and in "
            "the binder gate's negative basket or the screening library, so the gate "
            "would be scoring molecules it was trained on. Results are not trustworthy "
            "until the panel or those sets are changed.")
    else:
        note("<strong>Leakage check passed.</strong> No panel member appears in the "
             "binder gate's negative basket or in the wide library's target list, so "
             "the gate's verdict on a library molecule is a prediction, not recall.")

    if not campaign.validation.supports_selectivity_claim:
        st.error(
            "This campaign cannot make a selectivity claim: there is not enough "
            "cross-measured data to calibrate a gap interval. Screening it would "
            "produce a ranking with no honest uncertainty attached, which is exactly "
            "the failure mode the funnel exists to prevent.")
        return

    rounds = registry.rounds(campaign.campaign_id)
    if rounds:
        st.caption(f"Round history — {len(rounds)} recorded for this campaign.")
        st.dataframe(
            pd.DataFrame([{"round": r.index, "kind": r.kind, "when": r.created[:19],
                           "molecules": r.n_molecules, "code": r.code_version}
                          for r in reversed(rounds)]),
            width="stretch", hide_index=True)
    else:
        st.caption("No rounds recorded yet — running the funnel below logs the first.")

    if not campaign.model_ids:
        st.info(
            f"**Models are not built on this machine for {panel.name}.** Its ChEMBL "
            f"datasets ship with the repo, so the evidence above is real, but the "
            f"{len(panel.isoforms)} regressors, the applicability reference and this "
            "panel's own binder gate are trained on first run — several minutes, "
            "then cached. Only the validated JAK panel ships pre-trained.")

    provenance([("campaign", campaign.campaign_id), ("panel", panel.name),
                ("code version", campaign.code_version),
                ("models", ", ".join(sorted(campaign.model_ids.values())) or "not built here")])

    st.divider()
    render_funnel(panel_name)


# --------------------------------------------------------------------------- #
apply_theme()

with st.sidebar:
    st.markdown("### Mode")
    # segmented_control returns None when the active chip is clicked again.
    mode = st.segmented_control("Mode", ["Selectivity funnel", "Single molecule", "Campaigns"],
                                default="Selectivity funnel",
                                label_visibility="collapsed") or "Selectivity funnel"
    st.divider()

    # Every mode runs behind an explicit submit. Switching modes used to fire the
    # workflow immediately off the default text — landing on "Single molecule" scored
    # ruxolitinib and "Target screen" ran a full EGFR screen — so simply looking at a
    # tab cost seconds of compute nobody had asked for.
    if mode == "Single molecule":
        st.markdown("### Score a molecule")
        with st.form("single_form"):
            single_query = st.text_input("SMILES or compound name", value="ruxolitinib")
            single_go = st.form_submit_button("Score molecule", type="primary")
        st.caption("A SMILES string is parsed locally. Anything else is looked up by "
                   "name on PubChem and reduced to its neutral parent, so a salt form "
                   "scores as the drug it is.")
    elif mode == "Campaigns":
        from src.panels import PANELS

        st.markdown("### Choose a campaign")
        with st.form("campaign_form"):
            panel_choice = st.selectbox(
                "Selectivity panel", sorted(PANELS),
                format_func=lambda n: PANELS[n].label or n)
            campaign_go = st.form_submit_button("Open campaign", type="primary")
        st.caption("A panel is a target plus the off-targets it must be selective "
                   "against — the gap S is undefined without them. JAK is validated; "
                   "other panels build their models on first open and stay bootstrap "
                   "until their gates are re-run.")
    else:
        st.markdown("### Funnel")
        _, cost_line = funnel_cache_state()
        st.caption(cost_line)
        if st.session_state.get("funnel_started") and st.button("Reset run"):
            st.session_state["funnel_started"] = False
            run_funnel.clear()
            st.rerun()

masthead(["CPU-only", "campaign registry"] if mode == "Campaigns"
         else ["CPU-only", "JAK1 / JAK2 / JAK3", "conformal 90%"])

if mode == "Selectivity funnel":
    render_funnel()
elif mode == "Single molecule":
    # The submit flag is remembered, so later reruns (a table selection, an export)
    # keep showing the result instead of resetting to the idle state.
    if single_go:
        st.session_state["single_submitted"] = True
        st.session_state["single_query"] = single_query
    if st.session_state.get("single_submitted"):
        render_single(st.session_state.get("single_query", single_query))
    else:
        idle_prompt("Score a molecule",
                    "Enter a SMILES string or a compound name in the sidebar, then press "
                    "<strong>Score molecule</strong>. Scoring one structure takes about "
                    "0.3&nbsp;s; the library percentile it is ranked against is precomputed.")
else:
    if campaign_go:
        st.session_state["campaign_submitted"] = True
        st.session_state["campaign_panel"] = panel_choice
    if st.session_state.get("campaign_submitted"):
        render_campaign(st.session_state.get("campaign_panel", panel_choice))
    else:
        idle_prompt("Campaigns",
                    "Pick a selectivity panel in the sidebar and press "
                    "<strong>Open campaign</strong>. The campaign card shows what backs "
                    "the panel — cross-measured data, models, leakage check and its "
                    "validation tier — before anything is screened.")
