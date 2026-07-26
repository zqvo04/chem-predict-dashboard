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

from src.models.property_models import load_property_models
from src.pipeline import screen

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
def run_funnel():
    """Screen the wide library down the JAK-selectivity funnel (cached).

    The status container is created *inside* the cached function on purpose: a
    cached function may only drive Streamlit elements it owns, so that its
    replayed output has somewhere to land on a cache hit.
    """
    from src.funnel import screen_library

    status = st.status("Screening the wide library…", expanded=True)
    try:
        shortlist = screen_library(on_step=status.write)
    except Exception:
        status.update(label="Screening failed", state="error")
        raise
    status.update(label="Screen complete", state="complete", expanded=False)
    return shortlist


@st.cache_data(show_spinner=False)
def run_screen(target: str, expand: bool, max_records: int):
    tgt, model, scored = screen(target, expand=expand, max_records=max_records, use_cache=True)
    return tgt, model.metrics, scored


@st.cache_data(show_spinner=False)
def score_one(smiles: str):
    """Tier-1+2 score for a single molecule, plus its percentile in the library."""
    from src.funnel import gap_percentile, score_molecules

    scored = score_molecules([smiles])
    if scored.empty:
        return None, None
    return scored.iloc[0], gap_percentile(float(scored.iloc[0]["gap"]))


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
    from src import applicability as ad
    from src.conformal import BUNDLED_QUANTILES
    from src.data import library
    from src.models import isoform_regressor as ir
    from src.selectivity import OFFS, TARGET

    models = all(any((d / f"{iso}_reg.pkl").exists()
                     for d in (ir.MODEL_DIR, ir.BUNDLED_MODEL_DIR))
                 for iso in (TARGET, *OFFS))
    domain = all(any((d / f"{iso}.npz").exists()
                     for d in (ad.AD_CACHE_DIR, ad.BUNDLED_AD_DIR))
                 for iso in (TARGET, *OFFS))
    ready = models and any(p.exists() for p in (library.CACHE, library.BUNDLED))
    if ready and domain and BUNDLED_QUANTILES.exists():
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
def render_funnel() -> None:
    from src.funnel import screen_to_contract
    from src.selectivity import OFFS, POTENCY_FLOOR, TARGET

    section_head(
        f"{TARGET} selectivity funnel",
        f"A diverse, target-agnostic library run down the cost funnel — Ro5/PAINS, "
        f"then per-isoform QSAR to the selectivity gap <code>S = {TARGET} − max({', '.join(OFFS)})</code>, "
        f"then conformal intervals and applicability domain on the survivors only — "
        f"to a shortlist that is both selective and in-domain."
    )

    _, cost_line = funnel_cache_state()
    if not st.session_state.get("funnel_started"):
        note(f"<strong>Before you start.</strong> {cost_line}")
        if st.button("Run the funnel", type="primary", key="start_funnel"):
            st.session_state["funnel_started"] = True
            st.rerun()
        return

    try:
        sl = run_funnel()
    except Exception as err:                      # surface, don't swallow
        st.error(f"{type(err).__name__}: {err}")
        st.session_state["funnel_started"] = False
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
        selection_mode="multi-row", key="funnel_table",
        column_config={
            "SMILES": st.column_config.TextColumn("SMILES", width="large"),
            f"pred {TARGET}": st.column_config.NumberColumn(
                f"pred {TARGET}", format="%.2f", help=f"Predicted {TARGET} pChEMBL"),
            "gap S": st.column_config.NumberColumn(
                "gap S", format="%+.2f",
                help="Selectivity gap in log-units; +1 ≈ 10× selective"),
            "gap 90% CI": st.column_config.TextColumn(
                "gap 90% CI", help="Split-conformal interval, propagated from both isoforms"),
            "binder": st.column_config.NumberColumn(
                "binder", format="%.2f",
                help="Tier-0.5 binder-gate probability P(JAK binder). Every shortlisted "
                     "molecule cleared the gate; the value is its margin above the cutoff."),
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
        colab_handoff(screen_to_contract(chosen),
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
        row, percentile = score_one(smiles)
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
        ("Library percentile", f"{percentile:.1f}", "rank by gap S in the drug-like library"),
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
        if lo <= 0.0 <= hi:
            st.warning(
                f"The interval crosses zero, so this molecule cannot be called "
                f"{TARGET}-selective on its own evidence. What the model supports is its "
                f"**rank**: it sits at the {percentile:.1f}th percentile of the library. "
                f"Selectivity ranking was validated (Spearman 0.80 vs measured gaps, "
                f"4.5× enrichment); per-molecule intervals this wide were not.")
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
    colab_handoff(screen_to_contract(pd.DataFrame([row])), "Export contract — 1 molecule")


def render_target_screen(target: str, top_n: int, expand: bool, max_records: int) -> None:
    section_head(
        "Target screen",
        "The v1 single-target pipeline — ChEMBL retrieval, drug-likeness filter, "
        "per-target QSAR, composite ranking — with the known actives kept alongside "
        "as a positive control."
    )
    if not target:
        st.info("Enter a target name or ChEMBL id in the sidebar to run a screen.")
        return

    try:
        with st.spinner(f"Screening {target}…"):
            tgt, metrics, scored = run_screen(target.strip(), expand, max_records)
    except (ValueError, RuntimeError) as err:
        st.error(str(err))
        return

    known = scored[scored["source"] == "chembl_known"]
    novel = scored[scored["source"] == "pubchem_novel"]

    stat_row([
        ("Model R²", f"{metrics.r2:.3f}", "scaffold split — novel chemotypes"),
        ("RMSE", f"{metrics.rmse:.2f}", "pChEMBL units"),
        ("Known actives", f"{len(known)}", "drug-like, from ChEMBL"),
        ("Novel candidates", f"{len(novel)}", "PubChem analogues, not in training"),
    ], accent_first=True)

    prop = load_property_models()
    scoring = (
        f"<strong>{tgt.chembl_id} — {tgt.pref_name}.</strong> Activity model: "
        f"scaffold-split R² {metrics.r2:.3f}, RMSE {metrics.rmse:.2f} pChEMBL, "
        f"n = {metrics.n_molecules}."
    )
    if prop:
        scoring += (f" Solubility (ESOL) R² {prop.metrics['solubility']['r2']:.3f}; "
                    f"toxicity (Tox21 any-hit) ROC-AUC {prop.metrics['toxicity']['roc_auc']:.3f}.")
    scoring += (" Composite <code>= 0.5·activity + 0.2·QED + 0.15·solubility "
                "+ 0.15·(1 − tox risk)</code>.")
    note(scoring)

    display_cols = {
        "id": "ID", "pred_pchembl": "Predicted pChEMBL", "measured_pchembl": "Measured pChEMBL",
        "qed": "QED", "logS_pred": "logS (sol.)", "tox_prob": "Tox risk", "composite": "Score",
    }

    def track(df: pd.DataFrame, potency_col: str, potency_label: str, caption: str) -> None:
        if df.empty:
            st.info("No molecules in this track.")
            return
        top = df.head(top_n)
        st.caption(caption)
        img = mol_grid(top, potency_col, potency_label)
        if img is not None:
            st.image(img, width="stretch")
        st.dataframe(top[list(display_cols)].rename(columns=display_cols).round(3),
                     width="stretch", hide_index=True)

    tab_novel, tab_known = st.tabs(["Novel candidates", "Known actives (control)"])
    with tab_novel:
        track(novel, "pred_pchembl", "pred pChEMBL",
              "Molecules outside the model's training set, ranked by predicted potency "
              "× drug-likeness. This is the screening output.")
    with tab_known:
        track(known, "measured_pchembl", "measured pChEMBL",
              "Known ChEMBL actives scored on their measured potency — a positive "
              "control that the pipeline surfaces real binders.")

    provenance([("target", tgt.chembl_id), ("records", f"≤ {max_records}"),
                ("training molecules", f"{metrics.n_molecules}"),
                ("expansion", "PubChem analogues" if expand else "off")])


# --------------------------------------------------------------------------- #
apply_theme()

with st.sidebar:
    st.markdown("### Mode")
    # segmented_control returns None when the active chip is clicked again.
    mode = st.segmented_control("Mode", ["Selectivity funnel", "Single molecule", "Target screen"],
                                default="Selectivity funnel",
                                label_visibility="collapsed") or "Selectivity funnel"
    st.divider()

    if mode == "Single molecule":
        st.markdown("### Score a molecule")
        single_query = st.text_input("SMILES or compound name", value="ruxolitinib")
        st.caption("A SMILES string is parsed locally. Anything else is looked up by "
                   "name on PubChem and reduced to its neutral parent, so a salt form "
                   "scores as the drug it is.")
    elif mode == "Target screen":
        st.markdown("### Screen a target")
        target = st.text_input("Target name or ChEMBL id", value="EGFR")
        top_n = st.slider("Top N per track", 4, 24, 8)
        expand = st.checkbox("Expand with novel PubChem analogues", value=True)
        max_records = st.select_slider("Max ChEMBL records", [1000, 2000, 4000], value=4000)
        st.caption("EGFR ships with a pre-baked model and returns immediately. Other "
                   "targets train on first run (~30–40 s), then cache.")
    else:
        st.markdown("### Funnel")
        _, cost_line = funnel_cache_state()
        st.caption(cost_line)
        if st.session_state.get("funnel_started") and st.button("Reset run"):
            st.session_state["funnel_started"] = False
            run_funnel.clear()
            st.rerun()

masthead(["CPU-only", "ChEMBL + PubChem"] if mode == "Target screen"
         else ["CPU-only", "JAK1 / JAK2 / JAK3", "conformal 90%"])

if mode == "Selectivity funnel":
    render_funnel()
elif mode == "Single molecule":
    render_single(single_query)
else:
    render_target_screen(target, top_n, expand, max_records)
