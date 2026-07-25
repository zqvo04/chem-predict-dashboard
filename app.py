"""Streamlit screening dashboard.

    streamlit run app.py

Two modes over the same model layer:

  * **Selectivity funnel** — run the wide, target-agnostic library down the JAK1
    cost funnel (Ro5/PAINS -> per-isoform QSAR -> gap S -> conformal + applicability
    domain) and export the picked cases as a versioned loop contract.
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
_TINT_IN_DOMAIN = "rgba(26, 108, 94, 0.12)"    # --color-accent  #1A6C5E
_TINT_UNCERTAIN = "rgba(142, 83, 17, 0.12)"    # --color-warn    #8E5311

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
        "domain": sl["verdict"].map({"in_domain": "in-domain", "uncertain": "uncertain"}),
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
            "domain": st.column_config.TextColumn(
                "domain", help="Tanimoto-distance and descriptor-leverage signals combined"),
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
        contract = screen_to_contract(chosen)
        st.download_button(
            f"Export contract — {len(picked)} molecule{'s' if len(picked) != 1 else ''}",
            data=json.dumps(contract, indent=2),
            file_name=f"{contract['case_id']}.json",
            mime="application/json", type="primary",
        )
        prov = contract["provenance"]
        provenance(
            [("case id", contract["case_id"]),
             ("conformal α", f"{prov['conformal_alpha']}"),
             ("code version", prov["code_version"])]
            + [(f"{iso} model", mid) for iso, mid in prov["model_ids"].items()]
        )


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
    mode = st.segmented_control("Mode", ["Selectivity funnel", "Target screen"],
                                default="Selectivity funnel",
                                label_visibility="collapsed") or "Selectivity funnel"
    st.divider()

    if mode == "Target screen":
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

masthead(["CPU-only", "JAK1 / JAK2 / JAK3", "conformal 90%"]
         if mode == "Selectivity funnel" else ["CPU-only", "ChEMBL + PubChem"])

if mode == "Selectivity funnel":
    render_funnel()
else:
    render_target_screen(target, top_n, expand, max_records)
