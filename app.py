"""Phase 5: Streamlit screening dashboard.

    streamlit run app.py

Enter a target (e.g. EGFR), and the app runs the full pipeline — retrieve known
actives, expand with novel PubChem analogues, filter for drug-likeness, predict
per-target activity, and rank — then shows the two tracks with structures.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.models.property_models import load_property_models
from src.pipeline import screen

st.set_page_config(page_title="Target → Candidate Screening", page_icon="🧪", layout="wide")

_DISPLAY_COLS = {
    "id": "ID", "pred_pchembl": "Predicted pChEMBL", "measured_pchembl": "Measured pChEMBL",
    "qed": "QED", "logS_pred": "logS (sol.)", "tox_prob": "Tox risk", "composite": "Score",
}


@st.cache_data(show_spinner=False)
def run_screen(target: str, expand: bool, max_records: int):
    tgt, model, scored = screen(target, expand=expand, max_records=max_records, use_cache=True)
    return tgt, model.metrics, scored


def mol_grid(df: pd.DataFrame, score_col: str):
    mols, legends = [], []
    for row in df.itertuples():
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        if mol is None:
            continue
        mols.append(mol)
        legends.append(f"{row.id}  ({score_col}={getattr(row, score_col):.2f})")
    if not mols:
        return None
    return Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(230, 190), legends=legends)


def show_track(df: pd.DataFrame, n: int, potency_col: str, caption: str):
    if df.empty:
        st.info("No molecules in this track.")
        return
    top = df.head(n)
    st.caption(caption)
    img = mol_grid(top, potency_col)
    if img is not None:
        st.image(img, width='stretch')
    table = top[list(_DISPLAY_COLS)].rename(columns=_DISPLAY_COLS).round(3)
    st.dataframe(table, width='stretch', hide_index=True)


# --------------------------------------------------------------------------- #
st.title("🧪 Target → Candidate Screening")
st.write("A zero-cost, CPU-only end-to-end drug-discovery screen: "
         "**ChEMBL retrieval → drug-likeness filter → per-target QSAR → ranking**.")

with st.sidebar:
    st.header("Screen a target")
    target = st.text_input("Target name or ChEMBL id", value="EGFR")
    top_n = st.slider("Top N per track", 4, 24, 8)
    expand = st.checkbox("Expand with novel PubChem analogues", value=True)
    max_records = st.select_slider("Max ChEMBL records", [1000, 2000, 4000], value=4000)
    go = st.button("Run screening", type="primary")
    st.caption("EGFR ships with a pre-baked model (instant). Other targets train "
               "on first run (~30–40 s) and are then cached.")

if go or target:
    try:
        with st.spinner(f"Screening {target}…"):
            tgt, metrics, scored = run_screen(target.strip(), expand, max_records)
    except (ValueError, RuntimeError) as err:
        st.error(str(err))
        st.stop()

    known = scored[scored["source"] == "chembl_known"]
    novel = scored[scored["source"] == "pubchem_novel"]

    st.subheader(f"{tgt.chembl_id} — {tgt.pref_name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model R² (scaffold split)", f"{metrics.r2:.3f}")
    c2.metric("RMSE (pChEMBL)", f"{metrics.rmse:.2f}")
    c3.metric("Drug-like known actives", f"{len(known)}")
    c4.metric("Novel candidates", f"{len(novel)}")

    with st.expander("Model quality & scoring"):
        prop = load_property_models()
        st.markdown(
            f"- **Activity** (per-target QSAR): scaffold-split R² = {metrics.r2:.3f}, "
            f"RMSE = {metrics.rmse:.2f} pChEMBL, n = {metrics.n_molecules}\n"
            + (f"- **Solubility** (ESOL): R² = {prop.metrics['solubility']['r2']:.3f}\n"
               f"- **Toxicity** (Tox21 any-hit): ROC-AUC = {prop.metrics['toxicity']['roc_auc']:.3f}\n"
               if prop else "")
            + "- **Composite** = 0.5·activity + 0.2·QED + 0.15·solubility + 0.15·(1 − tox risk)"
        )

    tab_novel, tab_known = st.tabs(["🔬 Novel candidates (discovery)", "✅ Known actives (control)"])
    with tab_novel:
        show_track(novel, top_n, "pred_pchembl",
                   "Molecules not in the model's training set, ranked by predicted "
                   "potency × drug-likeness. This is the screening output.")
    with tab_known:
        show_track(known, top_n, "measured_pchembl",
                   "Known ChEMBL actives scored on their *measured* potency — a "
                   "positive control that the pipeline surfaces real binders.")
