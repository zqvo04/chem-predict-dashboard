"""D2 — does the indication actually drive the objective's sign?

Same panel, same models, opposite objectives. Autoimmune avoids JAK2 (EPO/TPO
signalling, hence anaemia and thrombocytopenia); myeloproliferative neoplasms
target JAK2 directly (JAK2 V617F). If the wiring is right, a JAK1-preferring drug
outranks a JAK2-preferring one under the first objective and the order reverses
under the second.

This is a WIRING check, not a ranking-performance result: all three drugs are in
the training data, so the predictions are recall. Claiming more would repeat v1's
mistake, and ROADMAP D2 says the same.

Note the degeneracy: for all three drugs JAK3 is the weakest isoform, so max(off)
picks the same isoform under both objectives and the two gaps come out exactly
antisymmetric. The test therefore asserts the two RANK ORDERS, which is the claim,
and not that the two scores carry independent information.
"""
import pandas as pd

from src import funnel
from src.selectivity import selectivity_gap

DRUGS = {
    "upadacitinib": "CC[C@@H]1CN(C(=O)NCC(F)(F)F)C[C@@H]1c1cnc2cnc3[nH]ccc3n12",
    "fedratinib": "Cc1cnc(Nc2ccc(OCCN3CCCC3)cc2)nc1Nc1cccc(S(=O)(=O)NC(C)(C)C)c1",
    "ruxolitinib": "N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1",
}


def _ranks():
    names = list(DRUGS)
    scored = funnel.score_molecules([DRUGS[n] for n in names])
    preds = pd.DataFrame({iso: scored[f"pred_{iso}"] for iso in ("JAK1", "JAK2", "JAK3")})
    frame = pd.DataFrame({
        "drug": names,
        "autoimmune": selectivity_gap(preds, "JAK1", ("JAK2", "JAK3")),
        "mpn": selectivity_gap(preds, "JAK2", ("JAK1", "JAK3")),
    })
    return (frame.sort_values("autoimmune", ascending=False)["drug"].tolist(),
            frame.sort_values("mpn", ascending=False)["drug"].tolist())


def test_indication_flips_the_ranking():
    autoimmune, mpn = _ranks()
    assert autoimmune == ["upadacitinib", "ruxolitinib", "fedratinib"]
    assert mpn == ["fedratinib", "ruxolitinib", "upadacitinib"]


def test_the_dual_inhibitor_is_middle_under_both_objectives():
    autoimmune, mpn = _ranks()
    assert autoimmune[1] == "ruxolitinib"
    assert mpn[1] == "ruxolitinib"
