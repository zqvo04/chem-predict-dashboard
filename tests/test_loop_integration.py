"""STEP 9: end-to-end loop integration test (fast, real code path, tiny data).

Runs B (screen) -> SELECT (contract) -> A (generate + re-score) through the *real*
funnel / deep_dive code, with the heavy data sources monkeypatched to a tiny
synthetic set so it runs in seconds. This is the guard that the loop can't silently
break (e.g. a missing symbol, a renamed column, a broken contract).
"""
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from src import deep_dive, funnel
from src.data import jak
from src.models import isoform_regressor as ir

# ~25 scaffolds x 3 substituent variants -> enough for the nested scaffold splits.
_SCAFFOLDS = [
    "c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "c1ccoc1",
    "c1ccsc1", "c1cnc2ccccc2c1", "c1ccc2ncccc2c1", "c1ccc2occc2c1", "c1ccc2sccc2c1",
    "O=C(O)c1ccccc1", "O=C(N)c1ccccc1", "c1ccc(-c2ccccc2)cc1", "c1ccc(Oc2ccccc2)cc1",
    "c1ccc(Cc2ccccc2)cc1", "c1ccc2c(c1)OCO2", "c1ccc2c(c1)CCC2", "c1ccc2c(c1)CCCC2",
    "c1cc2ccc3cccc4ccc(c1)c2c34", "c1ccc(-c2ccncc2)cc1", "c1ccc(-c2cccnc2)cc1",
    "c1ccc(-c2ccco2)cc1", "c1ccc(-c2cccs2)cc1", "Cc1ccccc1", "CCc1ccccc1",
]
_SUBS = ["", "C", "CC"]


def _synthetic(name, use_cache=True):
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    smis, pch = [], []
    for i, sc in enumerate(_SCAFFOLDS):
        for sub in _SUBS:
            mol = Chem.MolFromSmiles(sub + sc if sub else sc)
            if mol is None:                       # keep only valid canonical SMILES
                continue
            smis.append(Chem.MolToSmiles(mol))
            pch.append(6.5 + (i % 4) * 0.4 + rng.normal(0, 0.2))
    df = pd.DataFrame({"smi": smis, "pchembl": pch, "n_meas": 1})
    return df.drop_duplicates("smi").reset_index(drop=True)


@pytest.fixture
def tiny_world(tmp_path, monkeypatch):
    monkeypatch.setattr(ir, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(ir.jak, "build_isoform_dataset", _synthetic)
    monkeypatch.setattr(jak, "build_isoform_dataset", _synthetic)
    # patch the copy imported into conformal's namespace too
    import src.conformal as cf
    monkeypatch.setattr(cf.jak, "build_isoform_dataset", _synthetic)
    lib = pd.DataFrame({"smi": ["c1ccccc1C(=O)O", "c1ccc(O)cc1", "c1ccncc1C(=O)N",
                                "c1ccc2ccccc2c1", "Cc1ccc(cc1)C(=O)O"]})
    monkeypatch.setattr("src.funnel.load_library", lambda use_cache=True: lib)
    return tmp_path


def test_loop_runs_end_to_end(tiny_world):
    # B: screen the (tiny) library down the funnel
    sl = funnel.screen_library(tier1_keep=20, shortlist=5, use_cache=True)
    assert not sl.empty
    for col in ("gap", "gap_lo", "gap_hi", "in_domain", "verdict", "pred_JAK1"):
        assert col in sl.columns

    # SELECT: export a contract; its model_ids pin the current models
    contract = funnel.screen_to_contract(sl.head(2).reset_index(drop=True))
    assert contract["provenance"]["stage"] == "B_export"
    assert set(contract["provenance"]["model_ids"]) == {"JAK1", "JAK2", "JAK3"}

    # A: generate + re-score through the SAME models, emit an A_rescore contract
    result = deep_dive.run_deep_dive(contract, max_analogues_per_case=10)
    assert (result.after["origin"] == "generated").all()
    a_contract = deep_dive.rescore_contract(result)
    assert a_contract["provenance"]["stage"] == "A_rescore"
    assert "in-silico hypothesis" in deep_dive.report_markdown(result).lower()


def test_loop_blocks_on_model_mismatch(tiny_world):
    sl = funnel.screen_library(tier1_keep=20, shortlist=5, use_cache=True)
    contract = funnel.screen_to_contract(sl.head(1).reset_index(drop=True))
    contract["provenance"]["model_ids"]["JAK1"] = "CHEMBL2835@TAMPERED"
    with pytest.raises(ValueError, match="Model mismatch"):
        deep_dive.run_deep_dive(contract)
