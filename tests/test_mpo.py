"""MPO annotation: the property axes Tier 2 carries alongside the selectivity gap."""
import numpy as np
import pandas as pd

from src import mpo


def test_annotate_returns_one_row_per_input_with_the_expected_columns():
    out = mpo.annotate(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
    assert len(out) == 3
    assert set(out.columns) == {"qed", "logS_pred", "tox_prob", "mpo"}


def test_unparseable_smiles_yields_nan_rather_than_shifting_later_rows():
    out = mpo.annotate(["CCO", "not_a_smiles", "c1ccccc1"])
    assert len(out) == 3
    assert np.isnan(out["qed"].iloc[1]) and np.isnan(out["mpo"].iloc[1])
    assert not np.isnan(out["qed"].iloc[0]) and not np.isnan(out["qed"].iloc[2])


def test_one_disqualifying_property_drags_the_score_down():
    """The reason for a geometric mean over a weighted sum: a molecule that is
    excellent on two axes and unacceptable on the third must not average out."""
    good = np.array([0.9, 0.9, 0.9])
    one_bad = np.array([0.9, 0.9, 0.01])
    geo = lambda d: float(np.exp(np.mean(np.log(np.clip(d, 1e-6, 1.0)))))
    assert geo(one_bad) < 0.25 * geo(good)
    assert geo(one_bad) < one_bad.mean()      # a weighted sum would have said ~0.6


def test_ramp_clips_to_the_unit_interval_in_both_directions():
    # Solubility: higher logS is better, so the ramp runs poor -> good upwards.
    assert mpo._ramp(np.array([-9.0]), mpo.LOGS_POOR, mpo.LOGS_GOOD)[0] == 0.0
    assert mpo._ramp(np.array([-1.0]), mpo.LOGS_POOR, mpo.LOGS_GOOD)[0] == 1.0
    # Toxicity: lower probability is better, so poor > good and the ramp inverts.
    assert mpo._ramp(np.array([0.9]), mpo.TOX_POOR, mpo.TOX_GOOD)[0] == 0.0
    assert mpo._ramp(np.array([0.0]), mpo.TOX_POOR, mpo.TOX_GOOD)[0] == 1.0


def test_mpo_falls_back_to_qed_when_the_property_bundle_is_missing(monkeypatch):
    monkeypatch.setattr(mpo, "load_property_models", lambda: None)
    out = mpo.annotate(["CCO", "c1ccccc1"])
    assert out["logS_pred"].isna().all() and out["tox_prob"].isna().all()
    pd.testing.assert_series_equal(out["mpo"], out["qed"], check_names=False)
