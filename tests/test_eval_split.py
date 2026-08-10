"""The sealed evaluation split (N11 / G9).

STATE.md section 4c measured that a Tanimoto lookup ties the model on a scaffold
split and loses badly under a publication-year cut. The split sealed at round 0 is
therefore the time split, and these tests pin the properties every later round
depends on: the folds do not overlap, no eval molecule predates the cut, and the
file on disk is the authority rather than something recomputed per run.
"""
import pandas as pd

from src.data import panel_data
from src.panels import JAK

CUT = 2020


def test_eval_split_folds_are_disjoint_and_dated_correctly():
    split = panel_data.load_eval_split(JAK)
    train = split[split["fold"] == "train"]
    evalf = split[split["fold"] == "eval"]
    assert set(train["smi"]).isdisjoint(set(evalf["smi"]))
    assert (train["year_first"] <= CUT).all()
    assert (evalf["year_first"] > CUT).all()
    assert len(evalf) > 500          # a split too small to score is not a split


def test_eval_split_is_read_from_disk_not_recomputed():
    a = panel_data.load_eval_split(JAK)
    b = panel_data.load_eval_split(JAK)
    pd.testing.assert_frame_equal(a, b)
    assert (JAK.data_bundled / "eval_split.parquet").exists()


def test_year_first_covers_almost_every_training_molecule():
    year = panel_data.year_first(JAK)
    data = panel_data.build_isoform_dataset(JAK, "JAK1")
    assert data["smi"].map(year).notna().mean() > 0.99
