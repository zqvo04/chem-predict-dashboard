"""The measured-negative population and the discipline around it.

These are molecules somebody assayed against JAK and found weak — a censored
record, no pchembl anywhere in the panel. They are the negative class the gate
has never had. The invariants below are what keeps the experiment honest: the
sealed 414 stay out, the two folds do not overlap, and no scaffold crosses
between them, so a gate trained on the train fold has not seen the eval fold's
chemotypes.
"""
import pandas as pd

from src.data import panel_data
from src.models.scaffold_split import _scaffold
from src.panels import JAK


def test_population_excludes_the_sealed_414():
    """G6: the falsification audit's negative arm is not training data for anything."""
    population = panel_data.measured_negatives(JAK)
    sealed = set(panel_data.censored_library_molecules(JAK)["inchikey"])
    assert set(population["inchikey"]).isdisjoint(sealed)
    assert len(population) > 1000            # measured 1,720 on 2026-08-09


def test_population_has_no_quantified_molecule():
    """A molecule with a pchembl anywhere in the panel is training data, not a negative."""
    population = set(panel_data.measured_negatives(JAK)["smi"])
    for iso in JAK.isoforms:
        assert population.isdisjoint(set(panel_data.build_isoform_dataset(JAK, iso)["smi"]))


def test_folds_are_disjoint_and_scaffold_disjoint():
    split = panel_data.load_measured_negatives(JAK)
    train = split[split["fold"] == "train"]
    evalf = split[split["fold"] == "eval"]
    assert set(train["smi"]).isdisjoint(set(evalf["smi"]))
    assert len(evalf) > 200
    train_scaffolds = {_scaffold(s) for s in train["smi"]}
    eval_scaffolds = {_scaffold(s) for s in evalf["smi"]}
    assert train_scaffolds.isdisjoint(eval_scaffolds)


def test_the_split_is_read_from_disk_not_recomputed():
    a = panel_data.load_measured_negatives(JAK)
    b = panel_data.load_measured_negatives(JAK)
    pd.testing.assert_frame_equal(a, b)
    assert (JAK.data_bundled / "measured_negatives.parquet").exists()
