"""The time-split oracle: what it will answer, and what it must refuse (G2, G3).

The molecules it answers for are real and their measurements are real; what makes
them a test is that the campaign's models were refit without them. An oracle that
answers for a training molecule is measuring recall, so the refusal is the property
that matters.
"""
from src import oracle as orc
from src.data import panel_data
from src.panels import JAK


def test_answers_an_eval_fold_molecule_with_every_isoform():
    o = orc.TimeSplitOracle(JAK)
    answer = o.answer(o.remaining()[0])
    assert set(answer) == set(JAK.isoforms)
    assert all(isinstance(v, float) for v in answer.values())


def test_refuses_a_training_molecule():
    """G2: a molecule the campaign models trained on has no answer here."""
    split = panel_data.load_eval_split(JAK)
    train_smi = split.loc[split["fold"] == "train", "smi"].iloc[0]
    assert orc.TimeSplitOracle(JAK).answer(train_smi) is None


def test_refuses_a_molecule_nobody_measured():
    assert orc.TimeSplitOracle(JAK).answer("CCO") is None


def test_population_shrinks_as_it_is_asked_and_exhausts():
    o = orc.TimeSplitOracle(JAK)
    start = o.population()
    assert start > 1000                     # measured 1,078 on 2026-08-09
    o.answer(o.remaining()[0])
    assert o.population() == start - 1
    assert not o.exhausted


def test_asking_twice_does_not_double_count():
    o = orc.TimeSplitOracle(JAK)
    smi = o.remaining()[0]
    o.answer(smi)
    after = o.population()
    o.answer(smi)
    assert o.population() == after
