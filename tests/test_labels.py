"""The label ledger and the one invariant it exists to hold (G6).

A measurement is scored against a registered prediction or it is training data,
never both. The sealed molecules — the 414 measured non-binders and everything in
the sealed evaluation split — are the evaluation base, so a training reader that
returns even one of them has destroyed the thing every later round is measured on.
"""
import pytest

from src import labels
from src.panels import JAK


def test_append_and_read_roundtrip(tmp_path):
    labels.append(JAK, smi="CCO", isoform="JAK1", value=5.2, relation="=",
                  source="test", answer_status="answered", root=tmp_path)
    frame = labels.read(JAK, root=tmp_path)
    assert len(frame) == 1
    assert frame.loc[0, "smi"] == "CCO"
    assert frame.loc[0, "answer_status"] == "answered"
    assert not frame.loc[0, "sealed"]


def test_sealed_molecules_never_reach_training_rows(tmp_path):
    sealed = next(iter(labels.sealed_smiles(JAK)))
    labels.append(JAK, smi=sealed, isoform="JAK1", value=5.0, relation=">",
                  source="test", answer_status="answered", root=tmp_path)
    labels.append(JAK, smi="CCO", isoform="JAK1", value=5.2, relation="=",
                  source="test", answer_status="answered", root=tmp_path)
    train = labels.training_rows(JAK, root=tmp_path)
    assert set(train["smi"]) == {"CCO"}


def test_unanswered_rows_are_not_training_data(tmp_path):
    labels.append(JAK, smi="CCO", isoform="JAK1", value=None, relation=None,
                  source="test", answer_status="no_answer", root=tmp_path)
    labels.append(JAK, smi="CCC", isoform="JAK1", value=None, relation=None,
                  source="test", answer_status="failed", root=tmp_path)
    assert labels.training_rows(JAK, root=tmp_path).empty


def test_append_rejects_an_unknown_answer_status(tmp_path):
    with pytest.raises(ValueError):
        labels.append(JAK, smi="CCO", isoform="JAK1", value=1.0, relation="=",
                      source="test", answer_status="maybe", root=tmp_path)


def test_sealed_set_covers_both_populations():
    sealed = labels.sealed_smiles(JAK)
    assert len(sealed) > 1000      # 414 non-binders + the eval fold
