"""CI smoke for the asset-*producing* path (ROADMAP G11).

STATE.md section 1a: two rename breakages — `negatives.jak_positive_smiles` inside
`library._gate_training_smiles`, and `src.data.jak` inside `scripts/gate0_audit.py`
— sat undetected while 176 tests passed, because the unit tests monkeypatch the
exact symbols that were broken (`tests/test_library.py:26` replaces
`_gate_training_smiles` itself; `test_binder_gate.py` and `test_loop_integration.py`
do the same one layer down). The real call chain was therefore never executed.

These tests patch nothing. Both breakages were name-resolution failures, so simply
running the real chain against the committed assets raises ImportError or
AttributeError before any assertion is reached — which is the whole class of bug.

Deliberately not covered: the network rebuild (`use_cache=False`). CI is offline,
and neither breakage was in the fetch loop.
"""
import importlib

from src.data import library


def test_gate_training_smiles_runs_the_real_chain():
    # library._gate_training_smiles -> negatives.positive_smiles + build_negatives
    # -> panel_data.build_isoform_dataset -> committed assets/jak/*.parquet.
    # STATE.md's own reproduction command only *references* the attribute; the
    # broken import was in the function body, so it has to be called.
    smiles = library._gate_training_smiles(use_cache=True)
    assert len(smiles) > 10_000        # ~27.6k from the committed JAK assets


def test_gate0_audit_module_imports():
    importlib.import_module("scripts.gate0_audit")
