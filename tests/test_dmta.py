"""One DMTA round: the properties that make its numbers mean anything.

Every round carries a random control batch of the same size drawn from the same
pool (G5/N9). Wave 1 established twice — once on the falsification audit, once on
the learning curve — that a one-armed number cannot distinguish an effect from a
moved denominator, and a loop reporting its own improvement is exactly where that
mistake is most expensive.

The round assertions share a single `run_round` call on purpose. Scoring the pool
means a nearest-neighbour search of ~1,000 molecules against a ~12,600-molecule
training reference, and repeating it per assertion cost six minutes for four facts
about the same round.
"""
import pytest

from src import dmta, labels
from src import oracle as orc
from src.panels import JAK

BATCH = 20


@pytest.fixture(scope="module")
def context():
    return dmta.build_context(JAK)


@pytest.fixture(scope="module")
def one_round(context, tmp_path_factory):
    root = tmp_path_factory.mktemp("dmta")
    oracle = orc.TimeSplitOracle(JAK)
    before = oracle.population()
    result = dmta.run_round(JAK, context, oracle, batch=BATCH, seed=0, root=root)
    return result, oracle, before, root


def test_context_models_are_refit_not_the_deployed_ones(context):
    """G0: the campaign must not be scoring through the shipped models."""
    from src.models.isoform_regressor import train_and_cache
    assert context.models["JAK1"] is not train_and_cache(JAK, "JAK1").model
    assert set(context.models) == set(JAK.isoforms)


def test_round_reports_both_arms_and_a_comparable_ceiling(one_round):
    result, _, _, _ = one_round
    for arm in ("acquired", "random"):
        assert result[arm]["n"] == BATCH
        assert 0.0 <= result[arm]["hit_rate"] <= 1.0
        assert 0.0 <= result[arm]["falsification_rate"] <= 1.0
    assert result["enrichment_ceiling"] > 1.0
    assert result["advantage"] == pytest.approx(
        result["acquired"]["hit_rate"] - result["random"]["hit_rate"])


def test_round_records_every_answer_in_the_ledger_as_sealed(one_round):
    _, _, _, root = one_round
    ledger = labels.read(JAK, root=root)
    assert len(ledger) == 2 * BATCH * len(JAK.isoforms)   # both arms, every isoform
    assert ledger["sealed"].all()                         # G6
    assert labels.training_rows(JAK, root=root).empty


def test_round_consumes_the_oracle_population(one_round):
    result, oracle, before, _ = one_round
    assert oracle.population() == before - 2 * BATCH
    assert result["oracle_remaining"] == oracle.population()


def test_falsification_by_domain_bucket_is_reported(one_round):
    result, _, _, _ = one_round
    buckets = result["falsification_by_bucket"]
    assert buckets
    assert all(0.0 <= v <= 1.0 for v in buckets.values())


def test_penalty_is_larger_where_falsification_was_worse():
    penalty = dmta.penalty_from([{"falsification_by_bucket":
                                  {"0.00-0.30": 0.8, "0.60-1.01": 0.1}}])
    assert penalty["0.00-0.30"] > penalty["0.60-1.01"] >= 0.0


def test_penalty_averages_across_rounds_it_is_given():
    rounds = [{"falsification_by_bucket": {"0.00-0.30": 1.0}},
              {"falsification_by_bucket": {"0.00-0.30": 0.0}}]
    assert dmta.penalty_from(rounds)["0.00-0.30"] == pytest.approx(0.5)


def test_no_history_means_no_penalty():
    assert dmta.penalty_from([]) == {}
