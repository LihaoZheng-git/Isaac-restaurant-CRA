import pytest

from robot.evaluation.cra_experiments import (
    CRABaseline,
    EvidenceMode,
    PRIMARY_BASELINES,
    RestaurantTrialResult,
    primary_restaurant_scenarios,
    summarize_trials,
    validate_matched_trials,
)


def result(baseline, **changes):
    values = dict(
        scenario_id="chair",
        seed=1,
        baseline=baseline,
        evidence_mode=EvidenceMode.SENSOR,
        completed=True,
        fail_closed=False,
        collision=False,
        minimum_clearance_m=0.4,
        intervention_count=2,
        fallback_count=0,
        elapsed_seconds=8,
        minimum_cbf_margin=0.01,
        lambda_value=0 if baseline is CRABaseline.GEOMETRIC_VELOCITY_CBF else 1.25,
    )
    values.update(changes)
    return RestaurantTrialResult(**values)


def matched():
    return tuple(result(baseline) for baseline in PRIMARY_BASELINES)


def test_primary_plan_covers_four_hazard_families():
    scenarios = primary_restaurant_scenarios()
    assert {item.hazard_family for item in scenarios} == {
        "thermal",
        "contact",
        "potential",
        "kinetic",
    }
    assert all(item.evidence_mode is EvidenceMode.SENSOR for item in scenarios)


def test_matched_results_and_summary():
    validate_matched_trials(matched())
    summaries = summarize_trials(matched())
    assert len(summaries) == 3
    assert all(item.completion_rate == 1 for item in summaries)


def test_unmatched_and_duplicate_trials_rejected():
    with pytest.raises(ValueError, match="unmatched"):
        validate_matched_trials(matched()[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_matched_trials(matched() + (matched()[0],))


def test_primary_safe_sage_requires_fixed_lambda():
    with pytest.raises(ValueError, match="lambda=1.25"):
        result(CRABaseline.CONTINUOUS_CRA_SAFE_SAGE, lambda_value=1.4)
    ablation = result(CRABaseline.CRA_RISK_DEPENDENT_LAMBDA_ABLATION, lambda_value=1.4)
    assert ablation.lambda_value == 1.4


def test_collision_cannot_be_reported_as_completion():
    with pytest.raises(ValueError):
        result(CRABaseline.GEOMETRIC_VELOCITY_CBF, collision=True)


def test_oracle_and_prior_modes_remain_explicit():
    assert (
        primary_restaurant_scenarios(evidence_mode=EvidenceMode.ORACLE)[0].evidence_mode
        is EvidenceMode.ORACLE
    )
    assert (
        primary_restaurant_scenarios(evidence_mode=EvidenceMode.PRIOR)[0].evidence_mode
        is EvidenceMode.PRIOR
    )
