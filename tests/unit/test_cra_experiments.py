import pytest

from robot.evaluation.cra_experiments import (
    CRABaseline,
    EvidenceMode,
    FactorSensorReadiness,
    PRIMARY_BASELINES,
    RestaurantTrialResult,
    assess_restaurant_motion_readiness,
    primary_restaurant_scenarios,
    require_sensor_motion_readiness,
    summarize_trials,
    validate_matched_trials,
)
from robot.risk.cra import PhysicalRiskFactor


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
    assert {item.evidence_mode for item in summaries} == {EvidenceMode.SENSOR}


def test_summaries_never_pool_oracle_and_sensor_results():
    oracle = tuple(
        result(baseline, evidence_mode=EvidenceMode.ORACLE)
        for baseline in PRIMARY_BASELINES
    )
    summaries = summarize_trials(matched() + oracle)
    assert len(summaries) == 6
    assert {
        (item.evidence_mode, item.baseline, item.trial_count)
        for item in summaries
    } == {
        (mode, baseline, 1)
        for mode in (EvidenceMode.ORACLE, EvidenceMode.SENSOR)
        for baseline in PRIMARY_BASELINES
    }


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


def readiness(factor, **changes):
    values = dict(
        factor=factor,
        stationary_pipeline_validated=True,
        runtime_provenance_validated=True,
        sensor_domain_accuracy_validated=True,
    )
    values.update(changes)
    return FactorSensorReadiness(**values)


def test_all_four_sensor_factors_are_required_before_motion():
    report = assess_restaurant_motion_readiness(
        tuple(readiness(factor) for factor in PhysicalRiskFactor)
    )
    assert report.ready
    require_sensor_motion_readiness(report)


def test_unvalidated_models_block_motion_with_factor_specific_reasons():
    report = assess_restaurant_motion_readiness(
        (
            readiness(
                PhysicalRiskFactor.POTENTIAL,
                sensor_domain_accuracy_validated=False,
                limitations=("visual mass is not calibrated in Isaac",),
            ),
            readiness(
                PhysicalRiskFactor.KINETIC,
                sensor_domain_accuracy_validated=False,
            ),
            readiness(
                PhysicalRiskFactor.CONTACT,
                sensor_domain_accuracy_validated=False,
                limitations=("only exposed-edge masks are supported",),
            ),
            readiness(
                PhysicalRiskFactor.THERMAL,
                stationary_pipeline_validated=False,
                sensor_domain_accuracy_validated=False,
                limitations=("no trained hot-soup checkpoint",),
            ),
        )
    )
    assert not report.ready
    assert any(reason.startswith("thermal:") for reason in report.blocking_reasons)
    assert any("visual mass" in reason for reason in report.blocking_reasons)
    with pytest.raises(RuntimeError, match="not ready"):
        require_sensor_motion_readiness(report)


def test_missing_factor_readiness_is_not_treated_as_safe():
    report = assess_restaurant_motion_readiness(
        (readiness(PhysicalRiskFactor.POTENTIAL),)
    )
    assert not report.ready
    assert "thermal: readiness record unavailable" in report.blocking_reasons


def test_readiness_gate_rejects_invalid_report():
    with pytest.raises(TypeError):
        require_sensor_motion_readiness(object())
    with pytest.raises(ValueError, match="duplicates"):
        assess_restaurant_motion_readiness(
            (
                readiness(PhysicalRiskFactor.CONTACT),
                readiness(PhysicalRiskFactor.CONTACT),
            )
        )


def test_primary_scope_matches_current_hot_soup_and_contact_decisions():
    scenarios = primary_restaurant_scenarios()
    descriptions = {item.scenario_id: item.description for item in scenarios}
    assert "carried_hot_soup" in descriptions
    assert "visible-plume" in descriptions["carried_hot_soup"]
    assert descriptions["exposed_edge"] == "fixed-label exposed blade edge"
    assert "stove_flame" not in descriptions
