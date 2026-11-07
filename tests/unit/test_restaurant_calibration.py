"""The restaurant scoring candidate is explicit, bounded and source preserving."""

from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from robot.perception.physical_attributes import ScalarEstimate, EstimateStatus
from robot.risk.cra import (
    CRAAssessmentConfig, CRARiskAssessor, PhysicalRiskProfile,
    PhysicalRiskVector, RiskFactorEstimate,
)
from robot.risk.restaurant_calibration import (
    RestaurantCalibrationConfig, RestaurantFactorWeights,
    calibrate_object_policy, evaluate_restaurant_calibration,
)
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy


SCALES = dict(potential=100., kinetic=100.)


def fixture(contact=.39, kinetic=.004, *, confidence=.5, interaction=.15):
    missing = RiskFactorEstimate.unknown(source="original missing input")
    profile = PhysicalRiskProfile(thermal=missing, potential=missing,
        contact=RiskFactorEstimate(contact, interaction, confidence, "original_contact",
            ("original frame=12; source time=3",)),
        kinetic=RiskFactorEstimate(kinetic, interaction, .25, "original_energy",
            ("linear clipped energy severity; scale=100.0 J; bound=upper_bound",)))
    policy = CRASafeSAGEPolicy(risk_assessor=CRARiskAssessor(CRAAssessmentConfig(
        fallback_risks=PhysicalRiskVector(.25, .25, .25, .25)))).evaluate(profile)
    details = {name: dict(mode="evidence_supported" if name in ("contact", "kinetic")
                         else "configured_prior", basis="visual_silhouette_geometry_proxy"
                         if name == "contact" else "physical", input_rejections=[])
               for name in ("thermal", "contact", "potential", "kinetic")}
    return policy, details


def evaluate(policy, details, config=None, **kwargs):
    return evaluate_restaurant_calibration(policy.as_dict()["cra"], details,
        config or RestaurantCalibrationConfig(enabled=True), eligible=True,
        source_energy_scales_j=SCALES, **kwargs)


def proxy(value=.29, upper=.39, confidence=.5, timestamp=3., status=EstimateStatus.PRIOR):
    return ScalarEstimate(value, max(0., value-.1), upper, confidence, "1",
        "visual_silhouette_contact_proxy", timestamp, status,
        ("original contact clock retained",))


def route(value=.15, *, status=EstimateStatus.PRIOR, confidence=.5, timestamp=5.):
    return ScalarEstimate(value, value, value, confidence, "1", "original_route", timestamp, status)


def test_default_and_unassessed_policies_remain_exact_objects():
    original, details = fixture()
    assert calibrate_object_policy(original, details, RestaurantCalibrationConfig(), eligible=True) is original
    assert calibrate_object_policy(original, details, RestaurantCalibrationConfig(enabled=True), eligible=False) is original


def test_no_supported_evidence_keeps_original_policy_number():
    original, details = fixture()
    for item in details.values():
        item["mode"] = "configured_prior"
    result = evaluate(original, details)
    assert result.phi == original.risk
    assert result.reason == "no_supported_factor_original_policy_preserved"
    assert result.prior_baseline is None


def test_zero_supported_energy_does_not_inherit_flat_mass_uncertainty_score():
    original, details = fixture(kinetic=0)
    result = evaluate(original, details)
    kinetic = next(f for f in result.factors if f.factor == "kinetic")
    assert original.assessment.factor("kinetic").conservative_risk == .1875
    assert kinetic.score == 0
    assert kinetic.original_confidence == .25
    assert kinetic.original_factor_score == .1875


def test_missing_factor_priors_remain_explicit_and_pool_once():
    original, details = fixture(contact=0, kinetic=0)
    result = evaluate(original, details)
    assert result.phi == result.prior_baseline == .25
    assert {f.factor: f.score for f in result.factors if not f.supported} == dict(thermal=.25, potential=.25)


def test_supported_factor_response_is_monotone_and_strong_factor_can_dominate():
    values = [evaluate(*fixture(contact=x)).phi for x in np.linspace(0, 1, 31)]
    assert np.all(np.diff(values) >= 0)
    strong = evaluate(*fixture(contact=1, interaction=1))
    assert strong.phi == 1
    assert evaluate(*fixture(contact=.59)).phi > evaluate(*fixture(contact=.39)).phi


def test_lower_positive_confidence_never_reduces_supported_score():
    values = [evaluate(*fixture(confidence=x)).phi for x in np.linspace(1, .05, 25)]
    assert np.all(np.diff(values) >= 0)


def test_kinetic_exponent_changes_only_supported_kinetic_influence():
    original, details = fixture()
    equal = evaluate(original, details)
    emphasized = evaluate(original, details, RestaurantCalibrationConfig(
        enabled=True, weights=RestaurantFactorWeights(kinetic=1.5)))
    assert emphasized.phi > equal.phi
    assert [f.score for f in emphasized.factors] == [f.score for f in equal.factors]
    assert emphasized.prior_baseline == equal.prior_baseline


def test_source_energy_scale_is_required_and_clipping_cannot_be_inverted():
    original, details = fixture()
    with pytest.raises(ValueError, match="reference scales"):
        evaluate_restaurant_calibration(original.assessment.as_dict(), details,
            RestaurantCalibrationConfig(enabled=True), eligible=True)
    with pytest.raises(ValueError, match="cannot exceed"):
        evaluate(original, details, RestaurantCalibrationConfig(enabled=True, kinetic_energy_scale_j=101))


def test_human_protection_prior_is_separate_from_physical_factor_values():
    original, details = fixture()
    plain = evaluate(original, details)
    human = evaluate(original, details, verified_human=True)
    assert human.physical_phi == plain.phi
    assert human.phi == .8
    assert human.factors == plain.factors
    assert human.as_dict()["human_prior_uplift"] > 0
    unassessed = evaluate_restaurant_calibration(original.assessment.as_dict(), details,
        RestaurantCalibrationConfig(enabled=True), eligible=False, verified_human=True)
    assert unassessed.physical_phi == original.risk
    assert unassessed.phi == .8


def test_policy_protocol_reports_current_scores_with_legacy_arithmetic_separate():
    original, details = fixture()
    before = deepcopy((original.as_dict(), details))
    result = calibrate_object_policy(original, details, RestaurantCalibrationConfig(enabled=True),
        eligible=True, source_energy_scales_j=SCALES)
    assert result.risk == result.assessment.phi == result.as_dict()["cra"]["phi"]
    assert result.as_dict()["cra"]["aggregation"] == "restaurant_evidence_weighted_noisy_or"
    assert result.as_dict()["legacy_cra"] == original.assessment.as_dict()
    assert result.assessment.factor("kinetic").conservative_risk < original.assessment.factor("kinetic").conservative_risk
    assert result.boundary_flux == -(.5 + 1.2*result.risk)
    assert result.tangential_strength == original.tangential_strength == 1.25
    assert result.passing_side == original.passing_side
    assert (original.as_dict(), details) == before


def test_valid_retained_contact_survives_missing_motion_context_without_new_clock():
    original, details = fixture(confidence=0)
    details["contact"].update(mode="configured_prior", input_rejections=["contact_interaction_likelihood: unavailable"])
    saved = proxy()
    before = asdict(saved)
    result = evaluate(original, details, contact_proxy=saved,
        timestamp_seconds=5., maximum_contact_age_seconds=120.)
    contact = next(f for f in result.factors if f.factor == "contact")
    assert contact.supported
    assert contact.context_multiplier == 1
    assert contact.contact_context_basis.startswith("configured_upper_interaction_prior")
    assert contact.calibration_mode == "supported_visual_cue_with_explicit_context_prior"
    assert contact.original_confidence == 0
    assert contact.calibration_confidence == .5
    assert contact.score == pytest.approx(.39**2 * 1.125)
    assert contact.contact_proxy["timestamp_seconds"] == 3.
    assert asdict(saved) == before


def test_contact_recovery_does_not_supply_missing_kinetic_evidence():
    original, details = fixture(confidence=0)
    cra = original.assessment.as_dict()
    details["contact"].update(mode="configured_prior", input_rejections=["contact_interaction_likelihood: unavailable"])
    details["kinetic"]["mode"] = "configured_prior"
    cra["factors"]["kinetic"].update(confidence=0., conservative_risk=.25)
    result = evaluate_restaurant_calibration(cra, details, RestaurantCalibrationConfig(enabled=True),
        eligible=True, source_energy_scales_j=SCALES, contact_proxy=proxy(),
        timestamp_seconds=5., maximum_contact_age_seconds=120.)
    scores = {row.factor: row for row in result.factors}
    assert scores["contact"].supported
    assert scores["kinetic"].score == .25
    assert not scores["kinetic"].supported
    assert result.prior_baseline == .25


def test_proxy_qualification_uses_assessment_clock_and_original_approved_lifetime():
    original, details = fixture(confidence=0)
    details["contact"]["mode"] = "configured_prior"
    # The camera receipt can follow the LiDAR field time. Only its original
    # assessment clock qualifies this frame; no maximum-of-evidence time trick.
    result = evaluate(original, details, contact_proxy=proxy(timestamp=5.1),
        timestamp_seconds=5.1, maximum_contact_age_seconds=1.)
    assert next(row for row in result.factors if row.factor == "contact").supported
    too_early = evaluate(original, details, contact_proxy=proxy(timestamp=5.1),
        timestamp_seconds=5., maximum_contact_age_seconds=1.)
    assert not next(row for row in too_early.factors if row.factor == "contact").supported


def test_original_prior_confidence_cap_still_applies_to_retained_proxy():
    original, details = fixture(confidence=0)
    details["contact"]["mode"] = "configured_prior"
    result = evaluate(original, details, contact_proxy=proxy(confidence=.9),
        timestamp_seconds=5., maximum_contact_age_seconds=120., prior_confidence_cap=.4)
    contact = next(row for row in result.factors if row.factor == "contact")
    assert contact.calibration_confidence == .4
    assert contact.contact_proxy["confidence"] == .9


def test_valid_route_context_modulates_cue_with_its_original_source_time():
    original, details = fixture(confidence=0)
    details["contact"]["mode"] = "configured_prior"
    result = evaluate(original, details, contact_proxy=proxy(), contact_interaction=route(),
        timestamp_seconds=5., maximum_contact_age_seconds=120.)
    item = next(f for f in result.factors if f.factor == "contact")
    assert item.context_multiplier == pytest.approx(.75 + .25*.15)
    assert item.contact_context_basis == "current_route_interaction_upper_bound"


@pytest.mark.parametrize("fault", ["stale", "future", "fallback", "oracle", "source", "zero_confidence", "visual_rejection"])
def test_invalid_visual_evidence_is_never_recovered(fault):
    original, details = fixture(confidence=0)
    details["contact"]["mode"] = "configured_prior"
    saved = proxy()
    if fault == "stale":
        saved = replace(saved, timestamp_seconds=0)
    elif fault == "future":
        saved = replace(saved, timestamp_seconds=6)
    elif fault in {"fallback", "oracle"}:
        saved = replace(saved, status=EstimateStatus(fault))
    elif fault == "source":
        saved = replace(saved, source="authored object class")
    elif fault == "zero_confidence":
        saved = replace(saved, confidence=0)
    else:
        details["contact"]["input_rejections"] = ["visual_silhouette_geometry_proxy: stale evidence"]
    result = evaluate(original, details, contact_proxy=saved, timestamp_seconds=5., maximum_contact_age_seconds=3.)
    contact = next(f for f in result.factors if f.factor == "contact")
    assert not contact.supported and contact.contact_proxy_rejections


@pytest.mark.parametrize("kwargs", [dict(enabled=1), dict(uncertainty_gain=.51), dict(context_floor=-.1),
    dict(contact_power=0), dict(human_protection_floor=1.1), dict(kinetic_energy_scale_j=float("nan"))])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        RestaurantCalibrationConfig(**kwargs)


def test_exponents_must_keep_every_factor_active():
    for value in (0, -1, 5, float("nan"), True):
        with pytest.raises((ValueError, TypeError)):
            RestaurantFactorWeights(kinetic=value)
