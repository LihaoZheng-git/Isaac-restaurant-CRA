from dataclasses import replace

import numpy as np
import pytest

from test_human_mass_prior import inputs
from robot.perception.human_body_contact import (
    HumanBodyContactPriorConfig,
    HUMAN_BODY_CONTACT_SOURCE,
    select_visual_human_body_contact_prior,
)
from robot.perception.human_mass_prior import HumanMassPriorConfig
from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
    MassAttributes,
)
from robot.perception.visual_shape_contact import SHAPE_CONTACT_SOURCE
from robot.risk.cra import RiskFactorEstimate
from robot.risk.object_risk_policy import ObjectRiskPriorConfig, evaluate_object_risk
from robot.risk.physical_factor_adapter import (
    adapt_physical_factors,
    PhysicalFactorAdapterConfig,
)


def select(*, score=0.1, mass=70, timestamp=2.0, **changes):
    frame, association, people = inputs()
    if "frame" in changes:
        frame = changes["frame"]
    if "association" in changes:
        association = changes["association"]
    if "people" in changes:
        people = changes["people"]
    return select_visual_human_body_contact_prior(
        frame,
        association,
        people,
        track_id=7,
        timestamp_seconds=timestamp,
        association_config=HumanMassPriorConfig(
            "controlled adult", mass_kg=mass, lower_kg=40, upper_kg=120
        ),
        config=HumanBodyContactPriorConfig(score),
    )


def adapted_outline():
    unknown = PhysicalAttributes.unknown(track_id=7, timestamp_seconds=2.0)
    proxy = ScalarEstimate(
        1.0, 1.0, 1.0, 0.625, "1", SHAPE_CONTACT_SOURCE, 2.0, EstimateStatus.INFERRED
    )
    interaction = ScalarEstimate(
        1.0, 1.0, 1.0, 0.8, "1", "controlled_interaction", 2.0, EstimateStatus.INFERRED
    )
    return adapt_physical_factors(
        unknown,
        timestamp_seconds=2.0,
        visual_contact_proxy=proxy,
        visual_contact_basis="visual_silhouette_geometry_proxy",
        contact_interaction_likelihood=interaction,
        config=PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True),
    )


def test_fresh_person_selects_an_explicit_nonzero_policy_prior_without_measured_confidence():
    prior = select()
    assert prior.status is EstimateStatus.PRIOR and prior.value == 0.1
    assert prior.confidence == 0 and prior.source == HUMAN_BODY_CONTACT_SOURCE
    assert prior.timestamp_seconds == 2.0
    assert select(mass=100).value == prior.value


@pytest.mark.parametrize(
    "problem",
    ["unmatched", "class_only", "weak", "ambiguous", "epoch", "stale", "boundary"],
)
def test_class_text_or_unusable_person_evidence_cannot_lower_contact(problem):
    frame, association, people = inputs()
    timestamp = 2.0
    if problem == "unmatched":
        association = replace(association, status="ambiguous")
    elif problem == "class_only":
        people = replace(
            people, detections=(replace(people.detections[0], semantic_class="chair"),)
        )
    elif problem == "weak":
        people = replace(
            people, detections=(replace(people.detections[0], confidence=0.49),)
        )
    elif problem == "ambiguous":
        people = replace(people, detections=people.detections * 2)
    elif problem == "epoch":
        people = replace(people, source_frame_id=5)
    elif problem == "stale":
        timestamp = 2.2
    elif problem == "boundary":
        mask = np.array(people.detections[0].mask, copy=True)
        mask[0, 20] = True
        people = replace(people, detections=(replace(people.detections[0], mask=mask),))
    prior = select(
        frame=frame, association=association, people=people, timestamp=timestamp
    )
    assert not prior.is_available
    policy, detail = evaluate_object_risk(
        adapted_outline(), ObjectRiskPriorConfig(), human_body_contact_prior=prior
    )
    assert policy.risk == 1.0 and detail["contact"]["score"] == 1.0


def test_body_prior_replaces_only_outline_policy_and_preserves_raw_diagnostics_and_other_factors():
    adapted = adapted_outline()
    config = ObjectRiskPriorConfig()
    original, before = evaluate_object_risk(adapted, config)
    result, details = evaluate_object_risk(
        adapted, config, human_body_contact_prior=select()
    )
    assert original.risk == 1.0
    assert details["contact"]["score"] == 0.1
    assert details["contact"]["mode"] == "configured_human_body_contact_prior"
    assert details["contact"]["confidence"] == 0
    assert details["contact"]["raw_contact_policy_score"] == 1.0
    assert details["contact"]["raw_aggregate_phi"] == 1.0
    assert adapted.profile.contact.severity == 1.0  # Original estimator is untouched.
    for factor in ("thermal", "potential", "kinetic"):
        assert details[factor] == before[factor]
    assert result.risk == pytest.approx(1 - 0.9 * 0.75**3)
    assert result.tangential_strength == 1.25
    contact = next(f for f in result.assessment.factors if f.factor.value == "contact")
    assert (
        contact.estimate.confidence == 0
        and contact.estimate.source == HUMAN_BODY_CONTACT_SOURCE
    )


@pytest.mark.parametrize("score", [0.05, 0.1, 0.2])
def test_score_is_configurable_without_tuning_aggregation(score):
    _, details = evaluate_object_risk(
        adapted_outline(),
        ObjectRiskPriorConfig(),
        human_body_contact_prior=select(score=score),
    )
    assert details["contact"]["score"] == score


@pytest.mark.parametrize("change", ["old", "other_track", "oracle", "fake_source"])
def test_policy_rejects_old_or_wrong_identity_and_nonprior_inputs(change):
    prior = select()
    if change == "old":
        prior = replace(prior, timestamp_seconds=1.9)
    elif change == "other_track":
        prior = replace(prior, evidence=("track_id=8",))
    elif change == "oracle":
        prior = replace(prior, status=EstimateStatus.ORACLE)
    elif change == "fake_source":
        prior = replace(prior, source="authored_human_name")
    policy, _ = evaluate_object_risk(
        adapted_outline(), ObjectRiskPriorConfig(), human_body_contact_prior=prior
    )
    assert policy.risk == 1.0


def test_stronger_separate_edge_uses_its_original_uncertainty_policy():
    adapted = adapted_outline()
    edge = replace(
        adapted.contact,
        basis="visual_exposed_edge_proxy",
        estimate=RiskFactorEstimate(0.9, 1.0, 0.5, "controlled_edge"),
    )
    result, details = evaluate_object_risk(
        adapted,
        ObjectRiskPriorConfig(),
        human_body_contact_prior=select(),
        human_part_contact=edge,
    )
    assert details["contact"]["score"] == 0.9
    assert details["contact"]["basis"] == "visual_exposed_edge_proxy"
    assert details["contact"]["prior_risk"] == 0.25
    assert result.risk > 1 - 0.9 * 0.75**3


def test_complete_physical_or_direct_specialist_contact_is_never_suppressed():
    adapted = adapted_outline()
    for basis in ("physical", "visual_exposed_edge_proxy"):
        diagnostic = replace(adapted.contact, basis=basis)
        value = replace(adapted, contact=diagnostic)
        policy, detail = evaluate_object_risk(
            value, ObjectRiskPriorConfig(), human_body_contact_prior=select()
        )
        assert policy.risk == 1.0 and detail["contact"]["score"] == 1.0


def test_low_body_contact_does_not_limit_high_human_kinetic_risk():
    values = []
    for speed in (0.1, 5.0):
        attrs = PhysicalAttributes.unknown(track_id=7, timestamp_seconds=2.0)
        mass = ScalarEstimate(
            70, 45, 100, 0.5, "kg", "controlled_adult_prior", 2.0, EstimateStatus.PRIOR
        )
        motion = replace(
            attrs.motion,
            relative_speed_mps=ScalarEstimate(
                speed, speed, speed, 0.9, "m/s", "controlled_observed_speed", 2.0
            ),
            route_crossing_likelihood=ScalarEstimate(
                1.0, 1.0, 1.0, 0.9, "1", "controlled_interaction", 2.0
            ),
        )
        adapted = adapt_physical_factors(
            replace(attrs, mass=MassAttributes(mass), motion=motion),
            timestamp_seconds=2.0,
        )
        policy, detail = evaluate_object_risk(
            adapted, ObjectRiskPriorConfig(), human_body_contact_prior=select()
        )
        assert detail["contact"]["score"] == 0.1
        values.append(policy.risk)
    assert values[0] < values[1] == 1.0


def test_unknown_part_interaction_is_not_reinterpreted_as_a_low_body_score():
    adapted = adapted_outline()
    part = replace(
        adapted.contact,
        basis="visual_exposed_edge_proxy",
        estimate=RiskFactorEstimate.unknown(source="unresolved_part_interaction"),
        rejection_reasons=("part interaction unavailable",),
    )
    _, detail = evaluate_object_risk(
        adapted,
        ObjectRiskPriorConfig(),
        human_body_contact_prior=select(),
        human_part_contact=part,
    )
    assert detail["contact"]["score"] == 0.25
    assert detail["contact"]["mode"] == "configured_prior"


@pytest.mark.parametrize("score", [0, 1, -0.1, np.nan, np.inf, True, "low"])
def test_invalid_or_zero_human_prior_is_rejected(score):
    with pytest.raises(ValueError):
        HumanBodyContactPriorConfig(score)
