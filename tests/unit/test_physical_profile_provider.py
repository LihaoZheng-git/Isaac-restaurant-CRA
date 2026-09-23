from dataclasses import replace

from robot.perception.physical_attributes import (
    MassAttributes,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.risk.physical_profile_provider import PhysicalAttributeCRARiskProfileProvider
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy
from robot.safety.cra_track_policy import resolve_cra_track_policy
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.semantic_obstacle import SemanticObstacle


def scalar(value, unit):
    return ScalarEstimate(value, value, value, 1, unit, "sensor_fixture", 2)


def test_provider_maps_existing_lidar_id_and_exposes_diagnostics():
    base = PhysicalAttributes.unknown(track_id=4, timestamp_seconds=2)
    attributes = replace(
        base,
        mass=MassAttributes(scalar(10, "kg")),
        motion=replace(
            base.motion,
            relative_speed_mps=scalar(2, "m/s"),
            route_crossing_likelihood=scalar(0.5, "1"),
        ),
    )
    provider = PhysicalAttributeCRARiskProfileProvider(
        {4: attributes}, timestamp_seconds=2
    )
    track = TrackedObstacle(4, (1, 0), (1, 0), (0, 0), (0, 0), 5, 3, True)
    obstacle = SemanticObstacle(track, "person", 0.9, "camera_lidar")
    profile = provider.profile_for(obstacle)
    assert profile is not None and profile.kinetic.confidence == 1
    assert provider.diagnostics_by_track_id[4].track_id == 4


def test_provider_returns_none_for_track_without_physical_record():
    provider = PhysicalAttributeCRARiskProfileProvider({}, timestamp_seconds=2)
    track = TrackedObstacle(4, (1, 0), (1, 0), (0, 0), (0, 0), 5, 3, True)
    assert provider.profile_for(SemanticObstacle(track, "person", 1, "sensor")) is None


def test_live_style_unknowns_saturate_phi_and_keep_primary_lambda_fixed():
    base = PhysicalAttributes.unknown(track_id=4, timestamp_seconds=2)
    attributes = replace(base, mass=MassAttributes(scalar(8.5, "kg")))
    provider = PhysicalAttributeCRARiskProfileProvider(
        {4: attributes}, timestamp_seconds=2
    )
    track = TrackedObstacle(4, (1, 0), (1, 0), (0, 0), (0, 0), 5, 3, True)
    obstacle = SemanticObstacle(track, "chair", 0.9, "camera_lidar")

    result = resolve_cra_track_policy(
        (obstacle,), profile_provider=provider, policy=CRASafeSAGEPolicy()
    ).assignment_for(4)
    diagnostics = provider.diagnostics_by_track_id[4]

    assert diagnostics.potential.rejection_reasons == (
        "fall_height_m: unavailable",
        "harmful_fall_interaction: unavailable",
    )
    assert diagnostics.kinetic.rejection_reasons == (
        "relative_speed_mps: unavailable",
        "route_crossing_likelihood: unavailable",
    )
    assert diagnostics.contact.rejection_reasons
    assert diagnostics.thermal.rejection_reasons
    assert result.policy_result.assessment.nominal_phi == 0
    assert result.policy_result.assessment.phi == 1
    assert result.guidance_parameters.tangential_strength == 1.25
