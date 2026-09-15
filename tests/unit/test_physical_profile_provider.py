from dataclasses import replace

from robot.perception.physical_attributes import (
    MassAttributes,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.risk.physical_profile_provider import PhysicalAttributeCRARiskProfileProvider
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
