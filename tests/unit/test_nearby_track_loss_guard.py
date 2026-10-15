from dataclasses import replace

import pytest

from robot.evaluation.nearby_track_loss_guard import (
    NearbyTrackLossConfig,
    NearbyTrackLossGuard,
)
from robot.safety.obstacle_tracking import TrackedObstacle


def track(identity, x=1.3, y=0.0):
    return TrackedObstacle(identity, (x, y), (x, y), (0, 0), (0, 0), 20, 5, True)


def test_lost_low_object_stops_before_occupancy_expiry_implies_free_space():
    guard = NearbyTrackLossGuard()
    guard.observe((track(1),), source_time=1.0, robot_world_xy=(0, 0))
    assert not guard.observe((), source_time=1.1, robot_world_xy=(0.5, 0))["blocked"]
    result = guard.observe((), source_time=1.2, robot_world_xy=(0.6, 0))
    assert result["blocked"]
    assert result["blocking_tracks"][0]["original_last_seen_time"] == 1.0
    assert guard.observe((), source_time=1.3, robot_world_xy=(0.6, 0))[
        "blocking_tracks"
    ][0]["missing_seconds"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="epochs"):
        guard.observe((), source_time=1.3, robot_world_xy=(0.6, 0))


def test_current_positive_geometry_can_resolve_unique_id_change_only():
    guard = NearbyTrackLossGuard()
    guard.observe((track(1),), source_time=1, robot_world_xy=(0.5, 0))
    result = guard.observe((track(2, 1.35),), source_time=1.3, robot_world_xy=(0.6, 0))
    assert not result["blocked"]
    assert set(guard.observed) == {2}
    guard.observe((), source_time=1.4, robot_world_xy=(0.6, 0))
    assert guard.observe((), source_time=1.5, robot_world_xy=(0.6, 0))["blocked"]


def test_ambiguous_replacement_and_far_loss_are_not_assumed_clear():
    guard = NearbyTrackLossGuard()
    guard.observe((track(1),), source_time=1, robot_world_xy=(0, 0))
    assert not guard.observe((), source_time=1.3, robot_world_xy=(0, 0))["blocked"]
    result = guard.observe(
        (track(2, 1.25), track(3, 1.35)), source_time=1.4, robot_world_xy=(0.6, 0)
    )
    assert result["blocked"]
    assert result["blocking_tracks"][0]["track_id"] == 1


def test_guard_rejects_invalid_scope_and_duplicate_geometric_ids():
    with pytest.raises(ValueError):
        NearbyTrackLossConfig(maximum_missing_seconds=0)
    with pytest.raises(ValueError):
        replace(NearbyTrackLossConfig(), enrollment_distance_m=0.5)
    guard = NearbyTrackLossGuard()
    with pytest.raises(ValueError):
        guard.observe((track(1), track(1)), source_time=1, robot_world_xy=(0, 0))
