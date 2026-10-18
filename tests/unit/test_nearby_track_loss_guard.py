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


def merge_guard(enabled=True):
    guard = NearbyTrackLossGuard(
        NearbyTrackLossConfig(allow_observed_fragment_merges=enabled)
    )
    guard.observe((track(1, 1.1), track(2, 1.25)), source_time=1, robot_world_xy=(0, 0))
    return guard


def test_current_positive_points_resolve_one_fragment_rejoining_existing_body():
    guard = merge_guard()
    parent = replace(track(2, 1.25), point_count=4)
    result = guard.observe(
        (parent,),
        source_time=1.2,
        robot_world_xy=(0, 0),
        current_points_world_xy=[[1.1, 0], [1.1, 0.01], [1.1, -0.01], [1.7, 0]],
        current_point_track_ids=[2, 2, 2, 2],
        current_points_source_time=1.2,
    )
    assert not result["blocked"]
    assert len(result["observed_fragment_merges"]) == 1
    assert result["observed_fragment_merges"][0]["lost_track_id"] == 1
    assert set(guard.observed) == {2}
    assert guard.observe((), source_time=1.4, robot_world_xy=(0.2, 0))["blocked"]


@pytest.mark.parametrize(
    "problem", ["legacy", "missing", "too_few", "far", "ambiguous", "unclustered"]
)
def test_centroid_or_ambiguous_returns_never_resolve_missing_fragment(problem):
    guard = merge_guard(problem != "legacy")
    parent = replace(track(2, 1.25), point_count=4)
    tracks = (parent,)
    points = [[1.1, 0], [1.1, 0.01], [1.1, -0.01], [1.7, 0]]
    ids = [2, 2, 2, 2]
    if problem == "too_few":
        points[2] = [1.7, 0]
    if problem == "far":
        points = [[1.4, 0]] * 4
    if problem in {"ambiguous", "unclustered"}:
        points += [[1.1, 0.02]] * 4
        ids += [3 if problem == "ambiguous" else -1] * 4
        if problem == "ambiguous":
            tracks += (replace(track(3, 1.1), point_count=4),)
    kwargs = (
        {}
        if problem == "missing"
        else dict(
            current_points_world_xy=points,
            current_point_track_ids=ids,
            current_points_source_time=1.2,
        )
    )
    assert guard.observe(tracks, source_time=1.2, robot_world_xy=(0, 0), **kwargs)[
        "blocked"
    ]


def test_misaligned_geometry_never_commits_guard_state():
    guard = merge_guard()
    with pytest.raises(ValueError, match="aligned current"):
        guard.observe(
            (track(2),),
            source_time=1.2,
            robot_world_xy=(0, 0),
            current_points_world_xy=[[1.1, 0]],
            current_point_track_ids=[2],
            current_points_source_time=1.0,
        )
    assert guard.last_source_time == 1


def moving_fragment_guard():
    guard = NearbyTrackLossGuard(
        NearbyTrackLossConfig(
            allow_observed_fragment_merges=True, use_observed_merge_velocity=True
        )
    )
    # A swinging fragment's velocity can differ from its stable moving parent.
    lost = replace(track(1, 1.1), velocity_world_xy=(0.5, -0.7))
    parent = replace(track(2, 1.0), velocity_world_xy=(-0.7, 0))
    guard.observe((lost, parent), source_time=1, robot_world_xy=(0, 0))
    parent = replace(parent, centroid_world_xy=(0.93, 0), observation_count=6)
    guard.observe((parent,), source_time=1.1, robot_world_xy=(0, 0))
    return guard, replace(
        parent, centroid_world_xy=(0.86, 0), observation_count=7, point_count=4
    )


def test_moving_parent_hypothesis_requires_current_geometry_and_stable_sensor_velocity():
    guard, parent = moving_fragment_guard()
    result = guard.observe(
        (parent,),
        source_time=1.2,
        robot_world_xy=(0, 0),
        current_points_world_xy=[[0.96, 0], [0.96, 0.01], [0.96, -0.01], [0.56, 0]],
        current_point_track_ids=[2] * 4,
        current_points_source_time=1.2,
    )
    assert not result["blocked"]
    event = result["observed_fragment_merges"][0]
    assert event["basis"] == "inferred_current_cluster_velocity_positive_returns"
    assert event["translation_world_xy"] == pytest.approx([-0.14, 0])
    assert event["old_centroid_world_xy"] == [1.1, 0]
    assert event["positive_returns_near_query"] == 3
    assert set(guard.observed) == {2}


@pytest.mark.parametrize(
    "problem",
    [
        "no_points",
        "unstable_velocity",
        "too_fast",
        "too_far",
        "too_old",
        "insufficient_observations",
        "ambiguous",
    ],
)
def test_motion_hypothesis_never_replaces_actual_positive_evidence(problem):
    guard, parent = moving_fragment_guard()
    xyz = [[0.96, 0], [0.96, 0.01], [0.96, -0.01], [0.56, 0]]
    ids = [2] * 4
    tracks = (parent,)
    t = 1.2
    if problem == "unstable_velocity":
        tracks = (replace(parent, velocity_world_xy=(0.7, 0)),)
    if problem == "too_fast":
        tracks = (replace(parent, velocity_world_xy=(-3, 0)),)
    if problem == "too_far":
        xyz = [[0.3, 0]] * 4
    if problem == "too_old":
        t = 1.5
    if problem == "insufficient_observations":
        tracks = (replace(parent, observation_count=2),)
    if problem == "ambiguous":
        xyz += [[0.96, 0.02]] * 4
        ids += [3] * 4
        tracks += (replace(track(3, 0.96), point_count=4),)
    kwargs = (
        {}
        if problem == "no_points"
        else dict(
            current_points_world_xy=xyz,
            current_point_track_ids=ids,
            current_points_source_time=t,
        )
    )
    assert guard.observe(tracks, source_time=t, robot_world_xy=(0, 0), **kwargs)[
        "blocked"
    ]


def original_coobservation_guard(coexisted=True):
    guard = NearbyTrackLossGuard(
        NearbyTrackLossConfig(
            allow_observed_fragment_merges=True, use_observed_merge_velocity=True
        )
    )
    parent = replace(track(2, 0.6), velocity_world_xy=(0.5, 0), point_count=12)
    guard.observe(
        (track(1, 1.0), parent) if coexisted else (track(1, 1.0),),
        source_time=1.0,
        robot_world_xy=(-0.5, 0),
    )
    if not coexisted:
        guard.observe((parent,), source_time=1.05, robot_world_xy=(-0.5, 0))
    for t, x in [(1.1, 0.65), (1.2, 0.70)]:
        observe_coobserved_parent(guard, t, x)
    return guard


def observe_coobserved_parent(guard, t, x, problem=None):
    parent = replace(
        track(2, x), velocity_world_xy=(0.5, 0), observation_count=10, point_count=12
    )
    low = [
        v + 1.5 * (x - 0.65) for v in [0.85, 0.73, 0.61, 0.49, 0.37, 0.25, 0.37, 0.25]
    ]
    points = [[0.97, y] for y in [-0.02, -0.01, 0.01, 0.02]] + [[v, 0] for v in low]
    tracks, ids = (parent,), [2] * 12
    if problem == "unstable":
        tracks = (replace(parent, velocity_world_xy=(-0.5, 0)),)
    if problem == "unsupported":
        points = [[x, 0]] * 12
    if problem == "ambiguous":
        tracks += (replace(track(3, 1.0), point_count=4),)
        points += [[1.0, y] for y in [-0.02, -0.01, 0.01, 0.02]]
        ids += [3] * 4
    kwargs = (
        {}
        if problem == "missing"
        else dict(
            current_points_world_xy=points,
            current_point_track_ids=ids,
            current_points_source_time=t,
        )
    )
    return guard.observe(
        tracks, source_time=t, robot_world_xy=(-0.5, 0) if t < 1.3 else (0, 0), **kwargs
    )


def test_original_coobservation_survives_parent_updates_without_renewing_its_epoch():
    guard = original_coobservation_guard()
    assert guard.coobserved[1, 2] == 1.0
    result = observe_coobserved_parent(guard, 1.3, 0.76)
    assert not result["blocked"] and set(guard.observed) == {2}
    event = result["observed_fragment_merges"][0]
    assert event["coexistence_basis"] == "retained_original_coobservation"
    assert event["coobservation_source_time"] == 1.0
    assert event["original_last_seen_time"] == 1.0
    assert event["original_fragment_age_seconds"] == pytest.approx(0.3)
    assert event["basis"] == "old_centroid_positive_returns"
    assert event["positive_returns_near_query"] >= 3


@pytest.mark.parametrize(
    "problem",
    ["never_coexisted", "expired", "unstable", "unsupported", "missing", "ambiguous"],
)
def test_retained_coobservation_never_substitutes_for_fresh_positive_support(problem):
    guard = original_coobservation_guard(problem != "never_coexisted")
    t, x = 1.3, 0.76
    if problem == "expired":
        observe_coobserved_parent(guard, 1.3, 0.76, "unsupported")
        assert guard.coobserved[1, 2] == 1.0
        t, x = 1.4, 0.81
    result = observe_coobserved_parent(guard, t, x, problem)
    assert result["blocked"] and not result["observed_fragment_merges"]
    assert result["blocking_tracks"][0]["original_last_seen_time"] == 1.0
