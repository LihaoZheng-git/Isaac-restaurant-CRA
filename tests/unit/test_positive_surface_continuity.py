from dataclasses import replace

import numpy as np
import pytest

from robot.evaluation.nearby_track_loss_guard import (
    NearbyTrackLossConfig,
    NearbyTrackLossGuard,
)
from robot.safety.obstacle_tracking import TrackedObstacle


SURFACE = np.array([[.95, 0.], [1.25, 0.], [1.1, -.15], [1.1, .15]])
BODY = np.array([[1.3, 0.], [1.3, .01], [1.3, -.01]])


def tracked(identity, points):
    center = tuple(np.mean(points, axis=0))
    return TrackedObstacle(identity, center, center, (0, 0), (0, 0), len(points), 5, True)


def observe(guard, groups, now, *, extra=(), centers=None):
    tracks = tuple(tracked(key, pts) for key, pts in groups.items())
    if centers:
        tracks = tuple(replace(t, centroid_world_xy=centers.get(t.track_id, t.centroid_world_xy))
                       for t in tracks)
    points = np.vstack([*groups.values(), *[pts for _, pts in extra]])
    ids = np.concatenate([*[np.full(len(pts), key) for key, pts in groups.items()],
                          *[np.full(len(pts), key) for key, pts in extra]])
    original_points, original_ids = points.copy(), ids.copy()
    result = guard.observe(tracks, source_time=now, robot_world_xy=(0., 0.),
        current_points_world_xy=points, current_point_track_ids=ids,
        current_points_source_time=now)
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(ids, original_ids)
    return result


def prepared(*, enabled=True, coobserved=True, unique=False):
    guard = NearbyTrackLossGuard(NearbyTrackLossConfig(
        allow_observed_fragment_merges=True,
        allow_positive_extent_merges=True,
        allow_positive_surface_merges=enabled,
        allow_unique_full_surface_merges=unique))
    observe(guard, {1: SURFACE, 2: BODY} if coobserved else {1: SURFACE}, 1.)
    if not coobserved:
        observe(guard, {2: BODY}, 1.1)
    return guard


def test_complete_positive_surface_resolves_empty_centroid_without_wider_gate():
    merged = np.vstack([SURFACE, BODY])
    legacy = prepared(enabled=False)
    assert observe(legacy, {2: merged}, 1.2)["blocked"]
    assert legacy.positive_surfaces == {}
    guard = prepared()
    result = observe(guard, {2: merged}, 1.2)
    assert not result["blocked"]
    assert set(guard.observed) == {2}
    event, = result["observed_fragment_merges"]
    assert event["basis"] == "coobserved_whole_positive_surface_returns"
    assert event["positive_returns_near_query"] == 0
    assert event["original_last_seen_time"] == 1.
    assert event["coobservation_source_time"] == 1.
    assert event["translation_world_xy"] == [0., 0.]
    proof = event["positive_surface_proof"]
    assert proof["all_original_positive_returns_covered"]
    assert proof["original_surface_source_time"] == 1.
    assert proof["current_surface_source_time"] == 1.2
    assert proof["maximum_original_to_current_distance_m"] == 0.
    np.testing.assert_array_equal(proof["original_surface_points_world_xy"], SURFACE)
    assert proof["original_positive_return_count"] == len(SURFACE)
    assert proof["current_covering_return_count"] >= 3
    assert not result["body_prior_coverage_enabled"]


@pytest.mark.parametrize("problem", ["one_missing", "moved", "foreign", "unclustered", "far_centroid", "no_pair", "expired"])
def test_incomplete_ambiguous_or_unsupported_surface_still_stops(problem):
    guard = prepared(coobserved=problem != "no_pair")
    points = np.vstack([SURFACE, BODY])
    groups, extra, centers = {2: points}, (), None
    if problem == "one_missing":
        groups = {2: np.vstack([SURFACE[1:], BODY])}
    elif problem == "moved":
        groups = {2: points + [.11, 0.]}
    elif problem in {"foreign", "unclustered"}:
        competitor = np.array([[.95, 0.], [.95, .01], [.95, -.01]])
        if problem == "foreign":
            groups[3] = competitor
        else:
            extra = ((-1, competitor),)
    elif problem == "far_centroid":
        centers = {2: (1.5, 0.)}
    result = observe(guard, groups, 1.4 if problem == "expired" else 1.2,
                     extra=extra, centers=centers)
    assert result["blocked"]
    assert result["blocking_tracks"][0]["track_id"] == 1
    assert result["blocking_tracks"][0]["original_last_seen_time"] == 1.
    assert not result["observed_fragment_merges"]


def test_rejected_partial_surface_never_renews_missing_fragment_clock():
    guard = prepared()
    for now in (1.2, 1.3):
        result = observe(guard, {2: np.vstack([SURFACE[1:], BODY])}, now)
        assert result["blocked"]
        assert result["blocking_tracks"][0]["original_last_seen_time"] == 1.
        assert guard.positive_surfaces[1][1] == 1.
        np.testing.assert_array_equal(guard.positive_surfaces[1][0], SURFACE)
    assert result["positive_surface_merge_rejections"][0]["reason"] == "original_surface_not_fully_observed_now"
    np.testing.assert_array_equal(
        result["positive_surface_merge_rejections"][0]["original_surface_points_world_xy"],
        SURFACE,
    )


def test_missing_geometry_rejects_without_mutating_surface_or_source_state():
    guard = prepared()
    with pytest.raises(ValueError, match="aligned current geometry"):
        guard.observe((tracked(2, BODY),), source_time=1.2, robot_world_xy=(0, 0))
    assert guard.last_source_time == 1.
    assert guard.positive_surfaces[1][1] == 1.


def test_expired_surfaces_do_not_leave_live_geometric_evidence():
    guard = prepared()
    assert observe(guard, {2: BODY}, 1.4)["blocked"]
    assert 1 not in guard.positive_surfaces
    assert guard.observed[1][1] == 1.


def test_surface_opt_in_requires_existing_positive_association_scope():
    with pytest.raises(TypeError):
        NearbyTrackLossConfig(allow_positive_surface_merges=1)
    with pytest.raises(ValueError, match="positive extent"):
        NearbyTrackLossConfig(allow_positive_surface_merges=True)


def test_two_missing_surfaces_cannot_both_claim_one_current_parent():
    guard = NearbyTrackLossGuard(NearbyTrackLossConfig(
        allow_observed_fragment_merges=True, allow_positive_extent_merges=True,
        allow_positive_surface_merges=True))
    observe(guard, {1: SURFACE, 3: SURFACE, 2: BODY}, 1.)
    result = observe(guard, {2: np.vstack([SURFACE, BODY])}, 1.2)
    assert result["blocked"]
    assert {row["track_id"] for row in result["blocking_tracks"]} == {1, 3}
    assert not result["observed_fragment_merges"]


def test_unique_whole_surface_can_cover_a_far_centroid_with_a_partial_neighbor():
    body = np.column_stack((np.linspace(1.75, 1.85, 30), np.zeros(30)))
    patch = np.array([[.95, 0.], [.95, .01], [.95, -.01]])
    guards = [NearbyTrackLossGuard(NearbyTrackLossConfig(
        allow_observed_fragment_merges=True, allow_positive_extent_merges=True,
        allow_positive_surface_merges=True, allow_unique_full_surface_merges=enabled))
        for enabled in (False, True)]
    for guard in guards:
        observe(guard, {1: SURFACE, 2: body, 3: patch}, 1.)
    groups = {2: np.vstack([SURFACE, body]), 3: patch}
    assert observe(guards[0], groups, 1.2)["blocked"]
    result = observe(guards[1], groups, 1.2)
    assert not result["blocked"]
    event, = result["observed_fragment_merges"]
    proof = event["positive_surface_proof"]
    assert proof["correspondence_basis"] == "unique_whole_surface_positive_coverage"
    assert proof["parent_centroid_distance_m"] > .25
    assert proof["neighborhood_supported_track_ids"] == [2, 3]
    assert proof["whole_surface_supported_track_ids"] == [2]
    assert event["original_last_seen_time"] == event["coobservation_source_time"] == 1.
    assert event["translation_world_xy"] == [0., 0.]
    assert set(guards[1].observed) == {2, 3}


@pytest.mark.parametrize("problem", ["one_missing", "two_whole_parents", "unclustered", "no_pair", "expired"])
def test_unique_whole_surface_keeps_incomplete_ambiguous_or_expired_stops(problem):
    guard = prepared(unique=True, coobserved=problem != "no_pair")
    groups = {2: np.vstack([SURFACE, BODY])}
    extra = ()
    if problem == "one_missing":
        groups[2] = np.vstack([SURFACE[1:], BODY])
    elif problem == "two_whole_parents":
        groups[3] = SURFACE + .01
    elif problem == "unclustered":
        extra = ((-1, np.array([[.95, 0.], [.95, .01], [.95, -.01]])),)
    result = observe(guard, groups, 1.4 if problem == "expired" else 1.2, extra=extra)
    assert result["blocked"]
    assert result["blocking_tracks"][0]["original_last_seen_time"] == 1.
    assert not result["observed_fragment_merges"]


def test_unique_whole_surface_opt_in_requires_original_surface_scope():
    with pytest.raises(TypeError):
        NearbyTrackLossConfig(allow_unique_full_surface_merges=1)
    with pytest.raises(ValueError, match="requires positive surface"):
        NearbyTrackLossConfig(allow_unique_full_surface_merges=True)
