from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot.evaluation.nearby_track_loss_guard import NearbyTrackLossConfig, NearbyTrackLossGuard
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection, SemanticDetectionFrame
from robot.perception.visual_body_fragments import associate_visual_body_fragments
from robot.safety.obstacle_tracking import TrackedObstacle


def fixture():
    parent = np.column_stack((np.ones(40), np.linspace(-.05, .05, 40), np.ones(40)))
    part = np.array([[1.20, -.02, 1.], [1.20, 0., 1.], [1.20, .02, 1.]])
    points = np.vstack((parent, part))
    ids = np.r_[np.full(40, 1), np.full(3, 2)]
    tracks = tuple(TrackedObstacle(k, tuple(q[:, :2].mean(axis=0)), tuple(q[:, :2].mean(axis=0)),
        (0, 0), (0, 0), len(q), 4, True) for k, q in [(1, parent), (2, part)])
    tracking = SimpleNamespace(timestamp_seconds=1., point_track_ids=ids, tracked_obstacles=tracks)
    projection = SimpleNamespace(source_indices=np.arange(len(points)), pixel_indices_xy=np.tile([20, 20], (len(points), 1)))
    mask = np.zeros((40, 40), bool)
    mask[5:35, 5:35] = True
    person = SemanticDetection(0, "person", .85, PixelBoundingBox(5, 5, 35, 35), mask)
    frame = SemanticDetectionFrame(4, 1.05, 40, 40, (person,), "audited_person_model", .01)
    parents = [dict(track_id=1, lidar_timestamp_seconds=1., current_support_count=40,
                    identity_basis="retained_person_identity_prior_with_current_positive_LiDAR",
                    original_person_camera_timestamp=.8, original_person_camera_frame_id=2)]
    return tracking, points, projection, frame, parents


def associate(args):
    return associate_visual_body_fragments(*args, camera_timestamp_seconds=1.05, camera_frame_id=4)


def observe(guard, tracking, points, proofs=()):
    return guard.observe(tracking.tracked_obstacles, source_time=tracking.timestamp_seconds,
        robot_world_xy=(0, 0), current_points_world_xy=points[:, :2],
        current_point_track_ids=tracking.point_track_ids,
        current_points_source_time=tracking.timestamp_seconds,
        current_visual_part_associations=proofs)


def test_whole_current_part_association_preserves_original_inputs_and_identity_epochs():
    args = fixture()
    before = args[1].copy(), args[0].point_track_ids.copy()
    proof, = associate(args)
    assert (proof["child_track_id"], proof["parent_track_id"]) == (2, 1)
    assert proof["status"] == "inferred"
    assert proof["original_person_camera_timestamp"] == .8
    assert proof["source_timestamp_seconds"] == 1.
    np.testing.assert_array_equal(args[1], before[0])
    np.testing.assert_array_equal(args[0].point_track_ids, before[1])


@pytest.mark.parametrize("problem", ["one_outside", "one_unprojected", "far_3d", "duplicate_points",
    "competing_mask", "boundary", "weak_parent", "stale_parent", "no_parent"])
def test_partial_or_ambiguous_evidence_never_groups(problem):
    args = list(fixture())
    track, points, projection, frame, parents = args
    if problem == "one_outside":
        projection.pixel_indices_xy[-1] = [1, 1]
    elif problem == "one_unprojected":
        projection.source_indices = projection.source_indices[:-1]
        projection.pixel_indices_xy = projection.pixel_indices_xy[:-1]
    elif problem == "far_3d":
        points[-1, 2] += .31
    elif problem == "duplicate_points":
        points[-3:] = points[-1]
    elif problem == "competing_mask":
        args[3] = replace(frame, detections=frame.detections * 2)
    elif problem == "boundary":
        mask = frame.detections[0].mask.copy()
        mask[0, 20] = True
        args[3] = replace(frame, detections=(replace(frame.detections[0], mask=mask),))
    elif problem == "weak_parent":
        projection.pixel_indices_xy[:5] = [1, 1]
    elif problem == "stale_parent":
        parents[0]["lidar_timestamp_seconds"] = .9
    elif problem == "no_parent":
        args[4] = []
    assert associate(args) == []


def test_frame_mismatch_rejected_before_guard_state_changes():
    args = list(fixture())
    args[3] = replace(args[3], source_frame_id=5)
    with pytest.raises(ValueError, match="aligned current"):
        associate(args)


def test_guard_resolves_still_observed_part_but_retains_parent_loss_watch():
    tracking, points, *rest = args = fixture()
    proof, = associate(args)
    guard = NearbyTrackLossGuard(NearbyTrackLossConfig(allow_current_visual_part_associations=True))
    observe(guard, tracking, points)
    assert set(guard.observed) == {1, 2}
    tracking.timestamp_seconds = 1.1
    proof = {**proof, "source_timestamp_seconds": 1.1, "camera_timestamp_seconds": 1.15}
    result = observe(guard, tracking, points, [proof])
    assert not result["blocked"] and set(guard.observed) == {1}
    tracking.timestamp_seconds = 1.3
    tracking.tracked_obstacles = ()
    tracking.point_track_ids = np.empty(0, int)
    result = observe(guard, tracking, np.empty((0, 3)))
    assert result["blocked"]
    assert result["blocking_tracks"][0]["track_id"] == 1


def test_missing_part_cannot_be_cleared_with_old_visual_proof():
    args = fixture()
    tracking, points = args[:2]
    proof, = associate(args)
    guard = NearbyTrackLossGuard(NearbyTrackLossConfig(allow_current_visual_part_associations=True))
    observe(guard, tracking, points)
    tracking.timestamp_seconds = 1.2
    tracking.tracked_obstacles = tracking.tracked_obstacles[:1]
    tracking.point_track_ids = tracking.point_track_ids[:40]
    with pytest.raises(ValueError, match="invalid current positive"):
        observe(guard, tracking, points[:40], [proof])
    assert guard.last_source_time == 1. and set(guard.observed) == {1, 2}


def test_legacy_guard_rejects_unrequested_association():
    args = fixture()
    with pytest.raises(ValueError, match="explicit opt-in"):
        observe(NearbyTrackLossGuard(), args[0], args[1], associate(args))


def test_parent_is_enrolled_when_only_child_is_inside_enrollment_radius():
    args = fixture()
    tracking, points = args[:2]
    proof, = associate(args)
    tracking.tracked_obstacles = tuple(replace(t, centroid_world_xy=(2.05, 0.) if t.track_id == 1 else (1.9, 0.))
                                      for t in tracking.tracked_obstacles)
    guard = NearbyTrackLossGuard(NearbyTrackLossConfig(allow_current_visual_part_associations=True))
    observe(guard, tracking, points, [proof])
    assert set(guard.observed) == {1}


def dense_fixture(parent_count, child_count):
    args = list(fixture())
    tracking, _, projection, _, parents = args
    parent = np.column_stack((np.ones(parent_count), np.linspace(-.05, .05, parent_count), np.ones(parent_count)))
    child = np.column_stack((np.full(child_count, 1.2), np.linspace(-.02, .02, child_count), np.ones(child_count)))
    args[1] = np.vstack((parent, child))
    tracking.point_track_ids = np.r_[np.full(parent_count, 1), np.full(child_count, 2)]
    tracking.tracked_obstacles = tuple(replace(t, point_count=parent_count if t.track_id == 1 else child_count)
                                      for t in tracking.tracked_obstacles)
    projection.source_indices = np.arange(parent_count + child_count)
    projection.pixel_indices_xy = np.tile([20, 20], (parent_count + child_count, 1))
    parents[0]["current_support_count"] = parent_count
    return args


def test_association_does_not_depend_on_absolute_sampling_density():
    # Identical physical surfaces and support ratio, 20x denser sampling.
    for parent_count, child_count in [(40, 3), (800, 60), (1006, 69)]:
        args = dense_fixture(parent_count, child_count)
        proof, = associate(args)
        assert proof["child_positive_count"] == child_count
        guard = NearbyTrackLossGuard(NearbyTrackLossConfig(allow_current_visual_part_associations=True))
        observe(guard, args[0], args[1], [proof])
        assert set(guard.observed) == {1}


def test_large_relative_component_still_rejected():
    assert not associate(dense_fixture(1000, 101))
