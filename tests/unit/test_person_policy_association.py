import copy
from dataclasses import replace

import numpy as np
import pytest
from test_visual_body_fragments import fixture

from robot.perception.camera_lidar_fusion import ProjectedLidarPoints
from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE
from robot.perception.human_mass_prior import HumanMassPriorConfig
from robot.perception.person_policy_association import fresh_person_policy_associations
from robot.safety.motion_aware_passing import MotionAwarePassing, MotionAwarePassingConfig


def inputs():
    tracking, _, projected, frame, parents = fixture()
    projection = ProjectedLidarPoints(projected.source_indices,
        projected.pixel_indices_xy.astype(float), projected.pixel_indices_xy,
        np.full(len(projected.source_indices), 2.))
    parents[0].update(source=HUMAN_ENVELOPE_SOURCE, status="prior")
    return [tracking, projection, frame, parents]


def associate(args, **kwargs):
    return fresh_person_policy_associations(*args, camera_timestamp_seconds=1.05,
        camera_frame_id=4, config=HumanMassPriorConfig("bounded separated adult test"), **kwargs)


def test_current_detector_cue_keeps_old_identity_clocks_and_changes_no_physical_input():
    args = inputs()
    args[3][0]["original_person_camera_timestamp"] = .1
    before = copy.deepcopy(args[3]), args[0].point_track_ids.copy()
    selected = associate(args)
    assert set(selected) == {1}
    cue = selected[1]
    assert cue["status"] == "inferred" and cue["camera_timestamp_seconds"] == 1.05
    assert cue["original_parent_person_timestamp"] == .1
    assert cue["lidar_timestamp_seconds"] == 1.
    assert not cue["sam_instance_assessment"] and not cue["physical_evidence_refreshed"]
    assert cue["parent_mask_coverage"] == cue["parent_foreground_coverage"] == 1
    assert args[3] == before[0]
    np.testing.assert_array_equal(args[0].point_track_ids, before[1])


@pytest.mark.parametrize("problem", ["boundary", "low_confidence", "competitor", "low_coverage",
    "too_few", "behind_foreground", "missing_parent", "old_lidar", "future_identity", "wrong_source", "unmatched"])
def test_unsupported_or_ambiguous_current_cue_is_ineligible(problem):
    args = inputs()
    tracking, projection, frame, parents = args
    if problem == "boundary":
        mask = frame.detections[0].mask.copy()
        mask[0, 20] = True
        args[2] = replace(frame, detections=(replace(frame.detections[0], mask=mask),))
    elif problem == "low_confidence":
        args[2] = replace(frame, detections=(replace(frame.detections[0], confidence=.49),))
    elif problem == "competitor":
        args[2] = replace(frame, detections=frame.detections * 2)
    elif problem == "low_coverage":
        xy = projection.pixel_indices_xy.copy()
        xy[:5] = [1, 1]
        args[1] = replace(projection, pixel_indices_xy=xy, pixel_xy=xy.astype(float))
    elif problem == "too_few":
        args[1] = ProjectedLidarPoints(np.arange(19), projection.pixel_xy[:19],
            projection.pixel_indices_xy[:19], projection.camera_depths[:19])
    elif problem == "behind_foreground":
        tracking.point_track_ids = np.r_[tracking.point_track_ids, np.full(8, -1)]
        n = len(tracking.point_track_ids)
        xy = np.tile([20, 20], (n, 1))
        args[1] = ProjectedLidarPoints(np.arange(n), xy.astype(float), xy,
            np.r_[projection.camera_depths, np.full(8, .5)])
    elif problem == "missing_parent":
        args[3] = []
    elif problem == "old_lidar":
        parents[0]["lidar_timestamp_seconds"] = .9
    elif problem == "future_identity":
        parents[0]["original_person_camera_timestamp"] = 2.
    elif problem == "wrong_source":
        parents[0]["source"] = "oracle_actor_name"
    elif problem == "unmatched":
        tracking.tracked_obstacles = tuple(replace(t, matched_previous_track=False)
                                          for t in tracking.tracked_obstacles)
    assert associate(args) == {}


@pytest.mark.parametrize("problem", ["old_image", "wrong_frame", "outside_pixels", "duplicate_projection"])
def test_misaligned_source_rejected(problem):
    args = inputs()
    if problem == "old_image":
        args[2] = replace(args[2], source_timestamp_seconds=.8)
    elif problem == "wrong_frame":
        args[2] = replace(args[2], source_frame_id=5)
    elif problem == "outside_pixels":
        xy = args[1].pixel_indices_xy.copy()
        xy[0] = [50, 20]
        args[1] = replace(args[1], pixel_indices_xy=xy)
    elif problem == "duplicate_projection":
        ids = args[1].source_indices.copy()
        ids[-1] = ids[0]
        args[1] = replace(args[1], source_indices=ids)
    with pytest.raises(ValueError):
        associate(args)


def test_two_validated_parents_cannot_share_one_current_person_mask():
    args = inputs()
    tracking, projection, _, parents = args
    parent = tracking.tracked_obstacles[0]
    tracking.tracked_obstacles = (parent, replace(parent, track_id=2))
    tracking.point_track_ids = np.r_[np.full(40, 1), np.full(40, 2)]
    xy = np.tile([20, 20], (80, 1))
    args[1] = ProjectedLidarPoints(np.arange(80), xy.astype(float), xy, np.full(80, 2.))
    args[3] = parents + [{**parents[0], "track_id": 2}]
    assert associate(args) == {}


def test_fresh_class_cue_enables_already_qualified_motion_without_lowering_motion_gates():
    args = inputs()
    tracking = args[0]
    original = tracking.tracked_obstacles[0]
    child = tracking.tracked_obstacles[1]
    passing = MotionAwarePassing(MotionAwarePassingConfig(require_visual_person_selection=True))
    without_class = copy.deepcopy(passing)
    for time, x in [(.8, -.6), (.9, -.55), (1., -.5)]:
        tracking.timestamp_seconds = time
        tracking.tracked_obstacles = (replace(original, centroid_world_xy=(x, 1.),
            velocity_world_xy=(.5, 0.), observation_count=4 + round(10 * (time-.8))), child)
        fresh = associate(args) if time == 1. else {}
        _, absent = without_class.update(tracking, robot_xy=(0., 0.), robot_velocity_xy=(0., .1),
            goal_xy=(0., 3.), visual_person_track_ids=())
        sides, audit = passing.update(tracking, robot_xy=(0., 0.), robot_velocity_xy=(0., .1),
            goal_xy=(0., 3.), visual_person_track_ids=tuple(fresh))
    assert absent["decision"] == "unverified_visual_person_target"
    assert sides and audit["decision"] == "selected_trailing_tangent"
    assert audit["original_selection_visual_person_verified"] is True
    assert audit["encounter_selected_timestamp"] == 1.
    assert passing.config.minimum_consistent_observations == 3
