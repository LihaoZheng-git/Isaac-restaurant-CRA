from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot.perception.carried_object import (
    CarriedObjectConfig,
    CarryObservation,
    CarriedRelationHistory,
    propose_carried_object,
)
from robot.perception.camera_lidar_fusion import ProjectedLidarPoints
from robot.perception.human_pose import HumanPose, HumanPoseFrame, match_person_pose
from robot.perception.semantic_detection import SemanticDetection, PixelBoundingBox
from robot.perception.sam_instance_segmentation import SamInstanceResult
from robot.risk.carried_group_policy import maximum_member_policy
from robot.risk.cra import PhysicalRiskProfile, RiskFactorEstimate
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig


def observation(t, frame, *, human=None, obj=None, **kwargs):
    h = (0.5 * t, 0.0, 1.0) if human is None else human
    return CarryObservation(
        1, 2, frame, t, h, (h[0], 0.2, 1.1) if obj is None else obj, 0.2, 0.9, **kwargs
    )


def test_association_requires_actual_separate_world_co_motion():
    history = CarriedRelationHistory(CarriedObjectConfig())
    for frame, t in enumerate((1.0, 1.2, 1.4)):
        result = history.update(observation(t, frame))
    assert result["confirmed"]
    assert result["original_window_start"] == 1.0
    assert result["relative_drift_m"] == pytest.approx(0)
    assert result["human_translation_m"] == pytest.approx(0.2)


@pytest.mark.parametrize("static_object", [False, True])
def test_stationary_proximity_or_unmatched_motion_does_not_establish_carrying(
    static_object,
):
    history = CarriedRelationHistory(CarriedObjectConfig())
    for frame, t in enumerate((1.0, 1.2, 1.4)):
        result = history.update(
            observation(
                t,
                frame,
                human=(0.0, 0.0, 1.0) if not static_object else None,
                obj=(0.0, 0.2, 1.1),
            )
        )
    assert not result["confirmed"]


def test_gap_resets_relation_without_refreshing_previous_observations():
    history = CarriedRelationHistory(CarriedObjectConfig())
    for frame, t in enumerate((1.0, 1.2, 1.4, 2.0)):
        result = history.update(observation(t, frame))
    assert not result["confirmed"]
    assert result["original_window_start"] == 2.0


def test_hand_relative_motion_accepts_a_carried_prop_during_a_person_turn():
    history = CarriedRelationHistory(CarriedObjectConfig())
    for frame, t in enumerate((1.0, 1.2, 1.4)):
        angle = frame * np.pi / 4
        h = np.array([0.5 * t, 0.0, 1.0])
        hand = h + [0.3 * np.cos(angle), 0.3 * np.sin(angle), 0.1]
        result = history.update(
            observation(
                t,
                frame,
                human=tuple(h),
                obj=tuple(hand + [0, 0, 0.05]),
                hand_world_xyz=tuple(hand),
            )
        )
    assert result["confirmed"]
    assert result["relative_drift_m"] < 1e-12
    assert "wrist" in result["motion_reference"]


def test_hand_reference_does_not_accept_a_stationary_object_beside_a_moving_person():
    history = CarriedRelationHistory(CarriedObjectConfig())
    for frame, t in enumerate((1.0, 1.2, 1.4)):
        result = history.update(
            observation(
                t,
                frame,
                obj=(0.5, 0.2, 1.1),
                hand_world_xyz=(0.5 * t, 0.2, 1.05),
            )
        )
    assert not result["confirmed"]
    assert not result["checks"]["co_motion"]


@pytest.mark.parametrize("hand", [(1, 2), (1, np.nan, 3)])
def test_invalid_hand_geometry_cannot_advance_relation_history(hand):
    history = CarriedRelationHistory(CarriedObjectConfig())
    history.update(observation(1.0, 1))
    with pytest.raises(ValueError, match="finite same-frame"):
        history.update(observation(1.2, 2, hand_world_xyz=hand))
    assert len(history.samples[(1, 2)]) == 1


def test_missing_instance_clears_relationship_and_repeated_epoch_rejected():
    history = CarriedRelationHistory(CarriedObjectConfig())
    history.update(observation(1.0, 1))
    with pytest.raises(ValueError, match="epochs"):
        history.update(observation(1.0, 1))
    assert len(history.samples[(1, 2)]) == 1
    result = history.update(observation(1.2, 2, fresh_separate_instances=False))
    assert not result["confirmed"] and not history.samples


def policy(risk):
    absent = RiskFactorEstimate(0.0, 0.0, 1.0, "controlled_numeric_zero")
    return CRASafeSAGEPolicy().evaluate(
        PhysicalRiskProfile(
            absent,
            absent,
            RiskFactorEstimate(risk, 1.0, 1.0, "controlled_numeric_test"),
            absent,
        )
    )


@pytest.mark.parametrize("human,bowl", [(0.5, 0.8), (0.9, 0.2), (0.6, 0.6)])
def test_group_is_exact_maximum_and_never_blends_member_factors(human, bowl):
    members = {1: policy(human), 2: policy(bowl)}
    original = [p.as_dict() for p in members.values()]
    shared, detail = maximum_member_policy(members)
    assert shared.risk == max(human, bowl)
    assert shared.tangential_strength == 1.25
    assert [p.as_dict() for p in members.values()] == original
    assert detail["member_phi"] == pytest.approx({"1": human, "2": bowl})
    assert detail["dominant_member"] == (1 if human >= bowl else 2)


def test_instance_partition_preserves_touching_geometry_and_own_motion():
    tracker = ObstacleTracker(ObstacleTrackerConfig(minimum_cluster_points=4))
    pts = np.array(
        [[x, y] for x in (0.0, 0.02) for y in (0.0, 0.02)]
        + [[x, y] for x in (0.04, 0.06) for y in (0.0, 0.02)]
    )
    groups = (np.arange(4), np.arange(4, 8))
    first = tracker.update(
        pts,
        robot_position_world_xy=np.zeros(2),
        robot_yaw=0.0,
        timestamp_seconds=1.0,
        point_instance_groups=groups,
    )
    assert len(first.tracked_obstacles) == 2
    assert sum(t.point_count for t in first.tracked_obstacles) == 8
    moved = pts.copy()
    moved[4:, 1] += 0.04
    second = tracker.update(
        moved,
        robot_position_world_xy=np.zeros(2),
        robot_yaw=0.0,
        timestamp_seconds=1.2,
        point_instance_groups=groups,
    )
    assert second.tracked_obstacles[0].velocity_world_xy[1] == 0.0
    assert second.tracked_obstacles[1].velocity_world_xy[1] > 0.0


@pytest.mark.parametrize(
    "groups",
    [
        (np.array([0, 1, 2, 3]), np.array([0, 4, 5, 6])),
        (np.array([0, 1, 2, 99]),),
        (np.array([0.0, 1.0, 2.0, 3.0]),),
    ],
)
def test_invalid_overlapping_or_out_of_range_partitions_rejected(groups):
    with pytest.raises(ValueError, match="disjoint"):
        ObstacleTracker().update(
            np.zeros((8, 2)),
            robot_position_world_xy=np.zeros(2),
            robot_yaw=0.0,
            timestamp_seconds=1.0,
            point_instance_groups=groups,
        )


def fixture(reject=False):
    mask = np.zeros((300, 200), bool)
    mask[40:260, 70:110] = True
    person = SemanticDetection(
        0, "person", 0.9, PixelBoundingBox(70, 40, 110, 260), mask
    )
    kp = np.zeros((17, 3))
    kp[9] = [108, 145, 0.9]
    pose = HumanPose((70, 40, 110, 260), 0.9, kp)
    pixels = np.array(
        [[100, 140], [102, 143], [105, 145]] + [[115 + i, 135 + i] for i in range(8)]
    )
    xyz = np.array(
        [[2.0, 0.0, 1.0]] * 3 + [[2.0, 0.2 + i * 0.002, 1.0] for i in range(8)]
    )
    projection = ProjectedLidarPoints(
        np.arange(len(xyz)), pixels.astype(float), pixels, np.full(len(xyz), 2.0)
    )
    packet = SimpleNamespace(
        rgb=np.zeros((300, 200, 3), np.uint8),
        obstacle_xyz=xyz,
        camera_frame_id=7,
        camera_timestamp_seconds=1.0,
    )

    class Segmenter:
        def set_image(self, rgb):
            self.shape = rgb.shape[:2]

        def propose(self, ids, local, **kwargs):
            if reject:
                return SamInstanceResult(None, ("controlled rejected mask",))
            crop = np.zeros(self.shape, bool)
            crop[123:143, 70:88] = True
            detection = SemanticDetection(
                0, "unknown", 0.75, PixelBoundingBox(70, 123, 88, 143), crop
            )
            return SamInstanceResult(
                detection, (), prompt_xy=(75, 130), model_quality=0.9
            )

    return packet, projection, person, pose, Segmenter()


def test_crop_preserves_source_pixels_quality_and_exact_positive_ownership():
    packet, proj, person, pose, sam = fixture()
    result, detail = propose_carried_object(
        packet, proj, person, pose, sam, CarriedObjectConfig()
    )
    assert result is not None
    np.testing.assert_array_equal(np.flatnonzero(result["selected"]), np.arange(3, 11))
    assert result["proposal"].detection.mask.shape == packet.rgb.shape[:2]
    assert result["proposal"].model_quality == 0.9
    assert result["person_mask_overlap"] == 0.0
    assert detail[0]["source_frame_id"] == 7


def test_rejected_segmentation_never_becomes_a_low_risk_object():
    args = fixture(reject=True)
    result, details = propose_carried_object(*args, CarriedObjectConfig())
    assert result is None and details[-1]["accepted_candidates"] == 0


def test_pose_match_rejects_competing_people_and_copies_keypoints():
    _, _, person, pose, _ = fixture()
    frame = HumanPoseFrame(7, 1.0, (300, 200), (pose,))
    assert match_person_pose(person, frame) is pose
    assert not pose.keypoints.flags.writeable
    assert match_person_pose(person, replace(frame, people=(pose, pose))) is None
