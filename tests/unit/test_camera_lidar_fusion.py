import numpy as np
import pytest

from robot.perception.camera_lidar_fusion import (
    FUSION_LABEL_SOURCE,
    CameraLidarFusionConfig,
    CameraLidarSemanticFusion,
    FThetaCameraCalibration,
    project_fisheye_polynomial_points,
)
from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)
from robot.perception.visualization import render_camera_lidar_fusion_overlay
from robot.safety.obstacle_tracking import (
    ObstacleTrackingResult,
    TrackedObstacle,
)


def make_calibration() -> FThetaCameraCalibration:
    return FThetaCameraCalibration(
        nominal_width=100,
        nominal_height=100,
        optical_center_x=50.0,
        optical_center_y=50.0,
        maximum_fov_degrees=180.0,
        polynomial_coefficients=(0.0, 0.01, 0.0, 0.0, 0.0),
        output_width=100,
        output_height=100,
    )


def make_track(track_id: int, point_count: int = 2) -> TrackedObstacle:
    return TrackedObstacle(
        track_id=track_id,
        centroid_world_xy=(float(track_id), 0.0),
        centroid_robot_xy=(float(track_id), 0.0),
        velocity_world_xy=(0.0, 0.0),
        velocity_robot_xy=(0.0, 0.0),
        point_count=point_count,
        observation_count=1,
        matched_previous_track=False,
    )


def make_tracking_result(point_track_ids: list[int]) -> ObstacleTrackingResult:
    track_ids = sorted(set(point_track_ids))
    return ObstacleTrackingResult(
        timestamp_seconds=1.0,
        point_velocities_robot_xy=np.zeros((len(point_track_ids), 2)),
        point_track_ids=np.asarray(point_track_ids, dtype=np.int64),
        tracked_obstacles=tuple(
            make_track(track_id, point_track_ids.count(track_id))
            for track_id in track_ids
        ),
        cluster_count=len(track_ids),
        matched_track_count=0,
        new_track_count=len(track_ids),
    )


def make_detection_frame(*masks: np.ndarray) -> SemanticDetectionFrame:
    detections = tuple(
        SemanticDetection(
            class_id=index,
            semantic_class=("chair" if index == 0 else "person"),
            confidence=(0.8 if index == 0 else 0.9),
            bounding_box=PixelBoundingBox(1.0, 1.0, 99.0, 99.0),
            mask=np.asarray(mask, dtype=np.bool_),
        )
        for index, mask in enumerate(masks)
    )
    return SemanticDetectionFrame(
        source_frame_id=1,
        source_timestamp_seconds=1.0,
        image_height=100,
        image_width=100,
        detections=detections,
        model_name="test-seg",
        inference_wall_seconds=0.01,
    )


def test_fisheye_projection_uses_camera_minus_z_and_image_y_down() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, -2.0],
            [0.2, 0.0, -2.0],
            [0.0, 0.2, -2.0],
            [0.0, 0.0, 2.0],
        ]
    )

    result = project_fisheye_polynomial_points(
        points,
        camera_from_source=np.eye(4),
        calibration=make_calibration(),
    )

    np.testing.assert_array_equal(result.source_indices, [0, 1, 2])
    np.testing.assert_allclose(result.pixel_xy[0], [50.0, 50.0])
    assert result.pixel_xy[1, 0] > 50.0
    assert result.pixel_xy[2, 1] < 50.0
    np.testing.assert_allclose(result.camera_depths, [2.0, 2.0, 2.0])
    assert not result.pixel_xy.flags.writeable


def test_projection_applies_source_to_camera_transform() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [0.2, 0.0, -2.0]

    result = project_fisheye_polynomial_points(
        np.asarray([[0.0, 0.0, 0.0]]),
        camera_from_source=transform,
        calibration=make_calibration(),
    )

    assert result.source_indices.tolist() == [0]
    assert result.pixel_xy[0, 0] > 50.0


def test_fusion_attaches_mask_class_to_existing_track_id() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, -2.0],
            [0.1, 0.0, -2.0],
            [-0.8, 0.0, -2.0],
            [-0.7, 0.0, -2.0],
        ]
    )
    chair_mask = np.zeros((100, 100), dtype=np.bool_)
    chair_mask[45:56, 45:57] = True

    result = CameraLidarSemanticFusion(
        CameraLidarFusionConfig(mask_dilation_radius_pixels=0)
    ).fuse(
        detection_frame=make_detection_frame(chair_mask),
        obstacle_points_source_xyz=points,
        tracking_result=make_tracking_result([7, 7, 9, 9]),
        camera_from_source=np.eye(4),
        calibration=make_calibration(),
    )

    obstacles = {
        obstacle.track_id: obstacle
        for obstacle in result.semantic_labels.semantic_obstacles
    }
    assert obstacles[7].semantic_class == "chair"
    assert obstacles[7].label_source == FUSION_LABEL_SOURCE
    assert obstacles[7].semantic_confidence == pytest.approx(0.8)
    assert obstacles[9].is_unknown
    assert [item.status for item in result.associations] == [
        "matched",
        "insufficient_mask_overlap",
    ]
    assert result.matched_detection_indices == (0,)
    assert result.semantic_labels.unmatched_label_count == 0

    overlay = render_camera_lidar_fusion_overlay(
        np.zeros((100, 100, 3), dtype=np.uint8),
        make_detection_frame(chair_mask),
        result,
    )
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8
    assert np.any(overlay)


def test_fusion_falls_back_to_unknown_when_two_classes_are_ambiguous() -> None:
    points = np.asarray([[0.0, 0.0, -2.0], [0.1, 0.0, -2.0]])
    first_mask = np.zeros((100, 100), dtype=np.bool_)
    first_mask[45:56, 45:57] = True
    second_mask = first_mask.copy()

    result = CameraLidarSemanticFusion(
        CameraLidarFusionConfig(
            mask_dilation_radius_pixels=0,
            ambiguity_score_margin=0.11,
        )
    ).fuse(
        detection_frame=make_detection_frame(first_mask, second_mask),
        obstacle_points_source_xyz=points,
        tracking_result=make_tracking_result([3, 3]),
        camera_from_source=np.eye(4),
        calibration=make_calibration(),
    )

    assert result.semantic_labels.semantic_obstacles[0].is_unknown
    assert result.associations[0].status == "ambiguous"
    assert result.semantic_labels.unmatched_label_count == 2


def test_duplicate_same_class_masks_are_not_semantically_ambiguous() -> None:
    points = np.asarray([[0.0, 0.0, -2.0], [0.1, 0.0, -2.0]])
    mask = np.zeros((100, 100), dtype=np.bool_)
    mask[45:56, 45:57] = True
    detections = tuple(
        SemanticDetection(
            class_id=0,
            semantic_class="chair",
            confidence=confidence,
            bounding_box=PixelBoundingBox(45.0, 45.0, 57.0, 56.0),
            mask=mask,
        )
        for confidence in (0.80, 0.78, 0.76)
    )
    detection_frame = SemanticDetectionFrame(
        source_frame_id=1,
        source_timestamp_seconds=1.0,
        image_height=100,
        image_width=100,
        detections=detections,
        model_name="test-seg",
        inference_wall_seconds=0.01,
    )

    result = CameraLidarSemanticFusion(
        CameraLidarFusionConfig(
            mask_dilation_radius_pixels=0,
            ambiguity_score_margin=0.50,
        )
    ).fuse(
        detection_frame=detection_frame,
        obstacle_points_source_xyz=points,
        tracking_result=make_tracking_result([3, 3]),
        camera_from_source=np.eye(4),
        calibration=make_calibration(),
    )

    obstacle = result.semantic_labels.semantic_obstacles[0]
    assert obstacle.semantic_class == "chair"
    assert result.associations[0].status == "matched"
    assert result.associations[0].detection_index == 0
    assert result.semantic_labels.unmatched_label_count == 2


def test_duplicate_masks_can_fall_back_to_an_unused_same_class_instance() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, -2.0],
            [0.1, 0.0, -2.0],
            [0.0, 0.0, -2.0],
            [0.1, 0.0, -2.0],
        ]
    )
    mask = np.zeros((100, 100), dtype=np.bool_)
    mask[45:56, 45:57] = True
    detections = tuple(
        SemanticDetection(
            class_id=0,
            semantic_class="chair",
            confidence=0.8,
            bounding_box=PixelBoundingBox(45.0, 45.0, 57.0, 56.0),
            mask=mask,
        )
        for _ in range(2)
    )
    detection_frame = SemanticDetectionFrame(
        source_frame_id=1,
        source_timestamp_seconds=1.0,
        image_height=100,
        image_width=100,
        detections=detections,
        model_name="test-seg",
        inference_wall_seconds=0.01,
    )

    result = CameraLidarSemanticFusion(
        CameraLidarFusionConfig(mask_dilation_radius_pixels=0)
    ).fuse(
        detection_frame=detection_frame,
        obstacle_points_source_xyz=points,
        tracking_result=make_tracking_result([3, 3, 4, 4]),
        camera_from_source=np.eye(4),
        calibration=make_calibration(),
    )

    assert all(
        association.status == "matched"
        for association in result.associations
    )
    assert result.matched_detection_indices == (0, 1)


def test_fusion_rejects_track_that_only_grazes_a_detection() -> None:
    inside_points = [[0.0, 0.0, -2.0], [0.1, 0.0, -2.0]]
    outside_points = [[-0.8, 0.0, -2.0]] * 6
    points = np.asarray(inside_points + outside_points)
    mask = np.zeros((100, 100), dtype=np.bool_)
    mask[45:56, 45:57] = True
    detection_frame = SemanticDetectionFrame(
        source_frame_id=1,
        source_timestamp_seconds=1.0,
        image_height=100,
        image_width=100,
        detections=(
            SemanticDetection(
                class_id=0,
                semantic_class="chair",
                confidence=0.8,
                bounding_box=PixelBoundingBox(45.0, 45.0, 57.0, 56.0),
                mask=mask,
            ),
        ),
        model_name="test-seg",
        inference_wall_seconds=0.01,
    )

    result = CameraLidarSemanticFusion(
        CameraLidarFusionConfig(
            minimum_mask_hits=2,
            minimum_overlap_fraction=0.20,
            mask_dilation_radius_pixels=0,
            projected_median_box_margin_pixels=0,
        )
    ).fuse(
        detection_frame=detection_frame,
        obstacle_points_source_xyz=points,
        tracking_result=make_tracking_result([6] * len(points)),
        camera_from_source=np.eye(4),
        calibration=make_calibration(),
    )

    assert result.semantic_labels.semantic_obstacles[0].is_unknown
    assert result.associations[0].status == "projected_median_outside_box"
    assert result.associations[0].mask_hit_count == 2
    assert not result.associations[0].projected_median_inside_box


def test_mask_dilation_handles_small_projection_calibration_error() -> None:
    points = np.asarray([[0.0, 0.0, -2.0], [0.02, 0.0, -2.0]])
    mask = np.zeros((100, 100), dtype=np.bool_)
    mask[50, 53] = True

    result = CameraLidarSemanticFusion(
        CameraLidarFusionConfig(
            minimum_mask_hits=2,
            mask_dilation_radius_pixels=3,
        )
    ).fuse(
        detection_frame=make_detection_frame(mask),
        obstacle_points_source_xyz=points,
        tracking_result=make_tracking_result([4, 4]),
        camera_from_source=np.eye(4),
        calibration=make_calibration(),
    )

    assert result.associations[0].is_matched


def test_fusion_rejects_tracker_point_alignment_mismatch() -> None:
    with pytest.raises(ValueError, match="must align"):
        CameraLidarSemanticFusion().fuse(
            detection_frame=make_detection_frame(
                np.zeros((100, 100), dtype=np.bool_)
            ),
            obstacle_points_source_xyz=np.zeros((2, 3)),
            tracking_result=make_tracking_result([1]),
            camera_from_source=np.eye(4),
            calibration=make_calibration(),
        )


def test_invalid_fisheye_calibration_is_rejected() -> None:
    with pytest.raises(ValueError, match="five"):
        FThetaCameraCalibration(
            nominal_width=100,
            nominal_height=100,
            optical_center_x=50.0,
            optical_center_y=50.0,
            maximum_fov_degrees=180.0,
            polynomial_coefficients=(0.0, 0.01),
            output_width=100,
            output_height=100,
        )
