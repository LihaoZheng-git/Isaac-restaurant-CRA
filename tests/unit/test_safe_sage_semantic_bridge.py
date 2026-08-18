from dataclasses import replace

import numpy as np
import pytest

from robot.perception.camera_lidar_fusion import (
    CameraLidarFusionConfig,
    CameraLidarSemanticFusion,
    FThetaCameraCalibration,
)
from robot.perception.safe_sage_semantic_bridge import (
    SafeSAGECameraLidarObservation,
    SafeSAGESemanticPerceptionBridge,
)
from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)
from robot.perception.semantic_track_memory import SemanticTrackMemory
from robot.safety.class_safety_policy import (
    ClassSafetyParameters,
    ClassSafetyPolicy,
)
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig
from robot.safety.poisson_safety_function import PoissonSafetyFunctionConfig
from robot.safety.safe_sage_configuration_space import (
    CircularRobotFootprint,
    SafeSAGEConfigurationSpaceConfig,
)
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig
from robot.safety.safe_sage_guidance_policy import (
    PassingSide,
    SafeSAGEGuidanceParameters,
    SafeSAGEGuidancePolicy,
)
from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
from robot.safety.safe_sage_runtime import (
    SafeSAGEPerceptionObservation,
    SafeSAGEPerceptionRuntime,
)
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
)


OBSTACLE_POINTS_ROBOT_XY = np.array(
    [[1.9, -0.1], [1.9, 0.1], [2.1, -0.1], [2.1, 0.1]],
    dtype=np.float64,
)
OBSTACLE_POINTS_SOURCE_XYZ = np.array(
    [
        [-0.05, -0.05, -2.0],
        [-0.05, 0.05, -2.0],
        [0.05, -0.05, -2.0],
        [0.05, 0.05, -2.0],
    ],
    dtype=np.float64,
)


def make_bridge() -> SafeSAGESemanticPerceptionBridge:
    unknown = ClassSafetyParameters(
        risk=1.0,
        boundary_flux_magnitude=4.0,
    )
    chair = ClassSafetyParameters(
        risk=0.4,
        boundary_flux_magnitude=2.0,
    )
    class_policy = ClassSafetyPolicy({"unknown": unknown, "chair": chair})
    guidance_policy = SafeSAGEGuidancePolicy(
        {
            "unknown": SafeSAGEGuidanceParameters(
                safety_parameters=unknown,
                tangential_strength=0.0,
                passing_side=PassingSide.RIGHT,
            ),
            "chair": SafeSAGEGuidanceParameters(
                safety_parameters=chair,
                tangential_strength=0.0,
                passing_side=PassingSide.RIGHT,
            ),
        }
    )
    runtime = SafeSAGEPerceptionRuntime(
        obstacle_tracker=ObstacleTracker(
            ObstacleTrackerConfig(
                cluster_radius=0.5,
                minimum_cluster_points=2,
                maximum_association_distance=0.8,
                velocity_smoothing_factor=1.0,
            )
        ),
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=LocalOccupancyGridConfig(
            resolution=1.0,
            minimum_x=-5.0,
            maximum_x=8.0,
            minimum_y=-6.0,
            maximum_y=6.0,
        ),
        field_config=SafeSAGEFieldPipelineConfig(
            configuration_space_config=SafeSAGEConfigurationSpaceConfig(
                footprint=CircularRobotFootprint(radius=0.5),
                yaw_slice_count=3,
            ),
            guidance_field_config=TangentialLaplaceGuidanceFieldConfig(
                convergence_tolerance=1e-6,
                maximum_iterations=10_000,
                residual_check_interval=5,
                tangential_buffer_radius=2.0,
            ),
            h_full_config=SafeSAGEHFullConfig(
                yaw_slice_count=3,
                poisson_config=PoissonSafetyFunctionConfig(
                    convergence_tolerance=1e-6,
                    maximum_iterations=10_000,
                    residual_check_interval=5,
                ),
            ),
        ),
    )
    return SafeSAGESemanticPerceptionBridge(
        perception_runtime=runtime,
        fusion=CameraLidarSemanticFusion(
            CameraLidarFusionConfig(mask_dilation_radius_pixels=0)
        ),
        semantic_memory=SemanticTrackMemory(),
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


def make_detection_frame(
    frame_id: int,
    timestamp: float,
    *,
    include_chair: bool = True,
) -> SemanticDetectionFrame:
    detections: tuple[SemanticDetection, ...] = ()
    if include_chair:
        mask = np.zeros((100, 100), dtype=np.bool_)
        mask[40:61, 40:61] = True
        detections = (
            SemanticDetection(
                class_id=56,
                semantic_class="chair",
                confidence=0.8,
                bounding_box=PixelBoundingBox(40.0, 40.0, 61.0, 61.0),
                mask=mask,
            ),
        )
    return SemanticDetectionFrame(
        source_frame_id=frame_id,
        source_timestamp_seconds=timestamp,
        image_height=100,
        image_width=100,
        detections=detections,
        model_name="test-seg",
        inference_wall_seconds=0.01,
    )


def make_observation(
    frame_id: int,
    timestamp: float,
    *,
    points_robot_xy: np.ndarray = OBSTACLE_POINTS_ROBOT_XY,
    detection_timestamp: float | None = None,
    include_chair: bool = True,
) -> SafeSAGECameraLidarObservation:
    resolved_detection_timestamp = (
        timestamp if detection_timestamp is None else detection_timestamp
    )
    return SafeSAGECameraLidarObservation(
        perception=SafeSAGEPerceptionObservation(
            sensor_frame_id=frame_id,
            sensor_timestamp_seconds=timestamp,
            simulation_time_seconds=timestamp,
            obstacle_points_robot_xy=points_robot_xy,
            robot_position_world_xy=np.zeros(2),
            robot_yaw=0.0,
        ),
        obstacle_points_source_xyz=OBSTACLE_POINTS_SOURCE_XYZ,
        detection_frame=make_detection_frame(
            frame_id,
            resolved_detection_timestamp,
            include_chair=include_chair,
        ),
        camera_from_source=np.eye(4),
        calibration=make_calibration(),
    )


def occupied_risks(result) -> np.ndarray:
    grid = result.field_status.snapshot.payload.pipeline.configuration_space
    first_yaw_grid = grid.occupancy_grids[0]
    return first_yaw_grid.cell_risks[first_yaw_grid.occupied_mask]


def test_bridge_uses_unknown_policy_until_visual_label_is_confirmed() -> None:
    bridge = make_bridge()

    first = bridge.initialize(make_observation(1, 1.0))
    second = bridge.update(make_observation(2, 1.1))

    assert first.known_semantic_track_count == 0
    assert first.unknown_semantic_track_count == 1
    assert np.allclose(occupied_risks(first), 1.0)
    assert second.known_semantic_track_count == 1
    assert second.unknown_semantic_track_count == 0
    assert np.allclose(occupied_risks(second), 0.4)
    assert second.temporal_semantics.tracks[0].semantic_class == "chair"


def test_bridge_tracks_each_lidar_frame_exactly_once() -> None:
    bridge = make_bridge()

    first = bridge.initialize(make_observation(1, 1.0))
    second = bridge.update(make_observation(2, 1.1))

    first_track = first.field_status.snapshot.payload.tracking_result
    second_track = second.field_status.snapshot.payload.tracking_result
    assert first_track.tracked_obstacles[0].observation_count == 1
    assert second_track.tracked_obstacles[0].observation_count == 2
    assert first_track.tracked_obstacles[0].track_id == 0
    assert second_track.tracked_obstacles[0].track_id == 0


def test_short_camera_dropout_persists_label_on_same_geometry() -> None:
    bridge = make_bridge()
    bridge.initialize(make_observation(1, 1.0))
    bridge.update(make_observation(2, 1.1))

    dropout = bridge.update(
        make_observation(3, 1.2, include_chair=False)
    )

    assert dropout.known_semantic_track_count == 1
    assert dropout.temporal_semantics.tracks[0].status == "persisted"
    assert dropout.field_status.snapshot.payload.selected_track_ids == (0,)


def test_sensor_synchronization_failure_preserves_committed_state() -> None:
    bridge = make_bridge()
    accepted = bridge.initialize(make_observation(1, 1.0))

    with pytest.raises(RuntimeError, match="timestamp separation"):
        bridge.update(
            make_observation(2, 1.1, detection_timestamp=1.3)
        )

    assert bridge.last_result is accepted
    assert bridge.perception_runtime.field_buffer.generation == 1
    assert bridge.semantic_memory_track_count == 1


def test_observation_rejects_broken_point_alignment() -> None:
    observation = make_observation(1, 1.0)

    with pytest.raises(ValueError, match="one-to-one alignment"):
        replace(
            observation,
            obstacle_points_source_xyz=np.zeros((3, 3)),
        )
