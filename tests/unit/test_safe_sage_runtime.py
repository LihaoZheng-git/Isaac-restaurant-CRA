"""Unit tests for the reusable live Safe-SAGE runtime."""

from dataclasses import replace

import numpy as np
import pytest

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
from robot.safety.safe_sage_live_field import SafeSAGELiveFieldConfig
from robot.safety.safe_sage_runtime import (
    SafeSAGEPerceptionObservation,
    SafeSAGEPerceptionRuntime,
    robot_points_in_anchor_frame,
)
from robot.safety.safe_sage_temporal_derivative import (
    SafeSAGETemporalDerivativeConfig,
    SafeSAGETemporalDerivativeEstimator,
    SafeSAGETemporalDerivativeResult,
)
from robot.safety.semantic_label_provider import (
    NearestWorldPositionSemanticLabelProvider,
    SemanticClassLabel,
)
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
)


STATIC_OBSTACLE_POINTS = np.array(
    [
        [1.90, -0.10],
        [1.90, 0.10],
        [2.10, -0.10],
        [2.10, 0.10],
    ],
    dtype=np.float64,
)


def make_policies() -> tuple[ClassSafetyPolicy, SafeSAGEGuidancePolicy]:
    """Return matching lightweight semantic policies."""

    class_policy = ClassSafetyPolicy(
        {
            "unknown": ClassSafetyParameters(
                risk=1.0,
                boundary_flux_magnitude=4.0,
            ),
            "person": ClassSafetyParameters(
                risk=1.0,
                boundary_flux_magnitude=4.0,
            ),
        }
    )
    guidance_policy = SafeSAGEGuidancePolicy(
        {
            "unknown": SafeSAGEGuidanceParameters(
                safety_parameters=class_policy.parameters_for("unknown"),
                tangential_strength=0.0,
                passing_side=PassingSide.RIGHT,
            ),
            "person": SafeSAGEGuidanceParameters(
                safety_parameters=class_policy.parameters_for("person"),
                tangential_strength=0.0,
                passing_side=PassingSide.RIGHT,
            ),
        }
    )
    return class_policy, guidance_policy


def make_runtime(
    *,
    enable_temporal_updates: bool = True,
) -> SafeSAGEPerceptionRuntime:
    """Create a small deterministic runtime with real numerical solvers."""

    class_policy, guidance_policy = make_policies()
    return SafeSAGEPerceptionRuntime(
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
        temporal_estimator=SafeSAGETemporalDerivativeEstimator(
            SafeSAGETemporalDerivativeConfig(
                smoothing_factor=1.0,
                maximum_time_delta_seconds=0.5,
            )
        ),
        live_field_config=SafeSAGELiveFieldConfig(
            maximum_field_age_seconds=0.4,
            maximum_reused_control_cycles=1,
        ),
        enable_temporal_updates=enable_temporal_updates,
    )


def make_observation(
    frame_id: int,
    timestamp: float,
    *,
    robot_position_world_xy: tuple[float, float] = (0.0, 0.0),
    obstacle_world_offset_xy: tuple[float, float] = (0.0, 0.0),
) -> SafeSAGEPerceptionObservation:
    """Express a controlled world obstacle in the current robot frame."""

    robot_position = np.asarray(robot_position_world_xy, dtype=np.float64)
    world_points = STATIC_OBSTACLE_POINTS + np.asarray(
        obstacle_world_offset_xy,
        dtype=np.float64,
    )
    return SafeSAGEPerceptionObservation(
        sensor_frame_id=frame_id,
        sensor_timestamp_seconds=timestamp,
        simulation_time_seconds=timestamp,
        obstacle_points_robot_xy=world_points - robot_position,
        robot_position_world_xy=robot_position,
        robot_yaw=0.0,
    )


def make_provider(
    target_world_xy: tuple[float, float] = (2.0, 0.0),
):
    """Return the controlled manual person-label provider."""

    return NearestWorldPositionSemanticLabelProvider(
        target_world_xy,
        SemanticClassLabel("person"),
        maximum_association_distance=0.5,
    )


def test_runtime_initializes_complete_fixed_anchor_field() -> None:
    runtime = make_runtime()

    status = runtime.initialize(make_observation(1, 1.0), make_provider())

    assert status.generation == 1
    assert status.field_updated
    assert not status.fail_closed
    assert runtime.field.sensor_frame_id == 1
    assert runtime.field.all_obstacle_point_count == 4
    assert runtime.field.selected_obstacle_point_count == 4
    assert runtime.field.selected_track_ids == (0,)
    assert runtime.field.primary_association_distance == pytest.approx(0.0)
    assert runtime.field.temporal_safety_derivative == 0.0


def test_accepted_field_builds_live_geometry_motion_without_fall_substitution() -> None:
    from robot.perception.live_physical_evidence import (
        LivePhysicalEvidenceContext,
        build_live_physical_attributes,
        build_tracking_physical_attributes,
    )

    runtime = make_runtime()
    observation = make_observation(1, 1.0)
    runtime.initialize(observation, make_provider())
    points_xyz = np.column_stack(
        (observation.obstacle_points_robot_xy, np.full(4, 0.5))
    )
    records = build_live_physical_attributes(
        runtime.field,
        LivePhysicalEvidenceContext(
            obstacle_points_robot_xyz=points_xyz,
            floor_height_robot_m=0.0,
            robot_position_world_xy=(0.0, 0.0),
            waypoint_world_xy=(4.0, 0.0),
            measured_robot_velocity_world_xy=(0.1, 0.0),
        ),
    )
    assert len(records) == 1
    assert records[0].geometry.width_m.is_available
    assert not records[0].geometry.fall_height_m.is_available
    assert not records[0].mass.mass_kg.is_available
    assert not records[0].motion.relative_speed_mps.is_available  # new track

    direct_records = build_tracking_physical_attributes(
        tracking_result=runtime.field.tracking_result,
        label_result=runtime.field.label_result,
        sensor_timestamp_seconds=runtime.field.sensor_timestamp_seconds,
        context=LivePhysicalEvidenceContext(
            obstacle_points_robot_xyz=points_xyz,
            floor_height_robot_m=0.0,
            robot_position_world_xy=(0.0, 0.0),
            waypoint_world_xy=(4.0, 0.0),
            measured_robot_velocity_world_xy=(0.1, 0.0),
        ),
    )
    assert direct_records == records


def test_fixed_anchor_removes_robot_egomotion_from_static_field() -> None:
    runtime = make_runtime()
    runtime.initialize(make_observation(1, 1.0), make_provider())

    status = runtime.update(
        make_observation(
            2,
            1.2,
            robot_position_world_xy=(0.2, 0.0),
        ),
        make_provider(),
    )

    temporal = status.snapshot.payload.temporal_safety_derivative
    assert isinstance(temporal, SafeSAGETemporalDerivativeResult)
    np.testing.assert_allclose(temporal.values, 0.0, atol=1e-12)
    assert status.snapshot.payload.primary_track_id == 0
    assert status.snapshot.payload.maximum_selected_track_speed == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_moving_obstacle_produces_temporal_field_and_track_velocity() -> None:
    runtime = make_runtime()
    runtime.initialize(make_observation(1, 1.0), make_provider())

    status = runtime.update(
        make_observation(
            2,
            1.2,
            obstacle_world_offset_xy=(0.0, 0.2),
        ),
        make_provider((2.0, 0.2)),
    )

    field = status.snapshot.payload
    temporal = field.temporal_safety_derivative
    assert isinstance(temporal, SafeSAGETemporalDerivativeResult)
    assert temporal.maximum_absolute_derivative > 0.0
    assert field.maximum_selected_track_speed == pytest.approx(1.0)


def test_runtime_can_preserve_static_zero_temporal_baseline() -> None:
    runtime = make_runtime(enable_temporal_updates=False)
    runtime.initialize(make_observation(1, 1.0), make_provider())

    status = runtime.update(
        make_observation(2, 1.2),
        make_provider(),
    )

    assert status.snapshot.payload.temporal_safety_derivative == 0.0


def test_runtime_field_rejects_nonfinite_scalar_temporal_derivative() -> None:
    runtime = make_runtime()
    runtime.initialize(make_observation(1, 1.0), make_provider())

    with pytest.raises(ValueError, match="temporal_safety_derivative"):
        replace(runtime.field, temporal_safety_derivative=np.nan)


def test_failed_candidate_preserves_field_and_tracker_transaction() -> None:
    runtime = make_runtime()
    runtime.initialize(make_observation(1, 1.0), make_provider())
    original_field = runtime.field

    with pytest.raises(RuntimeError, match="No geometric track"):
        runtime.update(
            make_observation(2, 1.2),
            make_provider((7.0, 7.0)),
        )

    assert runtime.field is original_field
    assert runtime.field_buffer.generation == 1
    assert runtime.active_track_count == 1

    recovered = runtime.update(
        make_observation(2, 1.2),
        make_provider(),
    )
    assert recovered.generation == 2
    assert recovered.snapshot.payload.primary_track_id == 0


def test_runtime_delegates_reuse_and_inspect_freshness() -> None:
    runtime = make_runtime()
    runtime.initialize(make_observation(1, 1.0), make_provider())

    reused = runtime.reuse(current_simulation_time_seconds=1.2)
    inspected = runtime.inspect(current_simulation_time_seconds=1.3)

    assert reused.reused_control_cycles == 1
    assert not reused.fail_closed
    assert inspected.reused_control_cycles == 1
    assert not inspected.fail_closed


def test_execution_budget_shortens_reused_field_interval() -> None:
    runtime = make_runtime()
    runtime.initialize(make_observation(1, 1.0), make_provider())
    runtime.reuse(current_simulation_time_seconds=1.25)

    budget = runtime.fresh_execution_step_budget(
        current_simulation_time_seconds=1.25,
        physics_time_step_seconds=1.0 / 60.0,
        requested_steps=12,
    )

    assert budget == 9


def test_execution_budget_is_zero_for_stale_field() -> None:
    runtime = make_runtime()
    runtime.initialize(make_observation(1, 1.0), make_provider())

    budget = runtime.fresh_execution_step_budget(
        current_simulation_time_seconds=1.41,
        physics_time_step_seconds=1.0 / 60.0,
        requested_steps=12,
    )

    assert budget == 0


def test_robot_points_transform_through_world_into_anchor() -> None:
    transformed = robot_points_in_anchor_frame(
        np.array([[1.0, 0.0]]),
        robot_position_world_xy=np.array([2.0, 1.0]),
        robot_yaw=np.pi / 2.0,
        anchor_position_world_xy=np.array([1.0, 1.0]),
        anchor_yaw=np.pi / 2.0,
    )

    np.testing.assert_allclose(transformed, np.array([[1.0, -1.0]]))


def test_observation_owns_read_only_input_arrays() -> None:
    points = STATIC_OBSTACLE_POINTS.copy()
    robot_position = np.array([0.0, 0.0])
    observation = SafeSAGEPerceptionObservation(
        sensor_frame_id=1,
        sensor_timestamp_seconds=1.0,
        simulation_time_seconds=1.0,
        obstacle_points_robot_xy=points,
        robot_position_world_xy=robot_position,
        robot_yaw=0.0,
    )
    points[:] = 99.0
    robot_position[:] = 99.0

    np.testing.assert_allclose(
        observation.obstacle_points_robot_xy,
        STATIC_OBSTACLE_POINTS,
    )
    np.testing.assert_allclose(
        observation.robot_position_world_xy,
        np.zeros(2),
    )
    with pytest.raises(ValueError):
        observation.obstacle_points_robot_xy[0, 0] = 3.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"sensor_frame_id": -1},
        {"sensor_timestamp_seconds": -0.1},
        {"simulation_time_seconds": np.inf},
        {"robot_yaw": np.nan},
        {"obstacle_points_robot_xy": np.zeros((3, 3))},
    ],
)
def test_observation_rejects_invalid_inputs(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "sensor_frame_id": 1,
        "sensor_timestamp_seconds": 1.0,
        "simulation_time_seconds": 1.0,
        "obstacle_points_robot_xy": STATIC_OBSTACLE_POINTS,
        "robot_position_world_xy": np.zeros(2),
        "robot_yaw": 0.0,
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        SafeSAGEPerceptionObservation(**values)  # type: ignore[arg-type]
