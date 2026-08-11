"""Run closed-loop Safe-SAGE ``h_full`` navigation around a static proxy.

This demo begins from the stationary scenario validated by
``safe_sage_h_full_cbf_demo.py`` and adds wheel motion. The static blue cube
is observed through the real PandarXT-32 perception path and manually labeled
``person``. Its known world pose is used only for controlled track selection,
an independent policy-footprint clearance monitor, and result reporting.

At every new LiDAR frame the demo:

1. Transforms and filters the point cloud in the chassis frame.
2. Associates the controlled proxy with one tracked cluster.
3. Transforms that track into the fixed frame established by the first scan.
4. Builds an uninflated semantic occupancy grid from that track only.
5. Constructs conservative yaw-dependent configuration spaces using the
   measured Nova Carter footprint.
6. Solves semantic guidance and ``h_full(x, y, psi)``.

The field frame remains fixed at the robot pose associated with the first
scan. Every later observation is transformed into that same frame, and the
current look-ahead control point advances through it. The physical footprint
is expressed about that same point. This preserves the spatial approach seen
by the CBF and gives Nova Carter a realizable interface through which the
paper's tangential planar guidance can influence angular velocity.

The motion loop fails closed when LiDAR data becomes stale, perception or a
PDE solve fails, the query leaves the field, the CBF becomes infeasible, an
unsafe configuration is reached, or the independent ground-truth clearance
guard is crossed. The static-obstacle temporal derivative is explicitly zero.

Run from the project root:

    python -u scripts/demos/safe_sage_h_full_motion_demo.py

Press Play in Isaac Sim when requested.
"""

import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Importing the stationary demo initializes the one required SimulationApp.
# Its main() function is guarded and therefore is not executed here. Reusing
# this module keeps the already validated scene and sensor setup identical.
from scripts.demos import safe_sage_h_full_cbf_demo as stationary


simulation_app = stationary.simulation_app


import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd

from robot.control.differential_drive import NovaCarterController
from robot.control.unicycle_model import lookahead_state
from robot.navigation.waypoint_controller import (
    RobotPose2D,
    WaypointCommand,
    wrap_to_pi,
)
from robot.runtime.simulation_loop import (
    advance_playing_steps,
    wait_until_playing,
)
from robot.safety.class_safety_policy import ClassSafetyPolicy
from robot.safety.local_occupancy_grid import (
    LocalOccupancyGridConfig,
    build_local_occupancy_grid,
)
from robot.safety.obstacle_detection import ObstacleDetectionConfig
from robot.safety.obstacle_tracking import (
    ObstacleTracker,
    ObstacleTrackerConfig,
    TrackedObstacle,
)
from robot.safety.safe_sage_cbf_filter import (
    SafeSAGECBFConfig,
    SafeSAGECBFSafetyFilter,
)
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
)
from robot.safety.safe_sage_unicycle_cbf_filter import (
    SafeSAGEUnicycleCBFConfig,
    SafeSAGEUnicycleCBFFilter,
    SafeSAGEUnicycleCBFResult,
    shift_rectangular_footprint_to_lookahead,
)
from robot.safety.safe_sage_field_pipeline import (
    SafeSAGEFieldPipelineConfig,
    SafeSAGEFieldPipelineResult,
    build_safe_sage_field_pipeline,
)
from robot.safety.safe_sage_guidance_policy import (
    SafeSAGEGuidancePolicy,
)
from robot.safety.semantic_obstacle import SemanticObstacle
from robot.safety.safe_sage_temporal_derivative import (
    TemporalSafetyDerivativeInput,
)
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import compute_relative_prim_transform
from robot.sensor.lidar import LidarFrameInfo, NovaCarterLidar


# ---------------------------------------------------------------------------
# Closed-loop experiment configuration
# ---------------------------------------------------------------------------


MOTION_GRID_MINIMUM_X = -2.0
MOTION_GRID_MAXIMUM_X = 5.0

MOTION_SETTLE_STEPS = 60
MAXIMUM_MOTION_SECONDS = 20.0
MAXIMUM_MOTION_STEPS = math.ceil(
    MAXIMUM_MOTION_SECONDS / stationary.PHYSICS_DT
)
MAXIMUM_STALE_SCAN_SECONDS = 0.30
MAXIMUM_STALE_SCAN_STEPS = math.ceil(
    MAXIMUM_STALE_SCAN_SECONDS / stationary.PHYSICS_DT
)
STATUS_REPORT_INTERVAL_STEPS = 30

# Clearance is measured between the ground-truth proxy rectangle and the
# measured robot rectangle including the 0.05 m policy margin. This guard is
# independent of LiDAR and is never passed into the safety pipeline.
GROUND_TRUTH_CLEARANCE_ABORT = 0.02

PROXY_CLEARANCE_FORWARD_MARGIN = 0.30
MINIMUM_EXPECTED_RIGHTWARD_DISPLACEMENT = 0.05
SETTLED_LINEAR_SPEED = 0.02
SETTLED_ANGULAR_SPEED = 0.03
SETTLED_CONFIRMATION_STEPS = 60

TRANSIENT_LIDAR_ERRORS = stationary.TRANSIENT_LIDAR_ERRORS


@dataclass(frozen=True)
class MotionPipelineFrame:
    """One trial-anchored Safe-SAGE field and perception diagnostics."""

    sensor_frame_id: int
    sensor_timestamp_seconds: float
    anchor_pose_world: RobotPose2D
    all_obstacle_point_count: int
    proxy_point_count: int
    proxy_track_id: int
    proxy_association_error: float
    pipeline: SafeSAGEFieldPipelineResult
    build_wall_seconds: float
    proxy_velocity_world_xy: tuple[float, float] = (0.0, 0.0)
    temporal_safety_derivative: TemporalSafetyDerivativeInput = 0.0


@dataclass(frozen=True)
class MotionResult:
    """Closed-loop outcome and independent safety diagnostics."""

    termination_reason: str
    completed_steps: int
    received_scan_count: int
    initial_pose: RobotPose2D
    final_pose: RobotPose2D
    final_filter_result: SafeSAGEUnicycleCBFResult
    intervention_seen: bool
    first_intervention_step: int | None
    first_intervention_clearance: float
    minimum_policy_footprint_clearance: float
    minimum_sampled_safety: float
    minimum_cbf_margin: float
    maximum_rightward_displacement: float
    maximum_absolute_angular_velocity: float
    all_constraints_valid: bool
    cleared_proxy: bool
    route_complete: bool
    emergency_stop_seen: bool
    stale_scan_stop_seen: bool
    pipeline_failure_seen: bool
    ground_truth_abort_seen: bool
    application_stopped: bool
    mean_pipeline_build_wall_seconds: float
    maximum_pipeline_build_wall_seconds: float


def build_motion_pipeline_frame(
    *,
    points_sensor_frame: np.ndarray,
    frame_info: LidarFrameInfo,
    robot_pose: RobotPose2D,
    field_anchor_pose: RobotPose2D,
    obstacle_tracker: ObstacleTracker,
    robot_from_lidar: np.ndarray,
    detection_config: ObstacleDetectionConfig,
    class_policy: ClassSafetyPolicy,
    guidance_policy: SafeSAGEGuidancePolicy,
    grid_config: LocalOccupancyGridConfig,
    field_config: SafeSAGEFieldPipelineConfig,
    semantic_class: str = stationary.SEMANTIC_CLASS,
    controlled_proxy_position_world_xy: np.ndarray | None = None,
) -> MotionPipelineFrame:
    """Build one full Safe-SAGE field in the fixed trial-anchor frame.

    ``semantic_class`` defaults to the original controlled ``person`` label
    so existing motion demos retain their behavior. Diagnostic callers may
    override the label when reproducing a different paper baseline.
    """

    start_wall_time = time.perf_counter()
    obstacle_points = stationary.extract_obstacle_points(
        points_sensor_frame,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
    )
    if obstacle_points.shape[0] == 0:
        raise RuntimeError(
            "The current LiDAR scan produced no obstacle points."
        )

    tracking_result = obstacle_tracker.update(
        obstacle_points,
        robot_position_world_xy=np.array(
            [robot_pose.x, robot_pose.y],
            dtype=np.float64,
        ),
        robot_yaw=robot_pose.yaw,
        timestamp_seconds=(
            frame_info.sensor_timestamp_nanoseconds * 1e-9
        ),
    )
    if controlled_proxy_position_world_xy is None:
        proxy_track, association_error = stationary.find_proxy_track(
            tracking_result,
            robot_pose=robot_pose,
        )
    else:
        proxy_track, association_error = find_controlled_proxy_track_world(
            tracking_result.tracked_obstacles,
            controlled_proxy_position_world_xy,
        )
    if proxy_track is None:
        raise RuntimeError(
            "No LiDAR track matched the controlled proxy; nearest "
            f"association error={association_error:.3f} m."
        )

    proxy_mask = (
        tracking_result.point_track_ids == proxy_track.track_id
    )
    proxy_points = np.ascontiguousarray(
        obstacle_points[proxy_mask],
        dtype=np.float64,
    )
    proxy_track_ids = np.ascontiguousarray(
        tracking_result.point_track_ids[proxy_mask],
        dtype=np.int64,
    )
    if proxy_points.shape[0] == 0:
        raise RuntimeError(
            "The associated proxy track contained no aligned points."
        )

    proxy_points_anchor = robot_points_in_anchor_frame(
        proxy_points,
        robot_pose=robot_pose,
        anchor_pose=field_anchor_pose,
    )

    semantic_obstacle = SemanticObstacle(
        tracked_obstacle=proxy_track,
        semantic_class=semantic_class,
        semantic_confidence=1.0,
        label_source="manual",
    )
    source_grid = build_local_occupancy_grid(
        proxy_points_anchor,
        point_track_ids=proxy_track_ids,
        semantic_obstacles=(semantic_obstacle,),
        class_safety_policy=class_policy,
        config=grid_config,
    )
    pipeline = build_safe_sage_field_pipeline(
        source_grid,
        guidance_policy=guidance_policy,
        config=field_config,
        semantic_obstacles=(semantic_obstacle,),
    )
    return MotionPipelineFrame(
        sensor_frame_id=frame_info.sensor_frame_id,
        sensor_timestamp_seconds=(
            frame_info.sensor_timestamp_nanoseconds * 1e-9
        ),
        anchor_pose_world=field_anchor_pose,
        all_obstacle_point_count=int(obstacle_points.shape[0]),
        proxy_point_count=int(proxy_points.shape[0]),
        proxy_track_id=proxy_track.track_id,
        proxy_association_error=association_error,
        pipeline=pipeline,
        build_wall_seconds=time.perf_counter() - start_wall_time,
        proxy_velocity_world_xy=proxy_track.velocity_world_xy,
    )


def find_controlled_proxy_track_world(
    tracked_obstacles: tuple[TrackedObstacle, ...],
    controlled_proxy_position_world_xy: np.ndarray,
) -> tuple[TrackedObstacle | None, float]:
    """Associate one controlled semantic label by known proxy world pose.

    This helper is experimental infrastructure.  The known pose selects which
    real LiDAR track receives the manual label; it is not included in the
    occupancy geometry or supplied to either safety filter.
    """

    expected = np.asarray(
        controlled_proxy_position_world_xy,
        dtype=np.float64,
    )
    if expected.shape != (2,):
        raise ValueError(
            "controlled_proxy_position_world_xy must have shape (2,)."
        )
    if not np.all(np.isfinite(expected)):
        raise ValueError(
            "controlled_proxy_position_world_xy must contain finite values."
        )
    best_track: TrackedObstacle | None = None
    best_error = math.inf
    for tracked_obstacle in tracked_obstacles:
        centroid_world = np.asarray(
            tracked_obstacle.centroid_world_xy,
            dtype=np.float64,
        )
        error = float(np.linalg.norm(centroid_world - expected))
        if error < best_error:
            best_track = tracked_obstacle
            best_error = error
    if best_error > stationary.PROXY_ASSOCIATION_RADIUS:
        return None, best_error
    return best_track, best_error


def robot_points_in_anchor_frame(
    points_robot_xy: np.ndarray,
    *,
    robot_pose: RobotPose2D,
    anchor_pose: RobotPose2D,
) -> np.ndarray:
    """Transform planar points from the current robot to a fixed frame."""

    points = np.asarray(points_robot_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_robot_xy must have shape (N, 2).")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_robot_xy must contain finite values.")

    robot_cosine = math.cos(robot_pose.yaw)
    robot_sine = math.sin(robot_pose.yaw)
    world_x = (
        robot_pose.x
        + robot_cosine * points[:, 0]
        - robot_sine * points[:, 1]
    )
    world_y = (
        robot_pose.y
        + robot_sine * points[:, 0]
        + robot_cosine * points[:, 1]
    )

    delta_world_x = world_x - anchor_pose.x
    delta_world_y = world_y - anchor_pose.y
    anchor_cosine = math.cos(anchor_pose.yaw)
    anchor_sine = math.sin(anchor_pose.yaw)
    points_anchor = np.column_stack(
        (
            anchor_cosine * delta_world_x
            + anchor_sine * delta_world_y,
            -anchor_sine * delta_world_x
            + anchor_cosine * delta_world_y,
        )
    )
    return np.ascontiguousarray(points_anchor, dtype=np.float64)


def current_pose_in_field_frame(
    current_pose: RobotPose2D,
    *,
    anchor_pose: RobotPose2D,
) -> tuple[np.ndarray, float]:
    """Express a world pose in the fixed trial-anchor frame."""

    delta_world_x = current_pose.x - anchor_pose.x
    delta_world_y = current_pose.y - anchor_pose.y
    cosine = math.cos(anchor_pose.yaw)
    sine = math.sin(anchor_pose.yaw)
    position_grid_xy = np.array(
        [
            cosine * delta_world_x + sine * delta_world_y,
            -sine * delta_world_x + cosine * delta_world_y,
        ],
        dtype=np.float64,
    )
    relative_yaw = wrap_to_pi(current_pose.yaw - anchor_pose.yaw)
    return position_grid_xy, relative_yaw


def robot_policy_footprint_polygon(pose: RobotPose2D) -> np.ndarray:
    """Return world vertices for the measured footprint plus policy margin."""

    minimum_x = -stationary.ROBOT_REAR_EXTENT - (
        stationary.ROBOT_SAFETY_MARGIN
    )
    maximum_x = stationary.ROBOT_FRONT_EXTENT + (
        stationary.ROBOT_SAFETY_MARGIN
    )
    minimum_y = -stationary.ROBOT_RIGHT_EXTENT - (
        stationary.ROBOT_SAFETY_MARGIN
    )
    maximum_y = stationary.ROBOT_LEFT_EXTENT + (
        stationary.ROBOT_SAFETY_MARGIN
    )
    local_vertices = np.array(
        [
            [minimum_x, minimum_y],
            [maximum_x, minimum_y],
            [maximum_x, maximum_y],
            [minimum_x, maximum_y],
        ],
        dtype=np.float64,
    )
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )
    return local_vertices @ rotation.T + np.array(
        [pose.x, pose.y],
        dtype=np.float64,
    )


def proxy_footprint_polygon(
    proxy_position_world_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Return the controlled proxy's ground-truth world rectangle."""

    center = (
        stationary.STATIC_PROXY_POSITION_WORLD[:2]
        if proxy_position_world_xy is None
        else np.asarray(proxy_position_world_xy, dtype=np.float64)
    )
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("proxy_position_world_xy must be one finite XY point.")
    half_x = 0.5 * stationary.PROXY_DIMENSIONS[0]
    half_y = 0.5 * stationary.PROXY_DIMENSIONS[1]
    return np.array(
        [
            [center[0] - half_x, center[1] - half_y],
            [center[0] + half_x, center[1] - half_y],
            [center[0] + half_x, center[1] + half_y],
            [center[0] - half_x, center[1] + half_y],
        ],
        dtype=np.float64,
    )


def policy_footprint_clearance(
    pose: RobotPose2D,
    proxy_position_world_xy: np.ndarray | None = None,
) -> float:
    """Return exact polygon clearance for the controlled validation pair."""

    robot_polygon = robot_policy_footprint_polygon(pose)
    proxy_polygon = proxy_footprint_polygon(proxy_position_world_xy)
    if convex_polygons_overlap(robot_polygon, proxy_polygon):
        return 0.0

    distances: list[float] = []
    for point in robot_polygon:
        distances.extend(
            point_to_polygon_edge_distances(point, proxy_polygon)
        )
    for point in proxy_polygon:
        distances.extend(
            point_to_polygon_edge_distances(point, robot_polygon)
        )
    return min(distances)


def convex_polygons_overlap(
    first_polygon: np.ndarray,
    second_polygon: np.ndarray,
) -> bool:
    """Test two convex polygons using all edge-normal separating axes."""

    for polygon in (first_polygon, second_polygon):
        for vertex_index in range(polygon.shape[0]):
            edge = (
                polygon[(vertex_index + 1) % polygon.shape[0]]
                - polygon[vertex_index]
            )
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-12:
                continue
            axis /= axis_norm
            first_projection = first_polygon @ axis
            second_projection = second_polygon @ axis
            if (
                float(np.max(first_projection))
                < float(np.min(second_projection)) - 1e-12
                or float(np.max(second_projection))
                < float(np.min(first_projection)) - 1e-12
            ):
                return False
    return True


def point_to_polygon_edge_distances(
    point: np.ndarray,
    polygon: np.ndarray,
) -> list[float]:
    """Return distances from one point to every polygon edge segment."""

    distances: list[float] = []
    for vertex_index in range(polygon.shape[0]):
        segment_start = polygon[vertex_index]
        segment_end = polygon[(vertex_index + 1) % polygon.shape[0]]
        segment = segment_end - segment_start
        segment_squared = float(np.dot(segment, segment))
        if segment_squared <= 1e-24:
            distances.append(float(np.linalg.norm(point - segment_start)))
            continue
        interpolation = float(
            np.clip(
                np.dot(point - segment_start, segment) / segment_squared,
                0.0,
                1.0,
            )
        )
        nearest = segment_start + interpolation * segment
        distances.append(float(np.linalg.norm(point - nearest)))
    return distances


def forward_and_rightward_displacement(
    pose: RobotPose2D,
    *,
    initial_pose: RobotPose2D,
) -> tuple[float, float]:
    """Return displacement along the initial forward and right axes."""

    delta_x = pose.x - initial_pose.x
    delta_y = pose.y - initial_pose.y
    cosine = math.cos(initial_pose.yaw)
    sine = math.sin(initial_pose.yaw)
    forward = cosine * delta_x + sine * delta_y
    rightward = sine * delta_x - cosine * delta_y
    return forward, rightward


def proxy_clearance_forward_threshold(
    initial_pose: RobotPose2D,
    proxy_position_world_xy: np.ndarray | None = None,
) -> float:
    """Return forward progress required to pass the proxy longitudinally."""

    center = (
        stationary.STATIC_PROXY_POSITION_WORLD[:2]
        if proxy_position_world_xy is None
        else np.asarray(proxy_position_world_xy, dtype=np.float64)
    )
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("proxy_position_world_xy must be one finite XY point.")
    delta_x = center[0] - initial_pose.x
    delta_y = center[1] - initial_pose.y
    proxy_center_forward = (
        math.cos(initial_pose.yaw) * delta_x
        + math.sin(initial_pose.yaw) * delta_y
    )
    return (
        proxy_center_forward
        + 0.5 * stationary.PROXY_DIMENSIONS[0]
        + PROXY_CLEARANCE_FORWARD_MARGIN
    )


def evaluate_filter(
    *,
    safety_filter: SafeSAGEUnicycleCBFFilter,
    nominal_command: WaypointCommand,
    robot_pose: RobotPose2D,
    pipeline_frame: MotionPipelineFrame,
) -> tuple[SafeSAGEUnicycleCBFResult, np.ndarray, float]:
    """Evaluate the fixed field at the current look-ahead configuration."""

    chassis_position, query_yaw = current_pose_in_field_frame(
        robot_pose,
        anchor_pose=pipeline_frame.anchor_pose_world,
    )
    query_state = lookahead_state(
        np.array(
            [chassis_position[0], chassis_position[1], query_yaw],
            dtype=np.float64,
        ),
        lookahead_distance=(
            stationary.UNICYCLE_LOOKAHEAD_DISTANCE
        ),
    )
    query_position = query_state[:2]
    result = safety_filter.filter_command(
        nominal_command.linear_velocity,
        nominal_command.angular_velocity,
        query_position_grid_xy=query_position,
        query_yaw_radians=query_yaw,
        configuration_space=pipeline_frame.pipeline.configuration_space,
        h_full=pipeline_frame.pipeline.h_full,
        guidance_fields=pipeline_frame.pipeline.guidance_fields,
        temporal_safety_derivative=(
            stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE
        ),
    )
    return result, query_position, query_yaw


def print_live_status(
    *,
    completed_steps: int,
    pipeline_frame: MotionPipelineFrame,
    robot_pose: RobotPose2D,
    query_position: np.ndarray,
    query_yaw: float,
    clearance: float,
    rightward_displacement: float,
    nominal_command: WaypointCommand,
    filter_result: SafeSAGEUnicycleCBFResult,
) -> None:
    """Print one compact closed-loop status line."""

    filter_status = filter_result.emergency_reason or "ok"
    print(
        f"  step={completed_steps:<4} "
        f"scan={pipeline_frame.sensor_frame_id:<4} "
        f"pose=({robot_pose.x:.3f},{robot_pose.y:.3f},"
        f"{math.degrees(robot_pose.yaw):.1f}deg) "
        f"q=({query_position[0]:.3f},{query_position[1]:.3f},"
        f"{math.degrees(query_yaw):.1f}deg) "
        f"clear={clearance:.3f} "
        f"right={rightward_displacement:.3f} "
        f"h={filter_result.sampled_safety_value:.3f} "
        f"g=({filter_result.sampled_guidance_grid_xy[0]:.2f},"
        f"{filter_result.sampled_guidance_grid_xy[1]:.2f}) "
        f"v={nominal_command.linear_velocity:.3f}->"
        f"{filter_result.safe_linear_velocity:.3f} "
        f"w={nominal_command.angular_velocity:.3f}->"
        f"{filter_result.safe_angular_velocity:.3f} "
        f"margin={filter_result.cbf_margin:.2e} "
        f"intervention={filter_result.intervention_active} "
        f"status={filter_status}"
    )


def run_motion_trial(
    *,
    stage: Usd.Stage,
    lidar: NovaCarterLidar,
    robot_controller: NovaCarterController,
    robot_from_lidar: np.ndarray,
    detection_config: ObstacleDetectionConfig,
    class_policy: ClassSafetyPolicy,
    guidance_policy: SafeSAGEGuidancePolicy,
    grid_config: LocalOccupancyGridConfig,
    field_config: SafeSAGEFieldPipelineConfig,
    safety_filter: SafeSAGEUnicycleCBFFilter,
) -> MotionResult:
    """Run one fail-closed full-state Safe-SAGE motion trial."""

    stationary.position_robot_for_experiment(robot_controller)
    initial_pose = stationary.verify_robot_start_pose(stage)
    if not advance_playing_steps(
        simulation_app,
        physics_steps=MOTION_SETTLE_STEPS,
        on_step=robot_controller.stop,
    ):
        raise RuntimeError(
            "Isaac Sim stopped while preparing the motion trial."
        )

    waypoint_controller = stationary.create_waypoint_controller()
    obstacle_tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=stationary.TRACKER_CLUSTER_RADIUS,
            minimum_cluster_points=(
                stationary.TRACKER_MINIMUM_CLUSTER_POINTS
            ),
            maximum_association_distance=(
                stationary.TRACKER_MAXIMUM_ASSOCIATION_DISTANCE
            ),
        )
    )

    points_sensor_frame, frame_info, _ = stationary.wait_for_first_scan(
        lidar,
        robot_controller,
    )
    robot_pose = stationary.read_robot_pose(stage)
    field_anchor_pose = robot_pose
    pipeline_frame = build_motion_pipeline_frame(
        points_sensor_frame=points_sensor_frame,
        frame_info=frame_info,
        robot_pose=robot_pose,
        field_anchor_pose=field_anchor_pose,
        obstacle_tracker=obstacle_tracker,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=grid_config,
        field_config=field_config,
    )
    nominal_command = waypoint_controller.compute_command(robot_pose)
    filter_result, query_position, query_yaw = evaluate_filter(
        safety_filter=safety_filter,
        nominal_command=nominal_command,
        robot_pose=robot_pose,
        pipeline_frame=pipeline_frame,
    )

    completed_steps = 0
    received_scan_count = 1
    steps_since_new_scan = 0
    intervention_seen = filter_result.intervention_active
    first_intervention_step = 0 if intervention_seen else None
    current_clearance = policy_footprint_clearance(robot_pose)
    first_intervention_clearance = (
        current_clearance if intervention_seen else math.nan
    )
    minimum_clearance = current_clearance
    minimum_sampled_safety = filter_result.sampled_safety_value
    minimum_cbf_margin = filter_result.cbf_margin
    maximum_rightward_displacement = 0.0
    maximum_absolute_angular_velocity = abs(
        filter_result.safe_angular_velocity
    )
    all_constraints_valid = (
        filter_result.constraint_feasible
        and filter_result.constraint_satisfied
        and not filter_result.emergency_stop_active
    )
    cleared_proxy = False
    route_complete = False
    emergency_stop_seen = False
    stale_scan_stop_seen = False
    pipeline_failure_seen = False
    ground_truth_abort_seen = False
    application_stopped = False
    settled_confirmation_count = 0
    previous_intervention = filter_result.intervention_active
    pipeline_build_times = [pipeline_frame.build_wall_seconds]
    clearance_threshold = proxy_clearance_forward_threshold(initial_pose)
    termination_reason = "Maximum motion duration reached."

    print("\n" + "=" * 70)
    print("SAFE-SAGE H_FULL STAGE D - CLOSED-LOOP MOTION")
    print("=" * 70)
    print("\nControlled motion settings:")
    print("  Chassis state:                 (x, y, psi)")
    print("  Executable input:              (v, omega)")
    print(
        "  Look-ahead distance:           "
        f"{stationary.UNICYCLE_LOOKAHEAD_DISTANCE:.3f} m"
    )
    print(
        "  Maximum duration:              "
        f"{MAXIMUM_MOTION_SECONDS:.1f} s"
    )
    print(
        "  Stale-scan timeout:            "
        f"{MAXIMUM_STALE_SCAN_SECONDS:.2f} s"
    )
    print(
        "  Ground-truth clearance abort: "
        f"{GROUND_TRUTH_CLEARANCE_ABORT:.3f} m"
    )
    print(
        "  Passing side:                  "
        f"{guidance_policy.parameters_for(stationary.SEMANTIC_CLASS).passing_side.value}"
    )
    print(
        "  Forward clearance threshold:  "
        f"{clearance_threshold:.3f} m"
    )
    print(
        "  Field frame:                  fixed at first LiDAR scan"
    )
    print(
        "  Field anchor pose:            "
        f"({field_anchor_pose.x:.4f}, {field_anchor_pose.y:.4f}, "
        f"{math.degrees(field_anchor_pose.yaw):.2f} deg)"
    )
    print("\nLive motion status:")
    print_live_status(
        completed_steps=0,
        pipeline_frame=pipeline_frame,
        robot_pose=robot_pose,
        query_position=query_position,
        query_yaw=query_yaw,
        clearance=current_clearance,
        rightward_displacement=0.0,
        nominal_command=nominal_command,
        filter_result=filter_result,
    )

    while completed_steps < MAXIMUM_MOTION_STEPS:
        if not simulation_app.is_running():
            application_stopped = True
            termination_reason = "Isaac Sim stopped during the motion trial."
            break
        if not app_utils.is_playing():
            robot_controller.stop()
            simulation_app.update()
            continue

        robot_pose = stationary.read_robot_pose(stage)
        nominal_command = waypoint_controller.compute_command(robot_pose)
        if nominal_command.route_complete:
            route_complete = True
            termination_reason = "The nominal waypoint route completed."
            break

        steps_since_new_scan += 1
        new_scan_received = False
        try:
            current_points_sensor = lidar.get_point_cloud()
            current_frame_info = lidar.get_frame_info()
        except RuntimeError as error:
            if not any(
                message in str(error)
                for message in TRANSIENT_LIDAR_ERRORS
            ):
                raise
        else:
            if (
                current_frame_info.sensor_frame_id
                != pipeline_frame.sensor_frame_id
            ):
                try:
                    pipeline_frame = build_motion_pipeline_frame(
                        points_sensor_frame=current_points_sensor,
                        frame_info=current_frame_info,
                        robot_pose=robot_pose,
                        field_anchor_pose=field_anchor_pose,
                        obstacle_tracker=obstacle_tracker,
                        robot_from_lidar=robot_from_lidar,
                        detection_config=detection_config,
                        class_policy=class_policy,
                        guidance_policy=guidance_policy,
                        grid_config=grid_config,
                        field_config=field_config,
                    )
                except (RuntimeError, ValueError) as error:
                    pipeline_failure_seen = True
                    termination_reason = (
                        "Fail-closed perception/PDE stop: "
                        f"{error}"
                    )
                    break
                received_scan_count += 1
                steps_since_new_scan = 0
                new_scan_received = True
                pipeline_build_times.append(
                    pipeline_frame.build_wall_seconds
                )

        if steps_since_new_scan >= MAXIMUM_STALE_SCAN_STEPS:
            stale_scan_stop_seen = True
            termination_reason = (
                "LiDAR data became stale; fail-closed stop applied."
            )
            break

        try:
            filter_result, query_position, query_yaw = evaluate_filter(
                safety_filter=safety_filter,
                nominal_command=nominal_command,
                robot_pose=robot_pose,
                pipeline_frame=pipeline_frame,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            pipeline_failure_seen = True
            termination_reason = (
                "Fail-closed field-query stop: "
                f"{error}"
            )
            break

        current_clearance = policy_footprint_clearance(robot_pose)
        forward_displacement, rightward_displacement = (
            forward_and_rightward_displacement(
                robot_pose,
                initial_pose=initial_pose,
            )
        )
        minimum_clearance = min(minimum_clearance, current_clearance)
        minimum_sampled_safety = min(
            minimum_sampled_safety,
            filter_result.sampled_safety_value,
        )
        if not filter_result.emergency_stop_active:
            minimum_cbf_margin = min(
                minimum_cbf_margin,
                filter_result.cbf_margin,
            )
        maximum_rightward_displacement = max(
            maximum_rightward_displacement,
            rightward_displacement,
        )
        maximum_absolute_angular_velocity = max(
            maximum_absolute_angular_velocity,
            abs(filter_result.safe_angular_velocity),
        )
        current_constraint_valid = (
            filter_result.constraint_feasible
            and filter_result.constraint_satisfied
            and not filter_result.emergency_stop_active
        )
        all_constraints_valid = (
            all_constraints_valid and current_constraint_valid
        )

        if filter_result.intervention_active and not intervention_seen:
            intervention_seen = True
            first_intervention_step = completed_steps
            first_intervention_clearance = current_clearance

        intervention_changed = (
            filter_result.intervention_active != previous_intervention
        )
        periodic_report = (
            completed_steps > 0
            and completed_steps % STATUS_REPORT_INTERVAL_STEPS == 0
        )
        if new_scan_received or intervention_changed or periodic_report:
            print_live_status(
                completed_steps=completed_steps,
                pipeline_frame=pipeline_frame,
                robot_pose=robot_pose,
                query_position=query_position,
                query_yaw=query_yaw,
                clearance=current_clearance,
                rightward_displacement=rightward_displacement,
                nominal_command=nominal_command,
                filter_result=filter_result,
            )

        if current_clearance <= GROUND_TRUTH_CLEARANCE_ABORT:
            ground_truth_abort_seen = True
            termination_reason = (
                "Independent ground-truth clearance guard triggered."
            )
            break
        if not current_constraint_valid:
            emergency_stop_seen = True
            termination_reason = (
                "The full-state CBF requested a fail-closed stop."
            )
            break

        cleared_proxy = forward_displacement >= clearance_threshold
        if cleared_proxy:
            termination_reason = (
                "The robot cleared the proxy longitudinally."
            )
            break

        settled_command = (
            filter_result.intervention_active
            and filter_result.safe_linear_velocity
            <= SETTLED_LINEAR_SPEED
            and abs(filter_result.safe_angular_velocity)
            <= SETTLED_ANGULAR_SPEED
        )
        if settled_command:
            settled_confirmation_count += 1
        else:
            settled_confirmation_count = 0
        if settled_confirmation_count >= SETTLED_CONFIRMATION_STEPS:
            termination_reason = (
                "The safe command settled before proxy clearance."
            )
            break

        robot_controller.apply_command(
            filter_result.safe_linear_velocity,
            filter_result.safe_angular_velocity,
        )
        previous_intervention = filter_result.intervention_active
        simulation_app.update()
        completed_steps += 1

    if simulation_app.is_running():
        robot_controller.stop()
        app_utils.update_app(steps=5)
        final_pose = stationary.read_robot_pose(stage)
    else:
        final_pose = robot_pose

    final_clearance = policy_footprint_clearance(final_pose)
    minimum_clearance = min(minimum_clearance, final_clearance)
    _, final_rightward_displacement = forward_and_rightward_displacement(
        final_pose,
        initial_pose=initial_pose,
    )
    maximum_rightward_displacement = max(
        maximum_rightward_displacement,
        final_rightward_displacement,
    )
    mean_build_time = float(np.mean(pipeline_build_times))
    maximum_build_time = float(np.max(pipeline_build_times))

    return MotionResult(
        termination_reason=termination_reason,
        completed_steps=completed_steps,
        received_scan_count=received_scan_count,
        initial_pose=initial_pose,
        final_pose=final_pose,
        final_filter_result=filter_result,
        intervention_seen=intervention_seen,
        first_intervention_step=first_intervention_step,
        first_intervention_clearance=first_intervention_clearance,
        minimum_policy_footprint_clearance=minimum_clearance,
        minimum_sampled_safety=minimum_sampled_safety,
        minimum_cbf_margin=minimum_cbf_margin,
        maximum_rightward_displacement=(
            maximum_rightward_displacement
        ),
        maximum_absolute_angular_velocity=(
            maximum_absolute_angular_velocity
        ),
        all_constraints_valid=all_constraints_valid,
        cleared_proxy=cleared_proxy,
        route_complete=route_complete,
        emergency_stop_seen=emergency_stop_seen,
        stale_scan_stop_seen=stale_scan_stop_seen,
        pipeline_failure_seen=pipeline_failure_seen,
        ground_truth_abort_seen=ground_truth_abort_seen,
        application_stopped=application_stopped,
        mean_pipeline_build_wall_seconds=mean_build_time,
        maximum_pipeline_build_wall_seconds=maximum_build_time,
    )


def print_motion_result(result: MotionResult) -> None:
    """Print acceptance checks for one closed-loop motion trial."""

    final_forward, final_rightward = forward_and_rightward_displacement(
        result.final_pose,
        initial_pose=result.initial_pose,
    )
    rightward_response_seen = (
        result.maximum_rightward_displacement
        >= MINIMUM_EXPECTED_RIGHTWARD_DISPLACEMENT
    )
    accepted = (
        result.intervention_seen
        and result.all_constraints_valid
        and result.cleared_proxy
        and rightward_response_seen
        and result.minimum_policy_footprint_clearance
        > GROUND_TRUTH_CLEARANCE_ABORT
        and not result.emergency_stop_seen
        and not result.stale_scan_stop_seen
        and not result.pipeline_failure_seen
        and not result.ground_truth_abort_seen
        and not result.application_stopped
    )

    print("\n" + "=" * 70)
    print("SAFE-SAGE H_FULL STAGE D - MOTION RESULT")
    print("=" * 70)
    print(f"\nTermination: {result.termination_reason}")

    print("\nExecution:")
    print(f"  Completed physics steps:       {result.completed_steps}")
    print(f"  Received LiDAR scans:          {result.received_scan_count}")
    print(
        "  Mean pipeline wall time:      "
        f"{result.mean_pipeline_build_wall_seconds:.4f} s"
    )
    print(
        "  Maximum pipeline wall time:   "
        f"{result.maximum_pipeline_build_wall_seconds:.4f} s"
    )

    print("\nMotion:")
    print(
        "  Initial pose:                  "
        f"({result.initial_pose.x:.4f}, {result.initial_pose.y:.4f}, "
        f"{math.degrees(result.initial_pose.yaw):.2f} deg)"
    )
    print(
        "  Final pose:                    "
        f"({result.final_pose.x:.4f}, {result.final_pose.y:.4f}, "
        f"{math.degrees(result.final_pose.yaw):.2f} deg)"
    )
    print(f"  Final forward displacement:    {final_forward:.4f} m")
    print(f"  Final rightward displacement:  {final_rightward:.4f} m")
    print(
        "  Maximum rightward displacement:"
        f" {result.maximum_rightward_displacement:.4f} m"
    )
    print(f"  Proxy cleared:                 {result.cleared_proxy}")
    print(f"  Route complete:                {result.route_complete}")

    print("\nSafety:")
    print(f"  Intervention observed:         {result.intervention_seen}")
    print(f"  First intervention step:       {result.first_intervention_step}")
    print(
        "  First intervention clearance: "
        f"{result.first_intervention_clearance:.4f} m"
    )
    print(
        "  Minimum policy-footprint clearance: "
        f"{result.minimum_policy_footprint_clearance:.4f} m"
    )
    print(
        "  Minimum sampled h_full:       "
        f"{result.minimum_sampled_safety:.6f}"
    )
    print(f"  Minimum CBF margin:            {result.minimum_cbf_margin:.3e}")
    print(
        "  Maximum absolute angular speed: "
        f"{result.maximum_absolute_angular_velocity:.4f} rad/s"
    )
    print(f"  All constraints valid:         {result.all_constraints_valid}")
    print(f"  Emergency stop seen:           {result.emergency_stop_seen}")
    print(f"  Stale-scan stop seen:          {result.stale_scan_stop_seen}")
    print(f"  Pipeline failure seen:         {result.pipeline_failure_seen}")
    print(f"  Ground-truth abort seen:       {result.ground_truth_abort_seen}")
    print(
        "  Final emergency reason:       "
        f"{result.final_filter_result.emergency_reason}"
    )
    print(
        "  Final query on boundary:      "
        f"{result.final_filter_result.query_on_unsafe_boundary}"
    )
    print(
        "  Final yaw interpolation:      "
        f"{result.final_filter_result.lower_yaw_slice} -> "
        f"{result.final_filter_result.upper_yaw_slice}, "
        f"weight={result.final_filter_result.yaw_interpolation_weight:.6f}"
    )

    print("\nAcceptance checks:")
    print(f"  CBF intervention occurred:     {result.intervention_seen}")
    print(f"  Requested rightward response:  {rightward_response_seen}")
    print(f"  Cleared proxy:                 {result.cleared_proxy}")
    print(
        "  Positive independent clearance: "
        f"{result.minimum_policy_footprint_clearance > GROUND_TRUTH_CLEARANCE_ABORT}"
    )
    print(f"  No fail-closed event:          {not any((result.emergency_stop_seen, result.stale_scan_stop_seen, result.pipeline_failure_seen, result.ground_truth_abort_seen))}")

    print("\nInterpretation:")
    if accepted:
        print(
            "  [PASS] The closed-loop full-state Safe-SAGE filter "
            "intervened, produced the requested rightward response, and "
            "cleared the proxy while maintaining independent clearance."
        )
    else:
        print(
            "  [WARNING] The controlled motion trial did not pass every "
            "criterion. Keep this as diagnostic evidence and do not treat "
            "the controller as motion-validated yet."
        )
    print("=" * 70)


def main() -> None:
    """Run the controlled closed-loop full-state Safe-SAGE trial."""

    print("\n" + "=" * 70)
    print("NOVA CARTER SAFE-SAGE H_FULL CLOSED-LOOP DEMO")
    print("=" * 70)
    print(f"Guidance policy: {stationary.GUIDANCE_POLICY_PATH}")
    print("Semantic label source: manual controlled proxy")

    guidance_policy = SafeSAGEGuidancePolicy.from_json(
        stationary.GUIDANCE_POLICY_PATH
    )
    class_policy = stationary.class_policy_from_guidance_policy(
        guidance_policy
    )
    measured_footprint = RectangularRobotFootprint(
        front_extent=stationary.ROBOT_FRONT_EXTENT,
        rear_extent=stationary.ROBOT_REAR_EXTENT,
        left_extent=stationary.ROBOT_LEFT_EXTENT,
        right_extent=stationary.ROBOT_RIGHT_EXTENT,
        safety_margin=stationary.ROBOT_SAFETY_MARGIN,
    )
    footprint = shift_rectangular_footprint_to_lookahead(
        measured_footprint,
        lookahead_distance=(
            stationary.UNICYCLE_LOOKAHEAD_DISTANCE
        ),
    )
    field_config = stationary.create_field_pipeline_config(footprint)
    grid_config = LocalOccupancyGridConfig(
        resolution=stationary.GRID_RESOLUTION,
        minimum_x=MOTION_GRID_MINIMUM_X,
        maximum_x=MOTION_GRID_MAXIMUM_X,
        minimum_y=stationary.GRID_MINIMUM_Y,
        maximum_y=stationary.GRID_MAXIMUM_Y,
        obstacle_inflation_radius=0.0,
    )

    loaded_scene = load_robot_scene(
        scene_path=stationary.ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=stationary.ROBOT_PRIM_PATH,
        expected_meters_per_unit=(
            stationary.EXPECTED_METERS_PER_UNIT
        ),
    )
    stationary.create_static_proxy(loaded_scene.stage)

    robot_controller = NovaCarterController(
        robot_prim_path=stationary.ROBOT_PRIM_PATH
    )
    SimulationManager.setup_simulation(
        dt=stationary.PHYSICS_DT,
        device=stationary.PHYSICS_DEVICE,
    )

    print("\nPress Play in Isaac Sim to begin.")
    if not wait_until_playing(simulation_app):
        return

    print("\nPlay detected. Initializing physics...")
    app_utils.update_app(steps=10)
    robot_controller.verify_articulation()
    stationary.position_robot_for_experiment(robot_controller)
    stationary.verify_robot_start_pose(loaded_scene.stage)
    robot_controller.print_configuration()

    lidar = NovaCarterLidar(
        lidar_prim_path=stationary.PANDAR_XT32_LIDAR_PRIM_PATH
    )
    lidar.initialize()
    robot_from_lidar = compute_relative_prim_transform(
        loaded_scene.stage,
        source_prim_path=stationary.PANDAR_XT32_LIDAR_PRIM_PATH,
        target_prim_path=stationary.ROBOT_BASE_PRIM_PATH,
    )
    floor_height_robot = stationary.estimate_floor_height_in_robot_frame(
        loaded_scene.stage
    )
    detection_config = ObstacleDetectionConfig(
        minimum_height=(
            floor_height_robot + stationary.MINIMUM_HEIGHT_ABOVE_FLOOR
        ),
        maximum_height=(
            floor_height_robot + stationary.MAXIMUM_HEIGHT_ABOVE_FLOOR
        ),
        minimum_range=stationary.MINIMUM_DETECTION_RANGE,
        maximum_range=stationary.MAXIMUM_DETECTION_RANGE,
        forward_only=False,
        angular_bin_size_degrees=(
            stationary.ANGULAR_BIN_SIZE_DEGREES
        ),
    )
    paper_filter = SafeSAGECBFSafetyFilter(
        SafeSAGECBFConfig(
            gamma=stationary.CBF_GAMMA,
            minimum_planar_velocity=(
                -robot_controller.config.max_linear_speed
            ),
            maximum_planar_velocity=(
                robot_controller.config.max_linear_speed
            ),
            minimum_angular_velocity=(
                -robot_controller.config.max_angular_speed
            ),
            maximum_angular_velocity=(
                robot_controller.config.max_angular_speed
            ),
            planar_velocity_weight=(
                stationary.CBF_LINEAR_VELOCITY_WEIGHT
            ),
            angular_velocity_weight=(
                stationary.CBF_ANGULAR_VELOCITY_WEIGHT
            ),
        )
    )
    safety_filter = SafeSAGEUnicycleCBFFilter(
        paper_filter,
        SafeSAGEUnicycleCBFConfig(
            lookahead_distance=(
                stationary.UNICYCLE_LOOKAHEAD_DISTANCE
            ),
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=(
                robot_controller.config.max_linear_speed
            ),
            minimum_angular_velocity=(
                -robot_controller.config.max_angular_speed
            ),
            maximum_angular_velocity=(
                robot_controller.config.max_angular_speed
            ),
            linear_velocity_weight=(
                stationary.CBF_LINEAR_VELOCITY_WEIGHT
            ),
            angular_velocity_weight=(
                stationary.CBF_ANGULAR_VELOCITY_WEIGHT
            ),
        ),
    )

    print("\nStarting the fail-closed closed-loop motion trial...")
    result = run_motion_trial(
        stage=loaded_scene.stage,
        lidar=lidar,
        robot_controller=robot_controller,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=grid_config,
        field_config=field_config,
        safety_filter=safety_filter,
    )
    print_motion_result(result)

    robot_controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()

    print("\nExpected successful scene behavior:")
    print("  - Nova Carter initially drives toward the blue proxy cube.")
    print("  - The CBF intervenes before the policy footprint reaches it.")
    print("  - Passing-side semantics produce motion toward robot-right.")
    print("  - The robot clears the proxy without touching it.")
    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when finished.")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 70)
        print("ERROR DURING SAFE-SAGE H_FULL MOTION DEMO")
        print("=" * 70)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
