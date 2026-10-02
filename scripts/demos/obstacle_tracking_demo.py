"""Validate LiDAR obstacle-velocity tracking without robot motion.

This controlled Isaac Sim demo:

1. Loads the restaurant robot scene.
2. Places Nova Carter near world position (26.50, 5.00).
3. Keeps the robot stationary for the entire experiment.
4. Moves a visual proxy at fixed X=28.20 from Y=7.00 to Y=4.00.
5. Extracts obstacle points from each new PandarXT-32 scan.
6. Clusters and tracks those points without simulator ground truth.
7. Compares the proxy track's estimated velocity with the known 0.60 m/s
   trajectory.
8. Verifies that the estimate returns toward zero after the proxy stops.

Simulator ground truth is used only for evaluation and report association. It
is never passed into ``ObstacleTracker``.

Run from the project root:

    python -u scripts/demos/obstacle_tracking_demo.py

The demo starts playback automatically and closes after writing its report by
default. Set ``ISAAC_SIM_AUTOCLOSE=0`` to keep the window open for inspection.
"""

from dataclasses import asdict
import json
import math
import os
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from robot.config import (
    EXPECTED_METERS_PER_UNIT,
    PANDAR_XT32_LIDAR_PRIM_PATH,
    PHYSICS_DEVICE,
    PHYSICS_DT,
    ROBOT_BASE_PRIM_PATH,
    ROBOT_PRIM_PATH,
    ROBOT_SCENE_PATH,
)

from isaacsim import SimulationApp


RTX_HISTORY_TRANSFORM_COUNT = 20

simulation_app = SimulationApp(
    {
        "headless": False,
        "enable_motion_bvh": True,
        "extra_args": [
            (
                "--/rtx/scenedb/maxHistoryTransformCount="
                f"{RTX_HISTORY_TRANSFORM_COUNT}"
            ),
        ],
    }
)


import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Usd, UsdGeom

from robot.control.differential_drive import NovaCarterController
from robot.navigation.waypoint_controller import RobotPose2D
from robot.perception.motion_diagnostic import (
    MotionDiagnosticReport,
    evaluate_motion_diagnostic,
)
from robot.runtime.simulation_loop import (
    advance_playing_steps,
    wait_until_playing,
)
from robot.safety.obstacle_detection import (
    ObstacleDetectionConfig,
    detect_obstacle_points,
)
from robot.safety.obstacle_tracking import (
    ObstacleTracker,
    ObstacleTrackerConfig,
    ObstacleTrackingResult,
    TrackedObstacle,
)
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import (
    compute_relative_prim_transform,
    get_prim_planar_pose,
    get_prim_world_transform,
)
from robot.sensor.lidar import (
    LidarFrameInfo,
    NovaCarterLidar,
)
from robot.sensor.lidar_geometry import transform_points


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------


MOVING_OBSTACLE_PRIM_PATH = "/World/ObstacleTrackingTestProxy"

ROBOT_START_POSITION_WORLD = (
    26.4955,
    5.00175,
    0.07382,
)
ROBOT_START_YAW_RADIANS = 0.0

OBSTACLE_X_WORLD = 28.20
OBSTACLE_START_Y_WORLD = 7.00
OBSTACLE_END_Y_WORLD = 4.00
OBSTACLE_SPEED_METERS_PER_SECOND = 0.60

OBSTACLE_DIMENSIONS = (
    0.30,
    0.30,
    1.00,
)
OBSTACLE_COLOR = (
    0.10,
    0.75,
    1.00,
)

PROXY_REPORT_ASSOCIATION_RADIUS = 0.60

TRACKER_CLUSTER_RADIUS = 0.35
TRACKER_MINIMUM_CLUSTER_POINTS = 2
TRACKER_MAXIMUM_ASSOCIATION_DISTANCE = 0.60
TRACKER_VELOCITY_SMOOTHING_FACTOR = 0.35
TRACKER_STATIONARY_SPEED_THRESHOLD = 0.20
TRACKER_MINIMUM_CONSECUTIVE_MOTION_UPDATES = 3
TRACKER_MAXIMUM_MISSED_UPDATES = 3

SETTLE_STEPS = 60
MAXIMUM_SCAN_WAIT_STEPS = 300

POST_STOP_OBSERVATION_SECONDS = 1.50
STATIONARY_EVALUATION_DELAY_SECONDS = 0.50

TRAJECTORY_DURATION_SECONDS = (
    OBSTACLE_START_Y_WORLD
    - OBSTACLE_END_Y_WORLD
) / OBSTACLE_SPEED_METERS_PER_SECOND

MAXIMUM_EXPERIMENT_DURATION_SECONDS = (
    TRAJECTORY_DURATION_SECONDS
    + POST_STOP_OBSERVATION_SECONDS
)
MAXIMUM_EXPERIMENT_STEPS = math.ceil(
    MAXIMUM_EXPERIMENT_DURATION_SECONDS / PHYSICS_DT
)

MAXIMUM_STALE_SCAN_SECONDS = 0.50
MAXIMUM_STALE_SCAN_STEPS = math.ceil(
    MAXIMUM_STALE_SCAN_SECONDS / PHYSICS_DT
)

STATUS_REPORT_INTERVAL_SCANS = 5

MINIMUM_MOVING_ESTIMATE_COUNT = 10
MAXIMUM_MEDIAN_MOVING_VELOCITY_ERROR = 0.25
MAXIMUM_FINAL_STATIONARY_SPEED = 0.15
MAXIMUM_PROXY_TRACK_ID_SWITCHES = 2
MAXIMUM_CONFIRMATION_LATENCY_SECONDS = 1.0
MAXIMUM_ROBOT_POSITION_DRIFT_M = 0.01
MAXIMUM_ROBOT_YAW_DRIFT_DEGREES = 0.1

OUTPUT_DIRECTORY = Path(os.environ.get(
    "CRA_LIDAR_DIAGNOSTIC_DIR", str(PROJECT_ROOT / "outputs" / "lidar_motion_isaac")
))
AUTOCLOSE = os.environ.get("ISAAC_SIM_AUTOCLOSE", "1") != "0"

MINIMUM_HEIGHT_ABOVE_FLOOR = 0.05
MAXIMUM_HEIGHT_ABOVE_FLOOR = 1.00

MINIMUM_DETECTION_RANGE = 0.15
MAXIMUM_DETECTION_RANGE = 5.00
ANGULAR_BIN_SIZE_DEGREES = 2.0

TRANSIENT_LIDAR_ERRORS = (
    "No LiDAR data is available yet",
    "contains no points",
    "no valid XYZ points remained",
)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def yaw_to_quaternion_wxyz(
    yaw_radians: float,
) -> list[float]:
    """Return a scalar-first quaternion for a world-Z yaw rotation."""

    half_yaw = 0.5 * yaw_radians

    return [
        math.cos(half_yaw),
        0.0,
        0.0,
        math.sin(half_yaw),
    ]


def position_robot_for_experiment(
    robot_controller: NovaCarterController,
) -> None:
    """Teleport the initialized articulation to the experiment pose."""

    robot_controller.robot.set_world_poses(
        positions=[
            ROBOT_START_POSITION_WORLD,
        ],
        orientations=[
            yaw_to_quaternion_wxyz(
                ROBOT_START_YAW_RADIANS
            ),
        ],
    )
    robot_controller.stop()
    app_utils.update_app(steps=5)

    print("\nExperiment robot pose applied:")
    print(
        "  Position: "
        f"({ROBOT_START_POSITION_WORLD[0]:.4f}, "
        f"{ROBOT_START_POSITION_WORLD[1]:.4f}, "
        f"{ROBOT_START_POSITION_WORLD[2]:.4f}) m"
    )
    print(
        "  Yaw:      "
        f"{math.degrees(ROBOT_START_YAW_RADIANS):.2f} degrees"
    )
    print("  Motion:   disabled during tracking validation")


def read_robot_pose(
    stage: Usd.Stage,
) -> RobotPose2D:
    """Read the current chassis pose in the world frame."""

    x_position, y_position, yaw = get_prim_planar_pose(
        stage,
        ROBOT_BASE_PRIM_PATH,
    )

    return RobotPose2D(
        x=x_position,
        y=y_position,
        yaw=yaw,
    )


def verify_robot_start_pose(
    stage: Usd.Stage,
) -> RobotPose2D:
    """Validate the chassis pose after the articulation teleport."""

    pose = read_robot_pose(stage)

    planar_error = math.hypot(
        pose.x - ROBOT_START_POSITION_WORLD[0],
        pose.y - ROBOT_START_POSITION_WORLD[1],
    )
    yaw_error = abs(
        (
            pose.yaw
            - ROBOT_START_YAW_RADIANS
            + math.pi
        )
        % (2.0 * math.pi)
        - math.pi
    )

    print("\nApplied chassis pose:")
    print(f"  x:         {pose.x:.4f} m")
    print(f"  y:         {pose.y:.4f} m")
    print(
        "  yaw:       "
        f"{math.degrees(pose.yaw):.2f} degrees"
    )
    print(f"  XY error:  {planar_error:.4f} m")
    print(
        "  Yaw error: "
        f"{math.degrees(yaw_error):.2f} degrees"
    )

    if planar_error > 0.10:
        raise RuntimeError(
            "The robot articulation did not reach the requested "
            "experiment start position."
        )

    if yaw_error > math.radians(5.0):
        raise RuntimeError(
            "The robot articulation did not reach the requested "
            "experiment start orientation."
        )

    print("  [PASS] Runtime robot pose verified.")

    return pose


def create_moving_obstacle(
    stage: Usd.Stage,
) -> UsdGeom.XformOp:
    """Create the session-only visual proxy."""

    existing_prim = stage.GetPrimAtPath(
        MOVING_OBSTACLE_PRIM_PATH
    )

    if existing_prim.IsValid():
        raise RuntimeError(
            "The moving-obstacle prim already exists: "
            f"{MOVING_OBSTACLE_PRIM_PATH}"
        )

    obstacle_height = OBSTACLE_DIMENSIONS[2]

    with Usd.EditContext(
        stage,
        stage.GetSessionLayer(),
    ):
        cube = UsdGeom.Cube.Define(
            stage,
            MOVING_OBSTACLE_PRIM_PATH,
        )
        cube.CreateSizeAttr(1.0)
        cube.CreateExtentAttr(
            [
                Gf.Vec3f(-0.5, -0.5, -0.5),
                Gf.Vec3f(0.5, 0.5, 0.5),
            ]
        )
        cube.CreateDisplayColorAttr(
            [
                Gf.Vec3f(*OBSTACLE_COLOR),
            ]
        )

        translate_op = cube.AddTranslateOp()
        translate_op.Set(
            Gf.Vec3d(
                OBSTACLE_X_WORLD,
                OBSTACLE_START_Y_WORLD,
                obstacle_height / 2.0,
            )
        )

        cube.AddOrientOp().Set(Gf.Quatf(1.0))
        cube.AddScaleOp().Set(
            Gf.Vec3f(*OBSTACLE_DIMENSIONS)
        )

    print("\nMoving tracking proxy created:")
    print(f"  Prim:       {MOVING_OBSTACLE_PRIM_PATH}")
    print(
        "  Start:      "
        f"({OBSTACLE_X_WORLD:.3f}, "
        f"{OBSTACLE_START_Y_WORLD:.3f}) m"
    )
    print(
        "  End:        "
        f"({OBSTACLE_X_WORLD:.3f}, "
        f"{OBSTACLE_END_Y_WORLD:.3f}) m"
    )
    print(
        "  Velocity:   "
        f"(0.000, {-OBSTACLE_SPEED_METERS_PER_SECOND:.3f}) m/s"
    )
    print("  Collision:  disabled")
    print("  Layer:      session only")

    return translate_op


def obstacle_position_at_time(
    elapsed_seconds: float,
) -> np.ndarray:
    """Return the deterministic proxy center position."""

    travel_distance = min(
        OBSTACLE_SPEED_METERS_PER_SECOND
        * max(0.0, elapsed_seconds),
        OBSTACLE_START_Y_WORLD
        - OBSTACLE_END_Y_WORLD,
    )

    return np.array(
        [
            OBSTACLE_X_WORLD,
            OBSTACLE_START_Y_WORLD - travel_distance,
            OBSTACLE_DIMENSIONS[2] / 2.0,
        ],
        dtype=np.float64,
    )


def obstacle_velocity_world_at_time(
    elapsed_seconds: float,
) -> np.ndarray:
    """Return the proxy's ground-truth planar velocity."""

    if elapsed_seconds >= TRAJECTORY_DURATION_SECONDS:
        return np.zeros(2, dtype=np.float64)

    return np.array(
        [
            0.0,
            -OBSTACLE_SPEED_METERS_PER_SECOND,
        ],
        dtype=np.float64,
    )


def set_obstacle_position(
    stage: Usd.Stage,
    translate_op: UsdGeom.XformOp,
    position_world: np.ndarray,
) -> None:
    """Author one session-layer proxy position."""

    with Usd.EditContext(
        stage,
        stage.GetSessionLayer(),
    ):
        translate_op.Set(
            Gf.Vec3d(
                float(position_world[0]),
                float(position_world[1]),
                float(position_world[2]),
            )
        )


# ---------------------------------------------------------------------------
# LiDAR and evaluation helpers
# ---------------------------------------------------------------------------


def estimate_floor_height_in_robot_frame(
    stage: Usd.Stage,
) -> float:
    """Estimate world-Z-zero height in the current chassis frame."""

    robot_to_world = get_prim_world_transform(
        stage,
        ROBOT_BASE_PRIM_PATH,
    )
    robot_origin_world = (
        robot_to_world.ExtractTranslation()
    )
    ground_point_world = Gf.Vec3d(
        robot_origin_world[0],
        robot_origin_world[1],
        0.0,
    )
    ground_point_robot = (
        robot_to_world.GetInverse().Transform(
            ground_point_world
        )
    )

    return float(ground_point_robot[2])


def obstacle_center_in_robot_frame(
    pose: RobotPose2D,
    obstacle_position_world: np.ndarray,
) -> np.ndarray:
    """Express the proxy center in current chassis coordinates."""

    x_world_error = (
        float(obstacle_position_world[0]) - pose.x
    )
    y_world_error = (
        float(obstacle_position_world[1]) - pose.y
    )

    cosine_yaw = math.cos(pose.yaw)
    sine_yaw = math.sin(pose.yaw)

    return np.array(
        [
            (
                cosine_yaw * x_world_error
                + sine_yaw * y_world_error
            ),
            (
                -sine_yaw * x_world_error
                + cosine_yaw * y_world_error
            ),
        ],
        dtype=np.float64,
    )


def find_proxy_track(
    tracking_result: ObstacleTrackingResult,
    *,
    robot_pose: RobotPose2D,
    obstacle_position_world: np.ndarray,
) -> tuple[TrackedObstacle | None, float]:
    """Find the reported track nearest the known proxy center.

    This function is evaluation-only. The known center is not used by the
    tracker itself.
    """

    if not tracking_result.tracked_obstacles:
        return None, math.inf

    expected_center = obstacle_center_in_robot_frame(
        robot_pose,
        obstacle_position_world,
    )

    best_track: TrackedObstacle | None = None
    best_error = math.inf

    for obstacle in tracking_result.tracked_obstacles:
        centroid_robot = np.asarray(
            obstacle.centroid_robot_xy,
            dtype=np.float64,
        )
        error = float(
            np.linalg.norm(
                centroid_robot - expected_center
            )
        )

        if error < best_error:
            best_track = obstacle
            best_error = error

    if best_error > PROXY_REPORT_ASSOCIATION_RADIUS:
        return None, best_error

    return best_track, best_error


def wait_for_first_scan(
    lidar: NovaCarterLidar,
    robot_controller: NovaCarterController,
) -> tuple[np.ndarray, LidarFrameInfo, int]:
    """Advance until the first valid PandarXT-32 scan arrives."""

    last_transient_error: RuntimeError | None = None

    for completed_steps in range(
        1,
        MAXIMUM_SCAN_WAIT_STEPS + 1,
    ):
        if not simulation_app.is_running():
            raise RuntimeError(
                "Isaac Sim closed before a LiDAR scan arrived."
            )

        if not app_utils.is_playing():
            simulation_app.update()
            continue

        robot_controller.stop()
        simulation_app.update()

        try:
            points = lidar.get_point_cloud()
        except RuntimeError as error:
            is_transient = any(
                message in str(error)
                for message in TRANSIENT_LIDAR_ERRORS
            )

            if not is_transient:
                raise

            last_transient_error = error
            continue

        return (
            points,
            lidar.get_frame_info(),
            completed_steps,
        )

    detail = (
        f"\nLast LiDAR error: {last_transient_error}"
        if last_transient_error is not None
        else ""
    )

    raise TimeoutError(
        "No valid LiDAR scan arrived after "
        f"{MAXIMUM_SCAN_WAIT_STEPS} rendered updates."
        f"{detail}"
    )


def extract_obstacle_points(
    points_sensor_frame: np.ndarray,
    *,
    robot_from_lidar: np.ndarray,
    detection_config: ObstacleDetectionConfig,
) -> np.ndarray:
    """Transform one scan and extract planar obstacle points."""

    points_robot_frame = transform_points(
        points_sensor_frame,
        robot_from_lidar,
    )

    return detect_obstacle_points(
        points_robot_frame,
        detection_config,
    )


def format_value(
    value: float,
) -> str:
    """Format a finite value or an unavailable result."""

    if math.isfinite(value):
        return f"{value:.3f}"

    return "none"


def print_initial_report(
    *,
    pose: RobotPose2D,
    frame_info: LidarFrameInfo,
    obstacle_points: np.ndarray,
    tracking_result: ObstacleTrackingResult,
    proxy_track: TrackedObstacle | None,
    proxy_association_error: float,
) -> None:
    """Print stationary tracker initialization results."""

    print("\n" + "=" * 70)
    print("OBSTACLE TRACKING STAGE A - STATIONARY INITIALIZATION")
    print("=" * 70)
    print("\nRobot:")
    print(
        f"  Pose: x={pose.x:.4f} m, y={pose.y:.4f} m, "
        f"yaw={math.degrees(pose.yaw):.2f} degrees"
    )
    print("  Motion command: stopped")

    print("\nLiDAR:")
    print(f"  Frame ID:        {frame_info.sensor_frame_id}")
    print(
        "  Sensor time:     "
        f"{tracking_result.timestamp_seconds:.6f} s"
    )
    print(f"  Obstacle points: {obstacle_points.shape[0]}")
    print(f"  Clusters:        {tracking_result.cluster_count}")

    print("\nProxy evaluation:")
    print(f"  Track found:     {proxy_track is not None}")
    print(
        "  Association error: "
        f"{format_value(proxy_association_error)} m"
    )

    if proxy_track is not None:
        print(f"  Track ID:        {proxy_track.track_id}")
        print(
            "  Initial velocity: "
            f"({proxy_track.velocity_world_xy[0]:.3f}, "
            f"{proxy_track.velocity_world_xy[1]:.3f}) m/s"
        )

    print(
        "\nExpected: the first observation initializes velocity to zero."
    )
    print("The robot and proxy remained stationary during Stage A.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Tracking experiment
# ---------------------------------------------------------------------------


def run_tracking_experiment(
    *,
    stage: Usd.Stage,
    obstacle_translate_op: UsdGeom.XformOp,
    lidar: NovaCarterLidar,
    robot_controller: NovaCarterController,
    tracker: ObstacleTracker,
    detection_config: ObstacleDetectionConfig,
    robot_from_lidar: np.ndarray,
    initial_frame_id: int,
) -> MotionDiagnosticReport:
    """Move the proxy and validate LiDAR-only velocity estimates."""

    completed_steps = 0
    received_scan_count = 0
    steps_since_new_scan = 0
    latest_frame_id = initial_frame_id

    proxy_track_scan_count = 0
    proxy_track_id_switches = 0
    previous_proxy_track_id: int | None = None

    moving_velocity_errors: list[float] = []
    moving_estimated_speeds: list[float] = []
    moving_estimated_velocities: list[np.ndarray] = []
    stationary_estimated_speeds: list[float] = []
    first_confirmed_motion_seconds: float | None = None

    final_proxy_track: TrackedObstacle | None = None
    minimum_proxy_association_error = math.inf

    termination_reason = (
        "Maximum tracking duration reached."
    )

    print("\n" + "=" * 70)
    print("OBSTACLE TRACKING STAGE B - MOVING PROXY VALIDATION")
    print("=" * 70)
    print(
        "Ground-truth velocity while moving: "
        f"(0.000, {-OBSTACLE_SPEED_METERS_PER_SECOND:.3f}) m/s"
    )
    print(
        "Trajectory duration: "
        f"{TRAJECTORY_DURATION_SECONDS:.2f} s"
    )
    print(
        "Post-stop observation: "
        f"{POST_STOP_OBSERVATION_SECONDS:.2f} s"
    )
    print(
        "Stationary-speed deadband: "
        f"{TRACKER_STATIONARY_SPEED_THRESHOLD:.3f} m/s"
    )
    print(
        "Motion confirmation: "
        f"{TRACKER_MINIMUM_CONSECUTIVE_MOTION_UPDATES} scans"
    )
    print("\nLive status:")

    while (
        completed_steps < MAXIMUM_EXPERIMENT_STEPS
        and simulation_app.is_running()
    ):
        if not app_utils.is_playing():
            robot_controller.stop()
            simulation_app.update()
            continue

        elapsed_seconds = (
            completed_steps * PHYSICS_DT
        )
        obstacle_position_world = (
            obstacle_position_at_time(
                elapsed_seconds
            )
        )

        set_obstacle_position(
            stage,
            obstacle_translate_op,
            obstacle_position_world,
        )

        robot_controller.stop()
        simulation_app.update()
        completed_steps += 1
        steps_since_new_scan += 1

        try:
            points_sensor_frame = (
                lidar.get_point_cloud()
            )
            frame_info = lidar.get_frame_info()
        except RuntimeError as error:
            is_transient = any(
                message in str(error)
                for message in TRANSIENT_LIDAR_ERRORS
            )

            if not is_transient:
                raise

            if (
                steps_since_new_scan
                >= MAXIMUM_STALE_SCAN_STEPS
            ):
                termination_reason = (
                    "LiDAR data became stale."
                )
                break

            continue

        if frame_info.sensor_frame_id == latest_frame_id:
            if (
                steps_since_new_scan
                >= MAXIMUM_STALE_SCAN_STEPS
            ):
                termination_reason = (
                    "LiDAR data became stale."
                )
                break

            continue

        latest_frame_id = frame_info.sensor_frame_id
        steps_since_new_scan = 0
        received_scan_count += 1

        obstacle_points = extract_obstacle_points(
            points_sensor_frame,
            robot_from_lidar=robot_from_lidar,
            detection_config=detection_config,
        )

        pose = read_robot_pose(stage)
        sensor_timestamp_seconds = (
            frame_info.sensor_timestamp_nanoseconds
            * 1e-9
        )

        tracking_result = tracker.update(
            obstacle_points,
            robot_position_world_xy=np.array(
                [pose.x, pose.y],
                dtype=np.float64,
            ),
            robot_yaw=pose.yaw,
            timestamp_seconds=sensor_timestamp_seconds,
        )

        proxy_track, proxy_error = find_proxy_track(
            tracking_result,
            robot_pose=pose,
            obstacle_position_world=(
                obstacle_position_world
            ),
        )

        minimum_proxy_association_error = min(
            minimum_proxy_association_error,
            proxy_error,
        )

        expected_velocity_world = (
            obstacle_velocity_world_at_time(
                elapsed_seconds
            )
        )

        estimated_velocity_world: np.ndarray | None = None
        velocity_error = math.inf

        if proxy_track is not None:
            proxy_track_scan_count += 1
            final_proxy_track = proxy_track

            if (
                previous_proxy_track_id is not None
                and proxy_track.track_id
                != previous_proxy_track_id
            ):
                proxy_track_id_switches += 1

            previous_proxy_track_id = (
                proxy_track.track_id
            )

            estimated_velocity_world = np.asarray(
                proxy_track.velocity_world_xy,
                dtype=np.float64,
            )
            velocity_error = float(
                np.linalg.norm(
                    estimated_velocity_world
                    - expected_velocity_world
                )
            )
            estimated_speed = float(
                np.linalg.norm(
                    estimated_velocity_world
                )
            )

            moving_evaluation_active = (
                elapsed_seconds >= 0.20
                and elapsed_seconds
                < TRAJECTORY_DURATION_SECONDS
                and (
                    proxy_track.observation_count
                    >= (
                        TRACKER_MINIMUM_CONSECUTIVE_MOTION_UPDATES
                        + 1
                    )
                )
            )

            if moving_evaluation_active:
                moving_velocity_errors.append(
                    velocity_error
                )
                moving_estimated_speeds.append(
                    estimated_speed
                )
                moving_estimated_velocities.append(
                    estimated_velocity_world.copy()
                )

            if (
                first_confirmed_motion_seconds is None
                and elapsed_seconds < TRAJECTORY_DURATION_SECONDS
                and estimated_speed > 0.0
            ):
                first_confirmed_motion_seconds = elapsed_seconds

            stationary_evaluation_active = (
                elapsed_seconds
                >= (
                    TRAJECTORY_DURATION_SECONDS
                    + STATIONARY_EVALUATION_DELAY_SECONDS
                )
            )

            if stationary_evaluation_active:
                stationary_estimated_speeds.append(
                    estimated_speed
                )

        should_report = (
            received_scan_count == 1
            or received_scan_count
            % STATUS_REPORT_INTERVAL_SCANS
            == 0
            or abs(
                elapsed_seconds
                - TRAJECTORY_DURATION_SECONDS
            )
            <= PHYSICS_DT
        )

        if should_report:
            if estimated_velocity_world is None:
                estimated_velocity_text = "(none, none)"
                track_id_text = "none"
            else:
                estimated_velocity_text = (
                    f"({estimated_velocity_world[0]:.3f}, "
                    f"{estimated_velocity_world[1]:.3f})"
                )
                track_id_text = str(
                    proxy_track.track_id
                )

            print(
                f"  scan={received_scan_count:<3} "
                f"t={elapsed_seconds:5.2f} "
                f"frame={latest_frame_id:<4} "
                f"proxy_y={obstacle_position_world[1]:5.2f} "
                f"points={obstacle_points.shape[0]:<3} "
                f"clusters={tracking_result.cluster_count:<2} "
                f"track={track_id_text:<4} "
                f"assoc={format_value(proxy_error):>5} "
                f"v_est={estimated_velocity_text} "
                f"error={format_value(velocity_error)}"
            )

    robot_controller.stop()
    app_utils.update_app(steps=5)

    moving_estimate_count = len(
        moving_velocity_errors
    )

    median_moving_velocity_error = (
        float(np.median(moving_velocity_errors))
        if moving_velocity_errors
        else math.inf
    )
    median_moving_estimated_speed = (
        float(np.median(moving_estimated_speeds))
        if moving_estimated_speeds
        else math.inf
    )
    final_stationary_speed = (
        stationary_estimated_speeds[-1]
        if stationary_estimated_speeds
        else math.inf
    )

    report = evaluate_motion_diagnostic(
        np.asarray(moving_estimated_velocities, dtype=np.float64).reshape(-1, 2),
        expected_velocity_world_xy=(0.0, -OBSTACLE_SPEED_METERS_PER_SECOND),
        first_confirmed_motion_seconds=first_confirmed_motion_seconds,
        final_stationary_speed_mps=final_stationary_speed,
        proxy_track_id_switches=proxy_track_id_switches,
        minimum_association_error_m=minimum_proxy_association_error,
        minimum_estimates=MINIMUM_MOVING_ESTIMATE_COUNT,
        maximum_median_velocity_error_mps=MAXIMUM_MEDIAN_MOVING_VELOCITY_ERROR,
        maximum_confirmation_latency_seconds=MAXIMUM_CONFIRMATION_LATENCY_SECONDS,
        maximum_final_stationary_speed_mps=MAXIMUM_FINAL_STATIONARY_SPEED,
        maximum_track_id_switches=MAXIMUM_PROXY_TRACK_ID_SWITCHES,
    )
    enough_moving_estimates = moving_estimate_count >= MINIMUM_MOVING_ESTIMATE_COUNT
    moving_accuracy_passed = (
        median_moving_velocity_error <= MAXIMUM_MEDIAN_MOVING_VELOCITY_ERROR
    )
    stationary_accuracy_passed = final_stationary_speed <= MAXIMUM_FINAL_STATIONARY_SPEED
    identity_stability_passed = proxy_track_id_switches <= MAXIMUM_PROXY_TRACK_ID_SWITCHES

    print("\n" + "=" * 70)
    print("OBSTACLE TRACKING VALIDATION RESULT")
    print("=" * 70)
    print(f"Reason: {termination_reason}")
    print(f"Completed physics steps:       {completed_steps}")
    print(f"New LiDAR frames:              {received_scan_count}")
    print(f"Proxy-track frames:            {proxy_track_scan_count}")
    print(f"Moving velocity estimates:     {moving_estimate_count}")
    print(f"Proxy track-ID switches:       {proxy_track_id_switches}")
    print(
        "Minimum proxy association error: "
        f"{format_value(minimum_proxy_association_error)} m"
    )
    print(
        "Median estimated moving speed:   "
        f"{format_value(median_moving_estimated_speed)} m/s"
    )
    print(
        "Ground-truth moving speed:        "
        f"{OBSTACLE_SPEED_METERS_PER_SECOND:.3f} m/s"
    )
    print(
        "Median moving velocity error:     "
        f"{format_value(median_moving_velocity_error)} m/s"
    )
    print(
        "Final estimated stationary speed: "
        f"{format_value(final_stationary_speed)} m/s"
    )
    print(
        "Time to first confirmed motion:    "
        f"{format_value(first_confirmed_motion_seconds if first_confirmed_motion_seconds is not None else math.inf)} s"
    )

    if final_proxy_track is None:
        print("Final proxy track:               none")
    else:
        print(
            "Final proxy track:               "
            f"{final_proxy_track.track_id}"
        )

    print("\nChecks:")
    print(
        "  Enough moving estimates: "
        f"{enough_moving_estimates}"
    )
    print(
        "  Moving accuracy:         "
        f"{moving_accuracy_passed}"
    )
    print(
        "  Returns to stationary:   "
        f"{stationary_accuracy_passed}"
    )
    print(
        "  Track identity stable:   "
        f"{identity_stability_passed}"
    )

    print("\nInterpretation:")

    if report.passed:
        print(
            "  [PASS] LiDAR-only tracking recovered the proxy motion, "
            "maintained a stable identity, and returned toward zero "
            "velocity after the proxy stopped."
        )
    else:
        print(
            "  [REVIEW] One or more tracking checks did not pass. "
            "Inspect clustering, association, or smoothing parameters "
            "before connecting the tracker to the CBF."
        )

    print("=" * 70)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run stationary-robot obstacle-tracking validation."""

    print("\n" + "=" * 70)
    print("NOVA CARTER LIDAR OBSTACLE TRACKING DEMO")
    print("=" * 70)
    print("RTX Motion BVH requested: True")
    print(
        "RTX history transform count: "
        f"{RTX_HISTORY_TRANSFORM_COUNT}"
    )

    loaded_scene = load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=EXPECTED_METERS_PER_UNIT,
    )

    obstacle_translate_op = create_moving_obstacle(
        loaded_scene.stage
    )

    robot_controller = NovaCarterController(
        robot_prim_path=ROBOT_PRIM_PATH
    )

    SimulationManager.setup_simulation(
        dt=PHYSICS_DT,
        device=PHYSICS_DEVICE,
    )

    print("\nStarting Isaac Sim playback automatically...")
    app_utils.play()

    if not wait_until_playing(simulation_app):
        return

    print("\nPlay detected. Initializing physics...")
    app_utils.update_app(steps=10)

    robot_controller.verify_articulation()
    position_robot_for_experiment(
        robot_controller
    )
    verify_robot_start_pose(
        loaded_scene.stage
    )
    robot_controller.print_configuration()

    lidar = NovaCarterLidar(
        lidar_prim_path=PANDAR_XT32_LIDAR_PRIM_PATH
    )
    lidar.initialize()

    print(
        f"\nHolding the robot and proxy still for "
        f"{SETTLE_STEPS} physics steps..."
    )

    if not advance_playing_steps(
        simulation_app,
        physics_steps=SETTLE_STEPS,
        on_step=robot_controller.stop,
    ):
        return

    robot_from_lidar = compute_relative_prim_transform(
        loaded_scene.stage,
        source_prim_path=PANDAR_XT32_LIDAR_PRIM_PATH,
        target_prim_path=ROBOT_BASE_PRIM_PATH,
    )

    floor_height_robot = (
        estimate_floor_height_in_robot_frame(
            loaded_scene.stage
        )
    )

    detection_config = ObstacleDetectionConfig(
        minimum_height=(
            floor_height_robot
            + MINIMUM_HEIGHT_ABOVE_FLOOR
        ),
        maximum_height=(
            floor_height_robot
            + MAXIMUM_HEIGHT_ABOVE_FLOOR
        ),
        minimum_range=MINIMUM_DETECTION_RANGE,
        maximum_range=MAXIMUM_DETECTION_RANGE,
        forward_only=True,
        angular_bin_size_degrees=(
            ANGULAR_BIN_SIZE_DEGREES
        ),
    )

    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=TRACKER_CLUSTER_RADIUS,
            minimum_cluster_points=(
                TRACKER_MINIMUM_CLUSTER_POINTS
            ),
            maximum_association_distance=(
                TRACKER_MAXIMUM_ASSOCIATION_DISTANCE
            ),
            velocity_smoothing_factor=(
                TRACKER_VELOCITY_SMOOTHING_FACTOR
            ),
            stationary_speed_threshold=(
                TRACKER_STATIONARY_SPEED_THRESHOLD
            ),
            minimum_consecutive_motion_updates=(
                TRACKER_MINIMUM_CONSECUTIVE_MOTION_UPDATES
            ),
            maximum_missed_updates=(
                TRACKER_MAXIMUM_MISSED_UPDATES
            ),
        )
    )

    print("\nWaiting for a valid LiDAR scan...")

    (
        points_sensor_frame,
        initial_frame_info,
        scan_wait_steps,
    ) = wait_for_first_scan(
        lidar,
        robot_controller,
    )

    initial_obstacle_points = extract_obstacle_points(
        points_sensor_frame,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
    )

    if initial_obstacle_points.shape[0] == 0:
        raise RuntimeError(
            "The initial scan produced no obstacle points."
        )

    initial_pose = read_robot_pose(
        loaded_scene.stage
    )
    initial_tracking_result = tracker.update(
        initial_obstacle_points,
        robot_position_world_xy=np.array(
            [initial_pose.x, initial_pose.y],
            dtype=np.float64,
        ),
        robot_yaw=initial_pose.yaw,
        timestamp_seconds=(
            initial_frame_info.sensor_timestamp_nanoseconds
            * 1e-9
        ),
    )

    initial_obstacle_position = (
        obstacle_position_at_time(0.0)
    )
    (
        initial_proxy_track,
        initial_proxy_error,
    ) = find_proxy_track(
        initial_tracking_result,
        robot_pose=initial_pose,
        obstacle_position_world=(
            initial_obstacle_position
        ),
    )

    print(
        "Valid scan received after "
        f"{scan_wait_steps} update(s)."
    )

    print_initial_report(
        pose=initial_pose,
        frame_info=initial_frame_info,
        obstacle_points=initial_obstacle_points,
        tracking_result=initial_tracking_result,
        proxy_track=initial_proxy_track,
        proxy_association_error=(
            initial_proxy_error
        ),
    )

    report = run_tracking_experiment(
        stage=loaded_scene.stage,
        obstacle_translate_op=(
            obstacle_translate_op
        ),
        lidar=lidar,
        robot_controller=robot_controller,
        tracker=tracker,
        detection_config=detection_config,
        robot_from_lidar=robot_from_lidar,
        initial_frame_id=(
            initial_frame_info.sensor_frame_id
        ),
    )

    robot_controller.stop()
    app_utils.update_app(steps=5)
    final_pose = read_robot_pose(loaded_scene.stage)
    robot_drift_m = math.hypot(
        final_pose.x - initial_pose.x,
        final_pose.y - initial_pose.y,
    )
    robot_yaw_drift_degrees = math.degrees(
        abs(
            math.atan2(
                math.sin(final_pose.yaw - initial_pose.yaw),
                math.cos(final_pose.yaw - initial_pose.yaw),
            )
        )
    )
    blockers = list(report.blockers)
    if robot_drift_m > MAXIMUM_ROBOT_POSITION_DRIFT_M:
        blockers.append("robot position drift exceeded the stationary tolerance")
    if robot_yaw_drift_degrees > MAXIMUM_ROBOT_YAW_DRIFT_DEGREES:
        blockers.append("robot yaw drift exceeded the stationary tolerance")
    payload = {
        **asdict(report),
        "blockers": blockers,
        "passed": not blockers,
        "robot_motion_command": "zero",
        "robot_position_drift_m": robot_drift_m,
        "robot_yaw_drift_degrees": robot_yaw_drift_degrees,
        "runtime_tracker_config_changed": False,
        "slow_motion_validated": False,
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    app_utils.stop()

    if blockers:
        raise RuntimeError("stationary-robot LiDAR motion diagnostic did not pass")

    if not AUTOCLOSE:
        print("\nThe GUI will remain open for inspection.")
        print("Close the Isaac Sim window when finished.")
        while simulation_app.is_running():
            simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 70)
        print("ERROR DURING OBSTACLE TRACKING DEMO")
        print("=" * 70)
        print(error)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
