"""Compare static and velocity-aware CBFs against a crossing obstacle.

This controlled Isaac Sim demo uses a session-only visual cube as a moving
obstacle. The proxy does not have a physics collider: it is intended to test
LiDAR perception and safety-filter behavior without pushing the robot.

Scenario:

1. Nova Carter starts near world position (26.50, 5.00).
2. Its nominal waypoint is at approximately (29.00, 5.00).
3. A proxy obstacle moves at fixed X=28.20 from Y=7.00 to Y=4.00.
4. Static and velocity-aware CBFs evaluate the same LiDAR observations.
5. Known proxy velocity is assigned only to LiDAR points associated with the
   proxy; background points remain stationary.
6. The velocity-aware command controls the robot.
7. The demo compares intervention timing and reports safety and route metrics.

Expected first-run behavior:

    robot approaches
        -> proxy crosses the route
        -> CBF slows or stops the robot
        -> proxy clears
        -> CBF releases
        -> robot continues to the waypoint

The proxy trajectory and ground-truth association are experimental
infrastructure. Known proxy velocity is intentionally used as an oracle input
to isolate and validate the velocity-aware CBF mathematics. Velocity estimation
from sensor history belongs in a later tracking module.

Run from the project root:

    python -u scripts/demos/velocity_aware_cbf_demo.py

Press Play in Isaac Sim when requested.
"""

import math
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


# The RTX renderer requested 17 stored transforms in the validation run. Pass
# a slightly larger value at application startup so the setting is active
# before the renderer and RTX LiDAR initialize.
RTX_HISTORY_TRANSFORM_COUNT = 20

# Motion BVH is required for reliable motion effects from moving RTX geometry.
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
from robot.navigation.waypoint_controller import (
    RobotPose2D,
    Waypoint,
    WaypointCommand,
    WaypointController,
    WaypointControllerConfig,
)
from robot.runtime.simulation_loop import (
    advance_playing_steps,
    wait_until_playing,
)
from robot.safety.cbf_filter import (
    CBFConfig,
    CBFResult,
    CBFSafetyFilter,
)
from robot.safety.obstacle_detection import (
    ObstacleDetectionConfig,
    detect_obstacle_points,
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


MOVING_OBSTACLE_PRIM_PATH = "/World/DynamicCBFTestObstacle"

ROBOT_START_POSITION_WORLD = (
    26.4955,
    5.00175,
    0.07382,
)
ROBOT_START_YAW_RADIANS = 0.0

ROBOT_GOAL_WORLD = (
    29.00,
    5.00175,
)

OBSTACLE_X_WORLD = 28.20
OBSTACLE_START_Y_WORLD = 7.00
OBSTACLE_END_Y_WORLD = 4.00

# At 0.60 m/s, the proxy reaches the route centerline after about 3.33 s.
# Without CBF intervention, the robot would still be roughly 1.04 m behind the
# proxy center at that time. The 0.30 m proxy width leaves about 0.89 m to its
# nearest face, providing a controlled approach to the 0.80 m safety radius.
OBSTACLE_SPEED_METERS_PER_SECOND = 0.60

OBSTACLE_DIMENSIONS = (
    0.30,
    0.30,
    1.00,
)
OBSTACLE_COLOR = (
    1.0,
    0.15,
    0.05,
)

CBF_SAFETY_RADIUS = 0.80
CBF_ALPHA = 1.0

# Freeze the experiment if the known proxy surface reaches this distance,
# before the proxy can visually overlap the robot.
GROUND_TRUTH_ABORT_DISTANCE = 0.70

PROXY_ASSOCIATION_RADIUS = 0.60

SETTLE_STEPS = 60
MOTION_START_DELAY_STEPS = 120
MAXIMUM_SCAN_WAIT_STEPS = 300

MAXIMUM_EXPERIMENT_DURATION_SECONDS = 25.0
MAXIMUM_EXPERIMENT_STEPS = math.ceil(
    MAXIMUM_EXPERIMENT_DURATION_SECONDS / PHYSICS_DT
)

MAXIMUM_STALE_SCAN_SECONDS = 0.25
MAXIMUM_STALE_SCAN_STEPS = math.ceil(
    MAXIMUM_STALE_SCAN_SECONDS / PHYSICS_DT
)

STATUS_REPORT_INTERVAL_STEPS = 30

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
# Scene and trajectory helpers
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
    """Teleport the initialized articulation to the experiment start pose."""

    orientation = yaw_to_quaternion_wxyz(
        ROBOT_START_YAW_RADIANS
    )

    # Apply the pose after physics initializes. Setting it before Play may be
    # overwritten when the articulation creates its runtime physics state.
    robot_controller.robot.set_world_poses(
        positions=[
            ROBOT_START_POSITION_WORLD,
        ],
        orientations=[
            orientation,
        ],
    )
    robot_controller.stop()

    # Propagate the articulation pose to USD, rendering, and attached sensors.
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
    print("  Mode:     runtime articulation teleport")


def verify_robot_start_pose(
    stage: Usd.Stage,
) -> RobotPose2D:
    """Validate the current chassis pose after the articulation teleport."""

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
    """Create the session-only visual proxy and return its translate op."""

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

        cube.AddOrientOp().Set(
            Gf.Quatf(1.0)
        )
        cube.AddScaleOp().Set(
            Gf.Vec3f(
                *OBSTACLE_DIMENSIONS
            )
        )

    print("\nMoving LiDAR proxy created:")
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
        "  Speed:      "
        f"{OBSTACLE_SPEED_METERS_PER_SECOND:.3f} m/s"
    )
    print(
        "  Dimensions: "
        f"{OBSTACLE_DIMENSIONS} m"
    )
    print("  Collision:  disabled")
    print("  Layer:      session only")

    return translate_op


def obstacle_position_at_time(
    elapsed_seconds: float,
) -> np.ndarray:
    """Return the proxy center position for the deterministic trajectory."""

    travel_distance = min(
        OBSTACLE_SPEED_METERS_PER_SECOND
        * max(0.0, elapsed_seconds),
        OBSTACLE_START_Y_WORLD
        - OBSTACLE_END_Y_WORLD,
    )

    y_position = (
        OBSTACLE_START_Y_WORLD
        - travel_distance
    )

    return np.array(
        [
            OBSTACLE_X_WORLD,
            y_position,
            OBSTACLE_DIMENSIONS[2] / 2.0,
        ],
        dtype=np.float64,
    )


def obstacle_has_finished(
    position_world: np.ndarray,
) -> bool:
    """Whether the proxy has reached the end of its trajectory."""

    return bool(
        position_world[1]
        <= OBSTACLE_END_Y_WORLD + 1e-9
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
# Pose, geometry, and LiDAR helpers
# ---------------------------------------------------------------------------


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


def distance_to_obstacle_surface(
    pose: RobotPose2D,
    obstacle_position_world: np.ndarray,
) -> float:
    """Return planar distance from the chassis center to the proxy box."""

    half_x = 0.5 * OBSTACLE_DIMENSIONS[0]
    half_y = 0.5 * OBSTACLE_DIMENSIONS[1]

    x_separation = max(
        abs(
            pose.x
            - float(obstacle_position_world[0])
        )
        - half_x,
        0.0,
    )
    y_separation = max(
        abs(
            pose.y
            - float(obstacle_position_world[1])
        )
        - half_y,
        0.0,
    )

    return math.hypot(
        x_separation,
        y_separation,
    )


def obstacle_center_in_robot_frame(
    pose: RobotPose2D,
    obstacle_position_world: np.ndarray,
) -> np.ndarray:
    """Express the proxy center in the current planar chassis frame."""

    x_world_error = (
        float(obstacle_position_world[0])
        - pose.x
    )
    y_world_error = (
        float(obstacle_position_world[1])
        - pose.y
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


def obstacle_velocity_world_at_time(
    elapsed_seconds: float,
) -> np.ndarray:
    """Return the proxy's deterministic planar world-frame velocity."""

    trajectory_duration = (
        OBSTACLE_START_Y_WORLD
        - OBSTACLE_END_Y_WORLD
    ) / OBSTACLE_SPEED_METERS_PER_SECOND

    if elapsed_seconds >= trajectory_duration:
        return np.zeros(2, dtype=np.float64)

    return np.array(
        [
            0.0,
            -OBSTACLE_SPEED_METERS_PER_SECOND,
        ],
        dtype=np.float64,
    )


def world_velocity_to_robot_frame(
    velocity_world_xy: np.ndarray,
    *,
    robot_yaw: float,
) -> np.ndarray:
    """Resolve a planar world-frame vector into robot base-frame axes."""

    velocity_world = np.asarray(
        velocity_world_xy,
        dtype=np.float64,
    )

    if velocity_world.shape != (2,):
        raise ValueError(
            "velocity_world_xy must have shape (2,), "
            f"received {velocity_world.shape}."
        )

    if not np.isfinite(velocity_world).all():
        raise ValueError(
            "velocity_world_xy must contain only finite values."
        )

    cosine_yaw = math.cos(robot_yaw)
    sine_yaw = math.sin(robot_yaw)

    return np.array(
        [
            (
                cosine_yaw * velocity_world[0]
                + sine_yaw * velocity_world[1]
            ),
            (
                -sine_yaw * velocity_world[0]
                + cosine_yaw * velocity_world[1]
            ),
        ],
        dtype=np.float64,
    )


def assign_proxy_velocity_to_points(
    obstacle_points: np.ndarray,
    *,
    robot_pose: RobotPose2D,
    obstacle_position_world: np.ndarray,
    elapsed_seconds: float,
) -> tuple[np.ndarray, int]:
    """Assign oracle proxy velocity to associated LiDAR obstacle points.

    This association is validation infrastructure rather than a velocity
    estimator. Points outside the known proxy neighborhood receive zero
    velocity and are therefore treated as stationary background geometry.
    """

    point_velocities = np.zeros_like(
        obstacle_points,
        dtype=np.float64,
    )

    if obstacle_points.shape[0] == 0:
        return point_velocities, 0

    expected_center = obstacle_center_in_robot_frame(
        robot_pose,
        obstacle_position_world,
    )

    if expected_center[0] < 0.0:
        return point_velocities, 0

    association_errors = np.linalg.norm(
        obstacle_points - expected_center,
        axis=1,
    )
    association_mask = (
        association_errors
        <= PROXY_ASSOCIATION_RADIUS
    )

    associated_point_count = int(
        np.count_nonzero(association_mask)
    )

    if associated_point_count == 0:
        return point_velocities, 0

    obstacle_velocity_robot = (
        world_velocity_to_robot_frame(
            obstacle_velocity_world_at_time(
                elapsed_seconds
            ),
            robot_yaw=robot_pose.yaw,
        )
    )

    point_velocities[association_mask] = (
        obstacle_velocity_robot
    )

    return point_velocities, associated_point_count


def proxy_association_error(
    obstacle_points: np.ndarray,
    *,
    robot_pose: RobotPose2D,
    obstacle_position_world: np.ndarray,
) -> float:
    """Return distance from the expected proxy center to a LiDAR point."""

    if obstacle_points.shape[0] == 0:
        return math.inf

    expected_center = obstacle_center_in_robot_frame(
        robot_pose,
        obstacle_position_world,
    )

    if expected_center[0] < 0.0:
        return math.inf

    errors = np.linalg.norm(
        obstacle_points
        - expected_center,
        axis=1,
    )

    return float(
        np.min(errors)
    )


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
    """Transform one scan and return CBF-compatible obstacle points."""

    points_robot_frame = transform_points(
        points_sensor_frame,
        robot_from_lidar,
    )

    return detect_obstacle_points(
        points_robot_frame,
        detection_config,
    )


def format_distance(
    distance: float,
) -> str:
    """Format a finite or unavailable distance."""

    if math.isfinite(distance):
        return f"{distance:.3f}"

    return "none"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_initial_report(
    *,
    pose: RobotPose2D,
    obstacle_position_world: np.ndarray,
    frame_info: LidarFrameInfo,
    obstacle_points: np.ndarray,
    velocity_point_count: int,
    proxy_error: float,
    nominal_command: WaypointCommand,
    static_cbf_result: CBFResult,
    velocity_cbf_result: CBFResult,
) -> None:
    """Print the stationary perception and two-filter dry-run report."""

    ground_truth_distance = (
        distance_to_obstacle_surface(
            pose,
            obstacle_position_world,
        )
    )

    proxy_detected = (
        proxy_error
        <= PROXY_ASSOCIATION_RADIUS
    )

    print("\n" + "=" * 70)
    print("VELOCITY-AWARE CBF STAGE A/B - STATIONARY DRY RUN")
    print("=" * 70)

    print("\nRobot:")
    print(
        "  Pose: "
        f"x={pose.x:.4f} m, "
        f"y={pose.y:.4f} m, "
        f"yaw={math.degrees(pose.yaw):.2f} degrees"
    )
    print(
        "  Goal: "
        f"x={ROBOT_GOAL_WORLD[0]:.3f} m, "
        f"y={ROBOT_GOAL_WORLD[1]:.3f} m"
    )

    print("\nMoving proxy:")
    print(
        "  Position: "
        f"x={obstacle_position_world[0]:.3f} m, "
        f"y={obstacle_position_world[1]:.3f} m"
    )
    print(
        "  Ground-truth surface distance: "
        f"{ground_truth_distance:.3f} m"
    )

    print("\nLiDAR:")
    print(f"  Frame ID:          {frame_info.sensor_frame_id}")
    print(f"  Obstacle points:   {obstacle_points.shape[0]}")
    print(f"  Velocity points:   {velocity_point_count}")
    print(
        "  Proxy association: "
        f"{proxy_detected}"
    )
    print(
        "  Association error: "
        f"{format_distance(proxy_error)} m"
    )
    print(
        "  Nearest CBF point: "
        f"{format_distance(static_cbf_result.minimum_obstacle_distance)} m"
    )

    print("\nNominal command:")
    print(
        "  Linear velocity:  "
        f"{nominal_command.linear_velocity:.3f} m/s"
    )
    print(
        "  Angular velocity: "
        f"{nominal_command.angular_velocity:.3f} rad/s"
    )

    print("\nStatic CBF result:")
    print(
        "  Safe velocity: "
        f"{static_cbf_result.safe_linear_velocity:.3f} m/s"
    )
    print(
        "  Intervention: "
        f"{static_cbf_result.intervention_active}"
    )
    print(
        "  Feasible:     "
        f"{static_cbf_result.constraint_feasible}"
    )
    print(
        "  Emergency:    "
        f"{static_cbf_result.emergency_stop_active}"
    )

    print("\nVelocity-aware CBF result:")
    print(
        "  Safe velocity: "
        f"{velocity_cbf_result.safe_linear_velocity:.3f} m/s"
    )
    print(
        "  Intervention: "
        f"{velocity_cbf_result.intervention_active}"
    )
    print(
        "  Feasible:     "
        f"{velocity_cbf_result.constraint_feasible}"
    )
    print(
        "  Emergency:    "
        f"{velocity_cbf_result.emergency_stop_active}"
    )

    print(
        "\nThe robot and proxy remained stationary during this report; "
        "the velocity-aware result uses the proxy's upcoming velocity."
    )
    print("=" * 70)


def print_live_status(
    *,
    completed_steps: int,
    elapsed_seconds: float,
    frame_id: int,
    obstacle_position_world: np.ndarray,
    ground_truth_distance: float,
    proxy_error: float,
    velocity_point_count: int,
    nominal_command: WaypointCommand,
    static_cbf_result: CBFResult,
    velocity_cbf_result: CBFResult,
) -> None:
    """Print one compact dynamic-experiment status line."""

    proxy_detected = (
        proxy_error
        <= PROXY_ASSOCIATION_RADIUS
    )

    print(
        f"  step={completed_steps:<4} "
        f"t={elapsed_seconds:5.2f} "
        f"scan={frame_id:<4} "
        f"proxy_y={obstacle_position_world[1]:5.2f} "
        f"goal_d={nominal_command.distance_to_waypoint:.3f} "
        f"gt_d={ground_truth_distance:.3f} "
        f"lidar_d="
        f"{format_distance(velocity_cbf_result.minimum_obstacle_distance):>5} "
        f"proxy={str(proxy_detected):<5} "
        f"vel_pts={velocity_point_count:<3} "
        f"v_nom={nominal_command.linear_velocity:.3f} "
        f"v_static={static_cbf_result.safe_linear_velocity:.3f} "
        f"v_velocity={velocity_cbf_result.safe_linear_velocity:.3f} "
        f"static_i={static_cbf_result.intervention_active} "
        f"velocity_i={velocity_cbf_result.intervention_active}"
    )


# ---------------------------------------------------------------------------
# Integrated dynamic-obstacle experiment
# ---------------------------------------------------------------------------


def run_dynamic_obstacle_experiment(
    *,
    stage: Usd.Stage,
    obstacle_translate_op: UsdGeom.XformOp,
    lidar: NovaCarterLidar,
    robot_controller: NovaCarterController,
    waypoint_controller: WaypointController,
    safety_filter: CBFSafetyFilter,
    detection_config: ObstacleDetectionConfig,
    robot_from_lidar: np.ndarray,
    initial_obstacle_points: np.ndarray,
    initial_obstacle_velocities: np.ndarray,
    initial_velocity_point_count: int,
    initial_frame_info: LidarFrameInfo,
    initial_static_cbf_result: CBFResult,
    initial_cbf_result: CBFResult,
) -> None:
    """Run the robot under velocity-aware CBF with static comparison."""

    completed_steps = 0
    received_scan_count = 0
    associated_scan_count = 0
    steps_since_new_scan = 0

    obstacle_position_world = (
        obstacle_position_at_time(0.0)
    )

    latest_pose = read_robot_pose(stage)
    latest_nominal_command = (
        waypoint_controller.compute_command(
            latest_pose
        )
    )

    latest_obstacle_points = (
        initial_obstacle_points
    )
    latest_obstacle_velocities = (
        initial_obstacle_velocities
    )
    latest_velocity_point_count = (
        initial_velocity_point_count
    )
    latest_frame_id = (
        initial_frame_info.sensor_frame_id
    )
    latest_static_cbf_result = (
        initial_static_cbf_result
    )
    latest_cbf_result = initial_cbf_result

    latest_proxy_error = proxy_association_error(
        latest_obstacle_points,
        robot_pose=latest_pose,
        obstacle_position_world=(
            obstacle_position_world
        ),
    )

    previous_intervention = (
        initial_cbf_result.intervention_active
    )
    previous_static_intervention = (
        initial_static_cbf_result.intervention_active
    )

    intervention_seen = (
        initial_cbf_result.intervention_active
    )
    static_intervention_seen = (
        initial_static_cbf_result.intervention_active
    )
    intervention_release_seen = False

    first_intervention_step: int | None = (
        0 if intervention_seen else None
    )
    first_static_intervention_step: int | None = (
        0 if static_intervention_seen else None
    )
    first_release_step: int | None = None

    proxy_detection_seen = (
        latest_proxy_error
        <= PROXY_ASSOCIATION_RADIUS
    )

    crossing_reported = False
    finish_reported = False
    safety_boundary_violated = False

    minimum_ground_truth_distance = (
        distance_to_obstacle_surface(
            latest_pose,
            obstacle_position_world,
        )
    )
    minimum_lidar_distance = (
        initial_cbf_result.minimum_obstacle_distance
    )
    minimum_proxy_error = latest_proxy_error

    termination_reason = (
        "Maximum experiment duration reached."
    )

    print("\n" + "=" * 70)
    print("VELOCITY-AWARE CBF STAGE C - CLOSED-LOOP EXPERIMENT")
    print("=" * 70)
    print(
        "Robot nominal speed: "
        f"{waypoint_controller.config.maximum_linear_velocity:.3f} m/s"
    )
    print(
        "Proxy speed:         "
        f"{OBSTACLE_SPEED_METERS_PER_SECOND:.3f} m/s"
    )
    print(
        "CBF safety radius:   "
        f"{safety_filter.config.safety_radius:.3f} m"
    )
    print("Applied controller:  velocity-aware CBF")
    print("Comparison filter:   static CBF (diagnostic only)")
    print(
        "Ground-truth abort:  "
        f"{GROUND_TRUTH_ABORT_DISTANCE:.3f} m"
    )
    print("\nLive status:")

    while (
        completed_steps
        < MAXIMUM_EXPERIMENT_STEPS
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

        latest_pose = read_robot_pose(stage)
        latest_nominal_command = (
            waypoint_controller.compute_command(
                latest_pose
            )
        )

        if latest_nominal_command.route_complete:
            termination_reason = (
                "The robot reached the final waypoint."
            )
            robot_controller.stop()
            break

        steps_since_new_scan += 1

        try:
            points_sensor_frame = (
                lidar.get_point_cloud()
            )
            current_frame_info = (
                lidar.get_frame_info()
            )
        except RuntimeError as error:
            is_transient = any(
                message in str(error)
                for message in TRANSIENT_LIDAR_ERRORS
            )

            if not is_transient:
                raise
        else:
            is_new_frame = (
                current_frame_info.sensor_frame_id
                != latest_frame_id
            )

            if is_new_frame:
                latest_obstacle_points = (
                    extract_obstacle_points(
                        points_sensor_frame,
                        robot_from_lidar=(
                            robot_from_lidar
                        ),
                        detection_config=(
                            detection_config
                        ),
                    )
                )

                if (
                    latest_obstacle_points.shape[0]
                    == 0
                ):
                    termination_reason = (
                        "No obstacle points were detected. "
                        "Fail-closed stop applied."
                    )
                    break

                latest_frame_id = (
                    current_frame_info.sensor_frame_id
                )
                steps_since_new_scan = 0
                received_scan_count += 1

                latest_proxy_error = (
                    proxy_association_error(
                        latest_obstacle_points,
                        robot_pose=latest_pose,
                        obstacle_position_world=(
                            obstacle_position_world
                        ),
                    )
                )

                (
                    latest_obstacle_velocities,
                    latest_velocity_point_count,
                ) = assign_proxy_velocity_to_points(
                    latest_obstacle_points,
                    robot_pose=latest_pose,
                    obstacle_position_world=(
                        obstacle_position_world
                    ),
                    elapsed_seconds=elapsed_seconds,
                )

                minimum_proxy_error = min(
                    minimum_proxy_error,
                    latest_proxy_error,
                )

                if (
                    latest_proxy_error
                    <= PROXY_ASSOCIATION_RADIUS
                ):
                    proxy_detection_seen = True
                    associated_scan_count += 1

        if (
            steps_since_new_scan
            >= MAXIMUM_STALE_SCAN_STEPS
        ):
            termination_reason = (
                "LiDAR data became stale. "
                "Fail-closed stop applied."
            )
            break

        # Both filters receive the same positions and nominal command.
        # The static result is diagnostic only. The applied filter additionally
        # receives oracle velocity for proxy-associated points.
        latest_static_cbf_result = (
            safety_filter.filter_command(
                nominal_linear_velocity=(
                    latest_nominal_command.linear_velocity
                ),
                nominal_angular_velocity=(
                    latest_nominal_command.angular_velocity
                ),
                obstacle_points_xy=(
                    latest_obstacle_points
                ),
            )
        )

        latest_cbf_result = (
            safety_filter.filter_command(
                nominal_linear_velocity=(
                    latest_nominal_command.linear_velocity
                ),
                nominal_angular_velocity=(
                    latest_nominal_command.angular_velocity
                ),
                obstacle_points_xy=(
                    latest_obstacle_points
                ),
                obstacle_velocities_xy=(
                    latest_obstacle_velocities
                ),
            )
        )

        ground_truth_distance = (
            distance_to_obstacle_surface(
                latest_pose,
                obstacle_position_world,
            )
        )

        minimum_ground_truth_distance = min(
            minimum_ground_truth_distance,
            ground_truth_distance,
        )
        minimum_lidar_distance = min(
            minimum_lidar_distance,
            latest_cbf_result.minimum_obstacle_distance,
        )

        if (
            ground_truth_distance
            < CBF_SAFETY_RADIUS
        ):
            safety_boundary_violated = True

        if (
            latest_cbf_result.intervention_active
            and not intervention_seen
        ):
            intervention_seen = True
            first_intervention_step = (
                completed_steps
            )

        if (
            latest_static_cbf_result.intervention_active
            and not static_intervention_seen
        ):
            static_intervention_seen = True
            first_static_intervention_step = (
                completed_steps
            )

        intervention_released_now = (
            intervention_seen
            and previous_intervention
            and not (
                latest_cbf_result.intervention_active
            )
        )

        if intervention_released_now:
            intervention_release_seen = True

            if first_release_step is None:
                first_release_step = (
                    completed_steps
                )

        proxy_crossed_now = (
            not crossing_reported
            and obstacle_position_world[1]
            <= ROBOT_GOAL_WORLD[1]
        )

        if proxy_crossed_now:
            crossing_reported = True
            print(
                "\nProxy reached the robot route centerline."
            )

        proxy_finished_now = (
            not finish_reported
            and obstacle_has_finished(
                obstacle_position_world
            )
        )

        if proxy_finished_now:
            finish_reported = True
            print(
                "\nProxy reached the end of its trajectory."
            )

        emergency_stop = (
            not latest_cbf_result.constraint_feasible
            or latest_cbf_result.emergency_stop_active
        )

        ground_truth_abort = (
            ground_truth_distance
            <= GROUND_TRUTH_ABORT_DISTANCE
        )

        should_report = (
            completed_steps
            % STATUS_REPORT_INTERVAL_STEPS
            == 0
            or (
                latest_cbf_result.intervention_active
                != previous_intervention
            )
            or (
                latest_static_cbf_result.intervention_active
                != previous_static_intervention
            )
            or proxy_crossed_now
            or proxy_finished_now
            or emergency_stop
            or ground_truth_abort
        )

        if should_report:
            print_live_status(
                completed_steps=completed_steps,
                elapsed_seconds=elapsed_seconds,
                frame_id=latest_frame_id,
                obstacle_position_world=(
                    obstacle_position_world
                ),
                ground_truth_distance=(
                    ground_truth_distance
                ),
                proxy_error=latest_proxy_error,
                velocity_point_count=(
                    latest_velocity_point_count
                ),
                nominal_command=(
                    latest_nominal_command
                ),
                static_cbf_result=(
                    latest_static_cbf_result
                ),
                velocity_cbf_result=latest_cbf_result,
            )

        if emergency_stop:
            termination_reason = (
                "The velocity-aware CBF constraints became "
                "infeasible. Emergency stop applied."
            )
            break

        if ground_truth_abort:
            termination_reason = (
                "Ground-truth separation reached the "
                "experiment abort distance."
            )
            break

        robot_controller.apply_command(
            latest_cbf_result.safe_linear_velocity,
            latest_cbf_result.safe_angular_velocity,
        )

        previous_intervention = (
            latest_cbf_result.intervention_active
        )
        previous_static_intervention = (
            latest_static_cbf_result.intervention_active
        )

        simulation_app.update()
        completed_steps += 1

    robot_controller.stop()
    app_utils.update_app(steps=5)

    if simulation_app.is_running():
        latest_pose = read_robot_pose(stage)
        latest_nominal_command = (
            waypoint_controller.compute_command(
                latest_pose
            )
        )

    print("\n" + "=" * 70)
    print("VELOCITY-AWARE CBF COMPARISON RESULT")
    print("=" * 70)
    print(f"Reason: {termination_reason}")
    print(f"Completed physics steps: {completed_steps}")
    print(f"New LiDAR frames:        {received_scan_count}")
    print(f"Proxy-associated scans:  {associated_scan_count}")
    print(
        "Proxy detected:           "
        f"{proxy_detection_seen}"
    )
    print(
        "Velocity-aware intervention seen: "
        f"{intervention_seen}"
    )
    print(
        "Velocity-aware release seen:      "
        f"{intervention_release_seen}"
    )
    print(
        "Static intervention seen:         "
        f"{static_intervention_seen}"
    )

    if first_intervention_step is None:
        print("First velocity-aware intervention: none")
    else:
        print(
            "First velocity-aware intervention: "
            f"{first_intervention_step}"
        )

    if first_static_intervention_step is None:
        print("First static intervention step:   none")
    else:
        print(
            "First static intervention step:   "
            f"{first_static_intervention_step}"
        )

    velocity_intervened_earlier = (
        first_intervention_step is not None
        and (
            first_static_intervention_step is None
            or first_intervention_step
            < first_static_intervention_step
        )
    )

    print(
        "Velocity-aware intervened earlier: "
        f"{velocity_intervened_earlier}"
    )

    if first_release_step is None:
        print("First release step:       none")
    else:
        print(
            "First release step:       "
            f"{first_release_step}"
        )

    print(
        "Minimum ground-truth distance: "
        f"{minimum_ground_truth_distance:.4f} m"
    )
    print(
        "Minimum LiDAR/CBF distance:     "
        f"{format_distance(minimum_lidar_distance)} m"
    )
    print(
        "Minimum proxy association error:"
        f" {format_distance(minimum_proxy_error)} m"
    )
    print(
        "Safety boundary violated:       "
        f"{safety_boundary_violated}"
    )
    print(
        "Route complete:                 "
        f"{waypoint_controller.route_complete}"
    )

    print("\nFinal robot pose:")
    print(f"  x:   {latest_pose.x:.4f} m")
    print(f"  y:   {latest_pose.y:.4f} m")
    print(
        "  yaw: "
        f"{math.degrees(latest_pose.yaw):.2f} degrees"
    )

    print("\nInterpretation:")

    if safety_boundary_violated:
        print(
            "  [FAIL] The velocity-aware CBF did not "
            "maintain the ground-truth safety radius against the "
            "moving obstacle."
        )
    elif (
        waypoint_controller.route_complete
        and intervention_seen
        and intervention_release_seen
        and velocity_intervened_earlier
    ):
        print(
            "  [PASS] The velocity-aware CBF intervened before the "
            "static comparison, released after the proxy cleared, "
            "and completed the route."
        )
    elif (
        intervention_seen
        and not waypoint_controller.route_complete
    ):
        print(
            "  [PARTIAL] The velocity-aware CBF intervened safely, "
            "but the route "
            "did not complete. Inspect the termination reason."
        )
    elif waypoint_controller.route_complete:
        print(
            "  [WARNING] The route completed, but the expected earlier "
            "velocity-aware intervention was not demonstrated."
        )
    else:
        print(
            "  [WARNING] The experiment did not produce the expected "
            "intervention-and-recovery sequence."
        )

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the controlled velocity-aware CBF comparison."""

    print("\n" + "=" * 70)
    print("NOVA CARTER VELOCITY-AWARE CBF DEMO")
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

    obstacle_translate_op = (
        create_moving_obstacle(
            loaded_scene.stage
        )
    )

    robot_controller = NovaCarterController(
        robot_prim_path=ROBOT_PRIM_PATH
    )

    SimulationManager.setup_simulation(
        dt=PHYSICS_DT,
        device=PHYSICS_DEVICE,
    )

    print("\nPress Play in Isaac Sim to begin.")

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

    initial_pose = read_robot_pose(
        loaded_scene.stage
    )

    waypoint_controller = WaypointController(
        [
            Waypoint(
                x=ROBOT_GOAL_WORLD[0],
                y=ROBOT_GOAL_WORLD[1],
            )
        ],
        config=WaypointControllerConfig(
            maximum_linear_velocity=0.20,
            maximum_angular_velocity=0.60,
            linear_gain=0.80,
            angular_gain=1.50,
            position_tolerance=0.10,
            rotate_in_place_threshold_radians=(
                math.radians(25.0)
            ),
        ),
    )

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

    print("\nWaiting for a valid LiDAR scan...")

    (
        points_sensor_frame,
        initial_frame_info,
        scan_wait_steps,
    ) = wait_for_first_scan(
        lidar,
        robot_controller,
    )

    initial_obstacle_points = (
        extract_obstacle_points(
            points_sensor_frame,
            robot_from_lidar=robot_from_lidar,
            detection_config=detection_config,
        )
    )

    if initial_obstacle_points.shape[0] == 0:
        raise RuntimeError(
            "The initial scan produced no obstacle points."
        )

    print(
        "Valid scan received after "
        f"{scan_wait_steps} update(s)."
    )

    safety_filter = CBFSafetyFilter(
        CBFConfig(
            safety_radius=CBF_SAFETY_RADIUS,
            alpha=CBF_ALPHA,
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=(
                robot_controller.config.max_linear_speed
            ),
        )
    )

    initial_nominal_command = (
        waypoint_controller.compute_command(
            initial_pose
        )
    )

    initial_static_cbf_result = (
        safety_filter.filter_command(
            nominal_linear_velocity=(
                initial_nominal_command.linear_velocity
            ),
            nominal_angular_velocity=(
                initial_nominal_command.angular_velocity
            ),
            obstacle_points_xy=(
                initial_obstacle_points
            ),
        )
    )

    initial_obstacle_position = (
        obstacle_position_at_time(0.0)
    )

    (
        initial_obstacle_velocities,
        initial_velocity_point_count,
    ) = assign_proxy_velocity_to_points(
        initial_obstacle_points,
        robot_pose=initial_pose,
        obstacle_position_world=(
            initial_obstacle_position
        ),
        elapsed_seconds=0.0,
    )

    initial_cbf_result = (
        safety_filter.filter_command(
            nominal_linear_velocity=(
                initial_nominal_command.linear_velocity
            ),
            nominal_angular_velocity=(
                initial_nominal_command.angular_velocity
            ),
            obstacle_points_xy=(
                initial_obstacle_points
            ),
            obstacle_velocities_xy=(
                initial_obstacle_velocities
            ),
        )
    )

    initial_proxy_error = (
        proxy_association_error(
            initial_obstacle_points,
            robot_pose=initial_pose,
            obstacle_position_world=(
                initial_obstacle_position
            ),
        )
    )

    print_initial_report(
        pose=initial_pose,
        obstacle_position_world=(
            initial_obstacle_position
        ),
        frame_info=initial_frame_info,
        obstacle_points=initial_obstacle_points,
        velocity_point_count=(
            initial_velocity_point_count
        ),
        proxy_error=initial_proxy_error,
        nominal_command=initial_nominal_command,
        static_cbf_result=(
            initial_static_cbf_result
        ),
        velocity_cbf_result=initial_cbf_result,
    )

    print(
        "\nThe dynamic experiment begins after "
        f"{MOTION_START_DELAY_STEPS * PHYSICS_DT:.1f} "
        "simulation seconds..."
    )

    if not advance_playing_steps(
        simulation_app,
        physics_steps=MOTION_START_DELAY_STEPS,
        on_step=robot_controller.stop,
    ):
        return

    run_dynamic_obstacle_experiment(
        stage=loaded_scene.stage,
        obstacle_translate_op=(
            obstacle_translate_op
        ),
        lidar=lidar,
        robot_controller=robot_controller,
        waypoint_controller=(
            waypoint_controller
        ),
        safety_filter=safety_filter,
        detection_config=detection_config,
        robot_from_lidar=robot_from_lidar,
        initial_obstacle_points=(
            initial_obstacle_points
        ),
        initial_obstacle_velocities=(
            initial_obstacle_velocities
        ),
        initial_velocity_point_count=(
            initial_velocity_point_count
        ),
        initial_frame_info=initial_frame_info,
        initial_static_cbf_result=(
            initial_static_cbf_result
        ),
        initial_cbf_result=initial_cbf_result,
    )

    robot_controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()

    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when finished.")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 70)
        print("ERROR DURING VELOCITY-AWARE CBF DEMO")
        print("=" * 70)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
