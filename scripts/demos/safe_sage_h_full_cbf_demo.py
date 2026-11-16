"""Validate the full-state Safe-SAGE CBF pipeline without robot motion.

This controlled Isaac Sim demo uses one session-only cube as a semantic
proxy. The cube is sensed by the real PandarXT-32 pipeline; its known world
pose is used only to select the corresponding LiDAR track and to report an
independent distance check.

Stage A - perception and configuration-space geometry:

1. Load the restaurant robot scene.
2. Position Nova Carter in the spacious test area.
3. Create a static proxy directly ahead of the robot.
4. Capture one LiDAR scan and associate the proxy with one tracked cluster.
5. Rasterize only that track into an uninflated local occupancy grid.
6. Expand the grid over yaw using Nova Carter's measured rectangular
   collision footprint and a separate 0.05 m policy margin.

Stage B - semantic three-dimensional safety field:

1. Assign the manual ``person`` label to the proxy track.
2. Apply the configured Safe-SAGE risk, normal-flux, tangential-strength,
   and passing-side parameters.
3. Solve the semantic guidance field at every yaw slice.
4. Construct ``h_full(x, y, psi)`` and its spatial and yaw derivatives.

Stage C - stationary full-state CBF evaluation:

1. Generate one nominal straight-ahead waypoint command.
2. Query the robot-local field at ``(x, y, psi) = (0, 0, 0)``.
3. Evaluate

       (h_x cos(psi) + h_y sin(psi)) v
           + h_psi omega + h_t + gamma h >= 0.

4. Project the nominal command onto the actuator-limited safe set.
5. Print the filtered command and complete inequality diagnostics.

Stage C2 - stationary forward counterfactual sweep:

1. Reuse the same static ``h_full`` field and nominal command.
2. Evaluate hypothetical robot positions at fixed increments along ``+X``.
3. Stop at the first yaw-specific configuration-space boundary.
4. Verify that a feasible intervention occurs before that boundary and that
   every preceding safe command satisfies the full-state CBF inequality.

The obstacle is static, so ``h_t`` is explicitly zero. No nominal or
filtered command is sent to the wheels; the robot remains stationary during
all three stages. Camera/YOLO labels, temporal field estimation, and
closed-loop motion are deliberately outside this validation milestone.

Run from the project root:

    python -u scripts/demos/safe_sage_h_full_cbf_demo.py

Press Play in Isaac Sim when requested.
"""

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


RTX_HISTORY_TRANSFORM_COUNT = int(os.environ.get("ISAAC_RTX_HISTORY_TRANSFORM_COUNT", "20"))
if not 2 <= RTX_HISTORY_TRANSFORM_COUNT <= 256:
    raise ValueError("RTX history transform count must lie within [2, 256]")
SIMULATION_HEADLESS = os.environ.get(
    "ISAAC_SIM_HEADLESS",
    "0",
).strip().casefold() in {"1", "true", "yes", "on"}

simulation_app = SimulationApp(
    {
        "headless": SIMULATION_HEADLESS,
        "enable_motion_bvh": True,
        "extra_args": [
            *(["--enable", "omni.anim.behavior.core"]
              if os.environ.get("CRA_NATIVE_HUMAN") == "1" else []),
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
from robot.safety.class_safety_policy import ClassSafetyPolicy
from robot.safety.local_occupancy_grid import (
    LocalOccupancyGrid,
    LocalOccupancyGridConfig,
    build_local_occupancy_grid,
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
from robot.safety.poisson_safety_function import (
    PoissonSafetyFunctionConfig,
)
from robot.safety.safe_sage_cbf_filter import (
    SafeSAGECBFConfig,
    SafeSAGECBFSafetyFilter,
)
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
    SafeSAGEConfigurationSpaceConfig,
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
from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
from robot.safety.semantic_obstacle import SemanticObstacle
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
)
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import (
    compute_relative_prim_transform,
    get_prim_planar_pose,
    get_prim_world_transform,
)
from robot.sensor.lidar import LidarFrameInfo, NovaCarterLidar
from robot.sensor.lidar_geometry import transform_points


# ---------------------------------------------------------------------------
# Controlled scenario configuration
# ---------------------------------------------------------------------------


GUIDANCE_POLICY_PATH = (
    PROJECT_ROOT / "configs" / "safe_sage_class_safety.json"
)

STATIC_PROXY_PRIM_PATH = "/World/SafeSAGEHFullCBFTestObstacle"
SEMANTIC_CLASS = "person"

ROBOT_START_POSITION_WORLD = (
    26.4955,
    5.00175,
    0.07382,
)
ROBOT_START_YAW_RADIANS = 0.0
ROBOT_GOAL_WORLD = (
    29.0,
    5.00175,
)

# The nearest proxy face begins about 0.70 m ahead of the chassis reference.
STATIC_PROXY_POSITION_WORLD = np.array(
    [
        27.35,
        5.00175,
        0.50,
    ],
    dtype=np.float64,
)
PROXY_DIMENSIONS = (
    0.30,
    0.30,
    1.00,
)
PROXY_COLOR = (
    0.20,
    0.45,
    1.00,
)
PROXY_ASSOCIATION_RADIUS = 0.60

SETTLE_STEPS = 60
MAXIMUM_SCAN_WAIT_STEPS = 300

MINIMUM_HEIGHT_ABOVE_FLOOR = 0.05
MAXIMUM_HEIGHT_ABOVE_FLOOR = 1.00
MINIMUM_DETECTION_RANGE = 0.15
MAXIMUM_DETECTION_RANGE = 5.00
ANGULAR_BIN_SIZE_DEGREES = 2.0

TRACKER_CLUSTER_RADIUS = 0.35
TRACKER_MINIMUM_CLUSTER_POINTS = 2
TRACKER_MAXIMUM_ASSOCIATION_DISTANCE = 0.60

# The raw grid deliberately has no point-obstacle inflation. The measured
# robot footprint is applied exactly once in configuration space below.
GRID_RESOLUTION = 0.20
GRID_MINIMUM_X = -1.0
GRID_MAXIMUM_X = 5.0
GRID_MINIMUM_Y = -3.0
GRID_MAXIMUM_Y = 3.0
GRID_OBSTACLE_INFLATION_RADIUS = 0.0

# Measured from enabled robot-scale collision bounds in the chassis frame.
ROBOT_FRONT_EXTENT = 0.140000
ROBOT_REAR_EXTENT = 0.587351
ROBOT_LEFT_EXTENT = 0.246600
ROBOT_RIGHT_EXTENT = 0.246600
ROBOT_SAFETY_MARGIN = 0.050000

YAW_SLICE_COUNT = 10
LAPLACE_CONVERGENCE_TOLERANCE = 1e-6
TANGENTIAL_BUFFER_RADIUS = 0.60
H_FULL_SOFTPLUS_BETA = 1.0
H_FULL_MINIMUM_FORCING_MAGNITUDE = 1e-9
H_FULL_POISSON_CONVERGENCE_TOLERANCE = 1e-6

CBF_GAMMA = 1.0
CBF_LINEAR_VELOCITY_WEIGHT = 1.0
CBF_ANGULAR_VELOCITY_WEIGHT = 1.0
UNICYCLE_LOOKAHEAD_DISTANCE = 0.10
STATIC_TEMPORAL_SAFETY_DERIVATIVE = 0.0

FORWARD_PROBE_SPACING = 0.05
FORWARD_PROBE_MAXIMUM_X = 1.50

CURRENT_LOOKAHEAD_POSITION_GRID_XY = np.array(
    [UNICYCLE_LOOKAHEAD_DISTANCE, 0.0],
    dtype=np.float64,
)
CURRENT_CONFIGURATION_YAW_RADIANS = 0.0

MAXIMUM_STATIONARY_POSITION_DRIFT = 0.02
MAXIMUM_STATIONARY_YAW_DRIFT_RADIANS = math.radians(1.0)

TRANSIENT_LIDAR_ERRORS = (
    "No LiDAR data is available yet",
    "contains no points",
    "no valid XYZ points remained",
)


def yaw_to_quaternion_wxyz(yaw_radians: float) -> list[float]:
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
    """Teleport the initialized articulation to the controlled start pose."""

    robot_controller.robot.set_world_poses(
        positions=[ROBOT_START_POSITION_WORLD],
        orientations=[
            yaw_to_quaternion_wxyz(ROBOT_START_YAW_RADIANS)
        ],
    )
    robot_controller.stop()
    app_utils.update_app(steps=5)


def create_static_proxy(stage: Usd.Stage) -> None:
    """Create the session-only semantic proxy obstacle."""

    existing_prim = stage.GetPrimAtPath(STATIC_PROXY_PRIM_PATH)
    if existing_prim.IsValid():
        raise RuntimeError(
            "The h_full CBF test proxy already exists: "
            f"{STATIC_PROXY_PRIM_PATH}"
        )

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        cube = UsdGeom.Cube.Define(stage, STATIC_PROXY_PRIM_PATH)
        cube.CreateSizeAttr(1.0)
        cube.CreateExtentAttr(
            [
                Gf.Vec3f(-0.5, -0.5, -0.5),
                Gf.Vec3f(0.5, 0.5, 0.5),
            ]
        )
        cube.CreateDisplayColorAttr([Gf.Vec3f(*PROXY_COLOR)])
        cube.AddTranslateOp().Set(
            Gf.Vec3d(
                float(STATIC_PROXY_POSITION_WORLD[0]),
                float(STATIC_PROXY_POSITION_WORLD[1]),
                float(STATIC_PROXY_POSITION_WORLD[2]),
            )
        )
        cube.AddOrientOp().Set(Gf.Quatf(1.0))
        cube.AddScaleOp().Set(Gf.Vec3f(*PROXY_DIMENSIONS))


def read_robot_pose(stage: Usd.Stage) -> RobotPose2D:
    """Read the chassis planar pose from the stage."""

    x_position, y_position, yaw = get_prim_planar_pose(
        stage,
        ROBOT_BASE_PRIM_PATH,
    )
    return RobotPose2D(
        x=x_position,
        y=y_position,
        yaw=yaw,
    )


def verify_robot_start_pose(stage: Usd.Stage) -> RobotPose2D:
    """Validate that the articulation teleport reached the intended pose."""

    pose = read_robot_pose(stage)
    position_error = math.hypot(
        pose.x - ROBOT_START_POSITION_WORLD[0],
        pose.y - ROBOT_START_POSITION_WORLD[1],
    )
    yaw_error = absolute_wrapped_angle_difference(
        pose.yaw,
        ROBOT_START_YAW_RADIANS,
    )

    if position_error > 0.10:
        raise RuntimeError(
            "The robot did not reach the requested experiment position."
        )
    if yaw_error > math.radians(5.0):
        raise RuntimeError(
            "The robot did not reach the requested experiment yaw."
        )
    return pose


def absolute_wrapped_angle_difference(
    first: float,
    second: float,
) -> float:
    """Return the absolute shortest angular separation."""

    return abs((first - second + math.pi) % math.tau - math.pi)


def estimate_floor_height_in_robot_frame(stage: Usd.Stage) -> float:
    """Estimate world-Z-zero height in the current chassis frame."""

    robot_to_world = get_prim_world_transform(
        stage,
        ROBOT_BASE_PRIM_PATH,
    )
    robot_origin_world = robot_to_world.ExtractTranslation()
    ground_point_world = Gf.Vec3d(
        robot_origin_world[0],
        robot_origin_world[1],
        0.0,
    )
    ground_point_robot = robot_to_world.GetInverse().Transform(
        ground_point_world
    )
    return float(ground_point_robot[2])


def proxy_center_in_robot_frame(pose: RobotPose2D) -> np.ndarray:
    """Express the known proxy center in the current chassis frame."""

    x_world_error = STATIC_PROXY_POSITION_WORLD[0] - pose.x
    y_world_error = STATIC_PROXY_POSITION_WORLD[1] - pose.y
    cosine_yaw = math.cos(pose.yaw)
    sine_yaw = math.sin(pose.yaw)
    return np.array(
        [
            cosine_yaw * x_world_error + sine_yaw * y_world_error,
            -sine_yaw * x_world_error + cosine_yaw * y_world_error,
        ],
        dtype=np.float64,
    )


def distance_to_proxy_surface(pose: RobotPose2D) -> float:
    """Return the planar chassis-reference distance to the proxy surface."""

    half_x = 0.5 * PROXY_DIMENSIONS[0]
    half_y = 0.5 * PROXY_DIMENSIONS[1]
    x_separation = max(
        abs(pose.x - STATIC_PROXY_POSITION_WORLD[0]) - half_x,
        0.0,
    )
    y_separation = max(
        abs(pose.y - STATIC_PROXY_POSITION_WORLD[1]) - half_y,
        0.0,
    )
    return math.hypot(x_separation, y_separation)


def wait_for_first_scan(
    lidar: NovaCarterLidar,
    robot_controller: NovaCarterController,
) -> tuple[np.ndarray, LidarFrameInfo, int]:
    """Advance until the first valid PandarXT-32 scan arrives."""

    last_transient_error: RuntimeError | None = None
    for completed_steps in range(1, MAXIMUM_SCAN_WAIT_STEPS + 1):
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

        return points, lidar.get_frame_info(), completed_steps

    detail = (
        f" Last LiDAR error: {last_transient_error}"
        if last_transient_error is not None
        else ""
    )
    raise TimeoutError(
        "No valid LiDAR scan arrived after "
        f"{MAXIMUM_SCAN_WAIT_STEPS} updates.{detail}"
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


def find_proxy_track(
    tracking_result: ObstacleTrackingResult,
    *,
    robot_pose: RobotPose2D,
) -> tuple[TrackedObstacle | None, float]:
    """Associate the known proxy center with the nearest LiDAR track."""

    if not tracking_result.tracked_obstacles:
        return None, math.inf

    expected_center = proxy_center_in_robot_frame(robot_pose)
    best_track: TrackedObstacle | None = None
    best_error = math.inf

    for tracked_obstacle in tracking_result.tracked_obstacles:
        centroid = np.asarray(
            tracked_obstacle.centroid_robot_xy,
            dtype=np.float64,
        )
        error = float(np.linalg.norm(centroid - expected_center))
        if error < best_error:
            best_track = tracked_obstacle
            best_error = error

    if best_error > PROXY_ASSOCIATION_RADIUS:
        return None, best_error
    return best_track, best_error


def class_policy_from_guidance_policy(
    guidance_policy: SafeSAGEGuidancePolicy,
) -> ClassSafetyPolicy:
    """Expose the guidance policy's shared risk and normal-flux values."""

    return ClassSafetyPolicy(
        {
            semantic_class: parameters.safety_parameters
            for semantic_class, parameters
            in guidance_policy.class_parameters.items()
        },
        unknown_class=guidance_policy.unknown_class,
        risk_flux_mapping=guidance_policy.risk_flux_mapping,
    )


def create_waypoint_controller() -> WaypointController:
    """Create the nominal straight-ahead waypoint controller."""

    return WaypointController(
        [Waypoint(x=ROBOT_GOAL_WORLD[0], y=ROBOT_GOAL_WORLD[1])],
        config=WaypointControllerConfig(
            maximum_linear_velocity=0.20,
            maximum_angular_velocity=0.60,
            linear_gain=0.80,
            angular_gain=1.50,
            position_tolerance=0.10,
            rotate_in_place_threshold_radians=math.radians(25.0),
        ),
    )


def create_field_pipeline_config(
    footprint: RectangularRobotFootprint,
) -> SafeSAGEFieldPipelineConfig:
    """Return the shared configuration-space and PDE configuration."""

    return SafeSAGEFieldPipelineConfig(
        configuration_space_config=(
            SafeSAGEConfigurationSpaceConfig(
                footprint=footprint,
                yaw_slice_count=YAW_SLICE_COUNT,
            )
        ),
        guidance_field_config=(
            TangentialLaplaceGuidanceFieldConfig(
                convergence_tolerance=(
                    LAPLACE_CONVERGENCE_TOLERANCE
                ),
                tangential_buffer_radius=TANGENTIAL_BUFFER_RADIUS,
            )
        ),
        h_full_config=SafeSAGEHFullConfig(
            yaw_slice_count=YAW_SLICE_COUNT,
            softplus_beta=H_FULL_SOFTPLUS_BETA,
            minimum_forcing_magnitude=(
                H_FULL_MINIMUM_FORCING_MAGNITUDE
            ),
            poisson_config=PoissonSafetyFunctionConfig(
                convergence_tolerance=(
                    H_FULL_POISSON_CONVERGENCE_TOLERANCE
                ),
            ),
        ),
    )


def print_stage_a_report(
    *,
    robot_pose: RobotPose2D,
    frame_info: LidarFrameInfo,
    scan_wait_steps: int,
    obstacle_points: np.ndarray,
    tracking_result: ObstacleTrackingResult,
    proxy_track: TrackedObstacle,
    proxy_association_error: float,
    proxy_points: np.ndarray,
    source_grid: LocalOccupancyGrid,
    pipeline: SafeSAGEFieldPipelineResult,
    footprint: RectangularRobotFootprint,
) -> None:
    """Print perception and yaw-dependent footprint diagnostics."""

    nearest_proxy_distance = float(
        np.min(np.linalg.norm(proxy_points, axis=1))
    )
    known_surface_distance = distance_to_proxy_surface(robot_pose)
    footprint_counts = pipeline.configuration_space.footprint_cell_counts
    occupied_counts = np.array(
        [
            grid.occupied_cell_count
            for grid in pipeline.configuration_space.occupancy_grids
        ],
        dtype=np.int64,
    )

    print("\n" + "=" * 70)
    print("SAFE-SAGE H_FULL STAGE A - PERCEPTION AND CONFIGURATION SPACE")
    print("=" * 70)

    print("\nRobot pose:")
    print(f"  x:                           {robot_pose.x:.4f} m")
    print(f"  y:                           {robot_pose.y:.4f} m")
    print(
        "  world yaw:                   "
        f"{math.degrees(robot_pose.yaw):.2f} degrees"
    )

    print("\nLiDAR and controlled tracking:")
    print(f"  Frame ID:                     {frame_info.sensor_frame_id}")
    print(f"  Waited updates:               {scan_wait_steps}")
    print(f"  All obstacle points:          {obstacle_points.shape[0]}")
    print(f"  Detected clusters:            {tracking_result.cluster_count}")
    print(f"  Proxy track ID:               {proxy_track.track_id}")
    print(f"  Proxy track points:           {proxy_points.shape[0]}")
    print(
        "  Proxy association error:    "
        f"{proxy_association_error:.4f} m"
    )
    print(
        "  Nearest proxy return:       "
        f"{nearest_proxy_distance:.4f} m"
    )
    print(
        "  Known surface distance:     "
        f"{known_surface_distance:.4f} m"
    )

    print("\nRaw local occupancy grid:")
    print(f"  Shape:                        {source_grid.shape}")
    print(
        "  Resolution:                  "
        f"{source_grid.config.resolution:.3f} m"
    )
    print(
        "  Circular point inflation:    "
        f"{source_grid.config.obstacle_inflation_radius:.3f} m"
    )
    print(
        "  Occupied cells:              "
        f"{source_grid.occupied_cell_count}"
    )

    print("\nMeasured chassis-referenced Nova Carter footprint:")
    print(f"  Front extent (+X):            {ROBOT_FRONT_EXTENT:.6f} m")
    print(f"  Rear extent (-X):             {ROBOT_REAR_EXTENT:.6f} m")
    print(f"  Left extent (+Y):             {ROBOT_LEFT_EXTENT:.6f} m")
    print(f"  Right extent (-Y):            {ROBOT_RIGHT_EXTENT:.6f} m")

    print("\nEquivalent look-ahead-referenced footprint:")
    print(
        "  Look-ahead distance:          "
        f"{UNICYCLE_LOOKAHEAD_DISTANCE:.6f} m"
    )
    print(f"  Front extent (+X):            {footprint.front_extent:.6f} m")
    print(f"  Rear extent (-X):             {footprint.rear_extent:.6f} m")
    print(f"  Left extent (+Y):             {footprint.left_extent:.6f} m")
    print(f"  Right extent (-Y):            {footprint.right_extent:.6f} m")
    print(f"  Safety margin:                {footprint.safety_margin:.6f} m")

    print("\nYaw-dependent configuration space:")
    print(f"  Yaw slices:                   {pipeline.yaw_slice_count}")
    print(
        "  Yaw spacing:                 "
        f"{math.degrees(math.tau / pipeline.yaw_slice_count):.2f} degrees"
    )
    print(
        "  Footprint raster cells:      "
        f"[{int(np.min(footprint_counts))}, "
        f"{int(np.max(footprint_counts))}]"
    )
    print(
        "  Expanded occupied cells:     "
        f"[{int(np.min(occupied_counts))}, "
        f"{int(np.max(occupied_counts))}]"
    )

    print("\nControlled-data boundary:")
    print("  [PASS] A real LiDAR track was associated with the proxy.")
    print("  Only proxy-track points entered the raw occupancy grid.")
    print("  The known proxy pose was not passed to the safety pipeline.")
    print("  Footprint expansion was applied exactly once.")
    print("  The robot remained stationary during Stage A.")
    print("=" * 70)


def print_stage_b_report(
    *,
    guidance_policy: SafeSAGEGuidancePolicy,
    pipeline: SafeSAGEFieldPipelineResult,
) -> None:
    """Print semantic guidance and h_full solver diagnostics."""

    parameters = guidance_policy.parameters_for(SEMANTIC_CLASS)
    guidance_iterations = np.asarray(
        pipeline.guidance_iteration_counts,
        dtype=np.int64,
    )
    tangential_counts = np.asarray(
        pipeline.tangential_interface_cell_counts,
        dtype=np.int64,
    )
    guidance_residuals = np.asarray(
        [field.maximum_residual for field in pipeline.guidance_fields],
        dtype=np.float64,
    )
    h_full = pipeline.h_full

    print("\n" + "=" * 70)
    print("SAFE-SAGE H_FULL STAGE B - SEMANTIC 3D FIELD")
    print("=" * 70)

    print("\nSemantic guidance parameters:")
    print(f"  Semantic class:               {SEMANTIC_CLASS}")
    print(f"  Manual confidence:            {1.0:.3f}")
    print(f"  Risk:                         {parameters.risk:.3f}")
    print(
        "  Normal boundary flux b:      "
        f"{parameters.boundary_flux:.3f}"
    )
    print(
        "  Tangential strength lambda:  "
        f"{parameters.tangential_strength:.3f}"
    )
    print(
        "  Passing side:                "
        f"{parameters.passing_side.value}"
    )

    print("\nYaw-indexed semantic guidance solves:")
    print(
        "  Iteration range:              "
        f"[{int(np.min(guidance_iterations))}, "
        f"{int(np.max(guidance_iterations))}]"
    )
    print(
        "  Maximum residual:             "
        f"{float(np.max(guidance_residuals)):.3e}"
    )
    print(
        "  Tangential interface cells:  "
        f"[{int(np.min(tangential_counts))}, "
        f"{int(np.max(tangential_counts))}]"
    )

    print("\nFull-state Poisson safety field:")
    print(f"  Shape (yaw, rows, columns):   {h_full.shape}")
    print(
        "  Iteration range:              "
        f"[{int(np.min(h_full.iteration_counts))}, "
        f"{int(np.max(h_full.iteration_counts))}]"
    )
    print(
        "  Maximum residual:             "
        f"{float(np.max(h_full.maximum_residuals)):.3e}"
    )
    print(
        "  Safety-value range:           "
        f"[{float(np.min(h_full.values)):.6f}, "
        f"{float(np.max(h_full.values)):.6f}]"
    )
    print(
        "  Forcing range:                "
        f"[{float(np.min(h_full.forcing_fields)):.6f}, "
        f"{float(np.max(h_full.forcing_fields)):.6f}]"
    )

    semantic_checks = (
        pipeline.semantic_obstacle_count == 1,
        np.all(tangential_counts > 0),
        np.isfinite(h_full.values).all(),
        np.isfinite(h_full.spatial_gradients_robot_xy).all(),
        np.isfinite(h_full.yaw_derivatives).all(),
    )

    print("\nField checks:")
    print(
        "  One semantic obstacle:        "
        f"{pipeline.semantic_obstacle_count == 1}"
    )
    print(
        "  Tangential boundary in every yaw slice: "
        f"{bool(np.all(tangential_counts > 0))}"
    )
    print(
        "  All field values finite:      "
        f"{bool(np.isfinite(h_full.values).all())}"
    )

    print("\nInterpretation:")
    if all(semantic_checks):
        print(
            "  [PASS] Semantic guidance produced one finite, converged "
            "yaw-dependent h_full field."
        )
    else:
        print(
            "  [WARNING] The h_full stack is incomplete. Inspect the "
            "field checks before trusting the CBF result."
        )
    print("  The robot remained stationary during Stage B.")
    print("=" * 70)


def print_stage_c_report(
    *,
    nominal_command: WaypointCommand,
    filter_result: SafeSAGEUnicycleCBFResult,
    initial_pose: RobotPose2D,
    final_pose: RobotPose2D,
) -> None:
    """Print the stationary full-state CBF evaluation."""

    position_drift = math.hypot(
        final_pose.x - initial_pose.x,
        final_pose.y - initial_pose.y,
    )
    yaw_drift = absolute_wrapped_angle_difference(
        final_pose.yaw,
        initial_pose.yaw,
    )
    local_query_is_correct = (
        np.allclose(
            filter_result.query_position_grid_xy,
            CURRENT_LOOKAHEAD_POSITION_GRID_XY,
            rtol=0.0,
            atol=1e-12,
        )
        and math.isclose(
            filter_result.query_yaw_radians,
            CURRENT_CONFIGURATION_YAW_RADIANS,
            abs_tol=1e-12,
        )
    )
    stationary = (
        position_drift <= MAXIMUM_STATIONARY_POSITION_DRIFT
        and yaw_drift <= MAXIMUM_STATIONARY_YAW_DRIFT_RADIANS
    )
    accepted = (
        local_query_is_correct
        and filter_result.constraint_feasible
        and filter_result.constraint_satisfied
        and not filter_result.emergency_stop_active
        and stationary
    )

    gradient_x, gradient_y, gradient_yaw = (
        filter_result.sampled_gradient
    )
    guidance_x, guidance_y = filter_result.sampled_guidance_grid_xy

    print("\n" + "=" * 70)
    print("SAFE-SAGE H_FULL STAGE C - UNICYCLE CONTROL INTERFACE")
    print("=" * 70)

    print("\nControl model:")
    print("  Chassis state:                (x, y, psi)")
    print("  Executable input:             (v, omega)")
    print(
        "  Look-ahead distance:          "
        f"{UNICYCLE_LOOKAHEAD_DISTANCE:.3f} m"
    )

    print("\nLook-ahead field query:")
    print(
        "  Position (x, y):              "
        f"({filter_result.query_position_grid_xy[0]:.3f}, "
        f"{filter_result.query_position_grid_xy[1]:.3f}) m"
    )
    print(
        "  Relative yaw psi:             "
        f"{math.degrees(filter_result.query_yaw_radians):.2f} degrees"
    )
    print(f"  Grid cell:                    {filter_result.query_cell}")
    print(
        "  Yaw interpolation:            "
        f"slice {filter_result.lower_yaw_slice} -> "
        f"{filter_result.upper_yaw_slice}, "
        f"weight={filter_result.yaw_interpolation_weight:.3f}"
    )

    print("\nSampled full-state safety field:")
    print(
        "  h_full:                       "
        f"{filter_result.sampled_safety_value:.6f}"
    )
    print(f"  dh/dx:                        {gradient_x:.6f}")
    print(f"  dh/dy:                        {gradient_y:.6f}")
    print(f"  dh/dpsi:                      {gradient_yaw:.6f}")
    print(f"  Semantic LGF vx:              {guidance_x:.6f}")
    print(f"  Semantic LGF vy:              {guidance_y:.6f}")
    print(
        "  dh/dt (static obstacle):      "
        f"{filter_result.temporal_safety_derivative:.6f}"
    )

    print("\nNominal waypoint command:")
    print(
        "  Linear velocity:              "
        f"{nominal_command.linear_velocity:.6f} m/s"
    )
    print(
        "  Angular velocity:             "
        f"{nominal_command.angular_velocity:.6f} rad/s"
    )
    print(f"  Waypoint mode:                {nominal_command.mode.value}")

    print("\nFiltered command:")
    print(
        "  Safe linear velocity:         "
        f"{filter_result.safe_linear_velocity:.6f} m/s"
    )
    print(
        "  Safe angular velocity:        "
        f"{filter_result.safe_angular_velocity:.6f} rad/s"
    )
    print(
        "  Intervention active:          "
        f"{filter_result.intervention_active}"
    )
    print(
        "  Emergency stop:               "
        f"{filter_result.emergency_stop_active}"
    )
    if filter_result.emergency_reason is not None:
        print(
            "  Emergency reason:             "
            f"{filter_result.emergency_reason}"
        )

    print("\nFull-state CBF inequality:")
    print(
        "  Paper-space normal:           "
        f"{filter_result.constraint.constraint_normal}"
    )
    print(
        "  Linear coefficient:           "
        f"{filter_result.linear_velocity_coefficient:.6f}"
    )
    print(
        "  Angular coefficient:          "
        f"{filter_result.angular_velocity_coefficient:.6f}"
    )
    print(
        "  Required control LHS:         "
        f">= {filter_result.constraint_right_hand_side:.6f}"
    )
    print(
        "  Nominal control LHS:          "
        f"{filter_result.nominal_control_left_hand_side:.6f}"
    )
    print(
        "  Safe control LHS:             "
        f"{filter_result.safe_control_left_hand_side:.6f}"
    )
    print(
        "  Safe CBF margin:              "
        f"{filter_result.cbf_margin:.3e}"
    )

    print("\nValidation checks:")
    print(f"  Look-ahead query used:         {local_query_is_correct}")
    print(
        "  Constraint feasible:          "
        f"{filter_result.constraint_feasible}"
    )
    print(
        "  Constraint satisfied:         "
        f"{filter_result.constraint_satisfied}"
    )
    print(f"  Position drift:                {position_drift:.6f} m")
    print(
        "  Yaw drift:                     "
        f"{math.degrees(yaw_drift):.6f} degrees"
    )
    print(f"  Robot remained stationary:     {stationary}")

    print("\nInterpretation:")
    if accepted:
        print(
            "  [PASS] The measured yaw-dependent geometry, semantic "
            "LGF/h_full fields, paper constraint, and unicycle control "
            "interface produced a valid stationary safe command."
        )
    else:
        print(
            "  [WARNING] The stationary full-state integration did not "
            "pass every check. Do not enable robot motion yet."
        )
    if filter_result.intervention_active:
        print(
            "  The nominal command was modified, but neither command was "
            "applied to the wheels."
        )
    else:
        print(
            "  The nominal command was safe in this snapshot and passed "
            "through unchanged; it was not applied to the wheels."
        )
    print("=" * 70)


def evaluate_forward_counterfactuals(
    *,
    safety_filter: SafeSAGEUnicycleCBFFilter,
    nominal_command: WaypointCommand,
    pipeline: SafeSAGEFieldPipelineResult,
) -> tuple[SafeSAGEUnicycleCBFResult, ...]:
    """Evaluate virtual forward poses until the first unsafe boundary."""

    probe_results: list[SafeSAGEUnicycleCBFResult] = []
    probe_count = math.floor(
        FORWARD_PROBE_MAXIMUM_X / FORWARD_PROBE_SPACING
    ) + 1

    for probe_index in range(probe_count):
        probe_x = probe_index * FORWARD_PROBE_SPACING
        result = safety_filter.filter_command(
            nominal_command.linear_velocity,
            nominal_command.angular_velocity,
            query_position_grid_xy=np.array(
                [
                    probe_x
                    + UNICYCLE_LOOKAHEAD_DISTANCE,
                    0.0,
                ],
                dtype=np.float64,
            ),
            query_yaw_radians=CURRENT_CONFIGURATION_YAW_RADIANS,
            configuration_space=pipeline.configuration_space,
            h_full=pipeline.h_full,
            guidance_fields=pipeline.guidance_fields,
            temporal_safety_derivative=(
                STATIC_TEMPORAL_SAFETY_DERIVATIVE
            ),
        )
        probe_results.append(result)
        if result.query_on_unsafe_boundary:
            break

    return tuple(probe_results)


def print_forward_probe_report(
    *,
    probe_results: tuple[SafeSAGEUnicycleCBFResult, ...],
) -> None:
    """Print and validate the stationary forward safety transition."""

    if not probe_results:
        raise RuntimeError("The forward counterfactual sweep is empty.")

    boundary_result = next(
        (
            result
            for result in probe_results
            if result.query_on_unsafe_boundary
        ),
        None,
    )
    safe_results = tuple(
        result
        for result in probe_results
        if not result.emergency_stop_active
    )
    intervention_results = tuple(
        result
        for result in safe_results
        if result.intervention_active
    )
    first_intervention = (
        intervention_results[0]
        if intervention_results
        else None
    )

    all_safe_constraints_valid = bool(safe_results) and all(
        result.constraint_feasible
        and result.constraint_satisfied
        and result.cbf_margin >= -1e-9
        for result in safe_results
    )
    boundary_found = boundary_result is not None
    boundary_stopped = (
        boundary_result is not None
        and boundary_result.emergency_stop_active
        and boundary_result.safe_linear_velocity == 0.0
        and boundary_result.safe_angular_velocity == 0.0
        and boundary_result.emergency_reason
        == "unsafe_configuration_boundary"
    )
    intervention_before_boundary = (
        first_intervention is not None
        and boundary_result is not None
        and first_intervention.query_position_grid_xy[0]
        < boundary_result.query_position_grid_xy[0]
    )
    accepted = (
        all_safe_constraints_valid
        and boundary_found
        and boundary_stopped
        and intervention_before_boundary
    )

    print("\n" + "=" * 70)
    print("SAFE-SAGE H_FULL STAGE C2 - STATIONARY FORWARD PROBE")
    print("=" * 70)
    print(
        "\nThe following positions are counterfactual field queries; "
        "the robot did not move."
    )
    print(
        "\n   q_l.x      h       vsem_x   vsem_y   v_safe   w_safe  "
        "status"
    )
    print("  " + "-" * 72)

    for result in probe_results:
        if result.query_on_unsafe_boundary:
            status = "BOUNDARY STOP"
        elif result.intervention_active:
            status = "INTERVENE"
        else:
            status = "PASS"

        print(
            f"  {result.query_position_grid_xy[0]:6.2f}  "
            f"{result.sampled_safety_value:8.4f}  "
            f"{result.sampled_guidance_grid_xy[0]:8.3f}  "
            f"{result.sampled_guidance_grid_xy[1]:8.3f}  "
            f"{result.safe_linear_velocity:7.3f}  "
            f"{result.safe_angular_velocity:7.3f}  "
            f"{status}"
        )

    print("\nTransition checks:")
    print(f"  Safe probes evaluated:         {len(safe_results)}")
    print(
        "  All safe constraints valid:   "
        f"{all_safe_constraints_valid}"
    )
    print(f"  Intervention observed:         {bool(intervention_results)}")
    if first_intervention is not None:
        print(
            "  First intervention at x:      "
            f"{first_intervention.query_position_grid_xy[0]:.3f} m"
        )
    print(f"  Unsafe boundary found:         {boundary_found}")
    if boundary_result is not None:
        print(
            "  First boundary at x:          "
            f"{boundary_result.query_position_grid_xy[0]:.3f} m"
        )
    print(f"  Boundary produced stop:        {boundary_stopped}")
    print(
        "  Intervention before boundary: "
        f"{intervention_before_boundary}"
    )

    print("\nInterpretation:")
    if accepted:
        print(
            "  [PASS] The static full-state filter intervened before the "
            "yaw-specific collision boundary, maintained every feasible "
            "CBF constraint, and stopped at the boundary."
        )
    else:
        print(
            "  [WARNING] The expected pass/intervene/stop transition was "
            "not fully observed. Do not enable closed-loop motion yet."
        )
    print("  Every row above was diagnostic; no wheel command was applied.")
    print("=" * 70)


def main() -> None:
    """Run the stationary full-state Safe-SAGE CBF integration."""

    print("\n" + "=" * 70)
    print("NOVA CARTER SAFE-SAGE H_FULL CBF DEMO")
    print("=" * 70)
    print("Mode: stationary Stage A/B/C validation")
    print(f"Guidance policy: {GUIDANCE_POLICY_PATH}")

    guidance_policy = SafeSAGEGuidancePolicy.from_json(
        GUIDANCE_POLICY_PATH
    )
    if SEMANTIC_CLASS not in guidance_policy.class_parameters:
        raise RuntimeError(
            "Guidance policy is missing required class: "
            f"{SEMANTIC_CLASS}"
        )
    class_policy = class_policy_from_guidance_policy(guidance_policy)

    measured_footprint = RectangularRobotFootprint(
        front_extent=ROBOT_FRONT_EXTENT,
        rear_extent=ROBOT_REAR_EXTENT,
        left_extent=ROBOT_LEFT_EXTENT,
        right_extent=ROBOT_RIGHT_EXTENT,
        safety_margin=ROBOT_SAFETY_MARGIN,
    )
    footprint = shift_rectangular_footprint_to_lookahead(
        measured_footprint,
        lookahead_distance=UNICYCLE_LOOKAHEAD_DISTANCE,
    )
    field_pipeline_config = create_field_pipeline_config(footprint)

    loaded_scene = load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=EXPECTED_METERS_PER_UNIT,
    )
    create_static_proxy(loaded_scene.stage)

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
    position_robot_for_experiment(robot_controller)
    robot_pose = verify_robot_start_pose(loaded_scene.stage)
    robot_controller.print_configuration()

    lidar = NovaCarterLidar(
        lidar_prim_path=PANDAR_XT32_LIDAR_PRIM_PATH
    )
    lidar.initialize()

    print(
        "\nHolding the robot and proxy still for "
        f"{SETTLE_STEPS} physics steps..."
    )
    if not advance_playing_steps(
        simulation_app,
        physics_steps=SETTLE_STEPS,
        on_step=robot_controller.stop,
    ):
        return

    robot_pose = read_robot_pose(loaded_scene.stage)
    stationary_reference_pose = robot_pose
    robot_from_lidar = compute_relative_prim_transform(
        loaded_scene.stage,
        source_prim_path=PANDAR_XT32_LIDAR_PRIM_PATH,
        target_prim_path=ROBOT_BASE_PRIM_PATH,
    )
    floor_height_robot = estimate_floor_height_in_robot_frame(
        loaded_scene.stage
    )
    detection_config = ObstacleDetectionConfig(
        minimum_height=(
            floor_height_robot + MINIMUM_HEIGHT_ABOVE_FLOOR
        ),
        maximum_height=(
            floor_height_robot + MAXIMUM_HEIGHT_ABOVE_FLOOR
        ),
        minimum_range=MINIMUM_DETECTION_RANGE,
        maximum_range=MAXIMUM_DETECTION_RANGE,
        forward_only=True,
        angular_bin_size_degrees=ANGULAR_BIN_SIZE_DEGREES,
    )

    print("\nWaiting for a valid LiDAR scan...")
    points_sensor_frame, frame_info, scan_wait_steps = (
        wait_for_first_scan(lidar, robot_controller)
    )
    obstacle_points = extract_obstacle_points(
        points_sensor_frame,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
    )
    if obstacle_points.shape[0] == 0:
        raise RuntimeError("The scan produced no obstacle points.")

    obstacle_tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=TRACKER_CLUSTER_RADIUS,
            minimum_cluster_points=TRACKER_MINIMUM_CLUSTER_POINTS,
            maximum_association_distance=(
                TRACKER_MAXIMUM_ASSOCIATION_DISTANCE
            ),
        )
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
    proxy_track, proxy_association_error = find_proxy_track(
        tracking_result,
        robot_pose=robot_pose,
    )
    if proxy_track is None:
        raise RuntimeError(
            "No LiDAR track matched the controlled proxy. Nearest "
            f"association error was {proxy_association_error:.3f} m."
        )

    proxy_point_mask = (
        tracking_result.point_track_ids == proxy_track.track_id
    )
    proxy_points = np.ascontiguousarray(
        obstacle_points[proxy_point_mask],
        dtype=np.float64,
    )
    proxy_point_track_ids = np.ascontiguousarray(
        tracking_result.point_track_ids[proxy_point_mask],
        dtype=np.int64,
    )
    if proxy_points.shape[0] == 0:
        raise RuntimeError(
            "The associated proxy track contained no aligned points."
        )

    semantic_obstacle = SemanticObstacle(
        tracked_obstacle=proxy_track,
        semantic_class=SEMANTIC_CLASS,
        semantic_confidence=1.0,
        label_source="manual",
    )
    source_grid = build_local_occupancy_grid(
        proxy_points,
        point_track_ids=proxy_point_track_ids,
        semantic_obstacles=(semantic_obstacle,),
        class_safety_policy=class_policy,
        config=LocalOccupancyGridConfig(
            resolution=GRID_RESOLUTION,
            minimum_x=GRID_MINIMUM_X,
            maximum_x=GRID_MAXIMUM_X,
            minimum_y=GRID_MINIMUM_Y,
            maximum_y=GRID_MAXIMUM_Y,
            obstacle_inflation_radius=(
                GRID_OBSTACLE_INFLATION_RADIUS
            ),
        ),
    )

    print("\nBuilding yaw-dependent Safe-SAGE fields...")
    pipeline = build_safe_sage_field_pipeline(
        source_grid,
        guidance_policy=guidance_policy,
        config=field_pipeline_config,
        semantic_obstacles=(semantic_obstacle,),
    )

    nominal_command = create_waypoint_controller().compute_command(
        robot_pose
    )
    paper_filter = SafeSAGECBFSafetyFilter(
        SafeSAGECBFConfig(
            gamma=CBF_GAMMA,
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
            planar_velocity_weight=CBF_LINEAR_VELOCITY_WEIGHT,
            angular_velocity_weight=CBF_ANGULAR_VELOCITY_WEIGHT,
        )
    )
    safety_filter = SafeSAGEUnicycleCBFFilter(
        paper_filter,
        SafeSAGEUnicycleCBFConfig(
            lookahead_distance=(
                UNICYCLE_LOOKAHEAD_DISTANCE
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
            linear_velocity_weight=CBF_LINEAR_VELOCITY_WEIGHT,
            angular_velocity_weight=CBF_ANGULAR_VELOCITY_WEIGHT,
        ),
    )
    filter_result = safety_filter.filter_command(
        nominal_command.linear_velocity,
        nominal_command.angular_velocity,
        query_position_grid_xy=(
            CURRENT_LOOKAHEAD_POSITION_GRID_XY
        ),
        query_yaw_radians=CURRENT_CONFIGURATION_YAW_RADIANS,
        configuration_space=pipeline.configuration_space,
        h_full=pipeline.h_full,
        guidance_fields=pipeline.guidance_fields,
        temporal_safety_derivative=(
            STATIC_TEMPORAL_SAFETY_DERIVATIVE
        ),
    )
    forward_probe_results = evaluate_forward_counterfactuals(
        safety_filter=safety_filter,
        nominal_command=nominal_command,
        pipeline=pipeline,
    )

    robot_controller.stop()
    app_utils.update_app(steps=5)
    final_pose = read_robot_pose(loaded_scene.stage)

    print_stage_a_report(
        robot_pose=robot_pose,
        frame_info=frame_info,
        scan_wait_steps=scan_wait_steps,
        obstacle_points=obstacle_points,
        tracking_result=tracking_result,
        proxy_track=proxy_track,
        proxy_association_error=proxy_association_error,
        proxy_points=proxy_points,
        source_grid=source_grid,
        pipeline=pipeline,
        footprint=footprint,
    )
    print_stage_b_report(
        guidance_policy=guidance_policy,
        pipeline=pipeline,
    )
    print_stage_c_report(
        nominal_command=nominal_command,
        filter_result=filter_result,
        initial_pose=stationary_reference_pose,
        final_pose=final_pose,
    )
    print_forward_probe_report(
        probe_results=forward_probe_results,
    )

    robot_controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()

    print("\nExpected scene behavior:")
    print("  - Nova Carter remains at approximately (26.5, 5.0).")
    print("  - One blue cube remains directly ahead of the robot.")
    print("  - The wheels never receive the printed nominal or safe command.")
    print("  - Stage C2 positions are virtual; the robot does not follow them.")
    print("  - No bypass motion is expected in this stationary validation.")
    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when finished.")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 70)
        print("ERROR DURING SAFE-SAGE H_FULL CBF DEMO")
        print("=" * 70)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
