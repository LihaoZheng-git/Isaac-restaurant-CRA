"""Validate tangential-LGF integration with the existing PSF-CBF filter.

This controlled Isaac Sim demo uses one session-only static cube as a
semantic proxy. The robot remains stationary throughout the experiment.

Stage A - perception and geometry:

1. Load the restaurant robot scene.
2. Position Nova Carter in the spacious test area.
3. Create a static proxy directly ahead of the robot.
4. Capture one PandarXT-32 scan and extract obstacle points.
5. Associate the proxy with one real LiDAR track.
6. Retain only that track for the controlled comparison.

Stage B - stationary tangential-LGF integration:

1. Assign the manual ``person`` label to the proxy track.
2. Build one shared occupancy grid and geometric Poisson safety function.
3. Solve a normal-only Risk-Aware guidance field.
4. Solve pass-left and pass-right tangential guidance fields.
5. Filter the same nominal waypoint command with all three fields.
6. Verify that the normal boundary is preserved and the passing-side choice
   produces opposite lateral guidance and angular commands.

The proxy position is used only to select the controlled LiDAR track and to
report experimental distance. It is not passed to the occupancy builder,
PDE solvers, or safety filter. The semantic label is manual by design;
camera/YOLO fusion is a later stage.

This milestone keeps ``h`` geometric and shared. It validates the
tangential Laplace guidance vector inside the existing analytical PSF-CBF
filter. It does not yet implement semantic ``h_full``, temporal derivatives,
yaw-indexed fields, MPC, or closed-loop bypass motion.

Run from the project root:

    python -u scripts/demos/safe_sage_tangential_lgf_demo.py

Press Play in Isaac Sim when requested.
"""

import math
import sys
import traceback
from dataclasses import dataclass, replace
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
from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldResult,
    solve_laplace_guidance_field,
)
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
    PoissonSafetyFunctionResult,
    solve_poisson_safety_function,
)
from robot.safety.psf_cbf_filter import (
    PSFCBFConfig,
    PSFCBFResult,
    PSFCBFSafetyFilter,
)
from robot.safety.safe_sage_guidance_policy import (
    PassingSide,
    SafeSAGEGuidanceParameters,
    SafeSAGEGuidancePolicy,
)
from robot.safety.semantic_obstacle import SemanticObstacle
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
    TangentialLaplaceGuidanceFieldResult,
    solve_tangential_laplace_guidance_field,
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

STATIC_PROXY_PRIM_PATH = "/World/SafeSAGETangentialTestObstacle"
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

# The nearest proxy face begins about 0.70 m ahead of the chassis.
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

GRID_RESOLUTION = 0.20
GRID_MINIMUM_X = -1.0
GRID_MAXIMUM_X = 5.0
GRID_MINIMUM_Y = -3.0
GRID_MAXIMUM_Y = 3.0
GRID_OBSTACLE_INFLATION_RADIUS = 0.20

PSF_FORCING = -1.0
PSF_CONVERGENCE_TOLERANCE = 1e-6
LAPLACE_CONVERGENCE_TOLERANCE = 1e-6
TANGENTIAL_BUFFER_RADIUS = 0.60

PSF_CBF_GAMMA = 1.0

# This validation point is deliberately inside the harmonic region. At the
# previous 0.30 m setting, the close proxy places the sample on the fixed
# normal obstacle interface, where tangential guidance is correctly zero.
# Final controller tuning is outside the scope of this stationary demo.
PSF_CBF_LOOKAHEAD_DISTANCE = 0.15

GUIDANCE_DIRECTION_TOLERANCE = 1e-5
ANGULAR_RESPONSE_TOLERANCE = 1e-5

TRANSIENT_LIDAR_ERRORS = (
    "No LiDAR data is available yet",
    "contains no points",
    "no valid XYZ points remained",
)


@dataclass(frozen=True)
class GuidanceEvaluation:
    """One filtered result using a selected guidance construction."""

    name: str
    guidance_field: LaplaceGuidanceFieldResult
    filter_result: PSFCBFResult
    tangential_interface_cell_count: int


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
    """Teleport the initialized articulation to the controlled start pose."""

    robot_controller.robot.set_world_poses(
        positions=[ROBOT_START_POSITION_WORLD],
        orientations=[
            yaw_to_quaternion_wxyz(
                ROBOT_START_YAW_RADIANS
            )
        ],
    )
    robot_controller.stop()
    app_utils.update_app(steps=5)


def create_static_proxy(
    stage: Usd.Stage,
) -> None:
    """Create the session-only visual proxy."""

    existing_prim = stage.GetPrimAtPath(STATIC_PROXY_PRIM_PATH)
    if existing_prim.IsValid():
        raise RuntimeError(
            "The tangential-LGF test proxy already exists: "
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


def read_robot_pose(
    stage: Usd.Stage,
) -> RobotPose2D:
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


def verify_robot_start_pose(
    stage: Usd.Stage,
) -> RobotPose2D:
    """Validate that the runtime articulation teleport succeeded."""

    pose = read_robot_pose(stage)
    position_error = math.hypot(
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

    if position_error > 0.10:
        raise RuntimeError(
            "The robot did not reach the requested experiment position."
        )
    if yaw_error > math.radians(5.0):
        raise RuntimeError(
            "The robot did not reach the requested experiment yaw."
        )

    return pose


def estimate_floor_height_in_robot_frame(
    stage: Usd.Stage,
) -> float:
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


def proxy_center_in_robot_frame(
    pose: RobotPose2D,
) -> np.ndarray:
    """Express the known proxy center in current chassis-frame axes."""

    x_world_error = STATIC_PROXY_POSITION_WORLD[0] - pose.x
    y_world_error = STATIC_PROXY_POSITION_WORLD[1] - pose.y
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


def distance_to_proxy_surface(
    pose: RobotPose2D,
) -> float:
    """Return planar chassis-center distance to the proxy box surface."""

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
    """Expose exactly the guidance policy's normal-flux parameters."""

    return ClassSafetyPolicy(
        {
            semantic_class: parameters.safety_parameters
            for semantic_class, parameters
            in guidance_policy.class_parameters.items()
        },
        unknown_class=guidance_policy.unknown_class,
        risk_flux_mapping=guidance_policy.risk_flux_mapping,
    )


def policy_with_passing_side(
    policy: SafeSAGEGuidancePolicy,
    *,
    semantic_class: str,
    passing_side: PassingSide,
) -> SafeSAGEGuidancePolicy:
    """Return one counterfactual copy with an explicit passing side."""

    parameters: dict[str, SafeSAGEGuidanceParameters] = {}
    for class_name, class_parameters in policy.class_parameters.items():
        parameters[class_name] = (
            replace(class_parameters, passing_side=passing_side)
            if class_name == semantic_class
            else class_parameters
        )

    return SafeSAGEGuidancePolicy(
        parameters,
        unknown_class=policy.unknown_class,
        risk_flux_mapping=policy.risk_flux_mapping,
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


def evaluate_guidance(
    name: str,
    *,
    guidance_field: LaplaceGuidanceFieldResult,
    nominal_command: WaypointCommand,
    occupancy_grid: LocalOccupancyGrid,
    safety_function: PoissonSafetyFunctionResult,
    safety_filter: PSFCBFSafetyFilter,
) -> GuidanceEvaluation:
    """Filter the shared nominal command with one guidance field."""

    filter_result = safety_filter.filter_command(
        nominal_command.linear_velocity,
        nominal_command.angular_velocity,
        occupancy_grid,
        safety_function,
        guidance_field,
    )
    tangential_cell_count = (
        guidance_field.tangential_interface_cell_count
        if isinstance(
            guidance_field,
            TangentialLaplaceGuidanceFieldResult,
        )
        else 0
    )
    return GuidanceEvaluation(
        name=name,
        guidance_field=guidance_field,
        filter_result=filter_result,
        tangential_interface_cell_count=tangential_cell_count,
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
    occupancy_grid: LocalOccupancyGrid,
    safety_function: PoissonSafetyFunctionResult,
) -> None:
    """Print perception, association, and shared-geometry validation."""

    nearest_proxy_distance = float(
        np.min(np.linalg.norm(proxy_points, axis=1))
    )

    print("\n" + "=" * 70)
    print("TANGENTIAL-LGF STAGE A - PERCEPTION AND SHARED GEOMETRY")
    print("=" * 70)

    print("\nRobot pose:")
    print(f"  x:                       {robot_pose.x:.4f} m")
    print(f"  y:                       {robot_pose.y:.4f} m")
    print(
        "  yaw:                     "
        f"{math.degrees(robot_pose.yaw):.2f} degrees"
    )

    print("\nLiDAR and tracking:")
    print(f"  Frame ID:                 {frame_info.sensor_frame_id}")
    print(f"  Waited updates:            {scan_wait_steps}")
    print(f"  All obstacle points:       {obstacle_points.shape[0]}")
    print(f"  Detected clusters:         {tracking_result.cluster_count}")
    print(f"  Proxy track ID:            {proxy_track.track_id}")
    print(f"  Proxy track points:        {proxy_points.shape[0]}")
    print(
        "  Proxy association error: "
        f"{proxy_association_error:.4f} m"
    )
    print(
        "  Nearest proxy return:     "
        f"{nearest_proxy_distance:.4f} m"
    )
    print(
        "  Known surface distance:   "
        f"{distance_to_proxy_surface(robot_pose):.4f} m"
    )

    print("\nShared grid and geometric Poisson solve:")
    print(f"  Grid shape:                {occupancy_grid.shape}")
    print(
        "  Resolution:                "
        f"{occupancy_grid.config.resolution:.3f} m"
    )
    print(
        "  Occupied cells:            "
        f"{occupancy_grid.occupied_cell_count}"
    )
    print(
        "  Poisson iterations:        "
        f"{safety_function.iteration_count}"
    )
    print(
        "  Poisson maximum residual:  "
        f"{safety_function.maximum_residual:.3e}"
    )

    print("\nControlled-data boundary:")
    print("  [PASS] A real LiDAR track was associated with the proxy.")
    print("  Only proxy-track points enter the occupancy grid.")
    print("  Ground truth is not used by the field or safety filter.")
    print("  The robot remained stationary during Stage A.")
    print("=" * 70)


def print_evaluation(
    evaluation: GuidanceEvaluation,
) -> None:
    """Print one guidance and filtered-command evaluation."""

    result = evaluation.filter_result

    print(f"\n{evaluation.name}:")
    print(
        "  Tangential interface cells: "
        f"{evaluation.tangential_interface_cell_count}"
    )
    print(
        "  Laplace iterations:          "
        f"{evaluation.guidance_field.iteration_count}"
    )
    print(
        "  Laplace maximum residual:    "
        f"{evaluation.guidance_field.maximum_residual:.3e}"
    )
    print(
        "  Sampled safety h:            "
        f"{result.sampled_safety_value:.6f}"
    )
    print(
        "  Sampled guidance:            "
        f"({result.sampled_guidance_vector_robot_xy[0]:.6f}, "
        f"{result.sampled_guidance_vector_robot_xy[1]:.6f})"
    )
    print(
        "  Safe linear velocity:        "
        f"{result.safe_linear_velocity:.6f} m/s"
    )
    print(
        "  Safe angular velocity:       "
        f"{result.safe_angular_velocity:.6f} rad/s"
    )
    print(
        "  Constraint margin:           "
        f"{result.constraint_margin:.3e}"
    )
    print(f"  Intervention:                {result.intervention_active}")
    print(f"  Constraint satisfied:        {result.constraint_satisfied}")
    print(f"  Emergency stop:              {result.emergency_stop_active}")


def print_stage_b_report(
    *,
    parameters: SafeSAGEGuidanceParameters,
    nominal_command: WaypointCommand,
    normal_evaluation: GuidanceEvaluation,
    left_evaluation: GuidanceEvaluation,
    right_evaluation: GuidanceEvaluation,
) -> None:
    """Print and interpret the stationary integration comparison."""

    normal_result = normal_evaluation.filter_result
    left_result = left_evaluation.filter_result
    right_result = right_evaluation.filter_result

    normal_mask = normal_evaluation.guidance_field.obstacle_interface_mask
    left_normal_preserved = np.allclose(
        left_evaluation.guidance_field.vectors_robot_xy[normal_mask],
        normal_evaluation.guidance_field.vectors_robot_xy[normal_mask],
        rtol=0.0,
        atol=1e-12,
    )
    right_normal_preserved = np.allclose(
        right_evaluation.guidance_field.vectors_robot_xy[normal_mask],
        normal_evaluation.guidance_field.vectors_robot_xy[normal_mask],
        rtol=0.0,
        atol=1e-12,
    )

    left_guidance_y = (
        left_result.sampled_guidance_vector_robot_xy[1]
    )
    right_guidance_y = (
        right_result.sampled_guidance_vector_robot_xy[1]
    )
    opposite_lateral_guidance = (
        left_guidance_y > GUIDANCE_DIRECTION_TOLERANCE
        and right_guidance_y < -GUIDANCE_DIRECTION_TOLERANCE
    )
    opposite_angular_response = (
        left_result.safe_angular_velocity
        > normal_result.safe_angular_velocity
        + ANGULAR_RESPONSE_TOLERANCE
        and right_result.safe_angular_velocity
        < normal_result.safe_angular_velocity
        - ANGULAR_RESPONSE_TOLERANCE
    )
    same_sampled_safety = (
        left_result.sampled_safety_value
        == right_result.sampled_safety_value
        == normal_result.sampled_safety_value
    )
    tangential_interfaces_exist = (
        left_evaluation.tangential_interface_cell_count > 0
        and right_evaluation.tangential_interface_cell_count > 0
    )
    constraints_valid = all(
        result.constraint_feasible
        and result.constraint_satisfied
        and not result.emergency_stop_active
        for result in (
            normal_result,
            left_result,
            right_result,
        )
    )

    print("\n" + "=" * 70)
    print("TANGENTIAL-LGF STAGE B - STATIONARY FILTER INTEGRATION")
    print("=" * 70)

    print("\nShared semantic parameters:")
    print(f"  Semantic class:              {SEMANTIC_CLASS}")
    print(f"  Fixed risk:                  {parameters.risk:.3f}")
    print(f"  Normal boundary flux b:      {parameters.boundary_flux:.3f}")
    print(
        "  Tangential strength lambda: "
        f"{parameters.tangential_strength:.3f}"
    )
    print(
        "  Tangential buffer radius:   "
        f"{TANGENTIAL_BUFFER_RADIUS:.3f} m"
    )
    print(
        "  CBF look-ahead distance:    "
        f"{PSF_CBF_LOOKAHEAD_DISTANCE:.3f} m"
    )

    print("\nShared nominal waypoint command:")
    print(
        "  Linear velocity:             "
        f"{nominal_command.linear_velocity:.6f} m/s"
    )
    print(
        "  Angular velocity:            "
        f"{nominal_command.angular_velocity:.6f} rad/s"
    )
    print(f"  Waypoint mode:               {nominal_command.mode.value}")

    print_evaluation(normal_evaluation)
    print_evaluation(left_evaluation)
    print_evaluation(right_evaluation)

    print("\nControlled comparison checks:")
    print(f"  Shared geometric h:           {same_sampled_safety}")
    print(f"  Tangential interfaces exist: {tangential_interfaces_exist}")
    print(f"  Left normal boundary intact: {left_normal_preserved}")
    print(f"  Right normal boundary intact:{right_normal_preserved}")
    print(f"  Opposite lateral guidance:   {opposite_lateral_guidance}")
    print(f"  Opposite angular response:   {opposite_angular_response}")
    print(f"  All constraints valid:       {constraints_valid}")

    checks = (
        same_sampled_safety,
        tangential_interfaces_exist,
        left_normal_preserved,
        right_normal_preserved,
        opposite_lateral_guidance,
        opposite_angular_response,
        constraints_valid,
    )

    print("\nInterpretation:")
    if all(checks):
        print(
            "  [PASS] The shared geometric PSF and normal boundary were "
            "unchanged, while passing side produced opposite lateral "
            "guidance and safe angular commands."
        )
    else:
        print(
            "  [WARNING] The tangential field did not fully reach the "
            "filtered command as expected. Inspect the checks above "
            "before adding robot motion."
        )

    print("  All three commands were diagnostic; the robot did not move.")
    print("=" * 70)


def main() -> None:
    """Run stationary tangential-LGF integration validation."""

    print("\n" + "=" * 70)
    print("NOVA CARTER TANGENTIAL-LGF INTEGRATION DEMO")
    print("=" * 70)
    print("Mode: stationary Stage A/B validation")
    print(f"Guidance policy: {GUIDANCE_POLICY_PATH}")

    base_guidance_policy = SafeSAGEGuidancePolicy.from_json(
        GUIDANCE_POLICY_PATH
    )
    if SEMANTIC_CLASS not in base_guidance_policy.class_parameters:
        raise RuntimeError(
            "Guidance policy is missing required class: "
            f"{SEMANTIC_CLASS}"
        )

    class_policy = class_policy_from_guidance_policy(
        base_guidance_policy
    )
    left_policy = policy_with_passing_side(
        base_guidance_policy,
        semantic_class=SEMANTIC_CLASS,
        passing_side=PassingSide.LEFT,
    )
    right_policy = policy_with_passing_side(
        base_guidance_policy,
        semantic_class=SEMANTIC_CLASS,
        passing_side=PassingSide.RIGHT,
    )

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
    (
        points_sensor_frame,
        frame_info,
        scan_wait_steps,
    ) = wait_for_first_scan(lidar, robot_controller)
    obstacle_points = extract_obstacle_points(
        points_sensor_frame,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
    )
    if obstacle_points.shape[0] == 0:
        raise RuntimeError(
            "The initial scan produced no obstacle points."
        )

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
    proxy_track, proxy_error = find_proxy_track(
        tracking_result,
        robot_pose=robot_pose,
    )
    if proxy_track is None:
        raise RuntimeError(
            "No LiDAR track matched the controlled proxy. Nearest "
            f"association error was {proxy_error:.3f} m."
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
    grid_config = LocalOccupancyGridConfig(
        resolution=GRID_RESOLUTION,
        minimum_x=GRID_MINIMUM_X,
        maximum_x=GRID_MAXIMUM_X,
        minimum_y=GRID_MINIMUM_Y,
        maximum_y=GRID_MAXIMUM_Y,
        obstacle_inflation_radius=GRID_OBSTACLE_INFLATION_RADIUS,
    )
    occupancy_grid = build_local_occupancy_grid(
        proxy_points,
        point_track_ids=proxy_point_track_ids,
        semantic_obstacles=(semantic_obstacle,),
        class_safety_policy=class_policy,
        config=grid_config,
    )

    poisson_config = PoissonSafetyFunctionConfig(
        forcing=PSF_FORCING,
        convergence_tolerance=PSF_CONVERGENCE_TOLERANCE,
    )
    safety_function = solve_poisson_safety_function(
        occupancy_grid,
        poisson_config,
    )
    field_config = TangentialLaplaceGuidanceFieldConfig(
        convergence_tolerance=LAPLACE_CONVERGENCE_TOLERANCE,
        tangential_buffer_radius=TANGENTIAL_BUFFER_RADIUS,
    )

    normal_field = solve_laplace_guidance_field(
        occupancy_grid,
        field_config,
    )
    left_field = solve_tangential_laplace_guidance_field(
        occupancy_grid,
        guidance_policy=left_policy,
        semantic_obstacles=(semantic_obstacle,),
        config=field_config,
    )
    right_field = solve_tangential_laplace_guidance_field(
        occupancy_grid,
        guidance_policy=right_policy,
        semantic_obstacles=(semantic_obstacle,),
        config=field_config,
    )

    nominal_command = create_waypoint_controller().compute_command(
        robot_pose
    )
    safety_filter = PSFCBFSafetyFilter(
        PSFCBFConfig(
            gamma=PSF_CBF_GAMMA,
            lookahead_distance=PSF_CBF_LOOKAHEAD_DISTANCE,
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
        )
    )

    normal_evaluation = evaluate_guidance(
        "NORMAL-ONLY BASELINE",
        guidance_field=normal_field,
        nominal_command=nominal_command,
        occupancy_grid=occupancy_grid,
        safety_function=safety_function,
        safety_filter=safety_filter,
    )
    left_evaluation = evaluate_guidance(
        "TANGENTIAL PASS-LEFT",
        guidance_field=left_field,
        nominal_command=nominal_command,
        occupancy_grid=occupancy_grid,
        safety_function=safety_function,
        safety_filter=safety_filter,
    )
    right_evaluation = evaluate_guidance(
        "TANGENTIAL PASS-RIGHT",
        guidance_field=right_field,
        nominal_command=nominal_command,
        occupancy_grid=occupancy_grid,
        safety_function=safety_function,
        safety_filter=safety_filter,
    )

    print_stage_a_report(
        robot_pose=robot_pose,
        frame_info=frame_info,
        scan_wait_steps=scan_wait_steps,
        obstacle_points=obstacle_points,
        tracking_result=tracking_result,
        proxy_track=proxy_track,
        proxy_association_error=proxy_error,
        proxy_points=proxy_points,
        occupancy_grid=occupancy_grid,
        safety_function=safety_function,
    )
    print_stage_b_report(
        parameters=base_guidance_policy.parameters_for(SEMANTIC_CLASS),
        nominal_command=nominal_command,
        normal_evaluation=normal_evaluation,
        left_evaluation=left_evaluation,
        right_evaluation=right_evaluation,
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
        print("ERROR DURING TANGENTIAL-LGF INTEGRATION DEMO")
        print("=" * 70)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
