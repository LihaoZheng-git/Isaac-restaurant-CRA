"""Validate tangential-LGF integration with the existing PSF-CBF filter.

This controlled Isaac Sim demo uses one session-only static cube as a
semantic proxy. Stages A and B are stationary; Stage C is a resettable
closed-loop response ablation.

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

Stage C - resettable closed-loop comparison:

1. Move the proxy farther forward to provide an initially unmodified command.
2. Run a normal-only trial until it establishes a stable blocked response.
3. Reset the robot, waypoint controller, and obstacle tracker.
4. Run the configured pass-right tangential trial from the same initial pose.
5. Verify that an active tangential constraint changes the CBF correction
   toward the requested side.
6. Report physical displacement and proxy clearance as diagnostics, not as
   acceptance criteria for this incomplete Safe-SAGE stack.

The proxy position is used only to select the controlled LiDAR track and to
report experimental distance. It is not passed to the occupancy builder,
PDE solvers, or safety filter. The semantic label is manual by design;
camera/YOLO fusion is a later stage.

This milestone keeps ``h`` geometric and shared. It validates the tangential
Laplace guidance vector inside the existing analytical PSF-CBF filter. It
does not yet implement semantic ``h_full``, temporal derivatives,
yaw-indexed fields, or MPC. Stage C validates the closed-loop steering
correction and reports obstacle clearance separately. A full bypass is not
expected from this reactive ablation: the waypoint controller continues to
aim at the goal behind the obstacle, and the one-step safety projection may
satisfy its constraint mainly by slowing the robot. The controlled tracker
also intentionally retains only the proxy track.

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

# Stage C moves the proxy farther forward so both trials begin without an
# immediate intervention and have space to reveal their closed-loop response.
MOTION_PROXY_POSITION_WORLD = np.array(
    [
        27.90,
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

# Stage C retains the proxy briefly after it moves behind the chassis. The
# larger rear extent prevents its 0.60 m tangential buffer from touching the
# local grid boundary during clearance validation.
MOTION_GRID_MINIMUM_X = -2.0

PSF_FORCING = -1.0
PSF_CONVERGENCE_TOLERANCE = 1e-6
LAPLACE_CONVERGENCE_TOLERANCE = 1e-6
TANGENTIAL_BUFFER_RADIUS = 0.60

PSF_CBF_GAMMA = 1.0

# A smaller gamma activates the Stage C constraint earlier, while the sample
# still contains tangential information. Both resettable trials use this same
# value, so guidance construction remains their only controlled difference.
STAGE_C_PSF_CBF_GAMMA = 0.30

# This validation point is deliberately inside the harmonic region. At the
# previous 0.30 m setting, the close proxy places the sample on the fixed
# normal obstacle interface, where tangential guidance is correctly zero.
# Final controller tuning is outside the scope of this stationary demo.
PSF_CBF_LOOKAHEAD_DISTANCE = 0.15

GUIDANCE_DIRECTION_TOLERANCE = 1e-5
ANGULAR_RESPONSE_TOLERANCE = 1e-5

MOTION_TRIAL_SETTLE_STEPS = 60
MAXIMUM_MOTION_TRIAL_SECONDS = 20.0
MAXIMUM_MOTION_TRIAL_STEPS = math.ceil(
    MAXIMUM_MOTION_TRIAL_SECONDS / PHYSICS_DT
)
MAXIMUM_STALE_SCAN_SECONDS = 0.30
MAXIMUM_STALE_SCAN_STEPS = math.ceil(
    MAXIMUM_STALE_SCAN_SECONDS / PHYSICS_DT
)
STATUS_REPORT_INTERVAL_STEPS = 30
NORMAL_BLOCKED_LINEAR_SPEED = 0.08
NORMAL_BLOCKED_ANGULAR_SPEED = 0.03
NORMAL_BLOCKED_CONFIRMATION_STEPS = 30
TANGENTIAL_SETTLED_LINEAR_SPEED = 0.02
TANGENTIAL_SETTLED_ANGULAR_SPEED = 0.03
TANGENTIAL_SETTLED_CONFIRMATION_STEPS = 30
GROUND_TRUTH_ABORT_DISTANCE = 0.35
REQUESTED_SIDE_ANGULAR_CORRECTION_TOLERANCE = 0.05
BYPASS_LONGITUDINAL_MARGIN = 0.30

NORMAL_ONLY_MODE = "normal-only"
PASS_RIGHT_MODE = "pass-right"

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


@dataclass(frozen=True)
class PipelineFrame:
    """One LiDAR-rate PSF/LGF pipeline update."""

    sensor_frame_id: int
    all_obstacle_point_count: int
    proxy_point_count: int
    proxy_track_id: int
    proxy_association_error: float
    occupancy_grid: LocalOccupancyGrid
    safety_function: PoissonSafetyFunctionResult
    guidance_field: LaplaceGuidanceFieldResult


@dataclass(frozen=True)
class MotionTrialResult:
    """Summary of one resettable Stage C motion trial."""

    mode: str
    termination_reason: str
    completed_steps: int
    received_scan_count: int
    intervention_seen: bool
    first_intervention_step: int | None
    first_intervention_surface_distance: float
    minimum_surface_distance: float
    maximum_absolute_angular_velocity: float
    maximum_absolute_angular_correction: float
    maximum_requested_side_angular_correction: float
    maximum_sampled_guidance_jump: float
    maximum_rightward_displacement: float
    requested_side_correction_seen: bool
    cleared_proxy: bool
    final_pose: RobotPose2D
    final_filter_result: PSFCBFResult
    emergency_stop_seen: bool
    ground_truth_abort_seen: bool
    application_stopped: bool


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
) -> UsdGeom.XformOp:
    """Create the proxy and return its session-layer translation operation."""

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
        translate_op = cube.AddTranslateOp()
        translate_op.Set(
            Gf.Vec3d(
                float(STATIC_PROXY_POSITION_WORLD[0]),
                float(STATIC_PROXY_POSITION_WORLD[1]),
                float(STATIC_PROXY_POSITION_WORLD[2]),
            )
        )
        cube.AddOrientOp().Set(Gf.Quatf(1.0))
        cube.AddScaleOp().Set(Gf.Vec3f(*PROXY_DIMENSIONS))

    return translate_op


def set_proxy_position(
    stage: Usd.Stage,
    translate_op: UsdGeom.XformOp,
    position_world: np.ndarray,
) -> None:
    """Move the session-only proxy to one controlled world position."""

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        translate_op.Set(
            Gf.Vec3d(
                float(position_world[0]),
                float(position_world[1]),
                float(position_world[2]),
            )
        )


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
    *,
    proxy_position_world: np.ndarray,
) -> np.ndarray:
    """Express the known proxy center in current chassis-frame axes."""

    x_world_error = proxy_position_world[0] - pose.x
    y_world_error = proxy_position_world[1] - pose.y
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
    *,
    proxy_position_world: np.ndarray,
) -> float:
    """Return planar chassis-center distance to the proxy box surface."""

    half_x = 0.5 * PROXY_DIMENSIONS[0]
    half_y = 0.5 * PROXY_DIMENSIONS[1]
    x_separation = max(
        abs(pose.x - proxy_position_world[0]) - half_x,
        0.0,
    )
    y_separation = max(
        abs(pose.y - proxy_position_world[1]) - half_y,
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
    proxy_position_world: np.ndarray,
) -> tuple[TrackedObstacle | None, float]:
    """Associate the known proxy center with the nearest LiDAR track."""

    if not tracking_result.tracked_obstacles:
        return None, math.inf

    expected_center = proxy_center_in_robot_frame(
        robot_pose,
        proxy_position_world=proxy_position_world,
    )
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


def build_pipeline_frame(
    mode: str,
    *,
    points_sensor_frame: np.ndarray,
    frame_info: LidarFrameInfo,
    robot_pose: RobotPose2D,
    obstacle_tracker: ObstacleTracker,
    proxy_position_world: np.ndarray,
    robot_from_lidar: np.ndarray,
    detection_config: ObstacleDetectionConfig,
    class_policy: ClassSafetyPolicy,
    guidance_policy: SafeSAGEGuidancePolicy,
    grid_config: LocalOccupancyGridConfig,
    poisson_config: PoissonSafetyFunctionConfig,
    field_config: TangentialLaplaceGuidanceFieldConfig,
) -> PipelineFrame:
    """Run one LiDAR-rate PSF/LGF pipeline update."""

    obstacle_points = extract_obstacle_points(
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
    proxy_track, association_error = find_proxy_track(
        tracking_result,
        robot_pose=robot_pose,
        proxy_position_world=proxy_position_world,
    )
    if proxy_track is None:
        raise RuntimeError(
            "No current LiDAR track matched the controlled proxy; "
            f"nearest association error={association_error:.3f} m."
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
            "The associated proxy track has no aligned LiDAR points."
        )

    semantic_obstacle = SemanticObstacle(
        tracked_obstacle=proxy_track,
        semantic_class=SEMANTIC_CLASS,
        semantic_confidence=1.0,
        label_source="manual",
    )
    occupancy_grid = build_local_occupancy_grid(
        proxy_points,
        point_track_ids=proxy_track_ids,
        semantic_obstacles=(semantic_obstacle,),
        class_safety_policy=class_policy,
        config=grid_config,
    )
    safety_function = solve_poisson_safety_function(
        occupancy_grid,
        poisson_config,
    )

    if mode == NORMAL_ONLY_MODE:
        guidance_field = solve_laplace_guidance_field(
            occupancy_grid,
            field_config,
        )
    elif mode == PASS_RIGHT_MODE:
        guidance_field = solve_tangential_laplace_guidance_field(
            occupancy_grid,
            guidance_policy=guidance_policy,
            semantic_obstacles=(semantic_obstacle,),
            config=field_config,
        )
    else:
        raise ValueError(f"Unsupported motion-trial mode: {mode!r}.")

    return PipelineFrame(
        sensor_frame_id=frame_info.sensor_frame_id,
        all_obstacle_point_count=int(obstacle_points.shape[0]),
        proxy_point_count=int(proxy_points.shape[0]),
        proxy_track_id=proxy_track.track_id,
        proxy_association_error=association_error,
        occupancy_grid=occupancy_grid,
        safety_function=safety_function,
        guidance_field=guidance_field,
    )


def print_motion_status(
    *,
    mode: str,
    completed_steps: int,
    pipeline_frame: PipelineFrame,
    robot_pose: RobotPose2D,
    surface_distance: float,
    rightward_displacement: float,
    nominal_command: WaypointCommand,
    filter_result: PSFCBFResult,
) -> None:
    """Print one compact Stage C status line."""

    print(
        f"  mode={mode:<11} "
        f"step={completed_steps:<4} "
        f"scan={pipeline_frame.sensor_frame_id:<4} "
        f"pose=({robot_pose.x:.3f},{robot_pose.y:.3f}) "
        f"surface={surface_distance:.3f} "
        f"right={rightward_displacement:.3f} "
        f"v_nom={nominal_command.linear_velocity:.3f} "
        f"v_safe={filter_result.safe_linear_velocity:.3f} "
        f"w_safe={filter_result.safe_angular_velocity:.3f} "
        f"dw={filter_result.safe_angular_velocity - nominal_command.angular_velocity:.3f} "
        f"g_y={filter_result.sampled_guidance_vector_robot_xy[1]:.3f} "
        f"intervention={filter_result.intervention_active}"
    )


def run_motion_trial(
    mode: str,
    *,
    stage: Usd.Stage,
    lidar: NovaCarterLidar,
    robot_controller: NovaCarterController,
    safety_filter: PSFCBFSafetyFilter,
    proxy_position_world: np.ndarray,
    robot_from_lidar: np.ndarray,
    detection_config: ObstacleDetectionConfig,
    class_policy: ClassSafetyPolicy,
    guidance_policy: SafeSAGEGuidancePolicy,
    grid_config: LocalOccupancyGridConfig,
    poisson_config: PoissonSafetyFunctionConfig,
    field_config: TangentialLaplaceGuidanceFieldConfig,
) -> MotionTrialResult:
    """Run one resettable closed-loop guidance trial."""

    if mode not in (NORMAL_ONLY_MODE, PASS_RIGHT_MODE):
        raise ValueError(f"Unsupported motion-trial mode: {mode!r}.")

    print("\n" + "-" * 70)
    print(f"STAGE C MOTION TRIAL: {mode.upper()}")
    print("-" * 70)

    position_robot_for_experiment(robot_controller)
    initial_pose = verify_robot_start_pose(stage)

    if not advance_playing_steps(
        simulation_app,
        physics_steps=MOTION_TRIAL_SETTLE_STEPS,
        on_step=robot_controller.stop,
    ):
        raise RuntimeError(
            "Isaac Sim stopped while preparing a motion trial."
        )

    waypoint_controller = create_waypoint_controller()
    obstacle_tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=TRACKER_CLUSTER_RADIUS,
            minimum_cluster_points=TRACKER_MINIMUM_CLUSTER_POINTS,
            maximum_association_distance=(
                TRACKER_MAXIMUM_ASSOCIATION_DISTANCE
            ),
        )
    )

    (
        points_sensor_frame,
        frame_info,
        _,
    ) = wait_for_first_scan(lidar, robot_controller)
    robot_pose = read_robot_pose(stage)
    pipeline_frame = build_pipeline_frame(
        mode,
        points_sensor_frame=points_sensor_frame,
        frame_info=frame_info,
        robot_pose=robot_pose,
        obstacle_tracker=obstacle_tracker,
        proxy_position_world=proxy_position_world,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=grid_config,
        poisson_config=poisson_config,
        field_config=field_config,
    )
    nominal_command = waypoint_controller.compute_command(robot_pose)
    filter_result = safety_filter.filter_command(
        nominal_command.linear_velocity,
        nominal_command.angular_velocity,
        pipeline_frame.occupancy_grid,
        pipeline_frame.safety_function,
        pipeline_frame.guidance_field,
    )

    completed_steps = 0
    received_scan_count = 1
    steps_since_new_scan = 0
    blocked_confirmation_count = 0
    intervention_seen = filter_result.intervention_active
    first_intervention_step = 0 if intervention_seen else None
    initial_surface_distance = distance_to_proxy_surface(
        robot_pose,
        proxy_position_world=proxy_position_world,
    )
    first_intervention_surface_distance = (
        initial_surface_distance if intervention_seen else math.nan
    )
    minimum_surface_distance = initial_surface_distance
    maximum_absolute_angular_velocity = abs(
        filter_result.safe_angular_velocity
    )
    initial_angular_correction = (
        filter_result.safe_angular_velocity
        - nominal_command.angular_velocity
    )
    maximum_absolute_angular_correction = abs(
        initial_angular_correction
    )
    maximum_requested_side_angular_correction = max(
        0.0,
        -initial_angular_correction,
    )
    previous_sampled_guidance = np.asarray(
        filter_result.sampled_guidance_vector_robot_xy,
        dtype=np.float64,
    )
    maximum_sampled_guidance_jump = 0.0
    maximum_rightward_displacement = max(
        0.0,
        initial_pose.y - robot_pose.y,
    )
    previous_intervention = filter_result.intervention_active
    requested_side_correction_seen = (
        mode == PASS_RIGHT_MODE
        and filter_result.intervention_active
        and filter_result.sampled_guidance_vector_robot_xy[1]
        < -GUIDANCE_DIRECTION_TOLERANCE
        and -initial_angular_correction
        >= REQUESTED_SIDE_ANGULAR_CORRECTION_TOLERANCE
    )
    cleared_proxy = False
    emergency_stop_seen = False
    ground_truth_abort_seen = False
    application_stopped = False
    termination_reason = "Maximum motion-trial duration reached."

    print("\nLive motion status:")
    print_motion_status(
        mode=mode,
        completed_steps=0,
        pipeline_frame=pipeline_frame,
        robot_pose=robot_pose,
        surface_distance=initial_surface_distance,
        rightward_displacement=maximum_rightward_displacement,
        nominal_command=nominal_command,
        filter_result=filter_result,
    )

    while completed_steps < MAXIMUM_MOTION_TRIAL_STEPS:
        if not simulation_app.is_running():
            application_stopped = True
            termination_reason = (
                "Isaac Sim stopped running during the motion trial."
            )
            break

        if not app_utils.is_playing():
            robot_controller.stop()
            simulation_app.update()
            continue

        robot_pose = read_robot_pose(stage)
        nominal_command = waypoint_controller.compute_command(robot_pose)
        if nominal_command.route_complete:
            termination_reason = "The waypoint route completed."
            cleared_proxy = True
            robot_controller.stop()
            break

        steps_since_new_scan += 1

        try:
            current_points_sensor = lidar.get_point_cloud()
            current_frame_info = lidar.get_frame_info()
        except RuntimeError as error:
            is_transient = any(
                message in str(error)
                for message in TRANSIENT_LIDAR_ERRORS
            )
            if not is_transient:
                raise
        else:
            if (
                current_frame_info.sensor_frame_id
                != pipeline_frame.sensor_frame_id
            ):
                try:
                    pipeline_frame = build_pipeline_frame(
                        mode,
                        points_sensor_frame=current_points_sensor,
                        frame_info=current_frame_info,
                        robot_pose=robot_pose,
                        obstacle_tracker=obstacle_tracker,
                        proxy_position_world=proxy_position_world,
                        robot_from_lidar=robot_from_lidar,
                        detection_config=detection_config,
                        class_policy=class_policy,
                        guidance_policy=guidance_policy,
                        grid_config=grid_config,
                        poisson_config=poisson_config,
                        field_config=field_config,
                    )
                except (RuntimeError, ValueError) as error:
                    termination_reason = (
                        "Fail-closed perception/PDE stop: "
                        f"{error}"
                    )
                    break

                received_scan_count += 1
                steps_since_new_scan = 0

        if steps_since_new_scan >= MAXIMUM_STALE_SCAN_STEPS:
            termination_reason = (
                "LiDAR data became stale; fail-closed stop applied."
            )
            break

        filter_result = safety_filter.filter_command(
            nominal_command.linear_velocity,
            nominal_command.angular_velocity,
            pipeline_frame.occupancy_grid,
            pipeline_frame.safety_function,
            pipeline_frame.guidance_field,
        )
        surface_distance = distance_to_proxy_surface(
            robot_pose,
            proxy_position_world=proxy_position_world,
        )
        rightward_displacement = max(
            0.0,
            initial_pose.y - robot_pose.y,
        )
        minimum_surface_distance = min(
            minimum_surface_distance,
            surface_distance,
        )
        maximum_rightward_displacement = max(
            maximum_rightward_displacement,
            rightward_displacement,
        )
        maximum_absolute_angular_velocity = max(
            maximum_absolute_angular_velocity,
            abs(filter_result.safe_angular_velocity),
        )
        angular_correction = (
            filter_result.safe_angular_velocity
            - nominal_command.angular_velocity
        )
        maximum_absolute_angular_correction = max(
            maximum_absolute_angular_correction,
            abs(angular_correction),
        )
        requested_side_angular_correction = max(
            0.0,
            -angular_correction,
        )
        maximum_requested_side_angular_correction = max(
            maximum_requested_side_angular_correction,
            requested_side_angular_correction,
        )
        sampled_guidance = np.asarray(
            filter_result.sampled_guidance_vector_robot_xy,
            dtype=np.float64,
        )
        maximum_sampled_guidance_jump = max(
            maximum_sampled_guidance_jump,
            float(
                np.linalg.norm(
                    sampled_guidance - previous_sampled_guidance
                )
            ),
        )
        previous_sampled_guidance = sampled_guidance
        if (
            mode == PASS_RIGHT_MODE
            and filter_result.intervention_active
            and filter_result.sampled_guidance_vector_robot_xy[1]
            < -GUIDANCE_DIRECTION_TOLERANCE
            and requested_side_angular_correction
            >= REQUESTED_SIDE_ANGULAR_CORRECTION_TOLERANCE
        ):
            requested_side_correction_seen = True

        if filter_result.intervention_active and not intervention_seen:
            intervention_seen = True
            first_intervention_step = completed_steps
            first_intervention_surface_distance = surface_distance

        intervention_changed = (
            filter_result.intervention_active != previous_intervention
        )
        periodic_report = (
            completed_steps > 0
            and completed_steps % STATUS_REPORT_INTERVAL_STEPS == 0
        )
        if periodic_report or intervention_changed:
            print_motion_status(
                mode=mode,
                completed_steps=completed_steps,
                pipeline_frame=pipeline_frame,
                robot_pose=robot_pose,
                surface_distance=surface_distance,
                rightward_displacement=rightward_displacement,
                nominal_command=nominal_command,
                filter_result=filter_result,
            )

        if surface_distance <= GROUND_TRUTH_ABORT_DISTANCE:
            ground_truth_abort_seen = True
            termination_reason = (
                "Ground-truth abort distance reached; fail-closed stop "
                "applied."
            )
            break

        if (
            not filter_result.constraint_feasible
            or filter_result.emergency_stop_active
            or not filter_result.constraint_satisfied
        ):
            emergency_stop_seen = True
            termination_reason = (
                "The PSF-CBF requested an emergency stop."
            )
            break

        if mode == NORMAL_ONLY_MODE:
            stable_blocked_command = (
                filter_result.intervention_active
                and filter_result.safe_linear_velocity
                <= NORMAL_BLOCKED_LINEAR_SPEED
                and abs(filter_result.safe_angular_velocity)
                <= NORMAL_BLOCKED_ANGULAR_SPEED
            )
            if stable_blocked_command:
                blocked_confirmation_count += 1
            else:
                blocked_confirmation_count = 0

            if (
                blocked_confirmation_count
                >= NORMAL_BLOCKED_CONFIRMATION_STEPS
            ):
                termination_reason = (
                    "Normal-only guidance established a stable blocked "
                    "response."
                )
                break

        cleared_proxy = (
            robot_pose.x
            >= (
                proxy_position_world[0]
                + 0.5 * PROXY_DIMENSIONS[0]
                + BYPASS_LONGITUDINAL_MARGIN
            )
        )
        if mode == PASS_RIGHT_MODE and cleared_proxy:
            termination_reason = (
                "Tangential guidance cleared the proxy longitudinally."
            )
            break

        if mode == PASS_RIGHT_MODE:
            settled_tangential_command = (
                filter_result.intervention_active
                and filter_result.safe_linear_velocity
                <= TANGENTIAL_SETTLED_LINEAR_SPEED
                and abs(filter_result.safe_angular_velocity)
                <= TANGENTIAL_SETTLED_ANGULAR_SPEED
            )
            if settled_tangential_command:
                blocked_confirmation_count += 1
            else:
                blocked_confirmation_count = 0

            if (
                blocked_confirmation_count
                >= TANGENTIAL_SETTLED_CONFIRMATION_STEPS
            ):
                termination_reason = (
                    "Tangential closed-loop response settled before "
                    "proxy clearance."
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
        final_pose = read_robot_pose(stage)
    else:
        final_pose = robot_pose

    maximum_rightward_displacement = max(
        maximum_rightward_displacement,
        0.0,
        initial_pose.y - final_pose.y,
    )

    return MotionTrialResult(
        mode=mode,
        termination_reason=termination_reason,
        completed_steps=completed_steps,
        received_scan_count=received_scan_count,
        intervention_seen=intervention_seen,
        first_intervention_step=first_intervention_step,
        first_intervention_surface_distance=(
            first_intervention_surface_distance
        ),
        minimum_surface_distance=minimum_surface_distance,
        maximum_absolute_angular_velocity=(
            maximum_absolute_angular_velocity
        ),
        maximum_absolute_angular_correction=(
            maximum_absolute_angular_correction
        ),
        maximum_requested_side_angular_correction=(
            maximum_requested_side_angular_correction
        ),
        maximum_sampled_guidance_jump=(
            maximum_sampled_guidance_jump
        ),
        maximum_rightward_displacement=(
            maximum_rightward_displacement
        ),
        requested_side_correction_seen=(
            requested_side_correction_seen
        ),
        cleared_proxy=cleared_proxy,
        final_pose=final_pose,
        final_filter_result=filter_result,
        emergency_stop_seen=emergency_stop_seen,
        ground_truth_abort_seen=ground_truth_abort_seen,
        application_stopped=application_stopped,
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
    known_surface_distance = distance_to_proxy_surface(
        robot_pose,
        proxy_position_world=STATIC_PROXY_POSITION_WORLD,
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
        f"{known_surface_distance:.4f} m"
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


def format_optional_distance(
    value: float,
) -> str:
    """Format a possibly unavailable trial distance."""

    if math.isfinite(value):
        return f"{value:.4f} m"
    return "none"


def print_motion_trial_result(
    result: MotionTrialResult,
) -> None:
    """Print one completed Stage C trial summary."""

    print("\nMotion trial result:")
    print(f"  Mode:                         {result.mode}")
    print(f"  Reason:                       {result.termination_reason}")
    print(f"  Completed physics steps:      {result.completed_steps}")
    print(f"  Received LiDAR scans:         {result.received_scan_count}")
    print(f"  Intervention seen:            {result.intervention_seen}")
    print(
        "  First intervention step:      "
        f"{result.first_intervention_step}"
    )
    print(
        "  First intervention distance:  "
        f"{format_optional_distance(result.first_intervention_surface_distance)}"
    )
    print(
        "  Minimum surface distance:     "
        f"{result.minimum_surface_distance:.4f} m"
    )
    print(
        "  Maximum |angular velocity|:   "
        f"{result.maximum_absolute_angular_velocity:.4f} rad/s"
    )
    print(
        "  Maximum |CBF angular change|: "
        f"{result.maximum_absolute_angular_correction:.4f} rad/s"
    )
    print(
        "  Maximum rightward CBF change: "
        f"{result.maximum_requested_side_angular_correction:.4f} rad/s"
    )
    print(
        "  Maximum sampled-guidance jump:"
        f" {result.maximum_sampled_guidance_jump:.4f}"
    )
    print(
        "  Maximum rightward movement:   "
        f"{result.maximum_rightward_displacement:.4f} m"
    )
    print(
        "  Rightward CBF change observed: "
        f"{result.requested_side_correction_seen}"
    )
    print(f"  Proxy cleared:                {result.cleared_proxy}")
    print(
        "  Final safe command:           "
        f"v={result.final_filter_result.safe_linear_velocity:.4f} m/s, "
        f"w={result.final_filter_result.safe_angular_velocity:.4f} rad/s"
    )
    print(
        "  Final look-ahead cell:        "
        f"{result.final_filter_result.lookahead_cell}"
    )
    print(
        "  Final look-ahead occupied:    "
        f"{result.final_filter_result.lookahead_occupied}"
    )
    print(
        "  Final sampled safety h:       "
        f"{result.final_filter_result.sampled_safety_value:.6f}"
    )
    print(
        "  Final sampled guidance:       "
        f"({result.final_filter_result.sampled_guidance_vector_robot_xy[0]:.6f}, "
        f"{result.final_filter_result.sampled_guidance_vector_robot_xy[1]:.6f})"
    )
    print(
        "  Final constraint feasible:    "
        f"{result.final_filter_result.constraint_feasible}"
    )
    print(
        "  Final constraint satisfied:   "
        f"{result.final_filter_result.constraint_satisfied}"
    )
    print(f"  Emergency stop:               {result.emergency_stop_seen}")
    print(f"  Ground-truth abort:           {result.ground_truth_abort_seen}")
    print(
        "  Isaac Sim remained running:   "
        f"{not result.application_stopped}"
    )
    print(
        "  Final pose:                   "
        f"({result.final_pose.x:.4f}, {result.final_pose.y:.4f}, "
        f"{math.degrees(result.final_pose.yaw):.2f} deg)"
    )


def print_stage_c_comparison(
    normal_result: MotionTrialResult,
    tangential_result: MotionTrialResult,
) -> None:
    """Print the normal-only versus tangential motion comparison."""

    normal_blocked = (
        "stable blocked response"
        in normal_result.termination_reason
    )
    both_intervened = (
        normal_result.intervention_seen
        and tangential_result.intervention_seen
    )
    requested_side_correction_seen = (
        tangential_result.requested_side_correction_seen
    )
    tangential_correction_distinct = (
        tangential_result.maximum_requested_side_angular_correction
        > normal_result.maximum_absolute_angular_correction
        + ANGULAR_RESPONSE_TOLERANCE
    )
    tangential_cleared = tangential_result.cleared_proxy
    tangential_terminated_cleanly = (
        tangential_cleared
        or "response settled"
        in tangential_result.termination_reason
        or "waypoint route completed"
        in tangential_result.termination_reason
    )
    safe_separation = (
        not normal_result.ground_truth_abort_seen
        and not tangential_result.ground_truth_abort_seen
    )
    no_emergency = (
        not normal_result.emergency_stop_seen
        and not tangential_result.emergency_stop_seen
    )
    application_remained_running = (
        not normal_result.application_stopped
        and not tangential_result.application_stopped
    )

    print("\n" + "=" * 70)
    print("TANGENTIAL-LGF STAGE C - REACTIVE MOTION ABLATION")
    print("=" * 70)
    print_motion_trial_result(normal_result)
    print_motion_trial_result(tangential_result)

    print("\nComparison checks:")
    print(f"  Both trials intervened:        {both_intervened}")
    print(f"  Normal-only became blocked:    {normal_blocked}")
    print(
        "  Rightward CBF change observed: "
        f"{requested_side_correction_seen}"
    )
    print(
        "  Tangential correction distinct: "
        f"{tangential_correction_distinct}"
    )
    print(f"  Tangential trial cleared proxy:{tangential_cleared}")
    print(
        "  Tangential termination valid: "
        f"{tangential_terminated_cleanly}"
    )
    print(f"  No ground-truth abort:         {safe_separation}")
    print(f"  No emergency stop:             {no_emergency}")
    print(
        "  Isaac Sim remained running:   "
        f"{application_remained_running}"
    )

    checks = (
        both_intervened,
        normal_blocked,
        requested_side_correction_seen,
        tangential_correction_distinct,
        tangential_terminated_cleanly,
        safe_separation,
        no_emergency,
        application_remained_running,
    )

    print("\nInterpretation:")
    if all(checks):
        print(
            "  [PASS] Normal-only guidance produced a blocked response, "
            "while the active tangential constraint produced a distinct "
            "rightward CBF correction from the same starting pose."
        )
        if tangential_cleared:
            print("  [PASS] The tangential trial also cleared the proxy.")
        else:
            print(
                "  [EXPECTED LIMITATION] The reactive ablation did not "
                "clear the proxy. Physical bypass is not an acceptance "
                "criterion until h_full and predictive MPC are integrated."
            )
    else:
        print(
            "  [WARNING] The tangential field did not produce the expected "
            "requested-side CBF correction. Inspect the guidance sign and "
            "filter projection before proceeding."
        )

    print(
        "  Rightward displacement and proxy clearance are reported only "
        "as observations for this geometric-h/reactive-filter milestone."
    )
    print(
        "  Sampled-guidance jumps are diagnostic evidence for later "
        "occupancy-memory or temporal-smoothing work."
    )

    print(
        "  Ground truth was used only for evaluation and the independent "
        "abort guard."
    )
    print("=" * 70)


def main() -> None:
    """Run stationary and closed-loop tangential-LGF validation."""

    print("\n" + "=" * 70)
    print("NOVA CARTER TANGENTIAL-LGF INTEGRATION DEMO")
    print("=" * 70)
    print("Mode: stationary Stage A/B plus Stage C reactive ablation")
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
    proxy_translate_op = create_static_proxy(loaded_scene.stage)

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
    motion_detection_config = replace(
        detection_config,
        forward_only=False,
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
        proxy_position_world=STATIC_PROXY_POSITION_WORLD,
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
    motion_grid_config = replace(
        grid_config,
        minimum_x=MOTION_GRID_MINIMUM_X,
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

    print("\nPreparing Stage C controlled motion trials...")
    print(
        "  Moving proxy center to: "
        f"({MOTION_PROXY_POSITION_WORLD[0]:.3f}, "
        f"{MOTION_PROXY_POSITION_WORLD[1]:.3f}) m"
    )
    print(
        "  Requested tangential passing side: "
        f"{PassingSide.RIGHT.value}"
    )
    print(
        "  Shared Stage C CBF gamma: "
        f"{STAGE_C_PSF_CBF_GAMMA:.3f}"
    )
    print(
        "  Independent abort distance: "
        f"{GROUND_TRUTH_ABORT_DISTANCE:.3f} m"
    )
    set_proxy_position(
        loaded_scene.stage,
        proxy_translate_op,
        MOTION_PROXY_POSITION_WORLD,
    )
    robot_controller.stop()
    app_utils.update_app(steps=30)

    motion_safety_filter = PSFCBFSafetyFilter(
        replace(
            safety_filter.config,
            gamma=STAGE_C_PSF_CBF_GAMMA,
        )
    )

    normal_motion_result = run_motion_trial(
        NORMAL_ONLY_MODE,
        stage=loaded_scene.stage,
        lidar=lidar,
        robot_controller=robot_controller,
        safety_filter=motion_safety_filter,
        proxy_position_world=MOTION_PROXY_POSITION_WORLD,
        robot_from_lidar=robot_from_lidar,
        detection_config=motion_detection_config,
        class_policy=class_policy,
        guidance_policy=right_policy,
        grid_config=motion_grid_config,
        poisson_config=poisson_config,
        field_config=field_config,
    )
    if normal_motion_result.application_stopped:
        print_motion_trial_result(normal_motion_result)
        raise RuntimeError(
            "Stage C cannot continue because Isaac Sim stopped during "
            "the normal-only trial."
        )

    tangential_motion_result = run_motion_trial(
        PASS_RIGHT_MODE,
        stage=loaded_scene.stage,
        lidar=lidar,
        robot_controller=robot_controller,
        safety_filter=motion_safety_filter,
        proxy_position_world=MOTION_PROXY_POSITION_WORLD,
        robot_from_lidar=robot_from_lidar,
        detection_config=motion_detection_config,
        class_policy=class_policy,
        guidance_policy=right_policy,
        grid_config=motion_grid_config,
        poisson_config=poisson_config,
        field_config=field_config,
    )
    print_stage_c_comparison(
        normal_motion_result,
        tangential_motion_result,
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
