"""Validate the Risk-Aware PSF/LGF class-aware CBF baseline.

This controlled Isaac Sim demo uses one session-only static cube as a semantic
proxy. The cube has no physics collider and exists only for LiDAR perception
and safety-filter validation.

Stage A - perception and geometry:

1. Load the restaurant robot scene.
2. Position Nova Carter in the spacious test area.
3. Create a static proxy directly ahead of the robot.
4. Capture one PandarXT-32 scan and extract obstacle points.
5. Cluster the scan and associate the proxy with one real LiDAR track.
6. Retain only that track for this controlled semantic comparison.

Stage B - stationary risk-aware counterfactual:

1. Attach ``chair`` and ``person`` labels to the same geometric track.
2. Build two occupancy grids from the exact same LiDAR points.
3. Solve one shared Poisson safety function.
4. Solve class-dependent Laplace guidance fields.
5. Filter the same nominal waypoint command under both labels.
6. Compare flux, guidance magnitude, and safe commands.

Ground-truth proxy position is used only to select the controlled test track.
It is not passed to the obstacle tracker, PDE solvers, or PSF-CBF filter. The
semantic labels are manual by design; camera/YOLO fusion is a later stage.

This demo implements the normal-flux Risk-Aware PSF/LGF baseline. It does not
claim to implement Safe-SAGE's tangential social-flow interface, yaw-indexed
PSF, temporal derivative, or MPC safety layer.

Stage C - controlled motion comparison:

1. Move the proxy farther forward to expose intervention timing.
2. Apply the ``chair`` filter from the controlled robot start pose.
3. Reset the robot and all tracker state.
4. Apply the ``person`` filter from the identical start pose.
5. Compare intervention and controlled-slowdown surface distances.

LiDAR staleness, lost proxy association, PDE failure, an infeasible filter,
or the experiment-level ground-truth guard all produce a fail-closed stop.

Run from the project root:

    python -u scripts/demos/psf_class_aware_cbf_demo.py

Press Play in Isaac Sim when requested.
"""

import math
import sys
import traceback
from dataclasses import dataclass
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
from robot.safety.class_safety_policy import (
    ClassSafetyParameters,
    ClassSafetyPolicy,
)
from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldConfig,
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
from robot.safety.semantic_obstacle import SemanticObstacle
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
# Controlled scenario configuration
# ---------------------------------------------------------------------------


CLASS_SAFETY_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "class_safety.json"
)

STATIC_PROXY_PRIM_PATH = "/World/PSFClassAwareTestObstacle"

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

# The proxy is close enough to exercise the filter while the robot is held
# stationary. Its nearest face begins about 0.70 m ahead of the chassis.
STATIONARY_PROXY_POSITION_WORLD = np.array(
    [
        27.35,
        5.00175,
        0.50,
    ],
    dtype=np.float64,
)

# Stage C moves the same proxy farther forward so both trials begin with an
# unmodified nominal command and class-dependent intervention timing can be
# measured during approach.
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
    0.95,
    0.15,
    0.05,
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

PSF_CBF_GAMMA = 1.0
PSF_CBF_LOOKAHEAD_DISTANCE = 0.30

MOTION_TRIAL_CLASSES = (
    "chair",
    "person",
)
MOTION_TRIAL_SETTLE_STEPS = 60
MAXIMUM_MOTION_TRIAL_SECONDS = 15.0
MAXIMUM_MOTION_TRIAL_STEPS = math.ceil(
    MAXIMUM_MOTION_TRIAL_SECONDS / PHYSICS_DT
)
MAXIMUM_STALE_SCAN_SECONDS = 0.30
MAXIMUM_STALE_SCAN_STEPS = math.ceil(
    MAXIMUM_STALE_SCAN_SECONDS / PHYSICS_DT
)
STATUS_REPORT_INTERVAL_STEPS = 30
EVALUATION_LINEAR_SPEED_THRESHOLD = 0.13
EVALUATION_CONFIRMATION_STEPS = 12
GROUND_TRUTH_ABORT_DISTANCE = 0.35

TRANSIENT_LIDAR_ERRORS = (
    "No LiDAR data is available yet",
    "contains no points",
    "no valid XYZ points remained",
)


@dataclass(frozen=True)
class SemanticEvaluation:
    """One counterfactual class evaluation of the same LiDAR track."""

    semantic_class: str
    parameters: ClassSafetyParameters
    occupancy_grid: LocalOccupancyGrid
    guidance_field: LaplaceGuidanceFieldResult
    filter_result: PSFCBFResult


@dataclass(frozen=True)
class PipelineFrame:
    """One LiDAR-rate PSF pipeline update for a selected class."""

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
    """Summary of one applied semantic-class motion trial."""

    semantic_class: str
    termination_reason: str
    completed_steps: int
    received_scan_count: int
    route_complete: bool
    intervention_seen: bool
    first_intervention_step: int | None
    first_intervention_surface_distance: float
    minimum_surface_distance: float
    final_surface_distance: float
    final_pose: RobotPose2D
    final_filter_result: PSFCBFResult
    emergency_stop_seen: bool
    ground_truth_abort_seen: bool
    application_stopped: bool


# ---------------------------------------------------------------------------
# Scene and perception helpers
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
    """Create the proxy and return its session-layer translate operation."""

    existing_prim = stage.GetPrimAtPath(
        STATIC_PROXY_PRIM_PATH
    )
    if existing_prim.IsValid():
        raise RuntimeError(
            "The PSF test proxy already exists: "
            f"{STATIC_PROXY_PRIM_PATH}"
        )

    with Usd.EditContext(
        stage,
        stage.GetSessionLayer(),
    ):
        cube = UsdGeom.Cube.Define(
            stage,
            STATIC_PROXY_PRIM_PATH,
        )
        cube.CreateSizeAttr(1.0)
        cube.CreateExtentAttr(
            [
                Gf.Vec3f(-0.5, -0.5, -0.5),
                Gf.Vec3f(0.5, 0.5, 0.5),
            ]
        )
        cube.CreateDisplayColorAttr(
            [Gf.Vec3f(*PROXY_COLOR)]
        )
        translate_op = cube.AddTranslateOp()
        translate_op.Set(
            Gf.Vec3d(
                float(STATIONARY_PROXY_POSITION_WORLD[0]),
                float(STATIONARY_PROXY_POSITION_WORLD[1]),
                float(STATIONARY_PROXY_POSITION_WORLD[2]),
            )
        )
        cube.AddOrientOp().Set(Gf.Quatf(1.0))
        cube.AddScaleOp().Set(
            Gf.Vec3f(*PROXY_DIMENSIONS)
        )

    return translate_op


def set_proxy_position(
    stage: Usd.Stage,
    translate_op: UsdGeom.XformOp,
    position_world: np.ndarray,
) -> None:
    """Move the session-only proxy to one controlled world position."""

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


# ---------------------------------------------------------------------------
# Semantic PSF evaluation
# ---------------------------------------------------------------------------


def build_semantic_grid(
    semantic_class: str,
    *,
    proxy_track: TrackedObstacle,
    proxy_points: np.ndarray,
    proxy_point_track_ids: np.ndarray,
    class_policy: ClassSafetyPolicy,
    grid_config: LocalOccupancyGridConfig,
) -> LocalOccupancyGrid:
    """Build one counterfactual grid from the same proxy geometry."""

    semantic_obstacle = SemanticObstacle(
        tracked_obstacle=proxy_track,
        semantic_class=semantic_class,
        semantic_confidence=1.0,
        label_source="manual",
    )

    return build_local_occupancy_grid(
        proxy_points,
        point_track_ids=proxy_point_track_ids,
        semantic_obstacles=(semantic_obstacle,),
        class_safety_policy=class_policy,
        config=grid_config,
    )


def evaluate_semantic_class(
    semantic_class: str,
    *,
    occupancy_grid: LocalOccupancyGrid,
    safety_function: PoissonSafetyFunctionResult,
    nominal_command: WaypointCommand,
    class_policy: ClassSafetyPolicy,
    safety_filter: PSFCBFSafetyFilter,
    laplace_config: LaplaceGuidanceFieldConfig,
) -> SemanticEvaluation:
    """Solve guidance and filter one counterfactual class command."""

    guidance_field = solve_laplace_guidance_field(
        occupancy_grid,
        laplace_config,
    )
    filter_result = safety_filter.filter_command(
        nominal_command.linear_velocity,
        nominal_command.angular_velocity,
        occupancy_grid,
        safety_function,
        guidance_field,
    )

    return SemanticEvaluation(
        semantic_class=semantic_class,
        parameters=class_policy.parameters_for(semantic_class),
        occupancy_grid=occupancy_grid,
        guidance_field=guidance_field,
        filter_result=filter_result,
    )


def create_trial_waypoint_controller() -> WaypointController:
    """Create a fresh nominal controller for one motion trial."""

    return WaypointController(
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
            rotate_in_place_threshold_radians=math.radians(25.0),
        ),
    )


def build_pipeline_frame(
    semantic_class: str,
    *,
    points_sensor_frame: np.ndarray,
    frame_info: LidarFrameInfo,
    robot_pose: RobotPose2D,
    obstacle_tracker: ObstacleTracker,
    proxy_position_world: np.ndarray,
    robot_from_lidar: np.ndarray,
    detection_config: ObstacleDetectionConfig,
    class_policy: ClassSafetyPolicy,
    grid_config: LocalOccupancyGridConfig,
    poisson_config: PoissonSafetyFunctionConfig,
    laplace_config: LaplaceGuidanceFieldConfig,
) -> PipelineFrame:
    """Run one LiDAR-rate class-aware PSF update."""

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
        tracking_result.point_track_ids
        == proxy_track.track_id
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

    occupancy_grid = build_semantic_grid(
        semantic_class,
        proxy_track=proxy_track,
        proxy_points=proxy_points,
        proxy_point_track_ids=proxy_track_ids,
        class_policy=class_policy,
        grid_config=grid_config,
    )
    safety_function = solve_poisson_safety_function(
        occupancy_grid,
        poisson_config,
    )
    guidance_field = solve_laplace_guidance_field(
        occupancy_grid,
        laplace_config,
    )

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
    semantic_class: str,
    completed_steps: int,
    pipeline_frame: PipelineFrame,
    robot_pose: RobotPose2D,
    surface_distance: float,
    nominal_command: WaypointCommand,
    filter_result: PSFCBFResult,
) -> None:
    """Print one compact Stage C motion status line."""

    print(
        f"  class={semantic_class:<6} "
        f"step={completed_steps:<4} "
        f"scan={pipeline_frame.sensor_frame_id:<4} "
        f"x={robot_pose.x:.3f} "
        f"surface={surface_distance:.3f} "
        f"v_nom={nominal_command.linear_velocity:.3f} "
        f"v_safe={filter_result.safe_linear_velocity:.3f} "
        f"w_safe={filter_result.safe_angular_velocity:.3f} "
        f"h={filter_result.sampled_safety_value:.3f} "
        f"|g|={filter_result.sampled_guidance_magnitude:.3f} "
        f"intervention={filter_result.intervention_active}"
    )


def run_motion_trial(
    semantic_class: str,
    *,
    stage: Usd.Stage,
    lidar: NovaCarterLidar,
    robot_controller: NovaCarterController,
    safety_filter: PSFCBFSafetyFilter,
    proxy_position_world: np.ndarray,
    robot_from_lidar: np.ndarray,
    detection_config: ObstacleDetectionConfig,
    class_policy: ClassSafetyPolicy,
    grid_config: LocalOccupancyGridConfig,
    poisson_config: PoissonSafetyFunctionConfig,
    laplace_config: LaplaceGuidanceFieldConfig,
) -> MotionTrialResult:
    """Run one resettable closed-loop semantic-class motion trial."""

    print("\n" + "-" * 70)
    print(f"STAGE C MOTION TRIAL: {semantic_class.upper()}")
    print("-" * 70)

    position_robot_for_experiment(robot_controller)
    verify_robot_start_pose(stage)

    if not advance_playing_steps(
        simulation_app,
        physics_steps=MOTION_TRIAL_SETTLE_STEPS,
        on_step=robot_controller.stop,
    ):
        raise RuntimeError(
            "Isaac Sim stopped while preparing a motion trial."
        )

    waypoint_controller = create_trial_waypoint_controller()
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
        semantic_class,
        points_sensor_frame=points_sensor_frame,
        frame_info=frame_info,
        robot_pose=robot_pose,
        obstacle_tracker=obstacle_tracker,
        proxy_position_world=proxy_position_world,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
        class_policy=class_policy,
        grid_config=grid_config,
        poisson_config=poisson_config,
        laplace_config=laplace_config,
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
    slowdown_confirmation_count = 0
    intervention_seen = filter_result.intervention_active
    first_intervention_step = (
        0 if intervention_seen else None
    )
    initial_surface_distance = distance_to_proxy_surface(
        robot_pose,
        proxy_position_world=proxy_position_world,
    )
    first_intervention_surface_distance = (
        initial_surface_distance
        if intervention_seen
        else math.nan
    )
    minimum_surface_distance = initial_surface_distance
    previous_intervention = filter_result.intervention_active
    emergency_stop_seen = False
    ground_truth_abort_seen = False
    application_stopped = False
    termination_reason = "Maximum motion-trial duration reached."

    print("\nLive motion status:")
    print_motion_status(
        semantic_class=semantic_class,
        completed_steps=0,
        pipeline_frame=pipeline_frame,
        robot_pose=robot_pose,
        surface_distance=initial_surface_distance,
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
        nominal_command = waypoint_controller.compute_command(
            robot_pose
        )
        if nominal_command.route_complete:
            termination_reason = "The waypoint route completed."
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
                        semantic_class,
                        points_sensor_frame=current_points_sensor,
                        frame_info=current_frame_info,
                        robot_pose=robot_pose,
                        obstacle_tracker=obstacle_tracker,
                        proxy_position_world=proxy_position_world,
                        robot_from_lidar=robot_from_lidar,
                        detection_config=detection_config,
                        class_policy=class_policy,
                        grid_config=grid_config,
                        poisson_config=poisson_config,
                        laplace_config=laplace_config,
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
        minimum_surface_distance = min(
            minimum_surface_distance,
            surface_distance,
        )

        if filter_result.intervention_active and not intervention_seen:
            intervention_seen = True
            first_intervention_step = completed_steps
            first_intervention_surface_distance = surface_distance

        intervention_changed = (
            filter_result.intervention_active
            != previous_intervention
        )
        periodic_report = (
            completed_steps % STATUS_REPORT_INTERVAL_STEPS == 0
        )

        if periodic_report or intervention_changed:
            print_motion_status(
                semantic_class=semantic_class,
                completed_steps=completed_steps,
                pipeline_frame=pipeline_frame,
                robot_pose=robot_pose,
                surface_distance=surface_distance,
                nominal_command=nominal_command,
                filter_result=filter_result,
            )

        if surface_distance <= GROUND_TRUTH_ABORT_DISTANCE:
            ground_truth_abort_seen = True
            termination_reason = (
                "Ground-truth abort distance reached before a filtered "
                "stop; fail-closed stop applied."
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

        controlled_slowdown = (
            filter_result.intervention_active
            and nominal_command.linear_velocity
            > EVALUATION_LINEAR_SPEED_THRESHOLD
            and filter_result.safe_linear_velocity
            <= EVALUATION_LINEAR_SPEED_THRESHOLD
        )
        if controlled_slowdown:
            slowdown_confirmation_count += 1
        else:
            slowdown_confirmation_count = 0

        if (
            slowdown_confirmation_count
            >= EVALUATION_CONFIRMATION_STEPS
        ):
            termination_reason = (
                "The PSF-CBF maintained the controlled slowdown "
                "threshold."
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
        # The USD stage may fall back to authored transforms during shutdown.
        # Retain the latest pose observed while the application was valid.
        final_pose = robot_pose

    final_surface_distance = distance_to_proxy_surface(
        final_pose,
        proxy_position_world=proxy_position_world,
    )

    return MotionTrialResult(
        semantic_class=semantic_class,
        termination_reason=termination_reason,
        completed_steps=completed_steps,
        received_scan_count=received_scan_count,
        route_complete=waypoint_controller.route_complete,
        intervention_seen=intervention_seen,
        first_intervention_step=first_intervention_step,
        first_intervention_surface_distance=(
            first_intervention_surface_distance
        ),
        minimum_surface_distance=minimum_surface_distance,
        final_surface_distance=final_surface_distance,
        final_pose=final_pose,
        final_filter_result=filter_result,
        emergency_stop_seen=emergency_stop_seen,
        ground_truth_abort_seen=ground_truth_abort_seen,
        application_stopped=application_stopped,
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


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
    """Print perception, association, and Poisson validation."""

    nearest_proxy_distance = float(
        np.min(np.linalg.norm(proxy_points, axis=1))
    )
    known_surface_distance = distance_to_proxy_surface(
        robot_pose,
        proxy_position_world=(
            STATIONARY_PROXY_POSITION_WORLD
        ),
    )

    print("\n" + "=" * 70)
    print("RISK-AWARE PSF/LGF STAGE A - PERCEPTION AND GEOMETRY")
    print("=" * 70)

    print("\nRobot pose:")
    print(f"  x:                    {robot_pose.x:.4f} m")
    print(f"  y:                    {robot_pose.y:.4f} m")
    print(
        "  yaw:                  "
        f"{math.degrees(robot_pose.yaw):.2f} degrees"
    )

    print("\nLiDAR and tracking:")
    print(f"  Frame ID:              {frame_info.sensor_frame_id}")
    print(f"  Waited updates:         {scan_wait_steps}")
    print(f"  All obstacle points:    {obstacle_points.shape[0]}")
    print(f"  Detected clusters:      {tracking_result.cluster_count}")
    print(f"  Proxy track ID:         {proxy_track.track_id}")
    print(f"  Proxy track points:     {proxy_points.shape[0]}")
    print(
        "  Proxy association err: "
        f"{proxy_association_error:.4f} m"
    )
    print(
        "  Nearest proxy return:  "
        f"{nearest_proxy_distance:.4f} m"
    )
    print(
        "  Known surface distance:"
        f" {known_surface_distance:.4f} m"
    )

    print("\nLocal grid and Poisson solve:")
    print(f"  Grid shape:             {occupancy_grid.shape}")
    print(
        "  Resolution:             "
        f"{occupancy_grid.config.resolution:.3f} m"
    )
    print(
        "  Occupied cells:         "
        f"{occupancy_grid.occupied_cell_count}"
    )
    print(
        "  Poisson iterations:     "
        f"{safety_function.iteration_count}"
    )
    print(
        "  Poisson max residual:   "
        f"{safety_function.maximum_residual:.3e}"
    )

    print("\nControlled-data boundary:")
    print("  [PASS] A real LiDAR track was associated with the proxy.")
    print("  Only proxy-track points enter this semantic comparison.")
    print("  Ground truth is not used by the tracker or safety filter.")
    print("  The robot remained stationary during Stage A.")
    print("=" * 70)


def print_evaluation(
    evaluation: SemanticEvaluation,
) -> None:
    """Print one class-specific counterfactual result."""

    result = evaluation.filter_result

    print(f"\n{evaluation.semantic_class.upper()} evaluation:")
    print(f"  Fixed risk:             {evaluation.parameters.risk:.3f}")
    print(
        "  Boundary flux:          "
        f"{evaluation.parameters.boundary_flux:.3f}"
    )
    print(
        "  Laplace iterations:     "
        f"{evaluation.guidance_field.iteration_count}"
    )
    print(
        "  Laplace max residual:   "
        f"{evaluation.guidance_field.maximum_residual:.3e}"
    )
    print(
        "  Sampled safety h:       "
        f"{result.sampled_safety_value:.6f}"
    )
    print(
        "  Sampled guidance:       "
        f"({result.sampled_guidance_vector_robot_xy[0]:.6f}, "
        f"{result.sampled_guidance_vector_robot_xy[1]:.6f})"
    )
    print(
        "  Guidance magnitude:     "
        f"{result.sampled_guidance_magnitude:.6f}"
    )
    print(
        "  Safe linear velocity:   "
        f"{result.safe_linear_velocity:.6f} m/s"
    )
    print(
        "  Safe angular velocity:  "
        f"{result.safe_angular_velocity:.6f} rad/s"
    )
    print(
        "  Constraint margin:      "
        f"{result.constraint_margin:.3e}"
    )
    print(f"  Intervention:           {result.intervention_active}")
    print(f"  Feasible:               {result.constraint_feasible}")
    print(f"  Emergency stop:         {result.emergency_stop_active}")


def print_stage_b_report(
    *,
    nominal_command: WaypointCommand,
    chair_evaluation: SemanticEvaluation,
    person_evaluation: SemanticEvaluation,
) -> None:
    """Print and interpret the risk-aware counterfactual comparison."""

    chair_result = chair_evaluation.filter_result
    person_result = person_evaluation.filter_result

    same_occupancy = np.array_equal(
        chair_evaluation.occupancy_grid.occupied_mask,
        person_evaluation.occupancy_grid.occupied_mask,
    )
    stronger_person_flux = (
        person_evaluation.parameters.boundary_flux_magnitude
        > chair_evaluation.parameters.boundary_flux_magnitude
    )
    stronger_person_guidance = (
        person_result.sampled_guidance_magnitude
        > chair_result.sampled_guidance_magnitude + 1e-6
    )
    no_faster_person_command = (
        person_result.safe_linear_velocity
        <= chair_result.safe_linear_velocity + 1e-6
    )
    constraints_satisfied = (
        chair_result.constraint_satisfied
        and person_result.constraint_satisfied
    )

    print("\n" + "=" * 70)
    print("RISK-AWARE PSF/LGF STAGE B - STATIONARY COUNTERFACTUAL")
    print("=" * 70)

    print("\nShared nominal waypoint command:")
    print(
        "  Linear velocity:        "
        f"{nominal_command.linear_velocity:.6f} m/s"
    )
    print(
        "  Angular velocity:       "
        f"{nominal_command.angular_velocity:.6f} rad/s"
    )
    print(f"  Waypoint mode:          {nominal_command.mode.value}")

    print_evaluation(chair_evaluation)
    print_evaluation(person_evaluation)

    print("\nControlled comparison checks:")
    print(f"  Same occupancy geometry: {same_occupancy}")
    print(f"  Person flux is stronger: {stronger_person_flux}")
    print(f"  Person guidance stronger:{stronger_person_guidance}")
    print(f"  Person command no faster:{no_faster_person_command}")
    print(f"  Both constraints valid:  {constraints_satisfied}")

    checks = (
        same_occupancy,
        stronger_person_flux,
        stronger_person_guidance,
        no_faster_person_command,
        constraints_satisfied,
    )

    print("\nInterpretation:")
    if all(checks):
        print(
            "  [PASS] Identical geometry produced different behavior "
            "only because the semantic boundary flux changed."
        )
    else:
        print(
            "  [WARNING] The class-aware comparison was not fully "
            "separated. Inspect the checks above before adding motion."
        )

    print("  The robot remained stationary during Stage B.")
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
    print(f"  Class:                       {result.semantic_class}")
    print(f"  Reason:                      {result.termination_reason}")
    print(f"  Completed physics steps:     {result.completed_steps}")
    print(f"  Received LiDAR scans:        {result.received_scan_count}")
    print(f"  Route complete:              {result.route_complete}")
    print(f"  Intervention seen:           {result.intervention_seen}")
    print(
        "  First intervention step:     "
        f"{result.first_intervention_step}"
    )
    print(
        "  First intervention distance: "
        f"{format_optional_distance(result.first_intervention_surface_distance)}"
    )
    print(
        "  Minimum surface distance:    "
        f"{result.minimum_surface_distance:.4f} m"
    )
    print(
        "  Final surface distance:      "
        f"{result.final_surface_distance:.4f} m"
    )
    print(
        "  Final safe command:          "
        f"v={result.final_filter_result.safe_linear_velocity:.4f} m/s, "
        f"w={result.final_filter_result.safe_angular_velocity:.4f} rad/s"
    )
    print(f"  Emergency stop:              {result.emergency_stop_seen}")
    print(f"  Ground-truth abort:          {result.ground_truth_abort_seen}")
    print(f"  Isaac Sim remained running:  {not result.application_stopped}")
    print(
        "  Final pose:                  "
        f"({result.final_pose.x:.4f}, {result.final_pose.y:.4f}, "
        f"{math.degrees(result.final_pose.yaw):.2f} deg)"
    )


def print_stage_c_comparison(
    chair_result: MotionTrialResult,
    person_result: MotionTrialResult,
) -> None:
    """Print controlled motion comparison and expected ordering."""

    both_intervened = (
        chair_result.intervention_seen
        and person_result.intervention_seen
    )
    person_intervened_no_later = (
        math.isfinite(
            chair_result.first_intervention_surface_distance
        )
        and math.isfinite(
            person_result.first_intervention_surface_distance
        )
        and (
            person_result.first_intervention_surface_distance
            >= (
                chair_result.first_intervention_surface_distance
                - 0.03
            )
        )
    )
    person_finished_no_closer = (
        person_result.final_surface_distance
        >= chair_result.final_surface_distance - 0.03
    )
    no_abort = (
        not chair_result.ground_truth_abort_seen
        and not person_result.ground_truth_abort_seen
    )
    no_emergency = (
        not chair_result.emergency_stop_seen
        and not person_result.emergency_stop_seen
    )
    both_reached_slowdown_threshold = (
        "controlled slowdown threshold"
        in chair_result.termination_reason
        and "controlled slowdown threshold"
        in person_result.termination_reason
    )
    application_remained_running = (
        not chair_result.application_stopped
        and not person_result.application_stopped
    )
    person_minimum_no_closer = (
        person_result.minimum_surface_distance
        >= chair_result.minimum_surface_distance - 0.03
    )

    print("\n" + "=" * 70)
    print("RISK-AWARE PSF/LGF STAGE C - CONTROLLED MOTION COMPARISON")
    print("=" * 70)
    print_motion_trial_result(chair_result)
    print_motion_trial_result(person_result)

    print("\nComparison checks:")
    print(f"  Both classes intervened:       {both_intervened}")
    print(
        "  Person intervened no later:   "
        f"{person_intervened_no_later}"
    )
    print(
        "  Person finished no closer:    "
        f"{person_finished_no_closer}"
    )
    print(
        "  Person minimum no closer:     "
        f"{person_minimum_no_closer}"
    )
    print(
        "  Both reached slowdown limit:  "
        f"{both_reached_slowdown_threshold}"
    )
    print(f"  No ground-truth abort:         {no_abort}")
    print(f"  No emergency stop:             {no_emergency}")
    print(
        "  Isaac Sim remained running:   "
        f"{application_remained_running}"
    )

    print("\nInterpretation:")
    if all(
        (
            both_intervened,
            person_intervened_no_later,
            person_finished_no_closer,
            person_minimum_no_closer,
            both_reached_slowdown_threshold,
            no_abort,
            no_emergency,
            application_remained_running,
        )
    ):
        print(
            "  [PASS] The stronger person flux produced a more "
            "conservative closed-loop response from the same start pose."
        )
    else:
        print(
            "  [WARNING] Stage C did not satisfy every expected "
            "ordering. Inspect the trial reasons and live status."
        )

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Risk-Aware PSF/LGF CBF validation stages."""

    print("\n" + "=" * 70)
    print("NOVA CARTER RISK-AWARE PSF/LGF CBF DEMO")
    print("=" * 70)
    print("Mode: stationary Stage A/B validation")
    print(f"Class policy: {CLASS_SAFETY_CONFIG_PATH}")

    class_policy = ClassSafetyPolicy.from_json(
        CLASS_SAFETY_CONFIG_PATH
    )
    for required_class in ("chair", "person"):
        if required_class not in class_policy.class_parameters:
            raise RuntimeError(
                "Class safety policy is missing required class: "
                f"{required_class}"
            )

    loaded_scene = load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=EXPECTED_METERS_PER_UNIT,
    )
    proxy_translate_op = create_static_proxy(
        loaded_scene.stage
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
        proxy_position_world=(
            STATIONARY_PROXY_POSITION_WORLD
        ),
    )
    if proxy_track is None:
        raise RuntimeError(
            "No LiDAR track matched the controlled proxy. Nearest "
            f"association error was {proxy_error:.3f} m."
        )

    proxy_point_mask = (
        tracking_result.point_track_ids
        == proxy_track.track_id
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

    grid_config = LocalOccupancyGridConfig(
        resolution=GRID_RESOLUTION,
        minimum_x=GRID_MINIMUM_X,
        maximum_x=GRID_MAXIMUM_X,
        minimum_y=GRID_MINIMUM_Y,
        maximum_y=GRID_MAXIMUM_Y,
        obstacle_inflation_radius=(
            GRID_OBSTACLE_INFLATION_RADIUS
        ),
    )
    chair_grid = build_semantic_grid(
        "chair",
        proxy_track=proxy_track,
        proxy_points=proxy_points,
        proxy_point_track_ids=proxy_point_track_ids,
        class_policy=class_policy,
        grid_config=grid_config,
    )
    person_grid = build_semantic_grid(
        "person",
        proxy_track=proxy_track,
        proxy_points=proxy_points,
        proxy_point_track_ids=proxy_point_track_ids,
        class_policy=class_policy,
        grid_config=grid_config,
    )

    if not np.array_equal(
        chair_grid.occupied_mask,
        person_grid.occupied_mask,
    ):
        raise RuntimeError(
            "Counterfactual semantic grids changed occupancy geometry."
        )

    poisson_config = PoissonSafetyFunctionConfig(
        forcing=PSF_FORCING,
        convergence_tolerance=PSF_CONVERGENCE_TOLERANCE,
    )
    safety_function = solve_poisson_safety_function(
        chair_grid,
        poisson_config,
    )
    laplace_config = LaplaceGuidanceFieldConfig(
        convergence_tolerance=(
            LAPLACE_CONVERGENCE_TOLERANCE
        )
    )

    waypoint_controller = create_trial_waypoint_controller()
    nominal_command = waypoint_controller.compute_command(
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

    chair_evaluation = evaluate_semantic_class(
        "chair",
        occupancy_grid=chair_grid,
        safety_function=safety_function,
        nominal_command=nominal_command,
        class_policy=class_policy,
        safety_filter=safety_filter,
        laplace_config=laplace_config,
    )
    person_evaluation = evaluate_semantic_class(
        "person",
        occupancy_grid=person_grid,
        safety_function=safety_function,
        nominal_command=nominal_command,
        class_policy=class_policy,
        safety_filter=safety_filter,
        laplace_config=laplace_config,
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
        occupancy_grid=chair_grid,
        safety_function=safety_function,
    )
    print_stage_b_report(
        nominal_command=nominal_command,
        chair_evaluation=chair_evaluation,
        person_evaluation=person_evaluation,
    )

    print("\nPreparing Stage C controlled motion trials...")
    print(
        "  Moving proxy center to: "
        f"({MOTION_PROXY_POSITION_WORLD[0]:.3f}, "
        f"{MOTION_PROXY_POSITION_WORLD[1]:.3f}) m"
    )
    print(
        "  Controlled slowdown threshold: "
        f"{EVALUATION_LINEAR_SPEED_THRESHOLD:.3f} m/s for "
        f"{EVALUATION_CONFIRMATION_STEPS} physics steps"
    )
    set_proxy_position(
        loaded_scene.stage,
        proxy_translate_op,
        MOTION_PROXY_POSITION_WORLD,
    )
    robot_controller.stop()
    app_utils.update_app(steps=30)

    motion_results: dict[str, MotionTrialResult] = {}
    for semantic_class in MOTION_TRIAL_CLASSES:
        result = run_motion_trial(
            semantic_class,
            stage=loaded_scene.stage,
            lidar=lidar,
            robot_controller=robot_controller,
            safety_filter=safety_filter,
            proxy_position_world=MOTION_PROXY_POSITION_WORLD,
            robot_from_lidar=robot_from_lidar,
            detection_config=detection_config,
            class_policy=class_policy,
            grid_config=grid_config,
            poisson_config=poisson_config,
            laplace_config=laplace_config,
        )
        motion_results[semantic_class] = result

        if result.application_stopped:
            print_motion_trial_result(result)
            raise RuntimeError(
                "Stage C cannot continue because Isaac Sim stopped "
                "running. The report above preserves the last valid "
                "robot pose."
            )

    print_stage_c_comparison(
        motion_results["chair"],
        motion_results["person"],
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
        print("ERROR DURING RISK-AWARE PSF/LGF CBF DEMO")
        print("=" * 70)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
