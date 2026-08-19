"""Validate real YOLO semantics entering Safe-SAGE field construction.

Stage A - synchronized perception:

1. Hold Nova Carter stationary in the restaurant scene.
2. Capture paired front-camera and PandarXT-32 observations.
3. Extract and track real LiDAR obstacle returns exactly once per scan.
4. Associate YOLO instance masks with those existing track identifiers.

Stage B - temporal semantic policy:

1. Accumulate repeated class evidence on persistent LiDAR track IDs.
2. Keep unconfirmed, ambiguous, or missing labels as ``unknown``.
3. Resolve every current track through the configured class-risk policy.

Stage C - Safe-SAGE field integration:

1. Build occupancy, yaw-dependent configuration space, Laplace guidance, and
   ``h_full(x, y, psi)`` from the temporally labeled tracks.
2. Commit tracker state, semantic memory, and the complete field together.
3. Verify multiple newer fields, at least one real confirmed policy class,
   valid numerical fields, and a stationary robot.

Stages A--C intentionally send no command to MPC, CBF, or the wheels. After
that stationary gate passes, Stage D executes one short obstacle-unaware
waypoint through live semantic fields, Safe-SAGE MPC, the full-rate analytical
CBF, and Nova Carter's wheel controller. No bypass reference path is created.

Run from the project root in the cloned YOLO environment:

    $env:YOLO_MODEL_PATH = "C:\\path\\to\\yolo11n-seg.pt"
    python -u scripts/demos/yolo_safe_sage_integration_demo.py

Set ``ISAAC_SIM_HEADLESS=1`` for a headless run, or
``YOLO_SAFE_SAGE_AUTOCLOSE=1`` to close after a rendered report.
"""

import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from robot.config import (
    EXPECTED_METERS_PER_UNIT,
    FRONT_HAWK_LEFT_CAMERA_PRIM_PATH,
    PANDAR_XT32_LIDAR_PRIM_PATH,
    PHYSICS_DEVICE,
    PHYSICS_DT,
    ROBOT_BASE_PRIM_PATH,
    ROBOT_PRIM_PATH,
    ROBOT_SCENE_PATH,
)
from isaacsim import SimulationApp


def environment_flag(name: str) -> bool:
    """Interpret one conventional boolean environment variable."""

    return os.environ.get(name, "0").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


SIMULATION_HEADLESS = environment_flag("ISAAC_SIM_HEADLESS")
AUTOCLOSE_AFTER_REPORT = environment_flag("YOLO_SAFE_SAGE_AUTOCLOSE")
simulation_app = SimulationApp(
    {
        "headless": SIMULATION_HEADLESS,
        "enable_motion_bvh": True,
        "extra_args": ["--/rtx/scenedb/maxHistoryTransformCount=20"],
    }
)


import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager
from PIL import Image
from pxr import Gf, Usd

from robot.control.differential_drive import NovaCarterController
from robot.navigation.waypoint_controller import (
    RobotPose2D,
    Waypoint,
    WaypointController,
    WaypointControllerConfig,
)
from robot.perception.camera_lidar_fusion import (
    CameraLidarFusionConfig,
    CameraLidarSemanticFusion,
)
from robot.perception.safe_sage_semantic_bridge import (
    SafeSAGECameraLidarObservation,
    SafeSAGESemanticBridgeConfig,
    SafeSAGESemanticPerceptionBridge,
    SafeSAGESemanticPerceptionResult,
)
from robot.perception.semantic_track_memory import (
    SemanticTrackMemory,
    SemanticTrackMemoryConfig,
)
from robot.perception.visualization import render_camera_lidar_fusion_overlay
from robot.perception.yolo_detector import (
    YoloSegmentationConfig,
    YoloSegmentationDetector,
)
from robot.runtime.simulation_loop import advance_playing_steps, wait_until_playing
from robot.safety.class_safety_policy import ClassSafetyPolicy
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.obstacle_detection import (
    ObstacleDetectionConfig,
    detect_obstacle_points_3d,
)
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig
from robot.safety.poisson_safety_function import PoissonSafetyFunctionConfig
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
    SafeSAGEConfigurationSpaceConfig,
)
from robot.safety.safe_sage_cbf_filter import (
    SafeSAGECBFConfig,
    SafeSAGECBFSafetyFilter,
)
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidancePolicy
from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
from robot.safety.safe_sage_live_field import SafeSAGELiveFieldConfig
from robot.safety.safe_sage_runtime import (
    SafeSAGEControlRuntime,
    SafeSAGEPerceptionObservation,
    SafeSAGEPerceptionRuntime,
)
from robot.safety.safe_sage_temporal_derivative import (
    SafeSAGETemporalDerivativeConfig,
    SafeSAGETemporalDerivativeEstimator,
)
from robot.safety.safe_sage_unicycle_cbf_filter import (
    SafeSAGEUnicycleCBFConfig,
    SafeSAGEUnicycleCBFFilter,
    SafeSAGEUnicycleCBFResult,
    shift_rectangular_footprint_to_lookahead,
)
from robot.safety.safe_sage_unicycle_mpc import (
    SafeSAGEUnicycleMPC,
    SafeSAGEUnicycleMPCConfig,
    SafeSAGEUnicycleMPCResult,
)
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
)
from robot.scene.camera_calibration import read_fisheye_polynomial_calibration
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import (
    compute_relative_prim_transform,
    get_prim_planar_pose,
    get_prim_world_transform,
)
from robot.sensor.camera import CameraFrameInfo, NovaCarterCamera
from robot.sensor.lidar import LidarFrameInfo, NovaCarterLidar
from robot.sensor.lidar_geometry import transform_points


SAMPLE_COUNT = 4
SETTLE_STEPS = 60
MAXIMUM_NEW_SCAN_WAIT_STEPS = 180
MAXIMUM_SENSOR_TIME_OFFSET_SECONDS = 0.12
MAXIMUM_ALLOWED_POSITION_DRIFT = 0.005
MAXIMUM_ALLOWED_YAW_DRIFT = math.radians(0.10)

MOTION_GOAL_DISTANCE = 0.50
MOTION_MAXIMUM_CONTROL_CYCLES = 15
MOTION_CONTROL_PHYSICS_STEPS = 12
MOTION_MINIMUM_FORWARD_PROGRESS = 0.20
MOTION_MPC_HORIZON_STEPS = 12
MOTION_MPC_TIME_STEP = 0.20

DETECTION_HEIGHT_ABOVE_FLOOR = (0.05, 1.50)
DETECTION_MAXIMUM_RANGE = 5.0
DETECTION_ANGULAR_BIN_DEGREES = 0.25
SEMANTIC_CLUSTER_RADIUS = 0.07

GRID_RESOLUTION = 0.10
GRID_MINIMUM_X = -1.0
GRID_MAXIMUM_X = 5.0
GRID_MINIMUM_Y = -3.0
GRID_MAXIMUM_Y = 3.0
YAW_SLICE_COUNT = 10
UNICYCLE_LOOKAHEAD_DISTANCE = 0.10

ROBOT_FRONT_EXTENT = 0.140000
ROBOT_REAR_EXTENT = 0.587351
ROBOT_LEFT_EXTENT = 0.246600
ROBOT_RIGHT_EXTENT = 0.246600
ROBOT_SAFETY_MARGIN = 0.050000

GUIDANCE_POLICY_PATH = PROJECT_ROOT / "configs" / "safe_sage_class_safety.json"
YOLO_CONFIDENCE_THRESHOLD = 0.35
YOLO_IOU_THRESHOLD = 0.70
YOLO_IMAGE_SIZE = 640
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "0").strip() or "0"

FUSION_CONFIG = CameraLidarFusionConfig(
    minimum_visible_points=2,
    minimum_mask_hits=4,
    minimum_overlap_fraction=0.10,
    mask_dilation_radius_pixels=6,
    projected_median_box_margin_pixels=12,
)
MEMORY_CONFIG = SemanticTrackMemoryConfig(
    evidence_window_seconds=1.0,
    evidence_half_life_seconds=0.50,
    minimum_confirming_observations=2,
    minimum_observation_confidence=0.05,
    minimum_accumulated_evidence=0.12,
    minimum_class_evidence_margin=0.05,
    track_retention_seconds=2.0,
)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "yolo_safe_sage" / RUN_TIMESTAMP
TRANSIENT_SENSOR_ERRORS = (
    "No LiDAR data is available yet",
    "latest LiDAR update contains no points",
    "No RGB frame is available yet",
)


def resolve_model_path() -> Path:
    """Return the explicitly configured local segmentation checkpoint."""

    configured_path = os.environ.get("YOLO_MODEL_PATH", "").strip()
    if not configured_path:
        raise RuntimeError(
            "YOLO_MODEL_PATH is not set. Point it to yolo11n-seg.pt."
        )
    path = Path(configured_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"YOLO model weights were not found: {path}")
    return path


def make_detector(model_path: Path) -> YoloSegmentationDetector:
    """Initialize the optional Ultralytics adapter."""

    detector = YoloSegmentationDetector(
        YoloSegmentationConfig(
            model_path=model_path,
            confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
            iou_threshold=YOLO_IOU_THRESHOLD,
            image_size=YOLO_IMAGE_SIZE,
            device=YOLO_DEVICE,
        )
    )
    print("\nLoading YOLO11n segmentation weights...")
    detector.initialize()
    print("YOLO model initialized.")
    return detector


def read_chassis_pose(stage: Usd.Stage) -> np.ndarray:
    """Return the current world-frame ``(x, y, yaw)`` chassis pose."""

    return np.asarray(
        get_prim_planar_pose(stage, ROBOT_BASE_PRIM_PATH),
        dtype=np.float64,
    )


def simulation_time_seconds() -> float:
    """Return current Isaac timeline time in simulated seconds."""

    return float(omni.timeline.get_timeline_interface().get_current_time())


def wrapped_angle_difference(first: float, second: float) -> float:
    """Return the absolute wrapped difference between two yaw angles."""

    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def estimate_floor_height_in_robot_frame(stage: Usd.Stage) -> float:
    """Transform the restaurant world floor into the chassis frame."""

    robot_to_world = get_prim_world_transform(stage, ROBOT_BASE_PRIM_PATH)
    robot_origin_world = robot_to_world.ExtractTranslation()
    ground_world = Gf.Vec3d(
        robot_origin_world[0],
        robot_origin_world[1],
        0.0,
    )
    return float(robot_to_world.GetInverse().Transform(ground_world)[2])


def wait_for_new_sensor_pair(
    *,
    camera: NovaCarterCamera,
    lidar: NovaCarterLidar,
    controller: NovaCarterController,
    previous_lidar_frame_id: int | None,
) -> tuple[np.ndarray, CameraFrameInfo, np.ndarray, LidarFrameInfo, int]:
    """Return a paired image and strictly newer complete LiDAR scan."""

    last_error: RuntimeError | None = None
    for completed_updates in range(1, MAXIMUM_NEW_SCAN_WAIT_STEPS + 1):
        if not simulation_app.is_running():
            raise RuntimeError("Isaac Sim closed before sensor data arrived.")
        if not app_utils.is_playing():
            simulation_app.update()
            continue
        controller.stop()
        simulation_app.update()
        try:
            points_lidar = lidar.get_point_cloud()
            lidar_info = lidar.get_frame_info()
            image_rgb = camera.get_rgb()
            camera_info = camera.get_frame_info()
        except RuntimeError as error:
            if any(message in str(error) for message in TRANSIENT_SENSOR_ERRORS):
                last_error = error
                continue
            raise
        if (
            previous_lidar_frame_id is not None
            and lidar_info.sensor_frame_id <= previous_lidar_frame_id
        ):
            continue
        return image_rgb, camera_info, points_lidar, lidar_info, completed_updates
    detail = f" Last sensor error: {last_error}" if last_error else ""
    raise TimeoutError(
        "A newer paired camera--LiDAR observation did not arrive after "
        f"{MAXIMUM_NEW_SCAN_WAIT_STEPS} updates.{detail}"
    )


def create_policies_and_runtime(
    *,
    temporal_smoothing_factor: float | None = None,
    tangential_buffer_radius: float = 0.60,
    enable_temporal_updates: bool = True,
) -> tuple[
    ClassSafetyPolicy,
    SafeSAGEGuidancePolicy,
    SafeSAGEPerceptionRuntime,
]:
    """Create the production semantic policy and measured robot field."""

    guidance_policy = SafeSAGEGuidancePolicy.from_json(GUIDANCE_POLICY_PATH)
    class_policy = ClassSafetyPolicy(
        {
            semantic_class: parameters.safety_parameters
            for semantic_class, parameters
            in guidance_policy.class_parameters.items()
        },
        unknown_class=guidance_policy.unknown_class,
        risk_flux_mapping=guidance_policy.risk_flux_mapping,
    )
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
    grid_config = LocalOccupancyGridConfig(
        resolution=GRID_RESOLUTION,
        minimum_x=GRID_MINIMUM_X,
        maximum_x=GRID_MAXIMUM_X,
        minimum_y=GRID_MINIMUM_Y,
        maximum_y=GRID_MAXIMUM_Y,
        obstacle_inflation_radius=0.0,
    )
    field_config = SafeSAGEFieldPipelineConfig(
        configuration_space_config=SafeSAGEConfigurationSpaceConfig(
            footprint=footprint,
            yaw_slice_count=YAW_SLICE_COUNT,
        ),
        guidance_field_config=TangentialLaplaceGuidanceFieldConfig(
            convergence_tolerance=1e-6,
            tangential_buffer_radius=tangential_buffer_radius,
            allow_dense_scene_arbitration=True,
        ),
        h_full_config=SafeSAGEHFullConfig(
            yaw_slice_count=YAW_SLICE_COUNT,
            softplus_beta=1.0,
            minimum_forcing_magnitude=1e-9,
            poisson_config=PoissonSafetyFunctionConfig(
                convergence_tolerance=1e-6,
            ),
        ),
    )
    temporal_estimator = (
        None
        if temporal_smoothing_factor is None
        else SafeSAGETemporalDerivativeEstimator(
            SafeSAGETemporalDerivativeConfig(
                smoothing_factor=temporal_smoothing_factor,
            )
        )
    )
    runtime = SafeSAGEPerceptionRuntime(
        obstacle_tracker=ObstacleTracker(
            ObstacleTrackerConfig(
                cluster_radius=SEMANTIC_CLUSTER_RADIUS,
                minimum_cluster_points=2,
                maximum_association_distance=0.50,
            )
        ),
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=grid_config,
        field_config=field_config,
        temporal_estimator=temporal_estimator,
        live_field_config=SafeSAGELiveFieldConfig(
            # YOLO inference and a 10-slice dense-scene field currently take
            # longer than one control interval. Age is measured in simulated
            # time and remains bounded fail-closed between new sensor frames.
            maximum_field_age_seconds=2.50,
            maximum_reused_control_cycles=2,
        ),
        enable_temporal_updates=enable_temporal_updates,
    )
    return class_policy, guidance_policy, runtime


def numerical_field_is_valid(result: SafeSAGESemanticPerceptionResult) -> bool:
    """Return whether the accepted field contains finite aligned arrays."""

    field = result.field_status.snapshot.payload.pipeline
    h_values = field.h_full.values
    return bool(
        h_values.shape[0] == YAW_SLICE_COUNT
        and np.all(np.isfinite(h_values))
        and all(
            np.all(np.isfinite(guidance.vectors_robot_xy))
            for guidance in field.guidance_fields
        )
    )


@dataclass(frozen=True)
class SemanticMotionResult:
    """Outcome of the short end-to-end wheel-motion validation."""

    completed_control_cycles: int
    successful_field_updates: int
    forward_progress: float
    maximum_lateral_displacement: float
    mpc_intervention_seen: bool
    backup_intervention_seen: bool
    all_predictive_constraints_valid: bool
    all_backup_constraints_valid: bool
    route_complete: bool
    failure_reason: str | None


def robot_pose_from_array(pose: np.ndarray) -> RobotPose2D:
    """Convert one validated planar pose array to the controller type."""

    return RobotPose2D(
        x=float(pose[0]),
        y=float(pose[1]),
        yaw=float(pose[2]),
    )


def chassis_state_in_anchor(
    pose: RobotPose2D,
    *,
    anchor_pose: RobotPose2D,
) -> np.ndarray:
    """Return the measured chassis state in the immutable field frame."""

    delta_x = pose.x - anchor_pose.x
    delta_y = pose.y - anchor_pose.y
    cosine = math.cos(anchor_pose.yaw)
    sine = math.sin(anchor_pose.yaw)
    return np.array(
        [
            cosine * delta_x + sine * delta_y,
            -sine * delta_x + cosine * delta_y,
            math.atan2(
                math.sin(pose.yaw - anchor_pose.yaw),
                math.cos(pose.yaw - anchor_pose.yaw),
            ),
        ],
        dtype=np.float64,
    )


def create_control_runtime(
    controller: NovaCarterController,
    *,
    gamma: float = 1.0,
    mpc_horizon_steps: int = MOTION_MPC_HORIZON_STEPS,
    mpc_angular_velocity_weight: float = 0.02,
    backup_angular_velocity_weight: float = 1.0,
) -> SafeSAGEControlRuntime:
    """Create direct-unicycle MPC and the full-rate analytical CBF."""

    paper_filter = SafeSAGECBFSafetyFilter(
        SafeSAGECBFConfig(
            gamma=gamma,
            minimum_planar_velocity=-controller.config.max_linear_speed,
            maximum_planar_velocity=controller.config.max_linear_speed,
            minimum_angular_velocity=-controller.config.max_angular_speed,
            maximum_angular_velocity=controller.config.max_angular_speed,
            planar_velocity_weight=1.0,
            angular_velocity_weight=1.0,
        )
    )
    mpc = SafeSAGEUnicycleMPC(
        paper_filter,
        SafeSAGEUnicycleMPCConfig(
            horizon_steps=mpc_horizon_steps,
            time_step=MOTION_MPC_TIME_STEP,
            gamma=gamma,
            lookahead_distance=UNICYCLE_LOOKAHEAD_DISTANCE,
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=controller.config.max_linear_speed,
            minimum_angular_velocity=-controller.config.max_angular_speed,
            maximum_angular_velocity=controller.config.max_angular_speed,
            linear_velocity_weight=1.0,
            angular_velocity_weight=mpc_angular_velocity_weight,
            command_smoothness_weight=0.005,
            maximum_sqp_iterations=12,
            maximum_solver_iterations=4000,
            safety_substeps_per_interval=4,
        ),
    )
    backup = SafeSAGEUnicycleCBFFilter(
        paper_filter,
        SafeSAGEUnicycleCBFConfig(
            lookahead_distance=UNICYCLE_LOOKAHEAD_DISTANCE,
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=controller.config.max_linear_speed,
            minimum_angular_velocity=-controller.config.max_angular_speed,
            maximum_angular_velocity=controller.config.max_angular_speed,
            linear_velocity_weight=1.0,
            angular_velocity_weight=backup_angular_velocity_weight,
        ),
    )
    return SafeSAGEControlRuntime(mpc=mpc, backup_filter=backup)


def predictive_result_is_valid(result: SafeSAGEUnicycleMPCResult) -> bool:
    """Return whether an MPC plan is finite and satisfies exact DTCBF checks."""

    tolerance = 2e-6
    arrays = (
        result.command_sequence,
        result.predicted_chassis_states,
        result.predicted_safety_states,
        result.predicted_safety_values,
        result.dtcbf_margins,
    )
    return bool(
        result.success
        and not result.fallback_active
        and all(np.all(np.isfinite(array)) for array in arrays)
        and float(np.min(result.predicted_safety_values)) >= -tolerance
        and float(np.min(result.dtcbf_margins)) >= -tolerance
    )


def backup_result_is_valid(result: SafeSAGEUnicycleCBFResult) -> bool:
    """Return whether the analytical command remains executable and safe."""

    return bool(
        result.constraint_feasible
        and result.constraint_satisfied
        and not result.emergency_stop_active
        and math.isfinite(result.safe_linear_velocity)
        and math.isfinite(result.safe_angular_velocity)
        and math.isfinite(result.sampled_safety_value)
        and math.isfinite(result.cbf_margin)
    )


def run_semantic_motion(
    *,
    stage: Usd.Stage,
    controller: NovaCarterController,
    camera: NovaCarterCamera,
    lidar: NovaCarterLidar,
    detector: YoloSegmentationDetector,
    bridge: SafeSAGESemanticPerceptionBridge,
    robot_from_lidar: np.ndarray,
    camera_from_robot: np.ndarray,
    calibration,
    detection_config: ObstacleDetectionConfig,
    previous_lidar_frame_id: int,
) -> tuple[SemanticMotionResult, int]:
    """Execute a short obstacle-unaware waypoint under live semantic safety."""

    initial_pose = robot_pose_from_array(read_chassis_pose(stage))
    anchor_field = bridge.last_result.field_status.snapshot.payload
    anchor_pose = RobotPose2D(
        x=float(anchor_field.anchor_position_world_xy[0]),
        y=float(anchor_field.anchor_position_world_xy[1]),
        yaw=float(anchor_field.anchor_yaw),
    )
    goal = Waypoint(
        x=initial_pose.x + MOTION_GOAL_DISTANCE * math.cos(initial_pose.yaw),
        y=initial_pose.y + MOTION_GOAL_DISTANCE * math.sin(initial_pose.yaw),
    )
    waypoint_controller = WaypointController(
        [goal],
        config=WaypointControllerConfig(
            maximum_linear_velocity=0.15,
            maximum_angular_velocity=0.60,
            linear_gain=0.80,
            angular_gain=1.50,
            position_tolerance=0.10,
            minimum_alignment_scale=0.25,
            rotate_in_place_threshold_radians=math.radians(100.0),
        ),
    )
    control_runtime = create_control_runtime(controller)
    warm_start: np.ndarray | None = None
    completed_cycles = 0
    successful_updates = 0
    mpc_intervention_seen = False
    backup_intervention_seen = False
    all_predictive_valid = True
    all_backup_valid = True
    route_complete = False
    failure_reason: str | None = None
    maximum_lateral = 0.0

    print("\n" + "=" * 78)
    print("STAGE D - LIVE SEMANTIC MPC + CBF + WHEEL MOTION")
    print("=" * 78)
    print("  Nominal planner:             obstacle-unaware straight waypoint")
    print("  Safe predictive layer:      direct unicycle Safe-SAGE MPC")
    print("  Full-rate backup:           analytical CBF")
    print("  Semantic source:            temporal YOLO--LiDAR track fusion")
    print("  Precomputed bypass path:    none")
    print("\nLive motion status:")

    for cycle in range(MOTION_MAXIMUM_CONTROL_CYCLES):
        robot_pose = robot_pose_from_array(read_chassis_pose(stage))
        nominal_command = waypoint_controller.compute_command(robot_pose)
        if waypoint_controller.route_complete:
            route_complete = True
            break
        nominal_prediction = waypoint_controller.predict_commands(
            robot_pose,
            horizon_steps=control_runtime.mpc.config.horizon_steps,
            time_step=control_runtime.mpc.config.time_step,
        )
        nominal_sequence = np.asarray(
            [
                [command.linear_velocity, command.angular_velocity]
                for command in nominal_prediction
            ],
            dtype=np.float64,
        )
        chassis_state = chassis_state_in_anchor(
            robot_pose,
            anchor_pose=anchor_pose,
        )
        field = bridge.last_result.field_status.snapshot.payload
        solve_start = time.perf_counter()
        mpc_result = control_runtime.plan(
            chassis_state,
            nominal_sequence,
            field,
            initial_command_guess=warm_start,
            preferred_lateral_direction_grid_y=-1.0,
        )
        solve_wall_seconds = time.perf_counter() - solve_start
        predictive_valid = predictive_result_is_valid(mpc_result)
        all_predictive_valid &= predictive_valid
        if not predictive_valid:
            failure_reason = f"Fail-closed MPC stop: {mpc_result.status}"
            break
        warm_start = control_runtime.shifted_warm_start(mpc_result)
        mpc_intervention_seen |= mpc_result.intervention_active
        selected_command = np.array(
            [mpc_result.linear_velocity, mpc_result.angular_velocity],
            dtype=np.float64,
        )
        backup_result = control_runtime.filter_command(
            selected_command,
            chassis_state,
            field,
        )
        backup_valid = backup_result_is_valid(backup_result)
        all_backup_valid &= backup_valid
        if not backup_valid:
            failure_reason = "Fail-closed analytical backup stop."
            break
        backup_intervention_seen |= backup_result.intervention_active
        known_classes = sorted(
            {
                track.semantic_class
                for track in bridge.last_result.temporal_semantics.tracks
                if track.is_confirmed
            }
        )
        semantic_text = known_classes[0] if known_classes else "unknown"
        print(
            f"  cycle={cycle:<2} field={field.sensor_frame_id:<4} "
            f"class={semantic_text:<8} "
            f"q=({chassis_state[0]:.3f},{chassis_state[1]:.3f},"
            f"{math.degrees(chassis_state[2]):.1f}deg) "
            f"nom=({nominal_command.linear_velocity:.3f},"
            f"{nominal_command.angular_velocity:.3f}) "
            f"mpc=({mpc_result.linear_velocity:.3f},"
            f"{mpc_result.angular_velocity:.3f}) "
            f"out=({backup_result.safe_linear_velocity:.3f},"
            f"{backup_result.safe_angular_velocity:.3f}) "
            f"margin={mpc_result.minimum_dtcbf_margin:.2e} "
            f"solve={solve_wall_seconds:.3f}s"
        )

        execution_steps = bridge.perception_runtime.fresh_execution_step_budget(
            current_simulation_time_seconds=simulation_time_seconds(),
            physics_time_step_seconds=PHYSICS_DT,
            requested_steps=MOTION_CONTROL_PHYSICS_STEPS,
        )
        if execution_steps == 0:
            failure_reason = "Fail-closed stale semantic field stop."
            break
        command_interrupted = False
        for _ in range(execution_steps):
            if not simulation_app.is_running() or not app_utils.is_playing():
                failure_reason = "Isaac Sim stopped during wheel execution."
                command_interrupted = True
                break
            field_status = bridge.perception_runtime.inspect(
                current_simulation_time_seconds=simulation_time_seconds(),
            )
            if field_status.fail_closed:
                failure_reason = (
                    "Fail-closed stale semantic field stop: "
                    f"{field_status.failure_reason}"
                )
                command_interrupted = True
                break
            monitored_pose = robot_pose_from_array(read_chassis_pose(stage))
            monitored_state = chassis_state_in_anchor(
                monitored_pose,
                anchor_pose=anchor_pose,
            )
            step_backup = control_runtime.filter_command(
                selected_command,
                monitored_state,
                field,
            )
            step_valid = backup_result_is_valid(step_backup)
            all_backup_valid &= step_valid
            if not step_valid:
                failure_reason = "Fail-closed full-rate CBF stop."
                command_interrupted = True
                break
            backup_intervention_seen |= step_backup.intervention_active
            controller.apply_command(
                step_backup.safe_linear_velocity,
                step_backup.safe_angular_velocity,
            )
            simulation_app.update()
            monitored_state = chassis_state_in_anchor(
                robot_pose_from_array(read_chassis_pose(stage)),
                anchor_pose=anchor_pose,
            )
            maximum_lateral = max(maximum_lateral, abs(float(monitored_state[1])))
        controller.stop()
        if command_interrupted:
            break
        completed_cycles += 1

        (
            image_rgb,
            camera_info,
            points_lidar,
            lidar_info,
            _,
        ) = wait_for_new_sensor_pair(
            camera=camera,
            lidar=lidar,
            controller=controller,
            previous_lidar_frame_id=previous_lidar_frame_id,
        )
        previous_lidar_frame_id = lidar_info.sensor_frame_id
        points_robot = transform_points(points_lidar, robot_from_lidar)
        obstacle_points = detect_obstacle_points_3d(
            points_robot,
            detection_config,
        )
        current_pose = read_chassis_pose(stage)
        detections = detector.detect(
            image_rgb,
            source_frame_id=camera_info.read_index,
            source_timestamp_seconds=camera_info.simulation_time_seconds,
        )
        try:
            bridge.update(
                SafeSAGECameraLidarObservation(
                    perception=SafeSAGEPerceptionObservation(
                        sensor_frame_id=lidar_info.sensor_frame_id,
                        sensor_timestamp_seconds=(
                            lidar_info.sensor_timestamp_nanoseconds * 1e-9
                        ),
                        simulation_time_seconds=(
                            lidar_info.simulation_time_seconds
                        ),
                        obstacle_points_robot_xy=obstacle_points[:, :2],
                        robot_position_world_xy=current_pose[:2],
                        robot_yaw=float(current_pose[2]),
                    ),
                    obstacle_points_source_xyz=obstacle_points,
                    detection_frame=detections,
                    camera_from_source=camera_from_robot,
                    calibration=calibration,
                )
            )
        except (RuntimeError, TypeError, ValueError) as error:
            controller.stop()
            failure_reason = (
                "Fail-closed semantic field update stop: "
                f"{error}"
            )
            break
        successful_updates += 1

    controller.stop()
    app_utils.update_app(steps=3)
    final_state = chassis_state_in_anchor(
        robot_pose_from_array(read_chassis_pose(stage)),
        anchor_pose=anchor_pose,
    )
    return (
        SemanticMotionResult(
            completed_control_cycles=completed_cycles,
            successful_field_updates=successful_updates,
            forward_progress=float(final_state[0]),
            maximum_lateral_displacement=maximum_lateral,
            mpc_intervention_seen=mpc_intervention_seen,
            backup_intervention_seen=backup_intervention_seen,
            all_predictive_constraints_valid=all_predictive_valid,
            all_backup_constraints_valid=all_backup_valid,
            route_complete=route_complete,
            failure_reason=failure_reason,
        ),
        previous_lidar_frame_id,
    )


def main() -> None:
    """Run stationary real-sensor semantic Safe-SAGE integration."""

    print("\n" + "=" * 78)
    print("YOLO -> LIDAR TRACKS -> SAFE-SAGE H_FULL INTEGRATION")
    print("=" * 78)
    model_path = resolve_model_path()
    loaded_scene = load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=EXPECTED_METERS_PER_UNIT,
    )
    controller = NovaCarterController(robot_prim_path=ROBOT_PRIM_PATH)
    SimulationManager.setup_simulation(dt=PHYSICS_DT, device=PHYSICS_DEVICE)
    print("\nStarting Isaac Sim playback automatically...")
    app_utils.play()
    if not wait_until_playing(simulation_app):
        return
    app_utils.update_app(steps=10)
    controller.verify_articulation()
    controller.print_configuration()

    camera = NovaCarterCamera()
    lidar = NovaCarterLidar()
    camera.initialize()
    lidar.initialize()
    calibration = read_fisheye_polynomial_calibration(
        loaded_scene.stage,
        camera_prim_path=FRONT_HAWK_LEFT_CAMERA_PRIM_PATH,
        output_resolution=camera.resolution,
    )
    robot_from_lidar = compute_relative_prim_transform(
        loaded_scene.stage,
        source_prim_path=PANDAR_XT32_LIDAR_PRIM_PATH,
        target_prim_path=ROBOT_BASE_PRIM_PATH,
    )
    camera_from_robot = compute_relative_prim_transform(
        loaded_scene.stage,
        source_prim_path=ROBOT_BASE_PRIM_PATH,
        target_prim_path=FRONT_HAWK_LEFT_CAMERA_PRIM_PATH,
    )
    floor_height = estimate_floor_height_in_robot_frame(loaded_scene.stage)
    detection_config = ObstacleDetectionConfig(
        minimum_height=floor_height + DETECTION_HEIGHT_ABOVE_FLOOR[0],
        maximum_height=floor_height + DETECTION_HEIGHT_ABOVE_FLOOR[1],
        maximum_range=DETECTION_MAXIMUM_RANGE,
        forward_only=True,
        angular_bin_size_degrees=DETECTION_ANGULAR_BIN_DEGREES,
    )
    class_policy, _, perception_runtime = create_policies_and_runtime()
    bridge = SafeSAGESemanticPerceptionBridge(
        perception_runtime=perception_runtime,
        fusion=CameraLidarSemanticFusion(FUSION_CONFIG),
        semantic_memory=SemanticTrackMemory(MEMORY_CONFIG),
        config=SafeSAGESemanticBridgeConfig(
            maximum_sensor_time_offset_seconds=(
                MAXIMUM_SENSOR_TIME_OFFSET_SECONDS
            )
        ),
    )
    detector = make_detector(model_path)

    print(f"\nHolding Nova Carter still for {SETTLE_STEPS} physics steps...")
    if not advance_playing_steps(
        simulation_app,
        physics_steps=SETTLE_STEPS,
        on_step=controller.stop,
    ):
        return
    initial_pose = read_chassis_pose(loaded_scene.stage)
    previous_lidar_frame_id: int | None = None
    results: list[SafeSAGESemanticPerceptionResult] = []
    final_image: np.ndarray | None = None
    final_detections = None

    print("\n" + "=" * 78)
    print("STAGE A/B/C - LIVE SEMANTIC FIELD UPDATES")
    print("=" * 78)
    for sample_index in range(SAMPLE_COUNT):
        (
            image_rgb,
            camera_info,
            points_lidar,
            lidar_info,
            waited_updates,
        ) = wait_for_new_sensor_pair(
            camera=camera,
            lidar=lidar,
            controller=controller,
            previous_lidar_frame_id=previous_lidar_frame_id,
        )
        previous_lidar_frame_id = lidar_info.sensor_frame_id
        points_robot = transform_points(points_lidar, robot_from_lidar)
        obstacle_points = detect_obstacle_points_3d(
            points_robot,
            detection_config,
        )
        if obstacle_points.shape[0] == 0:
            raise RuntimeError("The paired scan produced no obstacle points.")
        pose = read_chassis_pose(loaded_scene.stage)
        detections = detector.detect(
            image_rgb,
            source_frame_id=camera_info.read_index,
            source_timestamp_seconds=camera_info.simulation_time_seconds,
        )
        lidar_timestamp = lidar_info.sensor_timestamp_nanoseconds * 1e-9
        observation = SafeSAGECameraLidarObservation(
            perception=SafeSAGEPerceptionObservation(
                sensor_frame_id=lidar_info.sensor_frame_id,
                sensor_timestamp_seconds=lidar_timestamp,
                simulation_time_seconds=lidar_info.simulation_time_seconds,
                obstacle_points_robot_xy=obstacle_points[:, :2],
                robot_position_world_xy=pose[:2],
                robot_yaw=float(pose[2]),
            ),
            obstacle_points_source_xyz=obstacle_points,
            detection_frame=detections,
            camera_from_source=camera_from_robot,
            calibration=calibration,
        )
        if sample_index == 0:
            result = bridge.initialize(
                observation,
                anchor_position_world_xy=pose[:2],
                anchor_yaw=float(pose[2]),
            )
        else:
            result = bridge.update(observation)
        results.append(result)
        raw_matches = sum(item.is_matched for item in result.fusion.associations)
        policy_classes = sorted(
            {
                item.semantic_class
                for item in result.temporal_semantics.tracks
                if item.is_confirmed
                and item.semantic_class in class_policy.known_classes
            }
        )
        print(
            f"  sample={sample_index + 1:<2} "
            f"scan={lidar_info.sensor_frame_id:<4} wait={waited_updates:<3} "
            f"detections={len(detections.detections):<2} "
            f"tracks={len(result.temporal_semantics.tracks):<3} "
            f"raw_matches={raw_matches:<2} "
            f"confirmed={result.known_semantic_track_count:<2} "
            f"policy_classes={policy_classes!s:<18} "
            f"field={result.field_status.generation:<2} "
            f"build={result.field_status.snapshot.build_wall_seconds:.3f}s "
            f"sync={result.sensor_time_offset_seconds:.3f}s"
        )
        final_image = np.array(image_rgb, copy=True)
        final_detections = detections

    controller.stop()
    app_utils.update_app(steps=5)
    final_pose = read_chassis_pose(loaded_scene.stage)

    if final_image is None or final_detections is None:
        raise RuntimeError("No camera result was retained for visualization.")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    overlay_path = OUTPUT_DIRECTORY / "yolo_safe_sage_fusion_overlay.png"
    overlay = render_camera_lidar_fusion_overlay(
        final_image,
        final_detections,
        results[-1].fusion,
        point_radius=2,
    )
    Image.fromarray(overlay).save(overlay_path)

    raw_match_seen = any(
        any(item.is_matched for item in result.fusion.associations)
        for result in results
    )
    confirmed_policy_classes = sorted(
        {
            item.semantic_class
            for result in results
            for item in result.temporal_semantics.tracks
            if item.is_confirmed
            and item.semantic_class in class_policy.known_classes
            and item.semantic_class != class_policy.unknown_class
        }
    )
    fields_valid = all(numerical_field_is_valid(result) for result in results)
    generations_valid = (
        [result.field_status.generation for result in results]
        == list(range(1, SAMPLE_COUNT + 1))
    )
    selected_geometry_valid = all(
        result.field_status.snapshot.payload.selected_obstacle_point_count > 0
        for result in results
    )
    maximum_sync_offset = max(
        result.sensor_time_offset_seconds for result in results
    )
    position_drift = float(np.linalg.norm(final_pose[:2] - initial_pose[:2]))
    yaw_drift = wrapped_angle_difference(final_pose[2], initial_pose[2])
    robot_stationary = (
        position_drift <= MAXIMUM_ALLOWED_POSITION_DRIFT
        and yaw_drift <= MAXIMUM_ALLOWED_YAW_DRIFT
    )
    passed = bool(
        len(results) == SAMPLE_COUNT
        and raw_match_seen
        and confirmed_policy_classes
        and fields_valid
        and generations_valid
        and selected_geometry_valid
        and maximum_sync_offset <= MAXIMUM_SENSOR_TIME_OFFSET_SECONDS
        and robot_stationary
    )

    print("\n" + "=" * 78)
    print("YOLO SAFE-SAGE INTEGRATION REPORT")
    print("=" * 78)
    print("\nPerception and semantics:")
    print(f"  Accepted paired samples:       {len(results)}")
    print(f"  Raw camera--LiDAR match seen:  {raw_match_seen}")
    print(f"  Confirmed policy classes:      {confirmed_policy_classes}")
    print(f"  Maximum sensor offset:         {maximum_sync_offset:.4f} s")
    print("\nSafe-SAGE field boundary:")
    print(f"  Sequential field generations:  {generations_valid}")
    print(f"  LiDAR geometry selected:       {selected_geometry_valid}")
    print(f"  All numerical fields finite:   {fields_valid}")
    print(f"  Final selected tracks:         {results[-1].field_status.snapshot.payload.selected_track_ids}")
    print("\nClass policy values observed:")
    for semantic_class in confirmed_policy_classes:
        parameters = class_policy.parameters_for(semantic_class)
        print(
            f"  {semantic_class:<12} risk={parameters.risk:.3f} "
            f"normal_flux={parameters.boundary_flux:.3f}"
        )
    print("\nControlled checks:")
    print(f"  Position drift:                {position_drift:.6f} m")
    print(f"  Yaw drift:                     {math.degrees(yaw_drift):.4f} degrees")
    print(f"  Robot remained stationary:     {robot_stationary}")
    print(f"\nSaved fusion overlay:           {overlay_path}")
    print("\nControlled-data boundary:")
    print("  LiDAR alone created geometry and persistent track IDs.")
    print("  YOLO masks only supplied class evidence for those tracks.")
    print("  Unknown labels remained obstacles under conservative policy.")
    print("  Stages A/B/C sent no command to MPC, CBF, or the wheels.")
    print("\nInterpretation:")
    if passed:
        print(
            "  [PASS] Real YOLO labels changed the class-aware Safe-SAGE "
            "field policy while LiDAR remained the sole geometry source."
        )
    else:
        print(
            "  [WARNING] The perception-to-field integration did not pass "
            "every controlled check. Inspect the sequence and overlay."
        )
    print("=" * 78)

    if not passed:
        print(
            "\nStage D was not enabled because the stationary semantic "
            "field gate did not pass."
        )
        return

    # Physics remains active across the gate so Isaac's articulation tensor
    # view and sensor producers retain their valid state. Stage D uses the
    # same bridge, not a precomputed reference path.
    motion_result, previous_lidar_frame_id = run_semantic_motion(
        stage=loaded_scene.stage,
        controller=controller,
        camera=camera,
        lidar=lidar,
        detector=detector,
        bridge=bridge,
        robot_from_lidar=robot_from_lidar,
        camera_from_robot=camera_from_robot,
        calibration=calibration,
        detection_config=detection_config,
        previous_lidar_frame_id=previous_lidar_frame_id,
    )
    del previous_lidar_frame_id
    motion_passed = bool(
        motion_result.completed_control_cycles > 0
        and motion_result.successful_field_updates > 0
        and motion_result.forward_progress >= MOTION_MINIMUM_FORWARD_PROGRESS
        and motion_result.all_predictive_constraints_valid
        and motion_result.all_backup_constraints_valid
        and motion_result.failure_reason is None
    )
    print("\n" + "=" * 78)
    print("YOLO SAFE-SAGE END-TO-END MOTION REPORT")
    print("=" * 78)
    print(f"  Completed control cycles:       {motion_result.completed_control_cycles}")
    print(f"  Live semantic field updates:    {motion_result.successful_field_updates}")
    print(f"  Forward progress:               {motion_result.forward_progress:.4f} m")
    print(
        "  Maximum lateral displacement:   "
        f"{motion_result.maximum_lateral_displacement:.4f} m"
    )
    print(f"  MPC intervention observed:      {motion_result.mpc_intervention_seen}")
    print(f"  Backup intervention observed:   {motion_result.backup_intervention_seen}")
    print(
        "  All predictive constraints:     "
        f"{motion_result.all_predictive_constraints_valid}"
    )
    print(
        "  All analytical constraints:     "
        f"{motion_result.all_backup_constraints_valid}"
    )
    print(f"  Route complete:                 {motion_result.route_complete}")
    print(f"  Failure reason:                 {motion_result.failure_reason}")
    print("\nInterpretation:")
    if motion_passed:
        print(
            "  [PASS] The obstacle-unaware waypoint flowed through live "
            "YOLO--LiDAR semantics, Safe-SAGE MPC, the full-rate CBF, and "
            "Nova Carter's wheel controller."
        )
    else:
        print(
            "  [WARNING] The short semantic wheel-motion gate did not pass "
            "every criterion. The controller stopped fail-closed."
        )
    print("=" * 78)
    controller.stop()
    app_utils.stop()

    if SIMULATION_HEADLESS or AUTOCLOSE_AFTER_REPORT:
        print("\nAutomated run complete; closing Isaac Sim.")
        return
    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when finished.")
    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - demo must close SimulationApp
        print("\n" + "=" * 78)
        print("ERROR DURING YOLO SAFE-SAGE INTEGRATION DEMO")
        print("=" * 78)
        print(error)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
