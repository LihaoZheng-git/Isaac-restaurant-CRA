"""Demonstrate semantic Safe-SAGE intervention around a real chair asset.

Stage A - controlled scene and stationary perception:

1. Load the restaurant and move one existing rendered chair into the open area
   using only an anonymous USD session-layer override.
2. Define one straight waypoint behind the chair and verify independently that
   the nominal line intersects the chair bounds.
3. Keep Nova Carter stationary while real camera and LiDAR frames create a
   persistent track and confirm the YOLO class ``chair``.

Stage B - class-aware field gate:

1. Resolve the confirmed chair through the repository safety policy.
2. Build finite live ``h_full(x, y, psi)`` fields from LiDAR geometry.
3. Confirm that the chair uses the configured risk, normal flux, tangential
   strength, and passing side without injecting its ground-truth class.

Stage C - live Safe-SAGE intervention and wheel motion:

1. Keep the nominal waypoint controller obstacle-unaware and aimed through the
   chair.
2. Update YOLO--LiDAR semantics and the class-aware field every control cycle.
3. Apply direct-unicycle Safe-SAGE MPC, then the full-rate analytical CBF, to
   Nova Carter's differential wheel controller.
4. Require predictive intervention, robot-right displacement, a complete
   physical bypass, valid safety constraints, and positive independent
   clearance.  No bypass reference path is generated.

The chair's authored position and bounds are used only to construct and score
the controlled experiment.  They are never passed to perception or control.

Run from the project root in the cloned YOLO environment:

    $env:YOLO_MODEL_PATH = "C:\\path\\to\\yolo11n-seg.pt"
    python -u scripts/demos/yolo_safe_sage_intervention_demo.py

Set ``ISAAC_SIM_HEADLESS=1`` for a headless run, or
``YOLO_SAFE_SAGE_INTERVENTION_AUTOCLOSE=1`` to close after the report.
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


# Importing the validated integration demo creates the single SimulationApp
# before any Isaac/pxr runtime modules are imported.
from scripts.demos import yolo_safe_sage_integration_demo as integration


import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
from isaacsim.core.simulation_manager import SimulationManager
from PIL import Image

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
from robot.control.differential_drive import NovaCarterController
from robot.navigation.waypoint_controller import (
    RobotPose2D,
    Waypoint,
    WaypointController,
    WaypointControllerConfig,
)
from robot.perception.camera_lidar_fusion import CameraLidarSemanticFusion
from robot.perception.safe_sage_semantic_bridge import (
    SafeSAGECameraLidarObservation,
    SafeSAGESemanticBridgeConfig,
    SafeSAGESemanticPerceptionBridge,
    SafeSAGESemanticPerceptionResult,
)
from robot.perception.semantic_track_memory import SemanticTrackMemory
from robot.perception.visualization import render_camera_lidar_fusion_overlay
from robot.runtime.simulation_loop import advance_playing_steps, wait_until_playing
from robot.safety.obstacle_detection import (
    ObstacleDetectionConfig,
    detect_obstacle_points_3d,
)
from robot.safety.safe_sage_runtime import SafeSAGEPerceptionObservation
from robot.scene.camera_calibration import read_fisheye_polynomial_calibration
from robot.scene.controlled_semantic_obstacle import (
    PlanarFootprint,
    WorldBounds3D,
    line_segment_intersects_planar_bounds,
    oriented_footprint_clearance_to_bounds,
    place_existing_prim_world_origin,
)
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import compute_relative_prim_transform
from robot.sensor.camera import NovaCarterCamera
from robot.sensor.lidar import NovaCarterLidar
from robot.sensor.lidar_geometry import transform_points


simulation_app = integration.simulation_app
SIMULATION_HEADLESS = integration.SIMULATION_HEADLESS
AUTOCLOSE_AFTER_REPORT = (
    integration.AUTOCLOSE_AFTER_REPORT
    or integration.environment_flag("YOLO_SAFE_SAGE_INTERVENTION_AUTOCLOSE")
)

CONTROLLED_CHAIR_PRIM_PATH = (
    "/World/Environment/Restaurant/assets_restaurant/"
    "Assets_Banquettes_Right/Geo_TableSet_09/Chair_Array"
)
ROBOT_SCENARIO_WORLD_ORIGIN = (26.4955, 5.00175, 0.07382)
CHAIR_FORWARD_OFFSET = 1.70
CHAIR_LATERAL_OFFSET = 0.0
NOMINAL_GOAL_FORWARD_DISTANCE = 3.50
EXPECTED_SEMANTIC_CLASS = "chair"
MAXIMUM_CHAIR_TRACK_ASSOCIATION_ERROR = 0.55
TANGENTIAL_BUFFER_RADIUS = 0.60
SAFETY_FILTER_GAMMA = 0.30
MPC_HORIZON_STEPS = 30
MPC_ANGULAR_VELOCITY_WEIGHT = 0.005
BACKUP_ANGULAR_VELOCITY_WEIGHT = 0.02

STATIONARY_SAMPLE_COUNT = 4
MOTION_MAXIMUM_CONTROL_CYCLES = int(
    os.environ.get("YOLO_SAFE_SAGE_MAX_CONTROL_CYCLES", "120")
)
MOTION_CONTROL_PHYSICS_STEPS = 12
MOTION_MINIMUM_RIGHTWARD_DISPLACEMENT = 0.30
MOTION_FULL_CLEAR_MARGIN = 0.08
GROUND_TRUTH_CLEARANCE_ABORT = 0.02

ROBOT_COLLISION_FOOTPRINT = PlanarFootprint(
    front_extent=integration.ROBOT_FRONT_EXTENT,
    rear_extent=integration.ROBOT_REAR_EXTENT,
    left_extent=integration.ROBOT_LEFT_EXTENT,
    right_extent=integration.ROBOT_RIGHT_EXTENT,
)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "yolo_safe_sage_intervention"
    / RUN_TIMESTAMP
)


@dataclass(frozen=True)
class ControlledTrackDiagnostic:
    """Nearest observed semantic track to the known chair position."""

    track_id: int
    association_error: float
    semantic_class: str
    semantic_confidence: float
    temporal_status: str
    current_observation_class: str
    reassociated_from_track_id: int | None

    @property
    def is_confirmed_chair(self) -> bool:
        """Return whether the nearest geometric track is a confirmed chair."""

        return (
            self.semantic_class == EXPECTED_SEMANTIC_CLASS
            and self.temporal_status
            in {"confirmed", "persisted", "reassociated"}
        )


@dataclass(frozen=True)
class InterventionMotionResult:
    """Outcome of the live controlled-chair bypass."""

    completed_control_cycles: int
    successful_field_updates: int
    chair_semantic_updates: int
    reassociated_semantic_updates: int
    controlled_track_id_changes: int
    forward_displacement: float
    maximum_rightward_displacement: float
    minimum_ground_truth_clearance: float
    mpc_intervention_seen: bool
    backup_intervention_seen: bool
    all_predictive_constraints_valid: bool
    all_backup_constraints_valid: bool
    chair_cleared: bool
    route_complete: bool
    failure_reason: str | None


def world_point_from_robot_offset(
    robot_pose: np.ndarray,
    *,
    forward: float,
    left: float,
) -> np.ndarray:
    """Resolve one planar robot-frame offset into the world frame."""

    cosine = math.cos(float(robot_pose[2]))
    sine = math.sin(float(robot_pose[2]))
    return np.array(
        [
            robot_pose[0] + cosine * forward - sine * left,
            robot_pose[1] + sine * forward + cosine * left,
        ],
        dtype=np.float64,
    )


def controlled_track_diagnostic(
    result: SafeSAGESemanticPerceptionResult,
    *,
    chair_bounds: WorldBounds3D,
) -> ControlledTrackDiagnostic:
    """Associate a ground-truth chair only for independent report scoring."""

    field = result.field_status.snapshot.payload
    obstacles = field.label_result.semantic_obstacles
    if not obstacles:
        raise RuntimeError("The accepted field contains no semantic obstacles.")
    chair_center = np.asarray(chair_bounds.center[:2], dtype=np.float64)
    nearest = min(
        obstacles,
        key=lambda item: float(
            np.linalg.norm(
                np.asarray(item.centroid_world_xy, dtype=np.float64)
                - chair_center
            )
        ),
    )
    association_error = float(
        np.linalg.norm(
            np.asarray(nearest.centroid_world_xy, dtype=np.float64)
            - chair_center
        )
    )
    temporal_by_track = {
        item.track_id: item for item in result.temporal_semantics.tracks
    }
    temporal = temporal_by_track.get(nearest.track_id)
    temporal_status = "missing" if temporal is None else temporal.status
    return ControlledTrackDiagnostic(
        track_id=nearest.track_id,
        association_error=association_error,
        semantic_class=nearest.semantic_class,
        semantic_confidence=nearest.semantic_confidence,
        temporal_status=temporal_status,
        current_observation_class=(
            "missing"
            if temporal is None
            else temporal.current_observation_class
        ),
        reassociated_from_track_id=(
            None
            if temporal is None
            else temporal.reassociated_from_track_id
        ),
    )


def capture_semantic_update(
    *,
    stage,
    camera: NovaCarterCamera,
    lidar: NovaCarterLidar,
    controller: NovaCarterController,
    detector,
    bridge: SafeSAGESemanticPerceptionBridge,
    robot_from_lidar: np.ndarray,
    camera_from_robot: np.ndarray,
    calibration,
    detection_config: ObstacleDetectionConfig,
    previous_lidar_frame_id: int | None,
    initialize: bool,
) -> tuple[SafeSAGESemanticPerceptionResult, np.ndarray, object, int, int]:
    """Capture one paired sensor update and transactionally build a field."""

    (
        image_rgb,
        camera_info,
        points_lidar,
        lidar_info,
        waited_updates,
    ) = integration.wait_for_new_sensor_pair(
        camera=camera,
        lidar=lidar,
        controller=controller,
        previous_lidar_frame_id=previous_lidar_frame_id,
    )
    points_robot = transform_points(points_lidar, robot_from_lidar)
    obstacle_points = detect_obstacle_points_3d(points_robot, detection_config)
    if obstacle_points.shape[0] == 0:
        raise RuntimeError("The paired scan produced no obstacle points.")
    pose = integration.read_chassis_pose(stage)
    detections = detector.detect(
        image_rgb,
        source_frame_id=camera_info.read_index,
        source_timestamp_seconds=camera_info.simulation_time_seconds,
    )
    observation = SafeSAGECameraLidarObservation(
        perception=SafeSAGEPerceptionObservation(
            sensor_frame_id=lidar_info.sensor_frame_id,
            sensor_timestamp_seconds=(
                lidar_info.sensor_timestamp_nanoseconds * 1e-9
            ),
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
    if initialize:
        result = bridge.initialize(
            observation,
            anchor_position_world_xy=pose[:2],
            anchor_yaw=float(pose[2]),
        )
    else:
        result = bridge.update(observation)
    return (
        result,
        np.array(image_rgb, copy=True),
        detections,
        lidar_info.sensor_frame_id,
        waited_updates,
    )


def chair_is_fully_cleared(
    pose: RobotPose2D,
    *,
    chair_bounds: WorldBounds3D,
) -> bool:
    """Return whether Nova Carter's rear is conservatively past the chair."""

    return (
        pose.x - integration.ROBOT_REAR_EXTENT
        > chair_bounds.maximum[0] + MOTION_FULL_CLEAR_MARGIN
    )


def run_intervention_motion(
    *,
    stage,
    controller: NovaCarterController,
    camera: NovaCarterCamera,
    lidar: NovaCarterLidar,
    detector,
    bridge: SafeSAGESemanticPerceptionBridge,
    robot_from_lidar: np.ndarray,
    camera_from_robot: np.ndarray,
    calibration,
    detection_config: ObstacleDetectionConfig,
    previous_lidar_frame_id: int,
    goal: Waypoint,
    chair_bounds: WorldBounds3D,
) -> tuple[InterventionMotionResult, int]:
    """Run a live obstacle-unaware waypoint through the semantic safety stack."""

    anchor_field = bridge.last_result.field_status.snapshot.payload
    anchor_pose = RobotPose2D(
        x=float(anchor_field.anchor_position_world_xy[0]),
        y=float(anchor_field.anchor_position_world_xy[1]),
        yaw=float(anchor_field.anchor_yaw),
    )
    waypoint_controller = WaypointController(
        [goal],
        config=WaypointControllerConfig(
            maximum_linear_velocity=0.20,
            maximum_angular_velocity=0.60,
            linear_gain=0.80,
            angular_gain=1.50,
            position_tolerance=0.10,
            minimum_alignment_scale=0.25,
            rotate_in_place_threshold_radians=math.radians(100.0),
        ),
    )
    control_runtime = integration.create_control_runtime(
        controller,
        gamma=SAFETY_FILTER_GAMMA,
        mpc_horizon_steps=MPC_HORIZON_STEPS,
        mpc_angular_velocity_weight=MPC_ANGULAR_VELOCITY_WEIGHT,
        backup_angular_velocity_weight=BACKUP_ANGULAR_VELOCITY_WEIGHT,
    )
    warm_start: np.ndarray | None = None
    completed_cycles = 0
    successful_updates = 0
    chair_semantic_updates = 0
    reassociated_semantic_updates = 0
    controlled_track_id_changes = 0
    initial_diagnostic = controlled_track_diagnostic(
        bridge.last_result,
        chair_bounds=chair_bounds,
    )
    previous_controlled_track_id = initial_diagnostic.track_id
    mpc_intervention_seen = False
    backup_intervention_seen = False
    all_predictive_valid = True
    all_backup_valid = True
    chair_cleared = False
    route_complete = False
    failure_reason: str | None = None
    maximum_rightward = 0.0
    minimum_clearance = math.inf

    print("\n" + "=" * 78)
    print("STAGE C - LIVE YOLO SAFE-SAGE CHAIR INTERVENTION")
    print("=" * 78)
    print("  Nominal planner:             one straight obstacle-unaware waypoint")
    print("  Nominal line crosses chair:  True")
    print("  Semantic source:             real YOLO masks on LiDAR track IDs")
    print("  Predictive safety:           direct-unicycle Safe-SAGE MPC")
    print("  Full-rate safety:            analytical CBF")
    print("  Requested passing side:      robot-right")
    print(f"  CBF/MPC gamma:               {SAFETY_FILTER_GAMMA:.3f}")
    print(f"  MPC horizon:                 {MPC_HORIZON_STEPS} steps")
    print(
        "  MPC angular tracking weight: "
        f"{MPC_ANGULAR_VELOCITY_WEIGHT:.4f}"
    )
    print(
        "  Backup angular weight:       "
        f"{BACKUP_ANGULAR_VELOCITY_WEIGHT:.3f}"
    )
    print("  Precomputed bypass path:     none")
    print("  Ground-truth controller use: none (abort/report only)")
    print("\nDirectional field audit at yaw=0:")
    print("  x       h_full       guidance(x,y)       coeff(v,omega)")
    audit_field = bridge.last_result.field_status.snapshot.payload
    for audit_x in (0.0, 0.30, 0.60, 0.90, 1.10, 1.20):
        audit_result = control_runtime.filter_command(
            np.zeros(2, dtype=np.float64),
            np.array([audit_x, 0.0, 0.0], dtype=np.float64),
            audit_field,
        )
        guidance_x, guidance_y = audit_result.sampled_guidance_grid_xy
        print(
            f"  {audit_x:4.2f}  {audit_result.sampled_safety_value:10.5f}  "
            f"({guidance_x:8.4f},{guidance_y:8.4f})  "
            f"({audit_result.linear_velocity_coefficient:8.4f},"
            f"{audit_result.angular_velocity_coefficient:8.4f})"
        )
    print("\nLive motion status:")

    for cycle in range(MOTION_MAXIMUM_CONTROL_CYCLES):
        robot_pose = integration.robot_pose_from_array(
            integration.read_chassis_pose(stage)
        )
        state = integration.chassis_state_in_anchor(
            robot_pose,
            anchor_pose=anchor_pose,
        )
        maximum_rightward = max(maximum_rightward, -float(state[1]))
        current_clearance = oriented_footprint_clearance_to_bounds(
            (robot_pose.x, robot_pose.y, robot_pose.yaw),
            footprint=ROBOT_COLLISION_FOOTPRINT,
            obstacle_bounds=chair_bounds,
        )
        minimum_clearance = min(minimum_clearance, current_clearance)
        chair_cleared = chair_is_fully_cleared(
            robot_pose,
            chair_bounds=chair_bounds,
        )
        if (
            chair_cleared
            and maximum_rightward >= MOTION_MINIMUM_RIGHTWARD_DISPLACEMENT
        ):
            break
        if current_clearance <= GROUND_TRUTH_CLEARANCE_ABORT:
            failure_reason = (
                "Independent ground-truth clearance guard stopped motion."
            )
            break

        nominal_command = waypoint_controller.compute_command(robot_pose)
        if waypoint_controller.route_complete:
            route_complete = True
            if not chair_cleared:
                failure_reason = "Nominal route ended before the chair was cleared."
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
        field = bridge.last_result.field_status.snapshot.payload
        if cycle == 0:
            stationary_diagnostic = control_runtime.filter_command(
                np.zeros(2, dtype=np.float64),
                state,
                field,
            )
            temporal_field = field.temporal_safety_derivative
            maximum_temporal_derivative = (
                0.0
                if isinstance(temporal_field, float)
                else temporal_field.maximum_absolute_derivative
            )
            print("\nInitial safety-state diagnostic:")
            print(
                "  Sampled h_full:             "
                f"{stationary_diagnostic.sampled_safety_value:.6f}"
            )
            print(
                "  Sampled temporal dh/dt:     "
                f"{stationary_diagnostic.temporal_safety_derivative:.6f}"
            )
            print(
                "  Maximum field |dh/dt|:      "
                f"{maximum_temporal_derivative:.6f}"
            )
            print(
                "  Zero-command CBF margin:    "
                f"{stationary_diagnostic.cbf_margin:.6e}"
            )
            print(
                "  Zero-command constraint ok: "
                f"{stationary_diagnostic.constraint_satisfied}"
            )
            print("\nLive motion status:")
        solve_start = time.perf_counter()
        mpc_result = control_runtime.plan(
            state,
            nominal_sequence,
            field,
            initial_command_guess=warm_start,
            preferred_lateral_direction_grid_y=-1.0,
        )
        solve_wall_seconds = time.perf_counter() - solve_start
        predictive_valid = integration.predictive_result_is_valid(mpc_result)
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
            state,
            field,
        )
        backup_valid = integration.backup_result_is_valid(backup_result)
        all_backup_valid &= backup_valid
        if not backup_valid:
            failure_reason = "Fail-closed analytical backup stop."
            break
        backup_intervention_seen |= backup_result.intervention_active
        try:
            diagnostic = controlled_track_diagnostic(
                bridge.last_result,
                chair_bounds=chair_bounds,
            )
            semantic_text = diagnostic.semantic_class
            track_text = str(diagnostic.track_id)
            status_text = diagnostic.temporal_status
            observation_text = diagnostic.current_observation_class
        except RuntimeError:
            semantic_text = "none"
            track_text = "-"
            status_text = "missing"
            observation_text = "none"
        print(
            f"  cycle={cycle:<3} field={field.sensor_frame_id:<4} "
            f"track={track_text:<3} class={semantic_text:<8} "
            f"status={status_text:<12} obs={observation_text:<8} "
            f"q=({state[0]:.3f},{state[1]:.3f},"
            f"{math.degrees(state[2]):.1f}deg) "
            f"clear={current_clearance:.3f} "
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
            current_simulation_time_seconds=integration.simulation_time_seconds(),
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
                current_simulation_time_seconds=(
                    integration.simulation_time_seconds()
                ),
            )
            if field_status.fail_closed:
                failure_reason = (
                    "Fail-closed stale semantic field stop: "
                    f"{field_status.failure_reason}"
                )
                command_interrupted = True
                break
            monitored_pose = integration.robot_pose_from_array(
                integration.read_chassis_pose(stage)
            )
            monitored_state = integration.chassis_state_in_anchor(
                monitored_pose,
                anchor_pose=anchor_pose,
            )
            step_backup = control_runtime.filter_command(
                selected_command,
                monitored_state,
                field,
            )
            step_valid = integration.backup_result_is_valid(step_backup)
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
            measured_pose = integration.robot_pose_from_array(
                integration.read_chassis_pose(stage)
            )
            measured_state = integration.chassis_state_in_anchor(
                measured_pose,
                anchor_pose=anchor_pose,
            )
            maximum_rightward = max(
                maximum_rightward,
                -float(measured_state[1]),
            )
            measured_clearance = oriented_footprint_clearance_to_bounds(
                (measured_pose.x, measured_pose.y, measured_pose.yaw),
                footprint=ROBOT_COLLISION_FOOTPRINT,
                obstacle_bounds=chair_bounds,
            )
            minimum_clearance = min(minimum_clearance, measured_clearance)
            if measured_clearance <= GROUND_TRUTH_CLEARANCE_ABORT:
                failure_reason = (
                    "Independent ground-truth clearance guard stopped motion."
                )
                command_interrupted = True
                break
        controller.stop()
        if command_interrupted:
            break
        completed_cycles += 1

        try:
            (
                updated_result,
                _,
                _,
                previous_lidar_frame_id,
                _,
            ) = capture_semantic_update(
                stage=stage,
                camera=camera,
                lidar=lidar,
                controller=controller,
                detector=detector,
                bridge=bridge,
                robot_from_lidar=robot_from_lidar,
                camera_from_robot=camera_from_robot,
                calibration=calibration,
                detection_config=detection_config,
                previous_lidar_frame_id=previous_lidar_frame_id,
                initialize=False,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            failure_reason = f"Fail-closed semantic field update stop: {error}"
            break
        successful_updates += 1
        try:
            updated_diagnostic = controlled_track_diagnostic(
                updated_result,
                chair_bounds=chair_bounds,
            )
            chair_semantic_updates += int(
                updated_diagnostic.is_confirmed_chair
                and updated_diagnostic.association_error
                <= MAXIMUM_CHAIR_TRACK_ASSOCIATION_ERROR
            )
            reassociated_semantic_updates += int(
                updated_diagnostic.temporal_status == "reassociated"
            )
            controlled_track_id_changes += int(
                updated_diagnostic.track_id != previous_controlled_track_id
            )
            previous_controlled_track_id = updated_diagnostic.track_id
        except RuntimeError:
            pass

    controller.stop()
    app_utils.update_app(steps=3)
    final_pose = integration.robot_pose_from_array(
        integration.read_chassis_pose(stage)
    )
    final_state = integration.chassis_state_in_anchor(
        final_pose,
        anchor_pose=anchor_pose,
    )
    final_clearance = oriented_footprint_clearance_to_bounds(
        (final_pose.x, final_pose.y, final_pose.yaw),
        footprint=ROBOT_COLLISION_FOOTPRINT,
        obstacle_bounds=chair_bounds,
    )
    minimum_clearance = min(minimum_clearance, final_clearance)
    chair_cleared = chair_is_fully_cleared(
        final_pose,
        chair_bounds=chair_bounds,
    )
    return (
        InterventionMotionResult(
            completed_control_cycles=completed_cycles,
            successful_field_updates=successful_updates,
            chair_semantic_updates=chair_semantic_updates,
            reassociated_semantic_updates=reassociated_semantic_updates,
            controlled_track_id_changes=controlled_track_id_changes,
            forward_displacement=float(final_state[0]),
            maximum_rightward_displacement=maximum_rightward,
            minimum_ground_truth_clearance=minimum_clearance,
            mpc_intervention_seen=mpc_intervention_seen,
            backup_intervention_seen=backup_intervention_seen,
            all_predictive_constraints_valid=all_predictive_valid,
            all_backup_constraints_valid=all_backup_valid,
            chair_cleared=chair_cleared,
            route_complete=route_complete,
            failure_reason=failure_reason,
        ),
        previous_lidar_frame_id,
    )


def main() -> None:
    """Run the controlled real-chair semantic intervention experiment."""

    print("\n" + "=" * 78)
    print("YOLO SAFE-SAGE CONTROLLED CHAIR INTERVENTION")
    print("=" * 78)
    model_path = integration.resolve_model_path()
    loaded_scene = load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=EXPECTED_METERS_PER_UNIT,
    )
    robot_placement = place_existing_prim_world_origin(
        loaded_scene.stage,
        prim_path=ROBOT_PRIM_PATH,
        target_world_origin=ROBOT_SCENARIO_WORLD_ORIGIN,
    )
    scene_robot_pose = integration.read_chassis_pose(loaded_scene.stage)
    chair_world_xy = world_point_from_robot_offset(
        scene_robot_pose,
        forward=CHAIR_FORWARD_OFFSET,
        left=CHAIR_LATERAL_OFFSET,
    )
    placement = place_existing_prim_world_origin(
        loaded_scene.stage,
        prim_path=CONTROLLED_CHAIR_PRIM_PATH,
        target_world_origin=(chair_world_xy[0], chair_world_xy[1], 0.0),
    )
    goal_world_xy = world_point_from_robot_offset(
        scene_robot_pose,
        forward=NOMINAL_GOAL_FORWARD_DISTANCE,
        left=0.0,
    )
    nominal_line_crosses_chair = line_segment_intersects_planar_bounds(
        scene_robot_pose[:2],
        goal_world_xy,
        obstacle_bounds=placement.placed_bounds,
    )
    if not nominal_line_crosses_chair:
        raise RuntimeError(
            "The controlled chair does not intersect the nominal waypoint line."
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
    floor_height = integration.estimate_floor_height_in_robot_frame(
        loaded_scene.stage
    )
    detection_config = ObstacleDetectionConfig(
        minimum_height=(
            floor_height + integration.DETECTION_HEIGHT_ABOVE_FLOOR[0]
        ),
        maximum_height=(
            floor_height + integration.DETECTION_HEIGHT_ABOVE_FLOOR[1]
        ),
        maximum_range=integration.DETECTION_MAXIMUM_RANGE,
        forward_only=True,
        angular_bin_size_degrees=(
            integration.DETECTION_ANGULAR_BIN_DEGREES
        ),
    )
    class_policy, guidance_policy, perception_runtime = (
        integration.create_policies_and_runtime(
            tangential_buffer_radius=TANGENTIAL_BUFFER_RADIUS,
            enable_temporal_updates=False,
        )
    )
    bridge = SafeSAGESemanticPerceptionBridge(
        perception_runtime=perception_runtime,
        fusion=CameraLidarSemanticFusion(integration.FUSION_CONFIG),
        semantic_memory=SemanticTrackMemory(integration.MEMORY_CONFIG),
        config=SafeSAGESemanticBridgeConfig(
            maximum_sensor_time_offset_seconds=(
                integration.MAXIMUM_SENSOR_TIME_OFFSET_SECONDS
            )
        ),
    )
    detector = integration.make_detector(model_path)

    print(
        f"\nHolding Nova Carter still for {integration.SETTLE_STEPS} "
        "physics steps..."
    )
    if not advance_playing_steps(
        simulation_app,
        physics_steps=integration.SETTLE_STEPS,
        on_step=controller.stop,
    ):
        return
    initial_pose = integration.read_chassis_pose(loaded_scene.stage)
    previous_lidar_frame_id: int | None = None
    results: list[SafeSAGESemanticPerceptionResult] = []
    diagnostics: list[ControlledTrackDiagnostic] = []
    final_image: np.ndarray | None = None
    final_detections = None

    print("\n" + "=" * 78)
    print("STAGE A/B - STATIONARY CHAIR SEMANTICS AND FIELD GATE")
    print("=" * 78)
    print(
        "  Robot scenario origin:       "
        f"({robot_placement.target_world_origin[0]:.3f}, "
        f"{robot_placement.target_world_origin[1]:.3f}, "
        f"{robot_placement.target_world_origin[2]:.3f}) m"
    )
    print(f"  Chair prim:                  {placement.prim_path}")
    print(
        "  Chair target origin:         "
        f"({placement.target_world_origin[0]:.3f}, "
        f"{placement.target_world_origin[1]:.3f}, "
        f"{placement.target_world_origin[2]:.3f}) m"
    )
    print(
        "  Chair world bounds XY:       "
        f"[{placement.placed_bounds.minimum[0]:.3f}, "
        f"{placement.placed_bounds.maximum[0]:.3f}] x "
        f"[{placement.placed_bounds.minimum[1]:.3f}, "
        f"{placement.placed_bounds.maximum[1]:.3f}] m"
    )
    print(
        "  Straight waypoint goal:      "
        f"({goal_world_xy[0]:.3f}, {goal_world_xy[1]:.3f}) m"
    )
    print(f"  Nominal line crosses chair:  {nominal_line_crosses_chair}")
    print("\nStationary paired observations:")

    for sample_index in range(STATIONARY_SAMPLE_COUNT):
        (
            result,
            image_rgb,
            detections,
            previous_lidar_frame_id,
            waited_updates,
        ) = capture_semantic_update(
            stage=loaded_scene.stage,
            camera=camera,
            lidar=lidar,
            controller=controller,
            detector=detector,
            bridge=bridge,
            robot_from_lidar=robot_from_lidar,
            camera_from_robot=camera_from_robot,
            calibration=calibration,
            detection_config=detection_config,
            previous_lidar_frame_id=previous_lidar_frame_id,
            initialize=(sample_index == 0),
        )
        diagnostic = controlled_track_diagnostic(
            result,
            chair_bounds=placement.placed_bounds,
        )
        results.append(result)
        diagnostics.append(diagnostic)
        print(
            f"  sample={sample_index + 1:<2} "
            f"scan={previous_lidar_frame_id:<4} wait={waited_updates:<3} "
            f"detections={len(detections.detections):<2} "
            f"tracks={len(result.temporal_semantics.tracks):<3} "
            f"chair_track={diagnostic.track_id:<3} "
            f"error={diagnostic.association_error:.3f}m "
            f"class={diagnostic.semantic_class:<8} "
            f"status={diagnostic.temporal_status:<9} "
            f"field={result.field_status.generation:<2} "
            f"build={result.field_status.snapshot.build_wall_seconds:.3f}s"
        )
        final_image = image_rgb
        final_detections = detections

    controller.stop()
    app_utils.update_app(steps=5)
    stationary_final_pose = integration.read_chassis_pose(loaded_scene.stage)
    if final_image is None or final_detections is None:
        raise RuntimeError("No stationary image was retained for visualization.")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    overlay_path = OUTPUT_DIRECTORY / "controlled_chair_fusion_overlay.png"
    overlay = render_camera_lidar_fusion_overlay(
        final_image,
        final_detections,
        results[-1].fusion,
        point_radius=2,
    )
    Image.fromarray(overlay).save(overlay_path)

    final_diagnostic = diagnostics[-1]
    fields_valid = all(integration.numerical_field_is_valid(item) for item in results)
    generations_valid = (
        [item.field_status.generation for item in results]
        == list(range(1, STATIONARY_SAMPLE_COUNT + 1))
    )
    maximum_sync_offset = max(
        item.sensor_time_offset_seconds for item in results
    )
    position_drift = float(
        np.linalg.norm(stationary_final_pose[:2] - initial_pose[:2])
    )
    yaw_drift = integration.wrapped_angle_difference(
        stationary_final_pose[2],
        initial_pose[2],
    )
    robot_stationary = (
        position_drift <= integration.MAXIMUM_ALLOWED_POSITION_DRIFT
        and yaw_drift <= integration.MAXIMUM_ALLOWED_YAW_DRIFT
    )
    chair_policy = class_policy.parameters_for(EXPECTED_SEMANTIC_CLASS)
    chair_guidance = guidance_policy.parameters_for(EXPECTED_SEMANTIC_CLASS)
    stationary_passed = bool(
        len(results) == STATIONARY_SAMPLE_COUNT
        and final_diagnostic.is_confirmed_chair
        and final_diagnostic.association_error
        <= MAXIMUM_CHAIR_TRACK_ASSOCIATION_ERROR
        and fields_valid
        and generations_valid
        and maximum_sync_offset
        <= integration.MAXIMUM_SENSOR_TIME_OFFSET_SECONDS
        and robot_stationary
        and nominal_line_crosses_chair
    )

    print("\nStationary gate report:")
    print(f"  Controlled track ID:         {final_diagnostic.track_id}")
    print(
        "  Controlled association err:  "
        f"{final_diagnostic.association_error:.4f} m"
    )
    print(f"  Confirmed semantic class:    {final_diagnostic.semantic_class}")
    print(f"  Temporal status:             {final_diagnostic.temporal_status}")
    print(f"  Chair risk:                  {chair_policy.risk:.3f}")
    print(f"  Chair normal flux b:         {chair_policy.boundary_flux:.3f}")
    print(
        "  Chair tangential lambda:     "
        f"{chair_guidance.tangential_strength:.3f}"
    )
    print(f"  Chair passing side:          {chair_guidance.passing_side}")
    print(f"  Tangential buffer radius:    {TANGENTIAL_BUFFER_RADIUS:.3f} m")
    print("  Static temporal derivative:  0.000 (controlled static chair)")
    print(f"  Sequential fields:           {generations_valid}")
    print(f"  Numerical fields finite:     {fields_valid}")
    print(f"  Maximum sensor offset:       {maximum_sync_offset:.4f} s")
    print(f"  Robot remained stationary:   {robot_stationary}")
    print(f"  Saved fusion overlay:        {overlay_path}")
    print("\nControlled-data boundary:")
    print("  Ground truth placed and scored the chair only.")
    print("  LiDAR created geometry and persistent track IDs.")
    print("  YOLO masks supplied class evidence for those tracks.")
    print("  Only the fused live field is visible to MPC and CBF.")
    if stationary_passed:
        print(
            "\n  [PASS] A real chair asset was confirmed by YOLO--LiDAR "
            "fusion and entered a finite class-aware Safe-SAGE field."
        )
    else:
        print(
            "\n  [WARNING] The stationary chair gate did not pass every "
            "criterion; wheel motion remains disabled."
        )
    print("=" * 78)

    if not stationary_passed:
        return

    motion_result, previous_lidar_frame_id = run_intervention_motion(
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
        goal=Waypoint(x=float(goal_world_xy[0]), y=float(goal_world_xy[1])),
        chair_bounds=placement.placed_bounds,
    )
    del previous_lidar_frame_id
    motion_passed = bool(
        motion_result.completed_control_cycles > 0
        and motion_result.successful_field_updates > 0
        and motion_result.chair_semantic_updates > 0
        and motion_result.mpc_intervention_seen
        and motion_result.maximum_rightward_displacement
        >= MOTION_MINIMUM_RIGHTWARD_DISPLACEMENT
        and motion_result.minimum_ground_truth_clearance
        > GROUND_TRUTH_CLEARANCE_ABORT
        and motion_result.chair_cleared
        and motion_result.all_predictive_constraints_valid
        and motion_result.all_backup_constraints_valid
        and motion_result.failure_reason is None
    )

    print("\n" + "=" * 78)
    print("YOLO SAFE-SAGE CONTROLLED CHAIR BYPASS RESULT")
    print("=" * 78)
    print(f"  Completed control cycles:       {motion_result.completed_control_cycles}")
    print(f"  Live semantic field updates:    {motion_result.successful_field_updates}")
    print(f"  Confirmed chair updates:        {motion_result.chair_semantic_updates}")
    print(
        "  Reassociated semantic updates: "
        f"{motion_result.reassociated_semantic_updates}"
    )
    print(
        "  Controlled track-ID changes:    "
        f"{motion_result.controlled_track_id_changes}"
    )
    print(
        "  Conservative fallback updates: "
        f"{motion_result.successful_field_updates - motion_result.chair_semantic_updates}"
    )
    print(f"  Forward displacement:           {motion_result.forward_displacement:.4f} m")
    print(
        "  Maximum robot-right motion:     "
        f"{motion_result.maximum_rightward_displacement:.4f} m"
    )
    print(
        "  Minimum independent clearance:  "
        f"{motion_result.minimum_ground_truth_clearance:.4f} m"
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
    print(f"  Chair fully cleared:             {motion_result.chair_cleared}")
    print(f"  Nominal route complete:          {motion_result.route_complete}")
    print(f"  Failure reason:                  {motion_result.failure_reason}")
    print("\nInterpretation:")
    if motion_passed:
        print(
            "  [PASS] The obstacle-unaware waypoint aimed through the chair, "
            "while live YOLO--LiDAR semantics caused Safe-SAGE MPC + CBF "
            "to intervene and physically bypass it on robot-right."
        )
        print(
            "  When the chair left the camera view, unlabeled LiDAR tracks "
            "used the configured conservative unknown-class policy."
        )
        if (
            motion_result.controlled_track_id_changes > 0
            and motion_result.reassociated_semantic_updates == 0
        ):
            print(
                "  Observed ID changes were ambiguous cluster fragments, so "
                "semantic evidence was not transferred between them."
            )
    else:
        print(
            "  [WARNING] The controlled intervention did not pass every "
            "criterion. The controller stopped fail-closed."
        )
    print("=" * 78)
    controller.stop()
    app_utils.stop()

    if SIMULATION_HEADLESS or AUTOCLOSE_AFTER_REPORT:
        print("\nAutomated run complete; closing Isaac Sim.")
        return
    print("\nExpected scene behavior:")
    print("  - One real restaurant chair is centered on the straight waypoint.")
    print("  - Nova Carter begins on a direct collision course with that chair.")
    print("  - Safe-SAGE turns the robot to its right; no bypass path is supplied.")
    print("  - The robot passes the chair without touching it.")
    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when finished.")
    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - demo must close SimulationApp
        print("\n" + "=" * 78)
        print("ERROR DURING YOLO SAFE-SAGE CHAIR INTERVENTION DEMO")
        print("=" * 78)
        print(error)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
