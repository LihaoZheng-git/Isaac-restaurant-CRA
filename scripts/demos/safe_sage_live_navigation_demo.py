"""Run direct Safe-SAGE control with a field rebuilt from live LiDAR.

This demo is the live-perception successor to
``safe_sage_unicycle_mpc_motion_demo.py``.  It deliberately preserves the
validated control architecture:

1. An obstacle-unaware waypoint controller produces the nominal command and
   nominal MPC rollout.
2. The newest complete LiDAR scan is transformed into a fixed trial frame.
3. Tracking, occupancy, configuration space, Laplace guidance, and
   ``h_full(x, y, psi)`` are rebuilt transactionally.
4. Direct unicycle Safe-SAGE MPC filters the nominal ``(v, omega)`` sequence.
5. An analytical CBF rechecks the first MPC command at every physics step.
6. Only the resulting safe command reaches Nova Carter's wheel controller.

No obstacle-aware reference path is generated.  The known blue proxy pose is
used only to assign its controlled semantic label and to provide an
independent collision-abort measurement.  It is never passed to MPC or the
analytical filter.  The proxy is static, so the motion-compensated fixed-frame
field uses ``partial h / partial t = 0`` in this experiment.

Run from the project root:

    python -u scripts/demos/safe_sage_live_navigation_demo.py

Playback starts automatically.  Set ``ISAAC_SIM_HEADLESS=1`` for a headless
run, or ``SAFE_SAGE_AUTOCLOSE=1`` to render normally and close after the
acceptance report (useful when RTX LiDAR needs an active renderer).
"""

import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# This import creates the one required SimulationApp.  Its guarded main()
# function is not executed.
from scripts.demos import safe_sage_unicycle_mpc_motion_demo as fixed


motion = fixed.motion
stationary = fixed.stationary
simulation_app = fixed.simulation_app


import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Usd, UsdGeom

from robot.control.differential_drive import NovaCarterController
from robot.navigation.waypoint_controller import RobotPose2D, WaypointCommand
from robot.runtime.simulation_loop import advance_playing_steps, wait_until_playing
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.obstacle_detection import ObstacleDetectionConfig
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig
from robot.safety.safe_sage_guidance_policy import PassingSide
from robot.safety.safe_sage_live_field import SafeSAGELiveFieldConfig
from robot.safety.safe_sage_runtime import (
    SafeSAGEControlRuntime,
    SafeSAGEPerceptionObservation,
    SafeSAGEPerceptionRuntime,
    SafeSAGERuntimeField,
)
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFResult
from robot.safety.safe_sage_unicycle_mpc import (
    SafeSAGEUnicycleMPC,
    SafeSAGEUnicycleMPCResult,
)
from robot.safety.semantic_label_provider import (
    NearestWorldPositionSemanticLabelProvider,
    SemanticClassLabel,
)
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import compute_relative_prim_transform
from robot.sensor.lidar import LidarFrameInfo, NovaCarterLidar


LIVE_FIELD_CONFIG = SafeSAGELiveFieldConfig(
    maximum_field_age_seconds=0.35,
    maximum_reused_control_cycles=2,
)
LIVE_SAFETY_SUBSTEPS_PER_INTERVAL = 4
AUTOCLOSE_AFTER_REPORT = os.environ.get("SAFE_SAGE_AUTOCLOSE", "0") == "1"
MINIMUM_REQUIRED_FIELD_UPDATES = 5
EXECUTED_MARKER_ROOT = "/World/SafeSAGELiveNavigation/Executed"
EXECUTED_MARKER_COLOR = (0.10, 0.95, 0.35)
FINAL_MARKER_COLOR = (1.00, 0.10, 0.80)
MARKER_RADIUS = 0.035
MARKER_HEIGHT_WORLD = 0.06


@dataclass(frozen=True)
class LiveNavigationResult:
    """Outcome and diagnostics for direct live-field Safe-SAGE control."""

    termination_reason: str
    completed_control_cycles: int
    completed_physics_steps: int
    initial_pose: RobotPose2D
    final_pose: RobotPose2D
    executed_chassis_states: np.ndarray
    initial_sensor_frame_id: int
    final_sensor_frame_id: int
    successful_field_updates: int
    reused_field_cycles: int
    maximum_consecutive_reuses: int
    maximum_field_age_seconds: float
    field_update_failures: int
    distinct_proxy_track_ids: tuple[int, ...]
    minimum_proxy_point_count: int
    maximum_proxy_association_error: float
    warm_start_revalidation_count: int
    mpc_intervention_seen: bool
    backup_intervention_seen: bool
    all_predictive_constraints_valid: bool
    all_backup_constraints_valid: bool
    cleared_proxy: bool
    mpc_failure_seen: bool
    backup_failure_seen: bool
    perception_failure_seen: bool
    stale_field_stop_seen: bool
    ground_truth_abort_seen: bool
    application_stopped: bool
    minimum_policy_clearance: float
    minimum_predicted_safety: float
    minimum_predicted_margin: float
    minimum_backup_safety: float
    minimum_backup_margin: float
    maximum_rightward_displacement: float
    mean_field_build_wall_seconds: float
    maximum_field_build_wall_seconds: float
    mean_mpc_solve_wall_seconds: float
    maximum_mpc_solve_wall_seconds: float


def simulation_time_seconds() -> float:
    """Return the current Isaac timeline time in simulated seconds."""

    return float(omni.timeline.get_timeline_interface().get_current_time())


def make_perception_observation(
    *,
    points_sensor_frame: np.ndarray,
    frame_info: LidarFrameInfo,
    robot_pose: RobotPose2D,
    robot_from_lidar: np.ndarray,
    detection_config: ObstacleDetectionConfig,
) -> SafeSAGEPerceptionObservation:
    """Adapt one real RTX LiDAR scan to the reusable runtime boundary."""

    obstacle_points = stationary.extract_obstacle_points(
        points_sensor_frame,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
    )
    return SafeSAGEPerceptionObservation(
        sensor_frame_id=frame_info.sensor_frame_id,
        sensor_timestamp_seconds=(
            frame_info.sensor_timestamp_nanoseconds * 1e-9
        ),
        simulation_time_seconds=frame_info.simulation_time_seconds,
        obstacle_points_robot_xy=obstacle_points,
        robot_position_world_xy=np.array(
            [robot_pose.x, robot_pose.y],
            dtype=np.float64,
        ),
        robot_yaw=robot_pose.yaw,
    )


def controlled_proxy_label_provider(
    proxy_position_world_xy: np.ndarray,
) -> NearestWorldPositionSemanticLabelProvider:
    """Assign the manual test label to the nearest real LiDAR track."""

    return NearestWorldPositionSemanticLabelProvider(
        proxy_position_world_xy,
        SemanticClassLabel(
            stationary.SEMANTIC_CLASS,
            confidence=1.0,
            source="manual",
        ),
        maximum_association_distance=stationary.PROXY_ASSOCIATION_RADIUS,
    )


def newest_lidar_frame(
    lidar: NovaCarterLidar,
    *,
    previous_frame_id: int,
) -> tuple[np.ndarray, LidarFrameInfo] | None:
    """Return a newer complete scan, or ``None`` if one is not ready."""

    try:
        points_sensor_frame = lidar.get_point_cloud()
        frame_info = lidar.get_frame_info()
    except RuntimeError as error:
        if any(
            message in str(error)
            for message in stationary.TRANSIENT_LIDAR_ERRORS
        ):
            return None
        raise
    if frame_info.sensor_frame_id < previous_frame_id:
        raise RuntimeError(
            "LiDAR frame ID moved backward from "
            f"{previous_frame_id} to {frame_info.sensor_frame_id}."
        )
    if frame_info.sensor_frame_id == previous_frame_id:
        return None
    return points_sensor_frame, frame_info


def nominal_sequence(
    waypoint_controller: fixed.WaypointController,
    robot_pose: RobotPose2D,
    *,
    mpc: SafeSAGEUnicycleMPC,
) -> tuple[WaypointCommand, np.ndarray]:
    """Roll out only the obstacle-unaware waypoint feedback policy."""

    command = waypoint_controller.compute_command(robot_pose)
    predicted = waypoint_controller.predict_commands(
        robot_pose,
        horizon_steps=mpc.config.horizon_steps,
        time_step=mpc.config.time_step,
    )
    sequence = np.array(
        [
            [item.linear_velocity, item.angular_velocity]
            for item in predicted
        ],
        dtype=np.float64,
    )
    return command, sequence


def chassis_state_in_field(
    pose: RobotPose2D,
    *,
    anchor_pose: RobotPose2D,
) -> np.ndarray:
    """Return ``(x, y, yaw)`` in the fixed first-scan frame."""

    position, yaw = motion.current_pose_in_field_frame(
        pose,
        anchor_pose=anchor_pose,
    )
    return np.array([position[0], position[1], yaw], dtype=np.float64)


def print_live_cycle(
    *,
    cycle: int,
    status_frame: SafeSAGERuntimeField,
    field_updated: bool,
    field_age: float,
    reused_cycles: int,
    chassis_state: np.ndarray,
    clearance: float,
    rightward: float,
    nominal_command: WaypointCommand,
    mpc_result: SafeSAGEUnicycleMPCResult,
    backup_result: SafeSAGEUnicycleCBFResult,
    solve_wall_seconds: float,
    warm_start_used: bool,
) -> None:
    """Print one compact live-field control status line."""

    print(
        f"  cycle={cycle:<3} scan={status_frame.sensor_frame_id:<4} "
        f"update={field_updated!s:<5} age={field_age:.3f}s "
        f"reuse={reused_cycles} track={status_frame.primary_track_id:<3} "
        f"q=({chassis_state[0]:.3f},{chassis_state[1]:.3f},"
        f"{math.degrees(chassis_state[2]):.1f}deg) "
        f"clear={clearance:.3f} right={rightward:.3f} "
        f"nom=({nominal_command.linear_velocity:.3f},"
        f"{nominal_command.angular_velocity:.3f}) "
        f"mpc=({mpc_result.linear_velocity:.3f},"
        f"{mpc_result.angular_velocity:.3f}) "
        f"out=({backup_result.safe_linear_velocity:.3f},"
        f"{backup_result.safe_angular_velocity:.3f}) "
        f"hmin={mpc_result.minimum_predicted_safety_value:.3f} "
        f"margin={mpc_result.minimum_dtcbf_margin:.2e} "
        f"solve={solve_wall_seconds:.3f}s warm={warm_start_used} "
        f"status={mpc_result.status}"
    )


def run_live_navigation(
    *,
    stage: Usd.Stage,
    lidar: NovaCarterLidar,
    robot_controller: NovaCarterController,
    perception_runtime: SafeSAGEPerceptionRuntime,
    control_runtime: SafeSAGEControlRuntime,
    robot_from_lidar: np.ndarray,
    detection_config: ObstacleDetectionConfig,
    grid_config: LocalOccupancyGridConfig,
    passing_side: PassingSide,
) -> LiveNavigationResult:
    """Run direct live-field MPC plus full-rate analytical filtering."""

    initial_pose = stationary.read_robot_pose(stage)
    initial_field = perception_runtime.field
    anchor_pose = RobotPose2D(
        x=float(initial_field.anchor_position_world_xy[0]),
        y=float(initial_field.anchor_position_world_xy[1]),
        yaw=initial_field.anchor_yaw,
    )
    initial_sensor_frame_id = initial_field.sensor_frame_id
    mpc = control_runtime.mpc
    waypoint_controller = fixed.create_motion_waypoint_controller()
    clearance_threshold = motion.proxy_clearance_forward_threshold(initial_pose)
    preferred_direction = -1.0 if passing_side is PassingSide.RIGHT else 1.0

    executed_states = [np.zeros(3, dtype=np.float64)]
    warm_start: np.ndarray | None = None
    completed_cycles = 0
    completed_physics_steps = 0
    successful_updates = 0
    reused_field_cycles = 0
    maximum_reuses = 0
    maximum_field_age = 0.0
    field_update_failures = 0
    warm_start_revalidations = 0
    proxy_track_ids = {initial_field.primary_track_id}
    minimum_proxy_points = initial_field.selected_obstacle_point_count
    maximum_association_error = float(
        initial_field.primary_association_distance or 0.0
    )
    field_build_times = [initial_field.build_wall_seconds]
    solve_wall_times: list[float] = []

    mpc_intervention_seen = False
    backup_intervention_seen = False
    all_predictive_valid = True
    all_backup_valid = True
    cleared_proxy = False
    mpc_failure_seen = False
    backup_failure_seen = False
    perception_failure_seen = False
    stale_field_stop_seen = False
    ground_truth_abort_seen = False
    application_stopped = False
    settled_confirmation_count = 0

    minimum_clearance = motion.policy_footprint_clearance(initial_pose)
    minimum_predicted_safety = math.inf
    minimum_predicted_margin = math.inf
    minimum_backup_safety = math.inf
    minimum_backup_margin = math.inf
    maximum_rightward = 0.0
    termination_reason = "Maximum live-navigation duration reached."

    print("\n" + "=" * 78)
    print("SAFE-SAGE STAGE D - DIRECT LIVE-FIELD CLOSED-LOOP MOTION")
    print("=" * 78)
    print("\nControlled settings:")
    print("  Nominal policy:                 obstacle-unaware waypoint feedback")
    print("  Safe controller:                direct unicycle Safe-SAGE MPC")
    print("  Backup layer:                   analytical CBF every physics step")
    print("  Field frame:                    fixed at the first LiDAR scan")
    print("  Field update:                   transactional at MPC boundaries")
    print("  Precomputed bypass reference:   none")
    print("  Static temporal derivative:     0.000000")
    print(f"  Live occupancy resolution:      {grid_config.resolution:.3f} m")
    print(
        "  Inter-sample geometry spacing:  "
        f"{mpc.config.time_step / mpc.config.safety_substeps_per_interval:.3f} s"
    )
    print(f"  Persistent passing side:        {passing_side.value}")
    print(
        "  Maximum accepted field age:     "
        f"{perception_runtime.field_buffer.config.maximum_field_age_seconds:.3f} s"
    )
    print(
        "  Maximum consecutive reuses:     "
        f"{perception_runtime.field_buffer.config.maximum_reused_control_cycles}"
    )
    print(f"  MPC control interval:           {fixed.MPC_CONTROL_INTERVAL_SECONDS:.3f} s")
    print(f"  Physics steps per command:      {fixed.CONTROL_PHYSICS_STEPS}")
    print(f"  Proxy-clear threshold:          {clearance_threshold:.3f} m")
    print("  MPC runtime acceptance limit:   none")
    print("\nLive navigation status:")

    for cycle in range(fixed.MAXIMUM_CONTROL_CYCLES):
        if not simulation_app.is_running():
            application_stopped = True
            termination_reason = "Isaac Sim stopped during live navigation."
            break
        while simulation_app.is_running() and not app_utils.is_playing():
            robot_controller.stop()
            simulation_app.update()
        if not simulation_app.is_running():
            application_stopped = True
            termination_reason = "Isaac Sim stopped during live navigation."
            break

        robot_controller.stop()
        robot_pose = stationary.read_robot_pose(stage)

        try:
            new_frame = newest_lidar_frame(
                lidar,
                previous_frame_id=(
                    perception_runtime.field_buffer.snapshot.sensor_frame_id
                ),
            )
        except (RuntimeError, TypeError, ValueError) as error:
            perception_failure_seen = True
            field_update_failures += 1
            termination_reason = f"Fail-closed LiDAR read stop: {error}"
            break

        if new_frame is None:
            reused_field_cycles += 1
            try:
                field_status = perception_runtime.reuse(
                    current_simulation_time_seconds=simulation_time_seconds(),
                )
            except (RuntimeError, TypeError, ValueError) as error:
                stale_field_stop_seen = True
                termination_reason = f"Fail-closed field-clock stop: {error}"
                break
        else:
            points_sensor_frame, frame_info = new_frame
            try:
                observation = make_perception_observation(
                    points_sensor_frame=points_sensor_frame,
                    frame_info=frame_info,
                    robot_pose=robot_pose,
                    robot_from_lidar=robot_from_lidar,
                    detection_config=detection_config,
                )
                field_status = perception_runtime.update(
                    observation,
                    controlled_proxy_label_provider(
                        stationary.STATIC_PROXY_POSITION_WORLD[:2]
                    ),
                )
            except (RuntimeError, TypeError, ValueError) as error:
                perception_failure_seen = True
                field_update_failures += 1
                termination_reason = (
                    "Fail-closed perception/PDE update stop: " f"{error}"
                )
                break
            candidate = field_status.snapshot.payload
            successful_updates += 1
            field_build_times.append(candidate.build_wall_seconds)
            proxy_track_ids.add(candidate.primary_track_id)
            minimum_proxy_points = min(
                minimum_proxy_points,
                candidate.selected_obstacle_point_count,
            )
            maximum_association_error = max(
                maximum_association_error,
                float(candidate.primary_association_distance or 0.0),
            )

        maximum_reuses = max(
            maximum_reuses,
            field_status.reused_control_cycles,
        )
        maximum_field_age = max(
            maximum_field_age,
            field_status.field_age_seconds,
        )
        if field_status.fail_closed:
            stale_field_stop_seen = True
            termination_reason = (
                "Fail-closed stale-field stop: "
                f"{field_status.failure_reason}"
            )
            break

        field = field_status.snapshot.payload
        nominal_command, nominal_commands = nominal_sequence(
            waypoint_controller,
            robot_pose,
            mpc=mpc,
        )
        if nominal_command.route_complete:
            termination_reason = "The nominal waypoint route completed."
            break

        chassis_state = chassis_state_in_field(
            robot_pose,
            anchor_pose=anchor_pose,
        )
        warm_start_used = warm_start is not None
        if warm_start_used:
            # The old-field plan is only a numerical seed.  MPC evaluates it
            # again against the newly committed field before any command can
            # be accepted.
            warm_start_revalidations += 1
        solve_start = time.perf_counter()
        mpc_result = control_runtime.plan(
            chassis_state,
            nominal_commands,
            field,
            initial_command_guess=warm_start,
            preferred_lateral_direction_grid_y=preferred_direction,
        )
        solve_wall_seconds = time.perf_counter() - solve_start
        solve_wall_times.append(solve_wall_seconds)
        completed_cycles += 1

        predictive_valid = fixed.predictive_result_is_valid(
            mpc_result,
            constraint_tolerance=mpc.config.constraint_tolerance,
        )
        all_predictive_valid &= predictive_valid
        if not predictive_valid:
            mpc_failure_seen = True
            termination_reason = f"Fail-closed MPC stop: {mpc_result.status}"
            break

        warm_start = control_runtime.shifted_warm_start(mpc_result)
        mpc_intervention_seen |= mpc_result.intervention_active
        minimum_predicted_safety = min(
            minimum_predicted_safety,
            float(np.min(mpc_result.predicted_safety_values)),
        )
        minimum_predicted_margin = min(
            minimum_predicted_margin,
            float(np.min(mpc_result.dtcbf_margins)),
        )

        selected_command = np.array(
            [mpc_result.linear_velocity, mpc_result.angular_velocity],
            dtype=np.float64,
        )
        backup_result = control_runtime.filter_command(
            selected_command,
            chassis_state,
            field,
        )
        backup_valid = fixed.backup_result_is_valid(backup_result)
        all_backup_valid &= backup_valid
        if not backup_valid:
            backup_failure_seen = True
            termination_reason = (
                "Fail-closed analytical backup stop: "
                f"{backup_result.emergency_reason or 'invalid constraint'}"
            )
            break

        backup_intervention_seen |= backup_result.intervention_active
        minimum_backup_safety = min(
            minimum_backup_safety,
            backup_result.sampled_safety_value,
        )
        minimum_backup_margin = min(
            minimum_backup_margin,
            backup_result.cbf_margin,
        )
        current_clearance = motion.policy_footprint_clearance(robot_pose)
        forward, rightward = motion.forward_and_rightward_displacement(
            robot_pose,
            initial_pose=initial_pose,
        )
        minimum_clearance = min(minimum_clearance, current_clearance)
        maximum_rightward = max(maximum_rightward, rightward)
        print_live_cycle(
            cycle=cycle,
            status_frame=field,
            field_updated=field_status.field_updated,
            field_age=field_status.field_age_seconds,
            reused_cycles=field_status.reused_control_cycles,
            chassis_state=chassis_state,
            clearance=current_clearance,
            rightward=rightward,
            nominal_command=nominal_command,
            mpc_result=mpc_result,
            backup_result=backup_result,
            solve_wall_seconds=solve_wall_seconds,
            warm_start_used=warm_start_used,
        )

        if current_clearance <= fixed.GROUND_TRUTH_CLEARANCE_ABORT:
            ground_truth_abort_seen = True
            termination_reason = "Independent ground-truth clearance guard triggered."
            break
        if forward >= clearance_threshold:
            cleared_proxy = True
            termination_reason = "The robot cleared the proxy longitudinally."
            break

        settled = (
            backup_result.safe_linear_velocity <= fixed.SETTLED_LINEAR_SPEED
            and abs(backup_result.safe_angular_velocity)
            <= fixed.SETTLED_ANGULAR_SPEED
        )
        settled_confirmation_count = (
            settled_confirmation_count + 1 if settled else 0
        )
        if settled_confirmation_count >= fixed.SETTLED_CONFIRMATION_CYCLES:
            termination_reason = "The safe command settled before proxy clearance."
            break

        execution_steps = perception_runtime.fresh_execution_step_budget(
            current_simulation_time_seconds=simulation_time_seconds(),
            physics_time_step_seconds=stationary.PHYSICS_DT,
            requested_steps=fixed.CONTROL_PHYSICS_STEPS,
        )
        if execution_steps == 0:
            # Waiting stationary for the next RTX frame is safer than
            # extending an old field beyond its freshness contract.
            robot_controller.stop()
            simulation_app.update()
            completed_physics_steps += 1
            continue

        command_segment_interrupted = False
        for _ in range(execution_steps):
            if not simulation_app.is_running():
                application_stopped = True
                termination_reason = "Isaac Sim stopped during command execution."
                command_segment_interrupted = True
                break
            while simulation_app.is_running() and not app_utils.is_playing():
                robot_controller.stop()
                simulation_app.update()
            if not simulation_app.is_running():
                application_stopped = True
                termination_reason = "Isaac Sim stopped during command execution."
                command_segment_interrupted = True
                break

            try:
                step_status = perception_runtime.inspect(
                    current_simulation_time_seconds=simulation_time_seconds(),
                )
            except (RuntimeError, TypeError, ValueError) as error:
                stale_field_stop_seen = True
                termination_reason = f"Fail-closed field-clock stop: {error}"
                command_segment_interrupted = True
                break
            maximum_field_age = max(
                maximum_field_age,
                step_status.field_age_seconds,
            )
            if step_status.fail_closed:
                stale_field_stop_seen = True
                termination_reason = (
                    "Fail-closed stale-field stop during command execution: "
                    f"{step_status.failure_reason}"
                )
                command_segment_interrupted = True
                break

            monitored_pose = stationary.read_robot_pose(stage)
            monitored_state = chassis_state_in_field(
                monitored_pose,
                anchor_pose=anchor_pose,
            )
            step_backup = control_runtime.filter_command(
                selected_command,
                monitored_state,
                step_status.snapshot.payload,
            )
            step_backup_valid = fixed.backup_result_is_valid(step_backup)
            all_backup_valid &= step_backup_valid
            if not step_backup_valid:
                backup_failure_seen = True
                termination_reason = (
                    "Fail-closed full-rate analytical backup stop: "
                    f"{step_backup.emergency_reason or 'invalid constraint'}"
                )
                command_segment_interrupted = True
                break

            backup_intervention_seen |= step_backup.intervention_active
            minimum_backup_safety = min(
                minimum_backup_safety,
                step_backup.sampled_safety_value,
            )
            minimum_backup_margin = min(
                minimum_backup_margin,
                step_backup.cbf_margin,
            )
            robot_controller.apply_command(
                step_backup.safe_linear_velocity,
                step_backup.safe_angular_velocity,
            )
            simulation_app.update()
            completed_physics_steps += 1

            monitored_pose = stationary.read_robot_pose(stage)
            monitored_state = chassis_state_in_field(
                monitored_pose,
                anchor_pose=anchor_pose,
            )
            executed_states.append(monitored_state)
            monitored_clearance = motion.policy_footprint_clearance(
                monitored_pose
            )
            monitored_forward, monitored_rightward = (
                motion.forward_and_rightward_displacement(
                    monitored_pose,
                    initial_pose=initial_pose,
                )
            )
            minimum_clearance = min(
                minimum_clearance,
                monitored_clearance,
            )
            maximum_rightward = max(
                maximum_rightward,
                monitored_rightward,
            )
            if monitored_clearance <= fixed.GROUND_TRUTH_CLEARANCE_ABORT:
                ground_truth_abort_seen = True
                termination_reason = (
                    "Independent ground-truth clearance guard triggered "
                    "during command execution."
                )
                command_segment_interrupted = True
                break
            if monitored_forward >= clearance_threshold:
                cleared_proxy = True
                termination_reason = "The robot cleared the proxy longitudinally."
                command_segment_interrupted = True
                break

        if command_segment_interrupted:
            break

    if simulation_app.is_running():
        robot_controller.stop()
        app_utils.update_app(steps=5)
        final_pose = stationary.read_robot_pose(stage)
    else:
        final_pose = stationary.read_robot_pose(stage)

    final_clearance = motion.policy_footprint_clearance(final_pose)
    minimum_clearance = min(minimum_clearance, final_clearance)
    _, final_rightward = motion.forward_and_rightward_displacement(
        final_pose,
        initial_pose=initial_pose,
    )
    maximum_rightward = max(maximum_rightward, final_rightward)
    mean_build_time = float(np.mean(field_build_times))
    maximum_build_time = float(np.max(field_build_times))
    mean_solve_time = (
        math.nan if not solve_wall_times else float(np.mean(solve_wall_times))
    )
    maximum_solve_time = (
        math.nan if not solve_wall_times else float(np.max(solve_wall_times))
    )
    return LiveNavigationResult(
        termination_reason=termination_reason,
        completed_control_cycles=completed_cycles,
        completed_physics_steps=completed_physics_steps,
        initial_pose=initial_pose,
        final_pose=final_pose,
        executed_chassis_states=np.ascontiguousarray(
            np.vstack(executed_states),
            dtype=np.float64,
        ),
        initial_sensor_frame_id=initial_sensor_frame_id,
        final_sensor_frame_id=(
            perception_runtime.field_buffer.snapshot.sensor_frame_id
        ),
        successful_field_updates=successful_updates,
        reused_field_cycles=reused_field_cycles,
        maximum_consecutive_reuses=maximum_reuses,
        maximum_field_age_seconds=maximum_field_age,
        field_update_failures=field_update_failures,
        distinct_proxy_track_ids=tuple(sorted(proxy_track_ids)),
        minimum_proxy_point_count=minimum_proxy_points,
        maximum_proxy_association_error=maximum_association_error,
        warm_start_revalidation_count=warm_start_revalidations,
        mpc_intervention_seen=mpc_intervention_seen,
        backup_intervention_seen=backup_intervention_seen,
        all_predictive_constraints_valid=all_predictive_valid,
        all_backup_constraints_valid=all_backup_valid,
        cleared_proxy=cleared_proxy,
        mpc_failure_seen=mpc_failure_seen,
        backup_failure_seen=backup_failure_seen,
        perception_failure_seen=perception_failure_seen,
        stale_field_stop_seen=stale_field_stop_seen,
        ground_truth_abort_seen=ground_truth_abort_seen,
        application_stopped=application_stopped,
        minimum_policy_clearance=minimum_clearance,
        minimum_predicted_safety=minimum_predicted_safety,
        minimum_predicted_margin=minimum_predicted_margin,
        minimum_backup_safety=minimum_backup_safety,
        minimum_backup_margin=minimum_backup_margin,
        maximum_rightward_displacement=maximum_rightward,
        mean_field_build_wall_seconds=mean_build_time,
        maximum_field_build_wall_seconds=maximum_build_time,
        mean_mpc_solve_wall_seconds=mean_solve_time,
        maximum_mpc_solve_wall_seconds=maximum_solve_time,
    )


def field_state_to_world_xy(
    state: np.ndarray,
    *,
    anchor_pose: RobotPose2D,
) -> tuple[float, float]:
    """Transform one fixed-field state into a world planar position."""

    cosine = math.cos(anchor_pose.yaw)
    sine = math.sin(anchor_pose.yaw)
    return (
        anchor_pose.x + cosine * float(state[0]) - sine * float(state[1]),
        anchor_pose.y + sine * float(state[0]) + cosine * float(state[1]),
    )


def create_executed_path_markers(
    stage: Usd.Stage,
    states: np.ndarray,
    *,
    anchor_pose: RobotPose2D,
) -> None:
    """Draw sparse, non-colliding markers for the physically executed path."""

    existing = stage.GetPrimAtPath(EXECUTED_MARKER_ROOT)
    if existing.IsValid():
        raise RuntimeError(
            f"Path-marker root already exists: {EXECUTED_MARKER_ROOT}"
        )
    indices = list(range(0, states.shape[0], fixed.CONTROL_PHYSICS_STEPS))
    if indices[-1] != states.shape[0] - 1:
        indices.append(states.shape[0] - 1)

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage, EXECUTED_MARKER_ROOT)
        for marker_index, state_index in enumerate(indices):
            sphere = UsdGeom.Sphere.Define(
                stage,
                f"{EXECUTED_MARKER_ROOT}/Point_{marker_index:03d}",
            )
            sphere.CreateRadiusAttr(MARKER_RADIUS)
            world_x, world_y = field_state_to_world_xy(
                states[state_index],
                anchor_pose=anchor_pose,
            )
            sphere.AddTranslateOp().Set(
                Gf.Vec3d(world_x, world_y, MARKER_HEIGHT_WORLD)
            )
            color = (
                FINAL_MARKER_COLOR
                if state_index == states.shape[0] - 1
                else EXECUTED_MARKER_COLOR
            )
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def print_result(result: LiveNavigationResult) -> bool:
    """Print the live-field acceptance report and return its decision."""

    final_forward, final_rightward = motion.forward_and_rightward_displacement(
        result.final_pose,
        initial_pose=result.initial_pose,
    )
    enough_updates = (
        result.successful_field_updates >= MINIMUM_REQUIRED_FIELD_UPDATES
    )
    stable_proxy_track = len(result.distinct_proxy_track_ids) == 1
    rightward_response = (
        result.maximum_rightward_displacement
        >= fixed.MINIMUM_EXPECTED_RIGHTWARD_DISPLACEMENT
    )
    accepted = bool(
        result.cleared_proxy
        and enough_updates
        and stable_proxy_track
        and rightward_response
        and result.mpc_intervention_seen
        and result.all_predictive_constraints_valid
        and result.all_backup_constraints_valid
        and result.minimum_policy_clearance
        > fixed.GROUND_TRUTH_CLEARANCE_ABORT
        and not result.mpc_failure_seen
        and not result.backup_failure_seen
        and not result.perception_failure_seen
        and not result.stale_field_stop_seen
        and not result.ground_truth_abort_seen
        and not result.application_stopped
    )

    print("\n" + "=" * 78)
    print("SAFE-SAGE STAGE D - LIVE-FIELD MOTION RESULT")
    print("=" * 78)
    print(f"Termination:                       {result.termination_reason}")
    print(f"Completed control cycles:          {result.completed_control_cycles}")
    print(f"Completed physics steps:           {result.completed_physics_steps}")
    print("\nLive perception and field:")
    print(
        "  Initial/final LiDAR frame:       "
        f"{result.initial_sensor_frame_id} / {result.final_sensor_frame_id}"
    )
    print(f"  Successful field updates:        {result.successful_field_updates}")
    print(f"  Reused field cycles:              {result.reused_field_cycles}")
    print(f"  Maximum consecutive reuses:      {result.maximum_consecutive_reuses}")
    print(f"  Maximum field age:               {result.maximum_field_age_seconds:.3f} s")
    print(f"  Field update failures:           {result.field_update_failures}")
    print(f"  Proxy track IDs:                 {result.distinct_proxy_track_ids}")
    print(f"  Minimum proxy points:            {result.minimum_proxy_point_count}")
    print(
        "  Maximum association error:       "
        f"{result.maximum_proxy_association_error:.4f} m"
    )
    print("\nPhysical motion:")
    print(f"  Forward displacement:            {final_forward:.4f} m")
    print(f"  Final rightward displacement:    {final_rightward:.4f} m")
    print(
        "  Maximum rightward displacement:  "
        f"{result.maximum_rightward_displacement:.4f} m"
    )
    print(f"  Minimum policy clearance:        {result.minimum_policy_clearance:.4f} m")
    print("\nPredictive and analytical safety:")
    print(f"  MPC intervention observed:       {result.mpc_intervention_seen}")
    print(f"  Backup intervention observed:    {result.backup_intervention_seen}")
    print(f"  All predictive constraints:      {result.all_predictive_constraints_valid}")
    print(f"  All analytical constraints:      {result.all_backup_constraints_valid}")
    print(f"  Minimum predicted h_full:        {result.minimum_predicted_safety:.6f}")
    print(f"  Minimum predicted DTCBF margin:  {result.minimum_predicted_margin:.3e}")
    print(f"  Minimum analytical h_full:       {result.minimum_backup_safety:.6f}")
    print(f"  Minimum analytical CBF margin:   {result.minimum_backup_margin:.3e}")
    print("\nUpdate and runtime diagnostics:")
    print(f"  Warm starts revalidated:         {result.warm_start_revalidation_count}")
    print(f"  Mean field build time:           {result.mean_field_build_wall_seconds:.3f} s")
    print(f"  Maximum field build time:        {result.maximum_field_build_wall_seconds:.3f} s")
    print(f"  Mean MPC solve time:             {result.mean_mpc_solve_wall_seconds:.3f} s")
    print(f"  Maximum MPC solve time:          {result.maximum_mpc_solve_wall_seconds:.3f} s")
    print("  Runtime is diagnostic, not an acceptance criterion.")
    print("\nControlled checks:")
    print(f"  Proxy cleared:                   {result.cleared_proxy}")
    print(f"  Enough live field updates:       {enough_updates}")
    print(f"  Stable proxy track:              {stable_proxy_track}")
    print(f"  Rightward response observed:     {rightward_response}")
    print(f"  MPC failure seen:                {result.mpc_failure_seen}")
    print(f"  Analytical failure seen:         {result.backup_failure_seen}")
    print(f"  Perception failure seen:         {result.perception_failure_seen}")
    print(f"  Stale-field stop seen:           {result.stale_field_stop_seen}")
    print(f"  Clearance abort seen:            {result.ground_truth_abort_seen}")
    print(f"  Application stopped:             {result.application_stopped}")
    print("\nInterpretation:")
    if accepted:
        print(
            "  [PASS] Nova Carter completed a direct Safe-SAGE bypass while "
            "the LiDAR-derived field was rebuilt online and every predictive "
            "and full-rate analytical safety check remained valid."
        )
    else:
        print(
            "  [WARNING] The live-field trial did not pass every criterion. "
            "The controller stopped fail-closed; inspect the diagnostics "
            "before enabling a dynamic-obstacle field."
        )
    print("=" * 78)
    return accepted


def main() -> None:
    """Run the complete direct live-perception Safe-SAGE trial."""

    print("\n" + "=" * 78)
    print("NOVA CARTER SAFE-SAGE LIVE-FIELD NAVIGATION DEMO")
    print("=" * 78)
    print("Mode: live LiDAR + direct unicycle MPC + analytical CBF + wheels")
    print(f"Guidance policy: {stationary.GUIDANCE_POLICY_PATH}")
    print("Semantic label source: manual controlled proxy")

    (
        class_policy,
        guidance_policy,
        grid_config,
        field_config,
    ) = fixed.create_fixed_field_configuration()
    passing_side = guidance_policy.parameters_for(
        stationary.SEMANTIC_CLASS
    ).passing_side

    loaded_scene = load_robot_scene(
        scene_path=stationary.ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=stationary.ROBOT_PRIM_PATH,
        expected_meters_per_unit=stationary.EXPECTED_METERS_PER_UNIT,
    )
    stationary.create_static_proxy(loaded_scene.stage)
    robot_controller = NovaCarterController(
        robot_prim_path=stationary.ROBOT_PRIM_PATH
    )
    SimulationManager.setup_simulation(
        dt=stationary.PHYSICS_DT,
        device=stationary.PHYSICS_DEVICE,
    )

    print("\nStarting Isaac Sim playback automatically...")
    app_utils.play()
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
        angular_bin_size_degrees=stationary.ANGULAR_BIN_SIZE_DEGREES,
    )
    obstacle_tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=stationary.TRACKER_CLUSTER_RADIUS,
            minimum_cluster_points=stationary.TRACKER_MINIMUM_CLUSTER_POINTS,
            maximum_association_distance=(
                stationary.TRACKER_MAXIMUM_ASSOCIATION_DISTANCE
            ),
        )
    )
    perception_runtime = SafeSAGEPerceptionRuntime(
        obstacle_tracker=obstacle_tracker,
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=grid_config,
        field_config=field_config,
        live_field_config=LIVE_FIELD_CONFIG,
        # This regression deliberately preserves the validated static-field
        # baseline.  The dynamic demo enables consecutive-field derivatives.
        enable_temporal_updates=False,
    )

    print(
        "\nHolding the robot and proxy still for "
        f"{stationary.SETTLE_STEPS} physics steps..."
    )
    if not advance_playing_steps(
        simulation_app,
        physics_steps=stationary.SETTLE_STEPS,
        on_step=robot_controller.stop,
    ):
        return

    anchor_pose = stationary.read_robot_pose(loaded_scene.stage)
    print("\nWaiting for the initial live-field LiDAR snapshot...")
    points_sensor_frame, frame_info, _ = stationary.wait_for_first_scan(
        lidar,
        robot_controller,
    )
    print("Building the first yaw-dependent Safe-SAGE field...")
    initial_status = perception_runtime.initialize(
        make_perception_observation(
            points_sensor_frame=points_sensor_frame,
            frame_info=frame_info,
            robot_pose=anchor_pose,
            robot_from_lidar=robot_from_lidar,
            detection_config=detection_config,
        ),
        controlled_proxy_label_provider(
            stationary.STATIC_PROXY_POSITION_WORLD[:2]
        ),
        anchor_position_world_xy=np.array(
            [anchor_pose.x, anchor_pose.y],
            dtype=np.float64,
        ),
        anchor_yaw=anchor_pose.yaw,
    )
    initial_field = initial_status.snapshot.payload
    print(
        f"Initial field committed from LiDAR frame {frame_info.sensor_frame_id} "
        f"in {initial_field.build_wall_seconds:.3f} s."
    )

    base_mpc, backup_filter = fixed.create_controller_stack(robot_controller)
    mpc = SafeSAGEUnicycleMPC(
        base_mpc.paper_filter,
        replace(
            base_mpc.config,
            safety_substeps_per_interval=(
                LIVE_SAFETY_SUBSTEPS_PER_INTERVAL
            ),
        ),
    )
    control_runtime = SafeSAGEControlRuntime(
        mpc=mpc,
        backup_filter=backup_filter,
    )
    fixed.print_directional_field_audit(
        pipeline_frame=initial_field,
        mpc=mpc,
        backup_filter=backup_filter,
        passing_side=passing_side,
    )
    initial_frame_id = perception_runtime.field.sensor_frame_id
    result = run_live_navigation(
        stage=loaded_scene.stage,
        lidar=lidar,
        robot_controller=robot_controller,
        perception_runtime=perception_runtime,
        control_runtime=control_runtime,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
        grid_config=grid_config,
        passing_side=passing_side,
    )
    if result.initial_sensor_frame_id != initial_frame_id:
        raise RuntimeError("Live-field result corrupted its initial frame ID.")
    create_executed_path_markers(
        loaded_scene.stage,
        result.executed_chassis_states,
        anchor_pose=anchor_pose,
    )
    print_result(result)

    robot_controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()

    print("\nExpected successful scene behavior:")
    print("  - Nova Carter begins an early robot-right turn.")
    print("  - The blue semantic proxy remains fixed and is not touched.")
    print("  - The obstacle-unaware waypoint still points through the proxy.")
    print("  - Safe-SAGE MPC and CBF alone create the physical bypass.")
    print("  - Green markers show only the physically executed path.")
    print("  - No cyan precomputed bypass-reference path appears.")
    print("  - The robot stops after clearing the proxy longitudinally.")

    if stationary.SIMULATION_HEADLESS or AUTOCLOSE_AFTER_REPORT:
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
        print("ERROR DURING SAFE-SAGE LIVE-FIELD NAVIGATION DEMO")
        print("=" * 78)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
