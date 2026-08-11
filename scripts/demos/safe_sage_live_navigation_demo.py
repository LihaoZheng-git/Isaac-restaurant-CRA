"""Run the complete live-perception Safe-SAGE bypass on Nova Carter.

This demo extends the validated fixed-field
``safe_sage_planar_to_unicycle_motion_demo.py`` milestone.

Stage A - controlled initialization:

1. Load the restaurant scene, Nova Carter, and the blue semantic proxy.
2. Capture one real PandarXT-32 scan while the robot is stationary.
3. Run the planar Safe-SAGE oracle and construct the validated, buffered
   unicycle tracking path.
4. Build the first look-ahead-referenced ``h_full`` field.

Stage B - live closed-loop navigation:

1. Read the newest completed LiDAR frame at each MPC boundary.
2. Transform detections into the fixed initial-robot frame.
3. Update obstacle tracking and rebuild occupancy, configuration-space,
   Laplace-guidance, and ``h_full`` fields.
4. Solve the unicycle MPC against the newest complete field.
5. Re-evaluate the analytical CBF before every physics command.
6. Apply only ``u_safe = (v_safe, omega_safe)`` to the wheel controller.
7. Stop after proxy clearance or immediately on any perception, predictive,
   analytical, staleness, or independent-clearance failure.

The controlled proxy pose is used only to associate the manually labelled
test object and to compute the independent clearance guard. It is never
provided to the MPC or analytical CBF. Because the proxy is static and every
scan is motion-compensated into one fixed frame, ``partial h / partial t`` is
explicitly zero in this experiment.

Run from the project root:

    python -u scripts/demos/safe_sage_live_navigation_demo.py

Playback starts automatically. Set ``ISAAC_SIM_HEADLESS=1`` for a headless
acceptance run.
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


# This import initializes the one required SimulationApp. Its guarded main
# function is not executed.
from scripts.demos import safe_sage_planar_to_unicycle_motion_demo as fixed

integration = fixed.integration
motion = fixed.motion
oracle = fixed.oracle
stationary = fixed.stationary
simulation_app = fixed.simulation_app


import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd
from robot.control.differential_drive import NovaCarterController
from robot.navigation.waypoint_controller import WaypointController
from robot.runtime.simulation_loop import advance_playing_steps, wait_until_playing
from robot.safety.class_safety_policy import ClassSafetyPolicy
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.obstacle_detection import ObstacleDetectionConfig
from robot.safety.obstacle_tracking import ObstacleTracker
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidancePolicy
from robot.safety.safe_sage_unicycle_cbf_filter import (
    SafeSAGEUnicycleCBFFilter,
    SafeSAGEUnicycleCBFResult,
    shift_rectangular_footprint_to_lookahead,
)
from robot.safety.safe_sage_unicycle_mpc import (
    SafeSAGEUnicycleMPC,
    SafeSAGEUnicycleMPCResult,
)
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import compute_relative_prim_transform
from robot.sensor.lidar import LidarFrameInfo, NovaCarterLidar

MAXIMUM_STALE_FIELD_CYCLES = 2
MINIMUM_REQUIRED_LIVE_FIELD_UPDATES = 5

REFERENCE_MARKER_ROOT = "/World/SafeSAGELiveNavigation/Reference"
EXECUTED_MARKER_ROOT = "/World/SafeSAGELiveNavigation/Executed"
REFERENCE_MARKER_COLOR = (0.10, 0.85, 1.00)
EXECUTED_MARKER_COLOR = (0.10, 0.95, 0.35)


@dataclass(frozen=True)
class LiveNavigationResult:
    """Outcome and diagnostics for the live physical bypass."""

    termination_reason: str
    completed_control_cycles: int
    completed_physics_steps: int
    executed_chassis_states: np.ndarray
    final_path_progress_index: int
    live_lidar_frames: int
    successful_field_updates: int
    field_update_failures: int
    maximum_stale_field_cycles: int
    initial_sensor_frame_id: int
    final_sensor_frame_id: int
    distinct_proxy_track_ids: tuple[int, ...]
    minimum_proxy_point_count: int
    maximum_proxy_association_error: float
    all_predictive_constraints_valid: bool
    all_analytical_constraints_valid: bool
    mpc_failure_seen: bool
    analytical_failure_seen: bool
    stale_field_stop_seen: bool
    perception_failure_seen: bool
    ground_truth_abort_seen: bool
    application_stopped: bool
    cleared_proxy: bool
    refinement_used_count: int
    refinement_failure_count: int
    feasibility_recovery_seen: bool
    analytical_intervention_seen: bool
    minimum_policy_clearance: float
    minimum_predicted_safety: float
    minimum_predicted_margin: float
    minimum_analytical_safety: float
    minimum_analytical_margin: float
    maximum_rightward_displacement: float
    maximum_path_error: float
    mean_field_build_wall_seconds: float
    maximum_field_build_wall_seconds: float
    mean_mpc_solve_wall_seconds: float
    maximum_mpc_solve_wall_seconds: float


def newest_lidar_frame(
    lidar: NovaCarterLidar,
    *,
    previous_frame_id: int,
) -> tuple[np.ndarray, LidarFrameInfo] | None:
    """Return a new complete LiDAR frame, or ``None`` for no new frame."""

    try:
        points_sensor_frame = lidar.get_point_cloud()
        frame_info = lidar.get_frame_info()
    except RuntimeError as error:
        if any(message in str(error) for message in stationary.TRANSIENT_LIDAR_ERRORS):
            return None
        raise
    if frame_info.sensor_frame_id == previous_frame_id:
        return None
    return points_sensor_frame, frame_info


def build_live_field(
    *,
    points_sensor_frame: np.ndarray,
    frame_info: LidarFrameInfo,
    robot_pose: stationary.RobotPose2D,
    anchor_pose: stationary.RobotPose2D,
    obstacle_tracker: ObstacleTracker,
    robot_from_lidar: np.ndarray,
    detection_config: ObstacleDetectionConfig,
    class_policy: ClassSafetyPolicy,
    guidance_policy: SafeSAGEGuidancePolicy,
    grid_config: LocalOccupancyGridConfig,
    field_config: SafeSAGEFieldPipelineConfig,
) -> motion.MotionPipelineFrame:
    """Build one live, fixed-anchor, look-ahead-referenced Safe-SAGE field."""

    return motion.build_motion_pipeline_frame(
        points_sensor_frame=points_sensor_frame,
        frame_info=frame_info,
        robot_pose=robot_pose,
        field_anchor_pose=anchor_pose,
        obstacle_tracker=obstacle_tracker,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=grid_config,
        field_config=field_config,
        semantic_class=oracle.PAPER_BASELINE_SEMANTIC_CLASS,
    )


def print_live_cycle(
    *,
    cycle: int,
    field_frame: motion.MotionPipelineFrame,
    field_updated: bool,
    stale_field_cycles: int,
    progress_index: int,
    path_state_count: int,
    chassis_state: np.ndarray,
    clearance: float,
    rightward: float,
    path_error: float,
    mpc_result: SafeSAGEUnicycleMPCResult,
    analytical_result: SafeSAGEUnicycleCBFResult,
    solve_wall_seconds: float,
    refinement_used: bool,
) -> None:
    """Print one compact live-perception control status line."""

    print(
        f"  cycle={cycle:<3} "
        f"scan={field_frame.sensor_frame_id:<4} "
        f"update={field_updated!s:<5} stale={stale_field_cycles} "
        f"track={field_frame.proxy_track_id:<3} "
        f"path={progress_index:02d}/{path_state_count - 1:02d} "
        f"q=({chassis_state[0]:.3f},{chassis_state[1]:.3f},"
        f"{math.degrees(chassis_state[2]):.1f}deg) "
        f"clear={clearance:.3f} right={rightward:.3f} "
        f"err={path_error:.3f} "
        f"mpc=({mpc_result.linear_velocity:.3f},"
        f"{mpc_result.angular_velocity:.3f}) "
        f"out=({analytical_result.safe_linear_velocity:.3f},"
        f"{analytical_result.safe_angular_velocity:.3f}) "
        f"hmin={mpc_result.minimum_predicted_safety_value:.3f} "
        f"margin={mpc_result.minimum_dtcbf_margin:.2e} "
        f"solve={solve_wall_seconds:.3f}s "
        f"refine={refinement_used} status={mpc_result.status}"
    )


def run_live_navigation(
    *,
    stage: Usd.Stage,
    lidar: NovaCarterLidar,
    robot_controller: NovaCarterController,
    waypoint_controller: WaypointController,
    planar_path_states: np.ndarray,
    initial_field_frame: motion.MotionPipelineFrame,
    obstacle_tracker: ObstacleTracker,
    robot_from_lidar: np.ndarray,
    detection_config: ObstacleDetectionConfig,
    class_policy: ClassSafetyPolicy,
    guidance_policy: SafeSAGEGuidancePolicy,
    grid_config: LocalOccupancyGridConfig,
    field_config: SafeSAGEFieldPipelineConfig,
    mpc: SafeSAGEUnicycleMPC,
    analytical_filter: SafeSAGEUnicycleCBFFilter,
    direction_grid_y: float,
) -> LiveNavigationResult:
    """Execute live-field MPC and full-rate analytical safety filtering."""

    initial_pose = stationary.read_robot_pose(stage)
    robot_pose = initial_pose
    anchor_pose = initial_field_frame.anchor_pose_world
    path_states = np.asarray(planar_path_states, dtype=np.float64)
    clearance_threshold = motion.proxy_clearance_forward_threshold(initial_pose)

    field_frame = initial_field_frame
    latest_frame_id = field_frame.sensor_frame_id
    progress_index = 0
    warm_start_commands: np.ndarray | None = None
    executed_states = [np.zeros(3, dtype=np.float64)]

    completed_control_cycles = 0
    completed_physics_steps = 0
    live_lidar_frames = 0
    successful_field_updates = 0
    field_update_failures = 0
    stale_field_cycles = 0
    maximum_stale_field_cycles = 0
    proxy_track_ids = {field_frame.proxy_track_id}
    minimum_proxy_point_count = field_frame.proxy_point_count
    maximum_proxy_association_error = field_frame.proxy_association_error

    all_predictive_valid = True
    all_analytical_valid = True
    mpc_failure_seen = False
    analytical_failure_seen = False
    stale_field_stop_seen = False
    perception_failure_seen = False
    ground_truth_abort_seen = False
    application_stopped = False
    cleared_proxy = False
    refinement_used_count = 0
    refinement_failure_count = 0
    feasibility_recovery_seen = False
    analytical_intervention_seen = False
    settled_confirmation_count = 0

    minimum_clearance = motion.policy_footprint_clearance(initial_pose)
    minimum_predicted_safety = math.inf
    minimum_predicted_margin = math.inf
    minimum_analytical_safety = math.inf
    minimum_analytical_margin = math.inf
    maximum_rightward = 0.0
    maximum_path_error = 0.0
    field_build_times = [field_frame.build_wall_seconds]
    solve_wall_times: list[float] = []
    termination_reason = "Maximum live-navigation duration reached."

    initial_nominal = waypoint_controller.compute_command(initial_pose)
    print("\n" + "=" * 78)
    print("SAFE-SAGE STAGE E - LIVE PERCEPTION AND PHYSICAL NAVIGATION")
    print("=" * 78)
    print("\nControlled settings:")
    print("  Navigation task:                straight waypoint goal")
    print(
        "  Initial nominal command:        "
        f"({initial_nominal.linear_velocity:.3f}, "
        f"{initial_nominal.angular_velocity:.3f})"
    )
    print("  Field frame:                    fixed at the first LiDAR scan")
    print("  Execution nominal:              feedback on Safe-SAGE path")
    print("  Field refresh:                  newest frame at MPC boundary")
    print("  Static temporal derivative:     0.000000")
    print("  Physical state/input:           (x, y, yaw) / (v, omega)")
    print("  MPC strategy:                   safe bootstrap + refinement")
    print(f"  MPC horizon:                    {mpc.config.horizon_steps} steps")
    print("  Analytical CBF rate:            every physics step")
    print(
        "  Lateral tracking-tube allowance:"
        f" {fixed.PATH_LATERAL_TRACKING_TUBE_ALLOWANCE:.3f} m"
    )
    print(
        "  Terminal path overrun:           "
        f"{fixed.PATH_TERMINAL_FORWARD_OVERRUN:.3f} m"
    )
    print(
        "  Maximum stale-field cycles:      "
        f"{MAXIMUM_STALE_FIELD_CYCLES}"
    )
    print(
        "  Proxy-clear forward threshold:   "
        f"{clearance_threshold:.3f} m"
    )
    print("  Ground-truth use:               association + abort only")
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
        nominal_command = waypoint_controller.compute_command(robot_pose)
        if nominal_command.route_complete:
            termination_reason = "The nominal waypoint route completed."
            break

        field_updated = False
        try:
            new_frame = newest_lidar_frame(
                lidar,
                previous_frame_id=latest_frame_id,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            perception_failure_seen = True
            field_update_failures += 1
            termination_reason = f"Fail-closed LiDAR read stop: {error}"
            break

        if new_frame is None:
            stale_field_cycles += 1
        else:
            live_lidar_frames += 1
            points_sensor_frame, frame_info = new_frame
            try:
                candidate_field = build_live_field(
                    points_sensor_frame=points_sensor_frame,
                    frame_info=frame_info,
                    robot_pose=robot_pose,
                    anchor_pose=anchor_pose,
                    obstacle_tracker=obstacle_tracker,
                    robot_from_lidar=robot_from_lidar,
                    detection_config=detection_config,
                    class_policy=class_policy,
                    guidance_policy=guidance_policy,
                    grid_config=grid_config,
                    field_config=field_config,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                perception_failure_seen = True
                field_update_failures += 1
                termination_reason = (
                    "Fail-closed perception/PDE update stop: " f"{error}"
                )
                break
            field_frame = candidate_field
            latest_frame_id = field_frame.sensor_frame_id
            stale_field_cycles = 0
            field_updated = True
            successful_field_updates += 1
            field_build_times.append(field_frame.build_wall_seconds)
            proxy_track_ids.add(field_frame.proxy_track_id)
            minimum_proxy_point_count = min(
                minimum_proxy_point_count,
                field_frame.proxy_point_count,
            )
            maximum_proxy_association_error = max(
                maximum_proxy_association_error,
                field_frame.proxy_association_error,
            )

        maximum_stale_field_cycles = max(
            maximum_stale_field_cycles,
            stale_field_cycles,
        )
        if stale_field_cycles > MAXIMUM_STALE_FIELD_CYCLES:
            stale_field_stop_seen = True
            termination_reason = "LiDAR field became stale; fail-closed stop applied."
            break

        field_position, field_yaw = motion.current_pose_in_field_frame(
            robot_pose,
            anchor_pose=anchor_pose,
        )
        chassis_state = np.array(
            [field_position[0], field_position[1], field_yaw],
            dtype=np.float64,
        )
        progress_index = fixed.select_path_progress_index(
            path_states,
            field_position,
            previous_index=progress_index,
        )
        reference = fixed.build_feedback_tracking_reference(
            path_states,
            current_chassis_state=chassis_state,
            progress_index=progress_index,
            mpc=mpc,
        )
        initial_guess = (
            reference.seed_commands
            if warm_start_commands is None
            else warm_start_commands
        )

        solve_start = time.perf_counter()
        candidate = mpc.plan(
            chassis_state,
            reference.seed_commands,
            initial_command_guess=initial_guess,
            target_chassis_states_grid=reference.target_states,
            preferred_lateral_direction_grid_y=direction_grid_y,
            configuration_space=field_frame.pipeline.configuration_space,
            h_full=field_frame.pipeline.h_full,
            guidance_fields=field_frame.pipeline.guidance_fields,
            temporal_safety_derivative=stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE,
        )
        candidate_valid = integration.predictive_result_is_valid(
            candidate,
            config=mpc.config,
        )
        all_predictive_valid &= candidate_valid
        if not candidate_valid:
            mpc_failure_seen = True
            termination_reason = f"Fail-closed MPC stop: {candidate.status}"
            break

        feasibility_recovery_seen |= "feasibility_recovered" in candidate.status
        refinement = mpc.plan(
            chassis_state,
            reference.seed_commands,
            initial_command_guess=candidate.command_sequence,
            target_chassis_states_grid=reference.target_states,
            preferred_lateral_direction_grid_y=direction_grid_y,
            configuration_space=field_frame.pipeline.configuration_space,
            h_full=field_frame.pipeline.h_full,
            guidance_fields=field_frame.pipeline.guidance_fields,
            temporal_safety_derivative=stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE,
        )
        refinement_valid = integration.predictive_result_is_valid(
            refinement,
            config=mpc.config,
        )
        if refinement_valid:
            mpc_result = refinement
            refinement_used = True
            refinement_used_count += 1
        else:
            mpc_result = candidate
            refinement_used = False
            refinement_failure_count += 1
        solve_wall_seconds = time.perf_counter() - solve_start
        solve_wall_times.append(solve_wall_seconds)
        completed_control_cycles += 1
        minimum_predicted_safety = min(
            minimum_predicted_safety,
            mpc_result.minimum_predicted_safety_value,
        )
        minimum_predicted_margin = min(
            minimum_predicted_margin,
            mpc_result.minimum_dtcbf_margin,
        )
        warm_start_commands = fixed.shifted_warm_start(
            mpc_result.command_sequence
        )

        command_segment_failed = False
        last_analytical_result: SafeSAGEUnicycleCBFResult | None = None
        for _ in range(fixed.CONTROL_PHYSICS_STEPS):
            if not simulation_app.is_running():
                application_stopped = True
                termination_reason = "Isaac Sim stopped during command execution."
                command_segment_failed = True
                break
            while simulation_app.is_running() and not app_utils.is_playing():
                robot_controller.stop()
                simulation_app.update()
            if not simulation_app.is_running():
                application_stopped = True
                termination_reason = "Isaac Sim stopped during command execution."
                command_segment_failed = True
                break

            monitored_pose = stationary.read_robot_pose(stage)
            monitored_position, monitored_yaw = motion.current_pose_in_field_frame(
                monitored_pose,
                anchor_pose=anchor_pose,
            )
            monitored_state = np.array(
                [monitored_position[0], monitored_position[1], monitored_yaw],
                dtype=np.float64,
            )
            monitored_clearance = motion.policy_footprint_clearance(monitored_pose)
            if monitored_clearance <= fixed.GROUND_TRUTH_CLEARANCE_ABORT:
                ground_truth_abort_seen = True
                termination_reason = (
                    "Independent clearance guard triggered before command "
                    "application."
                )
                command_segment_failed = True
                break

            analytical_result = fixed.evaluate_analytical_backup(
                safety_filter=analytical_filter,
                nominal_command=mpc_result.first_command,
                chassis_state=monitored_state,
                pipeline_frame=field_frame,
            )
            last_analytical_result = analytical_result
            analytical_valid = integration.analytical_result_is_valid(
                analytical_result
            )
            all_analytical_valid &= analytical_valid
            if not analytical_valid:
                analytical_failure_seen = True
                termination_reason = (
                    "Fail-closed analytical CBF stop: "
                    f"{analytical_result.emergency_reason or 'invalid constraint'}"
                )
                command_segment_failed = True
                break

            analytical_intervention_seen |= analytical_result.intervention_active
            minimum_analytical_safety = min(
                minimum_analytical_safety,
                analytical_result.sampled_safety_value,
            )
            minimum_analytical_margin = min(
                minimum_analytical_margin,
                analytical_result.cbf_margin,
            )
            robot_controller.apply_command(
                analytical_result.safe_linear_velocity,
                analytical_result.safe_angular_velocity,
            )
            simulation_app.update()
            completed_physics_steps += 1

            monitored_pose = stationary.read_robot_pose(stage)
            robot_pose = monitored_pose
            monitored_position, monitored_yaw = motion.current_pose_in_field_frame(
                monitored_pose,
                anchor_pose=anchor_pose,
            )
            monitored_state = np.array(
                [monitored_position[0], monitored_position[1], monitored_yaw],
                dtype=np.float64,
            )
            monitored_clearance = motion.policy_footprint_clearance(monitored_pose)
            forward, rightward = motion.forward_and_rightward_displacement(
                monitored_pose,
                initial_pose=initial_pose,
            )
            path_error = fixed.nearest_path_error(path_states, monitored_position)
            minimum_clearance = min(minimum_clearance, monitored_clearance)
            maximum_rightward = max(maximum_rightward, rightward)
            maximum_path_error = max(maximum_path_error, path_error)
            if monitored_clearance <= fixed.GROUND_TRUTH_CLEARANCE_ABORT:
                ground_truth_abort_seen = True
                termination_reason = (
                    "Independent clearance guard triggered during command "
                    "execution."
                )
                command_segment_failed = True
                break
            if forward >= clearance_threshold:
                cleared_proxy = True
                termination_reason = "Nova Carter cleared the proxy longitudinally."
                command_segment_failed = True
                break

        robot_controller.stop()
        if last_analytical_result is not None:
            field_position, field_yaw = motion.current_pose_in_field_frame(
                robot_pose,
                anchor_pose=anchor_pose,
            )
            chassis_state = np.array(
                [field_position[0], field_position[1], field_yaw],
                dtype=np.float64,
            )
            _, rightward = motion.forward_and_rightward_displacement(
                robot_pose,
                initial_pose=initial_pose,
            )
            current_clearance = motion.policy_footprint_clearance(robot_pose)
            current_path_error = fixed.nearest_path_error(
                path_states,
                field_position,
            )
            executed_states.append(chassis_state.copy())
            print_live_cycle(
                cycle=cycle,
                field_frame=field_frame,
                field_updated=field_updated,
                stale_field_cycles=stale_field_cycles,
                progress_index=progress_index,
                path_state_count=path_states.shape[0],
                chassis_state=chassis_state,
                clearance=current_clearance,
                rightward=rightward,
                path_error=current_path_error,
                mpc_result=mpc_result,
                analytical_result=last_analytical_result,
                solve_wall_seconds=solve_wall_seconds,
                refinement_used=refinement_used,
            )

            settled = bool(
                last_analytical_result.safe_linear_velocity
                <= fixed.SETTLED_LINEAR_SPEED
                and abs(last_analytical_result.safe_angular_velocity)
                <= fixed.SETTLED_ANGULAR_SPEED
            )
            if settled:
                settled_confirmation_count += 1
            else:
                settled_confirmation_count = 0
            if (
                settled_confirmation_count >= fixed.SETTLED_CONFIRMATION_CYCLES
                and not cleared_proxy
            ):
                termination_reason = "The safe command settled before proxy clearance."
                command_segment_failed = True

        if command_segment_failed:
            break

    if simulation_app.is_running():
        robot_controller.stop()
        app_utils.update_app(steps=5)
        final_pose = stationary.read_robot_pose(stage)
    else:
        final_pose = robot_pose

    final_position, final_yaw = motion.current_pose_in_field_frame(
        final_pose,
        anchor_pose=anchor_pose,
    )
    final_state = np.array(
        [final_position[0], final_position[1], final_yaw],
        dtype=np.float64,
    )
    if not np.allclose(executed_states[-1], final_state, atol=1e-9):
        executed_states.append(final_state)
    final_clearance = motion.policy_footprint_clearance(final_pose)
    minimum_clearance = min(minimum_clearance, final_clearance)
    _, final_rightward = motion.forward_and_rightward_displacement(
        final_pose,
        initial_pose=initial_pose,
    )
    maximum_rightward = max(maximum_rightward, final_rightward)

    return LiveNavigationResult(
        termination_reason=termination_reason,
        completed_control_cycles=completed_control_cycles,
        completed_physics_steps=completed_physics_steps,
        executed_chassis_states=np.ascontiguousarray(
            np.asarray(executed_states, dtype=np.float64)
        ),
        final_path_progress_index=progress_index,
        live_lidar_frames=live_lidar_frames,
        successful_field_updates=successful_field_updates,
        field_update_failures=field_update_failures,
        maximum_stale_field_cycles=maximum_stale_field_cycles,
        initial_sensor_frame_id=initial_field_frame.sensor_frame_id,
        final_sensor_frame_id=latest_frame_id,
        distinct_proxy_track_ids=tuple(sorted(proxy_track_ids)),
        minimum_proxy_point_count=minimum_proxy_point_count,
        maximum_proxy_association_error=maximum_proxy_association_error,
        all_predictive_constraints_valid=all_predictive_valid,
        all_analytical_constraints_valid=all_analytical_valid,
        mpc_failure_seen=mpc_failure_seen,
        analytical_failure_seen=analytical_failure_seen,
        stale_field_stop_seen=stale_field_stop_seen,
        perception_failure_seen=perception_failure_seen,
        ground_truth_abort_seen=ground_truth_abort_seen,
        application_stopped=application_stopped,
        cleared_proxy=cleared_proxy,
        refinement_used_count=refinement_used_count,
        refinement_failure_count=refinement_failure_count,
        feasibility_recovery_seen=feasibility_recovery_seen,
        analytical_intervention_seen=analytical_intervention_seen,
        minimum_policy_clearance=minimum_clearance,
        minimum_predicted_safety=minimum_predicted_safety,
        minimum_predicted_margin=minimum_predicted_margin,
        minimum_analytical_safety=minimum_analytical_safety,
        minimum_analytical_margin=minimum_analytical_margin,
        maximum_rightward_displacement=maximum_rightward,
        maximum_path_error=maximum_path_error,
        mean_field_build_wall_seconds=float(np.mean(field_build_times)),
        maximum_field_build_wall_seconds=float(np.max(field_build_times)),
        mean_mpc_solve_wall_seconds=(
            math.nan
            if not solve_wall_times
            else float(np.mean(solve_wall_times))
        ),
        maximum_mpc_solve_wall_seconds=(
            math.nan
            if not solve_wall_times
            else float(np.max(solve_wall_times))
        ),
    )


def print_result(result: LiveNavigationResult) -> bool:
    """Print the live-navigation acceptance report."""

    final_state = result.executed_chassis_states[-1]
    sufficient_live_updates = (
        result.successful_field_updates >= MINIMUM_REQUIRED_LIVE_FIELD_UPDATES
    )
    rightward_response = (
        result.maximum_rightward_displacement
        >= fixed.MINIMUM_EXPECTED_RIGHTWARD_DISPLACEMENT
    )
    accepted = bool(
        result.cleared_proxy
        and sufficient_live_updates
        and rightward_response
        and result.all_predictive_constraints_valid
        and result.all_analytical_constraints_valid
        and result.field_update_failures == 0
        and not result.mpc_failure_seen
        and not result.analytical_failure_seen
        and not result.stale_field_stop_seen
        and not result.perception_failure_seen
        and not result.ground_truth_abort_seen
        and not result.application_stopped
    )

    print("\n" + "=" * 78)
    print("SAFE-SAGE STAGE E RESULT - LIVE PHYSICAL BYPASS")
    print("=" * 78)
    print(f"Termination:                       {result.termination_reason}")
    print(
        "Completed control cycles:          "
        f"{result.completed_control_cycles}"
    )
    print(
        "Completed physics steps:           "
        f"{result.completed_physics_steps}"
    )
    print(
        "Final path progress index:         "
        f"{result.final_path_progress_index}"
    )

    print("\nLive perception and field updates:")
    print(f"  Initial/final frame ID:          {result.initial_sensor_frame_id} / {result.final_sensor_frame_id}")
    print(f"  New LiDAR frames:                {result.live_lidar_frames}")
    print(f"  Successful field updates:        {result.successful_field_updates}")
    print(f"  Field update failures:           {result.field_update_failures}")
    print(f"  Maximum stale-field cycles:      {result.maximum_stale_field_cycles}")
    print(f"  Proxy track IDs:                 {result.distinct_proxy_track_ids}")
    print(f"  Minimum proxy points:            {result.minimum_proxy_point_count}")
    print(f"  Maximum association error:       {result.maximum_proxy_association_error:.4f} m")
    print(f"  Mean field build time:           {result.mean_field_build_wall_seconds:.3f} s")
    print(f"  Maximum field build time:        {result.maximum_field_build_wall_seconds:.3f} s")

    print("\nPhysical motion:")
    print(
        "  Final field-frame state:         "
        f"({final_state[0]:.4f}, {final_state[1]:.4f}, "
        f"{math.degrees(final_state[2]):.2f} deg)"
    )
    print(f"  Maximum rightward displacement:  {result.maximum_rightward_displacement:.4f} m")
    print(f"  Maximum sampled path error:      {result.maximum_path_error:.4f} m")
    print(f"  Minimum policy clearance:        {result.minimum_policy_clearance:.4f} m")

    print("\nPredictive and analytical safety:")
    print(f"  All predictive constraints:      {result.all_predictive_constraints_valid}")
    print(f"  All analytical constraints:      {result.all_analytical_constraints_valid}")
    print(f"  Minimum predicted h_full:        {result.minimum_predicted_safety:.6f}")
    print(f"  Minimum predicted DTCBF margin:  {result.minimum_predicted_margin:.3e}")
    print(f"  Minimum analytical h_full:       {result.minimum_analytical_safety:.6f}")
    print(f"  Minimum analytical CBF margin:   {result.minimum_analytical_margin:.3e}")
    print(f"  Analytical intervention seen:    {result.analytical_intervention_seen}")

    print("\nSolver diagnostics:")
    print(f"  Safe refinements used:           {result.refinement_used_count}")
    print(f"  Refinement failures:             {result.refinement_failure_count}")
    print(f"  Feasibility recovery observed:   {result.feasibility_recovery_seen}")
    print(f"  Mean two-pass solve time:        {result.mean_mpc_solve_wall_seconds:.3f} s")
    print(f"  Maximum two-pass solve time:     {result.maximum_mpc_solve_wall_seconds:.3f} s")

    print("\nControlled checks:")
    print(f"  Proxy cleared:                   {result.cleared_proxy}")
    print(f"  Enough live field updates:       {sufficient_live_updates}")
    print(f"  Rightward response observed:     {rightward_response}")
    print(f"  MPC failure seen:                {result.mpc_failure_seen}")
    print(f"  Analytical failure seen:         {result.analytical_failure_seen}")
    print(f"  Perception failure seen:         {result.perception_failure_seen}")
    print(f"  Stale-field stop seen:           {result.stale_field_stop_seen}")
    print(f"  Clearance abort seen:            {result.ground_truth_abort_seen}")
    print(f"  Application stopped:             {result.application_stopped}")

    print("\nInterpretation:")
    if accepted:
        print(
            "  [PASS] Nova Carter completed the semantic right-side bypass "
            "while rebuilding the real LiDAR-derived Safe-SAGE field online "
            "and retaining valid predictive and full-rate analytical safety "
            "constraints."
        )
    else:
        print(
            "  [WARNING] The live physical trial did not pass every "
            "criterion. The controller stopped fail-closed; inspect the "
            "termination and field-update diagnostics before proceeding."
        )
    print("=" * 78)
    return accepted


def main() -> None:
    """Run the complete live-perception physical bypass."""

    print("\n" + "=" * 78)
    print("SAFE-SAGE LIVE PERCEPTION AND PHYSICAL NAVIGATION DEMO")
    print("=" * 78)
    print("Mode: live LiDAR fields + unicycle MPC + analytical CBF + wheels")
    print(f"Guidance policy: {oracle.PAPER_BASELINE_GUIDANCE_POLICY_PATH}")

    class_policy, guidance_policy, grid_config, planar_field_config = (
        oracle.create_oracle_configuration()
    )
    semantic_parameters = guidance_policy.parameters_for(
        oracle.PAPER_BASELINE_SEMANTIC_CLASS
    )
    direction_grid_y = integration.passing_direction_grid_y(
        semantic_parameters.passing_side
    )

    loaded_scene = load_robot_scene(
        scene_path=stationary.ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=stationary.ROBOT_PRIM_PATH,
        expected_meters_per_unit=stationary.EXPECTED_METERS_PER_UNIT,
    )
    oracle.configure_oracle_proxy_position()
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
    detection_config = integration.create_detection_config(floor_height_robot)

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
    print("\nWaiting for the initial shared LiDAR snapshot...")
    points_sensor_frame, frame_info, _ = stationary.wait_for_first_scan(
        lidar,
        robot_controller,
    )

    print("Building the initial chassis-origin planar Safe-SAGE field...")
    planar_frame = motion.build_motion_pipeline_frame(
        points_sensor_frame=points_sensor_frame,
        frame_info=frame_info,
        robot_pose=anchor_pose,
        field_anchor_pose=anchor_pose,
        obstacle_tracker=integration.create_obstacle_tracker(),
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=grid_config,
        field_config=planar_field_config,
        semantic_class=oracle.PAPER_BASELINE_SEMANTIC_CLASS,
    )
    planar_mpc, paper_filter = oracle.create_controller_stack(robot_controller)
    planar_result = oracle.run_planar_oracle(
        pipeline_frame=planar_frame,
        planar_mpc=planar_mpc,
        paper_filter=paper_filter,
    )
    planar_valid = bool(
        planar_result.proxy_cleared
        and planar_result.rightward_response_seen
        and planar_result.all_predictive_constraints_valid
        and planar_result.all_backup_constraints_valid
        and not planar_result.mpc_fallback_seen
        and not planar_result.backup_failure_seen
    )
    if not planar_valid:
        raise RuntimeError(
            "The planar oracle did not provide an accepted live-navigation "
            f"reference: {planar_result.termination_reason}"
        )

    tracking_path_states = fixed.add_lateral_tracking_tube_allowance(
        planar_result.virtual_states,
        preferred_lateral_direction_grid_y=direction_grid_y,
        maximum_allowance=fixed.PATH_LATERAL_TRACKING_TUBE_ALLOWANCE,
    )
    tracking_path_states = fixed.append_terminal_path_overrun(
        tracking_path_states,
        overrun_distance=fixed.PATH_TERMINAL_FORWARD_OVERRUN,
    )

    measured_footprint = RectangularRobotFootprint(
        front_extent=stationary.ROBOT_FRONT_EXTENT,
        rear_extent=stationary.ROBOT_REAR_EXTENT,
        left_extent=stationary.ROBOT_LEFT_EXTENT,
        right_extent=stationary.ROBOT_RIGHT_EXTENT,
        safety_margin=stationary.ROBOT_SAFETY_MARGIN,
    )
    lookahead_footprint = shift_rectangular_footprint_to_lookahead(
        measured_footprint,
        lookahead_distance=stationary.UNICYCLE_LOOKAHEAD_DISTANCE,
    )
    live_field_config = stationary.create_field_pipeline_config(
        lookahead_footprint
    )
    live_obstacle_tracker = integration.create_obstacle_tracker()
    print("Building the first live look-ahead Safe-SAGE field...")
    initial_field_frame = build_live_field(
        points_sensor_frame=points_sensor_frame,
        frame_info=frame_info,
        robot_pose=anchor_pose,
        anchor_pose=anchor_pose,
        obstacle_tracker=live_obstacle_tracker,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=grid_config,
        field_config=live_field_config,
    )

    mpc = fixed.create_motion_mpc(
        robot_controller,
        paper_filter=paper_filter,
    )
    analytical_filter = integration.create_analytical_filter(
        robot_controller,
        paper_filter=paper_filter,
    )
    waypoint_controller = stationary.create_waypoint_controller()
    initial_waypoint_command = waypoint_controller.compute_command(anchor_pose)
    waypoint_nominal_vector = np.array(
        [
            initial_waypoint_command.linear_velocity,
            initial_waypoint_command.angular_velocity,
        ],
        dtype=np.float64,
    )
    linear_nominal_matches = math.isclose(
        waypoint_nominal_vector[0],
        float(oracle.NOMINAL_PLANAR_COMMAND[0]),
        abs_tol=1e-6,
    )
    startup_heading_correction_is_small = (
        abs(
            waypoint_nominal_vector[1]
            - float(oracle.NOMINAL_PLANAR_COMMAND[1])
        )
        <= 0.01
    )
    if not (
        linear_nominal_matches and startup_heading_correction_is_small
    ):
        raise RuntimeError(
            "The waypoint controller's initial nominal command is not "
            "equivalent to the planar Safe-SAGE oracle input within the "
            "measured startup-pose tolerance."
        )
    print(
        "Validated waypoint-to-planar nominal command: "
        f"({waypoint_nominal_vector[0]:.3f}, "
        f"{waypoint_nominal_vector[1]:.3f})"
    )

    oracle.create_path_markers(
        loaded_scene.stage,
        tracking_path_states,
        anchor_pose=anchor_pose,
        root_path=REFERENCE_MARKER_ROOT,
        color=REFERENCE_MARKER_COLOR,
        highlight_final=False,
    )
    result = run_live_navigation(
        stage=loaded_scene.stage,
        lidar=lidar,
        robot_controller=robot_controller,
        waypoint_controller=waypoint_controller,
        planar_path_states=tracking_path_states,
        initial_field_frame=initial_field_frame,
        obstacle_tracker=live_obstacle_tracker,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=grid_config,
        field_config=live_field_config,
        mpc=mpc,
        analytical_filter=analytical_filter,
        direction_grid_y=direction_grid_y,
    )
    oracle.create_path_markers(
        loaded_scene.stage,
        result.executed_chassis_states,
        anchor_pose=anchor_pose,
        root_path=EXECUTED_MARKER_ROOT,
        color=EXECUTED_MARKER_COLOR,
        highlight_final=True,
    )
    print_result(result)

    robot_controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()

    print("\nExpected successful scene behavior:")
    print("  - Cyan markers show the buffered Safe-SAGE reference path.")
    print("  - Nova Carter performs the same robot-right semantic bypass.")
    print("  - The blue proxy remains fixed and is not touched.")
    print("  - Green markers show the physically executed path.")
    print("  - The terminal reports a new LiDAR/field update almost every cycle.")
    print("  - The robot stops only after clearing the proxy longitudinally.")

    if stationary.SIMULATION_HEADLESS:
        print("\nHeadless run complete; closing Isaac Sim.")
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
        print("ERROR DURING SAFE-SAGE LIVE NAVIGATION DEMO")
        print("=" * 78)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
