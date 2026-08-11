"""Run fixed-field closed-loop Safe-SAGE unicycle MPC motion.

This is the first physical-motion stage after the stationary MPC validation.
It deliberately freezes perception and the Safe-SAGE field after one valid
PandarXT-32 scan so that only the predictive controller, analytical backup,
and Nova Carter wheel interface are under test.

Stage A - controlled fixed-field construction:

1. Position Nova Carter in the spacious restaurant test area.
2. Create and observe the session-only blue semantic proxy.
3. Associate its real LiDAR track and manually label it as ``person``.
4. Build one yaw-dependent configuration space, guidance field, and
   ``h_full(x, y, psi)`` in a frame fixed at the initial robot pose.

Stage B - receding-horizon wheel motion:

1. Roll out the obstacle-unaware waypoint feedback policy over the MPC horizon.
2. Keep that nominal policy continuous during safety-induced heading changes.
3. Shift the preceding safe horizon and append a stationary terminal command.
4. Revalidate and solve the Safe-SAGE unicycle MPC in the fixed field.
5. Pass the first MPC command through the analytical unicycle CBF backup.
6. Execute only that backed-up ``(v, omega)`` command for one MPC interval.
7. Repeat until the robot clears the proxy or a fail-closed condition occurs.

The known proxy pose is used only by an independent clearance monitor and
the final validation report. It is never supplied to perception, MPC, or the
analytical safety filter. Runtime is reported but is not an acceptance
criterion in this risk-assessment stage.

Run from the project root:

    python -u scripts/demos/safe_sage_unicycle_mpc_motion_demo.py

Press Play in Isaac Sim when requested.
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


# Importing the stationary MPC demo initializes the one required
# SimulationApp. Its guarded main() function is not executed.
from scripts.demos import safe_sage_unicycle_mpc_demo as prediction


motion = prediction.motion
stationary = motion.stationary
simulation_app = prediction.simulation_app


import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd

from robot.control.differential_drive import NovaCarterController
from robot.control.unicycle_model import (
    integrate_unicycle_euler,
    lookahead_state,
)
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
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.obstacle_detection import ObstacleDetectionConfig
from robot.safety.obstacle_tracking import (
    ObstacleTracker,
    ObstacleTrackerConfig,
)
from robot.safety.safe_sage_cbf_filter import (
    SafeSAGECBFConfig,
    SafeSAGECBFSafetyFilter,
)
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
)
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig
from robot.safety.safe_sage_guidance_policy import (
    PassingSide,
    SafeSAGEGuidancePolicy,
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
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import compute_relative_prim_transform
from robot.sensor.lidar import NovaCarterLidar


# ---------------------------------------------------------------------------
# Fixed-field motion configuration
# ---------------------------------------------------------------------------


MPC_CONTROL_INTERVAL_SECONDS = prediction.MPC_TIME_STEP
CONTROL_PHYSICS_STEPS = max(
    1,
    round(MPC_CONTROL_INTERVAL_SECONDS / stationary.PHYSICS_DT),
)
MAXIMUM_MOTION_SECONDS = 20.0
MAXIMUM_CONTROL_CYCLES = math.ceil(
    MAXIMUM_MOTION_SECONDS / MPC_CONTROL_INTERVAL_SECONDS
)

GROUND_TRUTH_CLEARANCE_ABORT = motion.GROUND_TRUTH_CLEARANCE_ABORT
MINIMUM_EXPECTED_RIGHTWARD_DISPLACEMENT = motion.MINIMUM_EXPECTED_RIGHTWARD_DISPLACEMENT
SETTLED_LINEAR_SPEED = motion.SETTLED_LINEAR_SPEED
SETTLED_ANGULAR_SPEED = motion.SETTLED_ANGULAR_SPEED
SETTLED_CONFIRMATION_CYCLES = 5
DIRECTIONAL_AUDIT_FORWARD_SAMPLES = (0.0, 0.15, 0.30, 0.45)
DIRECTIONAL_AUDIT_LINEAR_VELOCITY = 0.05
DIRECTIONAL_AUDIT_ANGULAR_VELOCITY = 0.30
DIRECTIONAL_COEFFICIENT_TOLERANCE = 1e-8


@dataclass(frozen=True)
class FixedFieldMotionResult:
    """Closed-loop outcome and independent safety diagnostics."""

    termination_reason: str
    completed_control_cycles: int
    completed_physics_steps: int
    initial_pose: RobotPose2D
    final_pose: RobotPose2D
    final_mpc_result: SafeSAGEUnicycleMPCResult | None
    final_backup_result: SafeSAGEUnicycleCBFResult | None
    mpc_intervention_seen: bool
    backup_intervention_seen: bool
    all_predictive_constraints_valid: bool
    all_backup_constraints_valid: bool
    cleared_proxy: bool
    route_complete: bool
    mpc_failure_seen: bool
    backup_failure_seen: bool
    ground_truth_abort_seen: bool
    application_stopped: bool
    minimum_policy_footprint_clearance: float
    minimum_predicted_safety: float
    minimum_predicted_dtcbf_margin: float
    minimum_backup_safety: float
    minimum_backup_cbf_margin: float
    maximum_rightward_displacement: float
    mean_mpc_solve_wall_seconds: float
    maximum_mpc_solve_wall_seconds: float


def predictive_result_is_valid(
    result: SafeSAGEUnicycleMPCResult,
    *,
    constraint_tolerance: float,
) -> bool:
    """Return whether one MPC result is finite and exactly DTCBF-valid."""

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
        and all(np.isfinite(array).all() for array in arrays)
        and float(np.min(result.predicted_safety_values)) >= -constraint_tolerance
        and float(np.min(result.dtcbf_margins)) >= -constraint_tolerance
    )


def backup_result_is_valid(result: SafeSAGEUnicycleCBFResult) -> bool:
    """Return whether the analytical backup accepted one safe command."""

    return bool(
        result.constraint_feasible
        and result.constraint_satisfied
        and not result.emergency_stop_active
        and math.isfinite(result.safe_linear_velocity)
        and math.isfinite(result.safe_angular_velocity)
        and math.isfinite(result.sampled_safety_value)
        and math.isfinite(result.cbf_margin)
    )


def print_live_status(
    *,
    control_cycle: int,
    robot_pose: RobotPose2D,
    field_position: np.ndarray,
    field_yaw: float,
    clearance: float,
    rightward_displacement: float,
    nominal_command: WaypointCommand,
    mpc_result: SafeSAGEUnicycleMPCResult,
    backup_result: SafeSAGEUnicycleCBFResult,
    solve_wall_seconds: float,
    warm_start_used: bool,
) -> None:
    """Print one compact fixed-field MPC status line."""

    angular_coefficient = backup_result.angular_velocity_coefficient
    if angular_coefficient < -DIRECTIONAL_COEFFICIENT_TOLERANCE:
        field_turn = "R"
    elif angular_coefficient > DIRECTIONAL_COEFFICIENT_TOLERANCE:
        field_turn = "L"
    else:
        field_turn = "N"
    predicted_y = mpc_result.predicted_chassis_states[:, 1]

    print(
        f"  cycle={control_cycle:<3} "
        f"pose=({robot_pose.x:.3f},{robot_pose.y:.3f},"
        f"{math.degrees(robot_pose.yaw):.1f}deg) "
        f"q=({field_position[0]:.3f},{field_position[1]:.3f},"
        f"{math.degrees(field_yaw):.1f}deg) "
        f"clear={clearance:.3f} "
        f"right={rightward_displacement:.3f} "
        f"nom=({nominal_command.linear_velocity:.3f},"
        f"{nominal_command.angular_velocity:.3f}) "
        f"mpc=({mpc_result.linear_velocity:.3f},"
        f"{mpc_result.angular_velocity:.3f}) "
        f"out=({backup_result.safe_linear_velocity:.3f},"
        f"{backup_result.safe_angular_velocity:.3f}) "
        f"hmin={float(np.min(mpc_result.predicted_safety_values)):.3f} "
        f"margin={float(np.min(mpc_result.dtcbf_margins)):.2e} "
        f"field={field_turn}[{angular_coefficient:.2e}] "
        f"y[min,max,end]=({float(np.min(predicted_y)):.3f},"
        f"{float(np.max(predicted_y)):.3f},{float(predicted_y[-1]):.3f}) "
        f"solve={solve_wall_seconds:.3f}s "
        f"warm={warm_start_used} "
        f"status={mpc_result.status}"
    )


def print_directional_field_audit(
    *,
    pipeline_frame: motion.MotionPipelineFrame,
    mpc: SafeSAGEUnicycleMPC,
    backup_filter: SafeSAGEUnicycleCBFFilter,
    passing_side: PassingSide,
) -> None:
    """Print field-side diagnostics without issuing a wheel command."""

    configuration_space = pipeline_frame.pipeline.configuration_space
    h_full = pipeline_frame.pipeline.h_full
    guidance_fields = pipeline_frame.pipeline.guidance_fields
    lookahead_distance = backup_filter.config.lookahead_distance

    def evaluate_chassis_state(
        chassis_state: np.ndarray,
    ) -> SafeSAGEUnicycleCBFResult:
        query_state = lookahead_state(
            chassis_state,
            lookahead_distance=lookahead_distance,
        )
        return backup_filter.filter_command(
            0.0,
            0.0,
            query_position_grid_xy=query_state[:2],
            query_yaw_radians=float(query_state[2]),
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
            temporal_safety_derivative=(
                stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE
            ),
        )

    def preference(coefficient: float) -> str:
        if coefficient < -DIRECTIONAL_COEFFICIENT_TOLERANCE:
            return "RIGHT"
        if coefficient > DIRECTIONAL_COEFFICIENT_TOLERANCE:
            return "LEFT"
        return "NEUTRAL"

    print("\n" + "=" * 78)
    print("SAFE-SAGE FIXED-FIELD DIRECTIONAL AUDIT - STATIONARY")
    print("=" * 78)
    expected_sign = "negative" if passing_side is PassingSide.RIGHT else "positive"
    print(
        f"Expected passing side: {passing_side.value.upper()} "
        f"({expected_sign} field-frame Y at the initial heading)"
    )
    print("No command in this audit reaches the wheels.")
    print("\nCenterline field samples:")
    print("  chassis_x      h       guidance_xy          a_v       a_omega  preference")
    for forward_position in DIRECTIONAL_AUDIT_FORWARD_SAMPLES:
        state = np.array([forward_position, 0.0, 0.0], dtype=np.float64)
        result = evaluate_chassis_state(state)
        guidance = result.sampled_guidance_grid_xy
        print(
            f"  {forward_position:8.3f}  "
            f"{result.sampled_safety_value:7.4f}  "
            f"({guidance[0]:8.4f},{guidance[1]:8.4f})  "
            f"{result.linear_velocity_coefficient:9.4f}  "
            f"{result.angular_velocity_coefficient:9.4f}  "
            f"{preference(result.angular_velocity_coefficient):>10}"
            + (
                "  [UNSAFE CELL]"
                if result.query_on_unsafe_boundary
                else ""
            )
        )

    initial_state = np.zeros(3, dtype=np.float64)
    initial_result = evaluate_chassis_state(initial_state)
    print("\nOne-step exact DTCBF probes from the initial state:")
    print("  probe       command(v,omega)      next_h      margin    unsafe")
    for label, angular_velocity in (
        ("RIGHT", -DIRECTIONAL_AUDIT_ANGULAR_VELOCITY),
        ("STRAIGHT", 0.0),
        ("LEFT", DIRECTIONAL_AUDIT_ANGULAR_VELOCITY),
    ):
        command = np.array(
            [DIRECTIONAL_AUDIT_LINEAR_VELOCITY, angular_velocity],
            dtype=np.float64,
        )
        next_state = integrate_unicycle_euler(
            initial_state,
            command,
            time_step=mpc.config.time_step,
        )
        next_result = evaluate_chassis_state(next_state)
        margin = (
            next_result.sampled_safety_value
            - mpc.config.dtcbf_decay * initial_result.sampled_safety_value
        )
        print(
            f"  {label:<8}  "
            f"({command[0]:6.3f},{command[1]:7.3f})  "
            f"{next_result.sampled_safety_value:10.5f}  "
            f"{margin:9.3e}  "
            f"{str(next_result.constraint.unsafe_reason):>8}"
        )
    print("=" * 78)


def create_motion_waypoint_controller() -> WaypointController:
    """Create a continuous obstacle-unaware go-to-goal policy.

    The generic waypoint controller's rotate-only mode is useful for ordinary
    navigation, but it can remove all nominal forward progress after a safety
    intervention. This motion stage instead uses the same goal and gains while
    retaining the continuous cosine-aligned command. It still has no obstacle,
    semantic, field, or passing-side input.
    """

    return WaypointController(
        [Waypoint(*stationary.ROBOT_GOAL_WORLD[:2])],
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


def run_fixed_field_motion_trial(
    *,
    stage: Usd.Stage,
    robot_controller: NovaCarterController,
    pipeline_frame: motion.MotionPipelineFrame,
    mpc: SafeSAGEUnicycleMPC,
    backup_filter: SafeSAGEUnicycleCBFFilter,
    passing_side: PassingSide,
) -> FixedFieldMotionResult:
    """Execute receding-horizon MPC against one frozen Safe-SAGE field."""

    initial_pose = stationary.read_robot_pose(stage)
    robot_pose = initial_pose
    waypoint_controller = create_motion_waypoint_controller()
    clearance_threshold = motion.proxy_clearance_forward_threshold(initial_pose)

    completed_control_cycles = 0
    completed_physics_steps = 0
    final_mpc_result: SafeSAGEUnicycleMPCResult | None = None
    final_backup_result: SafeSAGEUnicycleCBFResult | None = None
    mpc_intervention_seen = False
    backup_intervention_seen = False
    all_predictive_constraints_valid = True
    all_backup_constraints_valid = True
    cleared_proxy = False
    route_complete = False
    mpc_failure_seen = False
    backup_failure_seen = False
    ground_truth_abort_seen = False
    application_stopped = False
    settled_confirmation_count = 0

    initial_clearance = motion.policy_footprint_clearance(initial_pose)
    minimum_clearance = initial_clearance
    minimum_predicted_safety = math.inf
    minimum_predicted_margin = math.inf
    minimum_backup_safety = math.inf
    minimum_backup_margin = math.inf
    maximum_rightward_displacement = 0.0
    solve_wall_times: list[float] = []
    warm_start_commands: np.ndarray | None = None
    termination_reason = "Maximum motion duration reached."

    print("\n" + "=" * 78)
    print("SAFE-SAGE UNICYCLE MPC STAGE C - FIXED-FIELD CLOSED-LOOP MOTION")
    print("=" * 78)
    print("\nControlled motion settings:")
    print("  Perception/field updates:       one initial LiDAR snapshot")
    print("  Field frame:                    fixed at the initial robot pose")
    print("  Chassis state:                  (x, y, psi)")
    print("  Executable command:             (v, omega)")
    print("  Nominal MPC horizon:            waypoint feedback rollout")
    print(f"  Persistent passing side:        {passing_side.value}")
    print(f"  MPC control interval:           {MPC_CONTROL_INTERVAL_SECONDS:.3f} s")
    print(f"  Physics steps per command:      {CONTROL_PHYSICS_STEPS}")
    print(f"  Maximum simulated duration:     {MAXIMUM_MOTION_SECONDS:.1f} s")
    print(f"  Ground-truth clearance abort:   {GROUND_TRUTH_CLEARANCE_ABORT:.3f} m")
    print(f"  Forward clearance threshold:    {clearance_threshold:.3f} m")
    print("  Receding-horizon warm start:    shifted plan + terminal stop")
    print("  MPC runtime acceptance limit:   none in this stage")
    print("\nLive motion status:")

    for control_cycle in range(MAXIMUM_CONTROL_CYCLES):
        if not simulation_app.is_running():
            application_stopped = True
            termination_reason = "Isaac Sim stopped during the motion trial."
            break
        while simulation_app.is_running() and not app_utils.is_playing():
            robot_controller.stop()
            simulation_app.update()
        if not simulation_app.is_running():
            application_stopped = True
            termination_reason = "Isaac Sim stopped during the motion trial."
            break

        robot_pose = stationary.read_robot_pose(stage)
        nominal_command = waypoint_controller.compute_command(robot_pose)
        predicted_nominal_commands = waypoint_controller.predict_commands(
            robot_pose,
            horizon_steps=mpc.config.horizon_steps,
            time_step=mpc.config.time_step,
        )
        nominal_command_sequence = np.array(
            [
                [command.linear_velocity, command.angular_velocity]
                for command in predicted_nominal_commands
            ],
            dtype=np.float64,
        )
        if nominal_command.route_complete:
            route_complete = True
            termination_reason = "The nominal waypoint route completed."
            break

        field_position, field_yaw = motion.current_pose_in_field_frame(
            robot_pose,
            anchor_pose=pipeline_frame.anchor_pose_world,
        )
        chassis_state = np.array(
            [field_position[0], field_position[1], field_yaw],
            dtype=np.float64,
        )
        warm_start_used = warm_start_commands is not None
        solve_start = time.perf_counter()
        mpc_result = mpc.plan(
            chassis_state,
            nominal_command_sequence,
            initial_command_guess=warm_start_commands,
            configuration_space=(pipeline_frame.pipeline.configuration_space),
            h_full=pipeline_frame.pipeline.h_full,
            guidance_fields=pipeline_frame.pipeline.guidance_fields,
            temporal_safety_derivative=(stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE),
            preferred_lateral_direction_grid_y=(
                -1.0 if passing_side is PassingSide.RIGHT else 1.0
            ),
        )
        solve_wall_seconds = time.perf_counter() - solve_start
        solve_wall_times.append(solve_wall_seconds)
        final_mpc_result = mpc_result
        completed_control_cycles = control_cycle + 1

        predictive_valid = predictive_result_is_valid(
            mpc_result,
            constraint_tolerance=mpc.config.constraint_tolerance,
        )
        all_predictive_constraints_valid = (
            all_predictive_constraints_valid and predictive_valid
        )
        if not predictive_valid:
            mpc_failure_seen = True
            termination_reason = f"Fail-closed MPC stop: {mpc_result.status}"
            break

        # Receding-horizon MPC should continue from the preceding safe plan,
        # revalidated at the newly measured chassis state.  Shift away the
        # command that was just selected and append a stationary terminal
        # command. The obstacle-unaware waypoint feedback rollout remains the
        # MPC objective; this shifted safe sequence is only its numerical seed.
        warm_start_commands = np.ascontiguousarray(
            np.vstack(
                (
                    mpc_result.command_sequence[1:],
                    np.zeros((1, 2), dtype=np.float64),
                )
            ),
            dtype=np.float64,
        )

        mpc_intervention_seen = mpc_intervention_seen or mpc_result.intervention_active
        minimum_predicted_safety = min(
            minimum_predicted_safety,
            float(np.min(mpc_result.predicted_safety_values)),
        )
        minimum_predicted_margin = min(
            minimum_predicted_margin,
            float(np.min(mpc_result.dtcbf_margins)),
        )

        backup_result = backup_filter.filter_command(
            mpc_result.linear_velocity,
            mpc_result.angular_velocity,
            query_position_grid_xy=(mpc_result.predicted_safety_states[0, :2]),
            query_yaw_radians=float(mpc_result.predicted_safety_states[0, 2]),
            configuration_space=(pipeline_frame.pipeline.configuration_space),
            h_full=pipeline_frame.pipeline.h_full,
            guidance_fields=pipeline_frame.pipeline.guidance_fields,
            temporal_safety_derivative=(stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE),
        )
        final_backup_result = backup_result
        backup_valid = backup_result_is_valid(backup_result)
        all_backup_constraints_valid = all_backup_constraints_valid and backup_valid
        if not backup_valid:
            backup_failure_seen = True
            termination_reason = (
                "Fail-closed analytical backup stop: "
                f"{backup_result.emergency_reason or 'invalid constraint'}"
            )
            break

        backup_intervention_seen = (
            backup_intervention_seen or backup_result.intervention_active
        )
        minimum_backup_safety = min(
            minimum_backup_safety,
            backup_result.sampled_safety_value,
        )
        minimum_backup_margin = min(
            minimum_backup_margin,
            backup_result.cbf_margin,
        )

        current_clearance = motion.policy_footprint_clearance(robot_pose)
        forward_displacement, rightward_displacement = (
            motion.forward_and_rightward_displacement(
                robot_pose,
                initial_pose=initial_pose,
            )
        )
        minimum_clearance = min(minimum_clearance, current_clearance)
        maximum_rightward_displacement = max(
            maximum_rightward_displacement,
            rightward_displacement,
        )
        print_live_status(
            control_cycle=control_cycle,
            robot_pose=robot_pose,
            field_position=field_position,
            field_yaw=field_yaw,
            clearance=current_clearance,
            rightward_displacement=rightward_displacement,
            nominal_command=nominal_command,
            mpc_result=mpc_result,
            backup_result=backup_result,
            solve_wall_seconds=solve_wall_seconds,
            warm_start_used=warm_start_used,
        )

        if current_clearance <= GROUND_TRUTH_CLEARANCE_ABORT:
            ground_truth_abort_seen = True
            termination_reason = "Independent ground-truth clearance guard triggered."
            break
        if forward_displacement >= clearance_threshold:
            cleared_proxy = True
            termination_reason = "The robot cleared the proxy longitudinally."
            break

        settled_command = (
            backup_result.safe_linear_velocity <= SETTLED_LINEAR_SPEED
            and abs(backup_result.safe_angular_velocity) <= SETTLED_ANGULAR_SPEED
        )
        if settled_command:
            settled_confirmation_count += 1
        else:
            settled_confirmation_count = 0
        if settled_confirmation_count >= SETTLED_CONFIRMATION_CYCLES:
            termination_reason = "The safe command settled before proxy clearance."
            break

        command_segment_interrupted = False
        for _ in range(CONTROL_PHYSICS_STEPS):
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

            robot_controller.apply_command(
                backup_result.safe_linear_velocity,
                backup_result.safe_angular_velocity,
            )
            simulation_app.update()
            completed_physics_steps += 1

            monitored_pose = stationary.read_robot_pose(stage)
            robot_pose = monitored_pose
            monitored_clearance = motion.policy_footprint_clearance(monitored_pose)
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
            maximum_rightward_displacement = max(
                maximum_rightward_displacement,
                monitored_rightward,
            )
            if monitored_clearance <= GROUND_TRUTH_CLEARANCE_ABORT:
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
        final_pose = robot_pose

    final_clearance = motion.policy_footprint_clearance(final_pose)
    minimum_clearance = min(minimum_clearance, final_clearance)
    _, final_rightward = motion.forward_and_rightward_displacement(
        final_pose,
        initial_pose=initial_pose,
    )
    maximum_rightward_displacement = max(
        maximum_rightward_displacement,
        final_rightward,
    )

    mean_solve_time = (
        math.nan if not solve_wall_times else float(np.mean(solve_wall_times))
    )
    maximum_solve_time = (
        math.nan if not solve_wall_times else float(np.max(solve_wall_times))
    )
    return FixedFieldMotionResult(
        termination_reason=termination_reason,
        completed_control_cycles=completed_control_cycles,
        completed_physics_steps=completed_physics_steps,
        initial_pose=initial_pose,
        final_pose=final_pose,
        final_mpc_result=final_mpc_result,
        final_backup_result=final_backup_result,
        mpc_intervention_seen=mpc_intervention_seen,
        backup_intervention_seen=backup_intervention_seen,
        all_predictive_constraints_valid=(all_predictive_constraints_valid),
        all_backup_constraints_valid=all_backup_constraints_valid,
        cleared_proxy=cleared_proxy,
        route_complete=route_complete,
        mpc_failure_seen=mpc_failure_seen,
        backup_failure_seen=backup_failure_seen,
        ground_truth_abort_seen=ground_truth_abort_seen,
        application_stopped=application_stopped,
        minimum_policy_footprint_clearance=minimum_clearance,
        minimum_predicted_safety=minimum_predicted_safety,
        minimum_predicted_dtcbf_margin=minimum_predicted_margin,
        minimum_backup_safety=minimum_backup_safety,
        minimum_backup_cbf_margin=minimum_backup_margin,
        maximum_rightward_displacement=maximum_rightward_displacement,
        mean_mpc_solve_wall_seconds=mean_solve_time,
        maximum_mpc_solve_wall_seconds=maximum_solve_time,
    )


def print_motion_result(result: FixedFieldMotionResult) -> None:
    """Print acceptance checks for the fixed-field closed-loop trial."""

    final_forward, final_rightward = motion.forward_and_rightward_displacement(
        result.final_pose,
        initial_pose=result.initial_pose,
    )
    rightward_response_seen = (
        result.maximum_rightward_displacement >= MINIMUM_EXPECTED_RIGHTWARD_DISPLACEMENT
    )
    accepted = bool(
        result.mpc_intervention_seen
        and result.all_predictive_constraints_valid
        and result.all_backup_constraints_valid
        and result.cleared_proxy
        and rightward_response_seen
        and result.minimum_policy_footprint_clearance > GROUND_TRUTH_CLEARANCE_ABORT
        and not result.mpc_failure_seen
        and not result.backup_failure_seen
        and not result.ground_truth_abort_seen
        and not result.application_stopped
    )

    print("\n" + "=" * 78)
    print("SAFE-SAGE UNICYCLE MPC STAGE C RESULT")
    print("=" * 78)
    print(f"Termination:                     {result.termination_reason}")
    print(f"Completed control cycles:        {result.completed_control_cycles}")
    print(f"Completed physics steps:         {result.completed_physics_steps}")
    print("\nFinal chassis pose:")
    print(f"  x:                              {result.final_pose.x:.4f} m")
    print(f"  y:                              {result.final_pose.y:.4f} m")
    print(
        "  yaw:                            "
        f"{math.degrees(result.final_pose.yaw):.2f} degrees"
    )
    print("\nMotion summary:")
    print(f"  Forward displacement:           {final_forward:.4f} m")
    print(f"  Final rightward displacement:   {final_rightward:.4f} m")
    print(
        "  Maximum rightward displacement: "
        f"{result.maximum_rightward_displacement:.4f} m"
    )
    print(
        "  Minimum policy clearance:       "
        f"{result.minimum_policy_footprint_clearance:.4f} m"
    )
    print("\nPredictive and backup safety:")
    print(f"  MPC intervention observed:      {result.mpc_intervention_seen}")
    print(f"  Backup intervention observed:   {result.backup_intervention_seen}")
    print(
        f"  All predictive constraints:     {result.all_predictive_constraints_valid}"
    )
    print(f"  All backup constraints:         {result.all_backup_constraints_valid}")
    print(f"  Minimum predicted h_full:       {result.minimum_predicted_safety:.6f}")
    print(
        f"  Minimum predicted DTCBF margin: {result.minimum_predicted_dtcbf_margin:.3e}"
    )
    print(f"  Minimum backup safety:          {result.minimum_backup_safety:.6f}")
    print(f"  Minimum backup CBF margin:      {result.minimum_backup_cbf_margin:.3e}")
    print("\nRuntime diagnostics (not acceptance criteria):")
    print(
        f"  Mean MPC solve wall time:        {result.mean_mpc_solve_wall_seconds:.3f} s"
    )
    print(
        "  Maximum MPC solve wall time:     "
        f"{result.maximum_mpc_solve_wall_seconds:.3f} s"
    )
    print("\nControlled checks:")
    print(f"  Proxy cleared:                   {result.cleared_proxy}")
    print(f"  Rightward response observed:     {rightward_response_seen}")
    print(f"  MPC failure seen:                {result.mpc_failure_seen}")
    print(f"  Backup failure seen:             {result.backup_failure_seen}")
    print(f"  Ground-truth abort seen:         {result.ground_truth_abort_seen}")
    print(f"  Application stopped:             {result.application_stopped}")
    print("\nInterpretation:")
    if accepted:
        print(
            "  [PASS] Receding-horizon Safe-SAGE MPC produced physical "
            "right-side bypass motion while every predictive and analytical "
            "safety check remained valid."
        )
        print(
            "  Live LiDAR field refresh remains intentionally outside this "
            "fixed-field control stage."
        )
    else:
        print(
            "  [WARNING] The fixed-field motion trial did not pass every "
            "criterion. Keep the controller fail-closed and inspect the "
            "termination reason before enabling live field updates."
        )
    print("=" * 78)


def create_controller_stack(
    robot_controller: NovaCarterController,
) -> tuple[SafeSAGEUnicycleMPC, SafeSAGEUnicycleCBFFilter]:
    """Create the shared paper filter, MPC, and analytical backup."""

    paper_filter = SafeSAGECBFSafetyFilter(
        SafeSAGECBFConfig(
            gamma=stationary.CBF_GAMMA,
            minimum_planar_velocity=(-robot_controller.config.max_linear_speed),
            maximum_planar_velocity=(robot_controller.config.max_linear_speed),
            minimum_angular_velocity=(-robot_controller.config.max_angular_speed),
            maximum_angular_velocity=(robot_controller.config.max_angular_speed),
            planar_velocity_weight=(stationary.CBF_LINEAR_VELOCITY_WEIGHT),
            angular_velocity_weight=(stationary.CBF_ANGULAR_VELOCITY_WEIGHT),
        )
    )
    mpc = SafeSAGEUnicycleMPC(
        paper_filter,
        SafeSAGEUnicycleMPCConfig(
            horizon_steps=prediction.MPC_HORIZON_STEPS,
            time_step=prediction.MPC_TIME_STEP,
            gamma=stationary.CBF_GAMMA,
            lookahead_distance=stationary.UNICYCLE_LOOKAHEAD_DISTANCE,
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=(robot_controller.config.max_linear_speed),
            minimum_angular_velocity=(-robot_controller.config.max_angular_speed),
            maximum_angular_velocity=(robot_controller.config.max_angular_speed),
            linear_velocity_weight=(prediction.MPC_LINEAR_VELOCITY_WEIGHT),
            angular_velocity_weight=(prediction.MPC_ANGULAR_VELOCITY_WEIGHT),
            command_smoothness_weight=(prediction.MPC_COMMAND_SMOOTHNESS_WEIGHT),
            maximum_sqp_iterations=(prediction.MPC_MAXIMUM_SQP_ITERATIONS),
            maximum_solver_iterations=(prediction.MPC_MAXIMUM_SOLVER_ITERATIONS),
        ),
    )
    backup_filter = SafeSAGEUnicycleCBFFilter(
        paper_filter,
        SafeSAGEUnicycleCBFConfig(
            lookahead_distance=stationary.UNICYCLE_LOOKAHEAD_DISTANCE,
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=(robot_controller.config.max_linear_speed),
            minimum_angular_velocity=(-robot_controller.config.max_angular_speed),
            maximum_angular_velocity=(robot_controller.config.max_angular_speed),
            linear_velocity_weight=(stationary.CBF_LINEAR_VELOCITY_WEIGHT),
            angular_velocity_weight=(stationary.CBF_ANGULAR_VELOCITY_WEIGHT),
        ),
    )
    return mpc, backup_filter


def create_fixed_field_configuration() -> tuple[
    ClassSafetyPolicy,
    SafeSAGEGuidancePolicy,
    LocalOccupancyGridConfig,
    SafeSAGEFieldPipelineConfig,
]:
    """Create the policies, measured footprint, grid, and PDE settings."""

    guidance_policy = SafeSAGEGuidancePolicy.from_json(stationary.GUIDANCE_POLICY_PATH)
    class_policy = stationary.class_policy_from_guidance_policy(guidance_policy)
    measured_footprint = RectangularRobotFootprint(
        front_extent=stationary.ROBOT_FRONT_EXTENT,
        rear_extent=stationary.ROBOT_REAR_EXTENT,
        left_extent=stationary.ROBOT_LEFT_EXTENT,
        right_extent=stationary.ROBOT_RIGHT_EXTENT,
        safety_margin=stationary.ROBOT_SAFETY_MARGIN,
    )
    footprint = shift_rectangular_footprint_to_lookahead(
        measured_footprint,
        lookahead_distance=stationary.UNICYCLE_LOOKAHEAD_DISTANCE,
    )
    grid_config = LocalOccupancyGridConfig(
        resolution=stationary.GRID_RESOLUTION,
        minimum_x=motion.MOTION_GRID_MINIMUM_X,
        maximum_x=motion.MOTION_GRID_MAXIMUM_X,
        minimum_y=stationary.GRID_MINIMUM_Y,
        maximum_y=stationary.GRID_MAXIMUM_Y,
        obstacle_inflation_radius=0.0,
    )
    return (
        class_policy,
        guidance_policy,
        grid_config,
        stationary.create_field_pipeline_config(footprint),
    )


def main() -> None:
    """Run the controlled fixed-field Safe-SAGE MPC motion trial."""

    print("\n" + "=" * 78)
    print("NOVA CARTER SAFE-SAGE UNICYCLE MPC MOTION DEMO")
    print("=" * 78)
    print("Mode: Stage A/B fixed-field receding-horizon motion")
    print(f"Guidance policy: {stationary.GUIDANCE_POLICY_PATH}")
    print("Semantic label source: manual controlled proxy")

    (
        class_policy,
        guidance_policy,
        grid_config,
        field_config,
    ) = create_fixed_field_configuration()

    loaded_scene = load_robot_scene(
        scene_path=stationary.ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=stationary.ROBOT_PRIM_PATH,
        expected_meters_per_unit=stationary.EXPECTED_METERS_PER_UNIT,
    )
    stationary.create_static_proxy(loaded_scene.stage)
    robot_controller = NovaCarterController(robot_prim_path=stationary.ROBOT_PRIM_PATH)
    SimulationManager.setup_simulation(
        dt=stationary.PHYSICS_DT,
        device=stationary.PHYSICS_DEVICE,
    )

    print("\nPress Play in Isaac Sim to begin.")
    if not wait_until_playing(simulation_app):
        return

    print("\nPlay detected. Initializing physics...")
    app_utils.update_app(steps=10)
    robot_controller.verify_articulation()
    stationary.position_robot_for_experiment(robot_controller)
    stationary.verify_robot_start_pose(loaded_scene.stage)
    robot_controller.print_configuration()

    lidar = NovaCarterLidar(lidar_prim_path=stationary.PANDAR_XT32_LIDAR_PRIM_PATH)
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
        minimum_height=(floor_height_robot + stationary.MINIMUM_HEIGHT_ABOVE_FLOOR),
        maximum_height=(floor_height_robot + stationary.MAXIMUM_HEIGHT_ABOVE_FLOOR),
        minimum_range=stationary.MINIMUM_DETECTION_RANGE,
        maximum_range=stationary.MAXIMUM_DETECTION_RANGE,
        forward_only=False,
        angular_bin_size_degrees=stationary.ANGULAR_BIN_SIZE_DEGREES,
    )
    obstacle_tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=stationary.TRACKER_CLUSTER_RADIUS,
            minimum_cluster_points=(stationary.TRACKER_MINIMUM_CLUSTER_POINTS),
            maximum_association_distance=(
                stationary.TRACKER_MAXIMUM_ASSOCIATION_DISTANCE
            ),
        )
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
    print("\nWaiting for the fixed-field LiDAR snapshot...")
    points_sensor_frame, frame_info, _ = stationary.wait_for_first_scan(
        lidar,
        robot_controller,
    )
    print("Building the fixed yaw-dependent Safe-SAGE field...")
    pipeline_frame = motion.build_motion_pipeline_frame(
        points_sensor_frame=points_sensor_frame,
        frame_info=frame_info,
        robot_pose=anchor_pose,
        field_anchor_pose=anchor_pose,
        obstacle_tracker=obstacle_tracker,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=grid_config,
        field_config=field_config,
    )
    print(
        "Fixed field built from LiDAR frame "
        f"{pipeline_frame.sensor_frame_id} in "
        f"{pipeline_frame.build_wall_seconds:.3f} s."
    )

    mpc, backup_filter = create_controller_stack(robot_controller)
    passing_side = guidance_policy.parameters_for(
        stationary.SEMANTIC_CLASS
    ).passing_side
    print_directional_field_audit(
        pipeline_frame=pipeline_frame,
        mpc=mpc,
        backup_filter=backup_filter,
        passing_side=passing_side,
    )
    result = run_fixed_field_motion_trial(
        stage=loaded_scene.stage,
        robot_controller=robot_controller,
        pipeline_frame=pipeline_frame,
        mpc=mpc,
        backup_filter=backup_filter,
        passing_side=passing_side,
    )
    print_motion_result(result)

    robot_controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()

    print("\nExpected successful scene behavior:")
    print("  - Nova Carter begins an early robot-right turn.")
    print("  - The blue semantic proxy remains fixed in the scene.")
    print("  - The robot passes the proxy without touching it.")
    print("  - The waypoint command begins steering it back toward the route.")
    print("  - No later LiDAR frame is used during this fixed-field stage.")
    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when finished.")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 78)
        print("ERROR DURING SAFE-SAGE UNICYCLE MPC MOTION DEMO")
        print("=" * 78)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
