"""Execute the validated Safe-SAGE planar path on Nova Carter.

This is the controlled physical-motion stage after the stationary
``safe_sage_planar_to_unicycle_demo.py`` validation.

Stage A - fixed real-perception experiment:

1. Position Nova Carter and the blue proxy in the spacious restaurant area.
2. Capture one real PandarXT-32 scan and associate the proxy track.
3. Build the chassis-origin Safe-SAGE field and obtain the successful virtual
   planar bypass path.
4. Build the look-ahead-referenced field for the unchanged physical footprint.

Stage B - fail-closed receding-horizon motion:

1. Add a reported lateral tracking-tube allowance to the fixed planar path.
2. Convert a moving path horizon into feasible unicycle chassis targets.
3. Solve and refine the ``(v, omega)`` MPC horizon under exact DTCBF checks.
4. Re-evaluate the analytical ``h_full`` CBF at every physics step.
5. Apply only the backed-up command to Nova Carter's differential drive.
6. Stop after longitudinally clearing the proxy or on any failed guard.

The proxy's known pose is used only for the independent clearance abort and
final evaluation. It is never supplied to perception, MPC, or the analytical
safety filter. Perception and the safety fields remain fixed in this stage so
that path execution is the only new component under test.

Run from the project root:

    python -u scripts/demos/safe_sage_planar_to_unicycle_motion_demo.py

Playback starts automatically. Set ``ISAAC_SIM_HEADLESS=1`` for a headless
validation run.
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
from scripts.demos import safe_sage_planar_to_unicycle_demo as integration


oracle = integration.oracle
motion = integration.motion
stationary = integration.stationary
simulation_app = integration.simulation_app


import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd
from robot.control.differential_drive import NovaCarterController
from robot.control.planar_path_reference import (
    PlanarPathReference,
)
from robot.control.unicycle_model import (
    integrate_unicycle_euler,
    lookahead_state,
    wrap_to_pi,
)
from robot.navigation.waypoint_controller import RobotPose2D
from robot.runtime.simulation_loop import (
    advance_playing_steps,
    wait_until_playing,
)
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
)
from robot.safety.safe_sage_unicycle_cbf_filter import (
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
# Controlled physical-motion configuration
# ---------------------------------------------------------------------------


MPC_CONTROL_INTERVAL_SECONDS = integration.UNICYCLE_TIME_STEP
CONTROL_PHYSICS_STEPS = max(
    1,
    round(MPC_CONTROL_INTERVAL_SECONDS / stationary.PHYSICS_DT),
)
MAXIMUM_MOTION_SECONDS = 25.0
MAXIMUM_CONTROL_CYCLES = math.ceil(
    MAXIMUM_MOTION_SECONDS / MPC_CONTROL_INTERVAL_SECONDS
)

# A six-second stationary horizon proves that a complete bypass exists, but
# during physical receding-horizon tracking it sees the later horizontal path
# segment too early. Nova Carter then begins unwinding its heading before it
# reaches the planar path's lateral offset. A four-second execution horizon
# keeps the optimization focused on locally reachable path geometry. Safety
# remains governed by the unchanged exact DTCBF and analytical CBF layers.
MOTION_MPC_HORIZON_STEPS = 20
MOTION_POSITION_TRACKING_WEIGHT = 8.0
MOTION_YAW_TRACKING_WEIGHT = 0.05

PATH_TRACKING_LOOKAHEAD_DISTANCE = 0.25
PATH_TRACKING_LINEAR_SPEED = 0.20
PATH_TRACKING_HEADING_GAIN = 2.5
PATH_TRACKING_ROTATE_IN_PLACE_ANGLE = math.radians(65.0)
PATH_TRACKING_FINAL_POSITION_TOLERANCE = 0.05

# The ideal planar state can translate laterally without rotating. Nova
# Carter instead rotates its long rectangular footprint and has nonzero
# closed-loop tracking error. This allowance expands the geometric reference
# only in the selected passing direction; the Safe-SAGE field, DTCBF, and
# analytical CBF remain unchanged and still certify every executable command.
PATH_LATERAL_TRACKING_TUBE_ALLOWANCE = 0.20
PATH_TERMINAL_FORWARD_OVERRUN = 0.10

GROUND_TRUTH_CLEARANCE_ABORT = motion.GROUND_TRUTH_CLEARANCE_ABORT
MINIMUM_EXPECTED_RIGHTWARD_DISPLACEMENT = (
    motion.MINIMUM_EXPECTED_RIGHTWARD_DISPLACEMENT
)
SETTLED_LINEAR_SPEED = motion.SETTLED_LINEAR_SPEED
SETTLED_ANGULAR_SPEED = motion.SETTLED_ANGULAR_SPEED
SETTLED_CONFIRMATION_CYCLES = 8
PATH_PROGRESS_SEARCH_BACKTRACK = 2

REFERENCE_MARKER_ROOT = "/World/SafeSAGEPlanarToUnicycleMotion/Reference"
EXECUTED_MARKER_ROOT = "/World/SafeSAGEPlanarToUnicycleMotion/Executed"
REFERENCE_MARKER_COLOR = (0.10, 0.85, 1.00)
EXECUTED_MARKER_COLOR = (0.10, 0.95, 0.35)


@dataclass(frozen=True)
class PlanarToUnicycleMotionResult:
    """Physical closed-loop outcome and independent safety diagnostics."""

    termination_reason: str
    completed_control_cycles: int
    completed_physics_steps: int
    initial_pose: RobotPose2D
    final_pose: RobotPose2D
    executed_chassis_states: np.ndarray
    final_path_progress_index: int
    all_predictive_constraints_valid: bool
    all_backup_constraints_valid: bool
    mpc_failure_seen: bool
    backup_failure_seen: bool
    ground_truth_abort_seen: bool
    application_stopped: bool
    cleared_proxy: bool
    refinement_used_count: int
    refinement_failure_count: int
    feasibility_recovery_seen: bool
    backup_intervention_seen: bool
    minimum_policy_clearance: float
    minimum_predicted_safety: float
    minimum_predicted_margin: float
    minimum_backup_safety: float
    minimum_backup_margin: float
    maximum_rightward_displacement: float
    maximum_path_error: float
    mean_mpc_solve_wall_seconds: float
    maximum_mpc_solve_wall_seconds: float


def select_path_progress_index(
    path_states: np.ndarray,
    current_position_grid_xy: np.ndarray,
    *,
    previous_index: int,
) -> int:
    """Return a nearest, monotonically nondecreasing path index."""

    states = np.asarray(path_states, dtype=np.float64)
    position = np.asarray(current_position_grid_xy, dtype=np.float64)
    if states.ndim != 2 or states.shape[1:] != (3,):
        raise ValueError("path_states must have shape (N, 3).")
    if states.shape[0] < 2 or not np.isfinite(states).all():
        raise ValueError("path_states must be finite and contain two states.")
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError(
            "current_position_grid_xy must be one finite planar point."
        )
    if not 0 <= previous_index < states.shape[0]:
        raise ValueError("previous_index lies outside path_states.")

    search_start = max(
        0,
        previous_index - PATH_PROGRESS_SEARCH_BACKTRACK,
    )
    distances = np.linalg.norm(
        states[search_start:, :2] - position,
        axis=1,
    )
    nearest_index = search_start + int(np.argmin(distances))
    return max(previous_index, nearest_index)


def add_lateral_tracking_tube_allowance(
    path_states: np.ndarray,
    *,
    preferred_lateral_direction_grid_y: float,
    maximum_allowance: float,
) -> np.ndarray:
    """Move the path outward to cover nonholonomic tracking error.

    The allowance is zero at the centerline and grows linearly with the
    planar oracle's lateral excursion. The start state is therefore unchanged
    and the full allowance appears only after the oracle commits to the
    selected passing side.
    """

    path = np.asarray(path_states, dtype=np.float64)
    if path.ndim != 2 or path.shape[1:] != (3,) or path.shape[0] == 0:
        raise ValueError("path_states must have shape (N, 3) with N > 0.")
    if not np.isfinite(path).all():
        raise ValueError("path_states must contain only finite values.")
    direction = float(preferred_lateral_direction_grid_y)
    if not math.isclose(abs(direction), 1.0, abs_tol=1e-9):
        raise ValueError(
            "preferred_lateral_direction_grid_y must be -1.0 or 1.0."
        )
    allowance = float(maximum_allowance)
    if not math.isfinite(allowance) or allowance < 0.0:
        raise ValueError("maximum_allowance must be finite and nonnegative.")

    adapted = np.array(path, dtype=np.float64, copy=True)
    signed_excursion = np.maximum(0.0, direction * adapted[:, 1])
    peak_excursion = float(np.max(signed_excursion))
    if peak_excursion > 0.0 and allowance > 0.0:
        excursion_fraction = signed_excursion / peak_excursion
        adapted[:, 1] += direction * allowance * excursion_fraction
    return np.ascontiguousarray(adapted)


def append_terminal_path_overrun(
    path_states: np.ndarray,
    *,
    overrun_distance: float,
) -> np.ndarray:
    """Extend the final path tangent beyond the strict clearance threshold."""

    path = np.asarray(path_states, dtype=np.float64)
    if path.ndim != 2 or path.shape[1:] != (3,) or path.shape[0] < 2:
        raise ValueError("path_states must have shape (N, 3) with N >= 2.")
    if not np.isfinite(path).all():
        raise ValueError("path_states must contain only finite values.")
    overrun = float(overrun_distance)
    if not math.isfinite(overrun) or overrun < 0.0:
        raise ValueError("overrun_distance must be finite and nonnegative.")
    if overrun == 0.0:
        return np.ascontiguousarray(path.copy())

    terminal_delta = path[-1, :2] - path[-2, :2]
    terminal_length = float(np.linalg.norm(terminal_delta))
    if terminal_length <= 1e-12:
        raise ValueError("The final path segment must have nonzero length.")
    terminal_state = path[-1].copy()
    terminal_state[:2] += overrun * terminal_delta / terminal_length
    return np.ascontiguousarray(np.vstack((path, terminal_state)))


def path_lookahead_index(
    path_states: np.ndarray,
    *,
    start_index: int,
    lookahead_distance: float,
) -> int:
    """Return the first path index at least one arc distance ahead."""

    states = np.asarray(path_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1:] != (3,):
        raise ValueError("path_states must have shape (N, 3).")
    if states.shape[0] < 2 or not np.isfinite(states).all():
        raise ValueError("path_states must be finite and contain two states.")
    if not 0 <= start_index < states.shape[0]:
        raise ValueError("start_index lies outside path_states.")
    distance = float(lookahead_distance)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("lookahead_distance must be finite and positive.")

    accumulated_distance = 0.0
    for index in range(start_index + 1, states.shape[0]):
        accumulated_distance += float(
            np.linalg.norm(states[index, :2] - states[index - 1, :2])
        )
        if accumulated_distance >= distance:
            return index
    return states.shape[0] - 1


def build_feedback_tracking_reference(
    path_states: np.ndarray,
    *,
    current_chassis_state: np.ndarray,
    progress_index: int,
    mpc: SafeSAGEUnicycleMPC,
) -> PlanarPathReference:
    """Roll out a feasible unicycle tracker along the planar path geometry.

    The planar oracle path is geometric guidance here, not a holonomic time
    schedule. Every returned target state is generated by the same unicycle
    dynamics used inside the MPC, so the reference is reachable by
    construction while retaining feedback to the Safe-SAGE passing path.
    """

    path = np.asarray(path_states, dtype=np.float64)
    initial_state = np.asarray(current_chassis_state, dtype=np.float64)
    if path.ndim != 2 or path.shape[1:] != (3,):
        raise ValueError("path_states must have shape (N, 3).")
    if path.shape[0] < 2 or not np.isfinite(path).all():
        raise ValueError("path_states must be finite and contain two states.")
    if initial_state.shape != (3,) or not np.isfinite(initial_state).all():
        raise ValueError("current_chassis_state must be one finite state.")
    if not 0 <= progress_index < path.shape[0]:
        raise ValueError("progress_index lies outside path_states.")

    states = np.empty((mpc.config.horizon_steps + 1, 3), dtype=np.float64)
    commands = np.empty((mpc.config.horizon_steps, 2), dtype=np.float64)
    states[0] = initial_state
    rollout_progress = progress_index

    for command_index in range(mpc.config.horizon_steps):
        current_state = states[command_index]
        rollout_progress = select_path_progress_index(
            path,
            current_state[:2],
            previous_index=rollout_progress,
        )
        final_error = path[-1, :2] - current_state[:2]
        if (
            rollout_progress == path.shape[0] - 1
            and float(np.linalg.norm(final_error))
            <= PATH_TRACKING_FINAL_POSITION_TOLERANCE
        ):
            command = np.zeros(2, dtype=np.float64)
        else:
            target_index = path_lookahead_index(
                path,
                start_index=rollout_progress,
                lookahead_distance=PATH_TRACKING_LOOKAHEAD_DISTANCE,
            )
            target_error = path[target_index, :2] - current_state[:2]
            target_distance = float(np.linalg.norm(target_error))
            desired_heading = math.atan2(
                float(target_error[1]),
                float(target_error[0]),
            )
            heading_error = wrap_to_pi(
                desired_heading - float(current_state[2])
            )
            angular_velocity = float(
                np.clip(
                    PATH_TRACKING_HEADING_GAIN * heading_error,
                    mpc.config.minimum_angular_velocity,
                    mpc.config.maximum_angular_velocity,
                )
            )
            if abs(heading_error) >= PATH_TRACKING_ROTATE_IN_PLACE_ANGLE:
                linear_velocity = 0.0
            else:
                alignment = max(0.0, math.cos(heading_error))
                linear_velocity = float(
                    np.clip(
                        min(
                            PATH_TRACKING_LINEAR_SPEED,
                            target_distance / mpc.config.time_step,
                        )
                        * alignment,
                        mpc.config.minimum_linear_velocity,
                        mpc.config.maximum_linear_velocity,
                    )
                )
            command = np.array(
                [linear_velocity, angular_velocity],
                dtype=np.float64,
            )
        commands[command_index] = command
        states[command_index + 1] = integrate_unicycle_euler(
            current_state,
            command,
            time_step=mpc.config.time_step,
        )

    return PlanarPathReference(
        target_states=states,
        seed_chassis_states=states,
        seed_commands=commands,
    )


def shifted_warm_start(
    command_sequence: np.ndarray,
) -> np.ndarray:
    """Shift one safe MPC horizon and append a stationary terminal command."""

    commands = np.asarray(command_sequence, dtype=np.float64)
    if commands.ndim != 2 or commands.shape[1:] != (2,):
        raise ValueError("command_sequence must have shape (N, 2).")
    if commands.shape[0] < 1 or not np.isfinite(commands).all():
        raise ValueError("command_sequence must be finite and non-empty.")
    return np.ascontiguousarray(
        np.vstack((commands[1:], np.zeros((1, 2), dtype=np.float64))),
        dtype=np.float64,
    )


def create_motion_mpc(
    robot_controller: NovaCarterController,
    *,
    paper_filter: oracle.SafeSAGECBFSafetyFilter,
) -> SafeSAGEUnicycleMPC:
    """Create the locally focused physical path-tracking MPC."""

    return SafeSAGEUnicycleMPC(
        paper_filter,
        SafeSAGEUnicycleMPCConfig(
            horizon_steps=MOTION_MPC_HORIZON_STEPS,
            time_step=MPC_CONTROL_INTERVAL_SECONDS,
            gamma=stationary.CBF_GAMMA,
            lookahead_distance=stationary.UNICYCLE_LOOKAHEAD_DISTANCE,
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
            linear_velocity_weight=(
                integration.UNICYCLE_LINEAR_VELOCITY_WEIGHT
            ),
            angular_velocity_weight=(
                integration.UNICYCLE_ANGULAR_VELOCITY_WEIGHT
            ),
            command_smoothness_weight=(
                integration.UNICYCLE_COMMAND_SMOOTHNESS_WEIGHT
            ),
            position_tracking_weight=MOTION_POSITION_TRACKING_WEIGHT,
            yaw_tracking_weight=MOTION_YAW_TRACKING_WEIGHT,
            maximum_sqp_iterations=(
                integration.UNICYCLE_MAXIMUM_SQP_ITERATIONS
            ),
            maximum_restoration_iterations=(
                integration.UNICYCLE_MAXIMUM_RESTORATION_ITERATIONS
            ),
            maximum_solver_iterations=(
                integration.UNICYCLE_MAXIMUM_SOLVER_ITERATIONS
            ),
        ),
    )


def nearest_path_error(
    path_states: np.ndarray,
    position_grid_xy: np.ndarray,
) -> float:
    """Return Euclidean distance to the closest planar reference sample."""

    return float(
        np.min(
            np.linalg.norm(
                np.asarray(path_states, dtype=np.float64)[:, :2]
                - np.asarray(position_grid_xy, dtype=np.float64),
                axis=1,
            )
        )
    )


def evaluate_analytical_backup(
    *,
    safety_filter: SafeSAGEUnicycleCBFFilter,
    nominal_command: tuple[float, float],
    chassis_state: np.ndarray,
    pipeline_frame: motion.MotionPipelineFrame,
) -> SafeSAGEUnicycleCBFResult:
    """Evaluate the fixed field at the current physical look-ahead state."""

    query_state = lookahead_state(
        chassis_state,
        lookahead_distance=safety_filter.config.lookahead_distance,
    )
    return safety_filter.filter_command(
        nominal_command[0],
        nominal_command[1],
        query_position_grid_xy=query_state[:2],
        query_yaw_radians=float(query_state[2]),
        configuration_space=pipeline_frame.pipeline.configuration_space,
        h_full=pipeline_frame.pipeline.h_full,
        guidance_fields=pipeline_frame.pipeline.guidance_fields,
        temporal_safety_derivative=(
            stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE
        ),
    )


def print_live_status(
    *,
    cycle: int,
    progress_index: int,
    path_state_count: int,
    chassis_state: np.ndarray,
    clearance: float,
    rightward: float,
    path_error: float,
    mpc_result: SafeSAGEUnicycleMPCResult,
    backup_result: SafeSAGEUnicycleCBFResult,
    solve_wall_seconds: float,
    refinement_used: bool,
) -> None:
    """Print one compact closed-loop status line."""

    print(
        f"  cycle={cycle:<3} "
        f"path={progress_index:02d}/{path_state_count - 1:02d} "
        f"q=({chassis_state[0]:.3f},{chassis_state[1]:.3f},"
        f"{math.degrees(chassis_state[2]):.1f}deg) "
        f"clear={clearance:.3f} right={rightward:.3f} "
        f"path_err={path_error:.3f} "
        f"mpc=({mpc_result.linear_velocity:.3f},"
        f"{mpc_result.angular_velocity:.3f}) "
        f"out=({backup_result.safe_linear_velocity:.3f},"
        f"{backup_result.safe_angular_velocity:.3f}) "
        f"hmin={mpc_result.minimum_predicted_safety_value:.3f} "
        f"margin={mpc_result.minimum_dtcbf_margin:.2e} "
        f"solve={solve_wall_seconds:.3f}s "
        f"refine={refinement_used} status={mpc_result.status}"
    )


def run_motion_trial(
    *,
    stage: Usd.Stage,
    robot_controller: NovaCarterController,
    planar_path_states: np.ndarray,
    pipeline_frame: motion.MotionPipelineFrame,
    mpc: SafeSAGEUnicycleMPC,
    backup_filter: SafeSAGEUnicycleCBFFilter,
    direction_grid_y: float,
) -> PlanarToUnicycleMotionResult:
    """Execute the fixed-path, fixed-field fail-closed motion trial."""

    initial_pose = stationary.read_robot_pose(stage)
    robot_pose = initial_pose
    path_states = np.asarray(planar_path_states, dtype=np.float64)
    clearance_threshold = motion.proxy_clearance_forward_threshold(initial_pose)
    completed_control_cycles = 0
    completed_physics_steps = 0
    progress_index = 0
    warm_start_commands: np.ndarray | None = None
    executed_states = [np.zeros(3, dtype=np.float64)]
    all_predictive_valid = True
    all_backup_valid = True
    mpc_failure_seen = False
    backup_failure_seen = False
    ground_truth_abort_seen = False
    application_stopped = False
    cleared_proxy = False
    refinement_used_count = 0
    refinement_failure_count = 0
    feasibility_recovery_seen = False
    backup_intervention_seen = False
    settled_confirmation_count = 0
    minimum_clearance = motion.policy_footprint_clearance(initial_pose)
    minimum_predicted_safety = math.inf
    minimum_predicted_margin = math.inf
    minimum_backup_safety = math.inf
    minimum_backup_margin = math.inf
    maximum_rightward = 0.0
    maximum_path_error = 0.0
    solve_wall_times: list[float] = []
    termination_reason = "Maximum controlled motion duration reached."

    print("\n" + "=" * 78)
    print("SAFE-SAGE PLANAR-TO-UNICYCLE STAGE D - PHYSICAL CLOSED LOOP")
    print("=" * 78)
    print("\nControlled settings:")
    print("  Perception and field updates:   one initial LiDAR snapshot")
    print("  Planar reference:               fixed successful oracle path")
    print(
        "  Lateral tracking-tube allowance:"
        f" {PATH_LATERAL_TRACKING_TUBE_ALLOWANCE:.3f} m"
    )
    print(
        "  Terminal path overrun:           "
        f"{PATH_TERMINAL_FORWARD_OVERRUN:.3f} m"
    )
    print("  Physical state/input:           (x, y, yaw) / (v, omega)")
    print("  MPC strategy:                   safe bootstrap + refinement")
    print(f"  MPC horizon:                    {mpc.config.horizon_steps} steps")
    print("  Analytical CBF rate:            every physics step")
    print(f"  MPC control interval:           {MPC_CONTROL_INTERVAL_SECONDS:.3f} s")
    print(f"  Physics steps per MPC command:  {CONTROL_PHYSICS_STEPS}")
    print(f"  Maximum simulated duration:     {MAXIMUM_MOTION_SECONDS:.1f} s")
    print(f"  Ground-truth clearance abort:   {GROUND_TRUTH_CLEARANCE_ABORT:.3f} m")
    print(f"  Proxy-clear forward threshold:  {clearance_threshold:.3f} m")
    print("  MPC runtime acceptance limit:   none")
    print("\nLive physical status:")

    for cycle in range(MAXIMUM_CONTROL_CYCLES):
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

        robot_controller.stop()
        robot_pose = stationary.read_robot_pose(stage)
        field_position, field_yaw = motion.current_pose_in_field_frame(
            robot_pose,
            anchor_pose=pipeline_frame.anchor_pose_world,
        )
        chassis_state = np.array(
            [field_position[0], field_position[1], field_yaw],
            dtype=np.float64,
        )
        progress_index = select_path_progress_index(
            path_states,
            field_position,
            previous_index=progress_index,
        )
        reference = build_feedback_tracking_reference(
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
            configuration_space=pipeline_frame.pipeline.configuration_space,
            h_full=pipeline_frame.pipeline.h_full,
            guidance_fields=pipeline_frame.pipeline.guidance_fields,
            temporal_safety_derivative=(
                stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE
            ),
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

        feasibility_recovery_seen |= (
            "feasibility_recovered" in candidate.status
        )
        refinement = mpc.plan(
            chassis_state,
            reference.seed_commands,
            initial_command_guess=candidate.command_sequence,
            target_chassis_states_grid=reference.target_states,
            preferred_lateral_direction_grid_y=direction_grid_y,
            configuration_space=pipeline_frame.pipeline.configuration_space,
            h_full=pipeline_frame.pipeline.h_full,
            guidance_fields=pipeline_frame.pipeline.guidance_fields,
            temporal_safety_derivative=(
                stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE
            ),
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
            # The candidate is already exactly certified. A failed optional
            # objective refinement therefore does not justify discarding the
            # safe command, but it remains visible in the final diagnostics.
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
        warm_start_commands = shifted_warm_start(
            mpc_result.command_sequence
        )

        command_segment_failed = False
        last_backup_result: SafeSAGEUnicycleCBFResult | None = None
        for _ in range(CONTROL_PHYSICS_STEPS):
            if not simulation_app.is_running():
                application_stopped = True
                termination_reason = (
                    "Isaac Sim stopped during command execution."
                )
                command_segment_failed = True
                break
            while simulation_app.is_running() and not app_utils.is_playing():
                robot_controller.stop()
                simulation_app.update()
            if not simulation_app.is_running():
                application_stopped = True
                termination_reason = (
                    "Isaac Sim stopped during command execution."
                )
                command_segment_failed = True
                break

            monitored_pose = stationary.read_robot_pose(stage)
            monitored_position, monitored_yaw = (
                motion.current_pose_in_field_frame(
                    monitored_pose,
                    anchor_pose=pipeline_frame.anchor_pose_world,
                )
            )
            monitored_state = np.array(
                [
                    monitored_position[0],
                    monitored_position[1],
                    monitored_yaw,
                ],
                dtype=np.float64,
            )
            monitored_clearance = motion.policy_footprint_clearance(
                monitored_pose
            )
            if monitored_clearance <= GROUND_TRUTH_CLEARANCE_ABORT:
                ground_truth_abort_seen = True
                termination_reason = (
                    "Independent clearance guard triggered before command "
                    "application."
                )
                command_segment_failed = True
                break

            backup_result = evaluate_analytical_backup(
                safety_filter=backup_filter,
                nominal_command=mpc_result.first_command,
                chassis_state=monitored_state,
                pipeline_frame=pipeline_frame,
            )
            last_backup_result = backup_result
            backup_valid = integration.analytical_result_is_valid(
                backup_result
            )
            all_backup_valid &= backup_valid
            if not backup_valid:
                backup_failure_seen = True
                termination_reason = (
                    "Fail-closed analytical backup stop: "
                    f"{backup_result.emergency_reason or 'invalid constraint'}"
                )
                command_segment_failed = True
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
            robot_controller.apply_command(
                backup_result.safe_linear_velocity,
                backup_result.safe_angular_velocity,
            )
            simulation_app.update()
            completed_physics_steps += 1

            monitored_pose = stationary.read_robot_pose(stage)
            robot_pose = monitored_pose
            monitored_position, monitored_yaw = (
                motion.current_pose_in_field_frame(
                    monitored_pose,
                    anchor_pose=pipeline_frame.anchor_pose_world,
                )
            )
            monitored_state = np.array(
                [
                    monitored_position[0],
                    monitored_position[1],
                    monitored_yaw,
                ],
                dtype=np.float64,
            )
            monitored_clearance = motion.policy_footprint_clearance(
                monitored_pose
            )
            forward, rightward = motion.forward_and_rightward_displacement(
                monitored_pose,
                initial_pose=initial_pose,
            )
            path_error = nearest_path_error(path_states, monitored_position)
            minimum_clearance = min(minimum_clearance, monitored_clearance)
            maximum_rightward = max(maximum_rightward, rightward)
            maximum_path_error = max(maximum_path_error, path_error)
            if monitored_clearance <= GROUND_TRUTH_CLEARANCE_ABORT:
                ground_truth_abort_seen = True
                termination_reason = (
                    "Independent clearance guard triggered during command "
                    "execution."
                )
                command_segment_failed = True
                break
            if forward >= clearance_threshold:
                cleared_proxy = True
                termination_reason = (
                    "Nova Carter cleared the proxy longitudinally."
                )
                command_segment_failed = True
                break

        robot_controller.stop()
        if last_backup_result is not None:
            field_position, field_yaw = motion.current_pose_in_field_frame(
                robot_pose,
                anchor_pose=pipeline_frame.anchor_pose_world,
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
            current_path_error = nearest_path_error(
                path_states,
                field_position,
            )
            executed_states.append(chassis_state.copy())
            print_live_status(
                cycle=cycle,
                progress_index=progress_index,
                path_state_count=path_states.shape[0],
                chassis_state=chassis_state,
                clearance=current_clearance,
                rightward=rightward,
                path_error=current_path_error,
                mpc_result=mpc_result,
                backup_result=last_backup_result,
                solve_wall_seconds=solve_wall_seconds,
                refinement_used=refinement_used,
            )

            settled = bool(
                last_backup_result.safe_linear_velocity
                <= SETTLED_LINEAR_SPEED
                and abs(last_backup_result.safe_angular_velocity)
                <= SETTLED_ANGULAR_SPEED
            )
            if settled:
                settled_confirmation_count += 1
            else:
                settled_confirmation_count = 0
            if (
                settled_confirmation_count >= SETTLED_CONFIRMATION_CYCLES
                and not cleared_proxy
            ):
                termination_reason = (
                    "The safe command settled before proxy clearance."
                )
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
        anchor_pose=pipeline_frame.anchor_pose_world,
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
    mean_solve_time = (
        math.nan
        if not solve_wall_times
        else float(np.mean(solve_wall_times))
    )
    maximum_solve_time = (
        math.nan
        if not solve_wall_times
        else float(np.max(solve_wall_times))
    )
    return PlanarToUnicycleMotionResult(
        termination_reason=termination_reason,
        completed_control_cycles=completed_control_cycles,
        completed_physics_steps=completed_physics_steps,
        initial_pose=initial_pose,
        final_pose=final_pose,
        executed_chassis_states=np.ascontiguousarray(
            np.asarray(executed_states, dtype=np.float64)
        ),
        final_path_progress_index=progress_index,
        all_predictive_constraints_valid=all_predictive_valid,
        all_backup_constraints_valid=all_backup_valid,
        mpc_failure_seen=mpc_failure_seen,
        backup_failure_seen=backup_failure_seen,
        ground_truth_abort_seen=ground_truth_abort_seen,
        application_stopped=application_stopped,
        cleared_proxy=cleared_proxy,
        refinement_used_count=refinement_used_count,
        refinement_failure_count=refinement_failure_count,
        feasibility_recovery_seen=feasibility_recovery_seen,
        backup_intervention_seen=backup_intervention_seen,
        minimum_policy_clearance=minimum_clearance,
        minimum_predicted_safety=minimum_predicted_safety,
        minimum_predicted_margin=minimum_predicted_margin,
        minimum_backup_safety=minimum_backup_safety,
        minimum_backup_margin=minimum_backup_margin,
        maximum_rightward_displacement=maximum_rightward,
        maximum_path_error=maximum_path_error,
        mean_mpc_solve_wall_seconds=mean_solve_time,
        maximum_mpc_solve_wall_seconds=maximum_solve_time,
    )


def print_motion_result(
    result: PlanarToUnicycleMotionResult,
) -> bool:
    """Print the controlled physical-motion acceptance report."""

    final_forward, final_rightward = motion.forward_and_rightward_displacement(
        result.final_pose,
        initial_pose=result.initial_pose,
    )
    rightward_response = bool(
        result.maximum_rightward_displacement
        >= MINIMUM_EXPECTED_RIGHTWARD_DISPLACEMENT
    )
    accepted = bool(
        result.cleared_proxy
        and rightward_response
        and result.all_predictive_constraints_valid
        and result.all_backup_constraints_valid
        and result.minimum_policy_clearance > GROUND_TRUTH_CLEARANCE_ABORT
        and not result.mpc_failure_seen
        and not result.backup_failure_seen
        and not result.ground_truth_abort_seen
        and not result.application_stopped
    )

    print("\n" + "=" * 78)
    print("SAFE-SAGE PLANAR-TO-UNICYCLE STAGE D RESULT")
    print("=" * 78)
    print(f"Termination:                       {result.termination_reason}")
    print(f"Completed MPC cycles:              {result.completed_control_cycles}")
    print(f"Completed physics steps:           {result.completed_physics_steps}")
    print(f"Final path progress index:         {result.final_path_progress_index}")

    print("\nPhysical motion:")
    print(f"  Final pose:                      ({result.final_pose.x:.4f}, {result.final_pose.y:.4f}, {math.degrees(result.final_pose.yaw):.2f} deg)")
    print(f"  Forward displacement:            {final_forward:.4f} m")
    print(f"  Final rightward displacement:    {final_rightward:.4f} m")
    print(f"  Maximum rightward displacement:  {result.maximum_rightward_displacement:.4f} m")
    print(f"  Maximum sampled path error:      {result.maximum_path_error:.4f} m")
    print(f"  Minimum policy clearance:        {result.minimum_policy_clearance:.4f} m")

    print("\nPredictive and analytical safety:")
    print(f"  All predictive constraints:      {result.all_predictive_constraints_valid}")
    print(f"  All analytical constraints:      {result.all_backup_constraints_valid}")
    print(f"  Minimum predicted h_full:        {result.minimum_predicted_safety:.6f}")
    print(f"  Minimum predicted DTCBF margin:  {result.minimum_predicted_margin:.3e}")
    print(f"  Minimum analytical h_full:       {result.minimum_backup_safety:.6f}")
    print(f"  Minimum analytical CBF margin:   {result.minimum_backup_margin:.3e}")
    print(f"  Analytical intervention seen:    {result.backup_intervention_seen}")

    print("\nSolver diagnostics:")
    print(f"  Safe refinements used:           {result.refinement_used_count}")
    print(f"  Refinement failures:             {result.refinement_failure_count}")
    print(f"  Feasibility recovery observed:   {result.feasibility_recovery_seen}")
    print(f"  Mean two-pass solve time:        {result.mean_mpc_solve_wall_seconds:.3f} s")
    print(f"  Maximum two-pass solve time:     {result.maximum_mpc_solve_wall_seconds:.3f} s")

    print("\nControlled checks:")
    print(f"  Proxy cleared:                   {result.cleared_proxy}")
    print(f"  Rightward response observed:     {rightward_response}")
    print(f"  MPC failure seen:                {result.mpc_failure_seen}")
    print(f"  Analytical failure seen:         {result.backup_failure_seen}")
    print(f"  Clearance abort seen:            {result.ground_truth_abort_seen}")
    print(f"  Application stopped:             {result.application_stopped}")

    print("\nInterpretation:")
    if accepted:
        print(
            "  [PASS] Nova Carter physically followed the planar Safe-SAGE "
            "passing side, cleared the proxy, and retained valid predictive "
            "and full-rate analytical safety constraints throughout the "
            "controlled fixed-field trial."
        )
        print("  Live perception/field updates may proceed to the next stage.")
    else:
        print(
            "  [WARNING] The controlled physical trial did not pass every "
            "criterion. The robot was stopped fail-closed; inspect the "
            "termination reason before enabling live field updates."
        )
    print("=" * 78)
    return accepted


def main() -> None:
    """Run the fixed-field planar-reference Nova Carter motion trial."""

    print("\n" + "=" * 78)
    print("NOVA CARTER SAFE-SAGE PLANAR-TO-UNICYCLE MOTION DEMO")
    print("=" * 78)
    print("Mode: fixed perception + receding-horizon physical execution")
    print(f"Guidance policy: {oracle.PAPER_BASELINE_GUIDANCE_POLICY_PATH}")

    (
        class_policy,
        guidance_policy,
        grid_config,
        planar_field_config,
    ) = oracle.create_oracle_configuration()
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
    detection_config = integration.create_detection_config(
        floor_height_robot
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
    print("\nWaiting for the shared fixed-field LiDAR snapshot...")
    points_sensor_frame, frame_info, _ = stationary.wait_for_first_scan(
        lidar,
        robot_controller,
    )

    print("Building the chassis-origin planar Safe-SAGE field...")
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
            "The planar oracle did not provide an accepted physical "
            f"reference: {planar_result.termination_reason}"
        )
    tracking_path_states = add_lateral_tracking_tube_allowance(
        planar_result.virtual_states,
        preferred_lateral_direction_grid_y=direction_grid_y,
        maximum_allowance=PATH_LATERAL_TRACKING_TUBE_ALLOWANCE,
    )
    tracking_path_states = append_terminal_path_overrun(
        tracking_path_states,
        overrun_distance=PATH_TERMINAL_FORWARD_OVERRUN,
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
    unicycle_field_config = stationary.create_field_pipeline_config(
        lookahead_footprint
    )
    print("Building the look-ahead-referenced unicycle Safe-SAGE field...")
    unicycle_frame = motion.build_motion_pipeline_frame(
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
        field_config=unicycle_field_config,
        semantic_class=oracle.PAPER_BASELINE_SEMANTIC_CLASS,
    )
    mpc = create_motion_mpc(
        robot_controller,
        paper_filter=paper_filter,
    )
    backup_filter = integration.create_analytical_filter(
        robot_controller,
        paper_filter=paper_filter,
    )

    oracle.create_path_markers(
        loaded_scene.stage,
        tracking_path_states,
        anchor_pose=anchor_pose,
        root_path=REFERENCE_MARKER_ROOT,
        color=REFERENCE_MARKER_COLOR,
        highlight_final=False,
    )
    result = run_motion_trial(
        stage=loaded_scene.stage,
        robot_controller=robot_controller,
        planar_path_states=tracking_path_states,
        pipeline_frame=unicycle_frame,
        mpc=mpc,
        backup_filter=backup_filter,
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
    print_motion_result(result)

    robot_controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()

    print("\nExpected successful scene behavior:")
    print(
        "  - Cyan markers show the planar path plus the reported "
        "tracking-tube allowance."
    )
    print("  - Nova Carter turns toward robot-right and follows that path.")
    print("  - The blue proxy remains fixed and is not touched.")
    print("  - Green markers show the physically executed chassis path.")
    print("  - The robot stops after clearing the proxy longitudinally.")
    print("  - No later LiDAR frame changes the fixed field in this stage.")

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
        print("ERROR DURING PLANAR-TO-UNICYCLE PHYSICAL MOTION DEMO")
        print("=" * 78)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
