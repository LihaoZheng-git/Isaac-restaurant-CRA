"""Diagnose Safe-SAGE in its ideal planar reduced-order command space.

This controlled oracle separates the semantic safety field from Nova
Carter's differential-drive realization.  It reuses the restaurant scene,
real PandarXT-32 point cloud, proxy-track association, semantic policy, and
yaw-dependent ``h_full`` pipeline already used by the unicycle experiments.

Stage A - fixed real-perception field:

1. Position Nova Carter in the spacious restaurant test area.
2. Create and observe the session-only blue semantic proxy.
3. Associate one real LiDAR track with the proxy and label it ``person``.
4. Build one fixed Safe-SAGE field about the measured chassis footprint.

Stage B - virtual planar closed loop:

1. Keep Nova Carter physically stationary.
2. Initialize a virtual state ``(x, y, yaw) = (0, 0, 0)`` in the field frame.
3. Repeatedly solve planar MPC for ``(v_x, v_y, omega)`` using a constant
   operator-style nominal command toward positive X.
4. Pass the first MPC command through the paper-space analytical filter.
5. Integrate only the virtual planar state; no command reaches the wheels.
6. Stop when the virtual robot clears the proxy or fails closed.

White markers show the unsafe nominal centerline. Green markers show the
executed virtual Safe-SAGE path, and the final virtual state is magenta.

This is a diagnostic control-space oracle, not a physically executable
controller for Nova Carter.

Run from the project root:

    python -u scripts/demos/safe_sage_planar_oracle_demo.py

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


# Importing the validated motion demo initializes the one required
# SimulationApp. Its guarded main() function is not executed.
from scripts.demos import safe_sage_h_full_motion_demo as motion

stationary = motion.stationary
simulation_app = motion.simulation_app


import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Usd, UsdGeom
from robot.control.differential_drive import NovaCarterController
from robot.control.planar_single_integrator import integrate_planar_euler
from robot.navigation.waypoint_controller import RobotPose2D, wrap_to_pi
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
    SafeSAGECBFResult,
    SafeSAGECBFSafetyFilter,
)
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
)
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidancePolicy
from robot.safety.safe_sage_planar_mpc import (
    SafeSAGEPlanarMPC,
    SafeSAGEPlanarMPCConfig,
    SafeSAGEPlanarMPCResult,
)
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import compute_relative_prim_transform
from robot.sensor.lidar import NovaCarterLidar

# ---------------------------------------------------------------------------
# Oracle configuration
# ---------------------------------------------------------------------------


ORACLE_CONTROL_INTERVAL_SECONDS = 0.20
ORACLE_HORIZON_STEPS = 30
ORACLE_MAXIMUM_SECONDS = 20.0
ORACLE_MAXIMUM_CONTROL_CYCLES = math.ceil(
    ORACLE_MAXIMUM_SECONDS / ORACLE_CONTROL_INTERVAL_SECONDS
)
ORACLE_ANALYTICAL_SUBSTEPS = max(
    1,
    round(ORACLE_CONTROL_INTERVAL_SECONDS / stationary.PHYSICS_DT),
)
ORACLE_ANALYTICAL_TIME_STEP = (
    ORACLE_CONTROL_INTERVAL_SECONDS / ORACLE_ANALYTICAL_SUBSTEPS
)

NOMINAL_PLANAR_COMMAND = np.array([0.20, 0.0, 0.0], dtype=np.float64)
MINIMUM_RIGHTWARD_RESPONSE = 0.05
SETTLED_PLANAR_SPEED = 0.01
SETTLED_ANGULAR_SPEED = 0.02
SETTLED_CONFIRMATION_CYCLES = 8

MPC_PLANAR_VELOCITY_WEIGHT = 1.0
MPC_ANGULAR_VELOCITY_WEIGHT = 0.02
MPC_COMMAND_SMOOTHNESS_WEIGHT = 0.005
MPC_LATERAL_SEED_SPEED = 0.15
MPC_MAXIMUM_SOLVER_ITERATIONS = 500

NOMINAL_MARKER_ROOT = "/World/SafeSAGEPlanarOracle/NominalPath"
SAFE_MARKER_ROOT = "/World/SafeSAGEPlanarOracle/SafePath"
MARKER_RADIUS = 0.035
MARKER_HEIGHT_WORLD = 0.08
NOMINAL_MARKER_COLOR = (0.95, 0.95, 0.95)
SAFE_MARKER_COLOR = (0.10, 0.95, 0.35)
FINAL_MARKER_COLOR = (1.00, 0.20, 0.80)


@dataclass(frozen=True)
class PlanarOracleResult:
    """Virtual closed-loop outcome and controlled diagnostic checks."""

    termination_reason: str
    completed_control_cycles: int
    virtual_states: np.ndarray
    final_mpc_result: SafeSAGEPlanarMPCResult | None
    final_backup_result: SafeSAGECBFResult | None
    all_predictive_constraints_valid: bool
    all_backup_constraints_valid: bool
    mpc_fallback_seen: bool
    backup_failure_seen: bool
    proxy_cleared: bool
    rightward_response_seen: bool
    maximum_rightward_displacement: float
    forward_displacement: float
    minimum_policy_clearance: float
    minimum_predicted_safety: float
    minimum_predictive_margin: float
    minimum_backup_safety: float
    minimum_backup_margin: float
    mean_mpc_solve_wall_seconds: float
    maximum_mpc_solve_wall_seconds: float


def predictive_result_is_valid(
    result: SafeSAGEPlanarMPCResult,
    *,
    constraint_tolerance: float,
) -> bool:
    """Return whether one planar prediction passed exact acceptance."""

    arrays = (
        result.command_sequence,
        result.predicted_states,
        result.predicted_safety_values,
        result.dtcbf_margins,
    )
    return bool(
        result.success
        and not result.fallback_active
        and all(np.isfinite(array).all() for array in arrays)
        and result.minimum_predicted_safety_value >= -constraint_tolerance
        and result.minimum_dtcbf_margin >= -constraint_tolerance
    )


def backup_result_is_valid(result: SafeSAGECBFResult) -> bool:
    """Return whether the paper-space analytical filter accepted a command."""

    safe_command = np.array(
        [
            result.safe_planar_velocity_grid_xy[0],
            result.safe_planar_velocity_grid_xy[1],
            result.safe_angular_velocity,
        ],
        dtype=np.float64,
    )
    return bool(
        result.constraint_feasible
        and result.constraint_satisfied
        and not result.emergency_stop_active
        and np.isfinite(safe_command).all()
        and math.isfinite(result.sampled_safety_value)
        and math.isfinite(result.cbf_margin)
    )


def field_state_to_world_pose(
    state_grid: np.ndarray,
    *,
    anchor_pose: RobotPose2D,
) -> RobotPose2D:
    """Transform one virtual field state into a world-frame pose."""

    cosine = math.cos(anchor_pose.yaw)
    sine = math.sin(anchor_pose.yaw)
    return RobotPose2D(
        x=(
            anchor_pose.x
            + cosine * float(state_grid[0])
            - sine * float(state_grid[1])
        ),
        y=(
            anchor_pose.y
            + sine * float(state_grid[0])
            + cosine * float(state_grid[1])
        ),
        yaw=wrap_to_pi(anchor_pose.yaw + float(state_grid[2])),
    )


def create_path_markers(
    stage: Usd.Stage,
    states_grid: np.ndarray,
    *,
    anchor_pose: RobotPose2D,
    root_path: str,
    color: tuple[float, float, float],
    highlight_final: bool,
) -> None:
    """Draw one non-colliding virtual path in the session layer."""

    existing = stage.GetPrimAtPath(root_path)
    if existing.IsValid():
        raise RuntimeError(f"Path-marker root already exists: {root_path}")

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage, root_path)
        final_index = states_grid.shape[0] - 1
        for index, state in enumerate(states_grid):
            marker_path = f"{root_path}/Point_{index:03d}"
            sphere = UsdGeom.Sphere.Define(stage, marker_path)
            sphere.CreateRadiusAttr(MARKER_RADIUS)
            pose = field_state_to_world_pose(
                state,
                anchor_pose=anchor_pose,
            )
            sphere.AddTranslateOp().Set(
                Gf.Vec3d(pose.x, pose.y, MARKER_HEIGHT_WORLD)
            )
            marker_color = (
                FINAL_MARKER_COLOR
                if highlight_final and index == final_index
                else color
            )
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*marker_color)])


def nominal_centerline_states(
    *,
    forward_threshold: float,
) -> np.ndarray:
    """Return a marker-only straight nominal path through the proxy."""

    final_x = max(
        forward_threshold,
        NOMINAL_PLANAR_COMMAND[0] * ORACLE_MAXIMUM_SECONDS,
    )
    sample_count = max(2, math.ceil(final_x / 0.10) + 1)
    states = np.zeros((sample_count, 3), dtype=np.float64)
    states[:, 0] = np.linspace(0.0, final_x, sample_count)
    return states


def create_oracle_configuration() -> tuple[
    ClassSafetyPolicy,
    SafeSAGEGuidancePolicy,
    LocalOccupancyGridConfig,
    SafeSAGEFieldPipelineConfig,
]:
    """Create the unshifted paper-space field configuration."""

    guidance_policy = SafeSAGEGuidancePolicy.from_json(
        stationary.GUIDANCE_POLICY_PATH
    )
    class_policy = stationary.class_policy_from_guidance_policy(
        guidance_policy
    )
    measured_footprint = RectangularRobotFootprint(
        front_extent=stationary.ROBOT_FRONT_EXTENT,
        rear_extent=stationary.ROBOT_REAR_EXTENT,
        left_extent=stationary.ROBOT_LEFT_EXTENT,
        right_extent=stationary.ROBOT_RIGHT_EXTENT,
        safety_margin=stationary.ROBOT_SAFETY_MARGIN,
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
        stationary.create_field_pipeline_config(measured_footprint),
    )


def create_controller_stack(
    robot_controller: NovaCarterController,
) -> tuple[SafeSAGEPlanarMPC, SafeSAGECBFSafetyFilter]:
    """Create the ideal planar MPC and paper-space analytical backup."""

    paper_filter = SafeSAGECBFSafetyFilter(
        SafeSAGECBFConfig(
            gamma=stationary.CBF_GAMMA,
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
            planar_velocity_weight=(
                stationary.CBF_LINEAR_VELOCITY_WEIGHT
            ),
            angular_velocity_weight=(
                stationary.CBF_ANGULAR_VELOCITY_WEIGHT
            ),
        )
    )
    planar_mpc = SafeSAGEPlanarMPC(
        paper_filter,
        SafeSAGEPlanarMPCConfig(
            horizon_steps=ORACLE_HORIZON_STEPS,
            time_step=ORACLE_CONTROL_INTERVAL_SECONDS,
            gamma=stationary.CBF_GAMMA,
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
            planar_velocity_weight=MPC_PLANAR_VELOCITY_WEIGHT,
            angular_velocity_weight=MPC_ANGULAR_VELOCITY_WEIGHT,
            command_smoothness_weight=(
                MPC_COMMAND_SMOOTHNESS_WEIGHT
            ),
            lateral_seed_speed=MPC_LATERAL_SEED_SPEED,
            maximum_solver_iterations=MPC_MAXIMUM_SOLVER_ITERATIONS,
        ),
    )
    return planar_mpc, paper_filter


def run_planar_oracle(
    *,
    pipeline_frame: motion.MotionPipelineFrame,
    planar_mpc: SafeSAGEPlanarMPC,
    paper_filter: SafeSAGECBFSafetyFilter,
) -> PlanarOracleResult:
    """Run the virtual receding-horizon planar safety experiment."""

    state = np.zeros(3, dtype=np.float64)
    state_history = [state.copy()]
    warm_start: np.ndarray | None = None
    final_mpc_result: SafeSAGEPlanarMPCResult | None = None
    final_backup_result: SafeSAGECBFResult | None = None
    all_predictive_valid = True
    all_backup_valid = True
    mpc_fallback_seen = False
    backup_failure_seen = False
    proxy_cleared = False
    settled_cycles = 0
    minimum_clearance = math.inf
    minimum_predicted_safety = math.inf
    minimum_predictive_margin = math.inf
    minimum_backup_safety = math.inf
    minimum_backup_margin = math.inf
    maximum_rightward = 0.0
    solve_times: list[float] = []
    termination_reason = "Maximum virtual duration reached."
    completed_cycles = 0

    forward_threshold = motion.proxy_clearance_forward_threshold(
        pipeline_frame.anchor_pose_world
    )
    print("\n" + "=" * 78)
    print("SAFE-SAGE PLANAR ORACLE - VIRTUAL CLOSED LOOP")
    print("=" * 78)
    print("\nControlled settings:")
    print("  Physical wheel commands:       disabled")
    print("  Virtual state:                 (x, y, yaw)")
    print("  Virtual command:               (v_x, v_y, omega)")
    print("  Field reference point:         chassis origin")
    print("  Nominal command:               (0.200, 0.000, 0.000)")
    print(
        "  MPC horizon:                   "
        f"{ORACLE_HORIZON_STEPS} steps"
    )
    print(
        "  MPC interval:                  "
        f"{ORACLE_CONTROL_INTERVAL_SECONDS:.3f} s"
    )
    print(
        "  Analytical-filter interval:    "
        f"{ORACLE_ANALYTICAL_TIME_STEP:.4f} s"
    )
    print(
        "  Proxy-clear forward threshold: "
        f"{forward_threshold:.3f} m"
    )
    print("\nLive virtual status:")

    for cycle in range(ORACLE_MAXIMUM_CONTROL_CYCLES):
        solve_start = time.perf_counter()
        mpc_result = planar_mpc.plan(
            state,
            NOMINAL_PLANAR_COMMAND,
            initial_command_guess=warm_start,
            configuration_space=(
                pipeline_frame.pipeline.configuration_space
            ),
            h_full=pipeline_frame.pipeline.h_full,
            guidance_fields=pipeline_frame.pipeline.guidance_fields,
        )
        solve_seconds = time.perf_counter() - solve_start
        solve_times.append(solve_seconds)
        final_mpc_result = mpc_result
        completed_cycles = cycle + 1

        predictive_valid = predictive_result_is_valid(
            mpc_result,
            constraint_tolerance=planar_mpc.config.constraint_tolerance,
        )
        all_predictive_valid &= predictive_valid
        mpc_fallback_seen |= mpc_result.fallback_active
        if np.isfinite(mpc_result.predicted_safety_values).all():
            minimum_predicted_safety = min(
                minimum_predicted_safety,
                mpc_result.minimum_predicted_safety_value,
            )
        if np.isfinite(mpc_result.dtcbf_margins).all():
            minimum_predictive_margin = min(
                minimum_predictive_margin,
                mpc_result.minimum_dtcbf_margin,
            )
        if not predictive_valid:
            termination_reason = (
                "Fail-closed planar MPC stop: " + mpc_result.status
            )
            break

        mpc_command = np.asarray(
            mpc_result.first_command,
            dtype=np.float64,
        )
        cycle_backup_valid = True
        last_output = np.zeros(3, dtype=np.float64)
        for _ in range(ORACLE_ANALYTICAL_SUBSTEPS):
            backup_result = paper_filter.filter_command(
                mpc_command[:2],
                float(mpc_command[2]),
                query_position_grid_xy=state[:2],
                query_yaw_radians=float(state[2]),
                configuration_space=(
                    pipeline_frame.pipeline.configuration_space
                ),
                h_full=pipeline_frame.pipeline.h_full,
                guidance_fields=pipeline_frame.pipeline.guidance_fields,
            )
            final_backup_result = backup_result
            backup_valid = backup_result_is_valid(backup_result)
            cycle_backup_valid &= backup_valid
            all_backup_valid &= backup_valid
            backup_failure_seen |= not backup_valid
            minimum_backup_safety = min(
                minimum_backup_safety,
                backup_result.sampled_safety_value,
            )
            minimum_backup_margin = min(
                minimum_backup_margin,
                backup_result.cbf_margin,
            )
            if not backup_valid:
                termination_reason = (
                    "Fail-closed analytical-filter stop: "
                    f"{backup_result.emergency_reason or 'invalid constraint'}"
                )
                break
            last_output = np.array(
                [
                    backup_result.safe_planar_velocity_grid_xy[0],
                    backup_result.safe_planar_velocity_grid_xy[1],
                    backup_result.safe_angular_velocity,
                ],
                dtype=np.float64,
            )
            state = integrate_planar_euler(
                state,
                last_output,
                time_step=ORACLE_ANALYTICAL_TIME_STEP,
            )
        if not cycle_backup_valid:
            break

        state_history.append(state.copy())
        virtual_pose = field_state_to_world_pose(
            state,
            anchor_pose=pipeline_frame.anchor_pose_world,
        )
        clearance = motion.policy_footprint_clearance(virtual_pose)
        minimum_clearance = min(minimum_clearance, clearance)
        forward, rightward = motion.forward_and_rightward_displacement(
            virtual_pose,
            initial_pose=pipeline_frame.anchor_pose_world,
        )
        maximum_rightward = max(maximum_rightward, rightward)
        proxy_cleared = forward >= forward_threshold

        predicted_y = mpc_result.predicted_states[:, 1]
        print(
            f"  cycle={cycle:<3} "
            f"q=({state[0]:.3f},{state[1]:.3f},"
            f"{math.degrees(state[2]):.1f}deg) "
            f"forward={forward:.3f} right={rightward:.3f} "
            f"clear={clearance:.3f} "
            f"mpc=({mpc_command[0]:.3f},{mpc_command[1]:.3f},"
            f"{mpc_command[2]:.3f}) "
            f"out=({last_output[0]:.3f},{last_output[1]:.3f},"
            f"{last_output[2]:.3f}) "
            f"y[min,max,end]=({float(np.min(predicted_y)):.3f},"
            f"{float(np.max(predicted_y)):.3f},"
            f"{float(predicted_y[-1]):.3f}) "
            f"hmin={mpc_result.minimum_predicted_safety_value:.3f} "
            f"margin={mpc_result.minimum_dtcbf_margin:.2e} "
            f"solve={solve_seconds:.3f}s "
            f"status={mpc_result.status}"
        )

        if clearance <= motion.GROUND_TRUTH_CLEARANCE_ABORT:
            termination_reason = (
                "Independent virtual clearance guard was crossed."
            )
            break
        if proxy_cleared:
            termination_reason = "The virtual planar state cleared the proxy."
            break

        if (
            float(np.linalg.norm(last_output[:2])) <= SETTLED_PLANAR_SPEED
            and abs(float(last_output[2])) <= SETTLED_ANGULAR_SPEED
        ):
            settled_cycles += 1
        else:
            settled_cycles = 0
        if settled_cycles >= SETTLED_CONFIRMATION_CYCLES:
            termination_reason = (
                "The safe planar command settled before proxy clearance."
            )
            break

        warm_start = np.vstack(
            (
                mpc_result.command_sequence[1:],
                mpc_result.command_sequence[-1:],
            )
        )

    final_pose = field_state_to_world_pose(
        state,
        anchor_pose=pipeline_frame.anchor_pose_world,
    )
    forward_displacement, _ = motion.forward_and_rightward_displacement(
        final_pose,
        initial_pose=pipeline_frame.anchor_pose_world,
    )
    rightward_response_seen = (
        maximum_rightward >= MINIMUM_RIGHTWARD_RESPONSE
    )
    return PlanarOracleResult(
        termination_reason=termination_reason,
        completed_control_cycles=completed_cycles,
        virtual_states=np.ascontiguousarray(
            np.asarray(state_history, dtype=np.float64)
        ),
        final_mpc_result=final_mpc_result,
        final_backup_result=final_backup_result,
        all_predictive_constraints_valid=all_predictive_valid,
        all_backup_constraints_valid=all_backup_valid,
        mpc_fallback_seen=mpc_fallback_seen,
        backup_failure_seen=backup_failure_seen,
        proxy_cleared=proxy_cleared,
        rightward_response_seen=rightward_response_seen,
        maximum_rightward_displacement=maximum_rightward,
        forward_displacement=forward_displacement,
        minimum_policy_clearance=minimum_clearance,
        minimum_predicted_safety=minimum_predicted_safety,
        minimum_predictive_margin=minimum_predictive_margin,
        minimum_backup_safety=minimum_backup_safety,
        minimum_backup_margin=minimum_backup_margin,
        mean_mpc_solve_wall_seconds=(
            float(np.mean(solve_times)) if solve_times else math.nan
        ),
        maximum_mpc_solve_wall_seconds=(
            float(np.max(solve_times)) if solve_times else math.nan
        ),
    )


def print_oracle_result(
    result: PlanarOracleResult,
    *,
    physical_initial_pose: RobotPose2D,
    physical_final_pose: RobotPose2D,
) -> None:
    """Print the diagnostic verdict and physical-stationarity check."""

    physical_position_drift = math.hypot(
        physical_final_pose.x - physical_initial_pose.x,
        physical_final_pose.y - physical_initial_pose.y,
    )
    physical_yaw_drift = stationary.absolute_wrapped_angle_difference(
        physical_final_pose.yaw,
        physical_initial_pose.yaw,
    )
    physical_robot_stationary = bool(
        physical_position_drift
        <= stationary.MAXIMUM_STATIONARY_POSITION_DRIFT
        and physical_yaw_drift
        <= stationary.MAXIMUM_STATIONARY_YAW_DRIFT_RADIANS
    )
    accepted = bool(
        result.proxy_cleared
        and result.rightward_response_seen
        and result.all_predictive_constraints_valid
        and result.all_backup_constraints_valid
        and not result.mpc_fallback_seen
        and not result.backup_failure_seen
        and result.minimum_policy_clearance
        > motion.GROUND_TRUTH_CLEARANCE_ABORT
        and physical_robot_stationary
    )

    print("\n" + "=" * 78)
    print("SAFE-SAGE PLANAR ORACLE RESULT")
    print("=" * 78)
    print(f"Termination:                       {result.termination_reason}")
    print(
        f"Completed virtual control cycles:  "
        f"{result.completed_control_cycles}"
    )
    print("\nVirtual motion:")
    print(
        f"  Forward displacement:             "
        f"{result.forward_displacement:.4f} m"
    )
    print(
        f"  Maximum rightward displacement:   "
        f"{result.maximum_rightward_displacement:.4f} m"
    )
    print(
        f"  Minimum policy clearance:         "
        f"{result.minimum_policy_clearance:.4f} m"
    )
    print("\nSafety layers:")
    print(
        f"  All predictive constraints valid: "
        f"{result.all_predictive_constraints_valid}"
    )
    print(
        f"  All analytical constraints valid: "
        f"{result.all_backup_constraints_valid}"
    )
    print(
        f"  MPC fallback observed:            "
        f"{result.mpc_fallback_seen}"
    )
    print(
        f"  Analytical failure observed:      "
        f"{result.backup_failure_seen}"
    )
    print(
        f"  Minimum predicted h_full:         "
        f"{result.minimum_predicted_safety:.6f}"
    )
    print(
        f"  Minimum predictive margin:        "
        f"{result.minimum_predictive_margin:.3e}"
    )
    print(
        f"  Minimum analytical h_full:        "
        f"{result.minimum_backup_safety:.6f}"
    )
    print(
        f"  Minimum analytical CBF margin:    "
        f"{result.minimum_backup_margin:.3e}"
    )
    print("\nRuntime diagnostics:")
    print(
        f"  Mean planar MPC solve time:        "
        f"{result.mean_mpc_solve_wall_seconds:.3f} s"
    )
    print(
        f"  Maximum planar MPC solve time:     "
        f"{result.maximum_mpc_solve_wall_seconds:.3f} s"
    )
    print("\nControlled checks:")
    print(f"  Proxy cleared:                     {result.proxy_cleared}")
    print(
        f"  Rightward response observed:       "
        f"{result.rightward_response_seen}"
    )
    print(
        f"  Physical robot remained stationary:{physical_robot_stationary}"
    )
    print(
        f"  Physical position drift:           "
        f"{physical_position_drift:.6f} m"
    )
    print(
        f"  Physical yaw drift:                "
        f"{math.degrees(physical_yaw_drift):.4f} degrees"
    )
    print("\nInterpretation:")
    if accepted:
        print(
            "  [PASS] The same real-perception Safe-SAGE pipeline produced "
            "a safe right-side bypass when direct planar velocity was "
            "available. This isolates the remaining difficulty to the "
            "differential-drive adaptation."
        )
    else:
        print(
            "  [WARNING] The planar oracle did not complete a safe bypass. "
            "The limitation is therefore not only Nova Carter's unicycle "
            "mapping; inspect field resolution, b/lambda balance, and the "
            "termination reason before changing the wheel controller."
        )
    print("=" * 78)


def main() -> None:
    """Build one real field and run the stationary planar oracle."""

    print("\n" + "=" * 78)
    print("SAFE-SAGE PLANAR CONTROL-SPACE ORACLE DEMO")
    print("=" * 78)
    print("Mode: real perception + virtual planar closed loop")
    print("Physical Nova Carter wheel commands: disabled")
    print(f"Guidance policy: {stationary.GUIDANCE_POLICY_PATH}")

    (
        class_policy,
        guidance_policy,
        grid_config,
        field_config,
    ) = create_oracle_configuration()

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

    print("\nPress Play in Isaac Sim to begin.")
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
            minimum_cluster_points=(
                stationary.TRACKER_MINIMUM_CLUSTER_POINTS
            ),
            maximum_association_distance=(
                stationary.TRACKER_MAXIMUM_ASSOCIATION_DISTANCE
            ),
        )
    )

    print(
        "\nHolding the physical robot and proxy still for "
        f"{stationary.SETTLE_STEPS} physics steps..."
    )
    if not advance_playing_steps(
        simulation_app,
        physics_steps=stationary.SETTLE_STEPS,
        on_step=robot_controller.stop,
    ):
        return

    anchor_pose = stationary.read_robot_pose(loaded_scene.stage)
    physical_initial_pose = anchor_pose
    print("\nWaiting for the fixed-field LiDAR snapshot...")
    points_sensor_frame, frame_info, _ = stationary.wait_for_first_scan(
        lidar,
        robot_controller,
    )
    print("Building the unshifted paper-space Safe-SAGE field...")
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

    print("\n" + "=" * 78)
    print("SAFE-SAGE PLANAR ORACLE - PERCEPTION AND FIELD")
    print("=" * 78)
    print(f"  LiDAR frame ID:                 {pipeline_frame.sensor_frame_id}")
    print(
        f"  All obstacle points:            "
        f"{pipeline_frame.all_obstacle_point_count}"
    )
    print(
        f"  Proxy track ID:                 "
        f"{pipeline_frame.proxy_track_id}"
    )
    print(
        f"  Proxy track points:             "
        f"{pipeline_frame.proxy_point_count}"
    )
    print(
        f"  Proxy association error:        "
        f"{pipeline_frame.proxy_association_error:.4f} m"
    )
    print(
        f"  Field build wall time:          "
        f"{pipeline_frame.build_wall_seconds:.3f} s"
    )
    print(
        f"  Grid shape:                     "
        f"{pipeline_frame.pipeline.configuration_space.occupancy_grids[0].shape}"
    )
    print("  Footprint reference:            chassis origin (no look-ahead)")
    print("  Physical robot remained still:  True")
    print("=" * 78)

    planar_mpc, paper_filter = create_controller_stack(robot_controller)
    result = run_planar_oracle(
        pipeline_frame=pipeline_frame,
        planar_mpc=planar_mpc,
        paper_filter=paper_filter,
    )

    create_path_markers(
        loaded_scene.stage,
        nominal_centerline_states(
            forward_threshold=motion.proxy_clearance_forward_threshold(
                anchor_pose
            )
        ),
        anchor_pose=anchor_pose,
        root_path=NOMINAL_MARKER_ROOT,
        color=NOMINAL_MARKER_COLOR,
        highlight_final=False,
    )
    create_path_markers(
        loaded_scene.stage,
        result.virtual_states,
        anchor_pose=anchor_pose,
        root_path=SAFE_MARKER_ROOT,
        color=SAFE_MARKER_COLOR,
        highlight_final=True,
    )

    robot_controller.stop()
    app_utils.update_app(steps=5)
    physical_final_pose = stationary.read_robot_pose(loaded_scene.stage)
    app_utils.stop()
    print_oracle_result(
        result,
        physical_initial_pose=physical_initial_pose,
        physical_final_pose=physical_final_pose,
    )

    print("\nExpected scene behavior:")
    print("  - The physical Nova Carter remains at its initial pose.")
    print("  - The blue semantic proxy remains fixed ahead of the robot.")
    print("  - White markers continue straight through the proxy.")
    print("  - Green markers should curve toward robot-right and pass it.")
    print("  - The final virtual point is shown in magenta.")
    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when finished.")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - demo must close SimulationApp
        print("\n" + "=" * 78)
        print("ERROR DURING SAFE-SAGE PLANAR ORACLE DEMO")
        print("=" * 78)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
