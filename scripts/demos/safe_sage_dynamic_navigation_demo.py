"""Run direct Safe-SAGE navigation with a moving LiDAR proxy.

This is the dynamic successor to ``safe_sage_live_navigation_demo.py``.
It preserves the same direct control architecture:

1. An obstacle-unaware waypoint controller points straight through the proxy.
2. Real LiDAR scans update tracking and the semantic ``h_full`` field.
3. Consecutive fields produce the paper's motion-compensated ``dh_full/dt``.
4. Direct unicycle Safe-SAGE MPC filters the nominal command horizon.
5. The analytical CBF rechecks the selected command every physics step.
6. Only that safe ``(v, omega)`` command reaches Nova Carter's wheels.

The controlled blue proxy moves from robot-right to robot-left across the
nominal route.  Its known pose is used only to assign the manual ``person``
label to the corresponding real LiDAR track and to compute an independent
ground-truth clearance guard.  It is never supplied as safety geometry.

Run from the project root:

    python -u scripts/demos/safe_sage_dynamic_navigation_demo.py

The demo starts playback automatically.  Set ``SAFE_SAGE_AUTOCLOSE=1`` to
close the rendered application after the report.  RTX LiDAR may not publish
scans in a true no-window run on every Isaac Sim installation.
"""

import math
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Importing the validated live demo creates the one required SimulationApp.
from scripts.demos import safe_sage_live_navigation_demo as live


fixed = live.fixed
motion = live.motion
stationary = live.stationary
simulation_app = live.simulation_app


import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Usd, UsdGeom

from robot.control.differential_drive import NovaCarterController
from robot.navigation.waypoint_controller import RobotPose2D
from robot.runtime.simulation_loop import advance_playing_steps, wait_until_playing
from robot.safety.obstacle_detection import ObstacleDetectionConfig
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig
from robot.safety.safe_sage_guidance_policy import PassingSide
from robot.safety.safe_sage_runtime import (
    SafeSAGEControlRuntime,
    SafeSAGEPerceptionRuntime,
    SafeSAGERuntimeField,
)
from robot.safety.safe_sage_temporal_derivative import (
    SafeSAGETemporalDerivativeConfig,
    SafeSAGETemporalDerivativeEstimator,
    SafeSAGETemporalDerivativeResult,
)
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPC
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import compute_relative_prim_transform
from robot.sensor.lidar import NovaCarterLidar


DYNAMIC_PROXY_PRIM_PATH = "/World/SafeSAGEDynamicTestObstacle"
DYNAMIC_PROXY_X_WORLD = 28.20
DYNAMIC_PROXY_START_Y_WORLD = 4.00
DYNAMIC_PROXY_END_Y_WORLD = 6.50
DYNAMIC_PROXY_SPEED_METERS_PER_SECOND = 0.60
DYNAMIC_PROXY_DIMENSIONS = stationary.PROXY_DIMENSIONS
DYNAMIC_PROXY_COLOR = stationary.PROXY_COLOR
DYNAMIC_GRID_MINIMUM_Y = -4.0
DYNAMIC_GRID_MAXIMUM_Y = 4.0
MINIMUM_TRACKED_PROXY_SPEED = 0.20
MINIMUM_TEMPORAL_MAGNITUDE = 1e-3
MINIMUM_DYNAMIC_FIELD_UPDATES = 5

TEMPORAL_CONFIG = SafeSAGETemporalDerivativeConfig(
    smoothing_factor=0.35,
    minimum_time_delta_seconds=0.05,
    maximum_time_delta_seconds=0.35,
)


@dataclass
class DynamicProxyMotion:
    """Deterministic session-layer crossing trajectory."""

    stage: Usd.Stage
    translate_op: UsdGeom.XformOp
    elapsed_seconds: float = 0.0

    @property
    def position_world(self) -> np.ndarray:
        distance = min(
            DYNAMIC_PROXY_SPEED_METERS_PER_SECOND * self.elapsed_seconds,
            DYNAMIC_PROXY_END_Y_WORLD - DYNAMIC_PROXY_START_Y_WORLD,
        )
        return np.array(
            [
                DYNAMIC_PROXY_X_WORLD,
                DYNAMIC_PROXY_START_Y_WORLD + distance,
                0.5 * DYNAMIC_PROXY_DIMENSIONS[2],
            ],
            dtype=np.float64,
        )

    @property
    def crossed_route(self) -> bool:
        return self.position_world[1] >= stationary.ROBOT_GOAL_WORLD[1]

    @property
    def finished(self) -> bool:
        return self.position_world[1] >= DYNAMIC_PROXY_END_Y_WORLD - 1e-9

    def advance(self, time_step: float) -> None:
        """Advance the deterministic motion and author its new pose."""

        self.elapsed_seconds += time_step
        position = self.position_world
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.translate_op.Set(
                Gf.Vec3d(
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                )
            )


@dataclass(frozen=True)
class DynamicNavigationResult:
    """Dynamic-field outcome and controlled diagnostics."""

    termination_reason: str
    completed_control_cycles: int
    completed_physics_steps: int
    initial_pose: RobotPose2D
    final_pose: RobotPose2D
    executed_chassis_states: np.ndarray
    initial_sensor_frame_id: int
    final_sensor_frame_id: int
    successful_field_updates: int
    temporal_field_updates: int
    reused_field_cycles: int
    field_update_failures: int
    distinct_proxy_track_ids: tuple[int, ...]
    maximum_tracked_proxy_speed: float
    minimum_sampled_temporal_derivative: float
    maximum_sampled_temporal_derivative: float
    maximum_field_temporal_magnitude: float
    minimum_temporal_valid_fraction: float
    minimum_policy_clearance: float
    maximum_rightward_displacement: float
    minimum_predicted_safety: float
    minimum_predicted_margin: float
    minimum_backup_safety: float
    minimum_backup_margin: float
    mean_field_build_wall_seconds: float
    mean_mpc_solve_wall_seconds: float
    cleared_proxy: bool
    proxy_crossed_route: bool
    proxy_finished: bool
    mpc_intervention_seen: bool
    all_predictive_constraints_valid: bool
    all_backup_constraints_valid: bool
    mpc_failure_seen: bool
    backup_failure_seen: bool
    perception_failure_seen: bool
    stale_field_stop_seen: bool
    ground_truth_abort_seen: bool
    application_stopped: bool


def create_dynamic_proxy(stage: Usd.Stage) -> DynamicProxyMotion:
    """Create the non-colliding blue proxy in the session layer."""

    if stage.GetPrimAtPath(DYNAMIC_PROXY_PRIM_PATH).IsValid():
        raise RuntimeError(
            f"Dynamic proxy already exists: {DYNAMIC_PROXY_PRIM_PATH}"
        )
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        cube = UsdGeom.Cube.Define(stage, DYNAMIC_PROXY_PRIM_PATH)
        cube.CreateSizeAttr(1.0)
        cube.CreateExtentAttr(
            [
                Gf.Vec3f(-0.5, -0.5, -0.5),
                Gf.Vec3f(0.5, 0.5, 0.5),
            ]
        )
        cube.CreateDisplayColorAttr([Gf.Vec3f(*DYNAMIC_PROXY_COLOR)])
        translate_op = cube.AddTranslateOp()
        translate_op.Set(
            Gf.Vec3d(
                DYNAMIC_PROXY_X_WORLD,
                DYNAMIC_PROXY_START_Y_WORLD,
                0.5 * DYNAMIC_PROXY_DIMENSIONS[2],
            )
        )
        cube.AddOrientOp().Set(Gf.Quatf(1.0))
        cube.AddScaleOp().Set(Gf.Vec3f(*DYNAMIC_PROXY_DIMENSIONS))
    return DynamicProxyMotion(stage=stage, translate_op=translate_op)


def dynamic_clearance(
    pose: RobotPose2D,
    proxy_motion: DynamicProxyMotion,
) -> float:
    """Return independent policy-footprint clearance."""

    return motion.policy_footprint_clearance(
        pose,
        proxy_motion.position_world[:2],
    )


def temporal_diagnostics(
    field: SafeSAGERuntimeField,
) -> tuple[float, float, float, float]:
    """Return sampled-field extrema and validity diagnostics."""

    temporal = field.temporal_safety_derivative
    if not isinstance(temporal, SafeSAGETemporalDerivativeResult):
        return 0.0, 0.0, 0.0, 1.0
    return (
        float(np.min(temporal.values)),
        float(np.max(temporal.values)),
        temporal.maximum_absolute_derivative,
        temporal.valid_fraction,
    )


def run_dynamic_navigation(
    *,
    stage: Usd.Stage,
    proxy_motion: DynamicProxyMotion,
    lidar: NovaCarterLidar,
    robot_controller: NovaCarterController,
    perception_runtime: SafeSAGEPerceptionRuntime,
    control_runtime: SafeSAGEControlRuntime,
    robot_from_lidar: np.ndarray,
    detection_config: ObstacleDetectionConfig,
    passing_side: PassingSide,
) -> DynamicNavigationResult:
    """Run direct dynamic-field MPC and the full-rate analytical CBF."""

    initial_pose = stationary.read_robot_pose(stage)
    initial_field = perception_runtime.field
    anchor_pose = RobotPose2D(
        x=float(initial_field.anchor_position_world_xy[0]),
        y=float(initial_field.anchor_position_world_xy[1]),
        yaw=initial_field.anchor_yaw,
    )
    initial_frame_id = initial_field.sensor_frame_id
    mpc = control_runtime.mpc
    waypoint_controller = fixed.create_motion_waypoint_controller()
    clearance_threshold = motion.proxy_clearance_forward_threshold(
        initial_pose,
        np.array(
            [DYNAMIC_PROXY_X_WORLD, stationary.ROBOT_GOAL_WORLD[1]],
            dtype=np.float64,
        ),
    )
    preferred_direction = -1.0 if passing_side is PassingSide.RIGHT else 1.0

    executed_states = [np.zeros(3, dtype=np.float64)]
    warm_start: np.ndarray | None = None
    completed_cycles = 0
    completed_physics_steps = 0
    successful_updates = 0
    temporal_updates = 0
    reused_cycles = 0
    update_failures = 0
    proxy_track_ids = {initial_field.primary_track_id}
    maximum_track_speed = 0.0
    minimum_sampled_temporal = math.inf
    maximum_sampled_temporal = -math.inf
    maximum_field_temporal = 0.0
    minimum_valid_fraction = 1.0
    minimum_clearance = dynamic_clearance(initial_pose, proxy_motion)
    maximum_rightward = 0.0
    minimum_predicted_safety = math.inf
    minimum_predicted_margin = math.inf
    minimum_backup_safety = math.inf
    minimum_backup_margin = math.inf
    field_build_times = [initial_field.build_wall_seconds]
    solve_times: list[float] = []

    cleared_proxy = False
    mpc_intervention_seen = False
    all_predictive_valid = True
    all_backup_valid = True
    mpc_failure_seen = False
    backup_failure_seen = False
    perception_failure_seen = False
    stale_field_stop_seen = False
    ground_truth_abort_seen = False
    application_stopped = False
    termination_reason = "Maximum dynamic-navigation duration reached."

    print("\n" + "=" * 78)
    print("SAFE-SAGE STAGE E - DYNAMIC TEMPORAL CLOSED-LOOP MOTION")
    print("=" * 78)
    print("  Nominal policy:                 obstacle-unaware waypoint feedback")
    print("  Precomputed bypass reference:   none")
    print("  Temporal method:                motion-compensated field difference")
    print("  Field frame:                    fixed first-scan frame")
    print("  Ego-motion grid shift:          (0.000, 0.000) m")
    print(
        "  Temporal low-pass alpha:        "
        f"{perception_runtime.temporal_estimator.config.smoothing_factor:.3f}"
    )
    print(f"  Proxy speed:                    {DYNAMIC_PROXY_SPEED_METERS_PER_SECOND:.3f} m/s")
    print(f"  Proxy-clear threshold:          {clearance_threshold:.3f} m")
    print("  Runtime:                        diagnostic only")
    print("\nLive dynamic status:")

    for cycle in range(fixed.MAXIMUM_CONTROL_CYCLES):
        if not simulation_app.is_running():
            application_stopped = True
            termination_reason = "Isaac Sim stopped during dynamic navigation."
            break
        while simulation_app.is_running() and not app_utils.is_playing():
            robot_controller.stop()
            simulation_app.update()
        if not simulation_app.is_running():
            application_stopped = True
            termination_reason = "Isaac Sim stopped during dynamic navigation."
            break

        robot_controller.stop()
        robot_pose = stationary.read_robot_pose(stage)
        try:
            new_frame = live.newest_lidar_frame(
                lidar,
                previous_frame_id=(
                    perception_runtime.field_buffer.snapshot.sensor_frame_id
                ),
            )
        except (RuntimeError, TypeError, ValueError) as error:
            perception_failure_seen = True
            update_failures += 1
            termination_reason = f"Fail-closed LiDAR read stop: {error}"
            break

        if new_frame is None:
            reused_cycles += 1
            try:
                field_status = perception_runtime.reuse(
                    current_simulation_time_seconds=live.simulation_time_seconds(),
                )
            except (RuntimeError, TypeError, ValueError) as error:
                stale_field_stop_seen = True
                termination_reason = f"Fail-closed field-clock stop: {error}"
                break
        else:
            points_sensor_frame, frame_info = new_frame
            try:
                observation = live.make_perception_observation(
                    points_sensor_frame=points_sensor_frame,
                    frame_info=frame_info,
                    robot_pose=robot_pose,
                    robot_from_lidar=robot_from_lidar,
                    detection_config=detection_config,
                )
                field_status = perception_runtime.update(
                    observation,
                    live.controlled_proxy_label_provider(
                        proxy_motion.position_world[:2]
                    )
                )
            except (RuntimeError, TypeError, ValueError) as error:
                perception_failure_seen = True
                update_failures += 1
                termination_reason = (
                    "Fail-closed perception/temporal update stop: " f"{error}"
                )
                break
            candidate = field_status.snapshot.payload
            successful_updates += 1
            temporal_updates += 1
            field_build_times.append(candidate.build_wall_seconds)
            proxy_track_ids.add(candidate.primary_track_id)

        if field_status.fail_closed:
            stale_field_stop_seen = True
            termination_reason = (
                "Fail-closed stale-field stop: "
                f"{field_status.failure_reason}"
            )
            break

        field = field_status.snapshot.payload
        track_speed = field.maximum_selected_track_speed
        maximum_track_speed = max(maximum_track_speed, track_speed)
        temporal_minimum, temporal_maximum, temporal_magnitude, valid_fraction = (
            temporal_diagnostics(field)
        )
        maximum_field_temporal = max(
            maximum_field_temporal,
            temporal_magnitude,
        )
        minimum_valid_fraction = min(minimum_valid_fraction, valid_fraction)

        nominal_command, nominal_commands = live.nominal_sequence(
            waypoint_controller,
            robot_pose,
            mpc=mpc,
        )
        if nominal_command.route_complete:
            termination_reason = "The nominal waypoint route completed."
            break
        chassis_state = live.chassis_state_in_field(
            robot_pose,
            anchor_pose=anchor_pose,
        )
        solve_start = time.perf_counter()
        mpc_result = control_runtime.plan(
            chassis_state,
            nominal_commands,
            field,
            initial_command_guess=warm_start,
            preferred_lateral_direction_grid_y=preferred_direction,
        )
        solve_time = time.perf_counter() - solve_start
        solve_times.append(solve_time)
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
            mpc_result.minimum_predicted_safety_value,
        )
        minimum_predicted_margin = min(
            minimum_predicted_margin,
            mpc_result.minimum_dtcbf_margin,
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

        minimum_sampled_temporal = min(
            minimum_sampled_temporal,
            backup_result.temporal_safety_derivative,
        )
        maximum_sampled_temporal = max(
            maximum_sampled_temporal,
            backup_result.temporal_safety_derivative,
        )
        minimum_backup_safety = min(
            minimum_backup_safety,
            backup_result.sampled_safety_value,
        )
        minimum_backup_margin = min(
            minimum_backup_margin,
            backup_result.cbf_margin,
        )
        current_clearance = dynamic_clearance(robot_pose, proxy_motion)
        forward, rightward = motion.forward_and_rightward_displacement(
            robot_pose,
            initial_pose=initial_pose,
        )
        minimum_clearance = min(minimum_clearance, current_clearance)
        maximum_rightward = max(maximum_rightward, rightward)
        print(
            f"  cycle={cycle:<3} scan={field.sensor_frame_id:<4} "
            f"proxy_y={proxy_motion.position_world[1]:.3f} "
            f"track_v={track_speed:.3f} "
            f"q=({chassis_state[0]:.3f},{chassis_state[1]:.3f},"
            f"{math.degrees(chassis_state[2]):.1f}deg) "
            f"clear={current_clearance:.3f} "
            f"dht={backup_result.temporal_safety_derivative:+.3e} "
            f"field=[{temporal_minimum:+.2e},{temporal_maximum:+.2e}] "
            f"nom=({nominal_command.linear_velocity:.3f},"
            f"{nominal_command.angular_velocity:.3f}) "
            f"out=({backup_result.safe_linear_velocity:.3f},"
            f"{backup_result.safe_angular_velocity:.3f}) "
            f"margin={mpc_result.minimum_dtcbf_margin:.2e} "
            f"solve={solve_time:.3f}s status={mpc_result.status}"
        )

        if current_clearance <= fixed.GROUND_TRUTH_CLEARANCE_ABORT:
            ground_truth_abort_seen = True
            termination_reason = "Independent ground-truth clearance guard triggered."
            break
        if forward >= clearance_threshold:
            cleared_proxy = True
            termination_reason = "The robot cleared the crossing plane."
            break

        execution_steps = perception_runtime.fresh_execution_step_budget(
            current_simulation_time_seconds=live.simulation_time_seconds(),
            physics_time_step_seconds=stationary.PHYSICS_DT,
            requested_steps=fixed.CONTROL_PHYSICS_STEPS,
        )
        if execution_steps == 0:
            robot_controller.stop()
            proxy_motion.advance(stationary.PHYSICS_DT)
            simulation_app.update()
            completed_physics_steps += 1
            continue

        segment_interrupted = False
        for _ in range(execution_steps):
            if not simulation_app.is_running():
                application_stopped = True
                termination_reason = "Isaac Sim stopped during command execution."
                segment_interrupted = True
                break
            try:
                step_status = perception_runtime.inspect(
                    current_simulation_time_seconds=live.simulation_time_seconds(),
                )
            except (RuntimeError, TypeError, ValueError) as error:
                stale_field_stop_seen = True
                termination_reason = f"Fail-closed field-clock stop: {error}"
                segment_interrupted = True
                break
            if step_status.fail_closed:
                stale_field_stop_seen = True
                termination_reason = (
                    "Fail-closed stale-field stop during execution: "
                    f"{step_status.failure_reason}"
                )
                segment_interrupted = True
                break
            monitored_pose = stationary.read_robot_pose(stage)
            monitored_state = live.chassis_state_in_field(
                monitored_pose,
                anchor_pose=anchor_pose,
            )
            step_backup = control_runtime.filter_command(
                selected_command,
                monitored_state,
                step_status.snapshot.payload,
            )
            step_valid = fixed.backup_result_is_valid(step_backup)
            all_backup_valid &= step_valid
            if not step_valid:
                backup_failure_seen = True
                termination_reason = (
                    "Fail-closed full-rate analytical stop: "
                    f"{step_backup.emergency_reason or 'invalid constraint'}"
                )
                segment_interrupted = True
                break
            minimum_sampled_temporal = min(
                minimum_sampled_temporal,
                step_backup.temporal_safety_derivative,
            )
            maximum_sampled_temporal = max(
                maximum_sampled_temporal,
                step_backup.temporal_safety_derivative,
            )
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
            proxy_motion.advance(stationary.PHYSICS_DT)
            simulation_app.update()
            completed_physics_steps += 1

            monitored_pose = stationary.read_robot_pose(stage)
            monitored_state = live.chassis_state_in_field(
                monitored_pose,
                anchor_pose=anchor_pose,
            )
            executed_states.append(monitored_state)
            monitored_clearance = dynamic_clearance(
                monitored_pose,
                proxy_motion,
            )
            monitored_forward, monitored_rightward = (
                motion.forward_and_rightward_displacement(
                    monitored_pose,
                    initial_pose=initial_pose,
                )
            )
            minimum_clearance = min(minimum_clearance, monitored_clearance)
            maximum_rightward = max(maximum_rightward, monitored_rightward)
            if monitored_clearance <= fixed.GROUND_TRUTH_CLEARANCE_ABORT:
                ground_truth_abort_seen = True
                termination_reason = (
                    "Independent ground-truth clearance guard triggered "
                    "during execution."
                )
                segment_interrupted = True
                break
            if monitored_forward >= clearance_threshold:
                cleared_proxy = True
                termination_reason = "The robot cleared the crossing plane."
                segment_interrupted = True
                break
        if segment_interrupted:
            break

    if simulation_app.is_running():
        robot_controller.stop()
        app_utils.update_app(steps=5)
    final_pose = stationary.read_robot_pose(stage)
    minimum_clearance = min(
        minimum_clearance,
        dynamic_clearance(final_pose, proxy_motion),
    )
    _, final_rightward = motion.forward_and_rightward_displacement(
        final_pose,
        initial_pose=initial_pose,
    )
    maximum_rightward = max(maximum_rightward, final_rightward)
    return DynamicNavigationResult(
        termination_reason=termination_reason,
        completed_control_cycles=completed_cycles,
        completed_physics_steps=completed_physics_steps,
        initial_pose=initial_pose,
        final_pose=final_pose,
        executed_chassis_states=np.ascontiguousarray(
            np.vstack(executed_states),
            dtype=np.float64,
        ),
        initial_sensor_frame_id=initial_frame_id,
        final_sensor_frame_id=(
            perception_runtime.field_buffer.snapshot.sensor_frame_id
        ),
        successful_field_updates=successful_updates,
        temporal_field_updates=temporal_updates,
        reused_field_cycles=reused_cycles,
        field_update_failures=update_failures,
        distinct_proxy_track_ids=tuple(sorted(proxy_track_ids)),
        maximum_tracked_proxy_speed=maximum_track_speed,
        minimum_sampled_temporal_derivative=minimum_sampled_temporal,
        maximum_sampled_temporal_derivative=maximum_sampled_temporal,
        maximum_field_temporal_magnitude=maximum_field_temporal,
        minimum_temporal_valid_fraction=minimum_valid_fraction,
        minimum_policy_clearance=minimum_clearance,
        maximum_rightward_displacement=maximum_rightward,
        minimum_predicted_safety=minimum_predicted_safety,
        minimum_predicted_margin=minimum_predicted_margin,
        minimum_backup_safety=minimum_backup_safety,
        minimum_backup_margin=minimum_backup_margin,
        mean_field_build_wall_seconds=float(np.mean(field_build_times)),
        mean_mpc_solve_wall_seconds=(
            math.nan if not solve_times else float(np.mean(solve_times))
        ),
        cleared_proxy=cleared_proxy,
        proxy_crossed_route=proxy_motion.crossed_route,
        proxy_finished=proxy_motion.finished,
        mpc_intervention_seen=mpc_intervention_seen,
        all_predictive_constraints_valid=all_predictive_valid,
        all_backup_constraints_valid=all_backup_valid,
        mpc_failure_seen=mpc_failure_seen,
        backup_failure_seen=backup_failure_seen,
        perception_failure_seen=perception_failure_seen,
        stale_field_stop_seen=stale_field_stop_seen,
        ground_truth_abort_seen=ground_truth_abort_seen,
        application_stopped=application_stopped,
    )


def print_result(result: DynamicNavigationResult) -> bool:
    """Print the dynamic acceptance report and return its decision."""

    forward, final_rightward = motion.forward_and_rightward_displacement(
        result.final_pose,
        initial_pose=result.initial_pose,
    )
    temporal_response_seen = (
        result.maximum_field_temporal_magnitude >= MINIMUM_TEMPORAL_MAGNITUDE
    )
    track_motion_seen = (
        result.maximum_tracked_proxy_speed >= MINIMUM_TRACKED_PROXY_SPEED
    )
    enough_updates = (
        result.temporal_field_updates >= MINIMUM_DYNAMIC_FIELD_UPDATES
    )
    stable_track = len(result.distinct_proxy_track_ids) == 1
    accepted = bool(
        result.cleared_proxy
        and result.proxy_crossed_route
        and result.proxy_finished
        and temporal_response_seen
        and track_motion_seen
        and enough_updates
        and stable_track
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
    print("SAFE-SAGE STAGE E - DYNAMIC TEMPORAL RESULT")
    print("=" * 78)
    print(f"Termination:                       {result.termination_reason}")
    print(f"Completed control cycles:          {result.completed_control_cycles}")
    print(f"Completed physics steps:           {result.completed_physics_steps}")
    print("\nLive field and tracking:")
    print(
        "  Initial/final LiDAR frame:       "
        f"{result.initial_sensor_frame_id} / {result.final_sensor_frame_id}"
    )
    print(f"  Successful field updates:        {result.successful_field_updates}")
    print(f"  Temporal field updates:          {result.temporal_field_updates}")
    print(f"  Reused field cycles:             {result.reused_field_cycles}")
    print(f"  Field update failures:           {result.field_update_failures}")
    print(f"  Proxy track IDs:                 {result.distinct_proxy_track_ids}")
    print(f"  Maximum tracked proxy speed:     {result.maximum_tracked_proxy_speed:.4f} m/s")
    print("\nTemporal derivative:")
    print(
        "  Sampled derivative range:        "
        f"[{result.minimum_sampled_temporal_derivative:+.4e}, "
        f"{result.maximum_sampled_temporal_derivative:+.4e}]"
    )
    print(
        "  Maximum field magnitude:         "
        f"{result.maximum_field_temporal_magnitude:.4e}"
    )
    print(
        "  Minimum valid aligned fraction:  "
        f"{result.minimum_temporal_valid_fraction:.4f}"
    )
    print("\nPhysical motion:")
    print(f"  Forward displacement:            {forward:.4f} m")
    print(f"  Final rightward displacement:    {final_rightward:.4f} m")
    print(f"  Maximum rightward displacement:  {result.maximum_rightward_displacement:.4f} m")
    print(f"  Minimum policy clearance:        {result.minimum_policy_clearance:.4f} m")
    print("\nSafety layers:")
    print(f"  Minimum predicted h_full:        {result.minimum_predicted_safety:.6f}")
    print(f"  Minimum predicted DTCBF margin:  {result.minimum_predicted_margin:.3e}")
    print(f"  Minimum analytical h_full:       {result.minimum_backup_safety:.6f}")
    print(f"  Minimum analytical CBF margin:   {result.minimum_backup_margin:.3e}")
    print(f"  Mean field build time:           {result.mean_field_build_wall_seconds:.3f} s")
    print(f"  Mean MPC solve time:             {result.mean_mpc_solve_wall_seconds:.3f} s")
    print("  Runtime is diagnostic, not an acceptance criterion.")
    print("\nControlled checks:")
    print(f"  Proxy crossed route:             {result.proxy_crossed_route}")
    print(f"  Proxy finished crossing:         {result.proxy_finished}")
    print(f"  Robot cleared crossing plane:    {result.cleared_proxy}")
    print(f"  Stable proxy track:              {stable_track}")
    print(f"  Tracked motion observed:         {track_motion_seen}")
    print(f"  Temporal response observed:      {temporal_response_seen}")
    print(f"  Enough temporal updates:         {enough_updates}")
    print(f"  MPC intervention observed:       {result.mpc_intervention_seen}")
    print(f"  All predictive constraints:      {result.all_predictive_constraints_valid}")
    print(f"  All analytical constraints:      {result.all_backup_constraints_valid}")
    print(f"  MPC failure seen:                {result.mpc_failure_seen}")
    print(f"  Analytical failure seen:         {result.backup_failure_seen}")
    print(f"  Perception failure seen:         {result.perception_failure_seen}")
    print(f"  Stale-field stop seen:           {result.stale_field_stop_seen}")
    print(f"  Clearance abort seen:            {result.ground_truth_abort_seen}")
    print(f"  Application stopped:             {result.application_stopped}")
    print("\nInterpretation:")
    if accepted:
        print(
            "  [PASS] Nova Carter safely completed the dynamic crossing "
            "trial using live LiDAR fields, motion-compensated temporal "
            "variation, direct MPC, and the full-rate analytical CBF."
        )
    else:
        print(
            "  [WARNING] The dynamic temporal trial did not pass every "
            "criterion. The controller stopped fail-closed."
        )
    print("=" * 78)
    return accepted


def main() -> None:
    """Build the controlled scenario and run dynamic Safe-SAGE."""

    print("\n" + "=" * 78)
    print("NOVA CARTER SAFE-SAGE DYNAMIC TEMPORAL NAVIGATION DEMO")
    print("=" * 78)
    print("Mode: moving proxy + live h_full + dh/dt + direct MPC/CBF")
    print(f"Guidance policy: {stationary.GUIDANCE_POLICY_PATH}")
    print("Semantic label source: manual controlled proxy")

    class_policy, guidance_policy, grid_config, field_config = (
        fixed.create_fixed_field_configuration()
    )
    # The proxy finishes 1.5 m to the left of the initial chassis center.
    # Preserve the validated resolution while giving every yaw-inflated
    # footprint and its 0.6 m tangential ring a full PDE boundary margin.
    grid_config = replace(
        grid_config,
        minimum_y=DYNAMIC_GRID_MINIMUM_Y,
        maximum_y=DYNAMIC_GRID_MAXIMUM_Y,
    )
    passing_side = guidance_policy.parameters_for(
        stationary.SEMANTIC_CLASS
    ).passing_side
    loaded_scene = load_robot_scene(
        scene_path=stationary.ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=stationary.ROBOT_PRIM_PATH,
        expected_meters_per_unit=stationary.EXPECTED_METERS_PER_UNIT,
    )
    proxy_motion = create_dynamic_proxy(loaded_scene.stage)
    print(
        "\nDynamic proxy: "
        f"x={DYNAMIC_PROXY_X_WORLD:.2f}, "
        f"y={DYNAMIC_PROXY_START_Y_WORLD:.2f} -> "
        f"{DYNAMIC_PROXY_END_Y_WORLD:.2f}, "
        f"speed={DYNAMIC_PROXY_SPEED_METERS_PER_SECOND:.2f} m/s"
    )
    print("Collision: disabled; session layer only")

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
    floor_height = stationary.estimate_floor_height_in_robot_frame(
        loaded_scene.stage
    )
    detection_config = ObstacleDetectionConfig(
        minimum_height=floor_height + stationary.MINIMUM_HEIGHT_ABOVE_FLOOR,
        maximum_height=floor_height + stationary.MAXIMUM_HEIGHT_ABOVE_FLOOR,
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
        temporal_estimator=SafeSAGETemporalDerivativeEstimator(
            TEMPORAL_CONFIG
        ),
        live_field_config=live.LIVE_FIELD_CONFIG,
        enable_temporal_updates=True,
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
    print("\nWaiting for the initial dynamic-field LiDAR snapshot...")
    points_sensor_frame, frame_info, _ = stationary.wait_for_first_scan(
        lidar,
        robot_controller,
    )
    initial_status = perception_runtime.initialize(
        live.make_perception_observation(
            points_sensor_frame=points_sensor_frame,
            frame_info=frame_info,
            robot_pose=anchor_pose,
            robot_from_lidar=robot_from_lidar,
            detection_config=detection_config,
        ),
        live.controlled_proxy_label_provider(
            proxy_motion.position_world[:2]
        ),
        anchor_position_world_xy=np.array(
            [anchor_pose.x, anchor_pose.y],
            dtype=np.float64,
        ),
        anchor_yaw=anchor_pose.yaw,
    )
    initial_field = initial_status.snapshot.payload
    print(
        f"Initial field committed from frame {frame_info.sensor_frame_id} "
        f"in {initial_field.build_wall_seconds:.3f} s."
    )

    base_mpc, backup_filter = fixed.create_controller_stack(robot_controller)
    mpc = SafeSAGEUnicycleMPC(
        base_mpc.paper_filter,
        replace(
            base_mpc.config,
            safety_substeps_per_interval=(
                live.LIVE_SAFETY_SUBSTEPS_PER_INTERVAL
            ),
        ),
    )
    control_runtime = SafeSAGEControlRuntime(
        mpc=mpc,
        backup_filter=backup_filter,
    )
    result = run_dynamic_navigation(
        stage=loaded_scene.stage,
        proxy_motion=proxy_motion,
        lidar=lidar,
        robot_controller=robot_controller,
        perception_runtime=perception_runtime,
        control_runtime=control_runtime,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
        passing_side=passing_side,
    )
    live.create_executed_path_markers(
        loaded_scene.stage,
        result.executed_chassis_states,
        anchor_pose=anchor_pose,
    )
    print_result(result)

    robot_controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()
    print("\nExpected successful scene behavior:")
    print("  - The blue proxy moves from robot-right to robot-left.")
    print("  - The waypoint planner continues pointing through the crossing plane.")
    print("  - Safe-SAGE modifies the physical motion without a bypass reference.")
    print("  - Nova Carter never touches the proxy and continues past its X plane.")
    print("  - Green markers show only the physically executed robot path.")
    if stationary.SIMULATION_HEADLESS or live.AUTOCLOSE_AFTER_REPORT:
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
        print("ERROR DURING SAFE-SAGE DYNAMIC TEMPORAL NAVIGATION DEMO")
        print("=" * 78)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
