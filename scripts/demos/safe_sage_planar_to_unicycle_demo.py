"""Validate the planar Safe-SAGE to Nova Carter MPC interface.

This stationary integration demo connects the two control spaces that were
validated separately:

Stage A - real perception and planar Safe-SAGE oracle:

1. Load the controlled restaurant scene and keep Nova Carter stationary.
2. Observe the blue proxy with the real PandarXT-32 LiDAR pipeline.
3. Build the paper-space Safe-SAGE field about the chassis origin.
4. Run the virtual planar oracle until it safely passes the proxy.

Stage B - executable unicycle reference construction:

1. Take the first unicycle-MPC horizon from the successful planar path.
2. Preserve its ``(x, y)`` geometry and derive path-tangent yaw targets.
3. Construct a bounded, dynamically feasible ``(v, omega)`` seed.

Stage C - stationary Nova Carter prediction:

1. Re-express the unchanged measured footprint about the look-ahead point.
2. Build the corresponding yaw-dependent ``h_full`` from the same LiDAR scan.
3. Track the planar reference with the unicycle MPC under exact DTCBF checks.
4. Pass the first MPC command through the analytical unicycle CBF backup.
5. Visualize every path without sending a command to the wheels.

Marker colors:

* White: unsafe nominal centerline.
* Cyan: planar Safe-SAGE chassis reference.
* Yellow: dynamically feasible unicycle seed.
* Green: DTCBF-constrained unicycle MPC prediction.
* Magenta: final predicted unicycle state.

Run from the project root:

    python -u scripts/demos/safe_sage_planar_to_unicycle_demo.py

Playback begins automatically. Set ``ISAAC_SIM_HEADLESS=1`` to run without a
window. The physical robot remains stationary in both modes.
"""

import math
import sys
import time
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# This import initializes the one required SimulationApp. The guarded oracle
# main function is not executed.
from scripts.demos import safe_sage_planar_oracle_demo as oracle


motion = oracle.motion
stationary = oracle.stationary
simulation_app = oracle.simulation_app


import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
from isaacsim.core.simulation_manager import SimulationManager
from robot.control.differential_drive import NovaCarterController
from robot.control.planar_path_reference import (
    PlanarPathReference,
    PlanarPathReferenceConfig,
    build_planar_path_reference,
)
from robot.navigation.waypoint_controller import RobotPose2D, wrap_to_pi
from robot.runtime.simulation_loop import (
    advance_playing_steps,
    wait_until_playing,
)
from robot.safety.obstacle_detection import ObstacleDetectionConfig
from robot.safety.obstacle_tracking import (
    ObstacleTracker,
    ObstacleTrackerConfig,
)
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
)
from robot.safety.safe_sage_guidance_policy import PassingSide
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
# Stationary integration configuration
# ---------------------------------------------------------------------------


UNICYCLE_HORIZON_STEPS = oracle.ORACLE_HORIZON_STEPS
UNICYCLE_TIME_STEP = oracle.ORACLE_CONTROL_INTERVAL_SECONDS

UNICYCLE_LINEAR_VELOCITY_WEIGHT = 1.0
UNICYCLE_ANGULAR_VELOCITY_WEIGHT = 0.02
UNICYCLE_COMMAND_SMOOTHNESS_WEIGHT = 0.005
UNICYCLE_POSITION_TRACKING_WEIGHT = 4.0
UNICYCLE_YAW_TRACKING_WEIGHT = 0.20
UNICYCLE_REFERENCE_HEADING_GAIN = 2.0
UNICYCLE_MAXIMUM_SQP_ITERATIONS = 12
UNICYCLE_MAXIMUM_RESTORATION_ITERATIONS = 12
UNICYCLE_MAXIMUM_SOLVER_ITERATIONS = 4000

MINIMUM_REFERENCE_RIGHTWARD_DISPLACEMENT = 0.05
MINIMUM_PREDICTED_RIGHTWARD_DISPLACEMENT = 0.03
MINIMUM_EXPECTED_ANGULAR_SPEED = 1e-3
MAXIMUM_POSITION_TRACKING_RMSE = 0.30
MAXIMUM_FINAL_POSITION_ERROR = 0.45

MARKER_ROOT = "/World/SafeSAGEPlanarToUnicycle"
NOMINAL_MARKER_ROOT = f"{MARKER_ROOT}/NominalCenterline"
PLANAR_REFERENCE_MARKER_ROOT = f"{MARKER_ROOT}/PlanarReference"
UNICYCLE_SEED_MARKER_ROOT = f"{MARKER_ROOT}/UnicycleSeed"
UNICYCLE_PREDICTION_MARKER_ROOT = f"{MARKER_ROOT}/MPCPrediction"
NOMINAL_MARKER_COLOR = (0.95, 0.95, 0.95)
PLANAR_REFERENCE_MARKER_COLOR = (0.10, 0.85, 1.00)
UNICYCLE_SEED_MARKER_COLOR = (1.00, 0.80, 0.10)
UNICYCLE_PREDICTION_MARKER_COLOR = (0.10, 0.95, 0.35)


def create_obstacle_tracker() -> ObstacleTracker:
    """Return one tracker with the validated controlled-demo settings."""

    return ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=stationary.TRACKER_CLUSTER_RADIUS,
            minimum_cluster_points=stationary.TRACKER_MINIMUM_CLUSTER_POINTS,
            maximum_association_distance=(
                stationary.TRACKER_MAXIMUM_ASSOCIATION_DISTANCE
            ),
        )
    )


def create_detection_config(
    floor_height_robot: float,
) -> ObstacleDetectionConfig:
    """Return the common LiDAR obstacle-extraction configuration."""

    return ObstacleDetectionConfig(
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


def select_planar_reference_horizon(
    planar_states: np.ndarray,
) -> np.ndarray:
    """Return the first complete unicycle-MPC horizon from an oracle path."""

    states = np.asarray(planar_states, dtype=np.float64)
    expected_count = UNICYCLE_HORIZON_STEPS + 1
    if states.ndim != 2 or states.shape[1:] != (3,):
        raise ValueError("planar_states must have shape (N, 3).")
    if not np.isfinite(states).all():
        raise ValueError("planar_states must contain only finite values.")
    if states.shape[0] < expected_count:
        raise RuntimeError(
            "The successful planar oracle path does not contain a complete "
            f"{UNICYCLE_HORIZON_STEPS}-step unicycle MPC horizon."
        )
    return np.array(states[:expected_count], dtype=np.float64, copy=True)


def passing_direction_grid_y(
    passing_side: object,
) -> float:
    """Map the configured passing side to the local grid-Y sign."""

    if passing_side is PassingSide.RIGHT:
        return -1.0
    if passing_side is PassingSide.LEFT:
        return 1.0
    raise ValueError(f"Unsupported passing side: {passing_side!r}")


def expected_directional_displacement(
    states: np.ndarray,
    *,
    direction_grid_y: float,
) -> float:
    """Return maximum displacement toward one signed lateral direction."""

    if direction_grid_y < 0.0:
        return float(-np.min(states[:, 1]))
    return float(np.max(states[:, 1]))


def wrapped_yaw_errors(
    states: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Return shortest signed yaw errors for two aligned state paths."""

    return np.array(
        [
            wrap_to_pi(float(state[2]) - float(target[2]))
            for state, target in zip(states, targets, strict=True)
        ],
        dtype=np.float64,
    )


def predictive_result_is_valid(
    result: SafeSAGEUnicycleMPCResult,
    *,
    config: SafeSAGEUnicycleMPCConfig,
) -> bool:
    """Return whether the unicycle prediction passed exact acceptance."""

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
        and result.minimum_predicted_safety_value
        >= -config.constraint_tolerance
        and result.minimum_dtcbf_margin >= -config.constraint_tolerance
    )


def analytical_result_is_valid(
    result: SafeSAGEUnicycleCBFResult,
) -> bool:
    """Return whether the final analytical unicycle layer is executable."""

    command = np.array(
        [result.safe_linear_velocity, result.safe_angular_velocity],
        dtype=np.float64,
    )
    return bool(
        result.constraint_feasible
        and result.constraint_satisfied
        and not result.emergency_stop_active
        and np.isfinite(command).all()
        and math.isfinite(result.sampled_safety_value)
        and math.isfinite(result.cbf_margin)
    )


def create_unicycle_mpc(
    robot_controller: NovaCarterController,
    *,
    paper_filter: oracle.SafeSAGECBFSafetyFilter,
) -> tuple[SafeSAGEUnicycleMPC, SafeSAGEUnicycleMPCConfig]:
    """Create the path-tracking Nova Carter MPC."""

    config = SafeSAGEUnicycleMPCConfig(
        horizon_steps=UNICYCLE_HORIZON_STEPS,
        time_step=UNICYCLE_TIME_STEP,
        gamma=stationary.CBF_GAMMA,
        lookahead_distance=stationary.UNICYCLE_LOOKAHEAD_DISTANCE,
        minimum_linear_velocity=0.0,
        maximum_linear_velocity=robot_controller.config.max_linear_speed,
        minimum_angular_velocity=-robot_controller.config.max_angular_speed,
        maximum_angular_velocity=robot_controller.config.max_angular_speed,
        linear_velocity_weight=UNICYCLE_LINEAR_VELOCITY_WEIGHT,
        angular_velocity_weight=UNICYCLE_ANGULAR_VELOCITY_WEIGHT,
        command_smoothness_weight=UNICYCLE_COMMAND_SMOOTHNESS_WEIGHT,
        position_tracking_weight=UNICYCLE_POSITION_TRACKING_WEIGHT,
        yaw_tracking_weight=UNICYCLE_YAW_TRACKING_WEIGHT,
        maximum_sqp_iterations=UNICYCLE_MAXIMUM_SQP_ITERATIONS,
        maximum_restoration_iterations=(
            UNICYCLE_MAXIMUM_RESTORATION_ITERATIONS
        ),
        maximum_solver_iterations=UNICYCLE_MAXIMUM_SOLVER_ITERATIONS,
    )
    return SafeSAGEUnicycleMPC(paper_filter, config), config


def create_analytical_filter(
    robot_controller: NovaCarterController,
    *,
    paper_filter: oracle.SafeSAGECBFSafetyFilter,
) -> SafeSAGEUnicycleCBFFilter:
    """Create the final full-rate analytical safety layer."""

    return SafeSAGEUnicycleCBFFilter(
        paper_filter,
        SafeSAGEUnicycleCBFConfig(
            lookahead_distance=stationary.UNICYCLE_LOOKAHEAD_DISTANCE,
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=robot_controller.config.max_linear_speed,
            minimum_angular_velocity=-robot_controller.config.max_angular_speed,
            maximum_angular_velocity=robot_controller.config.max_angular_speed,
            linear_velocity_weight=stationary.CBF_LINEAR_VELOCITY_WEIGHT,
            angular_velocity_weight=stationary.CBF_ANGULAR_VELOCITY_WEIGHT,
        ),
    )


def print_integration_report(
    *,
    planar_result: oracle.PlanarOracleResult,
    reference: PlanarPathReference,
    unicycle_frame: motion.MotionPipelineFrame,
    mpc_config: SafeSAGEUnicycleMPCConfig,
    bootstrap_result: SafeSAGEUnicycleMPCResult,
    mpc_result: SafeSAGEUnicycleMPCResult,
    analytical_result: SafeSAGEUnicycleCBFResult,
    bootstrap_wall_seconds: float,
    solve_wall_seconds: float,
    direction_grid_y: float,
    initial_pose: RobotPose2D,
    final_pose: RobotPose2D,
) -> bool:
    """Print the controlled comparison and return the final verdict."""

    planar_valid = bool(
        planar_result.proxy_cleared
        and planar_result.rightward_response_seen
        and planar_result.all_predictive_constraints_valid
        and planar_result.all_backup_constraints_valid
        and not planar_result.mpc_fallback_seen
        and not planar_result.backup_failure_seen
    )
    bootstrap_valid = predictive_result_is_valid(
        bootstrap_result,
        config=mpc_config,
    )
    mpc_valid = predictive_result_is_valid(mpc_result, config=mpc_config)
    analytical_valid = analytical_result_is_valid(analytical_result)

    target_rightward = expected_directional_displacement(
        reference.target_states,
        direction_grid_y=direction_grid_y,
    )
    seed_rightward = expected_directional_displacement(
        reference.seed_chassis_states,
        direction_grid_y=direction_grid_y,
    )
    predicted_rightward = expected_directional_displacement(
        mpc_result.predicted_chassis_states,
        direction_grid_y=direction_grid_y,
    )
    if direction_grid_y < 0.0:
        first_turn_expected = (
            mpc_result.angular_velocity < -MINIMUM_EXPECTED_ANGULAR_SPEED
        )
    else:
        first_turn_expected = (
            mpc_result.angular_velocity > MINIMUM_EXPECTED_ANGULAR_SPEED
        )

    position_errors = (
        mpc_result.predicted_chassis_states[:, :2]
        - reference.target_states[:, :2]
    )
    bootstrap_position_errors = (
        bootstrap_result.predicted_chassis_states[:, :2]
        - reference.target_states[:, :2]
    )
    bootstrap_position_rmse = float(
        np.sqrt(np.mean(np.sum(bootstrap_position_errors**2, axis=1)))
    )
    position_rmse = float(
        np.sqrt(np.mean(np.sum(position_errors**2, axis=1)))
    )
    final_position_error = float(np.linalg.norm(position_errors[-1]))
    yaw_errors = wrapped_yaw_errors(
        mpc_result.predicted_chassis_states,
        reference.target_states,
    )
    yaw_rmse = float(np.sqrt(np.mean(yaw_errors**2)))
    path_tracking_valid = bool(
        target_rightward >= MINIMUM_REFERENCE_RIGHTWARD_DISPLACEMENT
        and predicted_rightward >= MINIMUM_PREDICTED_RIGHTWARD_DISPLACEMENT
        and first_turn_expected
        and position_rmse <= MAXIMUM_POSITION_TRACKING_RMSE
        and final_position_error <= MAXIMUM_FINAL_POSITION_ERROR
    )

    position_drift = math.hypot(
        final_pose.x - initial_pose.x,
        final_pose.y - initial_pose.y,
    )
    yaw_drift = stationary.absolute_wrapped_angle_difference(
        final_pose.yaw,
        initial_pose.yaw,
    )
    robot_stationary = bool(
        position_drift <= stationary.MAXIMUM_STATIONARY_POSITION_DRIFT
        and yaw_drift <= stationary.MAXIMUM_STATIONARY_YAW_DRIFT_RADIANS
    )
    accepted = bool(
        planar_valid
        and bootstrap_valid
        and mpc_valid
        and analytical_valid
        and path_tracking_valid
        and robot_stationary
    )

    print("\n" + "=" * 78)
    print("SAFE-SAGE PLANAR-TO-UNICYCLE STATIONARY INTEGRATION REPORT")
    print("=" * 78)
    print("\nControlled architecture:")
    print("  Planar oracle state/input:      (x, y, yaw) / (v_x, v_y, omega)")
    print("  Nova Carter state/input:        (x, y, yaw) / (v, omega)")
    print("  Planar command sent to wheels:  False")
    print("  Unicycle command sent to wheels:False")
    print("  Shared LiDAR snapshot:          True")
    print("  Physical footprint changed:     False")
    print("  Footprint reference changed:    chassis -> look-ahead point")

    print("\nPlanar Safe-SAGE oracle:")
    print(f"  Termination:                    {planar_result.termination_reason}")
    print(f"  Completed cycles:               {planar_result.completed_control_cycles}")
    print(f"  Proxy cleared:                  {planar_result.proxy_cleared}")
    print(f"  Rightward response:             {planar_result.rightward_response_seen}")
    print(f"  Predictive constraints valid:   {planar_result.all_predictive_constraints_valid}")
    print(f"  Analytical constraints valid:   {planar_result.all_backup_constraints_valid}")
    print(f"  Oracle accepted:                {planar_valid}")

    print("\nReference conversion:")
    print(f"  Horizon states/commands:        {reference.target_states.shape[0]} / {reference.seed_commands.shape[0]}")
    print(f"  Target rightward displacement:  {target_rightward:.4f} m")
    print(f"  Seed rightward displacement:    {seed_rightward:.4f} m")
    print(f"  Seed first command (v, omega):  ({reference.seed_commands[0, 0]:.6f}, {reference.seed_commands[0, 1]:.6f})")
    print("  Seed purpose:                   optimizer initialization only")

    print("\nUnicycle MPC prediction:")
    print(f"  LiDAR frame ID:                 {unicycle_frame.sensor_frame_id}")
    print(f"  Horizon / interval:             {mpc_config.horizon_steps} / {mpc_config.time_step:.3f} s")
    print(f"  Bootstrap status:               {bootstrap_result.status}")
    print(f"  Bootstrap accepted:             {bootstrap_valid}")
    print(f"  Bootstrap tracking RMSE:        {bootstrap_position_rmse:.4f} m")
    print(f"  Bootstrap solve wall time:      {bootstrap_wall_seconds:.3f} s")
    print(f"  Refined first command:          ({mpc_result.linear_velocity:.6f}, {mpc_result.angular_velocity:.6f})")
    print(f"  Refined status:                 {mpc_result.status}")
    print(f"  Success / fallback:             {mpc_result.success} / {mpc_result.fallback_active}")
    print(f"  Minimum predicted h_full:       {mpc_result.minimum_predicted_safety_value:.6f}")
    print(f"  Minimum DTCBF margin:           {mpc_result.minimum_dtcbf_margin:.3e}")
    print(f"  Predicted rightward motion:     {predicted_rightward:.4f} m")
    print(f"  Position tracking RMSE:         {position_rmse:.4f} m")
    print(f"  Final position error:           {final_position_error:.4f} m")
    print(f"  Yaw tracking RMSE:              {math.degrees(yaw_rmse):.2f} degrees")
    print(f"  First turn follows path:        {first_turn_expected}")
    print(f"  Refinement solve wall time:     {solve_wall_seconds:.3f} s")
    print(f"  MPC accepted:                   {mpc_valid}")
    print(f"  Path tracking accepted:         {path_tracking_valid}")

    print("\nAnalytical unicycle backup:")
    print(f"  Output (v, omega):              ({analytical_result.safe_linear_velocity:.6f}, {analytical_result.safe_angular_velocity:.6f})")
    print(f"  Intervention:                   {analytical_result.intervention_active}")
    print(f"  Constraint margin:              {analytical_result.cbf_margin:.3e}")
    print(f"  Accepted:                       {analytical_valid}")

    print("\nPhysical stationarity:")
    print(f"  Position drift:                 {position_drift:.6f} m")
    print(f"  Yaw drift:                      {math.degrees(yaw_drift):.4f} degrees")
    print(f"  Robot remained stationary:      {robot_stationary}")

    print("\nInterpretation:")
    if accepted:
        print(
            "  [PASS] The planar Safe-SAGE bypass was converted into a "
            "dynamically feasible Nova Carter reference. The unicycle MPC "
            "followed its passing side while satisfying every predicted "
            "DTCBF constraint, and the analytical backup accepted the first "
            "executable command."
        )
        print("  Controlled physical motion may proceed to the next stage.")
    else:
        print(
            "  [WARNING] At least one stationary integration criterion "
            "failed. Keep wheel execution disabled and inspect the failed "
            "check above."
        )
    print("  All paths and commands in this report were diagnostic only.")
    print("=" * 78)
    return accepted


def main() -> None:
    """Run the complete stationary planar-to-unicycle integration."""

    print("\n" + "=" * 78)
    print("SAFE-SAGE PLANAR-TO-UNICYCLE INTEGRATION DEMO")
    print("=" * 78)
    print("Mode: real perception + virtual oracle + stationary prediction")
    print("Physical Nova Carter wheel commands: disabled")
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
    direction_grid_y = passing_direction_grid_y(
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
    detection_config = create_detection_config(floor_height_robot)

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
    print("\nWaiting for the shared fixed-field LiDAR snapshot...")
    points_sensor_frame, frame_info, _ = stationary.wait_for_first_scan(
        lidar,
        robot_controller,
    )

    print("Building the paper-space chassis-origin Safe-SAGE field...")
    planar_frame = motion.build_motion_pipeline_frame(
        points_sensor_frame=points_sensor_frame,
        frame_info=frame_info,
        robot_pose=anchor_pose,
        field_anchor_pose=anchor_pose,
        obstacle_tracker=create_obstacle_tracker(),
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
    if not (
        planar_result.proxy_cleared
        and planar_result.all_predictive_constraints_valid
        and planar_result.all_backup_constraints_valid
    ):
        raise RuntimeError(
            "The planar oracle did not provide an accepted bypass path: "
            f"{planar_result.termination_reason}"
        )

    planar_reference_states = select_planar_reference_horizon(
        planar_result.virtual_states
    )
    initial_chassis_state = np.zeros(3, dtype=np.float64)
    reference = build_planar_path_reference(
        planar_reference_states,
        initial_chassis_state=initial_chassis_state,
        config=PlanarPathReferenceConfig(
            time_step=UNICYCLE_TIME_STEP,
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=robot_controller.config.max_linear_speed,
            minimum_angular_velocity=-robot_controller.config.max_angular_speed,
            maximum_angular_velocity=robot_controller.config.max_angular_speed,
            heading_gain=UNICYCLE_REFERENCE_HEADING_GAIN,
        ),
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
        obstacle_tracker=create_obstacle_tracker(),
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=grid_config,
        field_config=unicycle_field_config,
        semantic_class=oracle.PAPER_BASELINE_SEMANTIC_CLASS,
    )

    unicycle_mpc, unicycle_mpc_config = create_unicycle_mpc(
        robot_controller,
        paper_filter=paper_filter,
    )
    print("Running the stationary safe-feasibility bootstrap...")
    bootstrap_start = time.perf_counter()
    bootstrap_result = unicycle_mpc.plan(
        initial_chassis_state,
        reference.seed_commands,
        initial_command_guess=reference.seed_commands,
        target_chassis_states_grid=reference.target_states,
        preferred_lateral_direction_grid_y=direction_grid_y,
        configuration_space=unicycle_frame.pipeline.configuration_space,
        h_full=unicycle_frame.pipeline.h_full,
        guidance_fields=unicycle_frame.pipeline.guidance_fields,
        temporal_safety_derivative=(
            stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE
        ),
    )
    bootstrap_wall_seconds = time.perf_counter() - bootstrap_start
    if not predictive_result_is_valid(
        bootstrap_result,
        config=unicycle_mpc_config,
    ):
        raise RuntimeError(
            "The unicycle MPC could not construct a certified safe "
            f"bootstrap horizon: {bootstrap_result.status}"
        )

    print("Refining path tracking from the certified safe horizon...")
    solve_start = time.perf_counter()
    unicycle_result = unicycle_mpc.plan(
        initial_chassis_state,
        reference.seed_commands,
        initial_command_guess=bootstrap_result.command_sequence,
        target_chassis_states_grid=reference.target_states,
        preferred_lateral_direction_grid_y=direction_grid_y,
        configuration_space=unicycle_frame.pipeline.configuration_space,
        h_full=unicycle_frame.pipeline.h_full,
        guidance_fields=unicycle_frame.pipeline.guidance_fields,
        temporal_safety_derivative=(
            stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE
        ),
    )
    solve_wall_seconds = time.perf_counter() - solve_start

    analytical_filter = create_analytical_filter(
        robot_controller,
        paper_filter=paper_filter,
    )
    analytical_result = analytical_filter.filter_command(
        unicycle_result.linear_velocity,
        unicycle_result.angular_velocity,
        query_position_grid_xy=(
            unicycle_result.predicted_safety_states[0, :2]
        ),
        query_yaw_radians=float(
            unicycle_result.predicted_safety_states[0, 2]
        ),
        configuration_space=unicycle_frame.pipeline.configuration_space,
        h_full=unicycle_frame.pipeline.h_full,
        guidance_fields=unicycle_frame.pipeline.guidance_fields,
        temporal_safety_derivative=(
            stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE
        ),
    )

    oracle.create_path_markers(
        loaded_scene.stage,
        oracle.nominal_centerline_states(
            forward_threshold=motion.proxy_clearance_forward_threshold(
                anchor_pose
            )
        ),
        anchor_pose=anchor_pose,
        root_path=NOMINAL_MARKER_ROOT,
        color=NOMINAL_MARKER_COLOR,
        highlight_final=False,
    )
    oracle.create_path_markers(
        loaded_scene.stage,
        reference.target_states,
        anchor_pose=anchor_pose,
        root_path=PLANAR_REFERENCE_MARKER_ROOT,
        color=PLANAR_REFERENCE_MARKER_COLOR,
        highlight_final=False,
    )
    oracle.create_path_markers(
        loaded_scene.stage,
        reference.seed_chassis_states,
        anchor_pose=anchor_pose,
        root_path=UNICYCLE_SEED_MARKER_ROOT,
        color=UNICYCLE_SEED_MARKER_COLOR,
        highlight_final=False,
    )
    oracle.create_path_markers(
        loaded_scene.stage,
        unicycle_result.predicted_chassis_states,
        anchor_pose=anchor_pose,
        root_path=UNICYCLE_PREDICTION_MARKER_ROOT,
        color=UNICYCLE_PREDICTION_MARKER_COLOR,
        highlight_final=True,
    )

    robot_controller.stop()
    app_utils.update_app(steps=5)
    physical_final_pose = stationary.read_robot_pose(loaded_scene.stage)
    app_utils.stop()
    print_integration_report(
        planar_result=planar_result,
        reference=reference,
        unicycle_frame=unicycle_frame,
        mpc_config=unicycle_mpc_config,
        bootstrap_result=bootstrap_result,
        mpc_result=unicycle_result,
        analytical_result=analytical_result,
        bootstrap_wall_seconds=bootstrap_wall_seconds,
        solve_wall_seconds=solve_wall_seconds,
        direction_grid_y=direction_grid_y,
        initial_pose=physical_initial_pose,
        final_pose=physical_final_pose,
    )

    print("\nExpected scene behavior:")
    print("  - Nova Carter and the blue proxy remain physically stationary.")
    print("  - White markers continue through the proxy.")
    print("  - Cyan markers show the planar Safe-SAGE right-side path.")
    print("  - Yellow markers show the feasible unicycle seed.")
    print("  - Green markers follow the same side under unicycle DTCBF constraints.")
    print("  - The final green prediction point is magenta.")

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
        print("ERROR DURING PLANAR-TO-UNICYCLE INTEGRATION DEMO")
        print("=" * 78)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
