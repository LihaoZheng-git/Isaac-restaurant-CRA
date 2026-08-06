"""Validate Safe-SAGE unicycle MPC prediction without robot motion.

This demo reuses the controlled restaurant scenario, LiDAR association,
semantic field construction, and measured Nova Carter footprint already
validated by ``safe_sage_h_full_cbf_demo.py``.

Stage A - real perception and Safe-SAGE field:

1. Position Nova Carter in the spacious restaurant test area.
2. Create the session-only blue proxy directly ahead of the robot.
3. Associate a real PandarXT-32 track with that proxy.
4. Manually label only that controlled track as ``person``.
5. Build the yaw-dependent semantic guidance and ``h_full`` fields.

Stage B - stationary predictive safety filtering:

1. Generate the straight waypoint command ``(v_nom, omega_nom)``.
2. Predict executable unicycle states ``(x, y, psi)`` over a finite horizon.
3. Minimize deviation from the nominal command under actuator bounds and
   Safe-SAGE discrete-time CBF constraints.
4. Use semantic guidance for position linearization and ``dh/dpsi`` for the
   heading component in the sequential quadratic programming loop.
5. Pass the first MPC command through the analytical unicycle CBF as a final
   diagnostic safety layer.

No command is sent to the wheels. Small green spheres visualize the predicted
chassis path; they are session-layer markers and have no collision geometry.

Run from the project root:

    python -u scripts/demos/safe_sage_unicycle_mpc_demo.py

Press Play in Isaac Sim when requested.
"""

import math
import sys
import time
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Importing the motion demo initializes the one required SimulationApp via
# the stationary demo. Neither imported main() function is executed.
from scripts.demos import safe_sage_h_full_motion_demo as motion


stationary = motion.stationary
simulation_app = motion.simulation_app


import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Usd, UsdGeom

from robot.control.differential_drive import NovaCarterController
from robot.navigation.waypoint_controller import RobotPose2D
from robot.runtime.simulation_loop import (
    advance_playing_steps,
    wait_until_playing,
)
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
# Stationary predictive experiment configuration
# ---------------------------------------------------------------------------


# A 12-step / 2.4 s horizon ended before the nominal trajectory passed the
# proxy, so the least-cost safe prediction only slowed near the horizon end.
# Six seconds lets the optimizer compare stopping with an actual side-pass.
MPC_HORIZON_STEPS = 30
MPC_TIME_STEP = 0.20
MPC_LINEAR_VELOCITY_WEIGHT = 1.0
MPC_ANGULAR_VELOCITY_WEIGHT = 0.02
MPC_COMMAND_SMOOTHNESS_WEIGHT = 0.005
MPC_MAXIMUM_SQP_ITERATIONS = 12
MPC_MAXIMUM_SOLVER_ITERATIONS = 4000

MAXIMUM_ACCEPTED_MPC_SOLVE_WALL_SECONDS = 0.70

MINIMUM_PREDICTED_LATERAL_DISPLACEMENT = 0.03
MINIMUM_FIRST_ANGULAR_SPEED = 1e-3

PREDICTION_ROOT_PRIM_PATH = "/World/SafeSAGEMPCPrediction"
PREDICTION_MARKER_RADIUS = 0.035
PREDICTION_MARKER_HEIGHT_WORLD = 0.08
PREDICTION_MARKER_COLOR = (0.10, 0.95, 0.35)
PREDICTION_FINAL_MARKER_COLOR = (1.00, 0.20, 0.80)


def field_state_to_world_xy(
    state_grid: np.ndarray,
    *,
    anchor_pose: RobotPose2D,
) -> tuple[float, float]:
    """Transform one field-frame planar state into world coordinates."""

    cosine = math.cos(anchor_pose.yaw)
    sine = math.sin(anchor_pose.yaw)
    world_x = (
        anchor_pose.x + cosine * float(state_grid[0]) - sine * float(state_grid[1])
    )
    world_y = (
        anchor_pose.y + sine * float(state_grid[0]) + cosine * float(state_grid[1])
    )
    return world_x, world_y


def create_prediction_markers(
    stage: Usd.Stage,
    predicted_chassis_states: np.ndarray,
    *,
    anchor_pose: RobotPose2D,
) -> None:
    """Draw non-colliding session-layer spheres along the predicted path."""

    existing_prim = stage.GetPrimAtPath(PREDICTION_ROOT_PRIM_PATH)
    if existing_prim.IsValid():
        raise RuntimeError(
            "The MPC prediction marker root already exists: "
            f"{PREDICTION_ROOT_PRIM_PATH}"
        )

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage, PREDICTION_ROOT_PRIM_PATH)
        final_index = predicted_chassis_states.shape[0] - 1
        for index, state in enumerate(predicted_chassis_states):
            marker_path = f"{PREDICTION_ROOT_PRIM_PATH}/Point_{index:02d}"
            sphere = UsdGeom.Sphere.Define(stage, marker_path)
            sphere.CreateRadiusAttr(PREDICTION_MARKER_RADIUS)
            world_x, world_y = field_state_to_world_xy(
                state,
                anchor_pose=anchor_pose,
            )
            sphere.AddTranslateOp().Set(
                Gf.Vec3d(
                    world_x,
                    world_y,
                    PREDICTION_MARKER_HEIGHT_WORLD,
                )
            )
            color = (
                PREDICTION_FINAL_MARKER_COLOR
                if index == final_index
                else PREDICTION_MARKER_COLOR
            )
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def expected_directional_displacement(
    predicted_chassis_states: np.ndarray,
    *,
    passing_side: PassingSide,
) -> float:
    """Return maximum displacement toward the configured passing side."""

    lateral_positions = predicted_chassis_states[:, 1]
    if passing_side is PassingSide.RIGHT:
        return float(-np.min(lateral_positions))
    return float(np.max(lateral_positions))


def first_command_has_expected_direction(
    result: SafeSAGEUnicycleMPCResult,
    *,
    passing_side: PassingSide,
) -> bool:
    """Return whether the first angular command follows the policy side."""

    if passing_side is PassingSide.RIGHT:
        return result.angular_velocity < -MINIMUM_FIRST_ANGULAR_SPEED
    return result.angular_velocity > MINIMUM_FIRST_ANGULAR_SPEED


def print_prediction_report(
    *,
    frame: motion.MotionPipelineFrame,
    nominal_linear_velocity: float,
    nominal_angular_velocity: float,
    mpc_config: SafeSAGEUnicycleMPCConfig,
    mpc_result: SafeSAGEUnicycleMPCResult,
    mpc_solve_wall_seconds: float,
    analytical_result: SafeSAGEUnicycleCBFResult,
    passing_side: PassingSide,
    initial_pose: RobotPose2D,
    final_pose: RobotPose2D,
) -> None:
    """Print predictive trajectory, layered-filter checks, and verdict."""

    lateral_displacement = expected_directional_displacement(
        mpc_result.predicted_chassis_states,
        passing_side=passing_side,
    )
    position_drift = math.hypot(
        final_pose.x - initial_pose.x,
        final_pose.y - initial_pose.y,
    )
    yaw_drift = stationary.absolute_wrapped_angle_difference(
        final_pose.yaw,
        initial_pose.yaw,
    )
    robot_stationary = (
        position_drift <= stationary.MAXIMUM_STATIONARY_POSITION_DRIFT
        and yaw_drift <= stationary.MAXIMUM_STATIONARY_YAW_DRIFT_RADIANS
    )
    mpc_constraint_valid = bool(
        mpc_result.success
        and np.isfinite(mpc_result.dtcbf_margins).all()
        and mpc_result.minimum_dtcbf_margin >= -mpc_config.constraint_tolerance
    )
    predictive_turn_seen = bool(
        mpc_result.success
        and mpc_result.intervention_active
        and first_command_has_expected_direction(
            mpc_result,
            passing_side=passing_side,
        )
        and lateral_displacement >= MINIMUM_PREDICTED_LATERAL_DISPLACEMENT
    )
    analytical_backup_valid = bool(
        analytical_result.constraint_feasible
        and analytical_result.constraint_satisfied
        and not analytical_result.emergency_stop_active
    )
    solve_time_acceptable = bool(
        mpc_solve_wall_seconds <= MAXIMUM_ACCEPTED_MPC_SOLVE_WALL_SECONDS
    )
    accepted = bool(
        mpc_constraint_valid
        and predictive_turn_seen
        and analytical_backup_valid
        and robot_stationary
        and solve_time_acceptable
    )

    print("\n" + "=" * 78)
    print("SAFE-SAGE UNICYCLE MPC - STATIONARY PREDICTION REPORT")
    print("=" * 78)
    print("\nPerception and field:")
    print(f"  LiDAR frame ID:              {frame.sensor_frame_id}")
    print(f"  All obstacle points:         {frame.all_obstacle_point_count}")
    print(f"  Proxy track ID:              {frame.proxy_track_id}")
    print(f"  Proxy track points:          {frame.proxy_point_count}")
    print(f"  Proxy association error:     {frame.proxy_association_error:.4f} m")
    print(f"  Field build wall time:       {frame.build_wall_seconds:.3f} s")

    print("\nPredictive control configuration:")
    print("  State:                       (x, y, yaw)")
    print("  Executable input:            (v, omega)")
    print("  Sparse QP solver:            OSQP")
    print(f"  Horizon steps:               {mpc_config.horizon_steps}")
    print(f"  Time step:                   {mpc_config.time_step:.3f} s")
    print(
        f"  Prediction duration:         "
        f"{mpc_config.horizon_steps * mpc_config.time_step:.3f} s"
    )
    print(f"  DTCBF decay:                 {mpc_config.dtcbf_decay:.6f}")
    print(f"  Passing side:                {passing_side.value}")

    print("\nNominal and first optimized command:")
    print(
        f"  Nominal (v, omega):          "
        f"({nominal_linear_velocity:.6f}, "
        f"{nominal_angular_velocity:.6f})"
    )
    print(
        f"  MPC (v, omega):              "
        f"({mpc_result.linear_velocity:.6f}, "
        f"{mpc_result.angular_velocity:.6f})"
    )
    print(f"  MPC status:                  {mpc_result.status}")
    print(f"  MPC success:                 {mpc_result.success}")
    print(f"  MPC intervention:            {mpc_result.intervention_active}")
    print(f"  MPC fallback:                {mpc_result.fallback_active}")
    print(f"  Exact restoration used:      {mpc_result.restoration_used}")
    print(
        f"  SQP / solver iterations:     "
        f"{mpc_result.sqp_iteration_count} / "
        f"{mpc_result.solver_iteration_count}"
    )
    print(f"  Objective value:             {mpc_result.objective_value:.6e}")
    print(f"  MPC solve wall time:         {mpc_solve_wall_seconds:.3f} s")

    print("\nPredicted trajectory in the fixed field frame:")
    print(
        "  k    time      x       y      yaw       v      omega       "
        "h_full      margin"
    )
    for index, state in enumerate(mpc_result.predicted_chassis_states):
        time_seconds = index * mpc_config.time_step
        if index < mpc_config.horizon_steps:
            command = mpc_result.command_sequence[index]
            linear_velocity = f"{command[0]:7.3f}"
            angular_velocity = f"{command[1]:8.3f}"
        else:
            linear_velocity = "      -"
            angular_velocity = "       -"
        safety_value = mpc_result.predicted_safety_values[index]
        safety_text = (
            f"{safety_value:10.5f}"
            if math.isfinite(float(safety_value))
            else "       nan"
        )
        if index == 0:
            margin_text = "         -"
        else:
            margin = mpc_result.dtcbf_margins[index - 1]
            margin_text = (
                f"{margin:10.3e}" if math.isfinite(float(margin)) else "       nan"
            )
        print(
            f"  {index:2d}  {time_seconds:6.2f}  "
            f"{state[0]:6.3f}  {state[1]:6.3f}  "
            f"{math.degrees(state[2]):7.2f}  "
            f"{linear_velocity}  {angular_velocity}  "
            f"{safety_text}  {margin_text}"
        )

    print("\nPrediction summary:")
    print(
        f"  Minimum predicted h_full:    "
        f"{mpc_result.minimum_predicted_safety_value:.6f}"
    )
    print(f"  Minimum DTCBF margin:        {mpc_result.minimum_dtcbf_margin:.3e}")
    print(f"  Passing-side displacement:   {lateral_displacement:.4f} m")

    print("\nAnalytical backup applied to MPC first command:")
    print(
        f"  Backup output (v, omega):    "
        f"({analytical_result.safe_linear_velocity:.6f}, "
        f"{analytical_result.safe_angular_velocity:.6f})"
    )
    print(f"  Backup intervention:         {analytical_result.intervention_active}")
    print(f"  Backup constraint valid:     {analytical_backup_valid}")

    print("\nControlled checks:")
    print(f"  Predictive CBF valid:         {mpc_constraint_valid}")
    print(f"  Early passing turn predicted: {predictive_turn_seen}")
    print(f"  Analytical backup valid:      {analytical_backup_valid}")
    print(f"  MPC solve time acceptable:    {solve_time_acceptable}")
    print(f"  Robot remained stationary:    {robot_stationary}")
    print(f"  Measured position drift:       {position_drift:.6f} m")
    print(f"  Measured yaw drift:            {math.degrees(yaw_drift):.4f} degrees")

    print("\nInterpretation:")
    if accepted:
        print(
            "  [PASS] The real Safe-SAGE field caused predictive, "
            "passing-side steering before execution, every predicted "
            "DTCBF constraint was valid, and the analytical backup "
            "accepted the first MPC command."
        )
        print("  Closed-loop wheel motion may proceed to the next stage.")
    else:
        print(
            "  [WARNING] The stationary prediction did not satisfy every "
            "acceptance check. Do not enable MPC wheel motion yet."
        )
    print("  The trajectory and both commands above were diagnostic only.")
    print("=" * 78)


def main() -> None:
    """Run the real-field stationary unicycle MPC prediction."""

    print("\n" + "=" * 78)
    print("NOVA CARTER SAFE-SAGE UNICYCLE MPC DEMO")
    print("=" * 78)
    print("Mode: stationary Stage A/B prediction validation")
    print(f"Guidance policy: {stationary.GUIDANCE_POLICY_PATH}")

    guidance_policy = SafeSAGEGuidancePolicy.from_json(stationary.GUIDANCE_POLICY_PATH)
    class_policy = stationary.class_policy_from_guidance_policy(guidance_policy)
    semantic_parameters = guidance_policy.parameters_for(stationary.SEMANTIC_CLASS)

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
    field_config = stationary.create_field_pipeline_config(footprint)
    grid_config = LocalOccupancyGridConfig(
        resolution=stationary.GRID_RESOLUTION,
        minimum_x=motion.MOTION_GRID_MINIMUM_X,
        maximum_x=motion.MOTION_GRID_MAXIMUM_X,
        minimum_y=stationary.GRID_MINIMUM_Y,
        maximum_y=stationary.GRID_MAXIMUM_Y,
        obstacle_inflation_radius=0.0,
    )

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
        angular_bin_size_degrees=(stationary.ANGULAR_BIN_SIZE_DEGREES),
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
    print("\nWaiting for a valid LiDAR scan...")
    points_sensor_frame, frame_info, _ = stationary.wait_for_first_scan(
        lidar,
        robot_controller,
    )
    print("Building yaw-dependent Safe-SAGE fields...")
    frame = motion.build_motion_pipeline_frame(
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

    nominal_command = stationary.create_waypoint_controller().compute_command(
        anchor_pose
    )
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
    mpc_config = SafeSAGEUnicycleMPCConfig(
        horizon_steps=MPC_HORIZON_STEPS,
        time_step=MPC_TIME_STEP,
        gamma=stationary.CBF_GAMMA,
        lookahead_distance=stationary.UNICYCLE_LOOKAHEAD_DISTANCE,
        minimum_linear_velocity=0.0,
        maximum_linear_velocity=robot_controller.config.max_linear_speed,
        minimum_angular_velocity=(-robot_controller.config.max_angular_speed),
        maximum_angular_velocity=(robot_controller.config.max_angular_speed),
        linear_velocity_weight=MPC_LINEAR_VELOCITY_WEIGHT,
        angular_velocity_weight=MPC_ANGULAR_VELOCITY_WEIGHT,
        command_smoothness_weight=MPC_COMMAND_SMOOTHNESS_WEIGHT,
        maximum_sqp_iterations=MPC_MAXIMUM_SQP_ITERATIONS,
        maximum_solver_iterations=MPC_MAXIMUM_SOLVER_ITERATIONS,
    )
    mpc = SafeSAGEUnicycleMPC(paper_filter, mpc_config)
    initial_chassis_position, initial_yaw = motion.current_pose_in_field_frame(
        anchor_pose,
        anchor_pose=anchor_pose,
    )
    initial_chassis_state = np.array(
        [
            initial_chassis_position[0],
            initial_chassis_position[1],
            initial_yaw,
        ],
        dtype=np.float64,
    )

    print("Running stationary unicycle MPC prediction...")
    mpc_start_wall_time = time.perf_counter()
    mpc_result = mpc.plan(
        initial_chassis_state,
        np.array(
            [
                nominal_command.linear_velocity,
                nominal_command.angular_velocity,
            ],
            dtype=np.float64,
        ),
        configuration_space=frame.pipeline.configuration_space,
        h_full=frame.pipeline.h_full,
        guidance_fields=frame.pipeline.guidance_fields,
        temporal_safety_derivative=(stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE),
    )

    analytical_filter = SafeSAGEUnicycleCBFFilter(
        paper_filter,
        SafeSAGEUnicycleCBFConfig(
            lookahead_distance=(stationary.UNICYCLE_LOOKAHEAD_DISTANCE),
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=(robot_controller.config.max_linear_speed),
            minimum_angular_velocity=(-robot_controller.config.max_angular_speed),
            maximum_angular_velocity=(robot_controller.config.max_angular_speed),
            linear_velocity_weight=(stationary.CBF_LINEAR_VELOCITY_WEIGHT),
            angular_velocity_weight=(stationary.CBF_ANGULAR_VELOCITY_WEIGHT),
        ),
    )
    mpc_solve_wall_seconds = time.perf_counter() - mpc_start_wall_time
    analytical_result = analytical_filter.filter_command(
        mpc_result.linear_velocity,
        mpc_result.angular_velocity,
        query_position_grid_xy=(mpc_result.predicted_safety_states[0, :2]),
        query_yaw_radians=float(mpc_result.predicted_safety_states[0, 2]),
        configuration_space=frame.pipeline.configuration_space,
        h_full=frame.pipeline.h_full,
        guidance_fields=frame.pipeline.guidance_fields,
        temporal_safety_derivative=(stationary.STATIC_TEMPORAL_SAFETY_DERIVATIVE),
    )

    create_prediction_markers(
        loaded_scene.stage,
        mpc_result.predicted_chassis_states,
        anchor_pose=anchor_pose,
    )
    robot_controller.stop()
    app_utils.update_app(steps=5)
    final_pose = stationary.read_robot_pose(loaded_scene.stage)

    print_prediction_report(
        frame=frame,
        nominal_linear_velocity=nominal_command.linear_velocity,
        nominal_angular_velocity=nominal_command.angular_velocity,
        mpc_config=mpc_config,
        mpc_result=mpc_result,
        mpc_solve_wall_seconds=mpc_solve_wall_seconds,
        analytical_result=analytical_result,
        passing_side=semantic_parameters.passing_side,
        initial_pose=anchor_pose,
        final_pose=final_pose,
    )

    robot_controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()

    print("\nExpected scene behavior:")
    print("  - Nova Carter remains stationary near (26.5, 5.0).")
    print("  - The blue semantic proxy remains directly ahead.")
    print("  - Green prediction markers begin at the robot and curve right.")
    print("  - The final predicted point is shown in magenta.")
    print("  - No nominal, MPC, or analytical command reaches the wheels.")
    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when finished.")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 78)
        print("ERROR DURING SAFE-SAGE UNICYCLE MPC DEMO")
        print("=" * 78)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()
