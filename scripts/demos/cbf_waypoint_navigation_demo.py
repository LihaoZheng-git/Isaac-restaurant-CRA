"""Integrate waypoint navigation with the LiDAR-based CBF safety filter.

This demo connects the complete baseline control pipeline:

    robot pose
        -> waypoint controller
        -> nominal command
        -> LiDAR obstacle extraction
        -> CBF safety filter
        -> Nova Carter differential controller

The route contains two waypoints:

1. Drive a short distance forward in the restaurant aisle.
2. Turn toward the furniture and request motion toward it.

Expected behavior:

- Nova Carter reaches the first waypoint.
- The waypoint controller rotates toward the second waypoint.
- The waypoint controller requests forward motion toward the furniture.
- The CBF progressively reduces the nominal linear velocity.
- The robot stops before reaching the second waypoint.

The baseline CBF preserves angular velocity and only modifies linear
velocity. It does not find an alternative route around the obstacle.

Run from the project root:

    python -u scripts/demos/cbf_waypoint_navigation_demo.py

Press Play in Isaac Sim when requested.
"""

import math
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from robot.config import (
    EXPECTED_METERS_PER_UNIT,
    PANDAR_XT32_LIDAR_PRIM_PATH,
    PHYSICS_DEVICE,
    PHYSICS_DT,
    ROBOT_BASE_PRIM_PATH,
    ROBOT_PRIM_PATH,
    ROBOT_SCENE_PATH,
)

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})


import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Usd

from robot.control.differential_drive import NovaCarterController
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
from robot.safety.cbf_filter import (
    CBFConfig,
    CBFResult,
    CBFSafetyFilter,
)
from robot.safety.obstacle_detection import (
    ObstacleDetectionConfig,
    detect_obstacle_points,
)
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import (
    compute_relative_prim_transform,
    get_prim_planar_pose,
    get_prim_world_transform,
)
from robot.sensor.lidar import (
    LidarFrameInfo,
    NovaCarterLidar,
)
from robot.sensor.lidar_geometry import transform_points


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

SETTLE_STEPS = 60
MOTION_START_DELAY_STEPS = 120

MAXIMUM_SCAN_WAIT_STEPS = 300

MAXIMUM_NAVIGATION_DURATION_SECONDS = 20.0
MAXIMUM_NAVIGATION_STEPS = math.ceil(
    MAXIMUM_NAVIGATION_DURATION_SECONDS / PHYSICS_DT
)

MAXIMUM_STALE_SCAN_SECONDS = 0.25
MAXIMUM_STALE_SCAN_STEPS = math.ceil(
    MAXIMUM_STALE_SCAN_SECONDS / PHYSICS_DT
)

STATUS_REPORT_INTERVAL_STEPS = 30
STOP_SPEED_THRESHOLD = 0.01

CBF_SAFETY_RADIUS = 0.80
CBF_ALPHA = 1.0

MINIMUM_HEIGHT_ABOVE_FLOOR = 0.05
MAXIMUM_HEIGHT_ABOVE_FLOOR = 1.00

MINIMUM_DETECTION_RANGE = 0.15
MAXIMUM_DETECTION_RANGE = 5.00

ANGULAR_BIN_SIZE_DEGREES = 2.0

# Change this to True to turn toward the opposite side.
TURN_LEFT = True

FIRST_WAYPOINT_FORWARD_DISTANCE = 0.50
SECOND_WAYPOINT_LATERAL_DISTANCE = 1.00

SECOND_WAYPOINT_LEFT_OFFSET = (
    SECOND_WAYPOINT_LATERAL_DISTANCE
    if TURN_LEFT
    else -SECOND_WAYPOINT_LATERAL_DISTANCE
)

# Each entry is (forward, left) relative to the initial chassis frame.
# These are absolute offsets from the initial pose, not incremental offsets.
ROUTE_LOCAL_OFFSETS = (
    (FIRST_WAYPOINT_FORWARD_DISTANCE, 0.00),
    (
        FIRST_WAYPOINT_FORWARD_DISTANCE,
        SECOND_WAYPOINT_LEFT_OFFSET,
    ),
)

TRANSIENT_LIDAR_ERRORS = (
    "No LiDAR data is available yet",
    "contains no points",
    "no valid XYZ points remained",
)


# ---------------------------------------------------------------------------
# Pose and route helpers
# ---------------------------------------------------------------------------


def read_robot_pose(
    stage: Usd.Stage,
) -> RobotPose2D:
    """Read Nova Carter's current chassis pose in the world frame."""

    x_position, y_position, yaw = get_prim_planar_pose(
        stage,
        ROBOT_BASE_PRIM_PATH,
    )

    return RobotPose2D(
        x=x_position,
        y=y_position,
        yaw=yaw,
    )


def create_world_route(
    initial_pose: RobotPose2D,
) -> tuple[Waypoint, ...]:
    """Convert initial-frame route offsets into world-frame waypoints."""

    cosine_yaw = math.cos(initial_pose.yaw)
    sine_yaw = math.sin(initial_pose.yaw)

    world_waypoints = []

    for forward_offset, left_offset in ROUTE_LOCAL_OFFSETS:
        world_x = (
            initial_pose.x
            + cosine_yaw * forward_offset
            - sine_yaw * left_offset
        )
        world_y = (
            initial_pose.y
            + sine_yaw * forward_offset
            + cosine_yaw * left_offset
        )

        world_waypoints.append(
            Waypoint(
                x=world_x,
                y=world_y,
            )
        )

    return tuple(world_waypoints)


def estimate_floor_height_in_robot_frame(
    stage: Usd.Stage,
) -> float:
    """Estimate the world-Z-zero floor height in the chassis frame."""

    robot_to_world = get_prim_world_transform(
        stage,
        ROBOT_BASE_PRIM_PATH,
    )

    robot_origin_world = (
        robot_to_world.ExtractTranslation()
    )

    ground_point_world = Gf.Vec3d(
        robot_origin_world[0],
        robot_origin_world[1],
        0.0,
    )

    world_to_robot = robot_to_world.GetInverse()

    ground_point_robot = world_to_robot.Transform(
        ground_point_world
    )

    return float(ground_point_robot[2])


# ---------------------------------------------------------------------------
# LiDAR helpers
# ---------------------------------------------------------------------------


def wait_for_first_scan(
    lidar: NovaCarterLidar,
    robot_controller: NovaCarterController,
) -> tuple[np.ndarray, LidarFrameInfo, int]:
    """Advance the simulation until the first valid LiDAR scan arrives."""

    last_transient_error: RuntimeError | None = None

    for completed_steps in range(
        1,
        MAXIMUM_SCAN_WAIT_STEPS + 1,
    ):
        if not simulation_app.is_running():
            raise RuntimeError(
                "Isaac Sim closed before a LiDAR scan arrived."
            )

        if not app_utils.is_playing():
            simulation_app.update()
            continue

        robot_controller.stop()
        simulation_app.update()

        try:
            points = lidar.get_point_cloud()
        except RuntimeError as error:
            is_transient = any(
                message in str(error)
                for message in TRANSIENT_LIDAR_ERRORS
            )

            if not is_transient:
                raise

            last_transient_error = error
            continue

        frame_info = lidar.get_frame_info()

        return points, frame_info, completed_steps

    detail = (
        f"\nLast LiDAR error: {last_transient_error}"
        if last_transient_error is not None
        else ""
    )

    raise TimeoutError(
        "No valid LiDAR scan arrived after "
        f"{MAXIMUM_SCAN_WAIT_STEPS} rendered updates."
        f"{detail}"
    )


def extract_obstacle_points(
    points_sensor_frame: np.ndarray,
    *,
    robot_from_lidar: np.ndarray,
    detection_config: ObstacleDetectionConfig,
) -> np.ndarray:
    """Transform a LiDAR scan and extract CBF obstacle points."""

    points_robot_frame = transform_points(
        points_sensor_frame,
        robot_from_lidar,
    )

    return detect_obstacle_points(
        points_robot_frame,
        detection_config,
    )


def format_distance(distance: float) -> str:
    """Format a finite or infinite obstacle distance."""

    if math.isfinite(distance):
        return f"{distance:.3f}"

    return "none"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_stage_a_report(
    *,
    initial_pose: RobotPose2D,
    waypoints: tuple[Waypoint, ...],
    frame_info: LidarFrameInfo,
    obstacle_points: np.ndarray,
    nominal_command: WaypointCommand,
    cbf_result: CBFResult,
) -> None:
    """Print the stationary integration dry-run report."""

    turn_direction = "left" if TURN_LEFT else "right"

    print("\n" + "=" * 70)
    print("CBF-WAYPOINT STAGE A - STATIONARY DRY RUN")
    print("=" * 70)

    print("\nScenario:")
    print(
        "  First motion: "
        f"{FIRST_WAYPOINT_FORWARD_DISTANCE:.2f} m forward"
    )
    print(f"  Turn direction: {turn_direction}")
    print(
        "  Second-leg distance: "
        f"{SECOND_WAYPOINT_LATERAL_DISTANCE:.2f} m"
    )

    print("\nInitial chassis pose:")
    print(f"  x:   {initial_pose.x:.4f} m")
    print(f"  y:   {initial_pose.y:.4f} m")
    print(
        "  yaw: "
        f"{math.degrees(initial_pose.yaw):.2f} degrees"
    )

    print("\nWorld-frame waypoints:")

    for index, waypoint in enumerate(waypoints):
        print(
            f"  [{index + 1}] "
            f"x={waypoint.x:.4f} m, "
            f"y={waypoint.y:.4f} m"
        )

    print("\nLiDAR:")
    print(f"  Frame ID:         {frame_info.sensor_frame_id}")
    print(f"  Obstacle points:  {obstacle_points.shape[0]}")
    print(
        "  Nearest obstacle: "
        f"{format_distance(cbf_result.minimum_obstacle_distance)} m"
    )

    print("\nInitial waypoint command:")
    print(f"  Mode:    {nominal_command.mode.value}")
    print(
        "  v_nom:   "
        f"{nominal_command.linear_velocity:.4f} m/s"
    )
    print(
        "  omega:   "
        f"{nominal_command.angular_velocity:.4f} rad/s"
    )
    print(
        "  Distance:"
        f" {nominal_command.distance_to_waypoint:.4f} m"
    )

    print("\nInitial CBF result:")
    print(
        "  v_safe:       "
        f"{cbf_result.safe_linear_velocity:.4f} m/s"
    )
    print(
        "  Intervention: "
        f"{cbf_result.intervention_active}"
    )
    print(
        "  Feasible:     "
        f"{cbf_result.constraint_feasible}"
    )
    print(
        "  Emergency:    "
        f"{cbf_result.emergency_stop_active}"
    )

    print("\nThe robot remained stationary during Stage A.")
    print("=" * 70)


def print_live_status(
    *,
    completed_steps: int,
    frame_id: int,
    pose: RobotPose2D,
    nominal_command: WaypointCommand,
    cbf_result: CBFResult,
) -> None:
    """Print one compact integration status line."""

    target_text = (
        str(nominal_command.target_waypoint_index + 1)
        if nominal_command.target_waypoint_index is not None
        else "done"
    )

    obstacle_distance = format_distance(
        cbf_result.minimum_obstacle_distance
    )

    print(
        f"  step={completed_steps:<4} "
        f"scan={frame_id:<4} "
        f"target={target_text:<4} "
        f"mode={nominal_command.mode.value:<9} "
        f"distance={nominal_command.distance_to_waypoint:.3f} "
        f"heading={math.degrees(nominal_command.heading_error):6.1f} "
        f"v_nom={nominal_command.linear_velocity:.3f} "
        f"v_safe={cbf_result.safe_linear_velocity:.3f} "
        f"omega={cbf_result.safe_angular_velocity:.3f} "
        f"obstacle={obstacle_distance:>5} "
        f"intervention={cbf_result.intervention_active}"
    )


# ---------------------------------------------------------------------------
# Integrated control loop
# ---------------------------------------------------------------------------


def run_safe_waypoint_navigation(
    *,
    stage: Usd.Stage,
    lidar: NovaCarterLidar,
    robot_controller: NovaCarterController,
    waypoint_controller: WaypointController,
    safety_filter: CBFSafetyFilter,
    detection_config: ObstacleDetectionConfig,
    robot_from_lidar: np.ndarray,
    initial_obstacle_points: np.ndarray,
    initial_frame_info: LidarFrameInfo,
    initial_cbf_result: CBFResult,
) -> None:
    """Run waypoint navigation while filtering every nominal command."""

    completed_steps = 0
    received_scan_count = 0
    steps_since_new_scan = 0

    latest_obstacle_points = initial_obstacle_points
    latest_frame_id = initial_frame_info.sensor_frame_id
    latest_cbf_result = initial_cbf_result

    latest_pose = read_robot_pose(stage)
    latest_nominal_command = (
        waypoint_controller.compute_command(latest_pose)
    )

    previous_mode = None
    previous_target_index = (
        waypoint_controller.current_waypoint_index
    )
    previous_intervention = (
        initial_cbf_result.intervention_active
    )

    intervention_seen = (
        initial_cbf_result.intervention_active
    )
    first_intervention_step: int | None = (
        0 if intervention_seen else None
    )

    minimum_obstacle_distance = (
        initial_cbf_result.minimum_obstacle_distance
    )

    termination_reason = (
        "Maximum navigation duration reached."
    )

    print("\n" + "=" * 70)
    print("CBF-WAYPOINT STAGES B/C - CLOSED-LOOP MOTION")
    print("=" * 70)
    print(
        "Maximum duration: "
        f"{MAXIMUM_NAVIGATION_DURATION_SECONDS:.1f} s"
    )
    print(
        "CBF safety radius: "
        f"{safety_filter.config.safety_radius:.3f} m"
    )
    print(
        "Maximum LiDAR age: "
        f"{MAXIMUM_STALE_SCAN_SECONDS:.2f} s"
    )
    print("\nLive navigation status:")

    while (
        completed_steps < MAXIMUM_NAVIGATION_STEPS
        and simulation_app.is_running()
    ):
        if not app_utils.is_playing():
            robot_controller.stop()
            simulation_app.update()
            continue

        latest_pose = read_robot_pose(stage)

        latest_nominal_command = (
            waypoint_controller.compute_command(
                latest_pose
            )
        )

        if latest_nominal_command.waypoint_advanced:
            if latest_nominal_command.route_complete:
                print("\nFinal waypoint reached.")
            else:
                print(
                    "\nReached waypoint 1. Advancing to "
                    f"waypoint "
                    f"{latest_nominal_command.target_waypoint_index + 1}."
                )

        if latest_nominal_command.route_complete:
            termination_reason = (
                "Every waypoint was reached."
            )
            robot_controller.stop()
            break

        steps_since_new_scan += 1

        try:
            points_sensor_frame = lidar.get_point_cloud()
            current_frame_info = lidar.get_frame_info()
        except RuntimeError as error:
            is_transient = any(
                message in str(error)
                for message in TRANSIENT_LIDAR_ERRORS
            )

            if not is_transient:
                raise
        else:
            is_new_frame = (
                current_frame_info.sensor_frame_id
                != latest_frame_id
            )

            if is_new_frame:
                latest_obstacle_points = (
                    extract_obstacle_points(
                        points_sensor_frame,
                        robot_from_lidar=robot_from_lidar,
                        detection_config=detection_config,
                    )
                )

                if latest_obstacle_points.shape[0] == 0:
                    termination_reason = (
                        "No obstacle points were detected. "
                        "Fail-closed stop applied."
                    )
                    break

                latest_frame_id = (
                    current_frame_info.sensor_frame_id
                )
                steps_since_new_scan = 0
                received_scan_count += 1

        if (
            steps_since_new_scan
            >= MAXIMUM_STALE_SCAN_STEPS
        ):
            termination_reason = (
                "LiDAR data became stale. "
                "Fail-closed stop applied."
            )
            break

        # Obstacle points are updated at the LiDAR rate, but the current
        # nominal command is filtered at every physics step.
        latest_cbf_result = safety_filter.filter_command(
            nominal_linear_velocity=(
                latest_nominal_command.linear_velocity
            ),
            nominal_angular_velocity=(
                latest_nominal_command.angular_velocity
            ),
            obstacle_points_xy=latest_obstacle_points,
        )

        minimum_obstacle_distance = min(
            minimum_obstacle_distance,
            latest_cbf_result.minimum_obstacle_distance,
        )

        if (
            latest_cbf_result.intervention_active
            and not intervention_seen
        ):
            intervention_seen = True
            first_intervention_step = completed_steps

        current_target_index = (
            latest_nominal_command.target_waypoint_index
        )

        target_changed = (
            current_target_index
            != previous_target_index
        )
        mode_changed = (
            latest_nominal_command.mode
            != previous_mode
        )
        intervention_changed = (
            latest_cbf_result.intervention_active
            != previous_intervention
        )
        periodic_report = (
            completed_steps
            % STATUS_REPORT_INTERVAL_STEPS
            == 0
        )

        emergency_stop = (
            not latest_cbf_result.constraint_feasible
            or latest_cbf_result.emergency_stop_active
        )

        filtered_stop = (
            latest_cbf_result.intervention_active
            and latest_nominal_command.linear_velocity
            > STOP_SPEED_THRESHOLD
            and latest_cbf_result.safe_linear_velocity
            <= STOP_SPEED_THRESHOLD
        )

        if (
            periodic_report
            or target_changed
            or mode_changed
            or intervention_changed
            or emergency_stop
            or filtered_stop
        ):
            print_live_status(
                completed_steps=completed_steps,
                frame_id=latest_frame_id,
                pose=latest_pose,
                nominal_command=latest_nominal_command,
                cbf_result=latest_cbf_result,
            )

        if emergency_stop:
            termination_reason = (
                "The CBF requested an emergency stop."
            )
            break

        if filtered_stop:
            termination_reason = (
                "The CBF reduced forward velocity to the "
                "automatic stop threshold."
            )
            break

        robot_controller.apply_command(
            latest_cbf_result.safe_linear_velocity,
            latest_cbf_result.safe_angular_velocity,
        )

        previous_mode = latest_nominal_command.mode
        previous_target_index = current_target_index
        previous_intervention = (
            latest_cbf_result.intervention_active
        )

        simulation_app.update()
        completed_steps += 1

    robot_controller.stop()
    app_utils.update_app(steps=5)

    if simulation_app.is_running():
        latest_pose = read_robot_pose(stage)
        latest_nominal_command = (
            waypoint_controller.compute_command(
                latest_pose
            )
        )

    reached_waypoint_count = (
        len(waypoint_controller.waypoints)
        if waypoint_controller.route_complete
        else (
            waypoint_controller.current_waypoint_index
            or 0
        )
    )

    print("\n" + "=" * 70)
    print("CBF-WAYPOINT INTEGRATION RESULT")
    print("=" * 70)
    print(f"Reason: {termination_reason}")
    print(f"Completed physics steps: {completed_steps}")
    print(f"New LiDAR frames:        {received_scan_count}")
    print(
        "Reached waypoints:        "
        f"{reached_waypoint_count}/"
        f"{len(waypoint_controller.waypoints)}"
    )
    print(
        "Route complete:           "
        f"{waypoint_controller.route_complete}"
    )
    print(
        "CBF intervention seen:    "
        f"{intervention_seen}"
    )

    if first_intervention_step is None:
        print("First intervention step:  none")
    else:
        print(
            "First intervention step:  "
            f"{first_intervention_step}"
        )

    print(
        "Minimum obstacle distance: "
        f"{format_distance(minimum_obstacle_distance)} m"
    )
    print(
        "Final nominal velocity:    "
        f"{latest_nominal_command.linear_velocity:.4f} m/s"
    )
    print(
        "Final safe velocity:       "
        f"{latest_cbf_result.safe_linear_velocity:.4f} m/s"
    )

    print("\nFinal chassis pose:")
    print(f"  x:   {latest_pose.x:.4f} m")
    print(f"  y:   {latest_pose.y:.4f} m")
    print(
        "  yaw: "
        f"{math.degrees(latest_pose.yaw):.2f} degrees"
    )

    print("\nInterpretation:")

    if (
        intervention_seen
        and not waypoint_controller.route_complete
    ):
        print(
            "  [PASS] The waypoint controller requested progress, "
            "but the CBF prevented the unsafe route from completing."
        )
    elif (
        intervention_seen
        and waypoint_controller.route_complete
    ):
        print(
            "  [PASS] The CBF intervened and the route still "
            "completed safely."
        )
    elif waypoint_controller.route_complete:
        print(
            "  [WARNING] The route completed without CBF "
            "intervention. The route may not intersect furniture."
        )
    else:
        print(
            "  [WARNING] The route did not complete and no CBF "
            "intervention was observed. Inspect the termination reason."
        )

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the CBF-filtered waypoint-navigation demonstration."""

    print("\n" + "=" * 70)
    print("NOVA CARTER CBF-WAYPOINT INTEGRATION DEMO")
    print("=" * 70)
    print(
        "Turn direction: "
        f"{'left' if TURN_LEFT else 'right'}"
    )

    loaded_scene = load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=EXPECTED_METERS_PER_UNIT,
    )

    robot_controller = NovaCarterController(
        robot_prim_path=ROBOT_PRIM_PATH
    )

    SimulationManager.setup_simulation(
        dt=PHYSICS_DT,
        device=PHYSICS_DEVICE,
    )

    print("\nPress Play in Isaac Sim to begin.")

    if not wait_until_playing(simulation_app):
        return

    print("\nPlay detected. Initializing physics...")
    app_utils.update_app(steps=10)

    robot_controller.verify_articulation()
    robot_controller.print_configuration()

    lidar = NovaCarterLidar(
        lidar_prim_path=PANDAR_XT32_LIDAR_PRIM_PATH
    )
    lidar.initialize()

    print(
        f"\nHolding the robot still for "
        f"{SETTLE_STEPS} physics steps..."
    )

    if not advance_playing_steps(
        simulation_app,
        physics_steps=SETTLE_STEPS,
        on_step=robot_controller.stop,
    ):
        return

    initial_pose = read_robot_pose(
        loaded_scene.stage
    )

    waypoints = create_world_route(
        initial_pose
    )

    waypoint_config = WaypointControllerConfig(
        maximum_linear_velocity=0.20,
        maximum_angular_velocity=0.60,
        linear_gain=0.80,
        angular_gain=1.50,
        position_tolerance=0.10,
        rotate_in_place_threshold_radians=math.radians(25.0),
    )

    waypoint_controller = WaypointController(
        waypoints,
        config=waypoint_config,
    )

    robot_from_lidar = compute_relative_prim_transform(
        loaded_scene.stage,
        source_prim_path=PANDAR_XT32_LIDAR_PRIM_PATH,
        target_prim_path=ROBOT_BASE_PRIM_PATH,
    )

    floor_height_robot = (
        estimate_floor_height_in_robot_frame(
            loaded_scene.stage
        )
    )

    detection_config = ObstacleDetectionConfig(
        minimum_height=(
            floor_height_robot
            + MINIMUM_HEIGHT_ABOVE_FLOOR
        ),
        maximum_height=(
            floor_height_robot
            + MAXIMUM_HEIGHT_ABOVE_FLOOR
        ),
        minimum_range=MINIMUM_DETECTION_RANGE,
        maximum_range=MAXIMUM_DETECTION_RANGE,
        forward_only=True,
        angular_bin_size_degrees=(
            ANGULAR_BIN_SIZE_DEGREES
        ),
    )

    print("\nWaiting for a valid LiDAR scan...")

    (
        points_sensor_frame,
        initial_frame_info,
        scan_wait_steps,
    ) = wait_for_first_scan(
        lidar,
        robot_controller,
    )

    initial_obstacle_points = extract_obstacle_points(
        points_sensor_frame,
        robot_from_lidar=robot_from_lidar,
        detection_config=detection_config,
    )

    if initial_obstacle_points.shape[0] == 0:
        raise RuntimeError(
            "The initial LiDAR scan produced no obstacle points."
        )

    print(
        "Valid scan received after "
        f"{scan_wait_steps} update(s)."
    )

    safety_config = CBFConfig(
        safety_radius=CBF_SAFETY_RADIUS,
        alpha=CBF_ALPHA,
        minimum_linear_velocity=0.0,
        maximum_linear_velocity=(
            robot_controller.config.max_linear_speed
        ),
    )

    safety_filter = CBFSafetyFilter(
        safety_config
    )

    initial_nominal_command = (
        waypoint_controller.compute_command(
            initial_pose
        )
    )

    initial_cbf_result = safety_filter.filter_command(
        nominal_linear_velocity=(
            initial_nominal_command.linear_velocity
        ),
        nominal_angular_velocity=(
            initial_nominal_command.angular_velocity
        ),
        obstacle_points_xy=initial_obstacle_points,
    )

    print_stage_a_report(
        initial_pose=initial_pose,
        waypoints=waypoints,
        frame_info=initial_frame_info,
        obstacle_points=initial_obstacle_points,
        nominal_command=initial_nominal_command,
        cbf_result=initial_cbf_result,
    )

    print(
        "\nMotion begins after "
        f"{MOTION_START_DELAY_STEPS * PHYSICS_DT:.1f} "
        "simulation seconds..."
    )

    if not advance_playing_steps(
        simulation_app,
        physics_steps=MOTION_START_DELAY_STEPS,
        on_step=robot_controller.stop,
    ):
        return

    run_safe_waypoint_navigation(
        stage=loaded_scene.stage,
        lidar=lidar,
        robot_controller=robot_controller,
        waypoint_controller=waypoint_controller,
        safety_filter=safety_filter,
        detection_config=detection_config,
        robot_from_lidar=robot_from_lidar,
        initial_obstacle_points=initial_obstacle_points,
        initial_frame_info=initial_frame_info,
        initial_cbf_result=initial_cbf_result,
    )

    robot_controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()

    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when finished.")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 70)
        print("ERROR DURING CBF-WAYPOINT INTEGRATION DEMO")
        print("=" * 70)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()