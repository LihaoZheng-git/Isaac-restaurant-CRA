"""Validate nominal waypoint navigation in the restaurant scene.

This demo validates the connection between the pure waypoint controller
and Nova Carter in Isaac Sim.

Stage A -- stationary validation:

1. Loads the restaurant robot scene.
2. Keeps Nova Carter stationary while physics settles.
3. Reads the chassis world-frame planar pose.
4. Creates a waypoint relative to the robot's initial heading.
5. Computes and reports the first nominal command without moving.

Stage B -- waypoint motion:

1. Repeatedly reads the current chassis pose.
2. Computes a nominal waypoint command.
3. Applies that command through Nova Carter's differential controller.
4. Stops when the waypoint is reached or the time limit expires.
5. Reports the final pose and navigation result.

This demo intentionally does not use LiDAR, CBF, or CRA. Its output is the
nominal control command that will later be passed to the safety layer.

Make sure the path ahead of the robot is clear.

Run from the project root:

    python -u scripts/demos/waypoint_navigation_demo.py

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
    PHYSICS_DEVICE,
    PHYSICS_DT,
    ROBOT_BASE_PRIM_PATH,
    ROBOT_PRIM_PATH,
    ROBOT_SCENE_PATH,
)

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})


import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd

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
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import get_prim_planar_pose


SETTLE_STEPS = 60
MOTION_START_DELAY_STEPS = 120

MAXIMUM_NAVIGATION_DURATION_SECONDS = 12.0
MAXIMUM_NAVIGATION_STEPS = math.ceil(
    MAXIMUM_NAVIGATION_DURATION_SECONDS / PHYSICS_DT
)

STATUS_REPORT_INTERVAL_STEPS = 30

# Each entry is (forward, left) in the robot's initial frame.
#
# Keep the first test simple: one waypoint one meter straight ahead.
# These offsets are converted to fixed world-frame waypoints before
# the robot begins moving.
ROUTE_LOCAL_OFFSETS = (
    (1.00, 0.00),
)


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


def print_stage_a_report(
    *,
    initial_pose: RobotPose2D,
    waypoints: tuple[Waypoint, ...],
    command: WaypointCommand,
) -> None:
    """Print the stationary pose and initial-command validation."""

    print("\n" + "=" * 70)
    print("WAYPOINT STAGE A - STATIONARY VALIDATION")
    print("=" * 70)

    print("\nInitial chassis pose:")
    print(f"  x:   {initial_pose.x:.4f} m")
    print(f"  y:   {initial_pose.y:.4f} m")
    print(
        "  yaw: "
        f"{math.degrees(initial_pose.yaw):.2f} degrees"
    )

    print("\nWorld-frame route:")

    for index, waypoint in enumerate(waypoints):
        print(
            f"  [{index}] "
            f"x={waypoint.x:.4f} m, "
            f"y={waypoint.y:.4f} m"
        )

    print("\nInitial nominal command:")
    print(f"  Mode:             {command.mode.value}")
    print(
        "  Linear velocity:  "
        f"{command.linear_velocity:.4f} m/s"
    )
    print(
        "  Angular velocity: "
        f"{command.angular_velocity:.4f} rad/s"
    )
    print(
        "  Target distance:  "
        f"{command.distance_to_waypoint:.4f} m"
    )
    print(
        "  Heading error:    "
        f"{math.degrees(command.heading_error):.2f} degrees"
    )

    print("\nExpected interpretation:")

    if abs(command.heading_error) < math.radians(5.0):
        print("  [PASS] The waypoint is approximately straight ahead.")
    else:
        print(
            "  [WARNING] The initial heading error is larger "
            "than expected."
        )

    if command.linear_velocity > 0.0:
        print("  [PASS] The controller requests forward motion.")
    else:
        print(
            "  [INFO] The controller will rotate before driving."
        )

    print("  The robot remained stationary during Stage A.")
    print("=" * 70)


def print_live_status(
    *,
    completed_steps: int,
    pose: RobotPose2D,
    command: WaypointCommand,
) -> None:
    """Print one compact navigation status line."""

    waypoint_number = (
        command.target_waypoint_index + 1
        if command.target_waypoint_index is not None
        else 0
    )

    print(
        f"  step={completed_steps:<4} "
        f"target={waypoint_number:<2} "
        f"pose=({pose.x:.3f}, {pose.y:.3f}) "
        f"distance={command.distance_to_waypoint:.3f} m "
        f"heading={math.degrees(command.heading_error):6.2f} deg "
        f"v={command.linear_velocity:.3f} m/s "
        f"w={command.angular_velocity:.3f} rad/s "
        f"mode={command.mode.value}"
    )


def run_navigation(
    *,
    stage: Usd.Stage,
    robot_controller: NovaCarterController,
    waypoint_controller: WaypointController,
) -> None:
    """Run the closed-loop nominal waypoint controller."""

    completed_steps = 0
    termination_reason = "Maximum navigation duration reached."

    previous_mode = None
    previous_target_index = (
        waypoint_controller.current_waypoint_index
    )

    latest_pose = read_robot_pose(stage)
    latest_command = waypoint_controller.compute_command(
        latest_pose
    )

    print("\n" + "=" * 70)
    print("WAYPOINT STAGE B - CLOSED-LOOP MOTION")
    print("=" * 70)
    print(
        "Maximum duration: "
        f"{MAXIMUM_NAVIGATION_DURATION_SECONDS:.1f} s"
    )
    print(
        "Position tolerance: "
        f"{waypoint_controller.config.position_tolerance:.3f} m"
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
        latest_command = waypoint_controller.compute_command(
            latest_pose
        )

        current_target_index = (
            latest_command.target_waypoint_index
        )

        target_changed = (
            current_target_index != previous_target_index
        )
        mode_changed = latest_command.mode != previous_mode
        periodic_report = (
            completed_steps % STATUS_REPORT_INTERVAL_STEPS == 0
        )

        if (
            periodic_report
            or target_changed
            or mode_changed
            or latest_command.route_complete
        ):
            print_live_status(
                completed_steps=completed_steps,
                pose=latest_pose,
                command=latest_command,
            )

        if latest_command.waypoint_advanced:
            if latest_command.route_complete:
                print("\nFinal waypoint reached.")
            else:
                print(
                    "\nAdvanced to waypoint "
                    f"{latest_command.target_waypoint_index + 1}."
                )

        if latest_command.route_complete:
            termination_reason = (
                "Every waypoint was reached successfully."
            )
            robot_controller.stop()
            break

        robot_controller.apply_command(
            latest_command.linear_velocity,
            latest_command.angular_velocity,
        )

        previous_mode = latest_command.mode
        previous_target_index = current_target_index

        simulation_app.update()
        completed_steps += 1

    robot_controller.stop()
    app_utils.update_app(steps=5)

    if simulation_app.is_running():
        latest_pose = read_robot_pose(stage)
        latest_command = waypoint_controller.compute_command(
            latest_pose
        )

        if latest_command.route_complete:
            termination_reason = (
                "Every waypoint was reached successfully."
            )

    print("\n" + "=" * 70)
    print("WAYPOINT STAGE B RESULT")
    print("=" * 70)
    print(f"Reason: {termination_reason}")
    print(f"Completed physics steps: {completed_steps}")

    print("\nFinal chassis pose:")
    print(f"  x:   {latest_pose.x:.4f} m")
    print(f"  y:   {latest_pose.y:.4f} m")
    print(
        "  yaw: "
        f"{math.degrees(latest_pose.yaw):.2f} degrees"
    )

    print("\nFinal controller state:")
    print(f"  Route complete: {latest_command.route_complete}")
    print(f"  Mode:           {latest_command.mode.value}")

    if latest_command.route_complete:
        print("  Remaining distance: 0.0000 m")
        print("\n[PASS] Waypoint navigation completed.")
    else:
        print(
            "  Remaining distance: "
            f"{latest_command.distance_to_waypoint:.4f} m"
        )
        print(
            "\n[WARNING] Navigation stopped before the route "
            "was completed."
        )

    print("=" * 70)


def main() -> None:
    """Run stationary validation followed by waypoint motion."""

    print("\n" + "=" * 70)
    print("NOVA CARTER WAYPOINT NAVIGATION DEMO")
    print("=" * 70)
    print("This demo applies nominal commands without a safety filter.")

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

    waypoints = create_world_route(initial_pose)

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

    initial_command = waypoint_controller.compute_command(
        initial_pose
    )

    print_stage_a_report(
        initial_pose=initial_pose,
        waypoints=waypoints,
        command=initial_command,
    )

    print(
        "\nStage B motion begins after "
        f"{MOTION_START_DELAY_STEPS * PHYSICS_DT:.1f} "
        "simulation seconds..."
    )

    if not advance_playing_steps(
        simulation_app,
        physics_steps=MOTION_START_DELAY_STEPS,
        on_step=robot_controller.stop,
    ):
        return

    run_navigation(
        stage=loaded_scene.stage,
        robot_controller=robot_controller,
        waypoint_controller=waypoint_controller,
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
        print("ERROR DURING WAYPOINT NAVIGATION DEMO")
        print("=" * 70)
        print(error)
        traceback.print_exc()
    finally:
        simulation_app.close()