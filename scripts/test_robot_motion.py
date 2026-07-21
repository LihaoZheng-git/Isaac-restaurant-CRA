"""Test basic Nova Carter movement inside the restaurant."""

from pathlib import Path

from isaacsim import SimulationApp


# Start Isaac Sim before importing Isaac Sim modules.
simulation_app = SimulationApp({"headless": False})


import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.experimental.wheeled_robots.controllers import (
    DifferentialController,
)
from isaacsim.robot.experimental.wheeled_robots.robots import (
    WheeledRobot,
)


# ---------------------------------------------------------------------
# Scene and robot configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROBOT_SCENE_PATH = (
    PROJECT_ROOT / "scenes" / "generated" / "restaurant_robot.usda"
)

ROBOT_PRIM_PATH = "/World/Robot"

LEFT_WHEEL_DOF = "joint_wheel_left"
RIGHT_WHEEL_DOF = "joint_wheel_right"

# Official Nova Carter geometry values.
# Controller dimensions remain in meters even though our USD stage
# is authored in centimeters.
WHEEL_RADIUS_METERS = 0.14
WHEEL_BASE_METERS = 0.4132

PHYSICS_DT = 1.0 / 60.0

# Conservative test speeds.
FORWARD_SPEED = 0.15       # meters per second
TURN_SPEED = 0.50          # radians per second

# Test durations measured in physics steps at 60 Hz.
SETTLE_STEPS = 120         # 2 seconds
FORWARD_STEPS = 180        # 3 seconds
STOP_STEPS = 60            # 1 second
TURN_STEPS = 120           # 2 seconds
FINAL_STOP_STEPS = 60      # 1 second


def wait_for_stage_loading(max_updates=600, extra_updates=60):
    """Wait for the restaurant and robot USD assets to load."""

    update_count = 0

    while (
        stage_utils.is_stage_loading()
        and update_count < max_updates
        and simulation_app.is_running()
    ):
        simulation_app.update()
        update_count += 1

    for _ in range(extra_updates):
        if not simulation_app.is_running():
            return

        simulation_app.update()

    if stage_utils.is_stage_loading():
        print(
            "WARNING: Stage still reports that it is loading "
            f"after {max_updates} updates."
        )
    else:
        print(
            f"Stage loading completed after "
            f"{update_count} update(s)."
        )


def verify_robot(robot):
    """Check that Nova Carter contains the expected wheel joints."""

    print("\n" + "=" * 70)
    print("ROBOT ARTICULATION REPORT")
    print("=" * 70)

    print(f"Robot path: {ROBOT_PRIM_PATH}")
    print(f"Number of DOFs: {robot.num_dofs}")
    print("DOF names:")

    for index, dof_name in enumerate(robot.dof_names):
        print(f"  [{index}] {dof_name}")

    missing_dofs = []

    for required_dof in (
        LEFT_WHEEL_DOF,
        RIGHT_WHEEL_DOF,
    ):
        if required_dof not in robot.dof_names:
            missing_dofs.append(required_dof)

    if missing_dofs:
        raise RuntimeError(
            "Nova Carter is missing the expected wheel DOFs: "
            f"{missing_dofs}"
        )

    print("\n[PASS] Left wheel joint was found.")
    print("[PASS] Right wheel joint was found.")
    print(
        "[PASS] Wheel ordering is "
        "[joint_wheel_left, joint_wheel_right]."
    )


def run_motion_phase(
    robot,
    controller,
    phase_name,
    command,
    physics_steps,
):
    """
    Apply one motion command for a fixed number of physics steps.

    command:
        [linear velocity in m/s, angular velocity in rad/s]
    """

    wheel_velocities = controller.forward(command)

    print("\n" + "-" * 70)
    print(f"Motion phase: {phase_name}")
    print(f"Command [linear, angular]: {command}")
    print(
        "Wheel velocities [left, right]: "
        f"{wheel_velocities}"
    )
    print(f"Physics steps: {physics_steps}")
    print("-" * 70)

    completed_steps = 0

    while (
        completed_steps < physics_steps
        and simulation_app.is_running()
    ):
        # If the user pauses the timeline, wait without advancing
        # the phase counter.
        if app_utils.is_playing():
            robot.apply_wheel_actions(wheel_velocities)
            completed_steps += 1

        simulation_app.update()

    return simulation_app.is_running()


def main():
    """Load the robot scene and run the basic movement test."""

    print("\n" + "=" * 70)
    print("NOVA CARTER MOTION TEST")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1. Open the generated restaurant robot scene
    # -----------------------------------------------------------------

    if not ROBOT_SCENE_PATH.exists():
        raise FileNotFoundError(
            "Robot scene was not found:\n"
            f"{ROBOT_SCENE_PATH}\n\n"
            "Run create_robot_scene.py first."
        )

    print(f"Opening scene: {ROBOT_SCENE_PATH}")

    opened, stage = stage_utils.open_stage(
        str(ROBOT_SCENE_PATH)
    )

    if not opened or stage is None:
        raise RuntimeError(
            "Isaac Sim could not open the robot scene:\n"
            f"{ROBOT_SCENE_PATH}"
        )

    wait_for_stage_loading()

    meters_per_unit, kilograms_per_unit = (
        stage_utils.get_stage_units()
    )

    print("\nStage information:")
    print(f"  Meters per unit:    {meters_per_unit}")
    print(f"  Kilograms per unit: {kilograms_per_unit}")
    print(f"  Up axis:            {stage_utils.get_stage_up_axis()}")

    if abs(meters_per_unit - 0.01) > 1e-9:
        raise RuntimeError(
            "Unexpected stage units. Expected centimeters "
            f"(0.01 meters per unit), found {meters_per_unit}."
        )

    robot_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)

    if not robot_prim.IsValid():
        raise RuntimeError(
            f"No robot was found at {ROBOT_PRIM_PATH}."
        )

    # -----------------------------------------------------------------
    # 2. Wrap the existing Nova Carter articulation
    # -----------------------------------------------------------------

    robot = WheeledRobot(
        paths=ROBOT_PRIM_PATH,
        wheel_dof_names=[
            LEFT_WHEEL_DOF,
            RIGHT_WHEEL_DOF,
        ],
    )

    # -----------------------------------------------------------------
    # 3. Create the differential-drive controller
    # -----------------------------------------------------------------

    controller = DifferentialController(
        wheel_radius=WHEEL_RADIUS_METERS,
        wheel_base=WHEEL_BASE_METERS,
        max_linear_speed=0.25,
        max_angular_speed=0.80,
        max_wheel_speed=10.0,
    )

    print("\nDifferential controller:")
    print(f"  Wheel radius: {WHEEL_RADIUS_METERS} meters")
    print(f"  Wheel base:   {WHEEL_BASE_METERS} meters")
    print(f"  Forward test: {FORWARD_SPEED} m/s")
    print(f"  Turning test: {TURN_SPEED} rad/s")

    # -----------------------------------------------------------------
    # 4. Initialize and start physics
    # -----------------------------------------------------------------

    SimulationManager.setup_simulation(
        dt=PHYSICS_DT,
        device="cpu",
    )

    print("\n" + "=" * 70)
    print("READY FOR MOTION TEST")
    print("=" * 70)
    print("Press Play in the Isaac Sim GUI to begin.")

    # Wait until the user presses Play.
    while (
        simulation_app.is_running()
        and not app_utils.is_playing()
    ):
        simulation_app.update()

    if not simulation_app.is_running():
        return

    print("\nPlay detected. Initializing robot physics...")

    # Allow the articulation and physics tensors to initialize.
    app_utils.update_app(steps=10)

    verify_robot(robot)

    print("\nStarting the motion sequence.")

    # -----------------------------------------------------------------
    # 5. Run the movement sequence
    # -----------------------------------------------------------------

    phases = [
        (
            "Settle on floor",
            [0.0, 0.0],
            SETTLE_STEPS,
        ),
        (
            "Move forward",
            [FORWARD_SPEED, 0.0],
            FORWARD_STEPS,
        ),
        (
            "Stop after forward movement",
            [0.0, 0.0],
            STOP_STEPS,
        ),
        (
            "Turn left in place",
            [0.0, TURN_SPEED],
            TURN_STEPS,
        ),
        (
            "Final stop",
            [0.0, 0.0],
            FINAL_STOP_STEPS,
        ),
    ]

    for phase_name, command, steps in phases:
        still_running = run_motion_phase(
            robot=robot,
            controller=controller,
            phase_name=phase_name,
            command=command,
            physics_steps=steps,
        )

        if not still_running:
            return

    # Send one final zero-velocity command.
    robot.apply_wheel_actions(
        controller.forward([0.0, 0.0])
    )

    app_utils.update_app(steps=5)
    app_utils.stop()

    print("\n" + "=" * 70)
    print("MOTION TEST COMPLETED")
    print("=" * 70)
    print("Nova Carter completed:")
    print("  1. Floor settling")
    print("  2. Forward movement")
    print("  3. Stop")
    print("  4. In-place turn")
    print("  5. Final stop")
    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when you are finished.")

    # -----------------------------------------------------------------
    # 6. Keep the GUI open after the test
    # -----------------------------------------------------------------

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 70)
        print("ERROR DURING ROBOT MOTION TEST")
        print("=" * 70)
        print(error)
    finally:
        simulation_app.close()