"""DEMO basic Nova Carter movement inside the restaurant."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from robot.config import (
    PHYSICS_DEVICE,
    PHYSICS_DT,
    ROBOT_PRIM_PATH,
    ROBOT_SCENE_PATH,
)


from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})


import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager

from robot.control.differential_drive import NovaCarterController
from robot.scene.loader import load_robot_scene
from robot.runtime.simulation_loop import (
    advance_playing_steps,
    wait_until_playing,
)



FORWARD_SPEED = 0.15
TURN_SPEED = 0.50

SETTLE_STEPS = 120
FORWARD_STEPS = 180
STOP_STEPS = 60
TURN_STEPS = 120
FINAL_STOP_STEPS = 60


def wait_for_user_to_press_play() -> bool:
    """Wait until the user starts the timeline."""

    print("Press Play in the Isaac Sim GUI to begin.")
    return wait_until_playing(simulation_app)


def run_motion_phase(
    controller: NovaCarterController,
    *,
    phase_name: str,
    linear_velocity: float,
    angular_velocity: float,
    physics_steps: int,
) -> bool:
    """Apply one command for a fixed number of playing physics steps."""

    wheel_velocities = controller.wheel_velocities(
        [linear_velocity, angular_velocity]
    )

    print("\n" + "-" * 70)
    print(f"Motion phase: {phase_name}")
    print(
        "Command [linear m/s, angular rad/s]: "
        f"[{linear_velocity}, {angular_velocity}]"
    )
    print(
        "Wheel velocities [left, right] rad/s: "
        f"{wheel_velocities}"
    )
    print(f"Physics steps: {physics_steps}")
    print("-" * 70)

    def apply_wheel_command() -> None:
        controller.apply_wheel_velocities(
            wheel_velocities
        )

    return advance_playing_steps(
        simulation_app,
        physics_steps=physics_steps,
        on_step=apply_wheel_command,
    )


def main() -> None:
    """Load the robot scene and run the movement test."""

    print("\n" + "=" * 70)
    print("NOVA CARTER MOTION DEMO")
    print("=" * 70)

    load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
    )

    controller = NovaCarterController(
        robot_prim_path=ROBOT_PRIM_PATH
    )
    controller.print_configuration()

    SimulationManager.setup_simulation(
        dt=PHYSICS_DT,
        device=PHYSICS_DEVICE,
    )

    if not wait_for_user_to_press_play():
        return

    print("\nPlay detected. Initializing robot physics...")
    app_utils.update_app(steps=10)

    controller.verify_articulation()
    print("\nStarting the motion sequence.")

    phases = [
        ("Settle on floor", 0.0, 0.0, SETTLE_STEPS),
        ("Move forward", FORWARD_SPEED, 0.0, FORWARD_STEPS),
        ("Stop after forward movement", 0.0, 0.0, STOP_STEPS),
        ("Turn left in place", 0.0, TURN_SPEED, TURN_STEPS),
        ("Final stop", 0.0, 0.0, FINAL_STOP_STEPS),
    ]

    for phase_name, linear, angular, steps in phases:
        still_running = run_motion_phase(
            controller,
            phase_name=phase_name,
            linear_velocity=linear,
            angular_velocity=angular,
            physics_steps=steps,
        )

        if not still_running:
            return

    controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()

    print("\n" + "=" * 70)
    print("MOTION DEMO COMPLETED")
    print("=" * 70)
    print("Nova Carter moved forward, stopped, turned, and stopped.")
    print("The GUI will remain open for inspection.")
    print("Close the Isaac Sim window when you are finished.")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 70)
        print("ERROR DURING ROBOT MOTION DEMO")
        print("=" * 70)
        print(error)
    finally:
        simulation_app.close()