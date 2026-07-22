"""Test Nova Carter motion and RGB-camera capture in the restaurant.

The test:

1. Loads the meter-based restaurant_robot.usda scene.
2. Initializes Nova Carter's differential-drive controller.
3. Uses the front Hawk left camera.
4. Waits for the user to press Play.
5. Lets the robot settle.
6. Saves one stationary image.
7. Moves forward for approximately three seconds.
8. Captures RGB images at approximately 10 FPS.
9. Stops and saves one final image.

Run from the project root:

    python -u scripts/test/test_camera.py
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path


def find_project_root() -> Path:
    """Find the project directory containing robot/sensor and scripts."""

    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "robot").is_dir()
            and (candidate / "robot" / "sensor").is_dir()
            and (candidate / "scripts").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not find the project root containing "
        "robot/sensor/ and scripts/."
    )


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from isaacsim import SimulationApp


# SimulationApp must be created before importing other Isaac Sim modules.
simulation_app = SimulationApp({"headless": False})


import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from PIL import Image

from robot.controller import NovaCarterController
from robot.loader import load_robot_scene
from robot.sensor.camera import NovaCarterCamera


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROBOT_SCENE_PATH = (
    PROJECT_ROOT
    / "scenes"
    / "generated"
    / "restaurant_robot.usda"
)

ROBOT_PRIM_PATH = "/World/Robot"

# Create a new directory for every run so older frames are not overwritten.
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "camera_test"
    / RUN_TIMESTAMP
)


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

PHYSICS_DT = 1.0 / 60.0
PHYSICS_DEVICE = "cpu"

FORWARD_SPEED = 0.15
ANGULAR_SPEED = 0.0

# Two seconds for the robot to settle on the floor.
SETTLE_STEPS = 120

# Three seconds of forward motion at 60 physics steps per second.
FORWARD_STEPS = 180

# One second stopped after moving.
FINAL_STOP_STEPS = 60

# Capture every six physics steps:
#
#     60 physics steps per second / 6 = 10 captured frames per second
CAPTURE_EVERY_N_STEPS = 6

# Maximum additional rendered updates while waiting for the first image.
MAX_FIRST_FRAME_WAIT_STEPS = 300


def wait_for_user_to_press_play() -> bool:
    """Keep the GUI responsive until the user presses Play."""

    print("\n" + "=" * 70)
    print("READY FOR CAMERA AND MOTION TEST")
    print("=" * 70)
    print("Press Play in the Isaac Sim GUI to begin.")
    print("Leave the timeline playing until the test finishes.")

    while (
        simulation_app.is_running()
        and not app_utils.is_playing()
    ):
        simulation_app.update()

    return simulation_app.is_running()


def convert_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB frame into a PNG-compatible uint8 array."""

    array = np.asarray(rgb)

    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)

    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(
            array,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        # Some camera backends return floats from 0 to 1, while others
        # return floats from 0 to 255.
        maximum_value = float(np.max(array)) if array.size else 0.0

        if maximum_value <= 1.0:
            array = array * 255.0

    return np.ascontiguousarray(
        np.clip(array, 0.0, 255.0).astype(np.uint8)
    )


def save_rgb_frame(
    rgb: np.ndarray,
    filename: str,
) -> Path:
    """Save one RGB frame and print basic image statistics."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIRECTORY / filename
    png_rgb = convert_to_uint8(rgb)

    if png_rgb.ndim != 3 or png_rgb.shape[2] != 3:
        raise RuntimeError(
            "Cannot save the camera frame because its shape is "
            f"{png_rgb.shape}, not (height, width, 3)."
        )

    Image.fromarray(png_rgb).save(output_path)

    mean_brightness = float(np.mean(png_rgb))
    minimum_value = int(np.min(png_rgb))
    maximum_value = int(np.max(png_rgb))

    print(
        f"Saved {filename} | "
        f"shape={png_rgb.shape} | "
        f"min={minimum_value} | "
        f"max={maximum_value} | "
        f"mean={mean_brightness:.2f}"
    )

    if maximum_value <= 2:
        print(
            "  [WARNING] This frame is almost completely black."
        )

    return output_path


def run_motion_steps(
    controller: NovaCarterController,
    *,
    linear_velocity: float,
    angular_velocity: float,
    physics_steps: int,
) -> bool:
    """Run a motion command for a fixed number of playing physics steps."""

    wheel_velocities = controller.wheel_velocities(
        [linear_velocity, angular_velocity]
    )

    completed_steps = 0

    while (
        completed_steps < physics_steps
        and simulation_app.is_running()
    ):
        if app_utils.is_playing():
            controller.apply_wheel_velocities(wheel_velocities)
            simulation_app.update()
            completed_steps += 1
        else:
            # Keep the GUI responsive if the timeline is temporarily paused.
            simulation_app.update()

    return simulation_app.is_running()


def wait_for_first_rgb_frame(
    camera: NovaCarterCamera,
    controller: NovaCarterController,
) -> np.ndarray:
    """Wait for the camera pipeline to produce its first valid RGB image."""

    print("\nWaiting for the first rendered RGB frame...")

    zero_wheel_velocities = controller.wheel_velocities([0.0, 0.0])
    completed_updates = 0
    last_error: RuntimeError | None = None

    while (
        completed_updates < MAX_FIRST_FRAME_WAIT_STEPS
        and simulation_app.is_running()
    ):
        if not app_utils.is_playing():
            simulation_app.update()
            continue

        controller.apply_wheel_velocities(zero_wheel_velocities)
        simulation_app.update()
        completed_updates += 1

        try:
            rgb = camera.get_rgb()

            print(
                "First RGB frame received after "
                f"{completed_updates} rendered update(s)."
            )

            return rgb

        except RuntimeError as error:
            if "No RGB frame is available yet" not in str(error):
                raise

            last_error = error

    if not simulation_app.is_running():
        raise RuntimeError(
            "Isaac Sim was closed before the first camera frame arrived."
        )

    detail = (
        f"\nLast camera error: {last_error}"
        if last_error is not None
        else ""
    )

    raise TimeoutError(
        "No RGB frame arrived after "
        f"{MAX_FIRST_FRAME_WAIT_STEPS} rendered updates."
        f"{detail}"
    )


def capture_forward_motion(
    camera: NovaCarterCamera,
    controller: NovaCarterController,
    *,
    starting_frame_index: int,
) -> tuple[int, list[Path]]:
    """Move Nova Carter forward and capture RGB frames during motion."""

    wheel_velocities = controller.wheel_velocities(
        [FORWARD_SPEED, ANGULAR_SPEED]
    )

    print("\n" + "-" * 70)
    print("FORWARD MOTION AND IMAGE CAPTURE")
    print("-" * 70)
    print(f"Linear speed:          {FORWARD_SPEED} m/s")
    print(f"Angular speed:         {ANGULAR_SPEED} rad/s")
    print(f"Physics steps:         {FORWARD_STEPS}")
    print(f"Capture interval:      {CAPTURE_EVERY_N_STEPS} steps")
    print(
        "Expected moving frames: "
        f"{FORWARD_STEPS // CAPTURE_EVERY_N_STEPS}"
    )

    completed_steps = 0
    frame_index = starting_frame_index
    saved_paths: list[Path] = []

    while (
        completed_steps < FORWARD_STEPS
        and simulation_app.is_running()
    ):
        if not app_utils.is_playing():
            simulation_app.update()
            continue

        controller.apply_wheel_velocities(wheel_velocities)
        simulation_app.update()
        completed_steps += 1

        if completed_steps % CAPTURE_EVERY_N_STEPS != 0:
            continue

        rgb = camera.get_rgb()

        filename = f"frame_{frame_index:04d}_moving.png"
        saved_path = save_rgb_frame(rgb, filename)

        saved_paths.append(saved_path)
        frame_index += 1

    return frame_index, saved_paths


def main() -> None:
    """Run the complete robot-motion and camera-capture test."""

    print("\n" + "=" * 70)
    print("NOVA CARTER CAMERA AND MOTION TEST")
    print("=" * 70)

    loaded_scene = load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=1.0,
    )

    print(
        f"\nValidated scene units: "
        f"{loaded_scene.meters_per_unit} m/unit"
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

    # Allow Isaac Sim to initialize the articulation and physics handles.
    app_utils.update_app(steps=10)

    controller.verify_articulation()

    print("\nInitializing the front Hawk left camera...")

    # NovaCarterCamera() uses front_hawk/left/camera_left by default.
    camera = NovaCarterCamera()
    camera.initialize()

    print(
        f"\nLetting Nova Carter settle for "
        f"{SETTLE_STEPS} physics steps..."
    )

    if not run_motion_steps(
        controller,
        linear_velocity=0.0,
        angular_velocity=0.0,
        physics_steps=SETTLE_STEPS,
    ):
        return

    # ------------------------------------------------------------------
    # Capture the stationary reference image.
    # ------------------------------------------------------------------

    stationary_rgb = wait_for_first_rgb_frame(
        camera,
        controller,
    )

    saved_images: list[Path] = [
        save_rgb_frame(
            stationary_rgb,
            "frame_0000_stationary.png",
        )
    ]

    # ------------------------------------------------------------------
    # Move forward and capture the image sequence.
    # ------------------------------------------------------------------

    next_frame_index, moving_images = capture_forward_motion(
        camera,
        controller,
        starting_frame_index=1,
    )

    saved_images.extend(moving_images)

    if not simulation_app.is_running():
        return

    # ------------------------------------------------------------------
    # Stop and capture the final image.
    # ------------------------------------------------------------------

    print(
        f"\nStopping Nova Carter for "
        f"{FINAL_STOP_STEPS} physics steps..."
    )

    if not run_motion_steps(
        controller,
        linear_velocity=0.0,
        angular_velocity=0.0,
        physics_steps=FINAL_STOP_STEPS,
    ):
        return

    final_rgb = camera.get_rgb()

    saved_images.append(
        save_rgb_frame(
            final_rgb,
            f"frame_{next_frame_index:04d}_stopped.png",
        )
    )

    controller.stop()
    app_utils.update_app(steps=5)
    app_utils.stop()

    frame_info = camera.get_frame_info()

    print("\n" + "=" * 70)
    print("CAMERA AND MOTION TEST COMPLETED")
    print("=" * 70)
    print(f"Camera prim:       {frame_info.camera_prim_path}")
    print(
        "Final frame shape: "
        f"({frame_info.height}, "
        f"{frame_info.width}, "
        f"{frame_info.channels})"
    )
    print(f"Frame data type:   {frame_info.dtype}")
    print(f"Images saved:      {len(saved_images)}")
    print(f"Output directory:  {OUTPUT_DIRECTORY}")
    print("\nExpected result:")
    print("  - The first image shows the stationary robot view.")
    print("  - The middle images change as the robot moves forward.")
    print("  - The final image shows the stopped robot view.")
    print("\nThe GUI will remain open for inspection.")
    print("Close the Isaac Sim window when you are finished.")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\n" + "=" * 70)
        print("ERROR DURING CAMERA AND MOTION TEST")
        print("=" * 70)
        print(error)
        traceback.print_exc()

    finally:
        simulation_app.close()