"""Open restaurant_robot.usda in the Isaac Sim GUI for inspection.

Run from the project root:

    python -u scripts/visualization/open_robot_scene.py
"""

import sys
from pathlib import Path


def find_project_root() -> Path:
    """Find the project directory containing robot/ and scenes/."""

    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "robot").is_dir()
            and (candidate / "scenes").is_dir()
        ):
            return candidate

    raise RuntimeError("Could not find the project root.")


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# SimulationApp must be created before importing other Isaac Sim modules.
from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": False,
    }
)


# These imports must remain after SimulationApp is instantiated.
from isaacsim.core.rendering_manager import ViewportManager

from robot.loader import load_robot_scene


ROBOT_SCENE_PATH = (
    PROJECT_ROOT
    / "scenes"
    / "generated"
    / "restaurant_robot.usda"
)

ROBOT_PRIM_PATH = "/World/Robot"

INITIAL_VIEW_POSITION = [
    4.70906,
    4.53751,
    2.03889,
]

INITIAL_VIEW_TARGET = [
    14.56589,
    3.65490,
    0.60223,
]


def set_initial_view() -> None:
    """Set the initial Perspective viewport position."""

    ready, waited_frames = ViewportManager.wait_for_viewport(
        max_frames=60,
    )

    if not ready:
        print(
            "\n[WARNING] The viewport was not ready after "
            f"{waited_frames} frames."
        )
        print("Continuing with the default viewpoint.")
        return

    ViewportManager.set_camera(
        "/OmniverseKit_Persp"
    )

    perspective_camera = ViewportManager.get_camera()

    ViewportManager.set_camera_view(
        camera=perspective_camera,
        eye=INITIAL_VIEW_POSITION,
        target=INITIAL_VIEW_TARGET,
    )

    simulation_app.update()

    print(
        "\nInitial Perspective viewpoint applied successfully."
    )


def main() -> None:
    """Load the restaurant scene and keep the GUI open."""

    load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=1.0,
    )

    set_initial_view()

    print("\nRestaurant robot scene loaded successfully.")
    print(f"Scene: {ROBOT_SCENE_PATH}")
    print("You do not need to press Play.")
    print("Close the Isaac Sim window when finished.")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nFailed to open the restaurant robot scene.")
        print(error)
        raise

    finally:
        simulation_app.close()