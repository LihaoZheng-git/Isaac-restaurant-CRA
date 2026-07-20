"""Create the restaurant scene containing Nova Carter."""

from pathlib import Path

from isaacsim import SimulationApp


# Isaac Sim must start before importing Omniverse modules.
simulation_app = SimulationApp({"headless": False})


import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_SCENE_PATH = (
    PROJECT_ROOT
    / "scenes"
    / "generated"
    / "restaurant_base.usda"
)

OUTPUT_SCENE_PATH = (
    PROJECT_ROOT
    / "scenes"
    / "generated"
    / "restaurant_robot.usda"
)


ROBOT_PRIM_PATH = "/World/Robot"

# Restaurant stage coordinates are centimeters.
#
# The robot origin is near its chassis rather than at the bottom
# of its wheels. Starting at Z=10 allows it to fall slightly and
# settle near Z=0 when physics begins.
ROBOT_POSITION = [866.0, 308.0, 10.0]


def open_restaurant_base():
    """Open the completed restaurant base scene."""

    if not BASE_SCENE_PATH.is_file():
        raise FileNotFoundError(
            "Restaurant base scene was not found:\n"
            f"{BASE_SCENE_PATH}"
        )

    success, stage = stage_utils.open_stage(
        BASE_SCENE_PATH.as_posix()
    )

    if not success or stage is None:
        raise RuntimeError(
            "Isaac Sim could not open the restaurant base:\n"
            f"{BASE_SCENE_PATH}"
        )

    # Give the restaurant references time to begin loading.
    for _ in range(60):
        simulation_app.update()

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)

    print()
    print("=" * 70)
    print("RESTAURANT BASE LOADED")
    print("=" * 70)
    print(f"Scene: {BASE_SCENE_PATH}")
    print(f"Meters per unit: {meters_per_unit}")

    if abs(meters_per_unit - 0.01) > 1e-9:
        raise RuntimeError(
            "The restaurant stage is expected to use centimeters "
            "(metersPerUnit = 0.01)."
        )

    return stage


def add_nova_carter(stage):
    """Reference Nova Carter into the restaurant stage."""

    assets_root_path = get_assets_root_path()

    if assets_root_path is None:
        raise RuntimeError(
            "Isaac Sim could not locate its asset root."
        )

    robot_usd_path = (
        assets_root_path
        + "/Isaac/Robots/NVIDIA/NovaCarter/"
        + "nova_carter.usd"
    )

    existing_robot = stage.GetPrimAtPath(ROBOT_PRIM_PATH)

    if existing_robot.IsValid():
        raise RuntimeError(
            f"A prim already exists at {ROBOT_PRIM_PATH}."
        )

    # This unit-aware utility handles the difference between the
    # meter-based Nova Carter and our centimeter-based restaurant.
    robot_prim = stage_utils.add_reference_to_stage(
        usd_path=robot_usd_path,
        path=ROBOT_PRIM_PATH,
    )

    if not robot_prim.IsValid():
        raise RuntimeError(
            "Isaac Sim could not add the Nova Carter reference."
        )

    # Allow the referenced robot asset to load.
    for _ in range(60):
        simulation_app.update()

    # Set only the position. We intentionally leave the robot's
    # orientation and unit-conversion scale unchanged.
    robot_transform = XformPrim(ROBOT_PRIM_PATH)

    robot_transform.set_world_poses(
        positions=ROBOT_POSITION,
    )

    # Render several updates so the new pose becomes visible.
    for _ in range(30):
        simulation_app.update()

    print()
    print("=" * 70)
    print("NOVA CARTER ADDED")
    print("=" * 70)
    print(f"Asset: {robot_usd_path}")
    print(f"Prim: {ROBOT_PRIM_PATH}")
    print(f"Initial position: {ROBOT_POSITION}")
    print("User scale: 1.0")
    print(
        "Wheel joints: "
        "joint_wheel_left, joint_wheel_right"
    )

    return robot_prim


def save_robot_scene():
    """Save the generated restaurant-and-robot stage."""

    OUTPUT_SCENE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = stage_utils.save_stage(
        OUTPUT_SCENE_PATH.as_posix()
    )

    if not success:
        raise RuntimeError(
            "Isaac Sim could not save the robot scene:\n"
            f"{OUTPUT_SCENE_PATH}"
        )

    print()
    print("=" * 70)
    print("ROBOT SCENE SAVED")
    print("=" * 70)
    print(f"Output: {OUTPUT_SCENE_PATH}")
    print()
    print("Press Play to test the robot under gravity.")
    print("The robot should fall slightly and settle on the floor.")
    print("=" * 70)


def main():
    stage = open_restaurant_base()

    add_nova_carter(stage)

    save_robot_scene()

    # Keep Isaac Sim open until the window is closed.
    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()