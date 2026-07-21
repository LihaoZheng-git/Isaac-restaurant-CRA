"""Create the restaurant scene with a sensor-equipped Nova Carter robot."""

from pathlib import Path

from isaacsim import SimulationApp


# Start Isaac Sim before importing other Isaac Sim or USD modules.
simulation_app = SimulationApp({"headless": False})


import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.storage.native import get_assets_root_path
from pxr import Usd, UsdGeom


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESTAURANT_SCENE_PATH = (
    PROJECT_ROOT / "scenes" / "generated" / "restaurant_base.usda"
)

OUTPUT_SCENE_PATH = (
    PROJECT_ROOT / "scenes" / "generated" / "restaurant_robot.usda"
)


# ---------------------------------------------------------------------
# Nova Carter settings
# ---------------------------------------------------------------------

ROBOT_PRIM_PATH = "/World/Robot"

NOVA_CARTER_ASSET_PATH = (
    "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
)

# The restaurant uses centimeters:
# X = 1172 cm
# Y = 395 cm
# Z = 10 cm
ROBOT_POSITION = [1172.0, 395.0, 10.0]


def wait_for_stage_loading(max_updates=600, extra_updates=60):
    """Wait until referenced USD assets have finished loading."""

    updates = 0

    while stage_utils.is_stage_loading() and updates < max_updates:
        simulation_app.update()
        updates += 1

    # Give referenced payloads a few additional frames to compose.
    for _ in range(extra_updates):
        simulation_app.update()

    if stage_utils.is_stage_loading():
        print(
            "WARNING: The stage still reports that it is loading after "
            f"{max_updates} updates."
        )
    else:
        print(f"Stage loading completed after {updates} update(s).")

def select_enabled_variant(
    robot_prim,
    variant_set_name,
    preferred_names,
):
    """
    Select an enabled variant while respecting the names authored
    in the installed USD asset.
    """

    variant_sets = robot_prim.GetVariantSets()
    available_set_names = variant_sets.GetNames()

    if variant_set_name not in available_set_names:
        raise RuntimeError(
            f"Variant set {variant_set_name!r} was not found. "
            f"Available variant sets: {available_set_names}"
        )

    variant_set = variant_sets.GetVariantSet(variant_set_name)
    available_variants = variant_set.GetVariantNames()

    print(
        f"{variant_set_name} variants available: "
        f"{available_variants}"
    )

    # First try the known preferred names.
    for preferred_name in preferred_names:
        if preferred_name in available_variants:
            variant_set.SetVariantSelection(preferred_name)

            print(
                f"Selected {variant_set_name} variant: "
                f"{preferred_name}"
            )

            return preferred_name

    # Fall back to the first choice that does not look disabled.
    disabled_names = {
        "none",
        "off",
        "disabled",
        "no_physics",
        "no_sensors",
    }

    for available_variant in available_variants:
        normalized_name = (
            available_variant
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if normalized_name not in disabled_names:
            variant_set.SetVariantSelection(available_variant)

            print(
                f"Selected fallback {variant_set_name} variant: "
                f"{available_variant}"
            )

            return available_variant

    raise RuntimeError(
        f"Could not find an enabled choice for variant set "
        f"{variant_set_name!r}. Available choices: "
        f"{available_variants}"
    )


def print_variant_report(robot_prim):
    """Print the available and currently selected Nova Carter variants."""

    print("\n" + "=" * 70)
    print("NOVA CARTER VARIANT REPORT")
    print("=" * 70)

    variant_sets = robot_prim.GetVariantSets()
    variant_set_names = variant_sets.GetNames()

    print(f"Available variant sets: {variant_set_names}")

    for variant_set_name in variant_set_names:
        variant_set = variant_sets.GetVariantSet(variant_set_name)

        print(f"\nVariant set: {variant_set_name}")
        print(f"Available choices: {variant_set.GetVariantNames()}")
        print(f"Selected choice: {variant_set.GetVariantSelection()}")

    sensor_selection = ""

    if "Sensors" in variant_set_names:
        sensor_selection = (
            variant_sets
            .GetVariantSet("Sensors")
            .GetVariantSelection()
        )

    physics_selection = ""

    if "Physics" in variant_set_names:
        physics_selection = (
            variant_sets
            .GetVariantSet("Physics")
            .GetVariantSelection()
        )

    print("\nVerification:")

    if sensor_selection in (
        "All_Sensors",
        "sensors",
        "Sensors",
    ):
        print(
            f"[PASS] Sensors are attached using "
            f"{sensor_selection!r}."
        )
    else:
        print(
            "[WARNING] The selected sensor variant may not attach "
            f"sensors: {sensor_selection!r}."
        )

    if physics_selection in ("physx", "Physics_Base"):
        print(
            f"[PASS] Physics is enabled using "
            f"{physics_selection!r}."
        )
    else:
        print(
            "[WARNING] The selected physics variant may not enable "
            f"physics: {physics_selection!r}."
        )


def find_sensor_prims(robot_prim):
    """
    Search the Nova Carter hierarchy for camera, LiDAR, and IMU prims.

    Camera prims are detected using their USD type. LiDAR and IMU prims
    are detected using their prim names, paths, and type names.
    """

    sensor_paths = {
        "camera": set(),
        "lidar": set(),
        "imu": set(),
    }

    for prim in Usd.PrimRange(robot_prim):
        prim_path = str(prim.GetPath())
        prim_name = prim.GetName()
        prim_type = prim.GetTypeName()

        searchable_text = (
            f"{prim_path} {prim_name} {prim_type}"
        ).lower()

        # Detect actual USD Camera prims.
        if prim.IsA(UsdGeom.Camera):
            sensor_paths["camera"].add(prim_path)

        # Also recognize common Nova Carter camera naming.
        if any(
            keyword in searchable_text
            for keyword in ("camera", "hawk", "owl")
        ):
            sensor_paths["camera"].add(prim_path)

        # Recognize 2D/3D LiDAR names.
        if any(
            keyword in searchable_text
            for keyword in (
                "lidar",
                "rplidar",
                "xt32",
                "xt_32",
            )
        ):
            sensor_paths["lidar"].add(prim_path)

        # Recognize IMU names.
        if "imu" in searchable_text:
            sensor_paths["imu"].add(prim_path)

    return {
        sensor_type: sorted(paths)
        for sensor_type, paths in sensor_paths.items()
    }


def print_sensor_report(robot_prim, max_paths_per_type=12):
    """Print the sensor prims found under the Nova Carter robot."""

    sensors = find_sensor_prims(robot_prim)

    print("\n" + "=" * 70)
    print("NOVA CARTER SENSOR REPORT")
    print("=" * 70)

    labels = {
        "camera": "Camera-related prims",
        "lidar": "LiDAR-related prims",
        "imu": "IMU-related prims",
    }

    for sensor_type in ("camera", "lidar", "imu"):
        paths = sensors[sensor_type]

        print(f"\n{labels[sensor_type]}: {len(paths)}")

        for path in paths[:max_paths_per_type]:
            print(f"  - {path}")

        remaining = len(paths) - max_paths_per_type

        if remaining > 0:
            print(f"  ... and {remaining} more")

    print("\nSensors needed for our planned work:")

    if sensors["camera"]:
        print("[PASS] Camera-related prims were found.")
        print("       These can support future visual navigation.")
    else:
        print("[WARNING] No camera-related prims were found.")

    if sensors["lidar"]:
        print("[PASS] LiDAR-related prims were found.")
        print("       These can support obstacle detection and CBF testing.")
    else:
        print("[WARNING] No LiDAR-related prims were found.")

    if sensors["imu"]:
        print("[PASS] IMU-related prims were found.")
    else:
        print("[INFO] No IMU-related prims were identified by name.")

    print(
        "\nNOTE: Attached sensors are not automatically producing images, "
        "point clouds, or ROS 2 messages. We will activate and read the "
        "specific sensors in a later step."
    )


def main():
    """Load the restaurant, add Nova Carter, and save the robot scene."""

    print("\n" + "=" * 70)
    print("CREATING RESTAURANT ROBOT SCENE")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1. Validate the restaurant scene
    # -----------------------------------------------------------------

    if not RESTAURANT_SCENE_PATH.exists():
        raise FileNotFoundError(
            "Restaurant scene was not found:\n"
            f"{RESTAURANT_SCENE_PATH}"
        )

    print(f"Restaurant scene: {RESTAURANT_SCENE_PATH}")
    print(f"Output scene:     {OUTPUT_SCENE_PATH}")

    # -----------------------------------------------------------------
    # 2. Open the completed restaurant base scene
    # -----------------------------------------------------------------

    opened, stage = stage_utils.open_stage(
        str(RESTAURANT_SCENE_PATH)
    )

    if not opened or stage is None:
        raise RuntimeError(
            "Isaac Sim could not open the restaurant scene:\n"
            f"{RESTAURANT_SCENE_PATH}"
        )

    wait_for_stage_loading()

    meters_per_unit, kilograms_per_unit = (
        stage_utils.get_stage_units()
    )

    up_axis = stage_utils.get_stage_up_axis()

    print("\nRestaurant stage information:")
    print(f"  Meters per unit:    {meters_per_unit}")
    print(f"  Kilograms per unit: {kilograms_per_unit}")
    print(f"  Up axis:            {up_axis}")

    if abs(meters_per_unit - 0.01) > 1e-9:
        raise RuntimeError(
            "Unexpected restaurant stage units. "
            f"Expected 0.01 meters per unit, found {meters_per_unit}."
        )

    # -----------------------------------------------------------------
    # 3. Locate the official Isaac Sim Nova Carter asset
    # -----------------------------------------------------------------

    assets_root_path = get_assets_root_path()

    if not assets_root_path:
        raise RuntimeError(
            "Isaac Sim could not locate its asset root."
        )

    robot_usd_path = (
        assets_root_path + NOVA_CARTER_ASSET_PATH
    )

    print(f"\nNova Carter asset: {robot_usd_path}")
    print(f"Robot prim path:   {ROBOT_PRIM_PATH}")

    # -----------------------------------------------------------------
    # 4. Add Nova Carter with physics and all sensor attachments
    # -----------------------------------------------------------------

    robot_prim = stage_utils.add_reference_to_stage(
        usd_path=robot_usd_path,
        path=ROBOT_PRIM_PATH,
    )

    if not robot_prim or not robot_prim.IsValid():
        raise RuntimeError(
            f"Failed to create the robot prim at {ROBOT_PRIM_PATH}."
        )

    wait_for_stage_loading(extra_updates=60)

    selected_physics_variant = select_enabled_variant(
        robot_prim=robot_prim,
        variant_set_name="Physics",
        preferred_names=[
            "physx",          # Actual Isaac Sim 6.0 asset name
            "Physics_Base",   # Name used by some asset versions
        ],
    )

    selected_sensor_variant = select_enabled_variant(
        robot_prim=robot_prim,
        variant_set_name="Sensors",
        preferred_names=[
            "All_Sensors",
            "sensors",
            "Sensors",
        ],
    )

    # Allow the newly selected variants to compose.
    wait_for_stage_loading(extra_updates=120)

    if not robot_prim or not robot_prim.IsValid():
        raise RuntimeError(
            f"Failed to create the robot prim at {ROBOT_PRIM_PATH}."
        )

    wait_for_stage_loading(extra_updates=120)

    # -----------------------------------------------------------------
    # 5. Set the robot's initial position
    # -----------------------------------------------------------------

    robot_transform = XformPrim(ROBOT_PRIM_PATH)

    robot_transform.set_world_poses(
        positions=ROBOT_POSITION
    )

    # Update the viewport after changing the transform.
    for _ in range(30):
        simulation_app.update()

    print(f"\nRobot initial position: {ROBOT_POSITION}")
    print("Position units: centimeters")

    # -----------------------------------------------------------------
    # 6. Verify variants and attached sensors
    # -----------------------------------------------------------------

    print_variant_report(robot_prim)
    print_sensor_report(robot_prim)

    # -----------------------------------------------------------------
    # 7. Save a separate robot scene
    # -----------------------------------------------------------------

    OUTPUT_SCENE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = stage_utils.save_stage(
        str(OUTPUT_SCENE_PATH)
    )

    if not saved:
        raise RuntimeError(
            "Isaac Sim could not save the output scene:\n"
            f"{OUTPUT_SCENE_PATH}"
        )

    print("\n" + "=" * 70)
    print("SCENE CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Saved to: {OUTPUT_SCENE_PATH}")
    print(
        "\nYou can now press Play in Isaac Sim to let the robot "
        "settle onto the restaurant floor."
    )

    # -----------------------------------------------------------------
    # 8. Keep the GUI open
    # -----------------------------------------------------------------

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 70)
        print("ERROR WHILE CREATING ROBOT SCENE")
        print("=" * 70)
        print(error)

        # Keep the window available briefly for inspection.
        for _ in range(60):
            if simulation_app.is_running():
                simulation_app.update()
    finally:
        simulation_app.close()