"""Inspect Nova Carter wheel radius and wheel-base geometry."""

from pathlib import Path
from statistics import median

from isaacsim import SimulationApp


# No GUI is required for this geometry inspection.
simulation_app = SimulationApp({"headless": True})


import isaacsim.core.experimental.utils.stage as stage_utils
from pxr import Gf, Usd, UsdGeom


# ---------------------------------------------------------------------
# Scene and prim paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROBOT_SCENE_PATH = (
    PROJECT_ROOT / "scenes" / "generated" / "restaurant_robot.usda"
)

ROBOT_PRIM_PATH = "/World/Robot"

LEFT_WHEEL_PATH = f"{ROBOT_PRIM_PATH}/wheel_left"
RIGHT_WHEEL_PATH = f"{ROBOT_PRIM_PATH}/wheel_right"

WHEEL_COLLIDER_PATHS = [
    f"{LEFT_WHEEL_PATH}/Colliders/Cylinder",
    f"{LEFT_WHEEL_PATH}/Colliders/Cylinder_01",
    f"{LEFT_WHEEL_PATH}/Colliders/Cylinder_02",
    f"{RIGHT_WHEEL_PATH}/Colliders/Cylinder",
    f"{RIGHT_WHEEL_PATH}/Colliders/Cylinder_01",
    f"{RIGHT_WHEEL_PATH}/Colliders/Cylinder_02",
]

# Values currently used in test_robot_motion.py.
CURRENT_CONTROLLER_RADIUS = 0.04295
CURRENT_CONTROLLER_BASE = 0.4132


def wait_for_stage_loading(max_updates=600, extra_updates=60):
    """Wait for referenced USD assets to finish loading."""

    update_count = 0

    while (
        stage_utils.is_stage_loading()
        and update_count < max_updates
    ):
        simulation_app.update()
        update_count += 1

    for _ in range(extra_updates):
        simulation_app.update()

    print(
        f"Stage loading completed after "
        f"{update_count} update(s)."
    )


def get_world_position(prim, xform_cache):
    """Return the world-space position of a prim."""

    world_transform = xform_cache.GetLocalToWorldTransform(prim)
    translation = world_transform.ExtractTranslation()

    return [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    ]


def get_world_scale(prim, xform_cache):
    """Return the absolute world scale of a prim."""

    world_matrix = xform_cache.GetLocalToWorldTransform(prim)
    transform = Gf.Transform(world_matrix)
    scale = transform.GetScale()

    return [
        abs(float(scale[0])),
        abs(float(scale[1])),
        abs(float(scale[2])),
    ]


def calculate_distance(point_a, point_b):
    """Calculate the Euclidean distance between two 3D points."""

    return sum(
        (point_b[index] - point_a[index]) ** 2
        for index in range(3)
    ) ** 0.5


def get_cylinder_radius(
    stage,
    collider_path,
    xform_cache,
    meters_per_unit,
):
    """
    Read a cylinder collider and calculate its effective radius.

    The authored radius is multiplied by the collider's composed
    transform scale before conversion to meters.
    """

    prim = stage.GetPrimAtPath(collider_path)

    if not prim.IsValid():
        print(f"[WARNING] Collider not found: {collider_path}")
        return None

    if not prim.IsA(UsdGeom.Cylinder):
        print(
            f"[WARNING] Prim is not a USD cylinder: "
            f"{collider_path} ({prim.GetTypeName()})"
        )
        return None

    cylinder = UsdGeom.Cylinder(prim)

    authored_radius = cylinder.GetRadiusAttr().Get()
    cylinder_axis = cylinder.GetAxisAttr().Get()

    if authored_radius is None:
        print(
            f"[WARNING] Cylinder has no radius: "
            f"{collider_path}"
        )
        return None

    authored_radius = float(authored_radius)
    axis = str(cylinder_axis).upper()

    world_scale = get_world_scale(prim, xform_cache)

    # A cylinder's radius lies along the two axes perpendicular
    # to its principal cylinder axis.
    if axis == "X":
        radial_scales = [world_scale[1], world_scale[2]]
    elif axis == "Y":
        radial_scales = [world_scale[0], world_scale[2]]
    else:
        radial_scales = [world_scale[0], world_scale[1]]

    average_radial_scale = sum(radial_scales) / 2.0

    radius_stage_units = (
        authored_radius * average_radial_scale
    )

    radius_meters = (
        radius_stage_units * meters_per_unit
    )

    scale_difference = abs(
        radial_scales[0] - radial_scales[1]
    )

    if scale_difference > 1e-5:
        print(
            "  [WARNING] The collider has nonuniform radial "
            f"scaling: {radial_scales}"
        )

    return {
        "path": collider_path,
        "axis": axis,
        "authored_radius": authored_radius,
        "world_scale": world_scale,
        "radius_stage_units": radius_stage_units,
        "radius_meters": radius_meters,
    }


def main():
    """Open the robot scene and measure its wheel geometry."""

    print("\n" + "=" * 70)
    print("NOVA CARTER WHEEL GEOMETRY INSPECTION")
    print("=" * 70)

    if not ROBOT_SCENE_PATH.exists():
        raise FileNotFoundError(
            "Robot scene was not found:\n"
            f"{ROBOT_SCENE_PATH}"
        )

    print(f"Scene: {ROBOT_SCENE_PATH}")

    opened, stage = stage_utils.open_stage(
        str(ROBOT_SCENE_PATH)
    )

    if not opened or stage is None:
        raise RuntimeError(
            "Could not open the robot scene."
        )

    wait_for_stage_loading()

    meters_per_unit, kilograms_per_unit = (
        stage_utils.get_stage_units()
    )

    print("\nStage information:")
    print(f"  Meters per unit:    {meters_per_unit}")
    print(f"  Kilograms per unit: {kilograms_per_unit}")
    print(f"  Up axis:            {stage_utils.get_stage_up_axis()}")

    xform_cache = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    )

    # -----------------------------------------------------------------
    # Measure the wheel base
    # -----------------------------------------------------------------

    left_wheel_prim = stage.GetPrimAtPath(
        LEFT_WHEEL_PATH
    )

    right_wheel_prim = stage.GetPrimAtPath(
        RIGHT_WHEEL_PATH
    )

    if not left_wheel_prim.IsValid():
        raise RuntimeError(
            f"Left wheel prim was not found: {LEFT_WHEEL_PATH}"
        )

    if not right_wheel_prim.IsValid():
        raise RuntimeError(
            f"Right wheel prim was not found: {RIGHT_WHEEL_PATH}"
        )

    left_center = get_world_position(
        left_wheel_prim,
        xform_cache,
    )

    right_center = get_world_position(
        right_wheel_prim,
        xform_cache,
    )

    wheel_base_stage_units = calculate_distance(
        left_center,
        right_center,
    )

    wheel_base_meters = (
        wheel_base_stage_units * meters_per_unit
    )

    center_difference = [
        right_center[index] - left_center[index]
        for index in range(3)
    ]

    print("\n" + "=" * 70)
    print("WHEEL-BASE MEASUREMENT")
    print("=" * 70)

    print(f"Left wheel center:  {left_center}")
    print(f"Right wheel center: {right_center}")
    print(f"Center difference:  {center_difference}")

    print(
        f"\nMeasured wheel base: "
        f"{wheel_base_stage_units:.6f} stage units"
    )

    print(
        f"Measured wheel base: "
        f"{wheel_base_meters:.6f} meters"
    )

    # -----------------------------------------------------------------
    # Measure cylinder collider radii
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("WHEEL-RADIUS MEASUREMENT")
    print("=" * 70)

    measured_radii = []

    for collider_path in WHEEL_COLLIDER_PATHS:
        result = get_cylinder_radius(
            stage=stage,
            collider_path=collider_path,
            xform_cache=xform_cache,
            meters_per_unit=meters_per_unit,
        )

        if result is None:
            continue

        measured_radii.append(
            result["radius_meters"]
        )

        print(f"\nCollider: {result['path']}")
        print(f"  Cylinder axis:     {result['axis']}")
        print(
            f"  Authored radius:   "
            f"{result['authored_radius']}"
        )
        print(
            f"  Composed scale:    "
            f"{result['world_scale']}"
        )
        print(
            f"  Effective radius:  "
            f"{result['radius_stage_units']:.6f} "
            "stage units"
        )
        print(
            f"  Effective radius:  "
            f"{result['radius_meters']:.6f} meters"
        )

    if not measured_radii:
        raise RuntimeError(
            "No valid cylinder collider radii were found."
        )

    # The max avoids choosing one unusual collider if the wheel
    # uses multiple collision pieces.
    measured_wheel_radius = max(measured_radii)

    # -----------------------------------------------------------------
    # Final recommended values
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MEASURED CONTROLLER PARAMETERS")
    print("=" * 70)

    print(
        f"Wheel radius: {measured_wheel_radius:.6f} meters"
    )

    print(
        f"Wheel base:   {wheel_base_meters:.6f} meters"
    )

    print("\nCurrent controller parameters:")

    print(
        f"Wheel radius: {CURRENT_CONTROLLER_RADIUS:.6f} meters"
    )

    print(
        f"Wheel base:   {CURRENT_CONTROLLER_BASE:.6f} meters"
    )

    radius_difference = abs(
        measured_wheel_radius - CURRENT_CONTROLLER_RADIUS
    )

    base_difference = abs(
        wheel_base_meters - CURRENT_CONTROLLER_BASE
    )

    print("\nAbsolute differences:")

    print(
        f"Wheel radius difference: "
        f"{radius_difference:.6f} meters"
    )

    print(
        f"Wheel-base difference:   "
        f"{base_difference:.6f} meters"
    )

    print("\nCopy these measured values into the controller:")

    print(
        f"WHEEL_RADIUS_METERS = "
        f"{measured_wheel_radius:.6f}"
    )

    print(
        f"WHEEL_BASE_METERS = "
        f"{wheel_base_meters:.6f}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 70)
        print("WHEEL GEOMETRY INSPECTION FAILED")
        print("=" * 70)
        print(error)
    finally:
        simulation_app.close()