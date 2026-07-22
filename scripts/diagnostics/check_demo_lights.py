"""Inspect lights contained in the original NVIDIA Restaurant Demo USD files.

This script is read-only. It does not modify any of the NVIDIA USD files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------
# Project and asset paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "asset_paths.local.json"
)

BUILDING_RELATIVE_PATH = (
    Path("Assemblies")
    / "assembly_BaseBuildingL0.usd"
)

RESTAURANT_RELATIVE_PATH = (
    Path("Assemblies")
    / "assembly_Restaurant.usd"
)

# Depending on the demo-pack version, the world file may use either name.
WORLD_FILE_NAMES = (
    "World_RestaurantDemopack.usd",
    "World_Restaurant.usd",
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def load_restaurant_pack_root() -> Path:
    """Read restaurant_pack_root from the local JSON configuration."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    root_value = config.get("restaurant_pack_root")

    if not isinstance(root_value, str) or not root_value.strip():
        raise ValueError(
            f'{CONFIG_PATH} must contain a non-empty '
            '"restaurant_pack_root".'
        )

    restaurant_pack_root = Path(
        root_value
    ).expanduser()

    if not restaurant_pack_root.is_dir():
        raise FileNotFoundError(
            "Restaurant Demo directory not found: "
            f"{restaurant_pack_root}"
        )

    return restaurant_pack_root.resolve()


def collect_usd_files(
    restaurant_pack_root: Path,
) -> list[tuple[str, Path]]:
    """Collect the original USD files that should be inspected."""

    usd_files: list[tuple[str, Path]] = []

    restaurant_path = (
        restaurant_pack_root
        / RESTAURANT_RELATIVE_PATH
    )

    building_path = (
        restaurant_pack_root
        / BUILDING_RELATIVE_PATH
    )

    if restaurant_path.is_file():
        usd_files.append(
            (
                "RESTAURANT ASSEMBLY",
                restaurant_path,
            )
        )

    if building_path.is_file():
        usd_files.append(
            (
                "BUILDING ASSEMBLY",
                building_path,
            )
        )

    for world_file_name in WORLD_FILE_NAMES:
        world_path = (
            restaurant_pack_root
            / world_file_name
        )

        if world_path.is_file():
            usd_files.append(
                (
                    "FULL RESTAURANT DEMO",
                    world_path,
                )
            )

            # Only use the first matching world filename.
            break

    if not usd_files:
        raise FileNotFoundError(
            "No Restaurant Demo USD files were found."
        )

    return usd_files


# ---------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------

def get_attribute_value(
    prim,
    *possible_names: str,
):
    """Read the first valid attribute from a list of possible names."""

    for attribute_name in possible_names:
        attribute = prim.GetAttribute(
            attribute_name
        )

        if attribute and attribute.IsValid():
            value = attribute.Get()

            if value is not None:
                return value

    return None


def is_light_prim(
    prim,
    UsdLux,
) -> bool:
    """Return True when a prim represents a USD light."""

    type_name = str(prim.GetTypeName())

    # This detects typed lights such as RectLight and DomeLight.
    if type_name.endswith("Light"):
        return True

    # This detects lights represented through applied API schemas,
    # including possible mesh lights.
    try:
        if prim.HasAPI(UsdLux.LightAPI):
            return True
    except Exception:
        pass

    try:
        if prim.HasAPI(UsdLux.MeshLightAPI):
            return True
    except Exception:
        pass

    return False


def format_vector(vector) -> str:
    """Format a USD vector for terminal output."""

    if vector is None:
        return "not set"

    try:
        return (
            "("
            + ", ".join(
                f"{float(component):.3f}"
                for component in vector
            )
            + ")"
        )
    except TypeError:
        return str(vector)


# ---------------------------------------------------------------------
# Light reporting
# ---------------------------------------------------------------------

def print_light(
    prim,
    meters_per_unit: float,
    transform_cache,
    UsdGeom,
) -> None:
    """Print the important properties of a light prim."""

    light_path = prim.GetPath()
    light_type = str(prim.GetTypeName())

    if not light_type:
        light_type = "API-based light"

    world_transform = (
        transform_cache.GetLocalToWorldTransform(
            prim
        )
    )

    position_stage_units = (
        world_transform.ExtractTranslation()
    )

    position_meters = [
        float(component) * meters_per_unit
        for component in position_stage_units
    ]

    imageable = UsdGeom.Imageable(prim)

    if imageable:
        visibility = imageable.ComputeVisibility()
    else:
        visibility = "not available"

    intensity = get_attribute_value(
        prim,
        "inputs:intensity",
        "intensity",
    )

    exposure = get_attribute_value(
        prim,
        "inputs:exposure",
        "exposure",
    )

    color = get_attribute_value(
        prim,
        "inputs:color",
        "color",
    )

    normalize = get_attribute_value(
        prim,
        "inputs:normalize",
        "normalize",
    )

    enable_temperature = get_attribute_value(
        prim,
        "inputs:enableColorTemperature",
        "enableColorTemperature",
    )

    color_temperature = get_attribute_value(
        prim,
        "inputs:colorTemperature",
        "colorTemperature",
    )

    width = get_attribute_value(
        prim,
        "inputs:width",
        "width",
    )

    height = get_attribute_value(
        prim,
        "inputs:height",
        "height",
    )

    radius = get_attribute_value(
        prim,
        "inputs:radius",
        "radius",
    )

    length = get_attribute_value(
        prim,
        "inputs:length",
        "length",
    )

    angle = get_attribute_value(
        prim,
        "inputs:angle",
        "angle",
    )

    texture_file = get_attribute_value(
        prim,
        "inputs:texture:file",
        "texture:file",
    )

    cone_angle = get_attribute_value(
        prim,
        "inputs:shaping:cone:angle",
        "shaping:cone:angle",
    )

    cone_softness = get_attribute_value(
        prim,
        "inputs:shaping:cone:softness",
        "shaping:cone:softness",
    )

    print()
    print(f"Light path: {light_path}")
    print(f"Type: {light_type}")
    print(f"Active: {prim.IsActive()}")
    print(f"Visibility: {visibility}")

    print(
        "Position in stage units: "
        f"{format_vector(position_stage_units)}"
    )

    print(
        "Position in meters: "
        f"{format_vector(position_meters)}"
    )

    print(f"Intensity: {intensity}")
    print(f"Exposure: {exposure}")
    print(f"Color: {format_vector(color)}")
    print(f"Normalize: {normalize}")

    if enable_temperature is not None:
        print(
            "Color temperature enabled: "
            f"{enable_temperature}"
        )

    if color_temperature is not None:
        print(
            "Color temperature: "
            f"{color_temperature}"
        )

    if width is not None:
        print(f"Width: {width}")

    if height is not None:
        print(f"Height: {height}")

    if radius is not None:
        print(f"Radius: {radius}")

    if length is not None:
        print(f"Length: {length}")

    if angle is not None:
        print(f"Angle: {angle}")

    if texture_file is not None:
        print(f"Texture file: {texture_file}")

    if cone_angle is not None:
        print(f"Shaping cone angle: {cone_angle}")

    if cone_softness is not None:
        print(
            "Shaping cone softness: "
            f"{cone_softness}"
        )


def inspect_stage(
    label: str,
    usd_path: Path,
    Usd,
    UsdGeom,
    UsdLux,
) -> None:
    """Open one USD and report every active light."""

    print()
    print("=" * 80)
    print(label)
    print("=" * 80)
    print(f"File: {usd_path}")

    stage = Usd.Stage.Open(
        usd_path.as_posix(),
        load=Usd.Stage.LoadAll,
    )

    if stage is None:
        print("Could not open this USD file.")
        return

    meters_per_unit = float(
        UsdGeom.GetStageMetersPerUnit(stage)
    )

    print(
        "Meters per unit: "
        f"{meters_per_unit}"
    )

    transform_cache = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    )

    lights = []

    # stage.Traverse() searches all active and loaded prims.
    for prim in stage.Traverse():
        if is_light_prim(prim, UsdLux):
            lights.append(prim)

    print(f"Number of active lights found: {len(lights)}")

    if not lights:
        print(
            "No active UsdLux lights were found. "
            "Visible fixtures may only use emissive materials."
        )
        return

    light_type_counts = Counter(
        str(light.GetTypeName()) or "API-based light"
        for light in lights
    )

    print("Light types:")

    for light_type, count in sorted(
        light_type_counts.items()
    ):
        print(f"  {light_type}: {count}")

    for light_number, light_prim in enumerate(
        lights,
        start=1,
    ):
        print()
        print("-" * 80)
        print(
            f"LIGHT {light_number} OF {len(lights)}"
        )
        print("-" * 80)

        print_light(
            light_prim,
            meters_per_unit,
            transform_cache,
            UsdGeom,
        )

    composition_errors = list(
        stage.GetCompositionErrors()
    )

    print()
    print(
        "Composition errors: "
        f"{len(composition_errors)}"
    )

    for error in composition_errors[:10]:
        print(f"  - {error}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    restaurant_pack_root = (
        load_restaurant_pack_root()
    )

    usd_files = collect_usd_files(
        restaurant_pack_root
    )

    # Isaac Sim must start before importing pxr modules.
    from isaacsim.simulation_app import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
        }
    )

    try:
        # Pylance may not understand pxr's dynamically exposed modules.
        from pxr import (  # pyright: ignore[reportAttributeAccessIssue]
            Usd,
            UsdGeom,
            UsdLux,
        )

        for label, usd_path in usd_files:
            inspect_stage(
                label,
                usd_path,
                Usd,
                UsdGeom,
                UsdLux,
            )

    finally:
        if not simulation_app.is_exiting():
            simulation_app.close()


if __name__ == "__main__":
    main()