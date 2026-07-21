"""Build the CRA restaurant base scene in Isaac Sim 6.0.1.

The NVIDIA Restaurant Demo remains outside this repository. This script
reads its location from configs/asset_paths.local.json and references the
building and restaurant assemblies into a project-owned USD stage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


# ---------------------------------------------------------------------
# Lighting exposure parameters
# ---------------------------------------------------------------------

CEILING_LIGHT_EXPOSURE = 3.0
PENDANT_LIGHT_EXPOSURE = 2.0

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "asset_paths.local.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "scenes"
    / "generated"
    / "restaurant_base.usda"
)

BUILDING_ASSEMBLY = (
    Path("Assemblies")
    / "assembly_BaseBuildingL0.usd"
)

RESTAURANT_ASSEMBLY = (
    Path("Assemblies")
    / "assembly_Restaurant.usd"
)


# ---------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------

def load_asset_paths() -> tuple[Path, Path]:
    """Read the local configuration and validate the required USD files."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Missing local configuration: {CONFIG_PATH}\n"
            "Create configs/asset_paths.local.json first."
        )

    try:
        with CONFIG_PATH.open(
            "r",
            encoding="utf-8",
        ) as config_file:
            config = json.load(config_file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {CONFIG_PATH}: {error}"
        ) from error

    root_value = config.get(
        "restaurant_pack_root"
    )

    if (
        not isinstance(root_value, str)
        or not root_value.strip()
    ):
        raise ValueError(
            f'{CONFIG_PATH} must contain a non-empty '
            '"restaurant_pack_root".'
        )

    restaurant_pack_root = Path(
        root_value
    ).expanduser()

    building_usd = (
        restaurant_pack_root
        / BUILDING_ASSEMBLY
    )

    restaurant_usd = (
        restaurant_pack_root
        / RESTAURANT_ASSEMBLY
    )

    missing_files = [
        usd_path
        for usd_path in (
            building_usd,
            restaurant_usd,
        )
        if not usd_path.is_file()
    ]

    if missing_files:
        formatted_paths = "\n".join(
            f"  - {usd_path}"
            for usd_path in missing_files
        )

        raise FileNotFoundError(
            "The following Restaurant Demo files "
            "were not found:\n"
            f"{formatted_paths}\n"
            f"Check restaurant_pack_root in "
            f"{CONFIG_PATH}."
        )

    return (
        building_usd.resolve(),
        restaurant_usd.resolve(),
    )


# ---------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------

def wait_for_assets(
    simulation_app,
    stage_utils,
) -> None:
    """Wait for referenced USD assets to finish loading."""

    minimum_frames = 30
    maximum_frames = 600

    for frame in range(maximum_frames):
        simulation_app.update()

        if (
            frame >= minimum_frames
            and not stage_utils.is_stage_loading()
        ):
            return

    print(
        "Warning: asset loading is still reported "
        "after 600 frames."
    )


def configure_original_ceiling_lights(stage):
    """Adjust the main ceiling and pendant lights."""

    updated_ceiling_lights = 0
    updated_pendant_lights = 0

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())

        exposure_attribute = prim.GetAttribute(
            "inputs:exposure"
        )

        if not exposure_attribute:
            continue

        if (
            "/ClgLights_Front/" in prim_path
            or "/ClgLights_Back/" in prim_path
        ):
            exposure_attribute.Set(
                CEILING_LIGHT_EXPOSURE
            )

            updated_ceiling_lights += 1

        elif (
            "/assets_restaurant_pendants/"
            in prim_path
        ):
            exposure_attribute.Set(
                PENDANT_LIGHT_EXPOSURE
            )

            updated_pendant_lights += 1

    print(
        "Updated "
        f"{updated_ceiling_lights} ceiling lights."
    )

    print(
        "Updated "
        f"{updated_pendant_lights} pendant lights."
    )

# ---------------------------------------------------------------------
# Scene validation
# ---------------------------------------------------------------------

def print_environment_size(
    stage,
    Usd,
    UsdGeom,
) -> None:
    """Print the environment size as a simple scale check."""

    environment_prim = stage.GetPrimAtPath(
        "/World/Environment"
    )

    bounding_box_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
        ],
        useExtentsHint=True,
    )

    world_bounds = (
        bounding_box_cache.ComputeWorldBound(
            environment_prim
        )
    )

    aligned_bounds = (
        world_bounds.ComputeAlignedBox()
    )

    size = aligned_bounds.GetSize()

    values = tuple(
        float(component)
        for component in size
    )

    if all(
        math.isfinite(component)
        for component in values
    ):
        print(
            "Approximate environment size "
            "(stage units): "
            f"X={values[0]:.2f}, "
            f"Y={values[1]:.2f}, "
            f"Z={values[2]:.2f}"
        )
    else:
        print(
            "Warning: environment bounds could "
            "not be calculated."
        )


def add_dark_zone_fill_lights(stage, Gf, UsdGeom, UsdLux):
    """Add RectLights for Dark Zones 1 and 2."""

    lights_root_path = "/World/Lighting/DarkZoneFillLights"
    stage.DefinePrim(lights_root_path, "Xform")

    dark_zones = {
        "DarkZone1": [
            (2962, 1618, 500),
            (3181, 1855, 500),
            (3400, 2092, 495),
        ],
        "DarkZone2": [
            (708, 2020, 500),
            (187, 1600, 500),
            (345, 1948, 500),
            (485, 2163, 500),
        ],
    }

    number_of_lights = 0

    for zone_name, light_positions in dark_zones.items():
        zone_path = f"{lights_root_path}/{zone_name}"
        stage.DefinePrim(zone_path, "Xform")

        for light_index, position in enumerate(
            light_positions,
            start=1,
        ):
            light_path = (
                f"{zone_path}/FillLight_{light_index:02d}"
            )

            rect_light = UsdLux.RectLight.Define(stage, light_path)

            # Same settings used for Dark Zone 1.
            rect_light.CreateIntensityAttr(1.0)
            rect_light.CreateExposureAttr(15.0)

            rect_light.CreateWidthAttr(100.0)
            rect_light.CreateHeightAttr(100.0)

            rect_light.CreateColorAttr(
                Gf.Vec3f(1.0, 1.0, 1.0)
            )
            rect_light.CreateNormalizeAttr(False)
            rect_light.CreateEnableColorTemperatureAttr(False)

            light_xform = UsdGeom.Xformable(
                rect_light.GetPrim()
            )

            light_xform.AddTranslateOp(
                precision=UsdGeom.XformOp.PrecisionDouble
            ).Set(
                Gf.Vec3d(*position)
            )

            number_of_lights += 1

    print(
        f"Added {number_of_lights} dark-zone RectLights "
        "with intensity 1.0 and exposure 15.0."
    )


# ---------------------------------------------------------------------
# Main scene construction
# ---------------------------------------------------------------------

def main() -> None:
    """Create and display the restaurant scene."""

    # Validate the filesystem paths before starting Isaac Sim.
    building_usd, restaurant_usd = (
        load_asset_paths()
    )

    # SimulationApp must start before importing pxr and other
    # Omniverse runtime modules.
    from isaacsim.simulation_app import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": False,
            "width": 1280,
            "height": 720,
        }
    )

    try:
        import isaacsim.core.experimental.utils.stage as stage_utils

        # Pylance may not understand pxr's dynamically exposed modules.
        from pxr import (  # pyright: ignore[reportAttributeAccessIssue]
            Gf,
            Usd,
            UsdGeom,
            UsdLux,
            UsdPhysics,
        )

        # -------------------------------------------------------------
        # Create the stage
        # -------------------------------------------------------------

        stage = stage_utils.create_new_stage(
            template="empty"
        )

        # Both NVIDIA assemblies use centimeter coordinates.
        #
        # The building relies on USD's default 0.01 value, while the
        # restaurant explicitly authors 0.01. Declaring the combined
        # stage as centimeters prevents Metrics Assembler from adding
        # a scale:unitsResolve transform only to the restaurant.
        stage_utils.set_stage_units(
            meters_per_unit=0.01,
            kilograms_per_unit=1.0,
        )

        stage_utils.set_stage_up_axis("Z")

        world_prim = stage.GetPrimAtPath(
            "/World"
        )

        if not world_prim.IsValid():
            raise RuntimeError(
                "The new stage does not contain "
                "a valid /World prim."
            )

        stage.SetDefaultPrim(world_prim)

        stage_utils.define_prim(
            "/World/Environment",
            "Xform",
        )

        stage_utils.define_prim(
            "/World/Lighting",
            "Xform",
        )

        # -------------------------------------------------------------
        # Reference the NVIDIA restaurant assemblies
        # -------------------------------------------------------------

        # Native USD references preserve the authored coordinates.
        building_prim = stage.DefinePrim(
            "/World/Environment/BuildingShell",
            "Xform",
        )

        building_reference_added = (
            building_prim
            .GetReferences()
            .AddReference(
                building_usd.as_posix()
            )
        )

        restaurant_prim = stage.DefinePrim(
            "/World/Environment/Restaurant",
            "Xform",
        )

        restaurant_reference_added = (
            restaurant_prim
            .GetReferences()
            .AddReference(
                restaurant_usd.as_posix()
            )
        )

        if (
            not building_prim.IsValid()
            or not building_reference_added
        ):
            raise RuntimeError(
                "Could not create the building reference."
            )

        if (
            not restaurant_prim.IsValid()
            or not restaurant_reference_added
        ):
            raise RuntimeError(
                "Could not create the restaurant reference."
            )

        # -------------------------------------------------------------
        # Lighting
        # -------------------------------------------------------------

        # Soft ambient light for the indoor environment.
        ambient_light = UsdLux.DomeLight.Define(
            stage,
            "/World/Lighting/AmbientLight",
        )

        ambient_light.CreateIntensityAttr(
            350.0
        )

        ambient_light.CreateColorAttr(
            Gf.Vec3f(
                1.0,
                0.97,
                0.92,
            )
        )

        # Directional fill light.
        fill_light = UsdLux.DistantLight.Define(
            stage,
            "/World/Lighting/FillLight",
        )

        fill_light.CreateIntensityAttr(
            500.0
        )

        fill_light.CreateAngleAttr(
            1.0
        )

        fill_light.CreateColorAttr(
            Gf.Vec3f(
                1.0,
                0.95,
                0.88,
            )
        )

        fill_light_transform = UsdGeom.Xformable(
            fill_light.GetPrim()
        )

        fill_light_transform.AddRotateXYZOp().Set(
            Gf.Vec3f(
                -45.0,
                0.0,
                30.0,
            )
        )

        # -------------------------------------------------------------
        # Physics
        # -------------------------------------------------------------

        meters_per_unit = (
            UsdGeom.GetStageMetersPerUnit(stage)
        )

        if meters_per_unit <= 0.0:
            raise RuntimeError(
                "The stage reports an invalid "
                "meters-per-unit value."
            )

        physics_scene = UsdPhysics.Scene.Define(
            stage,
            "/World/PhysicsScene",
        )

        physics_scene.CreateGravityDirectionAttr(
            Gf.Vec3f(
                0.0,
                0.0,
                -1.0,
            )
        )

        # Convert 9.81 meters per second squared into stage units.
        # For a centimeter stage, this becomes 981 cm/s².
        physics_scene.CreateGravityMagnitudeAttr(
            9.81 / meters_per_unit
        )

        # -------------------------------------------------------------
        # Wait for assets and validate the stage
        # -------------------------------------------------------------

        wait_for_assets(
            simulation_app,
            stage_utils,
        )


        configure_original_ceiling_lights(stage)

        add_dark_zone_fill_lights(
            stage,
            Gf,
            UsdGeom,
            UsdLux,
        )

        (
            meters_per_unit,
            kilograms_per_unit,
        ) = stage_utils.get_stage_units()

        print(
            "Stage units: "
            f"{meters_per_unit} m/unit, "
            f"{kilograms_per_unit} kg/unit"
        )

        print(
            "Stage up axis: "
            f"{stage_utils.get_stage_up_axis()}"
        )

        print_environment_size(
            stage,
            Usd,
            UsdGeom,
        )

        # -------------------------------------------------------------
        # Save the generated scene
        # -------------------------------------------------------------

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_succeeded = stage_utils.save_stage(
            OUTPUT_PATH.as_posix()
        )

        if not save_succeeded:
            raise RuntimeError(
                f"Isaac Sim could not save "
                f"{OUTPUT_PATH}"
            )

        print(
            f"Saved generated scene to: "
            f"{OUTPUT_PATH}"
        )

        print(
            "The Isaac Sim window will stay open. "
            "Close it to finish the script."
        )

        # Keep the GUI responsive until the window is closed.
        while simulation_app.is_running():
            simulation_app.update()

    finally:
        if not simulation_app.is_exiting():
            simulation_app.close()


if __name__ == "__main__":
    main()