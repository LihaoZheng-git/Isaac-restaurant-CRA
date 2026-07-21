"""Build the meter-scale CRA restaurant base scene in Isaac Sim 6.0.1.

The NVIDIA Restaurant Demo remains outside this repository. This script
reads its location from configs/asset_paths.local.json and references the
building and restaurant assemblies into a project-owned USD stage.

The source assemblies are authored in centimeters, but the generated stage
uses meters so meter-authored robots such as Nova Carter can be added without
requiring a metric conversion inside the robot's instanced prim hierarchy.
"""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path


# ---------------------------------------------------------------------
# Lighting exposure parameters
# ---------------------------------------------------------------------

CEILING_LIGHT_EXPOSURE = 3.0
PENDANT_LIGHT_EXPOSURE = 2.0

# ---------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------

STAGE_METERS_PER_UNIT = 1.0
SOURCE_ASSET_METERS_PER_UNIT = 0.01
EXPECTED_REFERENCE_SCALE = (
    SOURCE_ASSET_METERS_PER_UNIT
    / STAGE_METERS_PER_UNIT
)

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
    / "restaurant_base_meters.usda"
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

def get_units_resolve_scale(prim) -> tuple[float, float, float] | None:
    """Return the Metrics Assembler scale authored on a reference prim."""

    scale_attribute = prim.GetAttribute(
        "xformOp:scale:unitsResolve"
    )

    if not scale_attribute:
        return None

    scale_value = scale_attribute.Get()

    if scale_value is None:
        return None

    try:
        return tuple(float(component) for component in scale_value)
    except TypeError:
        scalar_value = float(scale_value)
        return (scalar_value, scalar_value, scalar_value)


def ensure_reference_unit_conversion(
    reference_prims,
    Gf,
    UsdGeom,
) -> None:
    """Author exactly one centimeter-to-meter corrective per asset.

    Metrics Assembler automatically handles interactive drag-and-drop, but a
    raw USD reference added by this script is not guaranteed to receive that
    correction.  Reuse a correct existing ``unitsResolve`` operation when one
    is present; otherwise author it explicitly on the reference root.
    """

    expected_scale = Gf.Vec3d(
        EXPECTED_REFERENCE_SCALE,
        EXPECTED_REFERENCE_SCALE,
        EXPECTED_REFERENCE_SCALE,
    )

    for label, prim in reference_prims:
        existing_scale = get_units_resolve_scale(prim)

        if existing_scale is None:
            scale_op = UsdGeom.Xformable(prim).AddScaleOp(
                precision=UsdGeom.XformOp.PrecisionDouble,
                opSuffix="unitsResolve",
            )
            scale_op.Set(expected_scale)
            print(
                f"{label}: authored explicit "
                f"Scale:unitsResolve={tuple(expected_scale)}",
                flush=True,
            )
        else:
            print(
                f"{label}: reusing existing "
                f"Scale:unitsResolve={existing_scale}",
                flush=True,
            )


def validate_reference_unit_conversion(reference_prims) -> None:
    """Verify exactly one centimeter-to-meter corrective per asset."""

    tolerance = 1e-6

    for label, prim in reference_prims:
        scale = get_units_resolve_scale(prim)

        if scale is None:
            raise RuntimeError(
                f"{label} has no Scale:unitsResolve correction. "
                "The explicit unit conversion was not authored; "
                "the file was not saved."
            )

        if not all(
            math.isclose(
                component,
                EXPECTED_REFERENCE_SCALE,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            for component in scale
        ):
            raise RuntimeError(
                f"{label} has unexpected Scale:unitsResolve={scale}; "
                f"expected ({EXPECTED_REFERENCE_SCALE}, "
                f"{EXPECTED_REFERENCE_SCALE}, "
                f"{EXPECTED_REFERENCE_SCALE}). "
                "The file was not saved."
            )

        print(
            f"{label} unit conversion: "
            f"Scale:unitsResolve={scale}",
            flush=True,
        )


def print_environment_size(
    stage,
    Usd,
    UsdGeom,
) -> tuple[float, float, float]:
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
            "(meters): "
            f"X={values[0]:.2f}, "
            f"Y={values[1]:.2f}, "
            f"Z={values[2]:.2f}"
        )

        if max(values) > 200.0:
            raise RuntimeError(
                "The environment is unexpectedly larger than "
                "200 meters. The centimeter-to-meter correction "
                "may be missing; the file was not saved."
            )

        if max(values) < 1.0:
            raise RuntimeError(
                "The environment is unexpectedly smaller than "
                "1 meter. The unit correction may have been "
                "applied twice; the file was not saved."
            )

        return values
    else:
        raise RuntimeError(
            "Environment bounds could not be calculated; "
            "the file was not saved."
        )


def add_dark_zone_fill_lights(stage, Gf, UsdGeom, UsdLux):
    """Add RectLights for Dark Zones 1 and 2."""

    lights_root_path = "/World/Lighting/DarkZoneFillLights"
    stage.DefinePrim(lights_root_path, "Xform")

    dark_zones = {
        "DarkZone1": [
            (29.62, 16.18, 5.00),
            (31.81, 18.55, 5.00),
            (34.00, 20.92, 4.95),
        ],
        "DarkZone2": [
            (7.08, 20.20, 5.00),
            (1.87, 16.00, 5.00),
            (3.45, 19.48, 5.00),
            (4.85, 21.63, 5.00),
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

            rect_light.CreateWidthAttr(1.0)
            rect_light.CreateHeightAttr(1.0)

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

    print(
        "[checkpoint] SimulationApp created.",
        flush=True,
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

        print(
            "[checkpoint] Isaac Sim modules imported.",
            flush=True,
        )

        # -------------------------------------------------------------
        # Create the stage
        # -------------------------------------------------------------

        stage = stage_utils.create_new_stage(
            template="empty"
        )

        print(
            "[checkpoint] Empty stage created.",
            flush=True,
        )

        # The generated stage is meter-authored. This script explicitly adds
        # one Scale:unitsResolve=0.01 correction to each centimeter-authored
        # restaurant reference. Nova Carter is already authored in meters, so
        # adding it later requires no unit correction.
        stage_utils.set_stage_units(
            meters_per_unit=STAGE_METERS_PER_UNIT,
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

        # Native references keep the NVIDIA source files external. Their
        # corrective scale is authored after the references finish loading.
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

        print(
            "[checkpoint] Restaurant references added.",
            flush=True,
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
        # In this meter stage, the authored value remains 9.81.
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

        print(
            "[checkpoint] Restaurant assets loaded.",
            flush=True,
        )

        reference_prims = (
            ("BuildingShell", building_prim),
            ("Restaurant", restaurant_prim),
        )

        ensure_reference_unit_conversion(
            reference_prims,
            Gf,
            UsdGeom,
        )

        validate_reference_unit_conversion(
            reference_prims
        )

        print(
            "[checkpoint] Unit conversion verified.",
            flush=True,
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
            "Close it to finish the script.",
            flush=True,
        )

        # Keep the GUI responsive until the window is closed.
        while simulation_app.is_running():
            simulation_app.update()

    except BaseException:
        print(
            "\n[error] Restaurant scene generation failed:",
            flush=True,
        )
        traceback.print_exc()
        raise

    finally:
        if not simulation_app.is_exiting():
            simulation_app.close()


if __name__ == "__main__":
    main()