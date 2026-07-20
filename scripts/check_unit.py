"""Inspect the units and coordinate metadata of the restaurant USD files."""

from __future__ import annotations

import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "asset_paths.local.json"

BUILDING_RELATIVE_PATH = (
    Path("Assemblies") / "assembly_BaseBuildingL0.usd"
)
RESTAURANT_RELATIVE_PATH = (
    Path("Assemblies") / "assembly_Restaurant.usd"
)


def load_asset_paths() -> tuple[Path, Path]:
    """Load and validate the two assembly paths."""

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    restaurant_pack_root = Path(config["restaurant_pack_root"])

    building_path = restaurant_pack_root / BUILDING_RELATIVE_PATH
    restaurant_path = restaurant_pack_root / RESTAURANT_RELATIVE_PATH

    for usd_path in (building_path, restaurant_path):
        if not usd_path.is_file():
            raise FileNotFoundError(f"USD file not found: {usd_path}")

    return building_path.resolve(), restaurant_path.resolve()


def classify_unit(meters_per_unit: float) -> str:
    """Give common meters-per-unit values readable names."""

    known_units = {
        1.0: "meters",
        0.01: "centimeters",
        0.001: "millimeters",
        0.0254: "inches",
        0.3048: "feet",
    }

    for known_value, unit_name in known_units.items():
        if math.isclose(
            meters_per_unit,
            known_value,
            rel_tol=1e-6,
        ):
            return unit_name

    return "custom or unknown unit"


def format_vector(vector) -> str:
    """Format a USD vector for terminal output."""

    values = [f"{float(component):.3f}" for component in vector]
    return f"({', '.join(values)})"


def inspect_usd(label, usd_path, Usd, UsdGeom):
    """Open one USD file and report its metadata."""

    print()
    print("=" * 70)
    print(label)
    print("=" * 70)
    print(f"File: {usd_path}")

    stage = Usd.Stage.Open(
        usd_path.as_posix(),
        load=Usd.Stage.LoadAll,
    )

    if stage is None:
        raise RuntimeError(f"Could not open: {usd_path}")

    meters_per_unit = float(
        UsdGeom.GetStageMetersPerUnit(stage)
    )

    meters_per_unit_authored = bool(
        UsdGeom.StageHasAuthoredMetersPerUnit(stage)
    )

    up_axis = str(
        UsdGeom.GetStageUpAxis(stage)
    ).upper()

    up_axis_authored = bool(
        stage.HasAuthoredMetadata("upAxis")
    )

    print(
        f"Meters per unit: {meters_per_unit:g} "
        f"({classify_unit(meters_per_unit)})"
    )
    print(
        f"Meters-per-unit explicitly authored: "
        f"{meters_per_unit_authored}"
    )
    print(f"Up axis: {up_axis}")
    print(f"Up axis explicitly authored: {up_axis_authored}")

    default_prim = stage.GetDefaultPrim()

    if default_prim.IsValid():
        print(f"Default prim: {default_prim.GetPath()}")
    else:
        print("Default prim: NOT SET")

    root_prims = list(stage.GetPseudoRoot().GetChildren())

    root_prim_names = [
        str(prim.GetPath())
        for prim in root_prims
    ]

    print(
        "Root prims: "
        + (
            ", ".join(root_prim_names)
            if root_prim_names
            else "none"
        )
    )

    target_prim = None

    if default_prim.IsValid():
        target_prim = default_prim
    elif len(root_prims) == 1:
        target_prim = root_prims[0]

    if target_prim is not None:
        xformable = UsdGeom.Xformable(target_prim)

        if xformable:
            transform_operations = (
                xformable.GetOrderedXformOps()
            )

            if transform_operations:
                print("Root transform operations:")

                for operation in transform_operations:
                    print(
                        f"  {operation.GetOpName()}: "
                        f"{operation.Get()}"
                    )
            else:
                print("Root transform operations: none")

        bounds_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
            ],
            useExtentsHint=True,
        )

        world_bounds = bounds_cache.ComputeWorldBound(
            target_prim
        )

        aligned_bounds = world_bounds.ComputeAlignedBox()

        bounds_size = aligned_bounds.GetSize()
        bounds_center = aligned_bounds.GetMidpoint()

        physical_size = [
            float(component) * meters_per_unit
            for component in bounds_size
        ]

        print(
            "Approximate bounds size: "
            f"{format_vector(bounds_size)} stage units"
        )
        print(
            "Approximate bounds center: "
            f"{format_vector(bounds_center)}"
        )
        print(
            "Approximate physical size: "
            f"{format_vector(physical_size)} meters"
        )

    composition_errors = list(
        stage.GetCompositionErrors()
    )

    print(
        f"Composition errors: "
        f"{len(composition_errors)}"
    )

    for error in composition_errors[:10]:
        print(f"  - {error}")

    return {
        "meters_per_unit": meters_per_unit,
        "up_axis": up_axis,
    }


def compare_assets(building_report, restaurant_report):
    """Compare the metadata reported by the two assemblies."""

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    building_mpu = building_report["meters_per_unit"]
    restaurant_mpu = restaurant_report["meters_per_unit"]

    scale_ratio = restaurant_mpu / building_mpu

    print(
        "Restaurant MPU / building MPU: "
        f"{scale_ratio:g}"
    )

    if math.isclose(scale_ratio, 1.0, rel_tol=1e-6):
        print("Unit metadata matches.")
    else:
        print("Unit metadata DOES NOT match.")
        print(
            "Possible restaurant scale correction: "
            f"{scale_ratio:g}"
        )

    building_axis = building_report["up_axis"]
    restaurant_axis = restaurant_report["up_axis"]

    if building_axis == restaurant_axis:
        print("Up-axis metadata matches.")
    else:
        print("Up-axis metadata DOES NOT match.")
        print(f"Building up axis: {building_axis}")
        print(f"Restaurant up axis: {restaurant_axis}")


def main():
    building_path, restaurant_path = load_asset_paths()

    # SimulationApp must start before importing pxr modules.
    from isaacsim.simulation_app import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
        }
    )

    try:
        # Pylance may not understand pxr's dynamically exposed modules.
        from pxr import Usd, UsdGeom  # pyright: ignore[reportAttributeAccessIssue]

        building_report = inspect_usd(
            "BUILDING ASSEMBLY",
            building_path,
            Usd,
            UsdGeom,
        )

        restaurant_report = inspect_usd(
            "RESTAURANT ASSEMBLY",
            restaurant_path,
            Usd,
            UsdGeom,
        )

        compare_assets(
            building_report,
            restaurant_report,
        )

    finally:
        if not simulation_app.is_exiting():
            simulation_app.close()


if __name__ == "__main__":
    main()