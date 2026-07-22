"""Inspect LiDAR sensors included with Nova Carter.

This diagnostic loads restaurant_robot.usda and searches under /World/Robot
for LiDAR prims and LiDAR-related USD properties.

The script does not create, modify, or initialize a LiDAR sensor. It only
inspects the composed USD stage.

Run from the project root:

    python -u scripts/diagnostics/inspect_lidar.py
"""

import sys
import traceback
from pathlib import Path
from typing import Any


def find_project_root() -> Path:
    """Find the project containing robot/sensor and scripts."""

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


# SimulationApp must exist before importing other Isaac Sim modules.
simulation_app = SimulationApp({"headless": False})


import omni.usd
from pxr import Usd, UsdGeom

from robot.loader import load_robot_scene


ROBOT_SCENE_PATH = (
    PROJECT_ROOT
    / "scenes"
    / "generated"
    / "restaurant_robot.usda"
)

ROBOT_PRIM_PATH = "/World/Robot"

LIDAR_NAME_TERMS = (
    "lidar",
    "omnilidar",
    "hesai",
    "pandar",
    "xt32",
    "xt-32",
    "rplidar",
    "s2e",
    "laser",
)

LIDAR_PROPERTY_TERMS = (
    "lidar",
    "camerasensortype",
    "sensormodel",
    "scanrate",
    "reportrate",
    "numberofemitters",
    "numberofchannels",
    "nearrange",
    "farrange",
    "rangeresolution",
    "rangeaccuracy",
    "azimuth",
    "elevation",
    "tickrate",
)

DISPLAY_ATTRIBUTE_TERMS = (
    "lidar",
    "sensor",
    "scan",
    "report",
    "emitter",
    "channel",
    "range",
    "azimuth",
    "elevation",
    "tick",
    "config",
)


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return whether lowercase text contains one of the terms."""

    lowercase_text = text.lower()
    return any(term in lowercase_text for term in terms)


def format_value(value: Any, maximum_length: int = 180) -> str:
    """Format a USD property value without printing huge arrays."""

    value_text = repr(value)

    if len(value_text) <= maximum_length:
        return value_text

    return value_text[:maximum_length] + "... <truncated>"


def get_attribute_value(attribute: Any) -> Any:
    """Safely read a USD attribute value."""

    try:
        return attribute.Get()
    except Exception as error:
        return f"<could not read: {error}>"


def get_candidate_reasons(prim: Usd.Prim) -> list[str]:
    """Explain why a prim might represent or belong to a LiDAR."""

    reasons: list[str] = []

    prim_name = prim.GetName()
    type_name = prim.GetTypeName()
    applied_schemas = list(prim.GetAppliedSchemas())

    if contains_any(prim_name, LIDAR_NAME_TERMS):
        reasons.append(f"name={prim_name!r}")

    if contains_any(type_name, LIDAR_NAME_TERMS):
        reasons.append(f"type={type_name!r}")

    matching_schemas = [
        schema
        for schema in applied_schemas
        if contains_any(schema, LIDAR_NAME_TERMS)
    ]

    if matching_schemas:
        reasons.append(
            "schema=" + ", ".join(matching_schemas)
        )

    for attribute in prim.GetAttributes():
        attribute_name = attribute.GetName()
        lowercase_name = attribute_name.lower()

        if contains_any(
            lowercase_name,
            LIDAR_PROPERTY_TERMS,
        ):
            reasons.append(f"attribute={attribute_name!r}")
            continue

        # Older RTX LiDAR assets can be USD Camera prims whose attribute
        # values identify them as LiDAR sensors.
        if (
            "sensor" in lowercase_name
            or "config" in lowercase_name
            or "model" in lowercase_name
        ):
            value = get_attribute_value(attribute)

            if isinstance(value, str) and contains_any(
                value,
                LIDAR_NAME_TERMS,
            ):
                reasons.append(
                    f"{attribute_name}={value!r}"
                )

    # Remove duplicate reasons while preserving their order.
    return list(dict.fromkeys(reasons))


def is_primary_lidar_prim(prim: Usd.Prim) -> bool:
    """Determine whether a prim is the actual LiDAR sensor prim."""

    type_name = prim.GetTypeName().lower()

    if "omnilidar" in type_name:
        return True

    if any(
        "omnilidar" in schema.lower()
        for schema in prim.GetAppliedSchemas()
    ):
        return True

    # Compatibility with older RTX LiDAR assets represented as cameras.
    for attribute in prim.GetAttributes():
        attribute_name = attribute.GetName().lower()

        if "camerasensortype" not in attribute_name:
            continue

        value = get_attribute_value(attribute)

        if isinstance(value, str) and value.lower() == "lidar":
            return True

    return False


def get_world_position(
    prim: Usd.Prim,
    xform_cache: UsdGeom.XformCache,
) -> tuple[float, float, float] | None:
    """Return the prim's world position in stage units."""

    xformable = UsdGeom.Xformable(prim)

    if not xformable:
        return None

    try:
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()

        return (
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        )
    except Exception:
        return None


def get_relevant_attributes(
    prim: Usd.Prim,
) -> list[tuple[str, Any]]:
    """Return configuration attributes useful for LiDAR inspection."""

    relevant_attributes: list[tuple[str, Any]] = []

    for attribute in prim.GetAttributes():
        attribute_name = attribute.GetName()

        if contains_any(
            attribute_name,
            DISPLAY_ATTRIBUTE_TERMS,
        ):
            relevant_attributes.append(
                (
                    attribute_name,
                    get_attribute_value(attribute),
                )
            )

    return sorted(
        relevant_attributes,
        key=lambda item: item[0].lower(),
    )


def print_variant_selections(robot_prim: Usd.Prim) -> None:
    """Print sensor-related variant selections below the robot."""

    print("\n" + "=" * 70)
    print("SENSOR-RELATED VARIANT SELECTIONS")
    print("=" * 70)

    found_selection = False

    for prim in Usd.PrimRange(robot_prim):
        variant_sets = prim.GetVariantSets()

        for variant_set_name in variant_sets.GetNames():
            variant_set = variant_sets.GetVariantSet(
                variant_set_name
            )
            selection = variant_set.GetVariantSelection()

            combined_text = (
                f"{variant_set_name} {selection}"
            )

            if not contains_any(
                combined_text,
                (
                    "sensor",
                    "lidar",
                    "hesai",
                    "pandar",
                    "rplidar",
                    "all_sensors",
                ),
            ):
                continue

            found_selection = True

            print(f"Prim:      {prim.GetPath()}")
            print(f"Variant:   {variant_set_name}")
            print(f"Selection: {selection or '<none>'}")
            print()

    if not found_selection:
        print("No sensor-related variant selections were found.")


def print_lidar_prim(
    prim: Usd.Prim,
    reasons: list[str],
    xform_cache: UsdGeom.XformCache,
) -> None:
    """Print detailed information about one LiDAR candidate."""

    print(f"Path:            {prim.GetPath()}")
    print(f"Name:            {prim.GetName()}")
    print(f"USD type:        {prim.GetTypeName() or '<untyped>'}")
    print(f"Active:          {prim.IsActive()}")
    print(f"Loaded:          {prim.IsLoaded()}")
    print(f"Defined:         {prim.IsDefined()}")
    print(f"Instance:        {prim.IsInstance()}")

    world_position = get_world_position(prim, xform_cache)

    if world_position is not None:
        x, y, z = world_position
        print(
            "World position:  "
            f"({x:.4f}, {y:.4f}, {z:.4f}) meters"
        )
    else:
        print("World position:  <not an Xformable prim>")

    applied_schemas = list(prim.GetAppliedSchemas())

    if applied_schemas:
        print(
            "Applied schemas: "
            + ", ".join(applied_schemas)
        )
    else:
        print("Applied schemas: <none>")

    print("Matched by:")

    for reason in reasons:
        print(f"  - {reason}")

    attributes = get_relevant_attributes(prim)

    if attributes:
        print("Relevant attributes:")

        for name, value in attributes:
            print(f"  - {name} = {format_value(value)}")
    else:
        print("Relevant attributes: <none>")

    relationships = prim.GetRelationships()

    if relationships:
        print("Relationships:")

        for relationship in relationships:
            try:
                targets = relationship.GetTargets()
            except Exception as error:
                targets = [f"<could not read: {error}>"]

            print(
                f"  - {relationship.GetName()} = "
                f"{format_value(targets)}"
            )


def inspect_lidar_prims(robot_prim: Usd.Prim) -> None:
    """Find and print LiDAR prims below Nova Carter."""

    xform_cache = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    )

    primary_lidars: list[
        tuple[Usd.Prim, list[str]]
    ] = []

    related_prims: list[
        tuple[Usd.Prim, list[str]]
    ] = []

    inspected_prim_count = 0

    for prim in Usd.PrimRange(robot_prim):
        inspected_prim_count += 1
        reasons = get_candidate_reasons(prim)

        if not reasons:
            continue

        if is_primary_lidar_prim(prim):
            primary_lidars.append((prim, reasons))
        else:
            related_prims.append((prim, reasons))

    print("\n" + "=" * 70)
    print("PRIMARY LIDAR SENSOR PRIMS")
    print("=" * 70)
    print(f"Robot prims inspected: {inspected_prim_count}")
    print(f"Primary LiDAR prims:   {len(primary_lidars)}")
    print(f"Related prims:         {len(related_prims)}")

    if primary_lidars:
        for index, (prim, reasons) in enumerate(
            primary_lidars,
            start=1,
        ):
            print("\n" + "-" * 70)
            print(f"PRIMARY LIDAR {index}")
            print("-" * 70)
            print_lidar_prim(
                prim,
                reasons,
                xform_cache,
            )
    else:
        print(
            "\nNo primary OmniLidar prim was found.\n"
            "Check the related paths below. This can mean that the "
            "LiDAR sensor variant is not composed or enabled."
        )

    print("\n" + "=" * 70)
    print("OTHER LIDAR-RELATED PRIMS")
    print("=" * 70)

    if related_prims:
        for prim, reasons in related_prims:
            print(f"\nPath: {prim.GetPath()}")
            print(f"Type: {prim.GetTypeName() or '<untyped>'}")

            for reason in reasons:
                print(f"  - {reason}")
    else:
        print("No additional LiDAR-related prims were found.")

    print("\n" + "=" * 70)
    print("INSPECTION SUMMARY")
    print("=" * 70)

    if primary_lidars:
        print("Use one of these paths when we create lidar.py:")

        for prim, _ in primary_lidars:
            print(f'  "{prim.GetPath()}"')
    else:
        print(
            "No usable LiDAR sensor path was identified yet."
        )


def main() -> None:
    """Load the robot scene and inspect its LiDAR prims."""

    print("\n" + "=" * 70)
    print("NOVA CARTER LIDAR INSPECTION")
    print("=" * 70)

    loaded_scene = load_robot_scene(
        scene_path=ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=ROBOT_PRIM_PATH,
        expected_meters_per_unit=1.0,
    )

    print_variant_selections(loaded_scene.robot_prim)
    inspect_lidar_prims(loaded_scene.robot_prim)

    print("\nInspection completed.")
    print("You do not need to press Play.")
    print("The GUI will remain open for Stage inspection.")
    print("Close the Isaac Sim window when finished.")

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\n" + "=" * 70)
        print("ERROR DURING LIDAR INSPECTION")
        print("=" * 70)
        print(error)
        traceback.print_exc()

    finally:
        simulation_app.close()