"""Utilities for loading and validating a Nova Carter scene.

Import this module only after ``SimulationApp`` has been created.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import isaacsim.core.experimental.utils.stage as stage_utils


DEFAULT_ROBOT_PRIM_PATH = "/World/Robot"
EXPECTED_METERS_PER_UNIT = 0.01
UNIT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class LoadedRobotScene:
    """Validated stage information returned by :func:`load_robot_scene`."""

    path: Path
    stage: Any
    robot_prim: Any
    robot_prim_path: str
    meters_per_unit: float
    kilograms_per_unit: float
    up_axis: str


def wait_for_stage_loading(
    simulation_app: Any,
    *,
    max_updates: int = 600,
    extra_updates: int = 60,
) -> None:
    """Wait until referenced USD assets have finished composing."""

    if max_updates < 1:
        raise ValueError("max_updates must be at least 1.")

    if extra_updates < 0:
        raise ValueError("extra_updates cannot be negative.")

    update_count = 0

    while (
        stage_utils.is_stage_loading()
        and update_count < max_updates
        and simulation_app.is_running()
    ):
        simulation_app.update()
        update_count += 1

    for _ in range(extra_updates):
        if not simulation_app.is_running():
            return

        simulation_app.update()

    if stage_utils.is_stage_loading():
        raise TimeoutError(
            "The stage is still loading after "
            f"{max_updates} application updates."
        )

    print(
        "Stage loading completed after "
        f"{update_count} update(s)."
    )


def load_robot_scene(
    scene_path: str | Path,
    simulation_app: Any,
    *,
    robot_prim_path: str = DEFAULT_ROBOT_PRIM_PATH,
    expected_meters_per_unit: float = EXPECTED_METERS_PER_UNIT,
) -> LoadedRobotScene:
    """Open a USD scene and validate its units and robot prim."""

    resolved_path = Path(scene_path).expanduser().resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            "Robot scene was not found:\n"
            f"{resolved_path}\n\n"
            "Run create_robot_scene.py first."
        )

    print(f"Opening scene: {resolved_path}")

    opened, stage = stage_utils.open_stage(str(resolved_path))

    if not opened or stage is None:
        raise RuntimeError(
            "Isaac Sim could not open the robot scene:\n"
            f"{resolved_path}"
        )

    wait_for_stage_loading(simulation_app)

    meters_per_unit, kilograms_per_unit = (
        stage_utils.get_stage_units()
    )
    up_axis = str(stage_utils.get_stage_up_axis())

    if (
        abs(meters_per_unit - expected_meters_per_unit)
        > UNIT_TOLERANCE
    ):
        raise RuntimeError(
            "Unexpected stage units. Expected "
            f"{expected_meters_per_unit} meters per unit, found "
            f"{meters_per_unit}."
        )

    robot_prim = stage.GetPrimAtPath(robot_prim_path)

    if not robot_prim.IsValid():
        raise RuntimeError(
            f"No robot was found at {robot_prim_path}."
        )

    loaded_scene = LoadedRobotScene(
        path=resolved_path,
        stage=stage,
        robot_prim=robot_prim,
        robot_prim_path=robot_prim_path,
        meters_per_unit=float(meters_per_unit),
        kilograms_per_unit=float(kilograms_per_unit),
        up_axis=up_axis,
    )

    print_scene_report(loaded_scene)
    return loaded_scene


def print_scene_report(scene: LoadedRobotScene) -> None:
    """Print the essential properties of a loaded robot scene."""

    print("\nStage information:")
    print(f"  Scene:              {scene.path}")
    print(f"  Robot prim:         {scene.robot_prim_path}")
    print(f"  Meters per unit:    {scene.meters_per_unit}")
    print(f"  Kilograms per unit: {scene.kilograms_per_unit}")
    print(f"  Up axis:            {scene.up_axis}")
