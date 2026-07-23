"""Reusable Isaac Sim timeline-loop helpers.

Import this module only after SimulationApp has been created.
"""

from collections.abc import Callable
from typing import Any

import isaacsim.core.experimental.utils.app as app_utils


def wait_until_playing(simulation_app: Any) -> bool:
    """Keep the application responsive until the timeline plays."""

    while simulation_app.is_running():
        if app_utils.is_playing():
            return True

        simulation_app.update()

    return False


def advance_playing_steps(
    simulation_app: Any,
    *,
    physics_steps: int,
    on_step: Callable[[], None] | None = None,
) -> bool:
    """Advance a fixed number of playing simulation steps."""

    if physics_steps < 0:
        raise ValueError("physics_steps cannot be negative.")

    completed_steps = 0

    while (
        completed_steps < physics_steps
        and simulation_app.is_running()
    ):
        if not app_utils.is_playing():
            simulation_app.update()
            continue

        if on_step is not None:
            on_step()

        simulation_app.update()
        completed_steps += 1

    return completed_steps == physics_steps