"""Opt-in translating field window, retaining a fixed world-aligned anchor.

Only overlapping cells have a temporal derivative after a shift. Newly entered
cells are explicitly invalid until a second real source observes that domain;
neither scalar nor batched control sampling may treat them as zero dynamics.
"""

from dataclasses import dataclass, replace
import math

import numpy as np

from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.safe_sage_temporal_derivative import (
    SafeSAGETemporalDerivativeResult,
    _shift_spatial_stack,
    _validate_field_pair,
)


@dataclass(frozen=True)
class RollingLocalField:
    base: LocalOccupancyGridConfig
    step_m: float = 1.0

    def __post_init__(self):
        cells = self.step_m / self.base.resolution
        if (not math.isfinite(cells) or cells < 1
                or not math.isclose(cells, round(cells), abs_tol=1e-9)):
            raise ValueError("rolling step must be a positive integer number of cells")
        if self.step_m >= min(-self.base.minimum_x, self.base.maximum_x,
                              -self.base.minimum_y, self.base.maximum_y):
            raise ValueError("rolling step must retain an interior robot domain")

    def select(self, position_xy, current):
        position = np.asarray(position_xy, dtype=float)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("rolling position must be finite xy")
        offset = np.array([current.minimum_x - self.base.minimum_x,
                           current.minimum_y - self.base.minimum_y])
        move = np.abs(position - offset) >= self.step_m
        offset[move] = np.round(position[move] / self.step_m) * self.step_m
        return replace(self.base,
                       minimum_x=self.base.minimum_x + offset[0],
                       maximum_x=self.base.maximum_x + offset[0],
                       minimum_y=self.base.minimum_y + offset[1],
                       maximum_y=self.base.maximum_y + offset[1])


def rolling_temporal_derivative(estimator, previous_h, current_h, *, previous_grid,
                                current_grid, time_delta_seconds, previous_estimate=None):
    """Compare actual consecutive fields at identical physical cell locations.

    The domain-boundary change remains in dh/dt; old fields are not re-solved
    with new boundary conditions. Smoothing retains only valid aligned history.
    Source clocks and the estimator's existing maximum interval are unchanged.
    """
    _validate_field_pair(previous_h, current_h, grid_config=current_grid)
    if previous_grid.shape != current_grid.shape or previous_grid.resolution != current_grid.resolution:
        raise ValueError("rolling fields must preserve shape and resolution")
    dt = float(time_delta_seconds)
    if not math.isfinite(dt) or not (estimator.config.minimum_time_delta_seconds <= dt
                                     <= estimator.config.maximum_time_delta_seconds):
        raise ValueError("rolling temporal source interval outside existing limits")
    offset = np.array([current_grid.minimum_x - previous_grid.minimum_x,
                       current_grid.minimum_y - previous_grid.minimum_y]) / current_grid.resolution
    if not np.allclose(offset, np.round(offset), atol=1e-8, rtol=0):
        raise ValueError("rolling field origins must align to whole cells")
    offset = np.round(offset)
    aligned, overlap = _shift_spatial_stack(previous_h.values, offset_cells_xy=offset)
    valid = np.broadcast_to(overlap, current_h.shape).copy()
    if not valid.any():
        raise ValueError("rolling fields have no temporal overlap")
    raw = np.zeros_like(current_h.values)
    raw[valid] = (current_h.values[valid] - aligned[valid]) / dt
    alpha = estimator.config.smoothing_factor
    values = alpha * raw
    if previous_estimate is not None:
        if (not isinstance(previous_estimate, SafeSAGETemporalDerivativeResult)
                or previous_estimate.grid_config != previous_grid
                or previous_estimate.shape != current_h.shape
                or not np.allclose(previous_estimate.yaw_angles_radians,
                                   current_h.yaw_angles_radians, atol=1e-12, rtol=0)):
            raise ValueError("rolling temporal history does not match its original grid")
        history, _ = _shift_spatial_stack(previous_estimate.values, offset_cells_xy=offset)
        history_valid, _ = _shift_spatial_stack(previous_estimate.valid_mask.astype(float),
                                               offset_cells_xy=offset)
        use = valid & (history_valid >= 1 - 1e-12)
        values[use] += (1 - alpha) * history[use]
    return SafeSAGETemporalDerivativeResult(
        values=values, raw_values=raw, valid_mask=valid,
        yaw_angles_radians=current_h.yaw_angles_radians, grid_config=current_grid,
        time_delta_seconds=dt, egomotion_translation_grid_xy=tuple(offset * current_grid.resolution),
        smoothing_factor=alpha, require_valid_samples=True)
