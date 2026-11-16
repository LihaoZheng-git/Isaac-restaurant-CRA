"""Motion-compensated temporal derivatives for Safe-SAGE ``h_full``.

Safe-SAGE estimates explicit safety variation from consecutive field solves:

    dh/dt(i, j, l) =
        (h_current(i, j, l) - h_previous(i + dy, j + dx, l)) / dt

where ``(dx, dy)`` compensates for robot ego-motion between robot-centric
grids.  This module implements the same operation with bilinear sub-cell
sampling, followed by the paper's first-order low-pass filtering.  A caller
whose fields already share one fixed frame supplies zero ego-motion.

The module is pure NumPy and has no Isaac Sim dependency.
"""

import math
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np

from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.safe_sage_h_full import SafeSAGEHFullResult


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _finite_vector_xy(value: object, *, name: str) -> np.ndarray:
    """Return one finite planar vector."""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,):
        raise ValueError(f"{name} must have shape (2,).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return np.ascontiguousarray(result, dtype=np.float64)


def _frozen_array(value: object, *, dtype: np.dtype) -> np.ndarray:
    """Return an owned, immutable, contiguous NumPy array."""

    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SafeSAGETemporalDerivativeConfig:
    """Timing and first-order smoothing settings."""

    smoothing_factor: float = 0.35
    minimum_time_delta_seconds: float = 1e-4
    maximum_time_delta_seconds: float = 0.35

    def __post_init__(self) -> None:
        smoothing = _finite_real(
            self.smoothing_factor,
            name="smoothing_factor",
        )
        minimum_delta = _finite_real(
            self.minimum_time_delta_seconds,
            name="minimum_time_delta_seconds",
        )
        maximum_delta = _finite_real(
            self.maximum_time_delta_seconds,
            name="maximum_time_delta_seconds",
        )
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing_factor must lie in (0, 1].")
        if minimum_delta <= 0.0:
            raise ValueError(
                "minimum_time_delta_seconds must be greater than zero."
            )
        if maximum_delta < minimum_delta:
            raise ValueError(
                "maximum_time_delta_seconds cannot be smaller than "
                "minimum_time_delta_seconds."
            )
        object.__setattr__(self, "smoothing_factor", smoothing)
        object.__setattr__(
            self,
            "minimum_time_delta_seconds",
            minimum_delta,
        )
        object.__setattr__(
            self,
            "maximum_time_delta_seconds",
            maximum_delta,
        )


@dataclass(frozen=True)
class SafeSAGETemporalDerivativeResult:
    """One aligned and filtered ``dh_full/dt`` volume."""

    values: np.ndarray
    raw_values: np.ndarray
    valid_mask: np.ndarray
    yaw_angles_radians: np.ndarray
    grid_config: LocalOccupancyGridConfig
    time_delta_seconds: float
    egomotion_translation_grid_xy: tuple[float, float]
    smoothing_factor: float
    require_valid_samples: bool = False

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        raw_values = np.asarray(self.raw_values, dtype=np.float64)
        valid_mask = np.asarray(self.valid_mask, dtype=bool)
        yaw_angles = np.asarray(
            self.yaw_angles_radians,
            dtype=np.float64,
        )
        if values.ndim != 3:
            raise ValueError("values must have shape (yaw, rows, columns).")
        if raw_values.shape != values.shape:
            raise ValueError("raw_values must match values shape.")
        if valid_mask.shape != values.shape:
            raise ValueError("valid_mask must match values shape.")
        if yaw_angles.shape != (values.shape[0],):
            raise ValueError("yaw_angles_radians must match the yaw axis.")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must contain finite values.")
        if not np.all(np.isfinite(raw_values)):
            raise ValueError("raw_values must contain finite values.")
        if not np.all(np.isfinite(yaw_angles)):
            raise ValueError("yaw_angles_radians must contain finite values.")
        if not isinstance(self.grid_config, LocalOccupancyGridConfig):
            raise TypeError("grid_config must be a LocalOccupancyGridConfig.")
        if values.shape[1:] != self.grid_config.shape:
            raise ValueError("Temporal field shape must match grid_config.")
        time_delta = _finite_real(
            self.time_delta_seconds,
            name="time_delta_seconds",
        )
        if time_delta <= 0.0:
            raise ValueError("time_delta_seconds must be greater than zero.")
        egomotion = _finite_vector_xy(
            self.egomotion_translation_grid_xy,
            name="egomotion_translation_grid_xy",
        )
        smoothing = _finite_real(
            self.smoothing_factor,
            name="smoothing_factor",
        )
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing_factor must lie in (0, 1].")
        object.__setattr__(self, "values", _frozen_array(values, dtype=np.float64))
        object.__setattr__(
            self,
            "raw_values",
            _frozen_array(raw_values, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "valid_mask",
            _frozen_array(valid_mask, dtype=np.bool_),
        )
        object.__setattr__(
            self,
            "yaw_angles_radians",
            _frozen_array(yaw_angles, dtype=np.float64),
        )
        object.__setattr__(self, "time_delta_seconds", time_delta)
        object.__setattr__(
            self,
            "egomotion_translation_grid_xy",
            (float(egomotion[0]), float(egomotion[1])),
        )
        object.__setattr__(self, "smoothing_factor", smoothing)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(yaw slices, rows, columns)``."""

        return self.values.shape

    @property
    def maximum_absolute_derivative(self) -> float:
        """Return the largest filtered derivative magnitude."""

        return float(np.max(np.abs(self.values)))

    @property
    def valid_fraction(self) -> float:
        """Return the fraction retained after ego-motion alignment."""

        return float(np.mean(self.valid_mask))

    def sample(
        self,
        query_position_grid_xy: np.ndarray,
        query_yaw_radians: float,
    ) -> float:
        """Trilinearly sample the temporal field at one configuration."""

        point = _finite_vector_xy(
            query_position_grid_xy,
            name="query_position_grid_xy",
        )
        yaw = _finite_real(
            query_yaw_radians,
            name="query_yaw_radians",
        )
        lower_yaw, upper_yaw, yaw_weight = _yaw_interpolation(
            yaw,
            yaw_slice_count=self.shape[0],
        )
        if self.require_valid_samples:
            validity = [
                _bilinear_sample_cell_centers(self.valid_mask[index].astype(float),
                                             point_grid_xy=point, config=self.grid_config)
                for index in (lower_yaw, upper_yaw)
            ]
            if (1 - yaw_weight) * validity[0] + yaw_weight * validity[1] < 1 - 1e-12:
                raise ValueError("temporal query has no valid consecutive-source overlap")
        lower_value = _bilinear_sample_cell_centers(
            self.values[lower_yaw],
            point_grid_xy=point,
            config=self.grid_config,
        )
        upper_value = _bilinear_sample_cell_centers(
            self.values[upper_yaw],
            point_grid_xy=point,
            config=self.grid_config,
        )
        return float(
            (1.0 - yaw_weight) * lower_value
            + yaw_weight * upper_value
        )


TemporalSafetyDerivativeInput: TypeAlias = (
    float | SafeSAGETemporalDerivativeResult
)


class SafeSAGETemporalDerivativeEstimator:
    """Estimate and low-pass filter consecutive ``h_full`` volumes."""

    def __init__(
        self,
        config: SafeSAGETemporalDerivativeConfig | None = None,
    ) -> None:
        self.config = config or SafeSAGETemporalDerivativeConfig()
        if not isinstance(self.config, SafeSAGETemporalDerivativeConfig):
            raise TypeError(
                "config must be a SafeSAGETemporalDerivativeConfig."
            )

    def estimate(
        self,
        *,
        previous_h_full: SafeSAGEHFullResult,
        current_h_full: SafeSAGEHFullResult,
        grid_config: LocalOccupancyGridConfig,
        time_delta_seconds: float,
        egomotion_translation_grid_xy: object = (0.0, 0.0),
        previous_estimate: SafeSAGETemporalDerivativeResult | None = None,
    ) -> SafeSAGETemporalDerivativeResult:
        """Return the paper-style motion-compensated finite difference.

        ``egomotion_translation_grid_xy`` is the robot translation from the
        previous grid frame to the current grid frame, resolved in current
        grid axes and expressed in metres.  Thus a current cell samples the
        previous volume at ``(x + delta_x, y + delta_y)``.
        """

        _validate_field_pair(
            previous_h_full,
            current_h_full,
            grid_config=grid_config,
        )
        time_delta = _finite_real(
            time_delta_seconds,
            name="time_delta_seconds",
        )
        if time_delta < self.config.minimum_time_delta_seconds:
            raise ValueError(
                "time_delta_seconds is smaller than the configured minimum."
            )
        if time_delta > self.config.maximum_time_delta_seconds:
            raise ValueError(
                "time_delta_seconds exceeds the configured maximum."
            )
        egomotion = _finite_vector_xy(
            egomotion_translation_grid_xy,
            name="egomotion_translation_grid_xy",
        )
        offset_cells_xy = egomotion / grid_config.resolution
        aligned_previous, valid_spatial = _shift_spatial_stack(
            previous_h_full.values,
            offset_cells_xy=offset_cells_xy,
        )
        valid_mask = np.broadcast_to(
            valid_spatial[np.newaxis, ...],
            current_h_full.values.shape,
        ).copy()
        raw = np.zeros_like(current_h_full.values, dtype=np.float64)
        raw[valid_mask] = (
            current_h_full.values[valid_mask]
            - aligned_previous[valid_mask]
        ) / time_delta

        if previous_estimate is None:
            # Initialize the first-order filter at zero rather than exposing
            # one unsmoothed frame when temporal history first becomes
            # available.
            filtered = self.config.smoothing_factor * raw
        else:
            _validate_previous_estimate(
                previous_estimate,
                current_h_full=current_h_full,
                grid_config=grid_config,
            )
            aligned_filtered, filtered_valid_spatial = _shift_spatial_stack(
                previous_estimate.values,
                offset_cells_xy=offset_cells_xy,
            )
            filtered_valid = np.broadcast_to(
                filtered_valid_spatial[np.newaxis, ...],
                current_h_full.values.shape,
            )
            valid_mask &= filtered_valid
            alpha = self.config.smoothing_factor
            filtered = np.zeros_like(raw, dtype=np.float64)
            filtered[valid_mask] = (
                alpha * raw[valid_mask]
                + (1.0 - alpha) * aligned_filtered[valid_mask]
            )

        return SafeSAGETemporalDerivativeResult(
            values=filtered,
            raw_values=raw,
            valid_mask=valid_mask,
            yaw_angles_radians=current_h_full.yaw_angles_radians,
            grid_config=grid_config,
            time_delta_seconds=time_delta,
            egomotion_translation_grid_xy=(
                float(egomotion[0]),
                float(egomotion[1]),
            ),
            smoothing_factor=self.config.smoothing_factor,
        )


def sample_temporal_safety_derivative(
    temporal_derivative: TemporalSafetyDerivativeInput,
    *,
    query_position_grid_xy: np.ndarray,
    query_yaw_radians: float,
    h_full: SafeSAGEHFullResult,
    grid_config: LocalOccupancyGridConfig,
) -> float:
    """Return a scalar derivative while preserving legacy scalar inputs."""

    if isinstance(temporal_derivative, SafeSAGETemporalDerivativeResult):
        _validate_previous_estimate(
            temporal_derivative,
            current_h_full=h_full,
            grid_config=grid_config,
        )
        return temporal_derivative.sample(
            query_position_grid_xy,
            query_yaw_radians,
        )
    return _finite_real(
        temporal_derivative,
        name="temporal_safety_derivative",
    )


def validate_temporal_safety_derivative(
    temporal_derivative: TemporalSafetyDerivativeInput,
    *,
    h_full: SafeSAGEHFullResult,
    grid_config: LocalOccupancyGridConfig,
) -> TemporalSafetyDerivativeInput:
    """Validate one scalar or spatial temporal input once at an API edge."""

    if isinstance(temporal_derivative, SafeSAGETemporalDerivativeResult):
        _validate_previous_estimate(
            temporal_derivative,
            current_h_full=h_full,
            grid_config=grid_config,
        )
        return temporal_derivative
    return _finite_real(
        temporal_derivative,
        name="temporal_safety_derivative",
    )


def _validate_field_pair(
    previous: object,
    current: object,
    *,
    grid_config: LocalOccupancyGridConfig,
) -> None:
    if not isinstance(previous, SafeSAGEHFullResult):
        raise TypeError("previous_h_full must be a SafeSAGEHFullResult.")
    if not isinstance(current, SafeSAGEHFullResult):
        raise TypeError("current_h_full must be a SafeSAGEHFullResult.")
    if not isinstance(grid_config, LocalOccupancyGridConfig):
        raise TypeError("grid_config must be a LocalOccupancyGridConfig.")
    if previous.shape != current.shape:
        raise ValueError("Consecutive h_full shapes must match.")
    if current.shape[1:] != grid_config.shape:
        raise ValueError("h_full spatial shape must match grid_config.")
    if not np.allclose(
        previous.yaw_angles_radians,
        current.yaw_angles_radians,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Consecutive h_full yaw angles must match.")
    if not math.isclose(
        previous.resolution,
        grid_config.resolution,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ) or not math.isclose(
        current.resolution,
        grid_config.resolution,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("h_full resolutions must match grid_config.")


def _validate_previous_estimate(
    estimate: object,
    *,
    current_h_full: SafeSAGEHFullResult,
    grid_config: LocalOccupancyGridConfig,
) -> None:
    if not isinstance(estimate, SafeSAGETemporalDerivativeResult):
        raise TypeError(
            "previous_estimate must be a "
            "SafeSAGETemporalDerivativeResult."
        )
    if estimate.shape != current_h_full.shape:
        raise ValueError("Temporal derivative shape must match h_full.")
    if estimate.grid_config != grid_config:
        raise ValueError("Temporal derivative grid_config must match.")
    if not np.allclose(
        estimate.yaw_angles_radians,
        current_h_full.yaw_angles_radians,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Temporal derivative yaw angles must match h_full.")


def _shift_spatial_stack(
    values: np.ndarray,
    *,
    offset_cells_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``values`` at spatial indices shifted by ``(dx, dy)``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("values must have shape (yaw, rows, columns).")
    row_count, column_count = array.shape[1:]
    source_rows = (
        np.arange(row_count, dtype=np.float64)[:, np.newaxis]
        + float(offset_cells_xy[1])
    )
    source_columns = (
        np.arange(column_count, dtype=np.float64)[np.newaxis, :]
        + float(offset_cells_xy[0])
    )
    valid = (
        (source_rows >= 0.0)
        & (source_rows <= row_count - 1)
        & (source_columns >= 0.0)
        & (source_columns <= column_count - 1)
    )
    row_zero = np.floor(np.clip(source_rows, 0.0, row_count - 1)).astype(
        np.int64
    )
    column_zero = np.floor(
        np.clip(source_columns, 0.0, column_count - 1)
    ).astype(np.int64)
    row_one = np.minimum(row_zero + 1, row_count - 1)
    column_one = np.minimum(column_zero + 1, column_count - 1)
    row_weight = np.clip(source_rows - row_zero, 0.0, 1.0)
    column_weight = np.clip(source_columns - column_zero, 0.0, 1.0)

    value_00 = array[:, row_zero, column_zero]
    value_01 = array[:, row_zero, column_one]
    value_10 = array[:, row_one, column_zero]
    value_11 = array[:, row_one, column_one]
    lower = (
        (1.0 - column_weight) * value_00
        + column_weight * value_01
    )
    upper = (
        (1.0 - column_weight) * value_10
        + column_weight * value_11
    )
    shifted = (
        (1.0 - row_weight) * lower
        + row_weight * upper
    )
    return (
        np.ascontiguousarray(shifted, dtype=np.float64),
        np.ascontiguousarray(valid, dtype=bool),
    )


def _yaw_interpolation(
    yaw_radians: float,
    *,
    yaw_slice_count: int,
) -> tuple[int, int, float]:
    wrapped_yaw = (yaw_radians + math.pi) % math.tau - math.pi
    spacing = math.tau / yaw_slice_count
    position = (wrapped_yaw + math.pi) / spacing
    floor_position = math.floor(position)
    lower = floor_position % yaw_slice_count
    return lower, (lower + 1) % yaw_slice_count, float(position - floor_position)


def _bilinear_sample_cell_centers(
    values: np.ndarray,
    *,
    point_grid_xy: np.ndarray,
    config: LocalOccupancyGridConfig,
) -> float:
    column_position = (
        (point_grid_xy[0] - config.minimum_x) / config.resolution - 0.5
    )
    row_position = (
        (point_grid_xy[1] - config.minimum_y) / config.resolution - 0.5
    )
    tolerance = 1e-9
    if not (
        -tolerance <= row_position <= config.row_count - 1 + tolerance
        and -tolerance
        <= column_position
        <= config.column_count - 1 + tolerance
    ):
        raise ValueError(
            "query_position_grid_xy must lie within the "
            "grid-cell-center domain."
        )
    row_position = float(np.clip(row_position, 0.0, config.row_count - 1))
    column_position = float(
        np.clip(column_position, 0.0, config.column_count - 1)
    )
    row_zero = math.floor(row_position)
    column_zero = math.floor(column_position)
    row_one = min(row_zero + 1, config.row_count - 1)
    column_one = min(column_zero + 1, config.column_count - 1)
    row_weight = row_position - row_zero
    column_weight = column_position - column_zero
    lower = (
        (1.0 - column_weight) * values[row_zero, column_zero]
        + column_weight * values[row_zero, column_one]
    )
    upper = (
        (1.0 - column_weight) * values[row_one, column_zero]
        + column_weight * values[row_one, column_one]
    )
    return float((1.0 - row_weight) * lower + row_weight * upper)
