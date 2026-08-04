"""Construct the yaw-indexed Safe-SAGE safety field ``h_full``.

Safe-SAGE defines ``h_full(x, y, psi)`` over planar position and periodic
robot yaw. For every discrete yaw ``psi_k``, the spatial slice solves

    Laplacian_xy(h_full[k, ...]) = f_hat[k, ...]

with zero Dirichlet values on the yaw-dependent unsafe boundary. The
forcing is derived from the divergence of that slice's semantic Laplace
guidance field ``v_sem``:

    f_hat = -softplus(-beta * div(v_sem)) / beta.

The Laplacian is spatial: this is a periodic stack of two-dimensional
Poisson solves, not a three-dimensional Poisson equation containing a yaw
Laplacian. A centered periodic difference supplies ``partial h / partial
psi`` for the later full-state CBF constraint.

Each input occupancy grid must already represent the unsafe configuration
space for its corresponding yaw (for example, obstacle geometry expanded
by the rotated robot footprint). Repeating one grid is valid for a circular
footprint or a controlled numerical test, but it does not model a general
non-circular robot.

Array convention:

* axis 0 indexes yaw;
* axis 1 indexes robot-frame ``+Y`` rows;
* axis 2 indexes robot-frame ``+X`` columns;
* spatial-gradient components are ordered ``(d/dx, d/dy)``.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldResult,
)
from robot.safety.local_occupancy_grid import LocalOccupancyGrid
from robot.safety.poisson_safety_function import (
    PoissonSafetyFunctionConfig,
    PoissonSafetyFunctionResult,
    solve_poisson_safety_function,
)


def _validated_positive_real(
    value: object,
    *,
    name: str,
) -> float:
    """Return a finite, strictly positive real number."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")

    return result


def _validated_yaw_slice_count(value: object) -> int:
    """Return a yaw count suitable for centered periodic differences."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("yaw_slice_count must be an integer.")
    if value < 3:
        raise ValueError("yaw_slice_count must be at least three.")
    return value


@dataclass(frozen=True)
class SafeSAGEHFullConfig:
    """Yaw discretization, forcing transform, and Poisson settings."""

    yaw_slice_count: int = 10
    softplus_beta: float = 1.0
    minimum_forcing_magnitude: float = 1e-9
    poisson_config: PoissonSafetyFunctionConfig = field(
        default_factory=PoissonSafetyFunctionConfig
    )

    def __post_init__(self) -> None:
        yaw_slice_count = _validated_yaw_slice_count(
            self.yaw_slice_count
        )
        beta = _validated_positive_real(
            self.softplus_beta,
            name="softplus_beta",
        )
        forcing_floor = _validated_positive_real(
            self.minimum_forcing_magnitude,
            name="minimum_forcing_magnitude",
        )
        if not isinstance(
            self.poisson_config,
            PoissonSafetyFunctionConfig,
        ):
            raise TypeError(
                "poisson_config must be a "
                "PoissonSafetyFunctionConfig."
            )

        object.__setattr__(
            self,
            "yaw_slice_count",
            yaw_slice_count,
        )
        object.__setattr__(self, "softplus_beta", beta)
        object.__setattr__(
            self,
            "minimum_forcing_magnitude",
            forcing_floor,
        )

    @property
    def yaw_spacing_radians(self) -> float:
        """Return the uniform periodic spacing between yaw slices."""

        return math.tau / self.yaw_slice_count


@dataclass(frozen=True)
class _SafeSAGEHFullSliceResult:
    """One internal fixed-yaw semantic PSF slice and diagnostics."""

    safety_function: PoissonSafetyFunctionResult
    guidance_divergence: np.ndarray
    forcing_field: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """Return the common two-dimensional grid shape."""

        return self.safety_function.shape


@dataclass(frozen=True)
class SafeSAGEHFullResult:
    """Complete immutable ``h_full(x, y, psi)`` and its derivatives."""

    yaw_angles_radians: np.ndarray
    values: np.ndarray
    spatial_gradients_robot_xy: np.ndarray
    yaw_derivatives: np.ndarray
    dirichlet_boundary_masks: np.ndarray
    guidance_divergence: np.ndarray
    forcing_fields: np.ndarray
    iteration_counts: np.ndarray
    maximum_residuals: np.ndarray
    resolution: float
    softplus_beta: float
    minimum_forcing_magnitude: float

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(yaw slices, rows, columns)``."""

        return self.values.shape

    @property
    def yaw_slice_count(self) -> int:
        """Return the number of periodic yaw slices."""

        return self.shape[0]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return the shared ``(rows, columns)`` grid shape."""

        return self.shape[1], self.shape[2]

    @property
    def yaw_spacing_radians(self) -> float:
        """Return the uniform periodic yaw spacing."""

        return math.tau / self.yaw_slice_count

    def interpolate_yaw_slice(
        self,
        yaw_radians: float,
    ) -> np.ndarray:
        """Linearly interpolate a spatial safety slice at periodic yaw."""

        lower, upper, weight = self._yaw_interpolation(yaw_radians)
        interpolated = (
            (1.0 - weight) * self.values[lower]
            + weight * self.values[upper]
        )
        return _frozen_array(interpolated, dtype=np.float64)

    def value_at_cell(
        self,
        yaw_radians: float,
        row_index: int,
        column_index: int,
    ) -> float:
        """Return periodically interpolated ``h_full`` at one grid cell."""

        self._validate_cell_indices(row_index, column_index)
        lower, upper, weight = self._yaw_interpolation(yaw_radians)
        return float(
            (1.0 - weight)
            * self.values[lower, row_index, column_index]
            + weight
            * self.values[upper, row_index, column_index]
        )

    def gradient_at_cell(
        self,
        yaw_radians: float,
        row_index: int,
        column_index: int,
    ) -> tuple[float, float, float]:
        """Return interpolated ``(dh/dx, dh/dy, dh/dpsi)``."""

        self._validate_cell_indices(row_index, column_index)
        lower, upper, weight = self._yaw_interpolation(yaw_radians)

        spatial_gradient = (
            (1.0 - weight)
            * self.spatial_gradients_robot_xy[
                lower,
                row_index,
                column_index,
            ]
            + weight
            * self.spatial_gradients_robot_xy[
                upper,
                row_index,
                column_index,
            ]
        )
        yaw_derivative = (
            (1.0 - weight)
            * self.yaw_derivatives[lower, row_index, column_index]
            + weight
            * self.yaw_derivatives[upper, row_index, column_index]
        )
        return (
            float(spatial_gradient[0]),
            float(spatial_gradient[1]),
            float(yaw_derivative),
        )

    def _yaw_interpolation(
        self,
        yaw_radians: float,
    ) -> tuple[int, int, float]:
        """Return periodic lower/upper indices and interpolation weight."""

        yaw = _validated_finite_real(
            yaw_radians,
            name="yaw_radians",
        )
        wrapped_yaw = (yaw + math.pi) % math.tau - math.pi
        slice_position = (
            (wrapped_yaw + math.pi) / self.yaw_spacing_radians
        )
        lower_index = int(math.floor(slice_position))
        lower_index %= self.yaw_slice_count
        upper_index = (lower_index + 1) % self.yaw_slice_count
        weight = slice_position - math.floor(slice_position)
        return lower_index, upper_index, float(weight)

    def _validate_cell_indices(
        self,
        row_index: int,
        column_index: int,
    ) -> None:
        """Reject invalid spatial cell indices."""

        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or isinstance(column_index, bool)
            or not isinstance(column_index, int)
        ):
            raise TypeError("Grid cell indices must be integers.")

        row_count, column_count = self.spatial_shape
        if not 0 <= row_index < row_count:
            raise IndexError("row_index is outside the solution grid.")
        if not 0 <= column_index < column_count:
            raise IndexError("column_index is outside the solution grid.")


def uniform_yaw_angles(yaw_slice_count: int = 10) -> np.ndarray:
    """Return canonical uniform yaw slices on ``[-pi, pi)``."""

    slice_count = _validated_yaw_slice_count(yaw_slice_count)
    angles = np.linspace(
        -math.pi,
        math.pi,
        num=slice_count,
        endpoint=False,
        dtype=np.float64,
    )
    return _frozen_array(angles, dtype=np.float64)


def compute_guidance_divergence(
    occupancy_grid: LocalOccupancyGrid,
    guidance_field: LaplaceGuidanceFieldResult,
) -> np.ndarray:
    """Return finite-difference ``div(v_sem)`` in robot coordinates.

    Centered differences are used when both neighbors lie outside occupied
    cells. At an occupied boundary, a one-sided difference avoids extending
    the guidance field through the obstacle interior. The outer grid boundary
    remains a valid neighbor because the LGF supplies its Dirichlet value.
    """

    _validate_pipeline_alignment(occupancy_grid, guidance_field)

    vectors = guidance_field.vectors_robot_xy
    valid_mask = ~occupancy_grid.occupied_mask
    resolution = occupancy_grid.config.resolution

    dvx_dx = _masked_axis_derivative(
        vectors[..., 0],
        valid_mask=valid_mask,
        axis=1,
        spacing=resolution,
    )
    dvy_dy = _masked_axis_derivative(
        vectors[..., 1],
        valid_mask=valid_mask,
        axis=0,
        spacing=resolution,
    )
    divergence = dvx_dx + dvy_dy
    divergence[occupancy_grid.occupied_mask] = 0.0

    if not np.isfinite(divergence).all():
        raise ValueError("Computed guidance divergence must be finite.")

    return _frozen_array(divergence, dtype=np.float64)


def build_safe_sage_forcing_field(
    guidance_divergence: np.ndarray,
    config: SafeSAGEHFullConfig | None = None,
) -> np.ndarray:
    """Transform guidance divergence into smooth negative PSF forcing."""

    if not isinstance(guidance_divergence, np.ndarray):
        raise TypeError("guidance_divergence must be a numpy array.")
    if guidance_divergence.ndim != 2:
        raise ValueError("guidance_divergence must be two-dimensional.")

    divergence = np.asarray(guidance_divergence, dtype=np.float64)
    if not np.isfinite(divergence).all():
        raise ValueError("guidance_divergence values must be finite.")

    forcing_config = config or SafeSAGEHFullConfig()
    if not isinstance(forcing_config, SafeSAGEHFullConfig):
        raise TypeError("config must be a SafeSAGEHFullConfig.")

    beta = forcing_config.softplus_beta
    with np.errstate(over="ignore", invalid="ignore"):
        softplus_values = np.logaddexp(
            0.0,
            -beta * divergence,
        ) / beta

    if not np.isfinite(softplus_values).all():
        raise ValueError(
            "The Safe-SAGE softplus forcing calculation overflowed."
        )

    forcing = -np.maximum(
        softplus_values,
        forcing_config.minimum_forcing_magnitude,
    )
    if not np.isfinite(forcing).all() or np.any(forcing >= 0.0):
        raise RuntimeError(
            "Safe-SAGE forcing must remain finite and strictly negative."
        )

    return _frozen_array(forcing, dtype=np.float64)


def solve_safe_sage_h_full(
    occupancy_grids: Sequence[LocalOccupancyGrid],
    guidance_fields: Sequence[LaplaceGuidanceFieldResult],
    config: SafeSAGEHFullConfig | None = None,
) -> SafeSAGEHFullResult:
    """Construct periodic ``h_full(x, y, psi)`` from yaw-specific slices.

    ``occupancy_grids[k]`` and ``guidance_fields[k]`` must describe the same
    geometry and semantic field at ``uniform_yaw_angles(config.yaw_slice_count)[k]``.
    All slices must share one spatial grid configuration.
    """

    full_config = config or SafeSAGEHFullConfig()
    if not isinstance(full_config, SafeSAGEHFullConfig):
        raise TypeError("config must be a SafeSAGEHFullConfig.")

    occupancy_sequence = _validated_sequence(
        occupancy_grids,
        name="occupancy_grids",
    )
    guidance_sequence = _validated_sequence(
        guidance_fields,
        name="guidance_fields",
    )
    expected_count = full_config.yaw_slice_count
    if len(occupancy_sequence) != expected_count:
        raise ValueError(
            "occupancy_grids length must equal yaw_slice_count."
        )
    if len(guidance_sequence) != expected_count:
        raise ValueError(
            "guidance_fields length must equal yaw_slice_count."
        )

    _validate_full_stack_alignment(
        occupancy_sequence,
        guidance_sequence,
    )
    slice_results = tuple(
        _solve_safe_sage_h_full_slice(
            occupancy_grid,
            guidance_field,
            full_config,
        )
        for occupancy_grid, guidance_field in zip(
            occupancy_sequence,
            guidance_sequence,
            strict=True,
        )
    )

    values = np.stack(
        [result.safety_function.values for result in slice_results]
    )
    boundary_masks = np.stack(
        [
            result.safety_function.dirichlet_boundary_mask
            for result in slice_results
        ]
    )
    divergences = np.stack(
        [result.guidance_divergence for result in slice_results]
    )
    forcing_fields = np.stack(
        [result.forcing_field for result in slice_results]
    )
    iteration_counts = np.asarray(
        [
            result.safety_function.iteration_count
            for result in slice_results
        ],
        dtype=np.int64,
    )
    maximum_residuals = np.asarray(
        [
            result.safety_function.maximum_residual
            for result in slice_results
        ],
        dtype=np.float64,
    )
    resolution = occupancy_sequence[0].config.resolution

    spatial_gradients = _spatial_gradients(
        values,
        boundary_masks=boundary_masks,
        resolution=resolution,
    )
    yaw_spacing = full_config.yaw_spacing_radians
    yaw_derivatives = (
        np.roll(values, -1, axis=0)
        - np.roll(values, 1, axis=0)
    ) / (2.0 * yaw_spacing)

    return SafeSAGEHFullResult(
        yaw_angles_radians=uniform_yaw_angles(expected_count),
        values=_frozen_array(values, dtype=np.float64),
        spatial_gradients_robot_xy=_frozen_array(
            spatial_gradients,
            dtype=np.float64,
        ),
        yaw_derivatives=_frozen_array(
            yaw_derivatives,
            dtype=np.float64,
        ),
        dirichlet_boundary_masks=_frozen_array(
            boundary_masks,
            dtype=bool,
        ),
        guidance_divergence=_frozen_array(
            divergences,
            dtype=np.float64,
        ),
        forcing_fields=_frozen_array(
            forcing_fields,
            dtype=np.float64,
        ),
        iteration_counts=_frozen_array(
            iteration_counts,
            dtype=np.int64,
        ),
        maximum_residuals=_frozen_array(
            maximum_residuals,
            dtype=np.float64,
        ),
        resolution=float(resolution),
        softplus_beta=full_config.softplus_beta,
        minimum_forcing_magnitude=(
            full_config.minimum_forcing_magnitude
        ),
    )


def _solve_safe_sage_h_full_slice(
    occupancy_grid: LocalOccupancyGrid,
    guidance_field: LaplaceGuidanceFieldResult,
    config: SafeSAGEHFullConfig,
) -> _SafeSAGEHFullSliceResult:
    """Construct one spatial slice; public callers use the full solver."""

    divergence = compute_guidance_divergence(
        occupancy_grid,
        guidance_field,
    )
    forcing = build_safe_sage_forcing_field(divergence, config)
    safety_function = solve_poisson_safety_function(
        occupancy_grid,
        config.poisson_config,
        forcing_field=forcing,
    )
    return _SafeSAGEHFullSliceResult(
        safety_function=safety_function,
        guidance_divergence=divergence,
        forcing_field=forcing,
    )


def _validated_finite_real(value: object, *, name: str) -> float:
    """Return one finite real value."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _validated_sequence(value: object, *, name: str) -> tuple[object, ...]:
    """Materialize a non-string input sequence."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence.")
    return tuple(value)


def _validate_full_stack_alignment(
    occupancy_grids: tuple[object, ...],
    guidance_fields: tuple[object, ...],
) -> None:
    """Require every yaw slice to share one spatial discretization."""

    reference_grid = occupancy_grids[0]
    if not isinstance(reference_grid, LocalOccupancyGrid):
        raise TypeError(
            "Every occupancy_grids item must be a LocalOccupancyGrid."
        )
    reference_config = reference_grid.config

    for occupancy_grid, guidance_field in zip(
        occupancy_grids,
        guidance_fields,
        strict=True,
    ):
        if not isinstance(occupancy_grid, LocalOccupancyGrid):
            raise TypeError(
                "Every occupancy_grids item must be a LocalOccupancyGrid."
            )
        if occupancy_grid.config != reference_config:
            raise ValueError(
                "All occupancy grids must share one grid configuration."
            )
        _validate_pipeline_alignment(
            occupancy_grid,
            guidance_field,
        )


def _validate_pipeline_alignment(
    occupancy_grid: LocalOccupancyGrid,
    guidance_field: LaplaceGuidanceFieldResult,
) -> None:
    """Validate grid, guidance shape, resolution, and numerical values."""

    if not isinstance(occupancy_grid, LocalOccupancyGrid):
        raise TypeError("occupancy_grid must be a LocalOccupancyGrid.")
    if not isinstance(guidance_field, LaplaceGuidanceFieldResult):
        raise TypeError(
            "guidance_field must be a LaplaceGuidanceFieldResult."
        )
    if guidance_field.shape != occupancy_grid.shape:
        raise ValueError(
            "guidance_field shape must match the occupancy grid."
        )
    if not math.isclose(
        guidance_field.resolution,
        occupancy_grid.config.resolution,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "guidance_field resolution must match the occupancy grid."
        )
    if not np.isfinite(guidance_field.vectors_robot_xy).all():
        raise ValueError("guidance_field vectors must be finite.")


def _spatial_gradients(
    values: np.ndarray,
    *,
    boundary_masks: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """Return yaw-indexed spatial gradients in ``(x, y)`` order."""

    gradient_y, gradient_x = np.gradient(
        values,
        resolution,
        resolution,
        axis=(1, 2),
        edge_order=1,
    )
    gradients = np.stack((gradient_x, gradient_y), axis=-1)
    gradients[boundary_masks] = 0.0
    return gradients


def _masked_axis_derivative(
    values: np.ndarray,
    *,
    valid_mask: np.ndarray,
    axis: int,
    spacing: float,
) -> np.ndarray:
    """Differentiate one scalar grid without crossing occupied cells."""

    previous_values = np.zeros_like(values, dtype=np.float64)
    next_values = np.zeros_like(values, dtype=np.float64)
    previous_valid = np.zeros_like(valid_mask, dtype=bool)
    next_valid = np.zeros_like(valid_mask, dtype=bool)

    if axis == 0:
        previous_values[1:, :] = values[:-1, :]
        next_values[:-1, :] = values[1:, :]
        previous_valid[1:, :] = valid_mask[:-1, :]
        next_valid[:-1, :] = valid_mask[1:, :]
    elif axis == 1:
        previous_values[:, 1:] = values[:, :-1]
        next_values[:, :-1] = values[:, 1:]
        previous_valid[:, 1:] = valid_mask[:, :-1]
        next_valid[:, :-1] = valid_mask[:, 1:]
    else:
        raise ValueError("axis must be either 0 or 1.")

    centered = valid_mask & previous_valid & next_valid
    forward = valid_mask & ~previous_valid & next_valid
    backward = valid_mask & previous_valid & ~next_valid

    derivative = np.zeros_like(values, dtype=np.float64)
    derivative[centered] = (
        next_values[centered] - previous_values[centered]
    ) / (2.0 * spacing)
    derivative[forward] = (
        next_values[forward] - values[forward]
    ) / spacing
    derivative[backward] = (
        values[backward] - previous_values[backward]
    ) / spacing
    return derivative


def _frozen_array(values: np.ndarray, *, dtype: object) -> np.ndarray:
    """Return an owned, contiguous, read-only NumPy array."""

    frozen = np.array(values, dtype=dtype, copy=True, order="C")
    frozen.setflags(write=False)
    return frozen
