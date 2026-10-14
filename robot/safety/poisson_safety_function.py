"""Poisson safety function for the Risk-Aware PSF/LGF CBF baseline.

The scalar safety function solves

    Laplacian(h) = f,    f < 0

inside the free grid cells, with ``h = 0`` on occupied cells and the outer
grid boundary. The forcing may be one constant negative scalar or a spatially
varying negative field. The discrete maximum principle then gives positive
values in the free interior and zero values at unsafe boundaries.

The forcing is intentionally independent of semantic risk in this baseline.
Semantic risk is mapped to boundary flux in the separate Laplace guidance
field, and the scalar and vector fields meet in the CBF constraint. This is
the construction used by the Risk-Aware PSF/LGF method; it should not be
described as the complete Safe-SAGE field-synthesis pipeline.
"""

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from robot.safety.local_occupancy_grid import LocalOccupancyGrid


def _validated_real(
    value: object,
    *,
    name: str,
) -> float:
    """Return a finite real value while rejecting booleans and strings."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


def _validated_positive_integer(
    value: object,
    *,
    name: str,
) -> int:
    """Return a strictly positive integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value


@dataclass(frozen=True)
class PoissonSafetyFunctionConfig:
    """Numerical settings for the finite-difference Poisson solve."""

    forcing: float = -1.0
    convergence_tolerance: float = 1e-6
    maximum_iterations: int = 20_000
    residual_check_interval: int = 10
    solver_backend: str = "jacobi"

    def __post_init__(self) -> None:
        if self.solver_backend not in {"jacobi", "sparse_direct"}:
            raise ValueError("unsupported Poisson solver backend")
        forcing = _validated_real(
            self.forcing,
            name="forcing",
        )
        tolerance = _validated_real(
            self.convergence_tolerance,
            name="convergence_tolerance",
        )
        maximum_iterations = _validated_positive_integer(
            self.maximum_iterations,
            name="maximum_iterations",
        )
        check_interval = _validated_positive_integer(
            self.residual_check_interval,
            name="residual_check_interval",
        )

        if forcing >= 0.0:
            raise ValueError("forcing must be strictly negative.")
        if tolerance <= 0.0:
            raise ValueError(
                "convergence_tolerance must be greater than zero."
            )

        object.__setattr__(self, "forcing", forcing)
        object.__setattr__(
            self,
            "convergence_tolerance",
            tolerance,
        )
        object.__setattr__(
            self,
            "maximum_iterations",
            maximum_iterations,
        )
        object.__setattr__(
            self,
            "residual_check_interval",
            check_interval,
        )


@dataclass(frozen=True)
class PoissonSafetyFunctionResult:
    """Converged scalar safety values and solver diagnostics."""

    values: np.ndarray
    dirichlet_boundary_mask: np.ndarray
    iteration_count: int
    maximum_residual: float
    forcing: float | None
    resolution: float
    forcing_field: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the common grid shape."""

        return self.values.shape

    @property
    def minimum_value(self) -> float:
        """Return the minimum scalar safety value."""

        return float(np.min(self.values))

    @property
    def maximum_value(self) -> float:
        """Return the maximum scalar safety value."""

        return float(np.max(self.values))

    def value_at_cell(
        self,
        row_index: int,
        column_index: int,
    ) -> float:
        """Return the scalar safety value at one checked cell index."""

        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or isinstance(column_index, bool)
            or not isinstance(column_index, int)
        ):
            raise TypeError("Grid cell indices must be integers.")

        row_count, column_count = self.shape
        if not 0 <= row_index < row_count:
            raise IndexError("row_index is outside the solution grid.")
        if not 0 <= column_index < column_count:
            raise IndexError("column_index is outside the solution grid.")

        return float(self.values[row_index, column_index])


def solve_poisson_safety_function(
    occupancy_grid: LocalOccupancyGrid,
    config: PoissonSafetyFunctionConfig | None = None,
    *,
    forcing_field: np.ndarray | None = None,
) -> PoissonSafetyFunctionResult:
    """Solve the scalar Poisson safety function using Jacobi iteration.

    The outermost row and column are fixed at zero. Occupied cells are also
    fixed at zero. All remaining cells use a standard five-point Laplacian.
    When ``forcing_field`` is omitted, ``config.forcing`` is used uniformly.
    A supplied forcing field must match the grid and be strictly negative.

    Raises:
        RuntimeError:
            If the residual does not reach the configured tolerance within
            ``maximum_iterations``.
    """

    if not isinstance(occupancy_grid, LocalOccupancyGrid):
        raise TypeError("occupancy_grid must be a LocalOccupancyGrid.")

    solver_config = config or PoissonSafetyFunctionConfig()
    if not isinstance(solver_config, PoissonSafetyFunctionConfig):
        raise TypeError("config must be a PoissonSafetyFunctionConfig.")

    resolved_forcing = _resolve_forcing_field(
        occupancy_grid,
        scalar_forcing=solver_config.forcing,
        forcing_field=forcing_field,
    )
    scalar_forcing = (
        solver_config.forcing if forcing_field is None else None
    )

    boundary_mask = _dirichlet_boundary_mask(occupancy_grid)
    solution_mask = ~boundary_mask
    values = np.zeros(occupancy_grid.shape, dtype=np.float64)

    if not np.any(solution_mask):
        return _make_result(
            values,
            boundary_mask,
            iteration_count=0,
            maximum_residual=0.0,
            scalar_forcing=scalar_forcing,
            forcing_field=(
                None if forcing_field is None else resolved_forcing
            ),
            resolution=occupancy_grid.config.resolution,
        )

    resolution = occupancy_grid.config.resolution
    if solver_config.solver_backend == "sparse_direct":
        from robot.safety.sparse_dirichlet import solve_five_point_dirichlet

        values = solve_five_point_dirichlet(
            values, boundary_mask, forcing=resolved_forcing, resolution=resolution
        )
        residual = _maximum_poisson_residual(
            values,
            solution_mask=solution_mask,
            forcing_field=resolved_forcing,
            resolution=resolution,
        )
        if (
            not np.isfinite(values).all()
            or residual > solver_config.convergence_tolerance
        ):
            raise RuntimeError(
                "direct Poisson solution failed the unchanged residual gate"
            )
        return _make_result(
            values,
            boundary_mask,
            iteration_count=1,
            maximum_residual=residual,
            scalar_forcing=scalar_forcing,
            forcing_field=None if forcing_field is None else resolved_forcing,
            resolution=resolution,
        )
    source_term = -resolved_forcing[1:-1, 1:-1] * resolution**2
    maximum_residual = math.inf

    for iteration_count in range(
        1,
        solver_config.maximum_iterations + 1,
    ):
        neighbor_sum = (
            values[:-2, 1:-1]
            + values[2:, 1:-1]
            + values[1:-1, :-2]
            + values[1:-1, 2:]
        )
        candidate_values = 0.25 * (
            neighbor_sum + source_term
        )

        next_values = values.copy()
        interior_solution_mask = solution_mask[1:-1, 1:-1]
        next_values[1:-1, 1:-1][interior_solution_mask] = (
            candidate_values[interior_solution_mask]
        )
        values = next_values

        should_check_residual = (
            iteration_count == 1
            or iteration_count
            % solver_config.residual_check_interval
            == 0
            or iteration_count
            == solver_config.maximum_iterations
        )
        if not should_check_residual:
            continue

        maximum_residual = _maximum_poisson_residual(
            values,
            solution_mask=solution_mask,
            forcing_field=resolved_forcing,
            resolution=resolution,
        )
        if maximum_residual <= solver_config.convergence_tolerance:
            return _make_result(
                values,
                boundary_mask,
                iteration_count=iteration_count,
                maximum_residual=maximum_residual,
                scalar_forcing=scalar_forcing,
                forcing_field=(
                    None if forcing_field is None else resolved_forcing
                ),
                resolution=resolution,
            )

    raise RuntimeError(
        "Poisson safety-function solver did not converge after "
        f"{solver_config.maximum_iterations} iterations; maximum "
        f"residual={maximum_residual:.6e}, tolerance="
        f"{solver_config.convergence_tolerance:.6e}."
    )


def _dirichlet_boundary_mask(
    occupancy_grid: LocalOccupancyGrid,
) -> np.ndarray:
    """Return occupied cells plus the outer rectangular boundary."""

    boundary_mask = np.array(
        occupancy_grid.occupied_mask,
        dtype=bool,
        copy=True,
    )

    if boundary_mask.shape[0] > 0:
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
    if boundary_mask.shape[1] > 0:
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True

    return boundary_mask


def _resolve_forcing_field(
    occupancy_grid: LocalOccupancyGrid,
    *,
    scalar_forcing: float,
    forcing_field: np.ndarray | None,
) -> np.ndarray:
    """Return a validated negative forcing value for every grid cell."""

    if forcing_field is None:
        return np.full(
            occupancy_grid.shape,
            scalar_forcing,
            dtype=np.float64,
        )
    if not isinstance(forcing_field, np.ndarray):
        raise TypeError("forcing_field must be a numpy array.")
    if forcing_field.shape != occupancy_grid.shape:
        raise ValueError(
            "forcing_field shape must match the occupancy grid."
        )

    resolved = np.array(
        forcing_field,
        dtype=np.float64,
        copy=True,
        order="C",
    )
    if not np.isfinite(resolved).all():
        raise ValueError("forcing_field values must be finite.")
    if np.any(resolved >= 0.0):
        raise ValueError(
            "forcing_field values must be strictly negative."
        )

    return resolved


def _maximum_poisson_residual(
    values: np.ndarray,
    *,
    solution_mask: np.ndarray,
    forcing_field: np.ndarray,
    resolution: float,
) -> float:
    """Return the maximum absolute five-point-equation residual."""

    laplacian = (
        values[:-2, 1:-1]
        + values[2:, 1:-1]
        + values[1:-1, :-2]
        + values[1:-1, 2:]
        - 4.0 * values[1:-1, 1:-1]
    ) / resolution**2
    residuals = np.abs(
        laplacian - forcing_field[1:-1, 1:-1]
    )
    interior_solution_mask = solution_mask[1:-1, 1:-1]

    if not np.any(interior_solution_mask):
        return 0.0

    return float(np.max(residuals[interior_solution_mask]))


def _make_result(
    values: np.ndarray,
    boundary_mask: np.ndarray,
    *,
    iteration_count: int,
    maximum_residual: float,
    scalar_forcing: float | None,
    forcing_field: np.ndarray | None,
    resolution: float,
) -> PoissonSafetyFunctionResult:
    """Freeze solution arrays and construct a result object."""

    values = np.ascontiguousarray(values, dtype=np.float64)
    boundary_mask = np.ascontiguousarray(
        boundary_mask,
        dtype=bool,
    )
    frozen_forcing_field = None
    if forcing_field is not None:
        frozen_forcing_field = np.ascontiguousarray(
            forcing_field,
            dtype=np.float64,
        )
        frozen_forcing_field.setflags(write=False)
    values.setflags(write=False)
    boundary_mask.setflags(write=False)

    return PoissonSafetyFunctionResult(
        values=values,
        dirichlet_boundary_mask=boundary_mask,
        iteration_count=iteration_count,
        maximum_residual=float(maximum_residual),
        forcing=scalar_forcing,
        resolution=float(resolution),
        forcing_field=frozen_forcing_field,
    )
