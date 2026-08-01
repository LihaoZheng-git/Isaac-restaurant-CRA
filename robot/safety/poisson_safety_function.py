"""Finite-difference Poisson safety function on a local occupancy grid.

The scalar safety function solves

    Laplacian(h) = f,    f < 0

inside the free grid cells, with ``h = 0`` on occupied cells and the outer
grid boundary. The discrete maximum principle then gives positive values in
the free interior and zero values at unsafe boundaries.

Semantic boundary flux is intentionally not used here. It belongs to the
separate Laplace guidance-field equation implemented in the next module.
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

    def __post_init__(self) -> None:
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
    forcing: float
    resolution: float

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
) -> PoissonSafetyFunctionResult:
    """Solve the scalar Poisson safety function using Jacobi iteration.

    The outermost row and column are fixed at zero. Occupied cells are also
    fixed at zero. All remaining cells use a standard five-point Laplacian.

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

    boundary_mask = _dirichlet_boundary_mask(occupancy_grid)
    solution_mask = ~boundary_mask
    values = np.zeros(occupancy_grid.shape, dtype=np.float64)

    if not np.any(solution_mask):
        return _make_result(
            values,
            boundary_mask,
            iteration_count=0,
            maximum_residual=0.0,
            config=solver_config,
            resolution=occupancy_grid.config.resolution,
        )

    resolution = occupancy_grid.config.resolution
    source_term = -solver_config.forcing * resolution**2
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
            forcing=solver_config.forcing,
            resolution=resolution,
        )
        if maximum_residual <= solver_config.convergence_tolerance:
            return _make_result(
                values,
                boundary_mask,
                iteration_count=iteration_count,
                maximum_residual=maximum_residual,
                config=solver_config,
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


def _maximum_poisson_residual(
    values: np.ndarray,
    *,
    solution_mask: np.ndarray,
    forcing: float,
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
    residuals = np.abs(laplacian - forcing)
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
    config: PoissonSafetyFunctionConfig,
    resolution: float,
) -> PoissonSafetyFunctionResult:
    """Freeze solution arrays and construct a result object."""

    values = np.ascontiguousarray(values, dtype=np.float64)
    boundary_mask = np.ascontiguousarray(
        boundary_mask,
        dtype=bool,
    )
    values.setflags(write=False)
    boundary_mask.setflags(write=False)

    return PoissonSafetyFunctionResult(
        values=values,
        dirichlet_boundary_mask=boundary_mask,
        iteration_count=iteration_count,
        maximum_residual=float(maximum_residual),
        forcing=config.forcing,
        resolution=float(resolution),
    )
