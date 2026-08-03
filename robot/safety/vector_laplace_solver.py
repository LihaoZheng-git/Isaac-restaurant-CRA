"""Reusable finite-difference solver for vector Laplace equations.

The safety modules construct different Dirichlet boundary conditions, but
their numerical problem is the same: harmonically extend fixed vector values
through the remaining cells of a regular two-dimensional grid. Keeping that
iteration here prevents the Risk-Aware and tangential-LGF implementations
from drifting numerically.
"""

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


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
class VectorLaplaceSolverConfig:
    """Numerical settings shared by all vector Laplace fields."""

    convergence_tolerance: float = 1e-6
    maximum_iterations: int = 20_000
    residual_check_interval: int = 10
    relaxation_factor: float = 1.7

    def __post_init__(self) -> None:
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
        relaxation_factor = _validated_real(
            self.relaxation_factor,
            name="relaxation_factor",
        )

        if tolerance <= 0.0:
            raise ValueError(
                "convergence_tolerance must be greater than zero."
            )
        if not 0.0 < relaxation_factor < 2.0:
            raise ValueError(
                "relaxation_factor must be in the open interval (0, 2)."
            )

        object.__setattr__(self, "convergence_tolerance", tolerance)
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
        object.__setattr__(
            self,
            "relaxation_factor",
            relaxation_factor,
        )


@dataclass(frozen=True)
class VectorLaplaceSolveResult:
    """Converged vector values and numerical diagnostics."""

    vectors: np.ndarray
    iteration_count: int
    maximum_residual: float

    @property
    def shape(self) -> tuple[int, int]:
        """Return the two-dimensional grid shape."""

        return self.vectors.shape[:2]

    @property
    def component_count(self) -> int:
        """Return the number of independently solved vector components."""

        return int(self.vectors.shape[2])


def solve_vector_laplace_dirichlet(
    initial_vectors: np.ndarray,
    fixed_mask: np.ndarray,
    *,
    resolution: float,
    config: VectorLaplaceSolverConfig | None = None,
) -> VectorLaplaceSolveResult:
    """Harmonically extend fixed vector data through a regular grid.

    ``initial_vectors`` must have shape ``(rows, columns, components)`` and
    ``fixed_mask`` must have shape ``(rows, columns)``. Values on fixed cells
    are preserved exactly. Values on other cells are used only as the initial
    SOR iterate. Every outer-grid cell must be fixed because the finite-
    difference update operates on interior cells.

    The inputs are never modified. The returned vector array is contiguous
    and read-only.
    """

    solver_config = config or VectorLaplaceSolverConfig()
    if not isinstance(solver_config, VectorLaplaceSolverConfig):
        raise TypeError("config must be a VectorLaplaceSolverConfig.")

    vectors = _validated_vectors(initial_vectors)
    boundary_mask = _validated_fixed_mask(
        fixed_mask,
        expected_shape=vectors.shape[:2],
    )
    grid_resolution = _validated_real(resolution, name="resolution")
    if grid_resolution <= 0.0:
        raise ValueError("resolution must be greater than zero.")

    _validate_outer_boundary_is_fixed(boundary_mask)

    solution_mask = ~boundary_mask
    if not np.any(solution_mask):
        return _make_result(
            vectors,
            iteration_count=0,
            maximum_residual=0.0,
        )

    maximum_residual = math.inf
    interior_solution_mask = solution_mask[1:-1, 1:-1]
    interior_rows, interior_columns = np.indices(
        interior_solution_mask.shape
    )
    red_mask = interior_solution_mask & (
        (interior_rows + interior_columns) % 2 == 0
    )
    black_mask = interior_solution_mask & ~red_mask

    for iteration_count in range(
        1,
        solver_config.maximum_iterations + 1,
    ):
        for color_mask in (red_mask, black_mask):
            neighbor_sum = (
                vectors[:-2, 1:-1, :]
                + vectors[2:, 1:-1, :]
                + vectors[1:-1, :-2, :]
                + vectors[1:-1, 2:, :]
            )
            candidates = 0.25 * neighbor_sum
            interior_vectors = vectors[1:-1, 1:-1, :]
            interior_vectors[color_mask] += (
                solver_config.relaxation_factor
                * (
                    candidates[color_mask]
                    - interior_vectors[color_mask]
                )
            )

        should_check = (
            iteration_count == 1
            or iteration_count
            % solver_config.residual_check_interval
            == 0
            or iteration_count == solver_config.maximum_iterations
        )
        if not should_check:
            continue

        maximum_residual = _maximum_laplace_residual(
            vectors,
            solution_mask=solution_mask,
            resolution=grid_resolution,
        )
        if maximum_residual <= solver_config.convergence_tolerance:
            return _make_result(
                vectors,
                iteration_count=iteration_count,
                maximum_residual=maximum_residual,
            )

    raise RuntimeError(
        "Vector Laplace solver did not converge after "
        f"{solver_config.maximum_iterations} iterations; maximum "
        f"residual={maximum_residual:.6e}, tolerance="
        f"{solver_config.convergence_tolerance:.6e}."
    )


def _validated_vectors(initial_vectors: np.ndarray) -> np.ndarray:
    """Return a writable contiguous float64 copy of vector values."""

    try:
        vectors = np.asarray(initial_vectors, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "initial_vectors must contain real numbers."
        ) from error

    if vectors.ndim != 3 or vectors.shape[2] < 1:
        raise ValueError(
            "initial_vectors must have shape "
            "(rows, columns, components) with at least one component."
        )
    if vectors.shape[0] < 2 or vectors.shape[1] < 2:
        raise ValueError(
            "initial_vectors must contain at least two rows and columns."
        )
    if not np.isfinite(vectors).all():
        raise ValueError("initial_vectors must contain only finite values.")

    return np.array(vectors, dtype=np.float64, order="C", copy=True)


def _validated_fixed_mask(
    fixed_mask: np.ndarray,
    *,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Return a contiguous copy of a correctly shaped Boolean mask."""

    mask = np.asarray(fixed_mask)
    if mask.shape != expected_shape:
        raise ValueError(
            "fixed_mask must match the first two initial_vectors "
            f"dimensions {expected_shape}, received {mask.shape}."
        )
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError("fixed_mask must contain Boolean values.")

    return np.array(mask, dtype=bool, order="C", copy=True)


def _validate_outer_boundary_is_fixed(fixed_mask: np.ndarray) -> None:
    """Require a complete Dirichlet condition on the outer grid boundary."""

    outer_is_fixed = bool(
        np.all(fixed_mask[0, :])
        and np.all(fixed_mask[-1, :])
        and np.all(fixed_mask[:, 0])
        and np.all(fixed_mask[:, -1])
    )
    if not outer_is_fixed:
        raise ValueError(
            "Every outer-grid cell must be fixed by a Dirichlet boundary."
        )


def _maximum_laplace_residual(
    vectors: np.ndarray,
    *,
    solution_mask: np.ndarray,
    resolution: float,
) -> float:
    """Return the maximum absolute component-wise Laplace residual."""

    laplacian = (
        vectors[:-2, 1:-1, :]
        + vectors[2:, 1:-1, :]
        + vectors[1:-1, :-2, :]
        + vectors[1:-1, 2:, :]
        - 4.0 * vectors[1:-1, 1:-1, :]
    ) / resolution**2
    interior_solution_mask = solution_mask[1:-1, 1:-1]

    if not np.any(interior_solution_mask):
        return 0.0

    return float(np.max(np.abs(laplacian[interior_solution_mask])))


def _make_result(
    vectors: np.ndarray,
    *,
    iteration_count: int,
    maximum_residual: float,
) -> VectorLaplaceSolveResult:
    """Freeze the converged vector array and construct a result."""

    frozen_vectors = np.ascontiguousarray(vectors, dtype=np.float64)
    frozen_vectors.setflags(write=False)

    return VectorLaplaceSolveResult(
        vectors=frozen_vectors,
        iteration_count=int(iteration_count),
        maximum_residual=float(maximum_residual),
    )
