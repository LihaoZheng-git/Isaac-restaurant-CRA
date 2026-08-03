"""Normal-flux guidance field for the Risk-Aware PSF/LGF CBF baseline.

Each vector component solves the Laplace equation in free space:

    Laplacian(v_x) = 0
    Laplacian(v_y) = 0

On free cells adjacent to an obstacle, the prescribed vector is ``b * n``.
Here ``n`` points from free space into the obstacle and the configured
boundary flux ``b`` is negative, so the resulting vector points away from
the obstacle. The outer grid boundary and occupied cells are fixed at zero.

This solver intentionally implements only obstacle-normal boundary flux. It
does not implement Safe-SAGE's additional tangential Dirichlet interface for
social passing behavior. Keeping that distinction explicit preserves this
module as a reproducible risk-aware baseline.

Array convention follows ``LocalOccupancyGrid``: rows increase with
robot-frame ``+Y`` and columns increase with robot-frame ``+X``.
"""

import math
from dataclasses import dataclass

import numpy as np

from robot.safety.local_occupancy_grid import LocalOccupancyGrid
from robot.safety.vector_laplace_solver import (
    VectorLaplaceSolverConfig,
    solve_vector_laplace_dirichlet,
)

_NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class LaplaceGuidanceFieldConfig(VectorLaplaceSolverConfig):
    """Public numerical settings for the normal-only LGF baseline."""


@dataclass(frozen=True)
class LaplaceGuidanceFieldResult:
    """Converged harmonic vectors and boundary diagnostics."""

    vectors_robot_xy: np.ndarray
    dirichlet_boundary_mask: np.ndarray
    obstacle_interface_mask: np.ndarray
    obstacle_normals_robot_xy: np.ndarray
    iteration_count: int
    maximum_residual: float
    resolution: float

    @property
    def shape(self) -> tuple[int, int]:
        """Return the underlying two-dimensional grid shape."""

        return self.vectors_robot_xy.shape[:2]

    @property
    def maximum_magnitude(self) -> float:
        """Return the greatest vector magnitude in the grid."""

        magnitudes = np.linalg.norm(
            self.vectors_robot_xy,
            axis=2,
        )
        return float(np.max(magnitudes))

    def vector_at_cell(
        self,
        row_index: int,
        column_index: int,
    ) -> tuple[float, float]:
        """Return one checked robot-frame guidance vector."""

        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or isinstance(column_index, bool)
            or not isinstance(column_index, int)
        ):
            raise TypeError("Grid cell indices must be integers.")

        row_count, column_count = self.shape
        if not 0 <= row_index < row_count:
            raise IndexError("row_index is outside the guidance grid.")
        if not 0 <= column_index < column_count:
            raise IndexError("column_index is outside the guidance grid.")

        vector = self.vectors_robot_xy[row_index, column_index]
        return float(vector[0]), float(vector[1])


def solve_laplace_guidance_field(
    occupancy_grid: LocalOccupancyGrid,
    config: LaplaceGuidanceFieldConfig | None = None,
) -> LaplaceGuidanceFieldResult:
    """Solve the semantic vector Laplace guidance field.

    Raises:
        ValueError:
            If an occupied cell has a nonnegative boundary flux or a free
            interface cell has an ambiguous zero normal.
        RuntimeError:
            If the harmonic residual does not converge within the configured
            iteration limit.
    """

    if not isinstance(occupancy_grid, LocalOccupancyGrid):
        raise TypeError("occupancy_grid must be a LocalOccupancyGrid.")

    solver_config = config or LaplaceGuidanceFieldConfig()
    if not isinstance(solver_config, LaplaceGuidanceFieldConfig):
        raise TypeError("config must be a LaplaceGuidanceFieldConfig.")

    _validate_occupied_boundary_fluxes(occupancy_grid)

    (
        obstacle_interface_mask,
        obstacle_normals,
        prescribed_vectors,
    ) = _obstacle_interface_data(occupancy_grid)

    fixed_mask = _outer_boundary_mask(occupancy_grid.shape)
    fixed_mask |= occupancy_grid.occupied_mask
    fixed_mask |= obstacle_interface_mask

    vectors = np.zeros(
        occupancy_grid.shape + (2,),
        dtype=np.float64,
    )
    vectors[obstacle_interface_mask] = prescribed_vectors[
        obstacle_interface_mask
    ]

    resolution = occupancy_grid.config.resolution
    harmonic_result = solve_vector_laplace_dirichlet(
        vectors,
        fixed_mask,
        resolution=resolution,
        config=solver_config,
    )

    return _make_result(
        harmonic_result.vectors,
        fixed_mask,
        obstacle_interface_mask,
        obstacle_normals,
        iteration_count=harmonic_result.iteration_count,
        maximum_residual=harmonic_result.maximum_residual,
        resolution=resolution,
    )


def _validate_occupied_boundary_fluxes(
    occupancy_grid: LocalOccupancyGrid,
) -> None:
    """Require strictly negative flux on every occupied cell."""

    occupied_fluxes = occupancy_grid.cell_boundary_fluxes[
        occupancy_grid.occupied_mask
    ]
    if occupied_fluxes.size == 0:
        return
    if not np.isfinite(occupied_fluxes).all():
        raise ValueError(
            "Occupied-cell boundary fluxes must be finite."
        )
    if np.any(occupied_fluxes >= 0.0):
        raise ValueError(
            "Every occupied cell must have strictly negative "
            "boundary flux."
        )


def _obstacle_interface_data(
    occupancy_grid: LocalOccupancyGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct safe-side obstacle normals and prescribed vectors."""

    row_count, column_count = occupancy_grid.shape
    interface_mask = np.zeros(
        occupancy_grid.shape,
        dtype=bool,
    )
    normals = np.zeros(
        occupancy_grid.shape + (2,),
        dtype=np.float64,
    )
    prescribed_vectors = np.zeros_like(normals)

    for row_index in range(1, row_count - 1):
        for column_index in range(1, column_count - 1):
            if occupancy_grid.occupied_mask[
                row_index,
                column_index,
            ]:
                continue

            normal_sum = np.zeros(2, dtype=np.float64)
            adjacent_fluxes: list[float] = []

            for row_offset, column_offset in _NEIGHBOR_OFFSETS:
                neighbor_row = row_index + row_offset
                neighbor_column = column_index + column_offset

                if not occupancy_grid.occupied_mask[
                    neighbor_row,
                    neighbor_column,
                ]:
                    continue

                offset_norm = math.hypot(
                    row_offset,
                    column_offset,
                )
                normal_sum += np.array(
                    [
                        column_offset / offset_norm,
                        row_offset / offset_norm,
                    ],
                    dtype=np.float64,
                )
                adjacent_fluxes.append(
                    float(
                        occupancy_grid.cell_boundary_fluxes[
                            neighbor_row,
                            neighbor_column,
                        ]
                    )
                )

            if not adjacent_fluxes:
                continue

            normal_magnitude = float(np.linalg.norm(normal_sum))
            if normal_magnitude <= 1e-12:
                raise ValueError(
                    "Cannot determine a unique obstacle normal at cell "
                    f"({row_index}, {column_index}); increase obstacle "
                    "inflation or grid resolution."
                )

            normal = normal_sum / normal_magnitude
            boundary_flux = min(adjacent_fluxes)

            interface_mask[row_index, column_index] = True
            normals[row_index, column_index] = normal
            prescribed_vectors[row_index, column_index] = (
                boundary_flux * normal
            )

    return interface_mask, normals, prescribed_vectors


def _outer_boundary_mask(
    shape: tuple[int, int],
) -> np.ndarray:
    """Return a mask containing the outermost rows and columns."""

    boundary_mask = np.zeros(shape, dtype=bool)
    if shape[0] > 0:
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
    if shape[1] > 0:
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True

    return boundary_mask


def _make_result(
    vectors: np.ndarray,
    fixed_mask: np.ndarray,
    interface_mask: np.ndarray,
    normals: np.ndarray,
    *,
    iteration_count: int,
    maximum_residual: float,
    resolution: float,
) -> LaplaceGuidanceFieldResult:
    """Freeze result arrays and construct the diagnostic result."""

    vectors = np.ascontiguousarray(vectors, dtype=np.float64)
    fixed_mask = np.ascontiguousarray(fixed_mask, dtype=bool)
    interface_mask = np.ascontiguousarray(
        interface_mask,
        dtype=bool,
    )
    normals = np.ascontiguousarray(normals, dtype=np.float64)

    arrays = (
        vectors,
        fixed_mask,
        interface_mask,
        normals,
    )
    for array in arrays:
        array.setflags(write=False)

    return LaplaceGuidanceFieldResult(
        vectors_robot_xy=vectors,
        dirichlet_boundary_mask=fixed_mask,
        obstacle_interface_mask=interface_mask,
        obstacle_normals_robot_xy=normals,
        iteration_count=iteration_count,
        maximum_residual=float(maximum_residual),
        resolution=float(resolution),
    )
