"""Normal-and-tangential harmonic guidance on a local occupancy grid.

The Risk-Aware baseline prescribes normal repulsion on the safe side of an
obstacle boundary. This module adds the Safe-SAGE tangential-LGF component by
prescribing a second Dirichlet condition on a buffered free-space interface:

    obstacle interface:  v = b n
    buffered interface:  v = lambda tau

Here ``n`` points from free space toward the obstacle and ``b`` is negative,
so ``b * n`` points away from the obstacle. ``lambda`` is stored as a positive
strength; the class policy's explicit passing side selects the orientation of
the unit tangent ``tau``.

This milestone keeps the Poisson safety function independent of semantics.
Semantic propagation through ``div(v)`` into ``h_full`` is a later step.
"""

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np

from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldConfig,
    LaplaceGuidanceFieldResult,
)
from robot.safety.local_occupancy_grid import LocalOccupancyGrid
from robot.safety.safe_sage_guidance_policy import (
    PassingSide,
    SafeSAGEGuidanceParameters,
    SafeSAGEGuidancePolicy,
)
from robot.safety.semantic_obstacle import SemanticObstacle
from robot.safety.vector_laplace_solver import (
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


@dataclass(frozen=True)
class TangentialLaplaceGuidanceFieldConfig(LaplaceGuidanceFieldConfig):
    """Numerical settings and physical radius of the tangential interface."""

    tangential_buffer_radius: float = 0.6

    def __post_init__(self) -> None:
        super().__post_init__()

        radius = _validated_real(
            self.tangential_buffer_radius,
            name="tangential_buffer_radius",
        )
        if radius <= 0.0:
            raise ValueError(
                "tangential_buffer_radius must be greater than zero."
            )

        object.__setattr__(self, "tangential_buffer_radius", radius)


@dataclass(frozen=True)
class TangentialLaplaceGuidanceFieldResult(LaplaceGuidanceFieldResult):
    """Harmonic field with diagnostics for the buffered interface."""

    tangential_interface_mask: np.ndarray
    tangential_normals_robot_xy: np.ndarray
    tangential_boundary_vectors_robot_xy: np.ndarray
    obstacle_component_labels: np.ndarray
    tangential_buffer_radius: float

    @property
    def tangential_interface_cell_count(self) -> int:
        """Return the number of fixed tangential-interface cells."""

        return int(np.count_nonzero(self.tangential_interface_mask))


@dataclass(frozen=True)
class _ObstacleComponent:
    """One tracked obstacle or one connected untracked occupied region."""

    label: int
    occupied_indices: np.ndarray
    parameters: SafeSAGEGuidanceParameters


def solve_tangential_laplace_guidance_field(
    occupancy_grid: LocalOccupancyGrid,
    *,
    guidance_policy: SafeSAGEGuidancePolicy,
    semantic_obstacles: Iterable[SemanticObstacle] = (),
    config: TangentialLaplaceGuidanceFieldConfig | None = None,
) -> TangentialLaplaceGuidanceFieldResult:
    """Solve a Laplace field with normal and tangential interfaces.

    The tangential buffer is represented by cells whose centers lie within
    half a grid cell of ``tangential_buffer_radius`` from an occupied cell.
    A zero tangential strength disables the buffered interface for that
    obstacle, allowing an exact normal-only ablation.

    Raises:
        ValueError:
            If a positive-strength buffer is too small, reaches the outer
            grid boundary, overlaps another obstacle buffer, or intersects
            the normal obstacle interface.
        RuntimeError:
            If the harmonic residual does not converge.
    """

    if not isinstance(occupancy_grid, LocalOccupancyGrid):
        raise TypeError("occupancy_grid must be a LocalOccupancyGrid.")
    if not isinstance(guidance_policy, SafeSAGEGuidancePolicy):
        raise TypeError(
            "guidance_policy must be a SafeSAGEGuidancePolicy."
        )

    solver_config = config or TangentialLaplaceGuidanceFieldConfig()
    if not isinstance(
        solver_config,
        TangentialLaplaceGuidanceFieldConfig,
    ):
        raise TypeError(
            "config must be a TangentialLaplaceGuidanceFieldConfig."
        )

    semantic_by_track_id = _semantic_obstacles_by_track_id(
        semantic_obstacles
    )
    component_labels, components = _obstacle_components(
        occupancy_grid,
        guidance_policy=guidance_policy,
        semantic_by_track_id=semantic_by_track_id,
    )

    (
        obstacle_interface_mask,
        obstacle_normals,
        normal_boundary_vectors,
    ) = _normal_interface_data(
        occupancy_grid,
        component_labels=component_labels,
        components=components,
    )

    outer_boundary_mask = _outer_boundary_mask(occupancy_grid.shape)
    (
        tangential_interface_mask,
        tangential_normals,
        tangential_boundary_vectors,
    ) = _tangential_interface_data(
        occupancy_grid,
        components=components,
        obstacle_interface_mask=obstacle_interface_mask,
        outer_boundary_mask=outer_boundary_mask,
        buffer_radius=solver_config.tangential_buffer_radius,
    )

    fixed_mask = outer_boundary_mask.copy()
    fixed_mask |= occupancy_grid.occupied_mask
    fixed_mask |= obstacle_interface_mask
    fixed_mask |= tangential_interface_mask

    vectors = np.zeros(
        occupancy_grid.shape + (2,),
        dtype=np.float64,
    )
    vectors[obstacle_interface_mask] = normal_boundary_vectors[
        obstacle_interface_mask
    ]
    vectors[tangential_interface_mask] = tangential_boundary_vectors[
        tangential_interface_mask
    ]

    harmonic_result = solve_vector_laplace_dirichlet(
        vectors,
        fixed_mask,
        resolution=occupancy_grid.config.resolution,
        config=solver_config,
    )

    return _make_result(
        harmonic_result.vectors,
        fixed_mask=fixed_mask,
        obstacle_interface_mask=obstacle_interface_mask,
        obstacle_normals=obstacle_normals,
        tangential_interface_mask=tangential_interface_mask,
        tangential_normals=tangential_normals,
        tangential_boundary_vectors=tangential_boundary_vectors,
        component_labels=component_labels,
        iteration_count=harmonic_result.iteration_count,
        maximum_residual=harmonic_result.maximum_residual,
        resolution=occupancy_grid.config.resolution,
        buffer_radius=solver_config.tangential_buffer_radius,
    )


def _semantic_obstacles_by_track_id(
    semantic_obstacles: Iterable[SemanticObstacle],
) -> dict[int, SemanticObstacle]:
    """Validate semantic obstacles and index them by track identifier."""

    try:
        obstacles = tuple(semantic_obstacles)
    except TypeError as error:
        raise TypeError(
            "semantic_obstacles must be an iterable."
        ) from error

    by_track_id: dict[int, SemanticObstacle] = {}
    for obstacle in obstacles:
        if not isinstance(obstacle, SemanticObstacle):
            raise TypeError(
                "Every semantic obstacle must be a SemanticObstacle."
            )
        if obstacle.track_id < 0:
            raise ValueError(
                "Semantic obstacle track IDs cannot be negative."
            )
        if obstacle.track_id in by_track_id:
            raise ValueError(
                "semantic_obstacles contains duplicate track ID "
                f"{obstacle.track_id}."
            )

        by_track_id[obstacle.track_id] = obstacle

    return by_track_id


def _obstacle_components(
    occupancy_grid: LocalOccupancyGrid,
    *,
    guidance_policy: SafeSAGEGuidancePolicy,
    semantic_by_track_id: dict[int, SemanticObstacle],
) -> tuple[np.ndarray, tuple[_ObstacleComponent, ...]]:
    """Group occupied cells by track and connected unknown regions."""

    labels = np.full(occupancy_grid.shape, -1, dtype=np.int64)
    components: list[_ObstacleComponent] = []

    occupied_track_ids = occupancy_grid.cell_track_ids[
        occupancy_grid.occupied_mask
    ]
    if occupied_track_ids.size and np.any(occupied_track_ids < -1):
        raise ValueError(
            "Occupied-cell track IDs cannot be smaller than -1."
        )

    tracked_ids = sorted(
        int(track_id)
        for track_id in np.unique(occupied_track_ids)
        if track_id >= 0
    )
    for track_id in tracked_ids:
        occupied_indices = np.argwhere(
            occupancy_grid.occupied_mask
            & (occupancy_grid.cell_track_ids == track_id)
        )
        semantic_obstacle = semantic_by_track_id.get(track_id)
        parameters = (
            guidance_policy.parameters_for_obstacle(semantic_obstacle)
            if semantic_obstacle is not None
            else guidance_policy.parameters_for(
                guidance_policy.unknown_class
            )
        )
        label = len(components)
        labels[occupied_indices[:, 0], occupied_indices[:, 1]] = label
        components.append(
            _ObstacleComponent(
                label=label,
                occupied_indices=occupied_indices,
                parameters=parameters,
            )
        )

    unknown_mask = occupancy_grid.occupied_mask & (
        occupancy_grid.cell_track_ids == -1
    )
    visited = np.zeros(occupancy_grid.shape, dtype=bool)
    for seed_row, seed_column in np.argwhere(unknown_mask):
        if visited[seed_row, seed_column]:
            continue

        queue = deque([(int(seed_row), int(seed_column))])
        visited[seed_row, seed_column] = True
        component_cells: list[tuple[int, int]] = []

        while queue:
            row_index, column_index = queue.popleft()
            component_cells.append((row_index, column_index))

            for row_offset, column_offset in _NEIGHBOR_OFFSETS:
                neighbor_row = row_index + row_offset
                neighbor_column = column_index + column_offset
                if not (
                    0 <= neighbor_row < occupancy_grid.shape[0]
                    and 0 <= neighbor_column < occupancy_grid.shape[1]
                ):
                    continue
                if (
                    unknown_mask[neighbor_row, neighbor_column]
                    and not visited[neighbor_row, neighbor_column]
                ):
                    visited[neighbor_row, neighbor_column] = True
                    queue.append((neighbor_row, neighbor_column))

        occupied_indices = np.asarray(
            component_cells,
            dtype=np.int64,
        )
        label = len(components)
        labels[occupied_indices[:, 0], occupied_indices[:, 1]] = label
        components.append(
            _ObstacleComponent(
                label=label,
                occupied_indices=occupied_indices,
                parameters=guidance_policy.parameters_for(
                    guidance_policy.unknown_class
                ),
            )
        )

    return labels, tuple(components)


def _normal_interface_data(
    occupancy_grid: LocalOccupancyGrid,
    *,
    component_labels: np.ndarray,
    components: tuple[_ObstacleComponent, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct normal boundary data on free cells beside obstacles."""

    row_count, column_count = occupancy_grid.shape
    interface_mask = np.zeros(occupancy_grid.shape, dtype=bool)
    normals = np.zeros(
        occupancy_grid.shape + (2,),
        dtype=np.float64,
    )
    prescribed_vectors = np.zeros_like(normals)

    for row_index in range(1, row_count - 1):
        for column_index in range(1, column_count - 1):
            if occupancy_grid.occupied_mask[row_index, column_index]:
                continue

            normal_sum = np.zeros(2, dtype=np.float64)
            adjacent_fluxes: list[float] = []

            for row_offset, column_offset in _NEIGHBOR_OFFSETS:
                neighbor_row = row_index + row_offset
                neighbor_column = column_index + column_offset
                component_label = int(
                    component_labels[neighbor_row, neighbor_column]
                )
                if component_label < 0:
                    continue

                offset_norm = math.hypot(row_offset, column_offset)
                normal_sum += np.array(
                    [
                        column_offset / offset_norm,
                        row_offset / offset_norm,
                    ],
                    dtype=np.float64,
                )
                adjacent_fluxes.append(
                    components[component_label].parameters.boundary_flux
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


def _tangential_interface_data(
    occupancy_grid: LocalOccupancyGrid,
    *,
    components: tuple[_ObstacleComponent, ...],
    obstacle_interface_mask: np.ndarray,
    outer_boundary_mask: np.ndarray,
    buffer_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct buffered tangential Dirichlet cells and vectors."""

    shape = occupancy_grid.shape
    interface_mask = np.zeros(shape, dtype=bool)
    normals = np.zeros(shape + (2,), dtype=np.float64)
    prescribed_vectors = np.zeros_like(normals)

    active_components = tuple(
        component
        for component in components
        if component.parameters.has_tangential_guidance
    )
    if not active_components:
        return interface_mask, normals, prescribed_vectors

    resolution = occupancy_grid.config.resolution
    if buffer_radius < 2.0 * resolution - 1e-12:
        raise ValueError(
            "tangential_buffer_radius must be at least two grid cells "
            "when tangential guidance is enabled."
        )

    coverage_count = np.zeros(shape, dtype=np.int64)
    row_grid, column_grid = np.indices(shape)
    half_width = 0.5 * resolution

    distance_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for component in components:
        distances, nearest_normal_sums = _distance_and_normal_sums(
            row_grid,
            column_grid,
            component.occupied_indices,
            resolution=resolution,
        )
        distance_data[component.label] = (
            distances,
            nearest_normal_sums,
        )

        coverage_mask = (
            ~occupancy_grid.occupied_mask
            & (distances <= buffer_radius + half_width + 1e-12)
        )
        coverage_count[coverage_mask] += 1

    overlap_cells = np.argwhere(coverage_count > 1)
    if overlap_cells.size:
        row_index, column_index = overlap_cells[0]
        raise ValueError(
            "Tangential obstacle buffers overlap at grid cell "
            f"({int(row_index)}, {int(column_index)})."
        )

    for component in active_components:
        distances, nearest_normal_sums = distance_data[component.label]
        coverage_mask = (
            ~occupancy_grid.occupied_mask
            & (distances <= buffer_radius + half_width + 1e-12)
        )
        if np.any(coverage_mask & outer_boundary_mask):
            raise ValueError(
                "A tangential obstacle buffer reaches the outer grid "
                "boundary; enlarge the local grid or reduce the buffer."
            )

        ring_mask = (
            ~occupancy_grid.occupied_mask
            & (distances >= buffer_radius - half_width - 1e-12)
            & (distances <= buffer_radius + half_width + 1e-12)
        )
        if not np.any(ring_mask):
            raise ValueError(
                "No grid cells represent a requested tangential buffer; "
                "increase the grid resolution or buffer radius."
            )
        if np.any(ring_mask & obstacle_interface_mask):
            raise ValueError(
                "The tangential buffer intersects the normal obstacle "
                "interface; increase tangential_buffer_radius."
            )

        normal_magnitudes = np.linalg.norm(
            nearest_normal_sums,
            axis=2,
        )
        ambiguous_mask = ring_mask & (normal_magnitudes <= 1e-12)
        if np.any(ambiguous_mask):
            row_index, column_index = np.argwhere(ambiguous_mask)[0]
            raise ValueError(
                "Cannot determine a unique buffered-interface normal at "
                f"cell ({int(row_index)}, {int(column_index)})."
            )

        component_normals = np.zeros_like(normals)
        component_normals[ring_mask] = (
            nearest_normal_sums[ring_mask]
            / normal_magnitudes[ring_mask, None]
        )
        component_tangents = _oriented_tangents(
            component_normals,
            passing_side=component.parameters.passing_side,
        )

        interface_mask[ring_mask] = True
        normals[ring_mask] = component_normals[ring_mask]
        prescribed_vectors[ring_mask] = (
            component.parameters.tangential_strength
            * component_tangents[ring_mask]
        )

    return interface_mask, normals, prescribed_vectors


def _distance_and_normal_sums(
    row_grid: np.ndarray,
    column_grid: np.ndarray,
    occupied_indices: np.ndarray,
    *,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest occupied-cell distance and toward-obstacle vectors."""

    minimum_distance_squared = np.full(
        row_grid.shape,
        math.inf,
        dtype=np.float64,
    )
    normal_sums = np.zeros(row_grid.shape + (2,), dtype=np.float64)

    for occupied_row, occupied_column in occupied_indices:
        row_offsets = float(occupied_row) - row_grid
        column_offsets = float(occupied_column) - column_grid
        distance_squared = row_offsets**2 + column_offsets**2

        closer = distance_squared < minimum_distance_squared - 1e-12
        tied = np.isclose(
            distance_squared,
            minimum_distance_squared,
            rtol=0.0,
            atol=1e-12,
        )

        minimum_distance_squared[closer] = distance_squared[closer]
        normal_sums[closer, 0] = column_offsets[closer]
        normal_sums[closer, 1] = row_offsets[closer]
        tied &= ~closer
        normal_sums[tied, 0] += column_offsets[tied]
        normal_sums[tied, 1] += row_offsets[tied]

    distances = resolution * np.sqrt(minimum_distance_squared)
    return distances, normal_sums


def _oriented_tangents(
    normals: np.ndarray,
    *,
    passing_side: PassingSide,
) -> np.ndarray:
    """Rotate toward-obstacle normals into the requested passing direction."""

    tangents = np.zeros_like(normals)
    if passing_side is PassingSide.LEFT:
        tangents[..., 0] = -normals[..., 1]
        tangents[..., 1] = normals[..., 0]
    elif passing_side is PassingSide.RIGHT:
        tangents[..., 0] = normals[..., 1]
        tangents[..., 1] = -normals[..., 0]
    else:
        raise ValueError(f"Unsupported passing side: {passing_side!r}.")

    return tangents


def _outer_boundary_mask(shape: tuple[int, int]) -> np.ndarray:
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
    *,
    fixed_mask: np.ndarray,
    obstacle_interface_mask: np.ndarray,
    obstacle_normals: np.ndarray,
    tangential_interface_mask: np.ndarray,
    tangential_normals: np.ndarray,
    tangential_boundary_vectors: np.ndarray,
    component_labels: np.ndarray,
    iteration_count: int,
    maximum_residual: float,
    resolution: float,
    buffer_radius: float,
) -> TangentialLaplaceGuidanceFieldResult:
    """Freeze result arrays and construct the diagnostic result."""

    typed_arrays = (
        np.ascontiguousarray(vectors, dtype=np.float64),
        np.ascontiguousarray(fixed_mask, dtype=bool),
        np.ascontiguousarray(obstacle_interface_mask, dtype=bool),
        np.ascontiguousarray(obstacle_normals, dtype=np.float64),
        np.ascontiguousarray(tangential_interface_mask, dtype=bool),
        np.ascontiguousarray(tangential_normals, dtype=np.float64),
        np.ascontiguousarray(
            tangential_boundary_vectors,
            dtype=np.float64,
        ),
        np.ascontiguousarray(component_labels, dtype=np.int64),
    )
    for array in typed_arrays:
        array.setflags(write=False)

    (
        frozen_vectors,
        frozen_fixed_mask,
        frozen_obstacle_mask,
        frozen_obstacle_normals,
        frozen_tangential_mask,
        frozen_tangential_normals,
        frozen_tangential_vectors,
        frozen_component_labels,
    ) = typed_arrays

    return TangentialLaplaceGuidanceFieldResult(
        vectors_robot_xy=frozen_vectors,
        dirichlet_boundary_mask=frozen_fixed_mask,
        obstacle_interface_mask=frozen_obstacle_mask,
        obstacle_normals_robot_xy=frozen_obstacle_normals,
        iteration_count=int(iteration_count),
        maximum_residual=float(maximum_residual),
        resolution=float(resolution),
        tangential_interface_mask=frozen_tangential_mask,
        tangential_normals_robot_xy=frozen_tangential_normals,
        tangential_boundary_vectors_robot_xy=(
            frozen_tangential_vectors
        ),
        obstacle_component_labels=frozen_component_labels,
        tangential_buffer_radius=float(buffer_radius),
    )
