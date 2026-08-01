"""Robot-frame occupancy grid for Poisson safety-function construction.

The grid consumes the obstacle points and point-to-track identifiers produced
by the existing LiDAR perception pipeline. Semantic class policies are then
rasterized onto the same cells as the geometric obstacle points.

Array convention:

* columns increase with robot-frame ``+X`` (forward);
* rows increase with robot-frame ``+Y`` (left);
* cell indices are written as ``(row, column)`` or ``(y, x)``.

This module is pure NumPy and has no Isaac Sim dependency.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from robot.safety.class_safety_policy import (
    ClassSafetyParameters,
    ClassSafetyPolicy,
)
from robot.safety.semantic_obstacle import SemanticObstacle


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


def _validated_cell_count(
    span: float,
    resolution: float,
    *,
    axis_name: str,
) -> int:
    """Return an exact integer cell count for one configured axis."""

    raw_count = span / resolution
    cell_count = int(round(raw_count))

    if cell_count <= 0 or not math.isclose(
        raw_count,
        cell_count,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"The {axis_name} span must be an integer multiple of "
            "resolution."
        )

    return cell_count


@dataclass(frozen=True)
class LocalOccupancyGridConfig:
    """Geometry and obstacle-inflation settings for a local grid."""

    resolution: float = 0.10
    minimum_x: float = -1.0
    maximum_x: float = 5.0
    minimum_y: float = -3.0
    maximum_y: float = 3.0
    obstacle_inflation_radius: float = 0.0
    _row_count: int = field(init=False, repr=False)
    _column_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        resolution = _validated_real(
            self.resolution,
            name="resolution",
        )
        minimum_x = _validated_real(
            self.minimum_x,
            name="minimum_x",
        )
        maximum_x = _validated_real(
            self.maximum_x,
            name="maximum_x",
        )
        minimum_y = _validated_real(
            self.minimum_y,
            name="minimum_y",
        )
        maximum_y = _validated_real(
            self.maximum_y,
            name="maximum_y",
        )
        inflation_radius = _validated_real(
            self.obstacle_inflation_radius,
            name="obstacle_inflation_radius",
        )

        if resolution <= 0.0:
            raise ValueError("resolution must be greater than zero.")
        if maximum_x <= minimum_x:
            raise ValueError("maximum_x must be greater than minimum_x.")
        if maximum_y <= minimum_y:
            raise ValueError("maximum_y must be greater than minimum_y.")
        if inflation_radius < 0.0:
            raise ValueError(
                "obstacle_inflation_radius cannot be negative."
            )

        column_count = _validated_cell_count(
            maximum_x - minimum_x,
            resolution,
            axis_name="X",
        )
        row_count = _validated_cell_count(
            maximum_y - minimum_y,
            resolution,
            axis_name="Y",
        )

        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "minimum_x", minimum_x)
        object.__setattr__(self, "maximum_x", maximum_x)
        object.__setattr__(self, "minimum_y", minimum_y)
        object.__setattr__(self, "maximum_y", maximum_y)
        object.__setattr__(
            self,
            "obstacle_inflation_radius",
            inflation_radius,
        )
        object.__setattr__(self, "_row_count", row_count)
        object.__setattr__(self, "_column_count", column_count)

    @property
    def row_count(self) -> int:
        """Return the number of cells along robot-frame Y."""

        return self._row_count

    @property
    def column_count(self) -> int:
        """Return the number of cells along robot-frame X."""

        return self._column_count

    @property
    def shape(self) -> tuple[int, int]:
        """Return the NumPy array shape ``(rows, columns)``."""

        return self.row_count, self.column_count


@dataclass(frozen=True)
class LocalOccupancyGrid:
    """Rasterized geometry and semantic safety values."""

    config: LocalOccupancyGridConfig
    occupied_mask: np.ndarray
    cell_track_ids: np.ndarray
    cell_risks: np.ndarray
    cell_boundary_fluxes: np.ndarray
    input_point_count: int
    rasterized_point_count: int

    @property
    def shape(self) -> tuple[int, int]:
        """Return the common shape of every cell array."""

        return self.config.shape

    @property
    def occupied_cell_count(self) -> int:
        """Return the number of occupied cells after inflation."""

        return int(np.count_nonzero(self.occupied_mask))

    @property
    def discarded_point_count(self) -> int:
        """Return the number of input points outside the grid bounds."""

        return self.input_point_count - self.rasterized_point_count

    def robot_point_to_cell(
        self,
        point_robot_xy: np.ndarray,
    ) -> tuple[int, int] | None:
        """Map one robot-frame point to ``(row, column)``.

        Returns ``None`` when the point lies outside the half-open grid bounds.
        """

        point = _validated_points(
            np.asarray(point_robot_xy)[np.newaxis, :]
        )[0]

        if not _point_is_in_bounds(point, self.config):
            return None

        return _point_to_cell_unchecked(point, self.config)

    def cell_center_robot_xy(
        self,
        row_index: int,
        column_index: int,
    ) -> tuple[float, float]:
        """Return the robot-frame center of one grid cell."""

        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or isinstance(column_index, bool)
            or not isinstance(column_index, int)
        ):
            raise TypeError("Grid cell indices must be integers.")

        if not 0 <= row_index < self.config.row_count:
            raise IndexError("row_index is outside the grid.")
        if not 0 <= column_index < self.config.column_count:
            raise IndexError("column_index is outside the grid.")

        x_position = self.config.minimum_x + (
            column_index + 0.5
        ) * self.config.resolution
        y_position = self.config.minimum_y + (
            row_index + 0.5
        ) * self.config.resolution

        return float(x_position), float(y_position)


def build_local_occupancy_grid(
    obstacle_points_robot_xy: np.ndarray,
    *,
    class_safety_policy: ClassSafetyPolicy,
    point_track_ids: np.ndarray | None = None,
    semantic_obstacles: Iterable[SemanticObstacle] = (),
    config: LocalOccupancyGridConfig | None = None,
) -> LocalOccupancyGrid:
    """Rasterize tracked obstacle points and semantic PSF parameters.

    Points without a valid semantic association use the policy's conservative
    unknown-class parameters. Points outside the configured local bounds are
    discarded. If inflated obstacles overlap, the cell retains the parameters
    with the greater risk and then the greater flux magnitude.
    """

    if not isinstance(class_safety_policy, ClassSafetyPolicy):
        raise TypeError(
            "class_safety_policy must be a ClassSafetyPolicy."
        )

    grid_config = config or LocalOccupancyGridConfig()
    if not isinstance(grid_config, LocalOccupancyGridConfig):
        raise TypeError("config must be a LocalOccupancyGridConfig.")

    points = _validated_points(obstacle_points_robot_xy)
    track_ids = _validated_track_ids(
        point_track_ids,
        point_count=points.shape[0],
    )
    semantic_by_track_id = _semantic_obstacles_by_track_id(
        semantic_obstacles
    )

    occupied_mask = np.zeros(grid_config.shape, dtype=bool)
    cell_track_ids = np.full(
        grid_config.shape,
        -1,
        dtype=np.int64,
    )
    cell_risks = np.zeros(
        grid_config.shape,
        dtype=np.float64,
    )
    cell_boundary_fluxes = np.zeros(
        grid_config.shape,
        dtype=np.float64,
    )

    inflation_offsets = _inflation_cell_offsets(grid_config)
    unknown_parameters = class_safety_policy.parameters_for(
        class_safety_policy.unknown_class
    )
    rasterized_point_count = 0

    for point_index, point in enumerate(points):
        if not _point_is_in_bounds(point, grid_config):
            continue

        rasterized_point_count += 1
        source_row, source_column = _point_to_cell_unchecked(
            point,
            grid_config,
        )
        track_id = int(track_ids[point_index])
        parameters = _parameters_for_track(
            track_id,
            semantic_by_track_id=semantic_by_track_id,
            class_safety_policy=class_safety_policy,
            unknown_parameters=unknown_parameters,
        )

        for row_offset, column_offset in inflation_offsets:
            row_index = source_row + row_offset
            column_index = source_column + column_offset

            if not (
                0 <= row_index < grid_config.row_count
                and 0 <= column_index < grid_config.column_count
            ):
                continue

            if _candidate_is_safer(
                parameters,
                track_id=track_id,
                occupied=bool(occupied_mask[row_index, column_index]),
                existing_risk=float(cell_risks[row_index, column_index]),
                existing_boundary_flux=float(
                    cell_boundary_fluxes[row_index, column_index]
                ),
                existing_track_id=int(
                    cell_track_ids[row_index, column_index]
                ),
            ):
                cell_track_ids[row_index, column_index] = track_id
                cell_risks[row_index, column_index] = parameters.risk
                cell_boundary_fluxes[row_index, column_index] = (
                    parameters.boundary_flux
                )

            occupied_mask[row_index, column_index] = True

    arrays = (
        occupied_mask,
        cell_track_ids,
        cell_risks,
        cell_boundary_fluxes,
    )
    for array in arrays:
        array.setflags(write=False)

    return LocalOccupancyGrid(
        config=grid_config,
        occupied_mask=occupied_mask,
        cell_track_ids=cell_track_ids,
        cell_risks=cell_risks,
        cell_boundary_fluxes=cell_boundary_fluxes,
        input_point_count=int(points.shape[0]),
        rasterized_point_count=rasterized_point_count,
    )


def _validated_points(
    obstacle_points_robot_xy: np.ndarray,
) -> np.ndarray:
    """Return obstacle points as a validated contiguous float64 array."""

    try:
        points = np.asarray(
            obstacle_points_robot_xy,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "obstacle_points_robot_xy must contain real numbers."
        ) from error

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "obstacle_points_robot_xy must have shape (N, 2), "
            f"received {points.shape}."
        )
    if not np.isfinite(points).all():
        raise ValueError(
            "obstacle_points_robot_xy must contain only finite values."
        )

    return np.ascontiguousarray(points, dtype=np.float64)


def _validated_track_ids(
    point_track_ids: np.ndarray | None,
    *,
    point_count: int,
) -> np.ndarray:
    """Return aligned track identifiers, defaulting to unknown ``-1``."""

    if point_track_ids is None:
        return np.full(point_count, -1, dtype=np.int64)

    track_ids = np.asarray(point_track_ids)
    if track_ids.shape != (point_count,):
        raise ValueError(
            "point_track_ids must have shape "
            f"({point_count},), received {track_ids.shape}."
        )
    if np.issubdtype(track_ids.dtype, np.bool_) or not np.issubdtype(
        track_ids.dtype,
        np.integer,
    ):
        raise TypeError("point_track_ids must contain integers.")
    if np.any(track_ids < -1):
        raise ValueError("point_track_ids cannot contain values below -1.")

    return np.ascontiguousarray(track_ids, dtype=np.int64)


def _semantic_obstacles_by_track_id(
    semantic_obstacles: Iterable[SemanticObstacle],
) -> dict[int, SemanticObstacle]:
    """Validate semantics and index them by geometric track ID."""

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


def _parameters_for_track(
    track_id: int,
    *,
    semantic_by_track_id: dict[int, SemanticObstacle],
    class_safety_policy: ClassSafetyPolicy,
    unknown_parameters: ClassSafetyParameters,
) -> ClassSafetyParameters:
    """Return semantic parameters for one point-to-track association."""

    semantic_obstacle = semantic_by_track_id.get(track_id)
    if semantic_obstacle is None:
        return unknown_parameters

    return class_safety_policy.parameters_for_obstacle(
        semantic_obstacle
    )


def _point_is_in_bounds(
    point_robot_xy: np.ndarray,
    config: LocalOccupancyGridConfig,
) -> bool:
    """Return whether a point lies in the grid's half-open bounds."""

    return bool(
        config.minimum_x <= point_robot_xy[0] < config.maximum_x
        and config.minimum_y <= point_robot_xy[1] < config.maximum_y
    )


def _point_to_cell_unchecked(
    point_robot_xy: np.ndarray,
    config: LocalOccupancyGridConfig,
) -> tuple[int, int]:
    """Convert one known-in-bounds point to a grid cell."""

    column_index = int(
        math.floor(
            (point_robot_xy[0] - config.minimum_x)
            / config.resolution
        )
    )
    row_index = int(
        math.floor(
            (point_robot_xy[1] - config.minimum_y)
            / config.resolution
        )
    )

    return row_index, column_index


def _inflation_cell_offsets(
    config: LocalOccupancyGridConfig,
) -> tuple[tuple[int, int], ...]:
    """Return circular cell-center offsets for obstacle inflation."""

    radius = config.obstacle_inflation_radius
    maximum_offset = int(math.ceil(radius / config.resolution))
    offsets: list[tuple[int, int]] = []

    for row_offset in range(-maximum_offset, maximum_offset + 1):
        for column_offset in range(
            -maximum_offset,
            maximum_offset + 1,
        ):
            distance = config.resolution * math.hypot(
                row_offset,
                column_offset,
            )
            if distance <= radius + 1e-12:
                offsets.append((row_offset, column_offset))

    return tuple(offsets)


def _candidate_is_safer(
    parameters: ClassSafetyParameters,
    *,
    track_id: int,
    occupied: bool,
    existing_risk: float,
    existing_boundary_flux: float,
    existing_track_id: int,
) -> bool:
    """Return whether a candidate should own an overlapping grid cell."""

    if not occupied:
        return True

    candidate_priority = (
        parameters.risk,
        parameters.boundary_flux_magnitude,
    )
    existing_priority = (
        existing_risk,
        abs(existing_boundary_flux),
    )

    if candidate_priority != existing_priority:
        return candidate_priority > existing_priority

    return existing_track_id < 0 <= track_id
