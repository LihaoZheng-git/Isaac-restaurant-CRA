"""Yaw-dependent configuration-space grids for Safe-SAGE.

``h_full(x, y, psi)`` requires a different unsafe spatial domain for every
robot yaw when the robot footprint is not circular. This module converts one
raw obstacle occupancy grid into a periodic stack of configuration-space
grids by taking a discrete Minkowski difference with the rotated footprint.

For an obstacle at ``o`` and a footprint point ``p`` rotated by yaw ``psi``,
the corresponding unsafe robot-reference position is

    q = o - R(psi) p.

This reflection is important for footprints whose reference point is not at
their geometric center. Semantic track IDs, risks, and normal boundary fluxes
are propagated from source obstacle cells into every expanded cell. When
expanded obstacles overlap, the more conservative semantic parameters win.

Array convention follows ``LocalOccupancyGrid``: rows increase with
robot-frame ``+Y`` and columns increase with robot-frame ``+X``.
"""

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from robot.safety.local_occupancy_grid import LocalOccupancyGrid


def _validated_nonnegative_real(
    value: object,
    *,
    name: str,
) -> float:
    """Return one finite nonnegative real value."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if result < 0.0:
        raise ValueError(f"{name} cannot be negative.")
    return result


def _validated_positive_real(
    value: object,
    *,
    name: str,
) -> float:
    """Return one finite positive real value."""

    result = _validated_nonnegative_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _validated_yaw_slice_count(value: object) -> int:
    """Return a yaw count suitable for a periodic field."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("yaw_slice_count must be an integer.")
    if value < 3:
        raise ValueError("yaw_slice_count must be at least three.")
    return value


@dataclass(frozen=True)
class RectangularRobotFootprint:
    """Axis-aligned footprint extents in the robot chassis frame.

    Extents are measured from the control/reference point, so the footprint
    may be asymmetric. ``front`` is ``+X``, ``rear`` is ``-X``, ``left`` is
    ``+Y``, and ``right`` is ``-Y``.
    """

    front_extent: float
    rear_extent: float
    left_extent: float
    right_extent: float
    safety_margin: float = 0.0

    def __post_init__(self) -> None:
        front = _validated_nonnegative_real(
            self.front_extent,
            name="front_extent",
        )
        rear = _validated_nonnegative_real(
            self.rear_extent,
            name="rear_extent",
        )
        left = _validated_nonnegative_real(
            self.left_extent,
            name="left_extent",
        )
        right = _validated_nonnegative_real(
            self.right_extent,
            name="right_extent",
        )
        margin = _validated_nonnegative_real(
            self.safety_margin,
            name="safety_margin",
        )
        if front + rear <= 0.0:
            raise ValueError(
                "front_extent + rear_extent must be greater than zero."
            )
        if left + right <= 0.0:
            raise ValueError(
                "left_extent + right_extent must be greater than zero."
            )

        object.__setattr__(self, "front_extent", front)
        object.__setattr__(self, "rear_extent", rear)
        object.__setattr__(self, "left_extent", left)
        object.__setattr__(self, "right_extent", right)
        object.__setattr__(self, "safety_margin", margin)

    @classmethod
    def centered(
        cls,
        *,
        length: float,
        width: float,
        safety_margin: float = 0.0,
    ) -> "RectangularRobotFootprint":
        """Construct a rectangle centered on the robot reference point."""

        resolved_length = _validated_positive_real(
            length,
            name="length",
        )
        resolved_width = _validated_positive_real(
            width,
            name="width",
        )
        return cls(
            front_extent=0.5 * resolved_length,
            rear_extent=0.5 * resolved_length,
            left_extent=0.5 * resolved_width,
            right_extent=0.5 * resolved_width,
            safety_margin=safety_margin,
        )

    @property
    def length(self) -> float:
        """Return the front-to-rear footprint length without margin."""

        return self.front_extent + self.rear_extent

    @property
    def width(self) -> float:
        """Return the left-to-right footprint width without margin."""

        return self.left_extent + self.right_extent


@dataclass(frozen=True)
class CircularRobotFootprint:
    """Yaw-invariant circular footprint for baselines and validation."""

    radius: float
    safety_margin: float = 0.0

    def __post_init__(self) -> None:
        radius = _validated_positive_real(self.radius, name="radius")
        margin = _validated_nonnegative_real(
            self.safety_margin,
            name="safety_margin",
        )
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "safety_margin", margin)


RobotFootprint = RectangularRobotFootprint | CircularRobotFootprint


@dataclass(frozen=True)
class SafeSAGEConfigurationSpaceConfig:
    """Footprint and uniform yaw-discretization settings."""

    footprint: RobotFootprint
    yaw_slice_count: int = 10

    def __post_init__(self) -> None:
        if not isinstance(
            self.footprint,
            (RectangularRobotFootprint, CircularRobotFootprint),
        ):
            raise TypeError(
                "footprint must be a rectangular or circular footprint."
            )
        object.__setattr__(
            self,
            "yaw_slice_count",
            _validated_yaw_slice_count(self.yaw_slice_count),
        )


@dataclass(frozen=True)
class SafeSAGEConfigurationSpaceResult:
    """Yaw angles and their aligned configuration-space grids."""

    yaw_angles_radians: np.ndarray
    occupancy_grids: tuple[LocalOccupancyGrid, ...]
    footprint: RobotFootprint
    footprint_cell_counts: np.ndarray

    @property
    def yaw_slice_count(self) -> int:
        """Return the number of yaw-dependent grids."""

        return len(self.occupancy_grids)


def build_safe_sage_configuration_space(
    source_grid: LocalOccupancyGrid,
    config: SafeSAGEConfigurationSpaceConfig,
) -> SafeSAGEConfigurationSpaceResult:
    """Expand raw obstacles by the rotated footprint at each yaw slice.

    The source grid must have ``obstacle_inflation_radius == 0``. Applying
    circular point inflation before this transformation would silently count
    part of the robot clearance twice.
    """

    _validate_source_grid(source_grid)
    if not isinstance(config, SafeSAGEConfigurationSpaceConfig):
        raise TypeError(
            "config must be a SafeSAGEConfigurationSpaceConfig."
        )

    yaw_angles = uniform_yaw_angles(config.yaw_slice_count)
    expanded_grids: list[LocalOccupancyGrid] = []
    footprint_cell_counts: list[int] = []

    for yaw in yaw_angles:
        offsets = _footprint_cell_offsets(
            config.footprint,
            yaw_radians=float(yaw),
            resolution=source_grid.config.resolution,
        )
        expanded_grids.append(
            _expand_grid_with_offsets(source_grid, offsets)
        )
        footprint_cell_counts.append(len(offsets))

    frozen_counts = _frozen_array(
        footprint_cell_counts,
        dtype=np.int64,
    )
    return SafeSAGEConfigurationSpaceResult(
        yaw_angles_radians=yaw_angles,
        occupancy_grids=tuple(expanded_grids),
        footprint=config.footprint,
        footprint_cell_counts=frozen_counts,
    )


def uniform_yaw_angles(yaw_slice_count: int = 10) -> np.ndarray:
    """Return uniform periodic yaw slices on ``[-pi, pi)``."""

    slice_count = _validated_yaw_slice_count(yaw_slice_count)
    yaw_angles = np.linspace(
        -math.pi,
        math.pi,
        num=slice_count,
        endpoint=False,
        dtype=np.float64,
    )
    return _frozen_array(yaw_angles, dtype=np.float64)


def _validate_source_grid(source_grid: LocalOccupancyGrid) -> None:
    """Validate raw-grid type, metadata, and absence of prior inflation."""

    if not isinstance(source_grid, LocalOccupancyGrid):
        raise TypeError("source_grid must be a LocalOccupancyGrid.")
    if not math.isclose(
        source_grid.config.obstacle_inflation_radius,
        0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "source_grid must use obstacle_inflation_radius=0 to avoid "
            "double inflation."
        )

    cell_arrays = (
        source_grid.occupied_mask,
        source_grid.cell_track_ids,
        source_grid.cell_risks,
        source_grid.cell_boundary_fluxes,
    )
    if any(array.shape != source_grid.shape for array in cell_arrays):
        raise ValueError(
            "Every source-grid cell array must match the configured shape."
        )
    if not np.isfinite(source_grid.cell_risks).all():
        raise ValueError("source-grid risks must be finite.")
    if not np.isfinite(source_grid.cell_boundary_fluxes).all():
        raise ValueError("source-grid boundary fluxes must be finite.")
    if np.any(source_grid.cell_track_ids < -1):
        raise ValueError("source-grid track IDs cannot be below -1.")


def _footprint_cell_offsets(
    footprint: RobotFootprint,
    *,
    yaw_radians: float,
    resolution: float,
) -> tuple[tuple[int, int], ...]:
    """Conservatively rasterize reflected footprint cells for one yaw."""

    maximum_radius = _maximum_footprint_radius(footprint)
    cell_half_extent = 0.5 * resolution
    cell_half_diagonal = math.sqrt(2.0) * cell_half_extent
    maximum_offset = int(
        math.ceil(
            (maximum_radius + cell_half_diagonal) / resolution
        )
    )
    offsets: list[tuple[int, int]] = []

    for row_offset in range(-maximum_offset, maximum_offset + 1):
        for column_offset in range(
            -maximum_offset,
            maximum_offset + 1,
        ):
            candidate_minus_obstacle_x = column_offset * resolution
            candidate_minus_obstacle_y = row_offset * resolution
            obstacle_in_footprint_x = -candidate_minus_obstacle_x
            obstacle_in_footprint_y = -candidate_minus_obstacle_y

            if _footprint_overlaps_grid_cell(
                footprint,
                cell_center_grid_xy=(
                    obstacle_in_footprint_x,
                    obstacle_in_footprint_y,
                ),
                yaw_radians=yaw_radians,
                cell_half_extent=cell_half_extent,
            ):
                offsets.append((row_offset, column_offset))

    if (0, 0) not in offsets:
        raise RuntimeError(
            "The robot footprint must contain its reference point."
        )
    return tuple(offsets)


def _maximum_footprint_radius(footprint: RobotFootprint) -> float:
    """Return a yaw-independent bounding radius including safety margin."""

    if isinstance(footprint, CircularRobotFootprint):
        return footprint.radius + footprint.safety_margin

    maximum_x = max(
        footprint.front_extent,
        footprint.rear_extent,
    ) + footprint.safety_margin
    maximum_y = max(
        footprint.left_extent,
        footprint.right_extent,
    ) + footprint.safety_margin
    return math.hypot(maximum_x, maximum_y)


def _footprint_overlaps_grid_cell(
    footprint: RobotFootprint,
    *,
    cell_center_grid_xy: tuple[float, float],
    yaw_radians: float,
    cell_half_extent: float,
) -> bool:
    """Return whether an axis-aligned grid cell intersects the footprint."""

    center_x, center_y = cell_center_grid_xy
    tolerance = 1e-12
    if isinstance(footprint, CircularRobotFootprint):
        radius = footprint.radius + footprint.safety_margin
        nearest_x = max(abs(center_x) - cell_half_extent, 0.0)
        nearest_y = max(abs(center_y) - cell_half_extent, 0.0)
        return math.hypot(nearest_x, nearest_y) <= radius + tolerance

    cosine = math.cos(yaw_radians)
    sine = math.sin(yaw_radians)
    margin = footprint.safety_margin
    rectangle_half_x = (
        0.5 * (footprint.front_extent + footprint.rear_extent)
        + margin
    )
    rectangle_half_y = (
        0.5 * (footprint.left_extent + footprint.right_extent)
        + margin
    )
    rectangle_center_local_x = 0.5 * (
        footprint.front_extent - footprint.rear_extent
    )
    rectangle_center_local_y = 0.5 * (
        footprint.left_extent - footprint.right_extent
    )
    rectangle_center_grid_x = (
        cosine * rectangle_center_local_x
        - sine * rectangle_center_local_y
    )
    rectangle_center_grid_y = (
        sine * rectangle_center_local_x
        + cosine * rectangle_center_local_y
    )
    separation_x = center_x - rectangle_center_grid_x
    separation_y = center_y - rectangle_center_grid_y
    absolute_cosine = abs(cosine)
    absolute_sine = abs(sine)

    # Separating-axis test for an oriented rectangle and one axis-aligned
    # square. Marking every intersecting cell makes the discrete collision
    # set conservative without folding grid resolution into the policy
    # safety margin.
    grid_x_limit = (
        cell_half_extent
        + rectangle_half_x * absolute_cosine
        + rectangle_half_y * absolute_sine
    )
    if abs(separation_x) > grid_x_limit + tolerance:
        return False

    grid_y_limit = (
        cell_half_extent
        + rectangle_half_x * absolute_sine
        + rectangle_half_y * absolute_cosine
    )
    if abs(separation_y) > grid_y_limit + tolerance:
        return False

    projection_on_rectangle_x = abs(
        separation_x * cosine + separation_y * sine
    )
    rectangle_x_limit = (
        rectangle_half_x
        + cell_half_extent * (absolute_cosine + absolute_sine)
    )
    if projection_on_rectangle_x > rectangle_x_limit + tolerance:
        return False

    projection_on_rectangle_y = abs(
        -separation_x * sine + separation_y * cosine
    )
    rectangle_y_limit = (
        rectangle_half_y
        + cell_half_extent * (absolute_sine + absolute_cosine)
    )
    return projection_on_rectangle_y <= rectangle_y_limit + tolerance


def _expand_grid_with_offsets(
    source_grid: LocalOccupancyGrid,
    offsets: tuple[tuple[int, int], ...],
) -> LocalOccupancyGrid:
    """Dilate source cells and propagate conservative semantic metadata."""

    occupied_mask = np.zeros(source_grid.shape, dtype=bool)
    cell_track_ids = np.full(source_grid.shape, -1, dtype=np.int64)
    cell_risks = np.zeros(source_grid.shape, dtype=np.float64)
    cell_boundary_fluxes = np.zeros(
        source_grid.shape,
        dtype=np.float64,
    )
    row_count, column_count = source_grid.shape

    for source_row, source_column in np.argwhere(
        source_grid.occupied_mask
    ):
        track_id = int(
            source_grid.cell_track_ids[source_row, source_column]
        )
        risk = float(source_grid.cell_risks[source_row, source_column])
        boundary_flux = float(
            source_grid.cell_boundary_fluxes[
                source_row,
                source_column,
            ]
        )

        for row_offset, column_offset in offsets:
            row_index = int(source_row) + row_offset
            column_index = int(source_column) + column_offset
            if not (
                0 <= row_index < row_count
                and 0 <= column_index < column_count
            ):
                continue

            if _candidate_has_priority(
                risk=risk,
                boundary_flux=boundary_flux,
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
                cell_risks[row_index, column_index] = risk
                cell_boundary_fluxes[row_index, column_index] = (
                    boundary_flux
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
        config=source_grid.config,
        occupied_mask=occupied_mask,
        cell_track_ids=cell_track_ids,
        cell_risks=cell_risks,
        cell_boundary_fluxes=cell_boundary_fluxes,
        input_point_count=source_grid.input_point_count,
        rasterized_point_count=source_grid.rasterized_point_count,
    )


def _candidate_has_priority(
    *,
    risk: float,
    boundary_flux: float,
    track_id: int,
    occupied: bool,
    existing_risk: float,
    existing_boundary_flux: float,
    existing_track_id: int,
) -> bool:
    """Return whether new semantic metadata wins an expanded-cell overlap."""

    if not occupied:
        return True

    candidate_priority = (risk, abs(boundary_flux))
    existing_priority = (
        existing_risk,
        abs(existing_boundary_flux),
    )
    if candidate_priority != existing_priority:
        return candidate_priority > existing_priority
    return existing_track_id < 0 <= track_id


def _frozen_array(values: object, *, dtype: object) -> np.ndarray:
    """Return an owned, contiguous, read-only NumPy array."""

    frozen = np.array(values, dtype=dtype, copy=True, order="C")
    frozen.setflags(write=False)
    return frozen
