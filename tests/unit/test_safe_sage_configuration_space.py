import math

import numpy as np
import pytest

from robot.safety.local_occupancy_grid import (
    LocalOccupancyGrid,
    LocalOccupancyGridConfig,
)
from robot.safety.safe_sage_configuration_space import (
    CircularRobotFootprint,
    RectangularRobotFootprint,
    SafeSAGEConfigurationSpaceConfig,
    build_safe_sage_configuration_space,
    uniform_yaw_angles,
)
from robot.safety.safe_sage_h_full import (
    uniform_yaw_angles as h_full_yaw_angles,
)


def make_grid(
    *,
    size: int = 11,
    resolution: float = 1.0,
    inflation_radius: float = 0.0,
    occupied_cells: tuple[
        tuple[int, int, int, float, float], ...
    ] = (),
) -> LocalOccupancyGrid:
    """Return a raw grid with explicit semantic cell metadata."""

    config = LocalOccupancyGridConfig(
        resolution=resolution,
        minimum_x=-0.5 * size * resolution,
        maximum_x=0.5 * size * resolution,
        minimum_y=-0.5 * size * resolution,
        maximum_y=0.5 * size * resolution,
        obstacle_inflation_radius=inflation_radius,
    )
    occupied_mask = np.zeros(config.shape, dtype=bool)
    track_ids = np.full(config.shape, -1, dtype=np.int64)
    risks = np.zeros(config.shape, dtype=np.float64)
    boundary_fluxes = np.zeros(config.shape, dtype=np.float64)

    for row, column, track_id, risk, boundary_flux in occupied_cells:
        occupied_mask[row, column] = True
        track_ids[row, column] = track_id
        risks[row, column] = risk
        boundary_fluxes[row, column] = boundary_flux

    return LocalOccupancyGrid(
        config=config,
        occupied_mask=occupied_mask,
        cell_track_ids=track_ids,
        cell_risks=risks,
        cell_boundary_fluxes=boundary_fluxes,
        input_point_count=len(occupied_cells),
        rasterized_point_count=len(occupied_cells),
    )


def centered_obstacle_grid() -> LocalOccupancyGrid:
    """Return one semantic obstacle at the center of an 11x11 grid."""

    return make_grid(
        occupied_cells=((5, 5, 7, 0.9, -5.0),),
    )


def test_default_yaw_order_matches_h_full() -> None:
    result = build_safe_sage_configuration_space(
        centered_obstacle_grid(),
        SafeSAGEConfigurationSpaceConfig(
            footprint=CircularRobotFootprint(radius=1.0)
        ),
    )

    assert result.yaw_slice_count == 10
    np.testing.assert_allclose(
        result.yaw_angles_radians,
        uniform_yaw_angles(),
    )
    np.testing.assert_allclose(
        result.yaw_angles_radians,
        h_full_yaw_angles(),
    )


def test_centered_rectangle_is_equivalent_at_zero_and_pi() -> None:
    result = build_safe_sage_configuration_space(
        centered_obstacle_grid(),
        SafeSAGEConfigurationSpaceConfig(
            footprint=RectangularRobotFootprint.centered(
                length=4.0,
                width=2.0,
            ),
            yaw_slice_count=4,
        ),
    )

    mask_at_minus_pi = result.occupancy_grids[0].occupied_mask
    mask_at_zero = result.occupancy_grids[2].occupied_mask
    np.testing.assert_array_equal(mask_at_minus_pi, mask_at_zero)


def test_quarter_turn_swaps_rectangular_length_and_width() -> None:
    result = build_safe_sage_configuration_space(
        centered_obstacle_grid(),
        SafeSAGEConfigurationSpaceConfig(
            footprint=RectangularRobotFootprint.centered(
                length=4.0,
                width=2.0,
            ),
            yaw_slice_count=4,
        ),
    )

    mask_at_zero = result.occupancy_grids[2].occupied_mask
    mask_at_half_pi = result.occupancy_grids[3].occupied_mask
    np.testing.assert_array_equal(mask_at_half_pi, mask_at_zero.T)
    assert result.footprint_cell_counts[2] == 15
    assert result.footprint_cell_counts[3] == 15


def test_asymmetric_rectangle_uses_minkowski_reflection() -> None:
    result = build_safe_sage_configuration_space(
        centered_obstacle_grid(),
        SafeSAGEConfigurationSpaceConfig(
            footprint=RectangularRobotFootprint(
                front_extent=2.0,
                rear_extent=1.0,
                left_extent=0.5,
                right_extent=0.5,
            ),
            yaw_slice_count=4,
        ),
    )

    occupied_columns_at_zero = set(
        np.flatnonzero(
            result.occupancy_grids[2].occupied_mask[5]
        ).tolist()
    )
    assert occupied_columns_at_zero == {3, 4, 5, 6}


def test_subcell_front_extent_marks_overlapping_candidate_cell() -> None:
    source_grid = make_grid(
        resolution=0.2,
        occupied_cells=((5, 5, 7, 0.9, -5.0),),
    )
    result = build_safe_sage_configuration_space(
        source_grid,
        SafeSAGEConfigurationSpaceConfig(
            footprint=RectangularRobotFootprint(
                front_extent=0.14,
                rear_extent=0.587351,
                left_extent=0.2466,
                right_extent=0.2466,
                safety_margin=0.05,
            ),
            yaw_slice_count=4,
        ),
    )

    mask_at_zero = result.occupancy_grids[2].occupied_mask
    assert mask_at_zero[5, 4]
    assert not mask_at_zero[5, 3]


def test_circular_footprint_is_yaw_invariant() -> None:
    result = build_safe_sage_configuration_space(
        centered_obstacle_grid(),
        SafeSAGEConfigurationSpaceConfig(
            footprint=CircularRobotFootprint(
                radius=2.0,
                safety_margin=0.25,
            ),
            yaw_slice_count=8,
        ),
    )

    reference_mask = result.occupancy_grids[0].occupied_mask
    assert all(
        np.array_equal(grid.occupied_mask, reference_mask)
        for grid in result.occupancy_grids[1:]
    )
    assert np.all(
        result.footprint_cell_counts
        == result.footprint_cell_counts[0]
    )


def test_semantic_metadata_is_propagated_to_expanded_cells() -> None:
    result = build_safe_sage_configuration_space(
        centered_obstacle_grid(),
        SafeSAGEConfigurationSpaceConfig(
            footprint=CircularRobotFootprint(radius=1.0),
            yaw_slice_count=3,
        ),
    )

    for grid in result.occupancy_grids:
        occupied = grid.occupied_mask
        assert np.all(grid.cell_track_ids[occupied] == 7)
        assert np.all(grid.cell_risks[occupied] == pytest.approx(0.9))
        assert np.all(
            grid.cell_boundary_fluxes[occupied]
            == pytest.approx(-5.0)
        )


def test_overlap_retains_more_conservative_semantic_source() -> None:
    source_grid = make_grid(
        occupied_cells=(
            (5, 4, 1, 0.4, -3.0),
            (5, 6, 2, 0.9, -5.0),
        )
    )
    result = build_safe_sage_configuration_space(
        source_grid,
        SafeSAGEConfigurationSpaceConfig(
            footprint=CircularRobotFootprint(radius=1.0),
            yaw_slice_count=3,
        ),
    )

    overlap_cell = (5, 5)
    for grid in result.occupancy_grids:
        assert grid.cell_track_ids[overlap_cell] == 2
        assert grid.cell_risks[overlap_cell] == pytest.approx(0.9)
        assert grid.cell_boundary_fluxes[overlap_cell] == pytest.approx(
            -5.0
        )


def test_empty_source_produces_empty_aligned_grids() -> None:
    source_grid = make_grid()
    result = build_safe_sage_configuration_space(
        source_grid,
        SafeSAGEConfigurationSpaceConfig(
            footprint=CircularRobotFootprint(radius=1.0),
            yaw_slice_count=3,
        ),
    )

    assert all(
        not np.any(grid.occupied_mask)
        for grid in result.occupancy_grids
    )
    assert all(
        grid.config == source_grid.config
        for grid in result.occupancy_grids
    )


def test_outputs_are_read_only() -> None:
    result = build_safe_sage_configuration_space(
        centered_obstacle_grid(),
        SafeSAGEConfigurationSpaceConfig(
            footprint=CircularRobotFootprint(radius=1.0),
            yaw_slice_count=3,
        ),
    )

    assert not result.yaw_angles_radians.flags.writeable
    assert not result.footprint_cell_counts.flags.writeable
    for grid in result.occupancy_grids:
        assert not grid.occupied_mask.flags.writeable
        assert not grid.cell_track_ids.flags.writeable
        assert not grid.cell_risks.flags.writeable
        assert not grid.cell_boundary_fluxes.flags.writeable


def test_preinflated_source_grid_is_rejected() -> None:
    source_grid = make_grid(inflation_radius=0.2)

    with pytest.raises(ValueError, match="double inflation"):
        build_safe_sage_configuration_space(
            source_grid,
            SafeSAGEConfigurationSpaceConfig(
                footprint=CircularRobotFootprint(radius=1.0)
            ),
        )


@pytest.mark.parametrize(
    "constructor, expected_error",
    [
        (
            lambda: CircularRobotFootprint(radius=0.0),
            ValueError,
        ),
        (
            lambda: CircularRobotFootprint(radius=True),
            TypeError,
        ),
        (
            lambda: RectangularRobotFootprint.centered(
                length=-1.0,
                width=1.0,
            ),
            ValueError,
        ),
        (
            lambda: RectangularRobotFootprint(
                front_extent=0.0,
                rear_extent=0.0,
                left_extent=1.0,
                right_extent=1.0,
            ),
            ValueError,
        ),
        (
            lambda: SafeSAGEConfigurationSpaceConfig(
                footprint=CircularRobotFootprint(radius=1.0),
                yaw_slice_count=2,
            ),
            ValueError,
        ),
    ],
)
def test_invalid_configuration_is_rejected(
    constructor: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        constructor()


def test_periodic_angles_exclude_positive_pi() -> None:
    angles = uniform_yaw_angles(4)

    np.testing.assert_allclose(
        angles,
        [-math.pi, -0.5 * math.pi, 0.0, 0.5 * math.pi],
    )
    assert math.pi not in angles
