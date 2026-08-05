import math

import numpy as np
import pytest

from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldResult,
)
from robot.safety.local_occupancy_grid import (
    LocalOccupancyGrid,
    LocalOccupancyGridConfig,
)
from robot.safety.poisson_safety_function import (
    PoissonSafetyFunctionConfig,
)
from robot.safety.safe_sage_h_full import (
    SafeSAGEHFullConfig,
    build_safe_sage_forcing_field,
    compute_guidance_divergence,
    solve_safe_sage_h_full,
    uniform_yaw_angles,
)


def make_grid(
    *,
    size: int = 5,
    resolution: float = 1.0,
    occupied_cell: tuple[int, int] | None = None,
) -> LocalOccupancyGrid:
    """Return a deterministic grid without invoking perception code."""

    config = LocalOccupancyGridConfig(
        resolution=resolution,
        minimum_x=0.0,
        maximum_x=size * resolution,
        minimum_y=0.0,
        maximum_y=size * resolution,
    )
    occupied = np.zeros(config.shape, dtype=bool)
    track_ids = np.full(config.shape, -1, dtype=np.int64)
    risks = np.zeros(config.shape, dtype=np.float64)
    fluxes = np.zeros(config.shape, dtype=np.float64)
    if occupied_cell is not None:
        occupied[occupied_cell] = True
        track_ids[occupied_cell] = 1
        risks[occupied_cell] = 1.0
        fluxes[occupied_cell] = -6.0

    return LocalOccupancyGrid(
        config=config,
        occupied_mask=occupied,
        cell_track_ids=track_ids,
        cell_risks=risks,
        cell_boundary_fluxes=fluxes,
        input_point_count=int(occupied_cell is not None),
        rasterized_point_count=int(occupied_cell is not None),
    )


def make_guidance(
    grid: LocalOccupancyGrid,
    *,
    divergence: float,
) -> LaplaceGuidanceFieldResult:
    """Return a linear field with known constant divergence."""

    row_count, column_count = grid.shape
    x_coordinates = (
        np.arange(column_count, dtype=np.float64)
        * grid.config.resolution
    )
    vectors = np.zeros((row_count, column_count, 2), dtype=np.float64)
    vectors[..., 0] = divergence * x_coordinates[np.newaxis, :]
    mask = np.zeros(grid.shape, dtype=bool)
    normals = np.zeros(grid.shape + (2,), dtype=np.float64)
    return LaplaceGuidanceFieldResult(
        vectors_robot_xy=vectors,
        dirichlet_boundary_mask=mask,
        obstacle_interface_mask=mask.copy(),
        obstacle_normals_robot_xy=normals,
        iteration_count=0,
        maximum_residual=0.0,
        resolution=grid.config.resolution,
    )


def make_config(yaw_slice_count: int = 4) -> SafeSAGEHFullConfig:
    """Return fast, strict settings for the small synthetic grids."""

    return SafeSAGEHFullConfig(
        yaw_slice_count=yaw_slice_count,
        poisson_config=PoissonSafetyFunctionConfig(
            convergence_tolerance=1e-10,
            residual_check_interval=1,
        ),
    )


def test_default_matches_paper_yaw_discretization() -> None:
    config = SafeSAGEHFullConfig()

    assert config.yaw_slice_count == 10
    assert config.yaw_spacing_radians == pytest.approx(math.tau / 10.0)
    np.testing.assert_allclose(
        uniform_yaw_angles(),
        np.linspace(-math.pi, math.pi, 10, endpoint=False),
    )


def test_guidance_divergence_uses_robot_xy_axis_order() -> None:
    grid = make_grid()
    guidance = make_guidance(grid, divergence=2.5)

    divergence = compute_guidance_divergence(grid, guidance)

    np.testing.assert_allclose(divergence, 2.5)


def test_softplus_forcing_is_finite_strictly_negative_and_stable() -> None:
    divergence = np.array(
        [[-1000.0, 0.0, 1000.0]],
        dtype=np.float64,
    )

    forcing = build_safe_sage_forcing_field(
        divergence,
        SafeSAGEHFullConfig(yaw_slice_count=3),
    )

    assert np.isfinite(forcing).all()
    assert np.all(forcing < 0.0)
    assert forcing[0, 0] == pytest.approx(-1000.0)
    assert forcing[0, 1] == pytest.approx(-math.log(2.0))
    assert forcing[0, 2] == pytest.approx(-1e-9)


def test_full_solver_stacks_yaw_slices_and_builds_derivatives() -> None:
    grid = make_grid()
    divergences = (0.0, 1.0, 2.0, -1.0)
    result = solve_safe_sage_h_full(
        [grid] * 4,
        [
            make_guidance(grid, divergence=value)
            for value in divergences
        ],
        make_config(),
    )

    assert result.shape == (4, 5, 5)
    assert result.spatial_gradients_robot_xy.shape == (4, 5, 5, 2)
    assert result.yaw_derivatives.shape == result.shape
    assert result.guidance_divergence.shape == result.shape
    assert result.forcing_fields.shape == result.shape
    assert result.iteration_counts.shape == (4,)
    assert result.maximum_residuals.shape == (4,)
    np.testing.assert_allclose(
        result.guidance_divergence[:, 2, 2],
        divergences,
    )

    spacing = math.tau / 4.0
    expected_first_derivative = (
        result.values[1] - result.values[3]
    ) / (2.0 * spacing)
    np.testing.assert_allclose(
        result.yaw_derivatives[0],
        expected_first_derivative,
    )


def test_yaw_interpolation_wraps_across_pi_boundary() -> None:
    grid = make_grid()
    result = solve_safe_sage_h_full(
        [grid] * 4,
        [
            make_guidance(grid, divergence=value)
            for value in (0.0, 1.0, 2.0, -1.0)
        ],
        make_config(),
    )
    row_index = column_index = 2
    half_last_interval = math.pi - result.yaw_spacing_radians / 2.0

    expected = 0.5 * (
        result.values[3, row_index, column_index]
        + result.values[0, row_index, column_index]
    )
    assert result.value_at_cell(
        half_last_interval,
        row_index,
        column_index,
    ) == pytest.approx(expected)
    assert result.value_at_cell(
        math.pi,
        row_index,
        column_index,
    ) == pytest.approx(result.values[0, row_index, column_index])


def test_yaw_specific_unsafe_boundaries_are_preserved() -> None:
    grids = [
        make_grid(occupied_cell=cell)
        for cell in ((2, 1), (1, 2), (2, 3), (3, 2))
    ]
    result = solve_safe_sage_h_full(
        grids,
        [make_guidance(grid, divergence=0.0) for grid in grids],
        make_config(),
    )

    for yaw_index, cell in enumerate(
        ((2, 1), (1, 2), (2, 3), (3, 2))
    ):
        assert result.dirichlet_boundary_masks[
            yaw_index,
            cell[0],
            cell[1],
        ]
        assert result.values[yaw_index, cell[0], cell[1]] == 0.0


def test_result_arrays_are_owned_contiguous_and_read_only() -> None:
    grid = make_grid()
    result = solve_safe_sage_h_full(
        [grid] * 3,
        [make_guidance(grid, divergence=0.0)] * 3,
        make_config(yaw_slice_count=3),
    )

    arrays = (
        result.yaw_angles_radians,
        result.values,
        result.spatial_gradients_robot_xy,
        result.yaw_derivatives,
        result.dirichlet_boundary_masks,
        result.guidance_divergence,
        result.forcing_fields,
        result.iteration_counts,
        result.maximum_residuals,
    )
    assert all(array.flags.c_contiguous for array in arrays)
    assert all(not array.flags.writeable for array in arrays)


def test_rejects_stack_length_and_grid_configuration_mismatches() -> None:
    grid = make_grid()
    guidance = make_guidance(grid, divergence=0.0)
    config = make_config(yaw_slice_count=3)

    with pytest.raises(ValueError, match="length"):
        solve_safe_sage_h_full([grid] * 2, [guidance] * 2, config)

    different_grid = make_grid(resolution=0.5)
    with pytest.raises(ValueError, match="grid configuration"):
        solve_safe_sage_h_full(
            [grid, grid, different_grid],
            [
                guidance,
                guidance,
                make_guidance(different_grid, divergence=0.0),
            ],
            config,
        )


@pytest.mark.parametrize("yaw_slice_count", [True, 0, 1, 2, 3.0])
def test_invalid_yaw_slice_count_is_rejected(
    yaw_slice_count: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SafeSAGEHFullConfig(yaw_slice_count=yaw_slice_count)
