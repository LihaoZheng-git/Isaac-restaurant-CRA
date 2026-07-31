import numpy as np
import pytest

from robot.safety.class_safety_policy import (
    ClassSafetyParameters,
    ClassSafetyPolicy,
)
from robot.safety.local_occupancy_grid import (
    LocalOccupancyGridConfig,
    build_local_occupancy_grid,
)
from robot.safety.poisson_safety_function import (
    PoissonSafetyFunctionConfig,
    solve_poisson_safety_function,
)


def make_policy() -> ClassSafetyPolicy:
    """Return the minimal policy required by the grid builder."""

    return ClassSafetyPolicy(
        {
            "unknown": ClassSafetyParameters(
                risk=1.0,
                boundary_flux_magnitude=6.0,
            )
        }
    )


def make_grid(
    *,
    size: int = 7,
    resolution: float = 1.0,
    obstacle_points: np.ndarray | None = None,
):
    """Return one square local occupancy grid."""

    points = (
        np.empty((0, 2), dtype=np.float64)
        if obstacle_points is None
        else obstacle_points
    )
    config = LocalOccupancyGridConfig(
        resolution=resolution,
        minimum_x=0.0,
        maximum_x=size * resolution,
        minimum_y=0.0,
        maximum_y=size * resolution,
    )
    return build_local_occupancy_grid(
        points,
        class_safety_policy=make_policy(),
        config=config,
    )


def test_three_by_three_grid_has_exact_center_solution() -> None:
    result = solve_poisson_safety_function(
        make_grid(size=3),
        PoissonSafetyFunctionConfig(
            forcing=-1.0,
            convergence_tolerance=1e-12,
            residual_check_interval=1,
        ),
    )

    expected = np.zeros((3, 3), dtype=np.float64)
    expected[1, 1] = 0.25

    np.testing.assert_allclose(result.values, expected)
    assert result.iteration_count == 1
    assert result.maximum_residual == pytest.approx(0.0)


def test_outer_boundary_is_fixed_at_zero() -> None:
    result = solve_poisson_safety_function(make_grid())

    assert np.all(result.values[0, :] == 0.0)
    assert np.all(result.values[-1, :] == 0.0)
    assert np.all(result.values[:, 0] == 0.0)
    assert np.all(result.values[:, -1] == 0.0)
    assert np.all(result.dirichlet_boundary_mask[0, :])
    assert np.all(result.dirichlet_boundary_mask[-1, :])
    assert np.all(result.dirichlet_boundary_mask[:, 0])
    assert np.all(result.dirichlet_boundary_mask[:, -1])


def test_occupied_cells_are_zero_dirichlet_boundaries() -> None:
    grid = make_grid(
        size=7,
        obstacle_points=np.array([[3.5, 3.5]]),
    )
    result = solve_poisson_safety_function(grid)

    assert grid.occupied_mask[3, 3]
    assert result.dirichlet_boundary_mask[3, 3]
    assert result.values[3, 3] == pytest.approx(0.0)
    assert result.values[3, 2] > 0.0
    assert result.values[3, 4] > 0.0


def test_solution_is_positive_and_symmetric_without_obstacles() -> None:
    result = solve_poisson_safety_function(make_grid(size=7))

    assert result.minimum_value == pytest.approx(0.0)
    assert result.maximum_value > 0.0
    assert np.all(result.values[1:-1, 1:-1] > 0.0)
    np.testing.assert_allclose(result.values, result.values[::-1, :])
    np.testing.assert_allclose(result.values, result.values[:, ::-1])
    np.testing.assert_allclose(result.values, result.values.T)


def test_reported_residual_satisfies_tolerance() -> None:
    tolerance = 1e-7
    result = solve_poisson_safety_function(
        make_grid(size=9, resolution=0.2),
        PoissonSafetyFunctionConfig(
            convergence_tolerance=tolerance,
            residual_check_interval=3,
        ),
    )

    assert result.maximum_residual <= tolerance
    assert result.iteration_count > 0
    assert result.forcing == pytest.approx(-1.0)
    assert result.resolution == pytest.approx(0.2)


def test_solution_scales_linearly_with_forcing() -> None:
    grid = make_grid(size=7)
    result_one = solve_poisson_safety_function(
        grid,
        PoissonSafetyFunctionConfig(forcing=-1.0),
    )
    result_two = solve_poisson_safety_function(
        grid,
        PoissonSafetyFunctionConfig(forcing=-2.0),
    )

    np.testing.assert_allclose(
        result_two.values,
        2.0 * result_one.values,
        atol=2e-6,
    )


def test_all_cells_on_boundary_returns_without_iteration() -> None:
    result = solve_poisson_safety_function(make_grid(size=2))

    assert result.iteration_count == 0
    assert result.maximum_residual == pytest.approx(0.0)
    assert np.all(result.values == 0.0)
    assert np.all(result.dirichlet_boundary_mask)


def test_result_arrays_are_read_only_and_contiguous() -> None:
    result = solve_poisson_safety_function(make_grid())

    assert result.values.flags.c_contiguous
    assert result.dirichlet_boundary_mask.flags.c_contiguous
    assert not result.values.flags.writeable
    assert not result.dirichlet_boundary_mask.flags.writeable


def test_value_at_cell_returns_checked_scalar() -> None:
    result = solve_poisson_safety_function(make_grid(size=3))

    assert result.value_at_cell(1, 1) == pytest.approx(0.25)
    with pytest.raises(IndexError):
        result.value_at_cell(3, 0)
    with pytest.raises(TypeError):
        result.value_at_cell(True, 0)


def test_nonconvergence_raises_clear_error() -> None:
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_poisson_safety_function(
            make_grid(size=9),
            PoissonSafetyFunctionConfig(
                convergence_tolerance=1e-15,
                maximum_iterations=1,
                residual_check_interval=1,
            ),
        )


def test_invalid_occupancy_grid_is_rejected() -> None:
    with pytest.raises(TypeError, match="LocalOccupancyGrid"):
        solve_poisson_safety_function(np.zeros((5, 5)))


@pytest.mark.parametrize(
    "config_values",
    [
        {"forcing": 0.0},
        {"forcing": 1.0},
        {"forcing": float("nan")},
        {"convergence_tolerance": 0.0},
        {"maximum_iterations": 0},
        {"maximum_iterations": True},
        {"residual_check_interval": 0},
    ],
)
def test_invalid_solver_configuration_is_rejected(
    config_values: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        PoissonSafetyFunctionConfig(**config_values)
