from dataclasses import replace

import numpy as np
import pytest

from robot.safety.class_safety_policy import (
    ClassSafetyParameters,
    ClassSafetyPolicy,
)
from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldConfig,
    solve_laplace_guidance_field,
)
from robot.safety.local_occupancy_grid import (
    LocalOccupancyGridConfig,
    build_local_occupancy_grid,
)
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.semantic_obstacle import SemanticObstacle


def make_policy() -> ClassSafetyPolicy:
    """Return different semantic fluxes for guidance tests."""

    return ClassSafetyPolicy(
        {
            "unknown": ClassSafetyParameters(
                risk=1.0,
                boundary_flux_magnitude=6.0,
            ),
            "chair": ClassSafetyParameters(
                risk=0.4,
                boundary_flux_magnitude=3.0,
            ),
            "person": ClassSafetyParameters(
                risk=1.0,
                boundary_flux_magnitude=6.0,
            ),
        }
    )


def make_semantic_obstacle(
    track_id: int,
    semantic_class: str,
) -> SemanticObstacle:
    """Return one deterministic semantic obstacle."""

    track = TrackedObstacle(
        track_id=track_id,
        centroid_world_xy=(0.0, 0.0),
        centroid_robot_xy=(0.0, 0.0),
        velocity_world_xy=(0.0, 0.0),
        velocity_robot_xy=(0.0, 0.0),
        point_count=1,
        observation_count=1,
        matched_previous_track=False,
    )
    return SemanticObstacle(
        tracked_obstacle=track,
        semantic_class=semantic_class,
        semantic_confidence=1.0,
    )


def make_grid(
    *,
    size: int = 9,
    points: np.ndarray | None = None,
    track_ids: np.ndarray | None = None,
    semantics: tuple[SemanticObstacle, ...] = (),
):
    """Return a square unit-resolution occupancy grid."""

    obstacle_points = (
        np.empty((0, 2), dtype=np.float64)
        if points is None
        else points
    )
    config = LocalOccupancyGridConfig(
        resolution=1.0,
        minimum_x=0.0,
        maximum_x=float(size),
        minimum_y=0.0,
        maximum_y=float(size),
    )
    return build_local_occupancy_grid(
        obstacle_points,
        point_track_ids=track_ids,
        semantic_obstacles=semantics,
        class_safety_policy=make_policy(),
        config=config,
    )


def test_empty_grid_produces_zero_guidance() -> None:
    result = solve_laplace_guidance_field(make_grid())

    assert result.iteration_count == 1
    assert result.maximum_residual == pytest.approx(0.0)
    assert result.maximum_magnitude == pytest.approx(0.0)
    assert not np.any(result.obstacle_interface_mask)
    assert np.all(result.vectors_robot_xy == 0.0)


def test_single_obstacle_boundary_vectors_point_away() -> None:
    grid = make_grid(
        size=9,
        points=np.array([[4.5, 4.5]]),
    )
    result = solve_laplace_guidance_field(grid)

    assert result.vector_at_cell(4, 3) == pytest.approx((-6.0, 0.0))
    assert result.vector_at_cell(4, 5) == pytest.approx((6.0, 0.0))
    assert result.vector_at_cell(3, 4) == pytest.approx((0.0, -6.0))
    assert result.vector_at_cell(5, 4) == pytest.approx((0.0, 6.0))


def test_diagonal_obstacle_normal_is_normalized() -> None:
    grid = make_grid(
        size=9,
        points=np.array([[4.5, 4.5]]),
    )
    result = solve_laplace_guidance_field(grid)

    normal = result.obstacle_normals_robot_xy[3, 3]
    vector = result.vectors_robot_xy[3, 3]

    np.testing.assert_allclose(
        normal,
        np.array([1.0, 1.0]) / np.sqrt(2.0),
    )
    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert np.linalg.norm(vector) == pytest.approx(6.0)
    assert np.all(vector < 0.0)


def test_semantic_class_changes_interface_magnitude() -> None:
    grid = make_grid(
        size=11,
        points=np.array(
            [
                [3.5, 5.5],
                [7.5, 5.5],
            ]
        ),
        track_ids=np.array([2, 8]),
        semantics=(
            make_semantic_obstacle(2, "chair"),
            make_semantic_obstacle(8, "person"),
        ),
    )
    result = solve_laplace_guidance_field(grid)

    chair_vector = result.vectors_robot_xy[5, 2]
    person_vector = result.vectors_robot_xy[5, 8]

    assert np.linalg.norm(chair_vector) == pytest.approx(3.0)
    assert np.linalg.norm(person_vector) == pytest.approx(6.0)
    assert chair_vector[0] < 0.0
    assert person_vector[0] > 0.0


def test_outer_boundary_and_occupied_cells_are_zero() -> None:
    grid = make_grid(
        points=np.array([[4.5, 4.5]]),
    )
    result = solve_laplace_guidance_field(grid)

    assert np.all(result.vectors_robot_xy[0, :, :] == 0.0)
    assert np.all(result.vectors_robot_xy[-1, :, :] == 0.0)
    assert np.all(result.vectors_robot_xy[:, 0, :] == 0.0)
    assert np.all(result.vectors_robot_xy[:, -1, :] == 0.0)
    assert np.all(result.vectors_robot_xy[4, 4] == 0.0)
    assert result.dirichlet_boundary_mask[4, 4]


def test_harmonic_solution_meets_residual_tolerance() -> None:
    tolerance = 1e-7
    result = solve_laplace_guidance_field(
        make_grid(
            size=13,
            points=np.array([[6.5, 6.5]]),
        ),
        LaplaceGuidanceFieldConfig(
            convergence_tolerance=tolerance,
            residual_check_interval=3,
        ),
    )

    assert result.iteration_count > 0
    assert result.maximum_residual <= tolerance


def test_no_unfixed_cells_returns_without_iteration() -> None:
    result = solve_laplace_guidance_field(
        make_grid(
            size=3,
            points=np.array([[1.5, 1.5]]),
        )
    )

    assert result.iteration_count == 0
    assert result.maximum_residual == pytest.approx(0.0)
    assert np.all(result.dirichlet_boundary_mask)


def test_result_arrays_are_read_only_and_contiguous() -> None:
    result = solve_laplace_guidance_field(
        make_grid(points=np.array([[4.5, 4.5]]))
    )

    arrays = (
        result.vectors_robot_xy,
        result.dirichlet_boundary_mask,
        result.obstacle_interface_mask,
        result.obstacle_normals_robot_xy,
    )
    for array in arrays:
        assert array.flags.c_contiguous
        assert not array.flags.writeable


def test_vector_at_cell_checks_indices() -> None:
    result = solve_laplace_guidance_field(make_grid())

    assert result.vector_at_cell(4, 4) == pytest.approx((0.0, 0.0))
    with pytest.raises(IndexError):
        result.vector_at_cell(9, 0)
    with pytest.raises(TypeError):
        result.vector_at_cell(True, 0)


def test_nonnegative_occupied_flux_is_rejected() -> None:
    grid = make_grid(points=np.array([[4.5, 4.5]]))
    invalid_fluxes = np.array(
        grid.cell_boundary_fluxes,
        copy=True,
    )
    invalid_fluxes[grid.occupied_mask] = 0.0
    invalid_grid = replace(
        grid,
        cell_boundary_fluxes=invalid_fluxes,
    )

    with pytest.raises(ValueError, match="strictly negative"):
        solve_laplace_guidance_field(invalid_grid)


def test_ambiguous_obstacle_normal_is_rejected() -> None:
    grid = make_grid(
        size=7,
        points=np.array(
            [
                [2.5, 3.5],
                [4.5, 3.5],
            ]
        ),
    )

    with pytest.raises(ValueError, match="unique obstacle normal"):
        solve_laplace_guidance_field(grid)


def test_nonconvergence_raises_clear_error() -> None:
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_laplace_guidance_field(
            make_grid(
                size=13,
                points=np.array([[6.5, 6.5]]),
            ),
            LaplaceGuidanceFieldConfig(
                convergence_tolerance=1e-15,
                maximum_iterations=1,
                residual_check_interval=1,
            ),
        )


def test_invalid_occupancy_grid_is_rejected() -> None:
    with pytest.raises(TypeError, match="LocalOccupancyGrid"):
        solve_laplace_guidance_field(np.zeros((5, 5)))


@pytest.mark.parametrize(
    "config_values",
    [
        {"convergence_tolerance": 0.0},
        {"convergence_tolerance": float("nan")},
        {"maximum_iterations": 0},
        {"maximum_iterations": True},
        {"residual_check_interval": 0},
        {"relaxation_factor": 0.0},
        {"relaxation_factor": 2.0},
    ],
)
def test_invalid_solver_configuration_is_rejected(
    config_values: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        LaplaceGuidanceFieldConfig(**config_values)
