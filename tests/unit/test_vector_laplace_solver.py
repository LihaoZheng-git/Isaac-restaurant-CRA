from dataclasses import replace

import numpy as np
import pytest

from robot.safety.vector_laplace_solver import (
    VectorLaplaceSolverConfig,
    solve_vector_laplace_dirichlet,
)


def linear_boundary_problem(
    *,
    size: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a field whose exact harmonic solution is ``(x, y)``."""

    rows, columns = np.indices((size, size))
    expected = np.stack((columns, rows), axis=2).astype(np.float64)

    fixed_mask = np.zeros((size, size), dtype=bool)
    fixed_mask[0, :] = True
    fixed_mask[-1, :] = True
    fixed_mask[:, 0] = True
    fixed_mask[:, -1] = True

    initial_vectors = np.zeros_like(expected)
    initial_vectors[fixed_mask] = expected[fixed_mask]
    return initial_vectors, fixed_mask, expected


def test_solver_recovers_linear_harmonic_field() -> None:
    initial, fixed_mask, expected = linear_boundary_problem()

    result = solve_vector_laplace_dirichlet(
        initial,
        fixed_mask,
        resolution=1.0,
        config=VectorLaplaceSolverConfig(
            convergence_tolerance=1e-9,
            residual_check_interval=2,
        ),
    )

    np.testing.assert_allclose(result.vectors, expected, atol=1e-8)
    assert result.maximum_residual <= 1e-9
    assert result.component_count == 2
    assert result.shape == (7, 7)


def test_fixed_interior_vector_is_preserved() -> None:
    initial, fixed_mask, _ = linear_boundary_problem()
    fixed_mask[3, 3] = True
    initial[3, 3] = (20.0, -4.0)

    result = solve_vector_laplace_dirichlet(
        initial,
        fixed_mask,
        resolution=0.5,
    )

    assert result.vectors[3, 3] == pytest.approx((20.0, -4.0))


def test_solver_does_not_modify_inputs() -> None:
    initial, fixed_mask, _ = linear_boundary_problem()
    initial_before = initial.copy()
    mask_before = fixed_mask.copy()

    solve_vector_laplace_dirichlet(
        initial,
        fixed_mask,
        resolution=1.0,
    )

    np.testing.assert_array_equal(initial, initial_before)
    np.testing.assert_array_equal(fixed_mask, mask_before)


def test_result_vector_array_is_contiguous_and_read_only() -> None:
    initial, fixed_mask, _ = linear_boundary_problem()

    result = solve_vector_laplace_dirichlet(
        initial,
        fixed_mask,
        resolution=1.0,
    )

    assert result.vectors.flags.c_contiguous
    assert not result.vectors.flags.writeable
    with pytest.raises(ValueError):
        result.vectors[2, 2, 0] = 3.0


def test_all_fixed_cells_return_without_iteration() -> None:
    initial = np.ones((3, 4, 2), dtype=np.float64)
    fixed_mask = np.ones((3, 4), dtype=bool)

    result = solve_vector_laplace_dirichlet(
        initial,
        fixed_mask,
        resolution=1.0,
    )

    assert result.iteration_count == 0
    assert result.maximum_residual == pytest.approx(0.0)
    np.testing.assert_array_equal(result.vectors, initial)


@pytest.mark.parametrize(
    "vectors",
    [
        np.zeros((4, 4)),
        np.zeros((4, 4, 0)),
        np.zeros((1, 4, 2)),
        np.full((4, 4, 2), np.nan),
    ],
)
def test_invalid_vector_arrays_are_rejected(vectors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        solve_vector_laplace_dirichlet(
            vectors,
            np.ones(vectors.shape[:2], dtype=bool),
            resolution=1.0,
        )


def test_misaligned_fixed_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="fixed_mask"):
        solve_vector_laplace_dirichlet(
            np.zeros((5, 5, 2)),
            np.ones((4, 5), dtype=bool),
            resolution=1.0,
        )


def test_non_boolean_fixed_mask_is_rejected() -> None:
    with pytest.raises(TypeError, match="Boolean"):
        solve_vector_laplace_dirichlet(
            np.zeros((5, 5, 2)),
            np.ones((5, 5), dtype=np.int64),
            resolution=1.0,
        )


def test_unfixed_outer_boundary_is_rejected() -> None:
    initial, fixed_mask, _ = linear_boundary_problem()
    fixed_mask[0, 3] = False

    with pytest.raises(ValueError, match="outer-grid"):
        solve_vector_laplace_dirichlet(
            initial,
            fixed_mask,
            resolution=1.0,
        )


@pytest.mark.parametrize("resolution", [0.0, -1.0, np.inf, np.nan])
def test_invalid_resolution_is_rejected(resolution: float) -> None:
    initial, fixed_mask, _ = linear_boundary_problem()

    with pytest.raises(ValueError):
        solve_vector_laplace_dirichlet(
            initial,
            fixed_mask,
            resolution=resolution,
        )


def test_non_numeric_resolution_is_rejected() -> None:
    initial, fixed_mask, _ = linear_boundary_problem()

    with pytest.raises(TypeError, match="real number"):
        solve_vector_laplace_dirichlet(
            initial,
            fixed_mask,
            resolution="fine",
        )


def test_nonconvergence_is_reported() -> None:
    initial, fixed_mask, _ = linear_boundary_problem(size=9)
    config = replace(
        VectorLaplaceSolverConfig(),
        convergence_tolerance=1e-15,
        maximum_iterations=1,
        residual_check_interval=1,
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        solve_vector_laplace_dirichlet(
            initial,
            fixed_mask,
            resolution=1.0,
            config=config,
        )


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
def test_invalid_configuration_is_rejected(
    config_values: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        VectorLaplaceSolverConfig(**config_values)
