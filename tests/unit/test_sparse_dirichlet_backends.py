from dataclasses import replace

import numpy as np
import pytest

from robot.safety.local_occupancy_grid import (
    LocalOccupancyGrid,
    LocalOccupancyGridConfig,
)
from robot.safety.poisson_safety_function import (
    PoissonSafetyFunctionConfig,
    solve_poisson_safety_function,
)
from robot.safety.vector_laplace_solver import (
    VectorLaplaceSolverConfig,
    solve_vector_laplace_dirichlet,
)


def grid():
    config = LocalOccupancyGridConfig(
        resolution=0.2, minimum_x=0, maximum_x=3.2, minimum_y=0, maximum_y=2.0
    )
    mask = np.zeros(config.shape, bool)
    mask[:, 7] = True
    mask[3:5, 3:5] = True
    return LocalOccupancyGrid(
        config,
        mask,
        np.full(config.shape, -1),
        mask.astype(float),
        -mask.astype(float),
        int(mask.sum()),
        int(mask.sum()),
    )


def test_direct_poisson_matches_jacobi_with_obstacles_disconnected_regions_and_variable_forcing():
    scene = grid()
    forcing = -1.0 - np.arange(np.prod(scene.shape)).reshape(scene.shape) * 0.01
    config = PoissonSafetyFunctionConfig(convergence_tolerance=1e-7)
    legacy = solve_poisson_safety_function(scene, config, forcing_field=forcing)
    direct = solve_poisson_safety_function(
        scene, replace(config, solver_backend="sparse_direct"), forcing_field=forcing
    )
    np.testing.assert_allclose(direct.values, legacy.values, atol=1e-7)
    assert direct.maximum_residual <= config.convergence_tolerance
    assert (direct.values[direct.dirichlet_boundary_mask] == 0).all()
    assert not direct.values.flags.writeable


def test_direct_vector_matches_sor_and_preserves_exact_dirichlet_data():
    scene = grid()
    fixed = scene.occupied_mask.copy()
    fixed[[0, -1]] = True
    fixed[:, [0, -1]] = True
    initial = np.random.default_rng(42).normal(size=(*scene.shape, 3))
    before = initial.copy()
    config = VectorLaplaceSolverConfig(convergence_tolerance=1e-7)
    legacy = solve_vector_laplace_dirichlet(
        initial, fixed, resolution=0.2, config=config
    )
    direct = solve_vector_laplace_dirichlet(
        initial,
        fixed,
        resolution=0.2,
        config=replace(config, solver_backend="sparse_direct"),
    )
    np.testing.assert_allclose(direct.vectors, legacy.vectors, atol=1e-7)
    np.testing.assert_array_equal(direct.vectors[fixed], initial[fixed])
    np.testing.assert_array_equal(initial, before)
    assert direct.maximum_residual <= config.convergence_tolerance
    assert not direct.vectors.flags.writeable


def test_direct_backend_does_not_relax_residual_tolerance_or_accept_invalid_name():
    with pytest.raises(RuntimeError, match="residual gate"):
        solve_poisson_safety_function(
            grid(),
            PoissonSafetyFunctionConfig(
                solver_backend="sparse_direct", convergence_tolerance=1e-20
            ),
        )
    with pytest.raises(ValueError, match="backend"):
        VectorLaplaceSolverConfig(solver_backend="approximate")
