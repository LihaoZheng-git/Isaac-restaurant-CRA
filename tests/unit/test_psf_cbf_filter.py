from dataclasses import replace

import numpy as np
import pytest

from robot.safety.class_safety_policy import (
    ClassSafetyParameters,
    ClassSafetyPolicy,
)
from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldResult,
)
from robot.safety.local_occupancy_grid import (
    LocalOccupancyGridConfig,
    build_local_occupancy_grid,
)
from robot.safety.poisson_safety_function import (
    PoissonSafetyFunctionResult,
)
from robot.safety.psf_cbf_filter import (
    PSFCBFConfig,
    PSFCBFSafetyFilter,
)


def make_policy() -> ClassSafetyPolicy:
    """Return the minimal policy needed by the occupancy-grid builder."""

    return ClassSafetyPolicy(
        {
            "unknown": ClassSafetyParameters(
                risk=1.0,
                boundary_flux_magnitude=6.0,
            )
        }
    )


def make_filter(
    **config_overrides: object,
) -> PSFCBFSafetyFilter:
    """Return a filter with simple test-friendly command limits."""

    config_values = {
        "gamma": 1.0,
        "lookahead_distance": 0.5,
        "minimum_linear_velocity": 0.0,
        "maximum_linear_velocity": 0.5,
        "minimum_angular_velocity": -1.0,
        "maximum_angular_velocity": 1.0,
    }
    config_values.update(config_overrides)
    return PSFCBFSafetyFilter(PSFCBFConfig(**config_values))


def make_pipeline(
    *,
    safety_value: float = 1.0,
    guidance_vector: tuple[float, float] = (0.0, 0.0),
    lookahead_occupied: bool = False,
):
    """Return aligned synthetic grid, scalar, and vector PDE results."""

    grid_config = LocalOccupancyGridConfig(
        resolution=1.0,
        minimum_x=-1.0,
        maximum_x=4.0,
        minimum_y=-2.5,
        maximum_y=2.5,
    )
    obstacle_points = (
        np.array([[0.5, 0.0]])
        if lookahead_occupied
        else np.empty((0, 2))
    )
    grid = build_local_occupancy_grid(
        obstacle_points,
        class_safety_policy=make_policy(),
        config=grid_config,
    )

    safety_values = np.full(
        grid.shape,
        safety_value,
        dtype=np.float64,
    )
    guidance_vectors = np.empty(
        grid.shape + (2,),
        dtype=np.float64,
    )
    guidance_vectors[..., 0] = guidance_vector[0]
    guidance_vectors[..., 1] = guidance_vector[1]
    boundary_mask = np.zeros(grid.shape, dtype=bool)
    interface_mask = np.zeros(grid.shape, dtype=bool)
    normals = np.zeros(grid.shape + (2,), dtype=np.float64)

    safety_function = PoissonSafetyFunctionResult(
        values=safety_values,
        dirichlet_boundary_mask=boundary_mask,
        iteration_count=1,
        maximum_residual=0.0,
        forcing=-1.0,
        resolution=grid_config.resolution,
    )
    guidance_field = LaplaceGuidanceFieldResult(
        vectors_robot_xy=guidance_vectors,
        dirichlet_boundary_mask=boundary_mask,
        obstacle_interface_mask=interface_mask,
        obstacle_normals_robot_xy=normals,
        iteration_count=1,
        maximum_residual=0.0,
        resolution=grid_config.resolution,
    )

    return grid, safety_function, guidance_field


def apply_filter(
    safety_filter: PSFCBFSafetyFilter,
    *,
    nominal_linear: float,
    nominal_angular: float,
    safety_value: float,
    guidance_vector: tuple[float, float],
    lookahead_occupied: bool = False,
):
    """Apply a filter to one aligned synthetic pipeline."""

    grid, safety_function, guidance_field = make_pipeline(
        safety_value=safety_value,
        guidance_vector=guidance_vector,
        lookahead_occupied=lookahead_occupied,
    )
    return safety_filter.filter_command(
        nominal_linear,
        nominal_angular,
        grid,
        safety_function,
        guidance_field,
    )


def test_zero_guidance_preserves_nominal_command() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.2,
        nominal_angular=0.3,
        safety_value=1.0,
        guidance_vector=(0.0, 0.0),
    )

    assert result.safe_linear_velocity == pytest.approx(0.2)
    assert result.safe_angular_velocity == pytest.approx(0.3)
    assert result.constraint_feasible
    assert result.constraint_satisfied
    assert not result.intervention_active
    assert not result.emergency_stop_active


def test_actuator_limits_are_applied_without_barrier_intervention() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.8,
        nominal_angular=-1.5,
        safety_value=1.0,
        guidance_vector=(0.0, 0.0),
    )

    assert result.safe_linear_velocity == pytest.approx(0.5)
    assert result.safe_angular_velocity == pytest.approx(-1.0)
    assert result.intervention_active
    assert result.constraint_feasible


def test_front_obstacle_reduces_linear_velocity() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.3,
        nominal_angular=0.4,
        safety_value=0.2,
        guidance_vector=(-2.0, 0.0),
    )

    assert result.safe_linear_velocity == pytest.approx(0.1)
    assert result.safe_angular_velocity == pytest.approx(0.4)
    assert result.nominal_constraint_left_hand_side == pytest.approx(-0.6)
    assert result.safe_constraint_left_hand_side == pytest.approx(-0.2)
    assert result.constraint_right_hand_side == pytest.approx(-0.2)
    assert result.constraint_margin == pytest.approx(0.0)
    assert result.intervention_active


def test_side_obstacle_changes_angular_velocity() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.2,
        nominal_angular=0.6,
        safety_value=0.2,
        guidance_vector=(0.0, -2.0),
    )

    assert result.safe_linear_velocity == pytest.approx(0.2)
    assert result.safe_angular_velocity == pytest.approx(0.2)
    assert result.intervention_active
    assert result.constraint_satisfied


def test_combined_constraint_uses_box_edge_solution() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.5,
        nominal_angular=1.0,
        safety_value=0.2,
        guidance_vector=(-2.0, -1.0),
    )

    assert result.safe_linear_velocity == pytest.approx(0.0)
    assert result.safe_angular_velocity == pytest.approx(0.4)
    assert result.safe_constraint_left_hand_side == pytest.approx(-0.2)
    assert result.constraint_satisfied


def test_stronger_class_flux_intervenes_more() -> None:
    chair_result = apply_filter(
        make_filter(),
        nominal_linear=0.2,
        nominal_angular=0.0,
        safety_value=0.3,
        guidance_vector=(-3.0, 0.0),
    )
    person_result = apply_filter(
        make_filter(),
        nominal_linear=0.2,
        nominal_angular=0.0,
        safety_value=0.3,
        guidance_vector=(-6.0, 0.0),
    )

    assert chair_result.safe_linear_velocity == pytest.approx(0.1)
    assert person_result.safe_linear_velocity == pytest.approx(0.05)
    assert (
        person_result.safe_linear_velocity
        < chair_result.safe_linear_velocity
    )


def test_safe_nominal_command_passes_through() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.05,
        nominal_angular=0.0,
        safety_value=0.2,
        guidance_vector=(-2.0, 0.0),
    )

    assert result.safe_linear_velocity == pytest.approx(0.05)
    assert not result.intervention_active


def test_occupied_lookahead_point_triggers_emergency_stop() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.2,
        nominal_angular=0.3,
        safety_value=0.0,
        guidance_vector=(0.0, 0.0),
        lookahead_occupied=True,
    )

    assert result.safe_linear_velocity == pytest.approx(0.0)
    assert result.safe_angular_velocity == pytest.approx(0.0)
    assert result.lookahead_occupied
    assert not result.constraint_feasible
    assert not result.constraint_satisfied
    assert result.intervention_active
    assert result.emergency_stop_active


def test_negative_safety_value_triggers_emergency_stop() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.2,
        nominal_angular=0.0,
        safety_value=-0.1,
        guidance_vector=(1.0, 0.0),
    )

    assert result.safe_linear_velocity == pytest.approx(0.0)
    assert not result.constraint_feasible
    assert result.emergency_stop_active


def test_pipeline_shape_mismatch_is_rejected() -> None:
    grid, safety_function, guidance_field = make_pipeline()
    invalid_safety = replace(
        safety_function,
        values=np.zeros((4, 4)),
    )

    with pytest.raises(ValueError, match="shape"):
        make_filter().filter_command(
            0.2,
            0.0,
            grid,
            invalid_safety,
            guidance_field,
        )


def test_pipeline_resolution_mismatch_is_rejected() -> None:
    grid, safety_function, guidance_field = make_pipeline()
    invalid_guidance = replace(
        guidance_field,
        resolution=0.5,
    )

    with pytest.raises(ValueError, match="resolution"):
        make_filter().filter_command(
            0.2,
            0.0,
            grid,
            safety_function,
            invalid_guidance,
        )


def test_nonfinite_pipeline_value_is_rejected() -> None:
    grid, safety_function, guidance_field = make_pipeline()
    invalid_values = np.array(safety_function.values, copy=True)
    invalid_values[2, 1] = np.nan
    invalid_safety = replace(
        safety_function,
        values=invalid_values,
    )

    with pytest.raises(ValueError, match="finite"):
        make_filter().filter_command(
            0.2,
            0.0,
            grid,
            invalid_safety,
            guidance_field,
        )


def test_lookahead_outside_cell_center_domain_is_rejected() -> None:
    grid, safety_function, guidance_field = make_pipeline()
    safety_filter = make_filter(lookahead_distance=3.9)

    with pytest.raises(ValueError, match="cell-center domain"):
        safety_filter.filter_command(
            0.2,
            0.0,
            grid,
            safety_function,
            guidance_field,
        )


@pytest.mark.parametrize(
    "config_values",
    [
        {"gamma": 0.0},
        {"lookahead_distance": 0.0},
        {
            "minimum_linear_velocity": 0.5,
            "maximum_linear_velocity": 0.2,
        },
        {
            "minimum_angular_velocity": 0.0,
            "maximum_angular_velocity": -0.1,
        },
        {
            "minimum_linear_velocity": 0.1,
            "maximum_linear_velocity": 0.5,
        },
        {
            "minimum_angular_velocity": 0.1,
            "maximum_angular_velocity": 1.0,
        },
        {"numerical_tolerance": 0.0},
        {"intervention_tolerance": -1.0},
    ],
)
def test_invalid_filter_configuration_is_rejected(
    config_values: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        PSFCBFConfig(**config_values)


@pytest.mark.parametrize(
    ("linear_velocity", "angular_velocity"),
    [
        (float("nan"), 0.0),
        (0.2, float("inf")),
        (True, 0.0),
    ],
)
def test_invalid_nominal_command_is_rejected(
    linear_velocity: object,
    angular_velocity: object,
) -> None:
    grid, safety_function, guidance_field = make_pipeline()

    with pytest.raises((TypeError, ValueError)):
        make_filter().filter_command(
            linear_velocity,
            angular_velocity,
            grid,
            safety_function,
            guidance_field,
        )
