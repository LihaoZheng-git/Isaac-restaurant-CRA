from dataclasses import replace

import numpy as np
import pytest

from robot.safety.local_occupancy_grid import (
    LocalOccupancyGrid,
    LocalOccupancyGridConfig,
)
from robot.safety.safe_sage_cbf_filter import (
    SafeSAGECBFConfig,
    SafeSAGECBFSafetyFilter,
)
from robot.safety.safe_sage_configuration_space import (
    CircularRobotFootprint,
    SafeSAGEConfigurationSpaceResult,
    uniform_yaw_angles,
)
from robot.safety.safe_sage_h_full import SafeSAGEHFullResult


def make_filter(**overrides: float) -> SafeSAGECBFSafetyFilter:
    """Return one deterministic full-state filter."""

    values = {
        "gamma": 1.0,
        "minimum_linear_velocity": 0.0,
        "maximum_linear_velocity": 0.5,
        "minimum_angular_velocity": -1.0,
        "maximum_angular_velocity": 1.0,
        "linear_velocity_weight": 1.0,
        "angular_velocity_weight": 1.0,
    }
    values.update(overrides)
    return SafeSAGECBFSafetyFilter(SafeSAGECBFConfig(**values))


def make_grid(
    *,
    occupied: bool = False,
) -> LocalOccupancyGrid:
    """Return a 5x5 grid centered at the origin."""

    config = LocalOccupancyGridConfig(
        resolution=1.0,
        minimum_x=-2.5,
        maximum_x=2.5,
        minimum_y=-2.5,
        maximum_y=2.5,
    )
    occupied_mask = np.zeros(config.shape, dtype=bool)
    track_ids = np.full(config.shape, -1, dtype=np.int64)
    risks = np.zeros(config.shape, dtype=np.float64)
    fluxes = np.zeros(config.shape, dtype=np.float64)
    if occupied:
        occupied_mask[2, 2] = True
        risks[2, 2] = 1.0
        fluxes[2, 2] = -6.0

    return LocalOccupancyGrid(
        config=config,
        occupied_mask=occupied_mask,
        cell_track_ids=track_ids,
        cell_risks=risks,
        cell_boundary_fluxes=fluxes,
        input_point_count=int(occupied),
        rasterized_point_count=int(occupied),
    )


def make_fields(
    *,
    safety_value: float = 1.0,
    gradient: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw_slice_count: int = 4,
    unsafe_yaw_indices: tuple[int, ...] = (),
    per_yaw_safety_values: tuple[float, ...] | None = None,
) -> tuple[SafeSAGEConfigurationSpaceResult, SafeSAGEHFullResult]:
    """Return aligned synthetic configuration-space and h_full results."""

    yaw_angles = uniform_yaw_angles(yaw_slice_count)
    grids = tuple(
        make_grid(occupied=index in unsafe_yaw_indices)
        for index in range(yaw_slice_count)
    )
    shape = (yaw_slice_count,) + grids[0].shape
    if per_yaw_safety_values is None:
        values = np.full(shape, safety_value, dtype=np.float64)
    else:
        if len(per_yaw_safety_values) != yaw_slice_count:
            raise ValueError("per_yaw_safety_values length mismatch")
        values = np.empty(shape, dtype=np.float64)
        for yaw_index, value in enumerate(per_yaw_safety_values):
            values[yaw_index] = value

    spatial_gradients = np.empty(shape + (2,), dtype=np.float64)
    spatial_gradients[..., 0] = gradient[0]
    spatial_gradients[..., 1] = gradient[1]
    yaw_derivatives = np.full(shape, gradient[2], dtype=np.float64)
    boundary_masks = np.zeros(shape, dtype=bool)
    boundary_masks[:, 0, :] = True
    boundary_masks[:, -1, :] = True
    boundary_masks[:, :, 0] = True
    boundary_masks[:, :, -1] = True
    for yaw_index in unsafe_yaw_indices:
        boundary_masks[yaw_index, 2, 2] = True

    configuration_space = SafeSAGEConfigurationSpaceResult(
        yaw_angles_radians=yaw_angles,
        occupancy_grids=grids,
        footprint=CircularRobotFootprint(radius=0.25),
        footprint_cell_counts=np.ones(
            yaw_slice_count,
            dtype=np.int64,
        ),
    )
    h_full = SafeSAGEHFullResult(
        yaw_angles_radians=yaw_angles,
        values=values,
        spatial_gradients_robot_xy=spatial_gradients,
        yaw_derivatives=yaw_derivatives,
        dirichlet_boundary_masks=boundary_masks,
        guidance_divergence=np.zeros(shape, dtype=np.float64),
        forcing_fields=np.full(shape, -1.0, dtype=np.float64),
        iteration_counts=np.ones(yaw_slice_count, dtype=np.int64),
        maximum_residuals=np.zeros(
            yaw_slice_count,
            dtype=np.float64,
        ),
        resolution=1.0,
        softplus_beta=1.0,
        minimum_forcing_magnitude=1e-9,
    )
    return configuration_space, h_full


def apply_filter(
    safety_filter: SafeSAGECBFSafetyFilter,
    *,
    nominal_linear: float = 0.2,
    nominal_angular: float = 0.0,
    query_yaw: float = 0.0,
    temporal_derivative: float = 0.0,
    **field_kwargs: object,
):
    """Apply one filter to aligned synthetic fields at the origin."""

    configuration_space, h_full = make_fields(**field_kwargs)
    return safety_filter.filter_command(
        nominal_linear,
        nominal_angular,
        query_position_grid_xy=np.array([0.0, 0.0]),
        query_yaw_radians=query_yaw,
        configuration_space=configuration_space,
        h_full=h_full,
        temporal_safety_derivative=temporal_derivative,
    )


def test_safe_nominal_command_passes_through() -> None:
    result = apply_filter(
        make_filter(),
        safety_value=1.0,
        gradient=(-1.0, 0.0, 0.0),
    )

    assert result.safe_linear_velocity == pytest.approx(0.2)
    assert result.safe_angular_velocity == pytest.approx(0.0)
    assert result.constraint_feasible
    assert result.constraint_satisfied
    assert not result.intervention_active
    assert not result.emergency_stop_active


def test_spatial_gradient_reduces_forward_velocity() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.3,
        nominal_angular=0.4,
        safety_value=0.2,
        gradient=(-2.0, 0.0, 0.0),
    )

    assert result.linear_velocity_coefficient == pytest.approx(-2.0)
    assert result.angular_velocity_coefficient == pytest.approx(0.0)
    assert result.safe_linear_velocity == pytest.approx(0.1)
    assert result.safe_angular_velocity == pytest.approx(0.4)
    assert result.cbf_margin == pytest.approx(0.0)
    assert result.intervention_active


def test_yaw_gradient_directly_changes_angular_velocity() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.2,
        nominal_angular=0.6,
        safety_value=0.2,
        gradient=(0.0, 0.0, -1.0),
    )

    assert result.safe_linear_velocity == pytest.approx(0.2)
    assert result.safe_angular_velocity == pytest.approx(0.2)
    assert result.intervention_active
    assert result.constraint_satisfied


def test_robot_heading_rotates_spatial_gradient_into_forward_dynamics() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.3,
        query_yaw=0.5 * np.pi,
        safety_value=0.2,
        gradient=(0.0, -2.0, 0.0),
    )

    assert result.linear_velocity_coefficient == pytest.approx(-2.0)
    assert result.safe_linear_velocity == pytest.approx(0.1)


def test_negative_temporal_derivative_tightens_constraint() -> None:
    stationary = apply_filter(
        make_filter(),
        nominal_linear=0.3,
        safety_value=0.2,
        gradient=(-2.0, 0.0, 0.0),
    )
    approaching = apply_filter(
        make_filter(),
        nominal_linear=0.3,
        safety_value=0.2,
        gradient=(-2.0, 0.0, 0.0),
        temporal_derivative=-0.1,
    )

    assert stationary.safe_linear_velocity == pytest.approx(0.1)
    assert approaching.safe_linear_velocity == pytest.approx(0.05)
    assert (
        approaching.constraint_right_hand_side
        > stationary.constraint_right_hand_side
    )


def test_positive_temporal_derivative_relaxes_constraint() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.12,
        safety_value=0.2,
        gradient=(-2.0, 0.0, 0.0),
        temporal_derivative=0.1,
    )

    assert result.safe_linear_velocity == pytest.approx(0.12)
    assert not result.intervention_active


def test_periodic_yaw_interpolation_wraps_at_pi() -> None:
    result = apply_filter(
        make_filter(),
        query_yaw=0.75 * np.pi,
        per_yaw_safety_values=(1.0, 2.0, 3.0, 5.0),
        gradient=(0.0, 0.0, 0.0),
    )

    assert result.lower_yaw_slice == 3
    assert result.upper_yaw_slice == 0
    assert result.yaw_interpolation_weight == pytest.approx(0.5)
    assert result.sampled_safety_value == pytest.approx(3.0)


def test_unsafe_yaw_slice_triggers_conservative_emergency_stop() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.2,
        query_yaw=0.25 * np.pi,
        safety_value=0.2,
        gradient=(-1.0, 0.0, 0.0),
        unsafe_yaw_indices=(3,),
    )

    assert result.lower_yaw_slice == 2
    assert result.upper_yaw_slice == 3
    assert result.query_on_unsafe_boundary
    assert result.safe_linear_velocity == 0.0
    assert result.safe_angular_velocity == 0.0
    assert result.emergency_stop_active
    assert result.emergency_reason == "unsafe_configuration_boundary"


def test_infeasible_constraint_triggers_emergency_stop() -> None:
    result = apply_filter(
        make_filter(),
        nominal_angular=0.5,
        safety_value=0.0,
        gradient=(0.0, 0.0, -1.0),
        temporal_derivative=-2.0,
    )

    assert not result.constraint_feasible
    assert result.emergency_stop_active
    assert result.emergency_reason == "infeasible_cbf_constraint"


def test_actuator_limits_are_applied_without_cbf_intervention() -> None:
    result = apply_filter(
        make_filter(),
        nominal_linear=0.8,
        nominal_angular=-2.0,
        safety_value=1.0,
        gradient=(0.0, 0.0, 0.0),
    )

    assert result.safe_linear_velocity == pytest.approx(0.5)
    assert result.safe_angular_velocity == pytest.approx(-1.0)
    assert result.intervention_active


def test_angular_weight_changes_combined_projection() -> None:
    equal_weight = apply_filter(
        make_filter(),
        nominal_linear=0.2,
        nominal_angular=0.0,
        safety_value=0.0,
        gradient=(-1.0, 0.0, 1.0),
    )
    angular_expensive = apply_filter(
        make_filter(angular_velocity_weight=4.0),
        nominal_linear=0.2,
        nominal_angular=0.0,
        safety_value=0.0,
        gradient=(-1.0, 0.0, 1.0),
    )

    assert equal_weight.safe_linear_velocity == pytest.approx(0.1)
    assert equal_weight.safe_angular_velocity == pytest.approx(0.1)
    assert angular_expensive.safe_linear_velocity == pytest.approx(0.04)
    assert angular_expensive.safe_angular_velocity == pytest.approx(0.04)


def test_query_outside_cell_center_domain_is_rejected() -> None:
    configuration_space, h_full = make_fields()

    with pytest.raises(ValueError, match="cell-center domain"):
        make_filter().filter_command(
            0.2,
            0.0,
            query_position_grid_xy=np.array([2.4, 0.0]),
            query_yaw_radians=0.0,
            configuration_space=configuration_space,
            h_full=h_full,
        )


def test_field_yaw_mismatch_is_rejected() -> None:
    configuration_space, h_full = make_fields()
    mismatched_h_full = replace(
        h_full,
        yaw_angles_radians=np.array(
            h_full.yaw_angles_radians,
            copy=True,
        )
        + 0.1,
    )

    with pytest.raises(ValueError, match="yaw angles"):
        make_filter().filter_command(
            0.2,
            0.0,
            query_position_grid_xy=np.array([0.0, 0.0]),
            query_yaw_radians=0.0,
            configuration_space=configuration_space,
            h_full=mismatched_h_full,
        )


def test_nonuniform_yaw_grid_is_rejected() -> None:
    configuration_space, h_full = make_fields()
    nonuniform_angles = np.array(
        configuration_space.yaw_angles_radians,
        copy=True,
    )
    nonuniform_angles[1] += 0.1
    mismatched_configuration_space = replace(
        configuration_space,
        yaw_angles_radians=nonuniform_angles,
    )
    mismatched_h_full = replace(
        h_full,
        yaw_angles_radians=nonuniform_angles,
    )

    with pytest.raises(ValueError, match="uniformly spaced"):
        make_filter().filter_command(
            0.2,
            0.0,
            query_position_grid_xy=np.array([0.0, 0.0]),
            query_yaw_radians=0.0,
            configuration_space=mismatched_configuration_space,
            h_full=mismatched_h_full,
        )


def test_malformed_spatial_gradient_shape_is_rejected() -> None:
    configuration_space, h_full = make_fields()
    malformed_h_full = replace(
        h_full,
        spatial_gradients_robot_xy=np.zeros(h_full.values.shape),
    )

    with pytest.raises(ValueError, match="spatial gradients"):
        make_filter().filter_command(
            0.2,
            0.0,
            query_position_grid_xy=np.array([0.0, 0.0]),
            query_yaw_radians=0.0,
            configuration_space=configuration_space,
            h_full=malformed_h_full,
        )


@pytest.mark.parametrize(
    "config_values",
    [
        {"gamma": 0.0},
        {"minimum_linear_velocity": 0.2},
        {"maximum_angular_velocity": -0.1},
        {"linear_velocity_weight": 0.0},
        {"angular_velocity_weight": -1.0},
        {"numerical_tolerance": 0.0},
        {"intervention_tolerance": -1.0},
    ],
)
def test_invalid_configuration_is_rejected(
    config_values: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        SafeSAGECBFConfig(**config_values)


@pytest.mark.parametrize(
    ("linear", "angular", "yaw", "temporal"),
    [
        (float("nan"), 0.0, 0.0, 0.0),
        (0.2, float("inf"), 0.0, 0.0),
        (0.2, 0.0, float("nan"), 0.0),
        (0.2, 0.0, 0.0, float("inf")),
    ],
)
def test_nonfinite_filter_inputs_are_rejected(
    linear: float,
    angular: float,
    yaw: float,
    temporal: float,
) -> None:
    configuration_space, h_full = make_fields()

    with pytest.raises(ValueError):
        make_filter().filter_command(
            linear,
            angular,
            query_position_grid_xy=np.array([0.0, 0.0]),
            query_yaw_radians=yaw,
            configuration_space=configuration_space,
            h_full=h_full,
            temporal_safety_derivative=temporal,
        )
