from dataclasses import asdict, replace

import numpy as np
import pytest
from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldResult,
)
from robot.safety.local_occupancy_grid import (
    LocalOccupancyGrid,
    LocalOccupancyGridConfig,
)
from robot.safety.safe_sage_cbf_filter import (
    SafeSAGECBFConfig,
    SafeSAGECBFSafetyFilter,
    project_weighted_box_halfspace,
)
from robot.safety.safe_sage_configuration_space import (
    CircularRobotFootprint,
    SafeSAGEConfigurationSpaceResult,
    uniform_yaw_angles,
)
from robot.safety.safe_sage_h_full import SafeSAGEHFullResult
from robot.safety.safe_sage_temporal_derivative import (
    SafeSAGETemporalDerivativeResult,
)


def make_filter(**overrides: float) -> SafeSAGECBFSafetyFilter:
    """Return one deterministic paper-space filter."""

    values = {
        "gamma": 1.0,
        "minimum_planar_velocity": -0.5,
        "maximum_planar_velocity": 0.5,
        "minimum_angular_velocity": -1.0,
        "maximum_angular_velocity": 1.0,
        "planar_velocity_weight": 1.0,
        "angular_velocity_weight": 1.0,
    }
    values.update(overrides)
    return SafeSAGECBFSafetyFilter(SafeSAGECBFConfig(**values))


def make_grid(*, occupied: bool = False) -> LocalOccupancyGrid:
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


def make_guidance_field(
    grid: LocalOccupancyGrid,
    vector_xy: tuple[float, float],
) -> LaplaceGuidanceFieldResult:
    """Return one constant synthetic guidance slice."""

    vectors = np.empty(grid.shape + (2,), dtype=np.float64)
    vectors[..., 0] = vector_xy[0]
    vectors[..., 1] = vector_xy[1]
    return LaplaceGuidanceFieldResult(
        vectors_robot_xy=vectors,
        dirichlet_boundary_mask=np.zeros(grid.shape, dtype=bool),
        obstacle_interface_mask=np.zeros(grid.shape, dtype=bool),
        obstacle_normals_robot_xy=np.zeros(
            grid.shape + (2,),
            dtype=np.float64,
        ),
        iteration_count=1,
        maximum_residual=0.0,
        resolution=grid.config.resolution,
    )


def make_fields(
    *,
    safety_value: float = 1.0,
    gradient: tuple[float, float, float] = (0.0, 0.0, 0.0),
    guidance: tuple[float, float] = (0.0, 0.0),
    per_yaw_guidance: tuple[tuple[float, float], ...] | None = None,
    yaw_slice_count: int = 4,
    unsafe_yaw_indices: tuple[int, ...] = (),
    per_yaw_safety_values: tuple[float, ...] | None = None,
) -> tuple[
    SafeSAGEConfigurationSpaceResult,
    SafeSAGEHFullResult,
    tuple[LaplaceGuidanceFieldResult, ...],
]:
    """Return aligned synthetic configuration, safety, and LGF fields."""

    yaw_angles = uniform_yaw_angles(yaw_slice_count)
    grids = tuple(
        make_grid(occupied=index in unsafe_yaw_indices)
        for index in range(yaw_slice_count)
    )
    shape = (yaw_slice_count,) + grids[0].shape
    grid_config = grids[0].config
    x_centers = (
        grid_config.minimum_x
        + (np.arange(grid_config.column_count, dtype=np.float64) + 0.5)
        * grid_config.resolution
    )
    y_centers = (
        grid_config.minimum_y
        + (np.arange(grid_config.row_count, dtype=np.float64) + 0.5)
        * grid_config.resolution
    )
    spatial_offsets = (
        gradient[0] * x_centers[np.newaxis, :] + gradient[1] * y_centers[:, np.newaxis]
    )
    values = np.empty(shape, dtype=np.float64)
    for yaw_index in range(yaw_slice_count):
        base_value = (
            safety_value
            if per_yaw_safety_values is None
            else per_yaw_safety_values[yaw_index]
        )
        values[yaw_index] = base_value + spatial_offsets

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
    if per_yaw_guidance is None:
        guidance_vectors = (guidance,) * yaw_slice_count
    else:
        if len(per_yaw_guidance) != yaw_slice_count:
            raise ValueError("per_yaw_guidance length mismatch")
        guidance_vectors = per_yaw_guidance
    guidance_fields = tuple(
        make_guidance_field(grids[index], guidance_vectors[index])
        for index in range(yaw_slice_count)
    )
    return configuration_space, h_full, guidance_fields


def apply_filter(
    safety_filter: SafeSAGECBFSafetyFilter,
    *,
    nominal_planar: tuple[float, float] = (0.2, 0.0),
    nominal_angular: float = 0.0,
    query_yaw: float = 0.0,
    temporal_derivative: float = 0.0,
    **field_kwargs: object,
):
    """Apply one filter to aligned synthetic fields at the origin."""

    configuration_space, h_full, guidance_fields = make_fields(**field_kwargs)
    return safety_filter.filter_command(
        np.asarray(nominal_planar, dtype=np.float64),
        nominal_angular,
        query_position_grid_xy=np.array([0.0, 0.0]),
        query_yaw_radians=query_yaw,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
        temporal_safety_derivative=temporal_derivative,
    )


def test_safe_nominal_command_passes_through() -> None:
    result = apply_filter(
        make_filter(),
        safety_value=1.0,
        guidance=(-1.0, 0.0),
    )

    assert result.safe_planar_velocity_grid_xy == pytest.approx((0.2, 0.0))
    assert result.safe_angular_velocity == pytest.approx(0.0)
    assert result.constraint.constraint_normal == pytest.approx((-1.0, 0.0, 0.0))
    assert result.constraint_feasible
    assert result.constraint_satisfied
    assert not result.intervention_active


def test_semantic_guidance_reduces_forward_velocity() -> None:
    result = apply_filter(
        make_filter(),
        nominal_planar=(0.3, 0.0),
        safety_value=0.2,
        guidance=(-2.0, 0.0),
    )

    assert result.safe_planar_velocity_grid_xy == pytest.approx((0.1, 0.0))
    assert result.cbf_margin == pytest.approx(0.0)
    assert result.intervention_active


def test_spatial_safety_gradient_does_not_replace_guidance() -> None:
    first = apply_filter(
        make_filter(),
        nominal_planar=(0.3, 0.0),
        safety_value=0.2,
        gradient=(50.0, -30.0, 0.0),
        guidance=(-2.0, 0.0),
    )
    second = apply_filter(
        make_filter(),
        nominal_planar=(0.3, 0.0),
        safety_value=0.2,
        gradient=(-50.0, 30.0, 0.0),
        guidance=(-2.0, 0.0),
    )

    assert first.constraint.constraint_normal == pytest.approx(
        second.constraint.constraint_normal
    )
    assert first.safe_planar_velocity_grid_xy == pytest.approx(
        second.safe_planar_velocity_grid_xy
    )


def test_opposite_tangential_guidance_produces_opposite_lateral_commands() -> None:
    pass_right = apply_filter(
        make_filter(),
        safety_value=0.0,
        guidance=(-1.0, -1.0),
    )
    pass_left = apply_filter(
        make_filter(),
        safety_value=0.0,
        guidance=(-1.0, 1.0),
    )

    assert pass_right.safe_planar_velocity_grid_xy == pytest.approx((0.1, -0.1))
    assert pass_left.safe_planar_velocity_grid_xy == pytest.approx((0.1, 0.1))


def test_yaw_derivative_directly_changes_angular_velocity() -> None:
    result = apply_filter(
        make_filter(),
        nominal_angular=0.6,
        safety_value=0.2,
        gradient=(0.0, 0.0, -1.0),
    )

    assert result.safe_angular_velocity == pytest.approx(0.2)
    assert result.intervention_active


def test_negative_temporal_derivative_tightens_constraint() -> None:
    stationary = apply_filter(
        make_filter(),
        nominal_planar=(0.3, 0.0),
        safety_value=0.2,
        gradient=(-2.0, 0.0, 0.0),
        guidance=(-2.0, 0.0),
    )
    approaching = apply_filter(
        make_filter(),
        nominal_planar=(0.3, 0.0),
        safety_value=0.2,
        gradient=(-2.0, 0.0, 0.0),
        guidance=(-2.0, 0.0),
        temporal_derivative=-0.1,
    )

    assert stationary.safe_planar_velocity_grid_xy[0] == pytest.approx(0.1)
    assert approaching.temporal_scaling == pytest.approx(1.0)
    assert approaching.safe_planar_velocity_grid_xy[0] == pytest.approx(0.05)


def test_spatial_temporal_field_is_sampled_at_query_configuration() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        safety_value=0.2,
        gradient=(-2.0, 0.0, 0.0),
        guidance=(-2.0, 0.0),
    )
    temporal_values = np.zeros_like(h_full.values)
    temporal_values[:, 2, 2] = -0.1
    temporal_field = SafeSAGETemporalDerivativeResult(
        values=temporal_values,
        raw_values=temporal_values,
        valid_mask=np.ones_like(temporal_values, dtype=bool),
        yaw_angles_radians=h_full.yaw_angles_radians,
        grid_config=configuration_space.occupancy_grids[0].config,
        time_delta_seconds=0.2,
        egomotion_translation_grid_xy=(0.0, 0.0),
        smoothing_factor=1.0,
    )

    result = make_filter().filter_command(
        np.array([0.3, 0.0]),
        0.0,
        query_position_grid_xy=np.array([0.0, 0.0]),
        query_yaw_radians=0.0,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
        temporal_safety_derivative=temporal_field,
    )

    assert result.temporal_safety_derivative == pytest.approx(-0.1)
    assert result.safe_planar_velocity_grid_xy[0] == pytest.approx(0.05)


def test_temporal_scaling_is_bounded() -> None:
    result = apply_filter(
        make_filter(),
        safety_value=1.0,
        gradient=(-0.1, 0.0, 0.0),
        guidance=(-10.0, 0.0),
        temporal_derivative=-0.1,
    )

    assert result.temporal_scaling == pytest.approx(1.0)


def test_periodic_yaw_interpolation_wraps_at_pi() -> None:
    result = apply_filter(
        make_filter(),
        query_yaw=0.75 * np.pi,
        per_yaw_safety_values=(1.0, 2.0, 3.0, 5.0),
        per_yaw_guidance=((1.0, 0.0),) * 4,
    )

    assert result.lower_yaw_slice == 3
    assert result.upper_yaw_slice == 0
    assert result.yaw_interpolation_weight == pytest.approx(0.5)
    assert result.sampled_safety_value == pytest.approx(3.0)


def test_interpolant_gradient_matches_sampled_value_finite_difference() -> None:
    query_yaw = -0.75 * np.pi
    epsilon = 1e-6
    field_arguments = {
        "gradient": (0.3, -0.2, 0.0),
        "per_yaw_safety_values": (1.0, 2.0, 3.0, 4.0),
    }
    result = apply_filter(
        make_filter(),
        query_yaw=query_yaw,
        **field_arguments,
    )
    positive = apply_filter(
        make_filter(),
        query_yaw=query_yaw + epsilon,
        **field_arguments,
    )
    negative = apply_filter(
        make_filter(),
        query_yaw=query_yaw - epsilon,
        **field_arguments,
    )
    finite_difference_yaw = (
        positive.sampled_safety_value - negative.sampled_safety_value
    ) / (2.0 * epsilon)

    assert result.interpolant_gradient[:2] == pytest.approx((0.3, -0.2))
    assert result.interpolant_gradient[2] == pytest.approx(finite_difference_yaw)
    assert result.interpolant_gradient[2] == pytest.approx(2.0 / np.pi)
    assert result.sampled_gradient[2] == pytest.approx(0.0)


def test_guidance_is_interpolated_across_yaw() -> None:
    result = apply_filter(
        make_filter(),
        query_yaw=0.75 * np.pi,
        per_yaw_guidance=(
            (2.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, -2.0),
        ),
    )

    assert result.sampled_guidance_grid_xy == pytest.approx((1.0, -1.0))


def test_unsafe_yaw_slice_triggers_emergency_stop() -> None:
    result = apply_filter(
        make_filter(),
        query_yaw=0.40 * np.pi,
        safety_value=0.2,
        guidance=(-1.0, 0.0),
        unsafe_yaw_indices=(3,),
    )

    assert result.query_on_unsafe_boundary
    assert result.safe_planar_velocity_grid_xy == (0.0, 0.0)
    assert result.safe_angular_velocity == 0.0
    assert result.emergency_stop_active
    assert result.emergency_reason == "unsafe_configuration_boundary"


def test_tiny_yaw_does_not_activate_distant_neighbor_boundary() -> None:
    result = apply_filter(
        make_filter(),
        query_yaw=0.01,
        safety_value=0.2,
        guidance=(-1.0, 0.0),
        unsafe_yaw_indices=(3,),
    )

    assert result.lower_yaw_slice == 2
    assert result.upper_yaw_slice == 3
    assert result.yaw_interpolation_weight < 0.5
    assert not result.query_on_unsafe_boundary


@pytest.mark.parametrize(
    ("query_yaw", "expected_unsafe"),
    [
        (0.40 * np.pi, True),
        (0.01, False),
    ],
)
def test_lightweight_configuration_query_matches_full_filter(
    query_yaw: float,
    expected_unsafe: bool,
) -> None:
    configuration_space, h_full, _ = make_fields(
        unsafe_yaw_indices=(3,),
    )

    is_unsafe = make_filter().is_unsafe_configuration(
        query_position_grid_xy=np.zeros(2, dtype=np.float64),
        query_yaw_radians=query_yaw,
        configuration_space=configuration_space,
        h_full=h_full,
    )

    assert is_unsafe is expected_unsafe


def test_infeasible_constraint_triggers_emergency_stop() -> None:
    result = apply_filter(
        make_filter(),
        safety_value=0.0,
        gradient=(0.1, 0.0, 0.0),
        guidance=(0.1, 0.0),
        temporal_derivative=-2.0,
    )

    assert not result.constraint_feasible
    assert result.emergency_stop_active
    assert result.emergency_reason == "infeasible_cbf_constraint"


def test_actuator_limits_are_applied() -> None:
    result = apply_filter(
        make_filter(),
        nominal_planar=(0.8, -0.7),
        nominal_angular=-2.0,
        safety_value=10.0,
        guidance=(0.0, 0.0),
    )

    assert result.safe_planar_velocity_grid_xy == pytest.approx((0.5, -0.5))
    assert result.safe_angular_velocity == pytest.approx(-1.0)


def test_angular_weight_changes_three_command_projection() -> None:
    equal_weight = apply_filter(
        make_filter(),
        safety_value=0.0,
        gradient=(0.0, 0.0, 1.0),
        guidance=(-1.0, 0.0),
    )
    angular_expensive = apply_filter(
        make_filter(angular_velocity_weight=4.0),
        safety_value=0.0,
        gradient=(0.0, 0.0, 1.0),
        guidance=(-1.0, 0.0),
    )

    assert equal_weight.safe_planar_velocity_grid_xy[0] == pytest.approx(0.1)
    assert equal_weight.safe_angular_velocity == pytest.approx(0.1)
    assert angular_expensive.safe_planar_velocity_grid_xy[0] == pytest.approx(0.04)
    assert angular_expensive.safe_angular_velocity == pytest.approx(0.04)


def test_generic_projection_supports_two_and_three_commands() -> None:
    for dimension in (2, 3):
        nominal = np.zeros(dimension, dtype=np.float64)
        normal = np.ones(dimension, dtype=np.float64)
        projected, feasible = project_weighted_box_halfspace(
            nominal,
            lower_bounds=np.full(dimension, -1.0),
            upper_bounds=np.full(dimension, 1.0),
            weights=np.ones(dimension),
            constraint_normal=normal,
            constraint_rhs=1.0,
            numerical_tolerance=1e-9,
        )

        assert feasible
        np.testing.assert_allclose(projected, np.full(dimension, 1 / dimension))


def test_query_outside_cell_center_domain_is_rejected() -> None:
    configuration_space, h_full, guidance_fields = make_fields()

    with pytest.raises(ValueError, match="cell-center domain"):
        make_filter().filter_command(
            np.array([0.2, 0.0]),
            0.0,
            query_position_grid_xy=np.array([2.4, 0.0]),
            query_yaw_radians=0.0,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
        )


def test_guidance_length_mismatch_is_rejected() -> None:
    configuration_space, h_full, guidance_fields = make_fields()

    with pytest.raises(ValueError, match="guidance_fields length"):
        make_filter().filter_command(
            np.array([0.2, 0.0]),
            0.0,
            query_position_grid_xy=np.array([0.0, 0.0]),
            query_yaw_radians=0.0,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields[:-1],
        )


def test_malformed_guidance_shape_is_rejected() -> None:
    configuration_space, h_full, guidance_fields = make_fields()
    malformed = replace(
        guidance_fields[0],
        vectors_robot_xy=np.zeros((4, 4, 2), dtype=np.float64),
    )

    with pytest.raises(ValueError, match="Guidance-field shape"):
        make_filter().filter_command(
            np.array([0.2, 0.0]),
            0.0,
            query_position_grid_xy=np.array([0.0, 0.0]),
            query_yaw_radians=0.0,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=(malformed,) + guidance_fields[1:],
        )


@pytest.mark.parametrize(
    "override",
    (
        {"gamma": 0.0},
        {"minimum_planar_velocity": 1.0},
        {"planar_velocity_weight": 0.0},
        {"temporal_scaling_epsilon": 0.0},
        {"temporal_scaling_kappa": 0.0},
    ),
)
def test_invalid_configuration_is_rejected(override: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SafeSAGECBFConfig(**override)


@pytest.mark.parametrize(
    ("planar", "angular"),
    (
        ((np.nan, 0.0), 0.0),
        ((0.0, np.inf), 0.0),
        ((0.0, 0.0), np.nan),
    ),
)
def test_nonfinite_filter_inputs_are_rejected(
    planar: tuple[float, float],
    angular: float,
) -> None:
    configuration_space, h_full, guidance_fields = make_fields()

    with pytest.raises(ValueError):
        make_filter().filter_command(
            np.asarray(planar),
            angular,
            query_position_grid_xy=np.array([0.0, 0.0]),
            query_yaw_radians=0.0,
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
        )


def test_batch_matches_scalar_sampling_at_yaw_and_cell_boundaries(monkeypatch):
    from robot.safety import safe_sage_cbf_filter as module

    configuration_space, h_full, guidance = make_fields(
        gradient=(0.3, -0.2, 0.1),
        per_yaw_safety_values=(0.5, 1.0, 1.5, 2.0),
        per_yaw_guidance=((1, 0), (0, 1), (-1, 0), (0, -1)),
        unsafe_yaw_indices=(2,),
    )
    states = np.array(
        [
            [x, y, yaw]
            for x, y in ((0, 0), (0.5, -0.5), (-2.0, 2.0))
            for yaw in (-np.pi, -np.pi / 4, 0, np.pi / 4, np.pi, 7 * np.pi)
        ]
    )
    filter_ = make_filter()
    kwargs = dict(
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance,
        temporal_safety_derivative=-0.03,
    )
    scalar = tuple(
        filter_.evaluate_constraint(
            query_position_grid_xy=state[:2],
            query_yaw_radians=float(state[2]),
            **kwargs,
        )
        for state in states
    )
    validations = []
    original = module._validate_field_alignment

    def validate(*args):
        validations.append(1)
        original(*args)

    monkeypatch.setattr(module, "_validate_field_alignment", validate)
    assert filter_.evaluate_constraints(states, **kwargs) == scalar
    assert len(validations) == 1
    assert filter_.evaluate_constraints(states, **kwargs) == scalar
    assert len(validations) == 2  # no stale validation cache


@pytest.mark.parametrize("states", [[], [[0, 0]], [[0, 0, np.nan]], [[3, 0, 0]]])
def test_batch_rejects_invalid_or_out_of_grid_queries(states):
    configuration_space, h_full, guidance = make_fields()
    with pytest.raises(ValueError):
        make_filter().evaluate_constraints(
            np.asarray(states),
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance,
        )


def test_batch_rejects_malformed_replacement_field():
    configuration_space, h_full, guidance = make_fields()
    malformed = replace(guidance[0], vectors_robot_xy=np.zeros((4, 4, 2)))
    with pytest.raises(ValueError, match="Guidance-field shape"):
        make_filter().evaluate_constraints(
            np.zeros((2, 3)),
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=(malformed,) + guidance[1:],
        )


@pytest.mark.parametrize("spatial_temporal", [False, True])
def test_batch_random_fields_match_independent_scalar_reference(spatial_temporal):
    rng = np.random.default_rng(854)
    cs, h, fields = make_fields(yaw_slice_count=8)
    masks = rng.random(h.shape) > 0.7
    h = replace(
        h, values=rng.normal(size=h.shape),
        yaw_derivatives=rng.normal(size=h.shape),
        dirichlet_boundary_masks=masks,
    )
    fields = tuple(replace(f, vectors_robot_xy=rng.normal(size=f.vectors_robot_xy.shape)) for f in fields)
    temporal_values = rng.normal(size=h.shape)
    temporal = SafeSAGETemporalDerivativeResult(
        values=temporal_values, raw_values=temporal_values,
        valid_mask=np.ones(h.shape, dtype=bool),
        yaw_angles_radians=h.yaw_angles_radians, grid_config=cs.occupancy_grids[0].config,
        time_delta_seconds=0.2, egomotion_translation_grid_xy=(0.0, 0.0), smoothing_factor=1.0,
    ) if spatial_temporal else -0.37
    filter_ = make_filter()
    tol = filter_.config.numerical_tolerance
    spacing = 2 * np.pi / h.yaw_slice_count
    states = np.vstack((
        np.column_stack((rng.uniform(-2, 2, (500, 2)), rng.uniform(-20 * np.pi, 20 * np.pi, 500))),
        [[x, y, -np.pi + spacing * fraction] for x, y in ((-2, -2), (2, 2), (2, -2), (0.5, -0.5))
         for fraction in (0, tol / 2, tol * 2, 0.5 - 2 * tol, 0.5, 0.5 + 2 * tol, 1, 8)],
    ))
    kwargs = dict(configuration_space=cs, h_full=h, guidance_fields=fields, temporal_safety_derivative=temporal)
    expected = tuple(filter_.evaluate_constraint(query_position_grid_xy=s[:2], query_yaw_radians=float(s[2]), **kwargs) for s in states)
    actual = filter_.evaluate_constraints(states, **kwargs)
    for batch, scalar in zip(actual, expected, strict=True):
        for key, value in asdict(scalar).items():
            if key in ("query_cell", "lower_yaw_slice", "upper_yaw_slice", "unsafe_reason", "query_on_unsafe_boundary"):
                assert getattr(batch, key) == value
            else:
                assert getattr(batch, key) == pytest.approx(value, abs=2e-13, rel=2e-13), key


@pytest.mark.parametrize("guidance", [(0.0, 0.0), (1.0, 0.0)])
def test_batch_zero_temporal_denominator_and_unsafe_priority(guidance):
    cs, h, fields = make_fields(safety_value=0.0, guidance=guidance)
    kwargs = dict(configuration_space=cs, h_full=h, guidance_fields=fields, temporal_safety_derivative=-0.4)
    filter_ = make_filter()
    assert filter_.evaluate_constraints(np.zeros((3, 3)), **kwargs) == tuple(
        filter_.evaluate_constraint(query_position_grid_xy=np.zeros(2), query_yaw_radians=0.0, **kwargs) for _ in range(3)
    )


@pytest.mark.parametrize("temporal", [True, np.nan, "0"])
def test_batch_rejects_invalid_temporal_input(temporal):
    cs, h, fields = make_fields()
    with pytest.raises((TypeError, ValueError)):
        make_filter().evaluate_constraints(np.zeros((2, 3)), configuration_space=cs, h_full=h, guidance_fields=fields, temporal_safety_derivative=temporal)


def test_vector_geometry_mask_matches_scalar_including_yaw_ties():
    configuration_space, h_full, _ = make_fields(unsafe_yaw_indices=(0, 2))
    generator = np.random.default_rng(42)
    states = np.column_stack((
        generator.uniform(-2.49, 2.49, (200, 2)),
        generator.uniform(-8 * np.pi, 8 * np.pi, 200),
    ))
    ties = np.array([[0, 0, yaw] for yaw in (
        -np.pi, -3 * np.pi / 4, -np.pi / 4, np.pi / 4, 3 * np.pi / 4,
    )])
    states = np.vstack((states, ties))
    filter_ = make_filter()
    kwargs = dict(configuration_space=configuration_space, h_full=h_full)
    expected = [filter_.is_unsafe_configuration(
        query_position_grid_xy=state[:2], query_yaw_radians=float(state[2]), **kwargs,
    ) for state in states]
    np.testing.assert_array_equal(filter_.unsafe_configuration_mask(states, **kwargs), expected)
    with pytest.raises(ValueError, match="outside"):
        filter_.unsafe_configuration_mask(np.array([[2.5, 0, 0]]), **kwargs)
