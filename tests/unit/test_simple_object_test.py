"""Evidence boundaries and fail-closed execution for the bounded object test."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot.evaluation.simple_object_test import (
    ControlledObjectInputs,
    controlled_object_adapter,
    disk_footprint_clearance,
    run_bounded_pass_by,
)
from robot.perception.physical_attributes import EstimateStatus, PhysicalAttributes
from robot.risk.physical_factor_adapter import adapt_physical_factors
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint


def policy(inputs):
    adapted = controlled_object_adapter(inputs, track_id=7, timestamp_seconds=1.0)
    return adapted, CRASafeSAGEPolicy().evaluate(adapted.profile)


def test_low_oracle_profile_keeps_relative_approach_energy_and_provenance():
    adapted, result = policy(ControlledObjectInputs())
    assert adapted.potential.energy_j.value == 0.0
    assert adapted.kinetic.energy_j.value == pytest.approx(0.005)
    assert result.assessment.phi == pytest.approx(1 - (1 - 0.00005) * 0.98 * 0.99)
    assert 0 < result.assessment.phi < 0.05
    for factor in (
        adapted.potential,
        adapted.kinetic,
        adapted.contact,
        adapted.thermal,
    ):
        assert all(item.status is EstimateStatus.ORACLE for _, item in factor.inputs)
    assert adapted.contact.basis == "visual_exposed_edge_proxy"
    assert result.tangential_strength == 1.25


@pytest.mark.parametrize(
    "changes",
    [
        {"fall_height_m": 0.8},
        {"relative_speed_mps": 2.0},
        {"contact_proxy": 0.8},
        {"thermal_cue": 0.8},
    ],
)
def test_increasing_each_factor_changes_flux_without_changing_lambda(changes):
    inputs = ControlledObjectInputs()
    _, low = policy(inputs)
    _, high = policy(replace(inputs, **changes))
    assert high.assessment.phi > low.assessment.phi
    assert (
        high.guidance_parameters.boundary_flux_magnitude
        > low.guidance_parameters.boundary_flux_magnitude
    )
    assert low.tangential_strength == high.tangential_strength == 1.25


def test_unknown_sphere_inputs_are_not_replaced_by_the_controlled_profile():
    sphere = PhysicalAttributes.unknown(
        track_id=7, timestamp_seconds=1.0, semantic_class="sphere"
    )
    adapted = adapt_physical_factors(sphere, timestamp_seconds=1.0)
    result = CRASafeSAGEPolicy().evaluate(adapted.profile)
    assert result.assessment.phi == 1.0
    assert all(
        factor.rejection_reasons
        for factor in (
            adapted.potential,
            adapted.kinetic,
            adapted.contact,
            adapted.thermal,
        )
    )
    assert sphere.contact.sharpness.status is EstimateStatus.UNAVAILABLE
    assert sphere.thermal.heat_source_likelihood.status is EstimateStatus.UNAVAILABLE


@pytest.mark.parametrize("initial_state,padding,warm_turn,flux", [
    ([.8904124, -.60043317, .35157854], {}, 0.0, .535818212),
    ([1.58971232, -.62837141, .02882322], {
        "configuration_position_padding_m": .02,
        "configuration_yaw_padding_radians": .08,
    }, 0.0, .535818212),
    ([.87604189, -.63662153, .29204763], {
        "configuration_position_padding_m": .02,
        "configuration_yaw_padding_radians": .08,
    }, .2, .98),
])
def test_online_turn_seed_escapes_stopped_horizon_without_weakening_safety(
    initial_state, padding, warm_turn, flux,
):
    """Synthetic front projection reproduces the wheel-run local minima."""
    from robot.navigation.waypoint_controller import (
        RobotPose2D,
        Waypoint,
        WaypointController,
        WaypointControllerConfig,
    )
    from robot.safety.class_safety_policy import (
        ClassSafetyParameters,
        ClassSafetyPolicy,
    )
    from robot.safety.local_occupancy_grid import (
        LocalOccupancyGridConfig,
        build_local_occupancy_grid,
    )
    from robot.safety.poisson_safety_function import PoissonSafetyFunctionConfig
    from robot.safety.safe_sage_configuration_space import (
        SafeSAGEConfigurationSpaceConfig,
    )
    from robot.safety.safe_sage_field_pipeline import (
        SafeSAGEFieldPipelineConfig,
        build_safe_sage_field_pipeline,
    )
    from robot.safety.safe_sage_guidance_policy import (
        SafeSAGEGuidanceParameters,
        SafeSAGEGuidancePolicy,
    )
    from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
    from robot.safety.safe_sage_unicycle_cbf_filter import (
        shift_rectangular_footprint_to_lookahead,
    )
    from robot.safety.safe_sage_unicycle_mpc import (
        SafeSAGEUnicycleMPC,
        SafeSAGEUnicycleMPCConfig,
    )
    from robot.safety.safe_sage_cbf_filter import SafeSAGECBFSafetyFilter
    from robot.safety.tangential_laplace_guidance_field import (
        TangentialLaplaceGuidanceFieldConfig,
    )

    x, y = np.meshgrid(np.linspace(1.065, 1.28, 40), np.linspace(-.235, .235, 40))
    visible = np.square(x - 1.3) + np.square(y) < .25 ** 2
    points = np.column_stack((x[visible], y[visible]))
    risk = .4 if flux == .98 else .02984851
    grid = build_local_occupancy_grid(
        points,
        class_safety_policy=ClassSafetyPolicy(
            {
                "unknown": ClassSafetyParameters(risk, flux),
            }
        ),
        config=LocalOccupancyGridConfig(
            resolution=0.1,
            minimum_x=-2,
            maximum_x=5,
            minimum_y=-3,
            maximum_y=3,
        ),
    )
    footprint = shift_rectangular_footprint_to_lookahead(
        RectangularRobotFootprint(0.14, 0.587351, 0.2466, 0.2466, 0.05),
        lookahead_distance=0.1,
    )
    field = build_safe_sage_field_pipeline(
        grid,
        guidance_policy=SafeSAGEGuidancePolicy(
            {
                    "unknown": SafeSAGEGuidanceParameters(
                        ClassSafetyParameters(risk, flux), 1.25, "right"
                    ),
            }
        ),
        config=SafeSAGEFieldPipelineConfig(
            SafeSAGEConfigurationSpaceConfig(footprint, 16),
            TangentialLaplaceGuidanceFieldConfig(
                convergence_tolerance=1e-6,
                tangential_buffer_radius=0.6,
            ),
            SafeSAGEHFullConfig(
                yaw_slice_count=16,
                softplus_beta=1.0,
                minimum_forcing_magnitude=1e-9,
                poisson_config=PoissonSafetyFunctionConfig(convergence_tolerance=1e-6),
            ),
        ),
    )
    state = np.array(initial_state)
    waypoint = WaypointController(
        [Waypoint(2.5, 0)],
        config=WaypointControllerConfig(
            maximum_linear_velocity=0.2,
            maximum_angular_velocity=0.6,
            linear_gain=0.8,
            angular_gain=1.5,
            position_tolerance=0.1,
            minimum_alignment_scale=0.25,
            rotate_in_place_threshold_radians=np.deg2rad(100),
        ),
    )
    nominal = np.array(
        [
            [q.linear_velocity, q.angular_velocity]
            for q in waypoint.predict_commands(
                RobotPose2D(*state),
                horizon_steps=30,
                time_step=0.2,
            )
        ]
    )
    config = SafeSAGEUnicycleMPCConfig(
        horizon_steps=30,
        time_step=0.2,
        safety_substeps_per_interval=12,
        integration_substeps_per_interval=12,
        angular_velocity_weight=0.02,
        command_smoothness_weight=0.005,
        maximum_sqp_iterations=12,
        **padding,
    )
    warm = np.zeros((30, 2))
    warm[:5, 1] = warm_turn
    kwargs = dict(
        configuration_space=field.configuration_space,
        h_full=field.h_full,
        guidance_fields=field.guidance_fields,
        initial_command_guess=warm,
    )
    original = SafeSAGEUnicycleMPC(SafeSAGECBFSafetyFilter(), config).plan(
        state,
        nominal,
        **kwargs,
    )
    improved = SafeSAGEUnicycleMPC(
        SafeSAGECBFSafetyFilter(),
        replace(config, enable_progress_seeds=True),
    ).plan(state, nominal, **kwargs)
    assert original.success and improved.success
    assert original.linear_velocity < 0.01
    if warm_turn:
        assert original.angular_velocity > .05
    assert improved.status == "exact_safe_progress_seed_selected"
    assert improved.angular_velocity <= -0.09
    assert improved.predicted_chassis_states[-1, 0] > state[0] + 0.4
    assert improved.objective_value < original.objective_value
    assert improved.minimum_dtcbf_margin >= -config.constraint_tolerance
    assert improved.minimum_predicted_safety_value >= 0
    filter_ = SafeSAGECBFSafetyFilter()
    for safety_state in improved.predicted_safety_states:
        assert not filter_.is_unsafe_configuration(
            query_position_grid_xy=safety_state[:2],
            query_yaw_radians=float(safety_state[2]),
            configuration_space=field.configuration_space,
            h_full=field.h_full,
        )


def test_clearance_monitor_handles_rotation_and_collision_independently():
    footprint = RectangularRobotFootprint(0.2, 0.6, 0.25, 0.25, 0.05)
    assert disk_footprint_clearance(
        [0, 0, 0], center_xy=[1, 0], radius_m=0.25, footprint=footprint
    ) == pytest.approx(0.5)
    assert disk_footprint_clearance(
        [0, 0, np.pi], center_xy=[1, 0], radius_m=0.25, footprint=footprint
    ) == pytest.approx(0.1)
    assert (
        disk_footprint_clearance(
            [0.6, 0, 0], center_xy=[1, 0], radius_m=0.25, footprint=footprint
        )
        < 0
    )


def runner(*, plan_success=True, backup_valid=True, initial_clearance=1.0):
    counters = {"commands": 0, "stops": 0, "backup_calls": 0}
    config = SimpleNamespace(time_step=0.2, horizon_steps=2, constraint_tolerance=1e-6)
    planned = SimpleNamespace(
        success=plan_success,
        fallback_active=False,
        command_sequence=np.zeros((2, 2)),
        predicted_chassis_states=np.zeros((3, 3)),
        predicted_safety_states=np.zeros((3, 3)),
        predicted_safety_values=np.ones(3),
        dtcbf_margins=np.ones(2),
        intervention_active=True,
        linear_velocity=0.1,
        angular_velocity=0.0,
    )
    mpc = SimpleNamespace(config=config, plan=lambda *a, **k: planned)

    def filter_command(*args, **kwargs):
        counters["backup_calls"] += 1
        return SimpleNamespace(
            constraint_feasible=backup_valid,
            constraint_satisfied=backup_valid,
            emergency_stop_active=not backup_valid,
            cbf_margin=1.0,
            intervention_active=False,
            safe_linear_velocity=0.1,
            safe_angular_velocity=0.0,
            emergency_reason="test_infeasible" if not backup_valid else None,
            constraint=SimpleNamespace(unsafe_reason=None),
            query_on_unsafe_boundary=False,
            sampled_safety_value=1.0,
        )

    backup = SimpleNamespace(
        config=SimpleNamespace(lookahead_distance=0.1), filter_command=filter_command
    )

    def advance(*args):
        counters["commands"] += 1

    def stop():
        counters["stops"] += 1

    result = run_bounded_pass_by(
        field=SimpleNamespace(
            configuration_space=None, h_full=None, guidance_fields=()
        ),
        mpc=mpc,
        backup=backup,
        read_state=lambda: np.zeros(3),
        advance=advance,
        stop=stop,
        clearance=lambda state: initial_clearance,
        physics_dt=0.05,
        maximum_seconds=0.2,
    )
    return result, counters


@pytest.mark.parametrize(
    "inputs,termination",
    [
        ({"plan_success": False}, "mpc_fail_closed"),
        ({"backup_valid": False}, "cbf_fail_closed"),
        ({"initial_clearance": 0.01}, "clearance_guard"),
    ],
)
def test_failure_guards_issue_no_motion_and_stop(inputs, termination):
    result, counters = runner(**inputs)
    assert not result["passed"]
    assert result["termination"] == termination
    assert counters["commands"] == 0
    assert counters["stops"] == 1


def test_analytical_filter_is_rechecked_before_every_physics_command():
    result, counters = runner()
    assert counters["commands"] == counters["backup_calls"] == 4
    assert counters["stops"] == 1
    assert not result["passed"]  # Command validity alone is not goal completion.
    assert result["termination"] == "duration_limit"
