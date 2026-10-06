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
