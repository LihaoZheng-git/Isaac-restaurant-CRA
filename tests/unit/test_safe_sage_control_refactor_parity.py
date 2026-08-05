"""Temporary migration guard for the Safe-SAGE control refactor.

This test intentionally depends on both the legacy differential-drive adapter
and the replacement unicycle filter.  Remove it together with the legacy
adapter only after the Isaac stationary and motion demos pass.
"""

import math

import numpy as np
import pytest
from robot.safety.safe_sage_differential_drive_interface import (
    SafeSAGEDifferentialDriveConfig,
    SafeSAGEDifferentialDriveSafetyFilter,
)
from robot.safety.safe_sage_unicycle_cbf_filter import (
    SafeSAGEUnicycleCBFConfig,
    SafeSAGEUnicycleCBFFilter,
)

from test_safe_sage_cbf_filter import make_fields, make_filter

LOOKAHEAD_DISTANCE = 0.10


def make_filter_pair() -> tuple[
    SafeSAGEDifferentialDriveSafetyFilter,
    SafeSAGEUnicycleCBFFilter,
]:
    """Return legacy and replacement filters with identical settings."""

    legacy = SafeSAGEDifferentialDriveSafetyFilter(
        make_filter(),
        SafeSAGEDifferentialDriveConfig(
            lookahead_distance=LOOKAHEAD_DISTANCE,
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=0.5,
            minimum_angular_velocity=-1.0,
            maximum_angular_velocity=1.0,
            linear_velocity_weight=1.3,
            angular_velocity_weight=0.8,
        ),
    )
    replacement = SafeSAGEUnicycleCBFFilter(
        make_filter(),
        SafeSAGEUnicycleCBFConfig(
            lookahead_distance=LOOKAHEAD_DISTANCE,
            minimum_linear_velocity=0.0,
            maximum_linear_velocity=0.5,
            minimum_angular_velocity=-1.0,
            maximum_angular_velocity=1.0,
            linear_velocity_weight=1.3,
            angular_velocity_weight=0.8,
        ),
    )
    return legacy, replacement


def assert_results_match(legacy, replacement) -> None:
    """Require numerical and status parity between both implementations."""

    float_attributes = (
        "nominal_linear_velocity",
        "nominal_angular_velocity",
        "safe_linear_velocity",
        "safe_angular_velocity",
        "linear_velocity_coefficient",
        "angular_velocity_coefficient",
        "nominal_control_left_hand_side",
        "safe_control_left_hand_side",
        "cbf_margin",
    )
    for attribute in float_attributes:
        assert getattr(replacement, attribute) == pytest.approx(
            getattr(legacy, attribute),
            abs=1e-12,
        )

    np.testing.assert_allclose(
        replacement.control_mapping,
        legacy.control_mapping,
        atol=1e-12,
        rtol=0.0,
    )
    assert replacement.constraint == legacy.constraint
    assert replacement.constraint_feasible == legacy.constraint_feasible
    assert replacement.constraint_satisfied == legacy.constraint_satisfied
    assert replacement.intervention_active == legacy.intervention_active
    assert (
        replacement.emergency_stop_active
        == legacy.emergency_stop_active
    )
    assert replacement.emergency_reason == legacy.emergency_reason


@pytest.mark.parametrize(
    "query_yaw",
    (-math.pi + 0.01, -0.5 * math.pi, 0.0, 0.73, math.pi - 0.01),
)
@pytest.mark.parametrize(
    "guidance",
    ((-2.0, 0.0), (-1.0, -1.0), (-1.0, 1.0), (0.4, -0.7)),
)
@pytest.mark.parametrize("yaw_derivative", (-0.7, 0.0, 0.7))
@pytest.mark.parametrize("safety_value", (0.0, 0.4))
@pytest.mark.parametrize("temporal_derivative", (-0.15, 0.0, 0.2))
def test_replacement_matches_legacy_across_control_space(
    query_yaw: float,
    guidance: tuple[float, float],
    yaw_derivative: float,
    safety_value: float,
    temporal_derivative: float,
) -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=guidance,
        safety_value=safety_value,
        gradient=(0.0, 0.0, yaw_derivative),
    )
    legacy_filter, replacement_filter = make_filter_pair()
    shared_arguments = {
        "query_position_grid_xy": np.array([0.0, 0.0]),
        "query_yaw_radians": query_yaw,
        "configuration_space": configuration_space,
        "h_full": h_full,
        "guidance_fields": guidance_fields,
        "temporal_safety_derivative": temporal_derivative,
    }
    legacy = legacy_filter.filter_command(
        0.2,
        0.13,
        **shared_arguments,
    )
    replacement = replacement_filter.filter_command(
        0.2,
        0.13,
        **shared_arguments,
    )
    assert_results_match(legacy, replacement)


def test_replacement_matches_legacy_fail_closed_result() -> None:
    configuration_space, h_full, guidance_fields = make_fields(
        guidance=(-1.0, 1.0),
        unsafe_yaw_indices=(3,),
    )
    legacy_filter, replacement_filter = make_filter_pair()
    shared_arguments = {
        "query_position_grid_xy": np.array([0.0, 0.0]),
        "query_yaw_radians": 0.40 * math.pi,
        "configuration_space": configuration_space,
        "h_full": h_full,
        "guidance_fields": guidance_fields,
    }
    legacy = legacy_filter.filter_command(
        0.2,
        -0.1,
        **shared_arguments,
    )
    replacement = replacement_filter.filter_command(
        0.2,
        -0.1,
        **shared_arguments,
    )
    assert_results_match(legacy, replacement)
