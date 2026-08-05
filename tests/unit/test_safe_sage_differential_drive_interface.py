import math

import numpy as np
import pytest
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
)
from robot.safety.safe_sage_differential_drive_interface import (
    SafeSAGEDifferentialDriveConfig,
    SafeSAGEDifferentialDriveSafetyFilter,
    differential_drive_control_mapping,
    lookahead_position_grid_xy,
    shift_rectangular_footprint_to_lookahead,
)

from test_safe_sage_cbf_filter import make_fields, make_filter


def make_drive_filter(
    **overrides: float,
) -> SafeSAGEDifferentialDriveSafetyFilter:
    """Return one deterministic Nova Carter interface."""

    values = {
        "lookahead_distance": 0.10,
        "minimum_linear_velocity": 0.0,
        "maximum_linear_velocity": 0.5,
        "minimum_angular_velocity": -1.0,
        "maximum_angular_velocity": 1.0,
        "linear_velocity_weight": 1.0,
        "angular_velocity_weight": 1.0,
    }
    values.update(overrides)
    return SafeSAGEDifferentialDriveSafetyFilter(
        make_filter(),
        SafeSAGEDifferentialDriveConfig(**values),
    )


def apply_drive_filter(
    *,
    guidance: tuple[float, float],
    safety_value: float = 0.0,
    gradient: tuple[float, float, float] = (0.0, 0.0, 0.0),
    query_yaw: float = 0.0,
    nominal_linear: float = 0.2,
    nominal_angular: float = 0.0,
    unsafe_yaw_indices: tuple[int, ...] = (),
):
    """Evaluate one synthetic differential-drive scenario."""

    configuration_space, h_full, guidance_fields = make_fields(
        guidance=guidance,
        safety_value=safety_value,
        gradient=gradient,
        unsafe_yaw_indices=unsafe_yaw_indices,
    )
    return make_drive_filter().filter_command(
        nominal_linear,
        nominal_angular,
        query_position_grid_xy=np.array([0.0, 0.0]),
        query_yaw_radians=query_yaw,
        configuration_space=configuration_space,
        h_full=h_full,
        guidance_fields=guidance_fields,
    )


def test_control_mapping_at_zero_yaw() -> None:
    mapping = differential_drive_control_mapping(
        0.0,
        lookahead_distance=0.1,
    )

    np.testing.assert_allclose(
        mapping,
        np.array(
            [
                [1.0, 0.0],
                [0.0, 0.1],
                [0.0, 1.0],
            ]
        ),
        atol=1e-12,
    )


def test_control_mapping_at_quarter_turn() -> None:
    mapping = differential_drive_control_mapping(
        0.5 * math.pi,
        lookahead_distance=0.2,
    )

    np.testing.assert_allclose(
        mapping,
        np.array(
            [
                [0.0, -0.2],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        atol=1e-12,
    )


def test_lookahead_position_uses_current_heading() -> None:
    position = lookahead_position_grid_xy(
        np.array([2.0, 3.0]),
        yaw_radians=0.5 * math.pi,
        lookahead_distance=0.1,
    )

    np.testing.assert_allclose(position, [2.0, 3.1], atol=1e-12)


def test_shifted_footprint_preserves_physical_rectangle() -> None:
    original = RectangularRobotFootprint(
        front_extent=0.14,
        rear_extent=0.587351,
        left_extent=0.2466,
        right_extent=0.2466,
        safety_margin=0.05,
    )
    shifted = shift_rectangular_footprint_to_lookahead(
        original,
        lookahead_distance=0.10,
    )

    assert shifted.front_extent == pytest.approx(0.04)
    assert shifted.rear_extent == pytest.approx(0.687351)
    assert shifted.length == pytest.approx(original.length)
    assert shifted.width == pytest.approx(original.width)
    assert shifted.safety_margin == original.safety_margin


def test_paper_normal_is_pushed_through_control_mapping() -> None:
    result = apply_drive_filter(
        guidance=(-2.0, -3.0),
        gradient=(0.0, 0.0, 0.5),
        safety_value=10.0,
    )

    assert result.linear_velocity_coefficient == pytest.approx(-2.0)
    assert result.angular_velocity_coefficient == pytest.approx(0.2)


def test_opposite_passing_sides_produce_opposite_steering() -> None:
    pass_right = apply_drive_filter(guidance=(-1.0, -1.0))
    pass_left = apply_drive_filter(guidance=(-1.0, 1.0))

    assert pass_right.safe_angular_velocity < 0.0
    assert pass_left.safe_angular_velocity > 0.0
    assert pass_right.safe_angular_velocity == pytest.approx(
        -pass_left.safe_angular_velocity
    )


def test_zero_tangential_component_only_reduces_forward_speed() -> None:
    result = apply_drive_filter(guidance=(-1.0, 0.0))

    assert result.safe_linear_velocity == pytest.approx(0.0)
    assert result.safe_angular_velocity == pytest.approx(0.0)


def test_yaw_derivative_remains_in_angular_coefficient() -> None:
    result = apply_drive_filter(
        guidance=(-1.0, 0.0),
        gradient=(0.0, 0.0, 0.7),
        safety_value=10.0,
    )

    assert result.angular_velocity_coefficient == pytest.approx(0.7)


def test_unsafe_configuration_fails_closed() -> None:
    result = apply_drive_filter(
        guidance=(-1.0, -1.0),
        query_yaw=0.40 * math.pi,
        unsafe_yaw_indices=(3,),
    )

    assert result.safe_linear_velocity == 0.0
    assert result.safe_angular_velocity == 0.0
    assert result.emergency_stop_active
    assert result.emergency_reason == "unsafe_configuration_boundary"


@pytest.mark.parametrize(
    "override",
    (
        {"lookahead_distance": 0.0},
        {"minimum_linear_velocity": 1.0},
        {"linear_velocity_weight": 0.0},
        {"numerical_tolerance": 0.0},
    ),
)
def test_invalid_interface_configuration_is_rejected(
    override: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        SafeSAGEDifferentialDriveConfig(**override)


def test_lookahead_cannot_exceed_supported_front_extent() -> None:
    footprint = RectangularRobotFootprint.centered(
        length=0.2,
        width=0.2,
    )

    with pytest.raises(ValueError, match="cannot exceed front_extent"):
        shift_rectangular_footprint_to_lookahead(
            footprint,
            lookahead_distance=0.11,
        )
