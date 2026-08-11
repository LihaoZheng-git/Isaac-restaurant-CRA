import math

import numpy as np
import pytest
from robot.control.planar_path_reference import (
    PlanarPathReferenceConfig,
    build_planar_path_reference,
)
from robot.control.unicycle_model import integrate_unicycle_euler


def test_straight_planar_path_produces_straight_unicycle_seed() -> None:
    result = build_planar_path_reference(
        [
            [0.00, 0.00, 0.70],
            [0.04, 0.00, 0.70],
            [0.08, 0.00, 0.70],
        ],
        initial_chassis_state=[0.0, 0.0, 0.0],
    )

    assert result.target_states[:, 2] == pytest.approx([0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        result.seed_commands,
        [[0.20, 0.0], [0.20, 0.0]],
    )
    np.testing.assert_allclose(
        result.seed_chassis_states,
        [
            [0.00, 0.00, 0.0],
            [0.04, 0.00, 0.0],
            [0.08, 0.00, 0.0],
        ],
    )


def test_right_side_path_requests_clockwise_turn() -> None:
    result = build_planar_path_reference(
        [
            [0.00, 0.00, 0.0],
            [0.04, -0.02, 0.0],
            [0.08, -0.05, 0.0],
        ],
        initial_chassis_state=[0.0, 0.0, 0.0],
    )

    assert result.target_states[1, 2] < 0.0
    assert result.seed_commands[0, 1] < 0.0
    assert np.all(result.seed_commands[:, 0] >= 0.0)
    assert np.all(result.seed_commands[:, 0] <= 0.25)
    assert np.all(np.abs(result.seed_commands[:, 1]) <= 0.8)


def test_planar_yaw_is_replaced_by_path_tangent() -> None:
    result = build_planar_path_reference(
        [
            [0.0, 0.0, 1.2],
            [0.0, 0.1, -2.0],
            [0.0, 0.2, 0.4],
        ],
        initial_chassis_state=[0.0, 0.0, 0.0],
    )

    assert result.target_states[:, 2] == pytest.approx(
        [0.0, math.pi / 2.0, math.pi / 2.0]
    )


def test_path_is_anchored_to_initial_chassis_position() -> None:
    result = build_planar_path_reference(
        [[2.0, 3.0, 0.0], [2.1, 3.0, 0.0]],
        initial_chassis_state=[10.0, -4.0, 0.0],
    )

    np.testing.assert_allclose(
        result.target_states[:, :2],
        [[10.0, -4.0], [10.1, -4.0]],
    )


def test_seed_rollout_exactly_uses_unicycle_dynamics() -> None:
    config = PlanarPathReferenceConfig(time_step=0.1)
    result = build_planar_path_reference(
        [
            [0.0, 0.0, 0.0],
            [0.02, -0.02, 0.0],
            [0.05, -0.05, 0.0],
        ],
        initial_chassis_state=[0.0, 0.0, 0.0],
        config=config,
    )

    for index, command in enumerate(result.seed_commands):
        expected = integrate_unicycle_euler(
            result.seed_chassis_states[index],
            command,
            time_step=config.time_step,
        )
        assert result.seed_chassis_states[index + 1] == pytest.approx(
            expected
        )


def test_zero_length_segment_keeps_heading_and_stops() -> None:
    result = build_planar_path_reference(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
        initial_chassis_state=[0.0, 0.0, -0.3],
    )

    assert result.target_states[1, 2] == pytest.approx(-0.3)
    assert result.seed_commands[0] == pytest.approx([0.0, 0.0])


def test_reference_arrays_are_owned_and_read_only() -> None:
    source = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    result = build_planar_path_reference(
        source,
        initial_chassis_state=[0.0, 0.0, 0.0],
    )
    source[1, 0] = 99.0

    assert result.target_states[1, 0] == pytest.approx(0.1)
    assert not result.target_states.flags.writeable
    assert not result.seed_chassis_states.flags.writeable
    assert not result.seed_commands.flags.writeable


@pytest.mark.parametrize(
    "planar_states",
    (
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0], [0.1, 0.0]],
        [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]],
    ),
)
def test_invalid_planar_path_is_rejected(planar_states) -> None:
    with pytest.raises(ValueError):
        build_planar_path_reference(
            planar_states,
            initial_chassis_state=[0.0, 0.0, 0.0],
        )


@pytest.mark.parametrize(
    "override",
    (
        {"time_step": 0.0},
        {"minimum_linear_velocity": 0.1},
        {"minimum_angular_velocity": 0.1},
        {"heading_gain": 0.0},
        {"stationary_position_tolerance": -1.0},
    ),
)
def test_invalid_configuration_is_rejected(override) -> None:
    with pytest.raises(ValueError):
        PlanarPathReferenceConfig(**override)
