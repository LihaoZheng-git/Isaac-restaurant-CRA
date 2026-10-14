import pytest

from robot.evaluation.kinetic_fixture import (
    actor_offset_velocity,
    summarize_kinetic_records,
)


def test_actor_profile_is_continuous_bounded_and_truth_is_separate():
    for boundary in (2, 5, 7, 10):
        left = actor_offset_velocity(boundary - 1e-8)[0]
        right = actor_offset_velocity(boundary)[0]
        assert left == pytest.approx(right, abs=1e-7)
    assert actor_offset_velocity(3) == (0.12, 0.12, "slow")
    assert actor_offset_velocity(8)[1] == -0.24
    assert actor_offset_velocity(11)[1] == 0
    with pytest.raises(ValueError):
        actor_offset_velocity(-1)
    assert not summarize_kinetic_records([])["passed"]
