import numpy as np
import pytest

from robot.evaluation.person_relative_rear_validation import (
    summarize_person_relative_rear,
)


def trace(side=1, late=False, wrong_side=False):
    rows = []
    for t in np.linspace(0, 4, 81):
        lateral = -side * 0.35 * np.sin(np.pi * t / 4) ** 2
        rows.append(
            dict(
                time=float(t),
                robot_xy=[float(t), float(-lateral if wrong_side else lateral)],
                actor_xyz=[2.0, side * 0.5 * (t - 2), 0.0],
                phase="walk_away" if late and t > 1.0 else "move",
            )
        )
    return rows


@pytest.mark.parametrize("side", [-1, 1])
def test_timed_curve_is_near_crossing_line_and_behind_current_walking_person(side):
    result = summarize_person_relative_rear(
        trace(side), nominal_goal_world_xy=(4.0, 0.0)
    )
    assert result["passed"]
    assert result["crossing_timestamp_seconds"] == pytest.approx(2.0)
    assert result["behind_projection_m"] == pytest.approx(-0.35)
    assert result["trailing_arc_near_line_m"] == pytest.approx(0.35)
    assert result["arc_distance_from_line_m"] <= 0.75


def test_late_and_wrong_side_curves_do_not_pass_the_new_gate():
    late = summarize_person_relative_rear(
        trace(late=True), nominal_goal_world_xy=(4.0, 0.0)
    )
    assert (
        not late["passed"]
        and not late["checks"]["crossing_line_during_transverse_walk"]
    )
    wrong = summarize_person_relative_rear(
        trace(wrong_side=True), nominal_goal_world_xy=(4.0, 0.0)
    )
    assert (
        not wrong["passed"] and not wrong["checks"]["behind_current_person_at_crossing"]
    )


def test_bad_truth_clock_or_phase_cannot_be_silently_accepted():
    rows = trace()
    rows[2]["time"] = rows[1]["time"]
    with pytest.raises(ValueError):
        summarize_person_relative_rear(rows, nominal_goal_world_xy=(4.0, 0.0))
    rows = trace()
    del rows[2]["phase"]
    with pytest.raises(ValueError):
        summarize_person_relative_rear(rows, nominal_goal_world_xy=(4.0, 0.0))
