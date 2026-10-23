import copy

import numpy as np
import pytest

from robot.evaluation.departing_pedestrian_validation import summarize_departing_pedestrian
from robot.scene.restaurant_departing_pedestrian import DepartingPedestrianRoute


def trace():
    rows = []
    for t in np.arange(0, 13, .05):
        x = 27.8 + min(2., max(0., t-1)*.5)
        y = 8.9 + max(0., t-5)*.5
        rows.append(dict(time=float(t), actor_xyz=[x, y, 0.],
            phase="idle_before" if t < 1 else "move" if t < 5 else "walk_away",
            left_foot_xyz=[.2*np.sin(8*t), 0, 0], right_foot_xyz=[0, 0, 0],
            actor_layout_clearance_m=.3, collider_alignment_error_m=.002,
            robot_xy=[28.8, 5.6], nonzero_wheel_command_count=0))
    return rows


def test_departure_is_one_actual_handoff_without_loop_or_authored_teleport():
    route = DepartingPedestrianRoute()
    assert route.goal == (29.8, 8.9, 0.)
    assert not route.advance((28.8, 8.9, 0.), 1.)
    assert route.advance((29.56, 8.9, 0.), 2.)
    assert route.goal == (29.8, 12.5, 0.)
    assert not route.advance(route.goal, 3.)
    assert route.index == 1 and len(route.events) == 1
    assert route.events[0]["actual_xyz"] == [29.56, 8.9, 0.]
    with pytest.raises(ValueError):
        route.advance((np.nan, 8.9, 0.), 4.)


def test_departure_checks_actual_crossing_motion_articulation_and_geometry():
    report = summarize_departing_pedestrian(trace(), parked_robot=True)
    assert report["passed"]
    assert report["median_moving_speed_mps"] == pytest.approx(.5)
    assert "actor_stopped" not in report["checks"]


def test_mirrored_departure_targets_do_not_mutate_the_legacy_route():
    route = DepartingPedestrianRoute(crossing_direction="right-to-left")
    assert route.goal == (27.8, 8.9, 0.)
    assert not route.advance((28.8, 8.9, 0.), 1.)
    assert route.advance((28.04, 8.9, 0.), 2.)
    assert route.goal == (27.8, 12.5, 0.)
    assert not route.advance(route.goal, 3.)
    assert len(route.events) == 1
    assert DepartingPedestrianRoute().goal == (29.8, 8.9, 0.)


def test_mirrored_actual_trace_needs_its_explicit_direction_and_all_geometry_gates():
    rows = copy.deepcopy(trace())
    for row in rows:
        row["actor_xyz"][0] = 57.6 - row["actor_xyz"][0]
    result = summarize_departing_pedestrian(rows, parked_robot=True, crossing_direction="right-to-left")
    assert result["passed"] and result["checks"]["actual_right_to_left_crossing"]
    assert not summarize_departing_pedestrian(rows, parked_robot=True)["passed"]
    assert not summarize_departing_pedestrian(trace(), parked_robot=True,
        crossing_direction="right-to-left")["passed"]
    rows[80]["actor_layout_clearance_m"] = .01
    assert not summarize_departing_pedestrian(rows, parked_robot=True,
        crossing_direction="right-to-left")["passed"]


@pytest.mark.parametrize("direction", [None, "forward", True])
def test_departure_rejects_unsupported_direction(direction):
    with pytest.raises(ValueError, match="direction"):
        DepartingPedestrianRoute(crossing_direction=direction)
    with pytest.raises(ValueError, match="direction"):
        summarize_departing_pedestrian(trace(), parked_robot=True, crossing_direction=direction)


@pytest.mark.parametrize("failure", ["no_crossing", "no_departure", "return", "no_articulation",
    "collision", "collider", "airborne", "wheels", "robot_drift", "nan", "phase", "time"])
def test_departure_cannot_pass_missing_or_invalid_actual_motion(failure):
    rows = copy.deepcopy(trace())
    if failure == "no_crossing":
        for r in rows:
            r["actor_xyz"][0] += 1.
    elif failure == "no_departure":
        for r in rows:
            r["actor_xyz"][1] = 8.9
    elif failure == "return":
        rows[-1]["actor_xyz"][1] = 9.
    elif failure == "no_articulation":
        for r in rows:
            r["left_foot_xyz"] = [0, 0, 0]
    elif failure == "collision":
        rows[80]["actor_layout_clearance_m"] = .01
    elif failure == "collider":
        rows[80]["collider_alignment_error_m"] = .04
    elif failure == "airborne":
        rows[80]["actor_xyz"][2] = .1
    elif failure == "wheels":
        rows[80]["nonzero_wheel_command_count"] = 1
    elif failure == "robot_drift":
        rows[80]["robot_xy"][1] += .1
    elif failure == "nan":
        rows[80]["actor_xyz"][0] = np.nan
    elif failure == "phase":
        rows[80]["phase"] = "stopped_after"
    elif failure == "time":
        rows[80]["time"] = rows[79]["time"]
    if failure in {"nan", "phase", "time"}:
        with pytest.raises(ValueError):
            summarize_departing_pedestrian(rows, parked_robot=True)
    else:
        assert not summarize_departing_pedestrian(rows, parked_robot=True)["passed"]
