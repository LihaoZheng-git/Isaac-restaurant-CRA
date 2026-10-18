import copy

import numpy as np
import pytest

from robot.evaluation.continuous_server_validation import summarize_continuous_server
from robot.scene.restaurant_continuous_server import ContinuousServerRoute
from robot.scene.restaurant_service_build import SHORT_START_WORLD_XYZ, restaurant_route_local_xy


def actual_records():
    rows = []
    for t in np.arange(0, 8, 0.02):
        elapsed = max(0, t - 1)
        xy = (
            (29.8 - 0.5 * elapsed, 8.9)
            if elapsed <= 3.6
            else (28.0, 8.9 + 0.5 * (elapsed - 3.6))
        )
        rows.append(
            dict(
                time=float(t),
                actor_xyz=[*xy, 0.0],
                phase="idle_before" if t < 1 else "move",
                left_foot_xyz=[0.1 * np.sin(8 * t), 0, 0],
                right_foot_xyz=[0, 0, 0],
                robot_xy=[28.8, 5.6],
                nonzero_wheel_command_count=0,
                actor_layout_clearance_m=0.2,
                collider_alignment_error_m=0.002,
            )
        )
    return rows


def test_continuous_circuit_uses_actual_progress_and_wraps():
    route = ContinuousServerRoute()
    assert not route.advance((29.8, 8.9, 0), 1.0)
    for i in range(4):
        target = route.goal
        assert route.advance(target, i + 2.0)
    assert route.index == 0 and len(route.events) == 4
    assert all(e["distance_m"] == 0 for e in route.events)
    with pytest.raises(ValueError):
        route.advance((float("nan"), 8.9, 0), 10.0)


def test_continuous_actor_accepts_actual_crossing_without_endpoint_stop_claim():
    report = summarize_continuous_server(
        actual_records(), requested_speed_mps=0.5, parked_robot=True
    )
    assert report["passed"]
    assert "actor_stopped" not in report["checks"]
    assert report["terminal_one_second_displacement_m"] >= 0.45


@pytest.mark.parametrize("view,start", [("approach-forward", SHORT_START_WORLD_XYZ[:2]), ("crossing-approach", (28.8,7.4))])
def test_forward_robot_start_preserves_same_fixed_world_service_waypoints(view, start):
    local = np.array(restaurant_route_local_xy("human-service", human_start_view=view))
    world = local @ np.array([[0,1],[-1,0]]) + start
    np.testing.assert_allclose(world, [(28.8,7.8),(28.8,8.7),(28.8,9.5)], atol=1e-10)
    with pytest.raises(ValueError):
        restaurant_route_local_xy("short", human_start_view=view)


@pytest.mark.parametrize(
    "failure",
    [
        "no_crossing",
        "static_end",
        "stopped_phase",
        "no_articulation",
        "layout_collision",
        "collider_lag",
        "outside",
        "wheel_command",
        "robot_drift",
        "nan",
    ],
)
def test_continuous_actor_still_rejects_missing_motion_or_geometry(failure):
    rows = copy.deepcopy(actual_records())
    if failure == "no_crossing":
        for r in rows:
            r["actor_xyz"][0] += 1.0
    elif failure == "static_end":
        for r in rows[-60:]:
            r["actor_xyz"] = rows[-61]["actor_xyz"]
    elif failure == "stopped_phase":
        rows[-1]["phase"] = "stopped_after"
    elif failure == "no_articulation":
        for r in rows:
            r["left_foot_xyz"] = [0, 0, 0]
    elif failure == "layout_collision":
        rows[80]["actor_layout_clearance_m"] = -0.01
    elif failure == "collider_lag":
        rows[80]["collider_alignment_error_m"] = 0.05
    elif failure == "outside":
        rows[80]["actor_xyz"][1] = 15.0
    elif failure == "wheel_command":
        rows[80]["nonzero_wheel_command_count"] = 1
    elif failure == "robot_drift":
        rows[80]["robot_xy"] = [28.8, 5.8]
    elif failure == "nan":
        rows[80]["actor_layout_clearance_m"] = float("nan")
    assert not summarize_continuous_server(
        rows, requested_speed_mps=0.5, parked_robot=True
    )["passed"]
