import json
from math import pi

import pytest

from scripts.evaluation.audit_crate_clearance import audit


def test_turn_and_braking_are_included_in_whole_chassis_passage(tmp_path):
    # A northbound world route and a crate along its right. At local x=2.6 the
    # center has passed the far edge; the rear of the padded chassis still overlaps
    # its longitudinal range. The entire body clears it during the braking sample.
    files = {
        "invocation.json": {
            "case": "crate-static",
            "stage": "motion",
            "source_revision": "test",
        },
        "fixture-monitor.json": {
            "anchor_world_xy_yaw": [10, 4, pi / 2],
            "fixtures": [
                {
                    "name": "BrokenCrate",
                    "center_world_xy": [10.6, 6],
                    "monitor_radius_m": 0.5,
                }
            ],
        },
        "restaurant-placement-monitor.json": {
            "props": [
                {
                    "name": "BrokenCrate",
                    "bounds_minimum": [10.3, 5.6, 0],
                    "bounds_maximum": [10.9, 6.4, 0.22],
                }
            ]
        },
        "report.json": {
            "reason": "goal_reached",
            "goal_local_xy": [4, 0],
            "nominal_route_policy": {"configured_anchors_local_xy": [[4, 0]]},
            "minimum_clearance_before_stop_m": 0.03,
            "minimum_clearance_during_stop_m": 0.15,
            "completed": True,
        },
        "trajectory.json": [[2, 0.12, 0.2], [2.6, 0, 0]],
        "commands.json": [{"v": 0.1}, {"v": 0.2}],
        "stopping-sweep.json": {
            "records": [
                {"frame": 0, "state": [3.99, 0, 0]},
                {"frame": 1, "state": [4, 0, 0]},
            ]
        },
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    result = audit(tmp_path)
    assert result["all_recorded"]["poses"] == 4
    assert result["first_entire_robot_past_crate"]["phase"] == "braking"
    assert result["reached_goal_with_entire_robot_past_crate"]
    assert result["max_abs_lateral_beside_crate_m"] == pytest.approx(0.12)
    assert result["last_issued"]["v"] == 0.2
