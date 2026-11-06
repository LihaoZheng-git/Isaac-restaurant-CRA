import json
from math import pi

import pytest

from scripts.evaluation.audit_crate_clearance import audit


@pytest.mark.parametrize(("label", "name"), [("crate", "BrokenCrate"), ("bin", "ServiceContainer")])
def test_turn_and_braking_are_included_in_whole_chassis_passage(tmp_path, label, name):
    # A northbound world route and a crate along its right. At local x=2.6 the
    # center has passed the far edge; the rear of the padded chassis still overlaps
    # its longitudinal range. The entire body clears it during the braking sample.
    files = {
        "invocation.json": {
            "case": f"{label}-static",
            "stage": "motion",
            "source_revision": "test",
        },
        "fixture-monitor.json": {
            "anchor_world_xy_yaw": [10, 4, pi / 2],
            "fixtures": [
                {
                    "name": name,
                    "center_world_xy": [10.6, 6],
                    "monitor_radius_m": 0.5,
                }
            ],
        },
        "restaurant-placement-monitor.json": {
            "props": [
                {
                    "name": name,
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
    assert result[f"first_entire_robot_past_{label}"]["phase"] == "braking"
    assert result[f"reached_goal_with_entire_robot_past_{label}"]
    assert result[f"max_abs_lateral_beside_{label}_m"] == pytest.approx(0.12)
    assert result["last_issued"]["v"] == 0.2
    files["fixture-monitor.json"]["fixtures"][0]["name"] = "AnotherProp"
    (tmp_path / "fixture-monitor.json").write_text(json.dumps(files["fixture-monitor.json"]))
    with pytest.raises(ValueError, match="fixture monitor"):
        audit(tmp_path)
