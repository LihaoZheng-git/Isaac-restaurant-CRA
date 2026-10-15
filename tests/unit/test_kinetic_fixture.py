import pytest
from copy import deepcopy

from robot.evaluation.kinetic_fixture import (
    actor_offset_velocity,
    summarize_kinetic_records,
    summarize_live_relative_motion,
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


def test_live_relative_motion_requires_velocity_difference_and_owned_evidence():
    record = {
        "sensor_frame_id": 10,
        "sensor_timestamp": 3.0,
        "actor_monitor": {
            "elapsed_seconds": 3.0,
            "phase": "slow",
            "velocity_world_xy": [0, 0.12],
            "center_world_xy": [1, 0],
        },
        "diagnostics": {"measured_robot_velocity_world_xy": [0, -0.08]},
        "objects": [
            {
                "attributes": {
                    "motion": {
                        "relative_speed_mps": {"value": 0.20, "status": "inferred"}
                    }
                },
                "diagnostics": {
                    "centroid_world_xy": [1, 0],
                    "velocity_world_xy": [0, 0.12],
                    "policy_return_count": 10,
                    "factor_scores": {"kinetic": {"mode": "evidence_supported"}},
                },
            }
        ],
    }

    def fresh_records(template):
        records = [deepcopy(template) for _ in range(5)]
        for i, item in enumerate(records):
            item["sensor_frame_id"] += i
            item["sensor_timestamp"] += i * 0.1
        return records

    assert summarize_live_relative_motion(fresh_records(record))["passed"]
    assert summarize_live_relative_motion(fresh_records(record))[
        "usable_updates_by_phase"
    ] == {"slow": 5}
    assert summarize_live_relative_motion([record] * 5)["usable_updates"] == 1
    for key in ("sensor_frame_id", "sensor_timestamp"):
        reused = fresh_records(record)
        for item in reused:
            item[key] = record[key]
        assert summarize_live_relative_motion(reused)["usable_updates"] == 1
    wrong = deepcopy(record)
    wrong["objects"][0]["attributes"]["motion"]["relative_speed_mps"]["value"] = 0.04
    assert not summarize_live_relative_motion(fresh_records(wrong))["passed"]
    for key, value in (("policy_return_count", 0), ("centroid_world_xy", [5, 0])):
        unavailable = deepcopy(record)
        unavailable["objects"][0]["diagnostics"][key] = value
        assert (
            summarize_live_relative_motion(fresh_records(unavailable))["usable_updates"]
            == 0
        )
    prior = deepcopy(record)
    prior["objects"][0]["attributes"]["motion"]["relative_speed_mps"]["status"] = (
        "prior"
    )
    assert not summarize_live_relative_motion(fresh_records(prior))["passed"]
    transition = deepcopy(record)
    transition["actor_monitor"]["elapsed_seconds"] = 2.1
    assert (
        summarize_live_relative_motion(fresh_records(transition))["usable_updates"] == 0
    )
