import numpy as np
import pytest

from robot.evaluation.human_navigation_validation import (
    human_stationary_checks,
    summarize_human_navigation,
)
from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE


def episode(mode="crossing"):
    def x(t):
        return max(27.8, 29.8 - 0.7 * max(0, t - 2)) if mode == "crossing" else 29.8

    truth = [
        {
            "time": float(t),
            "actor_xyz": [x(t), 8.9, 0],
            "phase": "stopped_after" if t > 5 else "move",
        }
        for t in np.arange(0, 6, 0.05)
    ]
    records = []
    for t in np.arange(2.5, 4.5, 0.1):
        records.append(
            {
                "sensor_timestamp": float(t),
                "commit_simulation_time": float(t + 0.15),
                "diagnostics": {"measured_robot_velocity_world_xy": [0, 0.2]},
                "objects": [
                    {
                        "track_id": 7,
                        "cra_control": {
                            "safe_sage": {"risk": 0.8, "tangential_strength": 1.25}
                        },
                        "attributes": {
                            "mass": {
                                "mass_kg": {
                                    "source": HUMAN_MASS_PRIOR_SOURCE,
                                    "status": "prior",
                                    "value": 70,
                                    "timestamp_seconds": float(t),
                                }
                            },
                            "motion": {
                                k: {
                                    "value": v,
                                    "status": "inferred",
                                    "timestamp_seconds": float(t),
                                }
                                for k, v in (
                                    (
                                        "velocity_x_mps",
                                        -0.7 if mode == "crossing" else 0,
                                    ),
                                    ("velocity_y_mps", 0),
                                    (
                                        "relative_speed_mps",
                                        0.7280109889280518
                                        if mode == "crossing"
                                        else 0.2,
                                    ),
                                )
                            },
                        },
                        "diagnostics": {
                            "accepted_instance": True,
                            "centroid_world_xy": [x(t), 8.9],
                            "policy_return_count": 20,
                            "factor_scores": {
                                n: {
                                    "mode": "evidence_supported"
                                    if n in {"kinetic", "contact"}
                                    else "configured_prior",
                                    "score": 0.25,
                                }
                                for n in ("potential", "kinetic", "contact", "thermal")
                            },
                            "property_evidence": {
                                n: {"mode": "fresh" if n == "mass" else "unavailable"}
                                for n in (
                                    "mass",
                                    "contact",
                                    "fall_height",
                                    "fall_interaction",
                                    "heat",
                                )
                            },
                        },
                    }
                ],
            }
        )
    return records, truth


@pytest.mark.parametrize("mode", ["standing", "crossing"])
def test_independent_current_human_motion_and_numeric_policy_checks(mode):
    records, truth = episode(mode)
    assert all(human_stationary_checks(records).values())
    report = summarize_human_navigation(records, truth, mode=mode)
    assert report["passed"]
    if mode == "crossing":
        assert report["simultaneous_rows"] >= 5
        assert report["median_steady_world_velocity_error_mps"] < 1e-8


@pytest.mark.parametrize(
    "failure",
    [
        "late",
        "future",
        "stale_motion",
        "prior_motion",
        "wrong_speed",
        "no_ownership",
        "no_simultaneous_motion",
        "borrowed_contact",
        "expired_mass",
        "future_mass",
        "changed_lambda",
        "wrong_relative_speed",
        "no_arrival",
        "nan",
    ],
)
def test_incomplete_or_unsafe_crossing_records_fail(failure):
    records, truth = episode()
    for r in records:
        o = r["objects"][0]
        if failure == "late":
            r["commit_simulation_time"] += 0.3
        elif failure == "future":
            r["commit_simulation_time"] = r["sensor_timestamp"] - 0.1
        elif failure == "stale_motion":
            o["attributes"]["motion"]["relative_speed_mps"]["timestamp_seconds"] -= 0.2
        elif failure == "prior_motion":
            o["attributes"]["motion"]["velocity_x_mps"]["status"] = "prior"
        elif failure == "wrong_speed":
            o["attributes"]["motion"]["velocity_x_mps"]["value"] = 0
        elif failure == "no_ownership":
            o["diagnostics"]["policy_return_count"] = 0
        elif failure == "no_simultaneous_motion":
            r["diagnostics"]["measured_robot_velocity_world_xy"] = [0, 0]
        elif failure == "borrowed_contact":
            o["diagnostics"]["property_evidence"]["contact"]["mode"] = (
                "persistence_prior"
            )
        elif failure == "expired_mass":
            o["attributes"]["mass"]["mass_kg"]["timestamp_seconds"] -= 4
        elif failure == "future_mass":
            o["attributes"]["mass"]["mass_kg"]["timestamp_seconds"] += 1
        elif failure == "changed_lambda":
            o["cra_control"]["safe_sage"]["tangential_strength"] = 1.8
        elif failure == "wrong_relative_speed":
            o["attributes"]["motion"]["relative_speed_mps"]["value"] = 1.4
        elif failure == "nan":
            o["diagnostics"]["factor_scores"]["contact"]["score"] = float("nan")
    if failure == "no_arrival":
        truth[-1]["phase"] = "move"
    assert not summarize_human_navigation(records, truth, mode="crossing")["passed"]


def test_stationary_gate_requires_actual_fresh_visual_mass_and_contact():
    records, _ = episode()
    for r in records:
        r["objects"][0]["diagnostics"]["property_evidence"]["mass"]["mode"] = (
            "persistence_prior"
        )
    assert not all(human_stationary_checks(records).values())


def test_standing_reference_rejects_translating_actor_and_wrong_world_velocity():
    records, truth = episode("standing")
    for r in truth:
        r["actor_xyz"][0] += 0.06 * r["time"]
    assert not summarize_human_navigation(records, truth, mode="standing")["passed"]
    records, truth = episode("standing")
    for r in records:
        r["objects"][0]["attributes"]["motion"]["velocity_x_mps"]["value"] = 0.4
    assert not summarize_human_navigation(records, truth, mode="standing")["passed"]
