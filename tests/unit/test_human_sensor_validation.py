import copy

import numpy as np
import pytest

from robot.evaluation.human_sensor_validation import summarize_human_sensor
from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE


def episode():
    def y(t):
        return 0.7 * max(0, t - 2.5)

    truth = [
        {"time": float(t), "actor_xyz": [10, y(t), 0]} for t in np.arange(0, 6, 0.05)
    ]
    records = []
    for t in [*np.arange(1, 2, 0.1), *np.arange(3, 4, 0.1)]:
        moving = t > 2.5
        records.append(
            {
                "time": float(t),
                "camera_time": float(t),
                "phase": "move" if moving else "idle_before",
                "completion_age_sim_seconds": 0.1,
                "objects": [
                    {
                        "attributes": {
                            "track_id": 4,
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
                                    "timestamp_seconds": float(t),
                                    "status": "inferred",
                                }
                                for k, v in (
                                    ("velocity_x_mps", 0),
                                    ("velocity_y_mps", 0.7 if moving else 0),
                                    ("relative_speed_mps", 0.7 if moving else 0),
                                )
                            },
                        },
                        "diagnostics": {
                            "accepted_instance": True,
                            "policy_return_count": 20,
                            "instance_centroid_world_xyz": [10, y(t), 0.7],
                            "factor_scores": {
                                k: {"score": 0.25, "mode": "configured_prior"}
                                for k in ("potential", "kinetic", "contact", "thermal")
                            },
                        },
                        "phi": 0.7,
                    }
                ],
            }
        )
    return records, truth


def test_fresh_prior_association_and_current_velocity_have_independent_acceptance():
    records, truth = episode()
    report = summarize_human_sensor(records, truth)
    assert report["passed"]
    assert report["median_moving_velocity_error_mps"] < 1e-8
    assert report["stationary_count"] >= 3 and report["steady_moving_count"] >= 5
    assert all(r["mass_status"] == "prior" for r in report["matched_fresh_prior_rows"])


@pytest.mark.parametrize(
    "problem",
    [
        "stale_mass",
        "stale_motion",
        "no_ownership",
        "wrong_person",
        "wrong_velocity",
        "borrowed_shape",
        "prior_motion",
    ],
)
def test_missing_or_stale_evidence_and_wrong_velocity_never_pass(problem):
    records, truth = episode()
    records = copy.deepcopy(records)
    for r in records:
        o = r["objects"][0]
        if problem == "stale_mass":
            o["attributes"]["mass"]["mass_kg"]["timestamp_seconds"] -= 0.2
        elif problem == "stale_motion":
            o["attributes"]["motion"]["relative_speed_mps"]["timestamp_seconds"] -= 0.2
        elif problem == "no_ownership":
            o["diagnostics"]["policy_return_count"] = 0
        elif problem == "wrong_person":
            o["diagnostics"]["instance_centroid_world_xyz"][0] += 1
        elif problem == "wrong_velocity":
            o["attributes"]["motion"]["velocity_x_mps"]["value"] = 1
        elif problem == "borrowed_shape":
            o["diagnostics"]["factor_scores"]["contact"]["mode"] = "persistence_prior"
        elif problem == "prior_motion":
            o["attributes"]["motion"]["relative_speed_mps"]["status"] = "prior"
    assert not summarize_human_sensor(records, truth)["passed"]


def test_oracle_evaluation_epochs_cannot_be_duplicated():
    records, truth = episode()
    truth[2]["time"] = truth[1]["time"]
    with pytest.raises(ValueError, match="epochs"):
        summarize_human_sensor(records, truth)


@pytest.mark.parametrize("second_human", [False, True])
def test_nearby_carried_object_is_not_a_second_human_but_competing_humans_are_ambiguous(
    second_human,
):
    records, truth = episode()
    for row in records:
        other = copy.deepcopy(row["objects"][0])
        other["attributes"]["track_id"] = 5
        if not second_human:
            other["attributes"]["mass"]["mass_kg"]["source"] = "visual_mass_runtime"
        row["objects"].append(other)
    assert summarize_human_sensor(records, truth)["passed"] is not second_human
