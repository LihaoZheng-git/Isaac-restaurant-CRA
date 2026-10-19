import copy

import numpy as np
import pytest

from robot.evaluation.motion_passing_validation import summarize_motion_passing


def example(curved=True):
    truth = []
    for t in np.linspace(0, 4, 81):
        truth.append(
            {
                "time": float(t),
                "actor_xyz": [-1 + 0.5 * t, 2, 0],
                "robot_xy": [-0.4 * np.sin(np.pi * t / 4) ** 2 if curved else 0.0, t],
            }
        )
    records = []
    for t in np.linspace(1, 3, 21):
        records.append(
            {
                "time": float(t),
                "motion_aware_passing": {
                    "preferences": {"7": "left"},
                    "encounter_selected_timestamp": 1.0,
                    "selection_velocity_world_xy": [0.5, 0],
                },
                "applied_field_policies": [
                    {"lambda": 1.25, "boundary_flux": -1.22, "applied_field_risk": 0.6}
                ],
                "objects": [
                    {
                        "attributes": {"track_id": 7},
                        "diagnostics": {"centroid_world_xy": [-1 + 0.5 * t, 2]},
                    }
                ],
            }
        )
    return records, truth


def test_arrival_alone_cannot_pass_the_curved_bypass_gate():
    records, truth = example(False)
    result = summarize_motion_passing(
        records, truth, moving_robot=True, nominal_goal_world_xy=(0, 4)
    )
    assert not result["passed"] and not result["checks"]["visible_trailing_side_arc"]
    assert result["checks"]["rejoined_original_route_goal"]
    records, truth = example(True)
    result = summarize_motion_passing(
        records, truth, moving_robot=True, nominal_goal_world_xy=(0, 4)
    )
    assert result["passed"]
    assert result["maximum_left_excursion_during_encounter_m"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "failure", ["identity", "epoch", "lambda", "no_policy", "missed_goal", "wrong_side"]
)
def test_bypass_claim_rejects_wrong_ownership_prior_epoch_or_missing_data(failure):
    records, truth = copy.deepcopy(example())
    if failure == "identity":
        records[0]["objects"][0]["diagnostics"]["centroid_world_xy"][0] += 2
    elif failure == "epoch":
        records[0]["motion_aware_passing"]["encounter_selected_timestamp"] = -9
    elif failure == "lambda":
        records[0]["applied_field_policies"][0]["lambda"] = 2
    elif failure == "no_policy":
        for record in records:
            record["motion_aware_passing"]["preferences"] = {}
    elif failure == "missed_goal":
        truth[-1]["robot_xy"][0] = 0.5
    elif failure == "wrong_side":
        records[0]["motion_aware_passing"]["preferences"] = {"7": "right"}
    assert not summarize_motion_passing(
        records, truth, moving_robot=True, nominal_goal_world_xy=(0, 4)
    )["passed"]
