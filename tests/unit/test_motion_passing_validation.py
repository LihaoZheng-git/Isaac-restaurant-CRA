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


def test_idle_root_animation_is_not_a_second_walking_encounter():
    records, truth = example()
    for row in truth:
        row["phase"] = "move"
    # After the real crossing, an idle root bob is closer to the robot and
    # briefly exceeds .3 m/s in the opposite direction. It is not walking.
    for time, x in ((4.1, .2), (4.15, .1), (4.2, .2)):
        truth.append(dict(time=time, phase="stopped_after", actor_xyz=[x, 4, 0], robot_xy=[0, 4]))
    corrected = summarize_motion_passing(records, truth, moving_robot=True, nominal_goal_world_xy=(0, 4))
    assert corrected["passed"]
    assert corrected["closest_moving_encounter_timestamp_seconds"] <= 4.
    assert corrected["excluded_nonwalking_speed_threshold_rows"] > 0
    assert "native walking phase AND" in corrected["encounter_evaluation_basis"]
    for row in truth:
        row.pop("phase")
    legacy = summarize_motion_passing(records, truth, moving_robot=True, nominal_goal_world_xy=(0, 4))
    assert not legacy["checks"]["passes_behind_moving_actor"]
    assert legacy["closest_moving_encounter_timestamp_seconds"] > 4.


def test_phase_metadata_cannot_invent_an_encounter_or_hide_a_wrong_side():
    records, truth = example()
    for row in truth:
        row["phase"] = "idle_before"
    no_encounter = summarize_motion_passing(records, truth, moving_robot=True, nominal_goal_world_xy=(0, 4))
    assert not no_encounter["passed"]
    assert not no_encounter["checks"]["passes_behind_moving_actor"]
    for row in truth:
        row["phase"] = "move"
        row["robot_xy"][0] = -row["robot_xy"][0]
    wrong_side = summarize_motion_passing(records, truth, moving_robot=True, nominal_goal_world_xy=(0, 4))
    assert not wrong_side["passed"]
    assert not wrong_side["checks"]["visible_trailing_side_arc"]


@pytest.mark.parametrize("invalid", ["partial", "unknown"])
def test_native_phase_metadata_must_be_complete_and_recognized(invalid):
    records, truth = example()
    for row in truth:
        row["phase"] = "move"
    if invalid == "partial":
        truth[0].pop("phase")
    else:
        truth[0]["phase"] = "unrecognized"
    with pytest.raises(ValueError, match="phase"):
        summarize_motion_passing(records, truth, moving_robot=True, nominal_goal_world_xy=(0, 4))


@pytest.mark.parametrize('yaw', [0., 1.1])
def test_progress_uses_actual_short_route_and_rejects_lateral_or_incomplete_displacement(yaw):
    records, truth = example()
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    for row in truth:
        row['robot_xy'] = (rotation @ (np.array(row['robot_xy']) * .7)).tolist()
    goal = rotation @ [0., 2.8]
    result = summarize_motion_passing(records, truth, moving_robot=True, nominal_goal_world_xy=goal)
    assert result['checks']['completed_route_progress']
    assert result['checks']['rejoined_original_route_goal']
    assert result['required_route_progress_m'] == pytest.approx(2.65)
    for endpoint in ([0., 2.6], [3., 0.]):
        truth[-1]['robot_xy'] = (rotation @ endpoint).tolist()
        failed = summarize_motion_passing(records, truth, moving_robot=True, nominal_goal_world_xy=goal)
        assert not failed['checks']['completed_route_progress']
        assert not failed['checks']['rejoined_original_route_goal']
