import copy
from types import SimpleNamespace

import numpy as np
import pytest
from test_encounter_guidance import inputs

from robot.evaluation.checked_rear_pass_benchmark import (
    CheckedRearPassBenchmarkRuntime,
    rear_pass_policy_adapter,
)
from robot.evaluation.encounter_control_benchmark import run_encounter_control_case


@pytest.mark.parametrize(
    "obstacle,mode,expected_lambda,ids",
    [
        ((3.3, -0.5), "rear", 1.25, (7,)),
        ((1.0, -0.5), "wait", 0.0, (7,)),
        ((3.3, 1.3), "none", 1.25, ()),
    ],
)
def test_behavior_changes_only_current_owned_lambda_with_unchanged_physical_policy(
    obstacle, mode, expected_lambda, ids
):
    g, a, c, p = inputs()
    before = copy.deepcopy((g, a, c, p))
    context = dict(
        state=np.zeros(3),
        goal=(3.9, 0.0),
        tracking=SimpleNamespace(
            tracked_obstacles=[
                SimpleNamespace(
                    track_id=7, centroid_world_xy=obstacle, velocity_world_xy=(0.0, 0.5)
                )
            ]
        ),
    )
    changed, rows, active = rear_pass_policy_adapter(
        g, a, c, p, source_time=2.0, context=context
    )
    assert active == ids
    assert all(changed[k].tangential_strength == expected_lambda for k in (7, 8, 70))
    assert all(changed[k].tangential_strength == 1.25 for k in (9, 1000000007))
    for k in g:
        assert (
            changed[k].risk == g[k].risk
            and changed[k].boundary_flux == g[k].boundary_flux
        )
    for row in rows:
        assert row["physical_CRA_lambda"] == 1.25
        assert row["lambda"] == changed[row["track_or_group_id"]].tangential_strength
        assert row["encounter_guidance_ablation"]["source_timestamp_seconds"] == 2.0
        if "rear_pass_behavior" in row:
            assert row["rear_pass_behavior"]["mode"] == mode
            assert (
                row["rear_pass_behavior"]["original_selection_timestamp_seconds"] == 1.0
            )
    assert (g, a, c, p) == before


def test_parallel_motion_keeps_normal_guidance_without_reusing_a_crossing_reference():
    g, a, c, p = inputs("parallel")
    changed, _, ids = rear_pass_policy_adapter(
        g,
        a,
        c,
        p,
        source_time=2.0,
        context=dict(tracking=SimpleNamespace(tracked_obstacles=[])),
    )
    assert not ids and changed == g


def test_short_oracle_trial_without_crossing_is_primary_and_cannot_claim_arrival():
    baseline = run_encounter_control_case(
        "cross-left-right", "encounter", maximum_seconds=0.2
    )
    experimental = run_encounter_control_case(
        "cross-left-right",
        "encounter",
        maximum_seconds=0.2,
        control_factory=CheckedRearPassBenchmarkRuntime,
        encounter_policy_adapter=rear_pass_policy_adapter,
    )
    assert not experimental["passed"]
    np.testing.assert_array_equal(
        experimental["rows"][0]["command"], baseline["rows"][0]["command"]
    )
    np.testing.assert_array_equal(
        experimental["endpoint_state"], baseline["endpoint_state"]
    )
    assert not experimental["rows"][0]["rear_pass_reference"]["selected"]
