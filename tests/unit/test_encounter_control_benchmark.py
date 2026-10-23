import copy
import json

import numpy as np
import pytest

from robot.evaluation.encounter_control_benchmark import (
    oracle_actor,
    run_encounter_control_case,
    summarize_oracle_rear_pass,
)


def test_oracle_fixtures_share_exogenous_onset_and_keep_mirrors_parallel_distinct():
    for case in (
        "cross-left-right",
        "cross-right-left",
        "parallel-same",
        "parallel-opposite",
        "cross-then-parallel",
    ):
        assert np.linalg.norm(oracle_actor(case, 10.0)[1]) == 0
    left, lv = oracle_actor("cross-left-right", 13.0)
    right, rv = oracle_actor("cross-right-left", 13.0)
    np.testing.assert_array_equal(left, [right[0], -right[1]])
    np.testing.assert_array_equal(lv, -rv)
    assert oracle_actor("parallel-same", 13.0)[1][0] > 0.4
    assert oracle_actor("parallel-opposite", 13.0)[1][0] < -0.4
    before, _ = oracle_actor("cross-then-parallel", 15.5 - 1e-7)
    after, velocity = oracle_actor("cross-then-parallel", 15.5)
    np.testing.assert_allclose(before, after, atol=1e-6)
    np.testing.assert_array_equal(velocity, [0.5, 0.0])


def test_short_bound_cannot_claim_arrival_and_preserves_other_obstacle_guidance():
    result = run_encounter_control_case(
        "cross-then-parallel", "encounter", maximum_seconds=0.2
    )
    assert not result["passed"] and result["arrival_seconds"] is None
    assert len(result["rows"]) == 1
    assert result["rows"][0]["lambdas"] == {7: 1.25, 9: 1.25}
    assert result["minimum_clearance_m"] > 0.02
    with pytest.raises(ValueError):
        run_encounter_control_case("unknown", "encounter")


def rear_trace(case="cross-left-right", *, amplitude=0.3, peak_time=14.0):
    sign = -1 if case == "cross-right-left" else 1
    rows = []
    for time in np.arange(0.0, 24.0, 0.2):
        end = time + 0.2
        y = -sign * amplitude * np.exp(-(((end - peak_time) / 1.5) ** 2))
        rows.append(
            dict(
                time=float(time),
                state=[min(3.9, end * 0.17), float(y), 0.0],
                cbf_margin=0.1,
                emergency=False,
            )
        )
    return dict(
        case=case,
        scope="ORACLE moving-disk virtual unicycle; unit scoring trace",
        passed=True,
        reason="goal_reached",
        arrival_seconds=23.0,
        minimum_clearance_m=0.4,
        rows=rows,
    )


@pytest.mark.parametrize(
    "case", ["cross-left-right", "cross-right-left", "cross-then-parallel"]
)
def test_rear_pass_scoring_accepts_trailing_arc_and_plane_crossing_in_both_directions(
    case,
):
    trial = rear_trace(case)
    before = copy.deepcopy(trial)
    result = summarize_oracle_rear_pass(trial)
    assert result["passed"]
    assert result["maximum_trailing_excursion_during_walking_m"] == pytest.approx(0.3)
    assert result["behind_projection_at_route_crossing_m"] < -0.1
    assert trial == before
    json.dumps(result)  # Diagnostics must remain serializable numeric evidence.


@pytest.mark.parametrize(
    "failure",
    ["small", "wrong_side", "late", "route", "clearance", "emergency", "margin"],
)
def test_arrival_or_large_absolute_curve_cannot_hide_failed_rear_pass_gate(failure):
    trial = rear_trace(
        amplitude=0.1
        if failure == "small"
        else -0.3
        if failure == "wrong_side"
        else 0.3,
        peak_time=18.0 if failure == "late" else 14.0,
    )
    if failure == "route":
        trial["passed"] = False
    if failure == "clearance":
        trial["minimum_clearance_m"] = 0.01
    if failure == "emergency":
        trial["rows"][0]["emergency"] = True
    if failure == "margin":
        trial["rows"][0]["cbf_margin"] = -0.01
    assert not summarize_oracle_rear_pass(trial)["passed"]


@pytest.mark.parametrize("failure", ["native_scope", "parallel", "missing_tick", "nan"])
def test_oracle_scoring_rejects_wrong_scope_or_unresolved_trace_clock(failure):
    trial = rear_trace()
    if failure == "native_scope":
        trial["scope"] = "native sensor data"
    if failure == "parallel":
        trial["case"] = "parallel-same"
    if failure == "missing_tick":
        del trial["rows"][2]
    if failure == "nan":
        trial["rows"][2]["state"][0] = np.nan
    with pytest.raises(ValueError):
        summarize_oracle_rear_pass(trial)
