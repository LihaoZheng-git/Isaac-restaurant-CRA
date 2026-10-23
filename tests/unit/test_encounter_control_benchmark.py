import numpy as np
import pytest

from robot.evaluation.encounter_control_benchmark import (
    oracle_actor,
    run_encounter_control_case,
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
