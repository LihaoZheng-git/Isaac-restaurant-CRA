import pytest

from test_crate_v3_controller import command, parse


@pytest.mark.parametrize("motion", [False, True])
def test_sensor_navigation_uses_primary_mpc_on_the_original_encounter(monkeypatch, motion):
    def arguments(name):
        values = command(name) + (["--motion"] if motion else [])
        return vars(parse(monkeypatch, values)["args"])

    previous = arguments("restaurant_crate_delay_robust_bypass")
    current = arguments("restaurant_crate_sensor_navigation")
    changed = {k for k in current if current[k] != previous[k]}
    assert changed <= {
        k for k in current if k.startswith("guidance_")
    } | {"observe_crate_clearance", "crate_v3_controller", "crate_delay_robust_bypass"}
    assert current["observe_crate_clearance"]
    assert current["experiment_baseline"] == "continuous_cra_safe_sage"
    assert current["guidance_tracking_weight"] == 0
    assert not current["crate_delay_robust_bypass"] and not current["crate_v3_controller"]
    assert not current["restaurant_human"] and not current["checked_rear_pass_reference"]
    assert current["static_occupancy_memory_prior"] and current["persistent_object_risk_map"]
    assert current["mpc_horizon_steps"] == 6


def test_longer_primary_prediction_changes_only_horizon(monkeypatch):
    def settings(name):
        return vars(parse(monkeypatch, command(name) + ["--motion"])["args"])

    original = settings("restaurant_crate_sensor_navigation")
    extended = settings("restaurant_crate_sensor_navigation_horizon12")
    assert extended == {**original, "mpc_horizon_steps": 12}
