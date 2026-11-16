"""Counterfactual field probes must not rewrite evidence or unknown policies."""

from copy import deepcopy

import pytest

from robot.evaluation.boundary_response import mapped_policies


def record():
    return dict(objects=[dict(track_id=7, diagnostics=dict(restaurant_calibration=dict(
        eligible=True, verified_human=False)), cra_control=dict(safe_sage=dict(
            risk=.3, boundary_flux=-.86)))], diagnostics=dict(applied_field_policies=[
                dict(track_or_group_id=7, applied_field_risk=.3, boundary_flux=-.86,
                     **{"lambda": 1.25}, applied_passing_side="left"),
                dict(track_or_group_id=8, applied_field_risk=.68359375, boundary_flux=-1.3203125,
                     **{"lambda": 1.25}),
            ]))


def test_only_qualified_boundary_changes():
    original = record()
    untouched = deepcopy(original)
    result, audit = mapped_policies(original, 2.9)
    assert result[7].risk == .3
    assert result[7].boundary_flux == pytest.approx(-1.22)
    assert result[7].passing_side.value == "left"
    assert result[7].tangential_strength == 1.25
    assert result[8].boundary_flux == -1.3203125
    assert result[8].risk == .68359375
    assert audit[1]["qualified"] is False
    assert original == untouched


def test_default_exact_policy_parity():
    result, _ = mapped_policies(record(), 1.7)
    assert result[7].boundary_flux == -.86


def test_floor_or_group_cannot_silently_use_individual_risk():
    value = record()
    value["diagnostics"]["applied_field_policies"][0]["applied_field_risk"] = .6
    with pytest.raises(ValueError, match="differs"):
        mapped_policies(value, 2.9)


@pytest.mark.parametrize("endpoint", [True, float("nan"), float("inf"), 1.6])
def test_invalid_endpoint(endpoint):
    with pytest.raises(ValueError):
        mapped_policies(record(), endpoint)


def test_duplicate_policy_rejects():
    value = record()
    value["diagnostics"]["applied_field_policies"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        mapped_policies(value, 2.9)


def test_nullable_carried_group_metadata():
    value = record()
    value["diagnostics"]["carried_object_groups"] = None
    assert mapped_policies(value, 2.9)[0][7].boundary_flux == pytest.approx(-1.22)
