"""Stronger field response must not fabricate risk, evidence or geometry."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from robot.evaluation.restaurant_calibration_runtime import restaurant_calibration_preset
from robot.evaluation.risk_palette import risk_color_rgb
from robot.risk.restaurant_calibration import RestaurantCalibrationConfig
from test_restaurant_calibration_runtime import apply, envelope, object_assessment, seeded_worker


def stronger(endpoint=2.9):
    return restaurant_calibration_preset("kinetic", normal_flux_magnitude_at_unit_risk=endpoint)


@pytest.mark.parametrize("endpoint", [True, float("nan"), float("inf"), 0, 1.69])
def test_invalid_boundary_strength_is_rejected(endpoint):
    with pytest.raises((ValueError, TypeError)):
        stronger(endpoint)


def test_no_silent_strength_change_when_calibration_disabled():
    with pytest.raises(ValueError, match="requires"):
        restaurant_calibration_preset("off", normal_flux_magnitude_at_unit_risk=2.9)
    with pytest.raises(ValueError, match="requires"):
        RestaurantCalibrationConfig(normal_flux_magnitude_at_unit_risk=2.9)


def test_verified_risk_evidence_and_colour_unchanged_with_stronger_boundary():
    items = [object_assessment(1), object_assessment(2, accepted=False),
             object_assessment(3, accepted=False)]
    before, old, _ = apply(items, envelopes=[envelope(3)])
    after, new, audit = apply(items, config=stronger(), envelopes=[envelope(3)])
    for previous, current in zip(before, after):
        assert current.attributes is previous.attributes
        assert current.adapted is previous.adapted
        key = current.attributes.track_id
        assert new[key].risk == old[key].risk
        assert new[key].assessment.as_dict()["factors"] == old[key].assessment.as_dict()["factors"]
        assert risk_color_rgb(new[key].risk, palette="restaurant_v1") == risk_color_rgb(old[key].risk, palette="restaurant_v1")
        assert new[key].tangential_strength == old[key].tangential_strength == 1.25
        assert new[key].passing_side == old[key].passing_side
    assert new[2] is old[2] is items[1].policy  # Ordinary unassociated obstacle.
    assert new[1].boundary_flux < old[1].boundary_flux
    assert new[3].boundary_flux < old[3].boundary_flux
    assert new[3].risk == .8  # Existing explicitly labeled human protection prior.
    assert audit["boundary_mapping"]["unit_risk_magnitude"] == 2.9
    assert audit["unknown_policy_unchanged"]
    assert asdict(stronger()) == dict(asdict(restaurant_calibration_preset("kinetic")),
                                    normal_flux_magnitude_at_unit_risk=2.9)


def test_boundary_endpoint_has_no_1_7_ceiling_and_preserves_risk_order():
    config = stronger(4.1)
    values = [config.normal_flux_magnitude_for_risk(r) for r in (0, .294, .449, .8, 1)]
    assert values == sorted(values)
    assert values[0] == .5 and values[-1] == pytest.approx(4.1)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        config.normal_flux_magnitude_for_risk(1.1)
    # Only the boundary configuration changes; normalized evidence parameters stay.
    assert replace(config, normal_flux_magnitude_at_unit_risk=1.7) == restaurant_calibration_preset("kinetic")


def test_real_field_changes_without_changing_occupancy_risk_or_evidence():
    from test_live_object_perception import packet
    baseline = seeded_worker(restaurant_calibration_preset("kinetic"))
    candidate = seeded_worker(stronger())
    old, new = baseline(packet(2, 1.1)), candidate(packet(2, 1.1))
    for before, after in zip(old.field.pipeline.configuration_space.occupancy_grids,
                             new.field.pipeline.configuration_space.occupancy_grids):
        np.testing.assert_array_equal(before.occupied_mask, after.occupied_mask)
        np.testing.assert_array_equal(before.cell_track_ids, after.cell_track_ids)
        np.testing.assert_array_equal(before.cell_risks, after.cell_risks)
    assert old.objects[0].attributes == new.objects[0].attributes
    assert old.objects[0].policy.risk == new.objects[0].policy.risk
    assert not np.allclose(old.field.pipeline.h_full.values, new.field.pipeline.h_full.values)


def test_unknown_only_field_is_bit_identical_even_with_stronger_endpoint():
    from test_object_risk_map import mapped_worker, multi_packet
    baseline, candidate = mapped_worker(), mapped_worker()
    baseline.restaurant_calibration_config = restaurant_calibration_preset("kinetic")
    candidate.restaurant_calibration_config = stronger(4.1)
    old, new = baseline(multi_packet()), candidate(multi_packet())
    np.testing.assert_array_equal(old.field.pipeline.h_full.values, new.field.pipeline.h_full.values)
    assert [o.policy.as_dict() for o in old.objects] == [o.policy.as_dict() for o in new.objects]


def test_nonstandard_boundary_cannot_enter_primary_without_explicit_configuration():
    from robot.evaluation.restaurant_baselines import applied_field_policies
    item = object_assessment()
    objects, policies, _ = apply([item], config=stronger())
    with pytest.raises(ValueError, match="primary comparison"):
        applied_field_policies(None, policies, objects)
    with pytest.raises(ValueError, match="mismatch"):
        applied_field_policies(None, policies, objects, restaurant_boundary_config=stronger(4.1))
    actual, audit = applied_field_policies(None, policies, objects,
        restaurant_boundary_config=stronger())
    assert actual[1].boundary_flux == policies[1].boundary_flux
    assert audit[0]["restaurant_boundary_response"]["unit_risk_magnitude"] == 2.9
    assert audit[0]["cra_phi"] == audit[0]["applied_field_risk"] == policies[1].risk
