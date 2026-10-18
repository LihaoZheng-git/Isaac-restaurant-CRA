from dataclasses import replace

import numpy as np
import pytest

from robot.perception.object_height import ObjectHeightConfig, estimate_object_height
from robot.perception.physical_attributes import EstimateStatus


def plane(z=0.0, xr=(-1.5, 1.5), yr=(-1.5, 1.5), n=41):
    xy = np.array([(x, y) for x in np.linspace(*xr, n) for y in np.linspace(*yr, n)])
    heights = z(xy) if callable(z) else np.full(len(xy), z)
    return np.column_stack((xy, heights))


def item(bottom=0.02, height=0.3, width=0.16):
    return np.array(
        [
            (x, y, z)
            for x in (-width / 2, 0, width / 2)
            for y in (-width / 2, 0, width / 2)
            for z in (bottom, bottom + height / 2, bottom + height)
        ]
    )


def estimate(scene, obj=None, **kwargs):
    return estimate_object_height(
        scene,
        item() if obj is None else obj,
        timestamp_seconds=2.1,
        scene_timestamp_seconds=2.0,
        floor_reference_m=0.0,
        reference_scope="synthetic metric Z-up test frame; known ground-band selection",
        **kwargs,
    )


def test_ground_support_is_separate_from_tall_object_center_height():
    obj = item(height=1.8)
    result = estimate(np.vstack((plane(), obj)), obj)
    assert result.support_state == "ground_supported"
    assert result.support_height_above_ground_m.value == pytest.approx(0.0)
    assert result.visible_center_height_m.value > 0.8
    assert result.fall_height_m.value == 0.0
    assert result.ground_height_m.status is EstimateStatus.INFERRED
    assert result.ground_height_m.timestamp_seconds == 2.0


def test_elevated_support_and_floor_give_drop_not_object_size():
    obj = item(bottom=0.82, height=0.3)
    table = plane(0.8, xr=(-0.4, 0.2), yr=(-0.5, 0.5))
    result = estimate(np.vstack((plane(), table, obj)), obj)
    assert result.support_state == "elevated_supported"
    assert result.support_height_above_ground_m.value == pytest.approx(0.8)
    assert result.visible_lower_height_m.value == pytest.approx(0.82)
    # Equal-height opposing landings need the existing explicit opt-in.
    config = ObjectHeightConfig()
    config = replace(
        config,
        surface_config=replace(
            config.surface_config, allow_equal_height_multiple_landings=True
        ),
    )
    result = estimate(np.vstack((plane(), table, obj)), obj, config=config)
    assert result.fall_height_m.value == pytest.approx(0.8)


def test_support_is_preserved_without_a_visible_lower_landing():
    obj = item(bottom=0.82)
    # Local tabletop patch; distant ground is outside the strict landing search.
    floor = plane(xr=(1.05, 1.85), yr=(-0.5, 0.5))
    table = plane(0.8, xr=(-0.4, 0.4), yr=(-0.4, 0.4))
    result = estimate(np.vstack((floor, table, obj)), obj)
    assert result.support_state == "elevated_supported"
    assert result.support_height_m.is_available
    assert not result.fall_height_m.is_available
    assert result.fall_rejection_reasons


def test_visible_upper_fragment_does_not_imply_elevated_support_or_drop():
    obj = item(bottom=0.7, height=0.8)
    result = estimate(np.vstack((plane(), obj)), obj)
    assert result.visible_lower_height_m.value == pytest.approx(0.7)
    assert result.support_state == "uncertain"
    assert not result.support_height_m.is_available
    assert not result.fall_height_m.is_available


def test_low_floor_patch_does_not_invent_zero_drop_at_unseen_edge():
    obj = item()
    result = estimate(np.vstack((plane(xr=(-0.4, 0.4), yr=(-0.4, 0.4)), obj)), obj)
    assert result.support_state == "ground_supported"
    assert not result.fall_height_m.is_available


def test_tabletop_is_not_mislabeled_as_ground_when_floor_unseen():
    obj = item(bottom=0.82)
    result = estimate(np.vstack((plane(0.8), obj)), obj)
    assert not result.ground_height_m.is_available
    assert not result.visible_lower_height_m.is_available
    assert result.support_state == "uncertain"
    assert result.support_height_m.is_available


def test_object_cannot_supply_its_own_ground():
    obj = plane(0.02)
    result = estimate(obj, obj)
    assert not result.ground_height_m.is_available
    assert result.floor_point_count == 0


def test_sloped_floor_is_fitted_instead_of_using_reference_height():
    ground = plane(lambda xy: 0.04 * xy[:, 0] + 0.025)
    obj = item(bottom=0.045)
    result = estimate(np.vstack((ground, obj)), obj)
    assert result.ground_height_m.value == pytest.approx(0.025)
    assert result.visible_lower_height_m.value == pytest.approx(0.02)


def test_reject_narrow_line_as_floor_even_if_both_axes_span():
    xy = np.linspace(-1, 1, 50)
    result = estimate(np.vstack((np.column_stack((xy, xy, np.zeros(50))), item())))
    assert not result.ground_height_m.is_available
    assert "two-dimensional" in result.rejection_reasons[-1]


def test_mixed_height_floor_band_is_rejected():
    result = estimate(np.vstack((plane(-0.08), plane(0.08), item())))
    assert not result.ground_height_m.is_available
    assert "planar" in result.rejection_reasons[-1]


@pytest.mark.parametrize("scene_time", [1.0, 2.2])
def test_reject_stale_or_future_clouds_without_renewing_source(scene_time):
    result = estimate_object_height(
        plane(),
        item(),
        timestamp_seconds=2.1,
        scene_timestamp_seconds=scene_time,
        floor_reference_m=0.0,
        reference_scope="test",
    )
    assert result.support_state == "uncertain"
    assert not result.fall_height_m.is_available
    assert result.ground_height_m.timestamp_seconds == scene_time


def test_coordinate_translation_preserves_elevation():
    obj, scene = item(), np.vstack((plane(), item()))
    offset = np.array([4.0, -2.0, -0.65])
    result = estimate_object_height(
        scene + offset,
        obj + offset,
        timestamp_seconds=2.1,
        scene_timestamp_seconds=2.0,
        floor_reference_m=-0.65,
        reference_scope="translated synthetic frame",
    )
    assert result.support_state == "ground_supported"
    assert result.visible_lower_height_m.value == pytest.approx(0.02)


def test_far_floor_extrapolation_does_not_establish_height():
    result = estimate(
        np.vstack((plane(xr=(1.8, 2.1), yr=(-0.5, 0.5)), item())),
        config=replace(ObjectHeightConfig(), minimum_floor_span_m=0.2),
    )
    assert not result.ground_height_m.is_available
    assert "too far" in result.rejection_reasons[-1]


def test_below_floor_association_is_rejected():
    obj = item(bottom=-0.5, height=0.1)
    result = estimate(np.vstack((plane(), obj)), obj)
    assert not result.visible_lower_height_m.is_available
    assert result.support_state == "uncertain"
    assert "below fitted floor" in result.rejection_reasons[-1]


def test_negative_support_offset_is_not_clipped_into_ground_support():
    obj = item(bottom=-0.13, height=0.35)
    result = estimate(np.vstack((plane(), plane(-0.15), obj)), obj)
    assert result.support_height_m.is_available
    assert result.support_state == "uncertain"
    assert not result.support_height_above_ground_m.is_available
    assert not result.fall_height_m.is_available
    assert "support lies below" in result.rejection_reasons[-1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_object_points": True},
        {"search_radius_m": float("nan")},
        {"confidence_cap": 1.2},
        {"maximum_floor_extrapolation_m": 3.0},
        {"minimum_elevated_support_m": 0.01},
        {"surface_config": object()},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises((ValueError, TypeError)):
        ObjectHeightConfig(**kwargs)


def test_nonfinite_geometry_is_rejected():
    with pytest.raises(ValueError, match="finite Nx3"):
        estimate([[0.0, 0.0, float("nan")]])


def test_multiple_plausible_support_levels_are_not_chosen_by_point_count():
    config = ObjectHeightConfig()
    config = replace(
        config,
        surface_config=replace(
            config.surface_config, support_maximum_gap_m=0.2, support_xy_margin_m=0.3
        ),
    )
    obj = item(bottom=0.17)
    result = estimate(np.vstack((plane(), plane(0.15), obj)), obj, config=config)
    assert result.support_state == "uncertain"
    assert not result.support_height_m.is_available
    assert not result.fall_height_m.is_available
    assert (
        max(result.support_candidate_heights_m)
        - min(result.support_candidate_heights_m)
        > 0.1
    )
    assert any("ambiguous" in reason for reason in result.rejection_reasons)


def test_sparse_returns_stay_unavailable_and_input_arrays_unchanged():
    scene, obj = plane(), item()[:4]
    scene_before, obj_before = scene.copy(), obj.copy()
    result = estimate(scene, obj)
    assert result.support_state == "uncertain"
    assert "insufficient object" in result.rejection_reasons[0]
    np.testing.assert_array_equal(scene, scene_before)
    np.testing.assert_array_equal(obj, obj_before)
