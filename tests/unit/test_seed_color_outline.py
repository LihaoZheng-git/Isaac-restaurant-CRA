"""RGB-derived protrusion recovery and conservative domain rejection."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from robot.perception.seed_color_outline import (
    SeedColorOutlineConfig,
    refine_seed_color_outline,
)
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection
from robot.perception.visual_shape_contact import estimate_silhouette_contact_proxy


def fixture(tip=False):
    seed = np.zeros((300, 300), np.uint8)
    cv2.circle(seed, (150, 150), 40, 1, -1)
    visible = seed.copy()
    if tip:
        cv2.fillPoly(visible, [np.array([[184, 141], [220, 150], [184, 159]])], 1)
    rgb = np.full((300, 300, 3), 90, np.uint8)
    rgb[visible.astype(bool)] = (30, 110, 220)
    detection = SemanticDetection(
        7, "arbitrary", 0.95, PixelBoundingBox(110, 110, 191, 191), seed.astype(bool)
    )
    return rgb, detection


def refine(rgb, item, **kwargs):
    return refine_seed_color_outline(
        rgb, item, uniform_color_domain_validated=True, **kwargs
    )


def test_rgb_recovers_tip_omitted_by_detector_and_raises_contact_cue():
    rgb, seed = fixture(True)
    result = refine(rgb, seed)
    assert result.detection is not None
    assert result.added_pixel_count > 150
    assert result.detection.mask[150, 219]
    assert result.detection.class_id == seed.class_id
    assert result.detection.confidence == 0.8
    old = estimate_silhouette_contact_proxy(seed, timestamp_seconds=1)
    new = estimate_silhouette_contact_proxy(result.detection, timestamp_seconds=1)
    assert new.value > old.value + 0.3
    assert result.source == "rgb_seed_color_connected_outline"


def test_round_rgb_does_not_invent_a_tip_or_modify_the_input():
    rgb, seed = fixture()
    before = seed.mask.copy()
    result = refine(rgb, seed)
    assert result.added_pixel_count == 0
    np.testing.assert_array_equal(result.detection.mask, before)
    np.testing.assert_array_equal(seed.mask, before)


def test_disconnected_same_color_neighbor_is_not_joined():
    rgb, seed = fixture()
    rgb[145:155, 204:212] = (30, 110, 220)
    result = refine(rgb, seed)
    assert result.detection is not None
    assert not result.detection.mask[150, 207]


@pytest.mark.parametrize("case", ["gray", "mixed", "flood", "tiny", "domain"])
def test_unsupported_domains_are_unavailable_without_low_outline_fallback(case):
    rgb, seed = fixture()
    if case == "gray":
        rgb[seed.mask] = 150
    elif case == "mixed":
        rgb[150:191][seed.mask[150:191]] = (220, 40, 20)
    elif case == "flood":
        rgb[:] = (30, 110, 220)
    elif case == "tiny":
        mask = np.zeros(seed.mask.shape, bool)
        mask[145:150, 145:150] = True
        seed = replace(seed, mask=mask)
    result = refine_seed_color_outline(
        rgb, seed, uniform_color_domain_validated=case != "domain"
    )
    assert result.detection is None
    assert result.rejection_reasons


def test_red_hue_wrap_is_treated_circularly():
    rgb, seed = fixture()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[seed.mask, 0] = 179
    hsv[150:191, :, 0][seed.mask[150:191]] = 1
    result = refine(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), seed)
    assert result.detection is not None
    assert result.seed_color_fraction == 1


@pytest.mark.parametrize(
    "change",
    [
        {"hue_tolerance": 90},
        {"minimum_saturation": 256},
        {"minimum_seed_pixels": True},
        {"box_margin_fraction": 0},
        {"maximum_growth_fraction": 1.1},
        {"confidence_cap": float("nan")},
    ],
)
def test_invalid_configuration_rejected(change):
    with pytest.raises(ValueError):
        SeedColorOutlineConfig(**change)


def test_rgb_shape_and_dtype_are_checked():
    rgb, seed = fixture()
    for invalid in (rgb.astype(float), rgb[:100]):
        with pytest.raises(ValueError):
            refine(invalid, seed)
