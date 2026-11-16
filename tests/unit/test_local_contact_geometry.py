"""Independent smooth, blunt, pointed and noisy boundary controls."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from robot.perception.local_contact_geometry import LocalContactGeometryConfig, local_convex_contact_features
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection
from robot.perception.visual_shape_contact import SilhouetteContactConfig, estimate_silhouette_contact_proxy

LOCAL = SilhouetteContactConfig(local_geometry=LocalContactGeometryConfig())


def mask_fixture(name):
    mask = np.zeros((256, 256), np.uint8)
    if name == "circle":
        cv2.circle(mask, (128, 128), 72, 1, -1)
    elif name == "ellipse":
        cv2.ellipse(mask, (128, 128), (35, 100), 0, 0, 360, 1, -1)
    elif name == "capsule":
        cv2.rectangle(mask, (100, 40), (156, 216), 1, -1)
        cv2.circle(mask, (128, 40), 28, 1, -1)
        cv2.circle(mask, (128, 216), 28, 1, -1)
    elif name == "rectangle":
        cv2.rectangle(mask, (110, 25), (146, 230), 1, -1)
    elif name == "triangle":
        cv2.fillPoly(mask, [np.array([[128, 20], [210, 218], [45, 218]])], 1)
    elif name == "nail":
        cv2.fillPoly(mask, [np.array([[128, 15], [140, 210], [159, 210], [159, 230],
                                    [97, 230], [97, 210], [116, 210]])], 1)
    elif name == "star":
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        radius = np.where(np.arange(16) % 2, 45, 100)
        cv2.fillPoly(mask, [np.column_stack([128+radius*np.cos(angles),
                                           128+radius*np.sin(angles)]).astype(int)], 1)
    else:
        raise ValueError(name)
    return mask.astype(bool)


def detection(mask, name="arbitrary"):
    return SemanticDetection(0, name, .95,
        PixelBoundingBox(0, 0, mask.shape[1], mask.shape[0]), mask)


def score(mask, config=LOCAL):
    return estimate_silhouette_contact_proxy(detection(mask), timestamp_seconds=1, config=config)


@pytest.mark.parametrize("rotation", [0, 27, 90, 143])
@pytest.mark.parametrize("scale", [.5, 1, 2])
def test_smooth_elongation_is_low_and_real_tips_remain_high(rotation, scale):
    values = {}
    for name in ("circle", "ellipse", "capsule", "triangle", "nail", "star"):
        mask = cv2.warpAffine(mask_fixture(name).astype(np.uint8),
            cv2.getRotationMatrix2D((128, 128), rotation, 1), (256, 256), flags=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).astype(bool)
        result = score(mask)
        assert result.is_available
        values[name] = result.value
    assert max(values[n] for n in ("circle", "ellipse", "capsule")) < .2
    assert min(values[n] for n in ("triangle", "nail", "star")) > .5


def test_blunt_corners_are_intermediate_not_elongation_saturation():
    assert score(mask_fixture("circle")).value < score(mask_fixture("rectangle")).value < .6
    assert score(mask_fixture("rectangle")).value < score(mask_fixture("triangle")).value
    # Legacy remains available and unchanged for existing demo/controller holds.
    assert score(mask_fixture("capsule"), None).value > .5
    assert score(mask_fixture("capsule")).value < .1


def test_pixel_spur_does_not_become_a_sharp_tip():
    mask = mask_fixture("circle")
    mask[128, 201] = True
    assert score(mask).value < .1


def test_inward_notch_does_not_win_over_outward_corners_and_winding_is_irrelevant():
    contour = np.array([[20, 20], [90, 20], [100, 90], [110, 20],
                        [180, 20], [180, 180], [20, 180]], dtype=float)
    forward = local_convex_contact_features(contour)
    backward = local_convex_contact_features(contour[::-1])
    assert forward.score == pytest.approx(backward.score, abs=.01)
    assert np.linalg.norm(np.array(forward.peak_pixel_xy) - [100, 90]) > 50
    assert 0 < forward.score < .6
    assert min(forward.signed_turns_degrees[:2]) > 0


def test_degenerate_inward_slit_is_not_flipped_into_a_convex_tip():
    mask = np.zeros((200, 200), np.uint8)
    mask[20:180, 20:180] = 1
    mask[20:110, 100] = 0
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    feature = local_convex_contact_features(contour)
    assert feature.score < .6
    assert np.linalg.norm(np.array(feature.peak_pixel_xy) - [100, 110]) > 50


def test_accepted_part_proxy_is_not_suppressed_by_local_smoothness():
    from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
    from robot.perception.visual_shape_contact import combine_shape_and_part_contact
    shape = score(mask_fixture("capsule"))
    part = ScalarEstimate(.8, .7, .9, .75, "1", "fixture", 1., EstimateStatus.INFERRED, ())
    combined = combine_shape_and_part_contact(shape, part)
    assert combined.value == .8 and combined.upper_bound == .9


@pytest.mark.parametrize("case", ["hole", "fragmented", "border", "tiny", "low_confidence"])
def test_original_mask_rejections_are_preserved(case):
    mask = mask_fixture("circle")
    if case == "hole":
        mask[120:130, 120:130] = False
    elif case == "fragmented":
        mask[120:130] = False
    elif case == "border":
        mask[:128, 125:130] = True
    elif case == "tiny":
        mask[:] = False
        mask[125:130, 125:130] = True
    item = detection(mask)
    if case == "low_confidence":
        item = replace(item, confidence=.1)
    result = estimate_silhouette_contact_proxy(item, timestamp_seconds=1, config=LOCAL)
    assert not result.is_available


def test_label_and_provenance_and_intervals_are_preserved():
    from robot.perception.physical_attributes import EstimateStatus
    mask = mask_fixture("triangle")
    a = estimate_silhouette_contact_proxy(detection(mask, "knife"), timestamp_seconds=4, config=LOCAL)
    b = estimate_silhouette_contact_proxy(detection(mask, "vase"), timestamp_seconds=4, config=LOCAL)
    assert a == b and a.timestamp_seconds == 4 and a.status is EstimateStatus.INFERRED
    assert a.lower_bound < a.value < a.upper_bound
    oracle = estimate_silhouette_contact_proxy(detection(mask), timestamp_seconds=4,
        config=LOCAL, evidence_status=EstimateStatus.ORACLE)
    assert oracle.status is EstimateStatus.ORACLE
    assert any("local_convex_multiscale" in item for item in a.evidence)


@pytest.mark.parametrize("kwargs", [{"sample_count": True}, {"sample_count": 2},
    {"minimum_support_pixels": float('nan')}, {"smoothing_perimeter_fraction": .1},
    {"support_perimeter_fractions": (.1, .01, .02)}, {"turn_full_degrees": 180},
    {"turn_start_degrees": 160}, {"persistence_full": .3}])
def test_invalid_local_parameters_reject(kwargs):
    with pytest.raises(ValueError):
        LocalContactGeometryConfig(**kwargs)


@pytest.mark.parametrize("contour", [[], [[0, 0], [1, 1], [2, 2]],
    [[0, 0], [1, 1], [float('nan'), 0]]])
def test_invalid_contours_reject(contour):
    with pytest.raises(ValueError):
        local_convex_contact_features(contour)
