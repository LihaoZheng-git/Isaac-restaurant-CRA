from dataclasses import replace

import cv2
import numpy as np
import pytest

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.moving_camera_plume import (
    MovingCameraObjectPlumeEstimator,
    TrackedRgbObjectFrame,
    register_background_pair,
)
from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)


def observed(index, *, plume=False, camera_motion=True, texture=True):
    rng = np.random.default_rng(8)
    image = (
        rng.integers(60, 125, size=(240, 320, 3), dtype=np.uint8)
        if texture
        else np.full((240, 320, 3), 100, np.uint8)
    )
    mask = np.zeros((240, 320), np.uint8)
    cv2.ellipse(mask, (160, 175), (35, 32), 0, 0, 360, 1, -1)
    image[mask.astype(bool)] = [20, 60, 180]
    if plume:
        cv2.circle(image, (160, 115), 14, (235, 235, 235), -1)
        cv2.circle(image, (160, 89), 11, (225, 225, 225), -1)
    transform = np.array(
        [
            [1.0, 0.0, 2 * index if camera_motion else 0.0],
            [0.0, 1.0, index if camera_motion else 0.0],
        ],
        np.float32,
    )
    image = cv2.warpAffine(image, transform, (320, 240), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, transform, (320, 240)).astype(bool)
    yy, xx = np.where(mask)
    detection = SemanticDetection(
        0,
        "unknown",
        0.75,
        PixelBoundingBox(
            float(xx.min()), float(yy.min()), float(xx.max() + 1), float(yy.max() + 1)
        ),
        mask,
    )
    frame = SemanticDetectionFrame(
        index, 1 + index * 0.1, 240, 320, (detection,), "test_instance", 0.01
    )
    link = TrackSemanticAssociation(
        4, "matched", 30, 0, 28, 0.93, 0.75, True, (float(xx.mean()), float(yy.mean()))
    )
    return TrackedRgbObjectFrame(image, frame, link, frame.source_timestamp_seconds)


def score(
    *, plume, camera_motion=True, negative_scope="synthetic registered-view domain"
):
    estimator = MovingCameraObjectPlumeEstimator()
    estimator.set_reference(
        observed(0, camera_motion=camera_motion), clean_reference_validated=True
    )
    for i in range(1, 5):
        result = estimator.update(
            observed(i, plume=plume, camera_motion=camera_motion),
            negative_validation_scope=negative_scope,
        )
    return estimator, result


def test_moving_camera_no_plume_is_low_only_in_a_validated_negative_domain():
    estimator, result = score(plume=False)
    assert result.thermal.heat_source_likelihood.is_available, result.rejection_reasons
    assert result.thermal.heat_source_likelihood.value == pytest.approx(0.0)
    assert estimator.last_diagnostics[4]["registered_source_iou"] > 0.9
    _, invalid = score(plume=False, negative_scope=None)
    assert not invalid.thermal.heat_source_likelihood.is_available


@pytest.mark.parametrize("camera_motion", [False, True])
def test_static_visible_plume_is_retained_after_registration(camera_motion):
    _, low = score(plume=False, camera_motion=camera_motion)
    estimator, high = score(plume=True, camera_motion=camera_motion)
    assert high.thermal.heat_source_likelihood.is_available, high.rejection_reasons
    assert (
        high.thermal.heat_source_likelihood.value
        > low.thermal.heat_source_likelihood.value + 0.1
    )
    assert high.positive_frames == 3
    assert estimator.last_diagnostics[4]["plume_pixels"] > 100
    assert not high.thermal.steam_likelihood.is_available


def test_first_frame_is_not_automatically_a_clean_reference():
    estimator = MovingCameraObjectPlumeEstimator()
    unknown = estimator.update(observed(0, plume=True))
    assert not unknown.thermal.heat_source_likelihood.is_available
    with pytest.raises(ValueError, match="validated"):
        estimator.set_reference(observed(0), clean_reference_validated=False)


def test_poor_background_registration_is_unknown_instead_of_negative():
    result = register_background_pair(
        observed(0, texture=False), observed(1, texture=False)
    )
    assert result.homography is None
    assert "background" in result.rejection_reasons[0]


@pytest.mark.parametrize(
    "case", ["stale_track", "different_track", "unmatched", "repeated", "future_track"]
)
def test_invalid_identity_or_epoch_cannot_produce_plume_evidence(case):
    left, right = observed(0), observed(1)
    if case == "stale_track":
        right = replace(right, track_timestamp_seconds=0.1)
    elif case == "different_track":
        right = replace(right, association=replace(right.association, track_id=9))
    elif case == "unmatched":
        right = replace(
            right, association=replace(right.association, status="ambiguous")
        )
    elif case == "repeated":
        right = left
    elif case == "future_track":
        right = replace(right, track_timestamp_seconds=2.0)
    assert register_background_pair(left, right).homography is None


def test_lighting_change_rejects_reference_subtraction():
    left, right = observed(0), observed(1)
    brighter = np.clip(right.rgb.astype(int) + 40, 0, 255).astype(np.uint8)
    result = register_background_pair(left, replace(right, rgb=brighter))
    assert result.homography is None


def test_missing_air_view_is_unknown_instead_of_low_plume():
    def high_source(index):
        item = observed(index)
        mask = np.zeros((240, 320), np.uint8)
        cv2.ellipse(mask, (160, 45), (35, 32), 0, 0, 360, 1, -1)
        detection = replace(
            item.detection,
            mask=mask.astype(bool),
            bounding_box=PixelBoundingBox(125, 13, 196, 78),
        )
        return replace(
            item, detections=replace(item.detections, detections=(detection,))
        )

    result = register_background_pair(high_source(0), high_source(1))
    assert result.homography is None
    assert "air region" in result.rejection_reasons[0]


def test_duplicate_update_cannot_fake_persistence():
    estimator = MovingCameraObjectPlumeEstimator()
    estimator.set_reference(observed(0), clean_reference_validated=True)
    first = observed(1, plume=True)
    estimator.update(first)
    result = estimator.update(first)
    assert not result.thermal.heat_source_likelihood.is_available
    assert "repeated" in result.rejection_reasons[0]


def test_multiple_tracks_cannot_claim_the_same_instance():
    estimator = MovingCameraObjectPlumeEstimator()
    a = observed(0)
    b = replace(a, association=replace(a.association, track_id=9))
    estimator.set_reference(a, clean_reference_validated=True)
    estimator.set_reference(b, clean_reference_validated=True)
    for i in range(1, 5):
        a = observed(i, plume=True)
        b = replace(a, association=replace(a.association, track_id=9))
        rows = estimator.update_many(
            (a, b), negative_validation_scope="synthetic domain"
        )
    assert not rows[4].thermal.heat_source_likelihood.is_available
    assert not rows[9].thermal.heat_source_likelihood.is_available
    assert "multiple tracks" in rows[4].rejection_reasons[0]


def test_plume_batch_rejects_mixed_epochs():
    with pytest.raises(ValueError, match="same RGB epoch"):
        MovingCameraObjectPlumeEstimator().update_many(
            (
                observed(0),
                replace(
                    observed(1),
                    association=replace(observed(1).association, track_id=9),
                ),
            )
        )


def test_clean_reference_has_a_declared_age_limit():
    from robot.perception.moving_camera_plume import MovingCameraPlumeConfig

    estimator = MovingCameraObjectPlumeEstimator(
        MovingCameraPlumeConfig(maximum_reference_age_seconds=0.2)
    )
    estimator.set_reference(observed(0), clean_reference_validated=True)
    result = estimator.update(observed(3, plume=True))
    assert not result.thermal.heat_source_likelihood.is_available
    assert "temporal domain" in result.rejection_reasons[0]


def test_identical_frame_numbers_do_not_allow_different_images_in_one_batch():
    first = observed(0)
    rgb = first.rgb.copy()
    rgb[0, 0] = 255
    second = replace(first, rgb=rgb, association=replace(first.association, track_id=9))
    with pytest.raises(ValueError, match="same camera image"):
        MovingCameraObjectPlumeEstimator().update_many((first, second))
