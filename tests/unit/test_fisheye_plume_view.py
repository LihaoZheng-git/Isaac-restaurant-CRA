from dataclasses import replace

import numpy as np
import pytest
from test_moving_camera_plume import observed

from robot.perception.camera_lidar_fusion import (
    FThetaCameraCalibration,
    project_fisheye_polynomial_points,
)
from robot.perception.fisheye_plume_view import pinhole_remap, rectify_plume_observation
from robot.perception.moving_camera_plume import register_background_pair


def calibration():
    return FThetaCameraCalibration(
        320, 240, 160.0, 120.0, 180.0, (0.0, 0.01, 0.0, 0.0, 0.0), 320, 240
    )


def test_rectification_maps_match_existing_calibrated_ray_projection():
    config = calibration()
    mx, my, valid = pinhole_remap(config)
    rows, columns = np.indices(valid.shape)
    focal = 160 / np.tan(np.deg2rad(70))
    rays = np.stack(
        ((columns - 160) / focal, -(rows - 120) / focal, -np.ones_like(rows)), axis=-1
    ).reshape(-1, 3)
    projected = project_fisheye_polynomial_points(
        rays, camera_from_source=np.eye(4), calibration=config
    )
    expected = np.stack((mx, my), axis=-1).reshape(-1, 2)[projected.source_indices]
    np.testing.assert_allclose(projected.pixel_xy, expected, atol=2e-5)
    assert not mx.flags.writeable and not valid.flags.writeable


def test_rectified_source_keeps_identity_confidence_and_both_epochs():
    original = observed(1)
    changed = rectify_plume_observation(original, calibration())
    assert changed.detections.source_frame_id == original.detections.source_frame_id
    assert (
        changed.detections.source_timestamp_seconds
        == original.detections.source_timestamp_seconds
    )
    assert changed.track_timestamp_seconds == original.track_timestamp_seconds
    assert changed.association is original.association
    assert changed.detection.confidence == original.detection.confidence
    assert changed.detection.semantic_class == "unknown"
    assert changed.detection.mask.sum() > 200
    assert not (changed.detection.mask & ~changed.valid_pixel_mask).any()
    with pytest.raises(ValueError):
        pinhole_remap(calibration(), 180.0)


def test_invalid_remap_pixels_cannot_support_background_registration_or_negative_plume():
    old = observed(0)
    current = observed(1)
    old = replace(old, valid_pixel_mask=np.zeros(old.rgb.shape[:2], bool))
    result = register_background_pair(old, current)
    assert result.homography is None
    assert "background features" in result.rejection_reasons[0]
