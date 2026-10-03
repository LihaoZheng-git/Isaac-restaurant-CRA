import numpy as np
import pytest

from robot.perception.rgb_plume import StationaryRgbPlumeEstimator
from robot.perception.semantic_detection import PixelBoundingBox


def estimator():
    return StationaryRgbPlumeEstimator(np.zeros((120, 160, 3), dtype=np.uint8),
                                       source_box=PixelBoundingBox(60, 80, 100, 110))


def test_rgb_changes_make_masks_without_oracle_segmentation():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[30:80, 70:90] = 140
    result = estimator().estimate(image, source_frame_id=1, timestamp_seconds=1)
    assert result.frame.observed_plume_count == 1
    assert result.frame.instances[0].mask_pixel_count == 1000
    assert result.spatial_association.value == 1
    assert result.frame.instances[0].plume_label == "visible_plume"


def test_empty_frame_is_invoked_but_not_automatically_valid_negative():
    result = estimator().estimate(np.zeros((120, 160, 3), dtype=np.uint8), source_frame_id=0, timestamp_seconds=0)
    assert result.frame.observed_plume_count == 0
    assert not result.spatial_association.is_available


def test_remote_plume_and_colored_foreground_do_not_match_source():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[20:40, 0:10] = 140
    image[30:80, 70:90, 0] = 200
    assert estimator().estimate(image, source_frame_id=0, timestamp_seconds=0).frame.observed_plume_count == 0


def test_global_camera_or_lighting_change_rejects_reference():
    result = estimator().estimate(np.full((120, 160, 3), 80, dtype=np.uint8), source_frame_id=1, timestamp_seconds=1)
    assert result.frame is None
    assert result.rejection_reasons


def test_bad_image_contract_is_rejected():
    with pytest.raises(ValueError, match="uint8"):
        estimator().estimate(np.zeros((120, 160, 3)), source_frame_id=0, timestamp_seconds=0)
