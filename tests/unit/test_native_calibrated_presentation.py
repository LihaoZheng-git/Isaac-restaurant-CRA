"""Native calibrated presentation changes colors/text, never recorded policy."""

import numpy as np
import pytest

from robot.evaluation.risk_camera_overlay import (
    camera_metadata_note,
    repair_camera_metadata_notes,
)
from scripts.evaluation.compose_restaurant_lidar_risk import presentation_options


def test_original_legacy_presentation_defaults_remain_available():
    value = presentation_options()
    assert value == dict(
        palette="legacy",
        title="Restaurant service | v4",
        video_name="restaurant-service-v4-lidar-risk.mp4",
        repair_camera_notes=False,
    )
    replay = presentation_options(replay=True)
    assert replay["palette"] == "restaurant_v1" and "replay" in replay["video_name"]


def test_native_calibrated_options_are_not_score_replay():
    value = presentation_options(
        palette="restaurant_v1",
        title="Restaurant service | calibrated V3",
        video_name="restaurant-service-v3-calibrated.mp4",
        repair_camera_notes=True,
    )
    assert value["palette"] == "restaurant_v1" and value["repair_camera_notes"]
    assert "replay" not in value["title"] and "replay" not in value["video_name"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(palette="per_object"),
        dict(title=""),
        dict(title="one\ntwo"),
        dict(video_name="../overwrite.mp4"),
        dict(video_name="C:/overwrite.mp4"),
        dict(video_name="sub\\video.mp4"),
        dict(video_name="video.exe"),
        dict(replay=True, repair_camera_notes=True),
    ],
)
def test_invalid_native_presentation_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        presentation_options(**kwargs)


def test_metadata_note_combines_both_counts_into_one_line():
    assert camera_metadata_note(2, 0) is None
    assert camera_metadata_note(3, 0) == "+ 1 masks; see ledger"
    assert camera_metadata_note(1, 2) == "2 records withheld; see ledger"
    assert camera_metadata_note(5, 2) == "+ 3 masks; 2 records withheld; see ledger"


def test_metadata_repair_is_confined_to_exact_strip_and_preserves_source_pixels():
    panel = np.random.default_rng(3).integers(
        0, 256, size=(600, 640, 3), dtype=np.uint8
    )
    before = panel.copy()
    epoch = dict(
        status="latest_assessed_frame",
        sensor_frame_id=4,
        camera_frame_id=9,
        masks=[{}, {}, {}, {}],
        rejected=[{}],
    )
    corrected, audit = repair_camera_metadata_notes(panel, epoch)
    np.testing.assert_array_equal(panel, before)
    np.testing.assert_array_equal(corrected[:502], before[:502])
    np.testing.assert_array_equal(corrected[525:], before[525:])
    assert not np.array_equal(corrected[502:525], before[502:525])
    assert audit["changed_row_range"] == [502, 525]
    assert audit["mask_count"] == 4 and audit["rejected_count"] == 1
    assert audit["sensor_frame_id"] == 4 and audit["camera_frame_id"] == 9
    assert audit["note"] == "+ 2 masks; 1 records withheld; see ledger"


def test_no_note_clears_only_stale_metadata_without_changing_mask_or_score_rows():
    panel = np.full((600, 640, 3), 255, np.uint8)
    corrected, audit = repair_camera_metadata_notes(
        panel, dict(status="latest_assessed_frame", masks=[], rejected=[])
    )
    assert audit["note"] is None
    assert np.all(corrected[502:525] == np.array([22, 27, 34]))
    assert np.all(corrected[:502] == 255) and np.all(corrected[525:] == 255)


@pytest.mark.parametrize(
    "panel", [np.zeros((600, 640, 4), np.uint8), np.zeros((600, 640, 3), float)]
)
def test_metadata_repair_rejects_non_native_panel(panel):
    with pytest.raises(ValueError, match="native camera panel"):
        repair_camera_metadata_notes(
            panel, dict(status="latest_assessed_frame", masks=[])
        )
