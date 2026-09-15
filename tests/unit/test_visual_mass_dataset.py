import json

import numpy as np
import pytest

from robot.perception.visual_mass_collection import (
    VisualMassCapture,
    persist_visual_mass_capture,
)
from robot.perception.visual_mass_dataset import (
    VisualMassSample,
    deterministic_object_split,
    evaluate_mass_predictions,
    read_visual_mass_manifest,
    validate_visual_mass_dataset,
    write_visual_mass_manifest,
)


def sample(**changes):
    values = dict(
        sample_id="chair-1-view-1",
        object_id="chair-1",
        semantic_class="chair",
        scene_id="controlled",
        split=deterministic_object_split("chair-1"),
        rgb_path="rgb/a.npy",
        mask_path="masks/a.npy",
        mass_kg=8,
        mass_source="isaac_authored_mass",
        width_m=0.5,
        depth_m=0.5,
        height_m=1,
        camera_id="front_hawk_left",
        simulation_seed=7,
    )
    values.update(changes)
    return VisualMassSample(**values)


def test_object_split_is_stable_and_keeps_views_together():
    assert deterministic_object_split("chair-1") == deterministic_object_split(
        "chair-1"
    )


def test_manifest_round_trip_is_deterministic(tmp_path):
    path = tmp_path / "manifest.jsonl"
    write_visual_mass_manifest([sample()], path)
    assert read_visual_mass_manifest(path) == (sample(),)
    assert json.loads(path.read_text().splitlines()[0])["schema_version"] == 1


def test_validation_detects_missing_files_and_object_leak(tmp_path):
    report = validate_visual_mass_dataset(
        [sample(split="train"), sample(sample_id="two", split="test")], tmp_path
    )
    assert report.object_split_leaks == ("chair-1",)
    assert report.missing_files and not report.valid


def test_capture_persists_rgb_mask_and_auditable_label(tmp_path):
    capture = VisualMassCapture(
        "bowl-1-view-1",
        "bowl-1",
        "bowl",
        "restaurant",
        "train",
        1.2,
        "measured_scale_mass",
        (0.2, 0.2, 0.1),
        "front_hawk_left",
        3,
        np.zeros((4, 5, 3), dtype=np.uint8),
        np.ones((4, 5), dtype=np.bool_),
    )
    item = persist_visual_mass_capture(capture, tmp_path)
    assert np.load(tmp_path / item.rgb_path, allow_pickle=False).shape == (4, 5, 3)
    assert validate_visual_mass_dataset([item], tmp_path).valid
    with pytest.raises(FileExistsError):
        persist_visual_mass_capture(capture, tmp_path)


def test_bad_capture_is_rejected(tmp_path):
    capture = VisualMassCapture(
        "x",
        "x",
        "chair",
        "s",
        "train",
        1,
        "isaac_authored_mass",
        (1, 1, 1),
        "c",
        0,
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.ones((2, 2), dtype=np.bool_),
    )
    with pytest.raises(ValueError):
        persist_visual_mass_capture(capture, tmp_path)


def test_metrics_preserve_truth_prediction_pairing():
    metrics = evaluate_mass_predictions([1, 10], [2, 8])
    assert metrics.mean_absolute_error_kg == 1.5
    assert metrics.median_absolute_error_kg == 1.5
    assert metrics.mean_absolute_percentage_error == pytest.approx(0.6)


@pytest.mark.parametrize(
    "change",
    [
        {"mass_kg": 0},
        {"mass_source": "semantic_prior"},
        {"rgb_path": "../escape.npy"},
        {"split": "random"},
        {"simulation_seed": -1},
    ],
)
def test_invalid_sample_rejected(change):
    with pytest.raises((TypeError, ValueError)):
        sample(**change)
