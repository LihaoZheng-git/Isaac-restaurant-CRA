"""LiDAR-prompted category-free masks with controlled numeric backend outputs."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from robot.perception.camera_lidar_fusion import ProjectedLidarPoints
from robot.perception.instance_lidar_geometry import masked_track_return_selection
from robot.perception.sam_instance_segmentation import (
    SAM_INSTANCE_SOURCE,
    SamInstanceConfig,
    SamInstanceSegmenter,
)


class Backend:
    def __init__(self, masks, scores):
        self.masks, self.scores = masks, scores
        self.prompts = []

    def set_image(self, image):
        self.image = image.copy()

    def predict(self, point):
        self.prompts.append(point)
        return self.masks, self.scores


def fixture(tmp_path, masks=None, scores=None):
    path = tmp_path / "sam2.1_t.pt"
    path.write_bytes(b"numeric-backend-test")
    if masks is None:
        masks = np.zeros((1, 100, 100), bool)
        masks[0, 20:70, 20:70] = True
    backend = Backend(masks, np.asarray([0.98] if scores is None else scores))
    config = SamInstanceConfig(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    segmenter = SamInstanceSegmenter(config, backend=backend)
    segmenter.set_image(np.full((100, 100, 3), 127, np.uint8))
    xy = np.asarray([[30 + i % 8, 30 + i // 8] for i in range(64)])
    projection = ProjectedLidarPoints(np.arange(64), xy.astype(float), xy, np.ones(64))
    return segmenter, backend, projection, np.full(64, 7)


def test_prompt_is_an_observed_return_and_category_stays_unknown(tmp_path):
    segmenter, backend, projection, ids = fixture(tmp_path)
    result = segmenter.propose(ids, projection, track_id=7)
    assert result.detection.semantic_class == "unknown"
    assert result.detection.confidence == 0.75
    assert result.source == SAM_INSTANCE_SOURCE
    assert result.model_quality == 0.98
    assert result.prompt_xy in [tuple(x) for x in projection.pixel_indices_xy]
    assert backend.prompts == [result.prompt_xy]


def test_nonprojected_and_other_track_occupancy_is_retained(tmp_path):
    segmenter, _, projection, ids = fixture(tmp_path)
    ids[0] = 8
    result = segmenter.propose(ids, projection, track_id=7)
    assert result.detection is not None
    ids = np.concatenate([ids, [7, 8]])
    selected = masked_track_return_selection(
        ids, projection, result.detection.mask, track_id=7, margin_pixels=0
    )
    assert not selected[0] and not selected[-2:].any()
    assert selected.sum() == 63


@pytest.mark.parametrize(
    "case", ["quality", "boundary", "foreign", "coverage", "area", "missing_track"]
)
def test_rejected_observations_remain_unavailable(tmp_path, case):
    segmenter, backend, projection, ids = fixture(tmp_path)
    if case == "quality":
        backend.scores[:] = 0.7
    if case == "boundary":
        backend.masks[0, 0:40, 20:70] = True
    if case == "foreign":
        ids[:20] = 8
    if case == "coverage":
        backend.masks[:] = False
        backend.masks[0, 33:60, 32:60] = True
        segmenter.config = replace(segmenter.config, minimum_track_coverage=0.9)
    if case == "area":
        backend.masks[:] = True
    result = segmenter.propose(
        ids, projection, track_id=9 if case == "missing_track" else 7
    )
    assert result.detection is None and result.rejection_reasons


def test_distinct_high_quality_masks_reject_ambiguous_extent(tmp_path):
    masks = np.zeros((2, 100, 100), bool)
    masks[0, 20:70, 20:70] = True
    masks[1, 25:65, 25:50] = True
    segmenter, _, projection, ids = fixture(tmp_path, masks, [0.98, 0.97])
    result = segmenter.propose(ids, projection, track_id=7)
    assert result.detection is None
    assert "ambiguous" in result.rejection_reasons[0]


def test_equivalent_multimasks_do_not_reject_one_instance(tmp_path):
    masks = np.zeros((2, 100, 100), bool)
    masks[0, 20:70, 20:70] = True
    masks[1, 21:70, 20:70] = True
    segmenter, _, projection, ids = fixture(tmp_path, masks, [0.98, 0.97])
    assert segmenter.propose(ids, projection, track_id=7).detection is not None


def test_mismatched_frame_output_cannot_be_used(tmp_path):
    segmenter, backend, projection, ids = fixture(tmp_path)
    backend.masks = np.ones((1, 10, 10), bool)
    with pytest.raises(ValueError, match="source RGB"):
        segmenter.propose(ids, projection, track_id=7)


def test_hash_mismatch_is_rejected_before_backend_initialization(tmp_path):
    path = tmp_path / "sam2.1_t.pt"
    path.write_bytes(b"not-the-release")
    with pytest.raises(ValueError, match="hash"):
        SamInstanceSegmenter(SamInstanceConfig(path))
