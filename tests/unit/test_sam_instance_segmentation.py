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
        self.boxes = []

    def set_image(self, image):
        self.image = image.copy()

    def predict(self, point, *, context_box_xyxy=None):
        self.prompts.append(point)
        self.boxes.append(context_box_xyxy)
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


def test_alternate_prompt_must_be_an_observed_return_of_the_same_track(tmp_path):
    segmenter, backend, projection, ids = fixture(tmp_path)
    result = segmenter.propose(ids, projection, track_id=7, prompt_source_index=0)
    assert result.prompt_xy == tuple(projection.pixel_indices_xy[0])
    assert result.detection.confidence == 0.75
    ids[0] = 8
    with pytest.raises(ValueError, match="owned return"):
        segmenter.propose(ids, projection, track_id=7, prompt_source_index=0)
    with pytest.raises(ValueError, match="owned return"):
        segmenter.propose(ids, projection, track_id=7, prompt_source_index=1000)


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


def test_rejected_mask_diagnostics_preserve_quality_and_association_metrics(tmp_path):
    segmenter, backend, projection, ids = fixture(tmp_path)
    backend.scores[:] = 0.7
    result = segmenter.propose(ids, projection, track_id=7)
    assert result.detection is None
    detail = result.candidate_diagnostics[0]
    assert detail["model_quality"] == 0.7
    assert detail["track_coverage"] == 1
    assert detail["foreign_hit_fraction"] == 0
    assert not detail["touches_image_boundary"]
    assert detail["contains_observed_prompt"]
    # A useful diagnostic must not convert a rejected mask into evidence.
    assert result.model_quality is None
    assert result.mask_track_hits == 0


def test_context_box_requires_opt_in_and_uses_only_owned_projected_pixels(tmp_path):
    segmenter, backend, projection, ids = fixture(tmp_path)
    with pytest.raises(ValueError, match="opt-in"):
        segmenter.propose(ids, projection, track_id=7, use_observed_context_box=True)
    segmenter.config = replace(segmenter.config, observed_context_box_retry=True)
    ids[-8:] = 8
    pixels = projection.pixel_indices_xy.copy()
    pixels[-8:] = [90, 90]
    projection = replace(projection, pixel_indices_xy=pixels)
    result = segmenter.propose(ids, projection, track_id=7, use_observed_context_box=True)
    assert result.detection is not None
    assert result.observed_context_box_xyxy == (22, 23, 46, 44)
    assert backend.boxes == [result.observed_context_box_xyxy]
    assert result.detection.semantic_class == "unknown"
    assert result.detection.confidence == .75
    # Ordinary point proposals retain their original backend contract.
    segmenter.propose(ids, projection, track_id=7)
    assert backend.boxes[-1] is None


@pytest.mark.parametrize("failure", ["quality", "foreign", "boundary", "ambiguous"])
def test_context_box_cannot_bypass_mask_acceptance(tmp_path, failure):
    segmenter, backend, projection, ids = fixture(tmp_path)
    segmenter.config = replace(segmenter.config, observed_context_box_retry=True)
    if failure == "quality":
        backend.scores[:] = .84
    elif failure == "foreign":
        ids[:20] = 8
    elif failure == "boundary":
        backend.masks[0, 0:40, 20:70] = True
    else:
        backend.masks = np.concatenate([backend.masks, backend.masks.copy()])
        backend.masks[1, :, 50:] = False
        backend.scores = np.array([.98, .97])
    result = segmenter.propose(ids, projection, track_id=7, use_observed_context_box=True)
    assert result.detection is None and result.rejection_reasons
    assert result.observed_context_box_xyxy is not None


def test_observed_context_box_clamps_to_image_without_inventing_ownership(tmp_path):
    segmenter, backend, projection, ids = fixture(tmp_path)
    segmenter.config = replace(segmenter.config, observed_context_box_retry=True)
    pixels = projection.pixel_indices_xy.copy()
    pixels[:32] = [1, 1]
    pixels[32:] = [98, 98]
    projection = replace(projection, pixel_indices_xy=pixels)
    result = segmenter.propose(ids, projection, track_id=7, use_observed_context_box=True)
    assert backend.boxes == [(0, 0, 100, 100)]
    assert result.detection is None  # The supplied mask contains neither return.


def extent_fixture(tmp_path, *, enabled=True):
    original = np.zeros((1, 100, 100), bool)
    original[0, 20:70, 30:35] = True  # Accepted part, 40 of 64 owned returns.
    expanded = np.zeros_like(original)
    expanded[0, 20:70, 20:70] = True
    segmenter, backend, projection, ids = fixture(tmp_path, original, [.91])
    segmenter.config = replace(segmenter.config, observed_context_box_retry=True,
                               track_extent_retry=enabled)
    backend.expanded, backend.expanded_scores = expanded, np.array([.90])

    def predict(point, *, context_box_xyxy=None):
        backend.prompts.append(point)
        backend.boxes.append(context_box_xyxy)
        return ((backend.masks, backend.scores) if context_box_xyxy is None
                else (backend.expanded, backend.expanded_scores))

    backend.predict = predict
    return segmenter, backend, projection, ids


def test_track_extent_retry_is_opt_in_and_replaces_only_with_checked_superset(tmp_path):
    segmenter, backend, projection, ids = extent_fixture(tmp_path, enabled=False)
    old = segmenter.propose(ids, projection, track_id=7)
    assert old.mask_track_hits == 40 and len(backend.prompts) == 1
    assert old.extent_retry_diagnostics is None
    segmenter.config = replace(segmenter.config, track_extent_retry=True)
    result = segmenter.propose(ids, projection, track_id=7)
    assert len(backend.prompts) == 3
    assert result.mask_track_hits == result.projected_track_points == 64
    assert np.all(result.detection.mask[old.detection.mask])
    assert result.detection.confidence == .75
    assert result.detection.semantic_class == "unknown"
    assert result.extent_retry_diagnostics["promoted"]
    assert result.extent_retry_diagnostics["original_track_coverage"] == .625
    assert result.extent_retry_diagnostics["original_mask_containment"] == 1
    # The box comes from current owned returns, not dimensions or class labels.
    assert backend.boxes[-1] == (28, 28, 40, 40)


@pytest.mark.parametrize("failure", ["quality", "foreign", "boundary", "ambiguous",
                                     "coverage", "containment"])
def test_track_extent_retry_cannot_weaken_acceptance_or_replace_different_region(tmp_path, failure):
    segmenter, backend, projection, ids = extent_fixture(tmp_path)
    if failure == "quality":
        backend.expanded_scores[:] = .84
    elif failure == "foreign":
        pixels = np.vstack([projection.pixel_indices_xy, np.tile([60, 60], (30, 1))])
        projection = ProjectedLidarPoints(np.arange(94), pixels.astype(float), pixels, np.ones(94))
        ids = np.r_[ids, np.full(30, 8)]
    elif failure == "boundary":
        backend.expanded[0, :30, 30] = True
    elif failure == "ambiguous":
        backend.expanded = np.concatenate([backend.expanded, backend.expanded.copy()])
        backend.expanded[1, :, 50:] = False
        backend.expanded_scores = np.array([.90, .91])
    elif failure == "coverage":
        backend.expanded[0, :, 35:] = False
    else:
        backend.expanded[0, 20:30, 30:35] = False
    result = segmenter.propose(ids, projection, track_id=7)
    assert result.detection is not None  # Original valid part survives failed retry.
    np.testing.assert_array_equal(result.detection.mask, backend.masks[0])
    assert not result.extent_retry_diagnostics["promoted"]
    assert result.mask_track_hits == 40
    assert len(backend.prompts) == 2


@pytest.mark.parametrize("mode", ["complete", "alternate", "box", "rejected"])
def test_extent_retry_never_recurses_or_retries_an_unaccepted_mask(tmp_path, mode):
    segmenter, backend, projection, ids = extent_fixture(tmp_path)
    kwargs = {}
    if mode == "complete":
        backend.masks = backend.expanded
    elif mode == "alternate":
        kwargs["prompt_source_index"] = 0
    elif mode == "box":
        kwargs["use_observed_context_box"] = True
    else:
        backend.scores[:] = .84
    result = segmenter.propose(ids, projection, track_id=7, **kwargs)
    assert len(backend.prompts) == 1
    assert result.extent_retry_diagnostics is None


def test_extent_retry_requires_explicit_boolean_and_context_permission(tmp_path):
    segmenter, _, _, _ = fixture(tmp_path)
    with pytest.raises(TypeError, match="boolean"):
        replace(segmenter.config, track_extent_retry=1)
    with pytest.raises(ValueError, match="context-box"):
        replace(segmenter.config, track_extent_retry=True)


def test_image_proposals_are_unassociated_and_preserve_quality_boundary_gates(tmp_path):
    segmenter, backend, projection, ids = fixture(tmp_path)
    invalid = backend.masks[0].copy()
    invalid[0, 25] = True
    backend.generate = lambda: (np.array([backend.masks[0], invalid, backend.masks[0]]), np.array([.94, .99, .7]))
    proposals = segmenter.image_proposals(ids, projection)
    assert proposals and all(p.detection.semantic_class == "unknown" for p in proposals)
    assert all(p.model_quality >= .85 and not p.detection.mask[0].any() for p in proposals)
    assert all(p.mask_track_hits == 0 and p.projected_track_points == 0 for p in proposals)
    assert any(p.candidate_diagnostics[0]["basis"] == "automatic_image_grid" for p in proposals)


def test_image_proposals_reject_backend_shape_mismatch(tmp_path):
    segmenter, backend, projection, ids = fixture(tmp_path)
    backend.generate = lambda: (np.zeros((1, 10, 10)), np.array([.99]))
    with pytest.raises(ValueError, match="shape mismatch"):
        segmenter.image_proposals(ids, projection)
