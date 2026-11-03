"""Adversarial positive-surface grouping and part/whole/source-clock checks."""

from dataclasses import replace

import numpy as np
import pytest

from robot.perception.camera_lidar_fusion import ProjectedLidarPoints
from robot.perception.image_object_association import ImageObjectConfig, StationaryImageObjects, associate_image_objects
from robot.perception.sam_instance_segmentation import SamInstanceResult
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection


def proposal(mask):
    yy, xx = np.where(mask)
    return SamInstanceResult(SemanticDetection(0, "unknown", .75,
        PixelBoundingBox(float(xx.min()), float(yy.min()), float(xx.max()+1), float(yy.max()+1)), mask), (), model_quality=.94)


def fixture():
    mask = np.zeros((80, 80), bool)
    mask[15:65, 15:65] = True
    xy = np.array([[20+i, 30] for i in range(8)] + [[50+i, 30] for i in range(8)])
    projection = ProjectedLidarPoints(np.arange(16), xy.astype(float), xy, np.ones(16)*2)
    xyz = np.column_stack((np.ones(16)*2, np.r_[np.linspace(-.3, -.25, 8), np.linspace(.25, .3, 8)], np.ones(16)*.2))
    ids = np.array([10]*8 + [20]*8)
    return mask, projection, xyz, ids


def test_disconnected_fragments_form_one_hypothesis_without_changing_inputs():
    mask, projection, xyz, ids = fixture()
    original = ids.copy()
    candidates, audit = associate_image_objects([proposal(mask)], projection, ids, xyz, ImageObjectConfig())
    assert len(candidates) == 1 and candidates[0]["selected"].sum() == 16
    assert [f["raw_track_id"] for f in audit[0]["fragments"]] == [10, 20]
    np.testing.assert_array_equal(ids, original)


@pytest.mark.parametrize("case", ["unclustered", "extent", "coverage"])
def test_unknown_ownership_large_geometry_or_partial_fragment_reject(case):
    mask, projection, xyz, ids = fixture()
    if case == "unclustered":
        ids[8:] = -1
    elif case == "extent":
        xyz[8:, 1] += 1
    else:
        mask[:, 53:] = False
    candidates, audit = associate_image_objects([proposal(mask)], projection, ids, xyz, ImageObjectConfig())
    assert not candidates and audit[0]["rejections"]


def test_background_through_holes_and_unprojected_points_cannot_get_object_policy():
    mask, projection, xyz, ids = fixture()
    depths = projection.camera_depths.copy()
    depths[8:] = 5
    projection = replace(projection, camera_depths=depths)
    xyz = np.vstack([xyz, [4, 4, .2]])
    ids = np.r_[ids, 99]
    candidates, _ = associate_image_objects([proposal(mask)], projection, ids, xyz, ImageObjectConfig())
    assert candidates[0]["selected"].sum() == 8
    assert not candidates[0]["selected"][8:].any()


def test_two_disjoint_viable_children_reject_merged_parent():
    mask, projection, xyz, ids = fixture()
    left, right = mask.copy(), mask.copy()
    left[:, 40:] = False
    right[:, :40] = False
    candidates, audit = associate_image_objects([proposal(mask), proposal(left), proposal(right)], projection, ids, xyz, ImageObjectConfig())
    assert len(candidates) == 2
    assert "merged mask" in audit[0]["rejections"][0]
    assert not np.any(candidates[0]["selected"] & candidates[1]["selected"])


def test_part_suppressed_when_larger_unique_mask_owns_both_fragments():
    mask, projection, xyz, ids = fixture()
    part = mask.copy()
    part[:, 40:] = False
    candidates, _ = associate_image_objects([proposal(part), proposal(mask)], projection, ids, xyz, ImageObjectConfig())
    assert len(candidates) == 1 and candidates[0]["selected"].sum() == 16


def test_duplicate_candidate_cannot_revive_a_rejected_nonnested_conflict():
    mask, projection, xyz, ids = fixture()
    a, b = mask.copy(), mask.copy()
    a[45:, :] = False
    b[15:25, :] = False
    candidates, _ = associate_image_objects([proposal(a), proposal(b), proposal(a)], projection, ids, xyz, ImageObjectConfig())
    assert not candidates


def test_repetition_gap_duplicate_and_partial_mask_never_refresh_original_evidence():
    mask, projection, xyz, ids = fixture()
    config = ImageObjectConfig()
    state = StationaryImageObjects(config)
    def observe(m, time):
        candidates, _ = associate_image_objects([proposal(m)], projection, ids, xyz, config)
        return state.update(candidates, xyz, time)
    assert not observe(mask, 1.)
    assert not observe(mask, 1.1)
    accepted = observe(mask, 1.2)
    assert len(accepted) == 1
    key = accepted[0]["object_id"]
    with pytest.raises(ValueError, match="increase"):
        observe(mask, 1.2)
    part = mask.copy()
    part[:, 40:] = False
    for t in (1.3, 1.4, 1.8, 2.0):
        assert not observe(part, t)
    assert state.records[key]["first"] == 1 and state.records[key]["last"] == 1.2
    assert not observe(mask, 2.1)  # broken support sequence requires three NEW sources
    assert not observe(mask, 2.2)
    assert len(observe(mask, 2.3)) == 1
    assert state.records[key]["first"] == 1


def test_raw_id_change_does_not_renew_visual_origin_or_require_identity_transfer():
    mask, projection, xyz, ids = fixture()
    state = StationaryImageObjects(ImageObjectConfig())
    keys = []
    for t in (1., 1.1, 1.2):
        ids += 10
        candidates, _ = associate_image_objects([proposal(mask)], projection, ids, xyz, state.config)
        accepted = state.update(candidates, xyz, t)
        keys.extend(c["object_id"] for c in accepted)
    assert len(keys) == 1 and len(state.records) == 1


def test_gradual_geometric_drift_is_compared_to_original_support():
    mask, projection, xyz, ids = fixture()
    state = StationaryImageObjects(ImageObjectConfig())
    for i in range(3):
        candidates, _ = associate_image_objects([proposal(mask)], projection, ids, xyz, state.config)
        accepted = state.update(candidates, xyz + [i*.04, 0, 0], 1+i*.1)
    assert not accepted


@pytest.mark.parametrize("kwargs", [{"maximum_extent_m": float("nan")}, {"minimum_observations": 2},
                                   {"minimum_mask_iou": 0}, {"maximum_reference_age_seconds": -1}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        ImageObjectConfig(**kwargs)
