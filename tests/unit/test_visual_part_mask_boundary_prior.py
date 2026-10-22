from dataclasses import replace

import numpy as np
import pytest

from test_visual_body_fragments import dense_fixture, associate, observe
from robot.perception.visual_body_fragments import associate_visual_body_fragments
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection
from robot.evaluation.nearby_track_loss_guard import NearbyTrackLossConfig, NearbyTrackLossGuard


def boundary_fixture():
    args = dense_fixture(400, 37)
    args[2].pixel_indices_xy[-1] = [4, 20]
    return args


def with_boundary(args, pixels=1):
    return associate_visual_body_fragments(*args, camera_timestamp_seconds=1.05,
        camera_frame_id=4, mask_boundary_tolerance_pixels=pixels)


def test_one_native_pixel_with_strong_original_mask_support_is_labeled_prior():
    args = boundary_fixture()
    points, mask = args[1].copy(), args[3].detections[0].mask.copy()
    assert associate(args) == []
    proof, = with_boundary(args)
    assert proof["status"] == "prior"
    assert proof["child_original_mask_coverage"] == 36/37
    assert proof["child_boundary_return_count"] == 1
    assert proof["mask_boundary_prior_pixels"] == 1
    assert proof["original_person_camera_timestamp"] == .8
    assert proof["source_timestamp_seconds"] == 1.
    np.testing.assert_array_equal(args[1], points)
    np.testing.assert_array_equal(args[3].detections[0].mask, mask)


@pytest.mark.parametrize("problem", ["two_pixels", "diagonal", "weak_original", "unprojected", "competing_boundary"])
def test_boundary_prior_never_accepts_larger_missing_or_competing_regions(problem):
    args = boundary_fixture()
    projection, frame = args[2], args[3]
    if problem == "two_pixels":
        projection.pixel_indices_xy[-1] = [3, 20]
    elif problem == "diagonal":
        projection.pixel_indices_xy[-1] = [4, 4]
    elif problem == "weak_original":
        projection.pixel_indices_xy[-2] = [4, 20]
    elif problem == "unprojected":
        projection.source_indices = projection.source_indices[:-1]
        projection.pixel_indices_xy = projection.pixel_indices_xy[:-1]
    else:
        mask = np.zeros((40, 40), bool)
        mask[20, 3] = True
        other = SemanticDetection(0, "person", .8, PixelBoundingBox(3, 20, 4, 21), mask)
        args[3] = replace(frame, detections=frame.detections+(other,))
    assert with_boundary(args) == []


@pytest.mark.parametrize("value", [-1, 2, True, 1.])
def test_invalid_boundary_prior_rejected(value):
    with pytest.raises(ValueError, match="native pixel"):
        with_boundary(boundary_fixture(), value)


def test_boundary_proof_requires_guard_opt_in_and_keeps_parent_loss_watch():
    args = boundary_fixture()
    proof, = with_boundary(args)
    strict = NearbyTrackLossGuard(NearbyTrackLossConfig(allow_current_visual_part_associations=True))
    with pytest.raises(ValueError, match="invalid current positive"):
        observe(strict, args[0], args[1], [proof])
    guard = NearbyTrackLossGuard(NearbyTrackLossConfig(
        allow_current_visual_part_associations=True, maximum_visual_mask_boundary_pixels=1))
    observe(guard, args[0], args[1], [proof])
    assert set(guard.observed) == {1}
    args[0].timestamp_seconds = 1.2
    args[0].tracked_obstacles = ()
    args[0].point_track_ids = np.empty(0, int)
    assert observe(guard, args[0], np.empty((0, 3)))["blocked"]


def test_boundary_guard_configuration_requires_explicit_visual_part_mode():
    with pytest.raises(ValueError, match="explicit current visual"):
        NearbyTrackLossConfig(maximum_visual_mask_boundary_pixels=1)
