"""Camera instance selection retains a separately accountable complement."""

import numpy as np
import pytest

from robot.perception.camera_lidar_fusion import ProjectedLidarPoints
from robot.perception.instance_lidar_geometry import (
    masked_track_return_selection,
    propose_chromatic_instance,
)


def projection():
    return ProjectedLidarPoints(
        np.array([0, 2, 3, 4]),
        np.array([[5, 5], [5, 6], [1, 1], [5, 5]]),
        np.array([[5, 5], [5, 6], [1, 1], [5, 5]]),
        np.ones(4),
    )


def test_only_projected_matched_instance_returns_are_selected():
    mask = np.zeros((10, 10), bool)
    mask[5, 5] = True
    ids = np.array([7, 7, 7, 7, 8, 7])
    selected = masked_track_return_selection(
        ids, projection(), mask, track_id=7, margin_pixels=0
    )
    assert selected.tolist() == [True, False, False, False, False, False]
    # Same-track support, nonprojected and other-track returns stay accounted for.
    assert np.flatnonzero(~selected).tolist() == [1, 2, 3, 4, 5]


def test_configured_registration_margin_is_explicit_and_bounded():
    mask = np.zeros((10, 10), bool)
    mask[5, 5] = True
    selected = masked_track_return_selection(
        np.full(5, 7), projection(), mask, track_id=7, margin_pixels=1
    )
    assert selected.tolist() == [True, False, True, False, True]


def test_missing_mask_hits_cannot_create_geometry():
    selected = masked_track_return_selection(
        np.full(5, 7), projection(), np.zeros((10, 10), bool), track_id=7
    )
    assert not selected.any()


@pytest.mark.parametrize("margin", [-1, True, 1.5])
def test_invalid_margin_rejected(margin):
    with pytest.raises(ValueError, match="margin_pixels"):
        masked_track_return_selection(
            np.full(5, 7),
            projection(),
            np.zeros((10, 10), bool),
            track_id=7,
            margin_pixels=margin,
        )


def test_projection_outside_supplied_mask_is_rejected():
    with pytest.raises(ValueError, match="projection indices"):
        masked_track_return_selection(
            np.full(5, 7), projection(), np.zeros((3, 3), bool), track_id=7
        )


def chromatic_fixture(two=False):
    rgb = np.full((100, 100, 3), 120, np.uint8)
    rgb[20:40, 20:40] = [20, 60, 220]
    xy = np.array([[20 + i % 20, 20 + i // 20] for i in range(80)])
    if two:
        rgb[60:80, 60:80] = [220, 30, 30]
        xy[40:] = [[60 + i % 20, 60 + i // 20] for i in range(40)]
    proj = ProjectedLidarPoints(np.arange(80), xy.astype(float), xy, np.ones(80))
    return rgb, proj


def test_lidar_seeded_rgb_component_is_observed_but_has_unknown_class():
    rgb, proj = chromatic_fixture()
    result = propose_chromatic_instance(
        rgb, np.full(80, 7), proj, track_id=7, separated_chromatic_domain_validated=True
    )
    assert result.detection is not None
    assert result.detection.semantic_class == "unknown"
    assert result.detection.confidence == 0.5
    assert result.detection.mask.sum() == 400
    assert result.source == "lidar_seeded_chromatic_rgb_component"


def test_border_connected_floor_is_background_not_a_bounded_neighbor():
    rgb, proj = chromatic_fixture()
    rgb[40:] = [160, 100, 50]
    result = propose_chromatic_instance(
        rgb, np.full(80, 7), proj, track_id=7, separated_chromatic_domain_validated=True
    )
    assert result.detection is not None and result.detection.mask.sum() == 400


@pytest.mark.parametrize(
    "case", ["domain", "gray", "ambiguous", "boundary", "multicolor", "wrong_track"]
)
def test_unsupported_chromatic_proposals_remain_unavailable(case):
    rgb, proj = chromatic_fixture(two=case == "ambiguous")
    if case == "gray":
        rgb[:] = 120
    if case == "boundary":
        rgb[:21, 20:40] = [20, 60, 220]
    if case == "multicolor":
        rgb[30:40, 20:40] = [220, 30, 30]
    result = propose_chromatic_instance(
        rgb,
        np.full(80, 7),
        proj,
        track_id=8 if case == "wrong_track" else 7,
        separated_chromatic_domain_validated=case != "domain",
    )
    assert result.detection is None and result.rejection_reasons


@pytest.mark.parametrize("large_gap", [False, True])
def test_only_tiny_specular_holes_can_be_inferred_as_external_outline(large_gap):
    rgb, proj = chromatic_fixture()
    rgb[30, 30] = 120
    if large_gap:
        rgb[28:33, 28:33] = 120
    result = propose_chromatic_instance(
        rgb, np.full(80, 7), proj, track_id=7, separated_chromatic_domain_validated=True
    )
    assert (result.detection is None) == large_gap
