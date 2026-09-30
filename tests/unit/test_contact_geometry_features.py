import numpy as np
import pytest

from robot.perception.contact_cues import DangerousPartSegmentation
from robot.perception.contact_geometry_features import (
    estimate_contact_geometry_features,
)
from robot.perception.physical_attributes import EstimateStatus


def segmentation(label, part_mask, confidence=0.8, timestamp=2.0):
    object_mask = np.ones((30, 30), dtype=bool)
    return DangerousPartSegmentation(
        object_mask,
        part_mask,
        label,
        confidence,
        timestamp,
        "geometry-fixture",
    )


def rectangle(y0, y1, x0, x1):
    mask = np.zeros((30, 30), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


@pytest.mark.parametrize(
    "label",
    ["exposed_edge", "pointed_tip", "protrusion", "jagged_surface"],
)
def test_general_taxonomy_produces_visual_descriptors_only(label):
    actual = estimate_contact_geometry_features(
        segmentation(label, rectangle(10, 14, 5, 25)),
        timestamp_seconds=2.0,
    )
    assert actual.geometry_class.value == label
    assert actual.exposure_fraction.is_available
    assert actual.elongation.is_available
    assert actual.boundary_complexity.is_available
    assert not actual.sharpness.is_available
    assert not actual.hardness.is_available
    assert not actual.orientation_toward_route.is_available


def test_long_thin_part_is_more_elongated_than_square_part():
    thin = estimate_contact_geometry_features(
        segmentation("exposed_edge", rectangle(12, 14, 3, 27)),
        timestamp_seconds=2.0,
    )
    square = estimate_contact_geometry_features(
        segmentation("protrusion", rectangle(8, 22, 8, 22)),
        timestamp_seconds=2.0,
    )
    assert thin.elongation.value > square.elongation.value


def test_jagged_boundary_has_greater_complexity_than_compact_square():
    compact_mask = rectangle(6, 24, 6, 24)
    jagged_mask = compact_mask.copy()
    jagged_mask[6:24:2, 6:12] = False
    compact = estimate_contact_geometry_features(
        segmentation("jagged_surface", compact_mask),
        timestamp_seconds=2.0,
    )
    jagged = estimate_contact_geometry_features(
        segmentation("jagged_surface", jagged_mask),
        timestamp_seconds=2.0,
    )
    assert jagged.boundary_complexity.value > compact.boundary_complexity.value


def test_oracle_fixture_preserves_oracle_provenance_without_physical_claim():
    actual = estimate_contact_geometry_features(
        segmentation("pointed_tip", rectangle(10, 20, 13, 17)),
        timestamp_seconds=2.0,
        evidence_status=EstimateStatus.ORACLE,
    )
    assert actual.geometry_class.status is EstimateStatus.ORACLE
    assert actual.elongation.status is EstimateStatus.ORACLE
    assert actual.sharpness.status is EstimateStatus.UNAVAILABLE


def test_unknown_geometry_class_is_rejected():
    with pytest.raises(ValueError, match="general taxonomy"):
        estimate_contact_geometry_features(
            segmentation("knife", rectangle(10, 14, 5, 25)),
            timestamp_seconds=2.0,
        )


def test_geometry_descriptors_keep_segmentation_timestamp():
    actual = estimate_contact_geometry_features(
        segmentation(
            "pointed_tip",
            rectangle(10, 20, 13, 17),
            timestamp=1.0,
        ),
        timestamp_seconds=2.0,
    )

    assert actual.geometry_class.timestamp_seconds == 1.0
    assert actual.exposure_fraction.timestamp_seconds == 1.0
