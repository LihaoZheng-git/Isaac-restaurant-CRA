import numpy as np
import pytest

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.contact_cues import DangerousPartSegmentation
from robot.perception.hot_soup_detector import (
    HotSoupDetection,
    HotSoupDetectionFrame,
    estimate_hot_soup_from_detections,
)
from robot.perception.semantic_detection import PixelBoundingBox
from robot.perception.specialist_track_evidence import (
    SpecialistAssociationFrame,
    build_track_specialist_evidence,
)


def association(track_id, detection_index, point=(50.0, 65.0), status="matched"):
    return TrackSemanticAssociation(
        track_id=track_id,
        status=status,
        visible_point_count=4,
        detection_index=detection_index,
        mask_hit_count=4,
        overlap_fraction=1.0,
        association_score=1.0,
        projected_median_inside_box=True,
        projected_median_xy=point,
    )


def association_frame(*items):
    return SpecialistAssociationFrame(7, 2.0, 100, 100, items)


def segmentation(timestamp=2.0):
    object_mask = np.zeros((100, 100), dtype=bool)
    object_mask[40:80, 30:70] = True
    danger_mask = np.zeros_like(object_mask)
    danger_mask[40:50, 30:70] = True
    return DangerousPartSegmentation(
        object_mask,
        danger_mask,
        "exposed_edge",
        0.8,
        timestamp,
        "contact-fixture",
    )


def hot_soup_frame(timestamp=2.0, frame_id=7):
    return HotSoupDetectionFrame(
        frame_id,
        timestamp,
        100,
        100,
        (
            HotSoupDetection(
                0,
                "visible_hot_soup_plume",
                0.9,
                PixelBoundingBox(40, 10, 60, 45),
            ),
            HotSoupDetection(
                1,
                "open_container",
                0.8,
                PixelBoundingBox(30, 45, 70, 80),
            ),
        ),
        "oracle-fixture",
        0.0,
    )


def test_aligned_specialists_become_fresh_per_track_providers():
    soup = hot_soup_frame()
    soup_result = estimate_hot_soup_from_detections(soup, timestamp_seconds=2.0)
    evidence = build_track_specialist_evidence(
        association_frame(association(4, 2)),
        contact_segmentations={2: segmentation()},
        hot_soup_frame=soup,
        hot_soup_result=soup_result,
    )

    contact = evidence.contact_for(4, 2.1)
    thermal = evidence.thermal_for(4, 2.1)
    assert contact.dangerous_part.value == "exposed_edge"
    assert contact.dangerous_part_exposure.value == pytest.approx(0.25)
    assert not contact.sharpness.is_available
    assert thermal.source_type.value == "visible_hot_soup_plume_proxy"
    assert not thermal.steam_likelihood.is_available
    assert evidence.diagnostics.contact_track_ids == (4,)
    assert evidence.diagnostics.thermal_track_ids == (4,)


def test_stale_or_unmatched_evidence_stays_unavailable():
    evidence = build_track_specialist_evidence(
        association_frame(association(4, 2)),
        contact_segmentations={2: segmentation()},
        maximum_age_seconds=0.5,
    )

    assert not evidence.contact_for(4, 2.6).dangerous_part_exposure.is_available
    assert not evidence.contact_for(9, 2.1).dangerous_part_exposure.is_available
    assert not evidence.thermal_for(4, 2.1).heat_source_likelihood.is_available


def test_multiple_tracks_in_container_are_ambiguous_and_unassigned():
    soup = hot_soup_frame()
    result = estimate_hot_soup_from_detections(soup, timestamp_seconds=2.0)
    evidence = build_track_specialist_evidence(
        association_frame(
            association(4, 0, point=(45, 65)),
            association(5, 1, point=(55, 65)),
        ),
        hot_soup_frame=soup,
        hot_soup_result=result,
    )

    assert evidence.diagnostics.ambiguous_thermal_track_ids == (4, 5)
    assert not evidence.thermal_for(4, 2.1).heat_source_likelihood.is_available


def test_specialist_frame_identity_and_timestamp_must_match():
    soup = hot_soup_frame(frame_id=8)
    result = estimate_hot_soup_from_detections(soup, timestamp_seconds=2.0)
    with pytest.raises(ValueError, match="frame IDs"):
        build_track_specialist_evidence(
            association_frame(association(4, 0)),
            hot_soup_frame=soup,
            hot_soup_result=result,
        )

    with pytest.raises(ValueError, match="timestamp"):
        build_track_specialist_evidence(
            association_frame(association(4, 0)),
            contact_segmentations={0: segmentation(timestamp=1.9)},
        )
