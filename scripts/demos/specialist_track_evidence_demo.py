"""Stationary oracle-fixture diagnostic for specialist-to-track plumbing.

This demo sends no wheel commands and runs no learned model. Its masks and
boxes are authored oracle fixtures used only to validate frame alignment,
LiDAR track assignment, freshness, and conservative unknowns.
"""

from __future__ import annotations

import numpy as np

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


def main() -> None:
    timestamp = 2.0
    object_mask = np.zeros((100, 100), dtype=bool)
    object_mask[40:80, 30:70] = True
    dangerous_mask = np.zeros_like(object_mask)
    dangerous_mask[40:50, 30:70] = True
    contact = DangerousPartSegmentation(
        object_mask,
        dangerous_mask,
        "exposed_edge",
        0.8,
        timestamp,
        "authored_oracle_fixture",
    )
    soup_frame = HotSoupDetectionFrame(
        7,
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
        "authored_oracle_fixture",
        0.0,
    )
    association = TrackSemanticAssociation(
        track_id=4,
        status="matched",
        visible_point_count=4,
        detection_index=2,
        mask_hit_count=4,
        overlap_fraction=1.0,
        association_score=1.0,
        projected_median_inside_box=True,
        projected_median_xy=(50.0, 65.0),
    )
    evidence = build_track_specialist_evidence(
        SpecialistAssociationFrame(7, timestamp, 100, 100, (association,)),
        contact_segmentations={2: contact},
        hot_soup_frame=soup_frame,
        hot_soup_result=estimate_hot_soup_from_detections(
            soup_frame,
            timestamp_seconds=timestamp,
        ),
    )

    attached_contact = evidence.contact_for(4, 2.1)
    attached_thermal = evidence.thermal_for(4, 2.1)
    stale_contact = evidence.contact_for(4, 3.0)
    assert attached_contact.dangerous_part.value == "exposed_edge"
    assert attached_thermal.source_type.value == "contextual_visible_plume_proxy"
    assert not attached_contact.sharpness.is_available
    assert not attached_thermal.steam_likelihood.is_available
    assert not stale_contact.dangerous_part_exposure.is_available
    print("stationary=True; wheel_commands=0; evidence=authored_oracle_fixture")
    print(f"contact_tracks={evidence.diagnostics.contact_track_ids}")
    print(f"thermal_tracks={evidence.diagnostics.thermal_track_ids}")
    print("unsupported_sharpness=unavailable; steam_identity=unavailable")
    print("stale_contact=unavailable")


if __name__ == "__main__":
    main()
