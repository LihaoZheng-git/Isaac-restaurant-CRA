"""Stationary oracle plumbing diagnostic for continuous plume-mask evidence."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.continuous_thermal_score import (
    estimate_continuous_thermal_score,
)
from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.perception.plume_features import (
    PlumeSourceObservation,
    extract_continuous_plume_features,
)
from robot.perception.plume_segmentation import (
    PlumeSegmentationFrame,
    VisualPlumeInstance,
)
from robot.perception.semantic_detection import PixelBoundingBox
from robot.perception.specialist_track_evidence import (
    SpecialistAssociationFrame,
    build_track_specialist_evidence,
)
from robot.perception.thermal_interaction import (
    estimate_thermal_route_interaction,
)
from robot.risk.cra import CRARiskAssessor
from robot.risk.physical_factor_adapter import adapt_physical_factors
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy


def oracle_cue(value: float, timestamp: float, source: str) -> ScalarEstimate:
    return ScalarEstimate(
        value,
        max(0.0, value - 0.05),
        min(1.0, value + 0.05),
        1.0,
        "1",
        source,
        timestamp,
        EstimateStatus.ORACLE,
    )


def plume_frame(frame_id: int, timestamp: float) -> PlumeSegmentationFrame:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:45, 40:60] = True
    instance = VisualPlumeInstance(
        "visible_plume",
        "visible_hot_soup_plume",
        0.9,
        PixelBoundingBox(40, 10, 60, 45),
        mask,
        0,
    )
    return PlumeSegmentationFrame(
        frame_id,
        timestamp,
        100,
        100,
        (instance,),
        "authored_oracle_mask_fixture",
        0.0,
        EstimateStatus.ORACLE,
    )


def main() -> None:
    timestamp = 2.0
    source_box = PixelBoundingBox(30, 45, 70, 80)
    observations = tuple(
        PlumeSourceObservation(
            plume_frame(frame_id, frame_timestamp),
            source_box,
            oracle_cue(0.8, frame_timestamp, "authored_source_context"),
            oracle_cue(0.9, frame_timestamp, "authored_spatial_association"),
            0,
        )
        for frame_id, frame_timestamp in ((5, 1.8), (6, 1.9), (7, 2.0))
    )
    features = extract_continuous_plume_features(
        observations,
        timestamp_seconds=timestamp,
    )
    score = estimate_continuous_thermal_score(
        features,
        timestamp_seconds=timestamp,
    )
    association = TrackSemanticAssociation(
        track_id=4,
        status="matched",
        visible_point_count=4,
        detection_index=0,
        mask_hit_count=4,
        overlap_fraction=1.0,
        association_score=1.0,
        projected_median_inside_box=True,
        projected_median_xy=(50.0, 65.0),
    )
    specialist = build_track_specialist_evidence(
        SpecialistAssociationFrame(7, timestamp, 100, 100, (association,)),
        thermal_attributes_by_detection_index={0: score.thermal},
    )
    thermal = specialist.thermal_for(4, timestamp)
    route = oracle_cue(0.4, timestamp, "authored_waypoint_corridor")
    interaction = estimate_thermal_route_interaction(
        thermal,
        route,
        timestamp_seconds=timestamp,
    )
    base = PhysicalAttributes.unknown(track_id=4, timestamp_seconds=timestamp)
    attributes = replace(
        base,
        thermal=thermal,
        motion=replace(base.motion, route_crossing_likelihood=route),
    )
    adapted = adapt_physical_factors(
        attributes,
        timestamp_seconds=timestamp,
        thermal_interaction_likelihood=interaction,
    )
    assessment = CRARiskAssessor().assess(adapted.profile)
    policy = CRASafeSAGEPolicy().evaluate(adapted.profile)

    print("CONTINUOUS THERMAL STATIONARY DIAGNOSTIC: authored oracle masks")
    print("wheel_commands=0; temperature=unavailable; hot_soup_class=unavailable")
    print(
        f"relative_area={features.relative_mask_area.value:.3f}; "
        f"vertical_extent={features.vertical_extent.value:.3f}; "
        f"persistence={features.persistence.value:.3f}"
    )
    print(
        f"plume_strength={score.plume_strength.value:.3f}; "
        f"heat_evidence={thermal.heat_source_likelihood.value:.3f}; "
        f"route_interaction={interaction.value:.3f}"
    )
    print(
        f"thermal_risk={adapted.profile.thermal.nominal_risk:.3f}; "
        f"nominal_phi={assessment.nominal_phi:.3f}; "
        f"conservative_phi={assessment.phi:.3f}; "
        f"lambda={policy.tangential_strength:.2f}"
    )
    assert thermal.heat_source_likelihood.status is EstimateStatus.ORACLE
    assert interaction.status is EstimateStatus.ORACLE
    assert specialist.diagnostics.thermal_track_ids == (4,)
    assert not adapted.thermal.rejection_reasons
    assert policy.tangential_strength == 1.25
    print("PASS: continuous thermal evidence reached CRA with oracle provenance")


if __name__ == "__main__":
    main()
