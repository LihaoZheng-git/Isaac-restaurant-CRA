"""Stationary oracle diagnostic from plume boxes to the CRA thermal factor."""

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.hot_soup_detector import (
    HotSoupDetection,
    HotSoupDetectionFrame,
    estimate_hot_soup_from_detections,
)
from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
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


def oracle_route_score(value: float, timestamp: float) -> ScalarEstimate:
    return ScalarEstimate(
        value,
        max(0.0, value - 0.1),
        min(1.0, value + 0.1),
        1.0,
        "1",
        "authored_route_oracle_fixture",
        timestamp,
        EstimateStatus.ORACLE,
        ("authored stationary waypoint-corridor relation",),
    )


def main() -> None:
    timestamp = 2.0
    frame = HotSoupDetectionFrame(
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
        evidence_status=EstimateStatus.ORACLE,
    )
    association = TrackSemanticAssociation(
        4,
        "matched",
        4,
        0,
        4,
        1.0,
        1.0,
        True,
        (50.0, 65.0),
    )
    result = estimate_hot_soup_from_detections(frame, timestamp_seconds=timestamp)
    specialist = build_track_specialist_evidence(
        SpecialistAssociationFrame(7, timestamp, 100, 100, (association,)),
        hot_soup_frame=frame,
        hot_soup_result=result,
    )
    thermal = specialist.thermal_for(4, timestamp)
    route_score = oracle_route_score(0.4, timestamp)
    interaction = estimate_thermal_route_interaction(
        thermal,
        route_score,
        timestamp_seconds=timestamp,
    )
    base = PhysicalAttributes.unknown(track_id=4, timestamp_seconds=timestamp)
    attributes = replace(
        base,
        thermal=thermal,
        motion=replace(base.motion, route_crossing_likelihood=route_score),
    )
    adapted = adapt_physical_factors(
        attributes,
        timestamp_seconds=timestamp,
        thermal_interaction_likelihood=interaction,
    )
    assessment = CRARiskAssessor().assess(adapted.profile)
    policy = CRASafeSAGEPolicy().evaluate(adapted.profile)

    print("STATIONARY THERMAL FACTOR DIAGNOSTIC: authored oracle boxes; no model")
    print("wheel_commands=0; temperature=unavailable; steam_identity=unavailable")
    print(
        f"heat_proxy={thermal.heat_source_likelihood.value:.3f}; "
        f"route_interaction={interaction.value:.3f}; "
        f"status={interaction.status.value}"
    )
    print(
        f"thermal_risk={adapted.profile.thermal.nominal_risk:.3f}; "
        f"nominal_phi={assessment.nominal_phi:.3f}; "
        f"conservative_phi={assessment.phi:.3f}; "
        f"lambda={policy.tangential_strength:.2f}"
    )
    assert thermal.heat_source_likelihood.status is EstimateStatus.ORACLE
    assert interaction.status is EstimateStatus.ORACLE
    assert not adapted.thermal.rejection_reasons
    assert adapted.thermal.estimate.confidence > 0
    assert policy.tangential_strength == 1.25
    print("PASS: provenance and unknown nonthermal factors remained explicit")


if __name__ == "__main__":
    main()
