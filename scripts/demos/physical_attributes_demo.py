"""Demonstrate the dependency-free CRA physical-attribute contracts.

This demo constructs four controlled records representing the kinds of
evidence that future camera, LiDAR, tracking, and learned estimators will
produce.  It does not run Isaac Sim, a vision model, CRA risk fusion, or
Safe-SAGE.  Values are illustrative software-validation examples and are not
calibrated research parameters.

Run from the project root:

    python -u scripts/demos/physical_attributes_demo.py
"""

from dataclasses import dataclass, replace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot.perception.physical_attributes import (
    CategoricalEstimate,
    EstimateStatus,
    MassAttributes,
    PhysicalAttributes,
    ScalarEstimate,
)


TIMESTAMP_SECONDS = 1.0


@dataclass(frozen=True)
class ControlledExample:
    """One illustrative physical-evidence record."""

    name: str
    interpretation: str
    attributes: PhysicalAttributes


def scalar(
    value: float,
    lower: float,
    upper: float,
    *,
    unit: str = "1",
    confidence: float,
    source: str,
    status: EstimateStatus = EstimateStatus.INFERRED,
    evidence: str,
) -> ScalarEstimate:
    """Return one concise controlled scalar estimate."""

    return ScalarEstimate(
        value=value,
        lower_bound=lower,
        upper_bound=upper,
        confidence=confidence,
        unit=unit,
        source=source,
        timestamp_seconds=TIMESTAMP_SECONDS,
        status=status,
        evidence=(evidence,),
    )


def category(
    value: str,
    *,
    confidence: float,
    source: str,
    evidence: str,
) -> CategoricalEstimate:
    """Return one concise controlled categorical estimate."""

    return CategoricalEstimate(
        value=value,
        confidence=confidence,
        source=source,
        timestamp_seconds=TIMESTAMP_SECONDS,
        evidence=(evidence,),
    )


def base_record(
    track_id: int,
    semantic_class: str,
    semantic_confidence: float,
) -> PhysicalAttributes:
    """Return a record whose not-yet-supplied attributes are unavailable."""

    return PhysicalAttributes.unknown(
        track_id=track_id,
        semantic_class=semantic_class,
        semantic_confidence=semantic_confidence,
        semantic_source="temporal_yolo_lidar_fusion",
        timestamp_seconds=TIMESTAMP_SECONDS,
        evidence=("specialist estimator has not supplied this attribute",),
    )


def controlled_examples() -> tuple[ControlledExample, ...]:
    """Return chair, person, knife, and hot-soup evidence examples."""

    chair = base_record(1, "chair", 0.91)
    chair = replace(
        chair,
        mass=MassAttributes(
            mass_kg=scalar(
                6.0,
                3.5,
                10.0,
                unit="kg",
                confidence=0.55,
                source="class_size_prior",
                status=EstimateStatus.PRIOR,
                evidence="chair class and LiDAR dimensions",
            )
        ),
        geometry=replace(
            chair.geometry,
            height_m=scalar(
                0.90,
                0.86,
                0.95,
                unit="m",
                confidence=0.90,
                source="camera_lidar_geometry",
                status=EstimateStatus.MEASURED,
                evidence="associated LiDAR vertical extent",
            ),
        ),
        stability=replace(
            chair.stability,
            instability_likelihood=scalar(
                0.10,
                0.03,
                0.25,
                confidence=0.70,
                source="support_geometry",
                evidence="center projection lies inside support region",
            ),
            support_state=category(
                "supported",
                confidence=0.80,
                source="support_geometry",
                evidence="chair feet intersect the floor support plane",
            ),
        ),
    )

    person = base_record(2, "person", 0.96)
    person = replace(
        person,
        mass=MassAttributes(
            mass_kg=scalar(
                75.0,
                45.0,
                110.0,
                unit="kg",
                confidence=0.45,
                source="adult_person_prior",
                status=EstimateStatus.PRIOR,
                evidence="broad anthropometric prior",
            )
        ),
        motion=replace(
            person.motion,
            velocity_x_mps=scalar(
                0.05,
                -0.05,
                0.15,
                unit="m/s",
                confidence=0.88,
                source="lidar_kalman_tracker",
                status=EstimateStatus.MEASURED,
                evidence="track displacement across LiDAR frames",
            ),
            velocity_y_mps=scalar(
                -1.20,
                -1.35,
                -1.05,
                unit="m/s",
                confidence=0.88,
                source="lidar_kalman_tracker",
                status=EstimateStatus.MEASURED,
                evidence="track displacement across LiDAR frames",
            ),
            relative_speed_mps=scalar(
                1.21,
                1.05,
                1.37,
                unit="m/s",
                confidence=0.84,
                source="relative_motion_estimator",
                evidence="robot and obstacle velocity vectors",
            ),
            route_crossing_likelihood=scalar(
                0.95,
                0.80,
                1.00,
                confidence=0.82,
                source="route_conflict_estimator",
                evidence="predicted track crosses the nominal route",
            ),
        ),
    )

    knife = base_record(3, "knife", 0.89)
    knife = replace(
        knife,
        mass=MassAttributes(
            mass_kg=scalar(
                0.08,
                0.03,
                0.15,
                unit="kg",
                confidence=0.65,
                source="utensil_size_prior",
                status=EstimateStatus.PRIOR,
                evidence="knife class and metric length",
            )
        ),
        contact=replace(
            knife.contact,
            sharpness=scalar(
                0.95,
                0.85,
                1.00,
                confidence=0.80,
                source="dangerous_part_segmenter",
                evidence="visible blade and pointed tip",
            ),
            hardness=scalar(
                0.90,
                0.75,
                1.00,
                confidence=0.72,
                source="visual_material_classifier",
                evidence="metal material classification",
            ),
            dangerous_part_exposure=scalar(
                0.90,
                0.75,
                1.00,
                confidence=0.84,
                source="dangerous_part_segmenter",
                evidence="most blade pixels are uncovered",
            ),
            orientation_toward_route=scalar(
                0.75,
                0.55,
                0.90,
                confidence=0.70,
                source="camera_lidar_orientation",
                evidence="tip direction intersects the traversable corridor",
            ),
            dangerous_part=category(
                "knife blade",
                confidence=0.88,
                source="dangerous_part_segmenter",
                evidence="blade region separated from handle",
            ),
        ),
    )

    hot_soup = base_record(4, "bowl", 0.87)
    hot_soup = replace(
        hot_soup,
        mass=MassAttributes(
            mass_kg=scalar(
                1.2,
                0.7,
                1.8,
                unit="kg",
                confidence=0.50,
                source="container_content_prior",
                status=EstimateStatus.PRIOR,
                evidence="bowl size and uncertain liquid fill level",
            )
        ),
        geometry=replace(
            hot_soup.geometry,
            fall_height_m=scalar(
                1.10,
                1.00,
                1.20,
                unit="m",
                confidence=0.82,
                source="camera_lidar_geometry",
                status=EstimateStatus.MEASURED,
                evidence="container height above the floor",
            ),
        ),
        thermal=replace(
            hot_soup.thermal,
            heat_source_likelihood=scalar(
                0.90,
                0.70,
                1.00,
                confidence=0.66,
                source="hot_liquid_cue_estimator",
                evidence="open bowl with persistent visible steam",
            ),
            steam_likelihood=scalar(
                0.85,
                0.65,
                0.98,
                confidence=0.76,
                source="temporal_steam_detector",
                evidence="rising plume persists across frames",
            ),
            open_container_likelihood=scalar(
                0.95,
                0.85,
                1.00,
                confidence=0.88,
                source="container_state_classifier",
                evidence="container opening is visible",
            ),
            carried_likelihood=scalar(
                0.90,
                0.75,
                1.00,
                confidence=0.81,
                source="human_object_relation_estimator",
                evidence="bowl moves with the waiter track",
            ),
            spill_likelihood=scalar(
                0.70,
                0.45,
                0.90,
                confidence=0.58,
                source="container_motion_estimator",
                evidence="open carried liquid can spill",
            ),
            source_type=category(
                "steaming liquid",
                confidence=0.79,
                source="thermal_cue_classifier",
                evidence="container and temporal steam cues agree",
            ),
        ),
    )

    return (
        ControlledExample(
            name="ordinary chair",
            interpretation="Mass, height, and support evidence are available.",
            attributes=chair,
        ),
        ControlledExample(
            name="crossing person",
            interpretation="Tracking supplies signed velocity and route conflict.",
            attributes=person,
        ),
        ControlledExample(
            name="exposed knife",
            interpretation="Part, material, exposure, and orientation remain separate.",
            attributes=knife,
        ),
        ControlledExample(
            name="hot soup",
            interpretation="Thermal cues are inferred, not temperature measurements.",
            attributes=hot_soup,
        ),
    )


def format_scalar(estimate: ScalarEstimate) -> str:
    """Return a compact scalar-estimate description."""

    if not estimate.is_available:
        return "unavailable"
    assert estimate.value is not None
    assert estimate.lower_bound is not None
    assert estimate.upper_bound is not None
    return (
        f"{estimate.value:.3f} {estimate.unit} "
        f"[{estimate.lower_bound:.3f}, {estimate.upper_bound:.3f}], "
        f"confidence={estimate.confidence:.2f}, status={estimate.status.value}"
    )


def print_example(example: ControlledExample) -> None:
    """Print the most relevant evidence from one example."""

    item = example.attributes
    print("\n" + "-" * 78)
    print(example.name.upper())
    print("-" * 78)
    print(f"  Interpretation:       {example.interpretation}")
    print(f"  Track ID:             {item.track_id}")
    print(f"  Semantic class:       {item.semantic_class}")
    print(f"  Semantic confidence:  {item.semantic_confidence:.2f}")
    print(
        "  Available estimates:  "
        f"{item.available_estimate_count}/{len(item.estimates())}"
    )
    print(f"  Mass:                 {format_scalar(item.mass.mass_kg)}")
    print(f"  Height:               {format_scalar(item.geometry.height_m)}")
    print(f"  Fall height:          {format_scalar(item.geometry.fall_height_m)}")
    print(
        "  Relative speed:       "
        f"{format_scalar(item.motion.relative_speed_mps)}"
    )
    print(f"  Sharpness:            {format_scalar(item.contact.sharpness)}")
    print(
        "  Heat-source likelihood:"
        f" {format_scalar(item.thermal.heat_source_likelihood)}"
    )


def main() -> None:
    """Print the controlled physical-attribute examples."""

    print("\n" + "=" * 78)
    print("CRA PHYSICAL-ATTRIBUTE CORE - CONTROLLED DATA-CONTRACT DEMO")
    print("=" * 78)
    print("  Values represent perception evidence, not CRA factor risks.")
    print("  Unavailable evidence remains missing rather than becoming zero.")
    print("  No Isaac Sim, vision model, CRA fusion, or controller is running.")

    examples = controlled_examples()
    for example in examples:
        print_example(example)

    all_valid = all(
        example.attributes.timestamp_seconds == TIMESTAMP_SECONDS
        for example in examples
    )
    print("\n" + "=" * 78)
    print("VALIDATION RESULT")
    print("=" * 78)
    if all_valid:
        print("  [PASS] All four physical-factor families are representable.")
        print("  The next branch can populate geometry and motion from sensors.")
    else:
        print("  [FAIL] At least one record failed timestamp validation.")
        raise RuntimeError("Controlled physical-attribute validation failed.")


if __name__ == "__main__":
    main()
