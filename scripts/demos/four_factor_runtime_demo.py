"""Stationary four-factor CRA-to-fixed-lambda policy diagnostic."""

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.physical_attributes import (
    MassAttributes,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.risk.cra import CRARiskAssessor
from robot.risk.physical_factor_adapter import adapt_physical_factors
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy


def scalar(value, unit="1"):
    return ScalarEstimate(
        value,
        max(0, value * 0.9),
        value if unit != "1" else min(1, value * 1.1),
        0.8,
        unit,
        "synthetic_specialist_fixture",
        2,
    )


def main():
    base = PhysicalAttributes.unknown(track_id=1, timestamp_seconds=2)
    attributes = replace(
        base,
        mass=MassAttributes(scalar(10, "kg")),
        geometry=replace(base.geometry, fall_height_m=scalar(1, "m")),
        motion=replace(
            base.motion,
            relative_speed_mps=scalar(1, "m/s"),
            route_crossing_likelihood=scalar(0.6),
        ),
        contact=replace(
            base.contact,
            sharpness=scalar(0.9),
            hardness=scalar(0.8),
            dangerous_part_exposure=scalar(0.5),
            orientation_toward_route=scalar(0.7),
        ),
        thermal=replace(base.thermal, heat_source_likelihood=scalar(0.6)),
    )
    result = adapt_physical_factors(
        attributes,
        timestamp_seconds=2,
        potential_interaction_likelihood=scalar(0.4),
        thermal_interaction_likelihood=scalar(0.3),
    )
    assessment = CRARiskAssessor().assess(result.profile)
    policy = CRASafeSAGEPolicy().evaluate(result.profile)
    print("STATIONARY FOUR-FACTOR DIAGNOSTIC: synthetic evidence; no wheel motion")
    for factor in assessment.factors:
        print(
            f"{factor.factor.value}: nominal={factor.nominal_risk:.3f}, "
            f"conservative={factor.conservative_risk:.3f}, confidence={factor.estimate.confidence:.2f}"
        )
    print(
        f"phi={assessment.phi:.3f}; flux={policy.boundary_flux:.3f}; "
        f"lambda={policy.tangential_strength:.2f}"
    )
    assert policy.tangential_strength == 1.25
    print("PASS: all factors explicit and primary lambda remains fixed")


if __name__ == "__main__":
    main()
