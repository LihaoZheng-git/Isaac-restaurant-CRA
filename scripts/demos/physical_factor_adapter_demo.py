"""Stationary synthetic diagnostic; no sensors, models, Isaac or wheel commands.

Run: python scripts/demos/physical_factor_adapter_demo.py
All values are illustrative fixtures, including values labeled measured to
exercise contracts. They are not observations collected from a real sensor.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot.perception.physical_attributes import (
    EstimateStatus,
    MassAttributes,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.risk.cra import CRARiskAssessor
from robot.risk.physical_factor_adapter import adapt_physical_factors


def scalar(value, unit="1", *, status=EstimateStatus.INFERRED, confidence=0.8):
    return ScalarEstimate(
        value,
        value * 0.8,
        min(1, value * 1.2) if unit == "1" else value * 1.2,
        confidence,
        unit,
        "synthetic_diagnostic_fixture",
        2.0,
        status,
    )


def main():
    print("STATIONARY SYNTHETIC DIAGNOSTIC: no sensing or robot motion")
    base = PhysicalAttributes.unknown(track_id=7, timestamp_seconds=2)
    supplied = replace(
        base,
        mass=MassAttributes(
            scalar(10, "kg", status=EstimateStatus.PRIOR, confidence=0.9)
        ),
        geometry=replace(
            base.geometry,
            fall_height_m=scalar(1, "m"),
            height_m=scalar(3, "m", status=EstimateStatus.MEASURED),
        ),
        motion=replace(
            base.motion,
            relative_speed_mps=scalar(2, "m/s", status=EstimateStatus.MEASURED),
            route_crossing_likelihood=scalar(0.6),
        ),
    )
    cases = (
        ("prior mass + geometry/motion fixtures", supplied, scalar(0.4), 2.0),
        (
            "object height only; fall height unavailable",
            replace(
                supplied,
                geometry=replace(
                    supplied.geometry, fall_height_m=base.geometry.fall_height_m
                ),
            ),
            scalar(0.4),
            2.0,
        ),
        ("missing harmful-fall interaction", supplied, None, 2.0),
        ("stale evidence", supplied, scalar(0.4), 4.0),
        ("all unavailable", base, None, 2.0),
    )
    for name, item, interaction, timestamp in cases:
        result = adapt_physical_factors(
            item,
            timestamp_seconds=timestamp,
            potential_interaction_likelihood=interaction,
        )
        assessment = CRARiskAssessor().assess(result.profile)
        print(
            json.dumps(
                {
                    "case": name,
                    "potential": result.potential.as_dict(),
                    "kinetic": result.kinetic.as_dict(),
                    "phi": assessment.phi,
                    "contact_and_thermal": "explicitly unknown",
                },
                indent=2,
            )
        )
    print("Default scales: 100 J; upper bounds; prior confidence cap 0.5; age 1 s.")
    print(
        "Unknown contact/thermal saturate default CRA phi; inspect factor diagnostics."
    )
    print("Energy scales and interaction proxies require calibration and evaluation.")


if __name__ == "__main__":
    main()
