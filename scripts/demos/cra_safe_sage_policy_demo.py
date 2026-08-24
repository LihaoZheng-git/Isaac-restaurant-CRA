"""Validate the CRA-to-Safe-SAGE policy without simulation or perception.

Every scenario below is assumed to have identical obstacle geometry.  Only
the controlled physical-risk profile changes, allowing the policy effect to
be inspected without distance, LiDAR, YOLO, field-solver, or controller
confounds.

The default policy reproduces the repository's paper-baseline endpoints:

* conservative CRA ``phi`` maps linearly to ``|b|`` from 0.5 to 1.7;
* tangential strength remains fixed at ``lambda = 1.25``; and
* passing side remains robot-right.

All factor values are illustrative software-validation hypotheses, not
calibrated research measurements.

Run from the project root:

    python -u scripts/demos/cra_safe_sage_policy_demo.py
"""

from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot.risk.cra import PhysicalRiskProfile, RiskFactorEstimate
from robot.safety.cra_safe_sage_policy import (
    CRASafeSAGEPolicy,
    CRASafeSAGEPolicyResult,
)


@dataclass(frozen=True)
class ControlledPolicyCase:
    """One manually specified same-geometry CRA policy case."""

    name: str
    profile: PhysicalRiskProfile


def absent(reason: str) -> RiskFactorEstimate:
    """Return one confidently absent physical factor."""

    return RiskFactorEstimate.known_absent(
        source="controlled_oracle",
        evidence=(reason,),
    )


def estimate(
    severity: float,
    interaction_likelihood: float,
    *,
    confidence: float = 1.0,
    evidence: str,
) -> RiskFactorEstimate:
    """Return one controlled physical-factor estimate."""

    return RiskFactorEstimate(
        severity=severity,
        interaction_likelihood=interaction_likelihood,
        confidence=confidence,
        source="controlled_oracle",
        evidence=(evidence,),
    )


def controlled_cases() -> tuple[ControlledPolicyCase, ...]:
    """Return ordered low-, contact-, compound-, and unknown-risk cases."""

    return (
        ControlledPolicyCase(
            name="ordinary chair",
            profile=PhysicalRiskProfile(
                thermal=absent("no heat source"),
                contact=estimate(
                    0.20,
                    0.20,
                    evidence="rounded furniture surface",
                ),
                potential=estimate(
                    0.30,
                    0.20,
                    evidence="chair is supported and stable",
                ),
                kinetic=absent("chair is stationary"),
            ),
        ),
        ControlledPolicyCase(
            name="exposed knife",
            profile=PhysicalRiskProfile(
                thermal=absent("no heat source"),
                contact=estimate(
                    0.95,
                    0.85,
                    evidence="unsheathed blade is exposed",
                ),
                potential=estimate(
                    0.15,
                    0.20,
                    evidence="knife is close to the floor",
                ),
                kinetic=absent("knife is stationary"),
            ),
        ),
        ControlledPolicyCase(
            name="waiter carrying hot soup",
            profile=PhysicalRiskProfile(
                thermal=estimate(
                    0.90,
                    0.80,
                    evidence="open hot liquid can spill",
                ),
                contact=estimate(
                    0.20,
                    0.25,
                    evidence="container has no sharp edge",
                ),
                potential=estimate(
                    0.45,
                    0.60,
                    evidence="liquid is carried above the floor",
                ),
                kinetic=estimate(
                    0.45,
                    0.70,
                    evidence="carrier crosses the nominal route",
                ),
            ),
        ),
        ControlledPolicyCase(
            name="unrecognized object",
            profile=PhysicalRiskProfile.unknown(source="unavailable"),
        ),
    )


def print_result(
    case: ControlledPolicyCase,
    result: CRASafeSAGEPolicyResult,
) -> None:
    """Print one explainable CRA-to-Safe-SAGE mapping."""

    dominant = (
        "none"
        if result.assessment.dominant_factor is None
        else result.assessment.dominant_factor.value
    )
    print(
        f"  {case.name:<27} "
        f"{result.assessment.nominal_phi:>8.4f}  "
        f"{result.risk:>8.4f}  "
        f"{result.boundary_flux:>8.4f}  "
        f"{result.tangential_strength:>8.4f}  "
        f"{dominant:<10}"
    )


def main() -> None:
    """Run the controlled same-geometry policy comparison."""

    print("\n" + "=" * 92)
    print("CRA -> SAFE-SAGE POLICY - CONTROLLED SAME-GEOMETRY DEMO")
    print("=" * 92)
    print("  Geometry:            identical and deliberately omitted")
    print("  Normal mapping:      |b| = 0.5 + phi * (1.7 - 0.5)")
    print("  Tangential mapping:  lambda = 1.25 (fixed paper baseline)")
    print("  Passing side:        right")
    print("  Values:              illustrative and uncalibrated")
    print(
        "\n  scenario                     phi_nom       phi         b    lambda  dominant"
    )
    print("  " + "-" * 88)

    policy = CRASafeSAGEPolicy()
    results = tuple(
        (case, policy.evaluate(case.profile)) for case in controlled_cases()
    )
    for case, result in results:
        print_result(case, result)

    flux_magnitudes = tuple(abs(result.boundary_flux) for _, result in results)
    strengths = tuple(result.tangential_strength for _, result in results)
    risks = tuple(result.risk for _, result in results)
    risk_ordering_valid = risks == tuple(sorted(risks))
    flux_ordering_valid = flux_magnitudes == tuple(sorted(flux_magnitudes))
    fixed_tangent_valid = all(abs(strength - 1.25) <= 1e-12 for strength in strengths)
    unknown_fail_closed = results[-1][1].risk == 1.0

    print("\n" + "=" * 92)
    print("CONTROLLED POLICY CHECKS")
    print("=" * 92)
    print(f"  Scenario risks are ordered:       {risk_ordering_valid}")
    print(f"  Higher risk strengthens |b|:      {flux_ordering_valid}")
    print(f"  Baseline lambda remains fixed:    {fixed_tangent_valid}")
    print(f"  Unknown input maps fail-closed:   {unknown_fail_closed}")
    passed = (
        risk_ordering_valid
        and flux_ordering_valid
        and fixed_tangent_valid
        and unknown_fail_closed
    )
    if passed:
        print(
            "\n  [PASS] CRA changes the Safe-SAGE normal response while "
            "the controlled geometry and baseline tangential policy stay fixed."
        )
    else:
        print("\n  [WARNING] A controlled CRA-to-Safe-SAGE policy check failed.")
    print("=" * 92)


if __name__ == "__main__":
    main()
