"""Validate the model-agnostic CRA mass-estimation pipeline.

This controlled, dependency-light demo exercises three decisions:

1. A visible LiDAR envelope is rejected as complete object volume, so a
   configured semantic prior remains the only usable chair-mass evidence.
2. Complete RGB-D geometry, a material-density estimate, a learned mass
   prediction, and a class prior are conservatively fused.
3. A class with no evidence remains unavailable rather than becoming zero.

All numbers below are illustrative software-validation values.  No camera,
neural network, Isaac Sim, CRA risk fusion, or robot control runs here.

Run from the project root:

    python -u scripts/demos/mass_estimation_demo.py
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from robot.perception.mass_estimation import (
    MASS_FUSION_SOURCE,
    ClassMassPrior,
    ClassMassPriorTable,
    estimate_track_mass,
)
from robot.perception.physical_attributes import (
    EstimateStatus,
    GeometryAttributes,
    ScalarEstimate,
)


TIMESTAMP_SECONDS = 2.0


def scalar(
    value: float,
    lower: float,
    upper: float,
    *,
    confidence: float,
    unit: str,
    source: str,
    status: EstimateStatus = EstimateStatus.INFERRED,
) -> ScalarEstimate:
    """Create one controlled physical estimate."""

    return ScalarEstimate(
        value=value,
        lower_bound=lower,
        upper_bound=upper,
        confidence=confidence,
        unit=unit,
        source=source,
        timestamp_seconds=TIMESTAMP_SECONDS,
        status=status,
        evidence=("illustrative software-validation input",),
    )


def geometry(source: str) -> GeometryAttributes:
    """Create controlled object dimensions from the named estimator."""

    def dimension(value: float, lower: float, upper: float) -> ScalarEstimate:
        return scalar(
            value,
            lower,
            upper,
            confidence=0.80,
            unit="m",
            source=source,
        )

    return GeometryAttributes(
        width_m=dimension(0.50, 0.45, 0.55),
        depth_m=dimension(0.40, 0.35, 0.45),
        height_m=dimension(1.00, 0.90, 1.10),
        center_height_m=dimension(0.50, 0.45, 0.55),
        fall_height_m=ScalarEstimate.unavailable(
            unit="m",
            source="support_and_stability_estimator_required",
            timestamp_seconds=TIMESTAMP_SECONDS,
        ),
    )


def main() -> None:
    """Run controlled prior, volume-density, fusion, and missing-data checks."""

    priors = ClassMassPriorTable(
        {
            "chair": ClassMassPrior(
                nominal_kg=6.0,
                lower_kg=3.0,
                upper_kg=12.0,
                confidence=0.35,
                source="controlled_demo_prior",
                evidence=("replace with cited or measured benchmark data",),
            )
        }
    )
    density = scalar(
        30.0,
        20.0,
        40.0,
        confidence=0.60,
        unit="kg/m^3",
        source="controlled_material_density_prior",
        status=EstimateStatus.PRIOR,
    )

    visible_lidar_result = estimate_track_mass(
        "chair",
        timestamp_seconds=TIMESTAMP_SECONDS,
        prior_table=priors,
        geometry=geometry("lidar_track_visible_envelope"),
        density_kg_per_m3=density,
    ).mass_kg

    learned_mass = scalar(
        7.0,
        4.0,
        11.0,
        confidence=0.70,
        unit="kg",
        source="controlled_rgbd_mass_model",
    )
    fused_result = estimate_track_mass(
        "chair",
        timestamp_seconds=TIMESTAMP_SECONDS,
        prior_table=priors,
        learned_mass=learned_mass,
        geometry=geometry("rgbd_complete_object_geometry"),
        density_kg_per_m3=density,
    ).mass_kg

    unknown_result = estimate_track_mass(
        "unknown_object",
        timestamp_seconds=TIMESTAMP_SECONDS,
        prior_table=priors,
    ).mass_kg

    checks = {
        "Visible envelope not used as volume": (
            visible_lidar_result.status is EstimateStatus.PRIOR
            and visible_lidar_result.value == 6.0
        ),
        "Independent evidence fused": fused_result.source == MASS_FUSION_SOURCE,
        "Fused bounds keep source hull": (
            fused_result.lower_bound <= 3.0
            and fused_result.upper_bound >= 12.0
        ),
        "Unknown mass remains unavailable": not unknown_result.is_available,
    }

    print("\n" + "=" * 72)
    print("CRA MASS-ESTIMATION VALIDATION REPORT")
    print("=" * 72)
    print("\nCase A - visible LiDAR envelope plus class prior:")
    print("  Geometry source:             lidar_track_visible_envelope")
    print("  Volume-density accepted:     False")
    print(f"  Selected mass:               {visible_lidar_result.value:.4f} kg")
    print(f"  Selected status:             {visible_lidar_result.status.value}")
    print(
        "  Selected interval:           "
        f"[{visible_lidar_result.lower_bound:.4f}, "
        f"{visible_lidar_result.upper_bound:.4f}] kg"
    )

    print("\nCase B - learned, complete RGB-D, density, and prior evidence:")
    print(f"  Fused nominal mass:          {fused_result.value:.4f} kg")
    print(
        "  Conservative interval hull: "
        f"[{fused_result.lower_bound:.4f}, "
        f"{fused_result.upper_bound:.4f}] kg"
    )
    print(f"  Fused confidence:            {fused_result.confidence:.4f}")
    print(f"  Result source:               {fused_result.source}")

    print("\nCase C - unknown class with no estimator evidence:")
    print(f"  Mass available:              {unknown_result.is_available}")
    print("  Numeric zero substituted:    False")

    print("\nControlled checks:")
    for name, passed in checks.items():
        print(f"  {'[PASS]' if passed else '[FAIL]'} {name}")

    print("\nInterpretation:")
    if all(checks.values()):
        print(
            "  [PASS] Mass evidence retained its provenance and uncertainty; "
            "partial geometry and missing evidence were handled explicitly."
        )
    else:
        print("  [FAIL] Inspect the failed controlled checks above.")
        raise RuntimeError("Mass-estimation demo failed.")
    print("  Demo values are illustrative and must not become paper parameters.")
    print("=" * 72)


if __name__ == "__main__":
    main()
