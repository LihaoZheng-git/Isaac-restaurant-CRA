"""Compare actual ground/raised reports; never supply or equalize factor scores."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def matched_geometry_fields(ground, raised, capture_path, output_directory):
    """Replay the computed policies on identical measured instance occupancy.

    This diagnostic excludes support/environment explicitly and never commands
    wheels. It cannot establish physical raised-platform navigation or isolate
    potential from other measured-factor/confidence differences.
    """
    import numpy as np
    from robot.safety.class_safety_policy import (
        ClassSafetyParameters,
        ClassSafetyPolicy,
    )
    from robot.safety.local_occupancy_grid import (
        LocalOccupancyGridConfig,
        build_local_occupancy_grid,
    )
    from robot.safety.obstacle_tracking import TrackedObstacle
    from robot.safety.poisson_safety_function import PoissonSafetyFunctionConfig
    from robot.safety.safe_sage_configuration_space import (
        RectangularRobotFootprint,
        SafeSAGEConfigurationSpaceConfig,
    )
    from robot.safety.safe_sage_field_pipeline import (
        SafeSAGEFieldPipelineConfig,
        build_safe_sage_field_pipeline,
    )
    from robot.safety.safe_sage_guidance_policy import (
        SafeSAGEGuidanceParameters,
        SafeSAGEGuidancePolicy,
    )
    from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
    from robot.safety.semantic_obstacle import SemanticObstacle
    from robot.safety.tangential_laplace_guidance_field import (
        TangentialLaplaceGuidanceFieldConfig,
    )

    points = np.load(capture_path)["points_anchor"]
    settings = ground["field_configuration"]
    cs = settings["configuration_space_config"]
    hf = settings["h_full_config"]
    config = SafeSAGEFieldPipelineConfig(
        SafeSAGEConfigurationSpaceConfig(
            RectangularRobotFootprint(**cs["footprint"]), cs["yaw_slice_count"]
        ),
        TangentialLaplaceGuidanceFieldConfig(**settings["guidance_field_config"]),
        SafeSAGEHFullConfig(
            **{
                **hf,
                "poisson_config": PoissonSafetyFunctionConfig(**hf["poisson_config"]),
            }
        ),
    )
    center = tuple(np.mean(points, axis=0))
    semantic = SemanticObstacle(
        TrackedObstacle(
            0, center, center, (0.0, 0.0), (0.0, 0.0), len(points), 12, True
        ),
        "unknown",
        0.5,
        "matched_measured_instance_policy_replay",
    )
    fields, rows = {}, {}
    for name, report in (("ground", ground), ("raised", raised)):
        source = report["policies"]["observed"]["safe_sage"]
        if (
            source["risk"] != report["sensor_phi"]
            or source["tangential_strength"] != 1.25
        ):
            raise ValueError(
                "policy replay requires unchanged observed phi and fixed lambda"
            )
        parameters = SafeSAGEGuidanceParameters(
            ClassSafetyParameters(source["risk"], source["boundary_flux_magnitude"]),
            source["tangential_strength"],
            source["passing_side"],
        )
        policy = SafeSAGEGuidancePolicy({"unknown": parameters})
        grid = build_local_occupancy_grid(
            points,
            point_track_ids=np.zeros(len(points), dtype=int),
            semantic_obstacles=(semantic,),
            class_safety_policy=ClassSafetyPolicy(
                {"unknown": parameters.safety_parameters}
            ),
            config=LocalOccupancyGridConfig(
                **{
                    key: value
                    for key, value in ground["grid_configuration"].items()
                    if not key.startswith("_")
                }
            ),
            track_safety_parameters={0: parameters.safety_parameters},
        )
        field = build_safe_sage_field_pipeline(
            grid,
            guidance_policy=policy,
            config=config,
            semantic_obstacles=(semantic,),
            track_guidance_parameters={0: parameters},
        )
        fields[name] = field.h_full.values
        np.savez_compressed(
            output_directory / f"matched-{name}-field.npz", h_full=field.h_full.values
        )
        rows[name] = {
            "computed_phi": source["risk"],
            "boundary_flux_magnitude": source["boundary_flux_magnitude"],
            "occupied_risks": np.unique(grid.cell_risks[grid.occupied_mask]).tolist(),
        }
    difference = float(np.max(np.abs(fields["raised"] - fields["ground"])))
    return {
        "scope": "offline computed-policy replay on the same observed ground-ball instance geometry; no wheels; support/environment excluded",
        "policies": rows,
        "maximum_absolute_h_full_difference": difference,
        "passed": difference > 1e-8
        and all(np.isfinite(value).all() for value in fields.values()),
        "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        "does_not_establish": "raised-platform physical safety or causal potential-only clearance ordering",
    }


def compare_reports(ground, raised):
    """Bounded matching tolerances declared independently of risk aggregation."""
    rows = {}
    for name, report in (("ground", ground), ("raised", raised)):
        rows[name] = {
            "passed": report["passed"],
            "four_factor_prior_ready": report["aggregate_prior_assisted_acceptance"],
            "fall_height": report["sensor_attributes"]["geometry"]["fall_height_m"],
            "mass": report["sensor_attributes"]["mass"]["mass_kg"],
            "factors": {
                key: value["nominal_risk"]
                for key, value in report["sensor_factors"]["factor_specific"].items()
            },
            "conservative_factors": {
                key: value["conservative_risk"]
                for key, value in report["sensor_factors"]["factor_specific"].items()
            },
            "phi": report["sensor_phi"],
            "field_risk_cells": report.get("field_risk_cells"),
            "motion": {
                key: {
                    field: value[field]
                    for field in (
                        "passed",
                        "route_complete",
                        "minimum_clearance_m",
                        "simulated_seconds",
                    )
                }
                for key, value in report["motion"].items()
            },
        }
    low, high = rows["ground"], rows["raised"]
    other_deltas = {
        factor: abs(high["factors"][factor] - low["factors"][factor])
        for factor in ("kinetic", "contact", "thermal")
    }
    mass_difference = abs(high["mass"]["value"] - low["mass"]["value"]) / max(
        high["mass"]["value"], low["mass"]["value"], 1e-9
    )
    checks = {
        "both_stationary_and_readiness_gates": all(
            report["stationary_passed"]
            and report["motion_gate_passed"]
            and report["aggregate_prior_assisted_acceptance"]
            and not report["sensor_factors"]["aggregate"]["unknown_fallback_factors"]
            for report in (ground, raised)
        ),
        "matched_radius_and_density_prior": ground["authored_fixture_radius_m"]
        == raised["authored_fixture_radius_m"]
        and ground["sphere_density_prior_kg_m3"]
        == raised["sphere_density_prior_kg_m3"],
        "matched_instance_method": ground["instance_mask_geometry_enabled"]
        and raised["instance_mask_geometry_enabled"]
        and ground["chromatic_instance_proposal_enabled"]
        == raised["chromatic_instance_proposal_enabled"],
        "fixed_lambda": ground["fixed_lambda"] == raised["fixed_lambda"] == 1.25,
        "scene_files_preserved": ground["scene_file_unchanged"]
        and raised["scene_file_unchanged"],
        "raised_support_retained": raised["unassessed_occupancy_return_count"] > 0
        and raised["unassessed_occupancy_phi"] == 1.0
        and len(raised["independent_scoring_disks"]) >= 2,
        "estimated_drop_ordering": high["fall_height"]["value"]
        > low["fall_height"]["value"],
        "potential_factor_ordering": high["factors"]["potential"]
        > low["factors"]["potential"],
        "aggregate_phi_ordering": high["phi"] > low["phi"],
        "other_nominal_factors_within_0_10": max(other_deltas.values()) <= 0.10,
        "rough_mass_within_25_percent": mass_difference <= 0.25,
    }
    return {
        "trials": rows,
        "checks": checks,
        "paired_perception_passed": all(checks.values()),
        "other_nominal_factor_absolute_differences": other_deltas,
        "relative_mass_difference": mass_difference,
        "both_wheel_passes_completed": all(
            report["motion"].get("observed", {}).get("passed", False)
            and report["motion"]["observed"]["route_complete"]
            and report["motion"]["observed"]["stopping"]["passed"]
            and report["motion"]["observed"]["motion_mode"]
            == "Isaac Nova Carter physical wheel commands"
            for report in (ground, raised)
        ),
        "limitations": [
            "prior-assisted mass and harmful-fall interaction; ground continuity is a prior",
            "different platform occupancy confounds a causal risk-to-clearance comparison",
            "unassessed support/environment uses conservative phi=1 and can dominate overlapping 2D cells",
            "stationary estimates and field are frozen during wheel trials; no real-time claim",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground", type=Path, required=True)
    parser.add_argument("--raised", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-motion", action="store_true")
    parser.add_argument("--matched-ground-capture", type=Path)
    args = parser.parse_args()
    ground, raised = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.ground, args.raised)
    ]
    result = compare_reports(ground, raised)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.matched_ground_capture is not None:
        result["matched_geometry_policy_replay"] = matched_geometry_fields(
            ground, raised, args.matched_ground_capture, args.output.parent
        )
    result["report_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (("ground", args.ground), ("raised", args.raised))
    }
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "checks",
                    "paired_perception_passed",
                    "both_wheel_passes_completed",
                )
            },
            indent=2,
        )
    )
    if (
        not result["paired_perception_passed"]
        or (args.require_motion and not result["both_wheel_passes_completed"])
        or not result.get("matched_geometry_policy_replay", {"passed": True})["passed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
