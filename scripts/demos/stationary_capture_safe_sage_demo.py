"""Replay measured chair geometry and stored physical evidence into Safe-SAGE.

This numerical field diagnostic preserves captured track IDs. It does not
command motion or claim four-factor perception acceptance. No evidence from
other objects/component demonstrations is spliced into the captured objects.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robot.evaluation.factor_specific_cra import diagnose_factor_specific_cra
from robot.perception.physical_attributes import CategoricalEstimate, PhysicalAttributes, ScalarEstimate
from robot.risk.physical_profile_provider import PhysicalAttributeCRARiskProfileProvider
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy
from robot.safety.cra_track_policy import resolve_cra_track_policy
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig, build_local_occupancy_grid
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.safe_sage_field_pipeline import build_safe_sage_field_pipeline
from robot.safety.semantic_obstacle import SemanticObstacle
from scripts.demos.cra_safe_sage_runtime_demo import make_runtime


def decode_attributes(payload):
    base = PhysicalAttributes.unknown(track_id=payload["track_id"],
        timestamp_seconds=payload["timestamp_seconds"], semantic_class=payload["semantic_class"],
        semantic_source=payload["semantic_source"], semantic_confidence=payload["semantic_confidence"])
    groups = {}
    for name in ("mass", "geometry", "motion", "contact", "thermal", "stability"):
        items = {key: (ScalarEstimate(**value) if "unit" in value else CategoricalEstimate(**value))
                 for key, value in payload[name].items()}
        groups[name] = replace(getattr(base, name), **items)
    return replace(base, **groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    if not capture["capture_passed"]:
        raise ValueError("input capture failed stationarity/sensor checks")
    inputs_path = Path(capture["artifact_paths"]["sensor_inputs"])
    inputs = np.load(inputs_path)
    # Explicitly select the matched objects with mass evidence, rather than
    # assigning semantic classes or inferred mass to the remaining clusters.
    rows = [row for row in capture["tracks"] if row["sensor_mass_available"]]
    if not rows:
        raise ValueError("no mass-estimated sensor track is available")
    attributes = {row["track_id"]: decode_attributes(row["physical_attributes"]) for row in rows}
    timestamp = max(item.timestamp_seconds for item in attributes.values())
    provider = PhysicalAttributeCRARiskProfileProvider(attributes, timestamp_seconds=timestamp)
    obstacles = []
    for row in rows:
        track_id = row["track_id"]
        item = attributes[track_id]
        points = inputs["obstacle_points_robot_xyz"][inputs["point_track_ids"] == track_id, :2]
        center = tuple(points.mean(axis=0).tolist())
        # Positions are local capture coordinates. This field-only diagnostic
        # anchors them to an identity frame; no world trajectory is inferred.
        track = TrackedObstacle(track_id, center, center, (0., 0.), (0., 0.), len(points), 1, False)
        obstacles.append(SemanticObstacle(track, item.semantic_class, item.semantic_confidence, item.semantic_source))
    policy = resolve_cra_track_policy(obstacles, profile_provider=provider, policy=CRASafeSAGEPolicy())
    runtime = make_runtime()
    selected = np.isin(inputs["point_track_ids"], tuple(attributes))
    grid_config = LocalOccupancyGridConfig(resolution=.25, minimum_x=-2., maximum_x=5., minimum_y=-3., maximum_y=3.)
    grid = build_local_occupancy_grid(inputs["obstacle_points_robot_xyz"][selected, :2],
        point_track_ids=inputs["point_track_ids"][selected], semantic_obstacles=obstacles,
        track_safety_parameters=policy.safety_parameters_by_track_id,
        class_safety_policy=runtime.class_policy, config=grid_config)
    field = build_safe_sage_field_pipeline(grid, guidance_policy=runtime.guidance_policy,
        config=runtime.field_config, semantic_obstacles=obstacles,
        track_guidance_parameters=policy.guidance_parameters_by_track_id)
    finite = bool(np.isfinite(field.h_full.values).all())
    fixed_lambda = all(item.policy_result.tangential_strength == 1.25 for item in policy.assignments)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"passed": finite and fixed_lambda, "source_capture_sha256": hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        "sensor_inputs_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
        "selected_track_ids": list(attributes), "scope": "saved sensor capture to numerical field; no motion",
        "aggregate_sensor_acceptance": False, "unknown_evidence_policy": "unchanged conservative fallback",
        "fixed_lambda": 1.25, "finite_field": finite, "field_shape": list(field.h_full.values.shape),
        "field_minimum": float(field.h_full.values.min()), "field_maximum": float(field.h_full.values.max()),
        "policy": policy.as_dict(), "factors": {track_id: diagnose_factor_specific_cra(result).to_dict()
            for track_id, result in provider.diagnostics_by_track_id.items()},
        "limitations": ["saved capture replay, not new closed-loop trial", "unknown factors can saturate phi",
                        "mass retains existing uncalibrated sensor-domain bounds", "world trajectory/velocity not reconstructed"]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output / "field.npz", h_full=field.h_full.values)
    fig, axis = plt.subplots(figsize=(7, 5))
    shown = axis.imshow(field.h_full.values[0], origin="lower", extent=(-2, 5, -3, 3), cmap="coolwarm")
    xy = inputs["obstacle_points_robot_xyz"][selected, :2]
    axis.scatter(xy[:, 0], xy[:, 1], s=7, c="black", label="captured LiDAR")
    axis.set(xlabel="Robot-frame x (m)", ylabel="Robot-frame y (m)",
             title="Stationary sensor replay → CRA → Safe-SAGE\nunknown factors retained; λ = 1.25")
    axis.legend()
    fig.colorbar(shown, ax=axis, label="h_full, yaw slice 0")
    fig.tight_layout()
    fig.savefig(args.output / "field.png", dpi=140)
    plt.close(fig)
    print(json.dumps({key: value for key, value in report.items() if key not in {"policy", "factors"}}, indent=2))
    if not report["passed"]:
        raise RuntimeError("stationary capture-to-field validation failed")


if __name__ == "__main__":
    main()
