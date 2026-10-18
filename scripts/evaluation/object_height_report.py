"""Replay and plot actual stationary height-component scans, without wheels.

Input directories are outputs of the explicitly selected general-height mode
in lidar_fall_height_isaac_diagnostic.py. Fixture selection and oracle frame
references remain declared; these are not autonomous object-discovery trials.
"""

import argparse
from dataclasses import asdict
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

from robot.perception.lidar_fall_surfaces import LidarFallSurfaceConfig
from robot.perception.object_height import ObjectHeightConfig, estimate_object_height
from robot.perception.physical_attributes import PhysicalAttributes
from robot.risk.object_risk_policy import ObjectRiskPriorConfig, evaluate_object_risk
from robot.risk.physical_factor_adapter import adapt_physical_factors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, clouds = [], []
    for path in args.trial:
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        if report.get("diagnostic_mode") != "general_object_height":
            raise ValueError(
                "input must be an explicit general-object height diagnostic"
            )
        with np.load(path / "points.npz", allow_pickle=False) as data:
            scene, obj = data["scene_points_robot_xyz"], data["object_points_robot_xyz"]
        with np.load(path / "scene_points.npz", allow_pickle=False) as data:
            reference = float(data["floor_height_robot_m"])
        original = report["object_height_evidence"]
        timestamp = original["ground_height_m"]["timestamp_seconds"]
        result = estimate_object_height(
            scene,
            obj,
            timestamp_seconds=timestamp,
            scene_timestamp_seconds=timestamp,
            floor_reference_m=reference,
            reference_scope=original["reference_scope"],
            config=ObjectHeightConfig(
                surface_config=LidarFallSurfaceConfig(
                    **report["experimental_surface_config"]
                )
            ),
        )
        support = result.support_height_above_ground_m
        oracle = report["oracle_fall_height_m"]
        expected = "ground_supported" if oracle == 0 else "elevated_supported"
        checks = {
            "original_height_gate_passed": report["passed"],
            "stationary_zero_wheels": report["robot_motion_command"] == "zero"
            and report["robot_position_drift_m"] <= 0.005,
            "source_scene_unchanged": report[
                "scene_file_and_fixture_root_layer_unchanged"
            ],
            "support_state": result.support_state == expected,
            "support_height_error": support.is_available
            and abs(support.value - oracle) <= 0.1,
            "support_interval_coverage": support.is_available
            and support.lower_bound <= oracle <= support.upper_bound,
            "replay_matches_recorded_support": result.support_state
            == original["support_state"]
            and support.value == original["support_height_above_ground_m"]["value"],
        }
        # Explicitly missing mass/interaction: this illustrates the prior label,
        # not a four-factor result or a controlled potential-energy comparison.
        adapted = adapt_physical_factors(
            PhysicalAttributes.unknown(track_id=0, timestamp_seconds=timestamp),
            timestamp_seconds=timestamp,
        )
        _, policies = evaluate_object_risk(
            adapted, ObjectRiskPriorConfig(), height_evidence=result
        )
        rows.append(
            {
                "path": str(path.resolve()),
                "passed": all(checks.values()),
                "checks": checks,
                "height_evidence": asdict(result),
                "oracle_support_elevation_m": oracle,
                "robot_position_drift_m": report["robot_position_drift_m"],
                "potential_prior_label_only": policies["potential"],
                "policy_scope": "mass/interaction deliberately unavailable; .25 policy demonstration only, no risk ordering claim",
                "source_sha256": {
                    name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                    for name in ("report.json", "points.npz", "scene_points.npz")
                },
            }
        )
        clouds.append((scene, obj, reference))

    fig, axes = plt.subplots(
        1, len(rows) + 1, figsize=(5 * (len(rows) + 1), 4.6), squeeze=False
    )
    for index, (row, (scene, obj, reference)) in enumerate(zip(rows, clouds)):
        ax = axes[0, index]
        xy = np.median(obj[:, :2], axis=0)
        selected = scene[
            (np.linalg.norm(scene[:, :2] - xy, axis=1) < 2.0)
            & (scene[:, 2] - reference < 1.2)
        ]
        ax.scatter(
            selected[:, 0],
            selected[:, 2] - reference,
            s=1,
            alpha=0.2,
            color="gray",
            label="actual scene returns",
        )
        ax.scatter(
            obj[:, 0],
            obj[:, 2] - reference,
            s=9,
            color="tab:orange",
            label="selected target returns",
        )
        support = row["height_evidence"]["support_height_m"]["value"]
        if support is not None:
            ax.axhline(
                support - reference,
                color="tab:blue",
                lw=1.5,
                label="inferred support elevation",
            )
        else:
            ax.text(0.03, 0.06, "support unavailable", transform=ax.transAxes)
        ax.set(
            title=f"{row['height_evidence']['support_state']}\nzero-wheel Isaac scan",
            xlabel="Chassis X (m)",
            ylabel="Z above declared floor reference (m)",
            ylim=(-0.08, 1.1),
        )
        ax.legend(fontsize=8, loc="upper right")
    ax = axes[0, -1]
    for i, row in enumerate(rows):
        h = row["height_evidence"]["support_height_above_ground_m"]
        if h["value"] is not None:
            ax.errorbar(
                i,
                h["value"],
                yerr=[[h["value"] - h["lower_bound"]], [h["upper_bound"] - h["value"]]],
                fmt="o",
                color="tab:blue",
                capsize=5,
                label="inferred support + bounds" if i == 0 else None,
            )
        else:
            ax.text(i, 0.05, "unknown", ha="center")
        ax.scatter(
            i + 0.12,
            row["oracle_support_elevation_m"],
            marker="x",
            color="black",
            label="fixture truth (scoring only)" if i == 0 else None,
        )
    ax.set(
        xticks=range(len(rows)),
        xticklabels=[
            r["height_evidence"]["support_state"].replace("_supported", "")
            for r in rows
        ],
        ylabel="Support elevation above fitted ground (m)",
        title="Support is separate from drop height",
    )
    ax.legend(fontsize=8)
    fig.text(
        0.5,
        0.015,
        "Actual stationary LiDAR data; configured target selection and oracle frame reference. Visible center/base ≠ fall height.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.output / "stationary-height.png", dpi=150)
    plt.close(fig)
    payload = {
        "passed": all(r["passed"] for r in rows),
        "trials": rows,
        "scope": "stationary replay of actual controlled sensor scans; no motion/carry association/complete four-factor acceptance",
    }
    (args.output / "report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": payload["passed"], "trial_count": len(rows)}, indent=2))
    if not payload["passed"]:
        raise RuntimeError("one or more original/replay height gates failed")


if __name__ == "__main__":
    main()
