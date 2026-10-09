"""Summarize actual category-free stationary captures without supplying scores."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot.perception.sam_instance_segmentation import SAM_INSTANCE_SOURCE
from robot.perception.visual_mass_runtime import (
    CATEGORY_FREE_MASS_EVIDENCE,
    VISUAL_MASS_RUNTIME_SOURCE,
)

FACTOR_NAMES = ("potential", "kinetic", "contact", "thermal")


def summarize_capture(report, mass_dataflow, sam_record):
    """Validate the recorded evidence path; acceptance remains bounded to it."""
    reasons = []
    for key in (
        "stationary_passed",
        "mapping_passed",
        "sensor_dataflow_passed",
        "sam_instance_proposal_enabled",
        "category_free_visual_mass_enabled",
        "instance_mask_geometry_enabled",
        "root_layer_unchanged_by_fixture",
    ):
        if report.get(key) is not True:
            reasons.append(f"{key} did not pass")
    if report.get("sphere_density_prior_kg_m3") is not None or report.get("profiles"):
        reasons.append("supplied density or oracle factor profiles are present")
    if report.get("fixed_lambda") != 1.25:
        reasons.append("primary lambda changed")
    attributes = report["sensor_attributes"]
    mass = attributes["mass"]["mass_kg"]
    if (
        attributes["semantic_source"] != SAM_INSTANCE_SOURCE
        or mass["source"] != VISUAL_MASS_RUNTIME_SOURCE
        or mass["status"] != "inferred"
        or CATEGORY_FREE_MASS_EVIDENCE not in mass["evidence"]
        or not mass_dataflow.get("live_dataflow_valid")
    ):
        reasons.append("category-free live mass provenance did not pass")
    factors = report["sensor_factors"]["factor_specific"]
    unavailable = [name for name in FACTOR_NAMES if not factors[name]["available"]]
    statuses = {
        item["status"] for name in FACTOR_NAMES for item in factors[name]["inputs"]
    }
    actual_mode = "prior" if "prior" in statuses else "sensor"
    if "oracle" in statuses or "fallback" in statuses:
        reasons.append("runtime factors contain oracle or fallback input provenance")
    readiness = report.get(
        "runtime_perception_readiness", report["prior_assisted_perception_readiness"]
    )
    if (
        not readiness["ready"]
        or readiness["evidence_mode"] != actual_mode
        or readiness["scope"] != "four_factor"
        or unavailable
    ):
        reasons.append(
            "complete four-factor readiness did not pass in the actual evidence mode"
        )
    if not report["motion_gate_passed"]:
        reasons.append("stationary motion gate did not pass")
    phi = report["sensor_phi"]
    mapped = report["policies"]["observed"]["safe_sage"]
    if (
        not math.isclose(phi, mapped["risk"], abs_tol=1e-12)
        or mapped["tangential_strength"] != 1.25
    ):
        reasons.append("computed phi or fixed lambda was changed before field mapping")
    observed_miss = (
        not any(
            item["overlap_fraction"] >= 0.5
            for item in sam_record.get("prior_detection_mask_overlaps", [])
        )
        if "prior_detection_mask_overlaps" in sam_record
        else None
    )
    if sam_record["rejection_reasons"] or sam_record["source"] != SAM_INSTANCE_SOURCE:
        reasons.append("SAM instance was rejected")
    fall = attributes["geometry"]["fall_height_m"]
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "category": attributes["semantic_class"],
        "evidence_mode": actual_mode,
        "yolo_mask_miss_on_sam_instance": observed_miss,
        "rough_mass_kg": mass["value"],
        "mass_confidence": mass["confidence"],
        "fall_height_m": fall["value"],
        "fall_height_status": fall["status"],
        "relative_speed_mps": attributes["motion"]["relative_speed_mps"]["value"],
        "phi": phi,
        "nominal_diagnostic_phi": report["sensor_factors"]["aggregate"]["nominal_phi"],
        "factor_specific": {
            name: {
                key: factors[name][key]
                for key in (
                    "available",
                    "nominal_risk",
                    "conservative_risk",
                    "confidence",
                    "basis",
                    "rejection_reasons",
                )
            }
            for name in FACTOR_NAMES
        },
        "unavailable_factors": unavailable,
        "unassessed_occupancy_return_count": report[
            "unassessed_occupancy_return_count"
        ],
        "scope": "stationary RGB/LiDAR proxy dataflow; not independent hazard accuracy or navigation validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", required=True, help="label=actual capture directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for specification in args.case:
        label, directory = specification.split("=", 1)
        path = Path(directory)
        report_path = path / "stationary.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        frame_id = json.loads((path / "vision.json").read_text(encoding="utf-8"))[
            "camera_info"
        ]["read_index"]
        mass_path = path / "live-mass-dataflow.json"
        sam_path = path / f"sam-mask-frame-{frame_id}.json"
        row = summarize_capture(
            report,
            json.loads(mass_path.read_text(encoding="utf-8")),
            json.loads(sam_path.read_text(encoding="utf-8")),
        )
        row["artifacts"] = {
            p.name: {
                "path": str(p.resolve()),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in (report_path, mass_path, sam_path)
        }
        rows[label] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cases": rows,
        "all_cases_accepted": all(row["accepted"] for row in rows.values()),
        "at_least_one_actual_yolo_mask_miss": any(
            row["yolo_mask_miss_on_sam_instance"] is True for row in rows.values()
        ),
        "scope": "configured category-free fixtures; ground/fall interaction priors remain explicit",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4), squeeze=False)
    for ax, (label, row) in zip(axes[0], rows.items()):
        x = np.arange(4)
        factors = row["factor_specific"]
        ax.bar(
            x - 0.18,
            [
                factors[name]["nominal_risk"] if factors[name]["available"] else np.nan
                for name in FACTOR_NAMES
            ],
            0.36,
            label="Nominal proxy diagnostic",
        )
        ax.bar(
            x + 0.18,
            [factors[name]["conservative_risk"] for name in FACTOR_NAMES],
            0.36,
            label="Conservative controller factor",
        )
        ax.set_xticks(x, FACTOR_NAMES)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"{label}: phi={row['phi']:.5f}\n{row['evidence_mode']} / accepted={row['accepted']}"
        )
        ax.set_ylabel("Risk score")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in ("all_cases_accepted", "at_least_one_actual_yolo_mask_miss")
            }
        )
    )


if __name__ == "__main__":
    main()
