"""Read-only summary of new native chapters; independent truth only evaluates."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from robot.evaluation.parallel_smoothing import summarize_parallel_smoothness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_directory
    rows = []
    for name in ("static", "parallel", "crossing"):
        source = folder / (name + "-motion-v6")
        if not (source / "report.json").exists():
            rows.append(dict(chapter=name, status="pending"))
            continue
        report = json.loads((source / "report.json").read_text())
        execution = (
            json.loads((source / "execution.json").read_text())
            if (source / "execution.json").exists()
            else {}
        )
        row = dict(
            chapter=name,
            passed=report["passed"]
            and execution.get("passed", False)
            and execution.get("source_unchanged", False),
            reason=report["reason"],
            source=str(source),
            report_sha256=hashlib.sha256(
                (source / "report.json").read_bytes()
            ).hexdigest(),
        )
        for key in (
            "waypoints_reached",
            "navigation_simulation_seconds",
            "minimum_clearance_m",
            "stop_drift_m",
            "minimum_cbf_margin",
            "live_field_count",
            "perception_evidence_counts",
            "all_object_policy_counts",
            "fixed_lambda",
        ):
            row[key] = report.get(key)
        human = report.get("human_navigation_validation") or {}
        row["supported_human_kinetic_rows"] = human.get("supported_moving_kinetic_rows")
        row["nearby_curve_m"] = human.get("parallel_curve", {}).get(
            "nearby_walking_excursion_m"
        )
        row["wall_seconds"] = execution.get("wall_seconds")
        if name == "parallel":
            truth = json.loads(
                (source / "human-truth-evaluation-only.json").read_text()
            )
            row["actual_smoothness"] = summarize_parallel_smoothness(report, truth)
        rows.append(row)
    result = dict(
        chapters=rows,
        all_chapters_passed=all(r.get("passed", False) for r in rows),
        scope="Separate new-scene prior-assisted trials. Not a continuous mission, sensor-only four-factor perception, matched baseline study or real-time result.",
    )
    (folder / "chapter-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
