"""Summarize recorded multi-object evidence without Isaac or wheel motion."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robot.evaluation.object_map_validation import (
    summarize_fixture_assessments,
    bounded_fixture_checks,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--require-fixture-evidence", action="store_true")
    args = parser.parse_args()
    records = json.loads((args.source / "stationary-fields.json").read_text())
    monitor = json.loads((args.source / "fixture-monitor.json").read_text())
    for fixture in monitor["fixtures"]:
        fixture.setdefault(
            "center_world_xyz",
            [*fixture["center_world_xy"], fixture["radius_m"] + fixture["drop_m"]],
        )
    report = summarize_fixture_assessments(
        records, monitor["fixtures"], maximum_distance_m=0.25
    )
    checks = bounded_fixture_checks(report)
    report["bounded_fixture_checks"] = checks
    report["passed"] = bool(checks) and all(checks.values())
    (args.source / "object-validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for name, row in report["fixtures"].items():
        latest = row["latest"]
        print(
            json.dumps(
                {
                    "fixture": name,
                    "matched": row["uniquely_matched_updates"],
                    "ambiguous": row["ambiguous_updates"],
                    "visual": row["accepted_visual_updates"],
                    "phi_range": row["phi_range"],
                    "factors": latest["diagnostics"]["factor_scores"]
                    if latest
                    else None,
                    "mass": latest["attributes"]["mass"]["mass_kg"] if latest else None,
                    "fall_height": latest["attributes"]["geometry"]["fall_height_m"]
                    if latest
                    else None,
                    "heat": latest["attributes"]["thermal"]["heat_source_likelihood"]
                    if latest
                    else None,
                }
            )
        )
    print(json.dumps({"bounded_checks": checks, "passed": report["passed"]}))
    if args.require_fixture_evidence and not report["passed"]:
        raise RuntimeError(
            "bounded fixture evidence gate failed; numeric priors do not establish successful perception"
        )


if __name__ == "__main__":
    main()
