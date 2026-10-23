"""Matched ORACLE reference/stop-go/primary comparisons; no native wheels."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from robot.evaluation.checked_rear_pass_benchmark import (
    CheckedRearPassBenchmarkRuntime,
    rear_pass_policy_adapter,
)
from robot.evaluation.encounter_control_benchmark import (
    run_encounter_control_case,
    summarize_oracle_rear_pass,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor-onset-seconds", type=float, default=11.5)
    parser.add_argument(
        "--waypoint-layout", choices=("legacy", "before-crossing"), default="legacy"
    )
    parser.add_argument(
        "--controller",
        nargs="+",
        choices=("primary", "stop-go", "rear"),
        default=("rear",),
    )
    parser.add_argument(
        "--case",
        nargs="+",
        choices=("cross-left-right", "cross-right-left", "cross-then-parallel"),
        default=("cross-left-right", "cross-right-left", "cross-then-parallel"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = (
        "robot/safety/checked_rear_pass_reference.py",
        "robot/evaluation/checked_rear_pass_benchmark.py",
        "robot/evaluation/encounter_control_benchmark.py",
        "scripts/evaluation/compare_checked_rear_pass.py",
    )
    manifest = dict(
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_sha256={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
        },
        actor_onset_seconds=args.actor_onset_seconds,
        waypoint_layout=args.waypoint_layout,
        controllers=args.controller,
        zero_wheels=True,
        scope=__doc__,
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    summary = []
    for case in args.case:
        for controller in args.controller:
            print(
                json.dumps(dict(starting_case=case, controller=controller)), flush=True
            )
            extra = (
                dict(
                    control_factory=CheckedRearPassBenchmarkRuntime,
                    encounter_policy_adapter=rear_pass_policy_adapter,
                )
                if controller == "rear"
                else {}
            )
            result = run_encounter_control_case(
                case,
                "fixed" if controller == "primary" else "encounter",
                actor_onset_seconds=args.actor_onset_seconds,
                waypoint_layout=args.waypoint_layout,
                **extra,
            )
            result["experimental_controller"] = controller
            result["rear_pass_assessment"] = summarize_oracle_rear_pass(result)
            (args.output / f"{case}-{controller}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            small = {k: v for k, v in result.items() if k != "rows"}
            summary.append(small)
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(json.dumps(small), flush=True)


if __name__ == "__main__":
    main()
