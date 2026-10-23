"""Matched ORACLE reference/stop-go/primary comparisons; no native wheels."""

import argparse
from dataclasses import asdict
from functools import partial
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
from robot.safety.checked_rear_pass_reference import CheckedRearPassReferenceConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor-onset-seconds", type=float, default=11.5)
    parser.add_argument(
        "--waypoint-layout", choices=("legacy", "before-crossing"), default="legacy"
    )
    parser.add_argument(
        "--waypoint-mode", choices=("checkpoints", "guides"), default="checkpoints"
    )
    parser.add_argument("--rear-minimum-lead", type=float, default=1.4)
    parser.add_argument("--rear-existing-turn-heading", type=float)
    parser.add_argument("--rear-minimum-first-translation", type=float, default=0.)
    parser.add_argument("--checked-wait-transition", action="store_true")
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
    rear_config = CheckedRearPassReferenceConfig(
        minimum_lead_m=args.rear_minimum_lead,
        existing_turn_heading_radians=args.rear_existing_turn_heading,
        minimum_first_translation_m=args.rear_minimum_first_translation,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    paths = (
        "robot/safety/checked_rear_pass_reference.py",
        "robot/evaluation/checked_rear_pass_benchmark.py",
        "robot/evaluation/encounter_control_benchmark.py",
        "robot/navigation/waypoint_controller.py",
        "robot/safety/checked_wait_transition.py",
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
        waypoint_mode=args.waypoint_mode,
        rear_pass_configuration=asdict(rear_config),
        checked_wait_transition=args.checked_wait_transition,
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
                    control_factory=partial(
                        CheckedRearPassBenchmarkRuntime, rear_pass_config=rear_config
                    ),
                    encounter_policy_adapter=partial(
                        rear_pass_policy_adapter, configuration=rear_config
                    ),
                )
                if controller == "rear"
                else {}
            )
            result = run_encounter_control_case(
                case,
                "fixed" if controller == "primary" else "encounter",
                actor_onset_seconds=args.actor_onset_seconds,
                waypoint_layout=args.waypoint_layout,
                waypoint_mode=args.waypoint_mode,
                checked_wait_transition=args.checked_wait_transition
                and controller != "primary",
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
