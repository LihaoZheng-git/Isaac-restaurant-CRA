"""Matched ORACLE crossing timing matrix; no perception or physical wheels."""

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
    CASES,
    run_encounter_control_case,
    summarize_oracle_rear_pass,
)
from robot.safety.checked_rear_pass_reference import CheckedRearPassReferenceConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onset", type=float, required=True)
    parser.add_argument("--case", choices=CASES, nargs="+", default=CASES)
    parser.add_argument(
        "--variant",
        choices=("primary", "uncorrected_wait", "checked_hybrid"),
        nargs="+",
        default=("primary", "uncorrected_wait", "checked_hybrid"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = CheckedRearPassReferenceConfig(existing_turn_heading_radians=0.2)
    source = (
        "robot/evaluation/encounter_control_benchmark.py",
        "robot/evaluation/checked_rear_pass_benchmark.py",
        "robot/evaluation/encounter_guidance.py",
        "robot/safety/checked_wait_transition.py",
        "robot/safety/checked_rear_pass_reference.py",
        "robot/safety/crossing_maneuver.py",
        "robot/navigation/waypoint_controller.py",
        "scripts/evaluation/compare_checked_crossing.py",
    )
    manifest = dict(
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_sha256={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source
        },
        onset_seconds=args.onset,
        cases=args.case,
        variants=args.variant,
        waypoint_mode="guides",
        waypoint_anchors=[2.2, 3.1, 3.9],
        primary_lambda=1.25,
        hybrid_reference_config=asdict(config),
        zero_wheels=True,
        scope=__doc__,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = []
    for case in args.case:
        for variant in args.variant:
            print(
                json.dumps(dict(starting=case, variant=variant, onset=args.onset)),
                flush=True,
            )
            extra = (
                dict(
                    control_factory=partial(
                        CheckedRearPassBenchmarkRuntime, rear_pass_config=config
                    ),
                    encounter_policy_adapter=partial(
                        rear_pass_policy_adapter, configuration=config
                    ),
                    checked_wait_transition=True,
                )
                if variant == "checked_hybrid"
                else {}
            )
            result = run_encounter_control_case(
                case,
                "fixed" if variant == "primary" else "encounter",
                actor_onset_seconds=args.onset,
                waypoint_mode="guides",
                **extra,
            )
            result["comparison_variant"] = variant
            result["rear_pass_assessment"] = (
                summarize_oracle_rear_pass(result) if case.startswith("cross") else None
            )
            transitions = [
                r["checked_wait_transition"]
                for r in result["rows"]
                if r["checked_wait_transition"]
            ]
            result["transition_summary"] = dict(
                proposed_wait_fields=len(transitions),
                rejected_wait_fields=sum(not t["accepted"] for t in transitions),
                accepted_wait_fields=sum(t["accepted"] for t in transitions),
                selected_yield_plans=sum(r["checked_yield"] for r in result["rows"]),
                original_source_pair_checks=all(
                    r["checked_wait_transition"]["source_timestamp_seconds"]
                    == r["field_source_timestamp_seconds"]
                    and r["checked_wait_transition"]["source_frame_id"]
                    == r["field_frame_id"]
                    and r["checked_wait_transition"][
                        "original_selection_timestamp_seconds"
                    ]
                    == r["original_selection_timestamp_seconds"]
                    and abs(
                        r["field_source_timestamp_seconds"]
                        - r["checked_wait_transition"][
                            "previous_accepted_source_timestamp_seconds"
                        ]
                        - 0.2
                    )
                    < 1e-8
                    for r in result["rows"]
                    if r["checked_wait_transition"]
                ),
            )
            (args.output / f"{case}-{variant}.json").write_text(
                json.dumps(result, indent=2)
            )
            small = {k: v for k, v in result.items() if k != "rows"}
            summary.append(small)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(small), flush=True)


if __name__ == "__main__":
    main()
