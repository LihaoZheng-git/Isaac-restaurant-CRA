"""Explicit ORACLE control matrix or recorded-sensor policy-only replay; zero wheels."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robot.evaluation.encounter_control_benchmark import (
    CASES,
    VARIANTS,
    run_encounter_control_case,
)
from robot.evaluation.encounter_guidance import apply_encounter_guidance
from robot.safety.class_safety_policy import ClassSafetyParameters
from robot.safety.encounter_motion import EncounterMotionClassifier
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidanceParameters


def sensor_policy_replay(path):
    classifier = EncounterMotionClassifier()
    results = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            d = r["diagnostics"]
            t = r["sensor_timestamp"]
            identities = d.get("human_body_envelope_priors", [])
            objects = {o["track_id"]: o for o in r["objects"]}
            tracks = []
            for identity in identities:
                key = identity["track_id"]
                o = objects[key]["diagnostics"]
                tracks.append(
                    NS(
                        track_id=key,
                        centroid_world_xy=o["centroid_world_xy"],
                        velocity_world_xy=o["velocity_world_xy"],
                        point_count=identity["current_support_count"],
                        observation_count=o["observation_count"],
                        matched_previous_track=o["observation_count"] >= 2,
                    )
                )
            passing = d.get("motion_aware_passing") or {}
            # Actual world waypoint is not stored in old field logs. Recover
            # it from the saved nominal route and active command checkpoints.
            cmd = next(
                (
                    c
                    for c in reversed(commands)
                    if c["time"] <= t - navigation_start + 1e-6
                ),
                commands[0],
            )
            goal = world_route[cmd.get("target_waypoint_index", 0)]
            classification = classifier.update(
                NS(timestamp_seconds=t, tracked_obstacles=tuple(tracks)),
                robot_xy=world_state(r["state_at_commit"])[:2],
                goal_xy=goal,
                human_track_ids=tuple(x.track_id for x in tracks),
            )
            audit = [
                dict(row, **{"lambda": 1.25}) for row in d["applied_field_policies"]
            ]
            guidance = {
                row["track_or_group_id"]: SafeSAGEGuidanceParameters(
                    ClassSafetyParameters(
                        row["applied_field_risk"], -row["boundary_flux"]
                    ),
                    1.25,
                    row.get("applied_passing_side", "right"),
                )
                for row in audit
            }
            changed, _, ids = apply_encounter_guidance(
                guidance, audit, classification, passing, source_time=t
            )
            results.append(
                dict(
                    time=t,
                    classification=classification,
                    yield_track_ids=ids,
                    applied_lambdas={
                        k: v.tangential_strength for k, v in changed.items()
                    },
                )
            )
    return dict(
        scope="recorded RGB/LiDAR-derived track policy-only replay; tracker matched flag reconstructed from observation count; waypoint context recovered from saved command/anchor clocks; no recomputed field, controller, fresh perception or wheel acceptance",
        records=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensor-replay", type=Path)
    parser.add_argument("--case", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--variant", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--maximum-seconds", type=float, default=40.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                source_revision=revision,
                zero_wheels=True,
                oracle_control_matrix=args.sensor_replay is None,
                sensor_policy_replay=str(args.sensor_replay),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.sensor_replay is not None:
        # These globals are confined to offline reconstruction, never runtime.
        import numpy as np

        global commands, navigation_start, world_route, world_state
        folder = args.sensor_replay.parent
        commands = json.loads((folder / "commands.json").read_text(encoding="utf-8"))
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        anchor = json.loads(
            (folder / "fixture-monitor.json").read_text(encoding="utf-8")
        )["anchor_world_xy_yaw"]
        c, s = np.cos(anchor[2]), np.sin(anchor[2])
        rot = np.array([[c, -s], [s, c]])
        world_route = np.array(report["nominal_route_local_xy"]) @ rot.T + anchor[:2]
        with args.sensor_replay.open(encoding="utf-8") as stream:
            first = json.loads(stream.readline())
        first_command = next(
            c
            for c in commands
            if c.get("generation") == 1 and c.get("field_age") is not None
        )
        navigation_start = (
            first["sensor_timestamp"]
            + first_command["field_age"]
            - first_command["time"]
        )

        def world_state(state):
            return np.array(state[:2]) @ rot.T + anchor[:2]

        result = sensor_policy_replay(args.sensor_replay)
        (args.output / "replay.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                dict(replayed_fields=len(result["records"]), scope=result["scope"])
            ),
            flush=True,
        )
        return
    summary = []
    for case in args.case:
        for variant in args.variant:
            print(json.dumps(dict(starting_case=case, variant=variant)), flush=True)
            try:
                result = run_encounter_control_case(
                    case, variant, maximum_seconds=args.maximum_seconds
                )
            except Exception as error:
                result = dict(
                    case=case,
                    variant=variant,
                    passed=False,
                    reason=repr(error),
                    scope="ORACLE control audit failed; no wheels",
                )
            (args.output / f"{case}-{variant}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            summary.append({k: v for k, v in result.items() if k != "rows"})
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(json.dumps(summary[-1]), flush=True)


if __name__ == "__main__":
    main()
