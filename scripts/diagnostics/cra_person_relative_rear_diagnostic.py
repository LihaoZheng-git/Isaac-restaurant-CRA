"""Audit trusted local rear-pass requests and score the stricter timing gate.

Numeric snapshots are produced by this repository, SHA checked against their
sidecar, and loaded only by explicit --trusted-native-run. No wheels or Isaac
are started. Results do not constitute counterfactual closed-loop navigation.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robot.evaluation.live_rear_pass import live_track_context
from robot.evaluation.person_relative_rear_validation import (
    summarize_person_relative_rear,
)
from robot.safety.person_relative_rear import (
    PersonRelativeRearConfig,
    PersonRelativeRearMemory,
)


def audit_native(directory):
    directory = Path(directory)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    sidecar = json.loads(
        (directory / "planner-replays/first-selected.json").read_text()
    )
    blob = (directory / "planner-replays" / sidecar["file"]).read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != sidecar["sha256"]:
        raise ValueError("original trusted-local planner snapshot hash mismatch")
    saved = pickle.loads(blob)
    control = saved["control"]
    state, _, field, _, _ = saved["request"]
    ids = field.motion_passing_reference_track_ids
    if len(ids) != 1:
        raise ValueError("one original selected person required")
    person, velocity = live_track_context(field, ids[0])
    cfg = PersonRelativeRearConfig(
        rear_distance_m=control.rear_pass_config.offset_m,
        maximum_linear_velocity=control.mpc.config.maximum_linear_velocity,
        maximum_angular_velocity=min(
            control.mpc.config.maximum_angular_velocity,
            abs(control.mpc.config.minimum_angular_velocity),
        ),
    )
    goal = np.array(report["nominal_route_local_xy"][-1])
    route = np.array(field.rear_pass_route_direction_grid_xy)
    memory = PersonRelativeRearMemory(cfg)
    reference = memory.update(
        track_id=ids[0],
        selection_timestamp=field.motion_passing_reference_source_timestamp_seconds,
        source_timestamp=field.sensor_timestamp_seconds,
        initial=state,
        goal=goal,
        person_xy=person,
        velocity_xy=velocity,
        route_direction=route,
    )
    initial, nominal, _, warm, active_goal = saved["request"]
    legacy = control.plan(
        initial,
        nominal,
        field,
        initial_command_guess=warm,
        nominal_goal_grid_xy=active_goal,
    )
    legacy_parity = np.array_equal(
        legacy.command_sequence, saved["expected_result"].command_sequence
    )
    if not legacy_parity:
        raise AssertionError(
            "disabled new mode changed the original numeric command sequence"
        )
    lead = float((person - state[:2]) @ route)
    anchor = json.loads((directory / "fixture-monitor.json").read_text())[
        "anchor_world_xy_yaw"
    ]
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    goal_world = np.array(anchor[:2]) + np.array([[c, -s], [s, c]]) @ goal
    strict = summarize_person_relative_rear(
        json.loads((directory / "human-truth-evaluation-only.json").read_text()),
        nominal_goal_world_xy=goal_world,
    )
    return dict(
        directory=str(directory),
        original_report_passed=report["passed"],
        original_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        trusted_snapshot_sha256=digest,
        original_source_timestamp_seconds=field.sensor_timestamp_seconds,
        original_selection_timestamp_seconds=field.motion_passing_reference_source_timestamp_seconds,
        original_planning_state=state.tolist(),
        observed_person_grid_xy=person.tolist(),
        inferred_policy_velocity_grid_xy=velocity.tolist(),
        configuration=asdict(cfg),
        current_forward_lead_m=lead,
        optimistic_forward_time_seconds=lead / cfg.maximum_linear_velocity,
        new_reference_eligible=reference is not None,
        disabled_mode_identical_original_command_sequence=bool(legacy_parity),
        maneuver=memory.diagnostic(),
        separate_stricter_recorded_trajectory_evaluation=strict,
        scope="Original planning-state/field numeric eligibility only; not the acquisition-pose worker replay or counterfactual navigation. Actor truth used only in separate posthoc evaluation.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trusted-native-run", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous diagnostic outputs")
    result = dict(
        runs=[audit_native(p) for p in args.trusted_native_run],
        scope="zero-wheel trusted local numeric/recorded-trajectory audit; original reports unchanged",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
