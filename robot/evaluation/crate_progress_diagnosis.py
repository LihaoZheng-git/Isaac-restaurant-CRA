"""Trusted local frozen-input diagnostic only. Never loaded by robot control."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from robot.navigation.waypoint_controller import (
    WaypointController,
    WaypointControllerConfig,
    Waypoint,
    RobotPose2D,
)
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPC
from robot.safety.laplace_guidance_tracking_mpc import LaplaceGuidanceTrackingMPC
from robot.evaluation.simple_object_test import disk_footprint_clearance


def load(meta_path):
    meta_path = Path(meta_path).resolve()
    meta = json.loads(meta_path.read_text())
    path = (meta_path.parent / meta["file"]).resolve()
    if path.parent != meta_path.parent or path.suffix != ".pickle":
        raise ValueError("metadata must name its sibling local pickle")
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
        raise ValueError("snapshot hash mismatch")
    return blob, meta


def nominal_for(state, steps, dt):
    controller = WaypointController(
        [Waypoint(4.0, 0.0)],
        config=WaypointControllerConfig(
            maximum_linear_velocity=0.2,
            maximum_angular_velocity=0.6,
            rotate_in_place_threshold_radians=math.radians(100),
            minimum_alignment_scale=0.25,
            intermediate_waypoints_as_guides=True,
            route_guide_lookahead_m=2.0,
            route_guide_max_cross_track_m=1.5,
        ),
        route_start=Waypoint(0.0, 0.0),
    )
    command = controller.compute_command(RobotPose2D(*state))
    sequence = controller.predict_commands(
        RobotPose2D(*state), horizon_steps=steps, time_step=dt
    )
    return np.array(
        [(q.linear_velocity, q.angular_velocity) for q in sequence]
    ), np.array([command.target_waypoint.x, command.target_waypoint.y])


def field_kwargs(field):
    return dict(
        configuration_space=field.pipeline.configuration_space,
        h_full=field.pipeline.h_full,
        guidance_fields=field.pipeline.guidance_fields,
        temporal_safety_derivative=field.temporal_safety_derivative,
    )


def analytical(control, state, field, command):
    result = control.filter_command(command, state, field)
    safe = np.array([result.safe_linear_velocity, result.safe_angular_velocity])
    return dict(
        command=safe.tolist(),
        margin=float(result.cbf_margin),
        feasible=bool(result.constraint_feasible),
        satisfied=bool(result.constraint_satisfied),
        emergency_stop=bool(result.emergency_stop_active),
        unchanged=bool(np.allclose(command, safe, atol=1e-9, rtol=0)),
    )


def trajectory_audit(mpc, state, commands, field, control, fixture=None):
    commands = np.asarray(commands, dtype=float)
    lower = np.array(
        [mpc.config.minimum_linear_velocity, mpc.config.minimum_angular_velocity]
    )
    upper = np.array(
        [mpc.config.maximum_linear_velocity, mpc.config.maximum_angular_velocity]
    )
    if (
        commands.shape != (mpc.config.horizon_steps, 2)
        or not np.isfinite(commands).all()
        or np.any(commands < lower)
        or np.any(commands > upper)
    ):
        raise ValueError(
            "audit requires a finite original bounded horizon; do not silently clip candidates"
        )
    kw = field_kwargs(field)
    endpoints = mpc._evaluate_trajectory(
        state, commands, **kw, check_dense_geometry=False
    )
    full = mpc._evaluate_trajectory(state, commands, **kw)
    if full is None:
        return dict(
            accepted=False, reason="sampling_or_geometry_evaluation_unavailable"
        )
    out = dict(
        accepted=mpc._trajectory_is_safe(full),
        endpoint_unsafe_indices=np.flatnonzero(
            endpoints.samples.unsafe_configuration_mask
        ).tolist(),
        dense_unsafe_indices=np.flatnonzero(
            full.samples.unsafe_configuration_mask
        ).tolist(),
        dtcbf_violation_indices=np.flatnonzero(
            full.margins < -mpc.config.constraint_tolerance
        ).tolist(),
        minimum_dtcbf=float(full.margins.min()),
        raw_h=full.samples.safety_values.tolist(),
        scaled_ht=full.samples.scaled_temporal_derivatives.tolist(),
        forecast_h=full.safety_values.tolist(),
        displacement=float(np.linalg.norm(full.chassis_states[-1, :2] - state[:2])),
        endpoint=full.chassis_states[-1].tolist(),
        first_command=np.asarray(commands)[0].tolist(),
        commands=np.asarray(commands).tolist(),
        states=full.chassis_states.tolist(),
        analytical=analytical(control, state, field, commands[0]),
    )
    if fixture is not None:
        center = (
            np.asarray(fixture["fixtures"][0]["center_world_xy"])
            - field.anchor_position_world_xy
        )
        c, s = math.cos(field.anchor_yaw), math.sin(field.anchor_yaw)
        center = center @ np.array([[c, -s], [s, c]])
        dense, _ = mpc._dense_chassis_queries(full.chassis_states, commands)
        reference_footprint = field.pipeline.configuration_space.footprint
        physical_footprint = replace(
            reference_footprint,
            front_extent=reference_footprint.front_extent
            + mpc.config.lookahead_distance,
            rear_extent=reference_footprint.rear_extent - mpc.config.lookahead_distance,
        )
        out["independent_fixture_clearance_min"] = min(
            disk_footprint_clearance(
                q,
                center_xy=center,
                radius_m=fixture["fixtures"][0]["monitor_radius_m"],
                footprint=physical_footprint,
            )
            for q in dense
        )
    return out


def rollout_return_trace(mpc, *args):
    events = []

    def trace(frame, event, arg):
        if frame.f_code.co_name == "_guidance_rollout" and event == "return":
            values = frame.f_locals
            events.append(
                dict(
                    line=frame.f_lineno,
                    step=values.get("i"),
                    returned_none=arg is None,
                    state=np.asarray(values.get("state")).tolist(),
                    preference=values.get("preference"),
                    unsafe_reason=getattr(values.get("sample"), "unsafe_reason", None),
                )
            )
        return trace

    old = sys.gettrace()
    try:
        sys.settrace(trace)
        result = mpc._guidance_rollout(*args)
    finally:
        sys.settrace(old)
    return result, events


def references(control, state, nominal, field, goal, fixture=None):
    mpc = control.mpc
    kw = field_kwargs(field)
    speed = float(np.max(nominal[:, 0]))
    rows = []
    for fraction in mpc.guidance_config.reference_speed_fractions():
        for cap, turn_first in (
            (0.4, False),
            (0.8, False),
            (0.1, True),
            (0.2, True),
            (0.4, True),
            (0.8, True),
        ):
            if turn_first and cap < mpc.guidance_config.minimum_turn_first_rate_cap:
                continue
            commands, trace = rollout_return_trace(
                mpc, state, goal, speed * fraction, cap, turn_first, kw
            )
            audit = (
                None
                if commands is None
                else trajectory_audit(mpc, state, commands, field, control, fixture)
            )
            rows.append(
                dict(
                    speed_fraction=fraction,
                    cap=cap,
                    turn_first=turn_first,
                    trace=trace,
                    audit=audit,
                )
            )
    return rows


def candidates(control, state, nominal, field, fixture=None):
    mpc = control.mpc
    n = mpc.config.horizon_steps
    rows = []
    seen = set()
    rejects = Counter()
    # Finite diagnostic pool, both signs; no authored bypass/fixture pose input.
    for speed, omega, turn_steps in itertools.product(
        (0.025, 0.05, 0.1, 0.15, 0.2),
        (-0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8),
        (0, 1, 2, 3, 5, 8, 12),
    ):
        if turn_steps >= n:
            continue
        for turning_speed in (0.0, speed) if turn_steps else (speed,):
            commands = np.tile([speed, 0.0], (n, 1))
            if turn_steps:
                commands[:turn_steps] = (turning_speed, omega)
            else:
                commands[:, 1] = omega
            key = commands.tobytes()
            if key in seen:
                continue
            seen.add(key)
            trial = mpc._evaluate_trajectory(
                state, commands, **field_kwargs(field), require_feasible=True
            )
            if trial is None or not mpc._trajectory_is_safe(trial):
                rejects["original_full_horizon_check"] += 1
                continue
            a = analytical(control, state, field, commands[0])
            if a["emergency_stop"] or not a["unchanged"] or not a["satisfied"]:
                rejects["analytical_first_command"] += 1
                continue
            audit = trajectory_audit(mpc, state, commands, field, control, fixture)
            audit["base_command_objective"] = mpc._objective_value(commands, nominal)
            rows.append(audit)
    rows.sort(key=lambda x: x["base_command_objective"])
    clearances = [
        row["independent_fixture_clearance_min"]
        for row in rows
        if "independent_fixture_clearance_min" in row
    ]
    return dict(
        tested=len(seen),
        passing=len(rows),
        rejections=dict(rejects),
        cheapest=rows[:3],
        largest_displacement=sorted(
            rows, key=lambda x: x["displacement"], reverse=True
        )[:3],
        independent_fixture_evaluation=(
            dict(
                count=len(clearances),
                minimum_clearance_m=min(clearances),
                below_original_monitor_002m=sum(value < 0.02 for value in clearances),
                scope="Evaluation only; never a candidate selection or field gate.",
            )
            if clearances
            else None
        ),
        scope="Finite offline witnesses only. No failed finite search establishes infeasibility; oracle fixture metric only reports after field-based selection.",
    )


def probe(blob, meta, fixture=None, pool=False):
    baseline = pickle.loads(blob)
    control = baseline["control"]
    state, nominal, field, warm, goal, *epoch = baseline["request"]
    if epoch:
        raise ValueError("static crate probe must not carry human prediction")
    original_field = pickle.dumps(field)
    generated, generated_goal = nominal_for(
        state, control.mpc.config.horizon_steps, control.mpc.config.time_step
    )
    np.testing.assert_allclose(generated, nominal, rtol=0, atol=1e-12)
    np.testing.assert_allclose(generated_goal, goal, rtol=0, atol=1e-12)
    start = time.perf_counter()
    result = control.plan(
        state, nominal, field, initial_command_guess=warm, nominal_goal_grid_xy=goal
    )
    np.testing.assert_allclose(
        result.command_sequence,
        baseline["expected_result"].command_sequence,
        rtol=0,
        atol=1e-10,
    )
    assert result.success == baseline["expected_result"].success
    original = dict(
        parity=True,
        wall_seconds=time.perf_counter() - start,
        status=result.status,
        audit=trajectory_audit(
            control.mpc, state, result.command_sequence, field, control, fixture
        ),
        guidance=asdict(result.guidance_tracking),
        warm_present=warm is not None,
    )
    assert pickle.dumps(field) == original_field
    variants = []
    for steps, guidance in itertools.product((6, 30), (False, True)):
        saved = pickle.loads(blob)
        c = saved["control"]
        state, _, field, _, goal = saved["request"]
        m = c.mpc
        config = replace(m.config, horizon_steps=steps)
        c.mpc = (
            LaplaceGuidanceTrackingMPC(
                m.paper_filter, config, guidance_config=m.guidance_config
            )
            if guidance
            else SafeSAGEUnicycleMPC(m.paper_filter, config)
        )
        nominal, goal = nominal_for(state, steps, config.time_step)
        before = pickle.dumps(field)
        start = time.perf_counter()
        r = c.plan(
            state,
            nominal,
            field,
            initial_command_guess=None,
            **({"nominal_goal_grid_xy": goal} if guidance else {}),
        )
        row = dict(
            steps=steps,
            guidance=guidance,
            cold_start=True,
            wall_seconds=time.perf_counter() - start,
            success=r.success,
            status=r.status,
            audit=trajectory_audit(c.mpc, state, r.command_sequence, field, c, fixture),
            configuration=asdict(config),
            nominal=nominal.tolist(),
        )
        if guidance:
            row["guidance_diagnostic"] = asdict(r.guidance_tracking)
            row["reference_candidates"] = references(
                c, state, nominal, field, goal, fixture
            )
        if pool and not guidance:
            row["bounded_pool"] = candidates(c, state, nominal, field, fixture)
        assert pickle.dumps(field) == before
        variants.append(row)
    return dict(
        meta=meta,
        state=np.asarray(state).tolist(),
        baseline=original,
        variants=variants,
        scope="Offline static crate replay. Native command parity first; matrix resets warm start in ALL cells and reconstructs the same pose-only waypoint horizon. Same field bytes/epochs/grid/phi/limits/solver budgets. No executable offline path or full-route acceptance.",
    )


def receipt_audit(run_directory):
    """Replay saved horizons at their recorded receipt poses, without a solve.

    Only rows with an identical field epoch at receipt establish native parity.
    A different field is explicitly a state-only counterfactual, never a match.
    This function loads trusted local pickles, just like ``probe``.
    """
    root = Path(run_directory)
    report = json.loads((root / "report.json").read_text())
    commands = json.loads((root / "commands.json").read_text())
    states = np.asarray(json.loads((root / "trajectory.json").read_text()))
    fields = [
        json.loads(line)
        for line in (root / "live-fields.jsonl").read_text().splitlines()
    ]
    times = np.asarray([row["time"] for row in commands])
    if len(states) != len(commands) or not fields:
        raise ValueError(
            "receipt audit requires aligned native trajectory and command rows"
        )
    index = int(
        np.argmin(np.linalg.norm(states - fields[0]["state_at_commit"], axis=1))
    )
    np.testing.assert_allclose(
        states[index], fields[0]["state_at_commit"], atol=1e-12, rtol=0
    )
    start = fields[0]["commit_simulation_time"] - times[index]
    rows = []
    for path in sorted((root / "planner-replays").glob("window-*.json")):
        blob, meta = load(path)
        saved = pickle.loads(blob)
        control = saved["control"]
        state, _, field, _, _ = saved["request"]
        result = saved["expected_result"]
        matches = [
            event
            for event in report["plan_events"]
            if np.max(
                np.abs(
                    np.asarray(
                        event["guidance_tracking_ablation"][
                            "initial_chassis_state_grid"
                        ]
                    )
                    - state
                )
            )
            < 1e-12
        ]
        if len(matches) != 1:
            raise ValueError("snapshot must match exactly one recorded planner request")
        event = matches[0]
        index = int(np.argmin(np.abs(times - event["time"])))
        if abs(times[index] - event["time"]) > 1e-10:
            raise ValueError("planner receipt has no aligned command/pose record")
        latest = max(
            (
                row
                for row in fields
                if row["commit_simulation_time"] <= start + times[index] + 1e-10
            ),
            key=lambda row: row["commit_simulation_time"],
        )
        same = (
            latest["sensor_frame_id"] == field.sensor_frame_id
            and abs(latest["sensor_timestamp"] - field.sensor_timestamp_seconds) < 1e-10
        )
        before = pickle.dumps(field)
        original = trajectory_audit(
            control.mpc, state, result.command_sequence, field, control
        )
        received = trajectory_audit(
            control.mpc, states[index], result.command_sequence, field, control
        )
        assert pickle.dumps(field) == before

        def compact(audit):
            return {
                key: audit.get(key)
                for key in (
                    "accepted",
                    "endpoint",
                    "dense_unsafe_indices",
                    "endpoint_unsafe_indices",
                    "dtcbf_violation_indices",
                    "minimum_dtcbf",
                    "analytical",
                )
            }

        rows.append(
            dict(
                snapshot=path.name,
                time=event["time"],
                source=meta["sensor_timestamp_seconds"],
                same_field_at_receipt=same,
                receipt_frame_id=latest["sensor_frame_id"],
                request_state=state.tolist(),
                receipt_state=states[index].tolist(),
                state_drift_m=float(np.linalg.norm(states[index, :2] - state[:2])),
                actual_stop=commands[index]["stop_reason"],
                original=compact(original),
                receipt=compact(received),
            )
        )
    if not rows:
        raise ValueError("no bounded-window snapshots found")
    return dict(
        start_time=start,
        rows=rows,
        scope="Same saved plan and field at original request versus actual receipt pose. Only "
        "same_field_at_receipt rows establish exact execution replay; others are state-only counterfactuals.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata", type=Path)
    source.add_argument("--receipt-run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture", type=Path)
    p.add_argument("--pool", action="store_true")
    p.add_argument("--trusted-local-replay", action="store_true")
    args = p.parse_args()
    if not args.trusted_local_replay or args.output.exists():
        p.error("acknowledge local replay and use a new output")
    if args.receipt_run is not None:
        if args.fixture is not None or args.pool:
            p.error("fixture/pool apply only to the planner matrix")
        result = receipt_audit(args.receipt_run)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                dict(
                    receipt_rows=len(result["rows"]),
                    exact_field_matches=sum(
                        row["same_field_at_receipt"] for row in result["rows"]
                    ),
                )
            )
        )
        return
    blob, meta = load(args.metadata)
    result = probe(
        blob,
        meta,
        None if args.fixture is None else json.loads(args.fixture.read_text()),
        args.pool,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                source=meta["sensor_timestamp_seconds"],
                baseline_parity=result["baseline"]["parity"],
                rows=[
                    dict(
                        steps=r["steps"],
                        guidance=r["guidance"],
                        success=r["success"],
                        first=r["audit"].get("first_command"),
                        accepted=r["audit"]["accepted"],
                        displacement=r["audit"].get("displacement"),
                        passing_candidates=r.get("bounded_pool", {}).get("passing"),
                    )
                    for r in result["variants"]
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
