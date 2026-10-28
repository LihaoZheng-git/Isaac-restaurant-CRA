from copy import deepcopy
from dataclasses import replace
import ast
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_predictive_human_geometry import FOOTPRINT
from test_predictive_human_mpc import inputs
from test_safe_sage_cbf_filter import make_filter

from robot.evaluation.observed_motion_reference_validation import (
    validate_observed_motion_reference,
)
from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE
from robot.safety.laplace_guidance_tracking_mpc import (
    LaplaceGuidanceTrackingConfig,
    LaplaceGuidanceTrackingMPC,
)
from robot.safety.observed_motion_reference import ObservedHumanMotionReference
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPCConfig


def context(center=(1.6, 0.8), velocity=(-0.5, 0.0)):
    return ObservedHumanMotionReference(
        7, 60, 1.0, center, velocity, 0.65, 0.25, 100, 0.9, 8
    )


def sample(c=None, **kwargs):
    times = np.linspace(0.0, 1.2, 73)
    states = np.column_stack((0.2 * times, np.zeros((len(times), 2))))
    defaults = dict(
        anchor=np.zeros(2),
        anchor_yaw=0.0,
        footprint=FOOTPRINT,
        lookahead=0.1,
        source_age=0.1,
        maximum_outward_fraction=0.35,
        growth_prior_mps=0.5,
    )
    return (c or context()).preference(states, times, **(defaults | kwargs))


def test_forecast_preference_is_continuous_bounded_and_not_a_risk_or_hard_constraint():
    values = [sample(context(center=(x, 0.8)))[0] for x in [3.0, 2.0, 1.6, 1.2]]
    assert values[0] == 0 and values[-1] == 0.35
    assert np.all(np.diff(values) >= 0)
    fraction, audit = sample()
    assert 0 < fraction <= 0.35 and audit["status"] == "prior"
    assert (
        audit["sensor_timestamp_seconds"] == 1.0
        and audit["original_person_camera_timestamp"] == 0.9
    )
    assert audit["horizon_seconds"] == pytest.approx(1.2)
    assert (
        "advisory" in audit["scope"]
        and "current_raw_LiDAR" in audit["hypothesis_basis"]
    )


@pytest.mark.parametrize("velocity", [(0.5, 0.0), (0.0, -0.5)])
def test_departing_or_crossing_motion_cannot_trigger_parallel_advisory(velocity):
    value, audit = sample(context(velocity=velocity))
    assert value == 0 and not audit["active"]


@pytest.mark.parametrize("age", [-0.01, 0.3501, np.nan, np.inf])
def test_actual_source_age_cannot_be_renewed_or_extended(age):
    with pytest.raises(ValueError):
        sample(source_age=age)


def test_rigid_world_frame_change_preserves_forecast():
    c = context()
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    anchor = np.array([28.0, 7.0])
    moved = replace(
        c,
        center_world_xy=tuple(np.array(c.center_world_xy) @ rotation.T + anchor),
        velocity_world_xy=tuple(np.array(c.velocity_world_xy) @ rotation.T),
    )
    a, b = sample(c), sample(moved, anchor=anchor, anchor_yaw=angle)
    assert a[0] == pytest.approx(b[0], abs=1e-12)
    assert a[1]["minimum_nominal_predicted_margin_m"] == pytest.approx(
        b[1]["minimum_nominal_predicted_margin_m"], abs=1e-12
    )


def observed(t=1.0, *, velocity=(-0.5, 0.0), points=100, matched=True):
    track = NS(
        track_id=7,
        centroid_world_xy=(1.6, 0.8),
        velocity_world_xy=velocity,
        point_count=points,
        matched_previous_track=matched,
        observation_count=8,
    )
    tracking = NS(timestamp_seconds=t, tracked_obstacles=(track,))
    body = dict(
        track_id=7,
        source=HUMAN_ENVELOPE_SOURCE,
        status="prior",
        applied_to_occupancy=True,
        lidar_timestamp_seconds=t,
        center_world_xy=track.centroid_world_xy,
        velocity_world_xy=velocity,
        radius_m=0.65,
        centroid_uncertainty_prior_m=0.25,
        current_support_count=points,
        original_person_camera_timestamp=0.9,
        original_person_camera_frame_id=8,
    )
    return tracking, [body]


@pytest.mark.parametrize(
    "change", [dict(velocity=(0.0, 0.0)), dict(points=19), dict(matched=False)]
)
def test_quiet_or_missing_identity_support_returns_unavailable_not_zero_motion(change):
    tracking, bodies = observed(**change)
    assert (
        ObservedHumanMotionReference.from_current(tracking, bodies, (7,), frame=60)
        is None
    )


def test_ambiguous_selection_and_body_are_not_accepted():
    t, b = observed()
    assert ObservedHumanMotionReference.from_current(t, b, (7, 8), frame=60) is None
    assert ObservedHumanMotionReference.from_current(t, b * 2, (7,), frame=60) is None
    b[0]["lidar_timestamp_seconds"] = 0.9
    with pytest.raises(ValueError):
        ObservedHumanMotionReference.from_current(t, b, (7,), frame=60)


def test_current_source_identity_velocity_and_clock_are_bound():
    tracking, bodies = observed()
    c = ObservedHumanMotionReference.from_current(tracking, bodies, (7,), frame=60)
    c.validate_current(tracking, 60, 1.0)
    for frame, epoch in [(61, 1.0), (60, 1.01)]:
        with pytest.raises(ValueError):
            c.validate_current(tracking, frame, epoch)
    tracking.tracked_obstacles[0].velocity_world_xy = (-0.4, 0.0)
    with pytest.raises(ValueError):
        c.validate_current(tracking, 60, 1.0)


def mpc(enabled):
    return LaplaceGuidanceTrackingMPC(
        make_filter(),
        SafeSAGEUnicycleMPCConfig(
            horizon_steps=6,
            safety_substeps_per_interval=12,
            integration_substeps_per_interval=12,
            maximum_solver_iterations=1000,
        ),
        guidance_config=LaplaceGuidanceTrackingConfig(
            weight=10,
            reference_mode="planar_forecast_velocity",
            tangent_velocity_fraction=1.0,
            reference_tangent_basis="selected_observed_obstacle",
            tangent_blend_mode="convex",
            motion_forecast_maximum_outward_fraction=0.35 if enabled else 0.0,
        ),
    )


def plan(m, **kw):
    return m.plan(
        np.zeros(3),
        (0.2, 0.0),
        nominal_goal_grid_xy=(2.0, 0.0),
        selected_passing_tangents=((7, (1.6, 0.8), "right"),),
        **inputs(),
        **kw,
    )


def test_missing_advisory_preserves_legacy_full_commands_and_shared_configuration():
    a, b = mpc(False), mpc(True)
    before = deepcopy(vars(b))
    ra, rb = plan(a), plan(b)
    np.testing.assert_array_equal(ra.command_sequence, rb.command_sequence)
    assert rb.guidance_tracking.observed_motion_forecast["status"] == "unavailable"
    assert b.guidance_config == before["guidance_config"]


def test_advisory_trajectory_still_passes_original_exact_horizon_checks():
    m = mpc(True)
    nominal = np.tile([0.2, 0.0], (6, 1))
    before = nominal.copy()
    r = m.plan(
        np.zeros(3),
        nominal,
        nominal_goal_grid_xy=(2.0, 0.0),
        selected_passing_tangents=((7, (1.6, 0.8), "right"),),
        observed_motion_forecast=context(),
        observed_motion_source_age_seconds=0.1,
        observed_motion_anchor=(0.0, 0.0, 0.0),
        **inputs(),
    )
    np.testing.assert_array_equal(nominal, before)
    assert r.guidance_tracking.observed_motion_forecast["outward_fraction"] > 0
    assert r.success and m.revalidate_commands(
        np.zeros(3), r.command_sequence, **inputs()
    )
    assert len(r.command_sequence) == 6 and m.config.time_step == 0.2
    assert m.predictive_geometry is None


def test_wrong_selected_identity_rejects_advisory_instead_of_transferring_it():
    with pytest.raises(ValueError):
        plan(
            mpc(True),
            observed_motion_forecast=replace(context(), track_id=8),
            observed_motion_source_age_seconds=0.1,
            observed_motion_anchor=(0.0, 0.0, 0.0),
        )


def test_source_dataflow_validation_rejects_missing_and_mismatched_records():
    t, b = observed()
    c = context().diagnostic()
    good = dict(
        time=1.0,
        sensor_frame_id=60,
        observed_motion_reference=c,
        human_body_envelope_priors=b,
        motion_aware_passing=dict(decision="selected_parallel_away_tangent"),
    )
    quiet = dict(time=0.8, sensor_frame_id=48, observed_motion_reference=None)
    rows = [deepcopy(good) for _ in range(20)] + [quiet] * 10
    # Repeated source rows are not twenty distinct observations.
    assert not validate_observed_motion_reference(rows)["passed"]
    assert not validate_observed_motion_reference([])["passed"]
    rows[0]["observed_motion_reference"]["velocity_world_xy"] = [-0.3, 0.0]
    assert not validate_observed_motion_reference(rows)["passed"]


def test_component_gate_accepts_only_distinct_current_sources_and_audits_all_rows():
    rows = []
    for i in range(30):
        t = 1.0 + i * 0.1
        frame = 60 + 6 * i
        tracking, bodies = observed(t)
        c = ObservedHumanMotionReference.from_current(
            tracking, bodies, (7,), frame=frame
        )
        rows.append(
            dict(
                time=t,
                sensor_frame_id=frame,
                observed_motion_reference=c.diagnostic() if i >= 10 else None,
                human_body_envelope_priors=bodies,
                motion_aware_passing=dict(
                    decision="selected_parallel_away_tangent"
                    if i >= 10
                    else "no_encounter",
                    source_timestamp=t,
                    encounter_selected_timestamp=t,
                    original_selection_visual_person_verified=True,
                    encounter_geometry="oncoming_parallel",
                    configuration=dict(maximum_encounter_seconds=8.0),
                    preferences={"7": "right"},
                ),
            )
        )
    assert validate_observed_motion_reference(rows)["passed"]
    held = deepcopy(rows)
    for row in held[11:]:
        row["motion_aware_passing"]["decision"] = "held_current_track_policy_preference"
        row["motion_aware_passing"]["encounter_selected_timestamp"] = 2.0
    assert validate_observed_motion_reference(held)["passed"]
    held[-1]["motion_aware_passing"]["encounter_selected_timestamp"] = 3.8
    assert not validate_observed_motion_reference(held)["passed"]
    for key in ["radius_prior_m", "sensor_timestamp_seconds", "current_support_count"]:
        bad = deepcopy(rows)
        bad[-1]["observed_motion_reference"][key] += 0.01
        assert not validate_observed_motion_reference(bad)["passed"]


@pytest.mark.parametrize(
    "change",
    [
        dict(encounter_selected_timestamp=4.0),
        dict(encounter_selected_timestamp=-1.0),
        dict(original_selection_visual_person_verified=False),
    ],
)
def test_held_current_selector_preserves_original_lease_and_visual_proof(change):
    from robot.evaluation.observed_motion_reference_validation import _current_selection

    s = dict(
        decision="held_current_track_policy_preference",
        source_timestamp=3.0,
        encounter_selected_timestamp=1.0,
        original_selection_visual_person_verified=True,
        encounter_geometry="oncoming_parallel",
        configuration=dict(maximum_encounter_seconds=8.0),
        preferences={7: "left"},
    )
    assert _current_selection(s, 3.0, 7)
    assert not _current_selection(s | change, 3.0, 7)


def current_runtime_field():
    from test_safe_sage_runtime import make_observation, make_provider, make_runtime

    runtime = make_runtime()
    for i in range(3):
        o = make_observation(
            i + 1, 1.0 + 0.1 * i, obstacle_world_offset_xy=(-0.05 * i, 0.0)
        )
        o = replace(
            o,
            obstacle_points_robot_xy=np.repeat(o.obstacle_points_robot_xy, 25, axis=0),
        )
        if i == 0:
            runtime.initialize(o, make_provider())
        else:
            runtime.update(o, make_provider())
    f = runtime.field
    t = f.tracking_result.tracked_obstacles[0]
    c = ObservedHumanMotionReference(
        t.track_id,
        f.sensor_frame_id,
        f.sensor_timestamp_seconds,
        t.centroid_world_xy,
        t.velocity_world_xy,
        0.65,
        0.25,
        t.point_count,
        0.9,
        8,
    )
    return replace(
        f,
        motion_passing_reference_track_ids=(t.track_id,),
        motion_passing_reference_sides=("right",),
        motion_passing_reference_source_timestamp_seconds=f.sensor_timestamp_seconds,
        observed_motion_forecast=c,
    )


def test_snapshot_rejects_transferred_advisory_identity_or_source():
    f = current_runtime_field()
    c = f.observed_motion_forecast
    for change in [
        dict(sensor_frame_id=99),
        dict(sensor_timestamp_seconds=1.1),
        dict(velocity_world_xy=(-0.7, 0.0)),
    ]:
        with pytest.raises(ValueError):
            replace(f, observed_motion_forecast=replace(c, **change))


def test_runtime_handoff_requires_actual_source_clock_and_never_enables_old_hard_prediction(
    monkeypatch,
):
    from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
    from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter
    from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPC

    f = current_runtime_field()
    m = mpc(True)
    result = SafeSAGEUnicycleMPC(m.paper_filter, m.config).plan(
        np.zeros(3), (0.2, 0.0), **inputs()
    )
    captured = {}

    def collect(*args, **kw):
        captured.update(kw)
        return result

    monkeypatch.setattr(m, "plan", collect)
    c = SafeSAGEControlRuntime(
        mpc=m, backup_filter=SafeSAGEUnicycleCBFFilter(m.paper_filter)
    )
    for epoch in [
        None,
        f.sensor_timestamp_seconds - 0.01,
        f.sensor_timestamp_seconds + 0.36,
    ]:
        with pytest.raises(ValueError):
            c.plan(
                np.zeros(3),
                (0.2, 0.0),
                f,
                nominal_goal_grid_xy=(2.0, 0.0),
                prediction_state_timestamp_seconds=epoch,
            )
    c.plan(
        np.zeros(3),
        (0.2, 0.0),
        f,
        nominal_goal_grid_xy=(2.0, 0.0),
        prediction_state_timestamp_seconds=f.sensor_timestamp_seconds + 0.1,
    )
    assert captured["observed_motion_forecast"] is f.observed_motion_forecast
    assert captured["observed_motion_source_age_seconds"] == pytest.approx(0.1)
    assert (
        captured["h_full"] is f.pipeline.h_full
        and captured["temporal_safety_derivative"] is f.temporal_safety_derivative
    )
    assert m.predictive_geometry is None


@pytest.mark.parametrize(
    "change",
    [
        "valid",
        "missing_body",
        "hard_prediction",
        "early_redesign",
        "short_horizon",
        "zero_weight",
    ],
)
def test_real_live_cli_parser_contract_before_isaac_import(monkeypatch, change):
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/demos/cra_live_object_isaac_demo.py"
    )
    tree = ast.parse(script.read_text(encoding="utf-8"))
    split = next(
        i
        for i, n in enumerate(tree.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "args" for t in n.targets)
    )
    guard = next(
        n
        for n in tree.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test).startswith("args.observed_motion_reference")
    )
    argv = [
        str(script),
        "--output",
        "unused_advisory_cli_contract",
        "--observed-motion-reference",
        "--parallel-guidance-ablation",
        "--guidance-tracking-weight",
        "10",
        "--guidance-tangent-basis",
        "selected_observed_obstacle",
        "--guidance-tangent-blend",
        "convex",
    ]
    if change != "missing_body":
        argv += ["--human-body-envelope-radius", ".4"]
    argv += {
        "hard_prediction": ["--predictive-human-geometry-ablation"],
        "early_redesign": ["--predictive-parallel-redesign"],
        "short_horizon": ["--mpc-horizon-steps", "4"],
        "zero_weight": ["--guidance-tracking-weight", "0"],
    }.get(change, [])
    monkeypatch.setattr(sys, "argv", argv)
    namespace = {"__file__": str(script), "__name__": "observed_motion_cli_contract"}
    # Only standard/NumPy imports and the ACTUAL parser declarations execute.
    exec(
        compile(
            ast.Module(body=tree.body[: split + 1], type_ignores=[]),
            str(script),
            "exec",
        ),
        namespace,
    )
    code = compile(ast.Module(body=[guard], type_ignores=[]), str(script), "exec")
    if change == "valid":
        exec(code, namespace)
    else:
        with pytest.raises(SystemExit) as error:
            exec(code, namespace)
        assert error.value.code == 2
