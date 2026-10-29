from dataclasses import replace
import ast
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
from test_laplace_guidance_tracking_mpc import config, field
from test_safe_sage_cbf_filter import make_filter

from robot.safety.checked_moving_arc_reference import search_checked_moving_arc
from robot.safety.laplace_guidance_tracking_mpc import (
    LaplaceGuidanceTrackingConfig,
    LaplaceGuidanceTrackingMPC,
)


def planner(**changes):
    guidance = LaplaceGuidanceTrackingConfig(
        weight=10,
        reference_mode="planar_forecast_velocity",
        reference_tangent_basis="selected_observed_obstacle",
        tangent_blend_mode="convex",
        tangent_velocity_fraction=0.5,
        minimum_reference_speed_fraction=0.25,
        moving_arc_reference=True,
    )
    return LaplaceGuidanceTrackingMPC(
        make_filter(), config(), guidance_config=replace(guidance, **changes)
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(moving_arc_reference=1),
        dict(weight=0),
        dict(reference_mode="filtered_goal_velocity"),
        dict(reference_tangent_basis="planar_gradient"),
        dict(tangent_blend_mode="additive"),
    ],
)
def test_invalid_arc_scope_rejected(changes):
    with pytest.raises(ValueError):
        planner(**changes)


def test_arcs_use_nominal_speed_limits_full_horizon_and_leave_inputs_unchanged():
    m = planner()
    f = field() | dict(selected_passing_tangents=((1, (1.0, -0.5), "left"),))
    before = pickle.dumps(f)
    initial, goal = np.zeros(3), np.array([2.0, 0.0])
    arc, _, _, audit = search_checked_moving_arc(m, initial, goal, 0.2, f)
    assert arc is not None and audit["checked"] == 132
    assert arc.commands.shape == (m.config.horizon_steps, 2)
    assert 0 < arc.commands[0, 0] <= 0.2
    assert np.all(arc.commands[:, 1] <= m.config.maximum_angular_velocity)
    assert np.all(arc.commands[:, 1] >= m.config.minimum_angular_velocity)
    assert (
        audit["minimum_forecast_margin"] >= -m.paper_filter.config.numerical_tolerance
    )
    assert m.revalidate_commands(initial, arc.commands, **field())
    assert pickle.dumps(f) == before


def test_every_arc_gets_full_geometry_and_forecast_checks(monkeypatch):
    m = planner()
    f = field() | dict(selected_passing_tangents=((1, (1.0, -0.5), "left"),))
    original = m._evaluate_trajectory
    calls = []

    def evaluate(*args, **kwargs):
        assert kwargs["require_feasible"]
        assert args[1].shape == (m.config.horizon_steps, 2)
        calls.append(1)
        e = original(*args, **kwargs)
        if e is not None:
            # Keep exact MPC checks unchanged; poison only the final auxiliary
            # half-space, proving it cannot accept just a good first command.
            e.samples.guidance_normals[-2] = (0.0, 0.0, 0.0)
            e.samples.scaled_temporal_derivatives[-2] = -1e6
        return e

    monkeypatch.setattr(m, "_evaluate_trajectory", evaluate)
    arc, _, _, audit = search_checked_moving_arc(
        m, np.zeros(3), np.array([2.0, 0.0]), 0.2, f
    )
    assert arc is None and len(calls) == 132
    assert audit["exact_safe"] > 0 and audit["forecast_safe"] == 0


def test_no_arc_can_override_exact_rejection(monkeypatch):
    m = planner()
    monkeypatch.setattr(m, "_evaluate_trajectory", lambda *a, **kw: None)
    arc, _, _, audit = search_checked_moving_arc(
        m,
        np.zeros(3),
        np.array([2.0, 0.0]),
        0.2,
        field() | dict(selected_passing_tangents=((1, (1.0, -0.5), "left"),)),
    )
    assert arc is None and audit["exact_safe"] == 0


def test_quiet_unselected_clone_never_searches_and_matches_disabled(monkeypatch):
    import robot.safety.checked_moving_arc_reference as module

    monkeypatch.setattr(
        module,
        "search_checked_moving_arc",
        lambda *a: pytest.fail("unselected arc search"),
    )
    a = planner(tangent_velocity_fraction=0.0)
    b = planner(tangent_velocity_fraction=0.0, moving_arc_reference=False)
    results = [
        m.plan(np.zeros(3), (0.2, 0.0), nominal_goal_grid_xy=(2.0, 0.0), **field())
        for m in (a, b)
    ]
    assert results[0].guidance_tracking.moving_arc_search is None
    np.testing.assert_array_equal(
        results[0].command_sequence, results[1].command_sequence
    )


def test_checked_arc_replaces_turn_first_without_changing_nominal(monkeypatch):
    m = planner()
    original = m._guidance_rollout
    # Retain only the actual turn-first generator, not a fabricated trajectory.
    monkeypatch.setattr(
        m,
        "_guidance_rollout",
        lambda initial, goal, speed, cap, turn, f: (
            original(initial, goal, speed, cap, turn, f) if turn else None
        ),
    )
    nominal = np.tile([0.2, 0.0], (m.config.horizon_steps, 1))
    before = nominal.copy()
    result = m.plan(
        np.zeros(3),
        nominal,
        nominal_goal_grid_xy=(2.0, 0.0),
        selected_passing_tangents=((1, (1.0, -0.5), "left"),),
        **field(),
    )
    d = result.guidance_tracking
    assert d.moving_arc_search["selected"] and not d.selected_turn_first
    assert d.first_reference_translation_delay_seconds == 0
    assert result.success and m.revalidate_commands(
        np.zeros(3), result.command_sequence, **field()
    )
    np.testing.assert_array_equal(nominal, before)


def test_rotating_current_scene_preserves_arc_selection():
    chosen = []
    for yaw in (0.0, np.pi / 2, -np.pi + 0.1):
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        f = field(g=rotation @ [1.0, 1.0]) | dict(
            selected_passing_tangents=((1, rotation @ [1.0, -0.5], "left"),)
        )
        arc, _, _, _ = search_checked_moving_arc(
            planner(), np.array([0.0, 0.0, yaw]), rotation @ [2.0, 0.0], 0.2, f
        )
        assert arc is not None
        chosen.append(arc.commands)
    for commands in chosen[1:]:
        np.testing.assert_allclose(commands, chosen[0], atol=1e-12)


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--mpc-unfinished-qp-proposal"],
        ["--mpc-horizon-steps", "4"],
        ["--guidance-tangent-velocity-fraction", "0"],
        ["--observed-motion-reference"],
        ["--predictive-parallel-redesign"],
    ],
)
def test_real_cli_scope_before_isaac(monkeypatch, extra):
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
        and ast.unparse(n.test).startswith("args.guidance_moving_arc_reference")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--output",
            "unused_arc_cli",
            "--guidance-moving-arc-reference",
            "--parallel-guidance-ablation",
            "--guidance-tracking-weight",
            "100",
            "--guidance-reference-mode",
            "planar_forecast_velocity",
            "--guidance-tangent-basis",
            "selected_observed_obstacle",
            "--guidance-tangent-blend",
            "convex",
            "--guidance-tangent-velocity-fraction",
            ".5",
            "--human-observed-passing-target",
            "--motion-aware-passing",
            *extra,
        ],
    )
    namespace = dict(__file__=str(script), __name__="arc_cli_test")
    exec(
        compile(
            ast.Module(body=tree.body[: split + 1], type_ignores=[]),
            str(script),
            "exec",
        ),
        namespace,
    )
    code = compile(ast.Module(body=[guard], type_ignores=[]), str(script), "exec")
    if extra:
        with pytest.raises(SystemExit) as error:
            exec(code, namespace)
        assert error.value.code == 2
    else:
        exec(code, namespace)
