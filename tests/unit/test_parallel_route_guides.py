"""Separate stronger travel preference; unchanged checks and pose-only nominal."""

import ast
from dataclasses import fields, replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_selected_obstacle_tangent_reference import selected_planner
from test_selected_tangent_reference import field, runtime_field

from robot.navigation.waypoint_controller import (
    RobotPose2D, Waypoint, WaypointController, WaypointControllerConfig,
)
from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter


def planner(mode="convex"):
    m = selected_planner()
    m.guidance_config = replace(m.guidance_config, tangent_blend_mode=mode)
    return m


@pytest.mark.parametrize("yaw", [0., np.pi / 2, -2.4])
@pytest.mark.parametrize("side,sign", [("left", 1.), ("right", -1.)])
def test_convex_preference_turns_further_away_without_exceeding_speed(yaw, side, sign):
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    initial, goal = np.array([0., 0., yaw]), rotation @ [2., 0.]
    center = rotation @ [1., -sign]
    kwargs = dict(field(gradient=(0., 0., 0.)), selected_passing_tangents=((7, center, side),))
    headings = []
    for mode in ("additive", "convex"):
        m = planner(mode)
        heading, speed = m._reference_preference(initial, goal, .2, m._sample(initial, kwargs), kwargs)
        assert 0 < speed <= .2
        headings.append(sign * ((heading - yaw + np.pi) % (2 * np.pi) - np.pi))
    assert 0 < headings[0] < headings[1] < np.pi / 2


@pytest.mark.parametrize("side,sign", [("left", 1.), ("right", -1.)])
def test_convex_executed_and_reference_horizons_still_pass_exact_checks(side, sign):
    m = planner()
    kwargs = field(-sign)
    nominal = np.tile((.2, 0.), (6, 1))
    original = nominal.copy()
    result = m.plan(np.zeros(3), nominal, nominal_goal_grid_xy=(1., 0.),
        selected_passing_tangents=((7, (1., -sign), side),), **kwargs)
    assert result.success and result.guidance_tracking.active
    assert sign * result.angular_velocity > .02
    assert result.guidance_tracking.tangent_blend_mode == "convex"
    for commands in (result.command_sequence, result.guidance_tracking.reference_commands):
        checked = m._evaluate_trajectory(np.zeros(3), np.asarray(commands),
            temporal_safety_derivative=0., **kwargs)
        assert checked is not None and m._trajectory_is_safe(checked)
    assert result.minimum_dtcbf_margin >= -m.config.constraint_tolerance
    np.testing.assert_array_equal(nominal, original)


def test_unsafe_temporal_forecast_is_not_made_usable_by_stronger_preference():
    m = planner()
    kwargs = dict(field(h=.1), temporal_safety_derivative=-1.)
    result = m.plan(np.zeros(3), (.2, 0.), nominal_goal_grid_xy=(1., 0.),
        selected_passing_tangents=((7, (1., -1.), "left"),), **kwargs)
    assert not result.success and result.first_command == (0., 0.)


def test_no_current_selection_disables_convex_tangent_without_mutating_sources():
    f, m = runtime_field(), planner()
    runtime = SafeSAGEControlRuntime(mpc=m, backup_filter=SafeSAGEUnicycleCBFFilter(m.paper_filter))
    result = runtime.plan(np.zeros(3), (.2, 0.), f, nominal_goal_grid_xy=(1., 0.))
    assert result.guidance_tracking.tangent_velocity_fraction == 0.
    assert result.guidance_tracking.tangent_blend_mode == "convex"
    assert m.guidance_config.tangent_velocity_fraction == 1.
    assert runtime.mpc is m and f.sensor_timestamp_seconds == 1.
    assert f.motion_passing_reference_track_ids == ()


@pytest.mark.parametrize("h,goal,fraction", [(2., (2., 0.), 1.), (.6, (2., 0.), 0.)])
def test_inactive_convex_preference_matches_additive(h, goal, fraction):
    kwargs = dict(field(h=h), selected_passing_tangents=((7, (1., -1.), "left"),))
    preferences = []
    for mode in ("additive", "convex"):
        m = planner(mode)
        m.guidance_config = replace(m.guidance_config, tangent_velocity_fraction=fraction)
        preferences.append(m._reference_preference(np.zeros(3), np.array(goal), .2,
            m._sample(np.zeros(3), kwargs), kwargs))
    np.testing.assert_array_equal(*preferences)


def test_additive_default_retains_identical_full_result():
    a, b = selected_planner(), planner("additive")
    results = [m.plan(np.zeros(3), (.2, 0.), nominal_goal_grid_xy=(1., 0.),
        selected_passing_tangents=((7, (1., -1.), "left"),), **field()) for m in (a, b)]
    for item in fields(results[0]):
        x, y = (getattr(r, item.name) for r in results)
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y)
        else:
            assert x == y


@pytest.mark.parametrize("changes", [dict(tangent_blend_mode="unknown"),
    dict(reference_mode="filtered_goal_velocity", tangent_velocity_fraction=0.),
    dict(reference_tangent_basis="planar_gradient"), dict(weight=0., tangent_velocity_fraction=0.)])
def test_invalid_convex_config_is_rejected(changes):
    with pytest.raises(ValueError, match="convex tangent blend"):
        replace(planner().guidance_config, **changes)


@pytest.mark.parametrize('predictive', [False, True])
def test_live_planner_uses_actual_pose_only_guide_target_instead_of_passed_checkpoint(predictive):
    # Exercise the actual submit expression without importing Isaac or the demo.
    # This catches a valid pose-only guide command paired with the old checkpoint
    # target in MPC, which would pull a detour back toward an already passed guide.
    script = Path(__file__).resolve().parents[2] / "scripts/demos/cra_live_object_isaac_demo.py"
    tree = ast.parse(script.read_text())
    submit = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "submit"
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "planner"]
    assert len(submit) == 1
    packet = next(k.value for k in submit[0].keywords if k.arg == "packet")
    # The explicit prediction request adds its actual state epoch at index5;
    # both legacy and predictive requests must retain the pose-only goal at4.
    packet = packet.body if predictive else packet.orelse
    expression = compile(ast.Expression(packet.elts[4]), str(script), "eval")
    for guides in (False, True):
        waypoint = WaypointController([Waypoint(1, 0), Waypoint(2, 0), Waypoint(3.5, 0)],
            route_start=Waypoint(0, 0),
            config=WaypointControllerConfig(intermediate_waypoints_as_guides=guides))
        nominal = waypoint.compute_command(RobotPose2D(.8, .4, 0.))
        args = SimpleNamespace(checked_rear_pass_reference=False, guidance_tracking_weight=100.)
        result = eval(expression, {"np": np, "args": args, "nominal": nominal})
        np.testing.assert_array_equal(result, [nominal.target_waypoint.x, nominal.target_waypoint.y])
        assert result[0] == pytest.approx(1.8 if guides else 1.)
        args.guidance_tracking_weight = 0.
        assert eval(expression, {"np": np, "args": args, "nominal": nominal}) is None


@pytest.mark.parametrize("extra", [[], ["--parallel-guidance-ablation"]])
def test_convex_cli_rejects_unlabeled_or_checkpoint_trial_before_isaac(tmp_path, extra):
    script = Path(__file__).resolve().parents[2] / "scripts/demos/cra_live_object_isaac_demo.py"
    result = subprocess.run([sys.executable, str(script), "--output", str(tmp_path / "unused"),
        "--guidance-tangent-blend", "convex", *extra], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "convex tangent blend requires" in result.stderr
    assert not (tmp_path / "unused").exists()
