"""The oracle benchmark can observe a simulated pass but cannot decide its wheel commands."""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot.evaluation.crate_clearance_observation import (
    summarize_clearance_samples,
    world_pose_from_local,
)


DEMO = Path('scripts/demos/cra_live_object_isaac_demo.py').resolve()


def parse_recipe(monkeypatch, recipe, extra_args):
    command = json.loads(Path(f'configs/{recipe}.json').read_text())['commands']['crate-static']
    monkeypatch.setattr('sys.argv', command + extra_args + ['--output', 'outputs/unused-crate-observation'])
    ns = {'__file__': str(DEMO)}
    exec(compile(DEMO.read_text().split('args.output.mkdir', 1)[0], str(DEMO), 'exec'), ns)
    return ns['args']


@pytest.mark.parametrize('recipe', ['restaurant_crate_route_probe', 'restaurant_crate_delay_robust_bypass'])
def test_passive_measurement_admits_both_primary_and_existing_ablation(monkeypatch, recipe):
    assert not parse_recipe(monkeypatch, recipe, []).observe_crate_clearance
    assert parse_recipe(monkeypatch, recipe, ['--observe-crate-clearance', '--motion']).observe_crate_clearance


@pytest.mark.parametrize('addition', [
    ['--experiment-baseline', 'geometric_velocity_cbf'],
    ['--moving-actor'],
    ['--progress-timeout-seconds', '10'],
])
def test_passive_measurement_rejects_other_experiments(monkeypatch, addition):
    with pytest.raises(SystemExit):
        parse_recipe(monkeypatch, 'restaurant_crate_route_probe', ['--observe-crate-clearance', *addition])


@pytest.mark.parametrize(('observed', 'expected'), [(False, 'clearance_guard'), (True, 'goal_reached')])
def test_actual_navigation_tick_only_legacy_guard_can_preempt_waypoint(observed, expected):
    tree = ast.parse(DEMO.read_text())
    loop, = [n for n in ast.walk(tree) if isinstance(n, ast.While) and 'args.maximum_seconds' in ast.unparse(n.test)]
    clearance_calls = []

    def oracle_clearance(state):
        clearance_calls.append(state)
        if observed:
            pytest.fail('navigation tick read oracle obstacle geometry')
        return 0.01

    ns = dict(
        args=SimpleNamespace(maximum_seconds=45, progress_timeout_seconds=0, observe_crate_clearance=observed),
        best_goal_distance=math.inf, minimum_clearance=math.inf, np=np,
        start_sim=0, timeline=SimpleNamespace(get_current_time=lambda: 1.0),
        state=lambda: np.array([4., 0., 0.]), GOAL=[4., 0.], recorded_crate_poses=[],
        independent_clearance=oracle_clearance, RobotPose2D=lambda *xyz: xyz,
        waypoint=SimpleNamespace(compute_command=lambda pose: SimpleNamespace(route_complete=True)),
    )
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(DEMO), 'exec'), ns)
    assert ns['reason'] == expected
    assert len(clearance_calls) == int(not observed)
    if observed:
        assert ns['recorded_crate_poses'] == [(1., [4., 0., 0.])]


def test_post_run_scoring_includes_exact_threshold_and_braking_crossings():
    rows = [dict(time_seconds=i/60, phase='navigation' if i < 3 else 'braking', clearance_m=x)
            for i, x in enumerate([.1, .02, .01, .04, -.001])]
    result = summarize_clearance_samples(rows)
    assert result['legacy_clearance_passed'] is False
    assert result['minimum_clearance_m'] == -.001
    assert result['samples_at_or_below_threshold'] == 3
    assert [(r['time_seconds'], r['transition']) for r in result['crossings']] == [
        (1/60, 'entered'), (3/60, 'exited'), (4/60, 'entered')]
    assert result['crossings'][-1]['phase'] == 'braking'


def test_rotated_anchor_reconstructs_world_pose_for_static_layout():
    np.testing.assert_allclose(
        world_pose_from_local((4., 1., -.3), (12., 40., math.pi/2)),
        (11., 44., math.pi/2-.3))
