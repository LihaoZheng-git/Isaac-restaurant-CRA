"""Keep the static repeat matched to V3 without fabricating a human reference."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from robot.evaluation.crate_v3_controller import V3_STATIC_CONTROLLER
from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter
from test_selected_obstacle_tangent_reference import selected_planner
from test_selected_tangent_reference import runtime_field


def command(name, case='crate-static'):
    return json.loads(Path(f'configs/{name}.json').read_text())['commands'][case]


def parse(monkeypatch, values):
    path = Path('scripts/demos/cra_live_object_isaac_demo.py').resolve()
    monkeypatch.setattr('sys.argv', values + ['--output', 'outputs/unused-v3-controller'])
    namespace = {'__file__': str(path)}
    exec(compile(path.read_text().split('args.output.mkdir', 1)[0], str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('motion', [False, True])
def test_profile_matches_v3_except_existing_no_human_fallback(monkeypatch, motion):
    values = command('restaurant_crate_v3_controller') + (['--motion'] if motion else [])
    result = parse(monkeypatch, values)
    original = parse(monkeypatch, command('restaurant_service_demo_v3_corrected_contact', 'long-parallel'))
    assert result['crate_v3_profile']['passed']
    for key in V3_STATIC_CONTROLLER:
        actual = getattr(result['args'], key)
        expected = getattr(original['args'], key)
        if key == 'guidance_tangent_velocity_fraction':
            assert (actual, expected) == (0., 1.)
        else:
            assert actual == expected, key


def test_all_other_parsed_probe_settings_remain_identical(monkeypatch):
    original = vars(parse(monkeypatch, command('restaurant_crate_route_probe'))['args'])
    current = vars(parse(monkeypatch, command('restaurant_crate_v3_controller'))['args'])
    allowed = set(V3_STATIC_CONTROLLER) | {'crate_v3_controller', 'save_failed_planner_replay'}
    assert {k: v for k, v in current.items() if k not in allowed} == {
        k: v for k, v in original.items() if k not in allowed}


@pytest.mark.parametrize('option,value', [
    ('--mpc-horizon-steps', '30'), ('--guidance-tangent-velocity-fraction', '1'),
    ('--guidance-tracking-weight', '0'), ('--simulation-rate', '.05'),
    ('--guidance-tangent-basis', 'combined_field'),
])
def test_profile_rejects_drift(monkeypatch, option, value):
    values = command('restaurant_crate_v3_controller')
    values[values.index(option) + 1] = value
    with pytest.raises(SystemExit):
        parse(monkeypatch, values)


@pytest.mark.parametrize('flag', [
    '--mpc-tapered-rescue-seed', '--mpc-consistent-temporal-jacobian',
    '--crate-route-probe', '--confirmed-prop-refresh',
])
def test_profile_rejects_missing_prerequisites(monkeypatch, flag):
    values = command('restaurant_crate_v3_controller')
    values.remove(flag)
    with pytest.raises(SystemExit):
        parse(monkeypatch, values)


@pytest.mark.parametrize('flag', ['--restaurant-human', '--motion-aware-passing',
    '--parallel-guidance-ablation', '--mpc-unfinished-qp-proposal'])
def test_profile_rejects_unapproved_policies(monkeypatch, flag):
    with pytest.raises(SystemExit):
        parse(monkeypatch, command('restaurant_crate_v3_controller') + [flag])


def test_explicit_zero_tangent_matches_actual_v3_runtime_fallback(monkeypatch):
    def unexpected_arc(*args, **kwargs):
        pytest.fail('no human reference exists: moving arc must remain inactive')

    monkeypatch.setattr('robot.safety.checked_moving_arc_reference.search_checked_moving_arc', unexpected_arc)
    mpc = selected_planner()
    mpc.guidance_config = replace(mpc.guidance_config, tangent_blend_mode='convex',
        moving_arc_reference=True)
    fallback = SafeSAGEControlRuntime(mpc=mpc,
        backup_filter=SafeSAGEUnicycleCBFFilter(mpc.paper_filter))
    explicit = selected_planner()
    explicit.guidance_config = replace(mpc.guidance_config, tangent_velocity_fraction=0.)
    static = SafeSAGEControlRuntime(mpc=explicit,
        backup_filter=SafeSAGEUnicycleCBFFilter(explicit.paper_filter))
    source = runtime_field()
    first = fallback.plan(np.zeros(3), (.2, 0.), source, nominal_goal_grid_xy=(1., 0.))
    second = static.plan(np.zeros(3), (.2, 0.), source, nominal_goal_grid_xy=(1., 0.))
    assert first.success == second.success
    np.testing.assert_array_equal(first.command_sequence, second.command_sequence)
    assert first.guidance_tracking.tangent_velocity_fraction == 0.
    assert mpc.guidance_config.tangent_velocity_fraction == 1.
    assert source.motion_passing_reference_track_ids == ()
    assert source.sensor_timestamp_seconds == 1.
