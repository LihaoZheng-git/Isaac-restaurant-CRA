"""The short harness changes setup/acceptance scope, never the full policy."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot.evaluation.service_crate_short_probe import (
    REFERENCE_RECIPE, ROUTE_LOCAL_XY, START_WORLD_XY_YAW,
    summarize_unexercised_human_scope,
)


SCRIPT = Path('scripts/demos/cra_live_object_isaac_demo.py').resolve()
RECIPE = Path('configs/restaurant_service_crate_short_full_controller.json')


def commands(path):
    return json.loads(Path(path).read_text())['commands']


def parse(monkeypatch, command, *, motion=False):
    calls = []
    monkeypatch.setattr('robot.evaluation.restaurant_long_service_validation.verify_near_human_gates',
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    values = command + ['--output', 'outputs/unused-service-crate-short-test']
    if motion:
        values += ['--motion', '--long-service-human-gates', 'component', 'near', 'idle']
    monkeypatch.setattr('sys.argv', values)
    namespace = {'__file__': str(SCRIPT)}
    exec(compile(SCRIPT.read_text().split('args.output.mkdir', 1)[0], str(SCRIPT), 'exec'), namespace)
    return namespace, calls


@pytest.mark.parametrize('motion', [False, True])
def test_full_arguments_and_existing_prerequisites_are_retained(monkeypatch, motion):
    result, calls = parse(monkeypatch, commands(RECIPE)['long-parallel'], motion=motion)
    reference, _ = parse(monkeypatch, commands(REFERENCE_RECIPE)['long-parallel'])
    args = result['args']
    assert result['service_crate_short_profile']['passed']
    assert args.mpc_horizon_steps == 6 and args.guidance_tracking_weight == 100
    assert args.stationary_furniture_memory and args.unknown_persistent_background_prior
    assert args.unknown_distinct_surface_witnesses and not args.static_occupancy_memory_prior
    allowed = {'service_crate_short_probe', 'maximum_seconds', 'motion', 'long_service_human_gates'}
    assert {k: v for k, v in vars(args).items() if k not in allowed} == {
        k: v for k, v in vars(reference['args']).items() if k not in allowed}
    assert len(calls) == int(motion)
    assert reference['service_crate_short_profile'] is None


def test_recipe_only_changes_selector_and_duration():
    candidate, reference = commands(RECIPE), commands(REFERENCE_RECIPE)
    assert candidate.keys() == reference.keys()
    for key in candidate:
        if key != 'long-parallel':
            assert candidate[key] == reference[key]
    cmd = candidate['long-parallel'].copy()
    cmd.remove('--service-crate-short-probe')
    cmd[cmd.index('--maximum-seconds')+1] = reference['long-parallel'][reference['long-parallel'].index('--maximum-seconds')+1]
    assert cmd == reference['long-parallel']


@pytest.mark.parametrize('flag,value', [
    ('--mpc-horizon-steps', '12'), ('--guidance-tracking-weight', '0'),
    ('--maximum-seconds', '90'), ('--restaurant-scene', 'scenes/other.usda'),
    ('--guidance-tangent-velocity-fraction', '0'), ('--visual-reassessment-interval', '1'),
])
def test_selector_rejects_policy_or_unbounded_setup_drift(monkeypatch, flag, value):
    command = commands(RECIPE)['long-parallel']
    command[command.index(flag)+1] = value
    with pytest.raises(SystemExit):
        parse(monkeypatch, command)


@pytest.mark.parametrize('flag', ['--unknown-persistent-background-prior', '--unknown-distinct-surface-witnesses',
                                 '--stationary-furniture-memory', '--service-static-props'])
def test_selector_rejects_removed_uncertainty_or_perception_policy(monkeypatch, flag):
    command = commands(RECIPE)['long-parallel']
    command.remove(flag)
    with pytest.raises(SystemExit):
        parse(monkeypatch, command)


def test_fixed_route_override_preserves_height_and_actor_settings():
    tree = ast.parse(SCRIPT.read_text())
    block = next(node for node in tree.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == 'args.service_crate_short_probe'
                 and any(isinstance(child, ast.Assign) and any(ast.unparse(t) == 'restaurant_target' for t in child.targets)
                         for child in node.body))
    stationary = SimpleNamespace(ROBOT_START_POSITION_WORLD=(7.8, 4.05, .2), ROBOT_START_YAW_RADIANS=1.)
    args = SimpleNamespace(service_crate_short_probe=True, human_crossing_trigger_y=9.)
    env = {'args': args, 'stationary': stationary}
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(SCRIPT), 'exec'), env)
    assert stationary.ROBOT_START_POSITION_WORLD == (*START_WORLD_XY_YAW[:2], .2)
    assert stationary.ROBOT_START_YAW_RADIANS == 0.
    assert env['ROUTE'] == ROUTE_LOCAL_XY and env['GOAL'] == (4., 0.)
    assert env['restaurant_target'] == 'BrokenCrate'
    assert args.human_crossing_trigger_y == 9.


def held_truth():
    return [dict(time=float(i), actor_xyz=[29.8, 23.3, 0.], robot_xy=[15.65+i*.1, 4.05],
                 phase='idle_before', initial_idle_held=True, native_agent_enabled=False)
            for i in range(4)]


def test_no_encounter_is_unexercised_never_a_human_pass():
    result = summarize_unexercised_human_scope([{'diagnostics': {}}], held_truth())
    assert result['unexercised'] and result['status'] == 'unexercised'
    assert result['human_encounter_passed'] is None


@pytest.mark.parametrize('key', ['human_body_envelope_priors', 'tracking_uncertainty_regions',
                                'motion_passing_reference_track_ids', 'crossing_yield_track_ids'])
def test_used_human_evidence_keeps_original_acceptance(key):
    result = summarize_unexercised_human_scope([{'diagnostics': {key: [1]}}], held_truth())
    assert not result['unexercised']


def test_generic_mask_label_does_not_invent_a_human_encounter():
    record = {'diagnostics': {'image_object_groups': {'unverified_person_mask': True}}}
    assert summarize_unexercised_human_scope([record], held_truth())['unexercised']


@pytest.mark.parametrize('change', ['walking', 'nearby', 'drift', 'unordered', 'nonfinite'])
def test_missing_or_contradictory_scope_never_excludes_original_human_checks(change):
    truth = deepcopy(held_truth())
    if change == 'walking':
        truth[-1]['phase'] = 'move'
    elif change == 'nearby':
        truth[-1]['robot_xy'] = truth[-1]['actor_xyz'][:2]
    elif change == 'drift':
        truth[-1]['actor_xyz'][0] += .002
    elif change == 'unordered':
        truth[-1]['time'] = 0.
    else:
        truth[-1]['actor_xyz'][0] = float('nan')
    assert not summarize_unexercised_human_scope([{'diagnostics': {}}], truth)['unexercised']


def test_empty_scope_cannot_exclude_human_acceptance():
    assert not summarize_unexercised_human_scope([], [])['unexercised']


@pytest.mark.parametrize('robot_completed,human_evidence,expected', [
    (False, {}, False),
    (True, {}, True),
    (True, {'human_body_envelope_priors': [1]}, False),
    (True, {'tracking_uncertainty_regions': [1]}, False),
])
def test_actual_report_scope_keeps_robot_and_exercised_human_failures(robot_completed, human_evidence, expected):
    tree = ast.parse(SCRIPT.read_text())
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == 'args.service_crate_short_probe'
                 and any(isinstance(child, ast.Assign) and any(ast.unparse(t) == 'service_crate_short_validation'
                         for t in child.targets) for child in node.body))
    saved = {}
    env = dict(args=SimpleNamespace(service_crate_short_probe=True),
               live_fields=[{'diagnostics': human_evidence}], human=SimpleNamespace(records=held_truth()),
               completed=False, robot_completion_before_human_audits=robot_completed,
               human_validation={'passed': False, 'checks': {'existing_raw_failure': False}},
               save_json=lambda name, value: saved.update({name: value}))
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(SCRIPT), 'exec'), env)
    assert env['completed'] is expected
    assert saved['human-navigation-validation.json']['passed'] is False
    assert saved['human-navigation-validation.json']['checks']['existing_raw_failure'] is False
    assert saved['service-crate-short-validation.json']['human_encounter_passed'] is None
