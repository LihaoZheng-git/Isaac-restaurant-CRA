import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from robot.scene.restaurant_background_walk import BackgroundServiceWalk, BACKGROUND_ROUTE, audit_background_walk
from robot.scene.restaurant_visual_steam import author_visual_steam, STEAM_PATH
from robot.evaluation.parallel_walking_validation import validate_parallel_actor
from test_parallel_walking_validation import example


def test_background_walk_uses_actual_arrival_and_reaches_original_parallel_segment():
    route = BackgroundServiceWalk()
    assert not route.advance((29.8, 10.5, 0), 0, encounter_due=True)
    assert route.phase == 'background_walk'  # A trigger cannot teleport or skip the current leg.
    for i, goal in enumerate(BACKGROUND_ROUTE):
        assert route.goal == goal
        assert route.advance(goal, i+1)
    assert route.phase == 'approach_parallel' and route.goal == (29.8, 10.5, 0)
    route.advance(route.goal, 5)
    assert route.phase == 'move' and route.goal == (29.8, 7., 0)
    route.advance(route.goal, 6)
    assert route.phase == 'continue_walk'
    for i in range(8):
        route.advance(route.goal, 7+i)
    assert route.phase == 'continue_walk'
    assert not any(e['next_phase'] == 'idle_before' for e in route.events)


def test_background_walk_does_not_exit_before_the_encounter_request():
    route = BackgroundServiceWalk()
    for i in range(12):
        route.advance(route.goal, i)
    assert route.phase == 'background_walk' and len(route.events) == 12
    with pytest.raises(ValueError):
        route.advance((np.nan, 1, 0), 15)


def test_background_gate_requires_measured_progress_complete_segment_and_geometry():
    rows = [dict(time=i*.1, actor_xyz=[29.8, 12.5-i*.05, 0.],
                 phase='background_walk' if i<40 else 'move' if i<60 else 'continue_walk',
                 actor_layout_clearance_m=.2, collider_alignment_error_m=0.) for i in range(121)]
    events = [dict(previous_phase='background_walk', previous_goal=BACKGROUND_ROUTE[-1]),
              dict(previous_phase='move', next_phase='continue_walk')]
    assert audit_background_walk(rows, events, minimum_background_seconds=3.)['passed']
    stationary = copy.deepcopy(rows)
    for row in stationary:
        row['actor_xyz'] = [29.8, 10., 0.]
    assert not audit_background_walk(stationary, events, minimum_background_seconds=3.)['checks']['measured_walking']
    assert not audit_background_walk(rows, events[:1], minimum_background_seconds=3.)['checks']['oncoming_segment_completed']
    rows[70]['actor_layout_clearance_m'] = .01
    assert not audit_background_walk(rows, events, minimum_background_seconds=3.)['checks']['independent_geometry']


def test_v2_parallel_validation_preserves_geometry_and_direction_failures():
    _, truth = example()
    for r in truth:
        r['background_service_walk'] = True
    before = copy.deepcopy(truth[0])
    before.update(time=-1., phase='background_walk', actor_xyz=[28.5, 12., 0.])
    rows = [before, *truth]
    assert validate_parallel_actor(rows, parked_robot=False)['passed']
    before['actor_layout_clearance_m'] = .01
    assert not validate_parallel_actor(rows, parked_robot=False)['passed']
    before['actor_layout_clearance_m'] = .3
    before['background_service_walk'] = False
    with pytest.raises(ValueError, match='explicit v2'):
        validate_parallel_actor(rows, parked_robot=False)


def test_rendered_steam_has_no_collider_mass_or_factor_label():
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(60)
    source = UsdGeom.Cube.Define(stage, '/World/Pot')
    source.AddTranslateOp().Set((3, 4, 1))
    result = author_visual_steam(stage, source_path='/World/Pot')
    assert not result['available_as_factor_input'] and not result['colliders']
    puffs = list(stage.GetPrimAtPath(STEAM_PATH).GetChildren())
    assert len(puffs) == 5
    for i, p in enumerate(puffs):
        assert not p.HasAPI(UsdPhysics.CollisionAPI) and not p.HasAPI(UsdPhysics.MassAPI)
        samples = p.GetAttribute('xformOp:translate').GetTimeSamples()
        assert samples[0] == 0 and samples[-1] == 18000
        for t in (0., 1., 120.):
            center = UsdGeom.XformCache(Usd.TimeCode(t*60)).GetLocalToWorldTransform(p).ExtractTranslation()
            np.testing.assert_allclose(center, [3+.018*np.sin(2*t+i),
                4+.012*np.cos(1.3*t+i), 2+.12+.075*i+.012*np.sin(3*t+i)])
    with pytest.raises(ValueError, match='already'):
        author_visual_steam(stage, source_path='/World/Pot')


@pytest.mark.parametrize('case', ['long-parallel', 'long-parallel-near', 'long-parallel-component',
                                'long-idle-component', 'bin-view', 'rounded-view',
                                'utensils-view', 'pot-view', 'pot-plume-component-view'])
@pytest.mark.parametrize('recipe', ['restaurant_service_demo_v2.json', 'restaurant_service_demo_v2_fragment_proof.json',
                                  'restaurant_service_demo_v2_short_fragment.json'])
def test_v2_recipes_pass_real_cli_checks_without_starting_kit(case, recipe, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = json.loads((root/'configs'/recipe).read_text())['commands'][case]
    source = root/command[0]
    prefix = source.read_text().split('args.output.mkdir', 1)[0]
    monkeypatch.setattr(sys, 'argv', [*command, '--output', str(tmp_path/'native')])
    monkeypatch.setattr(sys, 'path', list(sys.path))
    scope = dict(__file__=str(source), __name__='cli_validation_test')
    exec(compile(prefix, str(source), 'exec'), scope)
    assert not (tmp_path/'native').exists()


def test_fragment_recipe_changes_only_the_explicit_memory_option():
    root = Path(__file__).resolve().parents[2]
    original = json.loads((root/'configs/restaurant_service_demo_v2.json').read_text())['commands']
    candidate = json.loads((root/'configs/restaurant_service_demo_v2_fragment_proof.json').read_text())['commands']
    assert original.keys() == candidate.keys()
    for case, command in candidate.items():
        enabled = '--stationary-furniture-memory' in original[case]
        assert command.count('--stationary-furniture-fragment-proof') == int(enabled)
        assert [v for v in command if v != '--stationary-furniture-fragment-proof'] == original[case]


@pytest.mark.parametrize('case', ['long-parallel', 'long-parallel-component'])
def test_fragment_cli_rejects_without_memory_before_kit(case, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = json.loads((root/'configs/restaurant_service_demo_v2_fragment_proof.json').read_text())['commands'][case]
    command.remove('--stationary-furniture-memory')
    source = root/command[0]
    prefix = source.read_text().split('args.output.mkdir', 1)[0]
    monkeypatch.setattr(sys, 'argv', [*command, '--output', str(tmp_path/'native')])
    monkeypatch.setattr(sys, 'path', list(sys.path))
    with pytest.raises(SystemExit):
        exec(compile(prefix, str(source), 'exec'), dict(__file__=str(source), __name__='cli_test'))
    assert not (tmp_path/'native').exists()


def test_short_fragment_recipe_preserves_every_other_command_option():
    root = Path(__file__).resolve().parents[2]
    original = json.loads((root/'configs/restaurant_service_demo_v2_fragment_proof.json').read_text())['commands']
    candidate = json.loads((root/'configs/restaurant_service_demo_v2_short_fragment.json').read_text())['commands']
    assert original.keys() == candidate.keys()
    for case, command in candidate.items():
        enabled = case in ('long-parallel', 'long-parallel-near', 'long-parallel-component')
        assert command.count('--human-short-fragment-coverage') == int(enabled)
        assert [v for v in command if v != '--human-short-fragment-coverage'] == original[case]


@pytest.mark.parametrize('case,required', [('long-parallel','--human-fragment-coverage-prior'),
    ('long-parallel-component','--identity-selection-body-envelope')])
def test_short_fragment_cli_requires_explicit_parent_mode(case, required, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = json.loads((root/'configs/restaurant_service_demo_v2_short_fragment.json').read_text())['commands'][case]
    command.remove(required)
    source = root/command[0]
    monkeypatch.setattr(sys, 'argv', [*command, '--output', str(tmp_path/'native')])
    monkeypatch.setattr(sys, 'path', list(sys.path))
    with pytest.raises(SystemExit):
        exec(compile(source.read_text().split('args.output.mkdir', 1)[0], str(source), 'exec'),
             dict(__file__=str(source), __name__='cli_test'))
    assert not (tmp_path/'native').exists()
