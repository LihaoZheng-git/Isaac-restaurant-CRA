from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from robot.evaluation.unknown_obstacle_validation import summarize_unknown_obstacles
from robot.perception.unknown_obstacle_memory import UnknownObstacleConfig, UnknownObstacleMemory
from test_persistent_background_prior import proof_record
from test_stationary_furniture_memory import observe, GUIDANCE
from test_confirmed_background_motion import SHAPE, feed


POINTS = np.array([[1., -.1], [1.02, -.08], [1.16, .04], [1.18, .06]])


def memory():
    return UnknownObstacleMemory(UnknownObstacleConfig(supported_background_prior=True,
        persistent_background_prior=True, unclassified_static_prior=True))


@pytest.mark.parametrize('now', [1.2, 20., 300.])
def test_one_observation_stays_occupied_without_invented_stationarity_or_renewed_clocks(now):
    m = memory()
    observe(m, 1., points=POINTS)
    old = m.records[1]
    s = observe(m, now, points=np.empty((0, 2)))
    r, = s.regions
    assert r.mode == 'unclassified_static_geometry_prior' and r.radius_m == .30
    assert r.original.source_time == 1. and r.original.source_frame == 10
    assert r.original.stationary_support_time is None and r.original.background_support is None
    assert r.original.dynamic_evidence is None and r.original.guidance == old.guidance
    np.testing.assert_array_equal(r.original.points, POINTS)
    assert summarize_unknown_obstacles([proof_record(s)])['passed']


def test_option_off_preserves_original_expanding_unknown_motion():
    m = UnknownObstacleMemory(UnknownObstacleConfig(supported_background_prior=True, persistent_background_prior=True))
    observe(m, 1., points=POINTS)
    r, = observe(m, 20., points=np.empty((0, 2))).regions
    assert r.mode == 'unknown_motion_reachable_surface_prior' and r.radius_m > 19.


@pytest.mark.parametrize('kind', ['mask', 'body'])
@pytest.mark.parametrize('when', ['current', 'lost'])
def test_positive_human_exclusion_is_sticky_after_cue_disappears(kind, when):
    m = memory()
    kwargs = {'excluded_points': [[1., 0.]]} if kind == 'mask' else {'human_regions': [(np.array([1., 0.]), .65)]}
    observe(m, 1., points=POINTS, **(kwargs if when == 'current' else {}))
    observe(m, 1.2, points=np.empty((0, 2)), **(kwargs if when == 'lost' else {}))
    s = observe(m, 1.4, points=np.empty((0, 2)))
    r, = s.regions
    assert r.mode == 'unknown_motion_reachable_surface_prior'
    assert r.original.dynamic_evidence is not None and r.radius_m >= .30+.4+1.55-1e-9
    assert summarize_unknown_obstacles([proof_record(s)])['passed']


def test_confirmed_mover_remains_dynamic_after_current_velocity_and_id_disappear():
    m = memory()
    for t, shift in ((1., 0.), (1.2, .08), (1.4, .16)):
        s = feed(m, t, SHAPE+[1.+shift, 0.])
    assert s.observed_dynamic_evidence[0]['kind'] == 'confirmed_geometric_motion'
    r, = observe(m, 1.6, points=np.empty((0, 2))).regions
    assert r.mode == 'unknown_motion_reachable_surface_prior'
    assert r.original.source_time == 1.4
    assert r.original.dynamic_evidence['source_time'] == 1.4


def test_current_whole_surface_replacement_cannot_erase_dynamic_hazard():
    m = memory()
    for t, shift in ((1., 0.), (1.2, .08), (1.4, .16)):
        feed(m, t, SHAPE+[1.+shift, 0.])
    feed(m, 1.6, SHAPE+[1.18, 0.], key=3)
    assert m.records[3].dynamic_evidence['geometry_replacement']['previous_track_id'] == 2
    r, = observe(m, 1.8, points=np.empty((0, 2))).regions
    assert r.mode == 'unknown_motion_reachable_surface_prior'


def test_centroid_jump_alone_does_not_establish_physical_motion():
    m = memory()
    observe(m, 1., points=POINTS, velocity=(.8, 0.))
    observe(m, 1.2, points=POINTS[:3], velocity=(.8, 0.))
    r, = observe(m, 1.4, points=np.empty((0, 2))).regions
    assert r.mode == 'unclassified_static_geometry_prior'


def test_higher_original_risk_and_flux_are_retained():
    from robot.safety.class_safety_policy import ClassSafetyParameters
    m = memory()
    observe(m, 1., points=POINTS)
    high = replace(GUIDANCE, safety_parameters=ClassSafetyParameters(.9, 2.))
    m.records[1] = replace(m.records[1], guidance=high)
    r, = observe(m, 20., points=np.empty((0, 2))).regions
    assert r.original.guidance.risk == .9 and r.original.guidance.boundary_flux_magnitude == 2.


@pytest.mark.parametrize('kwargs', [dict(unclassified_static_prior=True),
    dict(supported_background_prior=True, unclassified_static_prior=True),
    dict(supported_background_prior=True, persistent_background_prior=True, unclassified_static_prior=1)])
def test_new_option_requires_explicit_dependencies(kwargs):
    with pytest.raises(ValueError):
        UnknownObstacleConfig(**kwargs)


@pytest.mark.parametrize('tamper', ['radius', 'opt_in', 'dynamic'])
def test_audit_rejects_invalid_static_default_claim(tamper):
    m = memory()
    observe(m, 1., points=POINTS)
    record = proof_record(observe(m, 2., points=np.empty((0, 2))))
    d = record['unknown_obstacle_memory']
    if tamper == 'radius':
        d['regions'][0]['radius_m'] = .1
    elif tamper == 'opt_in':
        d['config']['unclassified_static_prior'] = False
    else:
        d['regions'][0]['dynamic_evidence'] = dict(kind='positive_human_region', source_time=1.)
    assert not summarize_unknown_obstacles([record])['passed']


@pytest.mark.parametrize('case', ['long-parallel', 'long-parallel-near', 'long-parallel-component'])
def test_real_cli_recipe_adds_only_opt_in_and_checks_dependency(case, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    old = json.loads((root/'configs/restaurant_service_demo_v2_persistent_background.json').read_text())['commands'][case]
    command = json.loads((root/'configs/restaurant_service_demo_v2_unclassified_static.json').read_text())['commands'][case]
    assert command == [*old, '--unknown-unclassified-static-prior']
    source = root/command[0]
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'argv', [*command, '--output', str(tmp_path/'native')])
    code = compile(source.read_text().split('args.output.mkdir', 1)[0], str(source), 'exec')
    scope = dict(__file__=str(source), __name__='cli_test')
    exec(code, scope)
    assert scope['args'].unknown_unclassified_static_prior
    monkeypatch.setattr(sys, 'argv', [c for c in sys.argv if c != '--unknown-persistent-background-prior'])
    with pytest.raises(SystemExit):
        exec(code, dict(__file__=str(source), __name__='cli_test'))
    assert not (tmp_path/'native').exists()
