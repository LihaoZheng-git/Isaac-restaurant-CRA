import hashlib
import json

import pytest

from robot.evaluation.restaurant_long_service_validation import verify_near_human_gates


def test_distant_start_defers_exercise_count_but_never_provenance_failure():
    from robot.evaluation.restaurant_long_service_validation import distant_start_retention_audit
    report = dict(passed=False, checks=dict(original_fresh_visual_prior=True,
        current_positive_body_identity=True, original_source_expiry=True,
        decayed_confidence_widened_bounds=True, five_distinct_retained_sources=False))
    assert distant_start_retention_audit(report)['passed']
    assert not distant_start_retention_audit(report)['full_retention_gate_passed']
    for key in list(report['checks'])[:-1]:
        bad = dict(report, checks=dict(report['checks'], **{key: False}))
        assert not distant_start_retention_audit(bad)['passed']


@pytest.mark.parametrize('background,upper,straight', [(False,False,False),(True,False,False),(True,True,False),(False,False,True)])
def test_remote_start_cannot_replace_current_source_near_human_gates(tmp_path, background, upper, straight):
    (tmp_path/'robot').mkdir()
    source = tmp_path/'robot/controller.py'
    source.write_text('control')
    scene = tmp_path/'scene.usda'
    scene.write_text('scene')
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    folders = []
    options = dict(require_background_walk=background, require_upper_departure=upper,require_straight_walk=straight)
    command = (['--background-service-walk'] if background else []) + (
        ['--background-upper-departure'] if upper else []) + (['--straight-service-walk'] if straight else [])
    for case in ('long-parallel-component', 'long-parallel-near', 'long-idle-component'):
        folder = tmp_path/case
        folder.mkdir()
        folders.append(folder)
        (folder/'invocation.json').write_text(json.dumps(dict(case=case, command=command,
            source_sha256={'robot/controller.py': sha(source)}, scene_sha256={str(scene): sha(scene)})))
        (folder/'execution.json').write_text(json.dumps(dict(passed=True, source_unchanged=True)))
        if case == 'long-idle-component':
            native = folder/'native'
            native.mkdir()
            checks = {k: True for k in ('hold_at_least_180_seconds', 'held_agent_disabled',
                'idle_drift_below_1mm', 'released_native_walking', 'native_task_completed',
                'native_travel_over_3m', 'unchanged_geometry_guard')}
            if background:
                checks = {k: True for k in ('hold_at_least_180_seconds', 'held_agent_disabled',
                    'idle_drift_below_1mm', 'background_walk_exercised', 'background_loop_completed',
                    'oncoming_segment_completed', 'continuation_exercised', 'no_native_idle_after_start',
                    'measured_walking', 'independent_geometry', 'grounded', 'bounded_region')}
            if upper:
                checks.update(explicit_upper_departure=True, upper_aisle_reached=True)
            if straight:
                checks.update({k:True for k in ('explicit_one_way','native_one_way_completed','straight_aisle',
                    'forward_only_actor','complete_aisle_travel','independent_geometry','grounded','measured_walking')})
            (native/'validation.json').write_text(json.dumps(dict(passed=True, checks=checks)))
        elif case.endswith('component'):
            native = folder/'native'
            native.mkdir()
            for name in ('validation', 'human-sensor-validation'):
                (native/(name+'.json')).write_text(json.dumps(dict(passed=True)))
            (native/'robot-setup-validation.json').write_text(json.dumps(dict(
                configured_clearance_m=.5, settled_clearance_m=.5)))
        else:
            (folder/'stationary-gate.json').write_text(json.dumps(dict(passed=True,
                human_component_checks={'ownership': True}, human_body_envelope_checks={'contained': True})))
    with pytest.raises(ValueError, match='three'):
        verify_near_human_gates(folders[:2], tmp_path, scene, **options)
    assert verify_near_human_gates(folders, tmp_path, scene, **options)
    idle_gate = folders[2]/'native/validation.json'
    idle_report = json.loads(idle_gate.read_text())
    if background:
        with pytest.raises(ValueError, match='departure mode'):
            verify_near_human_gates(folders, tmp_path, scene,
                require_background_walk=True, require_upper_departure=not upper)
    if upper:
        for key in ('explicit_upper_departure', 'upper_aisle_reached'):
            for value in (False, None):
                bad = json.loads(json.dumps(idle_report))
                if value is None:
                    del bad['checks'][key]
                else:
                    bad['checks'][key] = value
                idle_gate.write_text(json.dumps(bad))
                with pytest.raises(ValueError, match='long idle'):
                    verify_near_human_gates(folders, tmp_path, scene, **options)
        idle_gate.write_text(json.dumps(idle_report))
    idle_report['checks']['idle_drift_below_1mm'] = False
    idle_gate.write_text(json.dumps(idle_report))
    with pytest.raises(ValueError, match='long idle'):
        verify_near_human_gates(folders, tmp_path, scene, **options)
    idle_report['checks']['idle_drift_below_1mm'] = True
    idle_gate.write_text(json.dumps(idle_report))
    source.write_text('changed control')
    with pytest.raises(ValueError, match='stale'):
        verify_near_human_gates(folders, tmp_path, scene, **options)
    source.write_text('control')
    gate = folders[1]/'stationary-gate.json'
    gate.write_text(json.dumps(dict(passed=True, human_component_checks={'ownership': True})))
    with pytest.raises(ValueError, match='near-view'):
        verify_near_human_gates(folders, tmp_path, scene, **options)


def test_upper_departure_preflight_requires_background_mode(tmp_path):
    with pytest.raises(ValueError, match='requires background'):
        verify_near_human_gates([], tmp_path, tmp_path/'scene', require_upper_departure=True)
