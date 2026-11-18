"""Gate the distant-start mission without claiming a distant human was observed."""

import hashlib
import json
import math
from pathlib import Path


def distant_start_retention_audit(report):
    """Exercise coverage belongs to the mandatory near-view gate; violations never do."""
    checks = report['checks']
    required = {k: v for k, v in checks.items() if k != 'five_distinct_retained_sources'}
    if set(required) != {'original_fresh_visual_prior', 'current_positive_body_identity',
                         'original_source_expiry', 'decayed_confidence_widened_bounds'}:
        raise ValueError('unexpected retention audit contract')
    return dict(passed=all(v is True for v in required.values()), checks=required,
                full_retention_gate_passed=report['passed'],
                scope='distant-start provenance audit only; five retained sources remain mandatory in the current-source near-view gate and complete navigation audit')


def component_reports_pass(folder):
    """Check the archived native solver reports, including measured clearances."""
    native = folder / 'native'
    if any(not (native / name).exists() for name in (
        'human-sensor-validation.json', 'robot-setup-validation.json'
    )):
        return False
    for path in native.glob('*validation.json'):
        report = json.loads(path.read_text())
        if path.name == 'robot-setup-validation.json':
            values = [report.get(key) for key in ('configured_clearance_m', 'settled_clearance_m')]
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(value) and value > .02 for value in values):
                return False
        elif report.get('passed') is not True:
            return False
    return True


def verify_near_human_gates(folders, root, scene, *, require_background_walk=False,
                           require_upper_departure=False, require_straight_walk=False,
                           require_left_outbound=False, held_dock_probe=False,
                           require_left_outbound_lounge_walk=False,
                           require_left_outbound_west_diner=False):
    """Require current runtime, scene and independent human prerequisites."""
    if held_dock_probe and not require_left_outbound:
        raise ValueError('held dock probe requires the left outbound actor profile')
    if require_upper_departure and not require_background_walk:
        raise ValueError('upper departure requires background walk')
    if require_straight_walk and (require_background_walk or require_upper_departure):
        raise ValueError('one-way gate is separate from the background circuit')
    if require_left_outbound and not require_straight_walk:
        raise ValueError('left outbound requires its one-way actor gate')
    if require_left_outbound_lounge_walk and not require_left_outbound:
        raise ValueError('the lounge continuation requires its left outbound actor gate')
    if require_left_outbound_west_diner and not require_left_outbound_lounge_walk:
        raise ValueError('the west diner requires the left outbound lounge continuation gate')
    root, scene = Path(root), Path(scene).resolve()
    runtime = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for directory in ('robot', 'scripts') for p in (root/directory).rglob('*.py')
               if p.relative_to(root).as_posix() != 'scripts/evaluation/run_restaurant_demo.py'}
    cases = [('long-parallel-component', 'native/validation.json')]
    if not held_dock_probe:
        cases.append(('long-parallel-near', 'stationary-gate.json'))
    cases.append(('long-idle-component', 'native/validation.json'))
    if len(folders) != len(cases):
        raise ValueError('two current-source held-actor gates required' if held_dock_probe
                         else 'three current-source human gates required')
    for folder, (case, filename) in zip(folders, cases):
        folder = Path(folder)
        meta = json.loads((folder/'invocation.json').read_text())
        execution = json.loads((folder/'execution.json').read_text())
        report = json.loads((folder/filename).read_text())
        previous = {p: h for p, h in meta['source_sha256'].items()
                    if p != 'scripts/evaluation/run_restaurant_demo.py'}
        scenes = {Path(p).resolve(): h for p, h in meta['scene_sha256'].items()}
        if (meta['case'] != case or not execution.get('passed')
            or not execution.get('source_unchanged') or report.get('passed') is not True
            or previous != runtime or scene not in scenes
            or any(hashlib.sha256(p.read_bytes()).hexdigest() != h for p, h in scenes.items())):
            raise ValueError('missing, failed or stale near-human gate')
        if ('--left-outbound-service' in meta['command']) != require_left_outbound:
            raise ValueError('left outbound actor profile must match each human gate')
        west_diner_gate = require_left_outbound_west_diner and case != 'long-parallel-component'
        if ('--left-outbound-west-diner' in meta['command']) != west_diner_gate:
            raise ValueError('west diner staging must match the near-view and native idle gates')
        if case.endswith('-near'):
            if not (report.get('human_component_checks') and report.get('human_body_envelope_checks')
                    and all(report['human_component_checks'].values())
                    and all(report['human_body_envelope_checks'].values())):
                raise ValueError('near-view human ownership and body envelope must pass')
        elif case == 'long-idle-component':
            if ('--straight-service-walk' in meta['command']) != require_straight_walk:
                raise ValueError('one-way actor mode must match the motion trial')
            if ('--left-outbound-lounge-walk' in meta['command']) != require_left_outbound_lounge_walk:
                raise ValueError('actor lounge continuation must match the motion trial')
            required = {'hold_at_least_180_seconds', 'held_agent_disabled',
                'idle_drift_below_1mm', 'released_native_walking', 'native_task_completed',
                'native_travel_over_3m', 'unchanged_geometry_guard'}
            if require_background_walk:
                required = {'hold_at_least_180_seconds', 'held_agent_disabled',
                    'idle_drift_below_1mm', 'background_walk_exercised', 'background_loop_completed',
                    'oncoming_segment_completed', 'continuation_exercised',
                    'no_native_idle_after_start', 'measured_walking', 'independent_geometry',
                    'grounded', 'bounded_region'}
                if '--background-service-walk' not in meta['command']:
                    raise ValueError('v2 requires its actual native background circuit gate')
                upper = '--background-upper-departure' in meta['command']
                if upper != require_upper_departure:
                    raise ValueError('actor departure mode must match the motion trial')
                if require_upper_departure:
                    required |= {'explicit_upper_departure', 'upper_aisle_reached'}
            if require_straight_walk:
                required |= {'explicit_one_way','native_one_way_completed','straight_aisle',
                    'forward_only_actor','complete_aisle_travel','independent_geometry','grounded','measured_walking'}
            if set(report.get('checks', {})) != required or not all(
                value is True for value in report['checks'].values()
            ):
                raise ValueError('long idle and native release must pass')
        else:
            if not component_reports_pass(folder):
                raise ValueError('all native component reports must pass')
    return True
