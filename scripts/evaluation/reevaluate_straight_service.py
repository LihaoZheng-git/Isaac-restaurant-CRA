"""Re-evaluate only the legacy actor-region mismatch on an archived v3 run.

Never overwrite native reports or claim a new wheel execution. Requires exact
legacy-summary reproduction, all other original gates, and unchanged inputs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robot.evaluation.human_navigation_validation import summarize_human_navigation
from robot.evaluation.parallel_walking_validation import anchored_nominal_route_world, validate_parallel_curve


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main(source, output):
    if output.exists():
        raise ValueError('preserve prior evaluations; choose a fresh output')
    names = ['report.json', 'execution.json', 'invocation.json', 'run-configuration.json',
        'fixture-monitor.json', 'human-truth-evaluation-only.json', 'live-fields.jsonl',
        'launch.log', 'stationary-gate.json', 'human-navigation-validation.json',
        'parallel-walking-validation.json', 'straight-walk-validation.json',
        'live-human-envelope-validation.json', 'live-human-mass-identity-retention-validation.json',
        'tracking-uncertainty-validation.json', 'unknown-obstacle-validation.json',
        'parallel-selection-validation.json']
    hashes = {name: digest(source/name) for name in names}
    def read(name):
        return json.loads((source/name).read_text())
    original, execution, invocation = read('report.json'), read('execution.json'), read('invocation.json')
    cfg = read('run-configuration.json')
    assert original['straight_service_walk'] and '--straight-service-walk' in invocation['command']
    assert cfg['straight_service_walk'] and cfg['restaurant_human'] == 'parallel'
    assert execution['source_unchanged'] and execution['launcher_exit_code'] == 0
    assert not execution['passed'] and not original['passed']
    log = (source/'launch.log').read_text(errors='replace')
    assert log.count('Traceback (most recent call last):') == 1
    assert 'RuntimeError: live wheel gate failed; inspect saved diagnostics' in log
    for flag in ('encounter_guidance_ablation', 'predictive_human_geometry_ablation',
                 'observed_motion_reference', 'predictive_parallel_redesign',
                 'human_carry_bowl', 'moving_actor'):
        assert not cfg[flag], flag
    assert cfg['crossing_strategy'] == 'legacy'
    assert original['post_goal_observation'] is None
    for prerequisite in invocation['prerequisites']:
        evidence = json.loads((Path(prerequisite['path'])/'execution.json').read_text())
        assert evidence['passed'] and evidence['source_unchanged']
    for name in names:
        if name.endswith('validation.json') and name not in (
                'human-navigation-validation.json', 'parallel-walking-validation.json'):
            assert read(name)['passed'], name
    assert read('stationary-gate.json')['passed']
    assert original['fixture_root_layer_unchanged']
    assert all(digest(Path(p)) == h for p, h in original['scene_source_sha256'].items())
    assert original['stationary_furniture_memory_validation']['source_aligned_human_audits_pass']
    assert original['completed'] and original['reason'] == 'goal_reached'
    assert original['waypoints_reached'] == 8 and original['minimum_clearance_m'] > .02
    assert original['stop_drift_m'] < .05
    assert original['live_field_count'] >= 10 and original['fields_committed_while_translating'] >= 10

    # Keep exactly the fields consumed by the validators, not large occupancy histories.
    records = []
    with (source/'live-fields.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            objects = []
            for obj in row['objects']:
                objects.append(dict(track_id=obj['track_id'], cra_control=dict(safe_sage=obj['cra_control']['safe_sage']),
                    attributes={k: obj['attributes'][k] for k in ('mass', 'motion')},
                    diagnostics={k: obj['diagnostics'][k] for k in ('centroid_world_xy',
                        'policy_return_count', 'factor_scores', 'property_evidence')}))
            records.append(dict(sensor_timestamp=row['sensor_timestamp'],
                commit_simulation_time=row['commit_simulation_time'], objects=objects,
                diagnostics={k: row['diagnostics'][k] for k in
                    ('measured_robot_velocity_world_xy', 'applied_field_policies')}))
    assert len(records) == original['live_field_count']
    truth = read('human-truth-evaluation-only.json')
    legacy_human = summarize_human_navigation(records, truth, mode='parallel')
    native_human = read('human-navigation-validation.json')
    assert all(native_human[k] == v for k, v in legacy_human.items())
    assert {k for k, v in legacy_human['checks'].items() if not v} == {'actual_parallel_walking'}
    route = anchored_nominal_route_world(read('fixture-monitor.json')['anchor_world_xy_yaw'],
                                         original['nominal_route_local_xy'])
    curve_args = dict(goal_world_xy=route[-1], nominal_route_world_xy=route)
    legacy_curve = validate_parallel_curve(records, truth, **curve_args)
    assert legacy_curve == read('parallel-walking-validation.json')
    assert {k for k, v in legacy_curve['checks'].items() if not v} == {'actual_parallel_actor'}
    assert {k for k, v in legacy_curve['actor_component']['checks'].items() if not v} == {'bounded_actor_region'}
    human = summarize_human_navigation(records, truth, mode='parallel', straight_service_walk=True)
    curve = validate_parallel_curve(records, truth, **curve_args, straight_service_walk=True)
    assert human['passed'] and curve['passed']
    assert all(digest(source/name) == h for name, h in hashes.items())
    corrected = copy.deepcopy(original)
    corrected['human_navigation_validation'].update(human)
    corrected['human_navigation_validation']['parallel_curve'] = curve
    corrected.update(passed=True, evaluation_scope='Post-run explicit v3 actor-protocol correction; original native report remains failed; no new wheel execution')
    result = dict(passed=True, runtime_revision=invocation['source_revision'],
        original_native_report_passed=False, original_native_execution_passed=False,
        no_native_rerun=True, legacy_results_exactly_reproduced=True,
        original_files_unchanged=True, original_file_sha256=hashes,
        corrected_check='Explicit one-way actor y=23.5 start, same 0.5m authored-region allowance; original physical/limb/speed gates plus complete one-way audit',
        scope='Reporting correction only. No trajectory, controller, margins, CRA, evidence, timestamps or acceptance tolerances modified.')
    output.mkdir(parents=True)
    for name, value in [('report.json', corrected), ('human-navigation-validation.json', human),
                        ('parallel-walking-validation.json', curve), ('correction.json', result)]:
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'original_file_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    main(args.source, args.output)
