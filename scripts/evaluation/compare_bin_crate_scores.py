"""Compare observed prop scores in two isolated, same-center service-aisle runs."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot.evaluation.image_object_runtime import coalesce_image_members_for_evaluation
from robot.evaluation.object_map_validation import summarize_fixture_assessments
from scripts.evaluation.audit_static_prop_motion import audit as audit_retention


METRICS = ('cue', 'contact', 'phi', 'kinetic', 'mass')
BIN_WIDTH_M = .25


def describe(values):
    if not values:
        return None
    if not all(math.isfinite(x) for x in values):
        raise ValueError('nonfinite observed score')
    return dict(count=len(values), median=statistics.median(values), minimum=min(values), maximum=max(values))


def summarize(rows):
    return {name: describe([r[name] for r in rows if r[name] is not None]) for name in METRICS}


def parked_observations(records, fixture):
    rows = []
    for record in coalesce_image_members_for_evaluation(records):
        # Reuse the native parked fixture matcher, including competitor rejection.
        match = summarize_fixture_assessments([record], [fixture], maximum_distance_m=.20)
        item = match['fixtures'][fixture['name']]['latest']
        if not item or not item['diagnostics']['accepted_instance']:
            continue
        diag = item['diagnostics']
        cra = item['cra_control']['cra']
        proxy = diag.get('contact_proxy')
        rows.append(dict(
            source=record['sensor_timestamp'], frame=record['sensor_frame_id'], track=item['track_id'],
            cue=proxy['value'] if proxy else None,
            contact=cra['factors']['contact']['conservative_risk'], phi=cra['phi'],
            kinetic=cra['factors']['kinetic']['conservative_risk'],
            mass=item['attributes']['mass']['mass_kg']['value'],
            source_evidence=diag.get('property_evidence', {}),
        ))
    return rows


def live_observations(audited_rows):
    all_rows, bins = [], {}
    for row in audited_rows:
        x = row['state_at_commit'][0]
        if not math.isfinite(x):
            raise ValueError('nonfinite along-route position')
        index = math.floor(x / BIN_WIDTH_M)
        bucket = bins.setdefault(index, dict(total_fields=0, targets=[]))
        bucket['total_fields'] += 1
        targets = row['targets']
        if len(targets) > 1:
            raise ValueError('ambiguous target identity in current-point replay')
        if targets:
            target = targets[0]
            all_rows.append(target)
            bucket['targets'].append(target)
    return all_rows, bins


def compare_bins(bin_buckets, crate_buckets):
    rows = []
    for index in sorted(bin_buckets.keys() | crate_buckets.keys()):
        row = dict(start_m=index*BIN_WIDTH_M, end_m=(index+1)*BIN_WIDTH_M)
        for key, buckets in [('bin', bin_buckets), ('crate', crate_buckets)]:
            bucket = buckets.get(index, dict(total_fields=0, targets=[]))
            targets = bucket['targets']
            row[key] = dict(total=bucket['total_fields'], qualified=len(targets),
                            available_speed=sum(t['relative_speed'] is not None for t in targets),
                            scores=summarize(targets))
        rows.append(row)
    common = [r for r in rows if r['bin']['qualified'] and r['crate']['qualified']]
    differences = {}
    for name in ('cue', 'contact', 'phi'):
        deltas = []
        for row in common:
            left, right = row['bin']['scores'][name], row['crate']['scores'][name]
            if left is not None and right is not None:
                deltas.append(right['median'] - left['median'])
        differences[name] = describe(deltas)
    return dict(width_m=BIN_WIDTH_M, common_intervals=len(common),
                crate_minus_bin_within_interval_medians=differences, intervals=rows)


def analyze_run(folder, expected_name):
    def read(name):
        return json.loads((folder / name).read_text())

    monitor = read('fixture-monitor.json')
    (fixture,) = monitor['fixtures']
    if fixture['name'] != expected_name:
        raise ValueError(f'expected isolated {expected_name} fixture')
    invocation = read('invocation.json')
    if invocation['stage'] != 'motion' or not read('stationary-gate.json')['passed']:
        raise ValueError('expected a motion run with a successful in-run parked check')
    report = read('report.json')
    if report['nominal_route_policy']['configured_anchors_local_xy'] != [[4., 0.]]:
        raise ValueError('expected the isolated straight four-metre waypoint')
    prop = next(p for p in read('restaurant-placement-monitor.json')['props'] if p['name'] == expected_name)
    source = read('stationary-fields.json')
    total_parked = len(source)
    parked = parked_observations(source, fixture)
    del source
    native = read('stationary-object-validation.json')['fixtures'][expected_name]
    if len(parked) != native['accepted_visual_updates']:
        raise ValueError('fresh accepted target count differs from the native parked summary')
    replay = audit_retention(folder)
    if replay['summary']['source_point_errors']:
        raise ValueError('native/source replay failed, inspect the standalone retention audit')
    targets, bins = live_observations(replay['rows'])
    with_speed = [t for t in targets if t['relative_speed'] is not None]
    command = invocation['command']
    start = command.index('--stationary-prop-world-pose') + 1
    result = dict(
        folder=str(folder), source_revision=invocation['source_revision'],
        expected_start_world_pose=[float(x) for x in command[start:start+3]],
        settled_anchor=monitor['anchor_world_xy_yaw'], target_center=fixture['center_world_xy'],
        target_box_world=[prop['bounds_minimum'], prop['bounds_maximum']],
        navigation_passed=report['passed'],
        parked=dict(total=total_parked, qualified=len(parked), scores=summarize(parked), rows=parked,
                    fresh_contact=sum(r['source_evidence'].get('contact', {}).get('mode') == 'fresh'
                                      for r in parked)),
        live=dict(total=len(replay['rows']), qualified=len(targets), available_speed=len(with_speed),
                  scores=summarize(targets), scores_with_available_speed=summarize(with_speed),
                  scores_without_available_speed=summarize([t for t in targets if t['relative_speed'] is None]),
                  original_property_frames=sorted({t['source_frame'] for t in targets}),
                  maximum_target_box_excess_m=max((t['independent_box_excess'] for t in targets), default=None),
                  target_rows_outside_box_allowance=sum(t['independent_box_excess'] > .03 for t in targets),
                  whole_center_diagnostic_pass_rows=replay['summary']['old_whole_center_rows_pass'],
                  source_point_errors=replay['summary']['source_point_errors']),
    )
    return result, bins


def compare(bin_folder, crate_folder):
    bin_run, bin_bins = analyze_run(bin_folder.resolve(), 'ServiceContainer')
    crate_run, crate_bins = analyze_run(crate_folder.resolve(), 'BrokenCrate')
    if (bin_run['expected_start_world_pose'] != crate_run['expected_start_world_pose']
            or math.dist(bin_run['target_center'], crate_run['target_center']) > 1e-5):
        raise ValueError('the two isolated runs must share authored robot start and target XY center')
    return dict(
        bin=bin_run, crate=crate_run,
        live_matched_progress=compare_bins(bin_bins, crate_bins),
        scope='Fresh accepted camera/LiDAR target observations at the parked start; separately, '
              'replayed current LiDAR membership and original mass/contact during motion. '
              'Intervals use local along-route metres at field commit. All available scores included. '
              'Per-field values are repeated observations within a single run. Different meshes, observed '
              'surfaces, uncertainty and interactions affect both contact risk and all-factor phi.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin', required=True, type=Path)
    parser.add_argument('--crate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new comparison output path')
    result = compare(args.bin, args.crate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for prop in ('bin', 'crate'):
        print(prop, json.dumps({phase: result[prop][phase]['scores'] for phase in ('parked', 'live')}))


if __name__ == '__main__':
    main()
