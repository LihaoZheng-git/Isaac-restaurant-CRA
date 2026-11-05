"""Replay retained prop sources and current LiDAR membership in a short trial.

Fixture truth selects reporting rows and checks mesh bounds only. It never
enters the replayed memory, matching, factor estimation or native controller.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot.perception.static_prop_evidence import StaticPropEvidence, StaticPropEvidenceConfig
from robot.perception.persistent_object_evidence import PersistentObjectEvidenceConfig
from robot.perception.physical_attributes import ScalarEstimate


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def audit(folder):
    hashes = {}

    def load(name):
        path = folder / name
        hashes[str(path)] = digest(path)
        return json.loads(path.read_text())

    stationary = load('stationary-fields.json')
    live_path = folder / 'live-fields.jsonl'
    hashes[str(live_path)] = digest(live_path)
    live = [json.loads(line) for line in live_path.read_text().splitlines() if line.strip()]
    fixture = load('fixture-monitor.json')['fixtures'][0]
    target = np.asarray(fixture['center_world_xyz'])
    prop = next(p for p in load('restaurant-placement-monitor.json')['props'] if p['name'] == fixture['name'])
    lo, hi = np.asarray(prop['bounds_minimum']), np.asarray(prop['bounds_maximum'])
    settings = stationary[0]['diagnostics']['image_object_groups']['static_prop_evidence']['configuration']
    memory = StaticPropEvidence(StaticPropEvidenceConfig(**settings), PersistentObjectEvidenceConfig(scope='source replay'))
    target_ids, errors, rows = set(), [], []
    for index, record in enumerate(stationary + live):
        frozen = index >= len(stationary)
        detail = record['diagnostics']['image_object_groups']
        path = Path(detail['candidate_archive'])
        hashes[str(path)] = digest(path)
        with np.load(path) as archive:
            candidates = json.loads(str(archive['candidates_json']))
            world, robot = archive['world_xyz'], archive['robot_xyz']
            point_ids, selected_sets = archive['raw_track_ids'], archive['selected']
        times = {o['attributes']['timestamp_seconds'] for o in record['objects']}
        if len(times) != 1:
            raise ValueError('current object attributes must share the original camera timestamp')
        timestamp = times.pop()
        memory.expire(timestamp)
        if detail['static_prop_evidence']['configuration'] != settings:
            errors.append([record['sensor_frame_id'], 'configuration changed'])
        matches = memory.match(world, point_ids, timestamp)[0] if frozen else {}
        # Recover the recorded rigid sensor placement, never a fixture pose.
        design = np.column_stack((robot[:, :2], np.ones(len(robot))))
        position = np.linalg.lstsq(design, world[:, :2], rcond=None)[0][2]
        for obj in record['objects']:
            d = obj['diagnostics']
            group = d.get('image_object_association')
            if not group or not d.get('accepted_instance'):
                continue
            which = next(i for i, c in enumerate(candidates) if c['proposal_index'] == group['proposal_index'])
            enrolled = memory.enroll(obj['track_id'], world[selected_sets[which]],
                ScalarEstimate(**obj['attributes']['mass']['mass_kg']),
                ScalarEstimate(**d['contact_proxy']) if d.get('contact_proxy') else None,
                frame_id=d['property_evidence']['mass']['source_frame_id'], source_time=timestamp,
                robot_xy=position, frozen=frozen)
            if enrolled and np.linalg.norm(np.asarray(d['instance_centroid_world_xyz']) - target) <= .20:
                target_ids.add(obj['track_id'])
        if not frozen:
            continue
        row = dict(source=record['sensor_timestamp'], camera_timestamp=timestamp, targets=[])
        for obj in record['objects']:
            d, key = obj['diagnostics'], obj['track_id']
            if not d.get('static_prop_policy'):
                continue
            prefix = [record['sensor_frame_id'], key]
            if key not in matches:
                errors.append([*prefix, 'unreplayed current match'])
                continue
            selected, proof = matches[key]
            original = memory.records[key]
            if int(selected.sum()) != d['policy_return_count']:
                errors.append([*prefix, 'point count mismatch'])
            for field in ('fragments', 'distinct_original_support', 'current_support_fraction',
                          'location_source_time', 'location_source_frame'):
                if proof[field] != d['static_prop_association'][field]:
                    errors.append([*prefix, 'association', field])
            for name, value in (('mass', obj['attributes']['mass']['mass_kg']), ('contact', d['contact_proxy'])):
                source = original[name]
                evidence = d['property_evidence'][name]
                for field in ('value', 'lower_bound', 'upper_bound', 'source', 'timestamp_seconds'):
                    if value[field] != getattr(source, field):
                        errors.append([*prefix, name, field])
                age = timestamp - source.timestamp_seconds
                expected = source.confidence * 2 ** (-age / memory.property_config.confidence_half_life_seconds)
                if (abs(value['confidence'] - expected) > 1e-12 or evidence['age_seconds'] != age
                        or value['status'] != 'prior' or evidence['source_frame_id'] != original['frame_id']):
                    errors.append([*prefix, name, 'original clock/confidence/provenance'])
            motion = obj['attributes']['motion']
            if any(v['timestamp_seconds'] != record['sensor_timestamp'] for v in motion.values()):
                errors.append([*prefix, 'stale motion'])
            if d['accepted_instance']:
                errors.append([*prefix, 'false fresh visual claim'])
            if key in target_ids:
                points = world[selected]
                row['targets'].append(dict(id=key, points=len(points), mass=original['mass'].value,
                    cue=original['contact'].value, contact=obj['cra_control']['cra']['factors']['contact']['conservative_risk'],
                    phi=obj['cra_control']['cra']['phi'],
                    kinetic=obj['cra_control']['cra']['factors']['kinetic']['conservative_risk'],
                    relative_speed=motion['relative_speed_mps']['value'],
                    original_center_error=float(np.linalg.norm(points.mean(axis=0)-target)),
                    independent_box_excess=float(np.maximum(np.maximum(lo-points, points-hi), 0).max()),
                    source_frame=original['frame_id']))
        rows.append(row)
    targets = [t for row in rows for t in row['targets']]
    ranges = {}
    for name in ('points', 'mass', 'cue', 'contact', 'phi', 'kinetic', 'relative_speed',
                 'original_center_error', 'independent_box_excess'):
        values = [t[name] for t in targets if t[name] is not None]
        ranges[name] = [min(values), max(values)] if values else None
    report = load('report.json')
    summary = dict(initial_target_ids=sorted(target_ids), live_rows=len(rows),
        target_rows=sum(bool(r['targets']) for r in rows), source_point_errors=errors,
        retention_pass=bool(rows) and all(len(r['targets']) == 1 for r in rows) and not errors,
        ranges=ranges, old_whole_center_rows_pass=sum(t['original_center_error'] <= .20 for t in targets),
        scope='Original intrinsic values/clocks with current LiDAR/motion; static-location association prior. Whole-center diagnostic reported separately.')
    summary['navigation'] = {k: report.get(k) for k in ('passed', 'minimum_clearance_m', 'minimum_cbf_margin',
        'stop_drift_m', 'live_field_count', 'fields_committed_while_translating', 'waypoints_reached', 'last_state')}
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise RuntimeError('source changed during audit')
    return dict(summary=summary, rows=rows, source_hashes=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh output path; preserve failed audits')
    result = audit(args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))
    return 0 if result['summary']['retention_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
