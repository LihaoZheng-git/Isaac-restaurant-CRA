"""Source/provenance audit, not validation of physical adult kilograms."""

import math

from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE
from robot.perception.human_mass_identity_retention import HUMAN_MASS_RETENTION_MARKER
from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE


def validate_human_mass_identity_retention(records):
    originals, used, rows = {}, set(), []
    checks = dict(original_fresh_visual_prior=True, current_positive_body_identity=True,
        original_source_expiry=True, decayed_confidence_widened_bounds=True)
    for record in records:
        diagnostic = record.get('diagnostics', record)
        audit = diagnostic.get('human_mass_body_identity_retention')
        if audit is None:
            continue
        now, frame = audit['camera_timestamp_seconds'], audit['camera_frame_id']
        sensor = record.get('sensor_timestamp', record.get('time'))
        objects = {o['attributes']['track_id']: o for o in record['objects']}
        bodies = {b['track_id']: b for b in diagnostic['human_body_envelope_priors']}
        config = audit['configuration']
        for row in audit['rows']:
            if row['mode'] == 'unavailable':
                continue
            key = row['track_id']
            obj, body = objects[key], bodies.get(key)
            mass = obj['attributes']['mass']['mass_kg']
            source_key = (key, row['source_timestamp'], row['source_frame_id'])
            checks['current_positive_body_identity'] &= (body is not None
                and body['source'] == HUMAN_ENVELOPE_SOURCE and body['status'] == 'prior'
                and body['lidar_timestamp_seconds'] == sensor and body['current_support_count'] >= 20
                and 0 <= now-sensor <= .25+1e-9)
            if row['mode'] == 'fresh':
                cue = obj['diagnostics']['geometry_evidence'].get('human_mass_prior')
                valid = (obj['diagnostics']['accepted_instance'] and cue == mass
                    and body is not None and body['original_person_camera_timestamp'] == now
                    and body['original_person_camera_frame_id'] == frame
                    and mass['source'] == HUMAN_MASS_PRIOR_SOURCE and mass['status'] == 'prior'
                    and mass['timestamp_seconds'] == now == row['source_timestamp']
                    and row['source_frame_id'] == frame and HUMAN_MASS_RETENTION_MARKER not in mass['evidence'])
                checks['original_fresh_visual_prior'] &= valid
                if valid:
                    originals[source_key] = mass
                continue
            source = originals.get(source_key)
            age = now-row['source_timestamp']
            valid_source = (source is not None and mass['source'] == HUMAN_MASS_PRIOR_SOURCE
                and body is not None and body['original_person_camera_timestamp'] >= row['source_timestamp']
                and body['original_person_camera_frame_id'] >= row['source_frame_id']
                and mass['status'] == 'prior' and mass['unit'] == 'kg'
                and HUMAN_MASS_RETENTION_MARKER in mass['evidence']
                and mass['timestamp_seconds'] == row['source_timestamp']
                and 0 <= age <= config['maximum_age_seconds']+1e-9 <= 3.+2e-9
                and abs(age-row['age_seconds']) <= 1e-9
                and row['current_lidar_timestamp'] == sensor)
            checks['original_source_expiry'] &= valid_source
            if valid_source:
                growth = config['bound_growth_fraction_per_second']*age
                checks['decayed_confidence_widened_bounds'] &= (
                    mass['value'] == source['value']
                    and math.isclose(mass['lower_bound'], max(0., source['lower_bound']*(1-growth)), abs_tol=1e-9)
                    and math.isclose(mass['upper_bound'], source['upper_bound']*(1+growth), abs_tol=1e-9)
                    and math.isclose(mass['confidence'], source['confidence']*2**(-age/config['confidence_half_life_seconds']), abs_tol=1e-9))
            used.add(sensor)
            rows.append(dict(track_id=key, camera_timestamp=now, current_lidar_timestamp=sensor,
                original_mass_timestamp=row['source_timestamp'], original_mass_frame=row['source_frame_id']))
    checks['five_distinct_retained_sources'] = len(used) >= 5
    checks = {key: bool(value) for key, value in checks.items()}
    return dict(passed=all(checks.values()), checks=checks, retained_source_count=len(used),
        original_prior_count=len(originals), rows=rows,
        scope='adult-mass PRIOR retention provenance only; original expiry/current body identity; not measured mass or new visual evidence')
