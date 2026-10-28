"""Audit the separate preparation policy; no actor truth or control authority."""

import math

import numpy as np


def validate_parallel_preparation(records):
    originals, used_sources, rows = {}, set(), []
    checks = dict(current_fit_and_source=True, unique_positive_candidate=True,
        original_selection_and_visual_proof=True, current_away_side=True, field_reference_provenance=True)
    for record in records:
        diagnostic = record.get('diagnostics', record)
        a = diagnostic.get('parallel_preparation')
        if a is None:
            continue
        now = record.get('sensor_timestamp', record.get('time'))
        source_frame = record.get('sensor_frame_id')
        preferences = a['preferences']
        if not preferences:
            continue
        selection = a['original_selection']
        candidates = {r['track_id']: r for r in a['qualified_candidates']}
        key = selection['track_id']
        start = selection['selected_timestamp_seconds']
        cfg = a['configuration']
        checks['current_fit_and_source'] &= (a['current_fit_ready'] and a['source_timestamp_seconds'] == now
            and a['source_frame_id'] == source_frame and a['status'] == 'prior')
        checks['unique_positive_candidate'] &= (len(candidates) == len(preferences) == 1 and key in candidates
            and candidates[key]['current_support_count'] >= 20
            and candidates[key]['current_fit_source_times'][-1] == now)
        selection_key = (key, start)
        if start == now:
            proof = a['current_visual_selection_evidence'].get(key, a['current_visual_selection_evidence'].get(str(key)))
            checks['original_selection_and_visual_proof'] &= (proof == selection['original_visual_evidence'] and proof is not None
                and proof.get('camera_frame_id') == a['camera_frame_id']
                and proof.get('camera_timestamp_seconds') == a['camera_timestamp_seconds']
                and 0 <= a['camera_timestamp_seconds']-now <= .25+1e-9)
            originals[selection_key] = selection.copy()
        checks['original_selection_and_visual_proof'] &= (selection_key in originals
            and originals.get(selection_key) == selection and start <= now <= start+cfg['maximum_selection_seconds']+1e-9)
        if key in candidates:
            p = np.asarray(candidates[key]['current_center_world_xy'])-a['current_robot_world_xy']
            d = np.asarray(a['nominal_route_direction_world_xy'])
            lateral = float(d[0]*p[1]-d[1]*p[0])
            speed = np.linalg.norm(candidates[key]['fitted_velocity_world_xy'])
            alignment = -float(d@candidates[key]['fitted_velocity_world_xy'])/max(speed, 1e-9)
            checks['current_away_side'] &= (abs(np.linalg.norm(d)-1) < 1e-9 and p@d > 0
                and cfg['minimum_lateral_separation_m'] <= abs(lateral) <= cfg['maximum_lateral_separation_m']
                and np.linalg.norm(p) <= cfg['maximum_distance_m']
                and selection['side'] == ('left' if lateral < 0 else 'right')
                and preferences.get(key, preferences.get(str(key))) == selection['side'] == candidates[key]['side']
                and (speed <= cfg['quiet_speed_mps'] or alignment >= cfg['minimum_oncoming_alignment']))
        if a['used_for_reference']:
            used_sources.add(now)
            ids = diagnostic.get('motion_passing_reference_track_ids')
            original_time = diagnostic.get('motion_passing_reference_source_timestamp_seconds')
            checks['field_reference_provenance'] &= ids is not None and list(ids) == [key] and original_time == start
            policies = [r for r in diagnostic.get('applied_field_policies', ()) if r['track_or_group_id'] == key]
            checks['field_reference_provenance'] &= bool(policies) and all(
                r['applied_passing_side'] == selection['side'] and r['lambda'] == 1.25 for r in policies)
        rows.append(dict(source_timestamp=now, track_id=key, original_selected_time=start,
            used_for_reference=a['used_for_reference']))
    checks['three_distinct_used_preparation_sources'] = len(used_sources) >= 3
    checks['finite_original_epochs'] = all(math.isfinite(r['source_timestamp']) for r in rows)
    return dict(passed=all(checks.values()), checks=checks, used_source_count=len(used_sources),
        original_selection_count=len(originals), rows=rows,
        scope='separate current-person preparation policy audit; static geometry is not future walking intent; original actor/factor/geometry/arrival/stop gates remain')
