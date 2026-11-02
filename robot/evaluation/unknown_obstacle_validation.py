"""Provenance/field and independent human-exclusion audits, not motion acceptance."""

import math

from robot.evaluation.furniture_memory_validation import audit_snapshot


def audit_static_human_exclusion(snapshot, truth):
    if snapshot is None:
        return None
    from types import SimpleNamespace
    # Human truth scores retained static surfaces only and never supplies any
    # runtime grouping, classification, geometry, policy or control parameter.
    static = [r.original for r in snapshot.regions if r.mode in (
        'stationary_positive_support_position_prior', 'retained_stationary_geometry_prior',
        'supported_background_position_prior', 'retained_supported_background_prior')]
    return audit_snapshot(SimpleNamespace(source_time=snapshot.source_time,
        surfaces=static, observed_surfaces=()), truth)


def summarize_unknown_obstacles(records):
    count = regions = replacements = 0
    errors = []
    modes = {}
    for record in records:
        d = record.get('diagnostics', record)
        memory = d.get('unknown_obstacle_memory')
        checks = d.get('unknown_obstacle_field_checks')
        if memory is None or checks is None:
            errors.append('missing mode provenance')
            continue
        count += 1
        now = memory['source_time']
        if len(memory['regions']) != len(checks):
            errors.append('region/field-check count mismatch')
        for region, proof in zip(memory['regions'], checks):
            regions += 1
            mode = region['mode']
            modes[mode] = modes.get(mode, 0)+1
            replacements += region['replacement_track_id'] is not None
            if (not proof or not proof.get('local_region_current_field_covered')
                    or proof['track_id'] != region['track_id']
                    or proof['original_source_time'] != region['original_source_time']
                    or proof['current_source_time'] != now
                    or not 0 <= region['original_source_time'] <= now
                    or not math.isfinite(region['radius_m']) or region['radius_m'] < 0
                    or region['applied_risk'] < region['fixed_unknown_phi']
                    or region.get('motion_support_time') is not None and not region['original_source_time'] <= region['motion_support_time'] <= now
                    or region['stationary_support_time'] is not None and region['stationary_support_time'] > now):
                errors.append('source/risk/geometry field provenance mismatch')
            support = region.get('background_support')
            if region['mode'] in ('supported_background_position_prior', 'retained_supported_background_prior'):
                cfg = memory['config']
                if (not cfg.get('supported_background_prior') or not support
                        or support['original_point_count'] != region['original_point_count']
                        or support['supported_fraction'] < cfg['background_support_fraction']
                        or support['supported_point_count']+support['unsupported_point_count'] != region['original_point_count']
                        or support['maximum_gap_m'] > cfg['background_maximum_gap_m']
                        or support['distinct_support_points'] < 3
                        or len(set(support['support_source_times'])) < 3
                        or max(support['support_source_times']) != region['stationary_support_time']
                        or max(support['support_source_times'])-min(support['support_source_times']) < .2-1e-9
                        or not support['whole_original_geometry_retained']
                        or region['radius_m'] < cfg['background_position_radius_m']):
                    errors.append('stationary-background support provenance mismatch')
        human = d.get('unknown_obstacle_static_human_audit')
        if human is not None and not human.get('passed'):
            errors.append('static memory overlaps independent human reference')
    return dict(passed=bool(count and not errors), sources=count, regions=regions,
        geometry_replacements=replacements, modes=modes, errors=errors[:20],
        scope='source/policy/current-grid provenance and available static-human exclusion only; not general uncertainty calibration or navigation acceptance')
