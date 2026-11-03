"""Provenance/field and independent human-exclusion audits, not motion acceptance."""

import math
import hashlib
import json
from pathlib import Path

import numpy as np

from robot.evaluation.furniture_memory_validation import audit_snapshot


def audit_static_human_exclusion(snapshot, truth):
    if snapshot is None:
        return None
    from types import SimpleNamespace
    # Human truth scores retained static surfaces only and never supplies any
    # runtime grouping, classification, geometry, policy or control parameter.
    static = [r.original for r in snapshot.regions if r.mode in (
        'stationary_positive_support_position_prior', 'retained_stationary_geometry_prior',
        'supported_background_position_prior', 'retained_supported_background_prior',
        'persistent_stationary_background_prior', 'unclassified_static_geometry_prior')]
    return audit_snapshot(SimpleNamespace(source_time=snapshot.source_time,
        surfaces=static, observed_surfaces=()), truth)


def summarize_unknown_obstacles(records):
    count = regions = replacements = 0
    point_motion_proofs = 0
    confirmed_motion_proofs = ambiguous_motion_proofs = 0
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
        for evidence in memory.get('observed_dynamic_evidence', ()):
            if (not memory['config'].get('unclassified_static_prior')
                    or evidence.get('source_time') != now or not _valid_dynamic_evidence(evidence, now,
                        verified_only=memory['config'].get('require_verified_dynamic_evidence', False))):
                errors.append('current dynamic evidence mismatch')
        for proof in memory.get('background_motion_proofs', ()):
            point_motion_proofs += 1
            distance = proof['minimum_unconfirmed_distance_m']
            times = proof['support_source_times']
            if (not memory['config'].get('persistent_background_prior')
                    or proof['current_source_time'] != now
                    or not 0 <= proof['original_source_time'] <= now
                    or proof['current_point_count'] <= 0
                    or proof['repeated_stationary_point_count'] < 0
                    or proof['unconfirmed_motion_point_count'] < 0
                    or proof['repeated_stationary_point_count']+proof['unconfirmed_motion_point_count'] != proof['current_point_count']
                    or (distance is None) != (proof['unconfirmed_motion_point_count'] == 0)
                    or distance is not None and (not math.isfinite(distance) or distance < 0)
                    or proof['stationary_overlap_only'] != (distance is None or distance > .06)
                    or proof['stationary_overlap_only'] and (
                        proof['repeated_stationary_point_count'] == 0
                        or len(set(times)) < 3 or max(times) != now
                        or max(times)-min(times) < .2-1e-9)):
                errors.append('point-motion contradiction proof mismatch')
            if memory.get('background_motion_policy') == 'repeated_geometry':
                confirmation = proof.get('motion_confirmation')
                current_distance = proof.get('minimum_current_distance_m', math.inf)
                if (not _valid_motion_confirmation(confirmation, now)
                        or not 0 <= current_distance <= .06):
                    errors.append('geometric-motion confirmation mismatch')
                elif proof.get('motion_contradiction') != confirmation['confirmed']:
                    errors.append('geometric-motion decision mismatch')
                elif confirmation['confirmed']:
                    confirmed_motion_proofs += 1
                else:
                    ambiguous_motion_proofs += 1
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
            evidence = region.get('dynamic_evidence')
            if mode == 'current_verified_part_geometry_replacement':
                recovery = region.get('verified_recovery')
                if not _valid_verified_part_recovery(recovery, region, d, now):
                    errors.append('verified part recovery provenance mismatch')
            if evidence is not None and not _valid_dynamic_evidence(evidence, now,
                    verified_only=memory['config'].get('require_verified_dynamic_evidence', False)):
                errors.append('retained dynamic evidence mismatch')
            if mode == 'unclassified_static_geometry_prior' and (
                    not memory['config'].get('unclassified_static_prior')
                    or region['radius_m'] != .30 or evidence is not None or support is not None):
                errors.append('unclassified static prior provenance mismatch')
            if region['mode'] in ('supported_background_position_prior', 'retained_supported_background_prior',
                    'persistent_stationary_background_prior'):
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
                if cfg.get('persistent_background_prior'):
                    if (region['radius_m'] > cfg['background_maximum_radius_m']
                            or cfg['background_maximum_radius_m'] > .30):
                        errors.append('persistent background position cap exceeded')
                if mode == 'persistent_stationary_background_prior' and (
                        not cfg.get('persistent_background_prior')
                        or region['stationary_support_time'] is None
                        or now-region['stationary_support_time'] <= cfg['stationary_lease_seconds']):
                    errors.append('persistent background opt-in/age mismatch')
        human = d.get('unknown_obstacle_static_human_audit')
        if human is not None and not human.get('passed'):
            errors.append('static memory overlaps independent human reference')
    return dict(passed=bool(count and not errors), sources=count, regions=regions,
        geometry_replacements=replacements, point_motion_proofs=point_motion_proofs,
        confirmed_motion_proofs=confirmed_motion_proofs, ambiguous_motion_proofs=ambiguous_motion_proofs,
        modes=modes, errors=errors[:20],
        scope='source/policy/current-grid provenance and available static-human exclusion only; not general uncertainty calibration or navigation acceptance')


def _valid_dynamic_evidence(evidence, now, *, verified_only=False):
    if (not isinstance(evidence, dict) or not isinstance(evidence.get('source_time'), (int, float))
            or not 0 <= evidence['source_time'] <= now):
        return False
    replacement = evidence.get('geometry_replacement')
    if replacement is not None and (
            not evidence['source_time'] <= replacement.get('source_time', -1) <= now
            or not 0 <= replacement.get('maximum_distance_m', math.inf) <= .10
            or replacement.get('previous_track_id', -1) < 0
            or replacement.get('current_track_id', -1) < 0
            or replacement['previous_track_id'] == replacement['current_track_id']):
        return False
    if evidence.get('kind') == 'confirmed_geometric_motion':
        return (_valid_motion_confirmation(evidence.get('motion_confirmation'), evidence['source_time'])
            and evidence['motion_confirmation']['confirmed'])
    if evidence.get('kind') == 'verified_human_association':
        proof = evidence.get('human_association') or {}
        return (proof.get('source_time') == evidence['source_time']
            and proof.get('track_id') == evidence.get('track_id')
            and proof.get('current_support_count', 0) >= 3
            and proof.get('basis') in ('validated_current_body', 'validated_current_visual_part')
            and 0 <= proof.get('original_person_camera_timestamp', -1) <= evidence['source_time']+.25
            and proof.get('original_person_camera_frame_id', -1) >= 0
            and proof.get('parent_track_id', -1) >= 0)
    return not verified_only and evidence.get('kind') in ('positive_human_region', 'positive_person_or_confirmed_motion_overlap')


def _valid_verified_part_recovery(proof, region, diagnostics, now):
    if not isinstance(proof, dict):
        return False
    cfg = diagnostics.get('human_track_recovery_config') or {}
    geometry, part = proof.get('geometry_coverage', {}), proof.get('visual_part_proof', {})
    bodies = [b for b in diagnostics.get('human_body_envelope_priors', ())
        if b['track_id'] == proof.get('track_id')]
    if (not cfg.get('verified_part_recovery') or len(bodies) != 1
            or proof not in diagnostics.get('human_track_recovery', ())
            or proof.get('decision') != 'reacquired_verified_visual_part_geometry'
            or proof.get('previous_track_id') != region['track_id']
            or proof.get('track_id') != region['replacement_track_id']
            or geometry.get('original_source_time') != region['original_source_time']
            or geometry.get('original_source_frame') != region['original_source_frame']
            or geometry.get('original_point_count') != region['original_point_count']
            or geometry.get('current_source_time') != now
            or geometry.get('original_risk') != region['applied_risk']
            or not 0 < geometry.get('missing_seconds', -1) <= .35+1e-9
            or abs(geometry['missing_seconds']-(now-region['original_source_time'])) > 1e-9
            or proof.get('source_gap_seconds') != geometry['missing_seconds']
            or geometry.get('maximum_point_speed_prior_mps') != cfg.get('maximum_human_speed_mps')
            or region['radius_m'] != geometry['missing_seconds']*geometry['maximum_point_speed_prior_mps']
            or not 0 <= proof.get('centroid_prediction_residual_m', math.inf) <= .25
            or {part.get('child_track_id'), part.get('parent_track_id')}
                != {proof['previous_track_id'], proof['track_id']}
            or not 0 < now-part.get('source_timestamp_seconds', -math.inf) <= cfg['maximum_gap_seconds']+1e-9):
        return False
    body = bodies[0]
    low, high = geometry.get('original_bounds_min_world_xy'), geometry.get('original_bounds_max_world_xy')
    if (np.shape(low) != (2,) or np.shape(high) != (2,) or not np.isfinite([low, high]).all()
            or np.any(np.asarray(low) > high) or body['lidar_timestamp_seconds'] != now
            or body['center_world_xy'] != geometry.get('current_center_world_xy')
            or body['radius_m'] != geometry.get('current_radius_m')
            or body['applied_cra_phi'] < geometry['original_risk']
            or body['original_person_camera_timestamp'] != proof.get('fresh_person_camera_timestamp')
            or body['original_person_camera_frame_id'] != proof.get('fresh_person_camera_frame_id')):
        return False
    corners = np.array([[low[0], low[1]], [low[0], high[1]], [high[0], low[1]], [high[0], high[1]]])
    extent = float(np.linalg.norm(corners-body['center_world_xy'], axis=1).max()+region['radius_m'])
    return (abs(extent-geometry.get('whole_bounds_plus_motion_extent_m', math.inf)) <= 1e-12
        and extent <= body['radius_m']+1e-9)


def _valid_motion_confirmation(proof, now):
    if not isinstance(proof, dict) or not isinstance(proof.get('confirmed'), bool):
        return False
    times = proof.get('source_times', [])
    sampled = proof.get('sampled_source_times', [])
    if times and (not np.isfinite(times).all() or min(times) < 0 or times[-1] != now
            or any(g <= 0 or g > .35+1e-9 for g in np.diff(times))
            or times[-1]-times[0] > .7+1e-9):
        return False
    if sampled and (len(sampled) != 3 or sampled != sorted(set(sampled))
            or any(t not in times for t in sampled) or sampled[0] != times[0] or sampled[-1] != now):
        return False
    if not proof['confirmed']:
        return proof.get('reason') in ('insufficient_continuous_geometry',
            'ambiguous_or_inconsistent_shape', 'unresolved_or_inconsistent_displacement')
    fits = proof.get('fits', [])
    if (len(times) < 3 or len(sampled) != 3 or sampled[-1]-sampled[0] < .2-1e-9
            or len(fits) != 3 or proof.get('reason') != 'consistent_observed_translation'):
        return False
    for fit in fits:
        if (np.shape(fit.get('translation_xy')) != (2,) or not np.isfinite(fit['translation_xy']).all()
                or not 0 <= fit.get('maximum_residual_m', math.inf) <= .03
                or not 0 <= fit.get('aligned_rms_m', math.inf) <= fit['maximum_residual_m']+1e-12
                or not math.isfinite(fit.get('stationary_rms_m', math.inf))
                or fit['stationary_rms_m']-fit['aligned_rms_m'] < .03):
            return False
    shifts = np.array([f['translation_xy'] for f in fits])
    velocities = shifts[:2]/np.diff(sampled)[:, None]
    speeds = np.linalg.norm(velocities, axis=1)
    return bool(np.linalg.norm(shifts[2]) > .12 and min(speeds) > .2
        and max(speeds)/min(speeds) <= 1.5
        and np.dot(*velocities)/(speeds.prod()) >= .9
        and np.linalg.norm(shifts[0]+shifts[1]-shifts[2]) <= .03)


class UnknownObstacleGeometryArchive:
    """Save every retained original surface, including unwatched fragments.

    Content-addressed original geometry is written once. Per-field diagnostics
    carry its evolving support, radius and applied policy separately. This is
    evaluation output only and never an input to perception or control.
    """

    def __init__(self, output):
        self.output = Path(output)
        self.saved = {}

    def save(self, snapshot):
        if snapshot is None:
            return None
        references = []
        for region in snapshot.regions:
            old = region.original
            metadata = dict(track_id=int(old.track_id), original_source_time=old.source_time,
                original_source_frame=int(old.source_frame), center_world_xy=list(old.center))
            encoded = json.dumps(metadata, sort_keys=True, allow_nan=False)
            points = np.ascontiguousarray(old.points, dtype='<f8')
            digest = hashlib.sha256(encoded.encode()+points.tobytes()).hexdigest()
            relative = f'unknown-geometry/{old.track_id}-{old.source_frame}-{digest}.npz'
            if digest not in self.saved:
                path = self.output/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    with np.load(path, allow_pickle=False) as stored:
                        if (str(stored['metadata_json']) != encoded
                                or not np.array_equal(stored['original_points_world_xy'], points)):
                            raise ValueError('original unknown geometry archive mismatch')
                else:
                    np.savez_compressed(path, metadata_json=np.array(encoded), original_points_world_xy=points)
                self.saved[digest] = hashlib.sha256(path.read_bytes()).hexdigest()
            references.append(dict(**metadata, original_point_count=len(points), path=relative,
                geometry_sha256=digest, file_sha256=self.saved[digest]))
        motion_references = []
        for key, time, cloud in snapshot.background_motion_geometry:
            metadata = dict(track_id=int(key), source_time=float(time))
            encoded = json.dumps(metadata, sort_keys=True, allow_nan=False)
            points = np.ascontiguousarray(cloud, dtype='<f8')
            digest = hashlib.sha256(encoded.encode()+points.tobytes()).hexdigest()
            relative = f'unknown-motion-geometry/{key}-{digest}.npz'
            if digest not in self.saved:
                path = self.output/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    with np.load(path, allow_pickle=False) as stored:
                        if (str(stored['metadata_json']) != encoded
                                or not np.array_equal(stored['observed_points_world_xy'], points)):
                            raise ValueError('observed motion geometry archive mismatch')
                else:
                    np.savez_compressed(path, metadata_json=np.array(encoded), observed_points_world_xy=points)
                self.saved[digest] = hashlib.sha256(path.read_bytes()).hexdigest()
            motion_references.append(dict(**metadata, point_count=len(points), path=relative,
                geometry_sha256=digest, file_sha256=self.saved[digest]))
        return dict(source_time=snapshot.source_time, regions=references, motion_evidence=motion_references,
            scope='exact original retained geometry; evaluation only, not renewed evidence')
