"""Bounded recovery across an already verified human parent/part ID swap.

No actor truth or new matching margin: a fresh accepted class cue strengthens
an existing positive part relationship. Complete old geometry must fit the
current body prior with the original all-direction missing-time allowance.
"""

import numpy as np

from dataclasses import replace

from robot.safety.class_safety_policy import ClassSafetyParameters


def surface_coverage(surface, *, center, radius, now, speed=1.):
    age = now-surface.source_time
    if not 0 < age <= .35+1e-9:
        return None
    low, high = surface.points.min(axis=0), surface.points.max(axis=0)
    corners = np.array([[low[0], low[1]], [low[0], high[1]],
                        [high[0], low[1]], [high[0], high[1]]])
    extent = float(np.linalg.norm(corners-center, axis=1).max()+speed*age)
    if extent > radius+1e-9:
        return None
    return dict(original_source_time=surface.source_time,
        original_source_frame=surface.source_frame, original_point_count=len(surface.points),
        original_bounds_min_world_xy=low.tolist(), original_bounds_max_world_xy=high.tolist(),
        current_source_time=now, current_center_world_xy=list(center), current_radius_m=radius,
        maximum_point_speed_prior_mps=speed, missing_seconds=age,
        whole_bounds_plus_motion_extent_m=extent, original_risk=surface.guidance.risk,
        original_boundary_flux_magnitude=surface.guidance.boundary_flux_magnitude)


def recovery_proposals(memory, tracking, fresh, surfaces, camera_time):
    """Propose unique NEW targets only; existing identities never merge."""
    cfg, now = memory.recovery_config, tracking.timestamp_seconds
    tracks = {t.track_id: t for t in tracking.tracked_obstacles}
    proposals = {}
    for old, previous in memory.identities.items():
        dt = now-previous['lidar_timestamp']
        surface = surfaces.get(old)
        if (old in tracks or not 0 < dt <= cfg.maximum_gap_seconds+1e-9
                or surface is None or surface.source_time != previous['lidar_timestamp']
                or not 0 <= camera_time-previous['camera_timestamp'] <= memory.config.maximum_identity_age_seconds
                or np.linalg.norm(previous['velocity']) > cfg.maximum_human_speed_mps):
            continue
        matches = []
        for new, cue in fresh.items():
            track = tracks.get(new)
            links = [p for p in memory.visual_part_links
                if {p['child_track_id'], p['parent_track_id']} == {old, new}
                and 0 < now-p['source_timestamp_seconds'] <= cfg.maximum_gap_seconds+1e-9]
            if (new in memory.identities or track is None or not links
                    or cue['camera_timestamp_seconds'] != camera_time
                    or track.point_count < memory.config.minimum_points
                    or np.linalg.norm(track.velocity_world_xy) > cfg.maximum_human_speed_mps):
                continue
            predicted = previous['center']+dt*previous['velocity']
            residual = float(np.linalg.norm(np.asarray(track.centroid_world_xy)-predicted))
            if residual > memory.config.maximum_centroid_residual_m:
                continue
            if any(other != old and other not in tracks
                    and 0 < now-state['lidar_timestamp'] <= cfg.maximum_gap_seconds+1e-9
                    and np.linalg.norm(np.asarray(track.centroid_world_xy)-state['center']
                        -(now-state['lidar_timestamp'])*state['velocity']) <= memory.config.maximum_centroid_residual_m
                    for other, state in memory.identities.items()):
                continue
            coverage = surface_coverage(surface, center=np.asarray(track.centroid_world_xy),
                radius=memory.config.body_radius_m+memory.config.centroid_uncertainty_m,
                now=now, speed=cfg.maximum_human_speed_mps)
            if coverage is None:
                continue
            matches.append(dict(previous_track_id=old, track_id=new,
                original_person_camera_timestamp=previous['camera_timestamp'],
                original_person_camera_frame_id=previous['camera_frame'],
                fresh_person_camera_timestamp=cue['camera_timestamp_seconds'],
                fresh_person_camera_frame_id=cue['camera_frame_id'],
                source_gap_seconds=dt, centroid_prediction_residual_m=residual,
                visual_part_proof=max(links, key=lambda p: p['source_timestamp_seconds']),
                geometry_coverage=coverage))
        if len(matches) == 1:
            proposals[old] = matches[0]
    # A new target claimed by multiple old people remains ambiguous.
    return {old:p for old, p in proposals.items()
        if sum(q['track_id'] == p['track_id'] for q in proposals.values()) == 1}


def apply_recovery_risk_floors(guidance, audit, recoveries, bodies):
    """Preserve previous applied policy separately from current CRA evidence."""
    body_ids = {b['track_id']:b['envelope_track_id'] for b in bodies}
    for proof in recoveries:
        key = proof['track_id']
        if key not in body_ids:
            raise ValueError('recovered identity requires its current body envelope')
        p = proof['geometry_coverage']
        for target in (key, body_ids[key]):
            old = guidance[target]
            guidance[target] = replace(old, safety_parameters=ClassSafetyParameters(
                max(old.risk, p['original_risk']),
                max(old.boundary_flux_magnitude, p['original_boundary_flux_magnitude'])))
            for row in audit:
                if row['track_or_group_id'] == target:
                    row['verified_recovery_risk_floor_prior'] = p['original_risk']
                    row['applied_field_risk'] = guidance[target].risk
                    row['boundary_flux'] = guidance[target].boundary_flux
        for body in bodies:
            if body['track_id'] == key:
                body['verified_recovery_risk_floor_prior'] = p['original_risk']
                body['applied_cra_phi'] = guidance[body_ids[key]].risk
