"""Join existing validated human associations to current positive geometry.

Raw person masks and expanded uncertainty regions are deliberately not inputs.
This preserves original visual epochs; it creates no new class/factor estimate.
"""


def verified_human_associations(tracking, bodies, visual_parts):
    now = tracking.timestamp_seconds
    tracks = {t.track_id: t for t in tracking.tracked_obstacles}
    parents = {}
    for body in bodies:
        key = body['track_id']
        if (key in parents or key not in tracks
                or body['lidar_timestamp_seconds'] != now
                or body['current_support_count'] != tracks[key].point_count
                or body['current_support_count'] < 3
                or not 0 <= body['identity_age_seconds'] <= 30.
                or not 0 <= body['original_person_camera_timestamp'] <= now+.25):
            raise ValueError('current validated body association required')
        parents[key] = dict(track_id=key, parent_track_id=key, source_time=now,
            current_support_count=tracks[key].point_count, basis='validated_current_body',
            original_person_camera_timestamp=body['original_person_camera_timestamp'],
            original_person_camera_frame_id=body['original_person_camera_frame_id'],
            identity_basis=body['identity_basis'])
    result = list(parents.values())
    for part in visual_parts:
        key, parent = part['child_track_id'], part['parent_track_id']
        if (key not in tracks or parent not in parents or key in {p['track_id'] for p in result}
                or part['source_timestamp_seconds'] != now
                or part['child_positive_count'] != tracks[key].point_count
                or part['parent_positive_count'] != tracks[parent].point_count
                or part['original_person_camera_timestamp'] != parents[parent]['original_person_camera_timestamp']
                or part['original_person_camera_frame_id'] != parents[parent]['original_person_camera_frame_id']):
            raise ValueError('unique current visual-part association with validated parent required')
        result.append(dict(parents[parent], track_id=key, basis='validated_current_visual_part',
            current_support_count=tracks[key].point_count, visual_part_proof=dict(part)))
    return tuple(result)
