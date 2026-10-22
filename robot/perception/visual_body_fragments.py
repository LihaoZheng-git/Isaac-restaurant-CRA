"""Current positive body-part association for the separated bare-human fixture.

This does not change clustering, occupancy, factors or human identity clocks.
A fresh RGB person mask and current measured 3D points can associate a small
fragment with a currently validated parent, before its cluster disappears.
"""

import math

import numpy as np
from scipy.spatial import cKDTree


def associate_visual_body_fragments(
    tracking, points_xyz, projection, person_frame, parent_records, *,
    camera_timestamp_seconds, camera_frame_id,
    mask_boundary_tolerance_pixels=0,
):
    """Return bounded inferred associations, never hidden-body/free-space evidence.

    Scope: separated adult, no carried objects. Every child return must project
    inside the same current mask, be within .30 m in 3D of actual parent returns,
    and belong to a small (>=3 returns, <=10% parent) cluster. The parent must
    already have current validated LiDAR identity support and >=90% of its
    projected returns inside one untruncated person mask. No SAM/class/factor
    acceptance is created by this association.

    An explicit one-native-pixel boundary PRIOR may retain >=95% original-mask
    support when every remaining projected return is an axial neighbor of that
    same mask. Expanded competing masks still reject. This does not change the
    strict default or infer any unobserved physical surface.
    """
    now = tracking.timestamp_seconds
    if (isinstance(mask_boundary_tolerance_pixels, bool)
        or not isinstance(mask_boundary_tolerance_pixels, int)
        or mask_boundary_tolerance_pixels not in (0, 1)):
        raise ValueError("visual part mask boundary prior must be zero or one native pixel")
    points = np.asarray(points_xyz)
    ids = tracking.point_track_ids
    if (points.shape != (len(ids), 3) or not np.isfinite(points).all()
        or not math.isfinite(camera_timestamp_seconds)
        or not 0 <= camera_timestamp_seconds - now <= .15 + 1e-9
        or person_frame.source_frame_id != camera_frame_id
        or person_frame.source_timestamp_seconds != camera_timestamp_seconds):
        raise ValueError("visual part association requires aligned current RGB/LiDAR")
    tracks = {t.track_id: t for t in tracking.tracked_obstacles}
    indices, pixels = projection.source_indices, projection.pixel_indices_xy
    projected_ids = ids[indices]
    people = [d for d in person_frame.detections
              if d.semantic_class == "person" and d.confidence >= .5]
    mask_hits = [d.mask[pixels[:, 1], pixels[:, 0]] for d in people]
    expanded_hits = mask_hits
    if mask_boundary_tolerance_pixels:
        expanded_hits = []
        for person in people:
            mask = person.mask
            expanded = mask.copy()
            # One Euclidean pixel: axial neighbors only, no diagonal growth.
            expanded[1:] |= mask[:-1]
            expanded[:-1] |= mask[1:]
            expanded[:, 1:] |= mask[:, :-1]
            expanded[:, :-1] |= mask[:, 1:]
            expanded_hits.append(expanded[pixels[:, 1], pixels[:, 0]])
    proofs = []
    for parent in parent_records:
        key = parent["track_id"]
        if (key not in tracks or parent["lidar_timestamp_seconds"] != now
            or parent["current_support_count"] != tracks[key].point_count):
            continue
        owned = projected_ids == key
        candidates = [i for i, hit in enumerate(mask_hits)
                      if np.count_nonzero(owned) >= 8 and np.mean(hit[owned]) >= .9]
        if len(candidates) != 1:
            continue
        index = candidates[0]
        person, hit = people[index], mask_hits[index]
        mask = person.mask
        if (person.mask_pixel_count < 200 or mask[0].any() or mask[-1].any()
            or mask[:, 0].any() or mask[:, -1].any()):
            continue
        tree = cKDTree(points[ids == key])
        for child, track in tracks.items():
            if child == key or not 3 <= track.point_count <= .1 * tracks[key].point_count:
                continue
            child_hits = projected_ids == child
            if (np.count_nonzero(child_hits) != track.point_count
                or any(np.any(other[child_hits]) for i, other in enumerate(expanded_hits) if i != index)):
                continue
            boundary_count = int(np.count_nonzero(~hit[child_hits]))
            coverage = float(np.mean(hit[child_hits]))
            if boundary_count and (not mask_boundary_tolerance_pixels or coverage < .95
                or not np.all(expanded_hits[index][child_hits])):
                continue
            child_points = points[ids == child]
            if len(np.unique(child_points, axis=0)) < 3:
                continue
            maximum_distance = float(np.max(tree.query(child_points)[0]))
            if maximum_distance > .3:
                continue
            proofs.append({
                "child_track_id": int(child), "parent_track_id": int(key),
                "source_timestamp_seconds": now,
                "camera_timestamp_seconds": camera_timestamp_seconds,
                "camera_frame_id": camera_frame_id,
                "child_positive_count": track.point_count,
                "parent_positive_count": tracks[key].point_count,
                "maximum_child_to_parent_3d_distance_m": maximum_distance,
                "parent_mask_coverage": float(np.mean(hit[owned])),
                "person_confidence": person.confidence,
                "parent_identity_basis": parent["identity_basis"],
                "original_person_camera_timestamp": parent["original_person_camera_timestamp"],
                "original_person_camera_frame_id": parent["original_person_camera_frame_id"],
                "status": "prior" if boundary_count else "inferred",
                "basis": "fresh_person_mask_boundary_prior_positive_small_fragment"
                    if boundary_count else "fresh_person_mask_whole_positive_small_fragment",
                **({"mask_boundary_prior_pixels": 1,
                    "child_original_mask_coverage": coverage,
                    "child_boundary_return_count": boundary_count,
                    "all_child_returns_inside_bounded_mask": True,
                    "mask_boundary_prior_scope": "one native-image axial pixel; >=95% original person-mask membership; positive 3D association only, no physical factor/geometry renewal"}
                   if boundary_count else {}),
                "scope": "separated bare adult; current positive part association only; raw occupancy/factors/identity clocks unchanged",
            })
    # A child fitting multiple parents has no unique identity. Reject every
    # competing proposal rather than selecting the nearest or first parent.
    return [p for p in proofs if sum(q["child_track_id"] == p["child_track_id"] for q in proofs) == 1
            and not any(q["child_track_id"] == p["parent_track_id"] for q in proofs)]
