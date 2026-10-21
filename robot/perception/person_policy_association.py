"""Fresh detector-to-LiDAR class cue for an already validated human identity.

This cue is for passing-policy eligibility only. It is not SAM acceptance,
physical evidence, a mass estimate, body occupancy or an identity-clock renewal.
"""

import math

import numpy as np

from robot.perception.camera_lidar_fusion import mask_foreground_hits
from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE
from robot.perception.human_mass_prior import HumanMassPriorConfig


def fresh_person_policy_associations(
    tracking, projection, person_frame, parent_records, *,
    camera_timestamp_seconds, camera_frame_id, config,
    minimum_projected_points=20, minimum_parent_coverage=.9, foreground_depth_gap_m=.5,
):
    """Use current person pixels and depth, retaining original parent provenance.

    The bounded separated-adult experiment already validates parent identity
    and current LiDAR continuity. This requires an additional fresh untruncated
    person detection, >=90% mask/foreground support and unique person/parent
    ownership. Missing/ambiguous cues remain ineligible; no physical input is
    filled or refreshed and all raw occupancy remains unchanged.
    """
    if not isinstance(config, HumanMassPriorConfig):
        raise TypeError("audited adult-person association configuration required")
    if (isinstance(minimum_projected_points, bool) or not isinstance(minimum_projected_points, int)
        or minimum_projected_points < 20
        or isinstance(minimum_parent_coverage, bool) or isinstance(foreground_depth_gap_m, bool)
        or not .9 <= minimum_parent_coverage <= 1
        or not .3 <= foreground_depth_gap_m <= .5):
        raise ValueError("invalid bounded person-policy association thresholds")
    now = tracking.timestamp_seconds
    if (not math.isfinite(camera_timestamp_seconds)
        or not 0 <= camera_timestamp_seconds - now <= config.maximum_frame_age_seconds + 1e-9
        or person_frame.source_frame_id != camera_frame_id
        or person_frame.source_timestamp_seconds != camera_timestamp_seconds):
        raise ValueError("fresh aligned camera/scan epochs required for person-policy selection")
    indices, xy = projection.source_indices, projection.pixel_indices_xy
    if (np.any(indices < 0) or np.any(indices >= len(tracking.point_track_ids))
        or len(np.unique(indices)) != len(indices)
        or np.any(xy < 0) or np.any(xy[:, 0] >= person_frame.image_width)
        or np.any(xy[:, 1] >= person_frame.image_height)):
        raise ValueError("person-policy projection differs from source frame")
    tracks = {t.track_id: t for t in tracking.tracked_obstacles}
    ids = tracking.point_track_ids[indices]
    people = [(i, d) for i, d in enumerate(person_frame.detections)
              if d.semantic_class == "person" and d.confidence >= config.minimum_person_confidence]
    proposals = []
    for parent in parent_records:
        key = parent["track_id"]
        track = tracks.get(key)
        if (track is None or not track.matched_previous_track or track.observation_count < 2
            or parent.get("source") != HUMAN_ENVELOPE_SOURCE or parent.get("status") != "prior"
            or parent["lidar_timestamp_seconds"] != now
            or parent["current_support_count"] != track.point_count
            or track.point_count != np.count_nonzero(tracking.point_track_ids == key)
            or not 0 <= parent["original_person_camera_timestamp"] <= camera_timestamp_seconds):
            continue
        owned = ids == key
        count = int(owned.sum())
        if count < minimum_projected_points:
            continue
        candidates = []
        for index, person in people:
            mask = person.mask
            coverage = float(np.mean(mask[xy[owned, 1], xy[owned, 0]]))
            if coverage < minimum_parent_coverage:
                continue
            if (person.mask_pixel_count < config.minimum_mask_pixels or mask[0].any()
                or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()):
                continue
            # Competing positive person masks reject, rather than choosing the
            # largest score and quietly assigning somebody else's class.
            if any(np.count_nonzero(mask & other.mask) / max(1, np.count_nonzero(mask | other.mask))
                   >= config.competitor_iou for i, other in people if i != index):
                continue
            foreground = mask_foreground_hits(projection, mask, gap_m=foreground_depth_gap_m)
            foreground_coverage = float(np.mean(foreground[owned]))
            if foreground_coverage < minimum_parent_coverage:
                continue
            candidates.append((index, {
                "basis": "fresh_person_detector_mask_with_current_validated_LiDAR_identity",
                "status": "inferred", "camera_frame_id": camera_frame_id,
                "camera_timestamp_seconds": camera_timestamp_seconds,
                "lidar_timestamp_seconds": now, "person_model": person_frame.model_name,
                "person_detection_index": index, "person_confidence": person.confidence,
                "projected_parent_returns": count, "parent_mask_coverage": coverage,
                "parent_foreground_coverage": foreground_coverage,
                "foreground_depth_gap_m": foreground_depth_gap_m,
                "foreground_depth_basis": "separated first-observed-depth-layer association hypothesis",
                "minimum_parent_coverage": minimum_parent_coverage,
                "minimum_projected_points": minimum_projected_points,
                "detector_domain_scope": config.scope,
                "original_parent_person_timestamp": parent["original_person_camera_timestamp"],
                "original_parent_person_frame_id": parent["original_person_camera_frame_id"],
                "parent_identity_basis": parent["identity_basis"],
                "sam_instance_assessment": False, "physical_evidence_refreshed": False,
                "scope": "bounded separated adult; class cue only for policy selection; no mass/contact/support/plume, geometry or identity-clock renewal",
            }))
        if len(candidates) == 1:
            index, evidence = candidates[0]
            proposals.append((key, index, evidence))
    return {key: evidence for key, index, evidence in proposals
            if sum(i == index for _, i, _ in proposals) == 1
            and sum(k == key for k, _, _ in proposals) == 1}
