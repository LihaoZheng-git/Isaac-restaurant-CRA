"""Bounded stationary image-object hypotheses across positive LiDAR fragments.

This is an inferred grouping in separated-prop views, not proof that disconnected
surfaces constitute one physical object. Never creates geometry or transfers raw
track identity. Missing/partial masks publish no fresh object measurement.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from robot.perception.camera_lidar_fusion import mask_foreground_hits


# Furniture retention owns 5B+raw_track_id. Image hypotheses are a different
# identity domain and must not overwrite that geometry's field policy.
IMAGE_OBJECT_ID_START = 8_000_000_000
IMAGE_OBJECT_ID_LIMIT = 9_000_000_000


@dataclass(frozen=True)
class ImageObjectConfig:
    maximum_extent_m: float = .9
    minimum_fragment_coverage: float = .8
    maximum_foreign_fraction: float = .15
    foreground_depth_gap_m: float = .5
    support_distance_m: float = .06
    maximum_gap_seconds: float = .35
    minimum_span_seconds: float = .2
    minimum_observations: int = 3
    minimum_mask_iou: float = .75
    maximum_reference_age_seconds: float = 10.
    hold_overlapping_hypotheses: bool = False
    confirmed_evidence_refresh: bool = False

    def __post_init__(self):
        if not isinstance(self.hold_overlapping_hypotheses, bool):
            raise TypeError("overlapping-hypothesis hold must be boolean")
        if not isinstance(self.confirmed_evidence_refresh, bool):
            raise TypeError('confirmed evidence refresh must be boolean')
        for name in ("maximum_extent_m", "foreground_depth_gap_m", "support_distance_m",
                     "maximum_gap_seconds", "minimum_span_seconds", "maximum_reference_age_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("minimum_fragment_coverage", "maximum_foreign_fraction", "minimum_mask_iou"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        if isinstance(self.minimum_observations, bool) or not isinstance(self.minimum_observations, int) or self.minimum_observations < 3:
            raise ValueError("at least three distinct observations required")


def associate_image_objects(proposals, projection, point_track_ids, xyz, config):
    """Select exact current visible returns, retaining original raw IDs separately."""
    xyz = np.asarray(xyz)
    ids = np.asarray(point_track_ids)
    if xyz.shape != (len(ids), 3) or not np.isfinite(xyz).all():
        raise ValueError("finite aligned XYZ and raw IDs required")
    projected_ids = ids[projection.source_indices]
    candidates, audit = [], []
    for index, proposal in enumerate(proposals):
        if proposal.detection is None:
            continue
        mask = proposal.detection.mask
        hits = mask_foreground_hits(projection, mask, gap_m=config.foreground_depth_gap_m)
        eligible, fragments = [], []
        for key in np.unique(projected_ids[hits]):
            own = projected_ids == key
            count = int(np.sum(own & hits))
            coverage = count / max(1, int(own.sum()))
            if key >= 0 and count >= 4 and coverage >= config.minimum_fragment_coverage:
                eligible.append(int(key))
                fragments.append({"raw_track_id": int(key), "hits": count, "visible_coverage": coverage})
        selected = np.zeros(len(ids), bool)
        selected[projection.source_indices[hits & np.isin(projected_ids, eligible)]] = True
        foreign = 1 - int(selected.sum()) / max(1, int(hits.sum()))
        reasons = []
        if selected.sum() < 8:
            reasons.append("insufficient qualified returns")
        if foreign > config.maximum_foreign_fraction:
            reasons.append("unresolved foreground ownership")
        if selected.any() and np.ptp(xyz[selected], axis=0).max() > config.maximum_extent_m:
            reasons.append("extent exceeds separated-prop assumption")
        row = {"proposal_index": index, "fragments": fragments, "selected_returns": int(selected.sum()),
               "foreign_fraction": foreign, "rejections": reasons,
               "model_quality":getattr(proposal,'model_quality',None)}
        audit.append(row)
        if not reasons:
            candidates.append({"proposal": proposal, "selected": selected, "diagnostic": row})
    # A merged mask containing two disjoint viable objects is ambiguous. Nested
    # parts are otherwise suppressed in favor of their viable larger outline.
    candidates.sort(key=lambda c: -c["proposal"].detection.mask_pixel_count)
    rejected = set()
    for i, a in enumerate(candidates):
        am = a["proposal"].detection.mask
        for j in range(i+1, len(candidates)):
            b = candidates[j]
            bm = b["proposal"].detection.mask
            overlap = np.sum(am & bm)
            equivalent = overlap / max(1, np.sum(am | bm)) >= .85
            nested = overlap >= .98 * min(am.sum(), bm.sum())
            if np.any(a["selected"] & b["selected"]) and not equivalent and not nested:
                rejected.update((i, j))
                a["diagnostic"]["rejections"].append("competing nonnested masks")
                b["diagnostic"]["rejections"].append("competing nonnested masks")
    for i, a in enumerate(candidates):
        am = a["proposal"].detection.mask
        children = [b for b in candidates[i+1:] if np.sum(am & b["proposal"].detection.mask)
                    >= .98 * b["proposal"].detection.mask_pixel_count]
        for j, b in enumerate(children):
            for c in children[j+1:]:
                bm, cm = b["proposal"].detection.mask, c["proposal"].detection.mask
                if np.sum(bm & cm) < .1 * min(bm.sum(), cm.sum()) and not np.any(b["selected"] & c["selected"]):
                    rejected.add(i)
        if children and i in rejected and "competing nonnested masks" not in a["diagnostic"]["rejections"]:
            a["diagnostic"]["rejections"].append("merged mask contains disjoint object hypotheses")
    chosen = []
    for i, a in enumerate(candidates):
        if i in rejected:
            continue
        overlaps = [b for b in chosen if np.any(a["selected"] & b["selected"])]
        if overlaps:
            a["diagnostic"]["rejections"].append("contained part or equivalent mask suppressed")
            continue
        chosen.append(a)
    return chosen, audit


class StationaryImageObjects:
    """Short repeated-current proof; stores no factor evidence or occupancy."""

    def __init__(self, config):
        self.config = config
        self.records = {}
        self.next_id = IMAGE_OBJECT_ID_START
        self.last_time = -math.inf
        if config.confirmed_evidence_refresh:
            from robot.perception.confirmed_view_refresh import ConfirmedViewRefresh
            self.refresh = ConfirmedViewRefresh()

    def update(self, candidates, world_xyz, timestamp, *, view=None):
        if not math.isfinite(timestamp) or timestamp <= self.last_time:
            raise ValueError("image object source epochs must increase")
        self.last_time = timestamp
        cfg = self.config
        # A bounded original-outline reference can reject part-only updates even
        # after a source gap. It supplies NO assessment, geometry or fresh clock.
        self.records = {k: v for k, v in self.records.items()
                        if timestamp - v["first"] <= cfg.maximum_reference_age_seconds}
        if cfg.confirmed_evidence_refresh:
            self.refresh.prune(self.records, timestamp, cfg.maximum_gap_seconds)
        accepted, seen = [], set()
        for candidate in candidates:
            mask = candidate["proposal"].detection.mask
            points = np.asarray(world_xyz)[candidate["selected"]]
            matches = []
            overlapping = []
            partial = False
            comparisons = []
            for key, old in self.records.items():
                if view is not None:
                    from robot.perception.moving_view_association import compare_moving_view
                    matched, overlap, comparison = compare_moving_view(old, mask, points, view, cfg)
                    comparison['object_id'] = key
                    comparisons.append(comparison)
                    if matched:
                        matches.append(key)
                    elif overlap:
                        overlapping.append(comparison)
                    continue
                intersection = np.sum(mask & old["mask"])
                iou = intersection / max(1, np.sum(mask | old["mask"]))
                current_to_old = cKDTree(old["points"]).query(points)[0]
                old_to_current = cKDTree(points).query(old["points"])[0]
                near = current_to_old.max() <= cfg.support_distance_m
                comparisons.append({"object_id": key, "original_source_time": old['first'],
                    "mask_iou": float(iou), "current_maximum_distance_m": float(current_to_old.max()),
                    "original_maximum_distance_m": float(old_to_current.max())})
                if near and intersection >= .98 * mask.sum() and mask.sum() < .8 * old["mask"].sum():
                    partial = True
                if iou < cfg.minimum_mask_iou:
                    continue
                distance = max(old_to_current.max(), current_to_old.max())
                if distance <= cfg.support_distance_m:
                    matches.append(key)
                elif cfg.hold_overlapping_hypotheses and current_to_old.min() <= cfg.support_distance_m:
                    overlapping.append({"object_id": key, "original_source_time": old["first"],
                        "mask_iou": float(iou), "maximum_support_distance_m": float(distance)})
            candidate['diagnostic']['reference_comparisons'] = comparisons
            if partial or len(matches) > 1 or any(k in seen for k in matches):
                candidate["diagnostic"]["temporal_rejection"] = "partial or ambiguous current object"
                continue
            if matches:
                key = matches[0]
                old = self.records[key]
                # Compare every current source to the original support and mask;
                # a chain of small drifts must not gradually shrink the object.
                if timestamp - old["last"] > cfg.maximum_gap_seconds + 1e-9:
                    old["count"], old["proof_start"] = 0, timestamp
                old["last"] = timestamp
                old["count"] += 1
            else:
                # A partly overlapping observation is insufficient proof of a
                # NEW object. Reject this source instead of creating competing
                # histories that later veto an unchanged valid original. This
                # supplies no geometry/identity/evidence and renews no clocks.
                # A future source must still pass the complete original support
                # test; expired references and spatially separate objects keep
                # the existing three-source enrollment path.
                if overlapping:
                    if cfg.confirmed_evidence_refresh and len(overlapping) == 1:
                        key = overlapping[0]['object_id']
                        if key not in seen and self.refresh.consider(key, self.records[key],
                                candidate, points, view, timestamp, cfg):
                            candidate['diagnostic'].update(object_id=key,
                                observations=candidate['diagnostic']['evidence_refresh']['observations'],
                                original_source_time=self.records[key]['first'], current_source_time=timestamp,
                                association_basis='confirmed_static_surface_growth',
                                reference_surface_centroid_world_xyz=self.records[key]['points'].mean(axis=0).tolist())
                            accepted.append({**candidate, 'object_id':key})
                            seen.add(key)
                            continue
                    candidate["diagnostic"]["temporal_rejection"] = "unmatched support overlaps an original image hypothesis"
                    candidate["diagnostic"]["overlapping_original_hypotheses"] = overlapping
                    continue
                if self.next_id >= IMAGE_OBJECT_ID_LIMIT:
                    raise RuntimeError("image-object identity range exhausted")
                key = self.next_id
                self.next_id += 1
                old = {"first": timestamp, "proof_start": timestamp, "last": timestamp, "count": 1,
                       "mask": mask.copy(), "points": points.copy(),
                       "model_quality":getattr(candidate['proposal'],'model_quality',None)}
                if view is not None:
                    old['camera_from_world'] = view.camera_from_world.copy()
                self.records[key] = old
            seen.add(key)
            candidate["diagnostic"].update(object_id=key, observations=old["count"],
                original_source_time=old["first"], current_source_time=timestamp)
            if view is not None:
                candidate['diagnostic'].update(
                    association_basis='bounded_static_prop_reprojected_outline_and_partial_surface_prior',
                    reference_surface_centroid_world_xyz=old['points'].mean(axis=0).tolist(),
                    reference_point_count=len(old['points']),
                    reference_geometry_scope='original observed surface retained for identity only; not complete physical center or current geometry')
            if old["count"] >= cfg.minimum_observations and timestamp - old["proof_start"] >= cfg.minimum_span_seconds - 1e-9:
                accepted.append({**candidate, "object_id": key})
        if cfg.confirmed_evidence_refresh:
            # A missing/ambiguous/interrupted observation cannot count toward a
            # later consecutive proof, even if its source gap would be short.
            self.refresh.pending = {k:v for k,v in self.refresh.pending.items()
                                    if v['last'] == timestamp}
        return accepted

    def commit_refresh(self, candidate, world_xyz, view, timestamp):
        from robot.perception.confirmed_view_refresh import refreshed_reference
        key = candidate['object_id']
        if not candidate['diagnostic'].get('evidence_refresh', {}).get('confirmed'):
            raise ValueError('cannot commit unconfirmed reference refresh')
        self.records[key] = refreshed_reference(self.records[key], candidate,
            np.asarray(world_xyz)[candidate['selected']], view, timestamp)
        self.refresh.pending.pop(key, None)
