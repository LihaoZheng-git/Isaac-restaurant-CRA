"""Bounded RGB-wrist/positive-LiDAR carried-instance proposals and co-motion.

One clearly observed adult and one distinct object are supported. A proposed
relationship is inferred, not a grasp measurement. No authored actor, prop,
joint, asset-name or heat information enters this module.
"""

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.spatial import cKDTree

from robot.perception.camera_lidar_fusion import mask_foreground_hits
from robot.perception.semantic_detection import PixelBoundingBox
from robot.safety.obstacle_tracking import _cluster_point_indices


@dataclass(frozen=True)
class CarriedObjectConfig:
    allow_occluding_subinstance: bool = False
    minimum_wrist_confidence: float = 0.6
    wrist_pixel_radius_body_fraction: float = 0.30
    wrist_depth_radius_body_fraction: float = 0.10
    maximum_hand_distance_m: float = 0.6
    candidate_cluster_radius_m: float = 0.10
    candidate_growth_m: float = 0.04
    maximum_person_mask_overlap: float = 0.20
    minimum_points: int = 8
    minimum_observations: int = 3
    minimum_span_seconds: float = 0.4
    maximum_gap_seconds: float = 0.35
    maximum_window_seconds: float = 2.0
    minimum_human_translation_m: float = 0.12
    maximum_relative_drift_m: float = 0.15
    maximum_relative_speed_mps: float = 0.3

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "allow_occluding_subinstance":
                if not isinstance(value, bool):
                    raise ValueError("occluding subinstance opt-in must be boolean")
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not isinstance(self.minimum_points, int)
            or self.minimum_points < 8
            or not isinstance(self.minimum_observations, int)
            or self.minimum_observations < 3
            or self.minimum_wrist_confidence > 1
            or self.maximum_person_mask_overlap >= 0.5
            or self.maximum_window_seconds < self.minimum_span_seconds
        ):
            raise ValueError("invalid bounded carry thresholds")


def _person_crop(packet, projection, person):
    height, width = packet.rgb.shape[:2]
    box = person.bounding_box
    lo = np.maximum(
        [0, 0], np.floor([box.x_min - 0.7 * box.width, box.y_min - 0.15 * box.height])
    ).astype(int)
    hi = np.minimum(
        [width, height],
        np.ceil([box.x_max + 0.7 * box.width, box.y_max + 0.15 * box.height]),
    ).astype(int)
    keep = np.all(
        (projection.pixel_indices_xy >= lo) & (projection.pixel_indices_xy < hi), axis=1
    )
    local = replace(
        projection,
        source_indices=projection.source_indices[keep],
        pixel_xy=projection.pixel_xy[keep] - lo,
        pixel_indices_xy=projection.pixel_indices_xy[keep] - lo,
        camera_depths=projection.camera_depths[keep],
    )
    return lo, hi, local


def _uncrop_proposal(proposal, lo, hi, shape):
    full = np.zeros(shape, bool)
    full[lo[1] : hi[1], lo[0] : hi[0]] = proposal.detection.mask
    yy, xx = np.where(full)
    detection = replace(
        proposal.detection,
        mask=full,
        bounding_box=PixelBoundingBox(
            float(xx.min()), float(yy.min()), float(xx.max() + 1), float(yy.max() + 1)
        ),
    )
    return replace(
        proposal,
        detection=detection,
        prompt_xy=None
        if proposal.prompt_xy is None
        else tuple((np.asarray(proposal.prompt_xy) + lo).tolist()),
    )


def propose_observed_person(packet, projection, person, pose, segmenter):
    """Join current person-mask depth fragments before unchanged SAM gates."""
    lo, hi, local = _person_crop(packet, projection, person)
    segmenter.set_image(packet.rgb[lo[1] : hi[1], lo[0] : hi[0]])
    hits = mask_foreground_hits(projection, person.mask, gap_m=0.5)
    ids = np.full(len(packet.obstacle_xyz), -1, dtype=np.int64)
    ids[projection.source_indices[hits]] = 0
    # At this scanner's vertical FOV the shoulders may have no LiDAR return;
    # an actual lower-body return avoids accidentally prompting the bowl.
    torso = pose.keypoints[[11, 12, 13, 14]]
    torso = torso[torso[:, 2] >= 0.6, :2]
    own = np.flatnonzero(ids[local.source_indices] == 0)
    if not len(torso) or not len(own):
        return None, {"rejection": "no observed torso prompt/depth"}
    prompt = int(
        local.source_indices[
            own[
                np.argmin(
                    np.linalg.norm(
                        local.pixel_xy[own] + lo - torso.mean(axis=0), axis=1
                    )
                )
            ]
        ]
    )
    proposal = segmenter.propose(ids, local, track_id=0, prompt_source_index=prompt)
    detail = {
        "basis": "same-frame observed COCO person-mask depth partition; cropped SAM torso prompt",
        "crop_xyxy": [*lo.tolist(), *hi.tolist()],
        "prompt_xy_original": None
        if proposal.prompt_xy is None
        else (np.array(proposal.prompt_xy) + lo).tolist(),
        "sam_rejections": proposal.rejection_reasons,
        "sam_candidates": proposal.candidate_diagnostics,
    }
    if proposal.detection is None:
        return None, detail
    full = _uncrop_proposal(proposal, lo, hi, packet.rgb.shape[:2])
    selected = np.zeros(len(ids), bool)
    selected[
        local.source_indices[
            mask_foreground_hits(local, proposal.detection.mask, gap_m=0.5)
            & (ids[local.source_indices] == 0)
        ]
    ] = True
    if selected.sum() < 8:
        return None, {**detail, "rejection": "insufficient exact-mask human returns"}
    return {"proposal": full, "selected": selected}, detail


def propose_carried_object(
    packet, projection, person, pose, segmenter, config, *, crop_is_set=False
):
    """Crop only observed person context; preserve native pixel/source mapping.

    Wrist-near non-person returns seed a provisional 3D component. A small
    positive-geometry growth covers segmentation seams before unchanged SAM
    quality/coverage/foreign/boundary gates. The final mask must be mostly
    outside the person mask. All rejected/unowned returns remain occupancy.
    """
    height, width = packet.rgb.shape[:2]
    box = person.bounding_box
    if (
        person.mask[0].any()
        or person.mask[-1].any()
        or person.mask[:, 0].any()
        or person.mask[:, -1].any()
    ):
        return None, [{"rejection": "person is truncated"}]
    bh = box.height
    lo, hi, local = _person_crop(packet, projection, person)
    pix = local.pixel_indices_xy + lo
    indices = local.source_indices
    xyz = packet.obstacle_xyz
    inside_person = person.mask[pix[:, 1], pix[:, 0]]
    if not crop_is_set:
        segmenter.set_image(packet.rgb[lo[1] : hi[1], lo[0] : hi[0]])
    accepted, diagnostics = [], []
    if config.allow_occluding_subinstance:
        wrists = pose.keypoints[[9, 10]]
        hand_positions = []
        for wrist in wrists:
            distance = np.linalg.norm(local.pixel_xy + lo - wrist[:2], axis=1)
            near = np.flatnonzero(
                inside_person
                & (distance < config.wrist_depth_radius_body_fraction * bh)
            )
            if len(near):
                hand_positions.append(xyz[indices[near[np.argmin(distance[near])]]])
        if (
            len(hand_positions) == 2
            and np.all(wrists[:, 2] >= config.minimum_wrist_confidence)
            and np.linalg.norm(wrists[0, :2] - wrists[1, :2]) < 0.5 * bh
        ):
            middle = wrists[:, :2].mean(axis=0)
            for target in (middle, middle - [0, 0.06 * bh]):
                own = np.flatnonzero(
                    inside_person
                    & (np.linalg.norm(local.pixel_xy + lo - target, axis=1) < 0.12 * bh)
                )
                if not len(own):
                    continue
                seed = int(
                    indices[
                        own[
                            np.argmin(
                                np.linalg.norm(
                                    local.pixel_xy[own] + lo - target, axis=1
                                )
                            )
                        ]
                    ]
                )
                region = (np.linalg.norm(xyz[indices] - xyz[seed], axis=1) < 0.4) & (
                    np.linalg.norm(local.pixel_xy + lo - middle, axis=1) < 0.25 * bh
                )
                # Separate foreground prop depth from the torso behind it.
                # Unselected real returns remain physical occupancy.
                seed_position = int(np.flatnonzero(indices == seed)[0])
                region &= (
                    np.abs(local.camera_depths - local.camera_depths[seed_position])
                    <= 0.15
                )
                ids = np.full(len(xyz), -1, dtype=np.int64)
                ids[indices[region]] = 0
                proposal = segmenter.propose(
                    ids, local, track_id=0, prompt_source_index=seed
                )
                detail = {
                    "basis": "occluding compact subinstance near both RGB wrists",
                    "source_frame_id": packet.camera_frame_id,
                    "source_time": packet.camera_timestamp_seconds,
                    "sam_rejections": proposal.rejection_reasons,
                    "sam_candidates": proposal.candidate_diagnostics,
                }
                diagnostics.append(detail)
                if proposal.detection is None:
                    continue
                full = _uncrop_proposal(proposal, lo, hi, (height, width))
                m = full.detection.mask
                compact = (
                    full.detection.bounding_box.height <= 0.35 * bh
                    and full.detection.bounding_box.width >= 0.08 * bh
                    and m.sum() <= 0.4 * person.mask.sum()
                )
                yy, xx = np.where(m)
                near_both = all(
                    np.min(np.hypot(xx - w[0], yy - w[1])) < 0.2 * bh for w in wrists
                )
                if not compact or not near_both:
                    detail["rejection"] = (
                        "noncompact/body/one-limb mask; two-wrist subinstance unresolved"
                    )
                    continue
                selected = np.zeros(len(xyz), bool)
                selected[
                    indices[
                        mask_foreground_hits(local, proposal.detection.mask, gap_m=0.5)
                        & region
                    ]
                ] = True
                if selected.sum() < config.minimum_points:
                    continue
                result = {
                    "proposal": full,
                    "selected": selected,
                    "hand_robot_xyz": np.mean(hand_positions, axis=0),
                    "wrist_index": -1,
                    "wrist_confidence": float(wrists[:, 2].min()),
                    "person_mask_overlap": float(
                        np.count_nonzero(m & person.mask) / m.sum()
                    ),
                    "basis": "two_wrist_occluding_compact_subinstance",
                }
                if not any(
                    np.count_nonzero(m & a["proposal"].detection.mask)
                    / np.count_nonzero(m | a["proposal"].detection.mask)
                    >= 0.85
                    for a in accepted
                ):
                    accepted.append(result)
            if accepted:
                if len(accepted) == 1:
                    return accepted[0], diagnostics
                return None, diagnostics + [
                    {"rejection": "ambiguous distinct two-wrist subinstances"}
                ]
    for wrist_index in (9, 10):
        wrist = pose.keypoints[wrist_index]
        if wrist[2] < config.minimum_wrist_confidence:
            diagnostics.append(
                {"wrist": wrist_index, "rejection": "low RGB wrist confidence"}
            )
            continue
        distance = np.linalg.norm(local.pixel_xy + lo - wrist[:2], axis=1)
        near = np.flatnonzero(
            inside_person & (distance < config.wrist_depth_radius_body_fraction * bh)
        )
        if not len(near):
            diagnostics.append(
                {
                    "wrist": wrist_index,
                    "rejection": "no same-source wrist depth support",
                }
            )
            continue
        hand = xyz[indices[near[np.argmin(distance[near])]]]
        candidate = (
            ~inside_person
            & (distance < config.wrist_pixel_radius_body_fraction * bh)
            & (
                np.linalg.norm(xyz[indices] - hand, axis=1)
                < config.maximum_hand_distance_m
            )
        )
        groups = _cluster_point_indices(
            xyz[indices[candidate]],
            cluster_radius=config.candidate_cluster_radius_m,
            minimum_cluster_points=config.minimum_points,
            backend="kdtree",
        )
        groups = sorted(
            groups,
            key=lambda g: np.linalg.norm(
                xyz[indices[candidate]][g].mean(axis=0) - hand
            ),
        )[:3]
        for group in groups:
            seed = indices[candidate][group]
            # Positive-neighbor growth may include returns on either side of a
            # person-mask seam; it never invents returns or erases occupancy.
            grow = (
                cKDTree(xyz[seed]).query(xyz[indices], k=1)[0]
                <= config.candidate_growth_m
            )
            owned = indices[grow]
            ids = np.full(len(xyz), -1, dtype=np.int64)
            ids[owned] = 0
            proposal = segmenter.propose(ids, local, track_id=0)
            detail = {
                "wrist": wrist_index,
                "crop_xyxy": [*lo.tolist(), *hi.tolist()],
                "seed_returns": len(seed),
                "grown_returns": len(owned),
                "sam_rejections": proposal.rejection_reasons,
                "sam_candidates": proposal.candidate_diagnostics,
                "source_frame_id": packet.camera_frame_id,
                "source_time": packet.camera_timestamp_seconds,
            }
            diagnostics.append(detail)
            if proposal.detection is None:
                continue
            full = np.zeros((height, width), bool)
            full[lo[1] : hi[1], lo[0] : hi[0]] = proposal.detection.mask
            overlap = float(np.count_nonzero(full & person.mask) / full.sum())
            detail["person_mask_overlap"] = overlap
            if overlap > config.maximum_person_mask_overlap:
                detail["rejection"] = (
                    "candidate overlaps person; limb/merged object unresolved"
                )
                continue
            hits = mask_foreground_hits(local, proposal.detection.mask, gap_m=0.5)
            selected = np.zeros(len(xyz), bool)
            selected[local.source_indices[hits & grow]] = True
            if selected.sum() < config.minimum_points:
                detail["rejection"] = (
                    "insufficient exact-mask positive instance returns"
                )
                continue
            yy, xx = np.where(full)
            detection = replace(
                proposal.detection,
                mask=full,
                bounding_box=PixelBoundingBox(
                    float(xx.min()),
                    float(yy.min()),
                    float(xx.max() + 1),
                    float(yy.max() + 1),
                ),
            )
            proposal = replace(
                proposal,
                detection=detection,
                prompt_xy=tuple((np.asarray(proposal.prompt_xy) + lo).tolist()),
            )
            candidate_result = {
                "proposal": proposal,
                "selected": selected,
                "hand_robot_xyz": hand,
                "wrist_index": wrist_index,
                "wrist_confidence": float(wrist[2]),
                "person_mask_overlap": overlap,
            }
            if any(
                np.count_nonzero(full & a["proposal"].detection.mask)
                / np.count_nonzero(full | a["proposal"].detection.mask)
                >= 0.85
                for a in accepted
            ):
                continue
            accepted.append(candidate_result)
    if len(accepted) != 1:
        diagnostics.append(
            {
                "rejection": "no unique distinct carried-object proposal",
                "accepted_candidates": len(accepted),
            }
        )
        return None, diagnostics
    return accepted[0], diagnostics


@dataclass(frozen=True)
class CarryObservation:
    human_id: int
    object_id: int
    source_frame_id: int
    timestamp: float
    human_world_xyz: tuple
    object_world_xyz: tuple
    hand_distance_m: float
    wrist_confidence: float
    fresh_separate_instances: bool = True
    hand_world_xyz: tuple | None = None


class CarriedRelationHistory:
    def __init__(self, config):
        self.config = config
        self.samples = {}

    def update(self, observation):
        c = self.config
        row = observation
        if (
            not isinstance(row, CarryObservation)
            or not math.isfinite(row.timestamp)
            or row.timestamp < 0
            or not np.isfinite(row.human_world_xyz).all()
            or not np.isfinite(row.object_world_xyz).all()
            or np.shape(row.human_world_xyz) != (3,)
            or np.shape(row.object_world_xyz) != (3,)
            or (
                row.hand_world_xyz is not None
                and (
                    np.shape(row.hand_world_xyz) != (3,)
                    or not np.isfinite(row.hand_world_xyz).all()
                )
            )
        ):
            raise ValueError(
                "carry observation needs finite same-frame metric geometry"
            )
        key = (row.human_id, row.object_id)
        history = self.samples.get(key, [])
        if history and (
            row.timestamp <= history[-1].timestamp
            or row.source_frame_id <= history[-1].source_frame_id
        ):
            raise ValueError("carry source epochs must increase")
        if not row.fresh_separate_instances or not (
            0 <= row.hand_distance_m <= c.maximum_hand_distance_m
            and c.minimum_wrist_confidence <= row.wrist_confidence <= 1
        ):
            self.samples.pop(key, None)
            return {
                "confirmed": False,
                "reason": "fresh separate instances/wrist proximity unavailable",
            }
        if history and row.timestamp - history[-1].timestamp > c.maximum_gap_seconds:
            history = []
        history = [
            r
            for r in history
            if row.timestamp - r.timestamp <= c.maximum_window_seconds
        ] + [row]
        self.samples = {
            k: v
            for k, v in self.samples.items()
            if row.timestamp - v[-1].timestamp <= c.maximum_gap_seconds
        }
        self.samples[key] = history
        span = row.timestamp - history[0].timestamp
        human = np.array([r.human_world_xyz for r in history])
        obj = np.array([r.object_world_xyz for r in history])
        hand_rows = [r.hand_world_xyz for r in history]
        if all(h is not None for h in hand_rows):
            hands = np.asarray(hand_rows, dtype=float)
            relative = obj - hands
            motion_basis = "RGB wrist plus same-source LiDAR depth proxy"
        else:
            relative = obj - human
            motion_basis = "member centroid offset (no hand proxy supplied)"
        drift = float(np.linalg.norm(relative - relative[0], axis=1).max())
        translation = float(np.linalg.norm(human[:, :2] - human[0, :2], axis=1).max())
        relative_speed = (
            float(np.linalg.norm(relative[-1] - relative[0]) / span)
            if span > 0
            else None
        )
        checks = {
            "enough_distinct_frames": len(history) >= c.minimum_observations,
            "enough_time": span >= c.minimum_span_seconds - 1e-9,
            "actual_world_translation": translation >= c.minimum_human_translation_m,
            "relative_position_consistent": drift <= c.maximum_relative_drift_m,
            "co_motion": relative_speed is not None
            and relative_speed <= c.maximum_relative_speed_mps,
        }
        return {
            "confirmed": all(checks.values()),
            "checks": checks,
            "human_id": row.human_id,
            "object_id": row.object_id,
            "source_frame_id": row.source_frame_id,
            "source_time": row.timestamp,
            "original_window_start": history[0].timestamp,
            "span_seconds": span,
            "human_translation_m": translation,
            "relative_drift_m": drift,
            "relative_speed_mps": relative_speed,
            "motion_reference": motion_basis,
            "basis": "RGB wrist proximity + separately owned LiDAR world co-motion; inferred relationship, not grasp sensing",
        }
