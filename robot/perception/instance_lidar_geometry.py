"""Select observed instance returns without deleting other obstacle geometry."""

import cv2
import numpy as np
from dataclasses import dataclass

from robot.perception.camera_lidar_fusion import ProjectedLidarPoints
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection


CHROMATIC_INSTANCE_SOURCE = "lidar_seeded_chromatic_rgb_component"


@dataclass(frozen=True)
class ChromaticInstanceResult:
    detection: SemanticDetection | None
    rejection_reasons: tuple[str, ...]
    source: str = CHROMATIC_INSTANCE_SOURCE


def propose_chromatic_instance(
    rgb, point_track_ids, projection, *, track_id, separated_chromatic_domain_validated
):
    """Bounded class-independent RGB proposal seeded by observed LiDAR pixels.

    This opt-in baseline requires one separated chromatic object against a
    contrasting local background. It uses no authored color, shape, height or mask.
    Gray/multicolor/touching objects and ambiguous competing components are
    unsupported. The output class stays unknown, with heuristic confidence .5.
    """
    if not isinstance(separated_chromatic_domain_validated, bool):
        raise TypeError("domain validation must be boolean")
    if not separated_chromatic_domain_validated:
        return ChromaticInstanceResult(
            None, ("separated chromatic domain not validated",)
        )
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 uint8")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    chromatic = (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 35)
    xy = projection.pixel_indices_xy
    masked_track_return_selection(
        point_track_ids, projection, chromatic, track_id=track_id, margin_pixels=0
    )
    selected = np.asarray(point_track_ids)[projection.source_indices] == track_id
    candidates = []
    # Overlapping circular hue bands separate differently colored neighbors.
    # No band is selected from authored fixture color.
    for center in range(0, 180, 12):
        distance = abs(hsv[:, :, 0].astype(float) - center)
        color = chromatic & (np.minimum(distance, 180 - distance) <= 12)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            color.astype(np.uint8), 8
        )
        hits = np.bincount(labels[xy[selected, 1], xy[selected, 0]], minlength=count)
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if (
                area < 200
                or hits[index] < 40
                or x == 0
                or y == 0
                or x + width == rgb.shape[1]
                or y + height == rgb.shape[0]
            ):
                continue
            mask = labels == index
            # Deduplicate overlapping bands belonging to the same instance.
            overlaps = [
                np.count_nonzero(mask & other) / max(1, min(mask.sum(), other.sum()))
                for _, other in candidates
            ]
            duplicate = next(
                (i for i, overlap in enumerate(overlaps) if overlap > 0.8), None
            )
            if duplicate is None:
                candidates.append((int(hits[index]), mask))
            elif hits[index] > candidates[duplicate][0]:
                candidates[duplicate] = (int(hits[index]), mask)
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return ChromaticInstanceResult(
            None, ("no separated chromatic component with sufficient LiDAR hits",)
        )
    if len(candidates) > 1 and candidates[0][0] < 2 * candidates[1][0]:
        return ChromaticInstanceResult(
            None, ("competing chromatic components are ambiguous",)
        )
    _, mask = candidates[0]
    boundary = (
        cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        & ~mask
    )
    # A border-connected background (for example the floor) is expected beside
    # an outline. A bounded adjacent chromatic region is ambiguous.
    _, adjacent_labels, adjacent_stats, _ = cv2.connectedComponentsWithStats(
        (chromatic & ~mask).astype(np.uint8), 8
    )
    for index in np.unique(adjacent_labels[boundary & chromatic]):
        if index == 0:
            continue
        x, y, width, height, area = adjacent_stats[index]
        if (
            area >= 200
            and x > 0
            and y > 0
            and x + width < rgb.shape[1]
            and y + height < rgb.shape[0]
        ):
            return ChromaticInstanceResult(
                None, ("component touches a bounded chromatic neighbor",)
            )
    hue = hsv[:, :, 0][mask].astype(float)
    angles = hue * (2 * np.pi / 180)
    concentration = np.hypot(np.sin(angles).mean(), np.cos(angles).mean())
    if concentration < 0.95:
        return ChromaticInstanceResult(None, ("component is not uniformly chromatic",))
    # A tiny neutral specular hole does not alter the external silhouette.
    # Large interiors remain ambiguous; this is explicit outline inference.
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    if filled.sum() - mask.sum() > 0.01 * mask.sum():
        return ChromaticInstanceResult(
            None, ("interior gap exceeds small specular-hole allowance",)
        )
    mask = filled.astype(bool)
    ys, xs = np.where(mask)
    x, y, width, height = xs.min(), ys.min(), np.ptp(xs) + 1, np.ptp(ys) + 1
    return ChromaticInstanceResult(
        SemanticDetection(
            0,
            "unknown",
            0.5,
            PixelBoundingBox(float(x), float(y), float(x + width), float(y + height)),
            mask,
        ),
        (),
    )


def masked_track_return_selection(
    point_track_ids,
    projection: ProjectedLidarPoints,
    instance_mask,
    *,
    track_id,
    margin_pixels=6,
):
    """Return a mask aligned with the original LiDAR obstacle array.

    Requires an already matched, unambiguous camera instance. The configured
    pixel margin accounts for registration error; it is not depth separation.
    Selected points must still pass the downstream shape/geometry gates.
    The complement remains occupied and must not inherit the instance's risk.
    """
    ids = np.asarray(point_track_ids)
    mask = np.asarray(instance_mask)
    if ids.ndim != 1 or mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("track IDs must be 1D and instance mask must be 2D boolean")
    if (
        isinstance(margin_pixels, bool)
        or not isinstance(margin_pixels, int)
        or margin_pixels < 0
    ):
        raise ValueError("margin_pixels must be a nonnegative integer")
    indices = projection.source_indices
    xy = projection.pixel_indices_xy
    if (
        np.any(indices < 0)
        or np.any(indices >= len(ids))
        or np.any(xy < 0)
        or np.any(xy[:, 0] >= mask.shape[1])
        or np.any(xy[:, 1] >= mask.shape[0])
    ):
        raise ValueError("projection indices fall outside the supplied arrays")
    if margin_pixels:
        size = 2 * margin_pixels + 1
        mask = cv2.dilate(
            mask.astype(np.uint8), np.ones((size, size), np.uint8)
        ).astype(bool)
    selected = np.zeros(len(ids), dtype=bool)
    hits = (ids[indices] == track_id) & mask[xy[:, 1], xy[:, 0]]
    selected[indices[hits]] = True
    return selected
