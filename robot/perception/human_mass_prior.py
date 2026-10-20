"""Explicit adult-human mass PRIOR selected by fresh, matched visual evidence.

The classifier supplies a domain cue, never a measured mass. Asset names,
authored poses, body dimensions and density are not accepted by this adapter.
"""

from dataclasses import dataclass
import math

import numpy as np

from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.perception.sam_instance_segmentation import SAM_INSTANCE_SOURCE

HUMAN_MASS_PRIOR_SOURCE = "configured_adult_human_mass_prior_visual_association"
PERSON_CHECKPOINT_SHA256 = (
    "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"
)


@dataclass(frozen=True)
class HumanMassPriorConfig:
    scope: str
    mass_kg: float = 70.0
    lower_kg: float = 45.0
    upper_kg: float = 100.0
    confidence: float = 0.5
    minimum_person_confidence: float = 0.5
    minimum_mask_iou: float = 0.5
    competitor_iou: float = 0.15
    maximum_frame_age_seconds: float = 0.15
    minimum_mask_pixels: int = 200
    catalog_domain_containment: float = 0.5

    def __post_init__(self):
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("human mass prior requires an explicit adult-human scope")
        if any(
            isinstance(v, bool) or not math.isfinite(v)
            for k, v in vars(self).items()
            if k != "scope"
        ):
            raise ValueError("human prior parameters must be finite numbers")
        if not 0 < self.lower_kg <= self.mass_kg <= self.upper_kg <= 300:
            raise ValueError("invalid human mass prior interval")
        if not (
            0 < self.confidence <= 0.5
            and 0.5 <= self.minimum_person_confidence <= 1
            and 0.5 <= self.minimum_mask_iou <= 1
            and 0 < self.competitor_iou < self.minimum_mask_iou
            and 0 < self.maximum_frame_age_seconds <= 0.5
            and 0 < self.catalog_domain_containment <= 1
        ):
            raise ValueError("invalid bounded human association thresholds")
        if (
            not isinstance(self.minimum_mask_pixels, int)
            or self.minimum_mask_pixels < 200
        ):
            raise ValueError("human mask must have at least 200 pixels")


def catalog_person_overlap_guard(
    instance_frame, association, person_frame, *, track_id, timestamp_seconds, config
):
    """Positive person pixels reject catalog inference; never assign identity/mass.

    Containment differs from whole-mask IoU: a small SAM fragment can be mostly
    inside a person while lacking sufficient ownership for the adult prior.
    Truncated/competing person masks may reject the catalog domain, but cannot
    grant a human prior. Missing/old positive cues do not prove nonhuman class.
    """
    if not isinstance(config, HumanMassPriorConfig):
        raise TypeError("config must be HumanMassPriorConfig")
    audit = {"blocked": False, "basis": "fresh_positive_person_mask_containment",
             "minimum_containment": config.catalog_domain_containment,
             "source_frame_id": instance_frame.source_frame_id,
             "source_timestamp_seconds": instance_frame.source_timestamp_seconds,
             "person_hits": []}
    if (association is None or not association.is_matched
        or association.track_id != track_id or association.detection_index is None
        or not 0 <= association.detection_index < len(instance_frame.detections)
        or SAM_INSTANCE_SOURCE not in instance_frame.model_name.split("+")
        or person_frame.source_frame_id != instance_frame.source_frame_id
        or person_frame.source_timestamp_seconds != instance_frame.source_timestamp_seconds
        or (person_frame.image_height, person_frame.image_width)
        != (instance_frame.image_height, instance_frame.image_width)
        or not 0 <= timestamp_seconds - instance_frame.source_timestamp_seconds
        <= config.maximum_frame_age_seconds):
        audit["rejection_reason"] = "fresh aligned accepted SAM/person frames required"
        return audit
    mask = instance_frame.detections[association.detection_index].mask
    count = int(np.count_nonzero(mask))
    for index, person in enumerate(person_frame.detections):
        if person.semantic_class != "person" or person.confidence < config.minimum_person_confidence:
            continue
        fraction = np.count_nonzero(mask & person.mask) / count if count else 0.
        if fraction >= config.catalog_domain_containment:
            audit["person_hits"].append({"index": index, "instance_containment": float(fraction)})
    audit["blocked"] = bool(audit["person_hits"])
    return audit


def estimate_visual_human_mass_prior(
    instance_frame, association, person_frame, *, track_id, timestamp_seconds, config
):
    """Require same-frame unambiguous mask overlap and an accepted SAM track."""

    def missing(reason):
        return ScalarEstimate.unavailable(
            unit="kg",
            source=HUMAN_MASS_PRIOR_SOURCE,
            timestamp_seconds=timestamp_seconds,
            evidence=(reason,),
        )

    if not isinstance(config, HumanMassPriorConfig):
        raise TypeError("config must be HumanMassPriorConfig")
    if (
        association is None
        or not association.is_matched
        or association.track_id != track_id
        or association.detection_index is None
        or not 0 <= association.detection_index < len(instance_frame.detections)
        or SAM_INSTANCE_SOURCE not in instance_frame.model_name.split("+")
    ):
        return missing("fresh accepted same-track SAM instance is required")
    if (
        person_frame.source_frame_id != instance_frame.source_frame_id
        or person_frame.source_timestamp_seconds
        != instance_frame.source_timestamp_seconds
        or (person_frame.image_height, person_frame.image_width)
        != (instance_frame.image_height, instance_frame.image_width)
        or not 0
        <= timestamp_seconds - instance_frame.source_timestamp_seconds
        <= config.maximum_frame_age_seconds
    ):
        return missing(
            "person and instance frames must share a fresh acquisition epoch"
        )
    instance = instance_frame.detections[association.detection_index]
    mask = instance.mask
    if (
        instance.confidence < 0.35
        or instance.mask_pixel_count < config.minimum_mask_pixels
        or mask[0].any()
        or mask[-1].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    ):
        return missing("instance mask is unresolved or truncated")
    candidates = []
    for index, person in enumerate(person_frame.detections):
        if (
            person.semantic_class != "person"
            or person.confidence < config.minimum_person_confidence
        ):
            continue
        p = person.mask
        overlap = int(np.count_nonzero(mask & p))
        union = int(np.count_nonzero(mask | p))
        iou = overlap / union if union else 0.0
        if iou < config.competitor_iou:
            continue
        if (
            person.mask_pixel_count < config.minimum_mask_pixels
            or p[0].any()
            or p[-1].any()
            or p[:, 0].any()
            or p[:, -1].any()
        ):
            return missing("overlapping person mask is unresolved or truncated")
        candidates.append((index, person, iou))
    if len(candidates) != 1 or candidates[0][2] < config.minimum_mask_iou:
        return missing("person/instance ownership is absent, weak or ambiguous")
    index, person, iou = candidates[0]
    return ScalarEstimate(
        config.mass_kg,
        config.lower_kg,
        config.upper_kg,
        min(config.confidence, instance.confidence, person.confidence),
        "kg",
        HUMAN_MASS_PRIOR_SOURCE,
        person_frame.source_timestamp_seconds,
        EstimateStatus.PRIOR,
        (
            config.scope,
            "configured engineering interval; not measured mass or a learned human-mass estimate",
            f"person_model={person_frame.model_name}; person_index={index}; mask_iou={iou}",
            f"track_id={track_id}; source_frame_id={person_frame.source_frame_id}",
            "adult-human demonstration only; no population, child or carried-load mass calibration",
        ),
    )
