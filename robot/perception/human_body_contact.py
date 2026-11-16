"""A declared blunt-body policy prior, selected by fresh visual person ownership.

This is not physical sharpness, hardness or an injury probability. Reuse the
existing person/SAM association gate; none of its configured mass numbers enter
the contact score. Carried objects require their own independent assessment.
"""

from dataclasses import dataclass
import math
from numbers import Real

from robot.perception.human_mass_prior import estimate_visual_human_mass_prior
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate


HUMAN_BODY_CONTACT_SOURCE = "configured_human_blunt_body_contact_prior"
HUMAN_BODY_CONTACT_BASIS = "human_blunt_body_contact_prior"


@dataclass(frozen=True)
class HumanBodyContactPriorConfig:
    score: float = 0.10
    scope: str = (
        "bounded adult restaurant body contact; low nonzero blunt-body prior; "
        "excludes carried objects and accepted hazardous-part evidence; uncalibrated"
    )

    def __post_init__(self):
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, Real)
            or not math.isfinite(self.score)
            or not 0 < self.score < 1
        ):
            raise ValueError(
                "human body contact prior must be finite and strictly between zero and one"
            )
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("human body contact prior requires an explicit scope")


def select_visual_human_body_contact_prior(
    instance_frame,
    association,
    person_frame,
    *,
    track_id,
    timestamp_seconds,
    association_config,
    config,
):
    if not isinstance(config, HumanBodyContactPriorConfig):
        raise TypeError("config must be HumanBodyContactPriorConfig")
    # Reuse only the validated visual ownership decision. Mass values are ignored.
    identity = estimate_visual_human_mass_prior(
        instance_frame,
        association,
        person_frame,
        track_id=track_id,
        timestamp_seconds=timestamp_seconds,
        config=association_config,
    )
    if not identity.is_available:
        return ScalarEstimate.unavailable(
            unit="1",
            source=HUMAN_BODY_CONTACT_SOURCE,
            timestamp_seconds=timestamp_seconds,
            evidence=(
                "fresh unambiguous person/SAM association required",
                *identity.evidence,
            ),
        )
    return ScalarEstimate(
        config.score,
        config.score,
        config.score,
        0.0,
        "1",
        HUMAN_BODY_CONTACT_SOURCE,
        identity.timestamp_seconds,
        EstimateStatus.PRIOR,
        (
            config.scope,
            f"track_id={track_id}",
            f"source_frame_id={instance_frame.source_frame_id}",
            "configured policy constant; equal bounds are not physical uncertainty bounds",
            "zero measurement confidence; person evidence selects a prior, not measured safety",
            "human mass values, body dimensions, speed and class text are not contact-score inputs",
            *(item for item in identity.evidence if item.startswith("person_model=")),
        ),
    )
