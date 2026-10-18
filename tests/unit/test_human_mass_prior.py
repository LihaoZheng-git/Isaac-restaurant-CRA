from dataclasses import replace

import numpy as np
import pytest

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.human_mass_prior import (
    HumanMassPriorConfig,
    estimate_visual_human_mass_prior,
)
from robot.perception.physical_attributes import EstimateStatus
from robot.perception.sam_instance_segmentation import SAM_INSTANCE_SOURCE
from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)


def inputs():
    mask = np.zeros((40, 40), bool)
    mask[5:35, 10:30] = True
    instance = SemanticDetection(
        0, "unknown", 0.5, PixelBoundingBox(10, 5, 30, 35), mask
    )
    frame = SemanticDetectionFrame(
        4, 2.0, 40, 40, (instance,), SAM_INSTANCE_SOURCE, 0.1
    )
    people = replace(
        frame,
        detections=(replace(instance, semantic_class="person", confidence=0.85),),
        model_name="yolo11n-seg.pt",
    )
    association = TrackSemanticAssociation(
        7, "matched", 20, 0, 18, 0.9, 0.8, True, (20, 20)
    )
    return frame, association, people


def estimate(frame, association, people, timestamp=2.0):
    return estimate_visual_human_mass_prior(
        frame,
        association,
        people,
        track_id=7,
        timestamp_seconds=timestamp,
        config=HumanMassPriorConfig("selected adult-human restaurant demonstration"),
    )


def test_visual_person_selects_configured_prior_without_relabeling_sam_or_measuring_mass():
    frame, association, people = inputs()
    result = estimate(frame, association, people)
    assert result.status is EstimateStatus.PRIOR
    assert (
        result.value,
        result.lower_bound,
        result.upper_bound,
        result.confidence,
    ) == (70, 45, 100, 0.5)
    assert (
        result.timestamp_seconds == 2
        and frame.detections[0].semantic_class == "unknown"
    )
    assert "not measured mass" in result.evidence[1]


@pytest.mark.parametrize(
    "failure",
    [
        "unmatched",
        "wrong_track",
        "frame",
        "time",
        "shape",
        "stale",
        "future",
        "non_person",
        "weak",
        "ambiguous",
        "boundary",
        "wrong_source",
    ],
)
def test_unavailable_person_identity_cannot_supply_a_human_mass_prior(failure):
    frame, association, people = inputs()
    timestamp = 2.0
    if failure == "unmatched":
        association = replace(association, status="ambiguous")
    elif failure == "wrong_track":
        association = replace(association, track_id=8)
    elif failure == "frame":
        people = replace(people, source_frame_id=5)
    elif failure == "time":
        people = replace(people, source_timestamp_seconds=2.1)
    elif failure == "shape":
        m = np.zeros((41, 40), bool)
        m[5:35, 10:30] = True
        people = replace(
            people, image_height=41, detections=(replace(people.detections[0], mask=m),)
        )
    elif failure == "stale":
        timestamp = 2.2
    elif failure == "future":
        timestamp = 1.9
    elif failure == "non_person":
        people = replace(
            people, detections=(replace(people.detections[0], semantic_class="chair"),)
        )
    elif failure == "weak":
        people = replace(
            people, detections=(replace(people.detections[0], confidence=0.49),)
        )
    elif failure == "ambiguous":
        people = replace(people, detections=people.detections * 2)
    elif failure == "boundary":
        m = people.detections[0].mask.copy()
        m[0, 20] = True
        people = replace(people, detections=(replace(people.detections[0], mask=m),))
    elif failure == "wrong_source":
        frame = replace(frame, model_name="authored_actor_label")
    result = estimate(frame, association, people, timestamp)
    assert result.status is EstimateStatus.UNAVAILABLE and result.value is None


def test_prior_interval_is_configurable_and_never_promoted_to_measurement():
    frame, association, people = inputs()
    result = estimate_visual_human_mass_prior(
        frame,
        association,
        people,
        track_id=7,
        timestamp_seconds=2.0,
        config=HumanMassPriorConfig(
            "explicit engineering assumption", mass_kg=60, lower_kg=40, upper_kg=90
        ),
    )
    assert result.value == 60 and result.status is EstimateStatus.PRIOR
    with pytest.raises(ValueError):
        HumanMassPriorConfig("", mass_kg=60)
    with pytest.raises(ValueError):
        HumanMassPriorConfig("scope", mass_kg=1)
