from dataclasses import replace

import numpy as np
import pytest

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.human_mass_prior import (
    HumanMassPriorConfig,
    estimate_visual_human_mass_prior,
    catalog_person_overlap_guard,
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


def test_small_contained_fragment_blocks_catalog_but_cannot_select_adult_mass():
    frame, association, people = inputs()
    mask = np.zeros((40, 40), bool)
    mask[10:20, 15:25] = True
    frame = replace(frame, detections=(replace(frame.detections[0], mask=mask),))
    result = catalog_person_overlap_guard(frame, association, people, track_id=7,
        timestamp_seconds=2., config=HumanMassPriorConfig("adult fixture"))
    assert result["blocked"]
    assert result["person_hits"][0]["instance_containment"] == 1.
    assert result["source_timestamp_seconds"] == 2.
    assert not estimate(frame, association, people).is_available


@pytest.mark.parametrize("problem", ["stale", "future", "wrong_frame", "wrong_source", "weak", "no_overlap", "unmatched"])
def test_domain_rejection_requires_current_positive_person_pixels(problem):
    frame, association, people = inputs()
    now = 2.
    if problem == "stale":
        now = 2.2
    elif problem == "future":
        now = 1.9
    elif problem == "wrong_frame":
        people = replace(people, source_frame_id=9)
    elif problem == "wrong_source":
        frame = replace(frame, model_name="authored")
    elif problem == "weak":
        people = replace(people, detections=(replace(people.detections[0], confidence=.4),))
    elif problem == "no_overlap":
        people = replace(people, detections=(replace(people.detections[0], mask=np.zeros((40,40), bool)),))
    elif problem == "unmatched":
        association = replace(association, status="ambiguous")
    assert not catalog_person_overlap_guard(frame, association, people, track_id=7,
        timestamp_seconds=now, config=HumanMassPriorConfig("adult fixture"))["blocked"]


def test_truncated_or_ambiguous_positive_person_pixels_only_reject_domain():
    frame, association, people = inputs()
    mask = people.detections[0].mask.copy()
    mask[0, 20] = True
    people = replace(people, detections=(replace(people.detections[0], mask=mask),)*2)
    assert catalog_person_overlap_guard(frame, association, people, track_id=7,
        timestamp_seconds=2., config=HumanMassPriorConfig("adult fixture"))["blocked"]
    assert not estimate(frame, association, people).is_available
