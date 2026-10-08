from dataclasses import replace

import numpy as np
import pytest

from robot.evaluation.cra_experiments import EvidenceMode, RestaurantScenario
from robot.evaluation.perception_readiness import (
    EvaluationScope,
    evaluate_perception_readiness,
)
from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.object_plume_evidence import (
    OBJECT_PLUME_BASIS,
    ObjectPlumeConfig,
    ObjectPlumeObservation,
    estimate_object_plume_evidence,
)
from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)
from robot.perception.specialist_track_evidence import SpecialistAssociationFrame
from robot.risk.physical_factor_adapter import (
    PhysicalFactorAdapterConfig,
    adapt_physical_factors,
)
from robot.risk.physical_profile_provider import PhysicalAttributeCRARiskProfileProvider


REFERENCE = np.zeros((180, 260, 3), np.uint8)


def observation(
    index,
    kind="none",
    boxes=((90, 110, 130, 150),),
    confidence=0.9,
    semantic_class="object",
    status=EstimateStatus.INFERRED,
):
    image = REFERENCE.copy()
    if kind == "small":
        image[100:110, 106:110] = 140
    if kind == "dense":
        image[60:110, 98:122] = 140
    if kind == "remote":
        image[60:110, 12:22] = 140
    if kind == "global":
        image[:] = 80
    timestamp = 1.1 + index * 0.1
    detections, links = [], []
    for track, coords in enumerate(boxes, 1):
        box = PixelBoundingBox(*coords)
        mask = np.zeros(image.shape[:2], bool)
        mask[int(box.y_min) : int(box.y_max), int(box.x_min) : int(box.x_max)] = True
        detections.append(
            SemanticDetection(track, semantic_class, confidence, box, mask)
        )
        links.append(
            TrackSemanticAssociation(
                track, "matched", 100, track - 1, 100, 1.0, 1.0, True, box.center_xy
            )
        )
    return ObjectPlumeObservation(
        image,
        SemanticDetectionFrame(
            index, timestamp, 180, 260, tuple(detections), "fixture", 0.0
        ),
        SpecialistAssociationFrame(index, timestamp, 180, 260, tuple(links)),
        timestamp,
        True,
        status,
    )


def estimate(kinds=("none",) * 3, **kwargs):
    observations = tuple(observation(i, kind) for i, kind in enumerate(kinds))
    return estimate_object_plume_evidence(
        REFERENCE,
        reference_timestamp_seconds=1.0,
        observations=observations,
        timestamp_seconds=1.3,
        clean_reference_validated=kwargs.pop("clean_reference_validated", True),
        negative_validation_scope=kwargs.pop(
            "negative_validation_scope", "controlled test domain"
        ),
        **kwargs,
    )


def test_no_plume_in_valid_window_is_low_continuous_cue_not_cold_class():
    result = estimate()[1]
    cue = result.thermal.heat_source_likelihood
    assert cue.is_available and cue.value == 0
    assert 0 < cue.upper_bound < 0.05
    assert 0 < cue.confidence < 1
    assert cue.status is EstimateStatus.INFERRED
    assert result.positive_frames == 0 and result.invoked_frames == 3
    assert not result.thermal.steam_likelihood.is_available
    assert not result.thermal.open_container_likelihood.is_available
    assert "not establish coldness" in " ".join(cue.evidence)


def test_continuous_strength_persistence_ordering_and_unrelated_plume():
    low, intermittent, dense, remote = [
        estimate(kinds)[1].thermal.heat_source_likelihood
        for kinds in (
            ("none",) * 3,
            ("small", "none", "small"),
            ("dense",) * 3,
            ("remote",) * 3,
        )
    ]
    assert low.value < intermittent.value < dense.value
    assert remote.value == low.value
    assert low.upper_bound < dense.lower_bound


def test_object_names_never_change_scores_or_infer_hot_soup():
    values = []
    for name in ("sports_ball", "bowl", "nail", "unknown"):
        frames = tuple(observation(i, "dense", semantic_class=name) for i in range(3))
        result = estimate_object_plume_evidence(
            REFERENCE,
            reference_timestamp_seconds=1.0,
            observations=frames,
            timestamp_seconds=1.3,
            clean_reference_validated=True,
        )
        values.append(result[1].thermal.heat_source_likelihood.value)
    assert len(set(values)) == 1


@pytest.mark.parametrize(
    "kwargs",
    [dict(clean_reference_validated=False), dict(negative_validation_scope=None)],
)
def test_missing_negative_validation_or_clean_reference_stays_unknown(kwargs):
    result = estimate(**kwargs)[1]
    assert not result.thermal.heat_source_likelihood.is_available
    assert result.rejection_reasons


def test_positive_only_window_does_not_require_empty_mask_validation():
    assert estimate(("dense",) * 3, negative_validation_scope=None)[
        1
    ].thermal.heat_source_likelihood.is_available
    assert not estimate(("dense", "none", "dense"), negative_validation_scope=None)[
        1
    ].thermal.heat_source_likelihood.is_available


@pytest.mark.parametrize(
    "case",
    [
        "moving_camera",
        "stale_track",
        "future_track",
        "ambiguous",
        "truncated",
        "small_region",
        "low_confidence",
        "source_moves",
        "clipped_air",
    ],
)
def test_invalid_object_observations_remain_unknown(case):
    frames = [observation(i) for i in range(3)]
    for i, item in enumerate(frames):
        if case == "moving_camera":
            item = replace(item, camera_stationary=False)
        elif case == "stale_track":
            item = replace(item, track_timestamp_seconds=0.1)
        elif case == "future_track":
            item = replace(item, track_timestamp_seconds=2.0)
        elif case == "ambiguous":
            item = replace(
                item,
                associations=replace(
                    item.associations,
                    associations=(
                        replace(item.associations.associations[0], status="ambiguous"),
                    ),
                ),
            )
        elif case in {
            "truncated",
            "small_region",
            "low_confidence",
            "source_moves",
            "clipped_air",
        }:
            coords = {
                "truncated": (0, 110, 40, 150),
                "small_region": (90, 110, 95, 115),
                "source_moves": (90 + i * 10, 110, 130 + i * 10, 150),
                "clipped_air": (90, 5, 130, 45),
            }
            item = observation(
                i,
                boxes=(coords.get(case, (90, 110, 130, 150)),),
                confidence=0.1 if case == "low_confidence" else 0.9,
            )
        frames[i] = item
    result = estimate_object_plume_evidence(
        REFERENCE,
        reference_timestamp_seconds=1.0,
        observations=tuple(frames),
        timestamp_seconds=1.3,
        clean_reference_validated=True,
        negative_validation_scope="controlled",
    )[1]
    assert not result.thermal.heat_source_likelihood.is_available
    assert result.rejection_reasons


def test_global_reference_change_rejected():
    assert not estimate(("global",) * 3)[1].thermal.heat_source_likelihood.is_available


def test_every_track_gets_a_record_and_shared_positive_plume_is_unknown():
    frames = tuple(
        observation(i, "dense", boxes=((90, 110, 130, 150), (96, 110, 136, 150)))
        for i in range(3)
    )
    result = estimate_object_plume_evidence(
        REFERENCE,
        reference_timestamp_seconds=1.0,
        observations=frames,
        timestamp_seconds=1.3,
        clean_reference_validated=True,
        negative_validation_scope="controlled",
    )
    assert set(result) == {1, 2}
    assert all("shared" in " ".join(item.rejection_reasons) for item in result.values())
    assert all(
        not item.thermal.heat_source_likelihood.is_available for item in result.values()
    )


def test_large_plume_on_other_source_does_not_invalidate_clean_object_background():
    frames = []
    for i in range(3):
        item = observation(i, boxes=((30, 110, 70, 150), (190, 110, 230, 150)))
        image = item.rgb.copy()
        image[55:110, 180:240] = 200
        frames.append(replace(item, rgb=image))
    result = estimate_object_plume_evidence(
        REFERENCE,
        reference_timestamp_seconds=1.0,
        observations=tuple(frames),
        timestamp_seconds=1.3,
        clean_reference_validated=True,
        negative_validation_scope="controlled",
    )
    assert result[1].thermal.heat_source_likelihood.value == 0
    assert result[2].thermal.heat_source_likelihood.value > 0.5
    global_frames = tuple(
        replace(item, rgb=np.full_like(item.rgb, 80)) for item in frames
    )
    invalid = estimate_object_plume_evidence(
        REFERENCE,
        reference_timestamp_seconds=1.0,
        observations=global_frames,
        timestamp_seconds=1.3,
        clean_reference_validated=True,
        negative_validation_scope="controlled",
    )
    assert all(
        not item.thermal.heat_source_likelihood.is_available
        for item in invalid.values()
    )


def test_distinct_objects_do_not_share_a_plume_score():
    frames = tuple(
        observation(i, "dense", boxes=((90, 110, 130, 150), (200, 110, 240, 150)))
        for i in range(3)
    )
    result = estimate_object_plume_evidence(
        REFERENCE,
        reference_timestamp_seconds=1.0,
        observations=frames,
        timestamp_seconds=1.3,
        clean_reference_validated=True,
        negative_validation_scope="controlled",
    )
    assert (
        result[1].thermal.heat_source_likelihood.value
        > result[2].thermal.heat_source_likelihood.value
    )


@pytest.mark.parametrize(
    "case", ["duplicate", "future", "stale", "reference_future", "too_short", "too_few"]
)
def test_bad_temporal_windows_rejected(case):
    frames = tuple(observation(i) for i in range(3))
    timestamp, reference = 1.3, 1.0
    if case == "duplicate":
        frames = (frames[0], frames[0], frames[2])
    if case == "future":
        timestamp = 1.2
    if case == "stale":
        timestamp = 2.0
    if case == "reference_future":
        reference = 1.2
    if case == "too_short":
        frames = frames[:2]
    if case == "too_few":
        frames = frames[:1]
    result = estimate_object_plume_evidence(
        REFERENCE,
        reference_timestamp_seconds=reference,
        observations=frames,
        timestamp_seconds=timestamp,
        clean_reference_validated=True,
        negative_validation_scope="controlled",
    )
    assert not result[1].thermal.heat_source_likelihood.is_available


def test_prior_regions_retain_prior_and_explicit_adapter_basis():
    frames = tuple(observation(i, status=EstimateStatus.PRIOR) for i in range(3))
    thermal = estimate_object_plume_evidence(
        REFERENCE,
        reference_timestamp_seconds=1.0,
        observations=frames,
        timestamp_seconds=1.3,
        clean_reference_validated=True,
        negative_validation_scope="controlled",
    )[1].thermal
    assert thermal.heat_source_likelihood.status is EstimateStatus.PRIOR
    base = PhysicalAttributes.unknown(track_id=1, timestamp_seconds=1.3)
    attributes = replace(base, thermal=thermal)
    route = ScalarEstimate(
        0.8, 0.7, 0.9, 0.9, "1", "route", 1.3, EstimateStatus.INFERRED
    )
    with pytest.raises(ValueError, match="opt-in"):
        adapt_physical_factors(
            attributes, timestamp_seconds=1.3, thermal_basis=OBJECT_PLUME_BASIS
        )
    provider = PhysicalAttributeCRARiskProfileProvider(
        {1: attributes},
        timestamp_seconds=1.3,
        thermal_interactions={1: route},
        thermal_bases={1: OBJECT_PLUME_BASIS},
        config=PhysicalFactorAdapterConfig(allow_visual_thermal_proxy=True),
    )
    result = provider.diagnostics_by_track_id[1]
    assert (
        result.thermal.basis == OBJECT_PLUME_BASIS
        and not result.thermal.rejection_reasons
    )
    prior = RestaurantScenario("plume", "thermal", "proxy", EvidenceMode.PRIOR, 0)
    sensor = replace(prior, evidence_mode=EvidenceMode.SENSOR)
    assert evaluate_perception_readiness(
        prior, result, scope=EvaluationScope.FACTOR_SPECIFIC
    ).ready
    assert not evaluate_perception_readiness(
        sensor, result, scope=EvaluationScope.FACTOR_SPECIFIC
    ).ready
    with pytest.raises(ValueError, match="opt-in"):
        evaluate_perception_readiness(
            prior, replace(result, config=PhysicalFactorAdapterConfig())
        )


def test_input_frames_must_align_and_invalid_config_rejected():
    item = observation(0)
    with pytest.raises(ValueError, match="align"):
        replace(item, associations=replace(item.associations, source_frame_id=99))
    with pytest.raises(ValueError, match="three"):
        ObjectPlumeConfig(minimum_frames=2)
