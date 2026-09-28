import pytest

from robot.perception.physical_attributes import (
    EstimateStatus,
    MassAttributes,
    ScalarEstimate,
)
from robot.perception.visual_mass_dataset import VisualMassSample
from robot.perception.visual_mass_evaluation import (
    VisualMassDomainEvaluationConfig,
    VisualMassDomainPrediction,
    evaluate_visual_mass_isaac_domain,
)
from robot.perception.visual_mass_runtime import VISUAL_MASS_RUNTIME_SOURCE


def sample(identifier, object_id, semantic_class, mass, split="test"):
    return VisualMassSample(
        identifier,
        object_id,
        semantic_class,
        "isaac-restaurant",
        split,
        f"rgb/{identifier}.npy",
        mass,
        "isaac_authored_mass",
        0.5,
        0.4,
        0.8,
        "front_hawk_left",
        1,
        mask_path=f"masks/{identifier}.npy",
    )


def prediction(identifier, value, *, digest="a" * 64):
    return VisualMassDomainPrediction(
        identifier,
        value,
        max(0.1, value - 1),
        value + 1,
        0.25,
        EstimateStatus.INFERRED,
        VISUAL_MASS_RUNTIME_SOURCE,
        digest,
    )


def test_repeated_views_are_averaged_per_independent_object():
    samples = (
        sample("chair-a-1", "chair-a", "chair", 5),
        sample("chair-a-2", "chair-a", "chair", 5),
        sample("bowl-b-1", "bowl-b", "bowl", 2),
    )
    predictions = (
        prediction("chair-a-1", 4),
        prediction("chair-a-2", 6),
        prediction("bowl-b-1", 3),
    )
    report = evaluate_visual_mass_isaac_domain(
        samples,
        predictions,
        config=VisualMassDomainEvaluationConfig(2, 1),
    )
    assert report.ready_for_sensor_domain_claim
    assert report.independent_object_count == 2
    assert report.sample_count == 3
    assert report.overall_metrics.mean_absolute_error_kg == pytest.approx(0.5)
    assert {item.semantic_class for item in report.per_class} == {"bowl", "chair"}


def test_missing_predictions_and_small_object_inventory_block_claim():
    samples = (
        sample("chair-a", "chair-a", "chair", 5),
        sample("chair-b", "chair-b", "chair", 6),
    )
    report = evaluate_visual_mass_isaac_domain(
        samples,
        (prediction("chair-a", 5),),
        config=VisualMassDomainEvaluationConfig(3, 2),
    )
    assert not report.ready_for_sensor_domain_claim
    assert any("missing predictions" in item for item in report.blockers)
    assert any("independent test objects" in item for item in report.blockers)


def test_multiple_checkpoints_block_one_evaluation_claim():
    samples = (
        sample("chair-a", "chair-a", "chair", 5),
        sample("bowl-b", "bowl-b", "bowl", 2),
    )
    report = evaluate_visual_mass_isaac_domain(
        samples,
        (
            prediction("chair-a", 5, digest="a" * 64),
            prediction("bowl-b", 2, digest="b" * 64),
        ),
        config=VisualMassDomainEvaluationConfig(2, 1),
    )
    assert not report.ready_for_sensor_domain_claim
    assert "predictions use multiple checkpoints" in report.blockers
    assert report.checkpoint_sha256 is None


def test_non_test_predictions_are_rejected_from_domain_report():
    samples = (sample("train-a", "a", "chair", 5, split="train"),)
    report = evaluate_visual_mass_isaac_domain(
        samples,
        (prediction("train-a", 5),),
        config=VisualMassDomainEvaluationConfig(1, 1),
    )
    assert not report.ready_for_sensor_domain_claim
    assert any("non-test" in item for item in report.blockers)
    assert report.overall_metrics is None


def test_prediction_can_be_built_only_from_available_inferred_runtime_mass():
    estimate = ScalarEstimate(
        4,
        3,
        5,
        0.25,
        "kg",
        VISUAL_MASS_RUNTIME_SOURCE,
        2,
        EstimateStatus.INFERRED,
    )
    result = VisualMassDomainPrediction.from_attributes(
        "sample",
        MassAttributes(estimate),
        checkpoint_sha256="a" * 64,
    )
    assert result.predicted_mass_kg == 4

    with pytest.raises(ValueError, match="available"):
        VisualMassDomainPrediction.from_attributes(
            "sample",
            MassAttributes(
                ScalarEstimate.unavailable(
                    unit="kg",
                    source=VISUAL_MASS_RUNTIME_SOURCE,
                    timestamp_seconds=2,
                )
            ),
            checkpoint_sha256="a" * 64,
        )
