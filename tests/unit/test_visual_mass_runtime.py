from pathlib import Path

import numpy as np
import pytest
import torch

from robot.perception.physical_attributes import (
    EstimateStatus,
    GeometryAttributes,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection
from robot.perception.visual_mass_runtime import (
    VisualMassRuntimeConfig,
    VisualMassRuntimeEstimator,
    mass_attributes_from_prediction,
)


def scalar(
    value,
    confidence=0.8,
    *,
    timestamp=2,
    status=EstimateStatus.INFERRED,
):
    return ScalarEstimate(
        value,
        value * 0.9,
        value * 1.1,
        confidence,
        "m",
        "lidar",
        timestamp,
        status,
    )


def geometry():
    unavailable = ScalarEstimate.unavailable(unit="m", source="none", timestamp_seconds=2)
    return GeometryAttributes(
        scalar(0.5), scalar(0.4), scalar(0.8), unavailable, unavailable
    )


def geometry_with(*, timestamp=2, status=EstimateStatus.INFERRED):
    unavailable = ScalarEstimate.unavailable(
        unit="m", source="none", timestamp_seconds=timestamp
    )
    return GeometryAttributes(
        scalar(0.5, timestamp=timestamp, status=status),
        scalar(0.4, timestamp=timestamp, status=status),
        scalar(0.8, timestamp=timestamp, status=status),
        unavailable,
        unavailable,
    )


def config(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    return VisualMassRuntimeConfig(checkpoint, validation_mae_kg=2, domain_confidence_cap=0.25)


class Backend:
    def predict_mass_kg(self, images, dimensions_m):
        assert images.shape == (1, 3, 4, 4)
        np.testing.assert_allclose(dimensions_m.cpu().numpy(), [[0.5, 0.4, 0.8]])
        return torch.tensor([5.0])


def transform(_image):
    return torch.ones((3, 4, 4))


def detection(name="chair"):
    mask = np.zeros((10, 12), dtype=np.bool_)
    mask[2:8, 3:10] = True
    return SemanticDetection(
        0, name, 0.9, PixelBoundingBox(3, 2, 10, 8), mask
    )


def test_prediction_is_inferred_and_domain_confidence_is_capped(tmp_path):
    result = mass_attributes_from_prediction(
        5,
        input_confidence=0.8,
        timestamp_seconds=2,
        config=config(tmp_path),
        checkpoint_sha256="abc",
    ).mass_kg
    assert result.status is EstimateStatus.INFERRED
    assert result.value == 5
    assert result.lower_bound == 3
    assert result.upper_bound == 7
    assert result.confidence == 0.25
    assert "not calibrated" in result.evidence[-1]


def test_ultralytics_numeric_device_is_normalized_for_pytorch(tmp_path):
    settings = VisualMassRuntimeConfig(
        config(tmp_path).checkpoint_path, device="0"
    )
    assert settings.device == "cuda:0"


def test_geometry_age_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="maximum_geometry_age_seconds"):
        VisualMassRuntimeConfig(
            config(tmp_path).checkpoint_path,
            maximum_geometry_age_seconds=0,
        )


def test_estimator_uses_masked_crop_and_lidar_dimensions(tmp_path):
    estimator = VisualMassRuntimeEstimator(
        config(tmp_path), backend=Backend(), transform=transform
    )
    image = np.full((10, 12, 3), 255, dtype=np.uint8)
    result = estimator.estimate(image, detection(), geometry(), timestamp_seconds=2)
    assert result.mass_kg.value == 5
    assert result.mass_kg.source.endswith("lidar_dimensions")


def test_unsupported_class_and_missing_dimensions_stay_unavailable(tmp_path):
    estimator = VisualMassRuntimeEstimator(
        config(tmp_path), backend=Backend(), transform=transform
    )
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    assert not estimator.estimate(
        image, detection("person"), geometry(), timestamp_seconds=2
    ).mass_kg.is_available
    missing = PhysicalAttributes.unknown(
        track_id=1, timestamp_seconds=2
    ).geometry
    assert not estimator.estimate(
        image, detection(), missing, timestamp_seconds=2
    ).mass_kg.is_available


def test_visual_mass_preserves_non_sensor_geometry_dependency(tmp_path):
    estimator = VisualMassRuntimeEstimator(
        config(tmp_path), backend=Backend(), transform=transform
    )
    image = np.zeros((10, 12, 3), dtype=np.uint8)

    result = estimator.estimate(
        image,
        detection(),
        geometry_with(status=EstimateStatus.ORACLE),
        timestamp_seconds=2,
    )

    assert result.mass_kg.status is EstimateStatus.ORACLE
    assert any("geometry_width: status=oracle" in item for item in result.mass_kg.evidence)


def test_visual_mass_rejects_stale_and_future_geometry(tmp_path):
    estimator = VisualMassRuntimeEstimator(
        config(tmp_path), backend=Backend(), transform=transform
    )
    image = np.zeros((10, 12, 3), dtype=np.uint8)

    stale = estimator.estimate(
        image, detection(), geometry_with(timestamp=1), timestamp_seconds=2
    )
    future = estimator.estimate(
        image, detection(), geometry_with(timestamp=3), timestamp_seconds=2
    )

    assert not stale.mass_kg.is_available
    assert "stale" in stale.mass_kg.evidence[0]
    assert not future.mass_kg.is_available
    assert "future" in future.mass_kg.evidence[0]
