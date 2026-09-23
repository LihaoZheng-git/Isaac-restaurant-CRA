"""Run the bounded ABO visual-mass baseline on one fused sensor track."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image
import torch
from torch import Tensor

from robot.perception.physical_attributes import (
    EstimateStatus,
    GeometryAttributes,
    MassAttributes,
    ScalarEstimate,
)
from robot.perception.semantic_detection import SemanticDetection
from robot.perception.visual_mass_model import (
    load_visual_mass_checkpoint,
    visual_mass_transform,
)


VISUAL_MASS_RUNTIME_SOURCE = "abo_mobilenet_rgb_lidar_dimensions"
ABO_SUPPORTED_CLASSES = (
    "bottle",
    "bowl",
    "chair",
    "cup",
    "knife",
    "pan",
    "plate",
    "stool",
    "table",
)


class VisualMassBackend(Protocol):
    def predict_mass_kg(self, images: Tensor, dimensions_m: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class VisualMassRuntimeConfig:
    checkpoint_path: Path
    device: str = "cpu"
    validation_mae_kg: float = 4.54224967956543
    domain_confidence_cap: float = 0.25
    supported_classes: tuple[str, ...] = ABO_SUPPORTED_CLASSES

    def __post_init__(self) -> None:
        path = Path(self.checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"visual mass checkpoint not found: {path}")
        mae = float(self.validation_mae_kg)
        cap = float(self.domain_confidence_cap)
        if not math.isfinite(mae) or mae <= 0:
            raise ValueError("validation_mae_kg must be positive and finite")
        if not math.isfinite(cap) or not 0 <= cap <= 1:
            raise ValueError("domain_confidence_cap must lie in [0, 1]")
        classes = tuple(sorted(set(self.supported_classes)))
        if not classes or any(not isinstance(item, str) or not item for item in classes):
            raise ValueError("supported_classes must contain nonempty class names")
        device = str(self.device).strip()
        if device.isdigit():
            device = f"cuda:{device}"
        try:
            torch.device(device)
        except RuntimeError as error:
            raise ValueError(f"invalid PyTorch device: {device}") from error
        object.__setattr__(self, "checkpoint_path", path)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "validation_mae_kg", mae)
        object.__setattr__(self, "domain_confidence_cap", cap)
        object.__setattr__(self, "supported_classes", classes)


def unavailable_mass(timestamp_seconds: float, reason: str) -> MassAttributes:
    return MassAttributes(
        ScalarEstimate.unavailable(
            unit="kg",
            source=VISUAL_MASS_RUNTIME_SOURCE,
            timestamp_seconds=timestamp_seconds,
            evidence=(reason,),
        )
    )


def mass_attributes_from_prediction(
    predicted_mass_kg: float,
    *,
    input_confidence: float,
    timestamp_seconds: float,
    config: VisualMassRuntimeConfig,
    checkpoint_sha256: str,
) -> MassAttributes:
    mass = float(predicted_mass_kg)
    confidence = min(float(input_confidence), config.domain_confidence_cap)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("predicted mass must be positive and finite")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("input confidence must lie in [0, 1]")
    error = config.validation_mae_kg
    return MassAttributes(
        ScalarEstimate(
            mass,
            max(0.001, mass - error),
            mass + error,
            confidence,
            "kg",
            VISUAL_MASS_RUNTIME_SOURCE,
            timestamp_seconds,
            EstimateStatus.INFERRED,
            (
                f"checkpoint_sha256={checkpoint_sha256}",
                f"catalog_validation_mae_kg={error}",
                "RGB instance crop plus LiDAR-derived dimensions",
                "ABO catalog domain; Isaac/restaurant mass is not calibrated",
            ),
        )
    )


class VisualMassRuntimeEstimator:
    """Checkpoint-backed estimator with explicit domain confidence cap."""

    def __init__(
        self,
        config: VisualMassRuntimeConfig,
        *,
        backend: VisualMassBackend | None = None,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        if not isinstance(config, VisualMassRuntimeConfig):
            raise TypeError("config must be VisualMassRuntimeConfig")
        self.config = config
        self.device = torch.device(config.device)
        if backend is None:
            backend, _ = load_visual_mass_checkpoint(
                config.checkpoint_path, map_location=self.device
            )
        self.backend = backend
        self.transform = transform or visual_mass_transform(training=False)
        self.checkpoint_sha256 = hashlib.sha256(
            config.checkpoint_path.read_bytes()
        ).hexdigest()

    def estimate(
        self,
        image_rgb: np.ndarray,
        detection: SemanticDetection,
        geometry: GeometryAttributes,
        *,
        timestamp_seconds: float,
    ) -> MassAttributes:
        timestamp = float(timestamp_seconds)
        if detection.semantic_class not in self.config.supported_classes:
            return unavailable_mass(
                timestamp, f"unsupported semantic class={detection.semantic_class}"
            )
        dimensions = (geometry.width_m, geometry.depth_m, geometry.height_m)
        if any(not item.is_available for item in dimensions):
            return unavailable_mass(timestamp, "one or more dimensions unavailable")
        values = tuple(float(item.value) for item in dimensions)
        if any(value <= 0 for value in values):
            return unavailable_mass(timestamp, "one or more dimensions nonpositive")

        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("image_rgb must have shape (height, width, 3 or 4)")
        if detection.mask.shape != image.shape[:2]:
            raise ValueError("detection mask must align with image_rgb")
        box = detection.bounding_box
        x0, y0 = int(math.floor(box.x_min)), int(math.floor(box.y_min))
        x1, y1 = int(math.ceil(box.x_max)), int(math.ceil(box.y_max))
        crop = np.array(image[y0:y1, x0:x1, :3], dtype=np.uint8, copy=True)
        crop_mask = detection.mask[y0:y1, x0:x1]
        if crop.size == 0 or not np.any(crop_mask):
            return unavailable_mass(timestamp, "empty instance crop")
        crop[~crop_mask] = 127
        pixels = self.transform(Image.fromarray(crop, mode="RGB")).unsqueeze(0)
        dimensions_tensor = torch.tensor(
            values, dtype=torch.float32
        ).unsqueeze(0)
        pixels = pixels.to(self.device)
        dimensions_tensor = dimensions_tensor.to(self.device)
        with torch.no_grad():
            prediction = self.backend.predict_mass_kg(pixels, dimensions_tensor)
        confidence = min(
            detection.confidence, *(item.confidence for item in dimensions)
        )
        return mass_attributes_from_prediction(
            float(prediction.reshape(-1)[0]),
            input_confidence=confidence,
            timestamp_seconds=timestamp,
            config=self.config,
            checkpoint_sha256=self.checkpoint_sha256,
        )
