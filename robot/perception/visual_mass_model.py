"""Lightweight image-and-dimensions visual mass regression contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from PIL import Image
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2

from robot.perception.abo_visual_mass import AboMassCandidate


VISUAL_MASS_MODEL_SCHEMA_VERSION = 1
VISUAL_MASS_ARCHITECTURE = "mobilenet_v3_small_rgb_log_dimensions"
IMAGENET_WEIGHTS_NAME = "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
IMAGENET_WEIGHTS_URL = MobileNet_V3_Small_Weights.IMAGENET1K_V1.url

_NORMALIZE = v2.Normalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)


def visual_mass_transform(*, training: bool) -> v2.Compose:
    """Return documented MobileNet preprocessing with bounded augmentation."""
    geometry = (
        v2.RandomResizedCrop((224, 224), scale=(0.8, 1.0), antialias=True)
        if training
        else v2.Compose(
            (v2.Resize(256, antialias=True), v2.CenterCrop((224, 224)))
        )
    )
    operations = [geometry]
    if training:
        operations.extend((v2.RandomHorizontalFlip(), v2.ColorJitter(0.1, 0.1, 0.1)))
    operations.extend((v2.ToImage(), v2.ToDtype(torch.float32, scale=True), _NORMALIZE))
    return v2.Compose(operations)


def log_dimension_statistics(
    candidates: Sequence[AboMassCandidate],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute train-only normalization values for positive SI dimensions."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    values = torch.tensor(
        [
            [math.log(item.width_m), math.log(item.depth_m), math.log(item.height_m)]
            for item in candidates
        ],
        dtype=torch.float32,
    )
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    return tuple(mean.tolist()), tuple(std.tolist())


class VisualMassRegressor(nn.Module):
    """MobileNetV3 image encoder fused with measured/inferred dimensions."""

    def __init__(
        self,
        dimension_mean: Sequence[float],
        dimension_std: Sequence[float],
        *,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(dimension_mean, dtype=torch.float32)
        std = torch.as_tensor(dimension_std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or torch.any(std <= 0):
            raise ValueError("dimension statistics must be three values with positive std")
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        feature_count = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        if freeze_backbone:
            for parameter in backbone.parameters():
                parameter.requires_grad = False
        self.backbone = backbone
        self.backbone_frozen = freeze_backbone
        self.regressor = nn.Sequential(
            nn.Linear(feature_count + 3, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )
        self.register_buffer("dimension_mean", mean)
        self.register_buffer("dimension_std", std)

    def train(self, mode: bool = True) -> VisualMassRegressor:
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, images: Tensor, dimensions_m: Tensor) -> Tensor:
        if dimensions_m.ndim != 2 or dimensions_m.shape[1] != 3:
            raise ValueError("dimensions_m must have shape (batch, 3)")
        if torch.any(dimensions_m <= 0):
            raise ValueError("dimensions_m must be positive")
        features = self.backbone(images)
        normalized = (torch.log(dimensions_m) - self.dimension_mean) / self.dimension_std
        return self.regressor(torch.cat((features, normalized), dim=1)).squeeze(1)

    def predict_mass_kg(self, images: Tensor, dimensions_m: Tensor) -> Tensor:
        """Return positive kilograms from the model's log-mass output."""
        return torch.exp(self(images, dimensions_m))


class AboVisualMassDataset(Dataset):
    """Load selected ABO images without exposing catalog labels to runtime."""

    def __init__(
        self,
        candidates: Sequence[AboMassCandidate],
        image_root: Path,
        *,
        training: bool,
    ) -> None:
        if not candidates:
            raise ValueError("dataset cannot be empty")
        self.candidates = tuple(candidates)
        self.image_root = Path(image_root)
        self.transform = visual_mass_transform(training=training)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, str, str]:
        item = self.candidates[index]
        path = self.image_root / item.image_path
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        dimensions = torch.tensor(
            (item.width_m, item.depth_m, item.height_m), dtype=torch.float32
        )
        log_mass = torch.tensor(math.log(item.mass_kg), dtype=torch.float32)
        return pixels, dimensions, log_mass, item.object_id, item.semantic_class


@dataclass(frozen=True)
class MassRegressionMetrics:
    count: int
    mean_absolute_error_kg: float
    median_absolute_error_kg: float
    mean_absolute_percentage_error: float


def mass_regression_metrics(
    target_log_mass: Tensor, predicted_log_mass: Tensor
) -> MassRegressionMetrics:
    if target_log_mass.shape != predicted_log_mass.shape or target_log_mass.numel() == 0:
        raise ValueError("target and prediction tensors must have the same nonzero shape")
    target = target_log_mass.detach().float().exp()
    predicted = predicted_log_mass.detach().float().exp()
    errors = (target - predicted).abs()
    return MassRegressionMetrics(
        count=target.numel(),
        mean_absolute_error_kg=float(errors.mean()),
        median_absolute_error_kg=float(torch.quantile(errors, 0.5)),
        mean_absolute_percentage_error=float((errors / target).mean()),
    )


def load_visual_mass_checkpoint(
    checkpoint_path: Path, *, map_location: str | torch.device = "cpu"
) -> tuple[VisualMassRegressor, dict[str, object]]:
    """Load a validated state-dict checkpoint without downloading weights."""
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("visual mass checkpoint must be a mapping")
    if payload.get("schema_version") != VISUAL_MASS_MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported visual mass checkpoint schema")
    if payload.get("architecture") != VISUAL_MASS_ARCHITECTURE:
        raise ValueError("unsupported visual mass architecture")
    model = VisualMassRegressor(
        payload["dimension_mean"],
        payload["dimension_std"],
        pretrained=False,
        freeze_backbone=bool(payload.get("backbone_frozen", True)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(map_location)
    model.eval()
    return model, payload


def fit_log_dimension_baseline(candidates: Sequence[AboMassCandidate]) -> Tensor:
    """Fit log(mass) from a bias and log dimensions with least squares."""
    if len(candidates) < 4:
        raise ValueError("at least four candidates are required")
    dimensions = torch.tensor(
        [
            [math.log(item.width_m), math.log(item.depth_m), math.log(item.height_m)]
            for item in candidates
        ],
        dtype=torch.float64,
    )
    design = torch.cat((torch.ones((len(candidates), 1), dtype=torch.float64), dimensions), dim=1)
    target = torch.tensor(
        [math.log(item.mass_kg) for item in candidates], dtype=torch.float64
    )
    return torch.linalg.lstsq(design, target).solution.float()


def predict_log_dimension_baseline(
    candidates: Sequence[AboMassCandidate], coefficients: Tensor
) -> Tensor:
    if coefficients.shape != (4,):
        raise ValueError("coefficients must contain bias plus three dimension terms")
    dimensions = torch.tensor(
        [
            [math.log(item.width_m), math.log(item.depth_m), math.log(item.height_m)]
            for item in candidates
        ],
        dtype=torch.float32,
    )
    design = torch.cat((torch.ones((len(candidates), 1)), dimensions), dim=1)
    return design @ coefficients
