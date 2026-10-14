"""Category-free SAM 2.1 masks prompted only by projected LiDAR returns.

The tracker still owns obstacle discovery and occupancy. SAM refines RGB
instances, with a model-quality gate and explicit rejection of ambiguous masks.
It supplies no class, material, density, physical sharpness, or temperature.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time

import numpy as np

from robot.perception.camera_lidar_fusion import ProjectedLidarPoints
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection


SAM_INSTANCE_SOURCE = "sam2_lidar_prompted_rgb_instance"
SAM_TINY_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"


@dataclass(frozen=True)
class SamInstanceConfig:
    checkpoint_path: Path
    device: str = "0"
    expected_sha256: str = SAM_TINY_SHA256
    minimum_projected_points: int = 8
    minimum_mask_pixels: int = 200
    maximum_image_fraction: float = 0.4
    minimum_model_quality: float = 0.85
    minimum_track_coverage: float = 0.2
    maximum_foreign_hit_fraction: float = 0.15
    equivalent_mask_iou: float = 0.85
    ambiguity_quality_gap: float = 0.05
    confidence_cap: float = 0.75

    def __post_init__(self):
        if not Path(self.checkpoint_path).is_file():
            raise FileNotFoundError("SAM checkpoint must be an existing local file")
        if not str(self.checkpoint_path).endswith("sam2.1_t.pt"):
            raise ValueError("this audited backend requires sam2.1_t.pt")
        for name in ("minimum_projected_points", "minimum_mask_pixels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "maximum_image_fraction",
            "minimum_model_quality",
            "minimum_track_coverage",
            "maximum_foreign_hit_fraction",
            "equivalent_mask_iou",
            "ambiguity_quality_gap",
            "confidence_cap",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")


@dataclass(frozen=True)
class SamInstanceResult:
    detection: SemanticDetection | None
    rejection_reasons: tuple[str, ...]
    prompt_xy: tuple[int, int] | None = None
    model_quality: float | None = None
    projected_track_points: int = 0
    mask_track_hits: int = 0
    inference_wall_seconds: float = 0.0
    image_encoder_wall_seconds: float = 0.0
    source: str = SAM_INSTANCE_SOURCE


class _UltralyticsBackend:
    def __init__(self, config):
        # Reuse installed packages. No installer or automatic asset fetch.
        import torch
        from ultralytics import SAM
        from ultralytics.models.sam import SAM2Predictor

        # Validate a tensor-only state dict before the installed model loader.
        state = torch.load(
            config.checkpoint_path, map_location="cpu", weights_only=True
        )
        if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
            raise ValueError("SAM release must contain a numeric model state dict")
        if not all(
            isinstance(value, torch.Tensor) for value in state["model"].values()
        ):
            raise ValueError("SAM state dict contains unsupported values")
        model = SAM(str(config.checkpoint_path))
        self.predictor = SAM2Predictor(
            overrides={
                "task": "segment",
                "mode": "predict",
                "device": config.device,
                "conf": 0.0,
                "verbose": False,
                "save": False,
                "imgsz": 1024,
            }
        )
        self.predictor.setup_model(model.model, verbose=False)
        self.image_bgr = None

    def set_image(self, rgb):
        import torch

        self.image_bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        self.predictor.reset_image()
        with torch.inference_mode():
            self.predictor.set_image(self.image_bgr)

    def predict(self, prompt_xy):
        result = self.predictor(
            self.image_bgr, points=[[prompt_xy]], labels=[[1]], multimask_output=True
        )[0]
        if result.masks is None:
            return np.empty((0, *self.image_bgr.shape[:2]), bool), np.empty(0)
        return (
            result.masks.data.cpu().numpy().astype(bool),
            result.boxes.conf.cpu().numpy(),
        )


class SamInstanceSegmenter:
    def __init__(self, config: SamInstanceConfig, *, backend=None):
        self.config = config
        digest = hashlib.sha256(Path(config.checkpoint_path).read_bytes()).hexdigest()
        if digest != config.expected_sha256:
            raise ValueError(
                "SAM checkpoint hash differs from configured audited release"
            )
        self.checkpoint_sha256 = digest
        self.backend = backend if backend is not None else _UltralyticsBackend(config)
        self.image_shape = None
        self.image_encoder_wall_seconds = 0.0

    def set_image(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("SAM image must be HxWx3 uint8 RGB")
        self.image_shape = rgb.shape[:2]
        start = time.perf_counter()
        self.backend.set_image(rgb)
        self.image_encoder_wall_seconds = time.perf_counter() - start

    def propose(
        self,
        point_track_ids,
        projection: ProjectedLidarPoints,
        *,
        track_id,
        prompt_source_index=None,
    ):
        if self.image_shape is None:
            raise RuntimeError("set_image is required for each camera frame")
        ids = np.asarray(point_track_ids)
        indices = projection.source_indices
        xy = projection.pixel_indices_xy
        height, width = self.image_shape
        if (
            ids.ndim != 1
            or (indices < 0).any()
            or (indices >= len(ids)).any()
            or (xy < 0).any()
            or (xy[:, 0] >= width).any()
            or (xy[:, 1] >= height).any()
        ):
            raise ValueError(
                "projected LiDAR indices must align with image/track arrays"
            )
        owned = ids[indices] == track_id
        points = xy[owned]
        count = len(points)
        if count < self.config.minimum_projected_points:
            return SamInstanceResult(
                None,
                ("insufficient projected track points",),
                projected_track_points=count,
            )
        # Use a real observed return nearest the robust image median. No authored
        # object location, class, color, mask, dimensions, or density is supplied.
        median = np.median(points, axis=0)
        prompt = tuple(
            int(v) for v in points[np.argmin(np.sum((points - median) ** 2, axis=1))]
        )
        if prompt_source_index is not None:
            if isinstance(prompt_source_index, bool) or not isinstance(
                prompt_source_index, (int, np.integer)
            ):
                raise ValueError("explicit prompt must name an observed owned return")
            matches = np.flatnonzero(owned & (indices == prompt_source_index))
            if len(matches) != 1:
                raise ValueError("explicit prompt must name an observed owned return")
            prompt = tuple(int(value) for value in xy[matches[0]])
        start = time.perf_counter()
        masks, qualities = self.backend.predict(prompt)
        elapsed = time.perf_counter() - start
        masks, qualities = np.asarray(masks), np.asarray(qualities)
        if masks.dtype != np.bool_ or masks.shape != (len(qualities), height, width):
            raise ValueError("SAM output masks must match the source RGB image")
        if (
            qualities.ndim != 1
            or not np.isfinite(qualities).all()
            or ((qualities < 0) | (qualities > 1)).any()
        ):
            raise ValueError("SAM model-quality scores must be finite values in [0, 1]")
        candidates = []
        for mask, quality in zip(masks, qualities):
            area = int(mask.sum())
            if (
                quality < self.config.minimum_model_quality
                or area < self.config.minimum_mask_pixels
                or area > self.config.maximum_image_fraction * height * width
                or not mask[prompt[1], prompt[0]]
                or mask[0].any()
                or mask[-1].any()
                or mask[:, 0].any()
                or mask[:, -1].any()
            ):
                continue
            hits = mask[xy[:, 1], xy[:, 0]]
            own_hits, foreign_hits = (
                int(np.count_nonzero(hits & owned)),
                int(np.count_nonzero(hits & ~owned)),
            )
            if (
                own_hits / count < self.config.minimum_track_coverage
                or foreign_hits / max(1, own_hits + foreign_hits)
                > self.config.maximum_foreign_hit_fraction
            ):
                continue
            candidates.append((float(quality), mask, own_hits))
        candidates.sort(key=lambda item: item[0], reverse=True)
        common = dict(
            prompt_xy=prompt,
            projected_track_points=count,
            inference_wall_seconds=elapsed,
            image_encoder_wall_seconds=self.image_encoder_wall_seconds,
        )
        if not candidates:
            return SamInstanceResult(
                None,
                ("no mask passes quality, boundary and LiDAR association gates",),
                **common,
            )
        quality, mask, hits = candidates[0]
        for other_quality, other, _ in candidates[1:]:
            iou = np.count_nonzero(mask & other) / max(
                1, np.count_nonzero(mask | other)
            )
            if (
                iou < self.config.equivalent_mask_iou
                and quality - other_quality < self.config.ambiguity_quality_gap
            ):
                return SamInstanceResult(
                    None,
                    ("competing non-equivalent SAM masks are ambiguous",),
                    **common,
                )
        yy, xx = np.where(mask)
        detection = SemanticDetection(
            0,
            "unknown",
            min(quality, self.config.confidence_cap),
            PixelBoundingBox(
                float(xx.min()),
                float(yy.min()),
                float(xx.max() + 1),
                float(yy.max() + 1),
            ),
            mask,
        )
        return SamInstanceResult(
            detection, (), model_quality=quality, mask_track_hits=hits, **common
        )
