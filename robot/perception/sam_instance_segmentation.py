"""Category-free SAM 2.1 masks prompted only by projected LiDAR returns.

The tracker still owns obstacle discovery and occupancy. SAM refines RGB
instances, with a model-quality gate and explicit rejection of ambiguous masks.
It supplies no class, material, density, physical sharpness, or temperature.
"""

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import time

import numpy as np

from robot.perception.camera_lidar_fusion import ProjectedLidarPoints, mask_foreground_hits
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection


SAM_INSTANCE_SOURCE = "sam2_lidar_prompted_rgb_instance"
SAM_IMAGE_OBJECT_SOURCE = "sam2_image_object_hypothesis"
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
    foreground_depth_gap_m: float | None = None
    observed_context_box_retry: bool = False
    context_box_padding_fraction: float = 1.0
    track_extent_retry: bool = False

    def __post_init__(self):
        if not isinstance(self.observed_context_box_retry, bool):
            raise TypeError("observed context-box opt-in must be boolean")
        if not isinstance(self.track_extent_retry, bool):
            raise TypeError("track-extent retry opt-in must be boolean")
        if self.track_extent_retry and not self.observed_context_box_retry:
            raise ValueError("track-extent retry requires observed context-box opt-in")
        if not math.isfinite(self.context_box_padding_fraction) or not 0 <= self.context_box_padding_fraction <= 1:
            raise ValueError("observed context-box padding must lie in [0, 1]")
        if self.foreground_depth_gap_m is not None and (
            not math.isfinite(self.foreground_depth_gap_m) or not 0.3 <= self.foreground_depth_gap_m <= 1.0
        ):
            raise ValueError("foreground depth gap must lie within [.3, 1] m")
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
    candidate_diagnostics: tuple[dict, ...] = ()
    observed_context_box_xyxy: tuple[int, int, int, int] | None = None
    extent_retry_diagnostics: dict | None = None


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

    def predict(self, prompt_xy, *, context_box_xyxy=None):
        box_args = {} if context_box_xyxy is None else {"bboxes": [context_box_xyxy]}
        result = self.predictor(
            self.image_bgr, points=[[prompt_xy]], labels=[[1]], multimask_output=True,
            **box_args,
        )[0]
        if result.masks is None:
            return np.empty((0, *self.image_bgr.shape[:2]), bool), np.empty(0)
        return (
            result.masks.data.cpu().numpy().astype(bool),
            result.boxes.conf.cpu().numpy(),
        )

    def generate(self):
        """Existing installed SAM backend; no labels, detector boxes or downloads."""
        result = self.predictor(
            self.image_bgr, points=None, bboxes=None, labels=None, masks=None,
            points_stride=32, points_batch_size=64, conf_thres=.85,
            stability_score_thresh=.95, crop_n_layers=0,
        )[0]
        if result.masks is None:
            return np.empty((0, *self.image_bgr.shape[:2]), bool), np.empty(0)
        return result.masks.data.cpu().numpy().astype(bool), result.boxes.conf.cpu().numpy()


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

    def image_proposals(self, point_track_ids, projection):
        """Image-first candidates plus observed extent prompts, NOT associations.

        Separate stationary experiment only. Original pixel/quality gates apply;
        multi-fragment ownership must be checked by the downstream adapter.
        No retained mask is used as a new observation.
        """
        import cv2
        from robot.perception.visual_shape_contact import SilhouetteContactConfig

        if self.image_shape is None:
            raise RuntimeError("set_image is required for each camera frame")
        batches = [("automatic_image_grid", *self.backend.generate())]
        ids = np.asarray(point_track_ids)[projection.source_indices]
        # Bounded same-source extent prompts recover instances missed by the
        # automatic point grid (e.g. a bin whose decorative panel is salient).
        eligible = [int(k) for k in np.unique(ids) if k >= 0 and np.sum(ids == k) >= 8]
        for key in eligible[:16]:
            xy = projection.pixel_indices_xy[ids == key]
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            padding = .25 * np.maximum(hi - lo, 1)
            a = np.maximum(0, np.floor(lo - padding)).astype(int)
            b = np.minimum(np.array(self.image_shape[::-1]) - 1, np.ceil(hi + padding)).astype(int)
            point = xy[np.argmin(np.sum((xy - np.median(xy, axis=0)) ** 2, axis=1))]
            masks, qualities = self.backend.predict(tuple(point), context_box_xyxy=(*a, *b))
            batches.append(("current_observed_extent", masks, qualities))
        proposals = []
        for basis, masks, qualities in batches:
            masks, qualities = np.asarray(masks), np.asarray(qualities)
            if masks.ndim != 3 or masks.shape[1:] != self.image_shape or qualities.shape != (len(masks),):
                raise ValueError("image proposal backend shape mismatch")
            for mask, quality in zip(masks.astype(bool), qualities):
                count = int(mask.sum())
                if (not np.isfinite(quality) or not self.config.minimum_model_quality <= quality <= 1
                    or count < self.config.minimum_mask_pixels
                    or count > mask.size * self.config.maximum_image_fraction
                    or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()):
                    continue
                # A larger mask with disconnected background islands is not a
                # better whole-object proposal. Reuse the existing silhouette
                # connected-component criterion; reject, never fill/crop masks.
                _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
                if stats[1:, cv2.CC_STAT_AREA].max() / count < SilhouetteContactConfig().minimum_component_fraction:
                    continue
                yy, xx = np.where(mask)
                detection = SemanticDetection(0, "unknown", min(float(quality), self.config.confidence_cap),
                    PixelBoundingBox(float(xx.min()), float(yy.min()), float(xx.max()+1), float(yy.max()+1)), mask)
                proposals.append(SamInstanceResult(detection, (), model_quality=float(quality),
                    source=SAM_IMAGE_OBJECT_SOURCE, candidate_diagnostics=({"basis": basis},)))
        # Stable maximum of 64 image hypotheses, without favoring any target.
        return tuple(sorted(proposals, key=lambda p: (-p.detection.mask_pixel_count, -p.model_quality))[:64])

    def propose(
        self, point_track_ids, projection, *, track_id,
        prompt_source_index=None, use_observed_context_box=False,
    ):
        """Optionally retry a valid but small part using this track's image extent.

        This default-OFF proposal change does not group tracks or weaken gates.
        A replacement must independently pass every original check, cover >=80%
        of the track's projected returns, and contain >=98% of the original mask.
        Neither fraction certifies a whole physical object: a track can be a part.
        """
        result = self._propose(
            point_track_ids, projection, track_id=track_id,
            prompt_source_index=prompt_source_index,
            use_observed_context_box=use_observed_context_box)
        if (not self.config.track_extent_retry or result.detection is None
                or use_observed_context_box or prompt_source_index is not None
                or result.mask_track_hits / max(1, result.projected_track_points) >= .8):
            return result
        retry = self._propose(
            point_track_ids, projection, track_id=track_id,
            use_observed_context_box=True, context_padding_fraction=.25)
        containment = (float(np.count_nonzero(result.detection.mask & retry.detection.mask))
                       / max(1, int(result.detection.mask.sum()))
                       if retry.detection is not None else 0.)
        coverage = retry.mask_track_hits / max(1, retry.projected_track_points)
        promote = bool(retry.detection is not None and coverage >= .8 and containment >= .98)
        diagnostic = {
            "basis": "same_source_owned_track_extent_proposal_only",
            "original_track_coverage": result.mask_track_hits / result.projected_track_points,
            "retry_track_coverage": coverage,
            "original_mask_containment": containment,
            "promoted": promote,
            "retry_context_box_xyxy": retry.observed_context_box_xyxy,
            "retry_rejections": retry.rejection_reasons,
            "original_candidates": result.candidate_diagnostics,
            "retry_candidates": retry.candidate_diagnostics,
            "scope": "same track and unchanged gates; not whole-object identity or new geometry",
        }
        selected = retry if promote else result
        return replace(selected, extent_retry_diagnostics=diagnostic,
                       inference_wall_seconds=result.inference_wall_seconds + retry.inference_wall_seconds)

    def _propose(
        self,
        point_track_ids,
        projection: ProjectedLidarPoints,
        *,
        track_id,
        prompt_source_index=None,
        use_observed_context_box=False,
        context_padding_fraction=None,
    ):
        if self.image_shape is None:
            raise RuntimeError("set_image is required for each camera frame")
        if not isinstance(use_observed_context_box, bool) or (
            use_observed_context_box and not self.config.observed_context_box_retry
        ):
            raise ValueError("observed context box requires explicit configuration opt-in")
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
        context_box = None
        if use_observed_context_box:
            # A sparse scan may cover only the lower body. Symmetric context is
            # an image proposal, not a claim about the object's physical extent.
            # Only this track's actually projected returns determine the box.
            low, high = points.min(axis=0), points.max(axis=0) + 1
            padding = (high - low) * (self.config.context_box_padding_fraction
                                      if context_padding_fraction is None else context_padding_fraction)
            context_box = tuple(int(v) for v in np.r_[
                np.maximum(0, np.floor(low - padding)),
                np.minimum([width, height], np.ceil(high + padding)),
            ])
        start = time.perf_counter()
        if context_box is None:
            masks, qualities = self.backend.predict(prompt)
        else:
            masks, qualities = self.backend.predict(prompt, context_box_xyxy=context_box)
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
        candidate_diagnostics = []
        for mask, quality in zip(masks, qualities):
            area = int(mask.sum())
            hits = mask[xy[:, 1], xy[:, 0]]
            original_hit_count = int(hits.sum())
            if self.config.foreground_depth_gap_m is not None:
                hits = mask_foreground_hits(projection, mask, gap_m=self.config.foreground_depth_gap_m)
            own_hits = int(np.count_nonzero(hits & owned))
            foreign_hits = int(np.count_nonzero(hits & ~owned))
            boundary = bool(
                mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
            )
            candidate_diagnostics.append(
                {
                    "model_quality": float(quality),
                    "mask_pixels": area,
                    "image_fraction": area / (height * width),
                    "contains_observed_prompt": bool(mask[prompt[1], prompt[0]]),
                    "touches_image_boundary": boundary,
                    "owned_mask_hits": own_hits,
                    "projected_owned_points": count,
                    "track_coverage": own_hits / count,
                    "foreign_mask_hits": foreign_hits,
                    "farther_depth_layer_mask_hits": original_hit_count - int(hits.sum()),
                    "foreign_hit_fraction": foreign_hits
                    / max(1, own_hits + foreign_hits),
                }
            )
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
            candidate_diagnostics=tuple(candidate_diagnostics),
            observed_context_box_xyxy=context_box,
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
