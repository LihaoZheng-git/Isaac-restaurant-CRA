"""Readiness and provenance contracts for hot-soup detector training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from robot.perception.hot_soup_dataset import (
    ALLOWED_SPLITS,
    HotSoupSample,
    validate_hot_soup_dataset,
)


@dataclass(frozen=True)
class HotSoupTrainingReadinessConfig:
    """Minimum functional gate; publication-scale sufficiency is separate."""

    minimum_real_capture_groups: int = 3
    minimum_real_positive_per_split: int = 1
    minimum_real_negative_per_split: int = 1
    minimum_evaluation_hard_negatives: int = 1

    def __post_init__(self) -> None:
        for name in (
            "minimum_real_capture_groups",
            "minimum_real_positive_per_split",
            "minimum_real_negative_per_split",
            "minimum_evaluation_hard_negatives",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class HotSoupTrainingReadiness:
    real_sample_count: int
    real_capture_group_count: int
    real_positive_by_split: tuple[tuple[str, int], ...]
    real_negative_by_split: tuple[tuple[str, int], ...]
    evaluation_hard_negative_count: int
    synthetic_sample_count: int
    external_training_sample_count: int
    blockers: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers


def assess_hot_soup_training_readiness(
    samples: Iterable[HotSoupSample],
    dataset_root: Path,
    *,
    config: HotSoupTrainingReadinessConfig | None = None,
) -> HotSoupTrainingReadiness:
    """Require valid, group-separated real positives and negatives."""

    items = tuple(samples)
    settings = config or HotSoupTrainingReadinessConfig()
    report = validate_hot_soup_dataset(items, dataset_root)
    blockers = []
    if not report.valid:
        blockers.append("dataset integrity validation failed")
    real = tuple(item for item in items if item.domain == "real_user_owned")
    synthetic = tuple(item for item in items if item.domain == "isaac_synthetic")
    external = tuple(
        item for item in items if item.domain == "licensed_external_training"
    )
    positive = {
        split: sum(item.split == split and item.plume_state == "present" for item in real)
        for split in sorted(ALLOWED_SPLITS)
    }
    negative = {
        split: sum(item.split == split and item.plume_state == "absent" for item in real)
        for split in sorted(ALLOWED_SPLITS)
    }
    groups = {item.capture_group_id for item in real}
    if len(groups) < settings.minimum_real_capture_groups:
        blockers.append(
            "real capture groups below minimum: "
            f"{len(groups)} < {settings.minimum_real_capture_groups}"
        )
    for split in sorted(ALLOWED_SPLITS):
        if positive[split] < settings.minimum_real_positive_per_split:
            blockers.append(
                f"{split}: real positive plume samples below minimum: "
                f"{positive[split]} < {settings.minimum_real_positive_per_split}"
            )
        if negative[split] < settings.minimum_real_negative_per_split:
            blockers.append(
                f"{split}: real negative plume samples below minimum: "
                f"{negative[split]} < {settings.minimum_real_negative_per_split}"
            )
    evaluation_hard_negatives = sum(
        item.split in {"val", "test"} and bool(item.hard_negative_tags)
        for item in real
    )
    if evaluation_hard_negatives < settings.minimum_evaluation_hard_negatives:
        blockers.append(
            "real validation/test hard negatives below minimum: "
            f"{evaluation_hard_negatives} < "
            f"{settings.minimum_evaluation_hard_negatives}"
        )
    return HotSoupTrainingReadiness(
        real_sample_count=len(real),
        real_capture_group_count=len(groups),
        real_positive_by_split=tuple(positive.items()),
        real_negative_by_split=tuple(negative.items()),
        evaluation_hard_negative_count=evaluation_hard_negatives,
        synthetic_sample_count=len(synthetic),
        external_training_sample_count=len(external),
        blockers=tuple(blockers),
    )


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_hot_soup_training_provenance(
    output_path: Path,
    *,
    readiness: HotSoupTrainingReadiness,
    manifest_path: Path,
    base_checkpoint_path: Path,
    trained_checkpoint_path: Path,
    package_version: str,
    package_license: str,
    training_arguments: dict[str, object],
) -> None:
    """Record exact data, code-package, and weight identities."""

    if not readiness.ready:
        raise ValueError("training provenance cannot mark a blocked dataset as trained")
    for path in (manifest_path, base_checkpoint_path, trained_checkpoint_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    payload = {
        "readiness": asdict(readiness),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "base_checkpoint": str(base_checkpoint_path),
        "base_checkpoint_sha256": sha256_path(base_checkpoint_path),
        "trained_checkpoint": str(trained_checkpoint_path),
        "trained_checkpoint_sha256": sha256_path(trained_checkpoint_path),
        "package_version": package_version,
        "package_license": package_license,
        "training_arguments": training_arguments,
        "scope": "visible plume and open-container detection; not temperature",
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
