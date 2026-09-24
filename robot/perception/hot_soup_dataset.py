"""Auditable dataset contracts for visible hot-soup plume detection.

These offline labels supervise a visible plume and an open container. They are
not temperature measurements, proof of steam, or CRA runtime evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Iterable


HOT_SOUP_DATASET_SCHEMA_VERSION = 1
HOT_SOUP_DETECTION_CLASSES = ("visible_hot_soup_plume", "open_container")
ALLOWED_SPLITS = frozenset({"train", "val", "test"})
ALLOWED_DOMAINS = frozenset({"real_user_owned", "isaac_synthetic"})
ALLOWED_ANNOTATION_SOURCES = frozenset(
    {"human_visual_annotation", "isaac_authored_oracle"}
)
ALLOWED_HARD_NEGATIVE_TAGS = frozenset(
    {"smoke", "glare", "haze", "white_background", "empty_container"}
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _identifier(value: object, name: str) -> str:
    result = _text(value, name)
    if not _SAFE_ID.fullmatch(result):
        raise ValueError(f"{name} contains unsupported filename characters")
    return result


def _relative_path(value: object, name: str) -> str:
    text = _text(value, name).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must stay inside the dataset root")
    return str(path)


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class HotSoupBox:
    """One normalized detection box with auditable annotation provenance."""

    class_name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    annotation_source: str

    def __post_init__(self) -> None:
        if self.class_name not in HOT_SOUP_DETECTION_CLASSES:
            raise ValueError("unsupported hot-soup detection class")
        if self.annotation_source not in ALLOWED_ANNOTATION_SOURCES:
            raise ValueError("unsupported annotation_source")
        for name in ("x_min", "y_min", "x_max", "y_max"):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("box must have positive normalized area")


@dataclass(frozen=True)
class HotSoupSample:
    """One completely annotated stationary image.

    Explicit present/absent states prevent missing annotations from being
    interpreted as safe negatives.
    """

    sample_id: str
    capture_group_id: str
    split: str
    image_path: str
    image_sha256: str
    width: int
    height: int
    domain: str
    source_reference: str
    usage_rights: str
    plume_state: str
    container_state: str
    boxes: tuple[HotSoupBox, ...] = ()
    hard_negative_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _identifier(self.sample_id, "sample_id"))
        object.__setattr__(
            self,
            "capture_group_id",
            _identifier(self.capture_group_id, "capture_group_id"),
        )
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}")
        object.__setattr__(
            self, "image_path", _relative_path(self.image_path, "image_path")
        )
        digest = _text(self.image_sha256, "image_sha256").lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("image_sha256 must be a hexadecimal SHA-256 digest")
        object.__setattr__(self, "image_sha256", digest)
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.domain not in ALLOWED_DOMAINS:
            raise ValueError(f"domain must be one of {sorted(ALLOWED_DOMAINS)}")
        for name in ("source_reference", "usage_rights"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("plume_state", "container_state"):
            if getattr(self, name) not in {"present", "absent"}:
                raise ValueError(f"{name} must be explicitly present or absent")
        boxes = tuple(self.boxes)
        if not all(isinstance(box, HotSoupBox) for box in boxes):
            raise TypeError("boxes must contain HotSoupBox values")
        object.__setattr__(self, "boxes", boxes)
        tags = tuple(sorted(set(self.hard_negative_tags)))
        if set(tags) - ALLOWED_HARD_NEGATIVE_TAGS:
            raise ValueError("unsupported hard-negative tag")
        object.__setattr__(self, "hard_negative_tags", tags)
        for class_name, state_name in (
            ("visible_hot_soup_plume", "plume_state"),
            ("open_container", "container_state"),
        ):
            count = sum(box.class_name == class_name for box in boxes)
            expected = getattr(self, state_name)
            if (expected == "present") != (count > 0):
                raise ValueError(f"{state_name} conflicts with {class_name} boxes")
        if self.plume_state == "present" and self.container_state != "present":
            raise ValueError("a positive hot-soup plume requires open-container context")
        if self.domain == "real_user_owned" and any(
            box.annotation_source == "isaac_authored_oracle" for box in boxes
        ):
            raise ValueError("real images cannot use Isaac oracle annotations")
        if self.hard_negative_tags and self.plume_state != "absent":
            raise ValueError("hard-negative tags require an absent hot-soup plume")


@dataclass(frozen=True)
class HotSoupDatasetReport:
    sample_count: int
    positive_plume_count: int
    negative_plume_count: int
    hard_negative_count: int
    domain_counts: tuple[tuple[str, int], ...]
    split_counts: tuple[tuple[str, int], ...]
    missing_files: tuple[str, ...]
    hash_mismatches: tuple[str, ...]
    capture_group_split_leaks: tuple[str, ...]
    duplicate_split_leaks: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not (
            self.missing_files
            or self.hash_mismatches
            or self.capture_group_split_leaks
            or self.duplicate_split_leaks
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_capture_group_split(
    capture_group_id: str,
    *,
    salt: str = "cra_hot_soup_v1",
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> str:
    """Keep adjacent frames from one capture session in a stable split."""
    key = _identifier(capture_group_id, "capture_group_id")
    train = float(train_fraction)
    val = float(val_fraction)
    if not 0 < train < 1 or not 0 <= val < 1 or train + val >= 1:
        raise ValueError("split fractions must leave nonzero train and test portions")
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < train:
        return "train"
    if fraction < train + val:
        return "val"
    return "test"


def write_hot_soup_manifest(
    samples: Iterable[HotSoupSample], manifest_path: Path
) -> None:
    items = sorted(tuple(samples), key=lambda item: item.sample_id)
    if len({item.sample_id for item in items}) != len(items):
        raise ValueError("sample_id values must be unique")
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(
                json.dumps(
                    {"schema_version": HOT_SOUP_DATASET_SCHEMA_VERSION, **asdict(item)},
                    sort_keys=True,
                )
                + "\n"
            )


def read_hot_soup_manifest(manifest_path: Path) -> tuple[HotSoupSample, ...]:
    items = []
    with Path(manifest_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.pop("schema_version", None) != HOT_SOUP_DATASET_SCHEMA_VERSION:
                raise ValueError(f"line {line_number}: unsupported schema version")
            payload["boxes"] = tuple(HotSoupBox(**box) for box in payload["boxes"])
            payload["hard_negative_tags"] = tuple(payload["hard_negative_tags"])
            items.append(HotSoupSample(**payload))
    if len({item.sample_id for item in items}) != len(items):
        raise ValueError("sample_id values must be unique")
    return tuple(items)


def validate_hot_soup_dataset(
    samples: Iterable[HotSoupSample], dataset_root: Path
) -> HotSoupDatasetReport:
    items = tuple(samples)
    root = Path(dataset_root).resolve()
    split_counts = {name: 0 for name in sorted(ALLOWED_SPLITS)}
    domain_counts = {name: 0 for name in sorted(ALLOWED_DOMAINS)}
    groups: dict[str, set[str]] = {}
    digests: dict[str, set[str]] = {}
    missing = []
    mismatches = []
    for item in items:
        split_counts[item.split] += 1
        domain_counts[item.domain] += 1
        groups.setdefault(item.capture_group_id, set()).add(item.split)
        digests.setdefault(item.image_sha256, set()).add(item.split)
        image_path = root / item.image_path
        if not image_path.is_file():
            missing.append(item.image_path)
        elif sha256_file(image_path) != item.image_sha256:
            mismatches.append(item.image_path)
    return HotSoupDatasetReport(
        sample_count=len(items),
        positive_plume_count=sum(item.plume_state == "present" for item in items),
        negative_plume_count=sum(item.plume_state == "absent" for item in items),
        hard_negative_count=sum(bool(item.hard_negative_tags) for item in items),
        domain_counts=tuple(domain_counts.items()),
        split_counts=tuple(split_counts.items()),
        missing_files=tuple(sorted(set(missing))),
        hash_mismatches=tuple(sorted(set(mismatches))),
        capture_group_split_leaks=tuple(
            sorted(group for group, splits in groups.items() if len(splits) > 1)
        ),
        duplicate_split_leaks=tuple(
            sorted(digest for digest, splits in digests.items() if len(splits) > 1)
        ),
    )


def export_yolo_hot_soup_dataset(
    samples: Iterable[HotSoupSample], dataset_root: Path, output_root: Path
) -> Path:
    """Copy a validated manifest into an Ultralytics detection layout."""
    items = tuple(samples)
    report = validate_hot_soup_dataset(items, dataset_root)
    if not report.valid:
        raise ValueError(f"hot-soup dataset is invalid: {report}")
    source_root = Path(dataset_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty YOLO export")
    used_splits = []
    for item in items:
        if item.split not in used_splits:
            used_splits.append(item.split)
        suffix = Path(item.image_path).suffix.lower() or ".png"
        image_target = destination / "images" / item.split / f"{item.sample_id}{suffix}"
        label_target = destination / "labels" / item.split / f"{item.sample_id}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / item.image_path, image_target)
        lines = []
        for box in item.boxes:
            class_id = HOT_SOUP_DETECTION_CLASSES.index(box.class_name)
            center_x = (box.x_min + box.x_max) / 2
            center_y = (box.y_min + box.y_max) / 2
            width = box.x_max - box.x_min
            height = box.y_max - box.y_min
            lines.append(
                f"{class_id} {center_x:.7f} {center_y:.7f} "
                f"{width:.7f} {height:.7f}"
            )
        label_target.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    config = destination / "dataset.yaml"
    lines = [f"path: {destination.as_posix()}"]
    if "train" in used_splits:
        lines.append("train: images/train")
    if "val" in used_splits:
        lines.append("val: images/val")
    if "test" in used_splits:
        lines.append("test: images/test")
    lines.append("names:")
    lines.extend(
        f"  {index}: {name}"
        for index, name in enumerate(HOT_SOUP_DETECTION_CLASSES)
    )
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config
