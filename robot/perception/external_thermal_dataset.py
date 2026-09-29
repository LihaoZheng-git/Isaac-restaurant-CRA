"""Audit and import external YOLO thermal cues without trusting their splits.

External boxes are candidate annotation aids. Imported images stay pending
project relabeling, and their only allowed project split is training. They do
not satisfy the project-owned real validation/test gate.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Iterable

from robot.perception.hot_soup_collection import (
    HotSoupCollectedImage,
    collect_hot_soup_images,
    write_hot_soup_collection_manifest,
)


EXTERNAL_THERMAL_AUDIT_SCHEMA_VERSION = 1
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
PLUME_CLASS_NAMES = frozenset({"steam"})
SOURCE_CLASS_NAMES = frozenset({"open_pot", "open_pot_boiling"})


@dataclass(frozen=True)
class ExternalThermalImageAudit:
    image_path: str
    published_split: str
    capture_group_id: str
    class_names: tuple[str, ...]
    label_file_present: bool
    label_box_count: int
    has_plume_box: bool
    has_source_box: bool


@dataclass(frozen=True)
class ExternalThermalDatasetAudit:
    image_count: int
    labeled_image_count: int
    missing_label_file_count: int
    source_group_count: int
    published_split_counts: tuple[tuple[str, int], ...]
    class_box_counts: tuple[tuple[str, int], ...]
    unsupported_class_box_counts: tuple[tuple[str, int], ...]
    published_split_leak_groups: tuple[str, ...]
    plume_without_source_count: int
    duplicate_content_count: int
    records: tuple[ExternalThermalImageAudit, ...]
    annotation_status: str = "pending_project_relabel"
    evaluation_status: str = "not_eligible_for_project_validation_or_test"


def read_yolo_class_names(config_path: Path) -> tuple[str, ...]:
    """Read the small ``names`` value from common YOLO YAML layouts."""

    lines = Path(config_path).read_text(encoding="utf-8").splitlines()
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped.startswith("names:"):
            continue
        remainder = stripped.removeprefix("names:").strip()
        if remainder:
            value = ast.literal_eval(remainder)
            if isinstance(value, (list, tuple)) and all(
                isinstance(item, str) and item.strip() for item in value
            ):
                return tuple(item.strip() for item in value)
            if isinstance(value, dict):
                ordered = sorted((int(key), str(item)) for key, item in value.items())
                if [key for key, _ in ordered] == list(range(len(ordered))):
                    return tuple(item for _, item in ordered)
            raise ValueError("YOLO names must be a string list or contiguous mapping")
        mapping = []
        for child in lines[index + 1 :]:
            if child and not child[0].isspace():
                break
            match = re.match(r"^\s*(\d+)\s*:\s*['\"]?(.+?)['\"]?\s*$", child)
            if match:
                mapping.append((int(match.group(1)), match.group(2)))
        mapping.sort()
        if mapping and [key for key, _ in mapping] == list(range(len(mapping))):
            return tuple(item for _, item in mapping)
        raise ValueError("YOLO names mapping must use contiguous integer keys")
    raise ValueError("YOLO config has no names field")


def external_capture_group_id(image_path: Path) -> str:
    """Recover an apparent source sequence while keeping an ASCII-safe ID."""

    stem = image_path.stem
    stem = re.sub(r"\.rf\.[A-Za-z0-9]+$", "", stem)
    stem = re.sub(r"_(?:jpg|jpeg|png|bmp|webp)$", "", stem, flags=re.IGNORECASE)
    source = re.sub(r"[-_](?:frame[-_]?)?\d+$", "", stem, flags=re.IGNORECASE)
    if not source:
        source = stem
    normalized = unicodedata.normalize("NFKD", source).encode("ascii", "ignore").decode()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", normalized).strip("-._")
    if not safe:
        safe = "source"
    digest = hashlib.sha256(source.casefold().encode("utf-8")).hexdigest()[:10]
    return f"external-{safe[:48]}-{digest}"


def _published_images(root: Path) -> tuple[tuple[str, Path, Path], ...]:
    candidates = []
    layouts = (
        ("train", root / "train" / "images", root / "train" / "labels"),
        ("val", root / "valid" / "images", root / "valid" / "labels"),
        ("val", root / "val" / "images", root / "val" / "labels"),
        ("test", root / "test" / "images", root / "test" / "labels"),
        ("train", root / "images" / "train", root / "labels" / "train"),
        ("val", root / "images" / "val", root / "labels" / "val"),
        ("test", root / "images" / "test", root / "labels" / "test"),
    )
    seen = set()
    for split, image_root, label_root in layouts:
        if not image_root.is_dir():
            continue
        for image in sorted(path for path in image_root.rglob("*") if path.is_file()):
            if image.suffix.casefold() not in IMAGE_SUFFIXES:
                continue
            key = image.resolve()
            if key in seen:
                continue
            seen.add(key)
            relative = image.relative_to(image_root).with_suffix(".txt")
            candidates.append((split, image, label_root / relative))
    if not candidates:
        raise ValueError("no images found in a supported YOLO split layout")
    return tuple(candidates)


def _label_class_ids(path: Path, class_count: int) -> tuple[int, ...]:
    if not path.is_file():
        return ()
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        token = line.split()[0]
        try:
            class_id = int(token)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid class id") from error
        if not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_number}: class id out of range")
        values.append(class_id)
    return tuple(values)


def audit_external_thermal_yolo(
    export_root: Path,
    *,
    class_names: Iterable[str] | None = None,
) -> ExternalThermalDatasetAudit:
    """Audit candidate boxes and published split leakage without accepting labels."""

    root = Path(export_root).resolve()
    names = (
        tuple(class_names)
        if class_names is not None
        else read_yolo_class_names(root / "data.yaml")
    )
    if not names or any(not isinstance(item, str) or not item.strip() for item in names):
        raise ValueError("class_names must contain nonempty strings")
    canonical = tuple(item.strip().casefold() for item in names)
    class_counts = {name: 0 for name in canonical}
    split_counts = {name: 0 for name in ("train", "val", "test")}
    group_splits: dict[str, set[str]] = {}
    records = []
    digests = []
    for split, image, label in _published_images(root):
        ids = _label_class_ids(label, len(names))
        observed = tuple(canonical[class_id] for class_id in ids)
        for name in observed:
            class_counts[name] += 1
        group = external_capture_group_id(image)
        group_splits.setdefault(group, set()).add(split)
        split_counts[split] += 1
        digests.append(hashlib.sha256(image.read_bytes()).hexdigest())
        records.append(
            ExternalThermalImageAudit(
                image_path=image.relative_to(root).as_posix(),
                published_split=split,
                capture_group_id=group,
                class_names=tuple(sorted(set(observed))),
                label_file_present=label.is_file(),
                label_box_count=len(ids),
                has_plume_box=bool(set(observed) & PLUME_CLASS_NAMES),
                has_source_box=bool(set(observed) & SOURCE_CLASS_NAMES),
            )
        )
    supported = PLUME_CLASS_NAMES | SOURCE_CLASS_NAMES
    unsupported = tuple(
        (name, count)
        for name, count in sorted(class_counts.items())
        if name not in supported and count
    )
    return ExternalThermalDatasetAudit(
        image_count=len(records),
        labeled_image_count=sum(item.label_file_present for item in records),
        missing_label_file_count=sum(
            not item.label_file_present for item in records
        ),
        source_group_count=len(group_splits),
        published_split_counts=tuple(split_counts.items()),
        class_box_counts=tuple(sorted(class_counts.items())),
        unsupported_class_box_counts=unsupported,
        published_split_leak_groups=tuple(
            sorted(group for group, splits in group_splits.items() if len(splits) > 1)
        ),
        plume_without_source_count=sum(
            item.has_plume_box and not item.has_source_box for item in records
        ),
        duplicate_content_count=len(digests) - len(set(digests)),
        records=tuple(records),
    )


def import_external_thermal_images(
    export_root: Path,
    dataset_root: Path,
    *,
    source_reference: str,
    usage_rights: str,
    class_names: Iterable[str] | None = None,
) -> tuple[ExternalThermalDatasetAudit, tuple[HotSoupCollectedImage, ...]]:
    """Copy candidate images as pending, training-only relabeling inputs."""

    audit = audit_external_thermal_yolo(export_root, class_names=class_names)
    source_root = Path(export_root).resolve()
    collected = []
    for record in audit.records:
        detail = (
            f"{source_reference}; original={record.image_path}; "
            f"published_split={record.published_split}; "
            f"candidate_classes={','.join(record.class_names) or 'none'}"
        )
        collected.extend(
            collect_hot_soup_images(
                [source_root / record.image_path],
                dataset_root,
                capture_group_id=record.capture_group_id,
                domain="external_training_candidate",
                source_kind="external_dataset_image",
                source_reference=detail,
                usage_rights=usage_rights,
                suggested_split="train",
            )
        )
    return audit, tuple(collected)


def write_external_thermal_audit(
    report: ExternalThermalDatasetAudit, output_path: Path
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": EXTERNAL_THERMAL_AUDIT_SCHEMA_VERSION, **asdict(report)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_external_thermal_collection(
    items: Iterable[HotSoupCollectedImage], output_path: Path
) -> None:
    write_hot_soup_collection_manifest(items, output_path)
