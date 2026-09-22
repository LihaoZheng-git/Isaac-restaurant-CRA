"""Inventory fixed-label PACO parts without loading full annotation files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import mmap
from pathlib import Path
from typing import Iterable, Iterator, Mapping


PACO_LVIS_ANNOTATION_SHA256 = (
    "02ac4edb22c251e07853e6231d69aec3fad0a180f03de2f8c880650322debc80"
)
PACO_REPOSITORY_URL = "https://github.com/facebookresearch/paco"
PACO_ANNOTATION_URL = (
    "https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip"
)
PACO_CONTACT_REPORT_SCHEMA_VERSION = 1
PACO_HAZARD_MANIFEST_SCHEMA_VERSION = 1

GENERAL_CONTACT_GEOMETRIES = (
    "exposed_edge",
    "pointed_tip",
    "protrusion",
    "jagged_surface",
)

# These labels supervise visible geometry only. They do not establish sharpness,
# material hardness, contact probability, or injury severity.
DEFAULT_PACO_HAZARD_PARTS = {
    "knife:blade": "exposed_edge",
    "scissors:blade": "exposed_edge",
    "screwdriver:tip": "pointed_tip",
}


@dataclass(frozen=True)
class PacoPartSplitCount:
    split: str
    category_id: int
    category_name: str
    image_count: int
    instance_count: int

    def __post_init__(self) -> None:
        if not self.split or not self.category_name:
            raise ValueError("split and category name must be nonempty")
        for name in ("category_id", "image_count", "instance_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class PacoContactInventoryReport:
    target_parts: tuple[str, ...]
    split_counts: tuple[PacoPartSplitCount, ...]
    total_images: int
    total_instances: int
    source_url: str = PACO_ANNOTATION_URL
    archive_sha256: str = PACO_LVIS_ANNOTATION_SHA256
    code_license: str = "MIT"
    annotation_license_status: str = "not_explicitly_stated_in_official_repository"
    runtime_model_status: str = "unavailable"


def _top_level_array_bytes(path: Path, key: str) -> bytes:
    """Extract one top-level JSON array using bounded memory."""
    marker = json.dumps(key).encode("ascii")
    with Path(path).open("rb") as handle, mmap.mmap(
        handle.fileno(), length=0, access=mmap.ACCESS_READ
    ) as data:
        search_start = 0
        marker_index = -1
        while True:
            candidate = data.find(marker, search_start)
            if candidate < 0:
                raise ValueError(f"top-level key {key!r} was not found")
            cursor = candidate + len(marker)
            while cursor < len(data) and chr(data[cursor]).isspace():
                cursor += 1
            if cursor < len(data) and data[cursor] == ord(":"):
                marker_index = candidate
                break
            search_start = candidate + len(marker)
        start = data.find(b"[", marker_index + len(marker))
        if start < 0:
            raise ValueError(f"top-level key {key!r} is not an array")
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(data)):
            value = data[index]
            if in_string:
                if escaped:
                    escaped = False
                elif value == ord("\\"):
                    escaped = True
                elif value == ord('"'):
                    in_string = False
                continue
            if value == ord('"'):
                in_string = True
            elif value == ord("["):
                depth += 1
            elif value == ord("]"):
                depth -= 1
                if depth == 0:
                    return data[start : index + 1]
        raise ValueError(f"top-level array {key!r} is unterminated")


def read_paco_categories(annotation_path: Path) -> tuple[dict[str, object], ...]:
    """Read only the small categories array from a large PACO JSON file."""
    values = json.loads(_top_level_array_bytes(annotation_path, "categories"))
    if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
        raise ValueError("PACO categories must be an array of objects")
    return tuple(values)


def iter_paco_array_objects(
    annotation_path: Path, key: str
) -> Iterator[dict[str, object]]:
    """Stream objects from one top-level PACO array with bounded memory."""
    marker = json.dumps(key).encode("ascii")
    with Path(annotation_path).open("rb") as handle, mmap.mmap(
        handle.fileno(), length=0, access=mmap.ACCESS_READ
    ) as data:
        marker_index = data.find(marker)
        if marker_index < 0:
            raise ValueError(f"top-level key {key!r} was not found")
        array_start = data.find(b"[", marker_index + len(marker))
        if array_start < 0:
            raise ValueError(f"top-level key {key!r} is not an array")
        object_start = None
        object_depth = 0
        array_depth = 1
        in_string = False
        escaped = False
        for index in range(array_start + 1, len(data)):
            value = data[index]
            if in_string:
                if escaped:
                    escaped = False
                elif value == ord("\\"):
                    escaped = True
                elif value == ord('"'):
                    in_string = False
                continue
            if value == ord('"'):
                in_string = True
            elif value == ord("["):
                array_depth += 1
            elif value == ord("]"):
                array_depth -= 1
                if array_depth == 0:
                    if object_start is not None:
                        raise ValueError(f"top-level array {key!r} is malformed")
                    return
            elif value == ord("{"):
                if object_depth == 0:
                    object_start = index
                object_depth += 1
            elif value == ord("}"):
                object_depth -= 1
                if object_depth < 0:
                    raise ValueError(f"top-level array {key!r} is malformed")
                if object_depth == 0 and object_start is not None:
                    item = json.loads(data[object_start : index + 1])
                    if not isinstance(item, dict):
                        raise ValueError(f"{key!r} must contain objects")
                    yield item
                    object_start = None
        raise ValueError(f"top-level array {key!r} is unterminated")


def build_paco_hazard_manifest(
    annotation_path: Path,
    *,
    part_mapping: Mapping[str, str] = DEFAULT_PACO_HAZARD_PARTS,
) -> dict[str, object]:
    """Build a provenance-rich geometry manifest without loading PACO in memory."""
    mapping = dict(part_mapping)
    if not mapping:
        raise ValueError("part mapping cannot be empty")
    invalid = set(mapping.values()) - set(GENERAL_CONTACT_GEOMETRIES)
    if invalid:
        raise ValueError(f"unsupported contact geometries: {sorted(invalid)}")

    categories = read_paco_categories(annotation_path)
    category_by_name = {str(item.get("name")): item for item in categories}
    category_name_by_id = {int(item["id"]): str(item["name"]) for item in categories}
    missing = sorted(set(mapping) - set(category_by_name))
    if missing:
        raise ValueError(f"missing PACO part categories: {missing}")
    category_to_geometry = {
        int(category_by_name[name]["id"]): geometry
        for name, geometry in mapping.items()
    }

    hazards = []
    parent_ids = set()
    image_ids = set()
    for annotation in iter_paco_array_objects(annotation_path, "annotations"):
        category_id = int(annotation["category_id"])
        if category_id not in category_to_geometry:
            continue
        parent_id = annotation.get("obj_ann_id")
        if parent_id is not None:
            parent_ids.add(int(parent_id))
        image_ids.add(int(annotation["image_id"]))
        hazards.append(
            {
                "annotation_id": int(annotation["id"]),
                "image_id": int(annotation["image_id"]),
                "source_part_category": category_name_by_id[category_id],
                "geometry_class": category_to_geometry[category_id],
                "part_segmentation": annotation.get("segmentation"),
                "part_bbox_xywh": annotation.get("bbox"),
                "part_area_pixels": annotation.get("area"),
                "parent_annotation_id": (
                    int(parent_id) if parent_id is not None else None
                ),
            }
        )

    parents = {}
    if parent_ids:
        for annotation in iter_paco_array_objects(annotation_path, "annotations"):
            annotation_id = int(annotation["id"])
            if annotation_id in parent_ids:
                parents[annotation_id] = {
                    "category": category_name_by_id[int(annotation["category_id"])],
                    "segmentation": annotation.get("segmentation"),
                    "bbox_xywh": annotation.get("bbox"),
                }
    image_fields = ("id", "width", "height", "file_name", "coco_url", "license")
    images = {
        int(image["id"]): {
            name: image[name] for name in image_fields if name in image
        }
        for image in iter_paco_array_objects(annotation_path, "images")
        if int(image["id"]) in image_ids
    }
    if set(images) != image_ids:
        raise ValueError("one or more selected PACO image records are missing")

    records = []
    for hazard in hazards:
        parent_id = hazard.pop("parent_annotation_id")
        records.append(
            {
                **hazard,
                "image": images[int(hazard["image_id"])],
                "parent_object": parents.get(parent_id),
            }
        )
    split = next(
        (
            name
            for name in ("train", "val", "test")
            if f"_{name}" in Path(annotation_path).stem.lower()
        ),
        Path(annotation_path).stem,
    )
    return {
        "schema_version": PACO_HAZARD_MANIFEST_SCHEMA_VERSION,
        "split": split,
        "source": {
            "dataset": "PACO-LVIS v1",
            "annotation_path": str(Path(annotation_path)),
            "annotation_archive_sha256": PACO_LVIS_ANNOTATION_SHA256,
            "code_license": "MIT",
            "annotation_license_status": (
                "not_explicitly_stated_in_official_repository"
            ),
        },
        "evidence_scope": {
            "supported": ["visible_part_geometry_class", "part_mask"],
            "unsupported": [
                "physical_sharpness",
                "material_hardness",
                "contact_probability",
                "injury_severity",
                "route_orientation",
            ],
        },
        "geometry_classes": list(GENERAL_CONTACT_GEOMETRIES),
        "part_mapping": mapping,
        "record_count": len(records),
        "image_count": len(image_ids),
        "records": records,
    }


def write_paco_hazard_manifest(manifest: Mapping[str, object], output_path: Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def inventory_paco_contact_parts(
    annotation_paths: Iterable[Path],
    *,
    target_parts: Iterable[str] = ("knife:blade",),
) -> PacoContactInventoryReport:
    """Count supervised dangerous-part masks from official split metadata."""
    targets = tuple(sorted(set(target_parts)))
    if not targets or any(not isinstance(item, str) or not item for item in targets):
        raise ValueError("target parts must contain nonempty names")
    counts = []
    for path in sorted(Path(item) for item in annotation_paths):
        stem = path.stem.lower()
        split = next(
            (name for name in ("train", "val", "test") if f"_{name}" in stem),
            stem,
        )
        by_name = {str(item.get("name")): item for item in read_paco_categories(path)}
        for target in targets:
            if target not in by_name:
                raise ValueError(f"{path}: missing PACO category {target!r}")
            item = by_name[target]
            if item.get("supercategory") != "PART":
                raise ValueError(f"{target!r} must be a PACO part category")
            counts.append(
                PacoPartSplitCount(
                    split=split,
                    category_id=int(item["id"]),
                    category_name=target,
                    image_count=int(item["image_count"]),
                    instance_count=int(item["instance_count"]),
                )
            )
    return PacoContactInventoryReport(
        target_parts=targets,
        split_counts=tuple(counts),
        total_images=sum(item.image_count for item in counts),
        total_instances=sum(item.instance_count for item in counts),
    )


def write_paco_contact_inventory(
    report: PacoContactInventoryReport, output_path: Path
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": PACO_CONTACT_REPORT_SCHEMA_VERSION, **asdict(report)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
