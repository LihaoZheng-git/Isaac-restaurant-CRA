"""Prepare an auditable ABO subset for offline visual-mass experiments.

ABO catalog weight and dimensions are external metadata claims.  They are
useful as offline training/evaluation labels, but they are neither robot sensor
measurements nor runtime CRA evidence.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Mapping, Sequence

from robot.perception.visual_mass_dataset import deterministic_object_split


ABO_SUBSET_SCHEMA_VERSION = 1
ABO_DATASET_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/index.html"
ABO_LICENSE = "CC BY 4.0"
ABO_MASS_PROVENANCE = "external_catalog_metadata"
ABO_DIMENSION_PROVENANCE = "external_catalog_metadata"
ABO_EVIDENCE_ROLE = "offline_training_or_evaluation_label"
ABO_RUNTIME_EVIDENCE_STATUS = "unavailable"

# Exact matches avoid errors such as treating device covers as drinking cups.
DEFAULT_RESTAURANT_PRODUCT_TYPES: Mapping[str, tuple[str, ...]] = {
    "chair": ("CHAIR", "BEAN_BAG_CHAIR"),
    "stool": ("STOOL_SEATING",),
    "table": ("TABLE",),
    "cup": ("DRINKING_CUP", "MEASURING_CUP"),
    "plate": ("DISHWARE_PLATE",),
    "bowl": ("DISHWARE_BOWL",),
    "bottle": ("BOTTLE",),
    "pan": ("SAUTE_FRY_PAN", "BAKING_PAN", "SHEET_PAN"),
    "knife": ("KITCHEN_KNIFE",),
}

_MASS_TO_KG = {
    "grams": 0.001,
    "kilograms": 1.0,
    "ounces": 0.028349523125,
    "pounds": 0.45359237,
}
_LENGTH_TO_M = {
    "centimeters": 0.01,
    "inches": 0.0254,
    "meters": 1.0,
    "millimeters": 0.001,
}


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _relative_path(value: object, name: str) -> str:
    text = _nonempty_text(value, name).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must stay inside the dataset root")
    return str(path)


@dataclass(frozen=True)
class AboMassCandidate:
    """One selected catalog image with offline physical-property labels."""

    object_id: str
    item_id: str
    domain_name: str
    semantic_class: str
    product_type: str
    image_id: str
    image_path: str
    mass_kg: float
    width_m: float
    depth_m: float
    height_m: float
    split: str
    mass_provenance: str = ABO_MASS_PROVENANCE
    dimension_provenance: str = ABO_DIMENSION_PROVENANCE
    evidence_role: str = ABO_EVIDENCE_ROLE
    runtime_evidence_status: str = ABO_RUNTIME_EVIDENCE_STATUS
    dataset_url: str = ABO_DATASET_URL
    license: str = ABO_LICENSE

    def __post_init__(self) -> None:
        for name in (
            "object_id",
            "item_id",
            "domain_name",
            "semantic_class",
            "product_type",
            "image_id",
            "mass_provenance",
            "dimension_provenance",
            "evidence_role",
            "runtime_evidence_status",
            "dataset_url",
            "license",
        ):
            object.__setattr__(self, name, _nonempty_text(getattr(self, name), name))
        object.__setattr__(self, "image_path", _relative_path(self.image_path, "image_path"))
        for name in ("mass_kg", "width_m", "depth_m", "height_m"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if self.mass_provenance != ABO_MASS_PROVENANCE:
            raise ValueError("ABO mass must remain labeled as external catalog metadata")
        if self.dimension_provenance != ABO_DIMENSION_PROVENANCE:
            raise ValueError("ABO dimensions must remain external catalog metadata")
        if self.evidence_role != ABO_EVIDENCE_ROLE:
            raise ValueError("ABO labels are restricted to offline model work")
        if self.runtime_evidence_status != ABO_RUNTIME_EVIDENCE_STATUS:
            raise ValueError("ABO labels are unavailable as runtime evidence")


@dataclass(frozen=True)
class AboSelectionReport:
    listings_scanned: int
    complete_physical_records: int
    restaurant_records: int
    selected_records: int
    missing_image_index_records: int
    class_counts: tuple[tuple[str, int], ...]
    split_counts: tuple[tuple[str, int], ...]
    rejection_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _PhysicalListing:
    object_id: str
    item_id: str
    domain_name: str
    semantic_class: str
    product_type: str
    image_id: str
    mass_kg: float
    width_m: float
    depth_m: float
    height_m: float


def iter_abo_listings(paths: Iterable[Path]) -> Iterator[dict[str, object]]:
    """Yield objects from ABO's gzip-compressed JSON Lines shards."""
    for path in sorted(Path(item) for item in paths):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield value


def _normalized_quantity(value: object, conversions: Mapping[str, float]) -> float:
    if not isinstance(value, Mapping):
        raise ValueError("quantity must be an object")
    normalized = value.get("normalized_value")
    quantity = normalized if isinstance(normalized, Mapping) else value
    unit = str(quantity.get("unit", "")).strip().lower()
    if unit not in conversions:
        raise ValueError(f"unsupported unit: {unit or '<missing>'}")
    return _positive(quantity.get("value"), "quantity") * conversions[unit]


def _product_types(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip().upper(),) if value.strip() else ()
    if not isinstance(value, Sequence):
        return ()
    result = []
    for item in value:
        raw = item.get("value") if isinstance(item, Mapping) else item
        if isinstance(raw, str) and raw.strip():
            result.append(raw.strip().upper())
    return tuple(result)


def _category_lookup(
    categories: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for semantic_class, product_types in categories.items():
        label = _nonempty_text(semantic_class, "semantic class")
        for product_type in product_types:
            normalized = _nonempty_text(product_type, "product type").upper()
            if normalized in result:
                raise ValueError(f"duplicate product type mapping: {normalized}")
            result[normalized] = label
    return result


def _parse_listing(
    item: Mapping[str, object], category_by_type: Mapping[str, str]
) -> tuple[_PhysicalListing | None, str]:
    matched = next(
        (
            (product_type, category_by_type[product_type])
            for product_type in _product_types(item.get("product_type"))
            if product_type in category_by_type
        ),
        None,
    )
    if matched is None:
        return None, "outside_selected_categories"
    product_type, semantic_class = matched
    try:
        item_id = _nonempty_text(item.get("item_id"), "item_id")
        domain_name = _nonempty_text(item.get("domain_name"), "domain_name")
        image_id = _nonempty_text(item.get("main_image_id"), "main_image_id")
        weights = item.get("item_weight")
        if not isinstance(weights, Sequence) or not weights:
            return None, "missing_mass"
        mass_kg = _normalized_quantity(weights[0], _MASS_TO_KG)
        dimensions = item.get("item_dimensions")
        if not isinstance(dimensions, Mapping):
            return None, "missing_dimensions"
        width_m = _normalized_quantity(dimensions.get("width"), _LENGTH_TO_M)
        depth_m = _normalized_quantity(dimensions.get("length"), _LENGTH_TO_M)
        height_m = _normalized_quantity(dimensions.get("height"), _LENGTH_TO_M)
    except (TypeError, ValueError):
        return None, "invalid_or_unsupported_physical_value"
    return (
        _PhysicalListing(
            object_id=f"{domain_name}:{item_id}",
            item_id=item_id,
            domain_name=domain_name,
            semantic_class=semantic_class,
            product_type=product_type,
            image_id=image_id,
            mass_kg=mass_kg,
            width_m=width_m,
            depth_m=depth_m,
            height_m=height_m,
        ),
        "accepted",
    )


def load_abo_image_paths(
    image_metadata_path: Path, image_ids: Iterable[str] | None = None
) -> dict[str, str]:
    """Resolve selected image IDs, or load the full official path index."""
    requested = None if image_ids is None else set(image_ids)
    result: dict[str, str] = {}
    with gzip.open(image_metadata_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row.get("image_id", "")
            if requested is None or image_id in requested:
                result[image_id] = _relative_path(row.get("path"), "image path")
    return result


def select_abo_mass_subset(
    listings: Iterable[Mapping[str, object]],
    image_paths: Mapping[str, str],
    *,
    categories: Mapping[str, Sequence[str]] = DEFAULT_RESTAURANT_PRODUCT_TYPES,
    max_objects_per_class: int = 200,
    salt: str = "cra_abo_visual_mass_v1",
) -> tuple[tuple[AboMassCandidate, ...], AboSelectionReport]:
    """Select a deterministic, class-capped subset with object-level splits."""
    if (
        isinstance(max_objects_per_class, bool)
        or not isinstance(max_objects_per_class, int)
        or max_objects_per_class <= 0
    ):
        raise ValueError("max_objects_per_class must be a positive integer")
    category_by_type = _category_lookup(categories)
    rejections: Counter[str] = Counter()
    grouped: dict[str, list[_PhysicalListing]] = {
        semantic_class: [] for semantic_class in categories
    }
    scanned = 0
    restaurant_records = 0
    complete_records = 0
    for item in listings:
        scanned += 1
        if any(
            product_type in category_by_type
            for product_type in _product_types(item.get("product_type"))
        ):
            restaurant_records += 1
        parsed, reason = _parse_listing(item, category_by_type)
        if parsed is None:
            rejections[reason] += 1
            continue
        complete_records += 1
        grouped[parsed.semantic_class].append(parsed)

    chosen: list[_PhysicalListing] = []
    for semantic_class in sorted(grouped):
        ranked = sorted(
            grouped[semantic_class],
            key=lambda item: hashlib.sha256(
                f"selection:{salt}:{item.object_id}".encode()
            ).digest(),
        )
        chosen.extend(ranked[:max_objects_per_class])

    candidates = []
    missing_images = 0
    for item in sorted(chosen, key=lambda value: value.object_id):
        path = image_paths.get(item.image_id)
        if path is None:
            missing_images += 1
            continue
        candidates.append(
            AboMassCandidate(
                **asdict(item),
                image_path=path,
                split=deterministic_object_split(
                    item.object_id, salt=f"split:{salt}"
                ),
            )
        )
    class_counts = Counter(item.semantic_class for item in candidates)
    split_counts = Counter(item.split for item in candidates)
    report = AboSelectionReport(
        listings_scanned=scanned,
        complete_physical_records=complete_records,
        restaurant_records=restaurant_records,
        selected_records=len(candidates),
        missing_image_index_records=missing_images,
        class_counts=tuple(sorted(class_counts.items())),
        split_counts=tuple(
            (name, split_counts.get(name, 0))
            for name in ("train", "validation", "test")
        ),
        rejection_counts=tuple(sorted(rejections.items())),
    )
    return tuple(candidates), report


def write_abo_subset_manifest(
    candidates: Iterable[AboMassCandidate], manifest_path: Path
) -> None:
    """Write a deterministic JSON Lines manifest with explicit provenance."""
    items = sorted(tuple(candidates), key=lambda item: item.object_id)
    if len({item.object_id for item in items}) != len(items):
        raise ValueError("object_id values must be unique")
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(
                json.dumps(
                    {"schema_version": ABO_SUBSET_SCHEMA_VERSION, **asdict(item)},
                    sort_keys=True,
                )
                + "\n"
            )


def write_abo_selection_report(
    report: AboSelectionReport, report_path: Path
) -> None:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
