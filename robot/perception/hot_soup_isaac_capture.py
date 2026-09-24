"""Convert stationary Isaac Replicator output into hot-soup dataset records.

The boxes produced here are authored-scene oracle annotations. They are not
detector measurements, temperature evidence, or evidence that a plume is steam.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from robot.perception.hot_soup_dataset import (
    HOT_SOUP_DETECTION_CLASSES,
    HotSoupBox,
    HotSoupSample,
    deterministic_capture_group_split,
    sha256_file,
)


_COORDINATE_FIELDS = {
    "x_min": ("x_min", "xMin"),
    "y_min": ("y_min", "yMin"),
    "x_max": ("x_max", "xMax"),
    "y_max": ("y_max", "yMax"),
}


def _value(record: Any, names: tuple[str, ...]) -> Any:
    if isinstance(record, Mapping):
        for name in names:
            if name in record:
                return record[name]
    fields = getattr(getattr(record, "dtype", None), "names", ()) or ()
    for name in names:
        if name in fields:
            return record[name]
    raise ValueError(f"Isaac box record is missing one of {names}")


def _class_name(label: Any) -> str | None:
    if isinstance(label, str):
        return label
    if isinstance(label, Mapping):
        value = label.get("class")
        return value if isinstance(value, str) else None
    return None


def oracle_boxes_from_replicator(
    records: Iterable[Any],
    id_to_labels: Mapping[Any, Any],
    *,
    width: int,
    height: int,
) -> tuple[HotSoupBox, ...]:
    """Union visible Replicator boxes by the two fixed project classes."""

    if width <= 0 or height <= 0:
        raise ValueError("capture dimensions must be positive")
    labels = {int(key): _class_name(value) for key, value in id_to_labels.items()}
    extents: dict[str, list[float]] = {}
    for record in records:
        semantic_id = int(_value(record, ("semanticId", "semantic_id")))
        class_name = labels.get(semantic_id)
        if class_name not in HOT_SOUP_DETECTION_CLASSES:
            continue
        coordinates = {
            name: float(_value(record, aliases))
            for name, aliases in _COORDINATE_FIELDS.items()
        }
        x_min = max(0.0, min(float(width), coordinates["x_min"]))
        y_min = max(0.0, min(float(height), coordinates["y_min"]))
        x_max = max(0.0, min(float(width), coordinates["x_max"]))
        y_max = max(0.0, min(float(height), coordinates["y_max"]))
        if x_min >= x_max or y_min >= y_max:
            continue
        if class_name not in extents:
            extents[class_name] = [x_min, y_min, x_max, y_max]
        else:
            union = extents[class_name]
            union[0] = min(union[0], x_min)
            union[1] = min(union[1], y_min)
            union[2] = max(union[2], x_max)
            union[3] = max(union[3], y_max)
    return tuple(
        HotSoupBox(
            class_name=class_name,
            x_min=extent[0] / width,
            y_min=extent[1] / height,
            x_max=extent[2] / width,
            y_max=extent[3] / height,
            annotation_source="isaac_authored_oracle",
        )
        for class_name in HOT_SOUP_DETECTION_CLASSES
        if (extent := extents.get(class_name)) is not None
    )


def build_isaac_hot_soup_sample(
    image_path: Path,
    dataset_root: Path,
    boxes: Iterable[HotSoupBox],
    *,
    sample_id: str,
    capture_group_id: str,
    width: int,
    height: int,
    source_reference: str,
) -> HotSoupSample:
    """Build an auditable synthetic sample after its rendered image is written."""

    image = Path(image_path).resolve()
    root = Path(dataset_root).resolve()
    try:
        relative = image.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("image_path must stay inside dataset_root") from error
    items = tuple(boxes)
    present = {box.class_name for box in items}
    return HotSoupSample(
        sample_id=sample_id,
        capture_group_id=capture_group_id,
        split=deterministic_capture_group_split(capture_group_id),
        image_path=relative,
        image_sha256=sha256_file(image),
        width=width,
        height=height,
        domain="isaac_synthetic",
        source_reference=source_reference,
        usage_rights="project-authored Isaac synthetic scene",
        plume_state=(
            "present" if "visible_hot_soup_plume" in present else "absent"
        ),
        container_state="present" if "open_container" in present else "absent",
        boxes=items,
    )
