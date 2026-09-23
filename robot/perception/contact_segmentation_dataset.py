"""Convert provenance-rich contact manifests into segmentation training data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from robot.perception.paco_contact_dataset import GENERAL_CONTACT_GEOMETRIES


def decode_coco_rle(segmentation: Mapping[str, object]) -> np.ndarray:
    """Decode an uncompressed or compressed COCO RLE mask without pycocotools."""
    size = segmentation.get("size")
    encoded = segmentation.get("counts")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in size)
        or any(value <= 0 for value in size)
    ):
        raise ValueError("RLE size must contain positive integer height and width")
    if isinstance(encoded, list):
        counts = encoded
    elif isinstance(encoded, str):
        counts = []
        cursor = 0
        while cursor < len(encoded):
            value = 0
            shift = 0
            while True:
                if cursor >= len(encoded):
                    raise ValueError("compressed RLE count is truncated")
                char = ord(encoded[cursor]) - 48
                cursor += 1
                if not 0 <= char <= 63:
                    raise ValueError("compressed RLE contains an invalid character")
                value |= (char & 0x1F) << shift
                shift += 5
                if not char & 0x20:
                    if char & 0x10:
                        value |= -1 << shift
                    break
            if len(counts) > 2:
                value += counts[-2]
            counts.append(value)
    else:
        raise ValueError("RLE counts must be a list or compressed string")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("RLE counts must be nonnegative integers")
    height, width = size
    if sum(counts) != height * width:
        raise ValueError("RLE counts do not cover the declared mask size")
    flat = np.zeros(height * width, dtype=np.bool_)
    offset = 0
    foreground = False
    for count in counts:
        if foreground:
            flat[offset : offset + count] = True
        offset += count
        foreground = not foreground
    return flat.reshape((height, width), order="F")


def segmentation_to_yolo_polygons(
    segmentation: object, *, width: int, height: int
) -> tuple[tuple[float, ...], ...]:
    """Convert PACO polygon/RLE segmentation into normalized exterior contours."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if isinstance(segmentation, dict):
        mask = decode_coco_rle(segmentation)
        if mask.shape != (height, width):
            raise ValueError("RLE dimensions do not match image metadata")
        import cv2

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons = [contour.reshape(-1, 2).astype(float) for contour in contours]
    elif isinstance(segmentation, list):
        polygons = []
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                continue
            polygons.append(np.asarray(polygon, dtype=float).reshape(-1, 2))
    else:
        raise ValueError("segmentation must be polygons or COCO RLE")
    normalized = []
    for polygon in polygons:
        if len(polygon) < 3 or not np.isfinite(polygon).all():
            continue
        polygon[:, 0] = np.clip(polygon[:, 0] / width, 0, 1)
        polygon[:, 1] = np.clip(polygon[:, 1] / height, 0, 1)
        values = tuple(float(value) for value in polygon.reshape(-1))
        if len(values) >= 6:
            normalized.append(values)
    return tuple(normalized)


def write_yolo_contact_split(
    manifest_path: Path, dataset_root: Path
) -> dict[str, int]:
    """Write YOLO segmentation labels for one PACO manifest split."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    split = str(manifest["split"])
    label_root = Path(dataset_root) / "labels" / split
    label_root.mkdir(parents=True, exist_ok=True)
    by_image: dict[int, list[str]] = {}
    skipped = 0
    polygon_count = 0
    for record in manifest["records"]:
        image = record["image"]
        image_id = int(record["image_id"])
        class_id = GENERAL_CONTACT_GEOMETRIES.index(record["geometry_class"])
        polygons = segmentation_to_yolo_polygons(
            record["part_segmentation"],
            width=int(image["width"]),
            height=int(image["height"]),
        )
        if not polygons:
            skipped += 1
            continue
        lines = by_image.setdefault(image_id, [])
        for polygon in polygons:
            coords = " ".join(f"{value:.7f}" for value in polygon)
            lines.append(f"{class_id} {coords}")
            polygon_count += 1
    for image_id, lines in by_image.items():
        (label_root / f"{image_id:012d}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return {
        "images": len(by_image),
        "polygons": polygon_count,
        "skipped_records": skipped,
    }


def write_yolo_dataset_yaml(dataset_root: Path, splits: Iterable[str]) -> Path:
    root = Path(dataset_root).resolve()
    path = root / "dataset.yaml"
    lines = [f"path: {root.as_posix()}"]
    lines.extend(f"{split}: images/{split}" for split in splits)
    lines.append("names:")
    lines.extend(
        f"  {index}: {name}" for index, name in enumerate(GENERAL_CONTACT_GEOMETRIES)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
