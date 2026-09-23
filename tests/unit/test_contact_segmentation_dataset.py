import json

import numpy as np
import pytest

from robot.perception.contact_segmentation_dataset import (
    decode_coco_rle,
    segmentation_to_yolo_polygons,
    write_yolo_contact_split,
    write_yolo_dataset_yaml,
)


def compress_counts(counts):
    output = []
    for index, original in enumerate(counts):
        value = original - counts[index - 2] if index > 2 else original
        while True:
            char = value & 0x1F
            value >>= 5
            more = value != (-1 if char & 0x10 else 0)
            if more:
                char |= 0x20
            output.append(chr(char + 48))
            if not more:
                break
    return "".join(output)


def mask_counts(mask):
    flat = mask.reshape(-1, order="F")
    counts = []
    value = False
    run = 0
    for item in flat:
        if bool(item) == value:
            run += 1
        else:
            counts.append(run)
            run = 1
            value = not value
    counts.append(run)
    return counts


def test_decode_compressed_coco_rle_round_trip():
    expected = np.zeros((5, 6), dtype=np.bool_)
    expected[1:4, 2:5] = True
    counts = mask_counts(expected)
    actual = decode_coco_rle(
        {"size": [5, 6], "counts": compress_counts(counts)}
    )
    assert np.array_equal(actual, expected)


def test_decode_rejects_bad_coverage():
    with pytest.raises(ValueError, match="do not cover"):
        decode_coco_rle({"size": [2, 2], "counts": [1, 1]})


def test_rle_and_polygon_inputs_become_normalized_polygons():
    mask = np.zeros((10, 20), dtype=np.bool_)
    mask[2:8, 5:15] = True
    rle = {"size": [10, 20], "counts": mask_counts(mask)}
    decoded = segmentation_to_yolo_polygons(rle, width=20, height=10)
    direct = segmentation_to_yolo_polygons(
        [[5, 2, 15, 2, 15, 8, 5, 8]], width=20, height=10
    )
    assert decoded and direct
    assert all(0 <= value <= 1 for value in decoded[0])
    assert direct[0] == (0.25, 0.2, 0.75, 0.2, 0.75, 0.8, 0.25, 0.8)


def test_write_yolo_split_and_config(tmp_path):
    manifest = {
        "split": "val",
        "records": [
            {
                "image_id": 7,
                "geometry_class": "exposed_edge",
                "part_segmentation": [[1, 1, 8, 1, 8, 3]],
                "image": {"width": 10, "height": 5},
            }
        ],
    }
    manifest_path = tmp_path / "val.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = write_yolo_contact_split(manifest_path, tmp_path / "dataset")
    assert report == {"images": 1, "polygons": 1, "skipped_records": 0}
    label = tmp_path / "dataset" / "labels" / "val" / "000000000007.txt"
    assert label.read_text(encoding="utf-8").startswith("0 ")
    config = write_yolo_dataset_yaml(tmp_path / "dataset", ["train", "val"])
    assert "3: jagged_surface" in config.read_text(encoding="utf-8")
