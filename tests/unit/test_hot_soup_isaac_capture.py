from pathlib import Path

import pytest

from robot.perception.hot_soup_dataset import validate_hot_soup_dataset
from robot.perception.hot_soup_isaac_capture import (
    build_isaac_hot_soup_sample,
    oracle_boxes_from_replicator,
)


def test_replicator_boxes_are_clipped_normalized_and_unioned_by_class():
    records = [
        {"semanticId": 4, "x_min": -5, "y_min": 10, "x_max": 40, "y_max": 50},
        {"semanticId": 4, "x_min": 30, "y_min": 20, "x_max": 70, "y_max": 90},
        {"semanticId": 9, "x_min": 20, "y_min": 70, "x_max": 80, "y_max": 110},
        {"semanticId": 12, "x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2},
    ]
    boxes = oracle_boxes_from_replicator(
        records,
        {"4": {"class": "visible_hot_soup_plume"}, 9: "open_container", 12: {"class": "table"}},
        width=100,
        height=100,
    )

    assert [box.class_name for box in boxes] == [
        "visible_hot_soup_plume",
        "open_container",
    ]
    assert (boxes[0].x_min, boxes[0].y_min, boxes[0].x_max, boxes[0].y_max) == (
        0.0,
        0.1,
        0.7,
        0.9,
    )
    assert boxes[1].y_max == 1.0
    assert all(box.annotation_source == "isaac_authored_oracle" for box in boxes)


def test_missing_semantic_class_is_not_fabricated():
    boxes = oracle_boxes_from_replicator(
        [{"semanticId": 7, "xMin": 1, "yMin": 2, "xMax": 5, "yMax": 8}],
        {7: {"class": "unrelated"}},
        width=10,
        height=10,
    )
    assert boxes == ()


def test_capture_dimensions_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        oracle_boxes_from_replicator([], {}, width=0, height=10)


def test_synthetic_sample_records_oracle_provenance(tmp_path: Path):
    image = tmp_path / "images" / "capture.png"
    image.parent.mkdir()
    image.write_bytes(b"rendered pixels")
    boxes = oracle_boxes_from_replicator(
        [
            {"semanticId": 1, "x_min": 10, "y_min": 5, "x_max": 30, "y_max": 35},
            {"semanticId": 2, "x_min": 8, "y_min": 30, "x_max": 35, "y_max": 45},
        ],
        {1: "visible_hot_soup_plume", 2: "open_container"},
        width=50,
        height=50,
    )
    sample = build_isaac_hot_soup_sample(
        image,
        tmp_path,
        boxes,
        sample_id="isaac-hot-soup-0001",
        capture_group_id="isaac-hot-soup-stationary-v1",
        width=50,
        height=50,
        source_reference="hot_soup_plume_proxy.usda",
    )

    assert sample.domain == "isaac_synthetic"
    assert sample.plume_state == sample.container_state == "present"
    assert sample.usage_rights == "project-authored Isaac synthetic scene"
    assert validate_hot_soup_dataset([sample], tmp_path).valid


def test_synthetic_sample_rejects_image_outside_dataset_root(tmp_path: Path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"pixels")
    root = tmp_path / "dataset"
    root.mkdir()
    with pytest.raises(ValueError, match="inside dataset_root"):
        build_isaac_hot_soup_sample(
            outside,
            root,
            (),
            sample_id="sample",
            capture_group_id="group",
            width=10,
            height=10,
            source_reference="scene.usda",
        )
