import hashlib

import cv2
import numpy as np
import pytest

from robot.perception.hot_soup_collection import (
    collect_hot_soup_images,
    write_hot_soup_collection_manifest,
)
from robot.perception.hot_soup_dataset import (
    HotSoupBox,
    HotSoupSample,
    deterministic_capture_group_split,
    export_yolo_hot_soup_dataset,
    read_hot_soup_manifest,
    sha256_file,
    validate_hot_soup_dataset,
    write_hot_soup_manifest,
)


def write_image(path, value=0):
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample(tmp_path, **changes):
    image_path = tmp_path / "images" / "sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        write_image(image_path)
    values = dict(
        sample_id="session-a-frame-1",
        capture_group_id="session-a",
        split=deterministic_capture_group_split("session-a"),
        image_path="images/sample.png",
        image_sha256=sha256_file(image_path),
        width=16,
        height=12,
        domain="real_user_owned",
        source_reference="owned stationary kitchen clip 1",
        usage_rights="captured by project author; research use permitted",
        plume_state="present",
        container_state="present",
        boxes=(
            HotSoupBox(
                "visible_hot_soup_plume",
                0.4,
                0.1,
                0.6,
                0.5,
                "human_visual_annotation",
            ),
            HotSoupBox(
                "open_container",
                0.2,
                0.45,
                0.8,
                0.9,
                "human_visual_annotation",
            ),
        ),
    )
    values.update(changes)
    return HotSoupSample(**values)


def test_manifest_round_trip_and_validation(tmp_path):
    item = sample(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    write_hot_soup_manifest([item], manifest)
    restored = read_hot_soup_manifest(manifest)
    assert restored == (item,)
    report = validate_hot_soup_dataset(restored, tmp_path)
    assert report.valid
    assert report.positive_plume_count == 1


def test_explicit_negative_can_represent_smoke_without_hot_soup_context(tmp_path):
    item = sample(
        tmp_path,
        plume_state="absent",
        container_state="absent",
        boxes=(),
        hard_negative_tags=("smoke",),
    )
    assert item.hard_negative_tags == ("smoke",)


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"plume_state": "absent"}, "conflicts"),
        ({"container_state": "absent"}, "conflicts"),
        (
            {
                "boxes": (
                    HotSoupBox(
                        "visible_hot_soup_plume",
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                        "human_visual_annotation",
                    ),
                )
            },
            "conflicts",
        ),
        (
            {
                "hard_negative_tags": ("smoke",),
            },
            "absent hot-soup plume",
        ),
    ],
)
def test_incomplete_or_contradictory_annotations_are_rejected(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        sample(tmp_path, **changes)


def test_real_image_cannot_use_isaac_oracle_box(tmp_path):
    with pytest.raises(ValueError, match="Isaac oracle"):
        sample(
            tmp_path,
            boxes=(
                HotSoupBox(
                    "visible_hot_soup_plume",
                    0.4,
                    0.1,
                    0.6,
                    0.5,
                    "isaac_authored_oracle",
                ),
                HotSoupBox(
                    "open_container",
                    0.2,
                    0.45,
                    0.8,
                    0.9,
                    "isaac_authored_oracle",
                ),
            ),
        )


def test_validation_detects_group_duplicate_and_hash_leakage(tmp_path):
    first = sample(tmp_path, split="train")
    second = sample(
        tmp_path,
        sample_id="session-a-frame-2",
        split="test",
    )
    report = validate_hot_soup_dataset([first, second], tmp_path)
    assert report.capture_group_split_leaks == ("session-a",)
    assert report.duplicate_split_leaks == (first.image_sha256,)
    assert not report.valid

    changed = sample(
        tmp_path,
        image_sha256="0" * 64,
    )
    assert validate_hot_soup_dataset([changed], tmp_path).hash_mismatches


def test_yolo_export_writes_positive_and_empty_negative_labels(tmp_path):
    positive = sample(tmp_path, split="train")
    negative_image = tmp_path / "images" / "negative.png"
    negative_digest = write_image(negative_image, 255)
    negative = sample(
        tmp_path,
        sample_id="session-b-frame-1",
        capture_group_id="session-b",
        split="val",
        image_path="images/negative.png",
        image_sha256=negative_digest,
        plume_state="absent",
        container_state="absent",
        boxes=(),
        hard_negative_tags=("white_background",),
    )
    output = tmp_path / "yolo"
    config = export_yolo_hot_soup_dataset([positive, negative], tmp_path, output)
    positive_label = output / "labels" / "train" / "session-a-frame-1.txt"
    negative_label = output / "labels" / "val" / "session-b-frame-1.txt"
    assert len(positive_label.read_text(encoding="utf-8").splitlines()) == 2
    assert negative_label.read_text(encoding="utf-8") == ""
    assert "1: open_container" in config.read_text(encoding="utf-8")


def test_collection_copies_deduplicates_and_stays_pending(tmp_path):
    source = tmp_path / "source.png"
    write_image(source, 80)
    root = tmp_path / "dataset"
    items = collect_hot_soup_images(
        [source, source],
        root,
        capture_group_id="owned-session-1",
        domain="real_user_owned",
        source_kind="real_image",
        source_reference="author camera session 1",
        usage_rights="captured by project author; research use permitted",
    )
    assert len(items) == 1
    assert items[0].annotation_status == "pending"
    assert (root / items[0].image_path).is_file()
    manifest = root / "collection.jsonl"
    write_hot_soup_collection_manifest(items, manifest)
    assert '"annotation_status": "pending"' in manifest.read_text(encoding="utf-8")


def test_capture_group_split_is_stable():
    assert deterministic_capture_group_split(
        "session-one"
    ) == deterministic_capture_group_split("session-one")
