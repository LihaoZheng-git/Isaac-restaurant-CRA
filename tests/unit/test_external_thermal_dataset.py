import json

import cv2
import numpy as np

from robot.perception.external_thermal_dataset import (
    audit_external_thermal_yolo,
    external_capture_group_id,
    import_external_thermal_images,
    read_yolo_class_names,
    write_external_thermal_audit,
    write_external_thermal_collection,
)


def write_image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((12, 16, 3), value, dtype=np.uint8))


def fixture_export(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "names:\n"
        "  0: Steam\n"
        "  1: open_pot\n"
        "  2: open_pot_boiling\n"
        "  3: person\n",
        encoding="utf-8",
    )
    examples = (
        (
            "train",
            "steam1_mp4-227_jpg.rf.aaa.jpg",
            "0 0.5 0.3 0.2 0.2\n",
        ),
        (
            "valid",
            "steam1_mp4-224_jpg.rf.bbb.jpg",
            "1 0.5 0.7 0.5 0.3\n",
        ),
        (
            "test",
            "other_clip-10_jpg.rf.ccc.jpg",
            "0 0.5 0.3 0.2 0.2\n2 0.5 0.7 0.5 0.3\n3 0.1 0.1 0.1 0.1\n",
        ),
    )
    for index, (split, name, labels) in enumerate(examples):
        image = root / split / "images" / name
        write_image(image, index * 50)
        label = root / split / "labels" / f"{image.stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(labels, encoding="utf-8")


def test_audit_reconstructs_groups_and_rejects_published_split_as_evaluation(tmp_path):
    fixture_export(tmp_path)
    report = audit_external_thermal_yolo(tmp_path)
    assert report.image_count == 3
    assert report.labeled_image_count == 3
    assert report.missing_label_file_count == 0
    assert report.source_group_count == 2
    assert len(report.published_split_leak_groups) == 1
    assert report.plume_without_source_count == 1
    assert report.unsupported_class_box_counts == (("person", 1),)
    assert report.annotation_status == "pending_project_relabel"
    assert report.evaluation_status.startswith("not_eligible")
    assert dict(report.class_box_counts)["steam"] == 2


def test_external_import_is_pending_and_training_only(tmp_path):
    source = tmp_path / "source"
    fixture_export(source)
    dataset = tmp_path / "dataset"
    report, items = import_external_thermal_images(
        source,
        dataset,
        source_reference="Kitchen Safety System v4 candidate",
        usage_rights="candidate CC BY 4.0; uploader-rights verification pending",
    )
    assert report.image_count == len(items) == 3
    assert {item.domain for item in items} == {"external_training_candidate"}
    assert {item.suggested_split for item in items} == {"train"}
    assert {item.source_kind for item in items} == {"external_dataset_image"}
    assert {item.annotation_status for item in items} == {"pending"}
    assert all((dataset / item.image_path).is_file() for item in items)

    audit_path = dataset / "audit.json"
    collection_path = dataset / "collection.jsonl"
    write_external_thermal_audit(report, audit_path)
    write_external_thermal_collection(items, collection_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["schema_version"] == 1
    assert "published_split_leak_groups" in audit
    assert '"annotation_status": "pending"' in collection_path.read_text(
        encoding="utf-8"
    )


def test_group_id_and_inline_class_names_are_stable(tmp_path):
    config = tmp_path / "data.yaml"
    config.write_text("names: ['Steam', 'open_pot']\n", encoding="utf-8")
    assert read_yolo_class_names(config) == ("Steam", "open_pot")
    first = external_capture_group_id(
        tmp_path / "höyry1_mp4-227_jpg.rf.aaa.jpg"
    )
    second = external_capture_group_id(
        tmp_path / "höyry1_mp4-224_jpg.rf.bbb.jpg"
    )
    assert first == second
