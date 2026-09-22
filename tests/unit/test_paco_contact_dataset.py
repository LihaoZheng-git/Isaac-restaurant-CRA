import json

import pytest

from robot.perception.paco_contact_dataset import (
    PACO_LVIS_ANNOTATION_SHA256,
    inventory_paco_contact_parts,
    read_paco_categories,
    write_paco_contact_inventory,
)


def annotation(path, split, *, name="knife:blade", supercategory="PART"):
    target = path / f"paco_lvis_v1_{split}.json"
    target.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "description": 'the word "categories"'}],
                "annotations": [{"id": 2, "segmentation": [[0, 1, 2, 3]]}],
                "categories": [
                    {
                        "id": 2041,
                        "name": name,
                        "supercategory": supercategory,
                        "image_count": 10,
                        "instance_count": 12,
                    }
                ],
                "attributes": [],
            }
        ),
        encoding="utf-8",
    )
    return target


def test_inventory_reads_bounded_category_metadata_and_sums_splits(tmp_path):
    paths = [annotation(tmp_path, split) for split in ("train", "val", "test")]
    report = inventory_paco_contact_parts(paths)
    assert report.total_images == 30
    assert report.total_instances == 36
    assert {item.split for item in report.split_counts} == {"train", "val", "test"}
    assert report.archive_sha256 == PACO_LVIS_ANNOTATION_SHA256
    assert report.annotation_license_status.startswith("not_explicitly")
    assert report.runtime_model_status == "unavailable"


def test_category_extractor_ignores_key_text_inside_strings(tmp_path):
    path = annotation(tmp_path, "val")
    categories = read_paco_categories(path)
    assert categories[0]["name"] == "knife:blade"


def test_missing_or_nonpart_target_is_rejected(tmp_path):
    missing = annotation(tmp_path, "train", name="knife:handle")
    with pytest.raises(ValueError, match="missing"):
        inventory_paco_contact_parts([missing])
    wrong = annotation(tmp_path, "val", supercategory="OBJECT")
    with pytest.raises(ValueError, match="part category"):
        inventory_paco_contact_parts([wrong])


def test_report_writer_records_schema_and_provenance(tmp_path):
    report = inventory_paco_contact_parts([annotation(tmp_path, "test")])
    output = tmp_path / "report.json"
    write_paco_contact_inventory(report, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["code_license"] == "MIT"
    assert payload["runtime_model_status"] == "unavailable"
