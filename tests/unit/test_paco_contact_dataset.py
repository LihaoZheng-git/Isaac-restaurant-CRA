import json

import pytest

from robot.perception.paco_contact_dataset import (
    DEFAULT_PACO_HAZARD_PARTS,
    PACO_LVIS_ANNOTATION_SHA256,
    build_paco_hazard_manifest,
    inventory_paco_contact_parts,
    iter_paco_array_objects,
    read_paco_categories,
    write_paco_hazard_manifest,
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


def hazard_annotation(path):
    target = path / "paco_lvis_v1_val.json"
    target.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 7,
                        "width": 20,
                        "height": 10,
                        "coco_url": "https://example.invalid/image.jpg",
                    }
                ],
                "annotations": [
                    {
                        "id": 10,
                        "image_id": 7,
                        "category_id": 615,
                        "bbox": [1, 2, 8, 4],
                        "segmentation": [[1, 2, 9, 2, 9, 6]],
                    },
                    {
                        "id": 11,
                        "image_id": 7,
                        "category_id": 2041,
                        "obj_ann_id": 10,
                        "bbox": [2, 3, 6, 1],
                        "area": 6,
                        "segmentation": {"size": [10, 20], "counts": "abc"},
                    },
                ],
                "categories": [
                    {"id": 615, "name": "knife", "supercategory": "OBJECT"},
                    {"id": 2041, "name": "knife:blade", "supercategory": "PART"},
                    {
                        "id": 2045,
                        "name": "scissors:blade",
                        "supercategory": "PART",
                    },
                    {
                        "id": 2421,
                        "name": "screwdriver:tip",
                        "supercategory": "PART",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return target


def test_streaming_array_reader_handles_nested_arrays_and_objects(tmp_path):
    path = hazard_annotation(tmp_path)
    annotations = list(iter_paco_array_objects(path, "annotations"))
    assert [item["id"] for item in annotations] == [10, 11]
    assert annotations[1]["segmentation"]["size"] == [10, 20]


def test_hazard_manifest_preserves_geometry_and_explicit_limits(tmp_path):
    path = hazard_annotation(tmp_path)
    manifest = build_paco_hazard_manifest(path)
    assert manifest["schema_version"] == 1
    assert manifest["split"] == "val"
    assert manifest["record_count"] == 1
    assert manifest["image_count"] == 1
    assert manifest["part_mapping"] == DEFAULT_PACO_HAZARD_PARTS
    record = manifest["records"][0]
    assert record["geometry_class"] == "exposed_edge"
    assert record["source_part_category"] == "knife:blade"
    assert record["parent_object"]["category"] == "knife"
    assert "physical_sharpness" in manifest["evidence_scope"]["unsupported"]


def test_hazard_manifest_rejects_unsupported_geometry(tmp_path):
    with pytest.raises(ValueError, match="unsupported contact geometries"):
        build_paco_hazard_manifest(
            hazard_annotation(tmp_path), part_mapping={"knife:blade": "knife"}
        )


def test_hazard_manifest_writer_records_provenance(tmp_path):
    manifest = build_paco_hazard_manifest(hazard_annotation(tmp_path))
    output = tmp_path / "hazards.json"
    write_paco_hazard_manifest(manifest, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source"]["annotation_archive_sha256"] == (
        PACO_LVIS_ANNOTATION_SHA256
    )
