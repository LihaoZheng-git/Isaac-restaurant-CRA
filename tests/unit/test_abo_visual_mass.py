import csv
import gzip
import json

import pytest

from robot.perception.abo_visual_mass import (
    ABO_EVIDENCE_ROLE,
    ABO_MASS_PROVENANCE,
    ABO_RUNTIME_EVIDENCE_STATUS,
    AboMassCandidate,
    iter_abo_listings,
    load_abo_image_paths,
    read_abo_subset_manifest,
    select_abo_mass_subset,
    write_abo_subset_manifest,
)


def listing(item_id, product_type="CHAIR", image_id=None):
    return {
        "item_id": item_id,
        "domain_name": "amazon.com",
        "product_type": [{"value": product_type}],
        "main_image_id": image_id or f"image-{item_id}",
        "item_weight": [
            {"normalized_value": {"unit": "pounds", "value": 10}}
        ],
        "item_dimensions": {
            "width": {"normalized_value": {"unit": "inches", "value": 20}},
            "length": {"normalized_value": {"unit": "inches", "value": 10}},
            "height": {"normalized_value": {"unit": "inches", "value": 30}},
        },
    }


def test_selection_converts_units_caps_classes_and_marks_offline_provenance():
    items = [listing("a"), listing("b"), listing("c", "TABLE")]
    paths = {f"image-{key}": f"aa/{key}.jpg" for key in ("a", "b", "c")}
    selected, report = select_abo_mass_subset(
        items,
        paths,
        categories={"chair": ("CHAIR",), "table": ("TABLE",)},
        max_objects_per_class=1,
    )
    assert len(selected) == 2
    chair = next(item for item in selected if item.semantic_class == "chair")
    assert chair.mass_kg == pytest.approx(4.5359237)
    assert (chair.width_m, chair.depth_m, chair.height_m) == pytest.approx(
        (0.508, 0.254, 0.762)
    )
    assert chair.mass_provenance == ABO_MASS_PROVENANCE
    assert chair.evidence_role == ABO_EVIDENCE_ROLE
    assert chair.runtime_evidence_status == ABO_RUNTIME_EVIDENCE_STATUS
    assert report.selected_records == 2
    assert dict(report.class_counts) == {"chair": 1, "table": 1}


def test_exact_product_type_matching_rejects_substring_false_positive():
    selected, report = select_abo_mass_subset(
        [listing("cover", "PORTABLE_ELECTRONIC_DEVICE_COVER")],
        {"image-cover": "aa/cover.jpg"},
        categories={"cup": ("DRINKING_CUP",)},
    )
    assert selected == ()
    assert dict(report.rejection_counts)["outside_selected_categories"] == 1


def test_missing_and_unsupported_physical_values_remain_rejected():
    missing = listing("missing")
    missing.pop("item_weight")
    bad_unit = listing("bad")
    bad_unit["item_weight"][0]["normalized_value"]["unit"] = "stones"
    selected, report = select_abo_mass_subset(
        [missing, bad_unit],
        {"image-missing": "aa/missing.jpg", "image-bad": "aa/bad.jpg"},
        categories={"chair": ("CHAIR",)},
    )
    assert selected == ()
    rejections = dict(report.rejection_counts)
    assert rejections["missing_mass"] == 1
    assert rejections["invalid_or_unsupported_physical_value"] == 1


def test_missing_image_index_is_reported_not_fabricated():
    selected, report = select_abo_mass_subset(
        [listing("a")], {}, categories={"chair": ("CHAIR",)}
    )
    assert selected == ()
    assert report.missing_image_index_records == 1


def test_cross_market_duplicate_products_and_images_are_not_split_as_independent():
    first = listing("same", image_id="shared")
    second = listing("same", image_id="other")
    second["domain_name"] = "amazon.ca"
    third = listing("different", image_id="shared")
    selected, report = select_abo_mass_subset(
        [first, second, third],
        {"shared": "aa/shared.jpg", "other": "bb/other.jpg"},
        categories={"chair": ("CHAIR",)},
    )
    assert len(selected) == 2
    assert len({item.item_id for item in selected}) == 2
    assert len({item.image_id for item in selected}) == 2
    assert all(item.split_group_id == item.item_id for item in selected)
    rejections = dict(report.rejection_counts)
    assert rejections["duplicate_product_id"] == 1


def test_class_capping_does_not_bias_object_splits():
    items = [listing(str(index)) for index in range(1000)]
    paths = {
        f"image-{index}": f"aa/{index}.jpg" for index in range(1000)
    }
    selected, report = select_abo_mass_subset(
        items,
        paths,
        categories={"chair": ("CHAIR",)},
        max_objects_per_class=200,
    )
    counts = dict(report.split_counts)
    assert len(selected) == 200
    assert 110 <= counts["train"] <= 170
    assert 10 <= counts["validation"] <= 50
    assert 10 <= counts["test"] <= 50


def test_gzip_jsonl_and_image_index_readers(tmp_path):
    listings_path = tmp_path / "listings_0.json.gz"
    with gzip.open(listings_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(listing("a")) + "\n")
    assert tuple(iter_abo_listings([listings_path])) == (listing("a"),)

    images_path = tmp_path / "images.csv.gz"
    with gzip.open(images_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", "height", "width", "path"))
        writer.writeheader()
        writer.writerow(
            {"image_id": "keep", "height": 10, "width": 20, "path": "aa/a.jpg"}
        )
        writer.writerow(
            {"image_id": "skip", "height": 10, "width": 20, "path": "bb/b.jpg"}
        )
    assert load_abo_image_paths(images_path, {"keep"}) == {"keep": "aa/a.jpg"}


def test_manifest_is_deterministic_and_rejects_duplicate_objects(tmp_path):
    candidate = AboMassCandidate(
        object_id="amazon.com:a",
        split_group_id="a",
        item_id="a",
        domain_name="amazon.com",
        semantic_class="chair",
        product_type="CHAIR",
        image_id="image-a",
        image_path="aa/a.jpg",
        mass_kg=1,
        width_m=1,
        depth_m=1,
        height_m=1,
        split="train",
    )
    path = tmp_path / "manifest.jsonl"
    write_abo_subset_manifest([candidate], path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["runtime_evidence_status"] == "unavailable"
    assert read_abo_subset_manifest(path) == (candidate,)
    with pytest.raises(ValueError, match="object_id"):
        write_abo_subset_manifest([candidate, candidate], path)


@pytest.mark.parametrize(
    "change",
    [
        {"mass_provenance": "measured"},
        {"evidence_role": "runtime"},
        {"runtime_evidence_status": "measured"},
        {"image_path": "../escape.jpg"},
    ],
)
def test_candidate_provenance_and_paths_cannot_be_misrepresented(change):
    values = dict(
        object_id="amazon.com:a",
        split_group_id="a",
        item_id="a",
        domain_name="amazon.com",
        semantic_class="chair",
        product_type="CHAIR",
        image_id="image-a",
        image_path="aa/a.jpg",
        mass_kg=1,
        width_m=1,
        depth_m=1,
        height_m=1,
        split="train",
    )
    values.update(change)
    with pytest.raises(ValueError):
        AboMassCandidate(**values)
