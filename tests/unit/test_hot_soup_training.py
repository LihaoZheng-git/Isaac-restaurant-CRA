import json

import pytest

from robot.perception.hot_soup_dataset import (
    HotSoupBox,
    HotSoupSample,
    sha256_file,
)
from robot.perception.hot_soup_training import (
    HotSoupTrainingReadiness,
    assess_hot_soup_training_readiness,
    write_hot_soup_training_provenance,
)


def sample(root, identifier, split, state, *, domain="real_user_owned", hard=()):
    image = root / f"{identifier}.png"
    image.write_bytes(f"image-{identifier}".encode())
    boxes = ()
    if state == "present":
        boxes = (
            HotSoupBox(
                "visible_hot_soup_plume",
                0.4,
                0.1,
                0.6,
                0.4,
                "human_visual_annotation",
            ),
            HotSoupBox(
                "open_container",
                0.3,
                0.4,
                0.7,
                0.8,
                "human_visual_annotation",
            ),
        )
    return HotSoupSample(
        identifier,
        f"group-{identifier}",
        split,
        image.name,
        sha256_file(image),
        100,
        100,
        domain,
        f"owned capture {identifier}",
        "owned by project",
        state,
        "present" if state == "present" else "absent",
        boxes,
        hard,
    )


def ready_samples(tmp_path):
    items = []
    for split in ("train", "val", "test"):
        items.append(sample(tmp_path, f"{split}-positive", split, "present"))
        items.append(
            sample(
                tmp_path,
                f"{split}-negative",
                split,
                "absent",
                hard=(("glare",) if split == "val" else ()),
            )
        )
    return tuple(items)


def test_real_split_coverage_and_hard_negative_pass_functional_gate(tmp_path):
    readiness = assess_hot_soup_training_readiness(
        ready_samples(tmp_path),
        tmp_path,
    )
    assert readiness.ready
    assert readiness.real_sample_count == 6
    assert readiness.evaluation_hard_negative_count == 1


def test_synthetic_samples_do_not_satisfy_real_training_gate(tmp_path):
    synthetic = sample(
        tmp_path,
        "synthetic",
        "train",
        "absent",
        domain="isaac_synthetic",
    )
    readiness = assess_hot_soup_training_readiness((synthetic,), tmp_path)
    assert not readiness.ready
    assert readiness.synthetic_sample_count == 1
    assert any("real capture groups" in item for item in readiness.blockers)
    assert any("test: real positive" in item for item in readiness.blockers)


def test_invalid_dataset_integrity_blocks_training(tmp_path):
    items = ready_samples(tmp_path)
    (tmp_path / items[0].image_path).write_bytes(b"changed")
    readiness = assess_hot_soup_training_readiness(items, tmp_path)
    assert not readiness.ready
    assert "dataset integrity validation failed" in readiness.blockers


def test_training_provenance_hashes_manifest_and_weights(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    base = tmp_path / "base.pt"
    trained = tmp_path / "best.pt"
    manifest.write_text("manifest\n", encoding="utf-8")
    base.write_bytes(b"base")
    trained.write_bytes(b"trained")
    readiness = HotSoupTrainingReadiness(6, 6, (), (), 1, 0, ())
    output = tmp_path / "provenance.json"
    write_hot_soup_training_provenance(
        output,
        readiness=readiness,
        manifest_path=manifest,
        base_checkpoint_path=base,
        trained_checkpoint_path=trained,
        package_version="1.2.3",
        package_license="AGPL-3.0",
        training_arguments={"epochs": 2},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["base_checkpoint_sha256"]) == 64
    assert len(payload["trained_checkpoint_sha256"]) == 64
    assert payload["package_license"] == "AGPL-3.0"
    assert "not temperature" in payload["scope"]


def test_blocked_dataset_cannot_write_trained_provenance(tmp_path):
    blocked = HotSoupTrainingReadiness(0, 0, (), (), 0, 0, ("blocked",))
    with pytest.raises(ValueError, match="blocked"):
        write_hot_soup_training_provenance(
            tmp_path / "report.json",
            readiness=blocked,
            manifest_path=tmp_path / "missing",
            base_checkpoint_path=tmp_path / "missing",
            trained_checkpoint_path=tmp_path / "missing",
            package_version="x",
            package_license="x",
            training_arguments={},
        )
