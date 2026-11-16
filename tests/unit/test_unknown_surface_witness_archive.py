"""Archive auditing must bind actual arrays and preserve a failed native result."""

import json

import numpy as np
import pytest

from robot.evaluation.unknown_obstacle_validation import UnknownObstacleGeometryArchive
from robot.perception.surface_support import whole_surface_replacement
from robot.perception.unknown_obstacle_memory import (
    UnknownObstacleConfig,
    UnknownObstacleSnapshot,
    UnknownSurface,
    UncertainSurface,
)
from scripts.evaluation.audit_unknown_surface_witnesses import audit


def archive(tmp_path):
    old = np.column_stack((1 + np.arange(5) * .001, np.zeros(5)))
    current = np.column_stack((np.linspace(1.08, 1.099, 5), np.zeros(5)))
    ids = np.full(5, 2)
    owner, distance, proof = whole_surface_replacement(old, current, ids)
    config = UnknownObstacleConfig(distinct_surface_witness_matching=True)
    proof.update(original_track_id=1, original_source_time=1., original_source_frame=10,
                 current_source_time=1.2, current_source_frame=12,
                 original_risk=config.unknown_phi, maximum_source_gap_seconds=.35)
    original = UnknownSurface(1, 1., 10, tuple(old.mean(axis=0)), old, config.guidance())
    region = UncertainSurface(original, .10, "current_whole_surface_geometry_replacement",
                              distance, owner, injective_support=proof)
    snapshot = UnknownObstacleSnapshot(1.2, config, (region,))
    memory = snapshot.diagnostic()
    # Only coverage provenance is under test here; this is not a field solve.
    coverage = dict(memory["regions"][0], current_source_time=1.2,
                    local_region_current_field_covered=True)
    source = tmp_path / "native"
    geometry = UnknownObstacleGeometryArchive(source).save(snapshot)
    candidate = source / "current.npz"
    np.savez(candidate, world_xyz=np.column_stack((current, np.zeros(5))),
             raw_track_ids=ids, source_frame_id=12, sensor_timestamp=1.2)
    record = dict(sensor_frame_id=12, sensor_timestamp=1.2, commit_simulation_time=1.3,
                  diagnostics=dict(unknown_obstacle_memory=memory,
                    unknown_obstacle_geometry_archive=geometry,
                    unknown_obstacle_field_checks=[coverage],
                    image_object_groups=dict(candidate_archive=str(candidate))))
    report = dict(passed=False, unknown_obstacle_validation=dict(
        sources=1, injective_surface_proofs=1, passed=False))
    (source / "report.json").write_text(json.dumps(report))
    (source / "live-fields.jsonl").write_text(json.dumps(record) + "\n")
    return source, record


def test_exact_certificate_pass_does_not_relabel_failed_native(tmp_path):
    source, _ = archive(tmp_path)
    result = audit(source)
    assert result["audit_passed"] and result["native_overall_passed"] is False
    assert result["counts"]["live_certificates"] == 1
    assert len(result["certificates"]) == 1 and len(result["source_hashes"]) == 4


@pytest.mark.parametrize("mutation", ["source_clock", "risk", "coverage", "witness", "summary_count", "archive_hash"])
def test_corrupt_certificate_or_source_rejected(tmp_path, mutation):
    source, record = archive(tmp_path)
    d = record["diagnostics"]
    region = d["unknown_obstacle_memory"]["regions"][0]
    if mutation == "source_clock":
        with np.load(source / "current.npz") as data:
            values = {key: data[key] for key in data.files}
        values["sensor_timestamp"] = 1.1
        np.savez(source / "current.npz", **values)
    elif mutation == "risk":
        region["applied_risk"] -= .1
    elif mutation == "coverage":
        d["unknown_obstacle_field_checks"][0]["local_region_current_field_covered"] = False
    elif mutation == "witness":
        region["injective_support"]["current_witness_indices"][0] = 999
    elif mutation == "summary_count":
        report = json.loads((source / "report.json").read_text())
        report["unknown_obstacle_validation"]["injective_surface_proofs"] = 0
        (source / "report.json").write_text(json.dumps(report))
    else:
        d["unknown_obstacle_geometry_archive"]["regions"][0]["file_sha256"] = "0" * 64
    (source / "live-fields.jsonl").write_text(json.dumps(record) + "\n")
    result = audit(source)
    assert not result["audit_passed"] and result["errors"]
    assert result["native_overall_passed"] is False
