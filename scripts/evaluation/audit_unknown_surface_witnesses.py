"""Bind recorded geometry certificates to source arrays; never grant native acceptance."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from robot.evaluation.unknown_obstacle_validation import validate_injective_support


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class Sources:
    def __init__(self, directory):
        self.directory = directory.resolve()
        self.hashes = {}

    def path(self, value):
        path = Path(value)
        path = (path if path.is_absolute() else self.directory / path).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("source reference escapes native archive")
        if path not in self.hashes:
            self.hashes[path] = digest(path)
        return path

    def json(self, value):
        return json.loads(self.path(value).read_text(encoding="utf-8"))


def audit_certificate(record, region, coverage, reference, metadata, old, current, ids):
    """Check exact geometry plus original clocks, risk and recorded raster coverage."""
    d = record["diagnostics"]
    memory = d["unknown_obstacle_memory"]
    config = memory["config"]
    proof = region["injective_support"]
    now = record["sensor_timestamp"]
    expected_metadata = dict(
        track_id=region["track_id"],
        original_source_time=region["original_source_time"],
        original_source_frame=region["original_source_frame"],
        center_world_xy=region["center_world_xy"],
    )
    if metadata != expected_metadata or any(
        reference.get(key) != value for key, value in expected_metadata.items()
    ):
        raise ValueError("original geometry metadata differs from region")
    if not validate_injective_support(
        proof, original_points=old, current_points=current, current_ids=ids
    ):
        raise ValueError("certificate does not bind complete distinct source support")
    if (
        not config.get("distinct_surface_witness_matching")
        or region["mode"] != "current_whole_surface_geometry_replacement"
        or proof["original_track_id"] != region["track_id"]
        or proof["current_track_id"] != region["replacement_track_id"]
        or proof["original_source_time"] != region["original_source_time"]
        or proof["original_source_frame"] != region["original_source_frame"]
        or proof["original_point_count"] != region["original_point_count"]
        or reference["original_point_count"] != len(old)
        or proof["original_risk"] != region["applied_risk"]
        or not config["unknown_phi"] <= region["applied_risk"] <= 1
        or region["fixed_unknown_phi"] != config["unknown_phi"]
        or proof["current_source_time"] != now or memory["source_time"] != now
        or proof["current_source_frame"] != record["sensor_frame_id"]
        or proof["maximum_source_gap_seconds"] != config["maximum_source_gap_seconds"]
        or not 0 < config["maximum_source_gap_seconds"] <= .35
        or not 0 < now - region["original_source_time"] <= config["maximum_source_gap_seconds"] + 1e-9
        or proof["maximum_nearest_distance_m"] != region["support_distance_m"]
        or region["radius_m"] != .10
        or not coverage or coverage.get("injective_support") != proof
        or coverage.get("track_id") != region["track_id"]
        or coverage.get("mode") != region["mode"]
        or coverage.get("replacement_track_id") != region["replacement_track_id"]
        or coverage.get("original_source_time") != region["original_source_time"]
        or coverage.get("original_source_frame") != region["original_source_frame"]
        or coverage.get("current_source_time") != now
        or coverage.get("original_point_count") != len(old)
        or coverage.get("applied_risk") != region["applied_risk"]
        or coverage.get("fixed_unknown_phi") != region["fixed_unknown_phi"]
        or coverage.get("radius_m") != .10
        or coverage.get("local_region_current_field_covered") is not True
        or record["commit_simulation_time"] < now
    ):
        raise ValueError("certificate source/risk/owner/age/coverage metadata differs")


def audit(source):
    sources = Sources(source)
    counts, errors, certificates = Counter(), [], []
    reports = {}
    for name in ("report.json", "stationary-gate.json", "execution.json"):
        if (source / name).exists():
            reports[name] = sources.json(name)
    for phase, name in (("stationary", "stationary-fields.json"), ("live", "live-fields.jsonl")):
        if not (source / name).exists():
            continue
        with sources.path(name).open(encoding="utf-8") as stream:
            records = json.load(stream) if phase == "stationary" else (json.loads(line) for line in stream)
            for record in records:
                counts[f"{phase}_fields"] += 1
                d = record["diagnostics"]
                memory = d.get("unknown_obstacle_memory")
                if memory is None:
                    continue
                regions = memory["regions"]
                counts[f"{phase}_memory_fields"] += 1
                selected = [i for i, r in enumerate(regions) if r.get("injective_support") is not None]
                for i in selected:
                    counts[f"{phase}_certificates"] += 1
                    region = regions[i]
                    label = dict(phase=phase, frame=record["sensor_frame_id"], track_id=region["track_id"])
                    try:
                        references = [r for r in d["unknown_obstacle_geometry_archive"]["regions"]
                                      if r["track_id"] == region["track_id"]]
                        if len(references) != 1:
                            raise ValueError("original geometry reference is not unique")
                        reference = references[0]
                        path = sources.path(reference["path"])
                        if sources.hashes[path] != reference["file_sha256"]:
                            raise ValueError("original geometry file hash differs")
                        with np.load(path, allow_pickle=False) as data:
                            encoded = str(data["metadata_json"])
                            metadata = json.loads(encoded)
                            old = data["original_points_world_xy"].copy()
                        encoded_geometry = encoded.encode() + np.ascontiguousarray(old, dtype="<f8").tobytes()
                        if hashlib.sha256(encoded_geometry).hexdigest() != reference["geometry_sha256"]:
                            raise ValueError("original geometry content hash differs")
                        path = sources.path(d["image_object_groups"]["candidate_archive"])
                        with np.load(path, allow_pickle=False) as data:
                            current, ids = data["world_xyz"][:, :2], data["raw_track_ids"]
                            if (int(data["source_frame_id"]) != record["sensor_frame_id"]
                                    or float(data["sensor_timestamp"]) != record["sensor_timestamp"]):
                                raise ValueError("current geometry source clock differs")
                        checks = d["unknown_obstacle_field_checks"]
                        if len(checks) != len(regions):
                            raise ValueError("region/raster coverage count differs")
                        audit_certificate(record, region, checks[i], reference, metadata, old, current, ids)
                        certificates.append(dict(**label, current_owner=region["replacement_track_id"],
                            original_source_time=region["original_source_time"],
                            current_source_time=record["sensor_timestamp"], original_point_count=len(old),
                            applied_risk=region["applied_risk"],
                            maximum_witness_distance_m=region["injective_support"]["maximum_witness_distance_m"]))
                    except (KeyError, ValueError, TypeError, IndexError, OSError) as error:
                        errors.append(dict(**label, error=str(error)))
    summaries = {}
    for phase, name in (("stationary", "stationary-gate.json"), ("live", "report.json")):
        summary = reports.get(name, {}).get("unknown_obstacle_validation")
        if summary is not None:
            expected = summary.get("injective_surface_proofs", 0)
            summaries[phase] = dict(expected_certificates=expected, native_validation_passed=summary.get("passed"))
            if expected != counts[f"{phase}_certificates"] or summary["sources"] != counts[f"{phase}_memory_fields"]:
                errors.append(dict(phase=phase, error="native summary source/certificate count differs"))
    if not counts["stationary_fields"] + counts["live_fields"]:
        errors.append(dict(error="no recorded fields"))
    for path, original in sources.hashes.items():
        if digest(path) != original:
            errors.append(dict(error="source changed during audit", path=str(path)))
    return dict(audit_passed=not errors, native_overall_passed=reports.get("report.json", {}).get("passed"),
        native_execution=reports.get("execution.json"), source_directory=str(source.resolve()),
        source_hashes={str(path): value for path, value in sources.hashes.items()},
        counts=dict(counts), summary_bindings=summaries, certificates=certificates, errors=errors,
        scope="Exact archived injective geometry certificates and recorded current-raster coverage metadata. "
              "No matching solver, identity/factor transfer, controller replay, or new native acceptance. "
              "A passed certificate audit does not override a failed native report.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source):
        parser.error("audit output must be outside the original native archive")
    result = audit(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("audit_passed", "native_overall_passed", "counts", "errors")}, indent=2))
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
