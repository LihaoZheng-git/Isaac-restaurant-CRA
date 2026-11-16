"""Check exported point colors against original committed controller records."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def audit_field(record, geometry, original_world):
    """Independent numeric parity check; no estimator or adapter is executed."""
    points = geometry["points_world_xy"]
    if not np.array_equal(points, original_world[:, :2]):
        raise ValueError("current source geometry changed")
    policies = {int(p["track_or_group_id"]): p["applied_field_risk"]
                for p in record["diagnostics"]["applied_field_policies"]}
    counts = {}
    for prefix, key in (("point", "points_world_xy"), ("retained", "retained_world_xy")):
        ids, phi = geometry[f"{prefix}_track_ids"], geometry[f"{prefix}_phi"]
        if ids.shape != phi.shape or len(ids) != len(geometry[key]):
            raise ValueError("point/policy array length differs")
        for identity in np.unique(ids):
            if int(identity) not in policies:
                raise ValueError("export has no recorded final policy")
            if not np.all(phi[ids == identity] == policies[int(identity)]):
                raise ValueError("color score differs from controller applied policy")
        counts[prefix] = len(ids)
    ids = geometry["point_track_ids"]
    for obj in record["objects"]:
        if int(np.count_nonzero(ids == obj["track_id"])) != obj["diagnostics"]["policy_return_count"]:
            raise ValueError("final per-object point count differs")
    original_count = record["diagnostics"].get("field_raw_placement_count")
    if original_count is not None and sum(counts.values()) != original_count:
        raise ValueError("current and retained placement total differs from controller")
    return counts


def audit(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(manifest["source_directory"])
    fields = {int(row["field_id"]): row for row in manifest["fields"]}
    if len(fields) != len(manifest["fields"]):
        raise ValueError("duplicate exported field")
    seen, current, retained = set(), 0, 0

    def check(record):
        nonlocal current, retained
        key = record["sensor_frame_id"]
        if key in seen or key not in fields:
            raise ValueError("duplicate or missing original field")
        field = fields[key]
        if (field["source_time"] != record["sensor_timestamp"]
                or field["commit_time"] != record["commit_simulation_time"]):
            raise ValueError("exported field clock differs")
        name = record["diagnostics"]["image_object_groups"]["candidate_archive"]
        with np.load(name, allow_pickle=False) as original:
            world = original["world_xyz"]
        path = (manifest_path.parent / field["geometry_file"]).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("geometry escaped export directory")
        with np.load(path, allow_pickle=False) as arrays:
            counts = audit_field(record, arrays, world)
        current += counts["point"]
        retained += counts["retained"]
        seen.add(key)

    for record in json.loads((source / "stationary-fields.json").read_text(encoding="utf-8")):
        check(record)
    with (source / "live-fields.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                check(json.loads(line))
    if seen != set(fields):
        raise ValueError("export contains non-original fields")
    epochs = json.loads((source / "risk-camera-epochs.json").read_text(encoding="utf-8"))["frames"]
    if len(epochs) != len(manifest["frames"]):
        raise ValueError("video epoch count differs")
    receipt_ties = 0
    for row, epoch in zip(manifest["frames"], epochs):
        if row["time"] != epoch["time"] or row["field_id"] != epoch.get("sensor_frame_id"):
            raise ValueError("video field receipt identity differs")
        if row["field_id"] is not None:
            field = fields[row["field_id"]]
            if field["commit_time"] > row["time"]:
                raise ValueError("video uses future controller field")
            # Count later commits at this timestamp that were correctly withheld.
            receipt_ties += any(f["commit_time"] == row["time"] and f["field_id"] != row["field_id"]
                                for f in fields.values())
    source_checks = manifest.get("source_hashes", {})
    if not source_checks:
        raise ValueError("original source hashes are required")
    for name, expected in source_checks.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"source changed: {name}")
    return dict(passed=True, fields=len(seen), video_frames=len(epochs), current_points=current,
                retained_geometry_samples=retained, original_files_verified=len(source_checks),
                later_same_timestamp_commits_withheld=receipt_ties,
                scope="Every original current XY, final policy ID count, applied phi, and video receipt checked; no new CRA or navigation acceptance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new audit output")
    result = audit(args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
