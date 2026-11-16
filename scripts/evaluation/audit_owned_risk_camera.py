"""Audit source-bound camera masks, including explicit zero-owner withholding.

No estimator, scorer or controller runs. Legacy recordings retain their original
all-accepted-mask contract. Calibrated recordings may withhold only the exact
zero-current-owner records documented by their native presentation capture.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot.perception.camera_lidar_fusion import (
    FThetaCameraCalibration,
    depth_visible_projection,
    project_fisheye_polynomial_points,
)


SOURCE_KEYS = (
    "sensor_frame_id",
    "sensor_timestamp_seconds",
    "camera_frame_id",
    "camera_timestamp_seconds",
    "masks",
    "rejected",
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_camera_metadata(meta, field, *, current_ownership_only):
    """Check the exact allowed object/mask/rejection sets for one source pair."""
    require(
        meta["sensor_timestamp_seconds"] == field["sensor_timestamp"],
        "camera LiDAR source time differs",
    )
    objects = {o["track_id"]: o for o in field["objects"]}
    require(
        len(objects) == len(field["objects"]), "duplicate original camera object ID"
    )
    masks = meta["masks"]
    shown = {m["track_id"] for m in masks}
    require(len(shown) == len(masks), "duplicate displayed mask ID")
    require(shown <= set(objects), "displayed mask lacks accepted original object")
    expected = {
        key
        for key, o in objects.items()
        if o["diagnostics"]["accepted_instance"]
        and (not current_ownership_only or o["diagnostics"]["policy_return_count"] > 0)
    }
    require(expected <= shown, "accepted current-owner mask is missing")
    zero = {
        key
        for key, o in objects.items()
        if o["diagnostics"]["policy_return_count"] == 0
    }
    withheld = set()
    for rejected in meta.get("rejected", []):
        key = rejected.get("track_id")
        require(
            current_ownership_only
            and key in zero
            and rejected.get("reason") == "no_current_policy_return_ownership"
            and rejected.get("policy_return_count") == 0,
            "unsupported or unjustified presentation rejection",
        )
        require(
            key not in withheld and key not in shown,
            "duplicate or displayed withheld record",
        )
        withheld.add(key)
    if current_ownership_only:
        require(not shown & zero, "zero-owner object received a colored mask")
        require(withheld == zero, "zero-owner presentation rejection is missing")
    for m in masks:
        obj = objects[m["track_id"]]
        d = obj["diagnostics"]
        require(
            m["phi"] == obj["cra_control"]["cra"]["phi"],
            "individual displayed CRA differs",
        )
        require(
            meta["camera_timestamp_seconds"] == obj["attributes"]["timestamp_seconds"],
            "object camera source differs",
        )
        require(
            m["factor_modes"] == {k: v["mode"] for k, v in d["factor_scores"].items()},
            "displayed factor provenance differs",
        )
        require(
            m["applied_phi"] == field["policy_phi"][m["track_id"]],
            "displayed policy/group score differs",
        )
    return objects, withheld


def verify_snapshot_clock(snapshot, meta):
    for source, shown in (
        ("lidar_id", "sensor_frame_id"),
        ("lidar_time", "sensor_timestamp_seconds"),
        ("camera_id", "camera_frame_id"),
        ("camera_time", "camera_timestamp_seconds"),
    ):
        require(
            snapshot[source] == meta[shown], "snapshot source tuple differs"
        )


def verify_camera_source(meta, field, snapshot, *, current_ownership_only):
    """Bind every assessment, with or without masks, to its sensor snapshot."""
    verify_snapshot_clock(snapshot, meta)
    return verify_camera_metadata(
        meta, field, current_ownership_only=current_ownership_only
    )


def audit_camera(folder):
    folder = Path(folder).resolve()
    hashes = {}

    def remember(path):
        path = Path(path).resolve()
        require(
            path.is_relative_to(folder), "camera audit source escaped native folder"
        )
        if str(path) not in hashes:
            hashes[str(path)] = digest(path)
        return path

    def read(path):
        return json.loads(remember(path).read_text(encoding="utf-8"))

    execution = read(folder / "execution.json")
    require(
        execution.get("passed") and execution.get("source_unchanged"),
        "passing frozen native execution required",
    )
    command = read(folder / "invocation.json")["command"]
    current_ownership_only = (
        "--restaurant-calibration" in command
        and command[command.index("--restaurant-calibration") + 1] != "off"
    )
    fields = {}

    def record_field(r):
        key = r["sensor_frame_id"]
        require(key not in fields, "duplicate field ID")
        fields[key] = dict(
            sensor_timestamp=r["sensor_timestamp"],
            commit_time=r["commit_simulation_time"],
            archive=r["diagnostics"]["image_object_groups"]["candidate_archive"],
            policy_phi={
                p["track_or_group_id"]: p["phi"]
                for p in r["diagnostics"]["occupancy_policies"]
            },
            objects=[
                o
                for o in r["objects"]
                if o["diagnostics"]["accepted_instance"]
                or o["diagnostics"].get("static_prop_policy")
            ],
        )

    for r in read(folder / "stationary-fields.json"):
        record_field(r)
    live = remember(folder / "live-fields.jsonl")
    with live.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record_field(json.loads(line))
    ledger = read(folder / "risk-camera-epochs.json")
    frames = ledger["frames"]
    overview = read(folder / "service-video-epochs.json")["frames"]
    require(
        [r["time"] for r in frames] == [r["time"] for r in overview],
        "native camera/video timing differs",
    )
    require(
        not ledger["rejected_assessments"],
        "native field/assessment presentation rejection",
    )
    unique = {}
    for epoch in frames:
        if epoch["status"] == "awaiting_assessment":
            require(not epoch["masks"], "mask before first assessment")
            continue
        require(
            epoch["status"] == "latest_assessed_frame",
            "unsupported camera presentation state",
        )
        require(
            epoch["sensor_frame_id"] in fields, "camera source has no recorded field"
        )
        require(
            fields[epoch["sensor_frame_id"]]["commit_time"] <= epoch["time"],
            "camera uses a future field commit",
        )
        age = epoch["time"] - epoch["camera_timestamp_seconds"]
        require(
            math.isfinite(age)
            and age >= 0
            and abs(epoch["source_age_seconds"] - age) < 1e-10,
            "camera source age differs",
        )
        meta = {
            k: epoch.get(k, []) if k == "rejected" else epoch[k] for k in SOURCE_KEYS
        }
        key = meta["camera_frame_id"]
        require(
            key not in unique or unique[key] == meta,
            "held camera source metadata changed",
        )
        unique[key] = meta
    files = {}
    for path in sorted((folder / "risk-camera-sources").glob("*.json")):
        meta = read(path)
        key = meta["camera_frame_id"]
        require(key not in files, "duplicate saved camera source")
        if key in unique:
            require(
                {
                    k: meta.get(k, []) if k == "rejected" else meta[k]
                    for k in SOURCE_KEYS
                }
                == unique[key],
                "saved/displayed camera metadata differ",
            )
        files[key] = (meta, path)
    calibration = FThetaCameraCalibration(**read(folder / "camera-calibration.json"))
    checks, withheld_count, no_mask_sources, source_pairs = [], 0, 0, 0
    for key in sorted(set(unique) | set(files)):
        meta = files[key][0] if key in files else unique[key]
        require(meta["sensor_frame_id"] in fields, "saved camera source has no field")
        field = fields[meta["sensor_frame_id"]]
        snapshot = read(
            folder / "sensor-snapshots" / f"{meta['sensor_frame_id']:06d}.json"
        )
        objects, withheld = verify_camera_source(
            meta, field, snapshot, current_ownership_only=current_ownership_only
        )
        source_pairs += 1
        withheld_count += len(withheld)
        if not meta["masks"]:
            no_mask_sources += 1
            continue
        require(key in files, "displayed masks lack saved source pixels")
        path = files[key][1].with_suffix(".npz")
        with np.load(remember(path), allow_pickle=False) as data:
            require(
                len(data["masks"]) == len(meta["masks"]), "saved mask count differs"
            )
            for m, mask in zip(meta["masks"], data["masks"]):
                obj = objects[m["track_id"]]
                d = obj["diagnostics"]
                require(
                    mask.dtype == np.bool_
                    and mask.shape == data["rgb"].shape[:2]
                    and int(mask.sum()) == m["pixels"],
                    "saved mask shape/type/pixel count differs",
                )
                if d.get("static_prop_policy"):
                    require(
                        m["basis"] == "retained evidence / LiDAR points",
                        "retained points presented as image mask",
                    )
                    with np.load(
                        remember(field["archive"]), allow_pickle=False
                    ) as original:
                        world = original["world_xyz"]
                        proof = d["static_prop_association"]
                        allowance = proof["position_allowance_m"]
                        selected = np.all(
                            (
                                world
                                >= np.array(proof["observed_bounds_minimum"])
                                - allowance
                            )
                            & (
                                world
                                <= np.array(proof["observed_bounds_maximum"])
                                + allowance
                            ),
                            axis=1,
                        )
                        selected &= np.isin(
                            original["raw_track_ids"],
                            [f["raw_track_id"] for f in proof["fragments"]],
                        )
                        require(
                            int(selected.sum()) == d["policy_return_count"],
                            "retained point ownership count differs",
                        )
                        projection = depth_visible_projection(
                            project_fisheye_polynomial_points(
                                world,
                                camera_from_source=original["camera_from_world"],
                                calibration=calibration,
                            ),
                            radius_pixels=4,
                        )
                        expected = np.zeros(mask.shape, np.uint8)
                        for x, y in projection.pixel_indices_xy[
                            selected[projection.source_indices]
                        ]:
                            cv2.circle(expected, (int(x), int(y)), 1, 1, -1)
                        require(
                            np.array_equal(mask, expected.astype(bool)),
                            "retained point projection differs",
                        )
                else:
                    require(
                        m["pixels"] == d["instance_mask_pixels"],
                        "accepted native mask pixel count differs",
                    )
                checks.append(
                    dict(camera_frame_id=key, track_id=m["track_id"], phi=m["phi"])
                )
    for name in ("service-navigation.mp4", "service-risk-camera.mp4"):
        cap = cv2.VideoCapture(str(remember(folder / name)))
        try:
            require(
                cap.isOpened()
                and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames),
                "native video frame count differs",
            )
        finally:
            cap.release()
    require(
        all(digest(path) == old for path, old in hashes.items()),
        "native camera audit source changed",
    )
    return dict(
        passed=True,
        exact_field_camera_score_mask_parity=True,
        video_timing_parity=True,
        video_frames=len(frames),
        accepted_masks=len(checks),
        unique_colored_tracks=len({c["track_id"] for c in checks}),
        phi_range=[min(c["phi"] for c in checks), max(c["phi"] for c in checks)]
        if checks
        else None,
        current_ownership_only=current_ownership_only,
        zero_owner_records_withheld=withheld_count,
        source_pairs_verified=source_pairs,
        no_mask_source_pairs_verified=no_mask_sources,
        source_unchanged=True,
        native_execution_passed=True,
        input_sha256=hashes,
        scope="Exact original individual/group score, mask/point projection, source pair and video timing parity; zero-owner withholding only when native opt-in enabled. No control acceptance or risk recalculation.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.folder / "camera-presentation-audit.json"
    if output.exists():
        parser.error("use a fresh audit output; preserve previous results")
    result = audit_camera(args.folder)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, indent=2)
    )


if __name__ == "__main__":
    main()
