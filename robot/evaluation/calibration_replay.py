"""Strict presentation of candidate scores on an unchanged archived trajectory."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from robot.evaluation.risk_camera_overlay import RiskCameraFrame, RiskCameraMask


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def candidate_geometry(geometry, record):
    """Only current-point risk changes; prior geometry and ownership stay exact."""
    policies = {int(r["track_id"]): r for r in record["policies"]}
    if len(policies) != len(record["policies"]):
        raise ValueError("duplicate candidate policy ID")
    result = dict(geometry)
    phi = np.asarray(geometry["point_phi"]).copy()
    for key in np.unique(geometry["point_track_ids"]):
        if int(key) not in policies:
            raise ValueError("current point has no candidate policy")
        row = policies[int(key)]
        selected = geometry["point_track_ids"] == key
        if not np.allclose(phi[selected], row["original_applied_phi"], rtol=0, atol=1e-12):
            raise ValueError("candidate does not reproduce original applied point risk")
        value = float(row["candidate_current_phi"])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid candidate point risk")
        phi[selected] = value
    result["point_phi"] = phi
    return result


class CalibrationReplay:
    """Bind score rows, original camera pixels and masks to exact field epochs."""

    def __init__(self, manifest_path, source):
        path = Path(manifest_path).resolve()
        self.source = Path(source).resolve()
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if Path(self.manifest["source_directory"]).resolve() != self.source:
            raise ValueError("calibration belongs to another native mission")
        fields = (path.parent / self.manifest["fields_file"]).resolve()
        if not fields.is_relative_to(path.parent) or digest(fields) != self.manifest["fields_sha256"]:
            raise ValueError("candidate field archive is invalid")
        self.hashes = {str(path): digest(path), str(fields): digest(fields)}
        self.hashes.update(self.manifest["source_hashes"])
        if "original_manifest" in self.manifest:
            original_manifest = Path(self.manifest["original_manifest"]).resolve()
            self.hashes[str(original_manifest)] = self.manifest["manifest_sha256"]
        self.verify()
        self.rows = {}
        with fields.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = int(row["sensor_frame_id"])
                if key in self.rows:
                    raise ValueError("duplicate candidate field epoch")
                self.rows[key] = row

    def verify(self):
        if any(digest(path) != expected for path, expected in self.hashes.items()):
            raise ValueError("calibration replay source changed")

    def field(self, frame):
        row = self.rows[int(frame["field_id"])]
        if (abs(row["sensor_timestamp"] - frame["source_time"]) > 1e-8 or
                abs(row["commit_simulation_time"] - frame["commit_time"]) > 1e-8):
            raise ValueError("candidate field source/commit mismatch")
        return row

    def _record_file(self, path):
        name = str(path.resolve())
        current = digest(path)
        if name in self.hashes and self.hashes[name] != current:
            raise ValueError("camera input changed during replay")
        self.hashes[name] = current

    def camera(self, epoch, row):
        if epoch["status"] == "awaiting_assessment":
            return None
        if int(epoch["sensor_frame_id"]) != row["sensor_frame_id"]:
            raise ValueError("camera does not match candidate field")
        original_masks = epoch["masks"]
        if original_masks:
            stem = self.source / "risk-camera-sources" / str(epoch["camera_frame_id"])
            metadata = stem.with_suffix(".json")
            archive = stem.with_suffix(".npz")
            self._record_file(metadata)
            self._record_file(archive)
            saved = json.loads(metadata.read_text(encoding="utf-8"))
            for key in ("sensor_frame_id", "sensor_timestamp_seconds", "camera_frame_id",
                        "camera_timestamp_seconds", "masks"):
                if saved[key] != epoch[key]:
                    raise ValueError("original camera mask/source mismatch")
            with np.load(archive, allow_pickle=False) as data:
                rgb, masks = data["rgb"], data["masks"]
            if len(masks) != len(original_masks):
                raise ValueError("original camera mask count mismatch")
        else:
            image_path = self.source / "sensor-snapshots" / f"{epoch['sensor_frame_id']:06d}.png"
            metadata = image_path.with_suffix(".json")
            self._record_file(metadata)
            snapshot = json.loads(metadata.read_text(encoding="utf-8"))
            for key, target in (("lidar_id", "sensor_frame_id"),
                                ("lidar_time", "sensor_timestamp_seconds"),
                                ("camera_id", "camera_frame_id"),
                                ("camera_time", "camera_timestamp_seconds")):
                if snapshot[key] != epoch[target]:
                    raise ValueError("no-mask RGB snapshot has a different source epoch")
            self._record_file(image_path)
            rgb = np.array(Image.open(image_path).convert("RGB"))
            masks = []
        objects = {r["track_id"]: r for r in row["objects"]}
        policies = {r["track_id"]: r for r in row["policies"]}
        result = []
        rejected = list(epoch.get("rejected", ()))
        for old, mask in zip(original_masks, masks):
            key = old["track_id"]
            obj = objects[key]
            if (abs(old["phi"] - obj["original_phi"]) > 1e-12 or
                    int(mask.sum()) != old["pixels"] or mask.shape != rgb.shape[:2]):
                raise ValueError("original object mask/score mismatch")
            if obj["policy_return_count"] <= 0:
                rejected.append(dict(track_id=key, reason="no_current_owned_policy_returns"))
                continue
            result.append(RiskCameraMask(key, obj["candidate_phi"],
                policies[key]["candidate_current_phi"], tuple(sorted(
                    (name, "evidence_supported" if obj.get("factors", {}).get(name, {}).get("supported") else mode)
                    for name, mode in old["factor_modes"].items())),
                mask, old["basis"] == "retained evidence / LiDAR points"))
        return RiskCameraFrame(epoch["sensor_frame_id"], epoch["sensor_timestamp_seconds"],
            epoch["camera_frame_id"], epoch["camera_timestamp_seconds"], rgb,
            tuple(result), tuple(rejected))
