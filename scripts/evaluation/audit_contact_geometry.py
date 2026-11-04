"""Compare legacy/local contact cues on original accepted image-object masks.

Offline diagnostic only: no geometry, identity, mass, prior or control changes.
All accepted grouped masks are audited; no fixture/category selects the score.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robot.perception.local_contact_geometry import LocalContactGeometryConfig
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection
from robot.perception.visual_shape_contact import SilhouetteContactConfig, estimate_silhouette_contact_proxy


def audit_source(folder, segmenter=None):
    fields_path = folder / "stationary-fields.json"
    records = {r["sensor_frame_id"]: r for r in json.loads(fields_path.read_text())}
    rows, hashes = [], {str(fields_path): hashlib.sha256(fields_path.read_bytes()).hexdigest()}
    for path in sorted((folder / "risk-camera-sources").glob("*.json"), key=lambda p: int(p.stem)):
        meta = json.loads(path.read_text())
        frame = records[meta["sensor_frame_id"]]
        if frame["sensor_timestamp"] != meta["sensor_timestamp_seconds"]:
            raise ValueError("camera mask metadata does not match original LiDAR source")
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path.with_suffix(".npz"))] = hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest()
        with np.load(path.with_suffix(".npz")) as data:
            for index, item in enumerate(meta["masks"]):
                candidates = [o for o in frame["objects"] if o["track_id"] == item["track_id"]
                              and o["diagnostics"].get("image_object_association")]
                if not candidates:
                    continue
                if len(candidates) != 1:
                    raise ValueError("ambiguous recorded object")
                obj = candidates[0]
                timestamp = meta["camera_timestamp_seconds"]
                if abs(obj["attributes"]["timestamp_seconds"] - timestamp) > 1e-9:
                    raise ValueError("only current same-image assessments can be replayed")
                mask = data["masks"][index].astype(bool)
                y, x = np.where(mask)
                detection = SemanticDetection(0, "unknown", obj["attributes"]["semantic_confidence"],
                    PixelBoundingBox(float(x.min()), float(y.min()), float(x.max()+1), float(y.max()+1)), mask)
                values = {}
                for method, config in (("legacy", None), ("local", SilhouetteContactConfig(
                        local_geometry=LocalContactGeometryConfig()))):
                    values[method] = asdict(estimate_silhouette_contact_proxy(
                        detection, timestamp_seconds=timestamp, config=config))
                if segmenter is not None:
                    from robot.perception.contact_mask_refinement import refine_contact_mask
                    refined, values["refinement_diagnostics"] = refine_contact_mask(
                        data["rgb"], detection, segmenter.backend)
                    values["refined"] = asdict(estimate_silhouette_contact_proxy(refined,
                        timestamp_seconds=timestamp,
                        config=SilhouetteContactConfig(local_geometry=LocalContactGeometryConfig())))
                recorded = obj["diagnostics"].get("contact_proxy")
                rows.append(dict(sensor_frame=meta["sensor_frame_id"], object_id=item["track_id"],
                    camera_timestamp=timestamp, mask_pixels=int(mask.sum()), **values,
                    recorded_contact_proxy=recorded,
                    recorded_contact_factor=obj["diagnostics"]["factor_scores"]["contact"],
                    recorded_phi=obj["cra_control"]["cra"]["phi"]))
    summary = {}
    for method in (("legacy", "local", "refined") if segmenter is not None else ("legacy", "local")):
        supported = [r[method]["value"] for r in rows if r[method]["value"] is not None]
        summary[method] = dict(available=len(supported), median=float(np.median(supported)) if supported else None)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise RuntimeError("source changed during replay")
    return dict(source=str(folder.resolve()), source_hashes=hashes, summary=summary, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refine-sam-checkpoint", type=Path,
                        help="Optional existing audited sam2.1_t.pt; no download or install")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output path; preserve previous diagnostics")
    sources = [p.resolve() for p in args.source]
    if len(sources) != len(set(sources)):
        parser.error("source folders must be distinct")
    segmenter = None
    if args.refine_sam_checkpoint is not None:
        from robot.perception.sam_instance_segmentation import SamInstanceConfig, SamInstanceSegmenter
        segmenter = SamInstanceSegmenter(SamInstanceConfig(args.refine_sam_checkpoint))
    result = dict(scope="Offline recorded-mask/RGB replay; visible boundary cues, not physical sharpness or newly executed CRA",
                  local_configuration=asdict(LocalContactGeometryConfig()),
                  refinement_checkpoint_sha256=segmenter.checkpoint_sha256 if segmenter is not None else None,
                  cases=[audit_source(p, segmenter) for p in sources])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: c[k] for k in ("source", "summary")} for c in result["cases"]], indent=2))


if __name__ == "__main__":
    main()
