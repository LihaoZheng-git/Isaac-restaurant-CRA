"""Replay archived stationary sensors; authored bounds score outputs only.

No runtime inputs, assets, checkpoints, clocks or obstacle filters are changed.
Sparse near-floor seeds can query SAM for diagnosis even when the ORIGINAL
eight-return association gate rejects them. An RGB pass is not an object pass.
"""

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from robot.evaluation.near_floor_diagnostic import diagnose_near_floor, NearFloorDiagnosticConfig
from robot.perception.camera_lidar_fusion import (
    FThetaCameraCalibration, depth_visible_projection, project_fisheye_polynomial_points,
)
from robot.perception.sam_instance_segmentation import SamInstanceConfig, SamInstanceSegmenter
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection
from robot.perception.sensor_epochs import gravity_aligned_robot_transform
from robot.perception.visual_shape_contact import estimate_silhouette_contact_proxy


def image_diagnostics(model, rgb, projection, candidates, residual, source_time):
    """Sensor-only image proposals; explicitly no authored fixture input."""
    model.set_image(rgb)
    start = time.perf_counter()
    batches = [("automatic_grid", *model.backend.generate())]
    for index, group in enumerate(candidates[:16]):
        owned = np.isin(projection.source_indices, group["indices"])
        xy = projection.pixel_indices_xy[owned]
        if not len(xy):
            continue
        point = xy[np.argmin(np.sum((xy - np.median(xy, axis=0))**2, axis=1))]
        batches.append((f"raw_near_floor_group_{index}", *model.backend.predict(tuple(point))))
    proposals = []
    for basis, masks, qualities in batches:
        for mask, quality in zip(masks, qualities):
            count = int(mask.sum())
            if not count:
                continue
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
            component = float(stats[1:, cv2.CC_STAT_AREA].max()/count)
            image_ok = bool(.85 <= quality <= 1 and 200 <= count <= mask.size*.4
                and component >= .98 and not (mask[0].any() or mask[-1].any()
                                             or mask[:, 0].any() or mask[:, -1].any()))
            xy = projection.pixel_indices_xy
            indices = projection.source_indices[mask[xy[:, 1], xy[:, 0]]]
            near = indices[(residual[indices] > .002) & (residual[indices] <= .05)]
            yy, xx = np.where(mask)
            detection = SemanticDetection(0, "unknown", min(float(quality), .75),
                PixelBoundingBox(float(xx.min()), float(yy.min()), float(xx.max()+1), float(yy.max()+1)), mask)
            cue = estimate_silhouette_contact_proxy(detection, timestamp_seconds=source_time) if image_ok else None
            proposals.append((mask, dict(basis=basis, quality=float(quality), pixels=count,
                image_gates=image_ok, dominant_component=component,
                visible_raw_hits=len(indices), visible_near_floor_hits=len(near),
                original_eight_return_gate=bool(len(near) >= model.config.minimum_projected_points),
                contact_outline_diagnostic=asdict(cue) if cue else None,
                runtime_admissible=False)))
    return proposals, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-fixture", type=Path, required=True)
    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--last-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh audit output required")
    args.output.mkdir(parents=True)
    settings = NearFloorDiagnosticConfig(candidate_residual_max_m=.05)
    model = SamInstanceSegmenter(SamInstanceConfig(ROOT/"outputs/sam2_integration/sam2.1_t.pt")) if args.sam else None
    report = dict(scope="offline stationary sensing diagnostic, not runtime/four-factor/navigation acceptance",
                  search_config=asdict(settings), sources={})
    for folder in args.source:
        records = json.loads((folder/"stationary-fields.json").read_text())
        if args.last_only:
            records = records[-1:]
        rows = []
        calibration = FThetaCameraCalibration(**json.loads((folder/"camera-calibration.json").read_text()))
        for record in records:
            frame = record["sensor_frame_id"]
            with np.load(folder/"sensor-snapshots"/f"{frame:06d}.npz") as scan:
                points = scan["scene_xyz"]
                meta = json.loads((folder/"sensor-snapshots"/f"{frame:06d}.json").read_text())
                result = diagnose_near_floor(points, floor_reference_m=-float(scan["world_from_robot"][2, 3]),
                    reference_scope="Isaac oracle ego pose/declared world floor band; plane fitted to current LiDAR",
                    source_time=meta["lidar_time"], observation_time=meta["camera_time"], config=settings)
                result["frame"] = frame
                proposals = []
                if model and result["floor_fit_available"]:
                    rgb = cv2.cvtColor(cv2.imread(str(folder/"sensor-snapshots"/f"{frame:06d}.png")), cv2.COLOR_BGR2RGB)
                    projection = depth_visible_projection(project_fisheye_polynomial_points(points,
                        camera_from_source=scan["camera_from_scan_robot"], calibration=calibration), radius_pixels=4)
                    residual = points[:, 2] - np.column_stack((points[:, :2], np.ones(len(points)))) @ result["plane"]
                    proposals, elapsed = image_diagnostics(model, rgb, projection, result["candidates"], residual, meta["camera_time"])
                    result["sam_wall_seconds"] = elapsed
                # Evaluation-only truth first enters AFTER candidate extraction and SAM.
                target = json.loads(args.evaluation_fixture.read_text())["target"]
                lo, hi = np.array(target["bounds_minimum"]), np.array(target["bounds_maximum"])
                transform = scan["world_from_robot"] @ np.linalg.inv(gravity_aligned_robot_transform(scan["world_from_robot"]))
                def truth_hits(xyz):
                    world = xyz @ transform[:3, :3].T + transform[:3, 3]
                    return ((world[:, :2] >= lo[:2]-.003).all(1) & (world[:, :2] <= hi[:2]+.003).all(1)
                            & (world[:, 2] > lo[2]+.002) & (world[:, 2] <= hi[2]+.003))
                truth = truth_hits(points)
                result["evaluation_only"] = dict(raw_region_returns=int(truth.sum()),
                    filtered_region_returns=int(truth_hits(scan["obstacle_xyz"]).sum()),
                    scope="authored bounding-region counts, not instance truth or runtime extraction")
                for candidate in result["candidates"]:
                    candidate["evaluation_region_hits"] = int(truth[candidate["indices"]].sum())
                corners = np.array(list(itertools.product(*zip(lo, hi))))
                local = (corners-transform[:3, 3]) @ transform[:3, :3]
                box_points = project_fisheye_polynomial_points(local,
                    camera_from_source=scan["camera_from_scan_robot"], calibration=calibration).pixel_indices_xy
                if not len(box_points):
                    raise ValueError("evaluation fixture outside current image")
                a, b = box_points.min(0), box_points.max(0)+1
                result["evaluation_only"]["image_box"] = [a.tolist(), b.tolist()]
                image_rows, masks = [], []
                for mask, item in proposals:
                    overlap = int(mask[a[1]:b[1], a[0]:b[0]].sum())
                    if not overlap:
                        continue
                    item["evaluation_box_overlap_fraction"] = overlap/item["pixels"]
                    if item["image_gates"] and overlap/item["pixels"] > .5:
                        item["saved_mask_index"] = len(masks)
                        masks.append(mask)
                    image_rows.append(item)
                result["image_proposals_overlapping_evaluation_box"] = image_rows
                if masks:
                    np.savez_compressed(args.output/f"{folder.name}-{frame:06d}-masks.npz", masks=np.stack(masks))
                rows.append(result)
        report["sources"][str(folder)] = rows
        print(json.dumps(dict(source=str(folder), frames=len(rows),
            raw_region_range=[min(r["evaluation_only"]["raw_region_returns"] for r in rows),
                              max(r["evaluation_only"]["raw_region_returns"] for r in rows)],
            image_candidate_frames=sum(any("saved_mask_index" in p for p in r["image_proposals_overlapping_evaluation_box"]) for r in rows))), flush=True)
        (args.output/"audit.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
