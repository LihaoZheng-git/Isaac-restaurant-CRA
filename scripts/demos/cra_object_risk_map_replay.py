"""Stationary recorded-sensor diagnostic of persistent all-track CRA mapping.

Uses existing audited local models, never starts Kit or commands wheels. The
optional missing-mask injection is a continuity diagnostic, not natural
occlusion. All-scene factor priors are engineering assumptions, not calibration.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from robot.evaluation.live_object_perception import (
    LiveObjectPacket,
    LiveObjectPerception,
)
from robot.perception.camera_lidar_fusion import (
    FThetaCameraCalibration,
    project_fisheye_polynomial_points,
)
from robot.perception.persistent_object_evidence import PersistentObjectEvidenceConfig
from robot.perception.sam_instance_segmentation import (
    SamInstanceConfig,
    SamInstanceResult,
    SamInstanceSegmenter,
)
from robot.perception.visual_mass_runtime import (
    VisualMassRuntimeConfig,
    VisualMassRuntimeEstimator,
)
from robot.perception.yolo_detector import (
    YoloSegmentationConfig,
    YoloSegmentationDetector,
)
from robot.risk.object_risk_policy import ObjectRiskPriorConfig
from robot.risk.cra import PhysicalRiskVector
from robot.safety.class_safety_policy import ClassSafetyPolicy
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
    SafeSAGEConfigurationSpaceConfig,
)
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidancePolicy
from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
from robot.safety.poisson_safety_function import PoissonSafetyFunctionConfig
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-gap-after", type=int)
    parser.add_argument("--prior-risk", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not 0 < args.prior_risk < 1:
        parser.error("prior risk must be strictly between zero and one")
    if args.visual_gap_after is not None and args.visual_gap_after < 2:
        parser.error("a gap needs at least two initial observations")
    source = args.source.resolve()
    gate = json.loads((source / "stationary-gate.json").read_text())
    if not gate["passed"] or gate["nonzero_wheel_commands"] != 0:
        raise ValueError("source must have an accepted stationary gate")
    original = json.loads((source / "stationary-fields.json").read_text())
    calibration = FThetaCameraCalibration(
        **json.loads((source / "camera-calibration.json").read_text())
    )
    args.output.mkdir(parents=True, exist_ok=True)
    weights = gate["weights"]
    for item in weights.values():
        path = Path(item["path"])
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError(
                "recorded local model hash differs; no download or substitution"
            )
    sam = SamInstanceSegmenter(
        SamInstanceConfig(Path(weights["sam"]["path"]), device=args.device)
    )
    mass = VisualMassRuntimeEstimator(
        VisualMassRuntimeConfig(
            Path(weights["mass"]["path"]),
            device=args.device,
            allow_category_free_instances=True,
        )
    )
    contact = YoloSegmentationDetector(
        YoloSegmentationConfig(
            Path(weights["contact"]["path"]),
            device=args.device,
            confidence_threshold=0.25,
            allowed_classes=("exposed_edge",),
        )
    )
    contact.initialize()

    class Gap:
        def __init__(self):
            self.index = 0

        def set_image(self, rgb):
            if args.visual_gap_after is None or self.index < args.visual_gap_after:
                sam.set_image(rgb)

        def propose(self, *positional, **keyword):
            if (
                args.visual_gap_after is not None
                and self.index >= args.visual_gap_after
            ):
                return SamInstanceResult(
                    None, ("stationary replay injected missing SAM output",)
                )
            return sam.propose(*positional, **keyword)

    segmenter = Gap()
    guidance = SafeSAGEGuidancePolicy.from_json(
        ROOT / "configs/safe_sage_paper_baseline.json"
    )
    classes = ClassSafetyPolicy(
        {
            name: item.safety_parameters
            for name, item in guidance.class_parameters.items()
        }
    )
    field_config = SafeSAGEFieldPipelineConfig(
        configuration_space_config=SafeSAGEConfigurationSpaceConfig(
            footprint=RectangularRobotFootprint(0.04, 0.687351, 0.2466, 0.2466, 0.05),
            yaw_slice_count=16,
        ),
        guidance_field_config=TangentialLaplaceGuidanceFieldConfig(
            allow_dense_scene_arbitration=True,
            solver_backend="sparse_direct",
            interface_backend="vectorized",
        ),
        h_full_config=SafeSAGEHFullConfig(
            yaw_slice_count=16,
            poisson_config=PoissonSafetyFunctionConfig(solver_backend="sparse_direct"),
        ),
    )
    runtime = None
    records, positions, yaws = [], [], []
    previous_position = previous_time = None
    started = time.perf_counter()
    for index, original_record in enumerate(original):
        stem = f"{original_record['sensor_frame_id']:06d}"
        snapshot = source / "sensor-snapshots" / stem
        meta = json.loads(snapshot.with_suffix(".json").read_text())
        with np.load(snapshot.with_suffix(".npz")) as data:
            pose = data["world_from_robot"]
            xy = pose[:2, 3]
            yaw = math.atan2(pose[1, 0], pose[0, 0])
            positions.append(xy.copy())
            yaws.append(yaw)
            if np.linalg.norm(xy - positions[0]) > 0.02 or abs(yaw - yaws[0]) > 0.02:
                raise ValueError("source exceeds this stationary replay scope")
            if runtime is None:
                c, s = math.cos(yaw), math.sin(yaw)
                goal = xy + np.array([2.5 * c, 2.5 * s])
                runtime = LiveObjectPerception(
                    segmenter=segmenter,
                    mass_estimator=mass,
                    contact_detector=contact,
                    calibration=calibration,
                    class_policy=classes,
                    guidance_policy=guidance,
                    grid_config=LocalOccupancyGridConfig(
                        resolution=0.1,
                        minimum_x=-2,
                        maximum_x=5,
                        minimum_y=-3,
                        maximum_y=3,
                    ),
                    field_config=field_config,
                    anchor_position_world_xy=xy,
                    anchor_yaw=yaw,
                    ground_prior_scope="original bounded separated sphere; observed floor with assumed unseen local continuity",
                    negative_plume_scope="original accepted stationary clean-reference rendered-plume scope; replay only",
                    rectify_plume=True,
                    persistent_property_config=PersistentObjectEvidenceConfig(
                        "stationary recorded static objects; mass, visible geometry, support and plume assumed unchanged for this 120 s session"
                    ),
                    object_factor_prior_config=ObjectRiskPriorConfig(
                        PhysicalRiskVector(
                            args.prior_risk,
                            args.prior_risk,
                            args.prior_risk,
                            args.prior_risk,
                        )
                    ),
                )
            velocity = (
                np.zeros(2)
                if previous_position is None
                else (xy - previous_position) / (meta["lidar_time"] - previous_time)
            )
            rgb = cv2.cvtColor(
                cv2.imread(str(snapshot.with_suffix(".png"))), cv2.COLOR_BGR2RGB
            )
            packet = LiveObjectPacket(
                meta["lidar_id"],
                meta["lidar_time"],
                meta["camera_id"],
                meta["camera_time"],
                rgb,
                data["scene_xyz"],
                data["obstacle_xyz"],
                tuple(xy),
                yaw,
                tuple(velocity),
                data["camera_from_scan_robot"],
                -pose[2, 3],
                tuple(goal),
                clean_reference_validated=index == 0,
            )
        segmenter.index = index
        if args.visual_gap_after is not None and index == args.visual_gap_after:
            runtime.persistent_properties.freeze_properties()
        result = runtime(packet)
        records.append(
            {
                "sensor_frame_id": packet.sensor_frame_id,
                "camera_timestamp": packet.camera_timestamp_seconds,
                "diagnostics": result.diagnostics,
                "objects": [
                    {
                        "track_id": item.attributes.track_id,
                        "policy": item.policy.as_dict(),
                        "attributes": asdict(item.attributes),
                        "readiness": asdict(item.readiness),
                    }
                    for item in result.objects
                ],
            }
        )
        print(
            json.dumps(
                {
                    "frame": index + 1,
                    "tracks": len(result.objects),
                    "accepted_instances": result.diagnostics["accepted_instance_count"],
                    "retained_properties": sum(
                        value["mode"] == "persistence_prior"
                        for item in result.objects
                        for value in item.diagnostics["property_evidence"].values()
                    ),
                    "phi_range": [
                        min(item.policy.risk for item in result.objects),
                        max(item.policy.risk for item in result.objects),
                    ],
                },
                default=serial,
            ),
            flush=True,
        )
        previous_position, previous_time = xy.copy(), meta["lidar_time"]
    scores = [item.policy.risk for item in result.objects]
    if not runtime.persistent_properties.frozen:
        runtime.persistent_properties.freeze_properties()
    report = {
        "passed": len(records) >= 2
        and all(
            np.isfinite(item["policy"]["cra"]["phi"])
            and len(item["policy"]["cra"]["factors"]) == 4
            for record in records
            for item in record["objects"]
        ),
        "scope": "stationary recorded RGB/LiDAR multi-track dataflow; not fresh Isaac acquisition, moving-camera validation, wheel acceptance or calibrated priors",
        "source": str(source),
        "source_plume_control": gate["plume_control"],
        "source_pose_and_floor_status": "oracle",
        "nonzero_wheel_commands": 0,
        "frames": len(records),
        "object_count_range": [
            min(record["diagnostics"]["object_count"] for record in records),
            max(record["diagnostics"]["object_count"] for record in records),
        ],
        "last_phi_range": [min(scores), max(scores)],
        "accepted_instance_updates": sum(
            record["diagnostics"]["accepted_instance_count"] for record in records
        ),
        "complete_perception_updates": sum(
            record["diagnostics"]["complete_perception_count"] for record in records
        ),
        "retained_property_uses": sum(
            value["mode"] == "persistence_prior"
            for record in records
            for obj in record["diagnostics"]["objects"]
            for value in obj["property_evidence"].values()
        ),
        "factor_prior_risk": args.prior_risk,
        "all_unknown_phi": 1 - (1 - args.prior_risk) ** 4,
        "visual_gap_after": args.visual_gap_after,
        "wall_seconds": time.perf_counter() - started,
        "weights": weights,
        "fixed_lambda": 1.25,
    }
    for name, value in (("report.json", report), ("frames.json", records)):
        (args.output / name).write_text(
            json.dumps(value, indent=2, default=serial, allow_nan=False) + "\n"
        )
    overlay = cv2.cvtColor(packet.rgb, cv2.COLOR_RGB2BGR)
    projection = project_fisheye_polynomial_points(
        packet.obstacle_xyz,
        camera_from_source=packet.camera_from_scan_robot,
        calibration=calibration,
    )
    for obj in result.objects:
        indices = (
            result.field.tracking_result.point_track_ids[projection.source_indices]
            == obj.attributes.track_id
        )
        if indices.any():
            xy_pixel = np.median(projection.pixel_xy[indices], axis=0).astype(int)
            cv2.putText(
                overlay,
                f"id{obj.attributes.track_id} phi={obj.policy.risk:.2f}",
                tuple(xy_pixel),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        overlay,
        "Stationary sensor replay; scores include labeled priors",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    cv2.imwrite(str(args.output / "object-risk-overlay.png"), overlay)
    print(json.dumps(report, default=serial), flush=True)
    if not report["passed"]:
        raise RuntimeError("multi-object stationary replay gate failed")


if __name__ == "__main__":
    main()
