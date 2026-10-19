"""Parked-robot articulated-human sensor diagnostic; import after SimulationApp.

Native actor positions enter only the separate evaluation history. The worker
receives copied RGB/LiDAR, robot pose/extrinsic references and waypoint context.
Asynchronous completion does not refresh acquisition epochs or command wheels.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import time

import cv2
import numpy as np

from robot.config import ROBOT_BASE_PRIM_PATH
from robot.evaluation.human_sensor_validation import summarize_human_sensor
from robot.evaluation.live_object_perception import (
    LiveObjectPacket,
    LiveObjectPerception,
)
from robot.perception.human_mass_prior import (
    HumanMassPriorConfig,
    HUMAN_MASS_PRIOR_SOURCE,
    PERSON_CHECKPOINT_SHA256,
)
from robot.perception.lidar_deskew import deskew_uncompensated_lidar
from robot.perception.physical_attributes import EstimateStatus
from robot.perception.persistent_object_evidence import PersistentObjectEvidenceConfig
from robot.perception.sam_instance_segmentation import (
    SamInstanceConfig,
    SamInstanceSegmenter,
)
from robot.perception.sensor_epochs import (
    EgoPoseHistory,
    EgoPoseSample,
    align_camera_lidar_epoch,
    gravity_aligned_robot_transform,
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
from robot.safety.class_safety_policy import ClassSafetyPolicy
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.obstacle_detection import (
    ObstacleDetectionConfig,
    detect_obstacle_points_3d,
)
from robot.safety.poisson_safety_function import PoissonSafetyFunctionConfig
from robot.safety.safe_sage_configuration_space import (
    RectangularRobotFootprint,
    SafeSAGEConfigurationSpaceConfig,
)
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidancePolicy
from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
from robot.safety.safe_sage_unicycle_cbf_filter import (
    shift_rectangular_footprint_to_lookahead,
)
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
)
from robot.scene.camera_calibration import read_fisheye_polynomial_calibration
from robot.scene.transforms import compute_relative_prim_transform
from robot.sensor.camera import NovaCarterCamera
from robot.sensor.lidar import NovaCarterLidar
from robot.sensor.lidar_geometry import transform_points


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class RestaurantHumanSensorDiagnostic:
    def __init__(
        self, stage, root, output, anchor, *, mass_interval=(70.0, 45.0, 100.0), cluster_radius_m=.12,
        carried_object_groups=False,
    ):
        self.stage, self.output = stage, output
        self.carried_object_groups=carried_object_groups
        self.camera = NovaCarterCamera(resolution=(400, 640))
        self.lidar = NovaCarterLidar()
        self.camera.initialize()
        self.lidar.initialize()
        self.robot_from_lidar = compute_relative_prim_transform(
            stage,
            source_prim_path=self.lidar.lidar_prim_path,
            target_prim_path=ROBOT_BASE_PRIM_PATH,
        )
        self.camera_from_robot = compute_relative_prim_transform(
            stage,
            source_prim_path=ROBOT_BASE_PRIM_PATH,
            target_prim_path=self.camera.camera_prim_path,
        )
        calibration = read_fisheye_polynomial_calibration(
            stage,
            camera_prim_path=self.camera.camera_prim_path,
            output_resolution=(400, 640),
        )
        paths = {
            "sam": root / "outputs/sam2_integration/sam2.1_t.pt",
            "mass": root
            / "outputs/visual_mass_data/abo/baseline_frozen_final_v2/best.pt",
            "contact": root
            / "runs/segment/outputs/contact_perception/training/yolo11n_seg_paco_geometry/weights/best.pt",
            "person": root / "outputs/restaurant_human_model_audit/yolo11n-seg.pt",
        }
        hashes = {
            k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()
        }
        if hashes["person"] != PERSON_CHECKPOINT_SHA256:
            raise ValueError(
                "person checkpoint differs from the audited official weights"
            )
        self.save(
            "sensor-model-audit.json",
            {
                "sha256": hashes,
                "paths": {k: str(p) for k, p in paths.items()},
                "person_source": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
                "person_license": "Ultralytics AGPL-3.0; externally downloaded, not vendored",
                "training_data": "COCO pretrained; no dataset downloaded or retraining",
                "calibration": asdict(calibration),
            },
        )
        person = YoloSegmentationDetector(
            YoloSegmentationConfig(
                paths["person"], confidence_threshold=0.5, allowed_classes=("person",)
            )
        )
        contact = YoloSegmentationDetector(
            YoloSegmentationConfig(
                paths["contact"],
                confidence_threshold=0.25,
                allowed_classes=("exposed_edge",),
            )
        )
        person.initialize()
        contact.initialize()
        pose_detector=None
        if carried_object_groups:
            from robot.perception.human_pose import YoloHumanPose,POSE_SHA256,POSE_URL
            from robot.perception.carried_object import CarriedObjectConfig
            from robot.perception.object_height import ObjectHeightConfig
            checkpoint=root/"outputs/carried_object_model_audit/yolo11n-pose.pt"
            pose_detector=YoloHumanPose(checkpoint)
            pose_detector.initialize()
            self.save("carried-pose-model-audit.json",{"path":str(checkpoint),"sha256":POSE_SHA256,
                "source":POSE_URL,"license":"Ultralytics AGPL-3.0 / Enterprise; not vendored",
                "training_data":"COCO keypoints pretrained; no retraining/data download",
                "scope":"RGB wrists, not measured physical hand position or grasp state"})
        guidance = SafeSAGEGuidancePolicy.from_json(
            root / "configs/safe_sage_class_safety.json"
        )
        policy = ClassSafetyPolicy(
            {key: p.safety_parameters for key, p in guidance.class_parameters.items()},
            unknown_class=guidance.unknown_class,
            risk_flux_mapping=guidance.risk_flux_mapping,
        )
        footprint = shift_rectangular_footprint_to_lookahead(
            RectangularRobotFootprint(0.14, 0.587351, 0.2466, 0.2466, 0.05),
            lookahead_distance=0.1,
        )
        self.worker = LiveObjectPerception(
            segmenter=SamInstanceSegmenter(
                SamInstanceConfig(
                    paths["sam"],
                    device="0",
                    observed_context_box_retry=True,
                    foreground_depth_gap_m=0.5,
                )
            ),
            mass_estimator=VisualMassRuntimeEstimator(
                VisualMassRuntimeConfig(
                    paths["mass"], device="0", allow_category_free_instances=True
                )
            ),
            contact_detector=contact,
            person_detector=person,
            human_mass_prior_config=HumanMassPriorConfig(
                "bounded adult-human restaurant component; configured mass interval, not calibration",
                mass_kg=mass_interval[0],
                lower_kg=mass_interval[1],
                upper_kg=mass_interval[2],
            ),
            calibration=calibration,
            class_policy=policy,
            guidance_policy=guidance,
            grid_config=LocalOccupancyGridConfig(
                resolution=0.1,
                minimum_x=-2,
                maximum_x=5,
                minimum_y=-3,
                maximum_y=3,
                obstacle_inflation_radius=0,
            ),
            field_config=SafeSAGEFieldPipelineConfig(
                SafeSAGEConfigurationSpaceConfig(
                    footprint=footprint,
                    yaw_slice_count=16,
                    expansion_backend="vectorized",
                ),
                TangentialLaplaceGuidanceFieldConfig(
                    convergence_tolerance=1e-6,
                    tangential_buffer_radius=0.6,
                    allow_dense_scene_arbitration=True,
                    solver_backend="sparse_direct",
                    interface_backend="vectorized",
                ),
                SafeSAGEHFullConfig(
                    yaw_slice_count=16,
                    poisson_config=PoissonSafetyFunctionConfig(
                        convergence_tolerance=1e-6, solver_backend="sparse_direct"
                    ),
                ),
            ),
            anchor_position_world_xy=anchor[:2],
            anchor_yaw=anchor[2],
            occupancy_retention_seconds=0.15,
            cluster_in_3d=True,
            cluster_radius_m=cluster_radius_m,
            camera_visibility_radius_pixels=4,
            mask_foreground_depth_gap_m=0.5,
            rectify_plume=True,
            persistent_property_config=PersistentObjectEvidenceConfig(
                "articulating human diagnostic: fresh inputs only; no unchanged-shape or support/plume persistence",
                retain_observed_properties=False,
                translate_instance_bounds_prior=False,
                minimum_visual_reassessment_interval_seconds=0.2,
                maximum_visual_tracks_per_update=2,
            ),
            object_factor_prior_config=ObjectRiskPriorConfig(),
            carried_object_config=CarriedObjectConfig() if carried_object_groups else None,
            pose_detector=pose_detector,
            object_height_config=ObjectHeightConfig() if carried_object_groups else None,
        )
        self.history = EgoPoseHistory(maximum_samples=2048)
        self.truth, self.records, self.rejections = [], [], []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.copied_scan = None
        self.last_capture = -math.inf
        self.last_lidar_id = -1

    def save(self, name, data):
        (self.output / name).write_text(
            json.dumps(data, indent=2, default=_json, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def tick(self, now, actor_xyz, phase):
        # This history is used only by finish() for independent matching/errors.
        self.truth.append({"time": now, "actor_xyz": list(actor_xyz), "phase": phase})
        pose = compute_relative_prim_transform(
            self.stage, source_prim_path=ROBOT_BASE_PRIM_PATH, target_prim_path="/World"
        )
        self.history.append(
            EgoPoseSample(now, pose, EstimateStatus.ORACLE, "isaac_usd_chassis_pose")
        )
        self._poll(now)
        if self.pending is not None or (
            self.copied_scan is None and now - self.last_capture < 0.1 - 1e-6
        ):
            return
        try:
            captured = self._capture(now, phase)
            if captured is None:
                return
            packet, source_phase = captured
        except (ValueError, RuntimeError) as e:
            self.copied_scan = None
            if not any(r["reason"] == str(e) for r in self.rejections):
                print("Human sensor capture rejection:", str(e), flush=True)
            self.rejections.append({"time": now, "stage": "capture", "reason": str(e)})
            return
        self.pending = (
            self.executor.submit(self.worker, packet),
            packet,
            source_phase,
            time.perf_counter(),
        )

    def _capture(self, now, phase):
        if self.copied_scan is None:
            raw = self.lidar.get_point_cloud().copy()
            info = self.lidar.get_frame_info()
            if info.sensor_frame_id <= self.last_lidar_id:
                return None
            rgb = self.camera.get_rgb().copy()
            camera_info = self.camera.get_frame_info()
            metadata = self.lidar.get_scan_metadata()
            self.last_lidar_id = info.sensor_frame_id
            self.copied_scan = (raw, info, rgb, camera_info, metadata, now, phase)
        raw, info, rgb, camera_info, metadata, queued_time, source_phase = (
            self.copied_scan
        )
        scan_time = info.sensor_timestamp_nanoseconds * 1e-9
        offsets = metadata.point_time_offsets_nanoseconds
        if offsets is None or not len(offsets):
            raise ValueError("copied scan lacks per-return acquisition epochs")
        last_return_time = scan_time + int(offsets.max()) * 1e-9
        self.save(
            "capture-epoch-audit.json",
            {
                "scan_time": scan_time,
                "camera_read_time": camera_info.simulation_time_seconds,
                "history_first": self.history.samples[0].timestamp_seconds,
                "history_last": self.history.samples[-1].timestamp_seconds,
                "lidar_frame_info": asdict(info),
                "scan_metadata": metadata.summary(),
                "last_return_time": last_return_time,
                "copied_scan_wait_seconds": now - queued_time,
            },
        )
        # Do not extrapolate a pose or shift a source timestamp. A copied scan
        # can wait at most .05 simulation seconds for its already-declared
        # return epochs to enter the pose history.
        if now - queued_time > 0.05 + 1e-9:
            raise ValueError("copied scan exceeded bounded pose-availability wait")
        if last_return_time > self.history.samples[-1].timestamp_seconds + 1e-9:
            return None
        epoch = align_camera_lidar_epoch(
            self.history,
            lidar_timestamp_seconds=scan_time,
            camera_timestamp_seconds=camera_info.simulation_time_seconds,
            camera_from_robot=self.camera_from_robot,
        )
        scene, deskew = deskew_uncompensated_lidar(
            raw,
            metadata,
            self.history,
            robot_from_sensor=self.robot_from_lidar,
            reference_timestamp_seconds=scan_time,
        )
        pose = epoch.lidar_pose.world_from_robot
        gravity = gravity_aligned_robot_transform(pose)
        scene = transform_points(scene, gravity)
        floor = -float(pose[2, 3])
        obstacles = detect_obstacle_points_3d(
            scene,
            ObstacleDetectionConfig(
                minimum_height=floor + 0.05,
                maximum_height=floor + 1.8,
                minimum_range=0.15,
                maximum_range=7.5,
                forward_only=False,
                angular_bin_size_degrees=None,
            ),
        )
        packet = LiveObjectPacket(
            info.sensor_frame_id,
            scan_time,
            camera_info.read_index,
            camera_info.simulation_time_seconds,
            rgb,
            scene,
            obstacles,
            tuple(pose[:2, 3]),
            math.atan2(pose[1, 0], pose[0, 0]),
            self.history.velocity_world_xy(scan_time),
            epoch.camera_from_scan_robot @ np.linalg.inv(gravity),
            floor,
            (30.6, 11.8),
            pose_source=epoch.lidar_pose.source,
            pose_status=epoch.lidar_pose.status,
        )
        self.copied_scan = None
        # Cadence limits successful packets, not rejected/duplicate attempts.
        # Retry a rejected scan next physics tick rather than phase-locking to
        # the same stale point in the 10 Hz accumulated-scan cycle.
        self.last_capture = queued_time
        cv2.imwrite(str(self.output / "robot-current.png"), rgb[:, :, ::-1])
        self.save("latest-scan-audit.json", deskew)
        return packet, source_phase

    def _poll(self, now, wait=False):
        if self.pending is None:
            return
        future, packet, phase, start = self.pending
        if not wait and not future.done():
            return
        self.pending = None
        try:
            result = future.result()
            if not self.worker.persistent_properties.frozen:
                self.worker.persistent_properties.freeze_properties()
            self.records.append(
                {
                    "time": packet.sensor_timestamp_seconds,
                    "camera_time": packet.camera_timestamp_seconds,
                    "phase": phase,
                    "completion_age_sim_seconds": now - packet.sensor_timestamp_seconds,
                    "processing_wall_seconds": time.perf_counter() - start,
                    "carried_object_groups":result.diagnostics.get("carried_object_groups"),
                    "objects": [
                        {
                            "attributes": asdict(o.attributes),
                            "diagnostics": o.diagnostics,
                            "phi": o.policy.risk,
                        }
                        for o in result.objects
                    ],
                }
            )
            if self.carried_object_groups and (len(self.records)<=24 or len(self.records)%10==0):
                snapshots=self.output/"carried-snapshots"
                snapshots.mkdir(exist_ok=True)
                stem=f"{packet.sensor_frame_id:06d}"
                cv2.imwrite(str(snapshots/f"{stem}.png"),packet.rgb[:,:,::-1])
                np.savez_compressed(snapshots/f"{stem}.npz",scene_xyz=packet.scene_xyz,
                    obstacle_xyz=packet.obstacle_xyz,camera_from_scan_robot=packet.camera_from_scan_robot,
                    sensor_time=packet.sensor_timestamp_seconds,camera_time=packet.camera_timestamp_seconds,
                    camera_frame_id=packet.camera_frame_id,robot_position_world_xy=packet.robot_position_world_xy,
                    robot_yaw=packet.robot_yaw,floor_reference_m=packet.floor_height_robot_m)
            with (self.output / "human-sensor-records.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(
                    json.dumps(self.records[-1], default=_json, allow_nan=False) + "\n"
                )
            print(
                "Human sensor field:",
                len(self.records),
                "tracks:",
                len(result.objects),
                "fresh visual human priors:",
                sum(
                    o.attributes.mass.mass_kg.source == HUMAN_MASS_PRIOR_SOURCE
                    for o in result.objects
                ),
                flush=True,
            )
        except Exception as e:
            print("Human sensor worker rejection:", str(e), flush=True)
            self.rejections.append(
                {
                    "time": packet.sensor_timestamp_seconds,
                    "stage": "worker",
                    "reason": str(e),
                }
            )
        self.save("human-sensor-current.json", self.records[-5:])

    @property
    def stationary_ready(self):
        return (
            sum(
                any(
                    o["attributes"]["mass"]["mass_kg"]["source"]
                    == HUMAN_MASS_PRIOR_SOURCE
                    and o["attributes"]["mass"]["mass_kg"]["status"] == "prior"
                    and o["diagnostics"]["accepted_instance"]
                    for o in r["objects"]
                )
                for r in self.records
                if r["phase"] == "idle_before"
            )
            >= 3
        )

    def close(self, now):
        self._poll(now, wait=True)
        self.executor.shutdown(wait=True)
        self.save("human-sensor-records.json", self.records)
        self.save("human-sensor-truth-evaluation-only.json", self.truth)
        self.save("human-sensor-rejections.json", self.rejections)
        report = summarize_human_sensor(self.records, self.truth)
        report["capture_worker_rejections"] = self.rejections
        self.save("human-sensor-validation.json", report)
        return report
