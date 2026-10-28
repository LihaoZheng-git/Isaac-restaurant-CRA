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
        human_body_contact_prior_score=None,
        motion_aware_passing=False,
        camera_prim_path=None,
        parallel_guidance_selection=False,
        maximum_identity_prior_selection_age_seconds=0.,
        identity_selection_body_envelope=False,
        predictive_human_geometry=False,
        human_prediction_uncertainty_rate=.25,
        human_prediction_uncertainty_floor=None,
        human_prediction_horizon_seconds=1.2,
        parallel_preparation=False,
        parallel_preparation_exit_lateral=None,
        human_mass_body_identity_retention=False,
        observed_motion_reference=False,
    ):
        self.stage, self.output = stage, output
        from robot.safety.motion_aware_passing import MotionAwarePassingConfig
        if parallel_guidance_selection and not motion_aware_passing:
            raise ValueError('parallel selection component requires motion-aware passing')
        if maximum_identity_prior_selection_age_seconds and not parallel_guidance_selection:
            raise ValueError('identity-selection component requires explicit parallel selector')
        if maximum_identity_prior_selection_age_seconds and not identity_selection_body_envelope:
            raise ValueError('identity-selection component requires explicit current body envelope')
        if identity_selection_body_envelope and not parallel_guidance_selection:
            raise ValueError('component body envelope requires explicit parallel selector')
        if predictive_human_geometry and not identity_selection_body_envelope:
            raise ValueError('prediction bench requires an explicit current body envelope')
        if parallel_preparation and not predictive_human_geometry:
            raise ValueError('early preparation requires explicit predictive geometry')
        if parallel_preparation_exit_lateral is not None and not parallel_preparation:
            raise ValueError('exit hysteresis requires explicit preparation')
        if human_prediction_uncertainty_floor is not None and not (parallel_preparation and predictive_human_geometry):
            raise ValueError('adaptive growth requires explicit predictive preparation')
        from robot.safety.predictive_human_geometry import HumanGeometryForecastConfig
        from robot.safety.parallel_preparation import ParallelPreparationConfig
        passing_config = MotionAwarePassingConfig(require_visual_person_selection=True,
            encounter_geometry='oncoming_parallel',
            maximum_identity_prior_selection_age_seconds=maximum_identity_prior_selection_age_seconds
            ) if parallel_guidance_selection else MotionAwarePassingConfig()
        self.passing_goal_world_xy = (28.8, 10.5) if parallel_guidance_selection else (28.8, 9.4) if motion_aware_passing else (30.6, 11.8)
        if motion_aware_passing:
            self.save("motion-passing-configuration.json", {
                "configuration":asdict(passing_config),
                "nominal_waypoint_context_world_xy":self.passing_goal_world_xy,
                "scope":"parked robot, current observed motion selection; hypothetical service waypoint context, zero wheels",
            })
        self.carried_object_groups=carried_object_groups
        from robot.perception.human_body_contact import HumanBodyContactPriorConfig
        body_contact_config = (None if human_body_contact_prior_score is None
            else HumanBodyContactPriorConfig(human_body_contact_prior_score))
        self.camera = NovaCarterCamera(resolution=(400, 640),
            **({"camera_prim_path": camera_prim_path} if camera_prim_path is not None else {}))
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
        if calibration.output_width * calibration.nominal_height != calibration.output_height * calibration.nominal_width:
            raise ValueError("selected camera acquisition must match its authored aspect ratio")
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
                "camera_prim_path": self.camera.camera_prim_path,
                "camera_scope": "one actual authored camera per trial; no multi-camera fusion or synthetic wider image",
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
            observed_motion_reference=observed_motion_reference,
            human_mass_body_identity_retention=human_mass_body_identity_retention,
            predictive_human_geometry_config=HumanGeometryForecastConfig(
                horizon_seconds=human_prediction_horizon_seconds,
                velocity_uncertainty_floor_prior_mps=human_prediction_uncertainty_floor,
                velocity_uncertainty_prior_mps=human_prediction_uncertainty_rate) if predictive_human_geometry else None,
            parallel_preparation_config=ParallelPreparationConfig(exit_lateral_separation_m=parallel_preparation_exit_lateral)
                if parallel_preparation else None,
            motion_passing_config=passing_config if motion_aware_passing else None,
            human_body_envelope_config=self._identity_selection_envelope_config()
                if identity_selection_body_envelope else None,
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
                maximum_property_age_seconds=3. if human_mass_body_identity_retention else 120.,
                translate_instance_bounds_prior=False,
                minimum_visual_reassessment_interval_seconds=0.2,
                maximum_visual_tracks_per_update=2,
            ),
            object_factor_prior_config=ObjectRiskPriorConfig(),
            carried_object_config=CarriedObjectConfig(allow_occluding_subinstance=True) if carried_object_groups else None,
            pose_detector=pose_detector,
            object_height_config=ObjectHeightConfig() if carried_object_groups else None,
            human_body_contact_config=body_contact_config,
        )
        if body_contact_config is not None:
            self.save("human-body-contact-configuration.json", asdict(body_contact_config))
        self.history = EgoPoseHistory(maximum_samples=2048)
        self.truth, self.records, self.rejections = [], [], []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.copied_scan = None
        self.last_capture = -math.inf
        self.last_lidar_id = -1

    @staticmethod
    def _identity_selection_envelope_config():
        from robot.perception.human_body_envelope import HumanBodyEnvelopeConfig
        return HumanBodyEnvelopeConfig(body_radius_m=.4, centroid_uncertainty_m=.25)

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
            self.passing_goal_world_xy,
            pose_source=epoch.lidar_pose.source,
            pose_status=epoch.lidar_pose.status,
            nominal_route_direction_world_xy=(tuple((np.asarray(self.passing_goal_world_xy)-self.worker.anchor)
                / np.linalg.norm(np.asarray(self.passing_goal_world_xy)-self.worker.anchor))
                if self.worker.parallel_preparation is not None else None),
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
                    "sensor_frame_id": packet.sensor_frame_id,
                    "camera_time": packet.camera_timestamp_seconds,
                    "phase": phase,
                    "completion_age_sim_seconds": now - packet.sensor_timestamp_seconds,
                    "processing_wall_seconds": time.perf_counter() - start,
                    "carried_object_groups":result.diagnostics.get("carried_object_groups"),
                    "occupancy_policies":result.diagnostics.get("occupancy_policies"),
                    "motion_aware_passing":result.diagnostics.get("motion_aware_passing"),
                    "human_body_envelope_priors":result.diagnostics.get("human_body_envelope_priors", ()),
                    "human_mass_body_identity_retention":result.diagnostics.get("human_mass_body_identity_retention"),
                    "predictive_human_geometry":result.diagnostics.get("predictive_human_geometry"),
                    "observed_motion_reference":result.diagnostics.get("observed_motion_reference"),
                    "parallel_preparation":result.diagnostics.get("parallel_preparation"),
                    "motion_passing_reference_track_ids":result.field.motion_passing_reference_track_ids,
                    "motion_passing_reference_source_timestamp_seconds":result.field.motion_passing_reference_source_timestamp_seconds,
                    "applied_field_policies":result.diagnostics.get("applied_field_policies"),
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
            if self.carried_object_groups and (len(self.records)<=60 or len(self.records)%10==0):
                snapshots=self.output/"carried-snapshots"
                snapshots.mkdir(exist_ok=True)
                stem=f"{packet.sensor_frame_id:06d}"
                cv2.imwrite(str(snapshots/f"{stem}.png"),packet.rgb[:,:,::-1])
                np.savez_compressed(snapshots/f"{stem}.npz",scene_xyz=packet.scene_xyz,
                    obstacle_xyz=packet.obstacle_xyz,camera_from_scan_robot=packet.camera_from_scan_robot,
                    sensor_time=packet.sensor_timestamp_seconds,camera_time=packet.camera_timestamp_seconds,
                    camera_frame_id=packet.camera_frame_id,robot_position_world_xy=packet.robot_position_world_xy,
                    robot_yaw=packet.robot_yaw,floor_reference_m=packet.floor_height_robot_m)
                debug=getattr(getattr(self.worker,"carried_runtime_state",None),"debug_masks",None)
                if debug and debug["source_frame_id"]==packet.camera_frame_id:
                    overlay=packet.rgb[:,:,::-1].copy()
                    for key,color in (("human",np.array([255,120,0])),("object",np.array([0,80,255]))):
                        if debug[key] is not None:
                            m=debug[key]
                            overlay[m]=(.6*overlay[m]+.4*color).astype(np.uint8)
                    cv2.imwrite(str(snapshots/f"{stem}-instances.png"),overlay)
                    np.savez_compressed(snapshots/f"{stem}-masks.npz",human=debug["human"],
                        object_mask=np.zeros_like(debug["human"]) if debug["object"] is None else debug["object"],
                        source_frame_id=packet.camera_frame_id)
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
