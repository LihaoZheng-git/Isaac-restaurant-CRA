"""Bounded live object evidence joined to transactional Safe-SAGE fields.

All inputs are copied sensor observations and declared pose/fixture-domain
references. No USD reads, authored object pose/size/mass, or wheel commands are
available to this worker. Unassessed returns always retain conservative risk.
"""

import copy
from dataclasses import asdict, dataclass, replace
import math
import time

import numpy as np

from robot.evaluation.cra_experiments import RestaurantScenario
from robot.evaluation.perception_readiness import (
    evaluate_perception_readiness,
    runtime_factor_evidence_mode,
)
from robot.evaluation.simple_object_test import build_assessed_object_field
from robot.perception.camera_lidar_fusion import (
    CameraLidarFusionConfig,
    CameraLidarSemanticFusion,
    project_fisheye_polynomial_points,
)
from robot.perception.contact_detection_adapter import (
    associate_contact_parts_to_objects,
)
from robot.perception.fall_height import estimate_fall_height
from robot.perception.ground_support import estimate_spherical_ground_support
from robot.perception.instance_lidar_geometry import masked_track_return_selection
from robot.perception.lidar_fall_surfaces import estimate_lidar_fall_surfaces
from robot.perception.motion_attributes import RouteMotionContext, estimate_track_motion
from robot.perception.moving_camera_plume import (
    MovingCameraObjectPlumeEstimator,
    MovingCameraPlumeConfig,
    TrackedRgbObjectFrame,
)
from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.perception.physical_geometry import estimate_track_geometry
from robot.perception.sam_instance_segmentation import SAM_INSTANCE_SOURCE
from robot.perception.semantic_detection import SemanticDetectionFrame
from robot.perception.specialist_track_evidence import (
    SpecialistAssociationFrame,
    build_track_shape_contact_proxies,
    build_track_visual_contact_proxies,
)
from robot.perception.spherical_mass import estimate_spherical_geometry
from robot.perception.thermal_interaction import estimate_thermal_route_interaction
from robot.perception.temporal_object_evidence import TemporalObjectEvidence
from robot.perception.visual_shape_contact import (
    SHAPE_CONTACT_BASIS,
    SHAPE_CONTACT_SOURCE,
)
from robot.risk.physical_factor_adapter import (
    PhysicalFactorAdapterConfig,
    adapt_physical_factors,
)
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig
from robot.safety.safe_sage_runtime import (
    SafeSAGERuntimeField,
    robot_points_in_anchor_frame,
)
from robot.safety.safe_sage_temporal_derivative import (
    SafeSAGETemporalDerivativeEstimator,
    SafeSAGETemporalDerivativeResult,
)
from robot.safety.semantic_label_provider import (
    SemanticLabelResult,
    SemanticTrackAssignment,
)
from robot.safety.semantic_obstacle import SemanticObstacle


@dataclass(frozen=True)
class LiveObjectPacket:
    sensor_frame_id: int
    sensor_timestamp_seconds: float
    camera_frame_id: int
    camera_timestamp_seconds: float
    rgb: np.ndarray
    scene_xyz: np.ndarray
    obstacle_xyz: np.ndarray
    robot_position_world_xy: tuple
    robot_yaw: float
    measured_robot_velocity_world_xy: tuple
    camera_from_scan_robot: np.ndarray
    floor_height_robot_m: float
    waypoint_world_xy: tuple
    clean_reference_validated: bool = False
    pose_source: str = "isaac_usd_pose_history"
    pose_status: EstimateStatus = EstimateStatus.ORACLE
    scan_pose_samples_world: np.ndarray | None = None
    scan_interval_scope: str | None = None

    def __post_init__(self):
        for name in ("sensor_frame_id", "camera_frame_id"):
            if (
                isinstance(getattr(self, name), bool)
                or not isinstance(getattr(self, name), int)
                or getattr(self, name) < 0
            ):
                raise ValueError("frame IDs must be nonnegative integers")
        for name in ("sensor_timestamp_seconds", "camera_timestamp_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("invalid sensor timestamp")
        if (
            not 0
            <= self.camera_timestamp_seconds - self.sensor_timestamp_seconds
            <= 0.15 + 1e-9
        ):
            raise ValueError("camera/LiDAR packet is stale or future")
        for name in ("rgb", "scene_xyz", "obstacle_xyz", "camera_from_scan_robot"):
            value = np.array(getattr(self, name), copy=True)
            if name == "rgb":
                if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
                    raise ValueError("RGB must be uint8 HxWx3")
            elif name.endswith("xyz"):
                if (
                    value.ndim != 2
                    or value.shape[1] != 3
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("point arrays must be finite Nx3")
            else:
                from robot.perception.sensor_epochs import rigid_transform

                value = rigid_transform(value)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not len(self.obstacle_xyz):
            raise ValueError("no obstacle returns in live packet")
        for name in (
            "robot_position_world_xy",
            "measured_robot_velocity_world_xy",
            "waypoint_world_xy",
        ):
            vector = np.asarray(getattr(self, name), dtype=float)
            if vector.shape != (2,) or not np.isfinite(vector).all():
                raise ValueError(f"{name} must be a finite planar vector")
            object.__setattr__(self, name, tuple(float(v) for v in vector))
        for name in ("robot_yaw", "floor_height_robot_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not isinstance(self.clean_reference_validated, bool):
            raise TypeError("clean reference flag must be boolean")
        if not isinstance(self.pose_source, str) or not self.pose_source.strip():
            raise ValueError("pose reference source must be explicit")
        if self.pose_status not in (
            EstimateStatus.MEASURED,
            EstimateStatus.INFERRED,
            EstimateStatus.ORACLE,
        ):
            raise ValueError("pose reference must be available")
        if self.scan_pose_samples_world is not None:
            from robot.perception.sensor_epochs import rigid_transform

            poses = np.array(
                [rigid_transform(value) for value in self.scan_pose_samples_world]
            )
            if poses.ndim != 3 or not len(poses):
                raise ValueError(
                    "scan poses must be a nonempty rigid-transform sequence"
                )
            if (
                not isinstance(self.scan_interval_scope, str)
                or not self.scan_interval_scope.strip()
            ):
                raise ValueError(
                    "scan interval occupancy needs an explicit timing/domain assumption"
                )
            poses.setflags(write=False)
            object.__setattr__(self, "scan_pose_samples_world", poses)


@dataclass(frozen=True)
class LiveObjectResult:
    field: SafeSAGERuntimeField
    attributes: PhysicalAttributes
    adapted: object
    policy: object
    readiness: object
    detections: SemanticDetectionFrame
    fusion: object | None
    diagnostics: dict
    objects: tuple = ()


class LiveObjectPerception:
    def __init__(
        self,
        *,
        segmenter,
        mass_estimator,
        contact_detector,
        calibration,
        class_policy,
        guidance_policy,
        grid_config,
        field_config,
        anchor_position_world_xy,
        anchor_yaw,
        ground_prior_scope=None,
        negative_plume_scope=None,
        rectify_plume=False,
        occupancy_retention_seconds=0.0,
        static_occupancy_memory_prior_scope=None,
        component_prompt_radius_m=0.04,
        evidence_retention_config=None,
        persistent_property_config=None,
        object_factor_prior_config=None,
    ):
        self.segmenter, self.mass_estimator, self.contact_detector = (
            segmenter,
            mass_estimator,
            contact_detector,
        )
        self.calibration = calibration
        self.evidence_memory = (
            TemporalObjectEvidence(evidence_retention_config)
            if evidence_retention_config is not None
            else None
        )
        from robot.perception.persistent_object_evidence import PersistentObjectEvidence
        from robot.risk.object_risk_policy import ObjectRiskPriorConfig

        if (persistent_property_config is None) != (object_factor_prior_config is None):
            raise ValueError(
                "object-map mode requires both property scope and factor-prior policy"
            )
        self.persistent_properties = (
            PersistentObjectEvidence(persistent_property_config)
            if persistent_property_config is not None
            else None
        )
        if object_factor_prior_config is not None and not isinstance(
            object_factor_prior_config, ObjectRiskPriorConfig
        ):
            raise TypeError("invalid object factor-prior policy")
        self.object_factor_prior_config = object_factor_prior_config
        self.last_visual_track_id = None
        if (
            not math.isfinite(component_prompt_radius_m)
            or not 0.005 <= component_prompt_radius_m <= 0.1
        ):
            raise ValueError("component prompt radius must lie within [.005, .1] m")
        self.component_prompt_radius_m = component_prompt_radius_m
        self.rectify_plume = rectify_plume
        if (
            not math.isfinite(occupancy_retention_seconds)
            or not 0 <= occupancy_retention_seconds <= 0.5
        ):
            raise ValueError("occupancy retention must lie within [0, .5] seconds")
        self.occupancy_retention_seconds = occupancy_retention_seconds
        self.retained = []
        if static_occupancy_memory_prior_scope is not None and (
            not isinstance(static_occupancy_memory_prior_scope, str)
            or not static_occupancy_memory_prior_scope.strip()
        ):
            raise ValueError("static occupancy memory requires an explicit scope")
        self.static_scope = static_occupancy_memory_prior_scope
        self.static_points = {}
        self.static_oldest_time = None
        self.class_policy, self.guidance_policy = class_policy, guidance_policy
        self.grid_config, self.field_config = grid_config, field_config
        self.anchor = np.array(anchor_position_world_xy, dtype=float)
        self.anchor_yaw = float(anchor_yaw)
        self.ground_scope, self.negative_scope = (
            ground_prior_scope,
            negative_plume_scope,
        )
        self.tracker = ObstacleTracker(
            ObstacleTrackerConfig(
                cluster_radius=0.20,
                minimum_cluster_points=4,
                maximum_association_distance=0.50,
                clustering_backend="kdtree",
            )
        )
        self.plume = MovingCameraObjectPlumeEstimator(
            MovingCameraPlumeConfig(
                allow_reference_recovery=rectify_plume,
                background_scope="above_source" if rectify_plume else "full_image",
            )
        )
        self.previous = None
        self.temporal = SafeSAGETemporalDerivativeEstimator()

    def __call__(self, packet: LiveObjectPacket):
        if self.persistent_properties is not None:
            from robot.evaluation.multi_object_perception import assess_all_objects

            return assess_all_objects(self, packet)
        started = time.perf_counter()
        if self.previous is not None and (
            packet.sensor_frame_id <= self.previous.sensor_frame_id
            or packet.sensor_timestamp_seconds <= self.previous.sensor_timestamp_seconds
        ):
            raise ValueError("live sensor epochs must increase")
        candidate_tracker = copy.deepcopy(self.tracker)
        candidate_plume = copy.deepcopy(self.plume)
        candidate_memory = copy.deepcopy(self.evidence_memory)
        tracking = candidate_tracker.update(
            packet.obstacle_xyz[:, :2],
            robot_position_world_xy=np.array(packet.robot_position_world_xy),
            robot_yaw=packet.robot_yaw,
            timestamp_seconds=packet.sensor_timestamp_seconds,
        )
        if not tracking.tracked_obstacles:
            raise RuntimeError("no geometric track is available")
        # Select an observable nearby geometric track, never an authored object
        # position or class. All other returns remain in the safety field.
        projection = project_fisheye_polynomial_points(
            packet.obstacle_xyz,
            camera_from_source=packet.camera_from_scan_robot,
            calibration=self.calibration,
        )
        eligible = [
            track
            for track in tracking.tracked_obstacles
            if -0.2 <= track.centroid_robot_xy[0] <= 2.5
            and abs(track.centroid_robot_xy[1]) <= 1.5
            and np.count_nonzero(
                tracking.point_track_ids[projection.source_indices] == track.track_id
            )
            >= 8
        ]
        eligible.sort(key=lambda track: np.linalg.norm(track.centroid_robot_xy))
        preferred = next(
            (
                track
                for track in tracking.tracked_obstacles
                if track.track_id == self.last_visual_track_id
                and candidate_memory is not None
                and candidate_memory.has_unexpired(
                    track.track_id, packet.camera_timestamp_seconds
                )
            ),
            None,
        )
        if preferred is not None and (preferred in eligible or not eligible):
            target = preferred
        elif eligible:
            target = eligible[0]
        else:
            target = min(
                tracking.tracked_obstacles,
                key=lambda track: np.linalg.norm(track.centroid_robot_xy),
            )
        assessment = self._assess_track(
            packet,
            tracking,
            projection,
            target,
            target in eligible,
            candidate_plume,
            candidate_memory,
        )
        attributes, adapted, frame, fusion, selected, accepted, base = (
            assessment[name]
            for name in (
                "attributes",
                "adapted",
                "frame",
                "fusion",
                "selected",
                "accepted",
                "base",
            )
        )
        timestamp = packet.camera_timestamp_seconds
        (
            association_done,
            physical_done,
            retention_diagnostics,
            geometry_diagnostics,
            sam_quality,
            prompt_basis,
            sam_candidates,
            reasons,
        ) = (
            assessment[name]
            for name in (
                "association_done",
                "physical_done",
                "retention_diagnostics",
                "geometry_diagnostics",
                "sam_quality",
                "prompt_basis",
                "sam_candidates",
                "reasons",
            )
        )
        semantic = SemanticObstacle(
            target, "unknown", base.semantic_confidence, base.semantic_source
        )

        def anchored(points):
            if packet.scan_pose_samples_world is not None:
                # With no per-ray acquisition times, protect every possible
                # placement through the recorded scan interval. This is an
                # explicit static-fixture timing assumption, not deskewed data.
                world = np.concatenate(
                    [
                        points @ pose[:3, :3].T + pose[:3, 3]
                        for pose in packet.scan_pose_samples_world
                    ]
                )
                delta = world[:, :2] - self.anchor
                c, s = math.cos(self.anchor_yaw), math.sin(self.anchor_yaw)
                return delta @ np.array([[c, -s], [s, c]])
            return robot_points_in_anchor_frame(
                points[:, :2],
                robot_position_world_xy=np.array(packet.robot_position_world_xy),
                robot_yaw=packet.robot_yaw,
                anchor_position_world_xy=self.anchor,
                anchor_yaw=self.anchor_yaw,
            )

        assessed_points = anchored(packet.obstacle_xyz[selected])
        unassessed_points = anchored(packet.obstacle_xyz[~selected])

        def coalesce_cells(points):
            # Same-group points in the same raster cell have identical policy
            # and inflation effects. Preserve first occurrence, exact cells,
            # and raw counts separately; this is not perception subsampling.
            if not len(points):
                return points
            indices = np.floor(
                (points - [self.grid_config.minimum_x, self.grid_config.minimum_y])
                / self.grid_config.resolution
            ).astype(np.int64)
            _, first = np.unique(indices, axis=0, return_index=True)
            return points[np.sort(first)]

        assessed_cells, unassessed_cells = (
            coalesce_cells(assessed_points),
            coalesce_cells(unassessed_points),
        )
        current_placements = (
            packet.sensor_timestamp_seconds,
            packet.sensor_frame_id,
            target.track_id,
            assessed_cells,
            unassessed_cells,
        )
        retained = [
            item
            for item in self.retained
            if 0
            < packet.sensor_timestamp_seconds - item[0]
            <= self.occupancy_retention_seconds + 1e-9
        ]
        assessed_history = [assessed_cells]
        unassessed_history = [unassessed_cells]
        for item in retained:
            (
                assessed_history
                if accepted and item[2] == target.track_id
                else unassessed_history
            ).append(item[3])
            unassessed_history.append(item[4])
        if self.static_scope:
            for old_track, old_points in self.static_points.items():
                (
                    assessed_history
                    if accepted and old_track == target.track_id
                    else unassessed_history
                ).append(old_points)
        assessed_cells = coalesce_cells(np.concatenate(assessed_history))
        unassessed_cells = coalesce_cells(np.concatenate(unassessed_history))
        candidate_static = dict(self.static_points)
        if self.static_scope:
            for key, points in (
                (target.track_id, current_placements[3]),
                (-1, current_placements[4]),
            ):
                candidate_static[key] = coalesce_cells(
                    np.concatenate(
                        (candidate_static.get(key, np.empty((0, 2))), points)
                    )
                )
        policy, pipeline = build_assessed_object_field(
            assessed_cells,
            adapted=adapted,
            semantic=semantic,
            class_policy=self.class_policy,
            guidance_policy=self.guidance_policy,
            grid_config=self.grid_config,
            field_config=self.field_config,
            unassessed_points_xy=unassessed_cells,
        )
        temporal = 0.0
        if self.previous is not None:
            previous_estimate = self.previous.temporal_safety_derivative
            temporal = self.temporal.estimate(
                previous_h_full=self.previous.pipeline.h_full,
                current_h_full=pipeline.h_full,
                grid_config=self.grid_config,
                time_delta_seconds=packet.sensor_timestamp_seconds
                - self.previous.sensor_timestamp_seconds,
                egomotion_translation_grid_xy=(0.0, 0.0),
                previous_estimate=previous_estimate
                if isinstance(previous_estimate, SafeSAGETemporalDerivativeResult)
                else None,
            )
        field = SafeSAGERuntimeField(
            packet.sensor_frame_id,
            packet.sensor_timestamp_seconds,
            self.anchor,
            self.anchor_yaw,
            len(packet.obstacle_xyz),
            len(packet.obstacle_xyz),
            tracking,
            SemanticLabelResult((SemanticTrackAssignment(semantic),)),
            pipeline,
            temporal,
            time.perf_counter() - started,
        )
        readiness = evaluate_perception_readiness(
            RestaurantScenario(
                "live_object",
                "potential",
                "live physical inputs",
                runtime_factor_evidence_mode(adapted),
                0,
            ),
            adapted,
        )
        self.tracker, self.previous, self.plume = (
            candidate_tracker,
            field,
            candidate_plume,
        )
        self.retained = (
            retained + [current_placements] if self.occupancy_retention_seconds else []
        )
        self.static_points = candidate_static
        self.evidence_memory = candidate_memory
        if accepted:
            self.last_visual_track_id = target.track_id
        if self.static_scope and self.static_oldest_time is None:
            self.static_oldest_time = packet.sensor_timestamp_seconds
        diagnostics = {
            "stage_wall_seconds": {
                "tracking_mask_association": association_done - started,
                "physical_estimators": physical_done - association_done,
                "factor_field_temporal": time.perf_counter() - physical_done,
            },
            "camera_frame_id": packet.camera_frame_id,
            "camera_timestamp": timestamp,
            "lidar_timestamp": packet.sensor_timestamp_seconds,
            "accepted_instance": accepted,
            "visual_processing_reason": None
            if accepted
            else (
                "no eligible projected geometric track"
                if target not in eligible
                else "mask proposal rejected"
                if not frame.detections
                else "camera-LiDAR association or exact-mask return selection rejected"
            ),
            "target_track_id": target.track_id,
            "target_point_count": target.point_count,
            "target_centroid_world_xy": target.centroid_world_xy,
            "target_velocity_world_xy": target.velocity_world_xy,
            "target_observation_count": target.observation_count,
            "target_matched_previous_track": target.matched_previous_track,
            "projected_target_return_count": int(
                np.count_nonzero(
                    tracking.point_track_ids[projection.source_indices]
                    == target.track_id
                )
            ),
            "temporal_object_evidence": retention_diagnostics,
            "geometry_evidence": geometry_diagnostics,
            "selected_return_count": int(selected.sum()),
            "unassessed_return_count": int((~selected).sum()),
            "sam_model_quality": sam_quality,
            "sam_prompt_basis": prompt_basis,
            "component_prompt_radius_m": self.component_prompt_radius_m,
            "sam_rejections": reasons,
            "sam_candidate_diagnostics": sam_candidates,
            "plume_registration": self.plume.last_diagnostics.get(target.track_id, {}),
            "fixed_lambda": policy.tangential_strength,
            "plume_coordinates": "calibrated_pinhole_140_degrees"
            if self.rectify_plume
            else "native_rgb",
            "scan_interval_scope": packet.scan_interval_scope,
            "scan_pose_sample_count": None
            if packet.scan_pose_samples_world is None
            else len(packet.scan_pose_samples_world),
            "field_raw_placement_count": len(assessed_points) + len(unassessed_points),
            "field_cell_representative_count": len(assessed_cells)
            + len(unassessed_cells),
            "retained_occupancy_epochs": [
                {"timestamp": item[0], "sensor_frame_id": item[1]} for item in retained
            ],
            "occupancy_retention_scope": "positive observed geometry retained as a conservative occupancy prior; no new sensor returns or negative free-space evidence"
            if retained
            else None,
            "static_occupancy_memory_prior_scope": self.static_scope,
            "static_oldest_source_time": self.static_oldest_time,
            "static_cached_track_ids": list(self.static_points),
            "unknown_factors": [
                name
                for name in ("potential", "kinetic", "contact", "thermal")
                if getattr(adapted, name).rejection_reasons
                or getattr(adapted, name).estimate.confidence <= 0
            ],
            "assumed_ground_scope": self.ground_scope,
            "reference_was_explicitly_validated": packet.clean_reference_validated,
        }
        return LiveObjectResult(
            field, attributes, adapted, policy, readiness, frame, fusion, diagnostics
        )

    def _assess_track(
        self,
        packet,
        tracking,
        projection,
        target,
        observable,
        candidate_plume,
        candidate_memory,
        *,
        image_is_set=False,
        candidate_persistent=None,
        shared_contact_frame=None,
    ):
        timestamp = packet.camera_timestamp_seconds
        frame = SemanticDetectionFrame(
            packet.camera_frame_id,
            timestamp,
            *packet.rgb.shape[:2],
            (),
            SAM_INSTANCE_SOURCE,
            0.0,
        )
        selected = np.zeros(len(packet.obstacle_xyz), bool)
        fusion = association = proxy = None
        reasons, sam_quality = [], None
        prompt_basis = "projected_track_median"
        sam_candidates = []
        if observable:
            if not image_is_set:
                self.segmenter.set_image(packet.rgb)
            proposal = self.segmenter.propose(
                tracking.point_track_ids, projection, track_id=target.track_id
            )
            sam_candidates.append(
                {
                    "prompt_basis": prompt_basis,
                    "prompt_xy": proposal.prompt_xy,
                    "candidates": getattr(proposal, "candidate_diagnostics", ()),
                }
            )
            if proposal.detection is None and proposal.rejection_reasons == (
                "no mask passes quality, boundary and LiDAR association gates",
            ):
                # A planar track can contain vertically separated components.
                # Retry one observed point in its nearest small 3D component,
                # retaining the full original association/quality gates and all
                # other occupancy. No class, authored mask/height or color.
                from robot.safety.obstacle_tracking import _cluster_point_indices

                owned = projection.source_indices[
                    tracking.point_track_ids[projection.source_indices]
                    == target.track_id
                ]
                groups = _cluster_point_indices(
                    packet.obstacle_xyz[owned],
                    cluster_radius=self.component_prompt_radius_m,
                    minimum_cluster_points=8,
                    backend="kdtree",
                )
                if groups:
                    group = min(
                        groups,
                        key=lambda indices: np.linalg.norm(
                            packet.obstacle_xyz[owned[indices]].mean(axis=0)
                        ),
                    )
                    component = packet.obstacle_xyz[owned[group]]
                    seed = int(
                        owned[
                            group[
                                np.argmin(
                                    np.sum(
                                        (component - np.median(component, axis=0)) ** 2,
                                        axis=1,
                                    )
                                )
                            ]
                        ]
                    )
                    proposal = self.segmenter.propose(
                        tracking.point_track_ids,
                        projection,
                        track_id=target.track_id,
                        prompt_source_index=seed,
                    )
                    prompt_basis = "nearest_observed_3d_component_retry"
                    sam_candidates.append(
                        {
                            "prompt_basis": prompt_basis,
                            "prompt_xy": proposal.prompt_xy,
                            "candidates": getattr(
                                proposal, "candidate_diagnostics", ()
                            ),
                        }
                    )
            reasons.extend(proposal.rejection_reasons)
            sam_quality = proposal.model_quality
            if proposal.detection is not None:
                frame = replace(
                    frame,
                    detections=(proposal.detection,),
                    inference_wall_seconds=proposal.inference_wall_seconds
                    + proposal.image_encoder_wall_seconds,
                )
                fusion = CameraLidarSemanticFusion(
                    CameraLidarFusionConfig(
                        minimum_mask_hits=4,
                        minimum_overlap_fraction=0.1,
                        mask_dilation_radius_pixels=6,
                        projected_median_box_margin_pixels=12,
                        require_projected_median_in_box=False,
                    )
                ).fuse(
                    tracking_result=tracking,
                    obstacle_points_source_xyz=packet.obstacle_xyz,
                    detection_frame=frame,
                    camera_from_source=packet.camera_from_scan_robot,
                    calibration=self.calibration,
                )
                association = next(
                    item
                    for item in fusion.associations
                    if item.track_id == target.track_id
                )
                if association.is_matched:
                    selected = masked_track_return_selection(
                        tracking.point_track_ids,
                        fusion.projection,
                        proposal.detection.mask,
                        track_id=target.track_id,
                    )
        accepted = (
            association is not None
            and association.is_matched
            and np.count_nonzero(selected) >= 4
        )
        association_done = time.perf_counter()
        base = PhysicalAttributes.unknown(
            track_id=target.track_id,
            timestamp_seconds=timestamp,
            semantic_class="unknown",
            semantic_source=SAM_INSTANCE_SOURCE
            if accepted
            else "geometric_track_without_visual_instance",
            semantic_confidence=frame.detections[0].confidence if accepted else 0.0,
        )
        motion = estimate_track_motion(
            target,
            timestamp_seconds=packet.sensor_timestamp_seconds,
            route_context=RouteMotionContext(
                packet.robot_position_world_xy,
                packet.waypoint_world_xy,
                0.0,
                packet.measured_robot_velocity_world_xy,
            ),
        )
        motion = replace(
            motion,
            **{
                name: replace(
                    getattr(motion, name),
                    evidence=(
                        *getattr(motion, name).evidence,
                        f"ego pose/velocity reference source={packet.pose_source}; status={packet.pose_status.value}",
                    ),
                )
                for name in (
                    "velocity_x_mps",
                    "velocity_y_mps",
                    "relative_speed_mps",
                    "route_crossing_likelihood",
                )
            },
        )
        attributes = replace(base, motion=motion)
        potential = ScalarEstimate.unavailable(
            unit="1",
            source="harmful_fall_opportunity_unavailable",
            timestamp_seconds=timestamp,
        )
        thermal_interaction = None
        geometry_diagnostics = {"accepted_instance": accepted}
        if accepted:
            geometry = estimate_track_geometry(
                packet.obstacle_xyz,
                tracking_result=replace(
                    tracking, point_track_ids=np.where(selected, target.track_id, -1)
                ),
                track_id=target.track_id,
                floor_height_robot_m=packet.floor_height_robot_m,
                association=association,
            )
            mass = self.mass_estimator.estimate_instance(
                packet.rgb,
                frame,
                association,
                geometry,
                track_id=target.track_id,
                timestamp_seconds=timestamp,
            )
            object_xyz = packet.obstacle_xyz[selected]
            surfaces = estimate_lidar_fall_surfaces(
                packet.scene_xyz,
                object_xyz,
                timestamp_seconds=packet.sensor_timestamp_seconds,
            )
            geometry_diagnostics["strict_fall_surfaces"] = asdict(surfaces)
            geometry = replace(
                geometry,
                fall_height_m=estimate_fall_height(
                    surfaces.evidence, timestamp_seconds=timestamp
                ),
            )
            if self.ground_scope is not None:
                sphere = estimate_spherical_geometry(
                    object_xyz, timestamp_seconds=packet.sensor_timestamp_seconds
                )
                ground = estimate_spherical_ground_support(
                    packet.scene_xyz,
                    sphere,
                    timestamp_seconds=timestamp,
                    scene_timestamp_seconds=packet.sensor_timestamp_seconds,
                    ground_continuity_prior=ScalarEstimate(
                        1.0,
                        1.0,
                        1.0,
                        0.5,
                        "1",
                        "configured_local_ground_continuity_prior",
                        packet.sensor_timestamp_seconds,
                        EstimateStatus.PRIOR,
                        (self.ground_scope,),
                    ),
                    prior_scope=self.ground_scope,
                )
                geometry_diagnostics["sphere_fit"] = asdict(sphere)
                geometry_diagnostics["ground_support"] = asdict(ground)
                if ground.fall_height_m.is_available:
                    geometry = replace(geometry, fall_height_m=ground.fall_height_m)
                    potential = ground.potential_interaction_likelihood
            links = SpecialistAssociationFrame(
                frame.source_frame_id,
                timestamp,
                frame.image_height,
                frame.image_width,
                fusion.associations,
            )
            parts = {}
            if self.contact_detector is not None:
                part_frame = shared_contact_frame or self.contact_detector.detect(
                    packet.rgb,
                    source_frame_id=packet.camera_frame_id,
                    source_timestamp_seconds=timestamp,
                )
                shared_contact_frame = part_frame
                part_links = associate_contact_parts_to_objects(frame, part_frame)
                parts = build_track_visual_contact_proxies(
                    links, part_links.segmentations_by_object_detection_index
                )
            proxy = build_track_shape_contact_proxies(
                frame, links, part_proxies=parts
            ).get(target.track_id)
            attributes = replace(attributes, geometry=geometry, mass=mass)
            if candidate_persistent is None or not candidate_persistent.has_thermal(
                target.track_id, timestamp
            ):
                observation = TrackedRgbObjectFrame(
                    packet.rgb,
                    frame,
                    association,
                    packet.sensor_timestamp_seconds,
                )
                if self.rectify_plume:
                    from robot.perception.fisheye_plume_view import (
                        rectify_plume_observation,
                    )

                    observation = rectify_plume_observation(
                        observation, self.calibration
                    )
                if packet.clean_reference_validated:
                    candidate_plume.set_reference(
                        observation, clean_reference_validated=True
                    )
                heat = candidate_plume.update(
                    observation, negative_validation_scope=self.negative_scope
                )
                attributes = replace(attributes, thermal=heat.thermal)
                thermal_interaction = estimate_thermal_route_interaction(
                    heat.thermal,
                    motion.route_crossing_likelihood,
                    timestamp_seconds=timestamp,
                )
        physical_done = time.perf_counter()
        retention_diagnostics = {"enabled": candidate_memory is not None}
        if candidate_memory is not None:
            retained_mass, proxy, detail = candidate_memory.resolve(
                track=target,
                tracked_obstacles=tracking.tracked_obstacles,
                sensor_timestamp_seconds=packet.sensor_timestamp_seconds,
                timestamp_seconds=timestamp,
                camera_frame_id=packet.camera_frame_id,
                ego_position_world_xy=packet.robot_position_world_xy,
                fresh_mass=attributes.mass.mass_kg,
                fresh_contact=proxy,
            )
            attributes = replace(
                attributes, mass=replace(attributes.mass, mass_kg=retained_mass)
            )
            retention_diagnostics.update(detail)
        persistence_diagnostics = None
        if candidate_persistent is not None:
            attributes, potential, proxy, persistence_diagnostics = (
                candidate_persistent.resolve(
                    track=target,
                    tracks=tracking.tracked_obstacles,
                    attributes=attributes,
                    potential=potential,
                    contact=proxy,
                    timestamp=timestamp,
                    camera_frame_id=packet.camera_frame_id,
                )
            )
            # Route interaction is current; retained heat carries its old epoch.
            route = motion.route_crossing_likelihood
            thermal_interaction = replace(
                route,
                evidence=(
                    *route.evidence,
                    "current same-track route interaction for session-persistent plume proxy; no spill/temperature claim",
                ),
            )
        adapted = adapt_physical_factors(
            attributes,
            timestamp_seconds=timestamp,
            potential_interaction_likelihood=potential,
            visual_contact_proxy=proxy,
            visual_contact_basis=SHAPE_CONTACT_BASIS
            if proxy is not None and proxy.source == SHAPE_CONTACT_SOURCE
            else "visual_exposed_edge_proxy",
            contact_interaction_likelihood=motion.route_crossing_likelihood
            if proxy is not None
            else None,
            thermal_interaction_likelihood=thermal_interaction,
            thermal_basis="visual_object_plume_proxy"
            if attributes.thermal.heat_source_likelihood.source
            in {"object_visible_plume_proxy", "registered_rgb_object_plume_proxy"}
            else "physical",
            config=PhysicalFactorAdapterConfig(
                allow_visual_contact_proxy=True,
                allow_visual_thermal_proxy=True,
                maximum_persistent_property_age_seconds=candidate_persistent.config.maximum_property_age_seconds
                if candidate_persistent is not None
                else None,
                maximum_retained_mass_age_seconds=candidate_memory.config.maximum_mass_age_seconds
                if candidate_memory is not None
                else None,
                maximum_retained_contact_age_seconds=candidate_memory.config.maximum_contact_age_seconds
                if candidate_memory is not None
                else None,
            ),
        )
        return {
            "attributes": attributes,
            "adapted": adapted,
            "frame": frame,
            "fusion": fusion,
            "selected": selected,
            "accepted": accepted,
            "base": base,
            "association_done": association_done,
            "physical_done": physical_done,
            "retention_diagnostics": retention_diagnostics,
            "geometry_diagnostics": geometry_diagnostics,
            "sam_quality": sam_quality,
            "prompt_basis": prompt_basis,
            "sam_candidates": sam_candidates,
            "reasons": reasons,
            "persistence": persistence_diagnostics,
            "contact_frame": shared_contact_frame,
        }
