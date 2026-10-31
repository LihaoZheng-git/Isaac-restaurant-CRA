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

from robot.evaluation.cra_experiments import CRABaseline, RestaurantScenario
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
from robot.perception.object_height import ObjectHeightConfig, estimate_object_height
from robot.perception.moving_camera_plume import (
    MovingCameraObjectPlumeEstimator,
    MovingCameraPlumeConfig,
    TrackedRgbObjectFrame,
)
from robot.perception.physical_attributes import (
    CategoricalEstimate,
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
    nominal_route_direction_world_xy: tuple | None = None
    nominal_route_terminal_goal_world_xy: tuple | None = None

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
        if self.nominal_route_direction_world_xy is not None:
            from robot.safety.encounter_motion import validated_route_direction
            direction = validated_route_direction(self.nominal_route_direction_world_xy)
            object.__setattr__(self, 'nominal_route_direction_world_xy', tuple(float(x) for x in direction))
        if self.nominal_route_terminal_goal_world_xy is not None:
            terminal = np.asarray(self.nominal_route_terminal_goal_world_xy, float)
            if terminal.shape != (2,) or not np.isfinite(terminal).all():
                raise ValueError("finite nominal terminal-route context required")
            object.__setattr__(self, 'nominal_route_terminal_goal_world_xy', tuple(float(v) for v in terminal))
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
    source_occupancy_grid: object | None = None
    furniture_memory_snapshot: object | None = None
    tracking_uncertainty_snapshot: object | None = None


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
        fall_surface_config=None,
        fall_interaction_prior_scope=None,
        cluster_in_3d=False,
        cluster_radius_m=None,
        camera_visibility_radius_pixels=0,
        mask_foreground_depth_gap_m=None,
        plume_view_fov_degrees=140.0,
        person_detector=None,
        human_mass_prior_config=None,
        object_height_config=None,
        carried_object_config=None,
        pose_detector=None,
        human_body_contact_config=None,
        experiment_baseline=None,
        motion_passing_config=None,
        human_body_envelope_config=None,
        human_track_recovery_config=None,
        human_tracking_uncertainty=False,
        human_fragment_coverage_config=None,
        human_observed_occupancy_only=False,
        human_boundary_flux=None,
        human_visual_fragment_association=False,
        human_fragment_mask_boundary_pixels=0,
        human_visual_part_field_policy=False,
        human_prior_geometry_associations=False,
        human_fresh_person_policy=False,
        zero_tangential_bias_ablation=False,
        encounter_motion_config=None,
        predictive_human_geometry_config=None,
        parallel_preparation_config=None,
        human_mass_body_identity_retention=False,
        observed_motion_reference=False,
    ):
        self.segmenter, self.mass_estimator, self.contact_detector = (
            segmenter,
            mass_estimator,
            contact_detector,
        )
        self.calibration = calibration
        from robot.perception.human_mass_prior import HumanMassPriorConfig

        if (person_detector is None) != (human_mass_prior_config is None):
            raise ValueError(
                "human mass prior and visual person detector must be supplied together"
            )
        if human_mass_prior_config is not None and not isinstance(
            human_mass_prior_config, HumanMassPriorConfig
        ):
            raise TypeError("invalid human mass prior configuration")
        self.person_detector = person_detector
        from robot.evaluation.restaurant_baselines import validate_baseline
        self.experiment_baseline = validate_baseline(experiment_baseline)
        from robot.safety.motion_aware_passing import MotionAwarePassing, MotionAwarePassingConfig
        if motion_passing_config is not None and (
            not isinstance(motion_passing_config, MotionAwarePassingConfig)
            or persistent_property_config is None or carried_object_config is not None
        ):
            raise ValueError("motion-aware passing requires standalone persistent object-map mode")
        if (motion_passing_config is not None and motion_passing_config.require_visual_person_selection
            and person_detector is None):
            raise ValueError("observed-human passing requires the audited visual person detector")
        self.motion_passing = MotionAwarePassing(motion_passing_config) if motion_passing_config is not None else None
        if self.experiment_baseline is not CRABaseline.CONTINUOUS_CRA_SAFE_SAGE and persistent_property_config is None:
            raise ValueError("comparison baseline requires persistent object-map mode")
        self.human_mass_prior_config = human_mass_prior_config
        from robot.perception.human_body_contact import HumanBodyContactPriorConfig
        if human_body_contact_config is not None and (
            not isinstance(human_body_contact_config, HumanBodyContactPriorConfig)
            or person_detector is None or persistent_property_config is None
        ):
            raise ValueError("human body contact prior requires fresh person association and object-map mode")
        self.human_body_contact_config = human_body_contact_config
        from robot.perception.human_body_envelope import HumanBodyEnvelope, HumanBodyEnvelopeConfig
        if human_body_envelope_config is not None and (
            not isinstance(human_body_envelope_config, HumanBodyEnvelopeConfig)
            or person_detector is None or persistent_property_config is None
            or carried_object_config is not None or static_occupancy_memory_prior_scope is not None
        ):
            raise ValueError("human envelope requires standalone live person/object-map mode, without static memory")
        if human_track_recovery_config is not None and human_body_envelope_config is None:
            raise ValueError('track recovery requires explicit human body geometry')
        self.human_body_envelope = HumanBodyEnvelope(human_body_envelope_config,
            recovery_config=human_track_recovery_config,
            planning_uncertainty=human_tracking_uncertainty) if human_body_envelope_config is not None else None
        if human_tracking_uncertainty and self.human_body_envelope is None:
            raise ValueError("tracking uncertainty requires current human geometry initialization")
        from robot.perception.human_mass_identity_retention import HumanMassIdentityRetention
        if (not isinstance(human_mass_body_identity_retention, bool)
            or human_mass_body_identity_retention and (self.human_body_envelope is None
                or human_mass_prior_config is None or persistent_property_config is None
                or persistent_property_config.maximum_property_age_seconds > 3.)):
            raise ValueError('adult mass identity retention requires explicit body identity and at most three-second property scope')
        self.human_mass_identity_retention = HumanMassIdentityRetention() if human_mass_body_identity_retention else None
        from robot.evaluation.restaurant_baselines import validate_human_boundary_flux
        self.human_boundary_flux = validate_human_boundary_flux(human_boundary_flux)
        if not isinstance(human_observed_occupancy_only, bool):
            raise ValueError("human observed-occupancy mode must be boolean")
        if (human_observed_occupancy_only or self.human_boundary_flux is not None) and (
            self.human_body_envelope is None
            or self.experiment_baseline is not CRABaseline.CONTINUOUS_CRA_SAFE_SAGE
        ):
            raise ValueError("human flux/observed-only ablation requires verified human identity and continuous CRA")
        if human_observed_occupancy_only and human_fragment_coverage_config is not None:
            raise ValueError("observed-only geometry cannot cover fragments with an unoccupied prior disk")
        self.human_observed_occupancy_only = human_observed_occupancy_only
        if human_tracking_uncertainty and (human_observed_occupancy_only or predictive_human_geometry_config is not None):
            raise ValueError("tracking uncertainty requires occupied body geometry without the separate predictive ablation")
        from robot.safety.predictive_human_geometry import HumanGeometryForecastConfig, PredictiveHumanGeometryMemory
        if predictive_human_geometry_config is not None and (
            not isinstance(predictive_human_geometry_config, HumanGeometryForecastConfig)
            or self.human_body_envelope is None or human_observed_occupancy_only):
            raise ValueError("prediction requires an explicit active verified human body-envelope prior")
        self.predictive_human_geometry = (None if predictive_human_geometry_config is None
            else PredictiveHumanGeometryMemory(predictive_human_geometry_config))
        if (not isinstance(observed_motion_reference, bool) or observed_motion_reference and (
            self.human_body_envelope is None or human_observed_occupancy_only
            or predictive_human_geometry_config is not None or parallel_preparation_config is not None
            or motion_passing_config is None or motion_passing_config.encounter_geometry != 'oncoming_parallel')):
            raise ValueError('observed-motion advisory requires current oncoming human/body scope without preparation or hard prediction')
        self.observed_motion_reference = observed_motion_reference
        from robot.safety.parallel_preparation import ParallelPreparation, ParallelPreparationConfig
        if parallel_preparation_config is not None and (
            not isinstance(parallel_preparation_config, ParallelPreparationConfig)
            or self.predictive_human_geometry is None or motion_passing_config is None
            or motion_passing_config.encounter_geometry != 'oncoming_parallel'):
            raise ValueError('early preparation requires explicit parallel passing and predictive human geometry')
        self.parallel_preparation = None if parallel_preparation_config is None else ParallelPreparation(parallel_preparation_config)
        if not isinstance(human_prior_geometry_associations, bool) or (
            human_prior_geometry_associations and (human_observed_occupancy_only
                or self.human_body_envelope is None or person_detector is None
                or carried_object_config is not None
                or not (human_visual_fragment_association or human_fresh_person_policy))):
            raise ValueError("prior-geometry associations require explicit body-prior geometry and current bare-human helpers")
        self.human_prior_geometry_associations = human_prior_geometry_associations
        association_scope = human_observed_occupancy_only or human_prior_geometry_associations
        if not isinstance(human_visual_fragment_association, bool) or (
            human_visual_fragment_association and (not association_scope
            or person_detector is None or carried_object_config is not None)
        ):
            raise ValueError("visual part association requires observed-only bare-human mode")
        self.human_visual_fragment_association = human_visual_fragment_association
        if (isinstance(human_fragment_mask_boundary_pixels, bool)
            or not isinstance(human_fragment_mask_boundary_pixels, int)
            or human_fragment_mask_boundary_pixels not in (0, 1)
            or human_fragment_mask_boundary_pixels and not human_visual_fragment_association):
            raise ValueError("one-pixel body-part boundary prior requires explicit visual part association")
        self.human_fragment_mask_boundary_pixels = human_fragment_mask_boundary_pixels
        if not isinstance(human_visual_part_field_policy, bool) or (
            human_visual_part_field_policy and (not human_visual_fragment_association
                or self.human_body_envelope is None or self.motion_passing is None)):
            raise ValueError("human-part field policy requires current visual parts, verified human identity and passing mode")
        self.human_visual_part_field_policy = human_visual_part_field_policy
        if not isinstance(human_fresh_person_policy, bool) or (
            human_fresh_person_policy and (not association_scope
                or person_detector is None or self.motion_passing is None
                or not self.motion_passing.config.require_visual_person_selection)
        ):
            raise ValueError("fresh person policy requires observed-only current-identity/person-selected passing mode")
        self.human_fresh_person_policy = human_fresh_person_policy
        if not isinstance(zero_tangential_bias_ablation, bool) or (
            zero_tangential_bias_ablation and (self.motion_passing is None
                or self.experiment_baseline is not CRABaseline.CONTINUOUS_CRA_SAFE_SAGE)):
            raise ValueError("zero tangential bias requires the explicit continuous-CRA crossing ablation")
        self.zero_tangential_bias_ablation = zero_tangential_bias_ablation
        from robot.safety.encounter_motion import EncounterMotionClassifier, EncounterMotionConfig
        if encounter_motion_config is not None and (
            not isinstance(encounter_motion_config, EncounterMotionConfig)
            or self.motion_passing is None or not self.motion_passing.config.require_visual_person_selection
            or self.human_body_envelope is None or zero_tangential_bias_ablation
            or self.experiment_baseline is not CRABaseline.CONTINUOUS_CRA_SAFE_SAGE):
            raise ValueError('encounter guidance requires current verified bare-human passing and cannot combine with global zero bias')
        self.encounter_motion = EncounterMotionClassifier(encounter_motion_config) if encounter_motion_config is not None else None
        from robot.perception.human_fragment_coverage import HumanFragmentCoverage
        if human_fragment_coverage_config is not None and (
            self.human_body_envelope is None or self.experiment_baseline is not CRABaseline.CONTINUOUS_CRA_SAFE_SAGE):
            raise ValueError("fragment coverage requires current human envelopes and continuous CRA mode")
        self.human_fragment_coverage = HumanFragmentCoverage(human_fragment_coverage_config) if human_fragment_coverage_config is not None else None
        if (motion_passing_config is not None
            and motion_passing_config.maximum_identity_prior_selection_age_seconds > 0
            and self.human_body_envelope is None):
            raise ValueError("identity-prior passing requires the current human body-envelope adapter")
        if (human_observed_occupancy_only and motion_passing_config is not None
                and motion_passing_config.maximum_identity_prior_selection_age_seconds > 0):
            raise ValueError("observed-only ablation retains fresh-only passing selection")
        self._person_packet = self._person_frame = None
        from robot.perception.lidar_fall_surfaces import LidarFallSurfaceConfig

        if fall_surface_config is not None and not isinstance(
            fall_surface_config, LidarFallSurfaceConfig
        ):
            raise TypeError("invalid fall surface configuration")
        if fall_interaction_prior_scope is not None and (
            not isinstance(fall_interaction_prior_scope, str)
            or not fall_interaction_prior_scope.strip()
        ):
            raise ValueError("fall interaction prior requires an explicit scope")
        self.fall_surface_config = fall_surface_config
        self.fall_interaction_prior_scope = fall_interaction_prior_scope
        if object_height_config is not None and not isinstance(object_height_config, ObjectHeightConfig):
            raise TypeError("object_height_config must be ObjectHeightConfig")
        if object_height_config is not None and (ground_prior_scope is not None or fall_surface_config is not None):
            raise ValueError("general object height and legacy spherical support modes must be selected separately")
        self.object_height_config = object_height_config
        from robot.perception.carried_object import CarriedObjectConfig
        if (carried_object_config is None) != (pose_detector is None):
            raise ValueError("carried-object mode requires both config and RGB pose detector")
        if carried_object_config is not None and (
            not isinstance(carried_object_config,CarriedObjectConfig)
            or person_detector is None or persistent_property_config is None
            or static_occupancy_memory_prior_scope is not None):
            raise ValueError("carried-object mode requires fresh person/multi-object inputs and no static memory")
        self.carried_object_config, self.pose_detector = carried_object_config, pose_detector
        if not isinstance(cluster_in_3d, bool):
            raise TypeError("3D clustering opt-in must be boolean")
        CameraLidarFusionConfig(
            visibility_radius_pixels=camera_visibility_radius_pixels,
            mask_foreground_depth_gap_m=mask_foreground_depth_gap_m,
        )
        self.cluster_in_3d = cluster_in_3d
        self.camera_visibility_radius_pixels = camera_visibility_radius_pixels
        self.mask_foreground_depth_gap_m = mask_foreground_depth_gap_m
        if (
            not math.isfinite(plume_view_fov_degrees)
            or not 30 <= plume_view_fov_degrees < 170
        ):
            raise ValueError("plume view FOV must lie within [30, 170) degrees")
        self.plume_view_fov_degrees = float(plume_view_fov_degrees)
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
                cluster_radius=(0.12 if cluster_in_3d else 0.20) if cluster_radius_m is None else cluster_radius_m,
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

    def visual_person_frame(self, packet):
        """Optional same-packet visual cue, shared across current track attempts."""
        if self.person_detector is None:
            return None
        if self._person_packet is not packet:
            self._person_frame = self.person_detector.detect(
                packet.rgb,
                source_frame_id=packet.camera_frame_id,
                source_timestamp_seconds=packet.camera_timestamp_seconds,
            )
            self._person_packet = packet
        return self._person_frame

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
            clustering_points_robot_xyz=packet.obstacle_xyz
            if self.cluster_in_3d
            else None,
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
        proposal_override=None,
        suppress_catalog_mass_for_human_identity=False,
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
            proposal = proposal_override or self.segmenter.propose(
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
            if (
                proposal.detection is None
                and proposal.rejection_reasons
                == ("no mask passes quality, boundary and LiDAR association gates",)
                and getattr(
                    getattr(self.segmenter, "config", None),
                    "observed_context_box_retry",
                    False,
                )
            ):
                proposal = self.segmenter.propose(
                    tracking.point_track_ids,
                    projection,
                    track_id=target.track_id,
                    use_observed_context_box=True,
                )
                prompt_basis = "observed_track_context_box_retry"
                sam_candidates.append(
                    {
                        "prompt_basis": prompt_basis,
                        "prompt_xy": proposal.prompt_xy,
                        "observed_context_box_xyxy": proposal.observed_context_box_xyxy,
                        "candidates": proposal.candidate_diagnostics,
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
                        visibility_radius_pixels=self.camera_visibility_radius_pixels,
                        mask_foreground_depth_gap_m=self.mask_foreground_depth_gap_m,
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
        height_evidence = None
        human_body_contact_prior = part_contact_diagnostic = None
        if self.object_height_config is not None and accepted:
            height_evidence = estimate_object_height(
                packet.scene_xyz,
                packet.obstacle_xyz[selected],
                timestamp_seconds=timestamp,
                scene_timestamp_seconds=packet.sensor_timestamp_seconds,
                floor_reference_m=packet.floor_height_robot_m,
                reference_scope=(f"gravity-aligned packet; pose={packet.pose_source}; status={packet.pose_status.value}; "
                    "floor-band selection uses declared packet ground reference; fitted height uses LiDAR returns"),
                config=self.object_height_config,
            )
            geometry_diagnostics["object_height"] = asdict(height_evidence)
            geometry_diagnostics["object_height_configuration"] = asdict(self.object_height_config)
            geometry_diagnostics["height_ownership_basis"] = "fresh_accepted_instance"
        elif self.object_height_config is not None:
            geometry_diagnostics["height_ownership_basis"] = "unresolved_lidar_cluster; fresh instance required for object height"
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
            from robot.perception.physical_attributes import MassAttributes
            person_frame = self.visual_person_frame(packet) if self.person_detector is not None else None
            domain_guard = None
            if person_frame is not None:
                from robot.perception.human_mass_prior import catalog_person_overlap_guard
                domain_guard = catalog_person_overlap_guard(frame, association, person_frame,
                    track_id=target.track_id, timestamp_seconds=timestamp,
                    config=self.human_mass_prior_config)
                geometry_diagnostics["catalog_person_overlap_guard"] = domain_guard
            block_catalog = suppress_catalog_mass_for_human_identity or bool(domain_guard and domain_guard["blocked"])
            mass = MassAttributes(ScalarEstimate.unavailable(unit="kg",
                source="catalog_human_domain_guard", timestamp_seconds=timestamp,
                evidence=("positive person domain cue blocks catalog-object inference; no adult mass/identity assigned or original expiry renewed",))) if block_catalog else self.mass_estimator.estimate_instance(
                packet.rgb,
                frame,
                association,
                geometry,
                track_id=target.track_id,
                timestamp_seconds=timestamp,
            )
            geometry_diagnostics["catalog_human_domain_guard"] = block_catalog
            if self.person_detector is not None:
                from robot.perception.human_mass_prior import (
                    estimate_visual_human_mass_prior,
                )
                from robot.perception.physical_attributes import MassAttributes

                prior = estimate_visual_human_mass_prior(
                    frame,
                    association,
                    person_frame,
                    track_id=target.track_id,
                    timestamp_seconds=timestamp,
                    config=self.human_mass_prior_config,
                )
                geometry_diagnostics["human_mass_prior"] = asdict(prior)
                if prior.is_available:
                    geometry_diagnostics["catalog_mass_before_human_prior"] = asdict(
                        mass.mass_kg
                    )
                    mass = MassAttributes(prior)
                if self.human_body_contact_config is not None:
                    from robot.perception.human_body_contact import select_visual_human_body_contact_prior
                    human_body_contact_prior = select_visual_human_body_contact_prior(
                        frame, association, person_frame,
                        track_id=target.track_id, timestamp_seconds=timestamp,
                        association_config=self.human_mass_prior_config,
                        config=self.human_body_contact_config,
                    )
                    geometry_diagnostics["human_body_contact_prior"] = asdict(human_body_contact_prior)
            object_xyz = packet.obstacle_xyz[selected]
            sphere = None
            if height_evidence is None:
                surfaces = estimate_lidar_fall_surfaces(
                    packet.scene_xyz, object_xyz,
                    timestamp_seconds=packet.sensor_timestamp_seconds,
                    config=self.fall_surface_config,
                )
                geometry_diagnostics["strict_fall_surfaces"] = asdict(surfaces)
                geometry_diagnostics["fall_surface_configuration"] = (
                    asdict(self.fall_surface_config) if self.fall_surface_config else None
                )
                support_consistent = None
                if self.ground_scope is not None or self.fall_surface_config is not None:
                    sphere = estimate_spherical_geometry(
                        object_xyz, timestamp_seconds=packet.sensor_timestamp_seconds
                    )
                if self.fall_surface_config is not None and surfaces.is_available:
                    support = surfaces.evidence.support_height_m
                    support_consistent = bool(
                        sphere.radius_m.is_available
                        and sphere.center_xyz is not None
                        and support.lower_bound
                        <= sphere.center_xyz[2] - sphere.radius_m.lower_bound
                        and support.upper_bound
                        >= sphere.center_xyz[2] - sphere.radius_m.upper_bound
                    )
                geometry_diagnostics["support_bottom_consistency"] = support_consistent
                geometry = replace(
                    geometry,
                    fall_height_m=estimate_fall_height(
                        surfaces.evidence if support_consistent is not False else None,
                        timestamp_seconds=timestamp,
                    ),
                )
            if height_evidence is not None:
                # General adjacent planar support is independent of spherical
                # fitting. Only the explicit opt-in can use this path; legacy
                # sphere bottom and ground-continuity gates remain unchanged.
                if height_evidence.visible_center_height_m.is_available:
                    geometry = replace(geometry, center_height_m=height_evidence.visible_center_height_m)
                geometry = replace(geometry, fall_height_m=height_evidence.fall_height_m)
                if height_evidence.support_state in {"ground_supported", "elevated_supported"}:
                    attributes = replace(attributes, stability=replace(attributes.stability,
                        support_state=CategoricalEstimate(height_evidence.support_state,
                            height_evidence.support_height_above_ground_m.confidence,
                            "lidar_adjacent_support_state", packet.sensor_timestamp_seconds,
                            EstimateStatus.INFERRED,
                            height_evidence.support_height_above_ground_m.evidence)))
            if self.ground_scope is not None:
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
                    if not geometry.fall_height_m.is_available:
                        geometry = replace(geometry, fall_height_m=ground.fall_height_m)
                    else:
                        geometry_diagnostics[
                            "strict_vs_ground_drop_bounds_disagree"
                        ] = bool(
                            geometry.fall_height_m.lower_bound
                            > ground.fall_height_m.upper_bound
                            or ground.fall_height_m.lower_bound
                            > geometry.fall_height_m.upper_bound
                        )
                    # A valid ground check supplies its separate worst-case
                    # interaction prior even when the strict drop is retained.
                    potential = ground.potential_interaction_likelihood
            if (
                geometry.fall_height_m.is_available
                and not potential.is_available
                and self.fall_interaction_prior_scope is not None
            ):
                potential = ScalarEstimate(
                    1.0,
                    1.0,
                    1.0,
                    0.5,
                    "1",
                    "configured_worst_case_harmful_fall_interaction_prior",
                    packet.sensor_timestamp_seconds,
                    EstimateStatus.PRIOR,
                    (self.fall_interaction_prior_scope,),
                )
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
            if (human_body_contact_prior is not None and human_body_contact_prior.is_available
                    and target.track_id in parts and parts[target.track_id].is_available):
                # Retain an accepted part cue separately: the combined outline
                # maximum is inappropriate for a human body, not for a real edge.
                part_contact_diagnostic = adapt_physical_factors(
                    attributes, timestamp_seconds=timestamp,
                    visual_contact_proxy=parts[target.track_id],
                    visual_contact_basis="visual_exposed_edge_proxy",
                    contact_interaction_likelihood=motion.route_crossing_likelihood,
                    config=PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True),
                ).contact
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
                        observation,
                        self.calibration,
                        horizontal_fov_degrees=self.plume_view_fov_degrees,
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
            "height_evidence": height_evidence if accepted else None,
            "sam_quality": sam_quality,
            "prompt_basis": prompt_basis,
            "sam_candidates": sam_candidates,
            "reasons": reasons,
            "persistence": persistence_diagnostics,
            "contact_frame": shared_contact_frame,
            "human_body_contact_prior": human_body_contact_prior,
            "human_part_contact": part_contact_diagnostic,
        }
