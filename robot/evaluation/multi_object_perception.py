"""Opt-in all-track CRA mapping using shared live perception components.

All current clustered occupancy receives numeric policies. Exact accepted mask
ownership is preserved; unclustered and unmatched support/environment returns
receive their own labeled factor priors, not a nearby object's low assessment.
"""

import copy
from dataclasses import dataclass, replace
import math
import time

import numpy as np

from robot.evaluation.cra_experiments import RestaurantScenario
from robot.evaluation.perception_readiness import (
    evaluate_perception_readiness,
    runtime_factor_evidence_mode,
)
from robot.perception.camera_lidar_fusion import (
    project_fisheye_polynomial_points,
    depth_visible_projection,
)
from robot.perception.physical_attributes import PhysicalAttributes
from robot.risk.object_risk_policy import evaluate_object_risk
from robot.risk.physical_factor_adapter import adapt_physical_factors
from robot.safety.local_occupancy_grid import build_local_occupancy_grid
from robot.safety.safe_sage_field_pipeline import build_safe_sage_field_pipeline
from robot.safety.safe_sage_runtime import SafeSAGERuntimeField
from robot.safety.safe_sage_temporal_derivative import SafeSAGETemporalDerivativeResult
from robot.safety.semantic_label_provider import (
    SemanticLabelResult,
    SemanticTrackAssignment,
)
from robot.safety.semantic_obstacle import SemanticObstacle


@dataclass(frozen=True)
class ObjectAssessment:
    attributes: object
    adapted: object
    policy: object
    readiness: object
    diagnostics: dict


def fresh_person_selection_evidence(objects, camera_timestamp, camera_frame_id):
    """Reuse the same-frame accepted person/SAM cue, never retained mass/class."""
    from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE

    eligible = {}
    for item in objects:
        d = item.diagnostics
        cue = d.get("geometry_evidence", {}).get("human_mass_prior", {})
        if (d.get("accepted_instance") and cue.get("source") == HUMAN_MASS_PRIOR_SOURCE
            and cue.get("status") == "prior" and cue.get("value") is not None
            and abs(cue.get("timestamp_seconds", -math.inf) - camera_timestamp) <= 1e-9):
            eligible[item.attributes.track_id] = {
                "basis": "fresh_same_frame_person_SAM_association; class cue only, not measured mass",
                "camera_frame_id": camera_frame_id,
                "camera_timestamp_seconds": camera_timestamp,
            }
    return eligible


def assess_all_objects(worker, packet):
    from robot.evaluation.live_object_perception import LiveObjectResult

    started = time.perf_counter()
    if worker.previous is not None and (
        packet.sensor_frame_id <= worker.previous.sensor_frame_id
        or packet.sensor_timestamp_seconds <= worker.previous.sensor_timestamp_seconds
    ):
        raise ValueError("live sensor epochs must increase")
    tracker = copy.deepcopy(worker.tracker)
    plume = copy.deepcopy(worker.plume)
    memory = copy.deepcopy(worker.persistent_properties)
    ownership = copy.deepcopy(getattr(worker, "object_ownership_bounds", {}))
    envelope_memory = copy.deepcopy(getattr(worker, "human_body_envelope", None))
    fragment_memory = copy.deepcopy(getattr(worker, "human_fragment_coverage", None))
    tracking = tracker.update(
        packet.obstacle_xyz[:, :2],
        robot_position_world_xy=np.array(packet.robot_position_world_xy),
        robot_yaw=packet.robot_yaw,
        timestamp_seconds=packet.sensor_timestamp_seconds,
        clustering_points_robot_xyz=packet.obstacle_xyz
        if worker.cluster_in_3d
        else None,
    )
    tracks = tracking.tracked_obstacles
    if not tracks:
        raise RuntimeError("no geometric track is available")
    projection = project_fisheye_polynomial_points(
        packet.obstacle_xyz,
        camera_from_source=packet.camera_from_scan_robot,
        calibration=worker.calibration,
    )
    projected_count = len(projection.source_indices)
    if worker.camera_visibility_radius_pixels:
        projection = depth_visible_projection(
            projection, radius_pixels=worker.camera_visibility_radius_pixels
        )
    visible = {
        track.track_id
        for track in tracks
        if np.count_nonzero(
            tracking.point_track_ids[projection.source_indices] == track.track_id
        )
        >= 8
    }
    attempted = {
        track_id
        for track_id in visible
        if not memory.frozen
        or packet.camera_timestamp_seconds
        - memory.last_visual_attempt.get(track_id, -math.inf)
        >= memory.config.minimum_visual_reassessment_interval_seconds
    }
    eligible_attempts = set(attempted)
    if memory.frozen:
        # Fair oldest-attempt scheduling bounds expensive vision while every
        # track still gets fresh motion and a numeric policy for this scan.
        attempted = set(
            sorted(
                attempted,
                key=lambda key: (memory.last_visual_attempt.get(key, -math.inf), key),
            )[: memory.config.maximum_visual_tracks_per_update]
        )
        # The bounded person opt-in reserves one visual slot from a fresh RGB
        # cue, leaving another for fair coverage. This is attention only; SAM,
        # foreign-return and final mask-ownership gates remain unchanged.
        person_frame = worker.visual_person_frame(packet)
        if (
            person_frame is not None
            and memory.config.maximum_visual_tracks_per_update >= 2
        ):
            person_hits = np.zeros(len(projection.source_indices), dtype=bool)
            for person in person_frame.detections:
                if person.semantic_class == "person" and person.confidence >= 0.5:
                    person_hits |= person.mask[
                        projection.pixel_indices_xy[:, 1],
                        projection.pixel_indices_xy[:, 0],
                    ]
            ids = tracking.point_track_ids[projection.source_indices]
            priority = [
                key
                for key in eligible_attempts
                if np.count_nonzero(person_hits & (ids == key)) >= 8
                and np.mean(person_hits[ids == key]) >= 0.5
            ]
            if priority:
                chosen = min(
                    priority,
                    key=lambda key: (
                        memory.last_visual_attempt.get(key, -math.inf),
                        key,
                    ),
                )
                fair = sorted(
                    eligible_attempts - {chosen},
                    key=lambda key: (
                        memory.last_visual_attempt.get(key, -math.inf),
                        key,
                    ),
                )
                attempted = {
                    chosen,
                    *fair[: memory.config.maximum_visual_tracks_per_update - 1],
                }
    if worker.carried_object_config is not None:
        # The dedicated current RGB person/depth partition resolves articulated
        # fragments and touching props. Do not also assess those same returns
        # as generic independent instances; their occupancy/priors remain.
        person_frame=worker.visual_person_frame(packet)
        person_hits=np.zeros(len(projection.source_indices),bool)
        for person in person_frame.detections:
            if person.semantic_class=="person" and person.confidence>=.5:
                person_hits |= person.mask[projection.pixel_indices_xy[:,1],projection.pixel_indices_xy[:,0]]
        person_tracks={track_id for track_id in attempted if np.count_nonzero(person_hits &
            (tracking.point_track_ids[projection.source_indices]==track_id))>=4}
        attempted-=person_tracks
    if attempted:
        worker.segmenter.set_image(packet.rgb)
    world = packet.obstacle_xyz.copy()
    c, s = math.cos(packet.robot_yaw), math.sin(packet.robot_yaw)
    world[:, :2] = (
        world[:, :2] @ np.array([[c, s], [-s, c]]) + packet.robot_position_world_xy
    )
    world[:, 2] -= packet.floor_height_robot_m
    points = (world[:, :2] - worker.anchor) @ np.array(
        [
            [math.cos(worker.anchor_yaw), -math.sin(worker.anchor_yaw)],
            [math.sin(worker.anchor_yaw), math.cos(worker.anchor_yaw)],
        ]
    )
    placement_count = 1
    if packet.scan_pose_samples_world is not None:
        copies = []
        for pose in packet.scan_pose_samples_world:
            placed = packet.obstacle_xyz @ pose[:3, :3].T + pose[:3, 3]
            copies.append(
                (placed[:, :2] - worker.anchor)
                @ np.array(
                    [
                        [math.cos(worker.anchor_yaw), -math.sin(worker.anchor_yaw)],
                        [math.sin(worker.anchor_yaw), math.cos(worker.anchor_yaw)],
                    ]
                )
            )
        points = np.concatenate(copies)
        placement_count = len(copies)
    point_ids = tracking.point_track_ids.copy()
    semantics, policies, objects, frames = [], {}, [], []
    selected_count = retained_count = 0
    contact_frame = None
    for track in tracks:
        if track.track_id in attempted:
            memory.last_visual_attempt[track.track_id] = packet.camera_timestamp_seconds
        assessment = worker._assess_track(
            packet,
            tracking,
            projection,
            track,
            track.track_id in attempted,
            plume,
            None,
            image_is_set=True,
            candidate_persistent=memory,
            shared_contact_frame=contact_frame,
            suppress_catalog_mass_for_human_identity=envelope_memory is not None and track.track_id in envelope_memory.identities,
        )
        contact_frame = assessment["contact_frame"]
        adapted = assessment["adapted"]
        policy, factor_details = evaluate_object_risk(
            adapted, worker.object_factor_prior_config,
            height_evidence=assessment.get("height_evidence"),
            human_body_contact_prior=assessment.get("human_body_contact_prior"),
            human_part_contact=assessment.get("human_part_contact"),
        )
        semantic = SemanticObstacle(
            track,
            "unknown",
            assessment["base"].semantic_confidence,
            assessment["base"].semantic_source,
        )
        semantics.append(semantic)
        policies[track.track_id] = policy
        selected = assessment["selected"]
        owned = tracking.point_track_ids == track.track_id
        ownership_translation = np.zeros(2)
        if assessment["accepted"]:
            xyz = world[selected]
            ownership[track.track_id] = (
                xyz.min(axis=0),
                xyz.max(axis=0),
                packet.camera_timestamp_seconds,
                track.centroid_world_xy,
            )
            owned = selected
            selected_count += int(selected.sum())
        elif (
            memory.config.allow_instance_bounds_prior
            and any(
                detail["mode"] == "persistence_prior"
                for detail in assessment["persistence"].values()
            )
            and track.track_id in ownership
        ):
            lo, hi, source_time, source_centroid = ownership[track.track_id]
            # This is a bounded static, axis-aligned positive association prior,
            # not hidden geometry, moving-actor tracking or scene free space.
            if (
                memory.config.translate_instance_bounds_prior
                and track.matched_previous_track
            ):
                ownership_translation = (
                    np.asarray(track.centroid_world_xy) - source_centroid
                )
            shift = np.array([*ownership_translation, 0.0])
            owned &= np.all(
                (world >= lo + shift - 0.05) & (world <= hi + shift + 0.05), axis=1
            )
            if (
                packet.camera_timestamp_seconds - source_time
                > memory.config.maximum_property_age_seconds
            ):
                owned[:] = False
            retained_count += int(owned.sum())
        elif any(
            detail["mode"] == "evidence_supported" for detail in factor_details.values()
        ):
            # Fresh motion alone does not establish ownership of old visual cues.
            if track.track_id in ownership:
                owned[:] = False
        leftovers = (tracking.point_track_ids == track.track_id) & ~owned
        if np.any(leftovers):
            auxiliary = track.track_id + 1_000_000_000
            point_ids[leftovers] = auxiliary
            unknown = adapt_physical_factors(
                PhysicalAttributes.unknown(
                    track_id=auxiliary,
                    timestamp_seconds=packet.camera_timestamp_seconds,
                ),
                timestamp_seconds=packet.camera_timestamp_seconds,
            )
            auxiliary_policy, _ = evaluate_object_risk(
                unknown, worker.object_factor_prior_config
            )
            semantics.append(
                SemanticObstacle(
                    replace(
                        track, track_id=auxiliary, point_count=int(leftovers.sum())
                    ),
                    "unknown",
                    0.0,
                    "unmatched_support_environment_factor_priors",
                )
            )
            policies[auxiliary] = auxiliary_policy
        readiness = evaluate_perception_readiness(
            RestaurantScenario(
                "persistent_object",
                "potential",
                "current and session-persistent inputs",
                runtime_factor_evidence_mode(adapted),
                0,
            ),
            adapted,
        )
        detail = {
            "track_id": track.track_id,
            "centroid_world_xy": track.centroid_world_xy,
            "velocity_world_xy": track.velocity_world_xy,
            "observation_count": track.observation_count,
            "camera_occluded_return_count": projected_count
            - len(projection.source_indices),
            "instance_centroid_world_xyz": world[selected].mean(axis=0).tolist()
            if assessment["accepted"]
            else None,
            "instance_box_xyxy": tuple(
                getattr(assessment["frame"].detections[0].bounding_box, name)
                for name in ("x_min", "y_min", "x_max", "y_max")
            )
            if assessment["accepted"]
            else None,
            "instance_mask_pixels": assessment["frame"].detections[0].mask_pixel_count
            if assessment["accepted"]
            else None,
            "instance_confidence": assessment["frame"].detections[0].confidence
            if assessment["accepted"]
            else None,
            "camera_observable": track.track_id in visible,
            "visual_attempted": track.track_id in attempted,
            "accepted_instance": assessment["accepted"],
            "selected_return_count": int(selected.sum()),
            "policy_return_count": int(owned.sum()),
            "instance_ownership_translation_xy": ownership_translation.tolist(),
            "instance_ownership_basis": "fresh_mask"
            if assessment["accepted"]
            else "tracked_translation_bounds_prior"
            if np.any(ownership_translation)
            else "static_bounds_or_geometric_prior",
            "factor_scores": factor_details,
            "property_evidence": assessment["persistence"],
            "geometry_evidence": assessment["geometry_diagnostics"],
            "sam_rejections": assessment["reasons"],
            "sam_candidate_diagnostics": assessment["sam_candidates"],
            "plume_registration": plume.last_diagnostics.get(track.track_id, {}),
        }
        objects.append(
            ObjectAssessment(
                assessment["attributes"], adapted, policy, readiness, detail
            )
        )
        frames.append(assessment)
    carried_state,carried_detail=None,None
    if worker.carried_object_config is not None:
        from robot.evaluation.carried_object_runtime import augment_carried_object
        carried_state,carried_detail=augment_carried_object(worker,packet,tracking,projection,
            world,objects,frames,policies,point_ids,semantics,contact_frame)
        selected_count=int(np.logical_or.reduce([f["selected"] for f in frames]).sum())
        for item in objects:
            item.diagnostics["policy_return_count"]=int(np.count_nonzero(point_ids==item.attributes.track_id))
    memory.finish_epoch(tracks, packet.camera_timestamp_seconds)
    active = {track.track_id for track in tracks}
    active.update(o.attributes.track_id for o in objects if o.diagnostics["accepted_instance"])
    ownership = {key: value for key, value in ownership.items() if key in active}
    # Unclustered returns stay occupied and get the same explicit prior policy.
    unclustered = point_ids < 0
    if np.any(unclustered):
        auxiliary = 2_000_000_000
        point_ids[unclustered] = auxiliary
        unknown = adapt_physical_factors(
            PhysicalAttributes.unknown(
                track_id=auxiliary, timestamp_seconds=packet.camera_timestamp_seconds
            ),
            timestamp_seconds=packet.camera_timestamp_seconds,
        )
        policies[auxiliary], _ = evaluate_object_risk(
            unknown, worker.object_factor_prior_config
        )
        semantics.append(
            SemanticObstacle(
                replace(
                    tracks[0], track_id=auxiliary, point_count=int(unclustered.sum())
                ),
                "unknown",
                0.0,
                "unclustered_occupancy_factor_priors",
            )
        )
    point_ids = np.tile(point_ids, placement_count)
    current_placement = (
        packet.sensor_timestamp_seconds,
        packet.sensor_frame_id,
        points.copy(),
        point_ids.copy(),
    )
    retained = [
        item
        for item in getattr(worker, "multi_retained", ())
        if 0
        < packet.sensor_timestamp_seconds - item[0]
        <= worker.occupancy_retention_seconds + 1e-9
    ]
    if not worker.static_scope:
        for item in retained:
            points = np.concatenate((points, item[2]))
            point_ids = np.concatenate((point_ids, item[3]))
    # Preserve positive occupancy memory separately from property evidence.
    static = dict(worker.static_points)
    if worker.static_scope:
        for track_id in np.unique(point_ids):
            key = int(track_id)
            static[key] = np.concatenate(
                (static.get(key, np.empty((0, 2))), points[point_ids == key])
            )
            cells = np.floor(
                (
                    static[key]
                    - [worker.grid_config.minimum_x, worker.grid_config.minimum_y]
                )
                / worker.grid_config.resolution
            ).astype(np.int64)
            _, unique = np.unique(cells, axis=0, return_index=True)
            static[key] = static[key][np.sort(unique)]
        for key, old_points in worker.static_points.items():
            points = np.concatenate((points, old_points))
            point_ids = np.concatenate((point_ids, np.full(len(old_points), key)))
            if key not in policies:
                unknown = adapt_physical_factors(
                    PhysicalAttributes.unknown(
                        track_id=key, timestamp_seconds=packet.camera_timestamp_seconds
                    ),
                    timestamp_seconds=packet.camera_timestamp_seconds,
                )
                policies[key], _ = evaluate_object_risk(
                    unknown, worker.object_factor_prior_config
                )
                semantics.append(
                    SemanticObstacle(
                        replace(tracks[0], track_id=key, point_count=len(old_points)),
                        "unknown",
                        0.0,
                        "unseen_static_occupancy_factor_priors",
                    )
                )
    for key in np.unique(point_ids):
        key = int(key)
        if key not in policies:
            unknown = adapt_physical_factors(
                PhysicalAttributes.unknown(
                    track_id=key, timestamp_seconds=packet.camera_timestamp_seconds
                ),
                timestamp_seconds=packet.camera_timestamp_seconds,
            )
            policies[key], _ = evaluate_object_risk(
                unknown, worker.object_factor_prior_config
            )
            semantics.append(
                SemanticObstacle(
                    replace(
                        tracks[0],
                        track_id=key,
                        point_count=int(np.count_nonzero(point_ids == key)),
                    ),
                    "unknown",
                    0.0,
                    "retained_occupancy_factor_priors",
                )
            )
    envelope_audit, envelope_origins = [], {}
    if envelope_memory is not None:
        from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE, envelope_grid_points
        envelope_audit = envelope_memory.update(tracking,
            fresh_person_selection_evidence(objects, packet.camera_timestamp_seconds, packet.camera_frame_id),
            camera_timestamp_seconds=packet.camera_timestamp_seconds)
        observed_only = worker.human_observed_occupancy_only
        for item in envelope_audit:
            item["applied_to_occupancy"] = not observed_only
        extra_points, extra_ids = envelope_grid_points([] if observed_only else envelope_audit,
            grid_config=worker.grid_config, anchor_xy=worker.anchor, anchor_yaw=worker.anchor_yaw)
        points, point_ids = np.concatenate((points, extra_points)), np.concatenate((point_ids, extra_ids))
        for item in envelope_audit:
            if observed_only:
                continue
            key, original = item["envelope_track_id"], item["track_id"]
            envelope_origins[key] = original
            unknown = adapt_physical_factors(PhysicalAttributes.unknown(track_id=key,
                timestamp_seconds=packet.camera_timestamp_seconds), timestamp_seconds=packet.camera_timestamp_seconds)
            conservative, _ = evaluate_object_risk(unknown, worker.object_factor_prior_config)
            policies[key] = max((policies[original], conservative), key=lambda value: value.risk)
            item["applied_cra_phi"] = policies[key].risk
            track = next(t for t in tracks if t.track_id == original)
            semantics.append(SemanticObstacle(replace(track, track_id=key,
                point_count=int(np.count_nonzero(extra_ids == key))), "unknown", 0., HUMAN_ENVELOPE_SOURCE))
    from robot.evaluation.restaurant_baselines import applied_field_policies
    visual_part_groups = []
    if getattr(worker, "human_visual_fragment_association", False):
        from robot.perception.visual_body_fragments import associate_visual_body_fragments
        visual_part_groups = associate_visual_body_fragments(
            tracking, packet.obstacle_xyz,
            project_fisheye_polynomial_points(packet.obstacle_xyz,
                camera_from_source=packet.camera_from_scan_robot, calibration=worker.calibration),
            worker.visual_person_frame(packet), envelope_audit,
            camera_timestamp_seconds=packet.camera_timestamp_seconds,
            camera_frame_id=packet.camera_frame_id,
            mask_boundary_tolerance_pixels=getattr(worker, "human_fragment_mask_boundary_pixels", 0))
    applied_guidance, baseline_audit = applied_field_policies(
        getattr(worker, "experiment_baseline", None), policies, objects, envelope_origins=envelope_origins,
        human_boundary_flux=worker.human_boundary_flux,
        verified_human_identities=envelope_audit,
        sensor_timestamp_seconds=packet.sensor_timestamp_seconds,
    )
    fragment_audit = None
    if fragment_memory is not None:
        from robot.perception.human_fragment_coverage import apply_coverage_risk_floors
        fragment_audit = fragment_memory.update(tracks, world[:, :2], tracking.point_track_ids,
            {t.track_id: applied_guidance[t.track_id].risk for t in tracks}, envelope_audit,
            source_time=packet.sensor_timestamp_seconds, camera_timestamp_seconds=packet.camera_timestamp_seconds)
        apply_coverage_risk_floors(applied_guidance, fragment_audit)
        for row in baseline_audit:
            key = row["track_or_group_id"]
            if any(p["envelope_track_id"] == key for p in fragment_audit["covered"]):
                row["fragment_coverage_risk_floor_prior"] = applied_guidance[key].risk
                row["applied_field_risk"] = applied_guidance[key].risk
                row["boundary_flux"] = applied_guidance[key].boundary_flux
    passing = copy.deepcopy(getattr(worker, "motion_passing", None))
    passing_audit = None
    passing_visual_evidence = copy.deepcopy(getattr(worker, "passing_visual_selection_evidence", None))
    if passing is not None:
        visual = fresh_person_selection_evidence(objects, packet.camera_timestamp_seconds,
            packet.camera_frame_id) if passing.config.require_visual_person_selection else None
        legacy_visual_ids = tuple(visual or {})
        direct = {}
        if getattr(worker, "human_fresh_person_policy", False):
            from robot.perception.person_policy_association import fresh_person_policy_associations
            direct = fresh_person_policy_associations(tracking, projection,
                worker.visual_person_frame(packet), envelope_audit,
                camera_timestamp_seconds=packet.camera_timestamp_seconds,
                camera_frame_id=packet.camera_frame_id, config=worker.human_mass_prior_config)
            # A same-frame accepted SAM cue remains the stronger existing path.
            # This alternative supplies class-policy evidence only.
            visual = {**direct, **(visual or {})}
        sides, passing_audit = passing.update(
            tracking, robot_xy=packet.robot_position_world_xy,
            robot_velocity_xy=packet.measured_robot_velocity_world_xy,
            goal_xy=packet.waypoint_world_xy,
            visual_person_track_ids=None if visual is None else tuple(visual),
            identity_prior_selection_evidence=envelope_memory.passing_selection_priors(
                tracking, camera_timestamp_seconds=packet.camera_timestamp_seconds,
                maximum_age_seconds=passing.config.maximum_identity_prior_selection_age_seconds)
            if passing.config.maximum_identity_prior_selection_age_seconds > 0 else None,
        )
        passing_audit["current_visual_selection_evidence"] = visual
        passing_audit["current_legacy_SAM_person_track_ids"] = legacy_visual_ids
        passing_audit["current_fresh_detector_person_track_ids"] = tuple(direct)
        if passing.encounter is None:
            passing_visual_evidence = None
        elif passing_audit["decision"] == "selected_trailing_tangent":
            passing_visual_evidence = copy.deepcopy((visual or {}).get(passing.encounter[0]))
        passing_audit["original_selection_visual_evidence"] = passing_visual_evidence
        passing_audit["ego_pose_reference"] = {"source":packet.pose_source,"status":packet.pose_status.value,
                                              "robot_velocity_source":"derivative of the declared ego pose history"}
        # A side preference can cover this track's unmatched returns too, but
        # their own risk/flux/unknown provenance and occupancy stay unchanged.
        for key, parameters in tuple(applied_guidance.items()):
            original = envelope_origins.get(key, key if key in sides else key-1_000_000_000)
            if original in sides:
                applied_guidance[key] = replace(parameters, passing_side=sides[original])
        for row in baseline_audit:
            key = row["track_or_group_id"]
            row["cra_passing_side"] = policies[key].passing_side.value
            row["applied_passing_side"] = applied_guidance[key].passing_side.value
    if getattr(worker, "human_visual_part_field_policy", False):
        from robot.evaluation.human_part_field_policy import apply_human_part_field_policy
        applied_guidance, baseline_audit = apply_human_part_field_policy(
            applied_guidance, baseline_audit, visual_part_groups, envelope_audit,
            sensor_timestamp_seconds=packet.sensor_timestamp_seconds,
            mask_boundary_prior_pixels=worker.human_fragment_mask_boundary_pixels)
    field_guidance_policy = worker.guidance_policy
    if getattr(worker, "zero_tangential_bias_ablation", False):
        from robot.evaluation.crossing_strategy import zero_tangential_bias
        field_guidance_policy, applied_guidance, baseline_audit = zero_tangential_bias(
            field_guidance_policy, applied_guidance, baseline_audit)
    cells = np.floor(
        (points - [worker.grid_config.minimum_x, worker.grid_config.minimum_y])
        / worker.grid_config.resolution
    ).astype(np.int64)
    _, first = np.unique(np.column_stack((cells, point_ids)), axis=0, return_index=True)
    first = np.sort(first)
    grid = build_local_occupancy_grid(
        points[first],
        point_track_ids=point_ids[first],
        semantic_obstacles=semantics,
        class_safety_policy=worker.class_policy,
        config=worker.grid_config,
        track_safety_parameters={
            key: value.safety_parameters for key, value in applied_guidance.items()
        },
    )
    pipeline_started = time.perf_counter()
    pipeline = build_safe_sage_field_pipeline(
        grid,
        guidance_policy=field_guidance_policy,
        config=replace(
            worker.field_config,
            guidance_field_config=replace(
                worker.field_config.guidance_field_config,
                merge_equal_policy_regions=True,
            ),
        ),
        semantic_obstacles=semantics,
        track_guidance_parameters=applied_guidance,
    )
    pipeline_seconds = time.perf_counter() - pipeline_started
    temporal = 0.0
    recovery = {"used": False}
    if worker.previous is not None:
        old = worker.previous.temporal_safety_derivative
        previous_h = worker.previous.pipeline.h_full
        previous_time = worker.previous.sensor_timestamp_seconds
        baseline = getattr(worker, "temporal_recovery_baseline", None)
        if (
            baseline is not None
            and previous_time < baseline[1] < packet.sensor_timestamp_seconds
            and baseline[0] < packet.sensor_frame_id
        ):
            previous_h, previous_time = baseline[2], baseline[1]
            old = None
            recovery = {
                "used": True,
                "baseline_frame_id": baseline[0],
                "baseline_timestamp": baseline[1],
                "scope": "two actual perception fields after a temporal gap; first bootstrap was not published for control",
            }
        delta = packet.sensor_timestamp_seconds - previous_time
        if delta > worker.temporal.config.maximum_time_delta_seconds:
            # A gap must stop control, but must not permanently prevent later
            # fresh fields from recovering. Keep only an unpublished numerical
            # baseline; tracker, evidence and accepted field stay transactional.
            worker.temporal_recovery_baseline = (
                packet.sensor_frame_id,
                packet.sensor_timestamp_seconds,
                pipeline.h_full,
            )
            raise ValueError(
                "temporal source gap; unpublished recovery baseline recorded; no control field"
            )
        temporal = worker.temporal.estimate(
            previous_h_full=previous_h,
            current_h_full=pipeline.h_full,
            grid_config=worker.grid_config,
            time_delta_seconds=delta,
            egomotion_translation_grid_xy=(0.0, 0.0),
            previous_estimate=old
            if isinstance(old, SafeSAGETemporalDerivativeResult)
            else None,
        )
    field = SafeSAGERuntimeField(
        packet.sensor_frame_id,
        packet.sensor_timestamp_seconds,
        worker.anchor,
        worker.anchor_yaw,
        len(packet.obstacle_xyz),
        len(packet.obstacle_xyz),
        tracking,
        SemanticLabelResult(
            tuple(SemanticTrackAssignment(semantic) for semantic in semantics)
        ),
        pipeline,
        temporal,
        time.perf_counter() - started,
        motion_passing_reference_track_ids=tuple(sorted(sides)) if passing is not None else (),
        motion_passing_reference_sides=tuple(sides[k].value for k in sorted(sides)) if passing is not None else (),
        motion_passing_reference_source_timestamp_seconds=passing.encounter[2] if passing is not None and sides else None,
    )
    # Transaction commit only after the whole multi-obstacle field succeeds.
    worker.tracker, worker.plume, worker.persistent_properties, worker.previous = (
        tracker,
        plume,
        memory,
        field,
    )
    worker.static_points, worker.object_ownership_bounds = static, ownership
    if passing is not None:
        worker.motion_passing = passing
        worker.passing_visual_selection_evidence = passing_visual_evidence
    if envelope_memory is not None:
        worker.human_body_envelope = envelope_memory
    if fragment_memory is not None:
        worker.human_fragment_coverage = fragment_memory
    worker.temporal_recovery_baseline = None
    if carried_state is not None:
        worker.carried_runtime_state=carried_state
    worker.multi_retained = (
        retained + [current_placement] if worker.occupancy_retention_seconds else []
    )
    primary, raw = objects[0], frames[0]
    diagnostics = {
        "stage_wall_seconds": {
            "perception_ownership_occupancy": pipeline_started - started,
            "field_solve": pipeline_seconds,
        },
        "measured_robot_velocity_world_xy": packet.measured_robot_velocity_world_xy,
        "mode": "persistent_multi_object_factor_priors",
        "experiment_baseline": worker.experiment_baseline.value,
        "applied_field_policies": baseline_audit,
        "motion_aware_passing": passing_audit,
        "motion_passing_reference_track_ids": field.motion_passing_reference_track_ids,
        "human_body_envelope_priors": envelope_audit,
        "current_visual_body_part_associations": visual_part_groups,
        "human_fragment_coverage_prior": fragment_audit,
        "object_count": len(objects),
        "objects": [item.diagnostics for item in objects],
        "carried_object_groups":carried_detail,
        "occupancy_policies": [
            {
                "track_or_group_id": key,
                "basis": "human_body_envelope_geometry_prior" if key in envelope_origins else "current_object_track"
                if key in active
                else "unmatched_or_remembered_occupancy_group",
                "phi": value.risk,
                "group_policy":next((group for group in (carried_detail or {}).get("confirmed_groups",[]) if key in group["members"]),None),
                "factor_score_basis":"dominant_member_assessment" if any(key in group["members"] for group in (carried_detail or {}).get("confirmed_groups",[])) else "individual_assessment",
                "factor_scores": {
                    item.factor.value: item.conservative_risk
                    for item in value.assessment.factors
                },
            }
            for key, value in policies.items()
        ],
        "accepted_instance_count": sum(
            item.diagnostics["accepted_instance"] for item in objects
        ),
        "complete_perception_count": sum(item.readiness.ready for item in objects),
        "selected_return_count": selected_count,
        "retained_association_return_count": retained_count,
        "factor_prior_config": {
            "scope": worker.object_factor_prior_config.scope,
            "factor_risks": worker.object_factor_prior_config.factor_risks.as_dict(),
        },
        "property_persistence_scope": memory.config.scope,
        "maximum_property_age_seconds": memory.config.maximum_property_age_seconds,
        "initial_plume_acquisition_frozen": memory.frozen,
        "visual_reassessment_interval_seconds": memory.config.minimum_visual_reassessment_interval_seconds,
        "maximum_visual_tracks_per_update": memory.config.maximum_visual_tracks_per_update,
        "retained_occupancy_epochs": [
            {"timestamp": item[0], "sensor_frame_id": item[1]} for item in retained
        ],
        "fixed_lambda": 0. if getattr(worker, "zero_tangential_bias_ablation", False) else 1.25,
        "physical_CRA_lambda": 1.25,
        "zero_tangential_bias_ablation": getattr(worker, "zero_tangential_bias_ablation", False),
        "human_control_ablation": {
            "observed_occupancy_only": worker.human_observed_occupancy_only,
            "requested_boundary_flux": worker.human_boundary_flux,
            "identity_scope": "unchanged bounded visual identity with current positive LiDAR; original factor clocks retained",
        },
        "merge_equal_policy_regions": True,
        "temporal_gap_recovery": recovery,
        "occupancy_retention_scope": worker.static_scope,
        "field_raw_placement_count": len(points),
        "field_cell_representative_count": len(first),
        "field_discarded_cell_representatives": grid.discarded_point_count,
        "controller_numeric_policy_ready": True,
        "sensor_perception_ready": all(item.readiness.ready for item in objects),
    }
    return LiveObjectResult(
        field,
        primary.attributes,
        primary.adapted,
        primary.policy,
        primary.readiness,
        raw["frame"],
        raw["fusion"],
        diagnostics,
        tuple(objects),
        grid,
    )
