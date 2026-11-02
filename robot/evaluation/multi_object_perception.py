"""Opt-in all-track CRA mapping using shared live perception components.

All current clustered occupancy receives numeric policies. Exact accepted mask
ownership is preserved; unclustered and unmatched support/environment returns
receive their own labeled factor priors, not a nearby object's low assessment.
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
    rolling = getattr(worker, "rolling_field", None)
    grid_config = worker.grid_config
    if rolling is not None:
        c, s = math.cos(worker.anchor_yaw), math.sin(worker.anchor_yaw)
        position = (np.asarray(packet.robot_position_world_xy) - worker.anchor) @ np.array([[c, -s], [s, c]])
        grid_config = rolling.select(position, grid_config)
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
    prediction_memory = copy.deepcopy(getattr(worker, "predictive_human_geometry", None))
    preparation_memory = copy.deepcopy(getattr(worker, "parallel_preparation", None))
    human_mass_memory = copy.deepcopy(getattr(worker, "human_mass_identity_retention", None))
    furniture_memory = copy.deepcopy(getattr(worker, "stationary_furniture_memory", None))
    unknown_memory = copy.deepcopy(getattr(worker, "unknown_obstacle_memory", None))
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
                    - [grid_config.minimum_x, grid_config.minimum_y]
                )
                / grid_config.resolution
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
    human_mass_audit = None
    if envelope_memory is not None:
        from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE, envelope_grid_points
        envelope_audit = envelope_memory.update(tracking,
            fresh_person_selection_evidence(objects, packet.camera_timestamp_seconds, packet.camera_frame_id),
            camera_timestamp_seconds=packet.camera_timestamp_seconds,
            robot_world_xy=packet.robot_position_world_xy)
        if human_mass_memory is not None:
            from robot.evaluation.human_mass_runtime import apply_body_identity_mass_retention
            objects, human_mass_audit = apply_body_identity_mass_retention(human_mass_memory,
                envelope_memory, tracking, packet, objects, frames, policies, worker.object_factor_prior_config)
        observed_only = worker.human_observed_occupancy_only
        for item in envelope_audit:
            item["applied_to_occupancy"] = not observed_only
        extra_points, extra_ids = envelope_grid_points([] if observed_only else envelope_audit,
            grid_config=grid_config, anchor_xy=worker.anchor, anchor_yaw=worker.anchor_yaw)
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
    prediction, prediction_audit = None, None
    if prediction_memory is not None:
        prediction, prediction_audit = prediction_memory.update(tracking, envelope_audit,
            sensor_frame_id=packet.sensor_frame_id, anchor=worker.anchor, anchor_yaw=worker.anchor_yaw)
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
    preparation_audit, reference_selection_time = None, None
    use_nominal_segment = (getattr(worker, 'encounter_motion', None) is not None
        or passing is not None and passing.config.encounter_geometry == 'oncoming_parallel')
    nominal_direction = getattr(packet, 'nominal_route_direction_world_xy', None) if use_nominal_segment else None
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
            nominal_route_direction_world_xy=nominal_direction,
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
        elif passing_audit["decision"] in {"selected_trailing_tangent", "selected_parallel_away_tangent"}:
            passing_visual_evidence = copy.deepcopy((visual or {}).get(passing.encounter[0]))
        passing_audit["original_selection_visual_evidence"] = passing_visual_evidence
        passing_audit["ego_pose_reference"] = {"source":packet.pose_source,"status":packet.pose_status.value,
                                              "robot_velocity_source":"derivative of the declared ego pose history"}
        normal_sides = dict(sides)
        if preparation_memory is not None:
            preparation_sides, preparation_audit = preparation_memory.update(prediction,
                robot_xy=packet.robot_position_world_xy, route_direction_world_xy=nominal_direction,
                visual_evidence=visual or {}, camera_timestamp_seconds=packet.camera_timestamp_seconds,
                camera_frame_id=packet.camera_frame_id)
            used = not sides and 'ambiguous' not in passing_audit['decision']
            if used:
                sides = preparation_sides
            preparation_audit['used_for_reference'] = bool(used and sides)
            preparation_audit['normal_motion_selector_has_precedence'] = bool(normal_sides)
        if sides:
            reference_selection_time = (passing.encounter[2] if normal_sides else
                preparation_audit['original_selection']['selected_timestamp_seconds'])
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
    normal_guidance = dict(applied_guidance)
    encounter = copy.deepcopy(getattr(worker, 'encounter_motion', None))
    motion_estimate = copy.deepcopy(getattr(worker, 'rear_encounter_motion_estimate', None))
    motion_estimate_audit = None
    rear_policy_motion = ()
    person_relative_memory = copy.deepcopy(getattr(worker, 'person_relative_rear_memory', None))
    encounter_audit, yield_ids = None, None
    rear_ids, rear_route = (), None
    if encounter is not None:
        from robot.evaluation.encounter_guidance import apply_encounter_guidance
        human_ids = tuple(r['track_id'] for r in envelope_audit)
        encounter_tracking = tracking
        if motion_estimate is not None:
            estimates, motion_estimate_audit = motion_estimate.update(tracking, human_ids)
            # Policy-only copy: physical kinetic/geometry/body-coverage inputs,
            # passing selection and rear target still use original tracking.
            encounter_tracking = replace(tracking, tracked_obstacles=tuple(
                replace(track, velocity_world_xy=estimates[track.track_id])
                if track.track_id in estimates else track for track in tracks))
        encounter_audit = encounter.update(encounter_tracking, robot_xy=packet.robot_position_world_xy,
            goal_xy=packet.waypoint_world_xy, human_track_ids=human_ids,
            nominal_route_direction_world_xy=nominal_direction)
        if motion_estimate_audit is not None:
            for key, row in encounter_audit['tracks'].items():
                row['policy_motion_fit'] = motion_estimate_audit['tracks'][key]
                if row['policy_motion_fit']['accepted']:
                    row['basis'] = 'inferred original-source-window centroid velocity for encounter policy; configured nominal route segment'
            if getattr(worker, 'rear_consistent_motion_fit', False):
                from robot.safety.encounter_motion_estimate import EncounterPolicyMotion
                rear_policy_motion = tuple(EncounterPolicyMotion.from_diagnostic(row, motion_estimate.config)
                    for key, row in motion_estimate_audit['tracks'].items()
                    if row['accepted'] and key in sides)
        applied_guidance, baseline_audit, yield_ids = apply_encounter_guidance(
            applied_guidance, baseline_audit, encounter_audit, passing_audit,
            source_time=packet.sensor_timestamp_seconds)
        if getattr(worker, "live_rear_pass_config", None) is not None:
            from robot.evaluation.live_rear_pass import adapt_live_rear_guidance
            applied_guidance, baseline_audit, yield_ids, rear_ids, rear_route = adapt_live_rear_guidance(
                normal_guidance, applied_guidance, baseline_audit, tracking, packet,
                anchor=worker.anchor, anchor_yaw=worker.anchor_yaw,
                crossing_ids=yield_ids, config=worker.live_rear_pass_config,
                policy_motion=rear_policy_motion, person_relative_memory=person_relative_memory)
    if getattr(worker, "zero_tangential_bias_ablation", False):
        from robot.evaluation.crossing_strategy import zero_tangential_bias
        field_guidance_policy, applied_guidance, baseline_audit = zero_tangential_bias(
            field_guidance_policy, applied_guidance, baseline_audit)
    uncertainty_snapshot, uncertainty_audit = None, []
    if envelope_memory is not None and envelope_memory.planning_uncertainty:
        from robot.perception.tracking_uncertainty import TrackingUncertaintySnapshot, SOURCE as UNCERTAINTY_SOURCE
        for body in envelope_audit:
            envelope_memory.geometry_policies[body['track_id']] = applied_guidance[body['envelope_track_id']]
        region_policies = []
        for region in envelope_memory.uncertainty_regions:
            previous_policy = envelope_memory.geometry_policies.get(region.track_id)
            if previous_policy is None:
                raise ValueError("uncertainty occupancy requires an original verified geometry policy")
            unknown = adapt_physical_factors(PhysicalAttributes.unknown(track_id=region.geometry_id,
                timestamp_seconds=packet.camera_timestamp_seconds), timestamp_seconds=packet.camera_timestamp_seconds)
            floor, _ = evaluate_object_risk(unknown, worker.object_factor_prior_config)
            guidance = max((previous_policy, floor.guidance_parameters,
                applied_guidance.get(region.track_id, previous_policy)), key=lambda g: g.risk)
            extra, _, _, _ = region.cells(grid_config, worker.anchor, worker.anchor_yaw)
            points = np.concatenate((points, extra))
            point_ids = np.concatenate((point_ids, np.full(len(extra), region.geometry_id, dtype=np.int64)))
            applied_guidance[region.geometry_id] = guidance
            c, s = math.cos(packet.robot_yaw), math.sin(packet.robot_yaw)
            center_robot = (np.asarray(region.center_world_xy)-packet.robot_position_world_xy) @ np.array([[c, -s], [s, c]])
            semantics.append(SemanticObstacle(replace(tracks[0], track_id=region.geometry_id,
                centroid_world_xy=region.center_world_xy, centroid_robot_xy=tuple(center_robot),
                point_count=len(extra), velocity_world_xy=(0., 0.), velocity_robot_xy=(0., 0.),
                observation_count=0, matched_previous_track=False), "unknown", 0., UNCERTAINTY_SOURCE))
            region_policies.append((region, guidance))
            row = dict(region.diagnostic(), applied_cra_phi=guidance.risk,
                field_occupied_cells=len(extra), original_applied_risk=previous_policy.risk)
            uncertainty_audit.append(row)
            baseline_audit.append(dict(track_or_group_id=region.geometry_id, observed_class="unknown",
                class_basis=UNCERTAINTY_SOURCE, cra_phi=guidance.risk, applied_field_risk=guidance.risk,
                boundary_flux=guidance.boundary_flux, **{"lambda": guidance.tangential_strength},
                field_used_for_control=True, factor_basis="original_risk_floor_and_configured_unknown_priors"))
        uncertainty_snapshot = TrackingUncertaintySnapshot(packet.sensor_timestamp_seconds, tuple(region_policies))
    furniture_snapshot, furniture_audit = None, None
    if furniture_memory is not None:
        from robot.perception.stationary_furniture_memory import (
            ID_OFFSET, SOURCE, person_exclusion_points,
        )
        excluded, person_exclusion_audit = person_exclusion_points(world, projection, worker.visual_person_frame(packet),
            packet.camera_frame_id, packet.camera_timestamp_seconds, envelope_audit)
        if uncertainty_snapshot is not None:
            for region, _ in uncertainty_snapshot.regions_and_guidance:
                selected = np.linalg.norm(world[:, :2]-region.center_world_xy, axis=1) <= region.radius_m
                excluded = np.concatenate((excluded, world[selected, :2]))
        # An original geometric track may contain separately owned instance and
        # unmatched support returns. Remember its strongest actually applied risk.
        original_guidance = {}
        for track in tracks:
            keys = np.unique(current_placement[3][:len(world)][tracking.point_track_ids == track.track_id])
            original_guidance[track.track_id] = max((applied_guidance[int(k)] for k in keys), key=lambda g: g.risk)
        furniture_snapshot, furniture_audit = furniture_memory.update(
            world[:, :2], tracking.point_track_ids, tracks, original_guidance,
            source_time=packet.sensor_timestamp_seconds, source_frame=packet.sensor_frame_id,
            robot_xy=packet.robot_position_world_xy, excluded_points=excluded,
            excluded_track_ids=[r["track_id"] for r in envelope_audit])
        furniture_audit["person_exclusion_observations"] = person_exclusion_audit
        extra, extra_ids = furniture_snapshot.placements(grid_config, worker.anchor, worker.anchor_yaw)
        points, point_ids = np.concatenate((points, extra)), np.concatenate((point_ids, extra_ids))
        for record in furniture_snapshot.surfaces:
            key = ID_OFFSET + record.track_id
            applied_guidance[key] = record.guidance
            semantics.append(SemanticObstacle(replace(tracks[0], track_id=key,
                point_count=int(np.count_nonzero(extra_ids == key))), "unknown", 0., SOURCE))
            baseline_audit.append(dict(track_or_group_id=key, observed_class="unknown",
                class_basis=SOURCE, original_track_id=record.track_id,
                original_source_time=record.source_time, original_source_frame=record.source_frame,
                cra_phi=record.guidance.risk, applied_field_risk=record.guidance.risk,
                boundary_flux=record.guidance.boundary_flux, lambda_value=record.guidance.tangential_strength,
                field_used_for_control=True, factor_basis="retained_applied_policy_not_new_factor_evidence"))
            baseline_audit[-1]["lambda"] = baseline_audit[-1].pop("lambda_value")
    unknown_snapshot = None
    if unknown_memory is not None:
        from robot.perception.unknown_obstacle_memory import SOURCE as UNKNOWN_SOURCE
        # A policy cannot be inferred from an authored object or nearby mask.
        # Preserve the strongest policy actually applied to each raw surface.
        raw_guidance = {}
        for track in tracks:
            keys = np.unique(current_placement[3][:len(world)][tracking.point_track_ids == track.track_id])
            raw_guidance[track.track_id] = max((applied_guidance[int(k)] for k in keys), key=lambda g: g.risk)
        human_regions = [(np.asarray(r["center_world_xy"]), r["radius_m"]) for r in envelope_audit]
        if uncertainty_snapshot is not None:
            human_regions.extend((np.asarray(r.center_world_xy), r.radius_m)
                for r, _ in uncertainty_snapshot.regions_and_guidance)
        unknown_snapshot = unknown_memory.update(world[:, :2], tracking.point_track_ids,
            tracks, raw_guidance, source_time=packet.sensor_timestamp_seconds,
            source_frame=packet.sensor_frame_id, robot_xy=packet.robot_position_world_xy,
            furniture_memory=furniture_memory,
            excluded_points=excluded if furniture_memory is not None else (),
            human_regions=human_regions,
            covered_human_fragments=fragment_audit.get("covered", ()) if fragment_audit else ())
        for region in unknown_snapshot.regions:
            extra, _, _, _ = region.cells(grid_config, worker.anchor, worker.anchor_yaw)
            key, guidance = region.geometry_id, region.original.guidance
            points = np.concatenate((points, extra))
            point_ids = np.concatenate((point_ids, np.full(len(extra), key, dtype=np.int64)))
            applied_guidance[key] = guidance
            semantics.append(SemanticObstacle(replace(tracks[0], track_id=key,
                point_count=len(extra), observation_count=0, matched_previous_track=False),
                "unknown", 0., UNKNOWN_SOURCE))
            baseline_audit.append(dict(track_or_group_id=key, observed_class="unknown",
                class_basis=UNKNOWN_SOURCE, cra_phi=guidance.risk, applied_field_risk=guidance.risk,
                boundary_flux=guidance.boundary_flux, **{"lambda": guidance.tangential_strength},
                field_used_for_control=True, factor_basis="fixed_unknown_phi_with_previous_risk_floor",
                original_track_id=region.original.track_id, original_source_time=region.original.source_time))
    cells = np.floor(
        (points - [grid_config.minimum_x, grid_config.minimum_y])
        / grid_config.resolution
    ).astype(np.int64)
    _, first = np.unique(np.column_stack((cells, point_ids)), axis=0, return_index=True)
    first = np.sort(first)
    grid = build_local_occupancy_grid(
        points[first],
        point_track_ids=point_ids[first],
        semantic_obstacles=semantics,
        class_safety_policy=worker.class_policy,
        config=grid_config,
        track_safety_parameters={
            key: value.safety_parameters for key, value in applied_guidance.items()
        },
    )
    if unknown_snapshot is not None:
        unknown_checks = [unknown_snapshot.coverage(r.original.track_id, r.original.source_time,
            r.original.center, grid, worker.anchor, worker.anchor_yaw) for r in unknown_snapshot.regions]
        if any(row is None for row in unknown_checks):
            raise ValueError("unknown obstacle geometry is not fully represented in the current local field")
    if fragment_memory is not None and fragment_memory.config.short_fragment_coverage:
        from robot.perception.human_fragment_coverage import validate_coverage_grid
        for proof in fragment_audit["covered"]:
            proof["field_coverage"] = validate_coverage_grid(
                proof, grid, worker.anchor, worker.anchor_yaw)
    pipeline_started = time.perf_counter()
    motion_reference = None
    if getattr(worker, 'observed_motion_reference', False):
        from robot.safety.observed_motion_reference import ObservedHumanMotionReference
        motion_reference = ObservedHumanMotionReference.from_current(tracking, envelope_audit,
            tuple(sorted(normal_sides)), frame=packet.sensor_frame_id)
    def build_candidate(guidance_parameters):
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
            track_guidance_parameters=guidance_parameters,
        )
        recovery = {"used": False}
        temporal = 0.0
        if worker.previous is not None:
            old = worker.previous.temporal_safety_derivative
            previous_h = worker.previous.pipeline.h_full
            previous_grid = worker.previous.pipeline.configuration_space.occupancy_grids[0].config
            previous_time = worker.previous.sensor_timestamp_seconds
            baseline = getattr(worker, "temporal_recovery_baseline", None)
            if (
                baseline is not None
                and previous_time < baseline[1] < packet.sensor_timestamp_seconds
                and baseline[0] < packet.sensor_frame_id
            ):
                previous_h, previous_time = baseline[2], baseline[1]
                previous_grid = baseline[3] if len(baseline) > 3 else previous_grid
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
                    grid_config,
                )
                raise ValueError(
                    "temporal source gap; unpublished recovery baseline recorded; no control field"
                )
            if rolling is not None:
                from robot.safety.rolling_local_field import rolling_temporal_derivative
                temporal = rolling_temporal_derivative(
                    worker.temporal, previous_h, pipeline.h_full,
                    previous_grid=previous_grid, current_grid=grid_config,
                    time_delta_seconds=delta, previous_estimate=old
                    if isinstance(old, SafeSAGETemporalDerivativeResult) else None)
                temporal.sample(query_position_grid_xy=position,
                                query_yaw_radians=packet.robot_yaw - worker.anchor_yaw)
            else:
                temporal = worker.temporal.estimate(
                    previous_h_full=previous_h,
                    current_h_full=pipeline.h_full,
                    grid_config=grid_config,
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
            motion_passing_reference_source_timestamp_seconds=reference_selection_time,
            crossing_yield_track_ids=yield_ids,
            rear_pass_reference_track_ids=rear_ids,
            rear_pass_route_direction_grid_xy=rear_route,
            rear_pass_policy_motion=rear_policy_motion,
            person_relative_rear_reference=(None if person_relative_memory is None else person_relative_memory.active_reference),
            predictive_human_geometry=prediction,
            observed_motion_forecast=motion_reference,
        )
        return field, recovery

    field, recovery = build_candidate(applied_guidance)
    transition_audit = None
    checked_control = getattr(worker, "checked_wait_control", None)
    if checked_control is not None and yield_ids:
        from robot.evaluation.live_checked_wait import check_wait, packet_state_in_field

        transition_audit = check_wait(checked_control, packet_state_in_field(packet, field), field)
        transition_audit["previous_accepted_source_timestamp_seconds"] = (
            worker.previous.sensor_timestamp_seconds if worker.previous is not None else None)
        transition_audit["temporal_recovery"] = recovery
        transition_audit["candidate_track_ids"] = yield_ids
        if transition_audit["accepted"]:
            transition_audit["selected_policy"] = "checked_zero_bias_wait_field"
        else:
            # A real rebuild from the same accepted source pair, never the
            # rejected candidate or its smoothed temporal derivative.
            field, recovery = build_candidate(normal_guidance)
            field = replace(field, crossing_yield_track_ids=())
            yield_ids = ()
            for row in baseline_audit:
                cue = row["encounter_guidance_ablation"]
                row["lambda"] = normal_guidance[row["track_or_group_id"]].tangential_strength
                cue.update(lambda_suppressed=False, basis="checked_wait_rejected_normal_guidance")
            transition_audit["selected_policy"] = "normal_guidance_current_field"
        applied_guidance = normal_guidance if not yield_ids else applied_guidance
    pipeline_seconds = time.perf_counter() - pipeline_started
    field = replace(field, build_wall_seconds=time.perf_counter() - started)
    rolling_diagnostic = None if rolling is None else {
        "previous_bounds": asdict(worker.grid_config),
        "current_bounds": asdict(grid_config),
        "recentered": grid_config != worker.grid_config,
        "step_m": rolling.step_m,
        "scope": "fixed world anchor; integer-cell translated window; original source epochs; temporal sampling requires consecutive-source overlap",
    }
    # Transaction commit only after the whole multi-obstacle field succeeds.
    worker.tracker, worker.plume, worker.persistent_properties, worker.previous = (
        tracker,
        plume,
        memory,
        field,
    )
    worker.static_points, worker.object_ownership_bounds = static, ownership
    worker.grid_config = grid_config
    if passing is not None:
        worker.motion_passing = passing
        worker.passing_visual_selection_evidence = passing_visual_evidence
    if encounter is not None:
        worker.encounter_motion = encounter
    if motion_estimate is not None:
        worker.rear_encounter_motion_estimate = motion_estimate
    if person_relative_memory is not None:
        worker.person_relative_rear_memory = person_relative_memory
    if envelope_memory is not None:
        worker.human_body_envelope = envelope_memory
    if fragment_memory is not None:
        worker.human_fragment_coverage = fragment_memory
    if prediction_memory is not None:
        worker.predictive_human_geometry = prediction_memory
    if preparation_memory is not None:
        worker.parallel_preparation = preparation_memory
    if human_mass_memory is not None:
        worker.human_mass_identity_retention = human_mass_memory
    if furniture_memory is not None:
        worker.stationary_furniture_memory = furniture_memory
    if unknown_memory is not None:
        worker.unknown_obstacle_memory = unknown_memory
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
        "encounter_motion_classification": encounter_audit,
        "rear_encounter_motion_estimate": motion_estimate_audit,
        "rear_policy_motion_estimates": [m.diagnostic() for m in rear_policy_motion],
        "person_relative_rear": None if person_relative_memory is None else person_relative_memory.diagnostic(),
        "crossing_yield_track_ids": yield_ids,
        "rear_pass_reference_track_ids": rear_ids,
        "rear_pass_route_direction_grid_xy": rear_route,
        "checked_wait_transition": transition_audit,
        "motion_passing_reference_track_ids": field.motion_passing_reference_track_ids,
        "motion_passing_reference_source_timestamp_seconds": field.motion_passing_reference_source_timestamp_seconds,
        "human_body_envelope_priors": envelope_audit,
        "human_track_recovery": None if envelope_memory is None else envelope_memory.recovery_audit,
        "tracking_uncertainty_regions": uncertainty_audit,
        "human_track_recovery_config": None if envelope_memory is None or envelope_memory.recovery_config is None
            else asdict(envelope_memory.recovery_config),
        "human_mass_body_identity_retention": human_mass_audit,
        "stationary_furniture_memory": furniture_audit,
        "unknown_obstacle_memory": None if unknown_snapshot is None else unknown_snapshot.diagnostic(),
        "unknown_obstacle_field_checks": None if unknown_snapshot is None else unknown_checks,
        "predictive_human_geometry": prediction_audit,
        "observed_motion_reference": None if motion_reference is None else motion_reference.diagnostic(),
        "parallel_preparation": preparation_audit,
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
        "fixed_lambda": None if encounter is not None else 0. if getattr(worker, "zero_tangential_bias_ablation", False) else 1.25,
        "encounter_guidance_ablation": encounter is not None,
        "physical_CRA_lambda": 1.25,
        "zero_tangential_bias_ablation": getattr(worker, "zero_tangential_bias_ablation", False),
        "human_control_ablation": {
            "observed_occupancy_only": worker.human_observed_occupancy_only,
            "requested_boundary_flux": worker.human_boundary_flux,
            "identity_scope": "unchanged bounded visual identity with current positive LiDAR; original factor clocks retained",
        },
        "merge_equal_policy_regions": True,
        "temporal_gap_recovery": recovery,
        "rolling_local_field": rolling_diagnostic,
        "occupancy_retention_scope": worker.static_scope,
        "field_raw_placement_count": len(points),
        "field_cell_representative_count": len(first),
        "field_discarded_cell_representatives": grid.discarded_point_count,
        "controller_numeric_policy_ready": True,
        "sensor_perception_ready": all(item.readiness.ready for item in objects),
    }
    risk_camera_frame = None
    if getattr(worker, "capture_risk_masks", False):
        from robot.evaluation.risk_camera_overlay import capture_risk_camera_frame
        risk_camera_frame = capture_risk_camera_frame(packet, objects, frames, policies)
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
        furniture_snapshot,
        uncertainty_snapshot,
        risk_camera_frame,
        unknown_snapshot,
    )
