"""Experimental CRA on verified groups of static props; optional evidence memory."""

import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from robot.evaluation.cra_experiments import RestaurantScenario
from robot.evaluation.perception_readiness import evaluate_perception_readiness, runtime_factor_evidence_mode
from robot.perception.image_object_association import StationaryImageObjects, associate_image_objects
from robot.risk.object_risk_policy import evaluate_object_risk
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.semantic_obstacle import SemanticObstacle


def current_target_group_memory_ready(objects, memory, center_world_xyz, source_timestamp):
    """Independent stationary test gate only; fixture truth never enters CRA."""
    if memory is None:
        return False
    for obj in objects:
        d = obj.diagnostics
        a = d.get('image_object_association', {})
        center = d.get('instance_centroid_world_xyz')
        if (d.get('accepted_instance') and d.get('policy_return_count', 0) > 0
            and a.get('current_source_time') == source_timestamp and center is not None
            and np.linalg.norm(np.asarray(center)-center_world_xyz) <= .20
            and all((obj.attributes.track_id, name) in memory.properties for name in ('mass','contact'))):
            return True
    return False


def coalesce_image_members_for_evaluation(records):
    """Do not count a verified object AND its raw member fragments as competitors.

    No truth is used here. Unrelated/ambiguous raw tracks remain competitors;
    factor values, original records and the independent matching gate stay intact.
    """
    result = []
    for record in records:
        members = set()
        for item in record["objects"]:
            d = item["diagnostics"]
            a = d.get("image_object_association", {})
            if (d.get("accepted_instance") and d.get("policy_return_count", 0) > 0
                and a.get("observations", 0) >= 3 and not a.get("rejections")
                and a.get("current_source_time") == record["sensor_timestamp"]):
                members.update(f["raw_track_id"] for f in a.get("fragments", []))
        result.append({**record, "objects": [o for o in record["objects"] if o["track_id"] not in members]})
    return result


def augment_image_objects(worker, packet, tracking, projection, world, objects,
                          frames, policies, point_ids, semantics, contact_frame, plume,
                          *, image_is_set=False):
    """Stage state until the current field commits; original occupancy is untouched."""
    from robot.evaluation.multi_object_perception import ObjectAssessment

    config = worker.image_object_config
    state = copy.deepcopy(getattr(worker, "image_object_state", None)) or StationaryImageObjects(config)
    property_memory = None
    if getattr(worker, "image_object_property_persistence", False):
        from robot.perception.persistent_object_evidence import PersistentObjectEvidence
        property_memory = getattr(state, "property_memory", None)
        if property_memory is None:
            property_memory = PersistentObjectEvidence(worker.persistent_properties.config)
        # Follow the existing explicit acquisition/freeze lifecycle. Borrowing
        # never renews original evidence clocks; group proof remains mandatory.
        property_memory.frozen = worker.persistent_properties.frozen
    memory_tracks = []
    static_config = getattr(worker, 'static_prop_evidence_config', None)
    static_memory = None
    static_matches, static_audit = {}, []
    excluded = np.zeros(len(world), bool)
    if getattr(worker, 'service_static_props', False):
        # A current person cue excludes static-property use; it does not label
        # unrelated background dynamic or erase any raw occupancy.
        person_frame = worker.visual_person_frame(packet)
        if person_frame is not None:
            if (person_frame.source_frame_id != packet.camera_frame_id or
                    person_frame.source_timestamp_seconds != packet.camera_timestamp_seconds):
                raise ValueError('static props require current person exclusions')
            for person in person_frame.detections:
                if person.semantic_class == 'person' and person.confidence >= .5:
                    hits = person.mask[projection.pixel_indices_xy[:,1],projection.pixel_indices_xy[:,0]]
                    excluded[projection.source_indices[hits]] = True
        identities = getattr(getattr(worker, 'human_body_envelope', None), 'identities', {})
        excluded |= np.isin(tracking.point_track_ids, list(identities))
        for obj in objects:
            if obj.attributes.semantic_class == 'person':
                excluded |= tracking.point_track_ids == obj.attributes.track_id
    if static_config is not None:
        from robot.perception.static_prop_evidence import StaticPropEvidence
        static_memory = getattr(state, 'static_properties', None)
        if static_memory is None:
            static_memory = StaticPropEvidence(static_config, worker.persistent_properties.config)
        static_memory.expire(packet.camera_timestamp_seconds)
        if worker.persistent_properties.frozen:
            static_matches, static_audit = static_memory.match(world, tracking.point_track_ids,
                packet.camera_timestamp_seconds, excluded=excluded)
    if not image_is_set:
        worker.segmenter.set_image(packet.rgb)
    proposals = worker.segmenter.image_proposals(tracking.point_track_ids, projection)
    candidates, audit = associate_image_objects(proposals, projection, tracking.point_track_ids,
                                                packet.obstacle_xyz, config)
    admitted = []
    for candidate in candidates:
        if np.any(candidate['selected'] & excluded):
            candidate['diagnostic']['static_scope_rejection'] = 'current person support; retain original human/raw policy'
        else:
            admitted.append(candidate)
    candidates = admitted
    view = None
    moving_config = getattr(worker, "moving_image_object_config", None)
    if moving_config is not None:
        from robot.perception.moving_view_association import image_object_view
        view = image_object_view(packet, worker.calibration, moving_config)
    archive_directory = getattr(worker, 'image_object_archive_directory', None)
    archive_path = None
    if archive_directory is not None:
        # Evaluation-only copies of actual candidate inputs, including failures.
        # Runtime never reads this directory; export is inside measured build time.
        archive = Path(archive_directory)
        archive.mkdir(parents=True, exist_ok=True)
        archive_path = archive / f'{packet.sensor_frame_id:06d}.npz'
        attempt = 1
        while archive_path.exists():
            archive_path = archive / f'{packet.sensor_frame_id:06d}-attempt{attempt}.npz'
            attempt += 1
        np.savez_compressed(archive_path,
            world_xyz=world, robot_xyz=packet.obstacle_xyz,
            raw_track_ids=tracking.point_track_ids,
            raw_velocities_robot_xy=tracking.point_velocities_robot_xy,
            selected=np.stack([c['selected'] for c in candidates]) if candidates else np.empty((0,len(world)),bool),
            masks=np.stack([c['proposal'].detection.mask for c in candidates]) if candidates else np.empty((0,*packet.rgb.shape[:2]),bool),
            camera_from_world=view.camera_from_world if view is not None else np.empty((0,0)),
            sensor_timestamp=packet.sensor_timestamp_seconds,
            source_frame_id=packet.sensor_frame_id,
            candidates_json=json.dumps([c['diagnostic'] for c in candidates]))
    verified = state.update(candidates, world, packet.sensor_timestamp_seconds, view=view)
    detail = {"scope": "stationary separated-prop image grouping hypothesis; no geometry creation or raw identity merge",
              "hold_overlapping_hypotheses": config.hold_overlapping_hypotheses,
              "source_frame": packet.sensor_frame_id, "source_time": packet.sensor_timestamp_seconds,
              "candidates": audit, "accepted_object_ids": []}
    if view is not None:
        detail.update(scope='bounded static-prop moving-view correspondence; inferred partial-surface identity, original raw geometry unchanged',
            ego_reference_source=packet.pose_source, ego_reference_status=packet.pose_status.value)
    if archive_path is not None:
        detail['candidate_archive'] = str(archive_path)
    for candidate_index, candidate in enumerate(verified):
        key, selected = candidate["object_id"], candidate["selected"]
        if static_memory is not None and worker.persistent_properties.frozen and (
                key in static_memory.records or any(
                    np.any(selected & match[0]) for match in static_matches.values())):
            candidate['diagnostic']['static_property_acquisition'] = 'retained original static-location evidence'
            continue
        center = packet.obstacle_xyz[selected, :2].mean(axis=0)
        world_center = world[selected, :2].mean(axis=0)
        # Do not turn a stationary-support hypothesis into measured zero speed.
        # Original per-return inferred velocities remain available to CRA.
        velocity = tracking.point_velocities_robot_xy[selected].mean(axis=0)
        c, s = np.cos(packet.robot_yaw), np.sin(packet.robot_yaw)
        world_velocity = velocity @ np.array([[c, s], [-s, c]])
        track = TrackedObstacle(key, tuple(world_center), tuple(center), tuple(world_velocity),
            tuple(velocity), int(selected.sum()), candidate["diagnostic"]["observations"], True)
        ids = tracking.point_track_ids.copy()
        ids[selected] = key
        members = {f["raw_track_id"] for f in candidate["diagnostic"]["fragments"]}
        competitors = (tuple(t for t in tracking.tracked_obstacles if t.track_id not in members)
                       if property_memory is not None else ())
        adapted_tracking = replace(tracking, point_track_ids=ids, tracked_obstacles=(track, *competitors))
        candidate_memory = copy.deepcopy(property_memory)
        assessment = worker._assess_track(packet, adapted_tracking, projection, track, True,
            plume, None, image_is_set=True, shared_contact_frame=contact_frame,
            candidate_persistent=candidate_memory,
            refine_contact_boundary=(worker.contact_mask_refinement_config is not None
                and candidate_index < worker.contact_mask_refinement_config.maximum_objects_per_frame),
            proposal_override=candidate["proposal"])
        candidate["diagnostic"]["final_assessment_rejections"] = assessment["reasons"]
        if not assessment["accepted"]:
            continue
        if static_memory is not None:
            from robot.evaluation.static_prop_runtime import enroll_static_group
            enroll_static_group(static_memory, assessment, key, world[selected], packet,
                                worker.persistent_properties.frozen)
        property_memory = candidate_memory
        memory_tracks.append(track)
        # Only the verified exact point set may acquire this object's policy.
        if np.any(assessment["selected"] & ~selected):
            raise RuntimeError("image-object assessment expanded verified ownership")
        policy, factors = evaluate_object_risk(assessment["adapted"], worker.object_factor_prior_config,
            height_evidence=assessment.get("height_evidence"))
        readiness = evaluate_perception_readiness(RestaurantScenario("image_object", "contact",
            "stationary current mask and repeated positive fragments", runtime_factor_evidence_mode(assessment["adapted"]), 0),
            assessment["adapted"])
        detection = assessment["frame"].detections[0]
        d = {"track_id": key, "accepted_instance": True, "camera_observable": True, "visual_attempted": True,
             "centroid_world_xy": tuple(world_center), "velocity_world_xy": tuple(world_velocity),
             "instance_centroid_world_xyz": world[selected].mean(axis=0).tolist(),
             "instance_geometry_scope": "centroid and extent of current observed returns; complete object center/extent not established",
             "instance_mask_pixels": detection.mask_pixel_count,
             "instance_box_xyxy": tuple(getattr(detection.bounding_box, n) for n in ("x_min", "y_min", "x_max", "y_max")),
             "selected_return_count": int(assessment["selected"].sum()), "policy_return_count": int(assessment["selected"].sum()),
             "factor_scores": factors, "geometry_evidence": assessment["geometry_diagnostics"],
             "property_evidence": assessment["persistence"] or {},
             "contact_proxy": assessment.get("contact_proxy"),
             "contact_mask_refinement": assessment.get("contact_mask_refinement"),
             "instance_ownership_basis": "current_image_mask_repeated_fragment_hypothesis",
             "image_object_association": candidate["diagnostic"], "sam_model_quality": candidate["proposal"].model_quality,
             "sam_rejections": [], "sam_candidate_diagnostics": candidate["proposal"].candidate_diagnostics}
        objects.append(ObjectAssessment(assessment["attributes"], assessment["adapted"], policy, readiness, d))
        frames.append(assessment)
        point_ids[assessment["selected"]] = key
        policies[key] = policy
        semantics.append(SemanticObstacle(track, "unknown", detection.confidence, "inferred_image_object_fragments"))
        detail["accepted_object_ids"].append(key)
    if property_memory is not None:
        # No rejected/absent group may borrow an old policy. Unmatched current
        # returns remain in the ordinary raw-track/unknown occupancy pipeline.
        property_memory.finish_epoch(memory_tracks, packet.camera_timestamp_seconds)
        state.property_memory = property_memory
        detail["property_persistence_scope"] = "existing guarded evidence memory on currently verified groups; original clocks, live motion; no partial-support identity transfer"
    if static_memory is not None:
        from robot.evaluation.static_prop_runtime import append_static_props
        append_static_props(worker, static_memory, static_matches, packet, tracking,
            projection, world, objects, frames, policies, point_ids, semantics, plume)
        state.static_properties = static_memory
        detail['static_prop_evidence'] = dict(
            records=list(static_memory.records), current_matches=list(static_matches),
            comparisons=static_audit, configuration=vars(static_config),
            scope='user-approved stationary separated-prop prior; original mass/contact clocks; current LiDAR geometry and motion')
    return state, detail
