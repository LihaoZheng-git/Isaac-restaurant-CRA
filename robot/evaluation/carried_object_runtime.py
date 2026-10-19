"""Transactional opt-in one-person/one-object instance and group adaptation."""

import copy
from dataclasses import asdict, replace

import numpy as np

from robot.evaluation.cra_experiments import RestaurantScenario
from robot.evaluation.perception_readiness import (
    evaluate_perception_readiness,
    runtime_factor_evidence_mode,
)
from robot.perception.carried_object import (
    CarryObservation,
    CarriedRelationHistory,
    propose_carried_object,
    propose_observed_person,
)
from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE
from robot.perception.human_pose import match_person_pose
from robot.risk.carried_group_policy import maximum_member_policy
from robot.risk.object_risk_policy import evaluate_object_risk
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig
from robot.safety.semantic_obstacle import SemanticObstacle


class CarriedRuntimeState:
    def __init__(self, config):
        self.relations = CarriedRelationHistory(config)
        self.trackers = {}
        self.last_times = {}
        self.offsets = {"human": 4_000_000_000, "object": 3_000_000_000}
        self.debug_masks = None


def augment_carried_object(
    worker,
    packet,
    tracking,
    projection,
    world,
    objects,
    frames,
    policies,
    point_ids,
    semantics,
    contact_frame,
):
    """Separate current positive depth partitions; retain all original geometry."""
    from robot.evaluation.multi_object_perception import ObjectAssessment
    from robot.perception.semantic_detection import PixelBoundingBox

    config = worker.carried_object_config
    original = copy.deepcopy(
        getattr(worker, "carried_runtime_state", None)
    ) or CarriedRuntimeState(config)
    detail = {
        "enabled": True,
        "confirmed_groups": [],
        "scope": "one clear adult and one separately segmented object; RGB wrist/positive geometry/co-motion hypothesis",
    }
    person_frame = worker.visual_person_frame(packet)
    people = [
        p
        for p in person_frame.detections
        if p.semantic_class == "person" and p.confidence >= 0.5
    ]
    if len(people) != 1:
        return original, {
            **detail,
            "rejection": "one unambiguous current visual person required",
        }
    pose_frame = worker.pose_detector.detect(
        packet.rgb,
        source_frame_id=packet.camera_frame_id,
        source_timestamp_seconds=packet.camera_timestamp_seconds,
    )
    if (
        pose_frame.source_frame_id != packet.camera_frame_id
        or pose_frame.source_timestamp_seconds != packet.camera_timestamp_seconds
        or pose_frame.image_shape != packet.rgb.shape[:2]
    ):
        raise ValueError("pose and instance source epochs/resolutions must agree")
    pose = match_person_pose(people[0], pose_frame)
    if pose is None:
        return original, {**detail, "rejection": "absent/ambiguous same-person pose"}
    parent, parent_detail = propose_observed_person(
        packet, projection, people[0], pose, worker.segmenter
    )
    detail["person_proposal"] = parent_detail
    if parent is None:
        return original, {
            **detail,
            "rejection": "fresh full-person SAM/depth partition rejected",
        }
    proposed, attempts = propose_carried_object(
        packet, projection, people[0], pose, worker.segmenter, config, crop_is_set=True
    )
    detail["proposals"] = attempts

    def assess(candidate, role, state):
        selected = candidate["selected"]
        xyz = packet.obstacle_xyz[selected]
        now = packet.sensor_timestamp_seconds
        if (
            role not in state.trackers
            or now - state.last_times[role] > config.maximum_gap_seconds
        ):
            state.trackers[role] = ObstacleTracker(
                ObstacleTrackerConfig(
                    minimum_cluster_points=4, maximum_association_distance=0.35
                )
            )
            state.offsets[role] += 1000
        state.last_times[role] = now
        tracked = state.trackers[role].update(
            xyz[:, :2],
            robot_position_world_xy=np.array(packet.robot_position_world_xy),
            robot_yaw=packet.robot_yaw,
            timestamp_seconds=now,
            point_instance_groups=(np.arange(len(xyz)),),
        )
        track = replace(
            tracked.tracked_obstacles[0],
            track_id=state.offsets[role] + tracked.tracked_obstacles[0].track_id,
        )
        ids = tracking.point_track_ids.copy()
        ids[selected] = track.track_id
        velocities = tracking.point_velocities_robot_xy.copy()
        velocities[selected] = tracked.point_velocities_robot_xy
        adapted_tracking = replace(
            tracking,
            point_track_ids=ids,
            point_velocities_robot_xy=velocities,
            tracked_obstacles=(track,),
        )
        assessment = worker._assess_track(
            packet,
            adapted_tracking,
            projection,
            track,
            True,
            copy.deepcopy(worker.plume),
            None,
            image_is_set=True,
            shared_contact_frame=contact_frame,
            proposal_override=candidate["proposal"],
        )
        return track, assessment

    state = copy.deepcopy(original)
    child = prop_track = None
    body_candidate = parent
    if proposed is not None:
        prop_track, child = assess(proposed, "object", state)
        if child["accepted"]:
            mask = (
                parent["proposal"].detection.mask & ~proposed["proposal"].detection.mask
            )
            yy, xx = np.where(mask)
            selected = parent["selected"].copy()
            pix = projection.pixel_indices_xy
            selected[projection.source_indices[~mask[pix[:, 1], pix[:, 0]]]] = False
            if len(xx) >= 200 and selected.sum() >= 8:
                detection = replace(
                    parent["proposal"].detection,
                    mask=mask,
                    bounding_box=PixelBoundingBox(
                        float(xx.min()),
                        float(yy.min()),
                        float(xx.max() + 1),
                        float(yy.max() + 1),
                    ),
                )
                body_candidate = {
                    "proposal": replace(parent["proposal"], detection=detection),
                    "selected": selected,
                }
            else:
                child = None
        else:
            child = None
    human_track, body = assess(body_candidate, "human", state)
    if (
        not body["accepted"]
        or body["attributes"].mass.mass_kg.source != HUMAN_MASS_PRIOR_SOURCE
    ):
        state = copy.deepcopy(original)
        child = None
        human_track, body = assess(parent, "human", state)
    if (
        not body["accepted"]
        or body["attributes"].mass.mass_kg.source != HUMAN_MASS_PRIOR_SOURCE
    ):
        return original, {
            **detail,
            "rejection": "person association/mass-prior gate failed after refinement",
        }
    if child is None:
        state.trackers.pop("object", None)
        state.last_times.pop("object", None)
        detail["rejection"] = (
            "no independently accepted separate object; person-only assessment"
        )
    removed = [
        i
        for i, o in enumerate(objects)
        if o.attributes.mass.mass_kg.source == HUMAN_MASS_PRIOR_SOURCE
    ]
    ownership = body["selected"] | (child["selected"] if child is not None else False)
    if any(
        np.any(ownership & f["selected"])
        for i, f in enumerate(frames)
        if i not in removed and f["accepted"]
    ):
        return original, {
            **detail,
            "rejection": "visual instances conflict with another accepted object",
        }
    for i in reversed(removed):
        point_ids[frames[i]["selected"]] = (
            objects[i].attributes.track_id + 1_000_000_000
        )
        objects.pop(i)
        frames.pop(i)

    members = []
    for role, track, assessment in (
        ("human", human_track, body),
        ("object", prop_track, child),
    ):
        if assessment is None:
            continue
        policy, factors = evaluate_object_risk(
            assessment["adapted"],
            worker.object_factor_prior_config,
            height_evidence=assessment.get("height_evidence"),
        )
        readiness = evaluate_perception_readiness(
            RestaurantScenario(
                "carried_member",
                "potential",
                "fresh separate visual member",
                runtime_factor_evidence_mode(assessment["adapted"]),
                0,
            ),
            assessment["adapted"],
        )
        xyz_world = world[assessment["selected"]].mean(axis=0)
        d = {
            "track_id": track.track_id,
            "accepted_instance": True,
            "carried_member": role,
            "selected_return_count": int(assessment["selected"].sum()),
            "policy_return_count": int(assessment["selected"].sum()),
            "factor_scores": factors,
            "geometry_evidence": assessment["geometry_diagnostics"],
            "property_evidence": {},
            "instance_ownership_basis": "fresh_RGB_depth_partition_cropped_SAM_exact_mask",
            "instance_centroid_world_xyz": xyz_world.tolist(),
            "centroid_world_xy": track.centroid_world_xy,
            "velocity_world_xy": track.velocity_world_xy,
            "source_frame_id": packet.camera_frame_id,
            "source_time": packet.camera_timestamp_seconds,
            "sam_model_quality": assessment["sam_quality"],
            "motion_basis": "separate accepted-instance LiDAR centroid history; no other member velocity copied",
            "mask_refinement": "object subtracted from human mask"
            if role == "human" and child is not None
            else None,
        }
        members.append(
            ObjectAssessment(
                assessment["attributes"], assessment["adapted"], policy, readiness, d
            )
        )
        objects.append(members[-1])
        frames.append(assessment)
        point_ids[assessment["selected"]] = track.track_id
        policies[track.track_id] = policy
        semantics.append(
            SemanticObstacle(
                track,
                "unknown",
                assessment["base"].semantic_confidence,
                "fresh_RGB_depth_partition_SAM_instance",
            )
        )
    state.debug_masks = {
        "source_frame_id": packet.camera_frame_id,
        "human": body["frame"].detections[0].mask,
        "object": None if child is None else child["frame"].detections[0].mask,
    }
    if child is None:
        return state, detail
    human, prop = members
    hxyz = np.asarray(human.diagnostics["instance_centroid_world_xyz"])
    oxyz = np.asarray(prop.diagnostics["instance_centroid_world_xyz"])
    hand_distance = float(
        np.linalg.norm(
            packet.obstacle_xyz[child["selected"]].mean(axis=0)
            - proposed["hand_robot_xyz"]
        )
    )
    relation = state.relations.update(
        CarryObservation(
            human_track.track_id,
            prop_track.track_id,
            packet.camera_frame_id,
            packet.sensor_timestamp_seconds,
            tuple(hxyz),
            tuple(oxyz),
            hand_distance,
            proposed["wrist_confidence"],
        )
    )
    detail["relationship"] = relation
    detail["members"] = {
        "human": human_track.track_id,
        "object": prop_track.track_id,
        "human_mass": asdict(human.attributes.mass.mass_kg),
        "object_mass": asdict(prop.attributes.mass.mass_kg),
        "hand_distance_m": hand_distance,
        "proposal_basis": proposed.get("basis", "non-person wrist-near component"),
    }
    if relation["confirmed"]:
        shared, group = maximum_member_policy(
            {human_track.track_id: human.policy, prop_track.track_id: prop.policy}
        )
        policies[human_track.track_id] = policies[prop_track.track_id] = shared
        detail["confirmed_groups"].append(group)
        human.diagnostics["group_policy"] = prop.diagnostics["group_policy"] = group
    return state, detail
