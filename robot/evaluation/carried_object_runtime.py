"""Transactional opt-in one-person/one-object instance and group adaptation."""

import copy
from dataclasses import asdict, replace

import numpy as np

from robot.evaluation.cra_experiments import RestaurantScenario
from robot.evaluation.perception_readiness import evaluate_perception_readiness, runtime_factor_evidence_mode
from robot.perception.carried_object import CarryObservation, CarriedRelationHistory, propose_carried_object
from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE
from robot.perception.human_pose import match_person_pose
from robot.risk.carried_group_policy import maximum_member_policy
from robot.risk.object_risk_policy import evaluate_object_risk
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig
from robot.safety.semantic_obstacle import SemanticObstacle


class CarriedRuntimeState:
    def __init__(self,config):
        self.relations=CarriedRelationHistory(config)
        self.tracker=None
        self.human_id=None
        self.last_time=None
        self.offset=3_000_000_000


def augment_carried_object(worker,packet,tracking,projection,world,objects,frames,policies,point_ids,semantics,contact_frame):
    """Never transfer physical factors between members; return candidate state."""
    from robot.evaluation.multi_object_perception import ObjectAssessment

    config=worker.carried_object_config
    state=copy.deepcopy(getattr(worker,"carried_runtime_state",None)) or CarriedRuntimeState(config)
    detail={"enabled":True,"confirmed_groups":[],"scope":"one clear adult and one separately segmented object; RGB wrist/positive geometry/co-motion hypothesis"}
    humans=[i for i,o in enumerate(objects) if o.diagnostics["accepted_instance"]
        and o.attributes.mass.mass_kg.source==HUMAN_MASS_PRIOR_SOURCE
        and o.attributes.mass.mass_kg.timestamp_seconds==packet.camera_timestamp_seconds]
    if len(humans)!=1:
        detail["rejection"]="one fresh separately owned human instance required"
        return state,detail
    hi=humans[0]
    human=objects[hi]
    person_frame=worker.visual_person_frame(packet)
    people=[p for p in person_frame.detections if p.semantic_class=="person" and p.confidence>=.5]
    if len(people)!=1:
        detail["rejection"]="multiple/unresolved visual people"
        return state,detail
    pose_frame=worker.pose_detector.detect(packet.rgb,source_frame_id=packet.camera_frame_id,
        source_timestamp_seconds=packet.camera_timestamp_seconds)
    if (pose_frame.source_frame_id!=packet.camera_frame_id
        or pose_frame.source_timestamp_seconds!=packet.camera_timestamp_seconds
        or pose_frame.image_shape!=packet.rgb.shape[:2]):
        raise ValueError("RGB pose must share the actual instance source epoch and resolution")
    pose=match_person_pose(people[0],pose_frame)
    if pose is None:
        detail["rejection"]="absent/ambiguous same-person RGB pose"
        return state,detail
    proposed,attempts=propose_carried_object(packet,projection,people[0],pose,worker.segmenter,config)
    detail["proposals"]=attempts
    if proposed is None:
        detail["rejection"]="no distinct accepted carried-object mask"
        return state,detail
    selected=proposed["selected"]
    if any(np.any(selected & frame["selected"]) for i,frame in enumerate(frames) if i!=hi):
        detail["rejection"]="candidate conflicts with another accepted instance"
        return state,detail
    now=packet.sensor_timestamp_seconds
    original_state=copy.deepcopy(state)
    if state.tracker is None or state.human_id!=human.attributes.track_id or now-state.last_time>config.maximum_gap_seconds:
        state.tracker=ObstacleTracker(ObstacleTrackerConfig(minimum_cluster_points=4,maximum_association_distance=.35))
        state.offset+=1000
    state.human_id,state.last_time=human.attributes.track_id,now
    xyz=packet.obstacle_xyz[selected]
    tracked=state.tracker.update(xyz[:,:2],robot_position_world_xy=np.array(packet.robot_position_world_xy),
        robot_yaw=packet.robot_yaw,timestamp_seconds=now,
        point_instance_groups=(np.arange(len(xyz)),))
    prop=replace(tracked.tracked_obstacles[0],track_id=state.offset+tracked.tracked_obstacles[0].track_id)
    ids=tracking.point_track_ids.copy()
    ids[selected]=prop.track_id
    velocities=tracking.point_velocities_robot_xy.copy()
    velocities[selected]=tracked.point_velocities_robot_xy
    prop_tracking=replace(tracking,point_track_ids=ids,point_velocities_robot_xy=velocities,
        tracked_obstacles=(prop,))
    # The child is evaluated through the exact same mass, geometry, contact,
    # plume and physical-factor adapters, with only its own positive returns.
    child=worker._assess_track(packet,prop_tracking,projection,prop,True,copy.deepcopy(worker.plume),None,
        image_is_set=True,shared_contact_frame=contact_frame,proposal_override=proposed["proposal"])
    if not child["accepted"]:
        detail["rejection"]="separate object association failed downstream exact-mask gates"
        return original_state,detail
    original=frames[hi]
    body_mask=original["frame"].detections[0].mask & ~proposed["proposal"].detection.mask
    if body_mask.sum()<200:
        detail["rejection"]="no resolved human mask remains after object separation"
        return original_state,detail
    yy,xx=np.where(body_mask)
    from robot.perception.semantic_detection import PixelBoundingBox
    body_detection=replace(original["frame"].detections[0],mask=body_mask,
        bounding_box=PixelBoundingBox(float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)))
    body_proposal=replace(proposed["proposal"],detection=body_detection,
        model_quality=original["sam_quality"],prompt_xy=None,
        candidate_diagnostics=tuple(original["sam_candidates"]))
    body_track=next(t for t in tracking.tracked_obstacles if t.track_id==human.attributes.track_id)
    body_tracking=replace(tracking,point_track_ids=ids)
    body=worker._assess_track(packet,body_tracking,projection,body_track,True,copy.deepcopy(worker.plume),None,
        image_is_set=True,shared_contact_frame=contact_frame,proposal_override=body_proposal)
    if not body["accepted"] or body["attributes"].mass.mass_kg.source!=HUMAN_MASS_PRIOR_SOURCE:
        detail["rejection"]="human ownership/visual prior failed after separate-mask refinement"
        return original_state,detail

    def evaluated(assessment):
        policy,factors=evaluate_object_risk(assessment["adapted"],worker.object_factor_prior_config,
            height_evidence=assessment.get("height_evidence"))
        readiness=evaluate_perception_readiness(RestaurantScenario("carried_member","potential",
            "fresh separate visual member",runtime_factor_evidence_mode(assessment["adapted"]),0),assessment["adapted"])
        return policy,factors,readiness
    hp,hf,hr=evaluated(body)
    op,of,oready=evaluated(child)
    # Keep discarded boundary/hand returns occupied with their existing prior.
    point_ids[original["selected"] & ~body["selected"]]=human.attributes.track_id+1_000_000_000
    point_ids[body["selected"]]=human.attributes.track_id
    point_ids[child["selected"]]=prop.track_id
    hd={**human.diagnostics,"factor_scores":hf,"geometry_evidence":body["geometry_diagnostics"],
        "selected_return_count":int(body["selected"].sum()),"policy_return_count":int(body["selected"].sum()),
        "carried_member":"human","mask_refinement":"original accepted human SAM mask minus separately accepted object mask; no occupancy removed"}
    objects[hi]=ObjectAssessment(body["attributes"],body["adapted"],hp,hr,hd)
    frames[hi]=body
    od={"track_id":prop.track_id,"accepted_instance":True,"carried_member":"object",
        "selected_return_count":int(child["selected"].sum()),"policy_return_count":int(child["selected"].sum()),
        "factor_scores":of,"geometry_evidence":child["geometry_diagnostics"],
        "instance_ownership_basis":"fresh_wrist_seeded_crop_SAM_exact_mask",
        "centroid_world_xy":prop.centroid_world_xy,"velocity_world_xy":prop.velocity_world_xy,
        "source_frame_id":packet.camera_frame_id,"source_time":packet.camera_timestamp_seconds,
        "sam_model_quality":proposed["proposal"].model_quality,"wrist_index":proposed["wrist_index"],
        "person_mask_overlap":proposed["person_mask_overlap"],
        "motion_basis":"separate accepted-instance LiDAR centroid tracker; no human velocity copied"}
    objects.append(ObjectAssessment(child["attributes"],child["adapted"],op,oready,od))
    frames.append(child)
    policies[human.attributes.track_id],policies[prop.track_id]=hp,op
    semantics.append(SemanticObstacle(prop,"unknown",child["base"].semantic_confidence,"fresh_wrist_seeded_SAM_instance"))
    hxyz=world[body["selected"]].mean(axis=0)
    oxyz=world[child["selected"]].mean(axis=0)
    hd["instance_centroid_world_xyz"]=hxyz.tolist()
    od["instance_centroid_world_xyz"]=oxyz.tolist()
    hand_distance=float(np.linalg.norm(packet.obstacle_xyz[child["selected"]].mean(axis=0)-proposed["hand_robot_xyz"]))
    relation=state.relations.update(CarryObservation(human.attributes.track_id,prop.track_id,
        packet.camera_frame_id,packet.sensor_timestamp_seconds,tuple(hxyz),tuple(oxyz),
        hand_distance,proposed["wrist_confidence"]))
    detail["relationship"]=relation
    detail["members"]={"human":human.attributes.track_id,"object":prop.track_id,
        "human_mass":asdict(body["attributes"].mass.mass_kg),"object_mass":asdict(child["attributes"].mass.mass_kg),
        "hand_distance_m":hand_distance}
    if relation["confirmed"]:
        shared,group=maximum_member_policy({human.attributes.track_id:hp,prop.track_id:op})
        policies[human.attributes.track_id]=policies[prop.track_id]=shared
        detail["confirmed_groups"].append(group)
        hd["group_policy"]=od["group_policy"]=group
    return state,detail
