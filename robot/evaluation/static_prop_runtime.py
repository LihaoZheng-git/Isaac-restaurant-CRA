"""Apply retained demo-prop evidence to uniquely supported current returns."""

from dataclasses import asdict, replace

import numpy as np

from robot.evaluation.cra_experiments import RestaurantScenario, EvidenceMode
from robot.evaluation.perception_readiness import evaluate_perception_readiness
from robot.perception.physical_attributes import ScalarEstimate
from robot.perception.visual_shape_contact import SHAPE_CONTACT_BASIS
from robot.risk.object_risk_policy import evaluate_object_risk
from robot.risk.physical_factor_adapter import adapt_physical_factors, PhysicalFactorAdapterConfig
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.semantic_obstacle import SemanticObstacle


def enroll_static_group(memory, assessment, key, world_points, packet, frozen, *, refresh_proof=None):
    proxy = assessment.get('contact_proxy')
    if proxy is not None:
        if refresh_proof is not None and key in memory.records:
            return memory.refresh(key, world_points, assessment['attributes'].mass.mass_kg,
                ScalarEstimate(**proxy), frame_id=packet.camera_frame_id,
                source_time=packet.camera_timestamp_seconds,
                robot_xy=packet.robot_position_world_xy, proof=refresh_proof)
        return memory.enroll(key, world_points, assessment['attributes'].mass.mass_kg,
            ScalarEstimate(**proxy), frame_id=packet.camera_frame_id,
            source_time=packet.camera_timestamp_seconds,
            robot_xy=packet.robot_position_world_xy, frozen=frozen)
    return False


def append_static_props(worker, memory, matches, packet, tracking, projection,
                        world, objects, frames, policies, point_ids, semantics, plume):
    from robot.evaluation.multi_object_perception import ObjectAssessment

    for key, (selected, proof) in matches.items():
        proof = {**proof, 'current_lidar_timestamp':packet.sensor_timestamp_seconds,
                 'current_camera_timestamp':packet.camera_timestamp_seconds}
        values, evidence = memory.retained(key, packet.camera_timestamp_seconds)
        center = packet.obstacle_xyz[selected, :2].mean(axis=0)
        world_center = world[selected, :2].mean(axis=0)
        velocity = tracking.point_velocities_robot_xy[selected].mean(axis=0)
        c, s = np.cos(packet.robot_yaw), np.sin(packet.robot_yaw)
        world_velocity = velocity @ np.array([[c,s],[-s,c]])
        members = {f['raw_track_id'] for f in proof['fragments']}
        raw_tracks = [t for t in tracking.tracked_obstacles if t.track_id in members]
        count = min(t.observation_count for t in raw_tracks)
        matched = all(t.matched_previous_track for t in raw_tracks)
        track = TrackedObstacle(key, tuple(world_center), tuple(center), tuple(world_velocity),
            tuple(velocity), int(selected.sum()), count, matched)
        ids = tracking.point_track_ids.copy()
        ids[selected] = key
        adapted_tracking = replace(tracking, point_track_ids=ids, tracked_obstacles=(track,))
        raw = worker._assess_track(packet, adapted_tracking, projection, track, False,
                                  plume, None)
        attributes = replace(raw['attributes'],
            mass=replace(raw['attributes'].mass, mass_kg=values['mass']))
        adapted = adapt_physical_factors(attributes,
            timestamp_seconds=packet.camera_timestamp_seconds,
            visual_contact_proxy=values['contact'], visual_contact_basis=SHAPE_CONTACT_BASIS,
            contact_interaction_likelihood=attributes.motion.route_crossing_likelihood,
            config=PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True,
                maximum_persistent_property_age_seconds=memory.property_config.maximum_property_age_seconds))
        policy, factors = evaluate_object_risk(adapted, worker.object_factor_prior_config)
        readiness = evaluate_perception_readiness(RestaurantScenario('static_prop','contact',
            'current LiDAR with retained static-location evidence', EvidenceMode.PRIOR, 0), adapted)
        d = dict(track_id=key, accepted_instance=False, static_prop_policy=True,
            camera_observable=False, visual_attempted=False,
            centroid_world_xy=tuple(world_center), velocity_world_xy=tuple(world_velocity),
            instance_centroid_world_xyz=world[selected].mean(axis=0).tolist(),
            instance_geometry_scope='current observed surface; stored region is a static association prior',
            selected_return_count=int(selected.sum()), policy_return_count=int(selected.sum()),
            factor_scores=factors, property_evidence=evidence,
            contact_proxy=asdict(values['contact']), geometry_evidence=raw['geometry_diagnostics'],
            static_prop_association=proof, instance_ownership_basis='static_prop_location_prior',
            presentation_basis='retained mass/contact + current LiDAR; not a fresh segmentation')
        raw.update(attributes=attributes, adapted=adapted, selected=selected,
            static_prop_policy=True,
            static_prop_display_pixels=projection.pixel_indices_xy[selected[projection.source_indices]].copy())
        objects.append(ObjectAssessment(attributes, adapted, policy, readiness, d))
        frames.append(raw)
        point_ids[selected] = key
        policies[key] = policy
        semantics.append(SemanticObstacle(track, 'unknown', 0., 'static_prop_location_prior'))
