from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot.evaluation.live_object_perception import (
    LiveObjectPacket,
    LiveObjectPerception,
)
from robot.perception.camera_lidar_fusion import FThetaCameraCalibration
from robot.perception.physical_attributes import EstimateStatus
from robot.safety.class_safety_policy import ClassSafetyParameters, ClassSafetyPolicy
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.safe_sage_configuration_space import (
    CircularRobotFootprint,
    SafeSAGEConfigurationSpaceConfig,
)
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig
from robot.safety.safe_sage_guidance_policy import (
    SafeSAGEGuidanceParameters,
    SafeSAGEGuidancePolicy,
)
from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
)


def packet(frame=1, t=1.0, x=0.0):
    world = np.array(
        [[1.0, -0.04, 0.2], [1.0, 0.04, 0.2], [1.04, -0.04, 0.3], [1.04, 0.04, 0.3]]
    )
    points = world - np.array([x, 0, 0])
    return LiveObjectPacket(
        frame,
        t,
        frame,
        t + 0.1,
        np.zeros((100, 100, 3), np.uint8),
        points,
        points,
        (x, 0),
        0.0,
        (0.2, 0),
        np.eye(4),
        0.0,
        (2.5, 0),
    )


def worker():
    policy = ClassSafetyPolicy(
        {"unknown": ClassSafetyParameters(risk=1, boundary_flux_magnitude=4)}
    )
    guidance = SafeSAGEGuidancePolicy(
        {
            "unknown": SafeSAGEGuidanceParameters(
                safety_parameters=policy.parameters_for("unknown"),
                tangential_strength=1.25,
                passing_side="right",
            )
        }
    )
    return LiveObjectPerception(
        segmenter=SimpleNamespace(),
        mass_estimator=None,
        contact_detector=None,
        calibration=FThetaCameraCalibration(
            100, 100, 50.0, 50.0, 180.0, (0.0, 0.01, 0.0, 0.0, 0.0), 100, 100
        ),
        class_policy=policy,
        guidance_policy=guidance,
        grid_config=LocalOccupancyGridConfig(
            resolution=0.5, minimum_x=-3, maximum_x=4, minimum_y=-3, maximum_y=3
        ),
        field_config=SafeSAGEFieldPipelineConfig(
            configuration_space_config=SafeSAGEConfigurationSpaceConfig(
                footprint=CircularRobotFootprint(radius=0.25), yaw_slice_count=4
            ),
            h_full_config=SafeSAGEHFullConfig(yaw_slice_count=4),
            guidance_field_config=TangentialLaplaceGuidanceFieldConfig(
                tangential_buffer_radius=1.0
            ),
        ),
        anchor_position_world_xy=(0, 0),
        anchor_yaw=0,
    )


def test_missing_visual_instance_retains_all_returns_and_conservative_policy():
    result = worker()(packet())
    assert result.diagnostics["accepted_instance"] is False
    assert result.diagnostics["selected_return_count"] == 0
    assert result.diagnostics["unassessed_return_count"] == 4
    assert result.policy.risk == 1.0
    assert result.field.all_obstacle_point_count == 4
    assert result.diagnostics["unknown_factors"] == [
        "potential",
        "kinetic",
        "contact",
        "thermal",
    ]
    assert not result.readiness.ready
    assert (
        np.count_nonzero(
            result.field.pipeline.configuration_space.occupancy_grids[0].occupied_mask
        )
        > 0
    )


def test_static_world_track_stays_static_during_ego_translation_and_keeps_scan_epoch():
    runtime = worker()
    runtime(packet())
    result = runtime(packet(2, 1.1, 0.02))
    assert result.attributes.motion.velocity_x_mps.value == pytest.approx(0, abs=1e-12)
    assert result.attributes.motion.relative_speed_mps.value == pytest.approx(0.2)
    assert result.field.sensor_timestamp_seconds == 1.1
    assert result.attributes.motion.relative_speed_mps.timestamp_seconds == 1.1
    assert result.attributes.timestamp_seconds == pytest.approx(1.2)
    assert any(
        "oracle" in item
        for item in result.attributes.motion.relative_speed_mps.evidence
    )
    assert result.field.temporal_safety_derivative.time_delta_seconds == pytest.approx(
        0.1
    )


def test_failed_field_transaction_does_not_advance_track_or_epoch():
    runtime = worker()
    accepted = runtime(packet())
    with pytest.raises(ValueError):
        runtime(packet(2, 1.5))
    assert runtime.previous is accepted.field
    recovered = runtime(packet(3, 1.1))
    assert recovered.field.tracking_result.tracked_obstacles[0].observation_count == 2
    with pytest.raises(ValueError, match="increase"):
        runtime(packet(4, 1.1))


def test_live_packet_copies_data_and_rejects_unavailable_or_stale_references():
    value = packet()
    assert not value.obstacle_xyz.flags.writeable
    for changes in (
        {"camera_timestamp_seconds": 1.2},
        {"robot_yaw": float("nan")},
        {"pose_status": EstimateStatus.UNAVAILABLE},
        {"measured_robot_velocity_world_xy": (float("inf"), 0)},
    ):
        with pytest.raises(ValueError):
            replace(value, **changes)


def test_scan_interval_union_keeps_observed_placements_without_dropping_returns():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 0.5
    value = replace(
        packet(),
        scan_pose_samples_world=poses,
        scan_interval_scope="synthetic static scan-interval prior",
    )
    result = worker()(value)
    assert result.field.all_obstacle_point_count == 4
    assert result.diagnostics["unassessed_return_count"] == 4
    assert result.diagnostics["scan_pose_sample_count"] == 2
    assert result.policy.risk == 1
    assert result.diagnostics["field_raw_placement_count"] == 8
    assert result.diagnostics["field_cell_representative_count"] == 4
    assert (
        result.field.pipeline.configuration_space.occupancy_grids[0].input_point_count
        == 4
    )
    assert not value.scan_pose_samples_world.flags.writeable
    with pytest.raises(ValueError, match="explicit"):
        replace(packet(), scan_pose_samples_world=poses)


def test_static_memory_retains_older_unseen_geometry_as_prior_without_retiming_it():
    runtime = worker()
    runtime.static_scope = "synthetic static scene occupancy prior"
    runtime.field_config = replace(
        runtime.field_config,
        guidance_field_config=replace(
            runtime.field_config.guidance_field_config,
            allow_dense_scene_arbitration=True,
        ),
    )
    runtime(packet())
    for index in range(1, 9):
        value = packet(index + 1, 1 + index * 0.1)
        value = replace(value, obstacle_xyz=value.obstacle_xyz + [1, 0, 0])
        latest = runtime(value)
    assert latest.field.sensor_timestamp_seconds == pytest.approx(1.8)
    assert latest.diagnostics["static_oldest_source_time"] == 1.0
    assert 0 in latest.diagnostics["static_cached_track_ids"]
    grid = latest.field.pipeline.configuration_space.occupancy_grids[0]
    assert grid.occupied_mask[6, 8]
    assert latest.policy.risk == 1.0
    assert not latest.readiness.ready


def seed_observed_properties(runtime, result):
    from robot.perception.physical_attributes import ScalarEstimate
    from robot.perception.temporal_object_evidence import TemporalObjectEvidence

    runtime.evidence_memory = TemporalObjectEvidence()
    track = result.field.tracking_result.tracked_obstacles[0]
    runtime.evidence_memory.resolve(
        track=track,
        tracked_obstacles=(track,),
        sensor_timestamp_seconds=1.0,
        timestamp_seconds=1.1,
        camera_frame_id=1,
        ego_position_world_xy=(0, 0),
        fresh_mass=ScalarEstimate(
            4, 3, 5, 0.25, "kg", "controlled_mass", 1.1, EstimateStatus.INFERRED
        ),
        fresh_contact=ScalarEstimate(
            0.2,
            0.1,
            0.3,
            0.7,
            "1",
            "visual_silhouette_contact_proxy",
            1.1,
            EstimateStatus.INFERRED,
        ),
    )


def test_missing_current_instance_borrows_only_stable_properties_with_fresh_motion():
    runtime = worker()
    first = runtime(packet())
    seed_observed_properties(runtime, first)
    latest = runtime(packet(2, 1.1, 0.02))
    assert not latest.diagnostics["accepted_instance"]
    assert latest.attributes.mass.mass_kg.value == 4
    assert latest.attributes.mass.mass_kg.timestamp_seconds == 1.1
    assert latest.attributes.mass.mass_kg.status is EstimateStatus.PRIOR
    assert latest.adapted.contact.basis == "visual_silhouette_geometry_proxy"
    assert latest.adapted.contact.inputs[0][1].timestamp_seconds == 1.1
    assert latest.attributes.motion.relative_speed_mps.timestamp_seconds == 1.1
    assert latest.attributes.motion.relative_speed_mps.value == pytest.approx(0.2)
    assert not latest.adapted.kinetic.rejection_reasons
    assert not latest.adapted.contact.rejection_reasons
    assert latest.diagnostics["unknown_factors"] == ["potential", "thermal"]
    assert latest.policy.risk == 1 and not latest.readiness.ready
    assert latest.field.all_obstacle_point_count == 4
    assert latest.diagnostics["unassessed_return_count"] == 4
    assert not latest.attributes.contact.sharpness.is_available


def test_failed_field_transaction_does_not_advance_retained_evidence_identity():
    runtime = worker()
    first = runtime(packet())
    seed_observed_properties(runtime, first)
    sources = runtime.evidence_memory.sources.copy()
    with pytest.raises(ValueError):
        runtime(packet(2, 1.5))
    assert runtime.evidence_memory.sources == sources
    latest = runtime(packet(3, 1.1))
    assert latest.attributes.mass.mass_kg.is_available
    assert latest.diagnostics["temporal_object_evidence"]["mass"]["mode"] == "retained"
