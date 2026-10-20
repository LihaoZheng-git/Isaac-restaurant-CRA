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


def worker(**overrides):
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
        **overrides,
    )


@pytest.mark.parametrize("radius,expected_clusters", [(.12, 2), (.18, 1)])
def test_explicit_3d_connectivity_changes_grouping_without_discarding_returns(radius, expected_clusters):
    runtime = worker(cluster_in_3d=True, cluster_radius_m=radius)
    points = np.array([[1.+dx, 0., .5] for dx in (0.,.01,.02,.03,.17,.18,.19,.20)])
    before = points.copy()
    result = runtime.tracker.update(points[:,:2], robot_position_world_xy=np.zeros(2),
        robot_yaw=0., timestamp_seconds=1., clustering_points_robot_xyz=points)
    assert len(result.tracked_obstacles) == expected_clusters
    assert sum(t.point_count for t in result.tracked_obstacles) == len(points)
    assert np.all(result.point_track_ids >= 0)
    np.testing.assert_array_equal(points, before)


def test_clustering_override_preserves_existing_default_radii():
    assert worker().tracker.config.cluster_radius == .20
    assert worker(cluster_in_3d=True).tracker.config.cluster_radius == .12


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


def test_opt_in_height_never_promotes_unresolved_cluster_to_owned_object():
    from robot.perception.object_height import ObjectHeightConfig

    result = worker(object_height_config=ObjectHeightConfig())(packet())
    evidence = result.diagnostics["geometry_evidence"]
    assert evidence["height_ownership_basis"].startswith("unresolved_lidar_cluster")
    assert "object_height" not in evidence
    assert not result.attributes.geometry.fall_height_m.is_available
    assert not result.attributes.stability.support_state.is_available
    assert result.field.all_obstacle_point_count == 4
    assert result.policy.risk == 1.


def test_general_height_cannot_silently_override_legacy_sphere_support_mode():
    from robot.perception.object_height import ObjectHeightConfig

    with pytest.raises(ValueError, match="selected separately"):
        worker(object_height_config=ObjectHeightConfig(),ground_prior_scope="legacy sphere fixture")


def test_accepted_category_free_instance_uses_general_support_without_sphere_fit():
    from robot.perception.object_height import ObjectHeightConfig
    from robot.perception.physical_attributes import MassAttributes, ScalarEstimate
    from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection

    obj = np.array([(1+x,y,z) for x in (-.08,0,.08)
        for y in (-.08,0,.08) for z in (.02,.18,.34)])
    floor = np.array([(x,y,0.) for x in np.linspace(-.2,2.2,41)
        for y in np.linspace(-1.2,1.2,41)])
    transform = np.eye(4)
    transform[:3,:3] = [[0,-1,0],[0,0,1],[-1,0,0]]
    value = replace(packet(), scene_xyz=np.vstack((floor,obj)), obstacle_xyz=obj,
        camera_from_scan_robot=transform)
    mask = np.zeros((100,100),bool)
    mask[10:90,10:90] = True
    detection = SemanticDetection(0,"unknown",.5,PixelBoundingBox(10,10,90,90),mask)
    runtime = worker(object_height_config=ObjectHeightConfig(),
        fall_interaction_prior_scope="unit test explicitly unmeasured interaction")
    runtime.segmenter = SimpleNamespace(set_image=lambda rgb: None,
        propose=lambda *a,**kw: SimpleNamespace(detection=detection, model_quality=.95,
            rejection_reasons=(), candidate_diagnostics=(), prompt_xy=(50,40),
            inference_wall_seconds=0., image_encoder_wall_seconds=0.))
    runtime.mass_estimator = SimpleNamespace(estimate_instance=lambda *a,**kw:
        MassAttributes(ScalarEstimate(2.,1.,3.,.5,"kg","controlled_test_mass",1.,EstimateStatus.INFERRED)))
    result = runtime(value)
    assert result.diagnostics["accepted_instance"]
    assert result.attributes.stability.support_state.value == "ground_supported"
    assert result.attributes.geometry.fall_height_m.value == 0.
    assert result.attributes.geometry.center_height_m.value > .1
    assert not result.adapted.potential.rejection_reasons
    assert result.adapted.potential.energy_j.value == 0.
    assert result.diagnostics["geometry_evidence"]["height_ownership_basis"] == "fresh_accepted_instance"
    assert result.field.all_obstacle_point_count == len(obj)


def test_explicit_human_identity_guard_never_calls_catalog_mass_network():
    from functools import partial
    from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection
    obj = np.array([(1+x,y,z) for x in (-.08,0,.08)
        for y in (-.08,0,.08) for z in (.02,.18,.34)])
    transform = np.eye(4)
    transform[:3,:3] = [[0,-1,0],[0,0,1],[-1,0,0]]
    value = replace(packet(), obstacle_xyz=obj, scene_xyz=obj,
        camera_from_scan_robot=transform)
    mask = np.zeros((100,100),bool)
    mask[10:90,10:90] = True
    detection = SemanticDetection(0,"unknown",.5,PixelBoundingBox(10,10,90,90),mask)
    runtime = worker()
    runtime.segmenter = SimpleNamespace(set_image=lambda rgb: None,
        propose=lambda *a,**kw: SimpleNamespace(detection=detection,model_quality=.95,
            rejection_reasons=(),candidate_diagnostics=(),prompt_xy=(50,40),
            inference_wall_seconds=0.,image_encoder_wall_seconds=0.))
    def forbidden(*a,**kw):
        raise AssertionError('catalog model cannot estimate a retained person identity')
    runtime.mass_estimator=SimpleNamespace(estimate_instance=forbidden)
    runtime._assess_track=partial(runtime._assess_track,suppress_catalog_mass_for_human_identity=True)
    result=runtime(value)
    assert result.diagnostics['accepted_instance']
    assert not result.attributes.mass.mass_kg.is_available
    assert result.attributes.mass.mass_kg.source=='catalog_human_domain_guard'
