from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from robot.evaluation.cra_experiments import CRABaseline
from robot.evaluation.restaurant_baselines import (
    applied_field_policies, crossing_due, geometric_cbf_command, validate_baseline,
)
from robot.risk.object_risk_policy import ObjectRiskPriorConfig, evaluate_object_risk
from robot.risk.physical_factor_adapter import adapt_physical_factors
from robot.perception.physical_attributes import PhysicalAttributes
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidancePolicy
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig


def policy():
    return evaluate_object_risk(adapt_physical_factors(PhysicalAttributes.unknown(
        track_id=7, timestamp_seconds=1., semantic_class="unknown",
    ), timestamp_seconds=1.), ObjectRiskPriorConfig())[0]


def test_fixed_class_retains_unknown_sam_and_physical_assessments():
    cra = policy()
    classes = SafeSAGEGuidancePolicy.from_json("configs/safe_sage_paper_baseline.json")
    obj = NS(attributes=NS(track_id=7), diagnostics={"accepted_instance": True})
    guidance, audit = applied_field_policies("fixed_class_safe_sage", {7: cra}, [obj], classes)
    assert guidance[7].risk == 1
    assert audit[0]["observed_class"] == "unknown"
    assert cra.risk == .68359375
    assert audit[0]["cra_phi"] == cra.risk
    assert guidance[7].tangential_strength == 1.25
    obj.diagnostics["geometry_evidence"] = {"human_mass_prior": {"value": 70, "status": "prior"}}
    _, audit = applied_field_policies("fixed_class_safe_sage", {7: cra}, [obj], classes)
    assert audit[0]["observed_class"] == "person"
    obj.diagnostics["accepted_instance"] = False
    _, audit = applied_field_policies("fixed_class_safe_sage", {7: cra}, [obj], classes)
    assert audit[0]["observed_class"] == "unknown"


def test_default_cra_guidance_is_identical_and_geometric_field_is_shadow_only():
    cra = policy()
    classes = SafeSAGEGuidancePolicy.from_json("configs/safe_sage_paper_baseline.json")
    actual, audit = applied_field_policies(None, {7: cra}, [], classes)
    assert actual[7] is cra.guidance_parameters
    assert audit[0]["field_used_for_control"]
    _, audit = applied_field_policies("geometric_velocity_cbf", {7: cra}, [], classes)
    assert not audit[0]["field_used_for_control"]
    with pytest.raises(ValueError):
        validate_baseline(CRABaseline.CRA_RISK_DEPENDENT_LAMBDA_ABLATION)


def geometric_fixture(distance=1., velocity=(-.5, 0.), anchor_yaw=0.):
    config = LocalOccupancyGridConfig(resolution=.1, minimum_x=0, maximum_x=3,
                                    minimum_y=-.05, maximum_y=.05)
    occupied = np.zeros(config.shape, bool)
    occupied[0, round((distance - .05) / .1)] = True
    grid = NS(config=config, occupied_mask=occupied, cell_track_ids=np.full(config.shape, 7))
    field = NS(sensor_timestamp_seconds=1., anchor_yaw=anchor_yaw,
               tracking_result=NS(tracked_obstacles=(NS(track_id=7,
                   velocity_world_xy=velocity, observation_count=3),)))
    return field, grid, RectangularRobotFootprint(.3, .3, .2, .2, .02)


def test_geometric_uses_observed_physical_velocity_and_rejects_stale_or_future():
    field, grid, footprint = geometric_fixture()
    moving, audit = geometric_cbf_command(field, grid, np.zeros(3), (.2, 0.), footprint, 1.1)
    field.tracking_result.tracked_obstacles[0].velocity_world_xy = (0., 0.)
    static, _ = geometric_cbf_command(field, grid, np.zeros(3), (.2, 0.), footprint, 1.1)
    assert moving[0] < static[0]
    assert audit["current_velocity_cells"] == 1
    for now in (.9, 1.36):
        command, audit = geometric_cbf_command(field, grid, np.zeros(3), (.2, .6), footprint, now)
        assert not command.any()
        assert audit["stop_reason"] == "stale_field"


def test_geometric_rotation_and_no_mass_or_class_input():
    field, grid, footprint = geometric_fixture(velocity=(0., -.5), anchor_yaw=np.pi/2)
    command, audit = geometric_cbf_command(field, grid, np.zeros(3), (.2, .3), footprint, 1.)
    reference, _ = geometric_cbf_command(*geometric_fixture()[:2], np.zeros(3), (.2, .3), footprint, 1.)
    np.testing.assert_allclose(command, reference, atol=1e-12)
    assert audit["safety_radius_m"] > np.hypot(.32, .22)
    assert audit["cbf_margin"] >= -1e-9 or audit["stop_reason"] is not None


def test_geometric_unknown_velocities_are_counted_as_priors():
    field, grid, footprint = geometric_fixture()
    field.tracking_result.tracked_obstacles = ()
    _, audit = geometric_cbf_command(field, grid, np.zeros(3), (.2, 0.), footprint, 1.)
    assert audit["zero_velocity_prior_cells"] == 1
    assert audit["current_velocity_cells"] == 0


def test_exogenous_crossing_schedule_is_independent_of_robot_progress():
    for robot_y in (-100, 100):
        assert not crossing_due(now=17.9, robot_y=robot_y, trigger_y=7., delay_seconds=8., navigation_start=10.)
        assert crossing_due(now=18., robot_y=robot_y, trigger_y=7., delay_seconds=8., navigation_start=10.)
        assert not crossing_due(now=100., robot_y=robot_y, trigger_y=7., delay_seconds=8., navigation_start=None)
    assert crossing_due(now=1., robot_y=7., trigger_y=7., delay_seconds=None, navigation_start=None)


def test_signed_clearance_preserves_failed_collision():
    from test_cra_experiments import result
    item = result(CRABaseline.FIXED_CLASS_SAFE_SAGE, completed=False, collision=True,
                  minimum_clearance_m=-.02)
    assert item.minimum_clearance_m == -.02
    with pytest.raises(ValueError, match="collision"):
        replace(item, collision=False)


def test_actual_field_changes_boundary_data_but_not_occupancy_or_cra():
    from test_object_risk_map import mapped_worker
    from test_live_object_perception import packet
    cra, fixed = mapped_worker(), mapped_worker()
    fixed.experiment_baseline = CRABaseline.FIXED_CLASS_SAFE_SAGE
    a, b = cra(packet()), fixed(packet())
    np.testing.assert_array_equal(a.source_occupancy_grid.occupied_mask, b.source_occupancy_grid.occupied_mask)
    np.testing.assert_array_equal(a.source_occupancy_grid.cell_track_ids, b.source_occupancy_grid.cell_track_ids)
    assert a.objects[0].policy.as_dict() == b.objects[0].policy.as_dict()
    for ga, gb in zip(a.field.pipeline.configuration_space.occupancy_grids,
                      b.field.pipeline.configuration_space.occupancy_grids, strict=True):
        np.testing.assert_array_equal(ga.occupied_mask, gb.occupied_mask)
    assert b.diagnostics["applied_field_policies"][0]["applied_field_risk"] == 1
    assert a.diagnostics["applied_field_policies"][0]["applied_field_risk"] < 1
    assert not np.array_equal(a.field.pipeline.h_full.values, b.field.pipeline.h_full.values)
