from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from robot.perception.human_body_envelope import (
    HumanBodyEnvelope,
    HumanBodyEnvelopeConfig,
)
from robot.perception.human_track_recovery import HumanTrackRecoveryConfig
from robot.perception.tracking_uncertainty import (
    ReachableTrackRegion,
    TrackingUncertaintySnapshot,
)
from robot.safety.class_safety_policy import ClassSafetyParameters, ClassSafetyPolicy
from robot.safety.local_occupancy_grid import (
    LocalOccupancyGridConfig,
    build_local_occupancy_grid,
)
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidanceParameters
from test_human_body_envelope import tracking, cue
from test_human_track_recovery import missing


@pytest.mark.parametrize(
    "case", ["long-parallel", "long-parallel-near", "long-parallel-component"]
)
def test_native_recipe_argument_validation_before_simulator_start(
    case, monkeypatch, tmp_path
):
    root = Path(__file__).resolve().parents[2]
    recipe = json.loads(
        (root / "configs/restaurant_tracking_uncertainty_v1.json").read_text()
    )
    command = recipe["commands"][case]
    source = root / command[0]
    # Exercise real parser/semantic checks without creating outputs or starting Kit.
    marker = "args.output.mkdir"
    content = source.read_text()
    assert marker in content
    prefix = content.split(marker, 1)[0]
    monkeypatch.setattr(sys, "argv", [*command, "--output", str(tmp_path / "native")])
    monkeypatch.setattr(sys, "path", list(sys.path))
    scope = {"__file__": str(source), "__name__": "cli_validation_test"}
    exec(compile(prefix, str(source), "exec"), scope)
    assert scope["args"].human_tracking_uncertainty
    assert not (tmp_path / "native").exists()


def memory(center=(1.0, 0.0), **kwargs):
    m = HumanBodyEnvelope(
        HumanBodyEnvelopeConfig(**kwargs),
        recovery_config=HumanTrackRecoveryConfig(),
        planning_uncertainty=True,
    )
    m.update(tracking(center=center), cue(), camera_timestamp_seconds=1.0)
    return m


def guidance(risk=0.8):
    return SafeSAGEGuidanceParameters(
        ClassSafetyParameters(risk=risk, boundary_flux_magnitude=0.5 + 1.2 * risk),
        tangential_strength=1.25,
        passing_side="right",
    )


def region(time=1.3, center=(1.0, 0.0)):
    return ReachableTrackRegion(
        7, center, 1.0, 1.0, 11, time, 0.6, 1.0, 1.55, "test_missing"
    )


def raster(r, config=None, risk=0.8):
    cfg = config or LocalOccupancyGridConfig(
        resolution=0.2, minimum_x=-8, maximum_x=8, minimum_y=-8, maximum_y=8
    )
    p, _, _, _ = r.cells(cfg, np.zeros(2), 0.0)
    g = guidance(risk)
    grid = build_local_occupancy_grid(
        p,
        point_track_ids=np.full(len(p), r.geometry_id),
        class_safety_policy=ClassSafetyPolicy({"unknown": g.safety_parameters}),
        track_safety_parameters={r.geometry_id: g.safety_parameters},
        config=cfg,
    )
    return grid, g


def test_loss_becomes_growing_geometry_after_old_timeout_without_renewing_sources():
    m = memory(center=(8.0, 0.0))
    for t in (1.2, 1.8, 3.0, 5.0):
        assert (
            m.update(
                missing(t), {}, camera_timestamp_seconds=t, robot_world_xy=(0.0, 0.0)
            )
            == []
        )
        (r,) = m.uncertainty_regions
        assert r.radius_m == pytest.approx(0.6 + t - 1 + 1.55)
        assert r.center_world_xy == (8.0, 0.0)
        assert r.original_camera_timestamp == r.original_lidar_timestamp == 1.0
        assert m.identities[7]["lidar_timestamp"] == 1.0
        assert not r.diagnostic()["current_identity_verified"]


def test_original_visual_expiry_does_not_relabel_current_lidar_as_fresh_identity():
    m = memory(maximum_identity_age_seconds=0.15)
    assert (
        m.update(
            tracking(1.2, (1.02, 0.0), 2, matched=True),
            {},
            camera_timestamp_seconds=1.2,
        )
        == []
    )
    assert m.uncertainty_regions[0].reason == "original_identity_age"
    assert (
        m.identities[7]["camera_timestamp"] == m.identities[7]["lidar_timestamp"] == 1.0
    )
    assert (
        m.passing_selection_priors(
            tracking(1.2), camera_timestamp_seconds=1.2, maximum_age_seconds=0.5
        )
        == {}
    )


@pytest.mark.parametrize("failure", ["jump", "unmatched", "ambiguous", "support"])
def test_association_failures_retain_original_positive_geometry(failure):
    m = memory()
    t = tracking(
        1.2,
        (3.0, 0.0) if failure == "jump" else (1.02, 0.0),
        2,
        matched=failure != "unmatched",
        other=failure == "ambiguous",
    )
    if failure == "support":
        t = replace(t, point_track_ids=np.full(3, 7))
    assert m.update(t, {}, camera_timestamp_seconds=1.2) == []
    assert m.uncertainty_regions[0].center_world_xy == (1.0, 0.0)
    assert m.identities[7]["lidar_timestamp"] == 1.0


def test_unique_short_gap_id_recovery_keeps_existing_contract_and_clears_region():
    m = memory()
    m.update(missing(1.1), {}, camera_timestamp_seconds=1.1)
    t = tracking(1.3, (1.03, 0.0), 1)
    t = replace(
        t,
        tracked_obstacles=(replace(t.tracked_obstacles[0], track_id=8),),
        point_track_ids=np.full(20, 8),
    )
    (row,) = m.update(t, {}, camera_timestamp_seconds=1.3)
    assert row["track_id"] == 8 and row["original_person_camera_timestamp"] == 1.0
    assert not m.uncertainty_regions and set(m.identities) == {8}


def test_future_clock_and_out_of_scope_speed_remain_errors():
    m = memory()
    with pytest.raises(ValueError):
        m.update(tracking(0.9), {}, camera_timestamp_seconds=0.9)
    t = tracking(1.1, (1.01, 0.0), 2, matched=True)
    t = replace(
        t,
        tracked_obstacles=(
            replace(t.tracked_obstacles[0], velocity_world_xy=(2.0, 0.0)),
        ),
    )
    with pytest.raises(ValueError, match="recovery_motion_scope"):
        m.update(t, {}, camera_timestamp_seconds=1.1)


def test_missing_geometry_never_supplies_retained_mass_even_with_positive_unverified_track():
    from robot.perception.human_mass_identity_retention import (
        HumanMassIdentityRetention,
    )

    m = memory(maximum_identity_age_seconds=0.15)
    t = tracking(1.2, (1.02, 0.0), 2, matched=True)
    m.update(t, {}, camera_timestamp_seconds=1.2)
    values, audit = HumanMassIdentityRetention().update(
        t, m, {}, timestamp_seconds=1.2, camera_frame_id=12
    )
    assert values == {} and audit["missing_geometry_mass_not_resolved"] == [7]


@pytest.mark.parametrize("change", ["speed", "horizon", "future", "nan"])
def test_invalid_uncertainty_bounds_reject(change):
    with pytest.raises(ValueError):
        replace(
            region(),
            **{
                "speed": dict(maximum_speed_mps=0.5),
                "horizon": dict(checked_horizon_seconds=0.2),
                "future": dict(original_lidar_timestamp=2.0),
                "nan": dict(center_world_xy=(float("nan"), 0.0)),
            }[change],
        )


def test_long_loss_raster_work_is_bounded_and_never_erases_uncertainty():
    r = region(time=10000.0)
    grid, _ = raster(r)
    assert grid.occupied_mask.all()
    outside, _ = raster(region(center=(100.0, 100.0)))
    assert not outside.occupied_mask.any()


@pytest.mark.parametrize(
    "failure", [None, "source", "risk", "hole", "clipped", "outside_original_body"]
)
def test_continuity_guard_requires_actual_whole_region_field_coverage(failure):
    from robot.evaluation.nearby_track_loss_guard import (
        NearbyTrackLossConfig,
        NearbyTrackLossGuard,
    )

    r = region()
    cfg = (
        None
        if failure != "clipped"
        else LocalOccupancyGridConfig(
            resolution=0.2, minimum_x=-1, maximum_x=3, minimum_y=-1, maximum_y=1
        )
    )
    grid, g = raster(r, cfg, risk=0.5 if failure == "risk" else 0.8)
    if failure == "hole":
        mask = grid.occupied_mask.copy()
        row, col = np.argwhere(mask)[0]
        mask[row, col] = False
        grid = replace(grid, occupied_mask=mask)
    snapshot = TrackingUncertaintySnapshot(1.3, ((r, guidance()),))
    guard = NearbyTrackLossGuard(NearbyTrackLossConfig(allow_tracking_uncertainty=True))
    p = np.array([[1.0 + dx, dy] for dx in (0.0, 0.01) for dy in (-0.01, 0.01)])
    if failure == "outside_original_body":
        p[0] = [1.0, 1.0]
    t = replace(tracking().tracked_obstacles[0], point_count=4)

    def args(time, snap):
        return dict(
            source_time=time,
            robot_world_xy=(0.0, 0.0),
            current_track_risks={7: 0.8},
            current_uncertainty_snapshot=snap,
            current_occupancy_grid=grid,
            current_grid_anchor_xy=np.zeros(2),
            current_grid_anchor_yaw=0.0,
        )

    guard.observe(
        (t,),
        current_points_world_xy=p,
        current_point_track_ids=np.full(4, 7),
        current_points_source_time=1.0,
        **args(1.0, TrackingUncertaintySnapshot(1.0)),
    )
    if failure == "source":
        guard.observed[7] = (np.array([1.0, 0.0]), 0.9)
    result = guard.observe(
        (),
        current_points_world_xy=np.empty((0, 2)),
        current_point_track_ids=np.empty(0, dtype=int),
        current_points_source_time=1.3,
        **args(1.3, snapshot),
    )
    assert result["blocked"] == (failure is not None)
    assert guard.observed[7][1] == (0.9 if failure == "source" else 1.0)


def test_actual_multi_object_field_contains_prior_without_observed_body_or_factor_refresh():
    from test_object_risk_map import mapped_worker
    from test_live_object_perception import packet

    w = mapped_worker()
    w.human_body_envelope = memory(center=(3.5, 0.0))
    w.human_body_envelope.geometry_policies[7] = guidance(0.9)
    result = w(packet(frame=8, t=1.8))
    rows = result.diagnostics["tracking_uncertainty_regions"]
    assert len(rows) == 1 and rows[0]["applied_cra_phi"] == 0.9
    assert rows[0]["original_lidar_timestamp_seconds"] == 1.0
    assert not result.diagnostics["human_body_envelope_priors"]
    assert result.tracking_uncertainty_snapshot.source_time == 1.8
    assert np.any(result.source_occupancy_grid.cell_track_ids == 6_000_000_007)
    assert result.field.tracking_result.point_track_ids.tolist() == [0] * 4


def test_independent_audit_rejects_actor_outside_region_and_never_requires_fabricated_future():
    from robot.evaluation.human_body_envelope_validation import (
        validate_tracking_uncertainty,
    )

    r = region()
    d = dict(r.diagnostic(), applied_cra_phi=0.8, original_applied_risk=0.8)
    record = {"diagnostics": {"tracking_uncertainty_regions": [d]}}
    truth = [dict(time=1.3, actor_xyz=[1.0, 0.0, 0.0])]
    audit = validate_tracking_uncertainty([record], truth, reference_radius_m=0.4)
    assert (
        audit["passed"]
        and audit["exercised"]
        and not audit["rows"][0]["full_future_available"]
    )
    truth[0]["actor_xyz"] = [6.0, 0.0, 0.0]
    assert not validate_tracking_uncertainty([record], truth, reference_radius_m=0.4)[
        "passed"
    ]


@pytest.mark.parametrize("center,allowed", [((0.0, 4.0), True), ((1.0, 0.0), False)])
def test_uncertain_occupancy_reaches_actual_cbf_and_preserves_real_conflict_stopping(
    center, allowed
):
    from robot.safety.cbf_filter import CBFSafetyFilter, CBFConfig

    r = region(center=center)
    grid, _ = raster(r)
    points, _, _, _ = r.cells(grid.config, np.zeros(2), 0.0)
    command = CBFSafetyFilter(CBFConfig(safety_radius=0.8)).filter_command(
        0.2, 0.0, points
    )
    assert (command.safe_linear_velocity > 0) == allowed
    if allowed:
        assert (
            command.constraint_feasible
            and command.safe_linear_velocity == pytest.approx(0.2)
        )
    else:
        assert command.emergency_stop_active


def test_field_failure_does_not_commit_uncertainty_or_tracker_state(monkeypatch):
    from test_object_risk_map import mapped_worker
    from test_live_object_perception import packet
    import robot.evaluation.multi_object_perception as module

    w = mapped_worker()
    w.human_body_envelope = memory(center=(3.5, 0.0))
    w.human_body_envelope.geometry_policies[7] = guidance()
    original = w.human_body_envelope

    def fail(*args, **kwargs):
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr(module, "build_safe_sage_field_pipeline", fail)
    with pytest.raises(RuntimeError, match="injected"):
        w(packet(frame=8, t=1.8))
    assert w.human_body_envelope is original and not original.uncertainty_regions
    assert w.previous is None
