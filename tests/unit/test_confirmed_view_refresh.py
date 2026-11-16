"""A better view can replace evidence; ambiguity cannot erase or renew it."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from robot.perception.confirmed_view_refresh import (
    ConfirmedViewRefreshConfig,
    growth_evidence,
)
from robot.perception.image_object_association import (
    ImageObjectConfig,
    StationaryImageObjects,
)
from robot.perception.physical_attributes import EstimateStatus
from test_moving_view_association import surface, context, mask_for, candidate
from test_static_prop_evidence import memory
from test_object_risk_map import scalar


def prepared():
    full = surface()
    small = full[(full[:, 1] >= -0.08) & (full[:, 1] <= 0.08)]
    state = StationaryImageObjects(
        ImageObjectConfig(
            hold_overlapping_hypotheses=True, confirmed_evidence_refresh=True
        )
    )
    for t in (1.0, 1.1, 1.2):
        accepted = state.update(
            [candidate(mask_for(small, context()), len(small))],
            small,
            t,
            view=context(),
        )
    return state, accepted[0]["object_id"], small, full


def test_growth_needs_three_sources_and_does_not_commit_before_fresh_assessment():
    state, key, small, full = prepared()
    view = context(0.6)
    for t in (1.3, 1.4):
        assert not state.update(
            [candidate(mask_for(full, view), len(full))], full, t, view=view
        )
        np.testing.assert_array_equal(state.records[key]["points"], small)
        assert state.records[key]["first"] == 1.0
    accepted = state.update(
        [candidate(mask_for(full, view), len(full))], full, 1.5, view=view
    )
    assert len(accepted) == 1 and accepted[0]["object_id"] == key
    assert accepted[0]["diagnostic"]["evidence_refresh"]["confirmed"]
    np.testing.assert_array_equal(state.records[key]["points"], small)
    state.commit_refresh(accepted[0], full, view, 1.5)
    np.testing.assert_array_equal(state.records[key]["points"], full)
    assert state.records[key]["first"] == 1.5
    assert state.records[key]["refresh_history"][0]["previous_source_time"] == 1.0


@pytest.mark.parametrize(
    "case",
    [
        "density_only",
        "lost_old_patch",
        "wrong_mask",
        "lower_quality",
        "oversize",
        "no_view",
    ],
)
def test_bad_view_cannot_confirm_growth(case):
    state, key, small, full = prepared()
    view = context(0.6)
    points = full.copy()
    if case == "density_only":
        points = np.repeat(small, 10, axis=0)
    if case == "lost_old_patch":
        points = full + [0.1, 0, 0]
    if case == "oversize":
        points = np.vstack((full, full + [0, 2, 0]))
    row = candidate(mask_for(points, view), len(points))
    if case == "wrong_mask":
        row["proposal"] = replace(
            row["proposal"],
            detection=replace(
                row["proposal"].detection, mask=np.zeros((100, 100), bool)
            ),
        )
    if case == "lower_quality":
        row["proposal"] = replace(row["proposal"], model_quality=0.7)
    audit = growth_evidence(
        state.records[key],
        row,
        points,
        None if case == "no_view" else view,
        state.config,
        ConfirmedViewRefreshConfig(),
    )
    assert audit["rejections"]
    np.testing.assert_array_equal(state.records[key]["points"], small)


def test_competing_references_cannot_be_selected_by_bigger_mask_or_score():
    state, key, small, full = prepared()
    state.records[key + 1] = copy.deepcopy(state.records[key])
    for t in (1.3, 1.4, 1.5):
        assert not state.update(
            [candidate(mask_for(full, context(0.6)), len(full))],
            full,
            t,
            view=context(0.6),
        )
    assert not state.refresh.pending
    assert state.records[key]["first"] == 1.0


def test_gap_restarts_confirmation_and_original_reference_expiry_stays():
    state, key, small, full = prepared()
    for t in (1.3, 1.7, 1.8):
        assert not state.update(
            [candidate(mask_for(full, context(0.6)), len(full))],
            full,
            t,
            view=context(0.6),
        )
    assert state.refresh.pending[key]["count"] == 2
    assert not state.update(
        [candidate(mask_for(full, context(0.6)), len(full))],
        full,
        11.1,
        view=context(0.6),
    )
    assert key not in state.records and key not in state.refresh.pending


def test_all_historical_anchors_prevent_successive_small_drift():
    state, key, small, full = prepared()
    state.records[key]["anchor_points"] = np.vstack((small, [[2, -0.5, 0.3]]))
    row = candidate(mask_for(full, context(0.6)), len(full))
    audit = growth_evidence(
        state.records[key],
        row,
        full,
        context(0.6),
        state.config,
        ConfirmedViewRefreshConfig(),
    )
    assert "not every historical anchor is currently supported" in audit["rejections"]


@pytest.mark.parametrize(
    "case",
    [
        "prior",
        "lower_confidence",
        "wider_bounds",
        "old_timestamp",
        "lost_support",
        "unconfirmed",
        "competitor",
    ],
)
def test_rejected_property_refresh_keeps_values_geometry_and_source(case):
    m, points = memory()
    old = copy.deepcopy(m.records[5])
    mass, contact = scalar(2.1, "kg", 2.0), scalar(0.4, "1", 2.0)
    proof = dict(confirmed=True, observations=3, rejections=[])
    if case == "prior":
        mass = replace(mass, status=EstimateStatus.PRIOR)
    if case == "lower_confidence":
        mass = replace(mass, confidence=0.01)
    if case == "wider_bounds":
        mass = replace(mass, upper_bound=100.0)
    if case == "old_timestamp":
        contact = replace(contact, timestamp_seconds=1.0)
    if case == "lost_support":
        points = points + [0.2, 0, 0]
    if case == "unconfirmed":
        proof["confirmed"] = False
    if case == "competitor":
        m.records[6] = copy.deepcopy(old)
    assert not m.refresh(
        5,
        points,
        mass,
        contact,
        frame_id=2,
        source_time=2.0,
        robot_xy=(0, 0),
        proof=proof,
    )
    assert (
        m.records[5]["mass"] == old["mass"]
        and m.records[5]["contact"] == old["contact"]
    )
    np.testing.assert_array_equal(m.records[5]["points"], old["points"])
    assert m.records[5]["location_source_time"] == old["location_source_time"]


def test_confirmed_fresh_property_replacement_is_not_score_selected():
    m, points = memory()
    original = m.records[5]["contact"]
    assert m.refresh(
        5,
        points,
        scalar(2.1, "kg", 2.0),
        scalar(0.1, "1", 2.0),
        frame_id=2,
        source_time=2.0,
        robot_xy=(0, 0),
        proof=dict(confirmed=True, observations=3, rejections=[]),
    )
    assert m.records[5]["contact"].value == 0.1  # lower score is allowed
    assert (
        m.records[5]["refresh_history"][0]["previous_contact"]["value"]
        == original.value
    )
    assert m.retained(5, 2.1)[0]["contact"].timestamp_seconds == 2.0


def test_default_has_no_refresh_behavior():
    assert not ImageObjectConfig().confirmed_evidence_refresh


def test_missing_intermediate_observation_resets_pending_proof():
    state, key, small, full = prepared()
    view = context(0.6)
    state.update([candidate(mask_for(full, view), len(full))], full, 1.3, view=view)
    state.update([], full, 1.4, view=view)
    assert not state.refresh.pending
    state.update([candidate(mask_for(full, view), len(full))], full, 1.5, view=view)
    assert state.refresh.pending[key]["count"] == 1


def test_runtime_refresh_is_transactional_and_kinetic_remains_current(monkeypatch):
    from test_image_object_runtime import setup, observed
    from test_image_object_association import proposal
    from robot.perception.moving_view_association import MovingViewAssociationConfig
    from robot.perception.static_prop_evidence import StaticPropEvidenceConfig
    import robot.evaluation.multi_object_perception as mapping

    runtime = setup()
    runtime.image_object_config = ImageObjectConfig(
        hold_overlapping_hypotheses=True, confirmed_evidence_refresh=True
    )
    runtime.image_object_property_persistence = True
    runtime.static_prop_evidence_config = StaticPropEvidenceConfig()
    runtime.moving_image_object_config = MovingViewAssociationConfig()
    full = surface()
    full[:, 2] *= 0.7  # Both views remain fully inside the unchanged test camera.
    small = full[full[:, 1] <= 0.15]
    current = [small, 0.7]
    runtime.segmenter.image_proposals = lambda *args: (
        proposal(mask_for(current[0], context(current[1]))),
    )

    def source(frame, t):
        points = current[0] - [current[1], 0, 0]
        return replace(
            observed(frame, t),
            obstacle_xyz=points,
            scene_xyz=points,
            robot_position_world_xy=(current[1], 0),
        )

    for i, t in enumerate((1.0, 1.1, 1.2), 1):
        runtime(source(i, t))
    memory = runtime.image_object_state.static_properties
    assert len(memory.records) == 1
    key = next(iter(memory.records))
    original = memory.records[key]["contact"]
    runtime.persistent_properties.freeze_properties()
    current[:] = [full, 1.0]
    for i, t in enumerate((1.3, 1.4), 4):
        runtime(source(i, t))
        assert (
            runtime.image_object_state.static_properties.records[key]["contact"]
            == original
        )
    before = runtime.image_object_state
    build = mapping.build_safe_sage_field_pipeline

    def fail(*args, **kwargs):
        raise RuntimeError("test field failure")

    monkeypatch.setattr(mapping, "build_safe_sage_field_pipeline", fail)
    with pytest.raises(RuntimeError, match="test field failure"):
        runtime(source(6, 1.5))
    assert runtime.image_object_state is before
    assert before.static_properties.records[key]["contact"] == original
    monkeypatch.setattr(mapping, "build_safe_sage_field_pipeline", build)
    result = runtime(source(7, 1.6))
    record = runtime.image_object_state.static_properties.records[key]
    assert record["contact"].timestamp_seconds == pytest.approx(1.7), [
        c.get("evidence_refresh", c.get("temporal_rejection"))
        for c in result.diagnostics["image_object_groups"]["candidates"]
    ]
    assert (
        record["refresh_history"][0]["previous_contact"]["timestamp_seconds"]
        == original.timestamp_seconds
    )
    obj = next(o for o in result.objects if o.attributes.track_id == key)
    assert obj.diagnostics["image_object_association"]["evidence_refresh"][
        "property_storage_accepted"
    ]
    assert obj.attributes.motion.relative_speed_mps.timestamp_seconds == pytest.approx(
        1.6
    )
    assert result.diagnostics["fixed_lambda"] == 1.25


@pytest.mark.parametrize("case", ["bin", "no_static", "mixed_scene"])
def test_cli_limits_experiment_to_crate(case, monkeypatch):
    import json
    from pathlib import Path

    path = Path("scripts/demos/cra_live_object_isaac_demo.py").resolve()
    cmd = json.loads(Path("configs/restaurant_contact_short_live.json").read_text())[
        "commands"
    ]["crate-static"]
    cmd += ["--confirmed-prop-refresh", "--output", "outputs/unused-refresh-cli"]
    if case == "bin":
        cmd[cmd.index("BrokenCrate")] = "ServiceContainer"
    if case == "no_static":
        cmd.remove("--static-prop-evidence")
    if case == "mixed_scene":
        cmd.append("--service-static-props")
    monkeypatch.setattr("sys.argv", cmd)
    code = compile(path.read_text().split("args.output.mkdir", 1)[0], str(path), "exec")
    with pytest.raises(SystemExit):
        exec(code, {"__file__": str(path)})
