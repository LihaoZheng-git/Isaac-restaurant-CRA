from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from test_motion_aware_passing import sample
from robot.evaluation.multi_object_perception import fresh_person_selection_evidence
from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE
from robot.safety.motion_aware_passing import (
    MotionAwarePassing,
    MotionAwarePassingConfig,
)


def observed_step(policy, t, eligible, ids=(7,)):
    tracking, _ = sample(t, ids=ids)
    return policy.update(
        tracking,
        robot_xy=(0, 0),
        robot_velocity_xy=(0, 0.2),
        goal_xy=(0, 4),
        visual_person_track_ids=eligible,
    )


def test_unverified_motion_never_selects_but_original_verified_policy_can_latch():
    p = MotionAwarePassing(
        MotionAwarePassingConfig(require_visual_person_selection=True)
    )
    for t in (0.1, 0.2, 0.3):
        sides, audit = observed_step(p, t, ())
    assert not sides and audit["decision"] == "unverified_visual_person_target"
    sides, audit = observed_step(p, 0.4, (7,))
    assert sides and audit["original_selection_visual_person_verified"] is True
    sides, held = observed_step(p, 0.5, ())
    assert sides and held["encounter_selected_timestamp"] == 0.4
    assert held["current_visual_person_track_ids"] == []
    assert held["original_selection_visual_person_verified"] is True
    sides, released = observed_step(p, 0.9, ())
    assert not sides and released["decision"] == "released_source_or_identity_gap"


def test_unknown_competing_geometry_still_releases_or_blocks_human_preference():
    p = MotionAwarePassing(
        MotionAwarePassingConfig(require_visual_person_selection=True)
    )
    for t in (0.1, 0.2, 0.3):
        sides, audit = observed_step(p, t, (7,), ids=(7, 8))
    assert not sides and audit["decision"] == "ambiguous_multiple_crossings"


@pytest.mark.parametrize("eligible", [None, (True,), (999,), (7, 7)])
def test_invalid_current_eligibility_is_rejected_transactionally(eligible):
    p = MotionAwarePassing(
        MotionAwarePassingConfig(require_visual_person_selection=True)
    )
    with pytest.raises(ValueError):
        observed_step(p, 0.1, eligible)
    assert p.last_timestamp is None and p.history == {}


@pytest.mark.parametrize(
    "problem",
    [None, "stale", "no_instance", "retained_only", "wrong_source", "unavailable"],
)
def test_person_selection_reuses_only_current_accepted_association(problem):
    cue = dict(
        value=70.0,
        status="prior",
        source=HUMAN_MASS_PRIOR_SOURCE,
        timestamp_seconds=10.0,
    )
    d = {"accepted_instance": True, "geometry_evidence": {"human_mass_prior": cue}}
    if problem == "stale":
        cue["timestamp_seconds"] = 9.9
    if problem == "no_instance":
        d["accepted_instance"] = False
    if problem == "retained_only":
        d["geometry_evidence"] = {}
        d["property_evidence"] = {"mass": {"mode": "persistence_prior"}}
    if problem == "wrong_source":
        cue["source"] = "visual_mass_network"
    if problem == "unavailable":
        cue.update(value=None, status="unavailable")
    obj = NS(attributes=NS(track_id=7), diagnostics=d)
    before = deepcopy(d)
    eligible = fresh_person_selection_evidence([obj], 10.0, 42)
    assert bool(eligible) == (problem is None)
    assert d == before
    if eligible:
        assert eligible[7]["camera_frame_id"] == 42
        assert eligible[7]["camera_timestamp_seconds"] == 10.0


def test_missing_person_cue_keeps_actual_all_obstacle_field_and_scores_unchanged():
    from dataclasses import replace
    import numpy as np
    from test_live_object_perception import packet
    from test_object_risk_map import mapped_worker

    actual, baseline = mapped_worker(), mapped_worker()
    actual.motion_passing = MotionAwarePassing(
        MotionAwarePassingConfig(
            minimum_point_count=3, require_visual_person_selection=True
        )
    )
    for i in range(6):
        source = packet(i + 1, 1.0 + 0.1 * i)
        points = source.obstacle_xyz + [0, 0.6 - 0.04 * i, 0]
        source = replace(source, obstacle_xyz=points, scene_xyz=points)
        result, reference = actual(source), baseline(source)
    audit = result.diagnostics["motion_aware_passing"]
    assert audit["preferences"] == {}
    assert audit["decision"] == "unverified_visual_person_target"
    assert (
        result.field.all_obstacle_point_count
        == reference.field.all_obstacle_point_count
        == 4
    )
    applied = result.diagnostics["applied_field_policies"]
    original = reference.diagnostics["applied_field_policies"]
    assert [{k: row[k] for k in base} for row, base in zip(applied, original)] == original
    assert all(row["applied_passing_side"] == row["cra_passing_side"] for row in applied)
    np.testing.assert_array_equal(
        result.field.pipeline.h_full.values, reference.field.pipeline.h_full.values
    )


def test_observed_selection_requires_actual_person_model_contract():
    from test_live_object_perception import worker
    from robot.perception.persistent_object_evidence import (
        PersistentObjectEvidenceConfig,
    )

    with pytest.raises(ValueError, match="visual person detector"):
        worker(
            persistent_property_config=PersistentObjectEvidenceConfig("unit scope"),
            motion_passing_config=MotionAwarePassingConfig(
                require_visual_person_selection=True
            ),
        )
