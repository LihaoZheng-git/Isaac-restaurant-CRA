import copy
from dataclasses import replace

import numpy as np
import pytest

from robot.safety.person_relative_rear import (
    PersonRelativeRearConfig,
    PersonRelativeRearMemory,
)
from robot.safety.checked_rear_pass_reference import (
    CheckedRearPassReferenceConfig,
    search_checked_rear_pass_reference,
)


def inputs(*, side=1, yaw=0.0, lead=0.6):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    return dict(
        track_id=7,
        selection_timestamp=1.0,
        source_timestamp=1.0,
        initial=(0.0, 0.0, yaw),
        goal=rotation @ [3.9, 0.0],
        person_xy=rotation @ [lead, -side * 0.5],
        velocity_xy=rotation @ [0.0, side * 0.5],
        route_direction=rotation @ [1.0, 0.0],
    )


def configuration():
    return PersonRelativeRearConfig(
        maximum_linear_velocity=0.25, maximum_angular_velocity=0.8
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(maximum_prediction_seconds=4.0),
        dict(minimum_prediction_seconds=3.0),
        dict(prediction_samples=1),
        dict(prediction_samples=True),
        dict(prediction_samples=26),
        dict(maximum_source_gap_seconds=0.3),
        dict(deadline_slack_seconds=0.6),
        dict(rear_distance_m=0.0),
        dict(path_lookahead_m=np.nan),
        dict(maximum_linear_velocity=True),
        dict(maximum_angular_velocity=-1.0),
    ],
)
def test_invalid_timing_geometry_or_actuator_configuration_rejected(change):
    with pytest.raises(ValueError):
        replace(configuration(), **change)


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("yaw", [0.0, 0.9])
def test_person_anchored_curve_has_a_motion_rear_passage_and_route_rejoin(side, yaw):
    args = inputs(side=side, yaw=yaw)
    memory = PersonRelativeRearMemory(configuration())
    ref = memory.update(**args)
    assert ref is not None
    assert ref.screening_seconds <= ref.passage_seconds
    velocity = np.array(args["velocity_xy"])
    velocity /= np.linalg.norm(velocity)
    assert (ref.passage_point - ref.predicted_person) @ velocity <= -1.2 + 1e-12
    assert np.linalg.norm(ref.passage_point - ref.rejoin_point) > 0
    np.testing.assert_allclose(ref.path[0], args["initial"][:2], atol=1e-12)
    np.testing.assert_allclose(ref.path[16], ref.passage_point, atol=1e-12)
    np.testing.assert_allclose(ref.path[-1], ref.rejoin_point, atol=1e-12)
    route = np.array(args["route_direction"])
    delta = ref.rejoin_point - np.array(args["goal"])
    assert abs(route[0] * delta[1] - route[1] * delta[0]) < 1e-10
    assert ref.context(args["initial"])["person_relative_phase"] == "approach"
    assert ref.context((*ref.rejoin_point, yaw))["person_relative_phase"] == "rejoin"
    assert ref.diagnostic()["status"] == "inferred"


def test_original_distant_crossing_is_rejected_instead_of_chasing_a_late_person():
    memory = PersonRelativeRearMemory(configuration())
    assert memory.update(**inputs(lead=1.37)) is None
    assert memory.reason == "no_passage_within_short_kinematic_scope"
    # A new current source may admit a slot, keeping the original selection cap.
    ref = memory.update(**dict(inputs(lead=0.6), source_timestamp=1.1))
    assert (
        ref is not None
        and ref.selection_timestamp == 1.0
        and ref.source_timestamp == 1.1
    )
    assert memory.key == (7, 1.0)


def test_reference_is_frozen_under_small_current_motion_variations_without_clock_renewal():
    memory = PersonRelativeRearMemory(configuration())
    args = inputs()
    original = memory.update(**args)
    path = original.path.copy()
    changed = dict(
        args,
        source_timestamp=1.1,
        person_xy=np.array(args["person_xy"]) + [0.01, 0.05],
        velocity_xy=(0.01, 0.49),
    )
    held = memory.update(**changed)
    assert held is original
    np.testing.assert_array_equal(held.path, path)
    assert held.source_timestamp == 1.0 and held.selection_timestamp == 1.0
    assert held.expires_at == original.expires_at


@pytest.mark.parametrize(
    "change",
    [
        dict(source_timestamp=4.1),
        dict(person_xy=(0.6, 0.8)),
        dict(velocity_xy=(0.0, -0.5)),
        dict(route_direction=(0.0, 1.0)),
    ],
)
def test_current_motion_gap_route_or_deadline_invalidates_without_reacquisition(change):
    memory = PersonRelativeRearMemory(configuration())
    args = inputs()
    memory.update(**args)
    update = dict(args, source_timestamp=1.1, person_xy=(0.6, -0.45))
    update.update(change)
    assert memory.update(**update) is None
    again = dict(args, source_timestamp=update["source_timestamp"] + 0.1)
    assert memory.update(**again) is None


def test_current_identity_loss_cannot_resume_the_old_maneuver_or_mutate_a_transaction_copy():
    memory = PersonRelativeRearMemory(configuration())
    args = inputs()
    ref = memory.update(**args)
    staged = copy.deepcopy(memory)
    assert staged.update(**dict(args, source_timestamp=1.1)) is not None
    staged.update(**dict(args, track_id=None, source_timestamp=1.2))
    assert memory.reference is ref and memory.last_source == 1.0
    assert staged.reference is None
    assert staged.update(**dict(args, source_timestamp=1.3)) is None


@pytest.mark.parametrize(
    "change",
    [
        dict(track_id=True),
        dict(source_timestamp=0.9),
        dict(passage_seconds=4.0),
        dict(initial_state=(np.nan, 0.0, 0.0)),
        dict(original_person_xy=(-0.1, -0.5)),
    ],
)
def test_immutable_reference_contract_rejects_corrupt_origin(change):
    ref = PersonRelativeRearMemory(configuration()).update(**inputs())
    with pytest.raises((ValueError, TypeError)):
        replace(ref, **change)


@pytest.mark.parametrize("failure", [None, "temporal", "analytical"])
def test_person_reference_still_requires_every_original_exact_and_first_cbf_check(
    failure,
):
    from test_checked_rear_pass_reference import search

    mpc, _, options, check = search(failure=failure)
    cfg = replace(
        configuration(),
        maximum_linear_velocity=min(0.25, mpc.config.maximum_linear_velocity),
        maximum_angular_velocity=min(
            0.8,
            mpc.config.maximum_angular_velocity,
            abs(mpc.config.minimum_angular_velocity),
        ),
    )
    args = inputs(lead=0.6)
    ref = PersonRelativeRearMemory(cfg).update(**args)
    assert ref is not None
    result = search_checked_rear_pass_reference(
        mpc,
        args["initial"],
        args["goal"],
        args["person_xy"],
        args["velocity_xy"],
        args["route_direction"],
        options,
        check,
        config=CheckedRearPassReferenceConfig(
            person_relative=cfg,
            feasibility_led_start=True,
            allow_turn_first_fallback=True,
        ),
        person_relative_reference=ref,
    )
    assert bool(result.trajectory is not None) == (failure is None)
    if result.trajectory is not None:
        assert mpc._trajectory_is_safe(
            mpc._evaluate_trajectory(
                np.array(args["initial"]),
                result.trajectory.commands,
                require_feasible=True,
                **options,
            )
        )
        assert not check(result.trajectory.commands[0]).emergency_stop_active


def test_missing_reference_does_not_call_search_or_invent_a_wait_command():
    result = search_checked_rear_pass_reference(
        None,
        [0, 0, 0],
        [3.9, 0],
        [1.37, -0.5],
        [0, 0.5],
        [1, 0],
        {},
        lambda _: pytest.fail("unqualified reference cannot issue a command"),
        config=CheckedRearPassReferenceConfig(person_relative=configuration()),
    )
    assert result.trajectory is None and result.diagnostics["counts"]["tested"] == 0


def test_terminal_route_context_is_validated_and_defaults_remain_unset():
    from test_live_object_perception import packet

    assert packet().nominal_route_terminal_goal_world_xy is None
    assert replace(
        packet(), nominal_route_terminal_goal_world_xy=(3.9, 0.0)
    ).nominal_route_terminal_goal_world_xy == (3.9, 0.0)
    with pytest.raises(ValueError):
        replace(packet(), nominal_route_terminal_goal_world_xy=(np.nan, 0.0))


def test_staged_all_object_memory_does_not_commit_on_discard():
    from test_live_checked_wait import configured
    from test_live_object_perception import packet
    from robot.evaluation.live_checked_wait import StagedLivePerception
    from robot.safety.encounter_motion import EncounterMotionClassifier

    worker = configured()
    worker.encounter_motion = EncounterMotionClassifier()
    worker.live_rear_pass_config = CheckedRearPassReferenceConfig(
        person_relative=configuration()
    )
    worker.person_relative_rear_memory = PersonRelativeRearMemory(configuration())
    original = worker.person_relative_rear_memory
    staged = StagedLivePerception(worker)
    source = replace(
        packet(),
        nominal_route_direction_world_xy=(1.0, 0.0),
        nominal_route_terminal_goal_world_xy=(3.9, 0.0),
    )
    result = staged(source)
    assert (
        worker.person_relative_rear_memory is original and original.last_source is None
    )
    staged.discard()
    result = staged(source)
    staged.accept(result)
    assert worker.person_relative_rear_memory is not original
    assert worker.person_relative_rear_memory.last_source == 1.0


@pytest.mark.parametrize(
    "change",
    [dict(track_id=8), dict(selection_timestamp=0.9), dict(source_timestamp=0.9)],
)
def test_reference_current_identity_and_source_cannot_be_substituted(change):
    args = inputs()
    ref = PersonRelativeRearMemory(configuration()).update(**args)
    current = dict(
        track_id=7,
        selection_timestamp=1.0,
        source_timestamp=1.0,
        person_xy=args["person_xy"],
        velocity_xy=args["velocity_xy"],
        route_direction=(1.0, 0.0),
    )
    current.update(change)
    with pytest.raises(ValueError):
        ref.validate_current(**current)


def test_a_new_passage_cannot_be_planned_beyond_original_policy_expiry():
    memory = PersonRelativeRearMemory(configuration())
    assert memory.update(**dict(inputs(), source_timestamp=8.8)) is None
    assert memory.key == (7, 1.0)


def test_new_field_rejects_reference_identity_and_motion_mismatch():
    from test_safe_sage_runtime import make_runtime, make_observation, make_provider

    runtime = make_runtime()
    runtime.initialize(make_observation(1, 1.0), make_provider())
    f = runtime.field
    t = replace(
        f.tracking_result.tracked_obstacles[0],
        point_count=30,
        observation_count=3,
        matched_previous_track=True,
        centroid_world_xy=(0.6, -0.5),
        velocity_world_xy=(0.0, 0.5),
    )
    tracking = replace(f.tracking_result, tracked_obstacles=(t,))
    ref = PersonRelativeRearMemory(configuration()).update(
        **dict(inputs(), track_id=t.track_id)
    )
    field = replace(
        f,
        anchor_position_world_xy=np.zeros(2),
        anchor_yaw=0.0,
        tracking_result=tracking,
        motion_passing_reference_track_ids=(t.track_id,),
        motion_passing_reference_sides=("right",),
        motion_passing_reference_source_timestamp_seconds=1.0,
        rear_pass_reference_track_ids=(t.track_id,),
        rear_pass_route_direction_grid_xy=(1.0, 0.0),
        person_relative_rear_reference=ref,
    )
    assert field.person_relative_rear_reference is ref
    with pytest.raises(ValueError, match="qualified rear identity"):
        replace(field, rear_pass_reference_track_ids=())
    with pytest.raises(ValueError, match="current person motion"):
        replace(
            field,
            tracking_result=replace(
                tracking, tracked_obstacles=(replace(t, velocity_world_xy=(0.0, -0.5)),)
            ),
        )


@pytest.mark.parametrize("has_terminal", [True, False])
def test_live_worker_uses_person_reference_or_explicit_context_rejection(has_terminal):
    from types import SimpleNamespace
    from test_encounter_guidance import inputs as policies
    from test_live_rear_pass import track
    from robot.evaluation.encounter_guidance import apply_encounter_guidance
    from robot.evaluation.live_rear_pass import adapt_live_rear_guidance

    cfg = configuration()
    memory = PersonRelativeRearMemory(cfg)
    tracking = SimpleNamespace(
        timestamp_seconds=2.0,
        tracked_obstacles=[track(position=(10.5, 20.6), velocity=(-0.5, 0.0))],
    )
    source = SimpleNamespace(
        sensor_timestamp_seconds=2.0,
        robot_position_world_xy=(10.0, 20.0),
        robot_yaw=np.pi / 2,
        waypoint_world_xy=(10.0, 21.0),
        nominal_route_direction_world_xy=(0.0, 1.0),
    )
    if has_terminal:
        source.nominal_route_terminal_goal_world_xy = (10.0, 23.9)
    normal, rows, classification, passing = policies()
    applied, rows, crossing = apply_encounter_guidance(
        normal, rows, classification, passing, source_time=2.0
    )
    _, new_rows, wait, rear, _ = adapt_live_rear_guidance(
        normal,
        applied,
        rows,
        tracking,
        source,
        anchor=np.array([10.0, 20.0]),
        anchor_yaw=np.pi / 2,
        crossing_ids=crossing,
        config=CheckedRearPassReferenceConfig(
            person_relative=cfg, feasibility_led_start=True
        ),
        person_relative_memory=memory,
    )
    if has_terminal:
        assert rear == (7,) and not wait and memory.reference is not None
        assert memory.reference.nominal_goal_xy == pytest.approx((3.9, 0.0))
    else:
        assert not rear and memory.reference is None
        assert any(
            "terminal-route" in r.get("live_rear_pass_behavior", {}).get("reason", "")
            for r in new_rows
        )
