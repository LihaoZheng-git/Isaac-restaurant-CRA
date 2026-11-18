import numpy as np
from test_laplace_guidance_tracking_mpc import config, field
from test_safe_sage_cbf_filter import make_filter

from robot.safety.committed_aisle_mpc import AisleGuideFrame
from robot.safety.field_alignment_mpc import FieldAlignmentConfig, FieldAlignmentMPC
from robot.safety.nominal_preserving_aisle_mpc import (
    NominalAisleRegion,
    NominalPreservingAisleMPC,
)
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter


def planners(*, guide_frame=None):
    paper = make_filter()
    settings = dict(
        alignment_config=FieldAlignmentConfig(velocity_weight=0.0, heading_weight=0.0),
        restaurant_mode=True,
    )
    return (
        FieldAlignmentMPC(paper, config(), **settings),
        NominalPreservingAisleMPC(
            paper,
            config(),
            backup_filter=SafeSAGEUnicycleCBFFilter(paper),
            guide_frame=guide_frame,
            **settings,
        ),
    )


def test_keeps_original_horizon_in_cruise_and_when_inside_the_pass_region(monkeypatch):
    native, mpc = planners()
    initial = np.zeros(3)
    nominal = np.tile((0.2, 0.0), (mpc.config.horizon_steps, 1))
    args = dict(
        active_guide_index=0,
        route_goal_grid_xy=(2.0, 0.0),
        nominal_goal_grid_xy=(2.0, 0.0),
    )
    fields = field(g=(0.0, 0.0), h=1.0)
    original = native.plan(initial, nominal, **args, **fields)
    monkeypatch.setattr(mpc, "_candidate_corridors", lambda *_: iter(()))
    cruise = mpc.plan(initial, nominal, **args, **fields)
    np.testing.assert_array_equal(cruise.command_sequence, original.command_sequence)

    region = NominalAisleRegion(0.0, 0.8, 1.6, 2.0, 0.0, 0.08, 0.2, 0.0)
    monkeypatch.setattr(mpc, "_candidate_corridors", lambda *_: iter((region,)))
    active = mpc.plan(initial, nominal, **args, **fields)
    assert active.corridor is region
    assert active.committed_bypass["nominal_selection"] == "native_preserved"
    assert active.objective_value == original.objective_value
    np.testing.assert_array_equal(active.command_sequence, original.command_sequence)


def test_finite_search_ranks_safe_candidates_by_original_nominal_cost(monkeypatch):
    _, mpc = planners()
    initial = np.zeros(3)
    region = NominalAisleRegion(0.0, 0.8, 1.6, 2.0, 0.0, 0.08, 0.2, 0.0)
    nominal = np.tile((0.1, 0.0), (mpc.config.horizon_steps, 1))
    first = np.tile((0.2, 0.0), (len(nominal), 1))
    closer = np.tile((0.12, 0.0), (len(nominal), 1))
    monkeypatch.setattr(mpc, "_alternative_sequences", lambda *_: iter((first, closer)))
    field_args = dict(temporal_safety_derivative=0.0, **field(g=(0.0, 0.0), h=1.0))
    result = mpc._plan_in_corridor(
        initial,
        nominal,
        region,
        field_args,
        "active_pass",
        lambda: mpc._fallback_result(initial, nominal, status="fixture"),
        None,
    )
    assert result.success
    assert (
        result.committed_bypass["nominal_selection"] == "minimum_cost_checked_candidate"
    )
    np.testing.assert_array_equal(result.command_sequence, closer)


def test_acquisition_can_turn_and_passage_requires_sampled_forward_motion():
    _, mpc = planners()
    region = NominalAisleRegion(0.0, 0.8, 1.6, 2.0, 0.0, 0.08, 0.2, 0.0)
    sequence = np.tile((0.2, 0.0), (mpc.config.horizon_steps, 1))
    sequence[0] = (0.0, 0.4)
    assert mpc.corridor_valid(np.zeros(3), sequence, region)
    assert not mpc.corridor_valid(np.array([0.8, 0.08, 0.0]), sequence, region)


def test_rotated_guide_uses_route_progress_and_preserves_safe_field_plan(monkeypatch):
    frame = AisleGuideFrame((0.0, 0.0), (0.0, -2.0))
    native, mpc = planners(guide_frame=frame)
    initial = frame.to_grid(np.zeros(3))
    nominal = np.tile((0.2, 0.0), (mpc.config.horizon_steps, 1))
    region = NominalAisleRegion(0.0, 0.8, 1.6, 2.0, 0.0, 0.08, 0.2, 0.0)
    monkeypatch.setattr(mpc, "_candidate_corridors", lambda *_: iter((region,)))
    args = dict(
        active_guide_index=2,
        route_goal_grid_xy=(0.0, -2.0),
        nominal_goal_grid_xy=(0.0, -2.0),
    )
    fields = field(g=(0.0, 0.0), h=1.0)
    original = native.plan(initial, nominal, **args, **fields)
    result = mpc.plan(initial, nominal, **args, **fields)
    assert result.committed_bypass["nominal_selection"] == "native_preserved"
    assert mpc.corridor_valid(initial, result.command_sequence, region)
    assert mpc.revalidate_commands(initial, result.command_sequence, **fields)
    np.testing.assert_array_equal(result.command_sequence, original.command_sequence)
    assert result.predicted_chassis_states[-1, 1] < 0.0
    assert mpc.route_states(result.predicted_chassis_states[-1])[0] > 0.0
