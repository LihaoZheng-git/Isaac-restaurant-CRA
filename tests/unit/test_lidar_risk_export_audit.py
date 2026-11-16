"""A rendered score must not silently become a different CRA/policy number."""

import numpy as np
import pytest

from scripts.evaluation.audit_lidar_risk_export import audit_field


def fixture():
    record = dict(phi=.99, diagnostics=dict(applied_field_policies=[
        dict(track_or_group_id=1, applied_field_risk=.61),
        dict(track_or_group_id=9, applied_field_risk=.74)]), objects=[
            dict(track_id=1, diagnostics=dict(policy_return_count=2))])
    world = np.array([[2., 3., .2], [4., 5., .3]])
    arrays = dict(points_world_xy=world[:, :2].copy(), point_track_ids=np.array([1, 1]),
                  point_phi=np.array([.61, .61]), retained_world_xy=np.array([[1., 2.]]),
                  retained_track_ids=np.array([9]), retained_phi=np.array([.74]))
    return record, arrays, world


def test_exact_original_policy_and_current_point_parity():
    assert audit_field(*fixture()) == dict(point=2, retained=1)


@pytest.mark.parametrize("fault", ["top_level_phi", "prior_floor", "point", "identity", "length"])
def test_corrupted_exports_fail_independently_of_renderer(fault):
    record, arrays, world = fixture()
    if fault == "top_level_phi":
        arrays["point_phi"][0] = record["phi"]
    elif fault == "prior_floor":
        arrays["retained_phi"][0] = .25
    elif fault == "point":
        arrays["points_world_xy"][0, 0] += .01
    elif fault == "identity":
        arrays["point_track_ids"][0] = 9
        arrays["point_phi"][0] = .74
    else:
        arrays["point_phi"] = arrays["point_phi"][:1]
    with pytest.raises(ValueError):
        audit_field(record, arrays, world)
