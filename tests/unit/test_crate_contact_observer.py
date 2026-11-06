from types import SimpleNamespace

import pytest

from robot.evaluation.crate_contact_observer import CrateContactObserver


@pytest.fixture
def observer():
    obs = CrateContactObserver("/World/Carter", "/World/Props/Crate")
    obs._stage_id = 15
    obs._path = str
    obs._metres_per_unit = 0.01
    obs.installed = True
    return obs


def header(left, right, event="CONTACT_FOUND", start=0, count=0):
    return SimpleNamespace(
        stage_id=15,
        type=SimpleNamespace(name=event),
        actor0=left,
        actor1=right,
        collider0=left,
        collider1=right,
        contact_data_offset=start,
        num_contact_data=count,
    )


def test_external_support_positive_control_is_required_in_each_phase(observer):
    wheel = "/World/Carter/wheel"
    floor = "/World/Floor/Collision"
    irrelevant = header(wheel, floor)
    irrelevant.stage_id = 16
    observer._on_contacts(
        [
            irrelevant,
            header("/World/Carter_similar", floor),
            header(wheel, "/World/Carter/chassis"),
            header(wheel, floor, "CONTACT_LOST"),
        ],
        [],
    )
    assert observer.report()["phases"]["parked"]["crate_contact_reported"] is None
    observer._on_contacts([header(wheel, floor)], [])
    assert observer.report()["phases"]["parked"]["crate_contact_reported"] is False
    observer.set_phase("navigation")
    assert observer.report()["phases"]["navigation"]["crate_contact_reported"] is None


def test_target_pair_order_nested_colliders_and_contact_data_slices(observer):
    wheel = "/World/Carter/wheel"
    h = header("/World/Props/Crate", wheel, "CONTACT_PERSIST", start=1, count=2)
    h.actor0 = "/World"
    h.collider0 = "/World/Props/Crate/Geometry"
    observer.set_phase("navigation")
    observer._on_contacts(
        [
            h,
            header(wheel, "/World/Props/Crate_duplicate/Geometry"),
            header(wheel, h.collider0, "CONTACT_LOST"),
        ],
        [SimpleNamespace(separation=x) for x in (-100.0, 0.1, -0.2)],
    )
    row = observer.report()["phases"]["navigation"]
    assert row["crate_contact_reported"] is True
    assert row["crate_contact_points"] == 2
    assert row["minimum_reported_crate_separation_m"] == pytest.approx(-0.002)
    assert row["crate_headers"] == {"CONTACT_PERSIST": 1, "CONTACT_LOST": 1}


def test_callback_fault_marks_outcome_unmeasured_without_interrupting(observer):
    observer._on_contacts([header("/World/Carter/link", "/World/Floor")], [])
    observer._on_contacts(
        [header("/World/Carter/link", observer.crate_root, start=2, count=1)], []
    )
    result = observer.report()
    assert "invalid PhysX contact point span" in result["error"]
    assert result["phases"]["parked"]["crate_contact_reported"] is None


def test_bin_and_crate_subscriptions_score_their_own_subtrees(observer):
    bin_obs = CrateContactObserver("/World/Carter", "/World/Props/Bin", target_label="bin")
    bin_obs._stage_id = observer._stage_id
    bin_obs._path = str
    bin_obs._metres_per_unit = observer._metres_per_unit
    assert bin_obs.report()["phases"]["parked"]["bin_contact_reported"] is None
    bin_obs.installed = True
    headers = [header("/World/Carter/wheel", "/World/Props/Bin/Mesh")]
    for channel in (observer, bin_obs):
        channel._on_contacts(headers, [])
    assert observer.report()["phases"]["parked"]["crate_contact_reported"] is False
    assert bin_obs.report()["phases"]["parked"]["bin_contact_reported"] is True
    assert "crate_headers" not in bin_obs.report()["phases"]["parked"]
    assert bin_obs.report()["target_prim_path"] == "/World/Props/Bin"
