from dataclasses import replace

import pytest

from robot.perception.physical_attributes import EstimateStatus, MassAttributes, PhysicalAttributes, ScalarEstimate
from robot.risk.physical_factor_adapter import PhysicalFactorAdapterConfig
from robot.risk.physical_profile_provider import PhysicalAttributeCRARiskProfileProvider
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy


def scalar(value, unit="1"):
    return ScalarEstimate(value, value, value, 1., unit, "controlled_oracle", 1., EstimateStatus.ORACLE)


def evidence(track_id):
    base = PhysicalAttributes.unknown(track_id=track_id, timestamp_seconds=1.)
    return replace(base, mass=MassAttributes(scalar(1., "kg")),
        geometry=replace(base.geometry, fall_height_m=scalar(0., "m")),
        motion=replace(base.motion, relative_speed_mps=scalar(0., "m/s"), route_crossing_likelihood=scalar(.8)),
        thermal=replace(base.thermal, heat_source_likelihood=scalar(0.)))


def test_proxy_changes_cra_flux_through_provider_while_lambda_stays_fixed():
    records = {track_id: evidence(track_id) for track_id in (1, 2)}
    interactions = {track_id: scalar(.8) for track_id in records}
    provider = PhysicalAttributeCRARiskProfileProvider(records, timestamp_seconds=1.,
        potential_interactions=interactions, thermal_interactions=interactions,
        contact_interactions=interactions, visual_contact_proxies={1: scalar(.1), 2: scalar(.8)},
        config=PhysicalFactorAdapterConfig(allow_visual_contact_proxy=True))
    first, second = (provider.diagnostics_by_track_id[track_id] for track_id in (1, 2))
    low, high = (CRASafeSAGEPolicy().evaluate(result.profile) for result in (first, second))
    assert low.assessment.phi < high.assessment.phi
    assert low.guidance_parameters.boundary_flux_magnitude < high.guidance_parameters.boundary_flux_magnitude
    assert low.tangential_strength == high.tangential_strength == 1.25
    assert first.contact.basis == "visual_exposed_edge_proxy"
    assert records[1].contact.sharpness.status is EstimateStatus.UNAVAILABLE


def test_provider_does_not_allow_proxy_without_opt_in():
    with pytest.raises(ValueError, match="opt-in"):
        PhysicalAttributeCRARiskProfileProvider({1: evidence(1)}, timestamp_seconds=1.,
                                               visual_contact_proxies={1: scalar(.1)})
