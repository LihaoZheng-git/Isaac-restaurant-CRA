"""Apply opt-in score calibration after ownership and human mass admission."""

from dataclasses import asdict, replace
import math

from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE
from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE
from robot.perception.physical_attributes import ScalarEstimate
from robot.risk.physical_factor_adapter import _maximum_input_age
from robot.risk.restaurant_calibration import (
    RestaurantCalibrationConfig, RestaurantFactorWeights, calibrate_object_policy,
)


def restaurant_calibration_preset(name, *, normal_flux_magnitude_at_unit_risk=1.7):
    if name == "off":
        if normal_flux_magnitude_at_unit_risk != 1.7:
            raise ValueError("alternative boundary response requires restaurant calibration")
        return None
    if name not in {"equal", "kinetic"}:
        raise ValueError("unknown restaurant calibration preset")
    return RestaurantCalibrationConfig(enabled=True, kinetic_energy_scale_j=50.,
        weights=RestaurantFactorWeights(kinetic=1.5 if name == "kinetic" else 1.),
        normal_flux_magnitude_at_unit_risk=normal_flux_magnitude_at_unit_risk)


def object_is_eligible(detail):
    """Current ownership is required even when intrinsic properties are stored."""
    return bool(detail.get("policy_return_count", 0) > 0 and (
        detail.get("accepted_instance") or detail.get("static_prop_policy") or
        any(detail.get("property_evidence", {}).get(name, {}).get("mode") == "persistence_prior"
            for name in ("mass", "contact"))))


def verified_human_ids(objects, envelopes, sensor_timestamp):
    """Use admitted source-bound person evidence; never semantic asset names."""
    result = set()
    for row in envelopes:
        original = row.get("original_person_camera_timestamp", math.nan)
        age = row.get("identity_age_seconds", math.nan)
        if (not all(isinstance(t, (float, int)) and not isinstance(t, bool)
                    and math.isfinite(t) and t >= 0 for t in (original, age))
                or row.get("lidar_timestamp_seconds") != sensor_timestamp
                or row.get("source") != HUMAN_ENVELOPE_SOURCE
                or row.get("status") != "prior" or row.get("current_support_count", 0) < 3
                or row["track_id"] in result
                or not -1e-9 <= original + age - sensor_timestamp <= .25 + 1e-9):
            raise ValueError("human calibration requires current verified identity")
        result.add(row["track_id"])
    for obj in objects:
        d = obj["diagnostics"]
        cue = d.get("geometry_evidence", {}).get("human_mass_prior", {})
        if (d.get("accepted_instance") and d.get("policy_return_count", 0) > 0
                and cue.get("source") == HUMAN_MASS_PRIOR_SOURCE and cue.get("status") == "prior"
                and cue.get("value") is not None
                and cue.get("timestamp_seconds") == obj["attributes"]["timestamp_seconds"]):
            result.add(obj["track_id"])
    return result


def apply_restaurant_calibration(objects, policies, config, envelopes, sensor_timestamp,
                                 *, carried_groups=None):
    """Return new assessments/policies; no geometry, IDs or evidence are changed."""
    if config is None or not config.enabled:
        return objects, policies, None
    records = [dict(track_id=o.attributes.track_id,
                    attributes=dict(timestamp_seconds=o.attributes.timestamp_seconds),
                    diagnostics=o.diagnostics) for o in objects]
    humans = verified_human_ids(records, envelopes, sensor_timestamp)
    updated, policies, rows = [], dict(policies), []
    for item in objects:
        key, detail = item.attributes.track_id, item.diagnostics
        cfg = item.adapted.config
        eligible = (object_is_eligible(detail) or
                    (key in humans and detail.get("policy_return_count", 0) > 0))
        if eligible and not cfg.use_upper_bounds:
            raise ValueError("restaurant calibration requires original upper-bound factor evidence")
        proxy = detail.get("contact_proxy")
        maximum_age = cfg.maximum_evidence_age_seconds
        if proxy is not None:
            proxy = proxy if isinstance(proxy, ScalarEstimate) else ScalarEstimate(**proxy)
            maximum_age = _maximum_input_age(
                detail["factor_scores"]["contact"]["basis"], proxy, cfg)
        contact_interaction = dict(item.adapted.contact.inputs).get("contact_interaction_likelihood")
        policy = calibrate_object_policy(item.policy, detail["factor_scores"], config,
            eligible=eligible,
            verified_human=key in humans,
            source_energy_scales_j=dict(potential=cfg.potential_energy_scale_j,
                                       kinetic=cfg.kinetic_energy_scale_j),
            contact_proxy=proxy, contact_interaction=contact_interaction,
            timestamp_seconds=item.adapted.timestamp_seconds,
            maximum_contact_age_seconds=maximum_age,
            maximum_interaction_age_seconds=cfg.maximum_evidence_age_seconds,
            prior_confidence_cap=cfg.prior_confidence_cap)
        # Earlier carried-group assembly may have replaced this entry with a
        # different member's policy. Restore the individual before regrouping.
        policies[key] = policy
        if policy is item.policy:
            updated.append(item)
            continue
        assessment = policy.calibration.as_dict()
        detail = dict(detail, restaurant_calibration=assessment,
                      legacy_factor_scores=detail["factor_scores"])
        if assessment["factors"]:
            detail["factor_scores"] = {
                name: dict(value, score=assessment["factors"][name]["score"],
                           original_mode=value["mode"],
                           mode="evidence_supported" if assessment["factors"][name]["supported"] else value["mode"],
                           calibration_mode=assessment["factors"][name]["calibration_mode"],
                           score_scope="restaurant_calibrated_factor; original estimate retained")
                for name, value in detail["factor_scores"].items()}
        updated.append(replace(item, policy=policy, diagnostics=detail))
        rows.append(dict(track_id=key, **assessment))
    # Preserve the established maximum-member rule; never sum two objects' mass.
    from robot.risk.carried_group_policy import maximum_member_policy
    calibrated_groups = []
    for group in (carried_groups or {}).get("confirmed_groups", []):
        members = group["members"]
        chosen, group_audit = maximum_member_policy({key: policies[key] for key in members})
        calibrated_groups.append(group_audit)
        for key in members:
            policies[key] = chosen
    return updated, policies, dict(config=asdict(config), objects=rows,
        verified_human_ids=sorted(humans), confirmed_groups=calibrated_groups,
        legacy_confirmed_groups=(carried_groups or {}).get("confirmed_groups", []),
        unknown_policy_unchanged=True,
        boundary_mapping=dict(zero_risk_magnitude=.5,
            unit_risk_magnitude=config.normal_flux_magnitude_at_unit_risk,
            formula="b=-(0.5+(unit_risk_magnitude-0.5)*phi)",
            scope="calibrated verified/retained object policies only; phi and palette unchanged"),
        scope="after original ownership and physical evidence; fixed lambda and geometry unchanged")
