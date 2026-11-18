"""Separate adult-prior continuity; current motion and all other cues stay live."""

from dataclasses import replace

from robot.evaluation.cra_experiments import RestaurantScenario
from robot.evaluation.perception_readiness import evaluate_perception_readiness, runtime_factor_evidence_mode
from robot.perception.human_mass_prior import HUMAN_MASS_PRIOR_SOURCE
from robot.perception.physical_attributes import ScalarEstimate
from robot.risk.object_risk_policy import evaluate_object_risk
from robot.risk.physical_factor_adapter import adapt_physical_factors


def apply_body_identity_mass_retention(memory, body_memory, tracking, packet, objects, frames, policies, prior_config):
    fresh = {}
    uncertain_ids = {r.track_id for r in body_memory.uncertainty_regions}
    for item, frame in zip(objects, frames):
        cue = frame.get('geometry_diagnostics', {}).get('human_mass_prior', {})
        if (item.attributes.track_id not in uncertain_ids and frame['accepted'] and cue.get('source') == HUMAN_MASS_PRIOR_SOURCE
            and cue.get('value') is not None and cue.get('timestamp_seconds') == packet.camera_timestamp_seconds):
            # Read the actual current specialist result, before the generic
            # property cache can freeze/replace it with an older observation.
            fresh[item.attributes.track_id] = ScalarEstimate(**cue)
    retained, audit = memory.update(tracking, body_memory, fresh,
        timestamp_seconds=packet.camera_timestamp_seconds, camera_frame_id=packet.camera_frame_id)
    rows = {r['track_id']: r for r in audit['rows']}
    retained.update({key: fresh[key] for key, row in rows.items() if row['mode'] == 'fresh'})
    updated = []
    for item, frame in zip(objects, frames):
        key = item.attributes.track_id
        mass = retained.get(key)
        row = rows.get(key)
        existing = item.attributes.mass.mass_kg
        # The generic property cache cannot supply the adult prior when the
        # current body-specific check withheld it.
        if (mass is None and body_memory.retire_unresolved and existing.source == HUMAN_MASS_PRIOR_SOURCE
                and existing.is_available):
            if row is None:
                row = dict(track_id=key, mode='unavailable', reason='no verified current body identity for adult mass')
                audit['rows'].append(row)
            mass = ScalarEstimate.unavailable(unit='kg', source=HUMAN_MASS_PRIOR_SOURCE,
                timestamp_seconds=packet.camera_timestamp_seconds, evidence=(row['reason'],))
        if mass is None:
            updated.append(item)
            continue
        attributes = replace(item.attributes, mass=replace(item.attributes.mass, mass_kg=mass))
        settings = replace(item.adapted.config, maximum_persistent_property_age_seconds=3.)
        physical = adapt_physical_factors(attributes, timestamp_seconds=packet.camera_timestamp_seconds,
            potential_interaction_likelihood=dict(item.adapted.potential.inputs)['harmful_fall_interaction'],
            config=settings)
        # Rebuild only the mass-dependent factors. Do not discard current visual
        # contact/plume inputs or invent their unsupported physical attributes.
        adapted = replace(item.adapted, config=settings, potential=physical.potential, kinetic=physical.kinetic,
            profile=replace(item.adapted.profile, potential=physical.potential.estimate, kinetic=physical.kinetic.estimate))
        policy, details = evaluate_object_risk(adapted, prior_config,
            height_evidence=frame.get('height_evidence'),
            human_body_contact_prior=frame.get('human_body_contact_prior'), human_part_contact=frame.get('human_part_contact'))
        diagnostics = dict(item.diagnostics)
        diagnostics['human_mass_identity_retention'] = dict(row,
            generic_property_result=diagnostics['property_evidence']['mass'])
        diagnostics['property_evidence'] = dict(diagnostics['property_evidence'], mass=row)
        diagnostics['factor_scores'] = details
        readiness = evaluate_perception_readiness(RestaurantScenario('persistent_object', 'potential',
            'body-identity adult mass prior with current physical evidence', runtime_factor_evidence_mode(adapted), 0), adapted)
        policies[key] = policy
        updated.append(replace(item, attributes=attributes, adapted=adapted, policy=policy,
            readiness=readiness, diagnostics=diagnostics))
    return updated, audit
