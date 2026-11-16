import pytest

from test_live_object_perception import worker
from robot.perception.human_mass_prior import HumanMassPriorConfig
from robot.perception.human_body_envelope import HumanBodyEnvelopeConfig
from robot.perception.persistent_object_evidence import PersistentObjectEvidenceConfig
from robot.safety.motion_aware_passing import MotionAwarePassingConfig
from robot.risk.object_risk_policy import ObjectRiskPriorConfig


def config():
    return dict(
        person_detector=object(),
        human_mass_prior_config=HumanMassPriorConfig("unit separated adult"),
        persistent_property_config=PersistentObjectEvidenceConfig(
            "unit separated adult"
        ),
        object_factor_prior_config=ObjectRiskPriorConfig(),
        human_body_envelope_config=HumanBodyEnvelopeConfig(
            body_radius_m=0.4, centroid_uncertainty_m=0.25
        ),
        human_boundary_flux=-1.7,
        motion_passing_config=MotionAwarePassingConfig(
            require_visual_person_selection=True
        ),
        human_observed_occupancy_only=False,
        human_visual_fragment_association=True,
        human_fresh_person_policy=True,
    )


def test_existing_observed_only_scope_remains_required_without_explicit_new_mode():
    with pytest.raises(ValueError, match="observed-only"):
        worker(**config())


def test_explicit_prior_geometry_scope_keeps_existing_bound_and_current_helpers():
    runtime = worker(
        **config(),
        human_prior_geometry_associations=True,
        human_fragment_mask_boundary_pixels=1,
        human_visual_part_field_policy=True,
    )
    assert not runtime.human_observed_occupancy_only
    assert (
        runtime.human_visual_fragment_association and runtime.human_fresh_person_policy
    )
    assert runtime.human_body_envelope.config.body_radius_m == 0.4
    assert runtime.human_body_envelope.config.centroid_uncertainty_m == 0.25
    assert runtime.human_prior_geometry_associations


@pytest.mark.parametrize("problem", ["boolean", "observed", "no_body", "no_helpers"])
def test_incompatible_prior_scope_is_rejected(problem):
    args = config()
    args["human_prior_geometry_associations"] = True
    if problem == "boolean":
        args["human_prior_geometry_associations"] = 1
    elif problem == "observed":
        args["human_observed_occupancy_only"] = True
    elif problem == "no_body":
        args["human_body_envelope_config"] = None
        args["human_boundary_flux"] = None
    else:
        args["human_visual_fragment_association"] = args[
            "human_fresh_person_policy"
        ] = False
    with pytest.raises(ValueError, match="prior-geometry"):
        worker(**args)
