"""A fixed short-route evaluation harness for the unchanged full service policy."""

import math


START_WORLD_XY_YAW = (15.65, 4.05, 0.0)
ROUTE_LOCAL_XY = ((4.0, 0.0),)
MAXIMUM_SECONDS = 45.0
LOUNGE_BIN_SCENE = "restaurant_lounge_bin_full_probe.usda"
LOUNGE_BIN_START_WORLD_XY_YAW = (25.35, 4.05, 0.)
LOUNGE_BIN_ROUTE_LOCAL_XY = ((3.05, 0.),)
REFERENCE_RECIPE = "experiment/configs/left_policy_reference.json"
SCOPE = (
    "4 m crate-only route using full service controller, perception and uncertainty settings; "
    "fresh observation history, not a full restaurant or human-encounter acceptance"
)


def validate_service_crate_short_probe(args, reference_args):
    """Only experiment output/gates, fixed route selector and duration may differ."""
    if not args.service_crate_short_probe or args.maximum_seconds != MAXIMUM_SECONDS:
        raise ValueError("service crate probe requires its explicit selector and fixed 45 s bound")
    allowed = {"output", "motion", "long_service_human_gates", "maximum_seconds", "service_crate_short_probe"}
    actual, expected = vars(args), vars(reference_args)
    changed = sorted(key for key in actual.keys() | expected.keys()
                     if key not in allowed and actual.get(key) != expected.get(key))
    if changed:
        raise ValueError("service crate probe differs from full calibrated service settings: " + ", ".join(changed))
    return dict(passed=True, scope=SCOPE, reference_recipe=REFERENCE_RECIPE,
                start_world_xy_yaw=START_WORLD_XY_YAW, nominal_route_local_xy=ROUTE_LOCAL_XY,
                maximum_seconds=MAXIMUM_SECONDS,
                controller_perception_uncertainty_arguments_match=True,
                changed_argument_names=sorted(key for key in actual if actual[key] != expected.get(key)),
                human_prerequisites="unchanged fresh component, near-view and idle-actor gates")


def validate_service_lounge_bin_probe(args, reference_args, reference_case):
    """Require the full V5 policy, with a different scene and evaluation route."""
    near_view = reference_case == "long-parallel-near"
    if (not args.service_lounge_bin_probe or args.service_crate_short_probe
        or not args.restaurant_scene or args.restaurant_scene.name != LOUNGE_BIN_SCENE
        or (near_view and args.motion) or (not near_view and args.maximum_seconds != MAXIMUM_SECONDS)):
        raise ValueError("lounge bin probe requires its separate scene and either parked near view or a 45 s drive bound")
    allowed = {"output", "motion", "long_service_human_gates",
               "service_lounge_bin_probe", "restaurant_scene", "restaurant_layout_survey"}
    if not near_view:
        allowed.add("maximum_seconds")
    actual, expected = vars(args), vars(reference_args)
    changed = sorted(key for key in actual.keys() | expected.keys()
                     if key not in allowed and actual.get(key) != expected.get(key))
    if changed:
        raise ValueError("lounge bin probe differs from full calibrated service settings: " + ", ".join(changed))
    if near_view:
        return dict(passed=True, reference_recipe=REFERENCE_RECIPE, reference_case=reference_case,
                    scope="stationary near-human prerequisite using its original pose and thresholds",
                    controller_perception_uncertainty_arguments_match=True)
    return dict(passed=True, reference_recipe=REFERENCE_RECIPE, reference_case=reference_case,
                target="ServiceContainer",
                start_world_xy_yaw=LOUNGE_BIN_START_WORLD_XY_YAW,
                nominal_route_local_xy=LOUNGE_BIN_ROUTE_LOCAL_XY, maximum_seconds=MAXIMUM_SECONDS,
                controller_perception_uncertainty_arguments_match=True,
                changed_argument_names=sorted(key for key in actual if actual[key] != expected.get(key)),
                scope="bounded lounge bin only; native legacy performer stays held; human exercise verified post-run")


def summarize_unexercised_human_scope(records, truth):
    """Exclude exercise requirements only when no human encounter was exercised.

    This does not modify any raw human audit. If any human evidence is present,
    the caller keeps every original human acceptance result. Actor truth is used
    for this post-run scope check only, never for control.
    """
    finite = bool(truth) and all(
        all(math.isfinite(v) for v in (*r["actor_xyz"], *r["robot_xy"], r["time"]))
        for r in truth
    )
    ordered = finite and all(b["time"] > a["time"] for a, b in zip(truth, truth[1:]))
    held = bool(truth) and all(r["phase"] == "idle_before" and r.get("initial_idle_held") is True
                              and r.get("native_agent_enabled") is False for r in truth)
    drift = max((math.dist(r["actor_xyz"][:2], truth[0]["actor_xyz"][:2]) for r in truth), default=math.inf)
    distance = min((math.dist(r["actor_xyz"][:2], r["robot_xy"]) for r in truth), default=-math.inf)
    evidence_keys = ("human_body_envelope_priors", "tracking_uncertainty_regions",
                     "motion_passing_reference_track_ids", "motion_passing_reference_sides",
                     "crossing_yield_track_ids", "rear_pass_reference_track_ids")
    no_runtime_human_evidence = bool(records) and all(
        not any(r["diagnostics"].get(key) for key in evidence_keys)
        and not any(row.get("mode") != "unavailable" for row in
                    (r["diagnostics"].get("human_mass_body_identity_retention") or {}).get("rows", ()))
        for r in records
    )
    checks = dict(finite_ordered_actor_records=bool(ordered), original_actor_remained_held=held,
                  original_idle_drift_below_1mm=finite and drift <= .001,
                  outside_original_2m_nearby_encounter_region=finite and distance > 2.0,
                  no_runtime_human_evidence=no_runtime_human_evidence)
    unexercised = all(checks.values())
    return dict(status="unexercised" if unexercised else "requires_original_human_acceptance",
                unexercised=unexercised, checks=checks,
                actor_drift_m=drift if finite else None,
                minimum_actor_robot_distance_m=distance if finite else None,
                human_encounter_passed=None,
                scope="post-run scope only; raw human audits retained; no human encounter pass claimed")
