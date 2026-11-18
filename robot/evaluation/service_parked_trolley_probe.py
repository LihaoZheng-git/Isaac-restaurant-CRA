"""Scoped parked trolley trial with unchanged full restaurant controls."""

import math

from robot.evaluation.service_crate_short_probe import MAXIMUM_SECONDS, REFERENCE_RECIPE


SCENE_NAME = "restaurant_parked_trolley_probe.usda"
TARGET_PATH = "/World/RestaurantLiveV1/PlateTrolley"
START_WORLD_XY_YAW = (28.35, 4.45, math.pi/2)
ROUTE_LOCAL_XY = ((3.4, 0.),)


def validate(args, reference_args, reference_case):
    near = reference_case == "long-parallel-near"
    if (not args.restaurant_scene or args.restaurant_scene.name != SCENE_NAME
            or args.service_crate_short_probe or args.service_lounge_bin_probe
            or (near and args.motion) or (not near and args.maximum_seconds != MAXIMUM_SECONDS)):
        raise ValueError("parked trolley trial requires its exact fixture and bounded route or parked near gate")
    allowed = {"output", "restaurant_scene", "restaurant_layout_survey", "service_parked_trolley_probe"}
    if not near:
        allowed.update(("maximum_seconds", "motion", "long_service_human_gates"))
    actual, expected = vars(args), vars(reference_args)
    changed = sorted(key for key in actual.keys() | expected.keys()
                     if key not in allowed and actual.get(key) != expected.get(key))
    if changed:
        raise ValueError("parked trolley probe changes the full policy: " + ", ".join(changed))
    return dict(passed=True, reference_recipe=REFERENCE_RECIPE, reference_case=reference_case,
                target="PlateTrolley", start_world_xy_yaw=START_WORLD_XY_YAW,
                nominal_route_local_xy=ROUTE_LOCAL_XY,
                controller_perception_uncertainty_arguments_match=True,
                scope="stationary trolley, remote legacy actor held for drive; moving trolley remains separate")
