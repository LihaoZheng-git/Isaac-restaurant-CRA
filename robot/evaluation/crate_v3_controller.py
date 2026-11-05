"""Explicit V3 controller settings for the isolated, no-human crate probe.

The zero tangent fraction is V3's existing runtime fallback when no current
human passing reference exists. It does not select the crate as a human.
"""

V3_STATIC_CONTROLLER = {
    'experiment_baseline': 'continuous_cra_safe_sage',
    'grid_resolution': .05,
    'mpc_horizon_steps': 6,
    'mpc_tapered_rescue_seed': True,
    'mpc_consistent_temporal_jacobian': True,
    'mpc_unfinished_qp_proposal': False,
    'qp_iteration_limit': 1000,
    'nonlinear_rescue_iterations': 20,
    'progress_seed_fraction': .9,
    'waypoint_mode': 'guides',
    'route_guide_lookahead': 2.,
    'route_guide_max_cross_track': 1.5,
    'guidance_tracking_weight': 100.,
    'guidance_activation_h': 2.,
    'guidance_max_heading_bias_degrees': 60.,
    'guidance_reference_goal_radius': .04,
    'guidance_fade_distance': .15,
    'guidance_minimum_reference_progress': .01,
    'guidance_turn_first_minimum_cap': .4,
    'guidance_reference_mode': 'planar_forecast_velocity',
    'guidance_reference_minimum_speed_fraction': 1.,
    'guidance_reference_search_width': 0,
    'guidance_reference_search_candidates': 4800,
    'guidance_tangent_velocity_fraction': 0.,
    'guidance_tangent_basis': 'selected_observed_obstacle',
    'guidance_tangent_blend': 'convex',
    'guidance_moving_arc_reference': True,
    'guidance_outward_reference_fraction': 0.,
    'guidance_turn_translation_fraction': 0.,
    'crossing_strategy': 'legacy',
    'simulation_rate': .025,
    'planner_result_max_age': .5,
    'immediate_replan': True,
}

NO_HUMAN_SELECTION = (
    'restaurant_human', 'restaurant_server', 'motion_aware_passing',
    'human_observed_passing_target', 'parallel_guidance_ablation',
    'predictive_human_geometry_ablation', 'predictive_parallel_redesign',
    'observed_motion_reference', 'rear_preparation', 'checked_rear_pass_reference',
    'zero_tangential_bias_ablation', 'encounter_guidance_ablation',
)


def validate_crate_v3_controller(args):
    """Reject a misconfigured opt-in before simulator startup; never retune it."""
    if not args.crate_route_probe or not args.confirmed_prop_refresh:
        raise ValueError('V3 crate controller requires the gated closer-crate refresh probe')
    if any(getattr(args, key) for key in NO_HUMAN_SELECTION):
        raise ValueError('V3 crate controller must have no human selection or predictive policy')
    different = [key for key, value in V3_STATIC_CONTROLLER.items()
                 if getattr(args, key) != value]
    if different:
        raise ValueError('V3 crate controller setting mismatch: ' + ', '.join(different))
    return dict(passed=True, settings=dict(V3_STATIC_CONTROLLER),
        scope='V3 controller with existing no-human-selection tangent fallback; same static crate sensing/occupancy and 4m straight route, not a mixed-human scene or isolated one-parameter ablation')
