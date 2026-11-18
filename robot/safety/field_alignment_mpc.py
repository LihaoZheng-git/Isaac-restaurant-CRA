"""Opt-in direct field/goal alignment objective for the static CRA pilot.

There is no grid route, authored bypass point, or prechecked-reference gate.
A current-field travel preference is frozen for each solve and supplies a soft
lookahead-velocity/heading cost. The base MPC still owns every executable bound,
exact DTCBF check, dense padded geometry check, and fallback. This is a local
planner objective experiment, not a global-convergence claim.
"""

from dataclasses import asdict, dataclass, fields
import math
from numbers import Integral, Real

import numpy as np
from scipy import sparse

from robot.safety.laplace_guidance_tracking_mpc import LaplaceGuidanceTrackingMPC
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPC, SafeSAGEUnicycleMPCResult


_LATERAL_GUIDANCE_SCALE = 1.25


@dataclass(frozen=True)
class FieldAlignmentConfig:
    velocity_weight: float = 4.
    heading_weight: float = 1.
    activation_safety_value: float = 2.
    maximum_tangent_fraction: float = 1.
    goal_fade_distance_m: float = 1.
    minimum_vector_norm: float = 1e-8
    minimum_guide_projection: float = 0.
    lateral_guidance_limit_radians: float = 0.

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f'{name} must be real')
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
            if name in {'activation_safety_value', 'goal_fade_distance_m', 'minimum_vector_norm'} and value == 0:
                raise ValueError(f'{name} must be positive')
            object.__setattr__(self, name, float(value))
        if self.maximum_tangent_fraction > 1:
            raise ValueError('maximum_tangent_fraction must be in [0, 1]')
        if self.minimum_guide_projection > 1:
            raise ValueError('minimum_guide_projection must be in [0, 1]')
        if self.lateral_guidance_limit_radians >= math.pi / 2:
            raise ValueError('lateral_guidance_limit_radians must be below pi/2')


@dataclass(frozen=True)
class FieldAlignmentMPCResult(SafeSAGEUnicycleMPCResult):
    field_alignment: dict


def _xy(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite XY vector')
    return result


class _AlignmentObjectiveMPC(SafeSAGEUnicycleMPC):
    """One request owns fixed targets; the nonlinear costs keep exact Jacobians."""

    def __init__(self, paper_filter, config, *, velocity, direction, speed_scale, alignment):
        super().__init__(paper_filter, config)
        self.velocity = np.array(velocity, dtype=float, copy=True)
        self.direction = np.array(direction, dtype=float, copy=True)
        self.velocity.setflags(write=False)
        self.direction.setflags(write=False)
        self.speed_scale = float(speed_scale)
        self.alignment = alignment

    def _alignment_residual(self, commands, chassis_states, sensitivities=None):
        """Residuals and optionally d residual / d flattened commands."""
        horizon = self.config.horizon_steps
        length = self.config.lookahead_distance
        residual = np.zeros((horizon, 4))
        jacobian = None if sensitivities is None else np.zeros((horizon, 4, 2*horizon))
        for index in range(horizon):
            yaw = chassis_states[index, 2]
            cosine, sine = math.cos(yaw), math.sin(yaw)
            speed, omega = commands[index]
            velocity = np.array([speed*cosine-length*omega*sine,
                                 speed*sine+length*omega*cosine])
            residual[index, :2] = (velocity-self.velocity)/self.speed_scale
            final_yaw = chassis_states[index+1, 2]
            heading = np.array([math.cos(final_yaw), math.sin(final_yaw)])
            residual[index, 2:] = heading-self.direction
            if jacobian is not None:
                velocity_yaw = np.array([-speed*sine-length*omega*cosine,
                                          speed*cosine-length*omega*sine])
                jacobian[index, :2] = velocity_yaw[:, None]*sensitivities[index, 2]
                jacobian[index, :2, 2*index:2*index+2] += np.array(
                    [[cosine, -length*sine], [sine, length*cosine]])
                jacobian[index, :2] /= self.speed_scale
                jacobian[index, 2:] = np.array([-heading[1], heading[0]])[:, None]*sensitivities[index+1, 2]
        return residual, jacobian

    def _residual_weights(self):
        return np.array([self.alignment.velocity_weight]*2+[self.alignment.heading_weight]*2)

    def _objective_value(self, commands, nominal_sequence, *, chassis_states=None, target_chassis_states=None):
        value = super()._objective_value(commands, nominal_sequence, chassis_states=chassis_states,
                                        target_chassis_states=target_chassis_states)
        if chassis_states is not None:
            residual, _ = self._alignment_residual(commands, chassis_states)
            value += float(np.sum(self._residual_weights()*residual**2))
        # Calls without states are the base search's command-only lower bound;
        # omitting this nonnegative cost preserves that bound.
        return value

    def _objective_gradient(self, commands, nominal_sequence, *, chassis_states, target_chassis_states):
        gradient = super()._objective_gradient(commands, nominal_sequence,
            chassis_states=chassis_states, target_chassis_states=target_chassis_states)
        sensitivity = self._rollout_state_sensitivities(chassis_states, commands)
        residual, jacobian = self._alignment_residual(commands, chassis_states, sensitivity)
        addition = 2*np.einsum('ijk,ij->k', jacobian, residual*self._residual_weights())
        return gradient+addition.reshape(commands.shape)

    def _quadratic_objective(self, nominal_sequence, *, reference, state_sensitivities,
                             target_chassis_states, slack_variable_count):
        matrix, vector = super()._quadratic_objective(nominal_sequence, reference=reference,
            state_sensitivities=state_sensitivities, target_chassis_states=target_chassis_states,
            slack_variable_count=slack_variable_count)
        residual, jacobian = self._alignment_residual(reference.commands, reference.chassis_states,
                                                     state_sensitivities)
        jacobian = jacobian.reshape(-1, 2*self.config.horizon_steps)
        weights = np.tile(self._residual_weights(), self.config.horizon_steps)
        offset = residual.ravel()-jacobian@reference.commands.ravel()
        hessian = 2*jacobian.T@(weights[:, None]*jacobian)
        addition = sparse.block_diag((sparse.csc_matrix(np.triu(hessian)),
            sparse.csc_matrix((slack_variable_count, slack_variable_count))), format='csc')
        vector = vector.copy()
        vector[:2*self.config.horizon_steps] += 2*jacobian.T@(weights*offset)
        return matrix+addition, vector


class FieldAlignmentMPC(LaplaceGuidanceTrackingMPC):
    """Direct field/goal alignment with a static pilot and an explicit restaurant mode.

The superclass provides compatibility with the existing nominal-goal runtime
interface only. Its checked-reference generation/tracking path is not called.
The field's circulation orientation is held for the mission goal in the static
pilot and for the active guide index in the restaurant. A current observed
passing selection supplies the restaurant's tangent during that encounter.
"""

    restaurant_mode = False

    def __init__(self, paper_filter, config=None, *, guidance_config=None, alignment_config=None,
                 route_goal_grid_xy=None, restaurant_mode=False):
        super().__init__(paper_filter, config, guidance_config=guidance_config)
        if not isinstance(restaurant_mode, bool):
            raise TypeError('restaurant_mode must be a boolean')
        self.restaurant_mode = restaurant_mode
        self.alignment_config = alignment_config or FieldAlignmentConfig()
        if not isinstance(self.alignment_config, FieldAlignmentConfig):
            raise TypeError('alignment_config must be FieldAlignmentConfig')
        self.route_goal_grid_xy = None if route_goal_grid_xy is None else tuple(_xy(route_goal_grid_xy, 'route_goal_grid_xy'))
        self._circulation_sign = None
        self._mission_goal = None
        self._validate_scope()

    def _validate_scope(self):
        guidance = self.guidance_config
        if (self.predictive_geometry is not None
                or guidance.reference_search_width != 0 or guidance.motion_forecast_maximum_outward_fraction != 0
                or (not self.restaurant_mode and
                    (guidance.tangent_velocity_fraction != 0 or guidance.moving_arc_reference))):
            raise ValueError('Field alignment is a separate static objective without moving, tangent-reference or beam modes')
        if self.config.position_tracking_weight == 0 and self.config.yaw_tracking_weight == 0:
            raise ValueError('Field alignment requires the existing nominal-state improvement path')

    def _travel_preference(self, initial, goal, speed, field, *, passing_selection=None):
        settings = self.alignment_config
        delta = goal-initial[:2]
        distance = float(np.linalg.norm(delta))
        goal_direction = delta/max(distance, settings.minimum_vector_norm)
        if distance <= settings.minimum_vector_norm:
            goal_direction = np.array([math.cos(initial[2]), math.sin(initial[2])])
        sample = self._sample(initial, field)
        gradient = np.asarray(sample.interpolant_gradient[:2])
        guide = np.asarray(sample.sampled_guidance_grid_xy)
        gradient_norm = float(np.linalg.norm(gradient))
        activation = math.sqrt(float(np.clip(1-sample.sampled_safety_value/settings.activation_safety_value, 0, 1)))
        fade = min(1., distance/settings.goal_fade_distance_m)**2
        if settings.lateral_guidance_limit_radians > 0. and passing_selection is None:
            # Keep the signed amplitude of the already-combined field. A
            # field with zero lateral component leaves the waypoint bearing.
            lateral = np.array([-goal_direction[1], goal_direction[0]])
            signed = float(np.dot(guide, lateral))
            normalized = float(np.clip(signed / _LATERAL_GUIDANCE_SCALE, -1., 1.))
            offset = math.tan(settings.lateral_guidance_limit_radians) * normalized * activation * fade
            direction = goal_direction + offset * lateral
            direction /= np.linalg.norm(direction)
            velocity = speed * direction
            return velocity, direction, dict(
                sampled_h=float(sample.sampled_safety_value), sampled_laplace_vector=guide.tolist(),
                sampled_planar_h_gradient=gradient.tolist(), sampled_unsafe_reason=sample.unsafe_reason,
                goal_direction=goal_direction.tolist(), tangent_source='signed_lateral_laplace',
                h_activation=activation, goal_distance_fade=fade, signed_lateral_guidance=signed,
                guidance_scale=_LATERAL_GUIDANCE_SCALE, applied_lateral_offset=offset,
                applied_heading_bias_radians=math.atan(offset), tangent_fraction=0., circulation_sign=None,
                preferred_velocity_grid_xy=velocity.tolist(), preferred_direction_grid_xy=direction.tolist(),
                preference_frozen_for_solve=True)
        occupied = any(grid.occupied_mask.any() for grid in field['configuration_space'].occupancy_grids)
        previous_sign = self._circulation_sign
        tangent = np.zeros(2)
        tangent_norm = 0.
        if passing_selection is not None:
            _, center, side = passing_selection
            radial = center-initial[:2]
            radial_norm = float(np.linalg.norm(radial))
            if radial_norm > settings.minimum_vector_norm:
                tangent = np.array([-radial[1], radial[0]]) / radial_norm
                tangent *= 1. if side == 'left' else -1.
                tangent_norm = 1.
        elif gradient_norm > settings.minimum_vector_norm and occupied:
            normal = gradient/gradient_norm
            tangent = guide-normal*np.dot(guide, normal)
            tangent_norm = float(np.linalg.norm(tangent))
            positive = np.array([-normal[1], normal[0]])
            if self._circulation_sign is None and tangent_norm > settings.minimum_vector_norm:
                self._circulation_sign = 1. if np.dot(tangent, positive) >= 0 else -1.
            if self._circulation_sign is not None:
                tangent = self._circulation_sign*positive
        raw_strength = settings.maximum_tangent_fraction*activation*fade if np.linalg.norm(tangent) > .5 else 0.
        strength = raw_strength
        projection_floor = settings.minimum_guide_projection
        if projection_floor > 0.:
            tangent_projection = float(np.clip(np.dot(tangent, goal_direction), -1., 1.))
            guide_loss = 1. - tangent_projection
            if 1. - strength * guide_loss < projection_floor:
                # The unnormalized blend has guide component 1 - strength * guide_loss.
                # Bound only the soft preference. Feasibility can still require waiting.
                strength = (1. - projection_floor) / guide_loss
        blended = (1-strength)*goal_direction+strength*tangent
        norm = float(np.linalg.norm(blended))
        direction = blended/norm if norm > settings.minimum_vector_norm else goal_direction
        velocity = speed*direction
        diagnostic = dict(sampled_h=float(sample.sampled_safety_value), sampled_laplace_vector=guide.tolist(),
            sampled_planar_h_gradient=gradient.tolist(), sampled_unsafe_reason=sample.unsafe_reason,
            goal_direction=goal_direction.tolist(), field_tangent_direction=tangent.tolist(),
            observed_tangent_norm=tangent_norm, circulation_sign=self._circulation_sign,
            circulation_selected_this_request=previous_sign is None and self._circulation_sign is not None,
            circulation_direction_held=previous_sign is not None,
            circulation_selection_basis='sign of current Laplace vector projected tangent to current planar h gradient',
            circulation_release_basis='reset on mission-goal change; tangent influence fades quadratically within goal_fade_distance_m',
            h_activation=activation, goal_distance_fade=fade, tangent_fraction=strength,
            preferred_velocity_grid_xy=velocity.tolist(), preferred_direction_grid_xy=direction.tolist(),
            preference_frozen_for_solve=True)
        if projection_floor > 0.:
            diagnostic.update(minimum_guide_projection=projection_floor,
                              uncapped_tangent_fraction=raw_strength,
                              tangent_guide_projection=tangent_projection,
                              blended_guide_projection=float(np.dot(blended, goal_direction)),
                              guide_projection_cap_applied=strength < raw_strength)
        if self.restaurant_mode:
            diagnostic['circulation_release_basis'] = (
                'reset on active guide change; tangent influence fades quadratically within goal_fade_distance_m')
            diagnostic['tangent_source'] = 'selected_observed_obstacle' if passing_selection is not None else 'laplace_field'
            if passing_selection is not None:
                key, center, side = passing_selection
                diagnostic.update(selected_passing_track_id=key, selected_passing_center_grid_xy=center.tolist(),
                                  selected_passing_side=side, selected_passing_tangent_available=tangent_norm > .5)
        return velocity, direction, diagnostic

    def plan(self, initial_chassis_state_grid, nominal_command, *, nominal_goal_grid_xy=None,
             route_goal_grid_xy=None, active_guide_index=None, selected_passing_tangents=(), observed_motion_forecast=None,
             observed_motion_source_age_seconds=None, observed_motion_anchor=None, **kwargs):
        self._validate_scope()
        references = tuple(selected_passing_tangents)
        if ((references and not self.restaurant_mode) or observed_motion_forecast is not None
                or observed_motion_source_age_seconds is not None or observed_motion_anchor is not None):
            raise ValueError('Field alignment cannot consume moving-obstacle preferences')
        passing_selection = None
        if self.restaurant_mode:
            if isinstance(active_guide_index, bool) or not isinstance(active_guide_index, Integral) or active_guide_index < 0:
                raise ValueError('Restaurant field alignment requires a nonnegative active guide index')
            active_guide_index = int(active_guide_index)
            if references:
                if len(references) != 1:
                    raise ValueError('Restaurant field alignment requires one current passing selection')
                (key, center, side), = references
                if isinstance(key, bool) or not isinstance(key, int) or key < 0 or side not in ('left', 'right'):
                    raise ValueError('Current passing selection requires a track ID and explicit side')
                passing_selection = (key, _xy(center, 'selected passing center'), side)
        elif active_guide_index is not None:
            raise ValueError('Active guide index requires restaurant field alignment')
        if kwargs.get('target_chassis_states_grid') is not None:
            raise ValueError('Field alignment cannot replace a separate state-path target')
        initial = np.asarray(initial_chassis_state_grid, dtype=float)
        if initial.shape != (3,) or not np.isfinite(initial).all():
            raise ValueError('initial_chassis_state_grid must be a finite state')
        nominal_goal = _xy(nominal_goal_grid_xy, 'nominal_goal_grid_xy')
        selected = self.route_goal_grid_xy if route_goal_grid_xy is None else route_goal_grid_xy
        lateral_mode = self.alignment_config.lateral_guidance_limit_radians > 0. and passing_selection is None
        goal = nominal_goal if selected is None or lateral_mode else _xy(selected, 'route_goal_grid_xy')
        circulation_key = active_guide_index if self.restaurant_mode else tuple(goal)
        goal_changed = self._mission_goal != circulation_key
        if goal_changed:
            self._mission_goal = circulation_key
            self._circulation_sign = None
        nominal = self._nominal_sequence(nominal_command)
        speed = float(np.clip(np.max(nominal[:, 0]), 0., self.config.maximum_linear_velocity))
        soft_objective = self.alignment_config.velocity_weight > 0. or self.alignment_config.heading_weight > 0.
        if soft_objective:
            velocity, direction, diagnostic = self._travel_preference(
                initial, goal, speed, kwargs, passing_selection=passing_selection)
            worker = _AlignmentObjectiveMPC(self.paper_filter, self.config,
                velocity=velocity, direction=direction, speed_scale=max(speed, .05), alignment=self.alignment_config)
        else:
            worker = SafeSAGEUnicycleMPC(self.paper_filter, self.config)
            diagnostic = dict(soft_alignment_objective_enabled=False, tangent_source='none', tangent_fraction=0.)
        # This is the original obstacle-unaware nominal rollout, not a bypass
        # reference. It activates the base optimizer's feasible-cost improvement
        # path even when the nominal commands already pass the safety checks.
        nominal_states = worker._rollout(initial, worker._clip_commands(nominal))
        kwargs = dict(kwargs, target_chassis_states_grid=nominal_states)
        result = worker.plan(initial, nominal, **kwargs)
        settings = asdict(self.alignment_config)
        if self.alignment_config.minimum_guide_projection == 0.:
            del settings['minimum_guide_projection']
        if self.alignment_config.lateral_guidance_limit_radians == 0.:
            del settings['lateral_guidance_limit_radians']
        diagnostic.update(basis=('direct_frozen_field_goal_alignment_soft_objective' if soft_objective
                                 else 'waypoint_only_objective_with_original_field_constraints'),
            settings=settings, initial_chassis_state_grid=initial.tolist(),
            nominal_goal_grid_xy=nominal_goal.tolist(), route_goal_grid_xy=goal.tolist(),
            nominal_commands_unchanged=True, original_nominal_state_tracking_active=True,
            no_checked_reference_required=True, no_grid_route=True,
            preference_speed_mps=speed, velocity_normalization_mps=max(speed, .05),
            inherited_checked_reference_weight_unused=self.guidance_config.weight,
            scope='Request-local soft velocity/heading cost; original nominal commands, CRA field, limits, exact DTCBF and padded geometry checks unchanged.')
        if self.restaurant_mode:
            diagnostic.update(active_guide_index=active_guide_index, circulation_reset_for_new_guide=goal_changed)
        else:
            diagnostic['circulation_reset_for_new_goal'] = goal_changed
        if lateral_mode:
            diagnostic['route_goal_grid_xy'] = nominal_goal.tolist() if selected is None else _xy(selected, 'route_goal_grid_xy').tolist()
            diagnostic['preference_goal_grid_xy'] = nominal_goal.tolist()
        return FieldAlignmentMPCResult(**{item.name: getattr(result, item.name) for item in fields(SafeSAGEUnicycleMPCResult)},
                                       field_alignment=diagnostic)
